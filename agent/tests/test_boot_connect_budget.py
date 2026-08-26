"""Boot connect budget (P0).

Agent and speech service start together; the agent then waits on SLV model
load, not on the session limiter. Measured on rk3576 (2026-08-25) the speech
service needed 102s to reach "Speech service ready" and the agent connected
with 16s of budget left under the old 75s deadline — surviving only because
its own init burned wall-clock first. The budget must cover model load and
be tunable per device.
"""
from __future__ import annotations

import pytest

from ovs_agent.app_base import BaseApp


def _resolve(monkeypatch, raw: str | None) -> float:
    if raw is None:
        monkeypatch.delenv("OVS_AGENT_BOOT_CONNECT_DEADLINE_S", raising=False)
    else:
        monkeypatch.setenv("OVS_AGENT_BOOT_CONNECT_DEADLINE_S", raw)
    return BaseApp._boot_connect_deadline_s(BaseApp)


def test_default_covers_observed_cold_start(monkeypatch):
    # 102s observed model load; the default needs real headroom over it.
    assert _resolve(monkeypatch, None) >= 150.0


def test_env_override(monkeypatch):
    assert _resolve(monkeypatch, "240") == 240.0


@pytest.mark.parametrize("raw", ["", "   ", "abc", "0", "-5"])
def test_bad_values_fall_back_to_default(monkeypatch, raw):
    assert _resolve(monkeypatch, raw) == BaseApp._BOOT_CONNECT_DEADLINE_S


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NaN", "Infinity"])
def test_non_finite_values_fall_back_to_default(monkeypatch, raw):
    """float() accepts nan/inf and `value <= 0` is False for both.

    nan would make the retry loop's `remaining <= 0` check never fire —
    unbounded retries at zero backoff; inf retries forever.
    """
    assert _resolve(monkeypatch, raw) == BaseApp._BOOT_CONNECT_DEADLINE_S


def test_resolved_budget_is_always_a_usable_deadline(monkeypatch):
    """Whatever the env says, the loop must get a finite positive budget."""
    import math

    for raw in [None, "", "abc", "0", "-5", "nan", "inf", "-inf", "240"]:
        value = _resolve(monkeypatch, raw)
        assert math.isfinite(value) and value > 0, f"{raw!r} -> {value!r}"


def test_instance_attribute_still_wins_when_env_unset(monkeypatch):
    """test_advertise_reconnect shrinks the budget via the instance attr."""
    monkeypatch.delenv("OVS_AGENT_BOOT_CONNECT_DEADLINE_S", raising=False)

    class _Stub:
        _BOOT_CONNECT_DEADLINE_S = 3.0
        _boot_connect_deadline_s = BaseApp._boot_connect_deadline_s

    assert _Stub()._boot_connect_deadline_s() == 3.0
