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


def test_instance_attribute_still_wins_when_env_unset(monkeypatch):
    """test_advertise_reconnect shrinks the budget via the instance attr."""
    monkeypatch.delenv("OVS_AGENT_BOOT_CONNECT_DEADLINE_S", raising=False)

    class _Stub:
        _BOOT_CONNECT_DEADLINE_S = 3.0
        _boot_connect_deadline_s = BaseApp._boot_connect_deadline_s

    assert _Stub()._boot_connect_deadline_s() == 3.0
