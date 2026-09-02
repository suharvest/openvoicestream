"""The arm may be plugged in AFTER the agent starts.

Regression guard for the 2026-09-01 outage: the stack had been up since
08-25, the B601-DM was plugged in a week later, and the arm stayed dead until
the container was restarted. Two independent one-shot assumptions caused it:

  1. ``_make_rebot_arm`` resolved the serial channel at CONSTRUCTION. With no
     arm on the bus that raised, ``ArmPlugin.setup()`` returned False, and the
     plugin was disabled for the rest of the process lifetime.
  2. ``ArmPlugin.start()`` called ``arm.connect()`` exactly once and fell back
     to cache-only forever on failure.

(The third cause — docker's ``devices:`` mapping a node once at container
start — is fixed in deploy/docker-compose.jetson-rebot.yml, not here.)
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

import pytest

import ovs_agent.apps.voice_rebot_arm.rebot_actuator as mod
from ovs_agent.apps.voice_rebot_arm.rebot_actuator import (
    RebotArmActuator,
    _make_rebot_arm,
)
from ovs_agent.plugins.actuator_actions import ArmPlugin


class _FakeRebotArm:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def connect(self, *a: Any, **k: Any) -> None: ...
    def enable(self, *a: Any, **k: Any) -> None: ...
    def set_gripper_enable(self, *a: Any, **k: Any) -> None: ...
    def get_tcp_pose(self):
        return [0.0] * 6


# ── construction must not touch the bus ─────────────────────────────


def test_builder_does_not_resolve_channel_at_construction(monkeypatch) -> None:
    """channel='auto' with no arm attached must still build an actuator.

    This is the exact call ArmPlugin.setup() makes. Before the fix it raised
    RuntimeError('no match') here and took the whole plugin down with it.
    """
    def _boom(*a: Any, **k: Any):
        raise RuntimeError("no serial port matched (arm not plugged in)")

    monkeypatch.setattr(mod, "resolve_serial_port", _boom)

    act = _make_rebot_arm({"channel": "auto"})
    assert isinstance(act, RebotArmActuator)
    assert act._channel_spec == "auto"  # noqa: SLF001


def test_connect_resolves_channel_late(monkeypatch) -> None:
    """Resolution must happen at connect() TIME, not just produce the right value.

    Asserting only that the resolved path reaches the SDK is not enough — that
    held before the fix too, when resolution ran in __init__. Count the calls.
    """
    captured: Dict[str, Any] = {}
    calls: list[str] = []

    def _resolve(*a: Any, **k: Any) -> str:
        calls.append("resolve")
        return "/dev/ttyACM7"

    monkeypatch.setattr(mod, "resolve_serial_port", _resolve)
    monkeypatch.setattr(
        mod, "RebotArm", lambda **kw: captured.update(kw) or _FakeRebotArm()
    )

    act = _make_rebot_arm({"channel": "auto"})
    assert calls == [], "construction must not touch the bus"
    act.connect()
    assert calls == ["resolve"]
    assert captured["channel"] == "/dev/ttyACM7"
    act.connect()
    assert calls == ["resolve", "resolve"], "each connect re-resolves"


def test_connect_picks_up_a_node_that_appears_later(monkeypatch) -> None:
    """A failed connect must not poison later attempts: re-resolve each time."""
    ports = [None, None, "/dev/ttyACM3"]
    captured: Dict[str, Any] = {}

    def _resolve(*a: Any, **k: Any) -> str:
        port = ports.pop(0)
        if port is None:
            raise RuntimeError("no serial port matched")
        return port

    monkeypatch.setattr(mod, "resolve_serial_port", _resolve)
    monkeypatch.setattr(
        mod, "RebotArm", lambda **kw: captured.update(kw) or _FakeRebotArm()
    )

    act = _make_rebot_arm({"channel": "auto"})
    for _ in range(2):
        with pytest.raises(RuntimeError):
            act.connect()
    act.connect()
    assert captured["channel"] == "/dev/ttyACM3"


# ── the plugin keeps retrying ───────────────────────────────────────


class _LateArm:
    """Actuator stub that refuses to connect until the Nth attempt."""

    def __init__(self, succeed_on: int) -> None:
        self.succeed_on = succeed_on
        self.attempts = 0
        self.connected = False

    def connect(self) -> None:
        self.attempts += 1
        if self.attempts < self.succeed_on:
            raise RuntimeError("no serial port matched")
        self.connected = True


def _plugin_with(arm: Any) -> ArmPlugin:
    plugin = ArmPlugin.__new__(ArmPlugin)  # skip setup-heavy __init__ deps
    ArmPlugin.__init__(
        plugin,
        app=object(),
        config={
            "actions_yaml_path": "/dev/null",
            "backend": "rebot_arm",
            "connect_retry_interval_s": 0.01,
        },
    )
    plugin.arm = arm
    return plugin


@pytest.mark.asyncio
async def test_connect_loop_retries_until_the_arm_appears() -> None:
    arm = _LateArm(succeed_on=3)
    plugin = _plugin_with(arm)
    await asyncio.wait_for(plugin._connect_loop(), timeout=5.0)  # noqa: SLF001
    assert arm.connected is True
    assert arm.attempts == 3


@pytest.mark.asyncio
async def test_connect_loop_stops_once_connected() -> None:
    """It must return on success, not keep hammering a live bus."""
    arm = _LateArm(succeed_on=1)
    plugin = _plugin_with(arm)
    await asyncio.wait_for(plugin._connect_loop(), timeout=5.0)  # noqa: SLF001
    await asyncio.sleep(0.05)
    assert arm.attempts == 1


@pytest.mark.asyncio
async def test_stop_cancels_a_pending_connect_loop() -> None:
    """An arm that never appears must not block shutdown."""
    arm = _LateArm(succeed_on=10_000)
    plugin = _plugin_with(arm)
    plugin._connect_task = asyncio.create_task(plugin._connect_loop())  # noqa: SLF001
    await asyncio.sleep(0.05)
    await plugin.stop()
    assert plugin._connect_task is None  # noqa: SLF001
    assert arm.connected is False


# ── the actuator must never publish a half-connected arm ────────────


class _FailingRebotArm(_FakeRebotArm):
    """Constructs fine, then fails the way a half-present bus does."""

    instances: list["_FailingRebotArm"] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.disconnected = False
        _FailingRebotArm.instances.append(self)

    def connect(self, *a: Any, **k: Any) -> None:
        raise RuntimeError("bus went away mid-handshake")

    def disconnect(self) -> None:
        self.disconnected = True


def test_failed_connect_leaves_no_half_built_arm(monkeypatch) -> None:
    """A wrapper that constructed but never enabled must not be published.

    Otherwise every `_robot is None` guard (observation loop, /observation,
    execute_sequence) sees a live-looking arm that was never enabled, and the
    next retry overwrites it without disconnecting.
    """
    _FailingRebotArm.instances.clear()
    monkeypatch.setattr(mod, "resolve_serial_port", lambda *a, **k: "/dev/ttyACM7")
    monkeypatch.setattr(mod, "RebotArm", _FailingRebotArm)

    act = _make_rebot_arm({"channel": "auto"})
    with pytest.raises(RuntimeError):
        act.connect()
    assert act.robot is None
    assert act.torque_enabled is False
    assert _FailingRebotArm.instances[0].disconnected is True


def test_disconnect_during_connect_does_not_energise_the_arm(monkeypatch) -> None:
    """stop() cannot kill the connect worker thread — it must still hand back.

    asyncio task cancellation interrupts the await, not the thread running
    inside asyncio.to_thread. Without a closed flag the abandoned thread runs
    to completion and leaves the motors enabled after shutdown.
    """
    built: list[Any] = []

    class _SlowArm(_FakeRebotArm):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.disconnected = False
            built.append(self)

        def connect(self, *a: Any, **k: Any) -> None:
            # stop() lands here, i.e. after construction but before publish.
            act.disconnect()

        def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(mod, "resolve_serial_port", lambda *a, **k: "/dev/ttyACM7")
    monkeypatch.setattr(mod, "RebotArm", _SlowArm)

    act = _make_rebot_arm({"channel": "auto"})
    with pytest.raises(RuntimeError):
        act.connect()
    assert act.robot is None
    assert act.torque_enabled is False
    assert built[0].disconnected is True


# ── retry cadence validation ────────────────────────────────────────


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", -5.0, 0.0, "", "abc", None])
def test_retry_interval_rejects_unusable_values(bad: Any) -> None:
    """float() takes 'nan'/'inf' and both compare False to <= 0.

    inf sleeps forever (the arm never reconnects), nan makes asyncio.sleep
    raise, and a negative value busy-loops on the serial bus.
    """
    plugin = _plugin_with(_LateArm(succeed_on=1))
    plugin.cfg["connect_retry_interval_s"] = bad
    val = plugin._retry_interval_s()  # noqa: SLF001
    assert val == plugin._CONNECT_RETRY_DEFAULT_S  # noqa: SLF001


def test_retry_interval_accepts_a_positive_finite_override() -> None:
    plugin = _plugin_with(_LateArm(succeed_on=1))
    plugin.cfg["connect_retry_interval_s"] = "2.5"
    assert plugin._retry_interval_s() == 2.5  # noqa: SLF001


def test_app_config_forwards_the_retry_interval() -> None:
    """The knob was documented but the app's plugin_cfg whitelist dropped it."""
    from ovs_agent.apps.voice_rebot_arm import app as arm_app

    src = Path(arm_app.__file__).read_text()
    assert '"connect_retry_interval_s",' in src
