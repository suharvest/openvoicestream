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
import threading
from pathlib import Path
from typing import Any, Dict

import pytest

import ovs_agent.apps.voice_rebot_arm.rebot_actuator as mod
from ovs_agent.apps.voice_rebot_arm.rebot_actuator import (
    RebotArmActuator,
    _make_rebot_arm,
)
from ovs_agent.actuators.base import ActuatorClosed
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


def test_second_connect_releases_the_first_arm(monkeypatch) -> None:
    """connect() twice with no disconnect between must not leak the first arm.

    Building a new wrapper over a live one left an energised arm with no handle
    to release it. Not reachable from the retry loop (it returns on success),
    but every direct caller in tools/ can do it.
    """
    built: list[Any] = []

    class _TrackedArm(_FakeRebotArm):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.disconnected = False
            built.append(self)

        def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(mod, "resolve_serial_port", lambda *a, **k: "/dev/ttyACM7")
    monkeypatch.setattr(mod, "RebotArm", _TrackedArm)

    act = _make_rebot_arm({"channel": "auto"})
    act.connect()
    act.connect()
    assert len(built) == 2
    assert built[0].disconnected is True, "first arm was leaked, still energised"
    assert built[1].disconnected is False
    assert act.robot is built[1]


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


# ── the real cross-thread interleaving ──────────────────────────────


def test_disconnect_from_another_thread_mid_connect(monkeypatch) -> None:
    """The A3 race for real: disconnect() on the main thread, connect() on a worker.

    The same-thread test above pins the flag logic but not the interleaving that
    actually happens in production — asyncio.to_thread runs connect() on a
    worker that task cancellation cannot touch, so stop() genuinely races it.
    Park the worker between enable and publish, disconnect from the main thread,
    then release it.
    """
    entered = threading.Event()
    release = threading.Event()
    built: list[Any] = []

    class _ParkedArm(_FakeRebotArm):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.disconnected = False
            built.append(self)

        def connect(self, *a: Any, **k: Any) -> None:
            entered.set()
            assert release.wait(timeout=5.0), "main thread never released us"

        def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(mod, "resolve_serial_port", lambda *a, **k: "/dev/ttyACM7")
    monkeypatch.setattr(mod, "RebotArm", _ParkedArm)

    act = _make_rebot_arm({"channel": "auto"})
    err: list[BaseException] = []

    def _worker() -> None:
        try:
            act.connect()
        except BaseException as e:  # noqa: BLE001 — recorded for the assert
            err.append(e)

    t = threading.Thread(target=_worker, name="connect-worker")
    t.start()
    assert entered.wait(timeout=5.0), "worker never reached connect()"
    act.disconnect()          # the shutdown, from the other thread
    release.set()
    t.join(timeout=5.0)
    assert not t.is_alive()

    assert err and isinstance(err[0], ActuatorClosed), f"got {err!r}"
    assert act.robot is None
    assert act.torque_enabled is False
    assert built[0].disconnected is True, "arm left energised after shutdown"


@pytest.mark.asyncio
async def test_connect_loop_does_not_retry_after_close() -> None:
    """A closed actuator must stop the loop, not look like a transient failure."""

    class _ClosingArm:
        def __init__(self) -> None:
            self.attempts = 0

        def connect(self) -> None:
            self.attempts += 1
            raise ActuatorClosed("closed")

    arm = _ClosingArm()
    plugin = _plugin_with(arm)
    await asyncio.wait_for(plugin._connect_loop(), timeout=5.0)  # noqa: SLF001
    await asyncio.sleep(0.05)
    assert arm.attempts == 1, "retried a shutdown"


def test_disconnect_waits_for_an_in_flight_reader(monkeypatch) -> None:
    """Teardown must not run while a reader holds the bus lock.

    _obs_loop and the /observation thread read through update_cache(), which
    holds _lock for the whole SDK call, so disconnect() has to queue behind it.

    NOTE ON WHAT THIS DOES *NOT* COVER: it does not discriminate the lock-scope
    fix in this commit. The version that set _closed under the lock and then
    dropped it before touching _robot also blocks here, because acquiring the
    lock at all is enough. The interleaving that fix actually addresses needs a
    SECOND reader arriving between disconnect()'s lock release and the teardown,
    which cannot be made deterministic without test hooks in production code.
    Unpublishing under the same lock removes the window by construction; this
    test only guards against a future refactor dropping the lock entirely.
    """
    in_read = threading.Event()
    finish_read = threading.Event()
    disconnected_during_read: list[bool] = []

    class _ParkedReaderArm(_FakeRebotArm):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.disconnected = False

        def get_tcp_pose(self):
            in_read.set()
            assert finish_read.wait(timeout=5.0)
            # If the lock were dropped early, teardown would already have run.
            disconnected_during_read.append(self.disconnected)
            return [0.0] * 6

        def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(mod, "resolve_serial_port", lambda *a, **k: "/dev/ttyACM7")
    monkeypatch.setattr(mod, "RebotArm", _ParkedReaderArm)

    act = _make_rebot_arm({"channel": "auto"})
    act.connect()
    in_read.clear()  # connect() primes the cache once; watch the next read

    reader = threading.Thread(target=act.update_cache, name="obs-reader")
    reader.start()
    assert in_read.wait(timeout=5.0), "reader never entered the SDK call"

    closer = threading.Thread(target=act.disconnect, name="closer")
    closer.start()
    closer.join(timeout=0.3)
    assert closer.is_alive(), "disconnect() tore down while a reader held the lock"

    finish_read.set()
    reader.join(timeout=5.0)
    closer.join(timeout=5.0)
    assert not reader.is_alive() and not closer.is_alive()
    assert disconnected_during_read == [False]
    assert act.robot is None


def test_overlapping_connects_cannot_publish_past_a_shutdown(monkeypatch) -> None:
    """Two connects on ONE actuator must not launder each other's closed flag.

    connect() clears _closed on entry. Without serialization: A enters and
    clears it, disconnect() sets it, B enters and clears it again, and A then
    publishes an arm the operator just released.
    """
    a_entered = threading.Event()
    a_release = threading.Event()
    built: list[Any] = []

    class _StagedArm(_FakeRebotArm):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.disconnected = False
            built.append(self)

        def connect(self, *a: Any, **k: Any) -> None:
            if len(built) == 1:          # only the first one parks
                a_entered.set()
                assert a_release.wait(timeout=5.0)

        def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(mod, "resolve_serial_port", lambda *a, **k: "/dev/ttyACM7")
    monkeypatch.setattr(mod, "RebotArm", _StagedArm)

    act = _make_rebot_arm({"channel": "auto"})
    errs: list[BaseException] = []

    def _run() -> None:
        try:
            act.connect()
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    a = threading.Thread(target=_run, name="connect-A")
    a.start()
    assert a_entered.wait(timeout=5.0)

    act.disconnect()                      # shutdown lands mid-connect
    b = threading.Thread(target=_run, name="connect-B")
    b.start()
    b.join(timeout=0.3)
    assert b.is_alive(), "B started while A held the connect lock"

    a_release.set()
    a.join(timeout=5.0)
    b.join(timeout=5.0)
    assert not a.is_alive() and not b.is_alive()

    # A must have been refused; B is a fresh, legitimate reconnect.
    assert errs and isinstance(errs[0], ActuatorClosed), f"got {errs!r}"
    assert built[0].disconnected is True, "A's arm left energised past shutdown"


# ── the plugin's own start() wiring ─────────────────────────────────


@pytest.mark.asyncio
async def test_start_does_not_block_on_a_missing_arm() -> None:
    """start() must schedule the connect, not await it.

    The original code did `await asyncio.to_thread(self.arm.connect)` inline,
    which is what made a missing arm permanent. If that ever comes back, an arm
    that never appears hangs agent startup instead of degrading to cache-only.
    """
    release = threading.Event()

    class _HangingArm:
        def __init__(self) -> None:
            self.connect_calls = 0
            self.cache_calls = 0

        def connect(self) -> None:
            self.connect_calls += 1
            release.wait(timeout=5.0)
            raise RuntimeError("no serial port matched")

        def update_cache(self) -> None:
            self.cache_calls += 1

        def disconnect(self) -> None: ...

    arm = _HangingArm()
    plugin = _plugin_with(arm)
    plugin.actions = None
    import ovs_agent.plugins.actuator_observation_server as obs_srv

    orig = obs_srv.start_observation_server
    obs_srv.start_observation_server = lambda *a, **k: None
    try:
        await asyncio.wait_for(plugin.start(), timeout=1.0)
        assert plugin._connect_task is not None   # noqa: SLF001
        assert plugin._obs_task is not None       # noqa: SLF001
        # The observation loop runs while the connect is still in flight —
        # that is the concurrency the actuator has to tolerate.
        await asyncio.sleep(0.15)
        assert arm.cache_calls > 0
        assert arm.connect_calls == 1
    finally:
        release.set()
        obs_srv.start_observation_server = orig
        await plugin.stop()
    assert plugin._connect_task is None           # noqa: SLF001
    assert plugin._obs_task is None               # noqa: SLF001
