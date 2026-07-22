"""Unit tests for RebotArmActuator (reBot B601-DM, Phase A).

Runs without the reBotArm_control_py SDK / motorbridge — we inject a
``_FakeRebotArm`` in place of the real RebotArm so frame→waypoint
translation, cancellation, the torque gate and factory registration can be
exercised on a developer Mac. Style mirrors test_actuator_actions.py.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Tuple

import pytest

from ovs_agent.actuators.factory import create_actuator
from ovs_agent.apps.voice_rebot_arm.rebot_actuator import (
    RebotArmActuator,
    _make_rebot_arm,
)


# ── fakes ───────────────────────────────────────────────────────────


class _FakeRebotArm:
    """Records move_to / gripper calls; no real bus."""

    def __init__(self) -> None:
        self.moves: List[Dict[str, float]] = []
        self.gripper_calls: List[str] = []
        self.gripper_args: List[Tuple[str, float]] = []  # (call, magnitude)
        self.connected = False
        self.gripper_inited = False
        self.disconnected = False
        self._arm = _FakeLowLevelArm()
        self._holding = False

    # lifecycle
    def connect(self, enable: bool = True) -> None:
        self.connected = True

    def init_gripper(self, cfg_path=None) -> None:
        self.gripper_inited = True

    def disconnect(self) -> None:
        self.disconnected = True
        self.connected = False

    # motion
    def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
        self.moves.append(
            {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch,
             "yaw": yaw, "duration": duration}
        )
        return True

    def safe_home(self, duration: float = 3.0) -> None:
        self.gripper_calls.append("safe_home")

    # gripper
    def open_gripper(self, distance_m: float = 0.09) -> None:
        self.gripper_calls.append("open")
        self.gripper_args.append(("open", float(distance_m)))

    def close_gripper(self) -> None:
        self.gripper_calls.append("close")
        self.gripper_args.append(("close", 0.0))

    def grasp(self, force=None, timeout: float = 5.0) -> bool:
        self.gripper_calls.append("grasp")
        self.gripper_args.append(("grasp", float(force) if force is not None else 0.0))
        self._holding = True
        return True

    def release_gripper(self, timeout: float = 4.0) -> None:
        self.gripper_calls.append("release")

    @property
    def gripper_is_holding(self) -> bool:
        return self._holding

    def get_gripper_state(self) -> Tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    # observation
    def get_tcp_pose(self):
        import numpy as np
        T = np.eye(4, dtype=float)
        T[0, 3] = 0.30
        T[1, 3] = 0.00
        T[2, 3] = 0.30
        return T


class _FakeLowLevelArm:
    def __init__(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False


def _make_connected_actuator(**overrides) -> Tuple[RebotArmActuator, _FakeRebotArm]:
    """Build an actuator with a fake arm already injected + torque on."""
    act = RebotArmActuator(channel="/dev/ttyACM1", **overrides)
    fake = _FakeRebotArm()
    act._robot = fake  # noqa: SLF001
    act._torque_state = "on"  # noqa: SLF001
    return act, fake


def _wp(delay: float = 0.05, **fields: float) -> Dict[str, Any]:
    """Build a {joints: {...}, delay} cartesian-waypoint frame."""
    joints: Dict[str, float] = {
        "x": 0.30, "y": 0.00, "z": 0.30,
        "roll": 0.0, "pitch": 0.7, "yaw": 0.0, "gripper": 0.0,
    }
    joints.update({k: float(v) for k, v in fields.items()})
    return {"joints": joints, "delay": delay}


# ── import / construction smoke ─────────────────────────────────────


def test_import_without_sdk() -> None:
    """The module imports + the actuator constructs on a Mac with no SDK."""
    act = RebotArmActuator(channel="/dev/ttyACM1")
    assert act.observation_features().keys() >= {
        "x", "y", "z", "roll", "pitch", "yaw", "gripper"
    }
    # Not connected → torque off, empty obs.
    assert act.torque_enabled is False
    assert act.get_cached_observation() == {}


# ── frame → waypoint translation ────────────────────────────────────


def test_execute_sequence_translates_pose_to_move_to() -> None:
    act, fake = _make_connected_actuator()
    ok = act.execute_sequence([
        _wp(x=0.30, y=0.10, z=0.35),
        _wp(x=0.30, y=-0.10, z=0.35),
    ])
    assert ok is True
    assert len(fake.moves) == 2
    assert fake.moves[0]["y"] == pytest.approx(0.10)
    assert fake.moves[1]["y"] == pytest.approx(-0.10)


def test_execute_sequence_uses_frame_duration_override() -> None:
    act, fake = _make_connected_actuator(move_duration=2.0)
    joints = _wp()["joints"]
    joints["duration"] = 1.25
    act.execute_sequence([{"joints": joints, "delay": 0.05}])
    assert fake.moves[0]["duration"] == pytest.approx(1.25)


def test_gripper_signed_magnitude_open_and_grasp() -> None:
    # +gripper → open to that width (m); -gripper → grasp with |g| force (N·m).
    act, fake = _make_connected_actuator()
    act.execute_sequence([_wp(gripper=0.06)])   # open 6 cm
    act.execute_sequence([_wp(gripper=-0.2)])   # grasp 0.2 N·m
    assert fake.gripper_args == [("open", 0.06), ("grasp", 0.2)]


def test_gripper_open_width_clamped_to_max() -> None:
    # open_distance_m is the max-open clamp; a wider request is clamped.
    act, fake = _make_connected_actuator(open_distance_m=0.09)
    act.execute_sequence([_wp(gripper=0.5)])  # request 0.5 m → clamp to 0.09
    assert fake.gripper_args == [("open", 0.09)]


def test_gripper_grasp_force_clamped_when_configured() -> None:
    # grasp_force is the max-force clamp; a larger |g| is clamped down.
    act, fake = _make_connected_actuator(grasp_force=0.3)
    act.execute_sequence([_wp(gripper=-1.5)])  # request 1.5 → clamp to 0.3
    assert fake.gripper_args == [("grasp", 0.3)]


def test_gripper_zero_holds() -> None:
    act, fake = _make_connected_actuator()
    act.execute_sequence([_wp(gripper=0.0)])
    assert fake.gripper_calls == []


def test_pose_only_frame_skips_gripper_and_gripper_only_skips_move() -> None:
    act, fake = _make_connected_actuator()
    # Gripper-only frame: no x/y/z/roll/pitch/yaw keys → no move_to.
    act.execute_sequence([{"joints": {"gripper": 1.0}, "delay": 0.05}])
    assert fake.moves == []
    assert fake.gripper_calls == ["open"]


# ── cancellation ────────────────────────────────────────────────────


def test_cancel_event_stops_sequence_midway() -> None:
    act, fake = _make_connected_actuator()
    cancel = threading.Event()

    frames = [_wp(y=0.1, delay=0.5), _wp(y=-0.1, delay=0.5), _wp(y=0.0, delay=0.5)]

    # Fire cancel shortly after start so only the first frame's move runs.
    def _trip() -> None:
        time.sleep(0.1)
        cancel.set()

    t = threading.Thread(target=_trip)
    t.start()
    ok = act.execute_sequence(frames, cancel_event=cancel)
    t.join()

    assert ok is True
    # First frame's move_to runs, then the long settle is interrupted, and
    # the loop breaks before frame 2's move.
    assert len(fake.moves) == 1


def test_cancel_before_any_frame_sends_nothing() -> None:
    act, fake = _make_connected_actuator()
    cancel = threading.Event()
    cancel.set()
    ok = act.execute_sequence([_wp(), _wp()], cancel_event=cancel)
    assert ok is True
    assert fake.moves == []


# ── torque gate ─────────────────────────────────────────────────────


def test_execute_sequence_refused_when_torque_off() -> None:
    act, fake = _make_connected_actuator()
    act._torque_state = "off"  # noqa: SLF001
    ok = act.execute_sequence([_wp(y=0.1)])
    assert ok is False
    assert fake.moves == []


def test_set_torque_toggles_state_and_low_level_arm() -> None:
    act, fake = _make_connected_actuator()
    act.set_torque(False)
    assert act.torque_enabled is False
    assert fake._arm.enabled is False  # noqa: SLF001
    act.set_torque(True)
    assert act.torque_enabled is True
    assert fake._arm.enabled is True  # noqa: SLF001


def test_set_torque_raises_when_not_connected() -> None:
    act = RebotArmActuator(channel="/dev/ttyACM1")
    with pytest.raises(RuntimeError):
        act.set_torque(True)


def test_torque_off_midsequence_aborts(monkeypatch) -> None:
    # Emergency disable mid-sequence: after the first frame's move we flip
    # torque off; the loop must abort before the next move and return False.
    act, fake = _make_connected_actuator()

    orig_move = fake.move_to

    def _move(*a, **kw):
        # After the first move, simulate an emergency set_torque(False).
        act._torque_state = "off"  # noqa: SLF001
        return orig_move(*a, **kw)

    monkeypatch.setattr(fake, "move_to", _move)

    ok = act.execute_sequence([_wp(y=0.1), _wp(y=-0.1), _wp(y=0.0)])
    assert ok is False
    # Only the first frame's move ran; the abort fired before frame 2.
    assert len(fake.moves) == 1


def test_set_torque_off_not_blocked_during_settle_sleep() -> None:
    # The lock must NOT be held across the settle sleep, so an emergency
    # set_torque(False) from another thread can proceed promptly while a
    # long-delay sequence is mid-settle.
    act, fake = _make_connected_actuator()
    entered = threading.Event()
    torque_done = threading.Event()

    orig_move = fake.move_to

    def _move(*a, **kw):
        entered.set()
        return orig_move(*a, **kw)

    fake.move_to = _move  # type: ignore[assignment]

    def _emergency() -> None:
        entered.wait(2.0)
        # If execute_sequence held the lock across the sleep, this set_torque
        # (which also takes self._lock) would block until the 2s sleep ends.
        t0 = time.monotonic()
        act.set_torque(False)
        torque_done.set()
        assert time.monotonic() - t0 < 1.0, "set_torque blocked on the sleep lock"

    t = threading.Thread(target=_emergency)
    t.start()
    # One frame with a long settle delay; set_torque off should land during it.
    act.execute_sequence([_wp(y=0.1, delay=2.0)])
    t.join()
    assert torque_done.is_set()


def test_move_failure_returns_false_and_stops_sequence() -> None:
    # A move_to that raises must abort the sequence and return False (not
    # silently swallow the error and report success).
    act, fake = _make_connected_actuator()

    def _boom(*a, **kw):
        raise RuntimeError("bus error")

    fake.move_to = _boom  # type: ignore[assignment]
    ok = act.execute_sequence([_wp(y=0.1), _wp(y=-0.1)])
    assert ok is False
    # Second frame never ran.
    assert fake.gripper_calls == []


def test_gripper_failure_returns_false_and_stops_sequence() -> None:
    act, fake = _make_connected_actuator()

    def _boom(*a, **kw):
        raise RuntimeError("gripper bus error")

    fake.grasp = _boom  # type: ignore[assignment]
    # gripper-only frames (no pose) so move never short-circuits first.
    ok = act.execute_sequence([
        {"joints": {"gripper": -0.2}, "delay": 0.01},
        {"joints": {"gripper": 0.05}, "delay": 0.01},
    ])
    assert ok is False
    # open (second frame) never ran because the grasp failure aborted.
    assert fake.gripper_calls == []


def test_acquire_motion_lock_is_same_lock() -> None:
    # The public motion-lock contextmanager grabs the SAME actuator lock so
    # the grasp pipeline stays mutually exclusive with execute_sequence.
    act, _fake = _make_connected_actuator()
    with act.acquire_motion_lock():
        # RLock is reentrant from this thread; from another thread it would
        # block. Assert acquire() fails non-blocking from a different thread.
        held = []

        def _try() -> None:
            held.append(act._lock.acquire(blocking=False))  # noqa: SLF001
            if held[-1]:
                act._lock.release()  # noqa: SLF001

        t = threading.Thread(target=_try)
        t.start()
        t.join()
        assert held == [False]


# ── amplitude-clamp validation (item 11) ────────────────────────────


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_open_distance_rejected(bad) -> None:
    with pytest.raises(ValueError):
        RebotArmActuator(channel="/dev/ttyACM1", open_distance_m=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.5])
def test_invalid_grasp_force_rejected(bad) -> None:
    with pytest.raises(ValueError):
        RebotArmActuator(channel="/dev/ttyACM1", grasp_force=bad)


@pytest.mark.parametrize("bad", [float("nan"), -2.0])
def test_invalid_move_duration_rejected(bad) -> None:
    with pytest.raises(ValueError):
        RebotArmActuator(channel="/dev/ttyACM1", move_duration=bad)


def test_grasp_force_none_allowed() -> None:
    # None is the "rely on SDK clamp" sentinel and must pass validation.
    act = RebotArmActuator(channel="/dev/ttyACM1", grasp_force=None)
    assert act._grasp_force is None  # noqa: SLF001


# ── RebotArm temp-cfg cleanup (item 5) — SDK-free ───────────────────


def test_rebotarm_cleanup_unlinks_temp_cfgs(tmp_path) -> None:
    # _cleanup_tmp_cfgs (called by disconnect() and on constructor failure)
    # must unlink every channel-override temp file and tolerate missing ones.
    from ovs_agent.apps.voice_rebot_arm.rebot_arm import RebotArm

    # Build a bare RebotArm without touching the SDK constructor.
    arm = RebotArm.__new__(RebotArm)
    f1 = tmp_path / "rebot_chan_a.yaml"
    f2 = tmp_path / "rebot_chan_b.yaml"
    f1.write_text("channel: /dev/ttyACM1\n")
    f2.write_text("channel: /dev/ttyACM1\n")
    arm._tmp_cfg_paths = [str(f1), str(f2), str(tmp_path / "already_gone.yaml")]  # noqa: SLF001

    arm._cleanup_tmp_cfgs()  # noqa: SLF001

    assert not f1.exists()
    assert not f2.exists()
    # Cleared, and idempotent on a second call.
    assert arm._tmp_cfg_paths == []  # noqa: SLF001
    arm._cleanup_tmp_cfgs()  # noqa: SLF001


def test_write_channel_override_creates_and_can_be_cleaned(tmp_path) -> None:
    # The override writer produces a real temp file with the channel patched;
    # cleanup removes it.
    from ovs_agent.apps.voice_rebot_arm.rebot_arm import (
        RebotArm,
        _write_channel_override_yaml,
    )

    src = tmp_path / "arm.yaml"
    src.write_text("channel: /dev/ttyACM0\nfoo: 1\n")
    tmp = _write_channel_override_yaml(str(src), "/dev/ttyACM1")
    import os as _os
    import yaml as _yaml

    assert _os.path.exists(tmp)
    with open(tmp) as f:
        data = _yaml.safe_load(f)
    assert data["channel"] == "/dev/ttyACM1"
    assert data["foo"] == 1  # other fields preserved

    arm = RebotArm.__new__(RebotArm)
    arm._tmp_cfg_paths = [tmp]  # noqa: SLF001
    arm._cleanup_tmp_cfgs()  # noqa: SLF001
    assert not _os.path.exists(tmp)


# ── RebotArm connect/disconnect safety (item 1) — SDK-free ──────────


class _FakeBareArm:
    def __init__(self) -> None:
        self.enabled = False
        self.disabled = False
        self.connected = False
        self.disconnected = False

    def connect(self) -> None:
        self.connected = True

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.disabled = True
        self.enabled = False

    def disconnect(self) -> None:
        self.disconnected = True


def _bare_rebotarm():
    """A RebotArm with internals stubbed enough for connect/disconnect, no SDK."""
    from ovs_agent.apps.voice_rebot_arm.rebot_arm import RebotArm

    arm = RebotArm.__new__(RebotArm)
    arm._arm = _FakeBareArm()  # noqa: SLF001
    arm._endpos_ctrl = None  # noqa: SLF001
    arm._connected = False  # noqa: SLF001
    arm._tmp_cfg_paths = []  # noqa: SLF001
    arm._gripper_mot = None  # noqa: SLF001
    arm._g_loop_running = False  # noqa: SLF001
    arm._g_loop_thread = None  # noqa: SLF001
    arm._g_lock = threading.Lock()  # noqa: SLF001
    arm._g_state = 0  # noqa: SLF001 (_GS.IDLE)
    return arm


def test_connect_disables_arm_if_post_enable_step_fails() -> None:
    # connect(enable=True): if ArmEndPos.start() raises AFTER enable(), the arm
    # must be best-effort disabled (no orphaned energised motors) and re-raise.
    arm = _bare_rebotarm()

    class _BoomEndPos:
        def __init__(self, _arm) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("controller start failed")

    arm._ArmEndPos = _BoomEndPos  # noqa: SLF001
    with pytest.raises(RuntimeError):
        arm.connect(enable=True)
    assert arm._arm.enabled is False  # noqa: SLF001
    assert arm._arm.disabled is True  # noqa: SLF001
    assert arm._connected is False  # noqa: SLF001


def test_disconnect_explicitly_disables_and_cleans(tmp_path) -> None:
    arm = _bare_rebotarm()
    tmpf = tmp_path / "rebot_chan_x.yaml"
    tmpf.write_text("channel: /dev/ttyACM1\n")
    arm._tmp_cfg_paths = [str(tmpf)]  # noqa: SLF001
    arm.disconnect()
    assert arm._arm.disabled is True  # noqa: SLF001 — explicit disable ran
    assert arm._arm.disconnected is True  # noqa: SLF001
    assert not tmpf.exists()  # temp cfg cleaned
    assert arm._connected is False  # noqa: SLF001


# ── RebotArm gripper stop / zero safety (items 4, 6) — SDK-free ─────


class _FakeGripperMot:
    def __init__(self) -> None:
        self.zero_calls = 0
        self.zero_raises: Exception | None = None

    def send_mit(self, *a, **kw) -> None:
        pass

    def request_feedback(self) -> None:
        pass

    def set_zero_position(self) -> None:
        self.zero_calls += 1
        if self.zero_raises is not None:
            raise self.zero_raises


def _bare_gripper_arm():
    arm = _bare_rebotarm()
    arm._gripper_mot = _FakeGripperMot()  # noqa: SLF001
    arm._gripper_ctrl = type("C", (), {"poll_feedback_once": lambda self: None})()  # noqa: SLF001
    arm._g_pos = 0.0  # noqa: SLF001
    return arm


def test_g_stop_loop_skips_softstop_when_thread_wont_die() -> None:
    # If the control thread does not stop within the join timeout, _g_stop_loop
    # must NOT send a soft-stop frame (would race the live thread) and must mark
    # the gripper unavailable.
    arm = _bare_gripper_arm()

    never_dies = threading.Event()

    def _spin() -> None:
        never_dies.wait(2.0)  # ignores the stop flag long enough to time out

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    arm._g_loop_thread = t  # noqa: SLF001
    arm._g_loop_running = True  # noqa: SLF001
    arm._g_loop_stop = threading.Event()  # noqa: SLF001

    mot = arm._gripper_mot  # noqa: SLF001
    sent = []
    mot.send_mit = lambda *a, **kw: sent.append(a)  # type: ignore[assignment]

    arm._g_stop_loop()  # noqa: SLF001 — joins 1.0s, thread still alive

    never_dies.set()
    t.join()
    assert sent == []                       # no soft-stop frame sent
    assert arm._gripper_mot is None         # noqa: SLF001 — marked unavailable
    assert arm._g_loop_running is False     # noqa: SLF001


def test_set_gripper_zero_restarts_loop_even_on_unexpected_error(monkeypatch) -> None:
    # set_zero_position raising a NON-CallError must still restart the control
    # loop via the finally block (no permanently-stopped gripper).
    arm = _bare_gripper_arm()
    arm._g_loop_stop = threading.Event()  # noqa: SLF001

    # Stub motorbridge.CallError import + the loop start/stop to be observable.
    import sys
    import types

    fake_mb = types.ModuleType("motorbridge")
    fake_mb.CallError = type("CallError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "motorbridge", fake_mb)

    started = []
    monkeypatch.setattr(arm, "_g_stop_loop", lambda: None)
    monkeypatch.setattr(arm, "_g_start_loop", lambda: started.append(True))

    arm._gripper_mot.zero_raises = RuntimeError("unexpected bus error")  # noqa: SLF001
    with pytest.raises(RuntimeError):
        arm.set_gripper_zero()
    assert started == [True]  # loop restarted despite the raise


# ── empty / disconnected ────────────────────────────────────────────


def test_execute_sequence_empty_returns_false() -> None:
    act, _fake = _make_connected_actuator()
    assert act.execute_sequence([]) is False


def test_execute_sequence_not_connected_returns_false() -> None:
    act = RebotArmActuator(channel="/dev/ttyACM1")
    assert act.execute_sequence([_wp()]) is False


def test_disconnect_best_effort() -> None:
    act, fake = _make_connected_actuator()
    act.disconnect()
    assert fake.disconnected is True
    assert act._robot is None  # noqa: SLF001
    assert act.torque_enabled is False
    # Idempotent: a second disconnect on an already-cleared actuator is a no-op.
    act.disconnect()


# ── observation cache ───────────────────────────────────────────────


def test_update_cache_reads_tcp_pose() -> None:
    act, _fake = _make_connected_actuator()
    obs = act.update_cache()
    assert obs["x"] == pytest.approx(0.30)
    assert obs["z"] == pytest.approx(0.30)
    assert "gripper" in obs
    # Cached read returns the same without touching the bus.
    assert act.get_cached_observation()["x"] == pytest.approx(0.30)


# ── factory registration ────────────────────────────────────────────


def test_registered_in_factory() -> None:
    # Importing rebot_actuator (done at module top) self-registers the
    # builder. create_actuator must resolve it.
    act = create_actuator("rebot_arm", {"channel": "/dev/ttyACM1"})
    assert isinstance(act, RebotArmActuator)


def test_builder_requires_channel() -> None:
    with pytest.raises(ValueError):
        _make_rebot_arm({})


def test_builder_threads_config_through() -> None:
    act = _make_rebot_arm({
        "channel": "/dev/ttyACM1",
        "repo_root": "/opt/rebot",
        "move_duration": 1.5,
        "grasp_force": 0.4,
        "open_distance_m": 0.05,
    })
    assert act._channel == "/dev/ttyACM1"  # noqa: SLF001
    assert act._repo_root == "/opt/rebot"  # noqa: SLF001


def test_builder_tolerates_empty_string_env_substitution() -> None:
    # "${VAR:-}" with VAR unset substitutes to "" — the builder must treat
    # empty/whitespace as "unset" for every optional field (the deploy bug:
    # grasp_force="" → float("") ValueError disabled the whole arm).
    act = _make_rebot_arm({
        "channel": "/dev/ttyACM1",
        "repo_root": "",
        "config_path": "  ",
        "urdf_path": "",
        "gripper_cfg_path": "",
        "move_duration": "",
        "grasp_force": "",
        "open_distance_m": "",
    })
    assert act._grasp_force is None       # noqa: SLF001
    assert act._repo_root is None         # noqa: SLF001
    assert act._config_path is None       # noqa: SLF001
    assert act._move_duration == 2.0      # noqa: SLF001  (default)
    assert act._open_distance_m == 0.09   # noqa: SLF001  (default)


# ── channel actually reaches the SDK (regression guard) ─────────────


def test_connect_passes_channel_to_rebotarm(monkeypatch) -> None:
    """connect() MUST forward channel to RebotArm, else the SDK falls back to
    its arm.yaml default /dev/ttyACM0 — the SO-ARM's port — and we would drive
    the wrong arm. This is the safety regression that motivated the fix."""
    import ovs_agent.apps.voice_rebot_arm.rebot_actuator as mod

    captured: Dict[str, Any] = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return _FakeRebotArm()

    monkeypatch.setattr(mod, "RebotArm", _spy)

    act = RebotArmActuator(channel="/dev/ttyACM1", repo_root="/opt/rebot")
    act.connect()

    assert captured.get("channel") == "/dev/ttyACM1", (
        f"connect() did not forward channel to RebotArm; got {captured!r}"
    )
    assert captured.get("repo_root") == "/opt/rebot"


def test_set_torque_delegates_to_robot_set_torque_when_present() -> None:
    """When the RebotArm wrapper exposes set_torque (restarts the ArmEndPos
    controller, not just _arm.enable), the actuator must PREFER it — a bare
    _arm.enable() leaves the controller stale → 'torque on but motors dead'."""
    act, fake = _make_connected_actuator()
    calls: list[bool] = []
    fake.set_torque = lambda enable: calls.append(enable)  # type: ignore[attr-defined]
    act.set_torque(False)
    assert calls == [False] and act.torque_enabled is False
    act.set_torque(True)
    assert calls == [False, True] and act.torque_enabled is True
    # delegated → the low-level _arm.enable/disable was NOT used as the path
    assert fake._arm.enabled is False  # untouched by the delegated path  # noqa: SLF001


def test_set_torque_falls_back_when_robot_set_torque_raises() -> None:
    """If the wrapper's set_torque fails, fall back to low-level enable/disable
    so torque control still works (degraded) instead of throwing."""
    act, fake = _make_connected_actuator()

    def _boom(enable: bool) -> None:
        raise RuntimeError("controller restart failed")

    fake.set_torque = _boom  # type: ignore[attr-defined]
    act.set_torque(True)  # must not raise
    assert act.torque_enabled is True
    assert fake._arm.enabled is True  # fell back to low-level enable  # noqa: SLF001
