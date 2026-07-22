"""Adaptive grasp-force ramp (RebotArm.grasp adaptive=True).

SDK-free: RebotArm is instantiated via __new__ with only the gripper-state
fields seeded, and a helper thread plays the 500Hz loop's role (flipping
CLOSING→HOLDING and simulating encoder creep). Asserts:
  * non-adaptive path is unchanged (target force = clip(force)),
  * adaptive starts LOW and ramps only while the gap creeps,
  * rigid object (no creep) settles at the start force,
  * soft object (creeps until a force level) ramps to that level, re-anchoring
    the hold angle, and never exceeds the cap,
  * grasp_service passes adaptive=True only when asked, with TypeError
    fallback for arms lacking the kwarg.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ovs_agent.apps.voice_rebot_arm import rebot_arm as ra


def _bare_arm() -> ra.RebotArm:
    arm = ra.RebotArm.__new__(ra.RebotArm)
    arm._gripper_mot = object()           # non-None → gripper present
    arm._g_lock = threading.Lock()
    arm._g_state = ra._GS.IDLE
    arm._g_pos = -3.0                     # partially open
    arm._g_vel = 0.0
    arm._g_torq = 0.0
    arm._g_pos_start = -3.0
    arm._g_close_elapsed = 0.0
    arm._g_q_contact = -3.0
    arm._g_target_force = ra._G_DEFAULT_FORCE
    arm._g_open_last_target = None
    return arm


def _holding_after(arm: ra.RebotArm, delay_s: float = 0.05) -> threading.Thread:
    """Background: flip CLOSING→HOLDING after a short delay (stand-in for the
    500Hz loop's contact detection)."""

    def _run() -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with arm._g_lock:
                if arm._g_state == ra._GS.CLOSING:
                    break
            time.sleep(0.005)
        time.sleep(delay_s)
        with arm._g_lock:
            arm._g_state = ra._GS.HOLDING

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def test_non_adaptive_path_unchanged():
    arm = _bare_arm()
    _holding_after(arm)
    ok = arm.grasp(force=0.65, timeout=2.0)
    assert ok is True
    assert arm._g_target_force == 0.65


def test_adaptive_rigid_object_settles_at_start_force():
    arm = _bare_arm()
    _holding_after(arm)
    # encoder never creeps (rigid box) → one window, stop at start force.
    ok = arm.grasp(force=0.8, timeout=2.0, adaptive=True,
                   adaptive_start=0.2, adaptive_step=0.1,
                   adaptive_window_s=0.05, adaptive_creep_rad=0.04)
    assert ok is True
    assert arm._g_target_force == 0.2


def test_adaptive_soft_object_ramps_until_stable_and_reanchors():
    arm = _bare_arm()
    _holding_after(arm)

    # Simulate a soft object: while hold force < 0.4, the gap keeps creeping
    # shut (pos moves toward 0); at >= 0.4 it stabilises.
    stop = threading.Event()

    def _creep() -> None:
        while not stop.is_set():
            with arm._g_lock:
                holding = arm._g_state == ra._GS.HOLDING
                tf = arm._g_target_force
            if holding and tf < 0.4:
                arm._g_pos = min(0.0, arm._g_pos + 0.06)  # creep shut
            time.sleep(0.01)

    t = threading.Thread(target=_creep, daemon=True)
    t.start()
    try:
        ok = arm.grasp(force=0.8, timeout=3.0, adaptive=True,
                       adaptive_start=0.2, adaptive_step=0.1,
                       adaptive_window_s=0.05, adaptive_creep_rad=0.04)
    finally:
        stop.set()
    assert ok is True
    # ramped 0.2 → 0.3 → 0.4, then stable; never hit the 0.8 cap.
    assert 0.4 <= arm._g_target_force < 0.8
    # hold angle re-anchored at the compressed position.
    assert arm._g_q_contact > -3.0


def test_adaptive_respects_cap():
    arm = _bare_arm()
    _holding_after(arm)
    stop = threading.Event()

    def _always_creep() -> None:
        while not stop.is_set():
            with arm._g_lock:
                holding = arm._g_state == ra._GS.HOLDING
            if holding:
                arm._g_pos = min(0.0, arm._g_pos + 0.06)
            time.sleep(0.01)

    t = threading.Thread(target=_always_creep, daemon=True)
    t.start()
    try:
        ok = arm.grasp(force=0.5, timeout=3.0, adaptive=True,
                       adaptive_start=0.2, adaptive_step=0.1,
                       adaptive_window_s=0.05, adaptive_creep_rad=0.04)
    finally:
        stop.set()
    assert ok is True
    assert arm._g_target_force == 0.5   # capped, never beyond


def test_grasp_service_typeerror_fallback_for_stub_arms():
    # An arm whose grasp() lacks the adaptive kwarg must still work when the
    # pipeline asks for adaptive force.
    from ovs_agent.apps.voice_rebot_arm.grasp_service import run_grasp_once
    import tests.test_grasp_service as tgs

    color, depth, K = tgs._scene()
    arm = tgs.FakeArm()  # grasp(self, force, timeout) — no adaptive kwarg
    cam = tgs.FakeCamera(color, depth, K)
    seg = tgs.FakeSegmenter(tgs._make_result())
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
        adaptive_force=True,
    )
    assert res["success"] is True
    assert res["adaptive_force"] is True


# ── release-verification false positive at full open (2026-06-12) ──────────
def test_holding_false_after_intentional_full_open_with_limit_shortfall():
    """Real-machine repro: open to 0.09 commanded (target -4.9), jaw parks at
    the soft-limit shortfall (-4.64 ≈ 0.0853m) with residual limit torque —
    must NOT report holding (it false-positived put_down's release verify)."""
    arm = _bare_arm()
    arm._g_state = ra._GS.IDLE
    arm._g_open_last_target = -4.9
    arm._g_pos = -4.64          # 0.26 rad shortfall — normal at full open
    arm._g_torq = 0.4           # residual limit torque
    assert arm.gripper_is_holding is False


def test_holding_true_when_open_blocked_by_object():
    """Open commanded full but the jaw stopped 1.2 rad short with torque —
    an object is genuinely still between the fingers."""
    arm = _bare_arm()
    arm._g_state = ra._GS.IDLE
    arm._g_open_last_target = -4.9
    arm._g_pos = -3.7           # blocked well short of target
    arm._g_torq = 0.4
    assert arm.gripper_is_holding is True


def test_holding_unaffected_without_open_intent():
    """No prior open intent (e.g. after a grasp): legacy gap+torque rule."""
    arm = _bare_arm()
    arm._g_state = ra._GS.IDLE
    arm._g_open_last_target = None
    arm._g_pos = -4.2
    arm._g_torq = 0.4
    assert arm.gripper_is_holding is True
    arm._g_torq = 0.01
    assert arm.gripper_is_holding is False
