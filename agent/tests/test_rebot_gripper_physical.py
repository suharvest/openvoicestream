"""Gripper physical-feedback truth tests (no hardware / SDK needed).

The control-mode enum is not a reliable record of "is an object held" —
any later gripper command rewrites it. These tests pin the physical-evidence
behaviours added for demo stability:

  * gripper_is_holding() = encoder gap + sustained grip torque (survives a
    misheard close_gripper while carrying);
  * OPENING completes via sustained stall at the physical open limit (the
    -4.9 rad soft-limit target is mechanically unreachable → arrive-tol alone
    never fires);
  * CLOSING enters CONTACT after a grace period even with no start-up travel
    (jaw already resting on the object).
"""

from __future__ import annotations

import threading

from ovs_agent.apps.voice_rebot_arm import rebot_arm as ra
from ovs_agent.apps.voice_rebot_arm.rebot_arm import RebotArm, _GS


def _bare_arm() -> RebotArm:
    """RebotArm with gripper state-machine fields only (no SDK / motors).

    _g_tick and _g_safe_mit tolerate absent hardware (their motor calls are
    wrapped in try/except), so driving the state machine directly is safe.
    """
    arm = RebotArm.__new__(RebotArm)
    arm._g_state = _GS.IDLE
    arm._g_lock = threading.Lock()
    arm._g_pos = 0.0
    arm._g_vel = 0.0
    arm._g_torq = 0.0
    arm._g_pos_start = 0.0
    arm._g_q_contact = 0.0
    arm._g_contact_elapsed = 0.0
    arm._g_open_q_des = ra._G_OPEN_SOFT_LIMIT
    arm._g_open_target = ra._G_OPEN_SOFT_LIMIT
    arm._g_open_stall_s = 0.0
    arm._g_close_elapsed = 0.0
    arm._g_target_force = ra._G_DEFAULT_FORCE
    arm._gripper_mot = None
    arm._gripper_ctrl = None
    return arm


# ── gripper_is_holding: physical evidence ───────────────────────────────


def test_holding_true_in_holding_and_contact_states() -> None:
    arm = _bare_arm()
    for s in (_GS.HOLDING, _GS.CONTACT):
        arm._g_state = s
        assert arm.gripper_is_holding is True


def test_holding_survives_misheard_close_while_carrying() -> None:
    """The 07:53 incident: a misheard close_gripper while carrying flips the
    enum to CLOSING — physically the jaw is still ajar exerting clamp torque,
    so holding must stay True (put_down used to refuse with 'nothing held')."""
    arm = _bare_arm()
    arm._g_state = _GS.CLOSING
    arm._g_pos = -4.3          # jaw around the 0.077m demo box
    arm._g_torq = 1.0          # close torque pushing into the object
    assert arm.gripper_is_holding is True


def test_holding_false_when_parked_empty_even_if_ajar() -> None:
    arm = _bare_arm()
    arm._g_state = _GS.IDLE
    arm._g_pos = -4.6          # parked open
    arm._g_torq = 0.02         # no grip force
    assert arm.gripper_is_holding is False


def test_holding_false_when_jaw_closed() -> None:
    arm = _bare_arm()
    arm._g_state = _GS.IDLE
    arm._g_pos = -0.05         # essentially closed → nothing between fingers
    arm._g_torq = 1.0
    assert arm.gripper_is_holding is False


def test_holding_false_while_opening() -> None:
    arm = _bare_arm()
    arm._g_state = _GS.OPENING
    arm._g_pos = -4.0
    arm._g_torq = 1.0
    assert arm.gripper_is_holding is False


def test_gripper_opening_m_maps_encoder_to_width() -> None:
    arm = _bare_arm()
    arm._g_pos = ra._G_ANGLE_OPEN  # full open angle
    assert abs(arm.gripper_opening_m() - ra._G_MAX_DIST_M) < 1e-9
    arm._g_pos = 0.0
    assert arm.gripper_opening_m() == 0.0


# ── OPENING: stall at the physical limit counts as completion ───────────


def test_opening_completes_via_stall_at_physical_limit() -> None:
    """Target -4.9 rad is past the mechanism's physical limit; the encoder
    parks short of arrive-tol with vel≈0. The sustained stall after the ramp
    must complete the open (IDLE) instead of pushing forever."""
    arm = _bare_arm()
    arm._g_state = _GS.OPENING
    arm._g_open_target = ra._G_OPEN_SOFT_LIMIT          # -4.9
    arm._g_open_q_des = ra._G_OPEN_SOFT_LIMIT           # ramp already done
    arm._g_pos = -4.55                                  # physical limit, > tol away
    arm._g_vel = 0.0
    for _ in range(int(ra._G_OPEN_STALL_S / 0.002) + 5):
        arm._g_tick(0.002)
        if arm._g_state == _GS.IDLE:
            break
    assert arm._g_state == _GS.IDLE
    assert arm._g_open_stall_s == 0.0  # counter reset for the next open


def test_opening_stall_counter_resets_while_still_moving() -> None:
    arm = _bare_arm()
    arm._g_state = _GS.OPENING
    arm._g_open_target = ra._G_OPEN_SOFT_LIMIT
    arm._g_open_q_des = ra._G_OPEN_SOFT_LIMIT
    arm._g_pos = -3.0
    arm._g_vel = -1.0  # still travelling toward open
    for _ in range(50):
        arm._g_tick(0.002)
    assert arm._g_state == _GS.OPENING  # no premature completion


def test_opening_completes_on_arrival_as_before() -> None:
    arm = _bare_arm()
    arm._g_state = _GS.OPENING
    arm._g_open_target = -3.0
    arm._g_open_q_des = -2.9
    arm._g_pos = -3.0  # within arrive tol of target
    arm._g_tick(0.002)
    assert arm._g_state == _GS.IDLE


# ── CLOSING: grace period covers jaw-already-on-object ──────────────────


def test_closing_reaches_contact_without_startup_travel() -> None:
    """Jaw already resting on the object cannot travel _G_STARTUP_DIST; after
    the grace period a stall must still mean contact → CONTACT → HOLDING
    (it used to sit in CLOSING pushing torque forever, with holding=False)."""
    arm = _bare_arm()
    arm._g_state = _GS.CLOSING
    arm._g_pos_start = -4.3
    arm._g_pos = -4.3   # cannot move — object in the way
    arm._g_vel = 0.0
    for _ in range(int(ra._G_CLOSE_GRACE_S / 0.002) + 5):
        arm._g_tick(0.002)
        if arm._g_state != _GS.CLOSING:
            break
    assert arm._g_state == _GS.CONTACT
    # CONTACT debounces into HOLDING shortly after.
    for _ in range(20):
        arm._g_tick(0.002)
        if arm._g_state == _GS.HOLDING:
            break
    assert arm._g_state == _GS.HOLDING


def test_closing_empty_still_idles_at_hard_stop() -> None:
    """Free close (empty jaw): travels past startup dist to the hard stop →
    IDLE, exactly as before the grace-period change."""
    arm = _bare_arm()
    arm._g_state = _GS.CLOSING
    arm._g_pos_start = -1.0
    arm._g_pos = -0.01  # travelled, now at the closed hard stop
    arm._g_vel = 0.0
    arm._g_tick(0.002)
    assert arm._g_state == _GS.IDLE
