"""Tests for the torch-free grasp pipeline (grasp_service.run_grasp_once).

All dependencies are mocked — no torch, no real camera/SDK, no onnxruntime.
We build a fake YoloResult by hand and a recording fake arm to assert:
  * full pipeline ordering (open → pregrasp → grasp_move → grasp → lift)
  * cancel_event mid-pipeline stops at the current stage and SAFE-PARKS the
    gripper (open_gripper called, no clamp left)
  * target filtering keeps only the requested class
  * no-detection / no-hand-eye early returns
"""

from __future__ import annotations

import contextlib
import threading
import time

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.grasp_service import (
    run_grasp_once,
    run_put_down_once,
)
from ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx import (
    YoloResult,
    _Box,
    _Boxes,
    _Masks,
)


# ── fakes ─────────────────────────────────────────────────────────────────
class FakeArm:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._holding = True

    def open_gripper(self, distance_m: float = 0.09) -> None:
        self.calls.append(("open_gripper", float(distance_m)))
        # Physical behaviour: an open jaw is no longer gripping.
        self._holding = False

    def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
        self.calls.append(("move_to", round(float(z), 4)))
        return True

    def wait_motion(self, duration: float, extra: float = 0.6) -> None:
        self.calls.append(("wait_motion",))

    def get_tcp_pose(self) -> np.ndarray:
        self.calls.append(("get_tcp_pose",))
        return np.eye(4, dtype=np.float64)

    def grasp(self, force=None, timeout: float = 5.0) -> bool:
        self.calls.append(("grasp", force))
        # Physical behaviour: a successful compliant grasp is holding the
        # object (mirrors the real arm's HOLDING state → encoder gap + torque).
        self._holding = True
        return True

    def release_gripper(self, timeout: float = 4.0) -> None:
        self.calls.append(("release_gripper",))

    def gripper_is_holding(self) -> bool:
        return self._holding

    def names_of_calls(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeCamera:
    def __init__(self, color, depth, K) -> None:
        self._color = color
        self._depth = depth
        self.K = K
        self.warmed = 0

    def warm_up(self, n: int) -> None:
        self.warmed += n

    def get_frame(self):
        return self._color, self._depth


class FakeSegmenter:
    """Returns a fixed YoloResult; records predict() conf."""

    def __init__(self, result) -> None:
        self._result = result
        self.predict_calls = 0
        self.last_only_names = None

    def predict(self, image_bgr, conf=0.25, iou=0.45, only_names=None):
        self.predict_calls += 1
        self.last_only_names = only_names
        return [self._result]


def _make_result(h=480, w=640, label="banana", cls_id=0, extra=None):
    names = {0: "banana", 1: "bottle"}
    # a solid central rectangular mask → min-area-rect succeeds, depth valid.
    mask = np.zeros((h, w), dtype=np.float32)
    mask[180:300, 260:380] = 1.0
    boxes = [_Box([260, 180, 380, 300], cls_id, 0.88)]
    masks = [mask]
    if extra is not None:
        # add a distractor detection of a different class.
        m2 = np.zeros((h, w), dtype=np.float32)
        m2[50:120, 50:160] = 1.0
        boxes.append(_Box([50, 50, 160, 120], extra, 0.80))
        masks.append(m2)
    return YoloResult(
        names=names,
        boxes=_Boxes(boxes),
        masks=_Masks(np.stack(masks, axis=0)),
        orig_shape=(h, w),
    )


def _scene():
    h, w = 480, 640
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 400, dtype=np.uint16)  # 400mm everywhere
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    return color, depth, K


# ── tests ─────────────────────────────────────────────────────────────────
def test_full_pipeline_order_and_success():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    res = run_grasp_once(
        "banana",
        arm=arm,
        segmenter=seg,
        camera=cam,
        K=K,
        T_hand_eye=np.eye(4),
        warm_up_frames=3,
        grasp_force=1.2,
    )

    assert res["success"] is True
    assert res["cancelled"] is False
    assert res["stage"] == "done"
    assert res["grasp_class"] == "banana"
    assert "grasp_pose" in res and len(res["grasp_pose"]) == 6
    # capture warm-up + the servo-correction re-look (servo_correct default).
    assert cam.warmed == 6
    # Median-always aggregation evaluates three frames for both the initial
    # observation and the servo-correction re-look.
    assert seg.predict_calls == 6

    names = arm.names_of_calls()
    # ordering: open before any move; grasp after both moves; force threaded.
    assert names.index("open_gripper") < names.index("move_to")
    assert names.count("move_to") >= 3   # pregrasp, grasp, lift
    assert "grasp" in names
    assert names.index("grasp") < len(names)
    # grasp received the configured force.
    grasp_call = next(c for c in arm.calls if c[0] == "grasp")
    assert grasp_call[1] == 1.2


def test_grasp_pipeline_uses_configured_safe_open_distance():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    res = run_grasp_once(
        "banana",
        arm=arm,
        segmenter=seg,
        camera=cam,
        K=K,
        T_hand_eye=np.eye(4),
        warm_up_frames=0,
        open_distance_m=0.06,
        move_duration=0.02,
    )

    assert res["success"] is True
    # The configured 0.06 is a FLOOR: the pipeline auto-widens the pre-grasp
    # open to the detected object width + margin (clamped to the 0.09
    # mechanical max) so the jaw clears objects wider than the configured
    # width (e.g. the 0.077m demo box).
    expected = min(0.09, max(0.06, res["jaw_width_m"] + 0.012))
    open_calls = [c for c in arm.calls if c[0] == "open_gripper"]
    assert open_calls == [("open_gripper", pytest.approx(expected))]
    assert res["open_distance_m"] == pytest.approx(expected)
    assert open_calls[0][1] >= 0.06


def test_cancel_before_motion_safe_parks_gripper():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    cancel = threading.Event()
    cancel.set()  # cancelled from the start

    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), cancel_event=cancel, open_distance_m=0.06,
    )

    assert res["success"] is False
    assert res["cancelled"] is True
    # safe-park = gripper opened; NO grasp (clamp) issued.
    assert "open_gripper" in arm.names_of_calls()
    open_call = next(c for c in arm.calls if c[0] == "open_gripper")
    assert open_call == ("open_gripper", 0.06)
    assert "grasp" not in arm.names_of_calls()


def test_cancel_midway_stops_after_pregrasp():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    # Arm that trips the cancel event right after the pregrasp move. The
    # interruptible settle-wait (_wait_motion_cancellable) now observes the
    # cancel DURING the pregrasp wait and safe-parks at stage="pregrasp" —
    # earlier than the old uninterruptible wait_motion, which only caught it
    # at the next stage's _check_cancel. Either way: only the pregrasp move
    # ran, the gripper was never clamped, and it safe-parks open last.
    cancel = threading.Event()

    class TrippingArm(FakeArm):
        def __init__(self, ev) -> None:
            super().__init__()
            self._ev = ev
            self._moves = 0

        def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
            self._moves += 1
            if self._moves == 1:  # pregrasp done → request stop
                self._ev.set()
            return super().move_to(x, y, z, roll, pitch, yaw, duration)

    arm = TrippingArm(cancel)
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), cancel_event=cancel, move_duration=0.3,
    )

    assert res["cancelled"] is True
    # Cancel is now caught during the pregrasp settle-wait (earlier).
    assert res["stage"] == "pregrasp"
    names = arm.names_of_calls()
    assert names.count("move_to") == 1     # only the pregrasp move ran
    assert "grasp" not in names            # never clamped
    assert names[-1] == "open_gripper"     # safe-parked last


def test_target_filter_keeps_only_requested_class():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    # scene has banana(0) + bottle(1); we request bottle.
    seg = FakeSegmenter(_make_result(label="banana", cls_id=0, extra=1))

    res = run_grasp_once(
        "bottle", arm=arm, segmenter=seg, camera=cam, K=K, T_hand_eye=np.eye(4)
    )
    # grasp executed and the chosen class is the bottle (filtered correctly).
    assert res.get("grasp_class") == "bottle"


def test_no_detection_returns_error():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result(cls_id=0))  # only banana present

    res = run_grasp_once(
        "wrench", arm=arm, segmenter=seg, camera=cam, K=K, T_hand_eye=np.eye(4)
    )
    assert res["success"] is False
    assert res["stage"] == "detect"
    assert "no valid grasp" in res["error"]
    assert "grasp" not in arm.names_of_calls()


def test_missing_hand_eye_returns_error_before_motion():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K, T_hand_eye=None
    )
    assert res["success"] is False
    assert res["stage"] == "transform"
    assert "hand-eye" in res["error"]
    # no arm motion issued.
    assert "move_to" not in arm.names_of_calls()


def test_missing_camera_returns_error():
    arm = FakeArm()
    seg = FakeSegmenter(_make_result())
    res = run_grasp_once("banana", arm=arm, segmenter=seg, camera=None)
    assert res["success"] is False
    assert "camera" in res["error"]


# ── actuator-lock coordination (item 2) ────────────────────────────────────
class _RecordingActuator:
    """Records lock acquire/release so we can assert the grasp pipeline holds
    the actuator lock around each bus op and releases it across waits."""

    def __init__(self) -> None:
        import threading as _t
        self._lock = _t.RLock()
        self.depth = 0
        self.max_depth = 0
        self.acquired_count = 0

    @contextlib.contextmanager
    def acquire_motion_lock(self):
        with self._lock:
            self.depth += 1
            self.acquired_count += 1
            self.max_depth = max(self.max_depth, self.depth)
            try:
                yield
            finally:
                self.depth -= 1


def test_grasp_holds_actuator_lock_per_op_and_releases_across_wait():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())
    actuator = _RecordingActuator()

    class LockAssertingArm(FakeArm):
        def __init__(self, act) -> None:
            super().__init__()
            self._act = act

        def move_to(self, *a, **kw) -> bool:
            # The lock MUST be held during the bus op.
            assert self._act.depth == 1, "move_to ran without the actuator lock"
            return super().move_to(*a, **kw)

        def wait_motion(self, duration, extra=0.6) -> None:
            # The lock MUST be released across the (blocking) settle wait.
            assert self._act.depth == 0, "wait_motion ran while holding the lock"
            super().wait_motion(duration, extra)

    arm = LockAssertingArm(actuator)
    res = run_grasp_once(
        "banana", arm=arm, actuator=actuator, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), move_duration=0.05, grasp_force=1.0,
    )
    assert res["success"] is True
    # Lock was acquired per-op (never nested) and acquired multiple times.
    assert actuator.max_depth == 1
    assert actuator.acquired_count >= 4  # tcp pose + open + 3 moves + grasp


def test_grasp_works_without_actuator_lock_interface():
    # When no actuator (or one without acquire_motion_lock) is passed, the
    # pipeline still runs via a null context — backward compatible.
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())
    res = run_grasp_once(
        "banana", arm=arm, actuator=object(), segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), move_duration=0.02,
    )
    assert res["success"] is True


# ── interruptible settle wait (item 10) ────────────────────────────────────
def test_cancel_during_settle_wait_safe_parks():
    # A long move_duration means the settle wait dominates; the cancel fires
    # during that wait and the pipeline must abort + safe-park (open) without
    # blocking for the full duration or clamping the gripper.
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())
    cancel = threading.Event()

    class WaitTrippingArm(FakeArm):
        def __init__(self, ev) -> None:
            super().__init__()
            self._ev = ev
            self._moves = 0

        def move_to(self, *a, **kw) -> bool:
            self._moves += 1
            if self._moves == 1:
                # Trip cancel just after the first move so the next settle
                # wait observes it.
                self._ev.set()
            return super().move_to(*a, **kw)

    arm = WaitTrippingArm(cancel)
    t0 = time.monotonic()
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), cancel_event=cancel, move_duration=5.0,
    )
    elapsed = time.monotonic() - t0
    assert res["cancelled"] is True
    assert elapsed < 2.0, "settle wait was not interruptible (blocked full 5s)"
    assert "grasp" not in arm.names_of_calls()
    assert arm.names_of_calls()[-1] == "open_gripper"


# ── put_down (place back where picked up) ──────────────────────────────────
class PoseRecordingArm(FakeArm):
    """FakeArm that records the full move_to pose, not just z."""

    def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
        self.calls.append(("move_to", (round(float(x), 4), round(float(y), 4),
                                       round(float(z), 4))))
        return True


def test_put_down_replays_recorded_grasp_poses_and_releases_wide():
    arm = PoseRecordingArm()
    grasp6 = [0.40, 0.05, 0.08, 0.0, 0.0, 0.3]
    pre6 = [0.38, 0.05, 0.16, 0.0, 0.0, 0.3]

    res = run_put_down_once(
        arm=arm,
        grasp_pose=grasp6,
        pregrasp_pose=pre6,
        open_distance_m=0.089,  # the grasp's auto-widened width
        move_duration=0.02,
    )

    assert res["success"] is True
    assert res["released"] is True
    assert res["used_recorded_pose"] is True
    assert res["placed_at"] == [0.40, 0.05, 0.08]

    moves = [c[1] for c in arm.calls if c[0] == "move_to"]
    # approach (pregrasp) → place (grasp) → VERTICAL HOP (+6cm, anti-tip) →
    # retreat (pregrasp) → home
    assert moves[0] == (0.38, 0.05, 0.16)
    assert moves[1] == (0.40, 0.05, 0.08)
    assert moves[2] == (0.40, 0.05, 0.14)
    assert moves[3] == (0.38, 0.05, 0.16)
    assert moves[4] == (0.27, 0.0, 0.24)
    # release uses the recorded (widened) width and happens AFTER the place
    # move and BEFORE the retreat.
    names = arm.names_of_calls()
    open_call = next(c for c in arm.calls if c[0] == "open_gripper")
    assert open_call == ("open_gripper", 0.089)
    open_idx = names.index("open_gripper")
    move_idxs = [i for i, n in enumerate(names) if n == "move_to"]
    assert move_idxs[1] < open_idx < move_idxs[2]


def test_put_down_fallback_pose_when_no_recorded_grasp():
    arm = PoseRecordingArm()
    res = run_put_down_once(
        arm=arm,
        place_pose=(0.30, 0.00, 0.15, 0.0, 0.0, 0.0),
        move_duration=0.02,
    )
    assert res["success"] is True
    assert res["used_recorded_pose"] is False
    assert res["placed_at"] == [0.30, 0.0, 0.15]
    moves = [c[1] for c in arm.calls if c[0] == "move_to"]
    # derived approach 0.08 above the fallback spot, then the spot itself.
    assert moves[0] == (0.30, 0.0, 0.23)
    assert moves[1] == (0.30, 0.0, 0.15)
    # full-open default release (no recorded width).
    open_call = next(c for c in arm.calls if c[0] == "open_gripper")
    assert open_call == ("open_gripper", 0.10)


def test_put_down_place_ik_failure_keeps_holding():
    # The place move MUST succeed; otherwise we keep holding (no release at an
    # arbitrary pose) and report the error.
    class PlaceFailArm(PoseRecordingArm):
        def __init__(self) -> None:
            super().__init__()
            self._moves = 0

        def move_to(self, *a, **kw) -> bool:
            self._moves += 1
            if self._moves == 2:  # approach ok, place fails
                self.calls.append(("move_to", "FAILED"))
                return False
            return super().move_to(*a, **kw)

    arm = PlaceFailArm()
    res = run_put_down_once(
        arm=arm,
        grasp_pose=[0.40, 0.0, 0.08, 0.0, 0.0, 0.0],
        pregrasp_pose=[0.38, 0.0, 0.16, 0.0, 0.0, 0.0],
        move_duration=0.02,
    )
    assert res["success"] is False
    assert res["released"] is False
    assert res["stage"] == "place"
    assert "still holding" in res["error"]
    assert "open_gripper" not in arm.names_of_calls()


def test_put_down_release_verified_with_retry_then_honest_failure():
    """Release is VERIFIED physically: still gripping after the recorded-width
    open → one retry at full mechanical open; still gripping → honest error
    (never a fake success that leaves the demo box clamped)."""
    class StuckJawArm(PoseRecordingArm):
        def gripper_is_holding(self) -> bool:
            return True  # physically never lets go

    arm = StuckJawArm()
    res = run_put_down_once(
        arm=arm,
        grasp_pose=[0.40, 0.0, 0.08, 0.0, 0.0, 0.0],
        pregrasp_pose=[0.38, 0.0, 0.16, 0.0, 0.0, 0.0],
        open_distance_m=0.089,
        move_duration=0.02,
    )
    assert res["success"] is False
    assert res["released"] is False
    assert res["stage"] == "release"
    assert "still gripping" in res["error"]
    opens = [c for c in arm.calls if c[0] == "open_gripper"]
    assert opens == [("open_gripper", 0.089), ("open_gripper", 0.10)]
    # No retreat/home after a failed release — the arm stays at the place
    # pose with the object still held.
    move_idxs = [i for i, n in enumerate(arm.names_of_calls()) if n == "move_to"]
    assert len(move_idxs) == 2  # approach + place only


def test_put_down_cancel_before_motion_safe_parks():
    arm = PoseRecordingArm()
    cancel = threading.Event()
    cancel.set()
    res = run_put_down_once(
        arm=arm,
        grasp_pose=[0.40, 0.0, 0.08, 0.0, 0.0, 0.0],
        cancel_event=cancel,
        move_duration=0.02,
    )
    assert res["success"] is False
    assert res["cancelled"] is True
    assert "move_to" not in arm.names_of_calls()
    # explicit user stop → safe-park (open) like the grasp pipeline.
    assert "open_gripper" in arm.names_of_calls()


def test_put_down_approach_ik_failure_falls_through_to_direct_place():
    class ApproachFailArm(PoseRecordingArm):
        def __init__(self) -> None:
            super().__init__()
            self._moves = 0

        def move_to(self, *a, **kw) -> bool:
            self._moves += 1
            if self._moves == 1:  # approach fails, the rest succeed
                return False
            return super().move_to(*a, **kw)

    arm = ApproachFailArm()
    res = run_put_down_once(
        arm=arm,
        grasp_pose=[0.40, 0.0, 0.08, 0.0, 0.0, 0.0],
        pregrasp_pose=[0.38, 0.0, 0.16, 0.0, 0.0, 0.0],
        move_duration=0.02,
    )
    assert res["success"] is True
    assert res["released"] is True
    moves = [c[1] for c in arm.calls if c[0] == "move_to"]
    # first recorded move is the direct place (approach returned False before
    # recording), then retreat + home.
    assert moves[0] == (0.40, 0.0, 0.08)


# ── holding-property handling (item 12) ────────────────────────────────────
def test_gripper_is_holding_property_not_called():
    # On the real RebotArm gripper_is_holding is a PROPERTY (bool), not a
    # method. The pipeline must not try to call it (TypeError).
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    class PropertyHoldingArm(FakeArm):
        @property
        def gripper_is_holding(self) -> bool:  # type: ignore[override]
            return True

    arm = PropertyHoldingArm()
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), move_duration=0.02,
    )
    assert res["success"] is True
    assert res["holding"] is True


# ── success-rate hardening (2026-06-12): retry / multi-frame / plausibility ─
class ColdCamera(FakeCamera):
    """Returns None frames for the first `cold` get_frame calls (Orbbec
    cold-start behaviour), then real frames."""

    def __init__(self, color, depth, K, cold=2) -> None:
        super().__init__(color, depth, K)
        self._cold = cold
        self.frames_served = 0

    def get_frame(self):
        self.frames_served += 1
        if self._cold > 0:
            self._cold -= 1
            return None, None
        return super().get_frame()


def test_multiframe_detection_recovers_cold_camera_frames():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = ColdCamera(color, depth, K, cold=2)
    seg = FakeSegmenter(_make_result())

    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
        detect_frames=3,
    )
    assert res["success"] is True
    assert res["attempt"] == 1          # recovered WITHIN the attempt
    # 2 cold + 3 valid median frames, +1 frame for the servo re-look.
    assert cam.frames_served == 6
    assert seg.predict_calls == 4       # three median frames + servo look


def test_detect_frame_floor_recovers_single_cold_frame():
    # Temporal stabilization (Item C) runs N = max(3, detect_frames) frames per
    # capture, so even with detect_frames=1 a single cold first frame is
    # recovered WITHIN the same attempt (the floor of 3 frames guarantees at
    # least a couple of real frames after a cold-start drop) — no retry needed.
    color, depth, K = _scene()
    arm = FakeArm()
    cam = ColdCamera(color, depth, K, cold=1)
    seg = FakeSegmenter(_make_result())

    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
        detect_frames=1, retries=0,
    )
    assert res["success"] is True
    assert res["attempt"] == 1


def test_closed_on_air_retries_and_succeeds():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    class AirThenGripArm(FakeArm):
        def __init__(self) -> None:
            super().__init__()
            self._grasps = 0

        def grasp(self, force=None, timeout: float = 5.0) -> bool:
            self._grasps += 1
            self.calls.append(("grasp", force))
            if self._grasps == 1:
                self._holding = False   # closed on air
                return False
            self._holding = True
            return True

    arm = AirThenGripArm()
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02, retries=1,
    )
    assert res["success"] is True
    assert res["attempt"] == 2
    # the failed attempt re-opened the jaw before retrying (safe + clears it).
    opens = [c for c in arm.calls if c[0] == "open_gripper"]
    assert len(opens) >= 2


def test_closed_on_air_no_retries_reports_failure():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    class AirArm(FakeArm):
        def grasp(self, force=None, timeout: float = 5.0) -> bool:
            self.calls.append(("grasp", force))
            self._holding = False
            return False

    res = run_grasp_once(
        "banana", arm=AirArm(), segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02, retries=0,
    )
    assert res["success"] is False
    assert res["stage"] == "grasp"
    assert "nothing held" in res["error"]


def test_lost_during_carry_retries():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    class SlipperyArm(FakeArm):
        """First attempt: grasps fine but loses the object during the carry
        (holding flips False after the carry-home move). Second attempt OK."""

        def __init__(self) -> None:
            super().__init__()
            self._attempt = 0
            self._moves_since_grasp = None

        def grasp(self, force=None, timeout: float = 5.0) -> bool:
            self.calls.append(("grasp", force))
            self._attempt += 1
            self._holding = True
            self._moves_since_grasp = 0
            return True

        def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
            ok = super().move_to(x, y, z, roll, pitch, yaw, duration)
            if self._moves_since_grasp is not None:
                self._moves_since_grasp += 1
                # clearance lift + carry home = 2 moves; drop on the carry of
                # the FIRST attempt only.
                if self._attempt == 1 and self._moves_since_grasp >= 2:
                    self._holding = False
            return ok

    arm = SlipperyArm()
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02, retries=1,
    )
    assert res["success"] is True
    assert res["attempt"] == 2


def test_plausible_box_rejects_garbage_position_and_is_optional():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    # Synthetic scene grasp lands near (0, 0, ~0.4) in this fake geometry —
    # a box demanding x>=0.2 must reject it (and exhaust retries).
    res = run_grasp_once(
        "banana", arm=FakeArm(), segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
        plausible_box=[0.2, 0.6, -0.3, 0.3, -0.02, 0.3], retries=1,
    )
    assert res["success"] is False
    assert res["stage"] == "plausibility"
    assert res["attempt"] == 2          # retriable → both attempts ran

    # No box (default) → same scene succeeds.
    res2 = run_grasp_once(
        "banana", arm=FakeArm(), segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
    )
    assert res2["success"] is True


def test_stage_timings_present_on_success_and_failure():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())
    res = run_grasp_once(
        "banana", arm=FakeArm(), segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
    )
    assert res["success"] is True
    assert "stage_ms" in res and "capture" in res["stage_ms"]


# ── oblique-view geometry (3D short axis) + IK orientation ladder ──────────
def test_short_axis_3d_measures_true_width_on_tilted_surface():
    # Depth ramps along x: the short edge spans a surface tilted in depth, so
    # the true 3D width must EXCEED the fronto-parallel (legacy) estimate and
    # the open axis must gain a z component.
    from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import (
        _short_axis_3d, _pixel_vec_to_3d,
    )
    h, w = 480, 640
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    depth = np.zeros((h, w), dtype=np.uint16)
    for x in range(w):
        depth[:, x] = 400 + (x - 320)  # 1mm per pixel ramp along x
    center = np.array([320.0, 240.0], dtype=np.float32)
    short_dir = np.array([1.0, 0.0], dtype=np.float32)  # short axis along x
    span_px = 120.0

    vec3d = _short_axis_3d(depth, K, center, short_dir, span_px)
    assert vec3d is not None
    width_3d = float(np.linalg.norm(vec3d))
    width_2d = float(np.linalg.norm(_pixel_vec_to_3d(short_dir * span_px, 0.4, K)))
    assert width_3d > width_2d            # tilt adds the depth component
    assert abs(float(vec3d[2])) > 0.01    # z component captured (≈0.12m over the span)


def test_short_axis_3d_falls_back_on_invalid_depth():
    from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import _short_axis_3d
    h, w = 480, 640
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    depth = np.zeros((h, w), dtype=np.uint16)  # all invalid
    out = _short_axis_3d(depth, K, np.array([320.0, 240.0]), np.array([1.0, 0.0]), 100.0)
    assert out is None


def test_orientation_ladder_relaxes_pitch_keeps_yaw():
    from ovs_agent.apps.voice_rebot_arm.grasp_service import _relax_orientation

    class TiltLimitedArm:
        """check_ik fails when |roll|+|pitch| > 0.2 (far-reach behaviour)."""
        def check_ik(self, x, y, z, r, p, yw):
            return (abs(r) + abs(p) <= 0.2), 0.0

    pre6d = [0.65, 0.0, 0.18, 0.1, 0.5, 0.45]
    grasp6d = [0.72, 0.0, 0.15, 0.1, 0.5, 0.45]
    out = _relax_orientation(TiltLimitedArm(), pre6d, grasp6d)
    assert out is not None
    new_pre, new_grasp = out
    assert new_grasp[5] == 0.45           # yaw preserved (jaw alignment)
    assert abs(new_grasp[3]) + abs(new_grasp[4]) <= 0.2  # flattened to feasible


def test_orientation_ladder_preserves_roll_alignment_when_only_pitch_blocks():
    """On the side-grasper the box-angle alignment is in the ROLL; when IK is
    blocked only by approach STEEPNESS (pitch), _relax must KEEP the roll
    (gripper still faces the angled box) and relax pitch — not flatten the roll.
    Regression for '夹爪不转头对着斜盒子'."""
    from ovs_agent.apps.voice_rebot_arm.grasp_service import _relax_orientation

    class PitchLimitedArm:
        """check_ik feasible for ANY roll, but only when pitch <= 0.3."""
        def check_ik(self, x, y, z, r, p, yw):
            return (abs(p) <= 0.3), 0.0

    # Large roll = aligned to a steeply-angled box; pitch too steep to reach.
    pre6d = [0.65, 0.0, 0.18, -0.95, 0.5, 0.0]
    grasp6d = [0.72, 0.0, 0.15, -0.95, 0.5, 0.0]
    out = _relax_orientation(PitchLimitedArm(), pre6d, grasp6d)
    assert out is not None
    _new_pre, new_grasp = out
    assert new_grasp[3] == -0.95          # ROLL (alignment) PRESERVED, not flattened
    assert abs(new_grasp[4]) <= 0.3       # pitch relaxed to feasible
    assert new_grasp[5] == 0.0            # yaw kept


def test_orientation_ladder_none_when_position_truly_unreachable():
    from ovs_agent.apps.voice_rebot_arm.grasp_service import _relax_orientation

    class NeverArm:
        def check_ik(self, *a):
            return False, 1.0

    out = _relax_orientation(NeverArm(), [0.9, 0, 0.1, 0, 0.5, 0.3],
                             [0.95, 0, 0.1, 0, 0.5, 0.3])
    assert out is None


def test_pregrasp_ik_failure_recovers_via_orientation_ladder():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    class TiltyArm(FakeArm):
        """move_to fails for any pose with |pitch| > 0.2; check_ik agrees.
        The synthetic grasp orientation has nonzero pitch, so the FIRST
        pregrasp move fails and the ladder must rescue the attempt."""
        def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
            if abs(roll) + abs(pitch) > 0.2:
                return False
            return super().move_to(x, y, z, roll, pitch, yaw, duration)

        def check_ik(self, x, y, z, r, p, yw):
            return (abs(r) + abs(p) <= 0.2), 0.0

    # Force a tilted grasp orientation by using a tilted hand-eye transform.
    import math
    a = 0.5
    T = np.eye(4)
    T[:3, :3] = np.array([
        [math.cos(a), 0, math.sin(a)],
        [0, 1, 0],
        [-math.sin(a), 0, math.cos(a)],
    ])
    arm = TiltyArm()
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=T, warm_up_frames=0, move_duration=0.02, retries=0,
    )
    # Either the original orientation was already feasible (geometry-dependent)
    # or the ladder kicked in — in both cases the grasp must succeed.
    assert res["success"] is True


# ── close-up re-observation (side-view width inflation fix) ────────────────
class ReobserveArm(FakeArm):
    """Records full move poses; get_tcp_pose returns identity (camera at
    origin) so the synthetic far/wide trigger can be steered via fakes."""

    def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
        self.calls.append(("move_to", (round(float(x), 3), round(float(y), 3),
                                       round(float(z), 3))))
        return True


class TwoStageSegmenter(FakeSegmenter):
    """Returns an inflated-width detection (side view) from the FIRST camera
    viewpoint, then a sane one (close-up top view) after the camera has moved
    for a re-observation.

    Multi-frame temporal stabilization (Item C) runs N detector calls per
    capture from the *same* viewpoint, so a side-view inflation must persist
    across the whole first capture (a real camera that hasn't moved keeps
    seeing the same inflated silhouette). We therefore switch on the number of
    captures (``wide_captures`` predicts of the wide result before flipping to
    the sane one), not on a single call, so the wide reading isn't averaged
    away by a magically-sane second frame from an unmoved camera."""

    def __init__(self, wide_result, good_result, wide_captures=3) -> None:
        super().__init__(good_result)
        self._wide = wide_result
        self.predict_calls = 0
        self._wide_captures = int(wide_captures)

    def predict(self, image_bgr, conf=0.25, iou=0.45, only_names=None):
        self.predict_calls += 1
        wide = self.predict_calls <= self._wide_captures
        return [self._wide if wide else self._result]


def _wide_result():
    # mask 300x280px at 400mm -> SHORT axis 280px ~ 0.19m, far beyond the jaw
    # (min-area-rect grasps across the SHORTER projected edge, so BOTH
    # dimensions must exceed the jaw to simulate side-view inflation).
    h, w = 480, 640
    mask = np.zeros((h, w), dtype=np.float32)
    mask[100:380, 170:470] = 1.0
    return YoloResult(
        names={0: "banana", 1: "bottle"},
        boxes=_Boxes([_Box([170, 100, 470, 380], 0, 0.7)]),
        masks=_Masks(np.stack([mask], axis=0)),
        orig_shape=(h, w),
    )


def test_reobserve_recovers_from_inflated_side_view_width():
    color, depth, K = _scene()
    arm = ReobserveArm()
    cam = FakeCamera(color, depth, K)
    seg = TwoStageSegmenter(_wide_result(), _make_result())

    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02, retries=0,
    )
    assert res["success"] is True
    assert res.get("reobserved") is True
    # final width is the close-up (sane) measurement, not the inflated one.
    assert res["jaw_width_m"] < 0.09
    assert seg.predict_calls >= 2


def test_wide_after_reobserve_is_rejected_not_executed():
    color, depth, K = _scene()
    arm = ReobserveArm()
    cam = FakeCamera(color, depth, K)
    # BOTH views report the inflated width → must reject, never close the jaw.
    seg = TwoStageSegmenter(_wide_result(), _wide_result())

    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02, retries=0,
    )
    assert res["success"] is False
    assert res["stage"] == "plausibility"
    assert "jaw width" in res["error"]
    assert "grasp" not in arm.names_of_calls()


def test_reobserve_disabled_keeps_single_view():
    color, depth, K = _scene()
    arm = ReobserveArm()
    cam = FakeCamera(color, depth, K)
    seg = TwoStageSegmenter(_wide_result(), _make_result())
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
        retries=0, reobserve=False,
    )
    # single (inflated) view → width gate rejects without a second look.
    # (multi-frame capture runs N detector calls in the ONE capture; the point
    # is that NO re-observation move happened — predict_calls stays within the
    # single capture's N-frame budget, never a second capture round.)
    assert res["success"] is False
    assert res.get("reobserved") is None
    assert 1 <= seg.predict_calls <= 3


# ── top-face plane grasp (Phase 1) + servo correction (Phase 2) ─────────────
def test_top_face_grasp_picks_top_plane_not_side():
    from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import _top_face_grasp
    h, w = 480, 640
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    depth = np.zeros((h, w), dtype=np.uint16)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Synthetic oblique box: TOP face (flat at 400mm, normal ≈ -z_cam) spans
    # rows 200-280; SIDE face (depth ramps with row → tilted plane) spans rows
    # 280-400 and is BIGGER. Camera looks straight down → up_cam = -z.
    mask[200:280, 250:390] = 1; depth[200:280, 250:390] = 400
    for r in range(280, 400):
        mask[r, 250:390] = 1
        depth[r, 250:390] = 400 + (r - 280) * 3   # steep ramp = vertical-ish face
    up_cam = np.array([0.0, 0.0, -1.0])
    out = _top_face_grasp(mask, depth, K, up_cam)
    assert out is not None
    center, open_axis, approach, width, length, n_in = out
    # the chosen face is the FLAT one at 0.4m (top), not the ramp (side):
    assert abs(float(center[2]) - 0.4) < 0.02
    # approach presses onto the face: along +z in camera coords (camera→face).
    assert float(approach[2]) > 0.9
    # width = minor extent of the 140x80px top face at 0.4m: 80px ≈ 0.053m.
    assert 0.03 < width < 0.08
    assert length > width


def test_top_face_grasp_none_without_enough_points():
    from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import _top_face_grasp
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:204, 200:204] = 1; depth[200:204, 200:204] = 400
    assert _top_face_grasp(mask, depth, K, np.array([0, 0, -1.0])) is None


def test_estimate_grasp_uses_top_face_with_up_hint():
    from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import estimate_grasps
    color, depth, K = _scene()
    res = _make_result()
    grasps = estimate_grasps([res], depth, K, up_hint_cam=np.array([0, 0, -1.0]))
    best = [g for g in grasps if g.is_valid]
    assert best, "top-face path must produce a valid grasp on the flat scene"
    # flat scene at 400mm → grasp depth ≈ 0.4m and sane width.
    assert abs(float(best[0].position[2]) - 0.4) < 0.02
    assert 0.01 < best[0].jaw_width_m < 0.09


def test_servo_correction_shifts_grasp_within_bounds():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    class DriftingTcpArm(FakeArm):
        """After the pregrasp move, get_tcp_pose returns a slightly SHIFTED
        pose — the servo re-detection then computes a drifted grasp point and
        the pipeline must shift its grasp x/y by that drift (≤3cm)."""
        def __init__(self) -> None:
            super().__init__()
            self._shift = 0.0

        def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
            ok = super().move_to(x, y, z, roll, pitch, yaw, duration)
            if len([c for c in self.calls if c[0] == "move_to"]) == 1:
                self._shift = 0.012   # 12mm drift appears after pregrasp
            return ok

        def get_tcp_pose(self) -> np.ndarray:
            self.calls.append(("get_tcp_pose",))
            T = np.eye(4, dtype=np.float64)
            T[0, 3] = self._shift
            return T

    arm = DriftingTcpArm()
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
        retries=0, servo_correct=True,
    )
    assert res["success"] is True
    assert res.get("servo_drift_mm") is not None
    assert 4.0 < res["servo_drift_mm"] <= 30.0


def test_servo_disabled_no_extra_detection():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())
    res = run_grasp_once(
        "banana", arm=FakeArm(), segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
        retries=0, servo_correct=False,
    )
    assert res["success"] is True
    assert "servo_drift_mm" not in res
    assert seg.predict_calls == 3


def test_reobserve_goes_high_when_top_face_not_visible():
    """First estimate from the legacy/silhouette path (camera can't see the
    object's top) + implausible width → the re-observation must move HIGH
    with a downward tilt (z 0.33, pitch 0.45), not the flat close-up."""
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = TwoStageSegmenter(_wide_result(), _make_result())

    class PoseLogArm(FakeArm):
        def __init__(self) -> None:
            super().__init__()
            self.poses: list[tuple] = []

        def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
            self.poses.append((round(z, 2), round(pitch, 2)))
            self.calls.append(("move_to", round(float(z), 4)))
            return True

    arm = PoseLogArm()
    # NO up_hint flows in tests via the FakeArm tcp (identity) — the wide
    # first result comes from the legacy path (method='legacy') because the
    # synthetic up-hint geometry rejects the flat scene? Force legacy by
    # relying on the wide fixture (top-face fit fails on it: erosion +
    # square-ish mask still fits a plane though...). Robust assertion: the
    # FIRST observation move (the re-observe) uses the high-tilt z/pitch.
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02, retries=0,
    )
    assert res.get("reobserved") is True
    first_move = arm.poses[0]
    # high-tilt variant unless the wide estimate already came from top_face
    assert first_move in [(0.33, 0.45), (0.26, 0.0)]
    if first_move == (0.33, 0.45):
        assert res["success"] is True


# ── side-face grasp (Phase: side MVP) ───────────────────────────────────────
def _tall_side_scene():
    """Synthetic tall object: camera sees ONLY a vertical front face (depth
    constant, normal toward camera, perpendicular to up). up = -y (camera
    horizontal). Face is 100px wide × 300px tall at 400mm ≈ 0.067m × 0.2m."""
    h, w = 480, 640
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.uint16)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[90:390, 290:390] = 1.0
    depth[90:390, 290:390] = 400
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    res = YoloResult(
        names={0: "box", 1: "bottle"},
        boxes=_Boxes([_Box([290, 90, 390, 390], 0, 0.8)]),
        masks=_Masks(np.stack([mask], axis=0)),
        orig_shape=(h, w),
    )
    return color, depth, K, res


def test_side_face_grasp_selected_for_tall_front_face():
    from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import estimate_grasps
    color, depth, K, res = _tall_side_scene()
    # up = -y in camera frame: the flat front face (normal -z, toward camera)
    # is perpendicular to up → side candidate; no top face in view.
    grasps = [g for g in estimate_grasps([res], depth, K,
                                         up_hint_cam=np.array([0.0, -1.0, 0.0]))
              if g.is_valid]
    assert grasps, "side-face path must produce a candidate"
    g = grasps[0]
    assert g.method == "side_face"
    # horizontal width of the face ≈ 100px at 0.4m ≈ 0.067m (within jaw).
    assert 0.04 < g.jaw_width_m < 0.085
    assert g.object_length_m > g.jaw_width_m   # vertical extent is longer


def test_side_grasp_too_low_rejected_retriable():
    color, depth, K, res = _tall_side_scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(res)

    class LowTcpArm(FakeArm):
        """TCP transform puts the side-face grasp point BELOW 45mm."""
        def get_tcp_pose(self) -> np.ndarray:
            self.calls.append(("get_tcp_pose",))
            T = np.eye(4, dtype=np.float64)
            # rotate camera frame so the face centroid lands at low base z:
            # base z = -y_cam mapping → centroid y_cam≈0 → z≈0. Identity works:
            return T

    res_run = run_grasp_once(
        "banana", arm=LowTcpArm(), segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
        retries=0, reobserve=False, servo_correct=False,
    )
    # with identity transforms the synthetic grasp z≈0 → below the 45mm gate
    if res_run.get("grasp_method") == "side_face":
        assert res_run["success"] is False
        assert "too low" in res_run["error"]


def test_side_grasp_skips_orientation_ladder():
    from ovs_agent.apps.voice_rebot_arm import grasp_service as gs
    color, depth, K, res = _tall_side_scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(res)

    calls = {"ladder": 0}
    orig = gs._relax_orientation

    def _spy(arm, pre, grasp):
        calls["ladder"] += 1
        return orig(arm, pre, grasp)

    class SideArm(FakeArm):
        """Pregrasp move fails once → ladder would be consulted for top
        grasps; must NOT be for side grasps. TCP lifts the scene so the
        side z-gate passes."""
        def __init__(self) -> None:
            super().__init__()
            self._fails = 1

        def get_tcp_pose(self) -> np.ndarray:
            self.calls.append(("get_tcp_pose",))
            T = np.eye(4, dtype=np.float64)
            T[2, 3] = 0.15  # raise base z above the 45mm side gate
            return T

        def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
            if self._fails > 0:
                self._fails -= 1
                return False
            return super().move_to(x, y, z, roll, pitch, yaw, duration)

    gs._relax_orientation = _spy
    try:
        out = run_grasp_once(
            "banana", arm=SideArm(), segmenter=seg, camera=cam, K=K,
            T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.02,
            retries=0, reobserve=False, servo_correct=False,
        )
    finally:
        gs._relax_orientation = orig
    if out.get("grasp_method") == "side_face":
        assert calls["ladder"] == 0          # ladder never consulted
        assert out["success"] is False       # move failed → honest failure
        assert out["error"] == "pregrasp IK failed"


# ── put_down table-boundary clamp (box fell off the table, 2026-06-12) ──


def test_put_down_clamps_edge_place_point_into_bounds() -> None:
    """A recorded grasp pose at the table edge must be pulled inward by the
    margin before release; the approach shifts by the same x/y delta so the
    descent geometry is preserved."""
    arm = PoseRecordingArm()
    res = run_put_down_once(
        arm=arm,
        grasp_pose=[0.46, 0.20, 0.08, 0.0, 0.0, 0.3],
        pregrasp_pose=[0.40, 0.18, 0.16, 0.0, 0.0, 0.3],
        open_distance_m=0.089,
        move_duration=0.02,
        place_bounds=[0.15, 0.48, -0.22, 0.22],
        place_margin_m=0.05,
    )
    assert res["success"] is True
    assert res["place_clamped"] is True
    assert res["place_original"] == [0.46, 0.20, 0.08]
    # x 0.46 → 0.43 (0.48-0.05), y 0.20 → 0.17 (0.22-0.05)
    assert res["placed_at"] == pytest.approx([0.43, 0.17, 0.08])
    moves = [c[1] for c in arm.calls if c[0] == "move_to"]
    # approach shifted by the same (-0.03, -0.03) delta.
    assert moves[0] == pytest.approx((0.37, 0.15, 0.16))
    assert moves[1] == pytest.approx((0.43, 0.17, 0.08))


def test_put_down_inside_bounds_is_untouched() -> None:
    arm = PoseRecordingArm()
    res = run_put_down_once(
        arm=arm,
        grasp_pose=[0.30, 0.00, 0.08, 0.0, 0.0, 0.0],
        pregrasp_pose=[0.28, 0.00, 0.16, 0.0, 0.0, 0.0],
        move_duration=0.02,
        place_bounds=[0.15, 0.48, -0.22, 0.22],
    )
    assert res["success"] is True
    assert "place_clamped" not in res
    assert res["placed_at"] == [0.30, 0.0, 0.08]


def test_put_down_malformed_bounds_ignored() -> None:
    arm = PoseRecordingArm()
    for bad in ([0.1, 0.2, 0.3], ["a", "b", "c", "d"], [0.5, 0.1, -0.2, 0.2]):
        res = run_put_down_once(
            arm=arm,
            grasp_pose=[0.46, 0.20, 0.08, 0.0, 0.0, 0.0],
            move_duration=0.02,
            place_bounds=bad,
        )
        assert res["success"] is True
        assert "place_clamped" not in res
        assert res["placed_at"] == [0.46, 0.2, 0.08]


def test_clamp_place_xy_margin_swallowing_axis_centers() -> None:
    import ovs_agent.apps.voice_rebot_arm.grasp_service as gs

    # y span 0.06 with margin 0.05 each side → no room: center at 0.0.
    out = gs._clamp_place_xy([0.50, 0.10, 0.08, 0, 0, 0], [0.2, 0.4, -0.03, 0.03], 0.05)
    assert out is not None
    assert out[0] == pytest.approx(0.35)  # 0.4 - 0.05
    assert out[1] == pytest.approx(0.0)   # centered


# ── over-wide top-face rejection (tall-box compromise plane, 2026-06-13) ──


def _fake_top(width: float, fill_side=None):
    """A stand-in _top_face_grasp returning a fixed width. fill_side, if given,
    is appended to side_out (simulating a collected side candidate)."""
    def _impl(mask, depth_mm, K, up_cam, *a, side_out=None, **kw):
        if fill_side is not None and side_out is not None:
            side_out.append(fill_side)
        # (center_cam, open_axis_cam, approach_cam, width, length, n_inliers)
        return (np.array([0.0, 0.0, 0.4]), np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, -1.0]), float(width), 0.12, 300)
    return _impl


def test_overwide_top_rejected_without_side_candidate(monkeypatch):
    """A tall box's compromise plane returns as 'top' with an inflated width
    BEFORE any side candidate is collected (side_out empty). The arbitration
    must still reject it (drop the 0.085+ width) and fall through to legacy —
    not pass the bogus 0.27m width to the plausibility gate."""
    import ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp as og

    monkeypatch.setattr(og, "_top_face_grasp", _fake_top(0.27))  # no side filled
    color, depth, K = _scene()
    grasps = og.estimate_grasps([_make_result()], depth, K,
                                up_hint_cam=np.array([0.0, 0.0, -1.0]))
    valid = [g for g in grasps if g.is_valid]
    assert valid, "should still produce a grasp via the legacy fallback"
    assert valid[0].method != "top_face", (
        f"over-wide top must be rejected, got method={valid[0].method!r} "
        f"width={valid[0].jaw_width_m}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PR #37: force-side intentionally suppresses top_face until its "
        "approach/roll is safe on hardware"
    ),
)
def test_normal_width_top_still_used(monkeypatch):
    """A normal flat box (top width < 0.085 jaw limit) keeps the top-face path
    byte-identically — the over-wide rejection must NOT touch it."""
    import ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp as og

    monkeypatch.setattr(og, "_top_face_grasp", _fake_top(0.06))
    color, depth, K = _scene()
    grasps = og.estimate_grasps([_make_result()], depth, K,
                                up_hint_cam=np.array([0.0, 0.0, -1.0]))
    valid = [g for g in grasps if g.is_valid]
    assert valid and valid[0].method == "top_face"
    assert abs(valid[0].jaw_width_m - 0.06) < 1e-6


def test_overwide_top_prefers_side_candidate_when_available(monkeypatch):
    """When a side candidate WAS collected, an over-wide top still yields a
    side_face grasp (the original arbitration intent, now also reached when
    side_cands is the only viable path)."""
    import ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp as og

    # side candidate: (centroid, horiz, n_cam, h_width<=0.085, v_len, n_in)
    side = (np.array([0.05, 0.0, 0.42]), np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0]), 0.058, 0.18, 250)
    monkeypatch.setattr(og, "_top_face_grasp", _fake_top(0.27, fill_side=side))
    color, depth, K = _scene()
    grasps = og.estimate_grasps([_make_result()], depth, K,
                                up_hint_cam=np.array([0.0, 0.0, -1.0]))
    valid = [g for g in grasps if g.is_valid]
    assert valid and valid[0].method == "side_face"
    assert valid[0].jaw_width_m <= 0.085


# ── reachability-aware multi-candidate selection (sim-grounded, 2026-06-15) ──
def _grasp(conf, pos, method="legacy"):
    """Minimal valid GraspPose at camera-frame position ``pos``."""
    from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import GraspPose
    return GraspPose(
        class_name="box", conf=float(conf), bbox_xyxy=(0, 0, 10, 10),
        center_px=(5, 5), position=np.asarray(pos, dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
        tcp_rotation=np.eye(3, dtype=np.float32),
        jaw_width_m=0.05, object_length_m=0.1, angle_deg=0.0,
        rect_points=np.zeros((4, 2), dtype=np.float32),
        short_edge_points=np.zeros((2, 2), dtype=np.float32),
        valid_depth_pixels=100, method=method,
    )


class _ReachArm:
    """check_ik feasible only when the base-frame x reach is below ``x_max``."""
    def __init__(self, x_max=0.5):
        self.x_max = x_max
        self.checks = 0

    def check_ik(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        self.checks += 1
        return (abs(x) <= self.x_max, None)


def test_reach_rank_prefers_reachable_over_higher_conf():
    """Two candidates: the higher-conf one transforms out of the IK envelope,
    the lower-conf one is reachable → the reachable one must win."""
    from ovs_agent.apps.voice_rebot_arm.grasp_service import _select_reachable_grasp
    T = np.eye(4, dtype=np.float64)  # camera frame == base frame
    near = _grasp(0.60, [0.30, 0.0, 0.0])   # base x≈0.30 reachable
    far = _grasp(0.95, [0.80, 0.0, 0.0])    # base x≈0.80 out of envelope
    pick = _select_reachable_grasp([far, near], _ReachArm(x_max=0.5), T,
                                   pregrasp_offset_m=0.08, insertion_depth_m=0.015)
    assert pick is near


def test_reach_rank_falls_back_to_conf_when_none_reachable():
    """If no candidate is reachable, keep the plain max-confidence pick (no
    regression vs the old select_best_grasp behaviour)."""
    from ovs_agent.apps.voice_rebot_arm.grasp_service import _select_reachable_grasp
    T = np.eye(4, dtype=np.float64)
    a = _grasp(0.60, [0.90, 0.0, 0.0])
    b = _grasp(0.95, [0.95, 0.0, 0.0])
    pick = _select_reachable_grasp([a, b], _ReachArm(x_max=0.5), T,
                                   pregrasp_offset_m=0.08, insertion_depth_m=0.015)
    assert pick is b  # max conf


def test_reach_rank_noop_without_check_ik():
    """An arm with no check_ik must yield the confidence pick unchanged."""
    from ovs_agent.apps.voice_rebot_arm.grasp_service import _select_reachable_grasp

    class _NoIK:
        pass

    T = np.eye(4, dtype=np.float64)
    a = _grasp(0.60, [0.30, 0.0, 0.0])
    b = _grasp(0.95, [0.80, 0.0, 0.0])
    pick = _select_reachable_grasp([a, b], _NoIK(), T,
                                   pregrasp_offset_m=0.08, insertion_depth_m=0.015)
    assert pick is b  # max conf, no reachability filtering


def test_reach_rank_among_reachable_breaks_tie_by_conf():
    """When several candidates are reachable, the highest confidence wins."""
    from ovs_agent.apps.voice_rebot_arm.grasp_service import _select_reachable_grasp
    T = np.eye(4, dtype=np.float64)
    lo = _grasp(0.55, [0.20, 0.0, 0.0])
    hi = _grasp(0.80, [0.30, 0.0, 0.0])
    pick = _select_reachable_grasp([lo, hi], _ReachArm(x_max=0.5), T,
                                   pregrasp_offset_m=0.08, insertion_depth_m=0.015)
    assert pick is hi
