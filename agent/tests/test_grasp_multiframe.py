"""Tests for multi-frame temporal stabilization (Item C) in grasp_service.

The aggregation math (cluster → per-component median → outlier rejection →
angular-MAD gate) is pure and torch/SDK-free, so we exercise it directly on
synthetic GraspPose objects, plus one fake-camera integration test that drives
the real ``_capture_and_detect`` loop with jittered per-frame detections.
"""

from __future__ import annotations

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.grasp_service import (
    _aggregate_cluster,
    _angle_diff_deg,
    _circular_median_deg,
    _cluster_grasps,
    run_grasp_once,
)
from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import GraspPose


def _pose(*, x=0.30, y=0.0, z=0.40, jaw=0.060, angle=10.0, conf=0.60,
          cls="box", cx=320, cy=240, bbox=(290, 210, 350, 270), method="legacy"):
    return GraspPose(
        class_name=cls,
        conf=float(conf),
        bbox_xyxy=tuple(int(v) for v in bbox),
        center_px=(int(cx), int(cy)),
        position=np.array([x, y, z], dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
        tcp_rotation=np.eye(3, dtype=np.float32),
        jaw_width_m=float(jaw),
        object_length_m=0.10,
        angle_deg=float(angle),
        rect_points=np.zeros((4, 2), dtype=np.float32),
        short_edge_points=np.zeros((2, 2), dtype=np.float32),
        valid_depth_pixels=500,
        method=method,
    )


# ── circular-median / angle helpers ─────────────────────────────────────────
def test_circular_median_basic():
    assert _circular_median_deg([10.0, 12.0, 14.0]) == pytest.approx(12.0, abs=0.5)


def test_circular_median_wraps_mod_180():
    # 179° and 1° are 2° apart (mod 180), median should land near the boundary.
    m = _circular_median_deg([179.0, 1.0, 0.0])
    assert _angle_diff_deg(m, 0.0) <= 2.5


def test_angle_diff_is_mod_180():
    assert _angle_diff_deg(10.0, 170.0) == pytest.approx(20.0)
    assert _angle_diff_deg(0.0, 179.0) == pytest.approx(1.0)


# ── clustering ───────────────────────────────────────────────────────────────
def test_cluster_groups_same_object():
    poses = [_pose(cx=320, cy=240), _pose(cx=325, cy=243), _pose(cx=318, cy=238)]
    clusters = _cluster_grasps(poses)
    assert len(clusters) == 1 and len(clusters[0]) == 3


def test_cluster_splits_far_centers():
    near = _pose(cx=320, cy=240, bbox=(290, 210, 350, 270))
    far = _pose(cx=500, cy=240, bbox=(470, 210, 530, 270))
    clusters = _cluster_grasps([near, far])
    assert len(clusters) == 2


def test_cluster_splits_different_class():
    a = _pose(cls="box")
    b = _pose(cls="banana")
    clusters = _cluster_grasps([a, b])
    assert len(clusters) == 2


# ── aggregation: median position / angle / width, max conf ───────────────────
def test_aggregate_is_per_component_median():
    cluster = [
        _pose(x=0.30, y=0.00, z=0.40, jaw=0.060, angle=10.0, conf=0.55),
        _pose(x=0.31, y=0.01, z=0.41, jaw=0.062, angle=12.0, conf=0.61),
        _pose(x=0.305, y=0.005, z=0.405, jaw=0.061, angle=11.0, conf=0.58),
    ]
    agg, info = _aggregate_cluster(cluster)
    assert agg is not None
    assert info["rejected"] == 0
    assert float(agg.position[0]) == pytest.approx(0.305, abs=1e-4)
    assert float(agg.position[2]) == pytest.approx(0.405, abs=1e-4)
    assert agg.jaw_width_m == pytest.approx(0.061, abs=1e-4)
    assert agg.angle_deg == pytest.approx(11.0, abs=0.5)
    # conf is the MAX across the cluster.
    assert agg.conf == pytest.approx(0.61)


def test_aggregate_rejects_position_outlier():
    cluster = [
        _pose(x=0.30, angle=10.0, conf=0.55),
        _pose(x=0.305, angle=11.0, conf=0.58),
        _pose(x=0.40, angle=10.0, conf=0.60),  # +95mm → position outlier
    ]
    agg, info = _aggregate_cluster(cluster)
    assert agg is not None
    assert info["rejected"] == 1
    # outlier dropped → median position from the two inliers (~0.3025).
    assert float(agg.position[0]) == pytest.approx(0.3025, abs=2e-3)


def test_aggregate_rejects_width_outlier():
    cluster = [
        _pose(jaw=0.060, conf=0.55),
        _pose(jaw=0.061, conf=0.58),
        _pose(jaw=0.090, conf=0.60),  # +30mm → width outlier
    ]
    agg, info = _aggregate_cluster(cluster)
    assert agg is not None
    assert info["rejected"] == 1
    assert agg.jaw_width_m == pytest.approx(0.0605, abs=1e-3)


def test_aggregate_angular_mad_gate_fires():
    # Angles scattered far apart (no consistent open-axis) → MAD > 18° → reject.
    cluster = [
        _pose(angle=0.0, conf=0.55),
        _pose(angle=40.0, conf=0.58),
        _pose(angle=85.0, conf=0.60),
    ]
    agg, info = _aggregate_cluster(cluster)
    assert agg is None
    assert info["angular_mad_deg"] > 18.0
    assert "MAD" in info["error"]


def test_aggregate_angular_mad_gate_skipped_for_round():
    # Same scatter but a round shape → angle is meaningless → gate skipped.
    cluster = [
        _pose(angle=0.0, conf=0.55, method="round"),
        _pose(angle=40.0, conf=0.58, method="round"),
        _pose(angle=85.0, conf=0.60, method="round"),
    ]
    agg, info = _aggregate_cluster(cluster)
    assert agg is not None
    assert info["angular_mad_deg"] > 18.0  # large, but tolerated for round


def test_aggregate_tight_angles_pass_gate():
    cluster = [
        _pose(angle=10.0, conf=0.55),
        _pose(angle=13.0, conf=0.58),
        _pose(angle=11.0, conf=0.60),
    ]
    agg, info = _aggregate_cluster(cluster)
    assert agg is not None
    assert info["angular_mad_deg"] <= 18.0


# ── integration: fake camera yielding N jittered frames ──────────────────────
class _FakeArm:
    def __init__(self):
        self.calls = []
        self._holding = True

    def open_gripper(self, distance_m=0.09):
        self.calls.append(("open_gripper", float(distance_m)))
        self._holding = False

    def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0):
        self.calls.append(("move_to",))
        return True

    def wait_motion(self, duration, extra=0.6):
        pass

    def get_tcp_pose(self):
        return np.eye(4, dtype=np.float64)

    def grasp(self, force=None, timeout=5.0):
        self.calls.append(("grasp", force))
        self._holding = True
        return True

    def gripper_is_holding(self):
        return self._holding


class _SeqCamera:
    """Yields a fixed color/depth; the segmenter is what varies per frame."""

    def __init__(self, color, depth, K):
        self._c, self._d, self.K = color, depth, K

    def warm_up(self, n):
        pass

    def get_frame(self):
        return self._c, self._d


def test_multiframe_integration_aggregates_and_drops_outlier(monkeypatch):
    """Drive the real _capture_and_detect: the segmenter is mocked and the
    per-frame grasp is injected via a patched estimate_grasps that returns a
    jittered pose per call, including one gross outlier frame. The committed
    grasp must be the median of the inliers (outlier excluded)."""
    import ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp as og

    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 400, dtype=np.uint16)

    # All conf < 0.70 so the early-out never fires → aggregation runs.
    seq = [
        _pose(x=0.300, y=0.000, z=0.400, jaw=0.060, angle=10.0, conf=0.60),
        _pose(x=0.305, y=0.005, z=0.405, jaw=0.061, angle=12.0, conf=0.62),
        _pose(x=0.400, y=0.050, z=0.500, jaw=0.085, angle=11.0, conf=0.64),  # outlier
    ]
    calls = {"i": 0}

    def fake_estimate(results, depth_mm, Kl, **kw):
        i = min(calls["i"], len(seq) - 1)
        calls["i"] += 1
        return [seq[i]]

    monkeypatch.setattr(og, "estimate_grasps", fake_estimate)

    # A real YoloResult so _filter_results_to_target works; estimate_grasps is
    # mocked, so the mask/geometry is irrelevant — only the box class matters.
    from ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx import (
        YoloResult, _Box, _Boxes, _Masks,
    )

    def _result():
        mask = np.zeros((480, 640), dtype=np.float32)
        mask[210:270, 290:350] = 1.0
        return YoloResult(
            names={0: "box"},
            boxes=_Boxes([_Box([290, 210, 350, 270], 0, 0.6)]),
            masks=_Masks(np.stack([mask], axis=0)),
            orig_shape=(480, 640),
        )

    class _Seg:
        def predict(self, image, conf=0.25, iou=0.45, only_names=None):
            return [_result()]

    arm = _FakeArm()
    res = run_grasp_once(
        "box", arm=arm, segmenter=_Seg(), camera=_SeqCamera(color, depth, K),
        K=K, T_hand_eye=np.eye(4), warm_up_frames=0, move_duration=0.0,
        detect_frames=3, retries=0, reobserve=False, servo_correct=False,
    )
    assert res["success"] is True
    # The committed width is the MEDIAN of the two inliers (~0.0605m), proving
    # the gross outlier frame (jaw 0.085, +24mm) was rejected and did NOT pull
    # the aggregate. (grasp_pose x is post-transform/insertion, so the width is
    # the clean pre-transform witness of the aggregation result.)
    assert res["jaw_width_m"] == pytest.approx(0.0605, abs=2e-3)
    # The grasp point sits in the inlier region (~0.30m), not the outlier's
    # 0.40m → confirms the outlier position was excluded too.
    assert res["grasp_pose"][0] < 0.35
    assert calls["i"] == 3  # all three frames detected before any motion
