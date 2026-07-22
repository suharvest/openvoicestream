"""GG-CNN refiner — runs the REAL exported ONNX (tools/artifacts) on
synthetic depth scenes; plus the consistency arbitration votes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.perception.ggcnn_refiner import (
    GgcnnGrasp,
    GgcnnRefiner,
    consistent,
)

ART = (Path(__file__).resolve().parents[1]
       / "ovs_agent/apps/voice_rebot_arm/tools/artifacts/ggcnn2-300.onnx")

pytestmark = pytest.mark.skipif(not ART.exists(), reason="ggcnn artifact missing")


def _scene_with_bar(angle_deg: float = 0.0):
    """A raised bar (graspable ridge) on a flat table plane, 400mm camera
    distance, bar 30mm tall. GG-CNN should place its best grasp on the bar."""
    h, w = 480, 640
    depth = np.full((h, w), 430, dtype=np.uint16)   # table at 430mm
    mask = np.zeros((h, w), dtype=np.uint8)
    import cv2
    canvas = np.zeros((h, w), dtype=np.uint8)
    center = (320, 240)
    rect = ((center), (200, 46), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(canvas, [box], 1)
    depth[canvas > 0] = 400                          # bar 30mm above table
    mask[canvas > 0] = 1
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    return depth, mask, K


def test_real_onnx_grasps_the_bar_inside_mask():
    refiner = GgcnnRefiner(str(ART))
    depth, mask, K = _scene_with_bar(0.0)
    g = refiner.predict(depth, mask, K)
    assert g is not None
    # grasp point lands ON the bar (inside the mask).
    assert mask[g.center_px[1], g.center_px[0]] == 1
    assert g.quality > 0
    assert 0.3 < g.depth_m < 0.5
    # metric width positive and below the crop scale sanity bound.
    assert 0.0 < g.width_m < 0.3


def test_angle_tracks_bar_orientation():
    refiner = GgcnnRefiner(str(ART))
    angles = {}
    for a in (0.0, 60.0):
        depth, mask, K = _scene_with_bar(a)
        g = refiner.predict(depth, mask, K)
        assert g is not None
        angles[a] = np.degrees(g.angle_rad) % 180.0
    # the two predicted angles must differ — the model reacts to orientation
    # (exact values are model-internal; we assert sensitivity, not identity).
    da = abs(angles[0.0] - angles[60.0]) % 180.0
    assert min(da, 180.0 - da) > 15.0


def test_none_when_mask_empty_or_artifact_bad(tmp_path):
    refiner = GgcnnRefiner(str(ART))
    depth, mask, K = _scene_with_bar()
    assert refiner.predict(depth, np.zeros_like(mask), K) is None
    bad = GgcnnRefiner(str(tmp_path / "missing.onnx"))
    assert bad.predict(depth, mask, K) is None


def test_consistency_vote():
    gg = GgcnnGrasp(center_px=(0, 0), quality=0.9,
                    angle_rad=np.radians(10.0), width_m=0.06, depth_m=0.4)
    assert consistent(plane_angle_deg=20.0, plane_width_m=0.07, gg=gg)        # within tol
    assert consistent(plane_angle_deg=190.0, plane_width_m=0.07, gg=gg)       # mod-180
    assert not consistent(plane_angle_deg=80.0, plane_width_m=0.06, gg=gg)    # angle off
    assert not consistent(plane_angle_deg=10.0, plane_width_m=0.10, gg=gg)    # width off
