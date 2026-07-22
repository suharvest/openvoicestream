"""Replay the grasp pipeline on REAL captured depth frames (no device).

Feeds a fixture (color/depth/mask/K/up_hint, optionally T_cam2base) saved by
``grasp_selfcheck --save-frames`` through the PRODUCTION grasp geometry
(estimate_grasps → finalize_grasp_pose), then compares the computed grasp to an
INDEPENDENT ground truth measured from the same real depth (the graspable
face's centroid + short axis). This is the faithful, deterministic, device-free
validator the analytic synthetic harness can't be — it carries the real depth
artifacts and (with T_cam2base) the real hand-eye calibration.

Fixture dir contents (from grasp_selfcheck --save-frames):
  best_depth_mm.npy, best_mask.npy, K.npy, up_hint_cam.npy,
  [T_cam2base.npy] [best_color.jpg]

Usage:
  python -m ovs_agent.apps.voice_rebot_arm.tools.grasp_replay <fixture_dir>
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from ..perception.ordinary_grasp import estimate_grasps, select_best_grasp
from ..perception.grasp_geometry import finalize_grasp_pose
from .synthetic_grasp_harness import make_detection


def _n(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _ground_truth(depth_mm, mask, K, up_cam):
    """Independently measure the graspable face: centroid + short axis, from the
    real depth. Returns (centroid_cam, short_axis_cam, width_m, slab_kind).

    Isolates the top slab (highest ~1.5cm along gravity-up) when the object
    presents a flat top; falls back to the camera-facing slab otherwise. PCA in
    the slab plane → minor axis = jaw direction, 5-95pct minor extent = width.
    """
    m = cv2.erode((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(m > 0)
    z = depth_mm[ys, xs].astype(np.float64)
    ok = z > 0
    xs, ys, z = xs[ok], ys[ok], z[ok] / 1000.0
    z_med = float(np.median(z))
    band = np.abs(z - z_med) <= 0.12
    xs, ys, z = xs[band], ys[band], z[band]
    K = np.asarray(K, np.float64).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.column_stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z])
    up = _n(np.asarray(up_cam, np.float64))
    h = pts @ up
    top = h >= h.max() - 0.015
    slab = pts[top] if top.sum() >= 80 else pts
    c = slab.mean(0)
    rel = slab - c
    rel = rel - np.outer(rel @ up, up)  # project into the horizontal slab plane
    _u, _s, vt = np.linalg.svd(rel, full_matrices=False)
    minor = vt[1]
    minor_c = rel @ minor
    width = float(np.percentile(minor_c, 95) - np.percentile(minor_c, 5))
    return c, _n(minor), width, ("top_slab" if top.sum() >= 80 else "full")


@dataclass
class ReplayResult:
    detected: bool
    method: str
    conf: float
    jaw_width_m: float
    # ground truth (independent measurement from the real depth)
    gt_width_m: float
    gt_slab: str
    # geometry checks (camera frame — frame-independent)
    centre_offset_mm: float        # computed grasp centre vs true face centroid
    jaw_vs_short_axis_deg: float   # computed open axis vs true short axis
    # base frame (only when T_cam2base present)
    grasp6d: Optional[list] = None
    pre6d: Optional[list] = None


def replay(fixture_dir: str, insertion_depth_m: float = 0.025,
           pregrasp_offset_m: float = 0.08) -> ReplayResult:
    d = fixture_dir.rstrip("/")
    depth = np.load(f"{d}/best_depth_mm.npy")
    mask = np.load(f"{d}/best_mask.npy")
    K = np.load(f"{d}/K.npy")
    up = np.load(f"{d}/up_hint_cam.npy")
    try:
        T_cam2base = np.load(f"{d}/T_cam2base.npy")
    except FileNotFoundError:
        T_cam2base = None

    det = make_detection((mask > 0).astype(np.uint8), K, "box")
    cands = [g for g in estimate_grasps([det], depth, np.asarray(K, np.float64),
                                        up_hint_cam=up) if g.is_valid]
    best = select_best_grasp(cands)
    if best is None:
        return ReplayResult(False, "none", 0.0, 0.0, 0.0, "n/a", float("nan"),
                            float("nan"))

    gt_c, gt_short, gt_w, gt_slab = _ground_truth(depth, mask, K, up)

    # computed grasp centre vs true face centroid (camera frame, mm)
    centre_off = float(np.linalg.norm(np.asarray(best.position, np.float64) - gt_c) * 1000)
    # computed jaw open axis (tcp_rotation col 1) vs true short axis
    open_cam = np.asarray(best.tcp_rotation, np.float64)[:, 1]
    cosang = abs(float(_n(open_cam) @ _n(gt_short)))
    jaw_deg = float(np.degrees(np.arccos(min(1.0, cosang))))

    g6 = p6 = None
    if T_cam2base is not None:
        g6, p6 = finalize_grasp_pose(best, T_cam2base, pregrasp_offset_m,
                                     insertion_depth_m)

    return ReplayResult(
        True, getattr(best, "method", "legacy"), float(best.conf),
        float(best.jaw_width_m), gt_w, gt_slab, centre_off, jaw_deg, g6, p6,
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: grasp_replay <fixture_dir>", file=sys.stderr)
        return 2
    r = replay(sys.argv[1])
    print(json.dumps(asdict(r), ensure_ascii=False, indent=2))
    if not r.detected:
        print("\nNO DETECTION", file=sys.stderr)
        return 1
    print(f"\nmethod={r.method} conf={r.conf:.2f}")
    print(f"jaw_width: pipeline={r.jaw_width_m*100:.1f}cm  ground-truth={r.gt_width_m*100:.1f}cm")
    print(f"grasp-centre offset from true centroid: {r.centre_offset_mm:.1f} mm")
    print(f"jaw vs true short-axis: {r.jaw_vs_short_axis_deg:.1f} deg")
    if r.grasp6d is not None:
        print(f"grasp6d (base): {[round(v,3) for v in r.grasp6d]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
