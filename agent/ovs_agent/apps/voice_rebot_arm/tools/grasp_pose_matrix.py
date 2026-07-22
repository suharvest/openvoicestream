"""Faithful pose-verification matrix for ONE box across rest-orientation × yaw ×
position × mask-bleed, using the now-audited synthetic harness (2026-06-18):

  * reachable() mirrors production (finalize_grasp_pose + pregrasp + side-pitch
    ladder),
  * render_box_depth(mask_bleed_px=...) reproduces the embin mask bleed,
  * the real TABLE height (z≈-0.05) is used,
  * a side-grasp HEAD-PITCH flag catches the shallow tilted-down bite.

Answers "which poses can this box be grasped in, and why the rest fail" without
the device. The one residual gap is the IK envelope being conservative on
low/level poses and the analytic scene having no far clutter (worst bleed) —
the real-frame replay (grasp_replay) covers those.

Run:
    python -m ovs_agent.apps.voice_rebot_arm.tools.grasp_pose_matrix
"""
from __future__ import annotations

import numpy as np

from ..perception.ordinary_grasp import estimate_grasps
from ..perception.grasp_geometry import finalize_grasp_pose
from ..perception.transforms import pose6d_to_mat4
from .synthetic_grasp_harness import (
    NoiseModel,
    default_K,
    default_T_cam2base,
    make_detection,
    reachable,
    render_box_depth,
    up_hint_from_extrinsic,
)

JAW_LIMIT = 0.088
TABLE_Z = -0.05            # real rig: table ~5cm below the arm base
HEAD_DOWN_MAX = 20.0       # deg; a side grip tilted more than this bites shallow
EDGES = (0.165, 0.085, 0.043)   # reComputer box edge lengths (m)


def _rest_orientations():
    a, b, c = EDGES
    return [
        ("flat   (165x85, 43 tall)", (a, b, c)),
        ("on-edge(165x43, 85 tall)", (a, c, b)),
        ("stand  ( 85x43,165 tall)", (b, c, a)),
    ]


def run():
    K = default_K(); T = default_T_cam2base(); up = up_hint_from_extrinsic(T)
    yaws = [0, 30, 45, 60, 90]
    positions = [(0.40, 0.0), (0.45, -0.08), (0.48, 0.08)]
    bleeds = [0, 25]   # clean vs realistic embin bleed
    counts = {"OK": 0, "too-wide": 0, "unreachable": 0, "shallow-tilt": 0,
              "no-grasp": 0}
    total = 0
    print(f"reComputer box edges {EDGES}m | jaw≤{JAW_LIMIT}m | table z={TABLE_Z}\n")
    for label, dims in _rest_orientations():
        short_h = min(dims[0], dims[1])
        print(f"=== {label}  short-footprint={short_h*100:.1f}cm "
              f"{'' if short_h<=JAW_LIMIT else '(TOO WIDE — ungraspable footprint)'} ===")
        for yaw in yaws:
            for (cx, cy) in positions:
                for bleed in bleeds:
                    total += 1
                    th = np.radians(yaw)
                    depth, mask = render_box_depth(
                        dims, (cx, cy, TABLE_Z, th), T, K,
                        noise=NoiseModel(seed=0), mask_bleed_px=bleed,
                    )
                    gs = [g for g in estimate_grasps(
                        [make_detection(mask, K, "box")], depth,
                        np.asarray(K, np.float64), up_hint_cam=up) if g.is_valid]
                    head_down = 0.0
                    reach_ok = False
                    if not gs:
                        verdict = "no-grasp"
                    else:
                        g = gs[0]
                        g6, _ = finalize_grasp_pose(g, T, 0.08, 0.025)
                        head_down = float(np.degrees(np.arcsin(
                            -np.asarray(pose6d_to_mat4(*g6), np.float64)[2, 0])))
                        reach_ok, _why = reachable(g, T)
                        # GEOMETRY verdict (faithful). Reachability is advisory —
                        # the CSV envelope is conservative on the low real table.
                        if g.jaw_width_m > JAW_LIMIT:
                            verdict = "too-wide"
                        elif g.method == "side_face" and head_down > HEAD_DOWN_MAX:
                            verdict = "shallow-tilt"
                        else:
                            verdict = "OK"
                    counts[verdict] = counts.get(verdict, 0) + 1
                    if bleed == 0:  # print clean row; bleed delta summarised below
                        m = gs[0].method if gs else "-"
                        jw = gs[0].jaw_width_m if gs else 0.0
                        flag = "OK " if verdict == "OK" else "XX "
                        rn = "reach✓" if reach_ok else "reach?"
                        print(f"  {flag}yaw={yaw:2d} pos=({cx:.2f},{cy:+.2f}) | "
                              f"{m:10s} jaw={jw*100:4.1f} head_down={head_down:3.0f}° "
                              f"| {verdict:11s} [{rn}]")
    print(f"\n=== SUMMARY ({total} cases incl. bleed variants) ===")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"  {k:14s}: {v}")


if __name__ == "__main__":
    run()
