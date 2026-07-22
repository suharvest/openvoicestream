"""Extension tests for the synthetic grasp harness (Mac, torch-free, no device).

Two additions on top of ``test_synthetic_grasp.py``:

  * ``test_noise_mode_reproduces_fusion`` — drive the D405-class
    :class:`NoiseModel` at a TALL FAR box and measure how wide the raw
    ``_top_face_grasp`` candidate gets under noise. The intent is to reproduce
    the real-machine top+side plane FUSION *naturally* (no value injection). In
    practice the upstream RANSAC + band-pass + erosion inside ``_top_face_grasp``
    robustly REJECTS the noisy fusion long before its width is computed, so even
    a swept set of aggressive noise params never produces an over-wide top
    (>0.085). We therefore REPORT the actual widest top width the noise reached
    and assert the WEAKER, still-load-bearing invariant: the production path
    (ordinary_grasp.py:161 guard) never emits a bogus over-wide ``top_face``,
    and the guard provably drops a >0.085 top.

  * ``test_dumped_frame_roundtrip`` — render a synthetic frame, persist it as
    ``color.jpg`` + ``depth.npy`` (+ ``K.npy``) to a tmp dir, reload via
    ``load_dumped_frame``, run ``plan_grasp_from_frame``, and assert the planned
    GraspPose matches the in-memory ``plan_grasp`` path (method + width).
"""

from __future__ import annotations

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.tools.synthetic_grasp_harness import (
    IMG_HW,
    NoiseModel,
    default_K,
    default_T_cam2base,
    load_dumped_frame,
    make_detection,
    plan_grasp,
    plan_grasp_from_frame,
    render_box_depth,
    up_hint_from_extrinsic,
)

TABLE_Z = 0.05


def _scene():
    return default_T_cam2base(), default_K()


def test_noise_mode_reproduces_fusion():
    """Sweep noise params on a TALL FAR box; report the widest raw top width the
    noise reaches and assert no over-wide top_face survives the production path.
    """
    from ovs_agent.apps.voice_rebot_arm.perception import ordinary_grasp as og

    T, K = _scene()
    up = up_hint_from_extrinsic(T)

    # Sweep (tall box height × far x × noise params). Each entry is
    # (dims, pose, noise_kwargs). The params escalate axial range-noise + edge
    # smear + band thickness — the three corruptions that physically fuse the
    # box top and upper side faces. Heights/x are the tall + far fusion regime.
    sweep = []
    for (h, x) in ((0.19, 0.55), (0.15, 0.55), (0.19, 0.50), (0.25, 0.55)):
        dims = (0.06, 0.06, h)
        pose = (x, 0.0, TABLE_Z, 0.0)
        for ps in (
            dict(seed=1),
            dict(seed=1, axial_b=0.006, edge_mix_m=0.05, edge_band_px=6),
            dict(seed=2, axial_b=0.008, edge_mix_m=0.03, edge_band_px=6, dropout_frac=0.02),
            dict(seed=3, axial_b=0.015, edge_mix_m=0.09, edge_band_px=18, dropout_frac=0.08),
            dict(seed=7, axial_b=0.020, edge_mix_m=0.12, edge_band_px=24, dropout_frac=0.10),
        ):
            sweep.append((dims, pose, ps))

    widest = 0.0
    widest_params = None
    for dims, pose, ps in sweep:
        nm = NoiseModel(**ps)
        depth_mm, mask = render_box_depth(dims, pose, T, K, IMG_HW, noise=nm)
        side_cands: list = []
        top = og._top_face_grasp(
            (mask > 0).astype(np.uint8), depth_mm,
            np.asarray(K, dtype=np.float64), np.asarray(up, dtype=np.float64),
            side_out=side_cands,
        )
        w = None if top is None else float(top[3])
        if w is not None:
            print(
                f"\n[noise-fusion] dims={dims} pose_x={pose[0]} params={ps} -> "
                f"raw top width={round(w, 4)} side_cands={len(side_cands)}"
            )
        if w is not None and w > widest:
            widest = w
            widest_params = (dims, pose[0], ps)

        # Whatever the noise did, the FULL production path must never emit a
        # bogus over-wide top_face (the line-161 guard is the safety net).
        g = plan_grasp(dims, pose, T, K, IMG_HW, noise=nm)
        if g is not None and g.method == "top_face":
            assert g.jaw_width_m <= 0.085 + 1e-6, (
                f"REGRESSION: noise produced top_face width {g.jaw_width_m:.4f} "
                f">0.085 (dims={dims} params={ps})"
            )

    print(
        f"\n[noise-fusion] WIDEST raw _top_face_grasp width reached under noise: "
        f"{widest:.4f} m  (params={widest_params})"
    )
    print(
        "[noise-fusion] upstream RANSAC + band-pass + erosion reject the noisy "
        "fusion before an over-wide top forms (widest stays < 0.085), so the "
        "test asserts the WEAKER invariant: the line-161 guard drops anything "
        ">0.085."
    )

    # WEAKER INVARIANT (fallback per task spec): directly verify the guard drops
    # a >0.085 top regardless of how it arose. Inject the real-machine 0.270 m
    # over-wide top (side_cands empty) and confirm the production path refuses
    # to surface it as a top_face.
    import unittest.mock as mock

    g_dims = (0.06, 0.06, 0.19)
    g_pose = (0.55, 0.0, TABLE_Z, 0.0)
    depth_mm, mask = render_box_depth(g_dims, g_pose, T, K, IMG_HW, noise=None)
    result = make_detection(mask, K, "box")
    centroid = np.array([0.0, 0.20, 0.50], dtype=np.float64)
    bogus_top = (
        centroid,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        0.270, 0.30, 500,
    )

    def _fake_top(mask_, depth_, K_, up_, side_out=None, **kw):
        return bogus_top

    with mock.patch.object(og, "_top_face_grasp", _fake_top):
        grasps = og.estimate_grasps(
            [result], depth_mm, np.asarray(K, dtype=np.float64),
            depth_quantile=0.5, up_hint_cam=up,
        )
    g_guarded = og.select_best_grasp(grasps)
    leaked = (
        g_guarded is not None
        and g_guarded.method == "top_face"
        and g_guarded.jaw_width_m > 0.085
    )
    print(
        f"[noise-fusion] guard isolation: injected 0.270 m top -> production "
        f"method={None if g_guarded is None else g_guarded.method} "
        f"(over-wide top leaked: {leaked})"
    )
    assert not leaked, "line-161 guard FAILED: over-wide top_face leaked"


def test_dumped_frame_roundtrip(tmp_path):
    """Render → save (color.jpg + depth.npy + K.npy) → load → plan_grasp_from_frame
    matches the in-memory plan_grasp path (method + width within tolerance).
    """
    import cv2

    T, K = _scene()
    dims = (0.12, 0.08, 0.04)  # flat box → clean top_face
    pose = (0.40, 0.0, TABLE_Z, 0.0)

    # in-memory reference path
    g_mem = plan_grasp(dims, pose, T, K, IMG_HW)
    assert g_mem is not None and g_mem.is_valid, "reference grasp should be valid"

    # render the frame and persist it as a real on-disk dump
    depth_mm, mask = render_box_depth(dims, pose, T, K, IMG_HW)
    # synthesize a plausible color image: gray table, lighter box silhouette
    color = np.full((IMG_HW[0], IMG_HW[1], 3), 60, dtype=np.uint8)
    color[mask > 0] = (200, 200, 200)

    cv2.imwrite(str(tmp_path / "color.jpg"), color)
    np.save(str(tmp_path / "depth.npy"), depth_mm)
    np.save(str(tmp_path / "K.npy"), np.asarray(K, dtype=np.float64))
    # a real dump carries the segmenter mask alongside the frame; persist it so
    # the round-trip detection matches the in-memory silhouette exactly (the
    # depth-median fallback is lossy and would drift the method label).
    np.save(str(tmp_path / "mask.npy"), mask)

    color_l, depth_l, K_l = load_dumped_frame(tmp_path, verbose=True)
    assert color_l.shape[:2] == IMG_HW
    assert depth_l.dtype == np.uint16
    assert np.allclose(K_l, K)
    mask_l = np.load(str(tmp_path / "mask.npy"))

    up = up_hint_from_extrinsic(T)
    g_frame = plan_grasp_from_frame(
        color_l, depth_l, K_l, up_hint_cam=up, segmenter=None, mask=mask_l
    )
    assert g_frame is not None and g_frame.is_valid, "dumped-frame grasp should be valid"

    print(
        f"\n[roundtrip] in-memory: method={g_mem.method} "
        f"width={g_mem.jaw_width_m:.4f} | from-frame: method={g_frame.method} "
        f"width={g_frame.jaw_width_m:.4f}"
    )

    # method label must match (both should land on top_face for this flat box)
    assert g_frame.method == g_mem.method, (
        f"method mismatch: frame={g_frame.method} vs mem={g_mem.method}"
    )
    # width within tolerance — the fallback depth-derived mask differs slightly
    # from the rendered silhouette, so allow a few-mm tolerance.
    assert abs(g_frame.jaw_width_m - g_mem.jaw_width_m) < 0.010, (
        f"width mismatch: frame={g_frame.jaw_width_m:.4f} vs "
        f"mem={g_mem.jaw_width_m:.4f}"
    )
