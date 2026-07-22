"""Torch-free synthetic-depth grasp REGRESSION harness (Mac, no GPU/camera/robot).

Turns the reBot grasp-PLANNING pipeline
(:mod:`perception.ordinary_grasp` + :mod:`perception.transforms`) into a fast,
deterministic, device-free function: render a metric oriented box on a table,
build the real :class:`perception.yolo_onnx.YoloResult`, run the production
estimator, transform to the base frame, and score reachability against the
*measured* IK envelope CSV. No ONNX model is run; no production code is touched.

Why analytic rendering (not Open3D/trimesh): neither is installed on this Mac,
and the geometry here is a single convex box + a planar table — a per-face
z-buffer with exact pinhole projection is both cleaner and trivially verifiable
(see ``test_renderer_backprojection_sane`` in the test module). Adds no deps;
uses only numpy + cv2 (already production deps).

────────────────────────────────────────────────────────────────────────────
ASSUMED CALIBRATION (no real hand_eye.npz / intrinsics.npz on this Mac — the
production paths read ``/opt/rebot-models/hand_eye.npz`` and the live Orbbec
SDK, neither present here). These are SYNTHETIC but self-consistent; swap them
for the real calibration when running against a device. The harness's value is
*relative* geometric consistency (render → estimate → transform round-trips),
not absolute calibration accuracy.

  * Intrinsics K (D405-class, 1280x720 — matches config.yaml color_width/height):
        fx = fy = 640.0,  cx = 640.0,  cy = 360.0
    (RealSense D405 native is ~1.93 px/deg-ish; 640 fx at 1280 wide ≈ 90° HFOV,
    a sane wide-baseline depth FOV. Documented so it can be replaced with the
    SDK's live K.)
  * Image size: 720 (H) x 1280 (W).
  * Eye-in-hand: ``T_cam2base = tcp_pose @ T_hand_eye`` (production convention,
    grasp_service.py:479). We synthesize T_cam2base DIRECTLY as a wrist camera
    looking DOWN-AND-FORWARD at the workspace from above — pitched ~50° below
    horizontal, mounted ~0.45 m up and slightly behind the work zone — which is
    the geometry the real observation TCP pose produces. Camera +Z (optical
    axis) points down-forward into the table; camera +X points to base +Y
    (image right), camera +Y points down-into-table. See
    :func:`default_T_cam2base`.

These choices put a box at base x≈0.35–0.5, z≈box_height squarely in the
camera's view and on the IK grid (x∈[0.25,0.55]), reproducing the real
observation pose's framing.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..perception.ordinary_grasp import (
    GraspPose,
    estimate_grasps,
    select_best_grasp,
)
from ..perception.transforms import pose6d_to_mat4, transform_grasp_pose_to_base
from ..perception.yolo_onnx import YoloResult, _Box, _Boxes, _Masks


# ── assumed calibration constants ────────────────────────────────────────────
IMG_HW: tuple[int, int] = (720, 1280)  # (H, W) — config.yaml color 1280x720

#: D405-class intrinsics at 1280x720 (see module docstring).
DEFAULT_K: np.ndarray = np.array(
    [
        [640.0, 0.0, 640.0],
        [0.0, 640.0, 360.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
_IK_CSV = _ARTIFACTS / "ik_envelope_b601dm.csv"


# ── D405-class depth noise model ─────────────────────────────────────────────
@dataclass
class NoiseModel:
    """RealSense D405-class depth-noise model applied to a CLEAN rendered depth.

    Three independent corruptions, each documented and seeded so a test is
    reproducible (RNG is ``np.random.default_rng(seed)``):

      1. **Axial (range) noise** — per-pixel Gaussian along the optical axis,
         std growing with range:  ``σ(z) = axial_a + axial_b·z²``  (metres).
         D405 datasheet quotes ≲ 2 % of range RMS at the working distance; the
         quadratic ``b·z²`` term reproduces the characteristic stereo-depth
         degradation with distance while ``a`` is a small near-range floor.
         Defaults a=0.001, b=0.0025 → at z=0.5 m σ≈1.6 mm, at z=0.9 m σ≈3 mm.
         This is the dominant fusion driver: it puffs the box-top points and the
         upper-side points toward a common slanted plane.

      2. **Lateral edge noise** — at depth discontinuities (the box silhouette,
         found by a morphological gradient of the valid-depth support), each
         edge band pixel is jittered toward EITHER the near (box) or far (table)
         surface by a few-pixel-equivalent depth perturbation. This is the
         "flying pixels / mixed pixels" effect that physically fuses the top and
         side faces into one cloud — exactly the corruption the 8fb88ac guard
         exists to survive. ``edge_band_px`` controls the band thickness and
         ``edge_mix_m`` the magnitude of the near/far smear.

      3. **Invalid dropout** — a fraction ``dropout_frac`` of valid pixels are
         zeroed (depth holes), like real specular / low-IR returns.

    Defaults reproduce the real-machine top+side fusion on a tall far box WITHOUT
    any value injection (see ``test_noise_mode_reproduces_fusion``).
    """

    axial_a: float = 0.001       # near-range σ floor (m)
    axial_b: float = 0.0025      # quadratic range coefficient (m per m²)
    edge_band_px: int = 6        # silhouette band thickness (px)
    edge_mix_m: float = 0.030    # near/far smear magnitude at edges (m)
    dropout_frac: float = 0.03   # fraction of valid pixels zeroed
    seed: int = 0

    def apply(self, depth_m: np.ndarray) -> np.ndarray:
        """Corrupt a clean float metres depth map (0 = invalid). Returns metres."""
        import cv2

        rng = np.random.default_rng(self.seed)
        out = depth_m.copy()
        valid = out > 0

        # (2) lateral edge noise FIRST (operates on the clean silhouette so the
        # discontinuity location is crisp). Edge band = morphological gradient of
        # the valid-support mask ∩ a depth-jump test against neighbours.
        vmask = valid.astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        grad = cv2.morphologyEx(vmask, cv2.MORPH_GRADIENT, k)  # support boundary
        # also catch internal box/table depth jumps (top-vs-table seam):
        dz = cv2.morphologyEx(
            out.astype(np.float32), cv2.MORPH_GRADIENT, k
        )
        jump = (dz > 0.02) & valid  # >2cm local jump → a real surface edge
        band = cv2.dilate(
            ((grad > 0) | jump).astype(np.uint8),
            np.ones((self.edge_band_px, self.edge_band_px), np.uint8),
        )
        band = (band > 0) & valid
        n_band = int(band.sum())
        if n_band:
            # half pulled toward NEAR surface, half toward FAR — a bimodal smear
            # that drags box-top points down onto the side and vice versa.
            smear = rng.uniform(-self.edge_mix_m, self.edge_mix_m, size=n_band)
            out[band] = out[band] + smear
            out[band] = np.clip(out[band], 1e-3, None)

        # (1) axial Gaussian noise on ALL valid pixels, σ(z)=a+b·z².
        valid = out > 0
        z = out[valid]
        sigma = self.axial_a + self.axial_b * z * z
        out[valid] = z + rng.normal(0.0, 1.0, size=z.shape) * sigma
        out[out < 1e-3] = 0.0

        # (3) random invalid dropout (holes).
        valid = out > 0
        vidx = np.flatnonzero(valid)
        if len(vidx):
            n_drop = int(round(self.dropout_frac * len(vidx)))
            if n_drop:
                drop = rng.choice(vidx, size=n_drop, replace=False)
                out.reshape(-1)[drop] = 0.0
        return out


def default_K() -> np.ndarray:
    return DEFAULT_K.copy()


def default_T_cam2base(
    cam_height_m: float = 0.45,
    cam_setback_m: float = -0.10,
    look_pitch_deg: float = 50.0,
) -> np.ndarray:
    """Synthetic eye-in-hand extrinsic: wrist camera looking down-forward.

    Builds ``T_cam2base`` (base ← camera) directly. Camera axes in base frame:
      * optical +Z aims DOWN-AND-FORWARD into the workspace (pitched
        ``look_pitch_deg`` below the base +X horizon),
      * image +X (right) → base +Y,
      * image +Y (down) → into the table (down-forward).
    Camera origin sits ``cam_height_m`` above the base plane and ``cam_setback_m``
    along base +X (negative = slightly behind the work zone).
    """
    p = np.radians(look_pitch_deg)
    # Optical axis (camera +Z) in base frame: forward (+X) tilted down (−Z).
    z_cam = np.array([np.cos(p), 0.0, -np.sin(p)], dtype=np.float64)
    z_cam /= np.linalg.norm(z_cam)
    # Image right (camera +X) → base +Y.
    x_cam = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    # Camera +Y = Z × X (completes right-handed frame; points down-forward).
    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)
    x_cam = np.cross(y_cam, z_cam)
    x_cam /= np.linalg.norm(x_cam)

    R = np.column_stack([x_cam, y_cam, z_cam])  # base ← camera rotation
    t = np.array([cam_setback_m, 0.0, cam_height_m], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def up_hint_from_extrinsic(T_cam2base: np.ndarray) -> np.ndarray:
    """Base +Z (world up) expressed in the camera frame.

    ``up_cam = R_cam2base.T @ [0,0,1]`` = third ROW of the rotation block.
    Production computes the same as ``(tcp @ hand_eye)[:3,:3]`` then uses the
    up-direction; here T_cam2base already folds both.
    """
    R = np.asarray(T_cam2base, dtype=np.float64)[:3, :3]
    up_base = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_cam = R.T @ up_base
    return up_cam.astype(np.float64)


# ── box geometry ─────────────────────────────────────────────────────────────
def _box_corners_base(
    box_dims_m: tuple[float, float, float],
    box_pose_base: tuple[float, float, float, float],
) -> np.ndarray:
    """8 corners of an axis-pose box resting ON the table (base frame).

    ``box_dims_m = (Lx, Ly, Lz)`` are full extents along the box's local X/Y/Z.
    ``box_pose_base = (cx, cy, table_z, yaw)``: center XY, the table surface z
    the box sits on, and yaw about base +Z. The box bottom is at ``table_z``;
    the box spans ``table_z .. table_z + Lz`` vertically.
    """
    lx, ly, lz = box_dims_m
    cx, cy, table_z, yaw = box_pose_base
    hx, hy = lx / 2.0, ly / 2.0
    # local corners (z from 0 = bottom to lz = top)
    local = np.array(
        [
            [-hx, -hy, 0.0],
            [hx, -hy, 0.0],
            [hx, hy, 0.0],
            [-hx, hy, 0.0],
            [-hx, -hy, lz],
            [hx, -hy, lz],
            [hx, hy, lz],
            [-hx, hy, lz],
        ],
        dtype=np.float64,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    world = (Rz @ local.T).T + np.array([cx, cy, table_z], dtype=np.float64)
    return world


# 6 faces as quads (corner index order, CCW); we don't need winding for z-buffer.
_BOX_FACES = (
    (0, 1, 2, 3),  # bottom
    (4, 5, 6, 7),  # top
    (0, 1, 5, 4),  # -Y side
    (1, 2, 6, 5),  # +X side
    (2, 3, 7, 6),  # +Y side
    (3, 0, 4, 7),  # -X side
)


def _to_cam(points_base: np.ndarray, T_cam2base: np.ndarray) -> np.ndarray:
    """base → camera frame (T_base2cam = inv(T_cam2base))."""
    T = np.asarray(T_cam2base, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    # inverse of a rigid transform
    return (R.T @ (np.asarray(points_base, dtype=np.float64).T - t.reshape(3, 1))).T


def _project_cam(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """camera-frame 3D → pixel (u,v); returns (N,2). z must be > 0."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = points_cam[:, 0] * fx / z + cx
    v = points_cam[:, 1] * fy / z + cy
    return np.column_stack([u, v])


def _rasterize_quad_zbuffer(
    quad_cam: np.ndarray,
    K: np.ndarray,
    depth_buf: np.ndarray,
    id_buf: Optional[np.ndarray],
    face_id: int,
) -> None:
    """Z-buffer one planar quad (4 cam-frame corners) into depth_buf (metres).

    Per-pixel exact depth: solve the supporting plane in camera frame, then for
    each pixel inside the projected polygon recover z analytically from the ray
    so the depth is the TRUE surface z (not interpolated corner z). This keeps
    back-projection exact regardless of perspective foreshortening.
    """
    import cv2  # local — production dep, kept off module import for clarity

    H, W = depth_buf.shape
    if np.any(quad_cam[:, 2] <= 1e-4):
        return  # behind / at camera — skip (boxes are well in front)

    # Plane through the quad in camera frame: n·X = d.
    p0, p1, p2 = quad_cam[0], quad_cam[1], quad_cam[2]
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return
    n = n / nn
    d = float(n @ p0)

    uv = _project_cam(quad_cam, K)
    poly = np.round(uv).astype(np.int32)
    x0 = max(int(np.floor(uv[:, 0].min())), 0)
    x1 = min(int(np.ceil(uv[:, 0].max())), W - 1)
    y0 = max(int(np.floor(uv[:, 1].min())), 0)
    y1 = min(int(np.ceil(uv[:, 1].max())), H - 1)
    if x1 < x0 or y1 < y0:
        return

    # fill polygon mask in the bbox sub-window
    sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    cv2.fillConvexPoly(sub, poly - np.array([x0, y0], dtype=np.int32), 1)
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return
    us = xs + x0
    vs = ys + y0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # ray dir (un-normalized) for each pixel: r = ((u-cx)/fx, (v-cy)/fy, 1)
    rx = (us - cx) / fx
    ry = (vs - cy) / fy
    # z = d / (n·r)  where r=(rx,ry,1)
    denom = n[0] * rx + n[1] * ry + n[2]
    valid = np.abs(denom) > 1e-9
    z = np.full(len(us), np.inf)
    z[valid] = d / denom[valid]
    z[z <= 0] = np.inf

    flat_idx = vs * W + us
    cur = depth_buf.reshape(-1)[flat_idx]
    cur_inf = np.where(cur > 0, cur, np.inf)
    closer = z < cur_inf
    upd = flat_idx[closer]
    depth_buf.reshape(-1)[upd] = z[closer]
    if id_buf is not None:
        id_buf.reshape(-1)[upd] = face_id


def render_box_depth(
    box_dims_m: tuple[float, float, float],
    box_pose_base: tuple[float, float, float, float],
    T_cam2base: np.ndarray,
    K: np.ndarray,
    img_hw: tuple[int, int] = IMG_HW,
    noise: Optional[NoiseModel] = None,
    mask_bleed_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a metric oriented box on a table plane via analytic z-buffer.

    Returns ``(depth_mm, mask)``:
      * ``depth_mm``: uint16 millimetres, 0 = invalid. Includes BOTH the table
        surface (z = box_pose_base[2], the plane the box rests on) and the box
        faces, so the estimator's plane-fit / depth-band logic sees realistic
        context.
      * ``mask``: uint8 HxW, 1 over the box silhouette ONLY (table excluded).

    ``noise=None`` (default) keeps the exact clean behaviour (existing tests
    stay byte-identical). Pass a :class:`NoiseModel` to corrupt the float-metres
    depth (axial Gaussian + lateral edge smear + dropout) BEFORE uint16-mm
    quantization — reproducing D405 sensor depth without value injection.
    """
    H, W = img_hw
    depth_buf = np.zeros((H, W), dtype=np.float64)  # metres; 0 = empty
    id_buf = np.full((H, W), -1, dtype=np.int32)  # 0=table, 1..6 box faces

    table_z = float(box_pose_base[2])

    # ── table plane: a large quad at z = table_z spanning the work zone ──
    # rendered as face_id 0; box faces get ids 1..6.
    half = 0.6
    cx_b, cy_b = float(box_pose_base[0]), float(box_pose_base[1])
    table_quad_base = np.array(
        [
            [cx_b - half, cy_b - half, table_z],
            [cx_b + half, cy_b - half, table_z],
            [cx_b + half, cy_b + half, table_z],
            [cx_b - half, cy_b + half, table_z],
        ],
        dtype=np.float64,
    )
    table_cam = _to_cam(table_quad_base, T_cam2base)
    _rasterize_quad_zbuffer(table_cam, K, depth_buf, id_buf, face_id=0)

    # ── box faces ──
    corners = _box_corners_base(box_dims_m, box_pose_base)
    corners_cam = _to_cam(corners, T_cam2base)
    for fi, face in enumerate(_BOX_FACES, start=1):
        quad = corners_cam[list(face)]
        _rasterize_quad_zbuffer(quad, K, depth_buf, id_buf, face_id=fi)

    if noise is not None:
        depth_buf = noise.apply(depth_buf)

    depth_mm = np.zeros((H, W), dtype=np.uint16)
    valid = depth_buf > 0
    depth_mm[valid] = np.clip(
        np.round(depth_buf[valid] * 1000.0), 0, 65535
    ).astype(np.uint16)

    mask = ((id_buf >= 1) & (id_buf <= 6)).astype(np.uint8)
    if mask_bleed_px > 0:
        # SEGMENTATION BLEED (2026-06-18 sim-vs-real audit, gap #1): the real
        # embin mask spills onto the surrounding table/background (captured
        # frames: mask 2-4× the box, masked-depth spread 0.2-0.6m vs the clean
        # silhouette's ~0.03m), which inflated the legacy width to 0.16-0.50m
        # and was wrongly rejected. The clean synthetic mask is the exact box
        # silhouette → it never reproduces this. Dilate the box mask onto
        # neighbouring rendered pixels (table + its recession to farther depth).
        # NOTE: the analytic scene has only box+table (no far clutter), so this
        # under-models the worst real bleed — the REAL-frame replay
        # (grasp_replay) remains the faithful width check.
        import cv2
        k = np.ones((int(mask_bleed_px), int(mask_bleed_px)), np.uint8)
        mask = cv2.dilate(mask, k)
        mask = (mask & (depth_mm > 0).astype(np.uint8)).astype(np.uint8)
    return depth_mm, mask


# ── analytic ray-cast renderers for non-box shapes ───────────────────────────
# Box rendering above z-buffers planar quads. Curved shapes (capsule/banana,
# sphere/orange, cylinder/bottle) have no quad decomposition, so they are
# rendered the dual way: cast ONE camera ray per pixel (exact pinhole) and
# intersect it analytically with the surface + the table plane, keeping the
# nearest positive hit. Same DEFAULT_K / T_cam2base convention as the box, so
# all renderers share the harness's extrinsic and the back-projection round-trip
# stays exact. Torch-free: pure numpy closed-form intersections.
def _camera_rays_base(
    T_cam2base: np.ndarray, K: np.ndarray, img_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel ray origin (camera center, base frame) + unit ray directions
    (base frame), plus camera +Z (base) for converting a base hit → camera z.

    Returns ``(origin (3,), dirs (H,W,3) unit, z_axis_base (3,))``.
    """
    H, W = img_hw
    T = np.asarray(T_cam2base, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    rx = (us - cx) / fx
    ry = (vs - cy) / fy
    rz = np.ones_like(rx)
    dir_cam = np.stack([rx, ry, rz], axis=-1)  # (H,W,3), z=1 (NOT unit)
    dir_base = dir_cam @ R.T  # rotate to base frame
    dir_base /= np.linalg.norm(dir_base, axis=-1, keepdims=True)
    z_axis_base = R[:, 2]  # camera optical axis in base frame
    return t, dir_base, z_axis_base


def _finalize_depth_mask(
    t_obj: np.ndarray,
    t_table: np.ndarray,
    origin: np.ndarray,
    dirs: np.ndarray,
    z_axis_base: np.ndarray,
    noise: Optional[NoiseModel],
) -> tuple[np.ndarray, np.ndarray]:
    """Shared tail: pick nearest hit (object vs table), convert to camera z,
    quantize to uint16 mm, build the object-only mask. ``t_*`` are ray params
    (∞ = miss). All in base frame; camera z = (hit - origin)·z_axis_base.
    """
    H, W, _ = dirs.shape
    obj_hit = np.isfinite(t_obj)
    table_hit = np.isfinite(t_table)
    # object wins where it is the nearer (smaller t) valid hit.
    obj_nearest = obj_hit & (~table_hit | (t_obj <= t_table))
    t_use = np.where(obj_nearest, t_obj, np.where(table_hit, t_table, np.inf))
    hit = np.isfinite(t_use)
    depth_m = np.zeros((H, W), dtype=np.float64)
    # camera-frame z of the hit point = t · (dir·z_axis) since |dir|=1.
    dir_dot_z = dirs @ z_axis_base  # (H,W)
    zc = t_use * dir_dot_z
    depth_m[hit] = zc[hit]
    depth_m[depth_m <= 0] = 0.0

    if noise is not None:
        depth_m = noise.apply(depth_m)

    depth_mm = np.zeros((H, W), dtype=np.uint16)
    valid = depth_m > 0
    depth_mm[valid] = np.clip(
        np.round(depth_m[valid] * 1000.0), 0, 65535
    ).astype(np.uint16)
    mask = obj_nearest.astype(np.uint8)
    return depth_mm, mask


def render_sphere_depth(
    center_base: tuple[float, float, float],
    radius_m: float,
    table_z: float,
    T_cam2base: np.ndarray,
    K: np.ndarray,
    img_hw: tuple[int, int] = IMG_HW,
    noise: Optional[NoiseModel] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a sphere (orange) resting on a table, analytic ray-sphere hit."""
    origin, dirs, z_axis = _camera_rays_base(T_cam2base, K, img_hw)
    c = np.asarray(center_base, dtype=np.float64)
    oc = origin - c  # (3,)
    b = dirs @ oc  # (H,W)
    cc = float(oc @ oc) - radius_m * radius_m
    disc = b * b - cc
    t_obj = np.full(dirs.shape[:2], np.inf)
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t0 = -b - sq
    t_obj[ok & (t0 > 0)] = t0[ok & (t0 > 0)]
    t_table = _ray_table(origin, dirs, z_axis, table_z)
    return _finalize_depth_mask(t_obj, t_table, origin, dirs, z_axis, noise)


def _ray_table(
    origin: np.ndarray, dirs: np.ndarray, z_axis: np.ndarray, table_z: float
) -> np.ndarray:
    """Ray param where each ray crosses the horizontal plane z=table_z (base)."""
    dz = dirs[..., 2]
    t = np.full(dirs.shape[:2], np.inf)
    moving = np.abs(dz) > 1e-9
    cand = (table_z - origin[2]) / np.where(moving, dz, 1.0)
    t[moving & (cand > 0)] = cand[moving & (cand > 0)]
    return t


def render_cylinder_depth(
    center_base: tuple[float, float, float],
    radius_m: float,
    length_m: float,
    axis_base: tuple[float, float, float],
    table_z: float,
    T_cam2base: np.ndarray,
    K: np.ndarray,
    img_hw: tuple[int, int] = IMG_HW,
    noise: Optional[NoiseModel] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a finite cylinder (bottle lying on its side) on a table.

    ``axis_base`` is the (unnormalized) cylinder-axis direction; ``length_m`` is
    the full length, ``radius_m`` the radius (diameter = 2·radius). Analytic
    ray-vs-infinite-cylinder, clamped to the finite extent along the axis.
    """
    origin, dirs, z_axis = _camera_rays_base(T_cam2base, K, img_hw)
    c = np.asarray(center_base, dtype=np.float64)
    a = np.asarray(axis_base, dtype=np.float64)
    a = a / np.linalg.norm(a)
    H, W, _ = dirs.shape

    # project ray dir and origin offset onto the plane ⊥ axis.
    d = dirs.reshape(-1, 3)
    dp = d - np.outer(d @ a, a)  # (N,3) dir ⊥ axis
    oc = origin - c
    op = oc - (oc @ a) * a  # (3,) offset ⊥ axis
    A = np.einsum("ij,ij->i", dp, dp)
    B = 2.0 * (dp @ op)
    C = float(op @ op) - radius_m * radius_m
    disc = B * B - 4 * A * C
    t = np.full(len(d), np.inf)
    ok = (disc >= 0) & (A > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t0 = (-B - sq) / np.where(A > 1e-12, 2 * A, 1.0)
    sel = ok & (t0 > 0)
    # clamp to finite length: axial coordinate of the hit within ±length/2.
    hit = origin + t0[:, None] * d
    axial = (hit - c) @ a
    sel &= np.abs(axial) <= (length_m / 2.0)
    t[sel] = t0[sel]
    t_obj = t.reshape(H, W)
    t_table = _ray_table(origin, dirs, z_axis, table_z)
    return _finalize_depth_mask(t_obj, t_table, origin, dirs, z_axis, noise)


def render_capsule_depth(
    center_base: tuple[float, float, float],
    radius_m: float,
    length_m: float,
    axis_base: tuple[float, float, float],
    table_z: float,
    T_cam2base: np.ndarray,
    K: np.ndarray,
    img_hw: tuple[int, int] = IMG_HW,
    curve_m: float = 0.0,
    noise: Optional[NoiseModel] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render an elongated capsule (banana-ish) on a table.

    A capsule = a cylinder of length ``length_m`` with hemispherical caps of
    ``radius_m``; the surface is the locus of points at distance ``radius_m``
    from the central SEGMENT. With ``curve_m>0`` the segment becomes a shallow
    parabolic arc (a banana bend) sampled into a polyline; the ray-vs-capsule
    distance is the min distance to any segment of the polyline. Analytic per
    segment (point-to-segment closest approach), torch-free.
    """
    origin, dirs, z_axis = _camera_rays_base(T_cam2base, K, img_hw)
    c = np.asarray(center_base, dtype=np.float64)
    a = np.asarray(axis_base, dtype=np.float64)
    a = a / np.linalg.norm(a)
    # build the (possibly curved) spine polyline as points along the axis with a
    # parabolic bend in the local up direction (perpendicular to axis & base-Z).
    bend_dir = np.cross(a, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(bend_dir) < 1e-6:
        bend_dir = np.array([1.0, 0.0, 0.0])
    bend_dir = bend_dir / np.linalg.norm(bend_dir)
    bend_up = np.cross(bend_dir, a)  # curve upward, away from table
    bend_up = bend_up / max(np.linalg.norm(bend_up), 1e-9)
    n_seg = 24
    s = np.linspace(-length_m / 2.0, length_m / 2.0, n_seg + 1)
    # parabola: offset = curve_m · (1 - (2s/L)²)
    off = curve_m * (1.0 - (2.0 * s / max(length_m, 1e-9)) ** 2)
    spine = c[None, :] + np.outer(s, a) + np.outer(off, bend_up)  # (n_seg+1,3)

    H, W, _ = dirs.shape
    d = dirs.reshape(-1, 3)
    N = len(d)
    t_best = np.full(N, np.inf)
    # For each spine segment, closed-form min over t of dist(ray(t), segment).
    # Cheap robust approach: sample the segment densely and do ray-vs-sphere on
    # each sample (sphere radius = capsule radius). n_seg×ray-sphere, vectorized.
    for p in spine:
        oc = origin - p
        b = d @ oc
        cc = float(oc @ oc) - radius_m * radius_m
        disc = b * b - cc
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t0 = -b - sq
        sel = ok & (t0 > 0) & (t0 < t_best)
        t_best[sel] = t0[sel]
    t_obj = t_best.reshape(H, W)
    t_table = _ray_table(origin, dirs, z_axis, table_z)
    return _finalize_depth_mask(t_obj, t_table, origin, dirs, z_axis, noise)


def make_detection(
    box_mask: np.ndarray,
    K: np.ndarray,
    class_name: str = "box",
) -> YoloResult:
    """Build a real YoloResult with one instance from the rendered box mask."""
    ys, xs = np.nonzero(box_mask > 0)
    if len(xs) == 0:
        raise ValueError("empty box mask — nothing rendered into view")
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    box = _Box(np.array([x1, y1, x2, y2], dtype=np.float32), cls_id=0, conf=0.9)
    boxes = _Boxes([box])
    masks = _Masks(np.asarray(box_mask, dtype=np.float32)[None, ...])  # (1,H,W)
    H, W = box_mask.shape
    return YoloResult(
        names={0: class_name},
        boxes=boxes,
        masks=masks,
        orig_shape=(H, W),
    )


def plan_grasp(
    box_dims: tuple[float, float, float],
    box_pose: tuple[float, float, float, float],
    T_cam2base: np.ndarray,
    K: np.ndarray,
    img_hw: tuple[int, int] = IMG_HW,
    depth_quantile: float = 0.5,
    with_up_hint: bool = True,
    noise: Optional[NoiseModel] = None,
) -> Optional[GraspPose]:
    """Glue: render → make detection → estimate_grasps → select_best_grasp."""
    depth_mm, mask = render_box_depth(
        box_dims, box_pose, T_cam2base, K, img_hw, noise=noise
    )
    result = make_detection(mask, K, class_name="box")
    up_hint = up_hint_from_extrinsic(T_cam2base) if with_up_hint else None
    grasps = estimate_grasps(
        [result],
        depth_mm,
        np.asarray(K, dtype=np.float64),
        depth_quantile=depth_quantile,
        up_hint_cam=up_hint,
    )
    return select_best_grasp(grasps)


def plan_grasp_from_depth_mask(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    T_cam2base: np.ndarray,
    K: np.ndarray,
    class_name: str = "object",
    depth_quantile: float = 0.5,
    with_up_hint: bool = True,
) -> Optional[GraspPose]:
    """Run the production estimator on an already-rendered (depth_mm, mask).

    Shape-general sibling of :func:`plan_grasp` (which is box-specific): the
    curved renderers (sphere/cylinder/capsule) return (depth_mm, mask) directly,
    so this glue makes the detection + runs ``estimate_grasps``.
    """
    result = make_detection(np.asarray(mask, dtype=np.uint8), K, class_name=class_name)
    up_hint = up_hint_from_extrinsic(T_cam2base) if with_up_hint else None
    grasps = estimate_grasps(
        [result],
        np.asarray(depth_mm),
        np.asarray(K, dtype=np.float64),
        depth_quantile=depth_quantile,
        up_hint_cam=up_hint,
    )
    return select_best_grasp(grasps)


# ── real dumped-frame ingestion ──────────────────────────────────────────────
# ``grasp_cycle_check.py --save-frames <dir>`` writes (grasp_cycle_check.py:87-90):
#     <dir>/cycle_color.jpg   uint8 BGR
#     <dir>/cycle_depth.npy   uint16 millimetres (0 = invalid)
# Those dumps live on the PRODUCTION device (Orbbec SDK + real hand-eye), not on
# this Mac, so we do NOT fetch them here. To analyse a real dump later:
#   1. scp the dir off the device (cycle_color.jpg + cycle_depth.npy).
#   2. depth.npy carries NO intrinsics — supply K via a sidecar in the SAME dir:
#        - ``K.npy``         a (3,3) float array, OR
#        - ``intrinsics.npz`` with a ``K`` (or ``camera_matrix``) entry.
#      If neither is present the harness DEFAULT_K is used (a warning is printed)
#      — fine for the synthetic round-trip test, but for a real device dump you
#      MUST supply the live SDK K or the metric width/length will be wrong.
#   3. ``color, depth, K = load_dumped_frame(dir)``
#      ``g = plan_grasp_from_frame(color, depth, K, up_hint_cam)``
#      where ``up_hint_cam = up_hint_from_extrinsic(T_cam2base)`` and
#      ``T_cam2base = tcp_pose @ hand_eye`` from the dump's metadata.

_COLOR_NAMES = ("cycle_color.jpg", "color.jpg")
_DEPTH_NAMES = ("cycle_depth.npy", "depth.npy")


def _first_existing(d: Path, names: tuple[str, ...]) -> Optional[Path]:
    for n in names:
        p = d / n
        if p.exists():
            return p
    return None


def load_dumped_frame(
    dir_path: str | os.PathLike, *, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a real dumped frame → ``(color_bgr, depth_mm, K)``.

    Accepts the ``grasp_cycle_check --save-frames`` names (``cycle_color.jpg`` /
    ``cycle_depth.npy``) and the prompt's generic ``color.jpg`` / ``depth.npy``.
    K is read from a ``K.npy`` / ``intrinsics.npz`` sidecar if present, else the
    harness :data:`DEFAULT_K` is used (a warning is printed in that case).
    """
    import cv2

    d = Path(dir_path)
    color_p = _first_existing(d, _COLOR_NAMES)
    depth_p = _first_existing(d, _DEPTH_NAMES)
    if color_p is None or depth_p is None:
        raise FileNotFoundError(
            f"dump dir {d} must contain one of {_COLOR_NAMES} and {_DEPTH_NAMES}"
        )
    color = cv2.imread(str(color_p), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"failed to read color image {color_p}")
    depth = np.load(str(depth_p))
    if depth.dtype != np.uint16:
        # tolerate float metres dumps by converting to uint16 mm
        if np.issubdtype(depth.dtype, np.floating):
            depth = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
        else:
            depth = depth.astype(np.uint16)

    K: Optional[np.ndarray] = None
    k_npy = d / "K.npy"
    intr_npz = d / "intrinsics.npz"
    if k_npy.exists():
        K = np.asarray(np.load(str(k_npy)), dtype=np.float64)
    elif intr_npz.exists():
        z = np.load(str(intr_npz))
        for key in ("K", "camera_matrix", "intrinsics"):
            if key in z:
                K = np.asarray(z[key], dtype=np.float64)
                break
    if K is None:
        if verbose:
            print(
                f"[load_dumped_frame] no K.npy/intrinsics.npz sidecar in {d} — "
                f"falling back to harness DEFAULT_K (supply the live SDK K for a "
                f"real device dump or metric width/length will be off)"
            )
        K = DEFAULT_K.copy()
    if K.shape != (3, 3):
        raise ValueError(f"sidecar K must be (3,3), got {K.shape}")
    return color, depth, K


def plan_grasp_from_frame(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    up_hint_cam: Optional[np.ndarray],
    segmenter: Any = None,
    class_name: str = "box",
    depth_quantile: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> Optional[GraspPose]:
    """Run the SAME estimate_grasps→select_best_grasp on a REAL dumped frame.

    Detection source (first that applies):
      * ``mask`` — an explicit HxW instance mask (uint8, 1 over the object). Use
        this when the dump carries a saved segmenter mask alongside the frame so
        the detection matches the production path EXACTLY (no depth heuristic).
      * ``segmenter`` — the production ``YoloOnnxSegmenter`` (on-device only) run
        on ``color_bgr``. On this Mac there is no ONNX model.
      * else — a fallback detection built from the near-depth cluster (box stands
        above the table). This is lossy (a depth-median threshold) and may not
        reproduce the segmenter mask; prefer passing ``mask`` for a faithful
        round-trip.
    """
    if mask is not None:
        results = [make_detection(np.asarray(mask, dtype=np.uint8), K, class_name=class_name)]
    elif segmenter is not None:
        results = segmenter.infer(color_bgr)
        if not isinstance(results, list):
            results = [results]
    else:
        # no model on this host: derive the instance mask from valid depth that
        # is closer than the table band (box stands above the table).
        valid = depth_mm > 0
        if not valid.any():
            return None
        z = depth_mm[valid].astype(np.float64)
        # box pixels are the NEAR cluster; table is the dominant FAR plane.
        thresh = float(np.median(z))
        mask = ((depth_mm > 0) & (depth_mm.astype(np.float64) < thresh)).astype(np.uint8)
        import cv2
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if int(mask.sum()) < 50:
            return None
        results = [make_detection(mask, K, class_name=class_name)]

    grasps = estimate_grasps(
        results,
        np.asarray(depth_mm),
        np.asarray(K, dtype=np.float64),
        depth_quantile=depth_quantile,
        up_hint_cam=None if up_hint_cam is None else np.asarray(up_hint_cam, dtype=np.float64),
    )
    return select_best_grasp(grasps)


# ── IK envelope reachability ─────────────────────────────────────────────────
class _IKEnvelope:
    """Nearest-grid feasibility lookup over the measured B601-DM IK CSV.

    The CSV is a discrete sweep over (x, y, z, pitch, yaw) → ok∈{0,1}. A base
    pose is reachable iff its nearest grid sample is ok=1 AND it lies inside the
    sampled bounds. The feasible PITCH band is derived FROM the data (per-pitch
    ok-rate), not hardcoded: see :attr:`pitch_sweet`.
    """

    def __init__(self, csv_path: Path = _IK_CSV) -> None:
        rows = []
        with open(csv_path) as f:
            for d in csv.DictReader(f):
                rows.append(d)
        self.x = np.array([float(r["x"]) for r in rows])
        self.y = np.array([float(r["y"]) for r in rows])
        self.z = np.array([float(r["z"]) for r in rows])
        self.pitch = np.array([float(r["pitch"]) for r in rows])
        self.yaw = np.array([float(r["yaw"]) for r in rows])
        self.ok = np.array([int(float(r["ok"])) for r in rows], dtype=bool)
        self._grid = np.column_stack([self.x, self.y, self.z, self.pitch, self.yaw])
        self.bounds = {
            "x": (self.x.min(), self.x.max()),
            "y": (self.y.min(), self.y.max()),
            "z": (self.z.min(), self.z.max()),
            "pitch": (self.pitch.min(), self.pitch.max()),
            "yaw": (self.yaw.min(), self.yaw.max()),
        }
        # per-pitch ok-rate → the feasible "sweet zone" the memory note refers to.
        self.pitch_okrate = {
            float(p): float(self.ok[self.pitch == p].mean())
            for p in np.unique(self.pitch)
        }
        # sweet zone = contiguous pitch values with ok-rate >= 0.90.
        good = sorted(p for p, r in self.pitch_okrate.items() if r >= 0.90)
        self.pitch_sweet = (min(good), max(good)) if good else (0.0, 0.0)

    def feasible(
        self, x: float, y: float, z: float, pitch: float, yaw: float
    ) -> tuple[bool, str]:
        # out-of-bounds on the measured grid → not reachable (extrapolation unsafe)
        for name, val in (("x", x), ("y", y), ("z", z), ("pitch", pitch), ("yaw", yaw)):
            lo, hi = self.bounds[name]
            # small tolerance equal to half the grid step is allowed via NN below;
            # hard-reject only well outside.
            span = hi - lo
            tol = 0.5 * span / 6.0 + 1e-6
            if val < lo - tol or val > hi + tol:
                return False, f"{name}={val:.3f} out of measured envelope [{lo:.3f},{hi:.3f}]"
        # nearest grid sample (normalize each axis by its span so no axis dominates)
        q = np.array([x, y, z, pitch, yaw], dtype=np.float64)
        scale = np.array(
            [self.bounds[k][1] - self.bounds[k][0] or 1.0
             for k in ("x", "y", "z", "pitch", "yaw")]
        )
        d = np.linalg.norm((self._grid - q) / scale, axis=1)
        j = int(np.argmin(d))
        if self.ok[j]:
            return True, (
                f"nearest grid ok=1 @ "
                f"(x={self.x[j]:.2f},y={self.y[j]:.2f},z={self.z[j]:.2f},"
                f"pitch={self.pitch[j]:.3f},yaw={self.yaw[j]:.2f})"
            )
        return False, (
            f"nearest grid ok=0 @ "
            f"(x={self.x[j]:.2f},y={self.y[j]:.2f},z={self.z[j]:.2f},"
            f"pitch={self.pitch[j]:.3f},yaw={self.yaw[j]:.2f})"
        )


_ENVELOPE: Optional[_IKEnvelope] = None


def _envelope() -> _IKEnvelope:
    global _ENVELOPE
    if _ENVELOPE is None:
        _ENVELOPE = _IKEnvelope()
    return _ENVELOPE


def reachable(
    grasp_pose: GraspPose,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float = 0.08,
    insertion_depth_m: float = 0.025,
) -> tuple[bool, str]:
    """Whether the EXECUTED grasp is reachable, mirroring grasp_service.

    Faithfulness fixes (2026-06-18 sim-vs-real audit): use the production
    ``finalize_grasp_pose`` (side face-normal insertion axis + deeper side
    insertion + top centring), and check BOTH the grasp AND the pregrasp
    standoff against the measured IK envelope — the bare grasp-only check let
    the sim pass poses whose real pregrasp failed (e.g. the level side reach).
    For side grasps that fail only on the pregrasp, model
    ``grasp_service._level_side_to_reachable``: tilt the head down in steps to
    the most-level reachable pitch.

    NOTE: the CSV envelope is conservative on low/level poses (the real
    pinocchio reaches z<0.08 / pitch<0.225 that the CSV flags), so this can be
    PESSIMISTIC vs the real arm — recalibrating the CSV from real check_ik logs
    is the remaining gap.
    """
    if grasp_pose is None or not grasp_pose.is_valid:
        return False, "grasp invalid / None"
    from ..perception.grasp_geometry import finalize_grasp_pose

    grasp6, pre6 = finalize_grasp_pose(
        grasp_pose, np.asarray(T_cam2base, dtype=np.float64),
        pregrasp_offset_m, insertion_depth_m,
    )
    env = _envelope()
    ok_g, why_g = env.feasible(grasp6[0], grasp6[1], grasp6[2], grasp6[4], grasp6[5])
    ok_p, why_p = env.feasible(pre6[0], pre6[1], pre6[2], pre6[4], pre6[5])
    if ok_g and ok_p:
        return True, f"grasp+pregrasp reachable (pitch={grasp6[4]:.3f})"
    # side-pitch ladder: tilt the head down to the most-level pitch where BOTH
    # grasp and pregrasp reach. Fires whenever the level pose is not fully
    # reachable (the level grasp itself can be out of reach, e.g. high yaw), not
    # only on pregrasp failure — matching grasp_service._level_side_to_reachable.
    if getattr(grasp_pose, "method", "") == "side_face":
        x, y, z, r, p0, yaw = grasp6
        for dp in (0.08, 0.16, 0.24, 0.32, 0.40, 0.48):
            p = p0 + dp
            tool_x = np.asarray(pose6d_to_mat4(x, y, z, r, p, yaw), dtype=np.float64)[:3, 0]
            pre = (x - float(tool_x[0]) * pregrasp_offset_m,
                   y - float(tool_x[1]) * pregrasp_offset_m,
                   z - float(tool_x[2]) * pregrasp_offset_m)
            okg, _ = env.feasible(x, y, z, p, yaw)
            okp, _ = env.feasible(pre[0], pre[1], pre[2], p, yaw)
            if okg and okp:
                return True, f"side ladder reachable @ pitch {p:.3f} (head tilted)"
        return False, f"side level unreachable: pregrasp {why_p}; ladder exhausted"
    return False, (
        f"grasp {'OK' if ok_g else why_g} / pregrasp {'OK' if ok_p else why_p}"
    )


__all__ = [
    "IMG_HW",
    "DEFAULT_K",
    "default_K",
    "default_T_cam2base",
    "up_hint_from_extrinsic",
    "render_box_depth",
    "render_sphere_depth",
    "render_cylinder_depth",
    "render_capsule_depth",
    "make_detection",
    "plan_grasp",
    "plan_grasp_from_depth_mask",
    "reachable",
    "NoiseModel",
    "load_dumped_frame",
    "plan_grasp_from_frame",
]
