"""Torch-free vision-grasp pipeline for the reBot B601-DM voice app.

``run_grasp_once(target, *, ...)`` runs one cancellable grasp attempt:

    camera frame  →  YoloOnnxSegmenter.predict (target class only)
                  →  estimate_grasps (short-axis OBB)
                  →  select_best_grasp
                  →  T_cam2base = arm.get_tcp_pose() @ T_hand_eye
                  →  transform_grasp_pose_to_base
                  →  open → pregrasp → grasp → arm.grasp(force) → lift
                  →  (optional release)

The ``cancel_event`` (``threading.Event``) is polled before every arm motion;
on cancel the gripper is driven to a safe (open) state and the pipeline stops
at the current stage, returning ``{"success": False, "cancelled": True, ...}``.

Heavy / device-only deps (onnxruntime via the segmenter, camera SDK) are
imported lazily so this module imports on a Mac without them. There is no
torch / ultralytics anywhere in this path.
"""

from __future__ import annotations

import contextlib
import os as _os
_DEF_SPILL_SAFE = '["cup", "water bottle"]'
import logging
import math
import threading
import time
from typing import Any, Optional

import numpy as np

from .rebot_actuator import sleep_cancellable

logger = logging.getLogger(__name__)


class GraspCancelled(Exception):
    """Raised internally when ``cancel_event`` fires mid-pipeline."""


# SDK mechanical max jaw opening (m) and the clearance added over the detected
# object width so the open jaw clears the object before the approach.
_GRIPPER_MAX_M = 0.100
_OPEN_MARGIN_M = 0.012


def _motion_lock(actuator: Any):
    """Return a context manager that holds the actuator's bus lock for one
    arm motion, or a null context when no actuator is supplied.

    SAFETY: the grasp pipeline drives the raw arm directly, bypassing
    ``execute_sequence``. Wrapping each discrete bus op (one move / one
    gripper command) in the actuator's lock keeps it mutually exclusive with
    a concurrent action sequence, the 500Hz gripper thread and observation
    reads. Like execute_sequence, the lock is per-op — the blocking
    ``wait_motion`` poll happens OUTSIDE the lock so torque-off / cache reads
    are not starved during the multi-second move.
    """
    acq = getattr(actuator, "acquire_motion_lock", None)
    if callable(acq):
        return acq()
    return contextlib.nullcontext()


def _safe_open_distance(value: float) -> float:
    dist = float(value)
    if not math.isfinite(dist) or dist < 0.0:
        raise ValueError(
            "open_distance_m must be a non-negative finite number; "
            f"got {value!r}"
        )
    return dist


# ── multi-frame temporal stabilization (Item C) ─────────────────────────────
# Aggregate several detector frames into ONE stable grasp pose instead of
# committing to the first valid frame. A single jittery frame can put the grasp
# point a couple of cm off or rotate the open-axis enough to grip the wrong
# edge; clustering N frames and taking the per-component median cancels that
# per-frame noise before any motion. The math below is camera-frame and
# torch-free so it runs (and is unit-tested) on a Mac with no model.

# Cluster / outlier thresholds (camera frame).
_CL_CENTER_PX = 35.0       # same-object center distance (px)
_CL_IOU = 0.45             # same-object bbox IoU
_OUT_POS_M = 0.025         # frame rejected if position deviates > 25mm
_OUT_ANGLE_DEG = 25.0      # ...or angle  > 25°
_OUT_WIDTH_M = 0.020       # ...or width  > 20mm
_ANGLE_MAD_MAX_DEG = 18.0  # final angular MAD gate (non-round shapes)
_CLUSTER_CONF_KEEP = 0.70  # keep a 1-member cluster only at conf >= this


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _circular_median_deg(angles_deg, mod: float = 180.0) -> float:
    """Median of angles taken modulo ``mod`` (open-axis is direction-agnostic,
    so 180° periodic). Computed via unit vectors on the doubled angle so the
    wrap at the mod boundary is handled, then mapped back into ``[0, mod)``."""
    if not angles_deg:
        return 0.0
    scale = 2.0 * math.pi / mod
    xs = [math.cos(a * scale) for a in angles_deg]
    ys = [math.sin(a * scale) for a in angles_deg]
    mx = float(np.median(xs))
    my = float(np.median(ys))
    ang = math.atan2(my, mx) / scale
    return ang % mod


def _angle_diff_deg(a: float, b: float, mod: float = 180.0) -> float:
    """Smallest absolute difference between two angles modulo ``mod``."""
    d = abs((a - b) % mod)
    return min(d, mod - d)


def _is_round_pose(pose: Any) -> bool:
    """A grasp whose method/shape is 'round' has no meaningful open-axis angle
    (curved object → the angle gate would spuriously fire). Best-effort: the
    estimator may tag the pose via ``method`` or a ``shape`` attribute."""
    method = str(getattr(pose, "method", "") or "").lower()
    shape = str(getattr(pose, "shape", "") or "").lower()
    return "round" in method or "round" in shape


def _cluster_grasps(valid):
    """Greedy single-linkage clustering of valid grasps by target class +
    center distance + bbox IoU. Returns a list of clusters (lists of poses)."""
    clusters: list[list[Any]] = []
    for g in valid:
        gc = np.asarray(g.center_px, dtype=np.float64)
        placed = False
        for cl in clusters:
            for member in cl:
                if member.class_name != g.class_name:
                    continue
                mc = np.asarray(member.center_px, dtype=np.float64)
                if float(np.linalg.norm(gc - mc)) > _CL_CENTER_PX:
                    continue
                if _bbox_iou(member.bbox_xyxy, g.bbox_xyxy) < _CL_IOU:
                    continue
                cl.append(g)
                placed = True
                break
            if placed:
                break
        if not placed:
            clusters.append([g])
    return clusters


def _aggregate_cluster(cluster):
    """Aggregate a cluster of valid GraspPose into one stabilized pose:
    position = per-component median (camera frame), jaw width = median,
    conf = max, angle = circular median (mod 180°). Frames whose pose deviates
    too far from the cluster median are dropped before the final aggregation.
    Returns (pose, info) or (None, info) when the angular-MAD gate fails.

    ``pose`` is the cluster member nearest the aggregated centroid, mutated to
    carry the median position / width / angle / max-conf — this preserves all
    the other fields (tcp_rotation, method, ggcnn_agree, rect_points, ...) the
    downstream pipeline reads while still committing to the stable estimate."""
    info: dict[str, Any] = {"n_in": len(cluster), "rejected": 0}
    positions = np.asarray([np.asarray(g.position, dtype=np.float64) for g in cluster])
    med_pos = np.median(positions, axis=0)
    widths = [float(g.jaw_width_m) for g in cluster]
    med_w = float(np.median(widths))
    angles = [float(g.angle_deg) for g in cluster]
    med_ang = _circular_median_deg(angles)

    # Outlier rejection vs the cluster median.
    kept = []
    for g in cluster:
        dp = float(np.linalg.norm(np.asarray(g.position, dtype=np.float64) - med_pos))
        da = _angle_diff_deg(float(g.angle_deg), med_ang)
        dw = abs(float(g.jaw_width_m) - med_w)
        if dp > _OUT_POS_M or da > _OUT_ANGLE_DEG or dw > _OUT_WIDTH_M:
            info["rejected"] += 1
            continue
        kept.append(g)
    if not kept:
        kept = list(cluster)

    # Recompute on the kept set.
    positions = np.asarray([np.asarray(g.position, dtype=np.float64) for g in kept])
    med_pos = np.median(positions, axis=0)
    med_w = float(np.median([float(g.jaw_width_m) for g in kept]))
    kept_angles = [float(g.angle_deg) for g in kept]
    med_ang = _circular_median_deg(kept_angles)
    max_conf = max(float(g.conf) for g in kept)

    # Angular MAD gate (skip for round shapes — no meaningful open-axis angle).
    round_cluster = any(_is_round_pose(g) for g in kept)
    mad = float(np.median([_angle_diff_deg(a, med_ang) for a in kept_angles]))
    info["angular_mad_deg"] = mad
    info["n_kept"] = len(kept)
    if not round_cluster and len(kept) >= 2 and mad > _ANGLE_MAD_MAX_DEG:
        info["error"] = f"angular MAD {mad:.1f}° exceeds {_ANGLE_MAD_MAX_DEG}°"
        return None, info

    # Pick the member nearest the aggregated centroid as the carrier pose and
    # overwrite its stabilized components.
    carrier = min(
        kept,
        key=lambda g: float(np.linalg.norm(np.asarray(g.position, dtype=np.float64) - med_pos)),
    )
    carrier.position = med_pos.astype(np.float32)
    carrier.jaw_width_m = med_w
    carrier.angle_deg = med_ang
    carrier.conf = max_conf
    return carrier, info


def _open_gripper_safe(arm: Any, open_distance_m: float) -> None:
    arm.open_gripper(open_distance_m)


def _gripper_holding(arm: Any, default: Optional[bool] = None) -> Optional[bool]:
    """Physical holding check (best-effort). On the real RebotArm
    ``gripper_is_holding`` is a method grounded in encoder gap + sustained
    grip torque; on some stubs it is a plain attribute. Unknown → default."""
    try:
        attr = getattr(arm, "gripper_is_holding", None)
        val = attr() if callable(attr) else attr
        return bool(val) if val is not None else default
    except Exception:
        return default


def _clamp_place_xy(
    place6: list, bounds: Any, margin_m: float
) -> Optional[list]:
    """Clamp a place pose's x/y into the table bounds shrunk inward by
    ``margin_m`` (objects released at the very edge get nudged off the table
    by the jaw retreat — real machine, 2026-06-12 night run). ``bounds`` is
    ``[x_min, x_max, y_min, y_max]`` in the base frame (see
    tools/table_bounds_calib.py). Returns the clamped pose, or None when the
    point is already inside / bounds are malformed."""
    try:
        x_min, x_max, y_min, y_max = (float(v) for v in bounds)
    except (TypeError, ValueError):
        logger.warning("put_down: ignoring malformed place_bounds %r", bounds)
        return None
    if not (x_min < x_max and y_min < y_max):
        logger.warning("put_down: ignoring inverted place_bounds %r", bounds)
        return None
    m = max(0.0, float(margin_m))

    def _axis(v: float, lo: float, hi: float) -> float:
        lo_m, hi_m = lo + m, hi - m
        if lo_m > hi_m:  # margin swallows the axis → safest point is center
            return (lo + hi) / 2.0
        return min(max(v, lo_m), hi_m)

    cx = _axis(place6[0], x_min, x_max)
    cy = _axis(place6[1], y_min, y_max)
    if abs(cx - place6[0]) < 1e-9 and abs(cy - place6[1]) < 1e-9:
        return None
    return [cx, cy, *place6[2:]]


def _check_cancel(
    cancel_event: Optional[threading.Event],
    arm: Any,
    open_distance_m: float,
) -> None:
    """Raise :class:`GraspCancelled` (after parking the gripper safe) if the
    cancel event is set. Safe-park = open the gripper so we never leave it
    clamped on a half-finished grasp."""
    if cancel_event is not None and cancel_event.is_set():
        try:
            _open_gripper_safe(arm, open_distance_m)
        except Exception:
            logger.exception("grasp cancel: open_gripper (safe-park) failed")
        raise GraspCancelled()


def _relax_orientation(arm: Any, pre6d, grasp6d):
    """IK fallback ladder for far/awkward grasps: the camera-derived approach
    orientation is often what makes a REACHABLE position IK-infeasible ("明明拉
    伸出去就可以夹取，但它认为超出范围").

    CRITICAL (2026-06-16, sim-verified): the jaw↔short-axis ALIGNMENT now lives
    in the approach AZIMUTH (base YAW) for moderately angled boxes — the grasp
    builder (`ordinary_grasp._approach_aligned_to_short_axis`) re-aims the
    forward camera-ray azimuth along the box long axis so the gripper turns its
    head to face the box. Near box-yaw 90 (short side pointing at the arm) that
    re-aim is skipped/capped and the residual alignment falls back to ROLL about
    the approach axis. So the ladder must KEEP YAW ALWAYS (that is where the
    alignment is) and PRESERVE the ROLL first (the fallback alignment), relaxing
    only the PITCH (approach steepness) toward the reachable band; the roll is
    sacrificed only as a last resort (a misaligned grasp beats none). The old
    ladder flattened roll+pitch together, which DESTROYED the box alignment
    whenever it fired ("夹爪不转头对着斜盒子"). A variant is accepted only when
    ``check_ik`` passes for BOTH the pregrasp and the grasp pose. Returns
    ``(pre6d', grasp6d')`` or ``None``. Zero cost on the success path — only
    called after the original orientation already failed IK.
    """
    chk = getattr(arm, "check_ik", None)
    if not callable(chk):
        return None
    xp, yp, zp, pr, pp, pyaw = (float(v) for v in pre6d)
    xg, yg, zg, gr, gp, gyaw = (float(v) for v in grasp6d)
    # (roll_scale, pitch_scale, label) — alignment-preserving variants first.
    for r_scale, p_scale, label in (
        (1.0, 0.5, "pitch 50%, roll/align KEPT"),
        (1.0, 0.0, "pitch flat, roll/align KEPT"),
        (0.5, 0.5, "roll+pitch 50%"),
        (0.5, 0.0, "roll 50%, pitch flat"),
        (0.0, 0.0, "roll+pitch flat"),
    ):
        cand_pre = [xp, yp, zp, pr * r_scale, pp * p_scale, pyaw]
        cand_grasp = [xg, yg, zg, gr * r_scale, gp * p_scale, gyaw]
        try:
            ok_pre, _ = chk(*cand_pre)
            ok_grasp, _ = chk(*cand_grasp)
        except Exception:
            logger.debug("orientation ladder: check_ik raised", exc_info=True)
            return None
        if ok_pre and ok_grasp:
            logger.info(
                "grasp: relaxed orientation (%s) to make IK feasible "
                "(yaw kept %.2f, roll %.2f→%.2f)",
                label, gyaw, gr, gr * r_scale,
            )
            return cand_pre, cand_grasp
    return None


def _level_side_to_reachable(arm: Any, pre6d, grasp6d, pregrasp_offset_m: float,
                             max_extra_pitch: Optional[float] = None):
    """Most-LEVEL reachable side approach.

    Side grasps approach the vertical face LEVEL (head ~horizontal) so the jaw
    bites flat-on at full depth instead of stabbing in tilted-down and catching
    a shallow edge (operator 2026-06-17). But a fully level reach at extension
    can push the pregrasp STANDOFF out of the IK envelope (real machine: grasp
    reachable, pregrasp not). Tilt the head DOWN from the level pitch in small
    steps — keeping the grasp position and recomputing the standoff along the
    new approach each step — and return the FIRST (most level) pitch where BOTH
    the grasp and its standoff are IK-reachable. Returns ``(pre6d', grasp6d')``
    or ``None``. Only called after the level pregrasp already failed IK.
    """
    chk = getattr(arm, "check_ik", None)
    if not callable(chk):
        return None
    from .perception.transforms import pose6d_to_mat4

    x, y, z, r, p0, yaw = (float(v) for v in grasp6d)
    for dp in (0.08, 0.16, 0.24, 0.32, 0.40, 0.48):
        if max_extra_pitch is not None and dp > max_extra_pitch:
            break          # spill-safe class: stay (nearly) level or give up
        p = p0 + dp
        g = [x, y, z, r, p, yaw]
        try:
            tool_x = np.asarray(pose6d_to_mat4(*g), dtype=np.float64)[:3, 0]
        except Exception:
            return None
        # standoff sits BACK along the approach (tool-X points into the box).
        pre = [
            x - float(tool_x[0]) * pregrasp_offset_m,
            y - float(tool_x[1]) * pregrasp_offset_m,
            z - float(tool_x[2]) * pregrasp_offset_m,
            r, p, yaw,
        ]
        try:
            ok_g, _ = chk(*g)
            ok_p, _ = chk(*pre)
        except Exception:
            logger.debug("side-level ladder: check_ik raised", exc_info=True)
            return None
        if ok_g and ok_p:
            logger.info(
                "grasp: side approach tilted to most-level reachable pitch "
                "%.2f→%.2f (head %.0f°→%.0f° down)",
                p0, p, np.degrees(p0), np.degrees(p),
            )
            return pre, g
    return None


def _grasp_is_reachable(arm: Any, pre6d, grasp6d, method: str) -> bool:
    """Predict whether the executor will be able to reach this grasp, mirroring
    the exact feasibility logic in the move stage: ``check_ik`` must pass for
    BOTH the pregrasp and the grasp pose, and for non-side grasps the
    orientation ladder (:func:`_relax_orientation`) counts as reachable too
    (the executor falls back to it). Used only to RANK multiple candidates —
    never to reject a sole candidate (an arm with no ``check_ik`` returns
    ``False`` for everything, so callers must treat "no reachable candidate"
    as "keep the confidence pick"). Zero motion, no side effects.
    """
    chk = getattr(arm, "check_ik", None)
    if not callable(chk):
        return False
    try:
        ok_pre, _ = chk(*(float(v) for v in pre6d))
        ok_grasp, _ = chk(*(float(v) for v in grasp6d))
    except Exception:
        logger.debug("reach rank: check_ik raised", exc_info=True)
        return False
    if ok_pre and ok_grasp:
        return True
    # Side grasps keep their pitch (it IS the face geometry — the executor
    # never relaxes them), so the raw check is final. Others get the ladder.
    if method == "side_face":
        return False
    return _relax_orientation(arm, pre6d, grasp6d) is not None


# Executed-pose geometry lives in a leaf module (no app deps) so it can be
# unit-tested and replayed on REAL captured frames offline. Re-exported here for
# backward-compatible imports.
from .perception.grasp_geometry import finalize_grasp_pose  # noqa: E402,F401


def _select_reachable_grasp(
    grasps: list,
    arm: Any,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    insertion_depth_m: float,
):
    """Reachability-aware pick among MULTIPLE camera-frame grasp candidates.

    The base :func:`select_best_grasp` ranks purely by detector confidence —
    with two objects of the target class on the table it can pick a
    high-confidence but kinematically out-of-envelope one (this arm is a
    side-grasper: far / steep poses fall outside the measured IK band) and
    then burn the retry budget on it. Here each valid candidate is transformed
    to base and IK-checked (same logic the executor uses); the pick prefers a
    REACHABLE candidate, breaking ties by confidence. If none are reachable —
    or the arm exposes no ``check_ik`` — this returns the plain max-confidence
    pick, so behaviour is byte-identical to the old path on the single-object
    case and whenever reachability data is unavailable (zero regression).
    """
    from .perception.transforms import transform_grasp_pose_to_base

    valid = [g for g in grasps if getattr(g, "is_valid", False)]
    if not valid:
        return None
    conf_pick = max(valid, key=lambda g: g.conf)
    if len(valid) < 2 or not callable(getattr(arm, "check_ik", None)):
        return conf_pick

    scored = []
    for g in valid:
        try:
            grasp6d, pre6d = transform_grasp_pose_to_base(
                g.position, g.tcp_rotation, T_cam2base,
                pregrasp_offset_m, insertion_depth_m=insertion_depth_m,
                offset_axis_cam=g.position,
            )
            reach = _grasp_is_reachable(
                arm, pre6d, grasp6d, getattr(g, "method", "legacy")
            )
        except Exception:
            logger.debug("reach rank: candidate transform/IK failed", exc_info=True)
            reach = False
        scored.append((g, reach))

    reachable = [g for g, r in scored if r]
    if not reachable:
        return conf_pick
    best = max(reachable, key=lambda g: g.conf)
    if best is not conf_pick:
        logger.info(
            "grasp: reachability-aware pick — chose reachable %s (conf %.2f) "
            "over higher-conf %s (conf %.2f, out of IK envelope)",
            best.class_name, best.conf, conf_pick.class_name, conf_pick.conf,
        )
    return best


def _wait_motion_cancellable(
    arm: Any,
    duration: float,
    cancel_event: Optional[threading.Event],
) -> bool:
    """Wait for a move to settle in small steps, polling ``cancel_event`` every
    ~0.1s. Returns True if it completed, False if cancelled (caller safe-parks
    and must NOT issue further motion). The arm's own ``wait_motion`` blocks
    uninterruptibly, so we poll instead and only consult it for the final
    settle when not cancelled."""
    if cancel_event is None:
        arm.wait_motion(duration)
        return True
    return sleep_cancellable(max(0.0, float(duration)), cancel_event)


def _widebox_orbit_pose(arm, box_x, box_y, cur_x, cur_y, side,
                        z_obs=0.20, pitch=-0.22):
    """First IK-reachable TCP viewpoint on a circle around a too-wide box
    (wide-box orbit). ``side`` +1/-1 picks the orbit direction from the
    current viewing azimuth. Ladder: truest side view first (90 deg, wide
    radius), degrading to shallower angles / shorter radii that stay inside
    the IK envelope — even a 50-60 deg view exposes the thin face to the
    side-candidate detector. z/pitch mirror the validated scan poses.
    Returns pose6d or None. NO motion — check_ik only."""
    chk = getattr(arm, "check_ik", None)
    if not callable(chk):
        return None
    a0 = math.atan2(cur_y - box_y, cur_x - box_x)  # azimuth box→camera now
    for ang_deg in (90.0, 75.0, 60.0, 50.0, 40.0, 32.0):
        for radius in (0.40, 0.34, 0.30, 0.25, 0.21):
            a = a0 + float(side) * math.radians(ang_deg)
            tx = box_x + radius * math.cos(a)
            ty = box_y + radius * math.sin(a)
            yaw_box = math.atan2(box_y - ty, box_x - tx)
            # The wrist cannot yaw far ACROSS the base-radial direction
            # (cross-yaw ~<=50deg demonstrated on this arm), so a dead-centre
            # aim is often IK-infeasible even where the POSITION is fine.
            # The camera FOV is wide: the box only needs to stay within
            # ~35deg of the optical axis. Try the true aim first, then relax
            # the yaw back toward the base-radial azimuth in steps that keep
            # the box inside the FOV (offsets 0 / 20 / 35 deg).
            for yaw_off in (0.0, 0.35, 0.61):
                yaw = yaw_box - float(side) * yaw_off
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                pose = [tx, ty, z_obs, 0.0, pitch, yaw]
                try:
                    ok, _ = chk(*pose)
                except Exception:
                    logger.debug("orbit pose: check_ik raised", exc_info=True)
                    return None
                if ok:
                    logger.info(
                        "grasp orbit: side %+d reachable at %.0fdeg r=%.2f "
                        "yaw_off=%.0fdeg -> %s",
                        int(side), ang_deg, radius, math.degrees(yaw_off),
                        [round(float(v), 3) for v in pose],
                    )
                    return pose
    return None


def run_grasp_once(
    target: str,
    *,
    arm: Any,
    actuator: Any = None,
    segmenter: Any = None,
    camera: Any = None,
    K: Optional[np.ndarray] = None,
    T_hand_eye: Optional[np.ndarray] = None,
    scan_poses: Optional[list] = None,
    cancel_event: Optional[threading.Event] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    depth_quantile: float = 0.5,
    pregrasp_offset_m: float = 0.08,
    insertion_depth_m: float = 0.015,
    lift_height_m: float = 0.12,
    home_pose: tuple = (0.27, 0.0, 0.24, 0.0, 0.0, 0.0),
    grasp_force: Optional[float] = None,
    open_distance_m: float = 0.06,
    move_duration: float = 2.0,
    warm_up_frames: int = 5,
    detect_frames: int = 3,
    retries: int = 1,
    plausible_box: Optional[list] = None,
    adaptive_force: bool = False,
    reobserve: bool = True,
    servo_correct: bool = True,
    ggcnn: Any = None,
    release_after: bool = False,
    frame_sink: Any = None,
) -> dict:
    """Run a single cancellable grasp attempt for ``target``.

    Args:
        target: object label to grasp (must be in the segmenter vocabulary).
        arm: a connected ``RebotArm`` (move_to / get_tcp_pose / grasp /
            open_gripper / release_gripper).
        segmenter: a :class:`..perception.yolo_onnx.YoloOnnxSegmenter`. If
            ``None`` the caller must have configured one elsewhere (raises).
        camera: an opened camera driver exposing ``get_frame()`` ->
            ``(color_bgr, depth_mm)`` and ``warm_up(n)`` / ``K``.
        K: camera intrinsics (3x3). Falls back to ``camera.K`` when ``None``.
        T_hand_eye: eye-in-hand transform (4x4, camera←TCP). Required to
            transform the camera-frame grasp into the base frame.
        cancel_event: polled before each motion; set → safe-park + abort.
        conf/iou: detection thresholds.
        depth_quantile/pregrasp_offset_m/insertion_depth_m: grasp geometry.
        lift_height_m: how far up to retreat after the compliant grasp.
        grasp_force: compliant-close force (Nm); ``None`` → arm default.
        open_distance_m: safe jaw opening distance (m) for pre-grasp and
            cancellation safe-park. The SDK mechanical max is 0.09m; this
            default intentionally stays at the validated 0.06m action width.
        move_duration: per-waypoint duration (s).
        warm_up_frames: frames to discard for exposure/AWB stability.
        detect_frames: max frames per detection attempt. The FIRST frame with
            a valid grasp candidate wins (early exit), so the common case adds
            ZERO latency; extra frames only run as insurance against dropped /
            cold-camera frames (``get_frame``→None) and single-frame exposure
            or depth noise.
        retries: extra full attempts (re-detect → re-grasp) after a RETRIABLE
            failure (implausible pose, IK fail, closed-on-air, object lost in
            carry). Failures with no recovery path (nothing detected after the
            scan sweep, missing calibration, cancel) never retry.
        release_after: open the gripper at the end (drop the object).

    Returns:
        ``{"success": bool, "target": str, "attempt": int, "stage_ms": {...},
        ...}``. On cancel: ``{"success": False, "cancelled": True,
        "stage": <str>, ...}``.
    """
    base: dict[str, Any] = {"success": False, "target": target, "cancelled": False}
    if segmenter is None:
        return {**base, "error": "no segmenter configured"}
    if camera is None:
        return {**base, "error": "no camera configured"}
    if T_hand_eye is None:
        return {**base, "stage": "transform", "error": "no hand-eye calibration"}

    # Final-failure cleanup: an unsuccessful attempt can leave the gripper
    # open with a visible gap (real machine 2026-07-13 — the closed-on-air
    # retry re-opens it via _open_gripper_safe and a subsequent non-retriable
    # detect failure returns straight away, so nothing ever closes it back
    # up). Cosmetic/operational only — close the jaw so the arm is not left
    # gaping after a miss. NOT applied on cancel: cancellation keeps the
    # existing open safe-park (fingers may be near the jaw when the user
    # cancels; closing then is the wrong safety call).
    def _close_on_failure(res: dict) -> dict:
        if res.get("success") or res.get("cancelled"):
            return res
        try:
            with _motion_lock(actuator):
                arm.close_gripper()
        except Exception:
            logger.debug("grasp: close-on-failure cleanup failed", exc_info=True)
        return res

    attempts = 1 + max(0, int(retries))
    last: dict[str, Any] = {**base, "error": "no attempt ran"}
    for attempt in range(1, attempts + 1):
        res = _grasp_attempt(
            target,
            arm=arm,
            actuator=actuator,
            segmenter=segmenter,
            camera=camera,
            K=K,
            T_hand_eye=T_hand_eye,
            scan_poses=scan_poses,
            cancel_event=cancel_event,
            conf=conf,
            iou=iou,
            depth_quantile=depth_quantile,
            pregrasp_offset_m=pregrasp_offset_m,
            insertion_depth_m=insertion_depth_m,
            lift_height_m=lift_height_m,
            home_pose=home_pose,
            grasp_force=grasp_force,
            open_distance_m=open_distance_m,
            move_duration=move_duration,
            warm_up_frames=warm_up_frames,
            detect_frames=detect_frames,
            plausible_box=plausible_box,
            adaptive_force=adaptive_force,
            reobserve=reobserve,
            servo_correct=servo_correct,
            ggcnn=ggcnn,
            release_after=release_after,
            frame_sink=frame_sink,
        )
        res["attempt"] = attempt
        retriable = bool(res.pop("_retriable", False))
        if res.get("success") or res.get("cancelled") or not retriable:
            return _close_on_failure(res)
        last = res
        if attempt < attempts:
            logger.info(
                "grasp attempt %d/%d failed at stage=%s (%s); retrying with a "
                "fresh detection",
                attempt, attempts, res.get("stage"), res.get("error"),
            )
    return _close_on_failure(last)


def _grasp_attempt(
    target: str,
    *,
    arm: Any,
    actuator: Any,
    segmenter: Any,
    camera: Any,
    K: Optional[np.ndarray],
    T_hand_eye: np.ndarray,
    scan_poses: Optional[list],
    cancel_event: Optional[threading.Event],
    conf: float,
    iou: float,
    depth_quantile: float,
    pregrasp_offset_m: float,
    insertion_depth_m: float,
    lift_height_m: float,
    home_pose: tuple,
    grasp_force: Optional[float],
    open_distance_m: float,
    move_duration: float,
    warm_up_frames: int,
    detect_frames: int,
    plausible_box: Optional[list],
    adaptive_force: bool,
    reobserve: bool,
    servo_correct: bool,
    ggcnn: Any,
    release_after: bool,
    frame_sink: Any = None,
) -> dict:
    """One full detect→grasp→carry attempt. Returns the result dict; a
    ``_retriable: True`` key marks failures worth a fresh attempt."""
    result: dict[str, Any] = {"success": False, "target": target, "cancelled": False}
    stage = "init"
    timings: dict[str, int] = {}
    t_last = time.monotonic()

    def _mark(next_stage: str) -> None:
        # Per-stage wall-clock (ms) — lets demo logs show WHERE a slow or
        # failed grasp spent its time without extra instrumentation.
        nonlocal stage, t_last
        now = time.monotonic()
        timings[stage] = int((now - t_last) * 1000)
        stage = next_stage
        t_last = now

    safe_open_m = _safe_open_distance(open_distance_m)
    try:
        # ── 1+2. detect from the current view; if nothing is seen AND
        # scan_poses are configured, sweep the camera across them until the
        # target appears (auto-search), so grasp_object does not silently fail
        # just because the object is not perfectly centered. ────────────
        from .perception.ordinary_grasp import estimate_grasps, select_best_grasp

        def _up_hint_cam():
            """Gravity-up expressed in the CAMERA frame at the current pose —
            lets the estimator pick the object's TOP face (and reject side
            faces) no matter how oblique the view is. Best-effort: None on
            any failure → estimator falls back to the silhouette path."""
            try:
                with _motion_lock(actuator):
                    tcp = np.asarray(arm.get_tcp_pose(), dtype=np.float64)
                r_cam2base = (tcp @ np.asarray(T_hand_eye, dtype=np.float64))[:3, :3]
                return r_cam2base.T @ np.array([0.0, 0.0, 1.0])
            except Exception:
                logger.debug("up-hint computation failed", exc_info=True)
                return None

        too_wide_seen: dict[str, Any] = {}

        def _capture_and_detect():
            if warm_up_frames > 0:
                try:
                    camera.warm_up(warm_up_frames)
                except Exception:
                    logger.debug("camera.warm_up failed (continuing)", exc_info=True)
            Kl = None
            nd = 0
            up_hint = _up_hint_cam()
            # Temporal stabilization (Item C): gather up to N = max(3,
            # detect_frames) frames, each contributing its best valid grasp,
            # then CLUSTER + per-component MEDIAN the candidates so per-frame
            # jitter (a few mm of center drift, a few degrees of open-axis
            # rotation) cancels before any motion. Early-out preserved for the
            # validated common path: the FIRST frame whose best grasp already
            # has conf >= _CLUSTER_CONF_KEEP returns immediately (a single
            # high-confidence frame is byte-identical to the old behaviour —
            # extra frames would add nothing but latency).
            n_frames = max(3, max(1, detect_frames))
            per_frame_best: list[Any] = []
            for i in range(n_frames):
                _check_cancel(cancel_event, arm, safe_open_m)
                color_bgr, depth_mm = camera.get_frame()
                if color_bgr is None or depth_mm is None:
                    logger.debug("grasp detect: frame %d empty (cold camera?)", i)
                    continue
                Kl = np.asarray(K if K is not None else camera.K, dtype=np.float32)
                results = segmenter.predict(color_bgr, conf=conf, iou=iou, only_names={target})
                results = _filter_results_to_target(results, target)
                nd = max(nd, sum(len(getattr(r, "boxes", []) or []) for r in results))
                cands = estimate_grasps(
                    results, depth_mm, Kl,
                    depth_quantile=depth_quantile,
                    up_hint_cam=up_hint,
                    ggcnn=ggcnn,
                )
                # With a single candidate the reachability rank is a no-op and
                # the conf pick is the answer — skip the extra TCP read so the
                # validated common path is byte-identical. Only when ≥2 of the
                # target class are in view do we break ties by reachability.
                valid_cands = [g for g in cands if getattr(g, "is_valid", False)]
                # Wide-box orbit signal: remember a measured-but-too-wide
                # box (base position + thinnest visible face) so the orbit
                # logic can move for a side view. Zero cost on the normal
                # path (only runs when a frame yields NO valid grasp).
                if not valid_cands:
                    for _g in cands:
                        if (getattr(_g, "method", "") != "too_wide"
                                or _g.position is None):
                            continue
                        try:
                            with _motion_lock(actuator):
                                _tcp_tw = np.asarray(
                                    arm.get_tcp_pose(), dtype=np.float64)
                            _pb = (
                                _tcp_tw
                                @ np.asarray(T_hand_eye, dtype=np.float64)
                            ) @ np.append(
                                np.asarray(_g.position, dtype=np.float64), 1.0)
                            _w = float(getattr(_g, "min_side_width_m", 0.0)
                                       or getattr(_g, "jaw_width_m", 0.0))
                            _prev = too_wide_seen.get("min_width")
                            too_wide_seen["pos_base"] = _pb[:3]
                            too_wide_seen["min_width"] = (
                                _w if _prev is None
                                else min(float(_prev), _w)
                            )
                        except Exception:
                            logger.debug("too-wide record failed",
                                         exc_info=True)
                        break
                if len(valid_cands) >= 2:
                    with _motion_lock(actuator):
                        tcp_now = np.asarray(arm.get_tcp_pose(), dtype=np.float64)
                    T_c2b = tcp_now @ np.asarray(T_hand_eye, dtype=np.float64)
                    b = _select_reachable_grasp(
                        cands, arm, T_c2b, pregrasp_offset_m, insertion_depth_m
                    )
                else:
                    b = select_best_grasp(cands)
                if frame_sink is not None:
                    # Dashboard decision-frame tee — never let visualization
                    # break (or measurably slow) the grasp itself.
                    try:
                        frame_sink(color_bgr, depth_mm, results, b, stage)
                    except Exception:
                        logger.debug("frame_sink failed", exc_info=True)
                if b is not None:
                    per_frame_best.append(b)
                    # Early-out: one already-confident frame ends the loop with
                    # the validated single-frame answer. DISABLED by default
                    # 2026-07-03: single-frame answer carries per-frame depth
                    # jitter -> intermittent few-mm side_face miss. Set
                    # REBOT_GRASP_EARLY_OUT=1 to restore old behaviour.
                    if (_os.environ.get("REBOT_GRASP_EARLY_OUT", "0") == "1"
                            and float(getattr(b, "conf", 0.0)) >= _CLUSTER_CONF_KEEP):
                        return b, Kl, nd
            if not per_frame_best:
                return None, Kl, nd
            # One valid frame total → collapse to it (byte-identical to the old
            # first-valid-frame path).
            if len(per_frame_best) == 1:
                return per_frame_best[0], Kl, nd
            # Cluster the per-frame bests; keep a cluster with >=2 valid grasps,
            # OR a single-member cluster only at high conf.
            clusters = _cluster_grasps(per_frame_best)
            eligible = [
                cl for cl in clusters
                if len(cl) >= 2 or max(float(g.conf) for g in cl) >= _CLUSTER_CONF_KEEP
            ]
            if not eligible:
                # No agreement across frames — fall back to the single best
                # frame (the existing gates downstream still apply).
                return select_best_grasp(per_frame_best), Kl, nd
            # Prefer the most-supported cluster, breaking ties by max conf.
            cluster = max(
                eligible,
                key=lambda cl: (len(cl), max(float(g.conf) for g in cl)),
            )
            agg, info = _aggregate_cluster(cluster)
            if agg is None:
                logger.info("grasp: temporal aggregation rejected (%s)",
                            info.get("error"))
                return None, Kl, nd
            logger.info(
                "grasp: temporal aggregate over %d/%d frames "
                "(mad=%.1f° kept=%d rejected=%d)",
                info.get("n_in", 0), len(per_frame_best),
                info.get("angular_mad_deg", 0.0),
                info.get("n_kept", 0), info.get("rejected", 0),
            )
            return agg, Kl, nd

        _mark("capture")
        _check_cancel(cancel_event, arm, safe_open_m)
        best, K_local, num_det = _capture_and_detect()
        if best is None and scan_poses:
            _mark("scan")
            for pose in scan_poses:
                _check_cancel(cancel_event, arm, safe_open_m)
                with _motion_lock(actuator):
                    mok = arm.move_to(*pose, duration=move_duration)
                if not mok:
                    continue
                if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                    _check_cancel(cancel_event, arm, safe_open_m)
                best, K_local, num_det = _capture_and_detect()
                if best is not None:
                    logger.info("grasp: target found while scanning at pose %r", pose)
                    break

        # ── WIDE-BOX ORBIT (opt-in REBOT_WIDEBOX_ORBIT=1, default OFF) ──
        # A box whose every visible side face exceeds the jaw may still be
        # grippable by its THICKNESS, edge-on/invisible from this view.
        # Orbit the camera around the box (left, then right), re-detect, and
        # grasp if the thin face fits; if both sides still measure too wide
        # -> clean "too big" verdict. Every orbit pose is check_ik-validated
        # BEFORE any motion; unreachable sides are skipped. Motions go via
        # the validated first scan pose (retract) so the arc never sweeps
        # over the box.
        if (
            best is None
            and _os.environ.get("REBOT_WIDEBOX_ORBIT", "0") == "1"
            and too_wide_seen.get("pos_base") is not None
        ):
            _mark("orbit")
            _pb = too_wide_seen["pos_base"]
            _bx, _by = float(_pb[0]), float(_pb[1])
            _minw0 = float(too_wide_seen.get("min_width") or 0.0)
            logger.info(
                "grasp orbit: box at base (%.2f, %.2f) too wide (thinnest "
                "face %.3fm > 0.085) — orbiting for a side view",
                _bx, _by, _minw0,
            )
            _retract = list(scan_poses[0]) if scan_poses else None
            for _side in (1.0, -1.0):
                _check_cancel(cancel_event, arm, safe_open_m)
                try:
                    with _motion_lock(actuator):
                        _tcp_o = np.asarray(arm.get_tcp_pose(), dtype=np.float64)
                    _cx, _cy = float(_tcp_o[0, 3]), float(_tcp_o[1, 3])
                except Exception:
                    logger.debug("orbit: tcp read failed", exc_info=True)
                    break
                _opose = _widebox_orbit_pose(arm, _bx, _by, _cx, _cy, _side)
                if _opose is None:
                    logger.info(
                        "grasp orbit: side %+d not IK-reachable — skipped",
                        int(_side))
                    continue
                if _retract is not None:
                    with _motion_lock(actuator):
                        arm.move_to(*_retract, duration=move_duration)
                    if not _wait_motion_cancellable(
                            arm, move_duration, cancel_event):
                        _check_cancel(cancel_event, arm, safe_open_m)
                _check_cancel(cancel_event, arm, safe_open_m)
                with _motion_lock(actuator):
                    _mok = arm.move_to(*_opose, duration=move_duration)
                if not _mok:
                    logger.info(
                        "grasp orbit: move to side %+d failed — skipped",
                        int(_side))
                    continue
                if not _wait_motion_cancellable(
                        arm, move_duration, cancel_event):
                    _check_cancel(cancel_event, arm, safe_open_m)
                best, K_local, num_det = _capture_and_detect()
                if best is not None:
                    logger.info(
                        "grasp orbit: graspable face found from side %+d "
                        "(jaw %.3fm)", int(_side),
                        float(getattr(best, "jaw_width_m", 0.0)))
                    break
                logger.info(
                    "grasp orbit: still no graspable face from side %+d",
                    int(_side))
            if best is None:
                # Verdict: genuinely too big (or no side view reachable).
                # Park back at the retract pose so the arm is not left
                # hovering at an orbit viewpoint.
                if _retract is not None:
                    try:
                        with _motion_lock(actuator):
                            arm.move_to(*_retract, duration=move_duration)
                        _wait_motion_cancellable(
                            arm, move_duration, cancel_event)
                    except Exception:
                        logger.debug("orbit retract failed", exc_info=True)
                _minw = float(too_wide_seen.get("min_width") or _minw0)
                return {
                    **result,
                    "stage": stage,
                    "stage_ms": timings,
                    "error": (
                        f"box too wide to grasp (thinnest visible face "
                        f"{_minw:.3f}m > 0.085m jaw)"
                    ),
                    "too_wide": True,
                    "num_detections": int(num_det),
                }

        _mark("detect")
        if best is None:
            # Clear TOO-BIG verdict (works with the orbit OFF too): when the
            # only thing detection measured was too-wide faces, say so — the
            # plugin speaks "The box is too big for me to grip." instead of
            # the misleading "I couldn't find the box."
            if too_wide_seen.get("min_width") is not None:
                _minw = float(too_wide_seen["min_width"])
                return {
                    **result,
                    "stage": stage,
                    "stage_ms": timings,
                    "error": (
                        f"box too wide to grasp (thinnest visible face "
                        f"{_minw:.3f}m > 0.085m jaw)"
                    ),
                    "too_wide": True,
                    "num_detections": int(num_det),
                }
            # NOT retriable: the scan sweep already covered every viewpoint —
            # an immediate identical retry would just repeat the whole sweep.
            return {
                **result,
                "stage": stage,
                "stage_ms": timings,
                "error": f"no valid grasp for target {target!r}",
                "num_detections": int(num_det),
            }
        K = K_local
        from .perception.transforms import transform_grasp_pose_to_base

        # Detection → transform → gates, with at most ONE close-up
        # re-observation round. A FAR / side-on viewpoint merges the box's
        # top and side faces into one silhouette, so the min-area-rect short
        # axis measures the WRONG physical dimension — the estimated width
        # comes out far wider than the real graspable face and the grasp
        # fails ("侧面对着 → 宽度虚大 → 夹不起来"). When the first detection
        # is far away or suspiciously wide, move the camera to the sweet-spot
        # viewing geometry (~0.22m short of the target, same height the
        # validated grasps used) and re-measure before committing.
        reobserved = False
        while True:
            result["grasp_class"] = best.class_name
            result["grasp_conf"] = float(best.conf)
            result["center_px"] = list(best.center_px)
            result["jaw_width_m"] = float(best.jaw_width_m)

            # ── 3. camera → base transform ──────────────────────────────
            _mark("transform")
            _check_cancel(cancel_event, arm, safe_open_m)
            with _motion_lock(actuator):
                tcp_pose = np.asarray(arm.get_tcp_pose(), dtype=np.float64)
            T_cam2base = tcp_pose @ np.asarray(T_hand_eye, dtype=np.float64)
            # SINGLE SOURCE OF TRUTH for the executed geometry (shared with the
            # offline real-frame replay harness — see finalize_grasp_pose).
            grasp6d, pre6d = finalize_grasp_pose(
                best, T_cam2base, pregrasp_offset_m, insertion_depth_m
            )
            result["grasp_pose"] = [float(v) for v in grasp6d]
            result["pregrasp_pose"] = [float(v) for v in pre6d]
            gx, gy, gz = (float(v) for v in grasp6d[:3])
            logger.info("FINDBG best.position=(%.4f %.4f %.4f) -> committed z=%.4f",
                        *[float(v) for v in best.position], gz)
            jaw = float(best.jaw_width_m)
            result["grasp_method"] = getattr(best, "method", "legacy")
            if getattr(best, "ggcnn_agree", None) is not None:
                result["ggcnn_agree"] = bool(best.ggcnn_agree)

            # SIDE grasps need jaw-body clearance above the TABLE: the fingers
            # wrap a vertical face, so a grip too close to the tabletop presses
            # the jaw into it. Two signals, and BOTH must look low to reject —
            # either one alone gives false rejects:
            #   • absolute z (gz): the OLD sole check (gz<0.045) assumed the
            #     table at base z≈0; on a table ~5cm below the arm base it
            #     wrongly rejected clear grasps (gz≈0.043) → fixed by also
            #     requiring the face signal.
            #   • face clearance (½ the visible vertical face below the centroid):
            #     assumes the visible face reaches the table, which FAILS when
            #     detection captures only the UPPER part of a tall box — a grip
            #     high on the box (gz≈0.158, ~20cm clear) then reads a tiny face
            #     clearance and gets wrongly rejected (real machine 2026-06-17).
            # So reject only when the grasp is BOTH absolutely low AND has little
            # face below it — a genuinely near-table grip. A high grasp (gz large)
            # or a full visible face passes.
            _side_clearance = 0.5 * float(getattr(best, "object_length_m", 0.0) or 0.0)
            # Table-aware (2026-07-13): the 0.045 bar assumed table at base
            # z=0 (its own comment admits the misfire). Measure clearance
            # from the REAL table height so short objects (cup) floored to
            # 0.040 by the side z-floor are not auto-rejected in [0.040,0.045).
            _table_z = float(_os.environ.get("REBOT_TABLE_Z", "0.0"))
            if (
                result["grasp_method"] == "side_face"
                and (gz - _table_z) < 0.045
                and _side_clearance < 0.045
            ):
                return {
                    **result,
                    "stage": "plausibility",
                    "stage_ms": timings,
                    "error": (
                        f"side grasp too low (z={gz:.3f}m, face clearance "
                        f"{_side_clearance:.3f}m)"
                    ),
                    "_retriable": True,
                }

            # Close-up re-observation trigger: far target (side-on view from
            # the home-height camera) or an estimate already wider than the
            # jaw can open (silhouette inflation). One round only.
            #
            # SKIP for a first-pass "round" classification (real machine
            # 2026-07-13, orange): the reobserve move for a non-top_face
            # object goes CLOSE + steep-down (obs_pitch=0.45, ~26°) to bring
            # a hidden top face into view. A round object has no top face to
            # reveal — that tilt instead foreshortens the sphere's visible
            # cap, and the PCA on that thin slice is unstable (elongation
            # read 1.18 clean at range, then climbed to 37 after the
            # close-up tilt on the SAME stationary orange). The first-pass
            # descriptor is already the reliable read for round bodies
            # (distance doesn't hide anything a closer angle would reveal),
            # so trust it and skip the move — also removes a full
            # move+settle+recapture round trip from the 20s+ grasp latency.
            if (
                reobserve
                and not reobserved
                and getattr(best, "method", "") != "round"
                and (gx > 0.50 or jaw > 0.085
                     or getattr(best, "ggcnn_agree", None) is False)
            ):
                reobserved = True
                obs_x = min(max(gx - 0.22, 0.20), 0.50)
                obs_y = min(max(gy * 0.8, -0.20), 0.20)
                obs_yaw = max(-0.5, min(0.5, math.atan2(gy - obs_y, max(gx - obs_x, 1e-3))))
                # Viewpoint strategy (real machine 2026-06-12): when the
                # FIRST estimate did not come from the top-face fit, the
                # camera most likely cannot SEE the object's top at all (a
                # tall box at camera height shows only its side faces — the
                # silhouette then measures the side width and gets rejected).
                # Re-observe from HIGH with a downward tilt so the top face
                # enters the view; otherwise the close-up sweet spot.
                if getattr(best, "method", "legacy") != "top_face":
                    obs_x = min(max(gx - 0.18, 0.20), 0.46)
                    obs_z, obs_pitch = 0.33, 0.45
                else:
                    obs_z, obs_pitch = 0.26, 0.0
                _mark("reobserve")
                _check_cancel(cancel_event, arm, safe_open_m)
                with _motion_lock(actuator):
                    obs_ok = arm.move_to(obs_x, obs_y, obs_z, 0.0, obs_pitch, obs_yaw,
                                         duration=move_duration)
                if obs_ok:
                    if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                        _check_cancel(cancel_event, arm, safe_open_m)
                    best2, K2, _nd2 = _capture_and_detect()
                    if best2 is not None:
                        logger.info(
                            "grasp: re-observed close up — width %.3f→%.3f m",
                            jaw, float(best2.jaw_width_m),
                        )
                        best = best2
                        result["reobserved"] = True
                        continue  # recompute transform from the new viewpoint
                    logger.info("grasp: close-up re-observation lost the target; "
                                "keeping the original detection")
                else:
                    logger.info("grasp: re-observe pose IK failed; keeping "
                                "original detection")
            break

        # Width gate AFTER the best available measurement: an object measured
        # wider than the jaw can physically open (soft limit ≈0.095m) cannot
        # be gripped — executing just slams the jaw at full open. Retriable:
        # a fresh frame / next attempt may measure saner.
        if not (0.010 <= jaw <= 0.095):
            return {
                **result,
                "stage": "plausibility",
                "stage_ms": timings,
                "error": f"implausible jaw width {jaw:.3f}m",
                "_retriable": True,
            }

        # Auto-size the pre-grasp open width to the detected object: a fixed
        # safe-open (e.g. 0.06m) is NARROWER than a wide box (e.g. 0.077m) and
        # the jaw would collide instead of going around it. Widen to
        # object_width + margin, clamped to the mechanical max. Never shrink
        # below the configured safe-open. The safe-park (cancel) open uses the
        # same widened value so a release always clears the object.
        widened = float(best.jaw_width_m) + _OPEN_MARGIN_M
        safe_open_m = min(_GRIPPER_MAX_M, max(safe_open_m, widened))
        result["open_distance_m"] = safe_open_m
        logger.info(
            "GDBG method=%s grasp6d=[%.4f %.4f %.4f | r%.3f p%.3f y%.3f] "
            "jaw=%.4f objlen=%.4f open=%.4f conf=%.2f",
            result.get("grasp_method"), gx, gy, gz,
            float(grasp6d[3]), float(grasp6d[4]), float(grasp6d[5]),
            jaw, float(getattr(best, "object_length_m", 0.0) or 0.0),
            safe_open_m, float(getattr(best, "conf", 0.0)),
        )

        # Plausibility gate (base frame, OPT-IN via plausible_box): the grasp
        # point must be a sane spot for an object on the table in front of
        # the arm. Bounds catch depth-noise garbage (z under the table,
        # target inside/behind the base), not borderline reach (IK rejects
        # those). Off by default — the box is rig-specific, so the app config
        # supplies it (config.yaml grasp.plausible_box).
        if plausible_box is not None:
            x0, x1, y0, y1, z0, z1 = (float(v) for v in plausible_box)
            if not (x0 <= gx <= x1 and y0 <= gy <= y1 and z0 <= gz <= z1):
                return {
                    **result,
                    "stage": "plausibility",
                    "stage_ms": timings,
                    "error": f"implausible grasp position ({gx:.2f},{gy:.2f},{gz:.2f})",
                    "_retriable": True,
                }

        # ── 4. execute: open → pregrasp → grasp pos → compliant grasp ───
        # SAFETY: each arm bus op is wrapped in the actuator lock (atomic vs a
        # concurrent action / gripper thread / cache read); the blocking
        # settle wait runs OUTSIDE the lock and is cancellable.
        _mark("open")
        _check_cancel(cancel_event, arm, safe_open_m)
        with _motion_lock(actuator):
            _open_gripper_safe(arm, safe_open_m)

        _mark("pregrasp")
        _check_cancel(cancel_event, arm, safe_open_m)
        xp, yp, zp, rxp, ryp, rzp = pre6d
        with _motion_lock(actuator):
            pregrasp_ok = arm.move_to(xp, yp, zp, rxp, ryp, rzp, duration=move_duration)
        if not pregrasp_ok:
            # Orientation ladder first: a far-but-reachable position is often
            # only infeasible because of the camera-derived roll/pitch. Keep
            # the yaw (jaw↔short-axis alignment), flatten the rest. NEVER for
            # side grasps — their pitch IS the grasp geometry; flattening it
            # would wipe the face alignment (IK envelope says side-band
            # poses are 91-100% feasible anyway, so the ladder buys nothing).
            relaxed = None
            if result.get("grasp_method") != "side_face":
                relaxed = _relax_orientation(arm, pre6d, grasp6d)
            else:
                # SIDE grasps approach LEVEL (head horizontal into the vertical
                # face). If the level standoff is out of reach, tilt the head
                # down to the most-level REACHABLE pitch rather than failing.
                # Spill-safe classes (cup, water bottle): contents pour
                # out of a tilted grasp, so cap the reachability tilt at
                # REBOT_SPILL_MAX_TILT (default 0.10 rad -> <=~8deg total
                # head-down). Unreachable within the cap -> retriable
                # decline; reposition the object closer instead.
                import json as _json
                _spill_raw = _os.environ.get("REBOT_SPILL_SAFE_CLASSES", None) or _DEF_SPILL_SAFE
                try:
                    _spill = {str(s).lower() for s in _json.loads(_spill_raw)}
                except Exception:
                    _spill = {"cup", "water bottle"}
                _max_tilt = None
                if str(result.get("grasp_class", "")).lower() in _spill:
                    _max_tilt = float(_os.environ.get("REBOT_SPILL_MAX_TILT", "0.10"))
                relaxed = _level_side_to_reachable(
                    arm, pre6d, grasp6d, pregrasp_offset_m,
                    max_extra_pitch=_max_tilt,
                )
            if relaxed is not None:
                pre6d, grasp6d = relaxed
                result["orientation_relaxed"] = True
                xp, yp, zp, rxp, ryp, rzp = pre6d
                with _motion_lock(actuator):
                    pregrasp_ok = arm.move_to(
                        xp, yp, zp, rxp, ryp, rzp, duration=move_duration
                    )
        if not pregrasp_ok:
            # Retriable: a fresh detection can yield a reachable pose (the
            # object may sit differently in the next frame).
            return {**result, "stage": stage, "stage_ms": timings,
                    "error": "pregrasp IK failed", "_retriable": True}
        # Overlap the servo-correction CAPTURE with the settle tail: the last
        # ~0.4s of the pregrasp move is damping/settle — the camera is already
        # essentially on target, so the frame grabbed there is sharp enough
        # for the bounded ≤3cm correction and the capture+inference cost
        # disappears into the wait (-0.4..1s per grasp).
        servo_pre = {}
        if servo_correct:
            head = max(0.0, float(move_duration) - 0.4)
            if not _wait_motion_cancellable(arm, head, cancel_event):
                _check_cancel(cancel_event, arm, safe_open_m)
            cap_box: dict = {}

            def _bg_capture() -> None:
                try:
                    cap_box["out"] = _capture_and_detect()
                except Exception:
                    logger.debug("overlapped servo capture failed", exc_info=True)

            t_cap = threading.Thread(target=_bg_capture, daemon=True)
            t_cap.start()
            if not _wait_motion_cancellable(arm, 0.4, cancel_event):
                _check_cancel(cancel_event, arm, safe_open_m)
            # 2026-07-14: was 3.0 — a capture takes 4.5-6s on CPU, so the
            # overlap almost never landed and the servo stage re-captured
            # from scratch (~6s duplicate). Wait it out instead.
            t_cap.join(timeout=8.0)
            servo_pre = cap_box
        else:
            if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                _check_cancel(cancel_event, arm, safe_open_m)  # safe-park + raise

        # Servo-lite correction (Phase 2): from the pregrasp the camera looks
        # straight at the target — one quick re-detection measures the drift
        # left by calibration residue and the earlier viewpoint's parallax,
        # and shifts the grasp x/y by it. Bounded: ignore sub-4mm noise and
        # anything >3cm (suspicious re-detection — trust the plan instead).
        if servo_correct:
            _mark("servo")
            _check_cancel(cancel_event, arm, safe_open_m)
            b3, _K3, _nd3 = servo_pre.get("out") or _capture_and_detect()
            logger.info(
                "SERVODBG redetect=%s num_det=%s",
                "MISS" if (b3 is None or b3.position is None) else "hit", _nd3,
            )
            if b3 is not None and b3.position is not None:
                try:
                    with _motion_lock(actuator):
                        tcp3 = np.asarray(arm.get_tcp_pose(), dtype=np.float64)
                    T3 = tcp3 @ np.asarray(T_hand_eye, dtype=np.float64)
                    g3, _p3 = transform_grasp_pose_to_base(
                        b3.position, b3.tcp_rotation, T3, pregrasp_offset_m,
                        insertion_depth_m=insertion_depth_m,
                        offset_axis_cam=b3.position,
                    )
                    dx = float(g3[0]) - float(grasp6d[0])
                    dy = float(g3[1]) - float(grasp6d[1])
                    drift = (dx * dx + dy * dy) ** 0.5
                    if 0.004 < drift <= 0.03:
                        grasp6d = [
                            float(grasp6d[0]) + dx, float(grasp6d[1]) + dy,
                            *(float(v) for v in grasp6d[2:]),
                        ]
                        result["grasp_pose"] = [float(v) for v in grasp6d]
                        result["servo_drift_mm"] = round(drift * 1000.0, 1)
                        logger.info("grasp: servo correction %.1fmm applied", drift * 1000)
                    elif drift > 0.03:
                        logger.info(
                            "grasp: servo re-detection drifted %.0fmm (>30mm) — "
                            "ignored as implausible", drift * 1000,
                        )
                    else:
                        logger.info(
                            "SERVODBG drift %.1fmm (<=4mm, no shift needed)",
                            drift * 1000,
                        )
                except Exception:
                    logger.info("servo correction failed (continuing)", exc_info=True)

        _mark("grasp_move")
        _check_cancel(cancel_event, arm, safe_open_m)
        xg, yg, zg, rxg, ryg, rzg = grasp6d
        grasp_dur = max(1.0, move_duration * 0.75)
        with _motion_lock(actuator):
            grasp_move_ok = arm.move_to(xg, yg, zg, rxg, ryg, rzg, duration=grasp_dur)
        if (not grasp_move_ok and not result.get("orientation_relaxed")
                and result.get("grasp_method") != "side_face"):
            # Same ladder if the grasp pose (not the pregrasp) is the
            # infeasible one. The arm is parked at the pregrasp — moving to a
            # flatter-orientation grasp from here is safe.
            relaxed = _relax_orientation(arm, pre6d, grasp6d)
            if relaxed is not None:
                _, grasp6d = relaxed
                result["orientation_relaxed"] = True
                xg, yg, zg, rxg, ryg, rzg = grasp6d
                with _motion_lock(actuator):
                    grasp_move_ok = arm.move_to(
                        xg, yg, zg, rxg, ryg, rzg, duration=grasp_dur
                    )
        if not grasp_move_ok:
            return {**result, "stage": stage, "stage_ms": timings,
                    "error": "grasp-pose IK failed", "_retriable": True}
        if not _wait_motion_cancellable(arm, grasp_dur, cancel_event):
            _check_cancel(cancel_event, arm, safe_open_m)  # safe-park + raise

        # Executed-vs-commanded pose at the moment of close: separates
        # perception error (arm ON target but box isn't there) from
        # trajectory/IK execution error (arm parked off-target).
        try:
            with _motion_lock(actuator):
                _tcp_g = np.asarray(arm.get_tcp_pose(), dtype=np.float64)
            logger.info(
                "EXECDBG grasp_move settled: actual=(%.4f %.4f %.4f) "
                "commanded=(%.4f %.4f %.4f) delta=(%+.1f %+.1f %+.1f)mm",
                _tcp_g[0, 3], _tcp_g[1, 3], _tcp_g[2, 3], xg, yg, zg,
                (_tcp_g[0, 3] - xg) * 1000.0,
                (_tcp_g[1, 3] - yg) * 1000.0,
                (_tcp_g[2, 3] - zg) * 1000.0,
            )
        except Exception:
            logger.debug("EXECDBG read failed", exc_info=True)

        _mark("grasp")
        _check_cancel(cancel_event, arm, safe_open_m)
        with _motion_lock(actuator):
            if adaptive_force:
                # Unconfigured object class: ramp from a gentle hold up to the
                # grasp_force ceiling until the gap stops creeping (see
                # RebotArm.grasp adaptive mode). TypeError fallback keeps
                # older/stub arms (no adaptive kwarg) working.
                try:
                    held = bool(arm.grasp(force=grasp_force, adaptive=True))
                except TypeError:
                    held = bool(arm.grasp(force=grasp_force))
            else:
                held = bool(arm.grasp(force=grasp_force))
        result["grasp_closed"] = held
        result["adaptive_force"] = bool(adaptive_force)

        # Closed-on-air check BEFORE carrying anything anywhere: when neither
        # the grasp state machine nor the physical holding signal (encoder gap
        # + grip torque) says we have the object, re-open and let the retry
        # re-detect — the object may have shifted when the jaw touched it.
        holding_now = _gripper_holding(arm, default=None)
        if not held and holding_now is not True:
            try:
                with _motion_lock(actuator):
                    _open_gripper_safe(arm, safe_open_m)
            except Exception:
                logger.exception("grasp: reopen after closed-on-air failed")
            return {**result, "stage": stage, "stage_ms": timings,
                    "error": "gripper closed but nothing held", "_retriable": True}

        # ── 5. lift clear of the table, then CARRY the object back to home ──
        # Demo flow: after grasping, the arm returns to its home/ready pose
        # holding the object (looks dynamic + parks it in a stable, centred,
        # IK-comfortable pose — a far grasp at the reach limit shakes/sags). The
        # recorded grasp_pose/pregrasp_pose (set above) let put_down replay the
        # pick spot to place it back, so carrying home loses nothing.
        _mark("lift")
        _check_cancel(cancel_event, arm, safe_open_m)
        lifted = False
        # (a) small straight-up clearance lift so the object does not drag
        #     across the table on the way home (best-effort; far grasps may fail
        #     IK here — the carry-home move lifts anyway).
        zc = zg + min(float(lift_height_m), 0.06)
        with _motion_lock(actuator):
            clr_ok = arm.move_to(xg, yg, zc, rxg, ryg, rzg,
                                 duration=max(1.0, move_duration * 0.6))
        if clr_ok and not _wait_motion_cancellable(arm, max(1.0, move_duration * 0.6), cancel_event):
            _check_cancel(cancel_event, arm, safe_open_m)
        # (b) carry home — the ready pose is always IK-reachable; this both
        #     lifts and re-centres, holding the object at home.
        _mark("carry_home")
        _check_cancel(cancel_event, arm, safe_open_m)
        with _motion_lock(actuator):
            home_ok = arm.move_to(*home_pose, duration=move_duration)
        if home_ok:
            if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                _check_cancel(cancel_event, arm, safe_open_m)
            lifted = True
            result["returned_home"] = True
        else:
            # Home unreachable (should not happen) — fall back to a small lift
            # so the object is at least raised off the table.
            logger.warning("carry_home IK failed; leaving arm at clearance lift")
            lifted = clr_ok
        result["lifted"] = lifted

        # ── 6. optional release ─────────────────────────────────────────
        if release_after:
            _mark("release")
            _check_cancel(cancel_event, arm, safe_open_m)
            with _motion_lock(actuator):
                _open_gripper_safe(arm, safe_open_m)

        # Holding check — PHYSICAL (encoder gap + grip torque on the real
        # arm), so a grasp() that timed out in software but is in fact
        # clamping the object still counts as success.
        holding = _gripper_holding(arm, default=None)
        result["holding"] = holding if holding is not None else held

        # Lost-in-carry check: the gripper closed on the object at the table
        # but the physical signal says it is gone after the carry — the box
        # slipped out en route. Retriable: it is most likely back near the
        # pickup spot, so a fresh detect→grasp recovers without the presenter
        # repeating the command.
        if not release_after and held and holding is False:
            _mark("verify")
            return {**result, "stage": "carry_home", "stage_ms": timings,
                    "error": "object lost during carry", "_retriable": True}

        result["success"] = bool(held or (not release_after and result["holding"]))
        _mark("done")
        result["stage"] = "done"
        result["stage_ms"] = timings
        return result

    except GraspCancelled:
        logger.info("grasp cancelled at stage=%s", stage)
        return {**result, "success": False, "cancelled": True, "stage": stage,
                "stage_ms": timings}
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("grasp pipeline failed at stage=%s", stage)
        # Best-effort safe-park on any unexpected failure.
        try:
            _open_gripper_safe(arm, safe_open_m)
        except Exception:
            pass
        return {**result, "stage": stage, "stage_ms": timings, "error": str(exc)}


def run_put_down_once(
    *,
    arm: Any,
    actuator: Any = None,
    grasp_pose: Optional[list] = None,
    pregrasp_pose: Optional[list] = None,
    open_distance_m: float = _GRIPPER_MAX_M,
    place_pose: tuple = (0.30, 0.00, 0.15, 0.0, 0.0, 0.0),
    home_pose: tuple = (0.27, 0.0, 0.24, 0.0, 0.0, 0.0),
    cancel_event: Optional[threading.Event] = None,
    move_duration: float = 2.0,
    place_bounds: Optional[list] = None,
    place_margin_m: float = 0.05,
) -> dict:
    """Put the held object DOWN — preferably back where it was picked up.

    Closes the pick-and-place loop with the camera in mind: the spot the last
    ``run_grasp_once`` picked the object from is, by construction, a spot the
    camera has just detected it at (and every waypoint there passed IK during
    the grasp). Releasing anywhere else (the old fixed place spot) can leave
    the object outside the camera's view, so the NEXT grasp fails. So:

    * ``grasp_pose`` / ``pregrasp_pose`` given (recorded from the last grasp):
      replay them — approach via ``pregrasp_pose``, descend to ``grasp_pose``,
      release, retreat back through ``pregrasp_pose``, go home. Zero new IK
      risk; the object lands exactly where the camera last saw it.
    * no recorded grasp (e.g. the user manually closed the gripper around
      something): fall back to ``place_pose``.

    Release width: ``open_distance_m`` should be the (auto-widened) width the
    grasp recorded — the jaw MUST open wider than the held object or the SDK's
    open ramp drives the jaws inward and never releases (0.06m cannot release
    the 0.077m demo box). Defaults to mechanical full-open.

    Failure policy: unlike cancel (explicit user stop → safe-park open), an
    UNEXPECTED failure keeps the object held and reports the error — never
    silently drop it at an arbitrary pose.

    Returns ``{"success": bool, "released": bool, "placed_at": [x,y,z], ...}``.
    """
    result: dict[str, Any] = {
        "success": False,
        "released": False,
        "cancelled": False,
        "used_recorded_pose": grasp_pose is not None,
    }
    stage = "init"
    safe_open_m = min(_GRIPPER_MAX_M, _safe_open_distance(open_distance_m))
    try:
        if grasp_pose is not None:
            place6 = [float(v) for v in grasp_pose]
            approach6 = (
                [float(v) for v in pregrasp_pose]
                if pregrasp_pose is not None
                else [place6[0], place6[1], place6[2] + 0.08, *place6[3:]]
            )
        else:
            place6 = [float(v) for v in place_pose]
            approach6 = [place6[0], place6[1], place6[2] + 0.08, *place6[3:]]

        # Table-boundary clamp: keep the release point away from the table
        # edge (recorded grasp poses CAN be near the edge — the box was
        # legitimately picked up there, but releasing there lets the retreat
        # nudge it off). Shift the approach by the same delta so the descent
        # stays vertical-ish and the recorded IK geometry survives.
        if place_bounds:
            clamped = _clamp_place_xy(place6, place_bounds, place_margin_m)
            if clamped is not None:
                result["place_clamped"] = True
                result["place_original"] = place6[:3]
                approach6[0] += clamped[0] - place6[0]
                approach6[1] += clamped[1] - place6[1]
                logger.warning(
                    "put_down: place point (%.3f, %.3f) outside safe table "
                    "bounds %s (margin %.2fm) — clamped to (%.3f, %.3f)",
                    place6[0], place6[1], list(place_bounds), place_margin_m,
                    clamped[0], clamped[1],
                )
                place6 = clamped

        # ── 1. approach above the place spot (IK-failure tolerated: fall
        # through to a direct move — the place pose itself is the gate). ──
        stage = "place_approach"
        _check_cancel(cancel_event, arm, safe_open_m)
        with _motion_lock(actuator):
            approach_ok = arm.move_to(*approach6, duration=move_duration)
        if approach_ok:
            if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                _check_cancel(cancel_event, arm, safe_open_m)  # safe-park + raise
        else:
            logger.info("put_down: approach IK failed; moving directly to place pose")

        # ── 2. descend to the place pose. This one MUST succeed — releasing
        # anywhere else drops the object at an unknown spot. ──────────────
        stage = "place"
        _check_cancel(cancel_event, arm, safe_open_m)
        with _motion_lock(actuator):
            place_ok = arm.move_to(*place6, duration=move_duration)
        if not place_ok:
            return {**result, "stage": stage, "error": "place-pose IK failed (still holding)"}
        if not _wait_motion_cancellable(arm, move_duration, cancel_event):
            _check_cancel(cancel_event, arm, safe_open_m)  # safe-park + raise

        # ── 3. release — VERIFIED physically, never assumed ─────────────
        # After the open, read back the gripper's physical holding evidence
        # (encoder gap + grip torque). Still clamping → retry once at full
        # mechanical open; still clamping after that → report the failure
        # honestly (arm stays at the place pose, object still held) instead
        # of claiming success.
        stage = "release"
        _check_cancel(cancel_event, arm, safe_open_m)
        with _motion_lock(actuator):
            _open_gripper_safe(arm, safe_open_m)
        released = _gripper_holding(arm, default=False) is not True
        if not released:
            logger.warning(
                "put_down: still gripping after open(%.3fm); retrying at full open",
                safe_open_m,
            )
            with _motion_lock(actuator):
                _open_gripper_safe(arm, _GRIPPER_MAX_M)
            released = _gripper_holding(arm, default=False) is not True
        result["released"] = released
        opening = getattr(arm, "gripper_opening_m", None)
        if callable(opening):
            try:
                result["release_opening_m"] = round(float(opening()), 4)
            except Exception:
                pass
        if not released:
            return {**result, "stage": stage,
                    "error": "release failed — jaw still gripping after full open"}
        result["placed_at"] = place6[:3]

        # ── 4. retreat: straight UP first, then back through the approach
        # pose. Retreating directly along the (often shallow) approach used
        # to brush tall objects and tip them over (real machine: round-1
        # put_down knocked the standing box flat). +6cm vertical clears the
        # object before any lateral motion; IK failure on the hop is
        # tolerated (fall through to the approach retreat).
        stage = "retreat"
        _check_cancel(cancel_event, arm, safe_open_m)
        hop = [place6[0], place6[1], place6[2] + 0.06, *place6[3:]]
        with _motion_lock(actuator):
            hop_ok = arm.move_to(*hop, duration=max(0.8, move_duration * 0.5))
        if hop_ok:
            if not _wait_motion_cancellable(arm, max(0.8, move_duration * 0.5), cancel_event):
                _check_cancel(cancel_event, arm, safe_open_m)
        with _motion_lock(actuator):
            retreat_ok = arm.move_to(*approach6, duration=move_duration)
        if retreat_ok:
            if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                _check_cancel(cancel_event, arm, safe_open_m)

        # ── 5. home ─────────────────────────────────────────────────────
        stage = "home"
        _check_cancel(cancel_event, arm, safe_open_m)
        with _motion_lock(actuator):
            home_ok = arm.move_to(*home_pose, duration=move_duration)
        if home_ok:
            if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                _check_cancel(cancel_event, arm, safe_open_m)

        # ── 6. rest state: claws CLOSED ─────────────────────────────────
        # The gripper's single-turn encoder only power-cycles safely with
        # the jaw at the closed stop, so the put-down flow — the natural
        # end of every carry — leaves the claw closed. Close only AFTER the
        # retreat + home (closing earlier would re-grab the object).
        # Best-effort: a failure here must not fail the successful put-down.
        stage = "rest_close"
        try:
            with _motion_lock(actuator):
                arm.grasp(force=0.3)
        except Exception:
            logger.warning("put_down: rest-close failed (non-fatal)",
                           exc_info=True)

        result["success"] = True
        result["stage"] = "done"
        return result

    except GraspCancelled:
        logger.info("put_down cancelled at stage=%s", stage)
        return {**result, "cancelled": True, "stage": stage}
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("put_down pipeline failed at stage=%s", stage)
        # Do NOT open the gripper here: an unexpected mid-carry failure must
        # not drop the object at an arbitrary pose. Cancel already safe-parks.
        return {**result, "stage": stage, "error": str(exc)}


def run_search_once(
    target: str,
    *,
    arm: Any,
    actuator: Any = None,
    segmenter: Any = None,
    camera: Any = None,
    T_hand_eye: Optional[np.ndarray] = None,
    scan_poses: Optional[list] = None,
    home_pose: tuple = (0.27, 0.0, 0.24, 0.0, 0.0, 0.0),
    cancel_event: Optional[threading.Event] = None,
    conf: float = 0.20,
    move_duration: float = 2.0,
    warm_up_frames: int = 3,
    frames: int = 6,
    indicate: bool = True,
    frame_sink: Any = None,
) -> dict:
    """Sweep the eye-in-hand camera across ``scan_poses`` to find ``target``.

    Unlike :func:`run_grasp_once` (which only looks from the current pose), this
    moves the arm through a list of observation poses, runs multi-frame
    detection at each, and stops at the first pose where the target is found —
    optionally reaching toward it ("pointing") without grasping. If no pose sees
    the target, the arm returns home. The gripper is never closed.

    Returns ``{"found": bool, "target": str, "scan_index": int,
    "position_base": [x,y,z], "conf": float, "indicated": bool, ...}``.
    """
    result: dict[str, Any] = {"found": False, "target": target, "cancelled": False}
    try:
        if segmenter is None or camera is None:
            return {**result, "error": "perception not configured"}
        poses = list(scan_poses) if scan_poses else [home_pose]
        K = np.asarray(camera.K, dtype=np.float32)
        scanned = 0
        for idx, pose in enumerate(poses):
            if cancel_event is not None and cancel_event.is_set():
                return {**result, "cancelled": True, "scan_index": idx}
            # move to the observation pose
            with _motion_lock(actuator):
                ok = arm.move_to(*pose, duration=move_duration)
            if not ok:
                logger.info("search: scan pose %d IK failed; skipping", idx)
                continue
            if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                return {**result, "cancelled": True, "scan_index": idx}
            scanned += 1
            # multi-frame detection at this view
            try:
                camera.warm_up(warm_up_frames)
            except Exception:
                logger.debug("camera.warm_up failed (continuing)", exc_info=True)
            best = None
            for _ in range(max(1, frames)):
                color_bgr, depth_mm = camera.get_frame()
                if color_bgr is None or depth_mm is None:
                    continue
                from .perception.ordinary_grasp import estimate_grasps, select_best_grasp
                results = segmenter.predict(color_bgr, conf=conf, only_names={target})
                results = _filter_results_to_target(results, target)
                cand = select_best_grasp(estimate_grasps(results, depth_mm, K))
                if frame_sink is not None:
                    try:
                        frame_sink(color_bgr, depth_mm, results, cand, f"search:{idx}")
                    except Exception:
                        logger.debug("frame_sink failed", exc_info=True)
                if cand is not None and (best is None or cand.conf > best.conf):
                    best = cand
            if best is None:
                continue
            # found — compute the target's base-frame position for reporting
            result["found"] = True
            result["scan_index"] = idx
            result["conf"] = float(best.conf)
            result["center_px"] = list(best.center_px)
            if T_hand_eye is not None:
                from .perception.transforms import transform_grasp_pose_to_base
                with _motion_lock(actuator):
                    tcp_pose = np.asarray(arm.get_tcp_pose(), dtype=np.float64)
                T_cam2base = tcp_pose @ np.asarray(T_hand_eye, dtype=np.float64)
                # Use the SAME transform the (validated) grasp path uses so the
                # reported/pointed position matches reality. grasp6d[:3] is the
                # box's base-frame xyz; we only point near it (no grasp).
                grasp6d, _pre = transform_grasp_pose_to_base(
                    best.position, best.tcp_rotation, T_cam2base,
                    pregrasp_offset_m=0.08, insertion_depth_m=0.0,
                )
                p_base = [float(v) for v in grasp6d[:3]]
                result["position_base"] = p_base
                # A single depth pixel from a far/high scan pose can yield a
                # physically impossible position (e.g. z below the base plane),
                # so only POINT at it when the computed location is plausible for
                # a box on the table AND IK-reachable. Otherwise we simply stay
                # at this scan pose — the camera is already centered on the box,
                # which is itself a natural "found it" indication. Never grasps.
                bx, by, bz = p_base
                # Bounds = HARDWARE-VALIDATED reach, not the old conservative
                # tuning box: real grasps land at x 0.55-0.62 (2026-06-12
                # production logs) and search detected the demo box at x=0.574
                # — an upper bound of 0.50 flagged every real position as
                # implausible, so search found the box but never pointed.
                plausible = (0.15 <= bx <= 0.68 and -0.25 <= by <= 0.25 and 0.0 <= bz <= 0.30)
                result["position_plausible"] = plausible
                if (indicate and plausible
                        and not (cancel_event is not None and cancel_event.is_set())):
                    # Pointing gesture pose: x=0.44 is IK-validated on hardware
                    # (the old 0.34 cap pointed visibly short of far boxes);
                    # check_ik below still gates the actual move.
                    px = min(0.44, max(0.20, bx))
                    py = min(0.14, max(-0.14, by))
                    pz = 0.20
                    try:
                        ik_ok, _ = arm.check_ik(px, py, pz, 0.0, 0.0, 0.0)
                    except Exception:
                        ik_ok = False
                    if ik_ok:
                        with _motion_lock(actuator):
                            arm.move_to(px, py, pz, 0.0, 0.0, 0.0, duration=move_duration)
                        _wait_motion_cancellable(arm, move_duration, cancel_event)
                        result["indicated"] = True
            result["scanned_poses"] = scanned
            return result
        # nothing found anywhere → return home
        if cancel_event is None or not cancel_event.is_set():
            with _motion_lock(actuator):
                arm.move_to(*home_pose, duration=move_duration)
        result["scanned_poses"] = scanned
        return result
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("search pipeline failed")
        return {**result, "error": str(exc)}


def _filter_results_to_target(results: list[Any], target: str) -> list[Any]:
    """Drop detections whose label != ``target`` so the grasp estimator only
    considers the requested object. Rebuilds each result's ``boxes`` / ``masks``
    in place via the same numpy containers the segmenter uses.

    Comparison is case-insensitive and also matches when the requested target
    is a substring of the detected label (e.g. ``"bottle"`` matches
    ``"water bottle"``).
    """
    from .perception.yolo_onnx import _Boxes, _Masks, YoloResult  # local import

    want = target.strip().lower()
    out: list[Any] = []
    for r in results:
        names = getattr(r, "names", {}) or {}
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if boxes is None:
            out.append(r)
            continue
        keep_idx: list[int] = []
        for i in range(len(boxes)):
            cls_id = int(np.asarray(boxes[i].cls[0]).reshape(-1)[0])
            label = (names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id))
            label_l = str(label).lower()
            if label_l == want or want in label_l or label_l in want:
                keep_idx.append(i)
        kept_boxes = _Boxes([boxes[i] for i in keep_idx])
        kept_masks = None
        if masks is not None and getattr(masks, "data", None) is not None and keep_idx:
            data = np.asarray(masks.data)
            kept_masks = _Masks(data[keep_idx])
        out.append(
            YoloResult(
                names=names,
                boxes=kept_boxes,
                masks=kept_masks,
                orig_shape=getattr(r, "orig_shape", (0, 0)),
            )
        )
    return out


__all__ = ["run_grasp_once", "run_put_down_once", "run_search_once", "GraspCancelled"]
