"""Pure grasp-pose geometry — the SINGLE SOURCE OF TRUTH for the executed
base-frame pose, shared by the live pipeline (grasp_service.run_grasp_once) and
the offline real-frame replay harness (tools/grasp_replay.py).

Leaf module: imports only numpy + transforms, NO app/voxedge deps, so the
geometry can be unit-tested and replayed on real captured depth frames without
booting the agent.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from .transforms import transform_grasp_pose_to_base

logger = logging.getLogger(__name__)


def _side_insertion_depth_m(default: float) -> float:
    """Deeper insertion for SIDE grasps so the jaw wraps the box body instead of
    catching its front edge (real machine 2026-06-17: shallow grip → slip).
    Env-tunable REBOT_SIDE_INSERT (m); default 0.04."""
    try:
        v = float(os.environ.get("REBOT_SIDE_INSERT", "") or 0.0)
    except (TypeError, ValueError):
        v = 0.0
    return max(default, v if v > 0 else 0.035)


def finalize_grasp_pose(best, T_cam2base, pregrasp_offset_m, insertion_depth_m):
    """Camera-frame grasp candidate → executed base-frame ``(grasp6d, pre6d)``.

    Pure — no arm, no camera, no I/O. Steps:

      1. camera→base transform with the insertion/pregrasp offset along the
         camera→object ray (``offset_axis_cam=best.position``) — decouples the
         insertion translation from the box-facing approach re-aim so the
         landing point does not swing when the gripper turns to face the box.
      2. TOP grasps: CENTRE on the top-face centroid + vertical bite. The
         insertion along the forward-tilted ray adds ~14mm useful downward bite
         but also ~21mm of HORIZONTAL shift toward the box's far edge (real
         machine 2026-06-17: jaw closed ~2cm off-centre → could not hold). The
         horizontal shift is a pure artifact, so pin x,y to the centroid and
         keep the bite via z (≥28mm below the measured top surface, ≥25mm above
         the table — the 2026-06-12 "抓得深" floor). The pregrasp tracks the same
         x,y shift so the approach path stays consistent.
    """
    method = getattr(best, "method", "legacy")
    # SIDE grasps: advance the insertion along the FACE NORMAL into the box (set
    # on the GraspPose) and deeper, so the jaw wraps the body rather than the
    # front edge. TOP/other grasps keep the camera→object ray (centring path).
    side_axis = getattr(best, "insertion_axis_cam", None)
    if method == "side_face" and side_axis is not None:
        offset_axis = side_axis
        insertion = _side_insertion_depth_m(insertion_depth_m)
    else:
        offset_axis = best.position
        insertion = insertion_depth_m
    grasp6d, pre6d = transform_grasp_pose_to_base(
        best.position,
        best.tcp_rotation,
        T_cam2base,
        pregrasp_offset_m,
        insertion_depth_m=insertion,
        offset_axis_cam=offset_axis,
    )
    try:
        _raw6d, _ = transform_grasp_pose_to_base(
            best.position, best.tcp_rotation, T_cam2base,
            pregrasp_offset_m, insertion_depth_m=0.0,
            offset_axis_cam=offset_axis)
        logger.info("FINDBG2 raw_base_z=%.4f insertion=%.3fm -> committed z=%.4f "
                    "(insertion drop %.1fmm)", float(_raw6d[2]), float(insertion),
                    float(grasp6d[2]), (float(_raw6d[2]) - float(grasp6d[2])) * 1000.0)
    except Exception:
        logger.debug("FINDBG2 failed", exc_info=True)
    if method == "top_face":
        surf = (
            np.asarray(T_cam2base, dtype=np.float64)
            @ np.append(np.asarray(best.position, dtype=np.float64), 1.0)
        )[:3]
        dx = float(surf[0]) - float(grasp6d[0])
        dy = float(surf[1]) - float(grasp6d[1])
        z_bite = max(float(surf[2]) - 0.018, 0.025)
        gz = min(float(grasp6d[2]), z_bite)
        logger.info(
            "grasp: centred top-grasp on centroid (dx %.3f dy %.3f) "
            "bite z %.3f→%.3f (surface %.3f)",
            dx, dy, float(grasp6d[2]), gz, float(surf[2]),
        )
        grasp6d = [float(surf[0]), float(surf[1]), gz,
                   *(float(v) for v in grasp6d[3:])]
        pre6d = [float(pre6d[0]) + dx, float(pre6d[1]) + dy,
                 *(float(v) for v in pre6d[2:])]
    # FWD_FIX: arm lands ~2-3cm short in forward x; nudge grasp+pregrasp forward
    import os as _os
    _fwd = float(_os.environ.get("REBOT_GRASP_FWD_M", "0.025"))
    grasp6d = [grasp6d[0] + _fwd, *grasp6d[1:]]
    pre6d = [pre6d[0] + _fwd, *pre6d[1:]]
    # SIDE-GRASP Z FLOOR (base frame): table-edge noise in the face fit can
    # plan the grip at table level — the wrist then drags the table and the
    # jaw closes on air/box-bottom. A side grasp below ~4cm is never right on
    # this rig (table sits ≈0.005-0.015m): clamp, and lift the pregrasp by the
    # same amount so the approach line stays level.
    if method == "side_face":
        _zmin = float(_os.environ.get("REBOT_SIDE_ZMIN", "0.040"))
        # Per-class floor (2026-07-14): at the generic 0.040 the jaw scrapes
        # the table on cups; short boxes must stay low, so only listed
        # classes get a taller floor.
        import json as _json
        try:
            _by_cls = _json.loads(_os.environ.get(
                "REBOT_SIDE_ZMIN_BY_CLASS", '{"cup": 0.055}'))
            _zmin = float(_by_cls.get(
                str(getattr(best, "class_name", "")).lower(), _zmin))
        except Exception:
            pass
        if float(grasp6d[2]) < _zmin:
            _dz = _zmin - float(grasp6d[2])
            logger.info(
                "grasp: side z %.3f below floor %.3f — raised %.0fmm",
                float(grasp6d[2]), _zmin, _dz * 1000.0,
            )
            grasp6d = [grasp6d[0], grasp6d[1], _zmin, *grasp6d[3:]]
            pre6d = [pre6d[0], pre6d[1], float(pre6d[2]) + _dz, *pre6d[3:]]
    # UNIVERSAL Z FLOOR (2026-07-13): the legacy route committed a grasp at
    # z=-0.005 — below base zero, finger tips into the table (cup incident).
    # NO method may commit a TCP below this floor. Kept low enough (0.015)
    # that legitimate top-face bites (>=0.025) and descriptor plans are
    # untouched; the side_face floor above stays stricter (0.040).
    _zuni = float(_os.environ.get("REBOT_Z_FLOOR_M", "0.015"))
    if float(grasp6d[2]) < _zuni:
        _dzu = _zuni - float(grasp6d[2])
        logger.info(
            "grasp: %s z %.3f below UNIVERSAL floor %.3f — raised %.0fmm",
            method, float(grasp6d[2]), _zuni, _dzu * 1000.0,
        )
        grasp6d = [grasp6d[0], grasp6d[1], _zuni, *grasp6d[3:]]
        pre6d = [pre6d[0], pre6d[1], float(pre6d[2]) + _dzu, *pre6d[3:]]
    return [float(v) for v in grasp6d], [float(v) for v in pre6d]
