"""Standalone grasp self-check — drive one real grasp without voice/LLM.

The grasp_object tool only fires via an LLM tool_call over a live voice turn,
and the agent process holds the serial bus + camera exclusively. To self-verify
the full grasp pipeline (detect → camera-frame pose → hand-eye → base pose →
pregrasp → grasp → lift) you must STOP the agent container first, then run this
in a temp container with the same device passthrough.

It wires the exact same objects the GraspPlugin builds at runtime:
  * RebotArmActuator via the rebot_arm factory (channel 'auto' = B601-DM USB id)
  * YoloOnnxSegmenter (box vocab, CPU EP)
  * Orbbec Gemini2 camera
  * hand-eye npz (T_result / T_hand_eye)

Three staged modes (safest → moves the arm):
  --detect-only  capture+detect; camera-frame pose only; NO arm at all.
  --pose-only    connect + read live TCP pose, compute the BASE-frame grasp
                 pose, print reachability; torque OFF, arm does NOT move.
  (default)      full grasp: open → pregrasp → grasp_move → grasp → lift.

Usage (inside the container, agent STOPPED so the bus/camera are free):
    python -m ovs_agent.apps.voice_rebot_arm.tools.grasp_selfcheck --detect-only
    python -m ovs_agent.apps.voice_rebot_arm.tools.grasp_selfcheck --pose-only
    python -m ovs_agent.apps.voice_rebot_arm.tools.grasp_selfcheck --target box

SAFETY: the default mode MOVES the real arm. A human must watch and e-stop.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading

import numpy as np


def _grasp_cfg() -> dict:
    """The metadata.grasp block resolved with container-default paths.

    Mirrors the env contract the GraspPlugin reads so this tool exercises the
    SAME detector as production. In particular the vocab-decoupled ("embin")
    detector: set REBOT_GRASP_MODEL → the embin engine and REBOT_TEXT_ENCODER →
    the text-PE encoder, and the box vocab via REBOT_GRASP_CLASSES (JSON list).
    Without REBOT_TEXT_ENCODER this stays on the legacy baked-vocab engine
    (whose close-up jaw-width estimate is unreliable — so embin is what a real
    pick validation needs).
    """
    classes_env = os.environ.get("REBOT_GRASP_CLASSES")
    classes = ["box", "cardboard box", "carton", "package"]
    if classes_env:
        try:
            parsed = json.loads(classes_env)
            if isinstance(parsed, list) and parsed:
                classes = [str(c) for c in parsed]
        except (ValueError, TypeError):
            pass
    return {
        "yolo_model_path": os.environ.get(
            "REBOT_GRASP_MODEL", "/opt/rebot-models/yoloe-26s-seg-box.onnx"
        ),
        "yolo_classes": classes,
        # embin (vocab-decoupled) detector — empty text_encoder ⇒ legacy baked
        # engine, byte-identical to before.
        "text_encoder_path": os.environ.get("REBOT_TEXT_ENCODER", ""),
        "embin_pad_slots": int(os.environ.get("REBOT_EMBIN_PAD_SLOTS", "16")),
        "onnx_providers": ["CPUExecutionProvider"],
        "camera": {
            "type": "orbbec_gemini2",
            "color_width": 1280,
            "color_height": 720,
            "fps": 30,
        },
        "hand_eye_path": os.environ.get(
            "REBOT_HAND_EYE", "/opt/rebot-models/hand_eye.npz"
        ),
        "open_distance_m": float(os.environ.get("REBOT_GRASP_OPEN_DIST", "0.06")),
        "grasp_force": float(os.environ.get("REBOT_GRASP_FORCE", "0.30")),
    }


def _make_segmenter(gcfg: dict, log=None):
    """Build the detector EXACTLY as the GraspPlugin does, so both self-check
    tools exercise the production path. With REBOT_TEXT_ENCODER set this is the
    vocab-decoupled ("embin") engine: encode the live yolo_classes to text-PE
    rows and feed them as class_embeddings (the engine carries no baked head).
    Without it, the legacy baked-vocab engine (byte-identical to before).
    """
    from ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx import YoloOnnxSegmenter

    if log is not None:
        log.info("loading segmenter %s", gcfg["yolo_model_path"])
    seg_kwargs: dict = {"providers": tuple(gcfg["onnx_providers"])}
    if gcfg.get("text_encoder_path"):
        from ovs_agent.apps.voice_rebot_arm.perception.text_pe import (
            TextPromptEncoder,
        )

        encoder = TextPromptEncoder(
            gcfg["text_encoder_path"], pad_slots=int(gcfg["embin_pad_slots"])
        )
        seg_kwargs["class_embeddings"] = encoder.encode(list(gcfg["yolo_classes"]))
        seg_kwargs["active_n"] = encoder.active_n
        if log is not None:
            log.info(
                "embin mode — %d class embeddings from %s",
                encoder.active_n, gcfg["text_encoder_path"],
            )
    return YoloOnnxSegmenter(
        gcfg["yolo_model_path"], list(gcfg["yolo_classes"]), **seg_kwargs
    )


def _actuator_cfg() -> dict:
    return {
        "channel": os.environ.get("REBOT_CHANNEL", "auto"),
        "channel_match": {"usb_id": ["2e88:4603"], "vendor": ["hdsc"]},
        "channel_ambiguous": "error",
        "repo_root": os.environ.get("REBOT_REPO_ROOT", "/opt/rebot"),
        "move_duration": float(os.environ.get("REBOT_MOVE_DURATION", "2.0")),
        "grasp_force": os.environ.get("REBOT_GRASP_FORCE", "0.30"),
        "open_distance_m": float(os.environ.get("REBOT_OPEN_DIST", "0.09")),
    }


def _prime_camera(cam, log, tries: int = 20) -> bool:
    """Drain the Orbbec startup so the stream is warm before any 500ms
    ``get_frame``. On a fresh open the first ~0.2-1s of frames arrive slower
    than the baked 500ms wait, so the high-level get_frame returns None until
    the depth/color sync settles. We poll the pipeline with a long timeout
    until a synced frame lands, then the steady-state 500ms calls succeed.
    """
    import time as _t

    pipe = getattr(cam, "_pipeline", None)
    if pipe is None:
        return True
    t0 = _t.time()
    for i in range(tries):
        try:
            fs = pipe.wait_for_frames(2000)
        except Exception:
            fs = None
        if fs is not None and fs.get_color_frame() is not None and fs.get_depth_frame() is not None:
            log.info("camera primed in %.2fs (%d polls)", _t.time() - t0, i + 1)
            return True
    log.warning("camera not primed after %d polls", tries)
    return False


def _load_hand_eye(path: str, log) -> "np.ndarray | None":
    if not os.path.exists(path):
        log.warning("hand-eye npz not found at %s", path)
        return None
    data = np.load(path)
    key = "T_hand_eye" if "T_hand_eye" in data else data.files[0]
    he = np.asarray(data[key], dtype=np.float64)
    log.info("hand-eye loaded key=%s\n%s", key, he)
    return he


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="box")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--detect-only", action="store_true",
                    help="Capture+detect only; camera-frame pose; NO arm.")
    ap.add_argument("--pose-only", action="store_true",
                    help="Connect+read TCP, compute base pose; arm does NOT move.")
    ap.add_argument("--search", action="store_true",
                    help="Sweep scan_poses to FIND the target + point at it (no grasp).")
    ap.add_argument("--save-frames", default=None,
                    help="dir to dump best color jpg + depth npy + mask + K + "
                         "up_hint for offline geometry diagnosis (no device)")
    ap.add_argument("--frames", type=int, default=8,
                    help="Frames to scan for the best detection (detect/pose).")
    ap.add_argument("--open-dist", type=float, default=None,
                    help="Override pre-grasp jaw open width (m); else config 0.06. "
                         "Must exceed the detected box width to fit around it.")
    ap.add_argument("--force", type=float, default=None,
                    help="Override compliant grasp force (Nm); else config 0.30.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("grasp_selfcheck")
    gcfg = _grasp_cfg()

    from ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx import YoloOnnxSegmenter
    from ovs_agent.apps.voice_rebot_arm.perception.camera import make_camera
    from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import (
        estimate_grasps,
        select_best_grasp,
    )

    seg = _make_segmenter(gcfg, log)
    log.info("opening camera %s", gcfg["camera"]["type"])
    cam = make_camera({"camera": dict(gcfg["camera"])})
    cam.open()
    _prime_camera(cam, log)
    hand_eye = _load_hand_eye(gcfg["hand_eye_path"], log)

    try:
        # ── stage 1: perception (camera-frame), multi-frame best ─────
        if args.detect_only or args.pose_only:
            from ovs_agent.apps.voice_rebot_arm.rebot_actuator import _make_rebot_arm
            from ovs_agent.apps.voice_rebot_arm.perception.transforms import (
                transform_grasp_pose_to_base,
            )

            K = np.asarray(cam.K, dtype=np.float32)
            # pose-only: connect the arm FIRST and read the live TCP pose so the
            # detector gets up_hint_cam (gravity-up in camera frame). Without it
            # estimate_grasp skips the TOP-FACE plane fit and falls back to the
            # 2D silhouette, which on an obliquely-viewed box MERGES the top+side
            # faces and reads a ~2x-inflated jaw width — the exact reason the
            # standalone check disagreed with production (which always supplies
            # up_hint). detect_only stays arm-free (pure camera probe, up=None).
            actuator = None
            up_hint = None
            tcp_pose = None
            if args.pose_only:
                log.info("connecting arm to read live TCP pose (no motion is commanded)")
                actuator = _make_rebot_arm(_actuator_cfg())
                actuator.connect()
                # NO motion: get_tcp_pose only calls the SDK _request_and_poll
                # (state read) + FK. Keep torque ON (connect() started the
                # controller); set_torque(False) would tear down the ArmEndPos
                # controller and its 'damiao' ctrl-map entry → KeyError.
                tcp_pose = np.asarray(actuator.robot.get_tcp_pose(), dtype=np.float64)
                r_cam2base = (tcp_pose @ np.asarray(hand_eye, dtype=np.float64))[:3, :3] \
                    if hand_eye is not None else None
                if r_cam2base is not None:
                    up_hint = r_cam2base.T @ np.array([0.0, 0.0, 1.0])
            try:
                best = None
                best_depth = None
                best_color = None
                best_mask = None
                ndet = 0
                for _f in range(args.frames):
                    cam.warm_up(1)
                    color_bgr, depth_mm = cam.get_frame()
                    if color_bgr is None or depth_mm is None:
                        continue
                    results = seg.predict(color_bgr, conf=args.conf, only_names={args.target})
                    n = sum(len(getattr(r, "boxes", []) or []) for r in results)
                    ndet = max(ndet, n)
                    cand = select_best_grasp(
                        estimate_grasps(results, depth_mm, K, depth_quantile=0.5,
                                        up_hint_cam=up_hint)
                    )
                    if cand is not None and (best is None or cand.conf > best.conf):
                        best, best_depth, best_color = cand, depth_mm, color_bgr
                        # keep the instance mask of the best detection too, for
                        # offline analysis of the width estimate.
                        from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import (
                            _depth_mask, _rect_points,
                        )
                        try:
                            r0 = results[0]
                            bb = tuple(int(v) for v in np.asarray(r0.boxes[0].xyxy[0])[:4])
                            rp = _rect_points(r0, 0, depth_mm.shape, bb)
                            best_mask = _depth_mask(r0, 0, depth_mm.shape, rp)
                        except Exception:
                            best_mask = None
                # Optional: dump the real color+depth(+mask) of the best frame so
                # the geometry can be replayed/diagnosed offline (no device).
                if args.save_frames and best_depth is not None:
                    import os as _os
                    import cv2 as _cv2
                    _os.makedirs(args.save_frames, exist_ok=True)
                    _cv2.imwrite(f"{args.save_frames}/best_color.jpg", best_color)
                    np.save(f"{args.save_frames}/best_depth_mm.npy", best_depth)
                    if best_mask is not None:
                        np.save(f"{args.save_frames}/best_mask.npy", best_mask)
                    np.save(f"{args.save_frames}/K.npy", np.asarray(K))
                    if up_hint is not None:
                        np.save(f"{args.save_frames}/up_hint_cam.npy", np.asarray(up_hint))
                    # T_cam2base = live TCP pose @ hand-eye — lets the offline
                    # replay reproduce the FULL base-frame pose incl. the real
                    # hand-eye calibration (pose-only path only; detect-only has
                    # no arm so tcp_pose is None).
                    if tcp_pose is not None and hand_eye is not None:
                        np.save(f"{args.save_frames}/T_cam2base.npy",
                                np.asarray(tcp_pose, dtype=np.float64)
                                @ np.asarray(hand_eye, dtype=np.float64))
                    log.info("saved frames to %s", args.save_frames)
                log.info("multi-frame: best conf=%s over %d frames",
                         None if best is None else round(float(best.conf), 3), args.frames)
                cam_out = {
                    "num_detections": ndet,
                    "best": None if best is None else {
                        "class": best.class_name,
                        "conf": float(best.conf),
                        "center_px": list(best.center_px),
                        "position_cam_m": [float(v) for v in best.position],
                        "jaw_width_m": float(best.jaw_width_m),
                        "method": getattr(best, "method", "legacy"),
                    },
                }
                log.info("camera-frame detection: %s", json.dumps(cam_out, ensure_ascii=False))

                if args.detect_only:
                    print("RESULT", json.dumps({"detect_only": True, **cam_out}, ensure_ascii=False))
                    return 0 if best is not None else 1

                if best is None:
                    print("RESULT", json.dumps({"pose_only": True, "error": "no detection", **cam_out}))
                    return 1

                if hand_eye is None:
                    print("RESULT", json.dumps({"pose_only": True, "error": "no hand-eye"}))
                    return 1
                T_cam2base = tcp_pose @ hand_eye
                # mirror production geometry: full insertion depth + the
                # camera→object ray offset axis (decouples the insertion
                # translation from the box-facing approach re-aim, 2026-06-16).
                grasp6d, pre6d = transform_grasp_pose_to_base(
                    best.position, best.tcp_rotation, T_cam2base, 0.08,
                    insertion_depth_m=0.025,
                    offset_axis_cam=best.position,
                )
                # Reachability: solve IK only (no motion) for both poses.
                pre_ok, pre_err = actuator.robot.check_ik(*pre6d)
                gr_ok, gr_err = actuator.robot.check_ik(*grasp6d)
                out = {
                    "pose_only": True,
                    **cam_out,
                    "tcp_pose_xyz": [float(tcp_pose[i, 3]) for i in range(3)],
                    "grasp_pose_base": [float(v) for v in grasp6d],
                    "pregrasp_pose_base": [float(v) for v in pre6d],
                    "pregrasp_ik": {"reachable": bool(pre_ok), "err": round(float(pre_err), 5)},
                    "grasp_ik": {"reachable": bool(gr_ok), "err": round(float(gr_err), 5)},
                }
                print("RESULT", json.dumps(out, ensure_ascii=False))
                return 0 if (pre_ok and gr_ok) else 2
            finally:
                if actuator is not None:
                    actuator.disconnect()

        # ── search: sweep scan_poses to FIND + point at (no grasp) ───
        if args.search:
            from ovs_agent.apps.voice_rebot_arm.rebot_actuator import _make_rebot_arm
            from ovs_agent.apps.voice_rebot_arm.grasp_service import run_search_once

            scan_poses = [
                (0.27, 0.00, 0.26, 0.0, 0.0, 0.0),
                (0.25, 0.10, 0.26, 0.0, 0.0, 0.35),
                (0.25, -0.10, 0.26, 0.0, 0.0, -0.35),
            ]
            log.info("building actuator (connect + torque on) — ARM WILL SWEEP")
            actuator = _make_rebot_arm(_actuator_cfg())
            actuator.connect()
            cancel = threading.Event()
            try:
                res = run_search_once(
                    args.target,
                    arm=actuator.robot,
                    actuator=actuator,
                    segmenter=seg,
                    camera=cam,
                    T_hand_eye=hand_eye,
                    scan_poses=scan_poses,
                    cancel_event=cancel,
                    conf=args.conf,
                )
                print("RESULT", json.dumps(res, ensure_ascii=False, default=str))
                return 0 if res.get("found") else 1
            finally:
                actuator.disconnect()

        # ── stage 3: full grasp (MOVES the arm) ──────────────────────
        from ovs_agent.apps.voice_rebot_arm.rebot_actuator import _make_rebot_arm
        from ovs_agent.apps.voice_rebot_arm.grasp_service import run_grasp_once

        log.info("building actuator (connect + torque on) — ARM WILL MOVE")
        actuator = _make_rebot_arm(_actuator_cfg())
        actuator.connect()
        log.info("torque_enabled=%s obs=%s",
                 actuator.torque_enabled, actuator.get_cached_observation())
        cancel = threading.Event()
        try:
            res = run_grasp_once(
                args.target,
                arm=actuator.robot,
                actuator=actuator,
                segmenter=seg,
                camera=cam,
                T_hand_eye=hand_eye,
                scan_poses=[
                    (0.27, 0.00, 0.26, 0.0, 0.0, 0.0),
                    (0.25, 0.10, 0.26, 0.0, 0.0, 0.35),
                    (0.25, -0.10, 0.26, 0.0, 0.0, -0.35),
                ],
                cancel_event=cancel,
                conf=args.conf,
                open_distance_m=args.open_dist if args.open_dist is not None else gcfg["open_distance_m"],
                grasp_force=args.force if args.force is not None else gcfg["grasp_force"],
            )
            print("RESULT", json.dumps(res, ensure_ascii=False, default=str))
            return 0 if res.get("success") else 1
        finally:
            actuator.disconnect()
    finally:
        try:
            cam.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
