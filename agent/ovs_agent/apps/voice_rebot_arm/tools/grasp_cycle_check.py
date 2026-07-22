"""Unattended grasp→put-back cycle check (real machine, ARM MOVES).

Drives N full pick-and-place cycles through the PRODUCTION pipeline
(run_grasp_once → run_put_down_once with the recorded poses) and reports a
machine-readable summary: success counts, per-stage timings, measured widths,
servo/re-observe activity. Aborts after ``--max-consec-fail`` consecutive
grasp failures (operator instruction: "拾取不上就停").

Run like grasp_selfcheck (agent stopped, privileged temp container):
    python grasp_cycle_check.py --cycles 5 --target box
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/opt/slv/agent")

from ovs_agent.apps.voice_rebot_arm.tools.grasp_selfcheck import (  # noqa: E402
    _actuator_cfg,
    _grasp_cfg,
    _load_hand_eye,
    _make_segmenter,
    _prime_camera,
)

# Mirror config.yaml metadata.grasp — keep in sync (this tool bypasses the
# plugin's config threading, so drift here silently tests a different system
# than production: move_duration 2.0-vs-1.4 and the old low scan poses both
# shipped that way before 2026-06-13).
SCAN_POSES = [
    (0.27, 0.00, 0.30, 0.0, 0.30, 0.0),
    (0.25, 0.10, 0.30, 0.0, 0.30, 0.35),
    (0.25, -0.10, 0.30, 0.0, 0.30, -0.35),
]
PLAUSIBLE_BOX = [0.05, 0.85, -0.50, 0.50, -0.02, 0.50]
PLACE_BOUNDS = [0.20, 0.60, -0.26, 0.40]
MOVE_DURATION = 1.4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--target", default="box")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--force", type=float, default=0.8)
    ap.add_argument("--insertion", type=float, default=0.025)
    ap.add_argument("--max-consec-fail", type=int, default=2)
    ap.add_argument("--move-duration", type=float, default=MOVE_DURATION)
    ap.add_argument("--save-frames", default=None,
                    help="dir to save first color jpg + depth npy")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("grasp_cycle")
    gcfg = _grasp_cfg()

    from ovs_agent.apps.voice_rebot_arm.perception.camera import make_camera
    from ovs_agent.apps.voice_rebot_arm.grasp_service import (
        run_grasp_once,
        run_put_down_once,
    )
    from ovs_agent.apps.voice_rebot_arm.rebot_actuator import _make_rebot_arm

    seg = _make_segmenter(gcfg, log)
    cam = make_camera({"camera": dict(gcfg["camera"])})
    cam.open()
    _prime_camera(cam, log)
    hand_eye = _load_hand_eye(gcfg["hand_eye_path"], log)
    if hand_eye is None:
        print("SUMMARY", json.dumps({"ok": False, "error": "no hand-eye"}))
        return 1

    if args.save_frames:
        import cv2
        for _ in range(10):
            c, d = cam.get_frame()
            if c is not None and d is not None:
                cv2.imwrite(f"{args.save_frames}/cycle_color.jpg", c)
                np.save(f"{args.save_frames}/cycle_depth.npy", d)
                break

    log.info("connecting actuator — ARM WILL MOVE")
    actuator = _make_rebot_arm(_actuator_cfg())
    actuator.connect()
    grasps: list[dict] = []
    putdowns: list[dict] = []
    consec_fail = 0
    try:
        for i in range(args.cycles):
            log.info("── cycle %d/%d: grasp ──", i + 1, args.cycles)
            res = run_grasp_once(
                args.target,
                arm=actuator.robot,
                actuator=actuator,
                segmenter=seg,
                camera=cam,
                T_hand_eye=hand_eye,
                scan_poses=SCAN_POSES,
                cancel_event=threading.Event(),
                conf=args.conf,
                insertion_depth_m=args.insertion,
                grasp_force=args.force,
                open_distance_m=float(str(gcfg["open_distance_m"]).strip() or 0.06)
                if not isinstance(gcfg["open_distance_m"], (int, float))
                else float(gcfg["open_distance_m"]),
                plausible_box=PLAUSIBLE_BOX,
                move_duration=args.move_duration,
            )
            grasps.append(res)
            print("GRASP", i + 1, json.dumps(res, ensure_ascii=False, default=str))
            if not res.get("success"):
                consec_fail += 1
                if consec_fail >= args.max_consec_fail:
                    log.warning("%d consecutive failures — stopping per policy",
                                consec_fail)
                    break
                continue
            consec_fail = 0
            time.sleep(1.0)
            pd = run_put_down_once(
                arm=actuator.robot,
                actuator=actuator,
                grasp_pose=res.get("grasp_pose"),
                pregrasp_pose=res.get("pregrasp_pose"),
                open_distance_m=float(res.get("open_distance_m", 0.09)),
                cancel_event=threading.Event(),
                move_duration=args.move_duration,
                place_bounds=PLACE_BOUNDS,
            )
            putdowns.append(pd)
            print("PUTDOWN", i + 1, json.dumps(pd, ensure_ascii=False, default=str))
            if not pd.get("success"):
                log.warning("put_down failed — stopping (box state unknown)")
                break
            time.sleep(1.0)
    finally:
        actuator.disconnect()
        try:
            cam.close()
        except Exception:
            pass

    ok_g = sum(1 for g in grasps if g.get("success"))
    ok_p = sum(1 for p in putdowns if p.get("success"))
    summary = {
        "cycles_run": len(grasps),
        "grasp_ok": ok_g,
        "putdown_ok": ok_p,
        "widths_m": [round(float(g.get("jaw_width_m", 0)), 4) for g in grasps],
        "attempts": [g.get("attempt") for g in grasps],
        "reobserved": [bool(g.get("reobserved")) for g in grasps],
        "servo_drift_mm": [g.get("servo_drift_mm") for g in grasps],
        "stages_failed": [g.get("stage") for g in grasps if not g.get("success")],
        "stage_ms_last": grasps[-1].get("stage_ms") if grasps else None,
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    return 0 if ok_g == len(grasps) and grasps else 1


if __name__ == "__main__":
    sys.exit(main())
