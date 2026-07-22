"""One-shot table-boundary calibration → `grasp.place_bounds` for put_down.

Why: put_down replays the recorded grasp pose, which can legitimately sit at
the table EDGE (the box was picked up there) — releasing there lets the jaw
retreat nudge the object off the table (real machine, 2026-06-12 night run).
run_put_down_once clamps the release x/y into `place_bounds` shrunk inward by
`place_margin_m`; this tool measures those bounds.

How: sweep the eye-in-hand camera over the production scan poses, deproject
each depth frame to a base-frame point cloud (T_cam2base = TCP @ hand-eye),
find the dominant horizontal plane near z≈0 (the tabletop) by z-histogram
mode, and take robust x/y percentiles of the plane inliers. The table must be
CLEAR (no objects) — anything on it shadows table area but cannot widen the
bounds, so clutter only makes the result conservative, never unsafe.

Outputs:
  * suggested config snippet (place_bounds: [x_min, x_max, y_min, y_max])
  * RESULT json line (machine-checkable: inlier count, per-pose coverage)
  * --debug-dir: top-down occupancy PNG + raw inlier npz for human review

Usage (agent STOPPED, temp container with device passthrough, same pattern as
grasp_selfcheck):
    python -m ovs_agent.apps.voice_rebot_arm.tools.table_bounds_calib
    python -m ovs_agent.apps.voice_rebot_arm.tools.table_bounds_calib --no-move

SAFETY: default mode MOVES the real arm through the scan poses. A human must
watch and e-stop. --no-move calibrates from the current pose only (narrower
view, still useful for a sanity re-check).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np

# Production observation sweep (config.yaml grasp.scan_poses) — keep in sync.
_SCAN_POSES = [
    (0.27, 0.00, 0.30, 0.0, 0.30, 0.0),
    (0.25, 0.10, 0.30, 0.0, 0.30, 0.35),
    (0.25, -0.10, 0.30, 0.0, 0.30, -0.35),
]

# Depth pixels worth deprojecting: closer than 0.15m is the gripper itself,
# beyond 1.2m is the far wall/floor.
_DEPTH_MIN_M, _DEPTH_MAX_M = 0.15, 1.20
# Tabletop search window around base z=0 and inlier band half-width. The B601
# base sits ON the table, so the surface is near z≈0 (grasp poses put the TCP
# at z 0.05-0.12 = half object height above it).
_Z_SEARCH_M, _Z_BIN_M, _Z_INLIER_M = 0.15, 0.005, 0.012
_STRIDE = 4  # deprojection subsample
_MIN_INLIERS = 20_000  # acceptance gate: fewer means the sweep saw no table


def _load_hand_eye(path: str, log) -> "np.ndarray | None":
    if not os.path.exists(path):
        log.warning("hand-eye npz not found at %s", path)
        return None
    data = np.load(path)
    key = "T_hand_eye" if "T_hand_eye" in data else data.files[0]
    return np.asarray(data[key], dtype=np.float64)


def _prime_camera(cam, log, tries: int = 20) -> bool:
    pipe = getattr(cam, "_pipeline", None)
    if pipe is None:
        return True
    t0 = time.time()
    for i in range(tries):
        try:
            fs = pipe.wait_for_frames(2000)
        except Exception:
            fs = None
        if fs is not None and fs.get_color_frame() is not None and fs.get_depth_frame() is not None:
            log.info("camera primed in %.2fs (%d polls)", time.time() - t0, i + 1)
            return True
    log.warning("camera not primed after %d polls", tries)
    return False


def _deproject_to_base(depth_mm: np.ndarray, K: np.ndarray, T_cam2base: np.ndarray) -> np.ndarray:
    """Subsampled depth image → Nx3 base-frame points (metres)."""
    z = depth_mm[::_STRIDE, ::_STRIDE].astype(np.float64) / 1000.0
    vs, us = np.mgrid[0 : depth_mm.shape[0] : _STRIDE, 0 : depth_mm.shape[1] : _STRIDE]
    ok = (z > _DEPTH_MIN_M) & (z < _DEPTH_MAX_M)
    z, us, vs = z[ok], us[ok].astype(np.float64), vs[ok].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pts_cam = np.column_stack([(us - cx) * z / fx, (vs - cy) * z / fy, z, np.ones_like(z)])
    return (pts_cam @ T_cam2base.T)[:, :3]


# Seed window for the connected-component walk: the IK-validated grasp zone.
# The table the DEMO happens on is by definition the surface under this zone;
# other surfaces at a similar height (far benches, shelves) are rejected by
# connectivity, not by z alone.
_SEED_X, _SEED_Y = (0.28, 0.55), (-0.25, 0.25)
_GRID_RES_M = 0.01
_CELL_MIN_PTS = 3


def _table_inliers(points: np.ndarray, log) -> "tuple[np.ndarray, float] | None":
    """Tilt-tolerant RANSAC plane near z≈0 (a z-histogram is NOT enough: a
    1-2° table/hand-eye tilt walks the surface out of a flat ±12mm band
    across 1m of table — first real-machine run lost the whole near zone)."""
    near = points[(points[:, 2] > -_Z_SEARCH_M) & (points[:, 2] < _Z_SEARCH_M)]
    if len(near) < 1000:
        log.warning("only %d points near z=0 — no table visible?", len(near))
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(400):
        p0, p1, p2 = near[rng.choice(len(near), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            continue
        n = n / norm
        if abs(n[2]) < 0.966:  # >15° from horizontal → not a tabletop
            continue
        if n[2] < 0:
            n = -n
        d = -float(n @ p0)
        cnt = int((np.abs(near @ n + d) < 0.010).sum())
        if best is None or cnt > best[0]:
            best = (cnt, n, d)
    if best is None:
        log.warning("RANSAC found no horizontal plane")
        return None
    _, n, d = best
    inl = near[np.abs(near @ n + d) < _Z_INLIER_M]
    z_table = float(-d / n[2])  # plane height at x=y=0
    log.info(
        "table plane z(0,0)=%.3fm tilt=%.1f° — %d inliers (of %d near-z points)",
        z_table, float(np.degrees(np.arccos(min(1.0, abs(n[2]))))), len(inl), len(near),
    )
    return inl, z_table


def _grasp_zone_component(inliers: np.ndarray, log) -> "np.ndarray | None":
    """Keep only the connected table patch under the grasp zone. Surfaces at
    the same height elsewhere (far bench, shelf edge) share the plane but not
    the demo table — connectivity on a 1cm occupancy grid separates them, and
    the component's extent IS the table edge (modulo camera FOV)."""
    xi = np.floor(inliers[:, 0] / _GRID_RES_M).astype(int)
    yi = np.floor(inliers[:, 1] / _GRID_RES_M).astype(int)
    from collections import Counter, deque

    counts = Counter(zip(xi.tolist(), yi.tolist()))
    occ = {c for c, k in counts.items() if k >= _CELL_MIN_PTS}
    seeds = [
        c for c in occ
        if _SEED_X[0] <= c[0] * _GRID_RES_M <= _SEED_X[1]
        and _SEED_Y[0] <= c[1] * _GRID_RES_M <= _SEED_Y[1]
    ]
    if not seeds:
        log.warning("no occupied cells in the grasp-zone seed window")
        return None
    seed = max(seeds, key=lambda c: counts[c])
    comp = {seed}
    q = deque([seed])
    while q:
        cx, cy = q.popleft()
        for dx in (-2, -1, 0, 1, 2):  # 2-cell reach jumps small speckle gaps
            for dy in (-2, -1, 0, 1, 2):
                nb = (cx + dx, cy + dy)
                if nb in occ and nb not in comp:
                    comp.add(nb)
                    q.append(nb)
    keep = np.fromiter(
        ((int(a), int(b)) in comp for a, b in zip(xi, yi)), bool, len(inliers)
    )
    log.info("grasp-zone component: %d/%d cells, %d/%d points",
             len(comp), len(occ), int(keep.sum()), len(inliers))
    return inliers[keep]


def _save_debug(inliers: np.ndarray, bounds: list, z_table: float, out_dir: str, log) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "table_inliers.npz"),
        inliers=inliers.astype(np.float32), bounds=np.asarray(bounds), z_table=z_table,
    )
    try:
        import cv2

        # Top-down occupancy: 1px = 5mm over the inlier extent + margin.
        res = 0.005
        x0, y0 = inliers[:, 0].min() - 0.05, inliers[:, 1].min() - 0.05
        w = int((inliers[:, 1].max() + 0.05 - y0) / res) + 1
        h = int((inliers[:, 0].max() + 0.05 - x0) / res) + 1
        img = np.zeros((h, w, 3), np.uint8)
        ui = ((inliers[:, 0] - x0) / res).astype(int)
        vi = ((inliers[:, 1] - y0) / res).astype(int)
        img[ui, vi] = (180, 180, 180)

        def px(x: float, y: float) -> tuple:
            return (int((y - y0) / res), int((x - x0) / res))

        cv2.rectangle(img, px(bounds[0], bounds[2]), px(bounds[1], bounds[3]), (0, 200, 0), 1)
        cv2.circle(img, px(0.0, 0.0), 4, (0, 0, 255), -1)  # arm base
        path = os.path.join(out_dir, "table_topdown.png")
        cv2.imwrite(path, img)
        log.info("debug: %s (grey=table inliers, green=bounds, red=arm base)", path)
    except Exception:
        log.exception("debug PNG failed (npz still saved)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-move", action="store_true",
                    help="Calibrate from the CURRENT pose only; arm does not move.")
    ap.add_argument("--frames", type=int, default=3, help="Depth frames per pose (median-merged).")
    ap.add_argument("--move-duration", type=float,
                    default=float(os.environ.get("REBOT_MOVE_DURATION", "1.4")))
    ap.add_argument("--debug-dir", default="/tmp/table-bounds",
                    help="Where to dump the inlier npz + top-down PNG ('' = skip).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("table_bounds_calib")

    from ovs_agent.apps.voice_rebot_arm.perception.camera import make_camera
    from ovs_agent.apps.voice_rebot_arm.rebot_actuator import _make_rebot_arm

    hand_eye = _load_hand_eye(
        os.environ.get("REBOT_HAND_EYE", "/opt/rebot-models/hand_eye.npz"), log
    )
    if hand_eye is None:
        print("RESULT", json.dumps({"error": "no hand-eye"}))
        return 1

    cam = make_camera({"camera": {
        "type": "orbbec_gemini2", "color_width": 1280, "color_height": 720, "fps": 30,
    }})
    cam.open()
    _prime_camera(cam, log)

    actuator = _make_rebot_arm({
        "channel": os.environ.get("REBOT_CHANNEL", "auto"),
        "channel_match": {"usb_id": ["2e88:4603"], "vendor": ["hdsc"]},
        "channel_ambiguous": "error",
        "repo_root": os.environ.get("REBOT_REPO_ROOT", "/opt/rebot"),
        "move_duration": args.move_duration,
        "grasp_force": "0.30",
        "open_distance_m": 0.09,
    })
    actuator.connect()
    arm = actuator.robot
    K = np.asarray(cam.K, dtype=np.float64)

    poses = [None] if args.no_move else list(_SCAN_POSES)
    all_pts: list[np.ndarray] = []
    per_pose: list[int] = []
    try:
        if not args.no_move:
            actuator.set_torque(True)
        for pose in poses:
            if pose is not None:
                log.info("moving to scan pose %s", pose)
                if not arm.move_to(*pose, duration=args.move_duration):
                    log.warning("scan pose IK failed, skipping: %s", pose)
                    per_pose.append(0)
                    continue
                time.sleep(args.move_duration + 0.4)  # settle: TCP read must match the frame
            tcp = np.asarray(arm.get_tcp_pose(), dtype=np.float64)
            T_cam2base = tcp @ hand_eye
            depths = []
            for _ in range(args.frames):
                cam.warm_up(1)
                _, depth_mm = cam.get_frame()
                if depth_mm is not None:
                    depths.append(depth_mm.astype(np.float32))
            if not depths:
                log.warning("no depth frames at pose %s", pose)
                per_pose.append(0)
                continue
            med = np.median(np.stack(depths), axis=0)
            pts = _deproject_to_base(med, K, T_cam2base)
            all_pts.append(pts)
            per_pose.append(len(pts))
            log.info("pose %s: %d points", pose, len(pts))
        if not args.no_move:
            log.info("returning home")
            arm.move_to(0.27, 0.0, 0.24, 0.0, 0.0, 0.0, duration=args.move_duration)
            time.sleep(args.move_duration + 0.2)
    finally:
        try:
            actuator.disconnect()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass

    if not all_pts:
        print("RESULT", json.dumps({"error": "no points captured", "per_pose": per_pose}))
        return 1
    pts = np.concatenate(all_pts)
    if args.debug_dir:
        # Raw near-plane cloud FIRST — lets the plane/component logic be
        # re-tuned offline without another arm sweep.
        os.makedirs(args.debug_dir, exist_ok=True)
        raw = pts[(pts[:, 2] > -_Z_SEARCH_M) & (pts[:, 2] < _Z_SEARCH_M)]
        np.savez_compressed(
            os.path.join(args.debug_dir, "near_cloud_raw.npz"),
            points=raw.astype(np.float32),
        )
    found = _table_inliers(pts, log)
    if found is None:
        print("RESULT", json.dumps({"error": "no table plane found", "per_pose": per_pose}))
        return 1
    inliers, z_table = found
    comp = _grasp_zone_component(inliers, log)
    if comp is None or len(comp) < 1000:
        print("RESULT", json.dumps({"error": "no table patch under grasp zone",
                                    "per_pose": per_pose}))
        return 1
    inliers = comp

    # Robust extent: 1/99 percentiles reject stray plane-height pixels
    # (sensor speckle, far shelf edges at table height).
    raw_bounds = [
        round(float(np.percentile(inliers[:, 0], 1)), 3),
        round(float(np.percentile(inliers[:, 0], 99)), 3),
        round(float(np.percentile(inliers[:, 1], 1)), 3),
        round(float(np.percentile(inliers[:, 1], 99)), 3),
    ]
    # Intersect with the IK-reachable box: a measured extent only matters
    # where the arm can place, and a min-side extent is usually the camera
    # FOV limit, NOT the table edge (the base itself sits on the table —
    # first real-machine run measured x_min 0.335 purely because the gripper
    # blocks the near view). Real edges show as below-plane drop-off points;
    # eyeball those in near_cloud_raw.npz before trusting a tighter value.
    # x_min: the base sits ON the table, so the near side is always table —
    # use the reach limit, never the (FOV-limited) measurement.
    reach = (0.15, 0.65, -0.45, 0.45)
    bounds = [
        reach[0],
        min(raw_bounds[1], reach[1]),
        max(raw_bounds[2], reach[2]),
        min(raw_bounds[3], reach[3]),
    ]
    if args.debug_dir:
        _save_debug(inliers, bounds, z_table, args.debug_dir, log)

    ok = len(inliers) >= _MIN_INLIERS
    print("RESULT", json.dumps({
        "ok": ok,
        "place_bounds": bounds,
        "raw_bounds": raw_bounds,
        "z_table_m": round(z_table, 4),
        "inliers": int(len(inliers)),
        "per_pose_points": per_pose,
        "span_m": [round(bounds[1] - bounds[0], 3), round(bounds[3] - bounds[2], 3)],
    }))
    print()
    print("# config.yaml → metadata.grasp:")
    print(f"place_bounds: [{bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]}]")
    if not ok:
        print(f"# WARNING: only {len(inliers)} inliers (<{_MIN_INLIERS}) — "
              "verify the camera actually saw the tabletop before trusting this.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
