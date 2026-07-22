"""Re-measure the B601-DM IK reachability envelope on the real arm.

Pure kinematics: connects the arm (NO calibration board, NO camera, NO object),
loops a pose grid, calls ``arm.check_ik`` for each — the SAME solver the grasp
pipeline uses — and writes a CSV (x,y,z,pitch,yaw,ok,err) in the format the
synthetic harness _IKEnvelope reads. The arm does NOT move (check_ik only solves
IK; no move_to is issued).

Extends the old grid DOWN to the real table height (z < 0.08): on this rig the
table sits ~5cm below the arm base, so side grasps land at z≈0.04 — which the
old envelope (z_min 0.08) wrongly called unreachable, making sim pessimistic.

Run (agent stopped, temp container with --device /dev/ttyACM0):
    python -m ovs_agent.apps.voice_rebot_arm.tools.ik_envelope_scan /out/ik_envelope.csv
"""
from __future__ import annotations

import csv
import sys
import time

# extended grid — adds low/negative z (real table) and x=0.60; keeps the old
# y/pitch/yaw discretisation so the envelope's per-pitch ok-rate stays comparable.
XS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
YS = [-0.20, -0.10, 0.0, 0.10, 0.20]
ZS = [-0.06, -0.02, 0.02, 0.06, 0.10, 0.14, 0.18, 0.22, 0.26]
PITCHES = [0.0, 0.225, 0.45, 0.675, 0.9, 1.2, 1.57]
YAWS = [-0.6, -0.3, 0.0, 0.3, 0.6]


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/out/ik_envelope.csv"
    from ovs_agent.apps.voice_rebot_arm.tools.grasp_selfcheck import _actuator_cfg
    from ovs_agent.apps.voice_rebot_arm.rebot_actuator import _make_rebot_arm

    print("connecting arm (torque on so the controller is live; NO motion is "
          "commanded — check_ik only solves IK)", flush=True)
    act = _make_rebot_arm(_actuator_cfg())
    act.connect()
    rows = []
    try:
        t0 = time.monotonic()
        n = 0
        total = len(XS) * len(YS) * len(ZS) * len(PITCHES) * len(YAWS)
        for x in XS:
            for y in YS:
                for z in ZS:
                    for p in PITCHES:
                        for yaw in YAWS:
                            try:
                                ok, err = act.robot.check_ik(x, y, z, 0.0, p, yaw)
                            except Exception:
                                ok, err = False, 9.999
                            rows.append((x, y, z, p, yaw, int(bool(ok)), float(err)))
                            n += 1
                    if n % 700 == 0:
                        print(f"  {n}/{total} ({time.monotonic()-t0:.0f}s)", flush=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "z", "pitch", "yaw", "ok", "err"])
            w.writerows(rows)
        ok_n = sum(r[5] for r in rows)
        print(f"IK SCAN DONE: {len(rows)} poses, {ok_n} reachable "
              f"({100*ok_n/len(rows):.0f}%), {time.monotonic()-t0:.0f}s -> {out}",
              flush=True)
        # quick low-z sanity: how many reachable at z<0.08 (the old envelope's
        # blind spot that made real low-table grasps read unreachable)?
        low = [r for r in rows if r[2] < 0.08]
        low_ok = sum(r[5] for r in low)
        print(f"  z<0.08 band: {low_ok}/{len(low)} reachable "
              f"(old envelope had NONE — z_min was 0.08)", flush=True)
    finally:
        act.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
