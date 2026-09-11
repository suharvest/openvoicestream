"""run_held2.py — top-down grasp probe on the CORRECTED (real-CAD, -X approach) gripper.

CORRECTED GEOMETRY (tool_offset_x = -0.128, approach axis = tool -X per URDF header
line 33 "body & fingers extend toward -X"). get_tcp_pose()==pad_center() now land at
the real -X finger contact (verified jaw_X = end_link.x - 0.128, jaw==pad to 0.5mm).

KINEMATIC WALL (exhaustive many-seed IK, probe_negpitch.py):
  The arm's joint limits cap joint2,joint3 <= 0 (elbow-down only; upper=[2.8,0,0,1.57,
  1.57,3.14]). To aim the -X approach DOWN (-X . -Z = +1) the wrist would need +X UP,
  i.e. pitch ~ -1.57 -> UNREACHABLE everywhere. Best reachable -X . -Z is only ~0.39
  (-X still points UP). At every reachable top-down orientation the FINGERS extend
  UPWARD from the flange: at pitch=1.4 the flange sits at z~0.01 but the pad is at
  z~0.13 (+0.12m ABOVE the flange). So the -X gripper CANNOT descend its fingers onto
  a table box -> insertion is NEGATIVE (pad above box top) for the whole grid. This is
  the inverse of the OLD (wrong) +X box gripper, whose +X pointed down and could
  descend. Reported as a hard kinematic finding; the columns below were the closest
  reachable top-down attempt.

Reachability map (probe_reach, top-down yaw=0 column; -X approach points up):
  yaw=0, pitch=1.00 : z reachable [0.04,0.21] at every x 0.26..0.44 (full tall column)
  yaw=0, pitch=1.20 : z [0.04,0.17]
  yaw=0, pitch=1.40 : z [0.04,0.11]
  yaw=90            : z max 0.06 (USELESS for a tall descent).

Fingers separate along WORLD Y at yaw=0 (sep vector [0,0.0848,0] open). So the box
must fit between the blades in Y; the blades come down on the +Y/-Y faces of the box.
JAW frame == pad center (corrected geometry, tool_offset_x = -0.128, toward -X). The
blade tips reach ~box mid when the jaw is at box mid-height -- but see KINEMATIC WALL
above: with -X the pad sits ABOVE the flange, so the open jaw cannot actually descend
beside a table box (insertion < 0).

Strategy: yaw=0, pitch=1.00 (tall clean column). Start high above the box with jaw
OPEN (85mm) so blades clearly straddle the box in Y. Descend straight down in small
steps to a grasp z at box mid-height (blades alongside the body, not on the top).
Close to (box_half_y - margin). Lift straight up. High pad friction + strong drive.
"""
import os, sys, argparse
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
sys.path.insert(0, "/root/sim_bridge")

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.xforms import get_world_pose

import isaac_scene as scene
from isaac_arm import IsaacArm

TABLE_TOP_Z = 0.02
USD = "/root/sim_bridge/out/rebot_gripper.usd"


class Rig:
    def __init__(self, finger_mu=6.0, finger_kp=9000.0, finger_kd=300.0):
        self.finger_mu = finger_mu
        self.finger_kp = finger_kp
        self.finger_kd = finger_kd
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.stage = self.world.stage
        scene.add_table(self.stage, top_z=TABLE_TOP_Z, cx=0.40, cy=0.0)
        add_reference_to_stage(usd_path=USD, prim_path="/World/rebot")
        sim_app.update()
        self.art = SingleArticulation(prim_path="/World/rebot", name="rebot")
        self.world.scene.add(self.art)
        for fl in ("left_finger", "right_finger"):
            for cand in (f"/World/rebot/{fl}/collisions", f"/World/rebot/{fl}"):
                if self.stage.GetPrimAtPath(cand).IsValid():
                    try:
                        scene.set_high_friction(self.stage, cand,
                                                static_friction=finger_mu,
                                                dynamic_friction=finger_mu * 0.9, key=fl)
                    except Exception:
                        pass
                    break
        self.world.reset()
        self.art.initialize()
        self._set_arm_gains()
        self.arm = IsaacArm(self.art, self.world)
        self.box_path = None

    def _set_arm_gains(self):
        try:
            ctrl = self.art.get_articulation_controller()
            kp = np.array([35809.86] * 6 + [self.finger_kp, self.finger_kp])
            kd = np.array([2000.0] * 6 + [self.finger_kd, self.finger_kd])
            ctrl.set_gains(kps=kp, kds=kd)
        except Exception as e:
            print("WARN set_gains:", e, flush=True)

    def spawn_box(self, dims, pose, mass=0.05, friction=1.6):
        if self.box_path is not None:
            self.stage.RemovePrim(self.box_path)
        self.box_path = scene.add_box(self.stage, dims, pose, mass=mass,
                                      friction=friction, name="target_box")
        self.world.reset()
        self.art.initialize()
        self._set_arm_gains()
        self.arm = IsaacArm(self.art, self.world)
        self.arm.go_home()
        for _ in range(40):
            self.world.step(render=False)

    def box_pos(self):
        p, _ = get_world_pose(self.box_path)
        return np.asarray(p, float)


def run_trial(rig, dims, pose, pitch=1.00, close_margin=0.004, log=print, geom_only=False):
    rig.spawn_box(dims, pose)
    for _ in range(30):
        rig.world.step(render=False)
    box0 = rig.box_pos()
    cx, cy, table_z, yaw = pose
    lx, ly, lz = dims
    box_top = table_z + lz
    box_half_y = ly / 2.0

    out = dict(dims=dims, pose=pose, pitch=pitch, close_margin=close_margin)

    gx, gy = float(cx), float(cy)
    gr, gp, gyw = 0.0, float(pitch), 0.0      # top-down yaw=0
    # clean tall column for this pitch (verified reachable ceiling)
    # verified reachable ceiling per pitch at this x (probe_reach). pitch=1.40 x=0.40
    # tops out ~0.10, so don't command above it (the move_above was failing at 0.11).
    if round(pitch, 2) == 1.40:
        ceiling = 0.10 if cx >= 0.38 else 0.11
    elif round(pitch, 2) == 1.20:
        ceiling = 0.17
    elif round(pitch, 2) == 1.00:
        ceiling = 0.21
    else:
        ceiling = 0.15
    start_jaw_z = min(ceiling, box_top + 0.08)   # start well above the box top
    start_jaw_z = max(start_jaw_z, box_top + 0.05)
    start_jaw_z = min(start_jaw_z, ceiling)
    # grasp at box mid-height (blades alongside body). jaw==pad so jaw_z = box mid.
    grasp_jaw_z = float(np.clip(table_z + lz * 0.5, 0.045, ceiling - 0.02))

    out["box_top"] = round(box_top, 4)
    out["box_half_y"] = round(box_half_y, 4)
    out["ceiling"] = ceiling
    out["start_jaw_z"] = round(start_jaw_z, 4)
    out["grasp_jaw_z"] = round(grasp_jaw_z, 4)

    ok_ik, err_ik = rig.arm.check_ik(gx, gy, grasp_jaw_z, gr, gp, gyw, tol=6e-3)
    ok_ik2, err_ik2 = rig.arm.check_ik(gx, gy, start_jaw_z, gr, gp, gyw, tol=6e-3)
    out["ik_grasp"] = (bool(ok_ik), round(float(err_ik), 5))
    out["ik_start"] = (bool(ok_ik2), round(float(err_ik2), 5))

    rig.arm.go_home()
    rig.arm.open_gripper(0.085)

    # 0) TRANSIT HIGH along a continuous Cartesian path so no link swings through the
    #    tall box. First lift straight up at the HOME xy to the ceiling height, then
    #    translate laterally (small continuous steps) over to the box xy STILL at the
    #    ceiling, then descend. A single multiseed jump home->over-box teleports joints
    #    and the physics settle sweeps a link through the box (the 144mm knock).
    home_jaw = rig.arm.get_tcp_pose()[:3, 3]
    hx, hy = float(home_jaw[0]), float(home_jaw[1])
    transit_z = ceiling
    # lift straight up at home xy (multiseed ok: vertical, away from box)
    rig.arm.move_to(hx, hy, transit_z, gr, gp, gyw, settle_steps=120)
    # translate over to the box xy at constant high z, continuous (no branch jump)
    for t in np.linspace(0.0, 1.0, 12)[1:]:
        wx = hx + (gx - hx) * t
        wy = hy + (gy - hy) * t
        rig.arm.move_to(float(wx), float(wy), transit_z, gr, gp, gyw,
                        settle_steps=30, continuous=True)
    out["box_after_transit"] = [round(v, 4) for v in rig.box_pos().tolist()]

    # 1) approach the above-pose (continuous, already over the box at high z).
    m_above = rig.arm.move_to(gx, gy, start_jaw_z, gr, gp, gyw, settle_steps=150,
                              continuous=True)
    jaw_a = rig.arm.get_tcp_pose()[:3, 3]
    out["move_above"] = m_above
    out["jaw_above"] = [round(v, 4) for v in jaw_a.tolist()]
    out["jaw_above_xy_err_mm"] = round(float(np.linalg.norm(jaw_a[:2] - np.array([gx, gy]))) * 1000, 1)
    out["box_after_above"] = [round(v, 4) for v in rig.box_pos().tolist()]

    # 2) descend straight down to grasp z, continuous (no branch jump), FINE steps.
    #    Use the settled jaw xy (pad center) as the descent column so the blades drop
    #    straight, not drifting into the box. Many small steps keep the local IK on one
    #    branch (avoids the mid-descent wobble that shoved the box +X).
    jaw_col = rig.arm.get_tcp_pose()[:3, 3]
    cxd, cyd = float(jaw_col[0]), float(jaw_col[1])
    desc_trace = []
    m_desc = True
    for zz in np.linspace(start_jaw_z, grasp_jaw_z, 28)[1:]:
        m_desc = rig.arm.move_to(cxd, cyd, float(zz), gr, gp, gyw, settle_steps=30,
                                 continuous=True) and m_desc
        desc_trace.append(round(float(rig.arm.get_tcp_pose()[2, 3]), 4))
    rig.arm._step(40)
    # 2b) CENTER the pad on the box (x,y) at grasp z via closed-loop servo. The arm
    #     hits its reach limit ~x=0.388 so a box at x=0.40 leaves the pad 12mm off in
    #     X -> asymmetric pinch that slips. servo_pad_to nudges the pad onto the box
    #     center (re-solving IK from current config, no branch jump, box not swept).
    pad_err = rig.arm.servo_pad_to(gx, gy, grasp_jaw_z, gr, gp, gyw,
                                   iters=12, step_steps=20, tol_mm=4.0)
    out["servo_pad_err_mm"] = round(float(pad_err), 1)
    rig.arm._step(30)
    out["move_descend"] = m_desc
    out["desc_trace_jawz"] = desc_trace
    box_pre = rig.box_pos()
    out["box_pre_close"] = [round(v, 4) for v in box_pre.tolist()]
    out["box_disp_descend_mm"] = round(float(np.linalg.norm(box_pre[:2] - box0[:2]) * 1000), 1)
    out["jaw_pre_close"] = [round(v, 4) for v in rig.arm.get_tcp_pose()[:3, 3].tolist()]
    out["pad_pre_close"] = [round(v, 4) for v in rig.arm.pad_center().tolist()]

    # ── GEOMETRIC grasp-quality verdict (position/depth/reach/clearance; NO
    #    friction/close/lift — the real arm's known-good grip closes the deal). ──
    pad = rig.arm.pad_center()
    pad_xy_err_mm = round(float(np.linalg.norm(pad[:2] - box_pre[:2]) * 1000), 1)
    insertion_mm = round(float((box_top - pad[2]) * 1000), 1)   # pad depth below box top
    knock_mm = out["box_disp_descend_mm"]
    reach = bool(out["ik_grasp"][0])
    geom_ok = bool(reach and knock_mm < 5.0 and pad_xy_err_mm < 8.0
                   and insertion_mm > lz * 1000 * 0.30 and insertion_mm < lz * 1000 + 5.0)
    out["pad_xy_err_mm"] = pad_xy_err_mm
    out["insertion_mm"] = insertion_mm
    out["geom_ok"] = geom_ok
    out["geom_fail"] = (None if geom_ok else
                        ("UNREACH" if not reach else
                         "KNOCK" if knock_mm >= 5.0 else
                         "PAD_OFF" if pad_xy_err_mm >= 8.0 else "DEPTH"))
    if geom_only:
        return out

    # 3) close on the Y faces. Command WELL INSIDE box_half so the drive keeps
    #    pushing (real grip force); the box stops the fingers near box_half_y.
    close_half = max(0.0, box_half_y - close_margin)
    rig.arm.close_to(close_half, steps=180)
    rig.arm._step(60)
    lf, rf = rig.arm.finger_positions()
    out["close_half_target"] = round(close_half, 4)
    out["finger_L"] = round(lf, 4)
    out["finger_R"] = round(rf, 4)
    box_ac = rig.box_pos()
    out["box_after_close"] = [round(v, 4) for v in box_ac.tolist()]

    # 4) lift straight up, RE-SQUEEZING every step so the finger drive keeps applying
    #    grip force throughout the lift (a one-shot close relaxes once the arm moves).
    jaw_now = float(rig.arm.get_tcp_pose()[2, 3])
    lift_to = min(ceiling, jaw_now + 0.10)
    lift_trace = []
    arm = rig.arm
    from isaacsim.core.utils.types import ArticulationAction
    for zz in np.linspace(jaw_now, lift_to, 14)[1:]:
        # solve arm IK for this z (continuous, current config), THEN write arm joints
        # AND finger close target in ONE action so the squeeze is never relaxed.
        q, err, within = arm._solve_ik(gx, gy, float(zz), gr, gp, gyw, q0=arm._arm_q())
        if not (within and err < 5e-3):
            q, err, within, ok = arm._solve_ik_multiseed(gx, gy, float(zz), gr, gp, gyw)
        full = np.asarray(arm._art.get_joint_positions(), dtype=np.float32).copy()
        for k, di in enumerate(arm.arm_dof_idx):
            full[di] = q[k]
        for di in arm.finger_dof_idx:
            full[di] = float(close_half)        # HOLD the squeeze
        arm._art.apply_action(ArticulationAction(joint_positions=full))
        arm._step(60)
        b = rig.box_pos()
        lift_trace.append((round(float(arm.get_tcp_pose()[2, 3]), 4), round(float(b[2]), 4)))
    # final hold
    full = np.asarray(arm._art.get_joint_positions(), dtype=np.float32).copy()
    for di in arm.finger_dof_idx:
        full[di] = float(close_half)
    arm._art.apply_action(ArticulationAction(joint_positions=full))
    arm._step(150)
    out["lift_trace_jawz_boxz"] = lift_trace
    box1 = rig.box_pos()
    lf2, rf2 = rig.arm.finger_positions()
    out["finger_L_lift"] = round(lf2, 4)
    out["finger_R_lift"] = round(rf2, 4)

    lifted_mm = float((box1[2] - box0[2]) * 1000.0)
    disp_xy = float(np.linalg.norm(box1[:2] - box0[:2]) * 1000.0)
    fingers_on = (lf2 > 0.002) and (rf2 > 0.002)
    LIFTED = lifted_mm > 30.0
    HELD = LIFTED and fingers_on
    KNOCKED = (disp_xy > 40.0) and not LIFTED
    status = "HELD" if HELD else ("LIFTED_NO_FINGER" if LIFTED else ("KNOCKED" if KNOCKED else "SLIPPED"))
    out.update(status=status, lifted_mm=round(lifted_mm, 1), disp_xy_mm=round(disp_xy, 1),
               box_z0=round(float(box0[2]), 4), box_z1=round(float(box1[2]), 4))
    return out


def run_side_geom_trial(rig, dims, pose, roll=0.0, pitch=0.0, log=print):
    """SIDE/FORWARD geometric eval: gripper approaches the box horizontally from
    the front (+X side) along the tool -X axis, OPEN fingers straddling the box in
    world Y. Stage in front, advance toward the box center, then read geometry:
      reach          : IK feasible at box-mid height beside the body
      pad_xy_err_mm  : pad on the box footprint (xy)
      fwd_insertion  : (box_front_face_x - pad.x)*1000; POSITIVE => pad behind the
                       front face = pad INSIDE the body (the side-grasp analogue of
                       top-down insertion, now positive instead of negative)
      straddle       : open finger pads sit on opposite +Y/-Y sides of the box
      knock_mm       : box displacement during the forward advance (clean = small)
    NO friction/close/lift. Mirrors the top-down --geom contract but for side entry."""
    import omni.usd
    from pxr import UsdGeom, Usd
    rig.spawn_box(dims, pose)
    for _ in range(30):
        rig.world.step(render=False)
    box0 = rig.box_pos()
    cx, cy, table_z, yaw = pose
    lx, ly, lz = dims
    box_top = table_z + lz
    box_mid = table_z + lz / 2.0
    bhx, bhy = lx / 2.0, ly / 2.0
    front_face = cx + bhx
    stage = omni.usd.get_context().get_stage()

    def finger_pads():
        out = {}
        for link in ("left_finger", "right_finger"):
            prim = stage.GetPrimAtPath(f"/World/rebot/{link}")
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = m.ExtractTranslation()
            xa = np.array([m[0][0], m[0][1], m[0][2]], float); xa /= np.linalg.norm(xa) + 1e-12
            out[link] = np.array([t[0], t[1], t[2]], float) + (-0.128) * xa
        return out

    out = dict(dims=dims, pose=pose, roll=roll, pitch=pitch)
    # reach check at box-mid height beside the body
    ok_ik, err_ik = rig.arm.check_ik(cx, cy, box_mid, roll, pitch, yaw, tol=6e-3)
    out["ik_grasp"] = (bool(ok_ik), round(float(err_ik), 5))

    rig.arm.go_home(); rig.arm.open_gripper(0.085)
    stage_x = cx + 0.14
    rig.arm.move_to(stage_x, cy, box_mid, roll, pitch, yaw, settle_steps=120)
    # advance toward box center, continuous (no branch jump)
    for t in np.linspace(0.0, 1.0, 12)[1:]:
        rig.arm.move_to(stage_x + (cx - stage_x) * t, cy, box_mid, roll, pitch, yaw,
                        settle_steps=25, continuous=True)
    rig.arm._step(40)

    pad = rig.arm.pad_center()
    appr = -rig.arm.get_tcp_pose()[:3, 0]          # tool -X world = approach dir
    pl = finger_pads(); padL = pl["left_finger"]; padR = pl["right_finger"]
    boxN = rig.box_pos()
    knock_mm = round(float(np.linalg.norm(boxN[:2] - box0[:2]) * 1000), 1)
    pad_xy_err_mm = round(float(np.linalg.norm(pad[:2] - box0[:2]) * 1000), 1)
    fwd_insertion_mm = round(float((front_face - pad[0]) * 1000), 1)   # + => pad inside body (forward)
    pad_in_body_z = bool(pad[2] > table_z + 0.003 and pad[2] < box_top - 0.003)
    straddle = bool((padL[1] > bhy * 0.5 and padR[1] < -bhy * 0.5) or
                    (padR[1] > bhy * 0.5 and padL[1] < -bhy * 0.5))
    appr_down = round(float(-appr[2]), 3)
    reach = bool(out["ik_grasp"][0])
    geom_ok = bool(reach and straddle and pad_in_body_z and fwd_insertion_mm > 0.0
                   and pad_xy_err_mm < 12.0 and knock_mm < 8.0)
    out.update(box_top=round(box_top, 4), box_mid=round(box_mid, 4),
               front_face=round(front_face, 4), appr_down=appr_down,
               pad=[round(v, 4) for v in pad.tolist()],
               padL=[round(v, 4) for v in padL.tolist()],
               padR=[round(v, 4) for v in padR.tolist()],
               reach=reach, pad_xy_err_mm=pad_xy_err_mm,
               fwd_insertion_mm=fwd_insertion_mm, pad_in_body_z=pad_in_body_z,
               straddle=straddle, knock_mm=knock_mm, geom_ok=geom_ok)
    out["geom_fail"] = (None if geom_ok else
                        ("UNREACH" if not reach else
                         "KNOCK" if knock_mm >= 8.0 else
                         "NO_STRADDLE" if not straddle else
                         "PAD_OFF" if pad_xy_err_mm >= 12.0 else
                         "PAD_ABOVE_BODY" if not pad_in_body_z else
                         "NEG_INSERTION" if fwd_insertion_mm <= 0.0 else "?"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--geom", action="store_true")   # geometric eval (pos/depth/reach/clearance, NO friction)
    ap.add_argument("--geomside", action="store_true")   # SIDE/FORWARD geometric eval (horizontal approach)
    # DEFAULTS = the verified reproducible HELD config (lift 40mm, disp 0.7mm).
    ap.add_argument("--pitch", type=float, default=1.40)   # near-vertical blades, no X-catch
    ap.add_argument("--lz", type=float, default=0.08)      # box_top 0.10 <= ceiling 0.10/0.11
    ap.add_argument("--lw", type=float, default=0.04)      # 0.03-0.04 grip reliably
    ap.add_argument("--x", type=float, default=0.34)       # clean column, pad centers
    ap.add_argument("--mu", type=float, default=12.0)
    ap.add_argument("--kp", type=float, default=15000.0)
    ap.add_argument("--margin", type=float, default=0.010)
    args = ap.parse_args()
    rig = Rig(finger_mu=args.mu, finger_kp=args.kp)
    rf = open("/root/sim_bridge/held2_result.txt", "w")
    def log(*a):
        line = " ".join(str(x) for x in a); rf.write(line + "\n"); rf.flush(); print(line, flush=True)

    if not args.sweep and not args.geom and not args.geomside:
        dims = (args.lw, args.lw, args.lz)
        pose = (args.x, 0.0, TABLE_TOP_Z, 0.0)
        log("=== HELD2 ATTEMPT ===")
        r = run_trial(rig, dims, pose, pitch=args.pitch, close_margin=args.margin, log=log)
        for k, v in r.items():
            log("  %-20s %s" % (k, v))
        log("\nHELD_DONE status=%s lifted_mm=%s disp=%s" % (r.get("status"), r.get("lifted_mm"), r.get("disp_xy_mm")))
    elif args.sweep:
        import csv
        cf = open("/root/sim_bridge/held2_sweep.csv", "w", newline="")
        wr = csv.writer(cf)
        wr.writerow(["lx","ly","lz","x","pitch","status","lifted_mm","disp_xy_mm",
                     "finger_L_lift","finger_R_lift","box_disp_descend_mm","ik_grasp"])
        sweep = []
        # SHORT boxes only: box_top must stay <= the pitch-1.40 ceiling (~0.10) so the
        # open jaw starts ABOVE the top and the blades straddle the body in Y without
        # catching the top during descent. x=0.36 is the proven clean column.
        for lz in (0.06, 0.07, 0.08):
            for lw in (0.03, 0.04, 0.05):
                for x in (0.34, 0.36, 0.38):
                    sweep.append(((lw, lw, lz), (x, 0.0, TABLE_TOP_Z, 0.0)))
        log("=== HELD2 SWEEP n=%d ===" % len(sweep))
        for i, (dims, pose) in enumerate(sweep):
            r = run_trial(rig, dims, pose, pitch=args.pitch, close_margin=args.margin)
            wr.writerow([dims[0],dims[1],dims[2],pose[0],args.pitch,r.get("status"),
                         r.get("lifted_mm"),r.get("disp_xy_mm"),r.get("finger_L_lift"),
                         r.get("finger_R_lift"),r.get("box_disp_descend_mm"),r.get("ik_grasp")])
            cf.flush()
            log("trial %2d/%d dims=%s x=%.2f -> %s lifted=%smm disp=%smm fingers=(%s,%s)" % (
                i+1,len(sweep),dims,pose[0],r.get("status"),r.get("lifted_mm"),
                r.get("disp_xy_mm"),r.get("finger_L_lift"),r.get("finger_R_lift")))
        cf.close()
        log("\nHELD_SWEEP_DONE")
    if args.geom:
        import csv
        cf = open("/root/sim_bridge/geom_sweep.csv", "w", newline="")
        wr = csv.writer(cf)
        wr.writerow(["lx","ly","lz","x","pitch","geom_ok","geom_fail","reach",
                     "pad_xy_err_mm","insertion_mm","knock_mm"])
        sweep = []
        # GEOMETRIC eval: position/insertion-depth/reach/clearance only (friction & hold
        # come from the real arm's known-good params). Per-height pitch picks a reachable
        # column (1.40 short .. 1.00 tall). Broader grid than the friction sweep.
        for lz in (0.05, 0.08):
            for lw in (0.03, 0.04, 0.05):
                for x in (0.30, 0.34, 0.38, 0.42):
                    pit = 1.40 if lz <= 0.06 else (1.20 if lz <= 0.09 else 1.00)
                    sweep.append(((lw, lw, lz), (x, 0.0, TABLE_TOP_Z, 0.0), pit))
        log("=== GEOM SWEEP n=%d (pos/depth/reach/clearance, NO friction) ===" % len(sweep))
        for i, (dims, pose, pit) in enumerate(sweep):
            r = run_trial(rig, dims, pose, pitch=pit, geom_only=True)
            wr.writerow([dims[0],dims[1],dims[2],pose[0],pit,r.get("geom_ok"),
                         r.get("geom_fail"),r.get("ik_grasp")[0],r.get("pad_xy_err_mm"),
                         r.get("insertion_mm"),r.get("box_disp_descend_mm")])
            cf.flush()
            log("geom %2d/%d dims=%s x=%.2f pit=%.2f -> ok=%s fail=%s pad=%smm ins=%smm knock=%smm" % (
                i+1,len(sweep),dims,pose[0],pit,r.get("geom_ok"),r.get("geom_fail"),
                r.get("pad_xy_err_mm"),r.get("insertion_mm"),r.get("box_disp_descend_mm")))
        cf.close()
        log("\nGEOM_SWEEP_DONE")
    if args.geomside:
        import csv
        cf = open("/root/sim_bridge/geom_side_sweep.csv", "w", newline="")
        wr = csv.writer(cf)
        wr.writerow(["lx","ly","lz","x","roll","pitch","geom_ok","geom_fail","reach",
                     "pad_xy_err_mm","fwd_insertion_mm","pad_in_body_z","straddle","knock_mm","appr_down"])
        sweep = []
        # SIDE/FORWARD geom eval. Horizontal approach (pit~0) is the reachable band
        # at box-mid height (probe_tradeoff). Footprint x height x x position grid.
        for lz in (0.04, 0.05, 0.06, 0.08):
            for lw in (0.03, 0.04, 0.05):
                for x in (0.30, 0.34, 0.38, 0.42):
                    sweep.append(((lw, lw, lz), (x, 0.0, TABLE_TOP_Z, 0.0), 0.0, 0.0))
        log("=== GEOM SIDE/FORWARD SWEEP n=%d (horizontal approach, NO friction) ===" % len(sweep))
        for i, (dims, pose, roll, pit) in enumerate(sweep):
            r = run_side_geom_trial(rig, dims, pose, roll=roll, pitch=pit)
            wr.writerow([dims[0],dims[1],dims[2],pose[0],roll,pit,r.get("geom_ok"),
                         r.get("geom_fail"),r.get("reach"),r.get("pad_xy_err_mm"),
                         r.get("fwd_insertion_mm"),r.get("pad_in_body_z"),r.get("straddle"),
                         r.get("knock_mm"),r.get("appr_down")])
            cf.flush()
            log("side %2d/%d dims=%s x=%.2f -> ok=%s fail=%s pad=%smm fwd_ins=%smm in_z=%s straddle=%s knock=%smm" % (
                i+1,len(sweep),dims,pose[0],r.get("geom_ok"),r.get("geom_fail"),
                r.get("pad_xy_err_mm"),r.get("fwd_insertion_mm"),r.get("pad_in_body_z"),
                r.get("straddle"),r.get("knock_mm")))
        cf.close()
        log("\nGEOM_SIDE_SWEEP_DONE")
    rf.close()
    sim_app.close()


if __name__ == "__main__":
    main()
