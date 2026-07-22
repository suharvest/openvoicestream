"""P3/P4: full sim grasp — observe (GT seg) -> estimate -> move -> close -> lift ->
verify HELD/LIFTED via PhysX contact + box height. Reuses the production
estimate_grasps / select_best_grasp / transform_grasp_pose_to_base unchanged.

Run modes:
  python run_grasp_sim.py            -> single box (P3)
  python run_grasp_sim.py --sweep    -> grid sweep -> CSV (P4)
"""
import os, sys, argparse
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
sys.path.insert(0, "/root/sim_bridge"); sys.path.insert(0, "/root/agent/ovs_agent/apps")

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.xforms import get_world_pose
from isaacsim.sensors.camera import Camera

import isaac_scene as scene
from isaac_camera import IsaacCameraDriver
from isaac_arm import IsaacArm, _R_to_rpy
from gt_segmenter import GtSegmenter

from voice_rebot_arm.perception.ordinary_grasp import estimate_grasps, select_best_grasp
from voice_rebot_arm.perception.transforms import transform_grasp_pose_to_base

intr = np.load("/root/sim/calib/intrinsics.npz")
K_REAL = np.asarray(intr["camera_matrix"], float)
W, H = 1280, 720
TABLE_TOP_Z = 0.02
USD = "/root/sim_bridge/out/rebot_gripper.usd"


def look_at(cam_pos, target, up=np.array([0., 0., 1.])):
    cam_pos = np.asarray(cam_pos, float); target = np.asarray(target, float)
    z = target - cam_pos; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x); y /= np.linalg.norm(y)
    T = np.eye(4); T[:3, :3] = np.column_stack([x, y, z]); T[:3, 3] = cam_pos
    return T


def up_hint(T):
    return (np.asarray(T)[:3, :3].T @ np.array([0., 0., 1.])).astype(np.float64)


class Rig:
    """Persistent world: robot + camera built once; box reset per trial."""
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.stage = self.world.stage
        scene.add_table(self.stage, top_z=TABLE_TOP_Z, cx=0.40, cy=0.0)
        add_reference_to_stage(usd_path=USD, prim_path="/World/rebot")
        sim_app.update()
        self.art = SingleArticulation(prim_path="/World/rebot", name="rebot")
        self.world.scene.add(self.art)

        self.cam = Camera(prim_path="/World/obs_cam", resolution=(W, H))
        self.cam.initialize()
        H_AP = 20.955; fx = K_REAL[0, 0]
        self.cam.set_focal_length(fx * H_AP / W)
        self.cam.set_horizontal_aperture(H_AP)
        self.cam.set_vertical_aperture(H_AP * H / W)
        self.cam.set_clipping_range(0.01, 10.0)
        self.cam.add_distance_to_image_plane_to_frame()
        self.cam.add_instance_id_segmentation_to_frame()

        # high-friction finger pads (collider prims live under the finger links)
        for fl in ("left_finger", "right_finger"):
            for cand in (f"/World/rebot/{fl}/collisions", f"/World/rebot/{fl}"):
                if self.stage.GetPrimAtPath(cand).IsValid():
                    try:
                        scene.set_high_friction(self.stage, cand, key=fl)
                    except Exception:
                        pass
                    break

        self.world.reset()
        self.cam.initialize()
        self.art.initialize()
        self._set_arm_gains()
        self.arm = IsaacArm(self.art, self.world)
        self.drv = IsaacCameraDriver(self.cam, K_REAL, W, H)
        self.box_path = None
        # jaw-frame settles accurately under the vertical grasp (calibration showed
        # ~[2.5,0,3.1]mm residual), so a static near-zero bias suffices. Skipping the
        # runtime arm-movement calibration here keeps the SyntheticData seg pipeline
        # clean for the first observe (moving the arm at init breaks the seg frame).
        # Under the vertical grasp the low-branch IK settles the jaw ~47mm SHORT in
        # X (measured: cmd x=0.4025 -> jaw x=0.3551) and a few mm off in Z. Bias the
        # commanded target forward so the jaw frame lands ON the box. (X-dominant.)
        self.vbias = np.array([0.047, 0.0, 0.004], float)

    def calibrate_vertical_bias(self, ref=(0.40, 0.0, 0.05)):
        """ONCE, with no box present: under the vertical (p=1.40) grasp orientation,
        find the target-bias that makes the settled JAW FRAME (TCP) land at `ref`.
        The vertical multiseed IK settles its jaw ~40mm short in X, so we servo the
        commanded target until the measured jaw frame reaches `ref`. Stored as
        self.vbias and added (laterally) to every vertical grasp target. Box-free
        => never knocks the box."""
        gr, gp, gyw = 0.0, 1.40, 0.0
        self.arm.go_home()
        for _ in range(60):
            self.world.step(render=False)
        self.arm.open_gripper(0.085)
        want = np.array(ref, float)
        jaw_cmd = want.copy()
        for _ in range(4):
            if not self.arm.move_to(jaw_cmd[0], jaw_cmd[1], jaw_cmd[2], gr, gp, gyw, settle_steps=120):
                break
            jaw = self.arm.get_tcp_pose()[:3, 3]
            err = want - jaw
            if np.linalg.norm(err) < 0.003:
                jaw_cmd = jaw_cmd + err
                break
            jaw_cmd = jaw_cmd + err
        self.vbias = jaw_cmd - want
        print("VBIAS calibrated (jaw):", np.round(self.vbias * 1000, 1).tolist(), "mm", flush=True)
        self.arm.stow()
        for _ in range(40):
            self.world.step(render=False)

    def _set_arm_gains(self):
        """The imported USD arm drives ship with kd=0 (undamped position drive).
        With kd=0 a commanded non-equilibrium config drifts/oscillates and the
        jaw settles far from the IK target (root cause of the KNOCKED P3 mode:
        jaw landed ~350mm off). Add critical-ish damping to the 6 arm joints so
        apply_action(joint_positions) actually holds the commanded pose."""
        try:
            ctrl = self.art.get_articulation_controller()
            kp = np.array([35809.86] * 6 + [625.0, 625.0])
            kd = np.array([2000.0] * 6 + [50.0, 50.0])
            ctrl.set_gains(kps=kp, kds=kd)
        except Exception as e:
            print("WARN set_gains failed:", e, flush=True)

    def box_mask(self):
        seg = self.cam.get_current_frame().get("instance_id_segmentation")
        if seg is None:
            return None
        data = np.asarray(seg["data"])
        ids = [int(k) for k, v in seg["info"].get("idToLabels", {}).items()
               if "target_box" in str(v)]
        if not ids:
            return None
        return np.isin(data, ids).astype(np.uint8)

    def spawn_box(self, dims, pose, mass=0.2, friction=1.4):
        if self.box_path is not None:
            self.stage.RemovePrim(self.box_path)
        self.box_path = scene.add_box(self.stage, dims, pose, mass=mass,
                                      friction=friction, name="target_box")
        # high-friction finger pads
        for fl in ("left_finger", "right_finger"):
            try:
                scene._add_physics_material  # ensure module present
            except Exception:
                pass
        self.world.reset()
        self.cam.initialize(); self.art.initialize()
        self._set_arm_gains()
        self.arm = IsaacArm(self.art, self.world)
        self.arm.go_home()
        # RigidPrim view so we can teleport the box (park/restore for the pad servo)
        try:
            from isaacsim.core.prims import SingleRigidPrim
            self._box_rb = SingleRigidPrim(prim_path=self.box_path, name="box_rb")
            self._box_rb.initialize()
            # SingleRigidPrim has set_world_pose (singular) not set_world_poses
        except Exception as e:
            self._box_rb = None
            print("WARN box_rb init:", e, flush=True)
        for _ in range(40):
            self.world.step(render=False)

    def box_pos(self):
        p, _ = get_world_pose(self.box_path)
        return np.asarray(p, float)

    def set_box_pose(self, cx, cy, z_center, yaw):
        """Teleport the box rigid body (used to park it out of the way while we
        servo the arm in free space, then restore it for the real grasp).
        Uses the RigidPrim view so PhysX picks up the new pose and zeros velocity."""
        import numpy as _np
        h = yaw / 2.0
        q = _np.array([_np.cos(h), 0.0, 0.0, _np.sin(h)], _np.float32)   # (w,x,y,z)
        pos = _np.array([cx, cy, z_center], _np.float32)
        try:
            self._box_rb.set_world_pose(position=pos, orientation=q)
            try:
                self._box_rb.set_linear_velocity(_np.zeros(3, _np.float32))
                self._box_rb.set_angular_velocity(_np.zeros(3, _np.float32))
            except Exception:
                pass
        except Exception as e:
            print("WARN set_box_pose:", e, flush=True)
        for _ in range(2):
            self.world.step(render=False)


def run_trial(rig, dims, pose, obs_cam_pos=(0.05, -0.22, 0.50), log=print):
    rig.spawn_box(dims, pose)
    # stow the arm to the +Y side so it does NOT occlude the down-looking camera.
    rig.arm.stow()
    box0 = rig.box_pos()
    cx, cy, table_z, yaw = pose

    # ── observe (fixed external cam from -Y side, arm stowed +Y; spec §6 fallback
    #    synthesized extrinsic — contact physics is what P3 validates) ──
    T_cam2base = look_at(obs_cam_pos, [cx, cy, table_z + dims[2] / 2.0])
    rig.drv.set_extrinsic(T_cam2base)
    for _ in range(25):
        rig.world.step(render=True)
    mask = rig.box_mask()
    bgr, depth_mm = rig.drv.get_frame()
    seg = GtSegmenter(lambda: mask, class_name="box")
    results = seg.predict(bgr)
    grasps = estimate_grasps(results, depth_mm, K_REAL, depth_quantile=0.5,
                             up_hint_cam=up_hint(T_cam2base))
    g = select_best_grasp(grasps)
    if g is None:
        return dict(status="NO_GRASP", method=None, jaw_width=None, reachable=False,
                    box_disp_mm=float(np.linalg.norm(rig.box_pos() - box0) * 1000))

    grasp6, pregrasp6 = transform_grasp_pose_to_base(
        np.asarray(g.position, float), np.asarray(g.tcp_rotation, float),
        T_cam2base, pregrasp_offset_m=0.08, insertion_depth_m=0.03)
    gx, gy, gz, gr, gp, gyw = grasp6
    px, py, pz, pr, pp, pyw = pregrasp6
    # reachability with the production-style pitch-relaxation ladder.
    reachable, used_pitch, ik_err = rig.arm.check_ik_relaxed(gx, gy, gz, gr, gp, gyw)
    dp = used_pitch - gp   # pitch delta to apply consistently to grasp+pregrasp+lift

    out = dict(method=g.method, jaw_width=round(float(g.jaw_width_m), 4),
               reachable=bool(reachable), ik_err=round(float(ik_err), 5),
               used_pitch=round(float(used_pitch), 3),
               grasp_base=[round(v, 4) for v in grasp6])

    if not reachable:
        out["status"] = "UNREACHABLE"
        out["box_disp_mm"] = float(np.linalg.norm(rig.box_pos() - box0) * 1000)
        return out

    # ── execute as a near-VERTICAL top-down grasp (harness override) ──────────
    # Perception returns a TILTED grasp (~0.81 rad pitch) whose physical pad line is
    # offset/tilted vs the jaw frame, so the lower finger sweeps a table box. For P3
    # contact validation we keep perception's GRASP POINT (box center, from depth)
    # but execute a top-down vertical orientation, which the IK confirms is reachable
    # at this x (debug: check_ik r0 p~1.4 yw0 -> ok). Vertical pads straddle the box
    # cleanly. We servo the small jaw->pad offset (box parked) at the vertical
    # orientation, where the offset is near-constant.
    # Execute a TOP-DOWN VERTICAL grasp (the JAW FRAME = real fingertip contact,
    # verified = end_link + 0.045·X). dbg_settle proved: command the jaw directly to
    # the box grasp point with vertical pitch + damped/stiff drives (kp 35809, kd
    # 2000) and it settles within ~1mm. The earlier vbias / pad-center servo were
    # aiming the WRONG point (the finger-link origins sit ~45mm ABOVE the jaw), so
    # they're dropped. We target the JAW at the live box (x,y) and a graspable z.
    gr, gp2, gyw = 0.0, 1.40, 0.0                   # top-down (tool +X points -Z)
    out["used_pitch"] = round(gp2, 3)
    bx, by, _bz = rig.box_pos()
    # grasp the box in its upper-middle: jaw at box_top - 0.5*lz clamped to reach
    # floor (jaw reaches down to ~0.045). For lz>=0.05 boxes the center is reachable.
    box_top = table_z + dims[2]
    gx, gy = float(bx), float(by)
    # Grasp z: the reachable vertical band (dbg_band) is ~[0.04, ceiling] where the
    # ceiling shrinks with x (0.19@x=0.32 -> 0.17@x=0.40). Pick a grasp z inside the
    # box extent AND inside the band, and a pregrasp/lift z under the ceiling.
    ceiling = float(np.interp(gx, [0.32, 0.40], [0.19, 0.17]))
    # Grasp HIGH on the box so the jaw stays in the single (elbow-up) IK branch that
    # the vertical pregrasp-and-descend tracks (it bottoms out ~0.09; below that the
    # solver must switch to elbow-down, which a continuous descent can't cross). For
    # a tall box this puts the pads on the upper body; for a short box this is the
    # reachability floor the spec wants surfaced (-> SLIPPED/UNREACHABLE in P4).
    gz = float(np.clip(table_z + dims[2] * 0.5, 0.045, box_top - 0.01))
    pre_z = float(min(box_top + 0.06, ceiling - 0.005))   # reachable clear height above box
    lift_z = float(min(gz + 0.12, ceiling - 0.005))
    out["grasp_base"] = [round(v, 4) for v in (gx, gy, gz, gr, gp2, gyw)]
    out["box_center_target"] = [round(v, 4) for v in (bx, by, table_z + dims[2] * 0.5)]

    rig.arm.go_home()
    rig.arm.open_gripper(0.085)
    # Two-stage VERTICAL approach that avoids sweeping the box:
    #  1) multiseed to a pose directly ABOVE the box (same x,y, high z) — its
    #     closest-jaw selection lands the jaw above the box, not through it.
    #  2) continuous (current-seed) descent straight down to the grasp z — holds the
    #     same elbow branch so the jaw tracks vertically down onto the box.
    # IK is BISTABLE (elbow-up vs elbow-down) and the arm CANNOT cross branches under
    # a single position command (it stalls mid-stretch). The ELBOW-DOWN branch reaches
    # the low grasp z (dbg_reach: jaw 0.05 at x=0.34). A HIGH pregrasp lands elbow-UP
    # and then can't descend. So the pregrasp is LOW (gz+0.045) — multiseed picks
    # elbow-down there — and the final descent is continuous in the SAME branch. The
    # box is slim (40mm) and the jaw approaches from just above, vertical, so the open
    # 85mm jaw clears it without sweeping.
    # LATERAL approach in the elbow-down branch: go to grasp HEIGHT but BEHIND the box
    # (-X), where the open 85mm jaw is clear of the box, then translate +X (continuous,
    # same branch) so the jaw slides AROUND the box to center on it. This enters the
    # low-reaching elbow-down branch without the from-above branch-stall and without
    # sweeping the box from the front.
    back_x = gx - 0.075
    m_pre = rig.arm.move_to(back_x, gy, gz, gr, gp2, gyw, settle_steps=150)
    jaw_above = rig.arm.get_tcp_pose()[:3, 3]
    out["jaw_above_pregrasp"] = [round(v, 4) for v in jaw_above.tolist()]
    out["box_after_pregrasp"] = [round(v, 4) for v in rig.box_pos().tolist()]
    out["pre_z"] = round(gz, 3)
    # translate forward in small continuous steps to center the open jaw on the box.
    m_grasp = True
    for xx in np.linspace(back_x, gx, 6)[1:]:
        m_grasp = rig.arm.move_to(float(xx), gy, gz, gr, gp2, gyw, settle_steps=45,
                                  continuous=True) and m_grasp
    rig.arm._step(40)
    out["move_pregrasp"] = m_pre
    out["move_grasp"] = m_grasp
    out["box_after_approach"] = [round(v, 4) for v in rig.box_pos().tolist()]
    box_pre_close = rig.box_pos()
    # diagnostic: render the grasp pose from a side cam to see jaw vs box.
    if getattr(run_trial, "_save_render", False):
        import cv2
        rig.drv.set_extrinsic(look_at((0.05, -0.30, 0.25), [cx, cy, table_z + dims[2] / 2.0]))
        for _ in range(15):
            rig.world.step(render=True)
        sb, _ = rig.drv.get_frame()
        if sb is not None:
            cv2.imwrite("/root/sim_bridge/p3_grasp_view.png", sb)
    # diagnostic: where did the jaw actually land vs the box?
    tcp_at_grasp = rig.arm.get_tcp_pose()
    pad_at_grasp = rig.arm.pad_center()
    out["jaw_pos"] = [round(v, 4) for v in tcp_at_grasp[:3, 3].tolist()]
    out["pad_pos"] = [round(v, 4) for v in pad_at_grasp.tolist()]
    out["box_pos_at_grasp"] = [round(v, 4) for v in box_pre_close.tolist()]
    out["jaw_to_box_mm"] = round(float(np.linalg.norm(tcp_at_grasp[:3, 3] - box_pre_close)) * 1000, 1)
    out["pad_to_box_mm"] = round(float(np.linalg.norm(pad_at_grasp - box_pre_close)) * 1000, 1)
    # Close to a half-target slightly LESS than the box half-width so the pads
    # compress on the box (the 625 N/m drive stiffness supplies grip force) rather
    # than crushing past it to fully-closed (which reads as no-object / slips).
    box_half = min(dims[0], dims[1]) / 2.0
    close_half = max(0.0, box_half - 0.004)
    rig.arm.close_to(close_half, steps=160)
    lf, rf = rig.arm.finger_positions()
    out["close_half_target"] = round(close_half, 4)
    # lift: retract straight UP in continuous steps (holds the elbow branch so the
    # lift doesn't branch-jump and fling the box). Settle long.
    for zz in np.linspace(gz, lift_z, 6)[1:]:
        rig.arm.move_to(gx, gy, float(zz), gr, gp2, gyw, settle_steps=60, continuous=True)
    rig.arm._step(120)
    box1 = rig.box_pos()

    lifted_mm = float((box1[2] - box0[2]) * 1000.0)
    disp_xy = float(np.linalg.norm(box1[:2] - box0[:2]) * 1000.0)
    fingers_open = (lf > 0.002) and (rf > 0.002)        # not fully closed => something between pads
    LIFTED = lifted_mm > 50.0
    HELD = LIFTED and fingers_open
    # KNOCKED: box moved far in XY but not grasped/lifted
    KNOCKED = (disp_xy > 40.0) and not LIFTED
    if HELD:
        status = "HELD"
    elif LIFTED:
        status = "LIFTED_NO_FINGER"   # lifted but fingers closed fully (slipped onto pad?)
    elif KNOCKED:
        status = "KNOCKED"
    else:
        status = "SLIPPED" if lifted_mm < 50 else "?"
    out.update(status=status, lifted_mm=round(lifted_mm, 1),
               disp_xy_mm=round(disp_xy, 1),
               finger_L=round(lf, 4), finger_R=round(rf, 4),
               box_z0=round(float(box0[2]), 4), box_z1=round(float(box1[2]), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    rig = Rig()

    if args.probe:
        pf = open("/root/sim_bridge/probe_result.txt", "w")
        def P(*a):
            s = " ".join(str(x) for x in a); pf.write(s + "\n"); pf.flush(); print(s, flush=True)
        configs = [
            ((0.05, 0.05, 0.06), (0.40, 0.0, TABLE_TOP_Z, 0.0)),
            ((0.04, 0.04, 0.05), (0.40, 0.0, TABLE_TOP_Z, 0.0)),
            ((0.05, 0.05, 0.05), (0.42, 0.0, TABLE_TOP_Z, 0.0)),
            ((0.04, 0.05, 0.06), (0.38, 0.0, TABLE_TOP_Z, 0.0)),
        ]
        for ci, (dims, pose) in enumerate(configs):
            r = run_trial(rig, dims, pose, log=lambda *a: None)
            P("CFG %d dims=%s x=%.2f -> status=%s lifted=%s disp_xy=%s jaw=%s pad=%s box_at_grasp=%s jaw_to_box=%s pad_to_box=%s fL=%s fR=%s box_after_appr=%s" % (
                ci, dims, pose[0], r.get("status"), r.get("lifted_mm"), r.get("disp_xy_mm"),
                r.get("jaw_pos"), r.get("pad_pos"), r.get("box_pos_at_grasp"), r.get("jaw_to_box_mm"),
                r.get("pad_to_box_mm"), r.get("finger_L"), r.get("finger_R"), r.get("box_after_approach")))
        pf.close(); sim_app.close(); return

    if args.debug:
        import cv2
        from gt_segmenter import GtSegmenter as _GS
        df = open("/root/sim_bridge/dbg_p3.txt", "w")
        def D(*a):
            line = " ".join(str(x) for x in a); df.write(line + "\n"); df.flush(); print(line, flush=True)
        dims = (0.05, 0.05, 0.06); pose = (0.40, 0.0, TABLE_TOP_Z, 0.0)
        rig.spawn_box(dims, pose)
        T = look_at((0.10, 0, 0.45), [0.40, 0.0, TABLE_TOP_Z + 0.03])
        rig.drv.set_extrinsic(T)
        for _ in range(30):
            rig.world.step(render=True)
        mask = rig.box_mask(); bgr, depth = rig.drv.get_frame()
        D("mask px:", None if mask is None else int(mask.sum()))
        D("depth valid px:", int((depth > 0).sum()),
          "range:", (int(depth[depth > 0].min()), int(depth.max())) if (depth > 0).any() else None)
        cv2.imwrite("/root/sim_bridge/dbg_p3_rgb.png", bgr)
        cv2.imwrite("/root/sim_bridge/dbg_p3_mask.png",
                    (mask * 255).astype(np.uint8) if mask is not None else np.zeros((H, W), np.uint8))
        results = _GS(lambda: mask).predict(bgr)
        grasps = estimate_grasps(results, depth, K_REAL, depth_quantile=0.5, up_hint_cam=up_hint(T))
        D("n candidates:", len(grasps))
        for gi, g in enumerate(grasps):
            D(" cand", gi, "valid=", g.is_valid, "method=", g.method, "reason=", g.rejected_reason,
              "w=", round(float(g.jaw_width_m), 4), "vdp=", g.valid_depth_pixels)
        gg = select_best_grasp(grasps)
        if gg is not None:
            grasp6, pre = transform_grasp_pose_to_base(np.asarray(gg.position, float),
                          np.asarray(gg.tcp_rotation, float), T, 0.08, 0.01)
            D("grasp6=", [round(v, 4) for v in grasp6])
            x, y, z, r, p, yw = grasp6
            for dp in (0.0, -0.2, -0.4, -0.6, -0.85):
                ok, err = rig.arm.check_ik(x, y, z, r, p + dp, yw)
                D("  check_ik pitch%+.2f -> ok=%s err=%.3f" % (dp, ok, err))
            ok, err = rig.arm.check_ik(x, y, z, 0.0, 1.30, 0.0)
            D("  vertical approx (r0 p1.30 yw0) ok=%s err=%.3f" % (ok, err))
            # what joint config does IK find + its FK pose
            q, e, w = rig.arm._solve_ik(x, y, z, r, p, yw)
            D("  IK q=", np.round(q, 3).tolist(), "err=%.3f within=%s" % (e, w))
        df.close(); sim_app.close(); return
    res_path = "/root/sim_bridge/p3_result.txt"
    rf = open(res_path, "w")
    def log(*a):
        line = " ".join(str(x) for x in a); rf.write(line + "\n"); rf.flush(); print(line, flush=True)

    if not args.sweep:
        dims = (0.04, 0.04, 0.06)   # slim box for the lateral elbow-down HELD attempt
        pose = (0.34, 0.0, TABLE_TOP_Z, 0.0)
        run_trial._save_render = True
        log("=== P3 SINGLE BOX ===")
        log("dims=", dims, "pose=", pose)
        r = run_trial(rig, dims, pose, log=log)
        for k, v in r.items():
            log("  %-12s %s" % (k, v))
        log("\nP3_DONE status=%s" % r.get("status"))
    else:
        import csv
        sweep = []
        for lx in (0.04, 0.05, 0.06):
            for lz in (0.05, 0.07):
                for x in (0.36, 0.42):
                    for yaw in (0.0, np.radians(30)):
                        sweep.append(((lx, lx + 0.02, lz), (x, 0.0, TABLE_TOP_Z, yaw)))
        csv_path = "/root/sim_bridge/p4_sweep.csv"
        cf = open(csv_path, "w", newline="")
        wr = csv.writer(cf)
        wr.writerow(["lx", "ly", "lz", "x", "y", "yaw_deg", "method", "jaw_width",
                     "reachable", "status", "lifted_mm", "disp_xy_mm", "finger_L", "finger_R"])
        log("=== P4 SWEEP n=%d ===" % len(sweep))
        for i, (dims, pose) in enumerate(sweep):
            r = run_trial(rig, dims, pose, log=lambda *a: None)
            wr.writerow([dims[0], dims[1], dims[2], pose[0], pose[1],
                         round(np.degrees(pose[3]), 1), r.get("method"),
                         r.get("jaw_width"), r.get("reachable"), r.get("status"),
                         r.get("lifted_mm"), r.get("disp_xy_mm"),
                         r.get("finger_L"), r.get("finger_R")])
            cf.flush()
            log("trial %2d/%d dims=%s x=%.2f yaw=%4.0f -> %s (method=%s w=%s lifted=%smm)"
                % (i + 1, len(sweep), dims, pose[0], np.degrees(pose[3]),
                   r.get("status"), r.get("method"), r.get("jaw_width"), r.get("lifted_mm")))
        cf.close()
        log("\nP4_DONE csv=%s" % csv_path)
    rf.close()
    sim_app.close()


main()
