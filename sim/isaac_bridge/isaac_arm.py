"""IsaacArm — duck-typed RebotArm backed by pinocchio IK + an Isaac articulation.

Kinematics: pinocchio on reBot-DevArm_fixend.urdf, frame end_link (the SAME model
the real arm uses), so check_ik / reachability are byte-aligned with Tier A.
Pose convention matches perception/transforms.py: 6D pose = (x,y,z,rx,ry,rz),
ZYX intrinsic euler  R = Rz(rz) @ Ry(ry) @ Rx(rx).
"""
import numpy as np
import pinocchio as pin

FIXEND_URDF = "/root/sim/rebot_b601dm_urdf/urdf/reBot-DevArm_fixend.urdf"
EE_FRAME = "end_link"
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGER_JOINTS = ["left_finger_joint", "right_finger_joint"]
FINGER_HALF_MAX = 0.0425           # per-finger stroke (m)
FINGER_LEAD_R = 0.0164             # CAD: torque(N·m)->linear force conversion radius


# ── pose helpers (match transforms.pose6d_to_mat4: R = Rz@Ry@Rx) ──
def _rpy_to_R(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _R_to_rpy(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1]); ry = np.arctan2(-R[2, 0], sy); rz = 0.0
    return np.array([rx, ry, rz])


class IsaacArm:
    def __init__(self, articulation, world, dt_settle_steps=80, tool_offset_x=-0.128):
        """articulation: SingleArticulation (already initialized & in a reset World).
        world: isaacsim World (for stepping).
        """
        from isaacsim.core.utils.types import ArticulationAction
        self._ArticulationAction = ArticulationAction
        self._art = articulation
        self._world = world
        self._settle = dt_settle_steps
        # REAL CAD gripper: approach axis = tool -X (URDF header line 33: "body &
        # fingers extend toward -X"). The jaw/TCP (finger contact) sits along end_link
        # local X by tool_offset_x; tool_offset_x is NEGATIVE (-0.128) so the jaw lands
        # at end_link + (-0.128)*X_local = the real -X finger contact (was +0.128 for
        # the OLD wrong +X box gripper). pad_center() reuses this signed value so it
        # auto-corrects with the sign flip.
        self.tool_offset_x = float(tool_offset_x)

        self.model = pin.buildModelFromUrdf(FIXEND_URDF)
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId(EE_FRAME)
        self.jl = self.model.lowerPositionLimit.copy()
        self.ju = self.model.upperPositionLimit.copy()

        names = list(self._art.dof_names)
        self.arm_dof_idx = [names.index(j) for j in ARM_JOINTS]
        self.finger_dof_idx = [names.index(j) for j in FINGER_JOINTS]
        # home = neutral arm config
        self._home_q = pin.neutral(self.model)

    # ── joint state bridge (Isaac articulation -> pinocchio q) ──
    def _arm_q(self):
        full = np.asarray(self._art.get_joint_positions(), dtype=np.float64)
        return full[self.arm_dof_idx].copy()

    def _step(self, n=None):
        for _ in range(self._settle if n is None else n):
            self._world.step(render=False)

    # ── FK ──
    def _fk(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_id].copy()

    def get_tcp_pose(self):
        """4x4 base<-JAW (TCP) at current sim joint state. The jaw frame is
        end_link translated by tool_offset_x (-0.128, i.e. toward tool -X = the real
        CAD approach axis) along its local X."""
        M = self._fk(self._arm_q())
        T = np.eye(4); T[:3, :3] = M.rotation
        T[:3, 3] = M.translation + M.rotation[:, 0] * self.tool_offset_x
        return T

    # ── IK (damped least squares to an SE3 target) ──
    def _solve_ik(self, x, y, z, rx, ry, rz, q0=None, iters=200, tol=1e-4):
        R = _rpy_to_R(rx, ry, rz)
        # target is the JAW pose; shift back along tool-X to the end_link target.
        p_jaw = np.array([x, y, z], dtype=np.float64)
        p_ee = p_jaw - R[:, 0] * self.tool_offset_x
        oMdes = pin.SE3(R, p_ee)
        q = (self._arm_q() if q0 is None else np.asarray(q0, float)).copy()
        damp = 1e-6
        err_norm = np.inf
        for _ in range(iters):
            M = self._fk(q)
            iMd = M.actInv(oMdes)
            err = pin.log(iMd).vector            # 6D twist error in EE frame
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
            q = pin.integrate(self.model, q, v * 1.0)
            q = np.clip(q, self.jl, self.ju)
        within = bool(np.all(q >= self.jl - 1e-6) and np.all(q <= self.ju + 1e-6))
        return q, err_norm, within

    def check_ik(self, x, y, z, roll, pitch, yaw, tol=5e-3):
        q, err, within, ok = self._solve_ik_multiseed(x, y, z, roll, pitch, yaw, tol=tol)
        return bool(ok), float(err)

    # ── motion ──
    def _solve_ik_multiseed(self, x, y, z, roll, pitch, yaw, tol=5e-3):
        """Solve IK from several seeds. Among the converged (err<tol, within-limits)
        solutions, return the one whose FK JAW is CLOSEST to the requested jaw
        target — this locks a consistent elbow branch and avoids the bistable
        vertical-IK branch-jump that left the jaw 30-50 mm off the box."""
        seeds = [None, self._home_q,
                 np.array([0.0, -1.0, -1.0, 0.0, 0.5, 0.0]),
                 np.array([0.3, -1.2, -0.8, 0.0, 0.6, 0.0]),
                 np.array([-0.3, -1.2, -0.8, 0.0, 0.6, 0.0]),
                 np.array([0.0, -1.2, -1.4, 0.0, 0.9, 0.0]),   # elbow-down (vertical)
                 np.array([0.0, -0.9, -1.6, 0.0, 1.1, 0.0])]
        jaw_tgt = np.array([x, y, z], dtype=np.float64)
        converged = []      # (jaw_dist, q, err, within)
        best_q, best_err, best_within = None, np.inf, False
        for s in seeds:
            q, err, within = self._solve_ik(x, y, z, roll, pitch, yaw, q0=s)
            if err < best_err:
                best_q, best_err, best_within = q, err, within
            if within and err < tol:
                M = self._fk(q)
                jaw_fk = M.translation + M.rotation[:, 0] * self.tool_offset_x
                converged.append((float(np.linalg.norm(jaw_fk - jaw_tgt)), q, err, within))
        if converged:
            converged.sort(key=lambda t: t[0])
            _, q, err, within = converged[0]
            return q, err, within, True
        return best_q, best_err, best_within, False

    def servo_jaw_to(self, x, y, z, roll, pitch, yaw, iters=6, step_steps=20,
                     tol_mm=5.0):
        """Closed-loop: nudge the jaw toward the target a few times, re-solving IK
        from the CURRENT config each iter (no branch jump). Safe under a vertical
        approach (box is not swept). Returns final jaw->target distance in mm."""
        for _ in range(iters):
            jaw = self.get_tcp_pose()[:3, 3]
            d = np.array([x, y, z]) - jaw
            dist_mm = float(np.linalg.norm(d)) * 1000.0
            if dist_mm < tol_mm:
                break
            # aim the IK at jaw + correction (over-correct slightly toward target)
            tgt = jaw + d
            q, err, within = self._solve_ik(tgt[0], tgt[1], tgt[2], roll, pitch, yaw,
                                            q0=self._arm_q())
            if not (within and err < 5e-3):
                break
            full = np.asarray(self._art.get_joint_positions(), dtype=np.float32).copy()
            for k, di in enumerate(self.arm_dof_idx):
                full[di] = q[k]
            self._art.apply_action(self._ArticulationAction(joint_positions=full))
            self._step(step_steps)
        return float(np.linalg.norm(np.array([x, y, z]) - self.get_tcp_pose()[:3, 3])) * 1000.0

    def servo_pad_to(self, x, y, z, roll, pitch, yaw, iters=10, step_steps=18,
                     tol_mm=6.0):
        """Closed-loop: drive the PAD CENTER (live finger-link midpoint) onto target.
        Each iter measures the pad->target error and asks IK to move the jaw by that
        delta (re-solved from current config, no branch jump). Returns final pad->
        target distance (mm)."""
        tgt = np.array([x, y, z], dtype=np.float64)
        for _ in range(iters):
            pad = self.pad_center()
            d = tgt - pad
            dist_mm = float(np.linalg.norm(d)) * 1000.0
            if dist_mm < tol_mm:
                break
            jaw = self.get_tcp_pose()[:3, 3]
            jaw_goal = jaw + d            # move jaw by the same world delta
            q, err, within = self._solve_ik(jaw_goal[0], jaw_goal[1], jaw_goal[2],
                                            roll, pitch, yaw, q0=self._arm_q())
            if not (within and err < 5e-3):
                break
            full = np.asarray(self._art.get_joint_positions(), dtype=np.float32).copy()
            for k, di in enumerate(self.arm_dof_idx):
                full[di] = q[k]
            self._art.apply_action(self._ArticulationAction(joint_positions=full))
            self._step(step_steps)
        return float(np.linalg.norm(tgt - self.pad_center())) * 1000.0

    def move_to(self, x, y, z, roll, pitch, yaw, duration=None, settle_steps=None,
                continuous=False):
        if continuous:
            # Only solve from the CURRENT joint config (no multiseed fallback), so a
            # fine descent cannot branch-jump to a different elbow solution.
            q, err, within = self._solve_ik(x, y, z, roll, pitch, yaw, q0=self._arm_q())
            ok = within and err < 5e-3
            if not ok:   # fall back to multiseed if the local solve fails
                q, err, within, ok = self._solve_ik_multiseed(x, y, z, roll, pitch, yaw)
        else:
            q, err, within, ok = self._solve_ik_multiseed(x, y, z, roll, pitch, yaw)
        if not ok:
            return False
        full = np.asarray(self._art.get_joint_positions(), dtype=np.float32).copy()
        for k, di in enumerate(self.arm_dof_idx):
            full[di] = q[k]
        self._art.apply_action(self._ArticulationAction(joint_positions=full))
        self._step(settle_steps)
        return True

    def stow(self, q=None):
        """Move arm to a stow config that keeps it out of a top-down camera's view
        of the workspace (rotate base to the side, fold up)."""
        if q is None:
            q = np.array([1.4, -0.5, -0.6, 0.0, 0.5, 0.0])  # joint1 swung +Y side
        q = np.clip(np.asarray(q, float), self.jl, self.ju)
        full = np.asarray(self._art.get_joint_positions(), dtype=np.float32).copy()
        for k, di in enumerate(self.arm_dof_idx):
            full[di] = q[k]
        self._art.apply_action(self._ArticulationAction(joint_positions=full))
        self._step(120)
        return True

    def check_ik_relaxed(self, x, y, z, roll, pitch, yaw,
                         pitch_relax=(0.0, -0.2, -0.4, -0.6, 0.2), tol=1e-3):
        """Reachability with a small pitch-relaxation ladder (mirrors
        grasp_service._relax_orientation). Returns (ok, used_pitch, err)."""
        for dp in pitch_relax:
            ok, err = self.check_ik(x, y, z, roll, pitch + dp, yaw, tol=tol)
            if ok:
                return True, pitch + dp, err
        return False, pitch, err

    def go_home(self):
        full = np.asarray(self._art.get_joint_positions(), dtype=np.float32).copy()
        for k, di in enumerate(self.arm_dof_idx):
            full[di] = self._home_q[k]
        self._art.apply_action(self._ArticulationAction(joint_positions=full))
        self._step()
        return True

    def wait_motion(self, duration):
        # duration seconds -> steps at 60 Hz physics
        self._step(max(1, int(duration * 60)))

    # ── gripper ──
    def open_gripper(self, distance_m=0.085):
        half = float(np.clip(distance_m / 2.0, 0.0, FINGER_HALF_MAX))
        full = np.asarray(self._art.get_joint_positions(), dtype=np.float32).copy()
        for di in self.finger_dof_idx:
            full[di] = half
        self._art.apply_action(self._ArticulationAction(joint_positions=full))
        self._step(60)
        return True

    def close_to(self, half_target=0.0, steps=80):
        full = np.asarray(self._art.get_joint_positions(), dtype=np.float32).copy()
        for di in self.finger_dof_idx:
            full[di] = float(half_target)
        self._art.apply_action(self._ArticulationAction(joint_positions=full))
        self._step(steps)
        return True

    def finger_positions(self):
        full = np.asarray(self._art.get_joint_positions(), dtype=np.float64)
        return float(full[self.finger_dof_idx[0]]), float(full[self.finger_dof_idx[1]])

    def pad_center(self):
        """World midpoint between the two finger-link origins at the CURRENT sim
        state. The IK jaw-frame (end_link + tool_offset_x) does NOT coincide with
        this physical pad midpoint under a tilted grasp, so the harness reads this
        to bias the grasp target onto the box (closes the ~25mm tool-frame error
        that was bumping the box instead of straddling it)."""
        import omni.usd
        from pxr import UsdGeom, Usd
        stage = omni.usd.get_context().get_stage()
        # The finger LINK origin sits at the gripper base (joint = end_link origin in
        # X), but the physical collision pad center is tool_offset_x (-0.128 m) along
        # the finger link local X = toward tool -X (real CAD finger-contact ~128 mm
        # toward -X from the flange). Reading the bare link origin under-reports the
        # pad by the full finger length, so add the signed offset along each finger's
        # world X axis (negative => -X, matching get_tcp_pose).
        PAD_OFF = float(self.tool_offset_x)  # -0.128 (signed; collision-box origin in link frame)
        pts = []
        for link in ("left_finger", "right_finger"):
            prim = stage.GetPrimAtPath(f"/World/rebot/{link}")
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = m.ExtractTranslation()
            xaxis = np.array([m[0][0], m[0][1], m[0][2]], float)  # world dir of link local +X
            xaxis = xaxis / (np.linalg.norm(xaxis) + 1e-12)
            pad = np.array([t[0], t[1], t[2]], float) + PAD_OFF * xaxis
            pts.append(pad)
        return (pts[0] + pts[1]) / 2.0

    def settle_to_q(self, q, steps=180):
        """Command the 6 arm joints to q and step until settled."""
        full = np.asarray(self._art.get_joint_positions(), dtype=np.float32).copy()
        for k, di in enumerate(self.arm_dof_idx):
            full[di] = q[k]
        self._art.apply_action(self._ArticulationAction(joint_positions=full))
        self._step(steps)
        return True
