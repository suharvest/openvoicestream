"""P2: IsaacArm pinocchio IK — FK round-trip + a reachable move_to that moves the arm.
Also cross-checks pinocchio FK(end_link) against Isaac's end_link world pose."""
import os, sys
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})
import numpy as np
sys.path.insert(0, "/root/sim_bridge"); sys.path.insert(0, "/root/agent/ovs_agent/apps")
_RES = open("/root/sim_bridge/p2_result.txt", "w")
def R(*a):
    line = " ".join(str(x) for x in a); _RES.write(line + "\n"); _RES.flush(); print(line, flush=True)

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaac_arm import IsaacArm, _rpy_to_R, _R_to_rpy

USD = "/root/sim_bridge/out/rebot_gripper.usd"
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD, prim_path="/World/rebot")
sim_app.update()
art = SingleArticulation(prim_path="/World/rebot", name="rebot")
world.scene.add(art)
world.reset(); art.initialize()

arm = IsaacArm(art, world)
R("arm DOF idx:", arm.arm_dof_idx, "finger idx:", arm.finger_dof_idx)
R("joint lower:", np.round(arm.jl, 3).tolist())
R("joint upper:", np.round(arm.ju, 3).tolist())

# ── FK round-trip: pick a few reachable joint configs, FK -> pose6d -> IK -> FK ──
R("\n=== FK ROUND-TRIP (FK->pose->IK->FK) ===")
import pinocchio as pin
test_qs = [
    np.array([0.0, -1.0, -1.0, 0.0, 0.5, 0.0]),
    np.array([0.3, -0.8, -1.2, 0.2, -0.3, 0.4]),
    np.array([-0.4, -1.4, -0.6, -0.3, 0.6, -0.2]),
]
for qi, q in enumerate(test_qs):
    q = np.clip(q, arm.jl, arm.ju)
    M = arm._fk(q)
    T = np.eye(4); T[:3, :3] = M.rotation; T[:3, 3] = M.translation
    pos = T[:3, 3]; rpy = _R_to_rpy(T[:3, :3])
    # solve IK back from a different seed
    q_sol, err, within = arm._solve_ik(pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2],
                                       q0=np.zeros(6))
    M2 = arm._fk(q_sol)
    pos_err = float(np.linalg.norm(M2.translation - pos))
    rot_err = float(np.linalg.norm(pin.log3(M2.rotation.T @ M.rotation)))
    R("case", qi, "target_pos=", np.round(pos, 4).tolist(),
      "rpy=", np.round(rpy, 4).tolist())
    R("   IK err6=%.2e  refit pos_err=%.2emm rot_err=%.2erad  within_limits=%s"
      % (err, pos_err * 1000, rot_err, within))

# ── pinocchio FK vs Isaac end_link world pose (validate the two agree) ──
R("\n=== pinocchio FK vs Isaac end_link ===")
from isaacsim.core.utils.xforms import get_world_pose
q_test = np.clip(np.array([0.2, -1.0, -1.0, 0.1, 0.3, 0.0]), arm.jl, arm.ju)
full = np.asarray(art.get_joint_positions(), dtype=np.float32).copy()
for k, di in enumerate(arm.arm_dof_idx):
    full[di] = q_test[k]
from isaacsim.core.utils.types import ArticulationAction
art.apply_action(ArticulationAction(joint_positions=full))
for _ in range(120):
    world.step(render=False)
Mp = arm._fk(arm._arm_q())
try:
    iso_pos, iso_quat = get_world_pose("/World/rebot/end_link")
    R("pinocchio end_link t=", np.round(Mp.translation, 4).tolist())
    R("isaac     end_link t=", np.round(np.asarray(iso_pos), 4).tolist())
    R("translation agreement err mm=", round(float(np.linalg.norm(Mp.translation - np.asarray(iso_pos))) * 1000, 2))
except Exception as e:
    R("isaac end_link pose err:", repr(e))

# ── a reachable move_to actually moves the arm ──
R("\n=== move_to reachable target ===")
arm.go_home()
tcp0 = arm.get_tcp_pose()
R("home TCP pos=", np.round(tcp0[:3, 3], 4).tolist())
# choose a target from a known-reachable joint config's FK
q_goal = np.clip(np.array([0.25, -1.1, -0.9, 0.0, 0.4, 0.0]), arm.jl, arm.ju)
Mg = arm._fk(q_goal); tpos = Mg.translation; trpy = _R_to_rpy(Mg.rotation)
ok_chk, err_chk = arm.check_ik(tpos[0], tpos[1], tpos[2], trpy[0], trpy[1], trpy[2])
R("check_ik(target)=", ok_chk, "err=%.2e" % err_chk)
moved = arm.move_to(tpos[0], tpos[1], tpos[2], trpy[0], trpy[1], trpy[2])
tcp1 = arm.get_tcp_pose()
R("move_to returned:", moved)
R("post-move TCP pos=", np.round(tcp1[:3, 3], 4).tolist(), "target pos=", np.round(tpos, 4).tolist())
R("reached error mm=", round(float(np.linalg.norm(tcp1[:3, 3] - tpos)) * 1000, 2))

# ── an unreachable target should fail check_ik ──
ok_far, err_far = arm.check_ik(1.5, 0.0, 0.2, 0.0, 0.0, 0.0)
R("\ncheck_ik(far 1.5m)=", ok_far, "err=%.3f (expect False)" % err_far)

R("\nP2_DONE")
_RES.close(); sim_app.close()
