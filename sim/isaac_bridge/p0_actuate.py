"""P0 final: load USD into a physics World, confirm articulation DOF, actuate fingers
open->close and observe joint positions change. Save a render screenshot."""
import os, sys
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})


import numpy as np
_RES = open("/root/sim_bridge/p0_actuate_result.txt", "w")
def R(*a):
    line=" ".join(str(x) for x in a)
    _RES.write(line+"\n"); _RES.flush()
    print(line, flush=True)
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.core.utils.prims as prim_utils

USD = "/root/sim_bridge/out/rebot_gripper.usd"
ROBOT_PRIM = "/World/rebot"

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD, prim_path=ROBOT_PRIM)
sim_app.update()

art = SingleArticulation(prim_path=ROBOT_PRIM, name="rebot")
world.scene.add(art)
world.reset()
art.initialize()
sim_app.update()

dof_names = art.dof_names
R("DOF count:", art.num_dof)
R("DOF names:", dof_names)

# find finger DOF indices
li = dof_names.index("left_finger_joint")
ri = dof_names.index("right_finger_joint")
arm_idx = [dof_names.index("joint%d" % k) for k in range(1, 7)]
R("arm DOF idx:", arm_idx, "finger idx:", li, ri)

def step(n):
    for _ in range(n):
        world.step(render=False)

# OPEN fingers (target 0.0425 each)
q = art.get_joint_positions()
R("initial finger pos: L=%.4f R=%.4f" % (q[li], q[ri]))
tgt = np.array(art.get_joint_positions(), dtype=np.float32)
tgt[li] = 0.0425; tgt[ri] = 0.0425
art.apply_action(ArticulationAction(joint_positions=tgt))
step(120)
q = art.get_joint_positions()
R("after OPEN target 0.0425: L=%.4f R=%.4f" % (q[li], q[ri]))

# CLOSE fingers (target 0.0)
tgt[li] = 0.0; tgt[ri] = 0.0
art.apply_action(ArticulationAction(joint_positions=tgt))
step(120)
q = art.get_joint_positions()
R("after CLOSE target 0.0: L=%.4f R=%.4f" % (q[li], q[ri]))

R("P0_ACTUATE_DONE")
_RES.close()
sim_app.close()
