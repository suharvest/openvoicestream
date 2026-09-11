import os
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})
import numpy as np, sys
sys.path.insert(0,"/root/sim_bridge")
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
import pinocchio as pin
import isaac_scene as scene
from isaac_arm import IsaacArm
OUT=open("/root/sim_bridge/probe_negpitch.txt","w")
def P(*a): s=" ".join(str(x) for x in a); OUT.write(s+"\n"); OUT.flush()
world = World(stage_units_in_meters=1.0); world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path="/root/sim_bridge/out/rebot_gripper.usd", prim_path="/World/rebot")
sim_app.update()
art=SingleArticulation(prim_path="/World/rebot", name="rebot"); world.scene.add(art)
world.reset(); art.initialize()
arm=IsaacArm(art, world)
# joint limits
P("joint lower:",np.round(arm.jl,3))
P("joint upper:",np.round(arm.ju,3))
# Direct IK with MANY seeds incl elbow-up/wrist-flip, target -X straight down (pitch=-1.57)
def many_seed_ik(x,y,z,rx,ry,rz):
    seeds=[None,arm._home_q]
    rng=np.random.default_rng(0)
    for _ in range(60):
        seeds.append(rng.uniform(arm.jl,arm.ju))
    best=(np.inf,None,False)
    for s in seeds:
        q,err,within=arm._solve_ik(x,y,z,rx,ry,rz,q0=s,iters=300,tol=1e-4)
        if within and err<best[0]: best=(err,q,within)
        if err<best[0] and best[1] is None: best=(err,q,within)
    return best
P("=== can we reach pitch=-1.57 (-X straight down) anywhere? many random seeds ===")
found=False
for x in (0.18,0.22,0.26,0.30,0.34,0.38,0.42):
  for z in (0.05,0.10,0.15,0.20,0.25):
    err,q,within=many_seed_ik(x,0.0,z,0.0,-1.5708,0.0)
    ok=within and err<6e-3
    if ok:
        found=True
        P(f"  REACHABLE x={x} z={z} err={err:.5f} q={np.round(q,3)}")
P("ANY pitch=-1.57 reachable:",found)
# Also the best achievable downward tilt: max over reachable of (+X dot +Z) i.e. -X dot -Z
P("=== max reachable -X-down (scan pitch incl negative, fine) ===")
best=(-9,)
for x in (0.20,0.26,0.30,0.34,0.40):
  for z in (0.06,0.12,0.18):
    for gp in np.arange(-1.6,-0.0,0.1):
        err,q,within=many_seed_ik(x,0.0,z,0.0,float(gp),0.0)
        if within and err<6e-3:
            R=arm._fk(q).rotation
            negX=-R[:,0]; dd=float(np.dot(negX,[0,0,-1.0]))
            if dd>best[0]: best=(dd,x,z,gp)
P("BEST negpitch reachable -X·(-Z)=%.3f at x=%.2f z=%.2f pitch=%.2f"%(best if len(best)>1 else (best[0],0,0,0)))
P("PROBE_DONE"); OUT.close(); sim_app.close()
