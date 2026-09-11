"""Last honest check: an ANGLED down-forward descent. Stage the open gripper
ABOVE and IN FRONT of the box, then descend along the approach axis (diagonal)
into the gap beside the box. Does coming from above-front avoid sweeping the box,
vs the pure-horizontal advance? Try the reachable angled band (pit 0.3..0.8)."""
import os
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
sim_app=SimulationApp({"headless":True})
import numpy as np, sys
sys.path.insert(0,"/root/sim_bridge")
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.xforms import get_world_pose
import isaac_scene as scene
from isaac_arm import IsaacArm
import omni.usd
from pxr import UsdGeom, Usd
TABLE_TOP_Z=0.02; USD="/root/sim_bridge/out/rebot_gripper.usd"
world=World(stage_units_in_meters=1.0); world.scene.add_default_ground_plane()
stage=world.stage; scene.add_table(stage,top_z=TABLE_TOP_Z,cx=0.40,cy=0.0)
add_reference_to_stage(usd_path=USD,prim_path="/World/rebot"); sim_app.update()
art=SingleArticulation(prim_path="/World/rebot",name="rebot"); world.scene.add(art)
world.reset(); art.initialize()
def gains(): 
    c=art.get_articulation_controller(); c.set_gains(kps=np.array([35809.86]*6+[9000.,9000.]),kds=np.array([2000.]*6+[300.,300.]))
gains(); arm=IsaacArm(art,world); box_path=[None]
def spawn(dims,pose):
    if box_path[0] is not None: stage.RemovePrim(box_path[0])
    box_path[0]=scene.add_box(stage,dims,pose,mass=0.05,friction=1.6,name="target_box")
    world.reset(); art.initialize(); gains()
    global arm; arm=IsaacArm(art,world); arm.go_home()
    for _ in range(40): world.step(render=False)
def boxpos(): p,_=get_world_pose(box_path[0]); return np.asarray(p,float)

def trial(lz,x,pit):
    dims=(0.04,0.04,lz); pose=(x,0.0,TABLE_TOP_Z,0.0); spawn(dims,pose)
    for _ in range(30): world.step(render=False)
    box0=boxpos(); box_mid=TABLE_TOP_Z+lz/2.; bhx=dims[0]/2.
    arm.go_home(); arm.open_gripper(0.085)
    # approach direction (tool -X) at this pitch points down-forward. Stage the jaw
    # BACK along the approach axis (so above+in front), then move along it to box_mid.
    # approach world dir:
    from isaac_arm import _rpy_to_R
    appr=-_rpy_to_R(0,pit,0)[:,0]  # commanded -X dir (down-forward)
    tgt=np.array([x,0.0,box_mid])
    back=0.16
    stage_p=tgt - appr*back
    ok=arm.move_to(float(stage_p[0]),0.0,float(stage_p[2]),0.0,pit,0.0,settle_steps=140)
    k_stage=np.linalg.norm(boxpos()[:2]-box0[:2])*1000
    # descend along approach axis
    maxk=k_stage
    for t in np.linspace(0,1,14)[1:]:
        p=stage_p+appr*back*t
        arm.move_to(float(p[0]),0.0,float(p[2]),0.0,pit,0.0,settle_steps=25,continuous=True)
        maxk=max(maxk,np.linalg.norm(boxpos()[:2]-box0[:2])*1000)
    pad=arm.pad_center(); appr_w=-arm.get_tcp_pose()[:3,0]; kfin=np.linalg.norm(boxpos()[:2]-box0[:2])*1000
    fwd_ins=( (x+bhx)-pad[0])*1000
    print(f"lz={lz} x={x} pit={pit:.2f}: stage_ok={ok} appr_down={-appr_w[2]:+.2f} "
          f"pad=[{pad[0]:.3f},{pad[1]:.3f},{pad[2]:.3f}] k_stage={k_stage:.1f} k_max={maxk:.1f} k_fin={kfin:.1f}mm fwd_ins={fwd_ins:+.1f}mm",flush=True)

for lz in (0.05,0.08):
    for x in (0.34,0.38):
        for pit in (0.3,0.5,0.7):
            trial(lz,x,pit)
print("\nPROBE_DONE",flush=True); sim_app.close()
