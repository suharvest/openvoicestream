"""Trace the forward advance: at what pad-x does the box start to move? Is there
ANY forward stop where the open fingers straddle the box (pad beside body) WITHOUT
knocking it? This isolates whether the side grasp is geometrically clean or the
gripper structure sweeps the box on entry."""
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
def pads():
    o={}
    for link in ("left_finger","right_finger"):
        prim=stage.GetPrimAtPath(f"/World/rebot/{link}")
        m=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t=m.ExtractTranslation(); xa=np.array([m[0][0],m[0][1],m[0][2]],float); xa/=np.linalg.norm(xa)+1e-12
        o[link]=np.array([t[0],t[1],t[2]],float)+(-0.128)*xa
    return o

def trace(lz,x):
    dims=(0.04,0.04,lz); pose=(x,0.0,TABLE_TOP_Z,0.0); spawn(dims,pose)
    for _ in range(30): world.step(render=False)
    box0=boxpos(); box_mid=TABLE_TOP_Z+lz/2.; bhx=dims[0]/2.; front=x+bhx
    arm.go_home(); arm.open_gripper(0.085)
    stage_x=x+0.16
    arm.move_to(stage_x,0.0,box_mid,0.0,0.0,0.0,settle_steps=120)
    print(f"\n## lz={lz} x={x} box_front_face_x={front:.3f} box_center_x={x:.3f}",flush=True)
    print(f"   (advance the jaw from {stage_x:.3f} toward {x-0.02:.3f}; watch pad.x, box knock)",flush=True)
    clean_stop=None
    for tgt in np.arange(stage_x, x-0.03, -0.01):
        arm.move_to(float(tgt),0.0,box_mid,0.0,0.0,0.0,settle_steps=30,continuous=True)
        pad=arm.pad_center(); b=boxpos(); knock=np.linalg.norm(b[:2]-box0[:2])*1000
        pl=pads(); padL=pl['left_finger']; padR=pl['right_finger']
        # pad beside body in x? (pad.x between back and front face)
        beside = (pad[0] < front) and (pad[0] > x-bhx)
        if knock<5 and beside and clean_stop is None:
            clean_stop=(tgt,pad.copy(),knock)
        print(f"   tgt={tgt:.3f} pad.x={pad[0]:.3f} pad.z={pad[2]:.3f} knock={knock:5.1f}mm beside_body={beside} "
              f"padL.x={padL[0]:.3f} padR.x={padR[0]:.3f}",flush=True)
        if knock>40: 
            print("   -> box swept >40mm, stop trace",flush=True); break
    if clean_stop:
        print(f"   CLEAN_STRADDLE at tgt={clean_stop[0]:.3f} pad={np.round(clean_stop[1],3)} knock={clean_stop[2]:.1f}mm",flush=True)
    else:
        print(f"   NO clean straddle stop (box swept before pad got beside body)",flush=True)

for lz in (0.05,0.08):
    for x in (0.34,0.38):
        trace(lz,x)
print("\nPROBE_DONE",flush=True); sim_app.close()
