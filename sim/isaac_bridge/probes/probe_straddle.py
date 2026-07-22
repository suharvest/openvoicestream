"""Self-contained: REAL gripper USD rig, HORIZONTAL FORWARD approach, confirm
OPEN fingers straddle a table box +Y/-Y faces with pad beside body (positive
forward insertion). NO friction/lift. Mirrors run_held2.Rig but inline so the
SimulationApp is created exactly once here."""
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

TABLE_TOP_Z=0.02
USD="/root/sim_bridge/out/rebot_gripper.usd"
world=World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage=world.stage
scene.add_table(stage,top_z=TABLE_TOP_Z,cx=0.40,cy=0.0)
add_reference_to_stage(usd_path=USD,prim_path="/World/rebot")
sim_app.update()
art=SingleArticulation(prim_path="/World/rebot",name="rebot")
world.scene.add(art)
world.reset(); art.initialize()
ctrl=art.get_articulation_controller()
ctrl.set_gains(kps=np.array([35809.86]*6+[9000.,9000.]),kds=np.array([2000.]*6+[300.,300.]))
arm=IsaacArm(art,world)
box_path=[None]

def spawn(dims,pose):
    if box_path[0] is not None: stage.RemovePrim(box_path[0])
    box_path[0]=scene.add_box(stage,dims,pose,mass=0.05,friction=1.6,name="target_box")
    world.reset(); art.initialize()
    ctrl=art.get_articulation_controller()
    ctrl.set_gains(kps=np.array([35809.86]*6+[9000.,9000.]),kds=np.array([2000.]*6+[300.,300.]))
    global arm; arm=IsaacArm(art,world); arm.go_home()
    for _ in range(40): world.step(render=False)

def boxpos():
    p,_=get_world_pose(box_path[0]); return np.asarray(p,float)

def finger_pads():
    out={}
    for link in ("left_finger","right_finger"):
        prim=stage.GetPrimAtPath(f"/World/rebot/{link}")
        m=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t=m.ExtractTranslation()
        xax=np.array([m[0][0],m[0][1],m[0][2]],float); xax/=np.linalg.norm(xax)+1e-12
        out[link]=np.array([t[0],t[1],t[2]],float)+(-0.128)*xax
    return out

def probe(lz,x,roll,pit):
    dims=(0.04,0.04,lz); pose=(x,0.0,TABLE_TOP_Z,0.0)
    spawn(dims,pose)
    for _ in range(30): world.step(render=False)
    box0=boxpos(); box_top=TABLE_TOP_Z+lz; box_mid=TABLE_TOP_Z+lz/2.; box_halfy=dims[1]/2.; box_halfx=dims[0]/2.
    arm.go_home(); arm.open_gripper(0.085)
    stage_x=x+0.12
    arm.move_to(stage_x,0.0,box_mid,roll,pit,0.0,settle_steps=120)
    for t in np.linspace(0,1,12)[1:]:
        arm.move_to(stage_x+(x-stage_x)*t,0.0,box_mid,roll,pit,0.0,settle_steps=25,continuous=True)
    arm._step(40)
    jaw=arm.get_tcp_pose()[:3,3]; pad=arm.pad_center(); appr=-arm.get_tcp_pose()[:3,0]
    padLR=finger_pads(); boxN=boxpos(); knock=np.linalg.norm(boxN[:2]-box0[:2])*1000
    fwd_ins=((x+box_halfx)-pad[0])*1000
    padL=padLR['left_finger']; padR=padLR['right_finger']
    straddleY=(padL[1]>box_halfy*0.5 and padR[1]<-box_halfy*0.5) or (padR[1]>box_halfy*0.5 and padL[1]<-box_halfy*0.5)
    pad_in_z=(pad[2]>TABLE_TOP_Z+0.003) and (pad[2]<box_top-0.003)
    print(f"\n## lz={lz} x={x} roll={roll:.2f} pit={pit:.2f}",flush=True)
    print(f"   appr(-X world)=[{appr[0]:.2f},{appr[1]:.2f},{appr[2]:.2f}] (FORWARD into box, down={-appr[2]:+.2f})",flush=True)
    print(f"   box_top={box_top:.3f} box_mid={box_mid:.3f} knock={knock:.1f}mm boxnow=[{boxN[0]:.3f},{boxN[1]:.3f},{boxN[2]:.3f}]",flush=True)
    print(f"   pad=[{pad[0]:.3f},{pad[1]:.3f},{pad[2]:.3f}] pad_in_body_z={pad_in_z} fwd_insertion={fwd_ins:+.1f}mm",flush=True)
    print(f"   padL=[{padL[0]:.3f},{padL[1]:+.3f},{padL[2]:.3f}] padR=[{padR[0]:.3f},{padR[1]:+.3f},{padR[2]:.3f}] box_halfY={box_halfy:.3f}",flush=True)
    print(f"   STRADDLE_Y={straddleY}",flush=True)
    return dict(lz=lz,x=x,knock=knock,fwd_ins=fwd_ins,pad_in_z=pad_in_z,straddle=straddleY)

res=[]
for lz in (0.04,0.05,0.08):
    for x in (0.30,0.34,0.38):
        res.append(probe(lz,x,0.0,0.0))
print("\n=== SUMMARY (horizontal forward approach, fingers straddle in Y) ===",flush=True)
for r in res:
    ok=r['straddle'] and r['pad_in_z'] and r['fwd_ins']>0 and r['knock']<8
    print(f"lz={r['lz']} x={r['x']}: straddle={r['straddle']} pad_in_z={r['pad_in_z']} fwd_ins={r['fwd_ins']:+.1f}mm knock={r['knock']:.1f}mm GEOM_OK={ok}",flush=True)
print("\nPROBE_DONE",flush=True)
sim_app.close()
