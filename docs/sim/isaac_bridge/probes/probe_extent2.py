import os
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
sim_app=SimulationApp({"headless":True})
import numpy as np, sys
sys.path.insert(0,"/root/sim_bridge")
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
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
c=art.get_articulation_controller(); c.set_gains(kps=np.array([35809.86]*6+[9000.,9000.]),kds=np.array([2000.]*6+[300.,300.]))
arm=IsaacArm(art,world); arm.go_home(); arm.open_gripper(0.085)
arm.move_to(0.45,0.0,0.06,0.0,0.0,0.0,settle_steps=150)
pad=arm.pad_center()
print(f"PAD=[{pad[0]:.3f},{pad[1]:.3f},{pad[2]:.3f}]",flush=True)
# gather ALL point-based geometry world xmax per top-level link
import numpy as np
def world_pts(prim):
    m=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    M=np.array([[m[i][j] for j in range(4)] for i in range(4)])
    g=UsdGeom.Mesh(prim) if prim.IsA(UsdGeom.Mesh) else None
    if g is None: return None
    pts=g.GetPointsAttr().Get()
    if not pts: return None
    P=np.array([[p[0],p[1],p[2],1.0] for p in pts])
    W=P@M  # row-vector convention (USD is row-major, point*matrix)
    return W[:,:3]
link_xmax={}
for prim in stage.Traverse():
    p=str(prim.GetPath())
    if not p.startswith("/World/rebot/"): continue
    if not prim.IsA(UsdGeom.Mesh): continue
    W=world_pts(prim)
    if W is None: continue
    link=p.split("/World/rebot/")[1].split("/")[0]
    xmax=float(W[:,0].max())
    if link not in link_xmax or xmax>link_xmax[link][0]:
        link_xmax[link]=(xmax, p)
print("\nPer-link forward(+X) extent (toward box) vs pad:",flush=True)
for link,(xmax,p) in sorted(link_xmax.items(),key=lambda kv:-kv[1][0]):
    print(f"  {link:18s} xmax={xmax:.3f}  fwd_of_pad={(xmax-pad[0])*1000:+.1f}mm",flush=True)
print("\nPROBE_DONE",flush=True); sim_app.close()
