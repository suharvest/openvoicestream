"""Map the reachable (jaw_z, approach_down) tradeoff per x, and find the SIDE/
FORWARD grasp sweet spot: lowest approach-angle that still reaches box-mid height
beside a table box. Pure pinocchio (no physics)."""
import os
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
sim_app=SimulationApp({"headless":True})
import numpy as np, sys
sys.path.insert(0,"/root/sim_bridge")
import pinocchio as pin
from isaac_arm import _rpy_to_R, FIXEND_URDF, EE_FRAME
model=pin.buildModelFromUrdf(FIXEND_URDF); data=model.createData()
ee_id=model.getFrameId(EE_FRAME); jl=model.lowerPositionLimit.copy(); ju=model.upperPositionLimit.copy()
TOOL=-0.128; TABLE=0.02
def fk(q):
    pin.forwardKinematics(model,data,q); pin.updateFramePlacements(model,data); return data.oMf[ee_id].copy()
def solve(x,y,z,rx,ry,rz,q0=None,iters=300,tol=1e-5):
    R=_rpy_to_R(rx,ry,rz); p_ee=np.array([x,y,z])-R[:,0]*TOOL; oMdes=pin.SE3(R,p_ee)
    q=(pin.neutral(model) if q0 is None else np.asarray(q0,float)).copy()
    for _ in range(iters):
        M=fk(q); iMd=M.actInv(oMdes); err=pin.log(iMd).vector
        if np.linalg.norm(err)<tol: break
        J=pin.computeFrameJacobian(model,data,q,ee_id); J=-np.dot(pin.Jlog6(iMd.inverse()),J)
        v=-J.T@np.linalg.solve(J@J.T+1e-6*np.eye(6),err); q=pin.integrate(model,q,v); q=np.clip(q,jl,ju)
    return q,float(np.linalg.norm(err)),bool(np.all(q>=jl-1e-6) and np.all(q<=ju+1e-6))
SEEDS=[None,pin.neutral(model),np.array([0.,-1.,-1.,0.,.5,0.]),np.array([.3,-1.2,-.8,0.,.6,0.]),
       np.array([-.3,-1.2,-.8,0.,.6,0.]),np.array([0.,-1.2,-1.4,0.,.9,0.]),np.array([0.,-.9,-1.6,0.,1.1,0.])]
def ms(x,y,z,rx,ry,rz,tol=4e-3):
    tgt=np.array([x,y,z]); best=None
    for s in SEEDS:
        q,err,w=solve(x,y,z,rx,ry,rz,q0=s)
        if w and err<tol:
            M=fk(q); jaw=M.translation+M.rotation[:,0]*TOOL; d=np.linalg.norm(jaw-tgt)
            if best is None or d<best[0]: best=(d,q,err,M)
    return best

# For each box height, find SIDE/FORWARD grasp: jaw at box-mid z, MAX approach_down reachable.
print("=== SIDE/FORWARD: jaw at box-MID z, max reachable approach_down per (lz,x) ===",flush=True)
print("(approach_down=1 straight down, 0 horizontal. Box-mid means pad beside body.)",flush=True)
for lz in (0.04,0.05,0.06,0.08,0.10):
    box_mid=TABLE+lz/2.; box_top=TABLE+lz
    print(f"\n-- lz={lz} box_mid_z={box_mid:.3f} box_top={box_top:.3f} --",flush=True)
    for x in (0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42):
        best=None
        for roll in (0.0,3.14159):
            for pit in np.arange(0.0,1.81,0.05):
                b=ms(x,0.0,box_mid,roll,pit,0.0)
                if b is None: continue
                d,q,err,M=b
                if d<0.004:
                    appr=-M.rotation[:,0]; down=float(-appr[2])
                    sep=M.rotation[:,1]  # separation axis ~ end_link Y
                    if best is None or down>best[0]:
                        best=(down,roll,pit,appr,sep)
        if best:
            down,roll,pit,appr,sep=best
            ang=np.degrees(np.arcsin(np.clip(down,-1,1)))
            print(f"  x={x:.2f}: reach@mid down={down:+.2f} ({ang:+.0f}deg below horiz) "
                  f"roll={roll:.2f} pit={pit:.2f} appr=[{appr[0]:.2f},{appr[1]:.2f},{appr[2]:.2f}] "
                  f"sepY=[{sep[0]:.2f},{sep[1]:.2f},{sep[2]:.2f}]",flush=True)
        else:
            print(f"  x={x:.2f}: UNREACHABLE at box-mid z",flush=True)
print("\nPROBE_DONE",flush=True)
sim_app.close()
