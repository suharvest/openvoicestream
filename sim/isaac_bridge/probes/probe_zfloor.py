"""Find the LOWEST reachable jaw/pad z at down-forward orientations, per x.
And ask: at the lowest reachable z, where is the pad relative to a table box?
Pure pinocchio IK FK (no physics). This tells us the hard kinematic floor."""
import os
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
sim_app=SimulationApp({"headless":True})
import numpy as np, sys
sys.path.insert(0,"/root/sim_bridge")
import pinocchio as pin
from isaac_arm import _rpy_to_R, FIXEND_URDF, EE_FRAME
model=pin.buildModelFromUrdf(FIXEND_URDF); data=model.createData()
ee_id=model.getFrameId(EE_FRAME)
jl=model.lowerPositionLimit.copy(); ju=model.upperPositionLimit.copy()
TOOL=-0.128; TABLE=0.02
def fk(q):
    pin.forwardKinematics(model,data,q); pin.updateFramePlacements(model,data)
    return data.oMf[ee_id].copy()
def solve(x,y,z,rx,ry,rz,q0=None,iters=300,tol=1e-5):
    R=_rpy_to_R(rx,ry,rz); p_ee=np.array([x,y,z])-R[:,0]*TOOL
    oMdes=pin.SE3(R,p_ee); q=(pin.neutral(model) if q0 is None else np.asarray(q0,float)).copy()
    for _ in range(iters):
        M=fk(q); iMd=M.actInv(oMdes); err=pin.log(iMd).vector
        if np.linalg.norm(err)<tol: break
        J=pin.computeFrameJacobian(model,data,q,ee_id); J=-np.dot(pin.Jlog6(iMd.inverse()),J)
        v=-J.T@np.linalg.solve(J@J.T+1e-6*np.eye(6),err); q=pin.integrate(model,q,v); q=np.clip(q,jl,ju)
    within=bool(np.all(q>=jl-1e-6) and np.all(q<=ju+1e-6))
    return q,float(np.linalg.norm(err)),within
SEEDS=[None,pin.neutral(model),np.array([0.,-1.,-1.,0.,.5,0.]),np.array([.3,-1.2,-.8,0.,.6,0.]),
       np.array([-.3,-1.2,-.8,0.,.6,0.]),np.array([0.,-1.2,-1.4,0.,.9,0.]),np.array([0.,-.9,-1.6,0.,1.1,0.])]
def ms(x,y,z,rx,ry,rz,tol=4e-3):
    tgt=np.array([x,y,z]); best=None
    for s in SEEDS:
        q,err,w=solve(x,y,z,rx,ry,rz,q0=s)
        if w and err<tol:
            M=fk(q); jaw=M.translation+M.rotation[:,0]*TOOL
            d=np.linalg.norm(jaw-tgt)
            if best is None or d<best[0]: best=(d,q,err,M)
    return best
def appr_down(q):
    M=fk(q); return float(-(-M.rotation[:,0])[2])  # = +appr points down

print("=== LOWEST reachable jaw z per (x, orientation band). Down-forward focus ===",flush=True)
print("For each x, scan z downward at the most-down reachable pitch; report floor.",flush=True)
for x in (0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44):
    # find, across pitch & roll, the lowest z that is reachable with jaw landing on target
    floor=None
    for roll in (0.0,3.14159):
        for pit in np.arange(0.3,2.41,0.1):
            # binary-ish scan z from 0.20 down to 0.02
            for z in np.arange(0.02,0.205,0.01):
                b=ms(x,0.0,z,roll,pit,0.0)
                if b is None: continue
                d,q,err,M=b
                if d<0.004:  # jaw actually lands where commanded (<4mm)
                    appr=-M.rotation[:,0]; down=float(-appr[2])
                    if floor is None or z<floor[0]:
                        floor=(z,roll,pit,down,M.translation+M.rotation[:,0]*TOOL)
    if floor:
        z,roll,pit,down,jaw=floor
        print(f"x={x:.2f}: jaw_z_FLOOR={z:.3f} (roll={roll:.2f} pit={pit:.2f} appr_down={down:+.2f}) "
              f"jaw=[{jaw[0]:.3f},{jaw[1]:.3f},{jaw[2]:.3f}]",flush=True)
    else:
        print(f"x={x:.2f}: NO reachable jaw landing (down-forward band)",flush=True)
print("\nPROBE_DONE",flush=True)
sim_app.close()
