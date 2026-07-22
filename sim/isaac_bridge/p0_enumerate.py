"""P0 enumeration: open imported USD, list joints/links. Uses Usd.Stage.Open."""
import os, sys
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics
OUT_USD = "/root/sim_bridge/out/rebot_gripper.usd"

stage = Usd.Stage.Open(OUT_USD)
sys.stdout.write("STAGE_OPENED=%s\n" % (stage is not None)); sys.stdout.flush()

revolute, prismatic, fixed, rbodies, roots = [], [], [], [], []
type_counts = {}
for prim in stage.Traverse():
    tn = str(prim.GetTypeName())
    type_counts[tn] = type_counts.get(tn, 0) + 1
    name = prim.GetName()
    path = prim.GetPath().pathString
    if tn == "PhysicsRevoluteJoint":
        j = UsdPhysics.RevoluteJoint(prim)
        revolute.append((name, j.GetAxisAttr().Get(), j.GetLowerLimitAttr().Get(),
                         j.GetUpperLimitAttr().Get(), path))
    elif tn == "PhysicsPrismaticJoint":
        j = UsdPhysics.PrismaticJoint(prim)
        prismatic.append((name, j.GetAxisAttr().Get(), j.GetLowerLimitAttr().Get(),
                          j.GetUpperLimitAttr().Get(), path))
    elif tn == "PhysicsFixedJoint":
        fixed.append((name, path))
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rbodies.append(path)
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        roots.append(path)

out = []
out.append("=== PRIM TYPE COUNTS ===")
for t, c in sorted(type_counts.items()):
    out.append("  %4d  %s" % (c, t))
out.append("\n=== ARTICULATION ROOT(S) — %d ===" % len(roots))
out += ["  " + r for r in roots]
out.append("\n=== REVOLUTE JOINTS (arm) — count=%d ===" % len(revolute))
for n, a, lo, hi, p in revolute:
    out.append("  %-12s axis=%s limit=[%s,%s]  %s" % (n, a, lo, hi, p))
out.append("\n=== PRISMATIC JOINTS (fingers) — count=%d ===" % len(prismatic))
for n, a, lo, hi, p in prismatic:
    out.append("  %-20s axis=%s limit=[%s,%s]  %s" % (n, a, lo, hi, p))
out.append("\n=== FIXED JOINTS — count=%d ===" % len(fixed))
for n, p in fixed:
    out.append("  %-22s %s" % (n, p))
out.append("\n=== RIGID BODY LINKS — count=%d ===" % len(rbodies))
out += ["  " + p for p in rbodies]
out.append("\nP0_ENUM_DONE revolute=%d prismatic=%d fixed=%d rbodies=%d roots=%d" %
           (len(revolute), len(prismatic), len(fixed), len(rbodies), len(roots)))
sys.stdout.write("\n".join(out) + "\n"); sys.stdout.flush()
sim_app.close()
