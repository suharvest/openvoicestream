"""P0: import reBot-DevArm_gripper.urdf -> USD via Isaac 4.5 URDF importer,
fixed base to world, headless. Confirm articulation loads with 6 arm joints +
2 finger prismatic joints. Print joint/link list.
"""
import os
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import sys
URDF = "/root/sim/rebot_b601dm_urdf/urdf/reBot-DevArm_gripper.urdf"
MESH_DIR = "/root/sim/rebot_b601dm_urdf/meshes"
OUT_USD = "/root/sim_bridge/out/rebot_gripper.usd"
os.makedirs("/root/sim_bridge/out", exist_ok=True)

# enable the URDF importer extension
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.urdf")
sim_app.update()

import omni.kit.commands
from pxr import Usd, UsdPhysics, Sdf
import isaacsim.asset.importer.urdf as urdf_ext  # noqa

# --- parse URDF import config ---
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
print("URDFCreateImportConfig status:", status)

import_config.merge_fixed_joints = False
import_config.convex_decomp = False        # keep simple for P0
import_config.fix_base = True               # base_link fixed to world
import_config.make_default_prim = True
import_config.self_collision = False
import_config.distance_scale = 1.0
import_config.density = 0.0
try:
    import_config.create_physics_scene = True
except Exception as e:
    print("create_physics_scene set skipped:", e)
# drive defaults: position drive for revolute, force/position for prismatic
try:
    from isaacsim.asset.importer.urdf import UrdfJointTargetType
    import_config.default_drive_type = 1  # position drive
except Exception as e:
    print("drive type set note:", e)

# The package:// prefix maps reBot-DevArm_description_fixend/meshes/ -> MESH_DIR.
# The importer resolves package:// relative to the urdf's directory by default; the
# meshes referenced as package://reBot-DevArm_description_fixend/meshes/X resolve to
# <urdf_dir>/../meshes? We pass the urdf path; importer searches relative dirs.

# Use URDFParseAndImportFile (single-shot parse + import)
result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF,
    import_config=import_config,
    dest_path=OUT_USD,
)
print("URDFParseAndImportFile result:", result, "prim_path:", prim_path)

# open the resulting stage and enumerate the articulation
import omni.usd
ctx = omni.usd.get_context()
ctx.open_stage(OUT_USD)
sim_app.update()
stage = ctx.get_stage()

print("\n=== STAGE PRIMS (joints + articulation root) ===")
arm_joints = []
finger_joints = []
links = []
for prim in stage.Traverse():
    tname = prim.GetTypeName()
    p = prim.GetPath().pathString
    if "Joint" in str(tname):
        # classify
        if prim.HasAPI(UsdPhysics.RevoluteJoint) or tname == "PhysicsRevoluteJoint":
            arm_joints.append((p, str(tname)))
        elif prim.HasAPI(UsdPhysics.PrismaticJoint) or tname == "PhysicsPrismaticJoint":
            finger_joints.append((p, str(tname)))
        else:
            # generic joint - check schema
            if "Revolute" in str(tname):
                arm_joints.append((p, str(tname)))
            elif "Prismatic" in str(tname):
                finger_joints.append((p, str(tname)))
    if tname == "Xform" and ("link" in p.lower() or "finger" in p.lower() or "base" in p.lower()):
        links.append(p)

print("\n--- REVOLUTE joints (expect 6 arm) ---")
for p, t in arm_joints:
    print("  ", t, p)
print("\n--- PRISMATIC joints (expect 2 fingers) ---")
for p, t in finger_joints:
    print("  ", t, p)

# also dump every Physics*Joint by name scan
print("\n--- ALL physics joints by name ---")
for prim in stage.Traverse():
    tn = prim.GetTypeName()
    if "PhysicsJoint" in str(tn) or "Joint" in str(tn) and "Physics" in str(tn):
        print("  ", tn, prim.GetName())

# Articulation root
print("\n--- ArticulationRoot prims ---")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print("  ROOT:", prim.GetPath().pathString)

print("\nP0_DONE n_revolute=%d n_prismatic=%d" % (len(arm_joints), len(finger_joints)))
sim_app.close()
