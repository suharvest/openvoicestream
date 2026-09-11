"""Scene builder: ground plane + table + parametrized box (PhysX rigid, friction)."""
import numpy as np
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, UsdShade, Sdf, Usd


def _add_physics_material(stage, path, static_friction, dynamic_friction, restitution):
    mat = UsdShade.Material.Define(stage, path)
    physx_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physx_mat.CreateStaticFrictionAttr().Set(static_friction)
    physx_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physx_mat.CreateRestitutionAttr().Set(restitution)
    return mat


def _bind_material(stage, prim_path, mat):
    prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI(prim).Bind(
        mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics")


def set_high_friction(stage, prim_path, static_friction=1.5, dynamic_friction=1.4,
                      restitution=0.0, key="pad"):
    """Bind a high-friction physics material to an existing collider prim."""
    mat = _add_physics_material(stage, f"/World/phys/{key}_mat",
                                static_friction, dynamic_friction, restitution)
    _bind_material(stage, prim_path, mat)
    return mat


def add_table(stage, top_z, cx=0.40, cy=0.0, size_x=0.6, size_y=0.8, thickness=0.04,
              friction=0.8):
    """Static table whose TOP surface is at world z=top_z."""
    path = "/World/table"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    cube.GetExtentAttr().Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    cube.CreateDisplayColorAttr().Set([(0.5, 0.5, 0.55)])
    half = thickness / 2.0
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, top_z - half))
    xf.AddScaleOp().Set(Gf.Vec3f(size_x, size_y, thickness))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    # static (no RigidBodyAPI) => fixed collider
    mat = _add_physics_material(stage, "/World/phys/table_mat", friction, friction, 0.0)
    _bind_material(stage, path, mat)
    return path


def add_box(stage, dims, pose, mass=0.2, friction=1.2, name="target_box",
            semantic="box"):
    """Dynamic PhysX rigid box.
    dims=(lx,ly,lz) full extents (m); pose=(cx,cy,table_z,yaw) box bottom at table_z.
    Returns prim_path.
    """
    lx, ly, lz = dims
    cx, cy, table_z, yaw = pose
    path = f"/World/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    cube.GetExtentAttr().Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    cube.CreateDisplayColorAttr().Set([(0.85, 0.25, 0.15)])
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, table_z + lz / 2.0))
    xf.AddRotateZOp().Set(np.degrees(yaw))
    xf.AddScaleOp().Set(Gf.Vec3f(lx, ly, lz))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    massapi = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    massapi.CreateMassAttr().Set(mass)
    mat = _add_physics_material(stage, f"/World/phys/{name}_mat", friction, friction, 0.0)
    _bind_material(stage, path, mat)
    # semantic label for GT segmentation (canonical Isaac 4.5 API)
    from isaacsim.core.utils.semantics import add_update_semantics
    add_update_semantics(cube.GetPrim(), semantic_label=semantic, type_label="class")
    return path
