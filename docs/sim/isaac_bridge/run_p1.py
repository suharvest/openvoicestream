"""P1: camera render product + GtSegmenter -> estimate_grasps -> GraspPose.

Builds ground+table+box, mounts an Isaac camera at a fixed observation extrinsic
T_cam2base (base<-camera, optical convention), sets its K to the real Orbbec
intrinsics, renders one RGB-D frame, pulls Isaac's GT instance-seg mask of the
box, runs the PRODUCTION estimate_grasps + select_best_grasp, prints the GraspPose.
"""
import os, sys
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
sys.path.insert(0, "/root/sim_bridge")
sys.path.insert(0, "/root/agent/ovs_agent/apps")

_RES = open("/root/sim_bridge/p1_result.txt", "w")
def R(*a):
    line = " ".join(str(x) for x in a)
    _RES.write(line + "\n"); _RES.flush()
    print(line, flush=True)

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep

from isaac_camera import IsaacCameraDriver, _OPT2USD
from gt_segmenter import GtSegmenter
import isaac_scene as scene

from voice_rebot_arm.perception.ordinary_grasp import estimate_grasps, select_best_grasp

# ── real calibration ──
intr = np.load("/root/sim/calib/intrinsics.npz")
K_REAL = np.asarray(intr["camera_matrix"], dtype=np.float64)
W, H = 1280, 720
R("K_REAL=\n" + str(K_REAL))

# ── observation extrinsic (fixed for P1; eye-in-hand realised in P2) ──
# wrist cam looking down-forward at the workspace; same geometry as the Tier-A
# synthetic harness default_T_cam2base (cam ~0.45 m up, pitched 50 deg down).
def look_at_extrinsic(cam_pos, target, world_up=np.array([0.0, 0.0, 1.0])):
    """Optical-convention T_cam2base: +Z toward target, +X right, +Y down."""
    cam_pos = np.asarray(cam_pos, float); target = np.asarray(target, float)
    z = target - cam_pos; z /= np.linalg.norm(z)
    x = np.cross(z, world_up); x /= np.linalg.norm(x)
    y = np.cross(z, x); y /= np.linalg.norm(y)
    T = np.eye(4); T[:3, :3] = np.column_stack([x, y, z]); T[:3, 3] = cam_pos
    return T

# observation pose: wrist cam ~0.45 m up, looking down-forward at the box.
T_cam2base = look_at_extrinsic([0.10, 0.0, 0.45], [0.40, 0.0, 0.05])
R("T_cam2base=\n" + str(T_cam2base))

def up_hint_from_extrinsic(T):
    Rm = np.asarray(T)[:3, :3]
    return (Rm.T @ np.array([0.0, 0.0, 1.0])).astype(np.float64)

# ── build scene ──
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = world.stage

TABLE_TOP_Z = 0.02   # table surface height (world). Box sits on it.
scene.add_table(stage, top_z=TABLE_TOP_Z, cx=0.40, cy=0.0)

BOX_DIMS = (0.05, 0.07, 0.06)        # lx,ly,lz
BOX_POSE = (0.40, 0.0, TABLE_TOP_Z, 0.0)  # cx,cy,table_z,yaw
box_path = scene.add_box(stage, BOX_DIMS, BOX_POSE, mass=0.2, friction=1.2,
                         name="target_box", semantic="box")
R("box_path=", box_path, "dims=", BOX_DIMS, "pose=", BOX_POSE)

# ── camera prim ──
cam_prim = "/World/obs_cam"
camera = Camera(prim_path=cam_prim, resolution=(W, H))
camera.initialize()
# set K: fx = focal * W / h_aperture ; pick h_aperture=20.955 (RS-ish), back out focal
H_APERTURE = 20.955
fx, fy = K_REAL[0, 0], K_REAL[1, 1]
cx, cy = K_REAL[0, 2], K_REAL[1, 2]
focal = fx * H_APERTURE / W
v_aperture = H_APERTURE * H / W
camera.set_focal_length(focal)
camera.set_horizontal_aperture(H_APERTURE)
camera.set_vertical_aperture(v_aperture)
camera.set_clipping_range(0.01, 10.0)
camera.add_distance_to_image_plane_to_frame()
camera.add_instance_id_segmentation_to_frame()  # prim-path based GT (robust)

world.reset()
camera.initialize()

# place camera at optical extrinsic
drv = IsaacCameraDriver(camera, K_REAL, W, H)
drv.set_extrinsic(T_cam2base)

# let physics settle + render a few frames so annotators populate
for _ in range(30):
    world.step(render=True)

K_sim = camera.get_intrinsics_matrix()
R("K_SIM (from Isaac camera)=\n" + str(np.asarray(K_sim)))

# ── GT instance-seg mask of the box ──
def box_mask_provider():
    frame = camera.get_current_frame()
    seg = frame.get("instance_id_segmentation")
    if seg is None:
        return None
    data = np.asarray(seg["data"])
    idToLabels = seg["info"].get("idToLabels", {})
    # prim-path based: select the id whose label is the target_box prim path
    target_ids = [int(k) for k, v in idToLabels.items() if "target_box" in str(v)]
    if not target_ids:
        return None
    return np.isin(data, target_ids).astype(np.uint8)

mask = box_mask_provider()
R("GT mask pixels:", None if mask is None else int(mask.sum()),
  "shape:", None if mask is None else mask.shape)

bgr, depth_mm = drv.get_frame()
R("rgb shape:", None if bgr is None else bgr.shape,
  "depth shape:", None if depth_mm is None else depth_mm.shape,
  "depth valid px:", None if depth_mm is None else int((depth_mm > 0).sum()),
  "depth range mm:", None if depth_mm is None else (int(depth_mm[depth_mm>0].min()), int(depth_mm.max())))

# save artifacts
import cv2
if bgr is not None:
    cv2.imwrite("/root/sim_bridge/p1_rgb.png", bgr)
if mask is not None:
    cv2.imwrite("/root/sim_bridge/p1_mask.png", (mask*255).astype(np.uint8))

# ── run grasp pipeline ──
seg = GtSegmenter(box_mask_provider, class_name="box")
results = seg.predict(bgr)
R("detections:", sum(len(getattr(r, 'boxes', [])) for r in results))

up_hint = up_hint_from_extrinsic(T_cam2base)
grasps = estimate_grasps(results, depth_mm, K_REAL, depth_quantile=0.5,
                         up_hint_cam=up_hint)
g = select_best_grasp(grasps)
R("\n=== GRASP RESULT ===")
if g is None:
    R("GRASP=None (no valid grasp)")
    for gr in grasps:
        R("  candidate rejected_reason=", getattr(gr, 'rejected_reason', '?'),
          "method=", getattr(gr, 'method', '?'))
else:
    R("method=", g.method)
    R("class_name=", g.class_name, "conf=", g.conf)
    R("jaw_width_m=", round(float(g.jaw_width_m), 4))
    R("object_length_m=", round(float(g.object_length_m), 4))
    R("angle_deg=", round(float(g.angle_deg), 2))
    R("position_cam (m)=", np.round(np.asarray(g.position, float), 4).tolist())
    R("center_px=", g.center_px, "valid_depth_pixels=", g.valid_depth_pixels)
    # transform to base to sanity-check absolute reachable region
    from voice_rebot_arm.perception.transforms import transform_grasp_pose_to_base
    grasp6, pregrasp6 = transform_grasp_pose_to_base(
        np.asarray(g.position, float), np.asarray(g.tcp_rotation, float),
        T_cam2base, pregrasp_offset_m=0.08, insertion_depth_m=0.0)
    R("grasp6 base (x,y,z,r,p,yw)=", [round(v, 4) for v in grasp6])

R("\nP1_DONE")
_RES.close()
sim_app.close()
