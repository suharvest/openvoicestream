# Isaac Sim ↔ reBot Grasp Pipeline — Closed-Loop Bridge Spec

**Status:** ready-to-implement (待命). Implement IN the Isaac Sim 4.5.0 container on
`wsl2-local` once the smoke test passes. Iterate in-container (write → run → fix
API → repeat); Isaac 4.5 APIs cannot be dry-run on Mac.

**Purpose:** validate the half Tier A (Mac synthetic-depth harness,
`tools/synthetic_grasp_harness.py` + `tools/grasp_sweep.py`) cannot — **contact
physics**: does the gripper actually hold the box, does it slip, does a search
sweep knock the box off the table (the real-machine incident). Geometry/IK is
already covered by Tier A; this is execution + contact.

---

## 0. The seam — why this needs ZERO pipeline changes

The whole grasp run is one dependency-injected entry point:

```
run_grasp_once(target, arm, *, segmenter=None, camera=None, K=None, ...)
  # agent/ovs_agent/apps/voice_rebot_arm/grasp_service.py:190
```
- `camera` — duck-typed `CameraDriver` (`perception/camera/base.py`): `get_frame() -> (color_bgr uint8 BGR, depth_mm uint16)`, `.K (3x3 float64)`, `.D`. Used at `grasp_service.py:399` (`camera.get_frame()`) and `:479` (`T_cam2base = arm.get_tcp_pose() @ T_hand_eye`).
- `arm` — duck-typed like `RebotArm` (`rebot_arm.py:191`). Methods the pipeline calls:
  | method | file:line | used for |
  |--------|-----------|----------|
  | `get_tcp_pose() -> 4x4` | rebot_arm.py:882 | camera→base extrinsic |
  | `check_ik(x,y,z,roll,pitch,yaw) -> (ok:bool, err:float)` | :894 | reachability + `_relax_orientation` ladder (grasp_service.py:140) |
  | `move_to(x,y,z,roll,pitch,yaw,duration=...)` | :918 | pregrasp / grasp / lift / home |
  | `open_gripper(distance_m) -> bool` | :686 | pre-grasp open |
  | `grasp(force=..., adaptive=...) -> bool` | :731 | compliant close |
  | `gripper_is_holding -> bool` (property) | :481 | success verify |
  | `wait_motion(duration)` | (rebot_arm) | settle blocking |
- `segmenter` — `.predict(bgr, only_names=...) -> [YoloResult]` (`perception/yolo_onnx.py:284`).

**Bridge = three sim-backed objects passed into the SAME `run_grasp_once`:**
`IsaacCameraDriver`, `IsaacArm`, `GtSegmenter`. No edit to grasp_service / ordinary_grasp / transforms.

---

## 1. Architecture

```
Isaac Sim 4.5.0 container (wsl2-local, headless, --gpus all)
 ├─ Scene
 │   ├─ ground plane + table (box, fixed) — table top z defines the world
 │   ├─ target box: parametrized (footprint/height/pose/yaw) PhysX rigid,
 │   │     high-friction material, mass ~0.1–0.3 kg
 │   └─ robot: reBot-DevArm_gripper.urdf → USD articulation,
 │         base_link FIXED to world (Articulation Root)
 ├─ IsaacCameraDriver(CameraDriver)   — sim D405 render product @ T_cam2base
 ├─ GtSegmenter                       — ground-truth instance seg → YoloResult
 ├─ IsaacArm  (duck-typed RebotArm)   — pinocchio IK on the SAME URDF → joint drives
 └─ sweep harness                     — grid of box poses → run_grasp_once → success
```

### Decision 1 — bypass YOLO with ground-truth segmentation
Render RGB-D, but build `YoloResult.masks.data[0]` from Isaac's **instance/semantic
segmentation** of the target box (Replicator `semantic_segmentation` or
`instance_segmentation` annotator), NOT by running the real ONNX YOLO on sim RGB.
Rationale: sim RGB ≠ real D405, so sim-YOLO would inject a domain-gap confound. We
want to test **geometry + execution + contact**, with perception held at ground
truth. (Real-YOLO-on-sim-RGB is a separate, later sim2real study.) The
`GtSegmenter.predict()` returns a `YoloResult` (build via `_Box(xyxy,cls,conf)` +
`_Masks(data=[HxW mask])` + `names={0:target}`, exactly as
`tools/synthetic_grasp_harness.py::make_detection` already does).

### Decision 2 — reuse the SDK kinematics (pinocchio), do NOT use Isaac IK
`IsaacArm.check_ik` / `move_to` solve IK with **pinocchio** (`pip install pin`) on
the **same URDF** (`reBot-DevArm_fixend.urdf`, frame `end_link`,
`forward_kinematics.py:20 _DEFAULT_FRAME="end_link"`; `robot_model.py:57
buildModelFromUrdf`) — i.e. the exact kinematics the real arm uses — then drive the
Isaac articulation joints to that solution. This keeps reachability/IK byte-aligned
with real and with Tier A's `ik_envelope` checks. Isaac's Lula IK would diverge.

### Decision 3 — camera = real hand-eye extrinsic + D405 intrinsics
Mount the sim camera so `T_cam2base = get_tcp_pose() @ T_hand_eye`, using the REAL
`hand_eye.npz` (`/opt/rebot-models/hand_eye.npz` on device — pull it read-only) and
real D405 intrinsics (resolution + fx/fy/cx/cy). Eye-in-hand: camera rides the
wrist, so it moves as the arm moves (re-render per observation pose).

---

## 2. Component contracts

### 2.1 IsaacCameraDriver(CameraDriver)
```
open()        -> create render product (RGB + distance-to-image-plane depth) on a
                 camera prim parented to the wrist/end_link at T_hand_eye
close()       -> destroy render product
get_frame()   -> (color_bgr uint8 HxWx3, depth_mm uint16 HxW)  # depth meters*1000, 0=invalid
.K            -> 3x3 from the sim camera focal/aperture set to match D405
.D            -> zeros (sim is undistorted) unless modeling distortion
```
Notes: set the sim camera focal length + horizontal aperture so the projected K
equals the real D405 K. Convert Isaac depth (meters, may be distance-to-plane) to
uint16 mm; clamp invalids to 0 to match real depth holes. Optionally add the Tier-A
`NoiseModel` on top for realism parity.

### 2.2 GtSegmenter
```
predict(bgr, only_names=None) -> [YoloResult]
   # from Isaac instance-seg of the target box: mask HxW, bbox from mask extents,
   # cls=0, conf=1.0, names={0: target}. Reuse make_detection() shape.
```

### 2.3 IsaacArm (duck-typed RebotArm)
```
get_tcp_pose() -> 4x4   # pinocchio FK(end_link) at current sim joint state
check_ik(x,y,z,r,p,yw) -> (ok, err)   # pinocchio IK; ok if converged within tol
move_to(x,y,z,r,p,yw, duration=…) -> # IK -> set 6 arm joint position targets ->
                                      # step sim until settled or timeout
open_gripper(distance_m) -> bool   # finger prismatic targets = distance/2 each
grasp(force=…, adaptive=…) -> bool # close fingers with force = force_Nm/ r(0.0164)
                                    # -> linear drive force (effort cap 91 N); step
gripper_is_holding (property) -> bool   # fingers NOT fully closed AND box in contact
                                        # with both pads AND box lifted vs table
wait_motion(duration) -> # step sim `duration` seconds
go_home() ->             # move_to home joint config
```
Gripper command mapping (matches `rebot_actuator.py::_apply_gripper` signed
magnitude, see `docs/sim/gripper_urdf_notes.md §6`): open width g→ finger target
g/2; grasp torque |g| N·m → linear force |g|/r. `r = 0.0164 m` (CAD, prismatic
`effort=91 N`).

---

## 3. Scene / physics setup (the Isaac-side TODO from gripper notes §5)
1. URDF→USD via the Isaac URDF importer; map `package://reBot-DevArm_description_fixend/` → local `meshes/` (importer mesh search path).
2. **Convex decomposition** on arm-link visual STLs for stable contact (importer "Convex Decomposition" collision approx). Gripper fingers already have box/mesh collision.
3. **High-friction PhysX material** on finger pads (static/dynamic μ≈1.0–1.5, low restitution); default lower μ elsewhere.
4. **Joint drives**: position stiffness/damping on the 6 arm revolute joints; force/position drive on the 2 prismatic finger joints; couple the two finger targets (equal) for a centered jaw.
5. `base_link` fixed to world (Articulation Root, fixed base).
6. Table top height = world reference; box spawns on it. **Match the table-to-base height to the real cell** (this is what makes Tier A's "flat box z<0.08 unreachable" finding reproducible/falsifiable in physics).

---

## 4. Sweep harness (the payoff)
Reuse the philosophy of `tools/grasp_cycle_check.py`. For each box config in a grid
(footprint × height × x × y × yaw — same axes as `tools/grasp_sweep.py`):
```
reset scene + spawn box at pose
result = run_grasp_once(target, isaac_arm, segmenter=gt_seg, camera=isaac_cam)
step sim through the motion
record: method, jaw_width, reachable(check_ik), GRASP SUCCESS METRIC, plus contact flags
```
**Success metric (what Tier A can't measure):**
- `LIFTED` — box center rose > N cm above table after lift.
- `HELD` — both finger pads in contact with box at end of lift (PhysX contact report).
- `SLIPPED` — contact lost during lift / box rotated out.
- `KNOCKED` — box displaced > M cm during the approach/observe sweep WITHOUT being grasped (the real-machine "swept the box off the table" failure — flag loudly).
Output CSV + summary matrix, same format as `grasp_sweep.py`, so Tier A (geometry)
and Tier B (physics) results are directly comparable per box config.

---

## 5. Phasing (in-container, iterate each before the next)
- **P0** import URDF→USD, fixed base, articulation loads, fingers actuate (open/close visibly). Headless screenshot proof.
- **P1** camera render product + GtSegmenter → `--detect-only`: feed one box, confirm `estimate_grasps` returns a sane GraspPose on sim RGB-D (method/width plausible).
- **P2** IsaacArm IK: `check_ik` + `move_to` to a target, FK round-trips, reachability matches `ik_envelope`.
- **P3** full `run_grasp_once` on ONE box → lift + `gripper_is_holding` true. Tune finger friction/force until a box is reliably held.
- **P4** the sweep grid + success matrix; compare KNOCKED/SLIPPED regions to Tier A's UNREACHABLE/WIDTH_MISMATCH regions.

---

## 6. Open items / prerequisites
- **`gripper_base_joint` origin** still a placeholder (identity). Needs the assembled flange→gripper transform: parse the full assembly STEP `reBot_B601_DM_v1.1_20260425.step` (positions all parts) OR measure on the real arm. Until then the grasp contact point is offset-approximate — fine for P0–P2, matters for P3 contact fidelity.
- **`hand_eye.npz`** + real D405 intrinsics: pull read-only from device `/opt/rebot-models/` (and the camera config). Until then use the Tier-A synthesized extrinsic (documented at top of `synthetic_grasp_harness.py`) — absolute z values are then approximate (same caveat as Tier A's reachability-floor finding).
- **pinocchio in the container**: `pip install pin` inside Isaac's python; load the fixend URDF.
- **Table-to-base height** must match real cell to make the flat-box reachability-floor finding physically testable.

## 7. Implementation guardrails (for the dispatched in-container agent)
- Work INSIDE the container via `docker run … nvcr.io/nvidia/isaac-sim:4.5.0 /isaac-sim/python.sh <script>`; mount the repo + `~/isaac-cache`. Iterate; expect Isaac 4.5 API churn (isaacsim vs omni.isaac.* namespaces).
- New code lives under `sim/isaac/` (e.g. `isaac_camera.py`, `isaac_arm.py`, `gt_segmenter.py`, `run_grasp_sim.py`). Do NOT modify production pipeline code — pass the sim objects into the existing `run_grasp_once`.
- Headless only (`SimulationApp({"headless": True})`); save screenshots/USD for proof. No GUI.
```
```
