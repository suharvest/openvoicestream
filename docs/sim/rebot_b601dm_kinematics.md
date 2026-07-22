# reBot B601-DM — Kinematic Spec Sheet (for hand-authored URDF / Isaac Sim)

**Purpose:** Everything recoverable about the B601-DM arm's kinematics from the
codebase + CAD inventory, to hand-author a URDF later. Every number is cited
to `file:line`. Numbers not present in code are flagged
**`MEASURE FROM CAD`** / **`MEASURE FROM ARM`** rather than invented.

> **CRITICAL CAVEAT (read first):** The numeric kinematics — DH parameters,
> link lengths, per-joint limits, the FK/IK chain — live entirely inside the
> vendored SDK package **`reBotArm_control_py`** (its `kinematics` module +
> shipped `urdf` + `config/arm.yaml`). That package is **NOT checked out on
> this Mac** — it is only present inside the rebot-arm container at
> `/opt/rebot/reBotArm_control_py/` (see `rebot_arm.py:42`,
> `rebot_arm.py:249-258`, `config.yaml:264-267`). The app code calls
> `load_robot_model()` / `compute_fk` / `solve_ik` from it but does not
> redefine any kinematics. **So the canonical URDF already exists inside the
> SDK** — the cleanest path is to pull
> `/opt/rebot/reBotArm_control_py/` (the `urdf/*.urdf` + meshes + the URDF that
> `load_robot_model()` loads) off the device rather than re-derive from CAD.
> Everything below is what's derivable *without* that package.

---

## 0. Hardware identity (from CAD BOM)

- Arm = **6 revolute joints + 1 gripper actuator = 7 Damiao brushless motors.**
  BOM: `4× DM4310(V4)` + `3× DM4340P(V4)`
  (`reBot-DevArm/hardware/reBot_B601_DM/readme.md:131-132`).
  v1.1 changelog: "Fixed Joint 1 model from 4310 to 4340P" and "Cable Restraint
  for the 3 end joint motors" (`readme.md:21`) ⇒ Joints 1–3 are the big
  DM4340P (175 USD), Joints 4–6 + gripper are DM4310, consistent with a
  3-big-base + 3-small-wrist layout. **Confirm the exact joint↔motor map from
  the SDK `arm.yaml` motor list — MEASURE FROM SDK.**
- Motor bus: Damiao DM-serial over an HDSC HC32 USB-CDC bridge,
  VID:PID `2e88:4603` (`rebot_actuator.py:453-459`).
- Bus channel realpath `/dev/ttyACM1` (NOT ttyACM0 = the SO-ARM)
  (`rebot_actuator.py:27-33`, `config.yaml:260-263`).

---

## 1. DOF / joint chain

**6-DOF serial revolute arm + 1 parallel-jaw gripper actuator (linear,
rack-driven).** Base → gripper.

| # | Joint | Type | Motor (likely) | CAD link part that follows it | Axis (see caveat) |
|---|-------|------|----------------|-------------------------------|-------------------|
| J1 | Base yaw | revolute | DM4340P | `Metal_Parts/03_Link1.step` (+ `02_Arm_Yaw_Limit.step` = yaw hard stop, `readme.md:93`) | base Z (vertical) |
| J2 | Shoulder pitch | revolute | DM4340P | `Metal_Parts/03_Link2.step` | horizontal |
| J3 | Elbow pitch | revolute | DM4340P | `Metal_Parts/03_Link3_L.step` + `03_Link3_R.step` (forked link) | horizontal, ∥ J2 |
| J4 | Forearm roll | revolute | DM4310 | `Metal_Parts/02_Lower_Upper_Link_L/R.step` | along forearm axis |
| J5 | Wrist pitch | revolute | DM4310 | `Metal_Parts/02_Lower_Wrist_Link_L/R.step`, `02_Wrist_Bracket.step` (`readme.md:97`) | horizontal |
| J6 | Wrist roll | revolute | DM4310 | `Metal_Parts/03_Link5.step`, `02_FLANGE.step` | tool axis |
| G  | Gripper | prismatic (parallel jaw, rack & pinion) | DM4310 | `02_Rack.step`/`02-RACK.step`, `02_Slider_Bracket.step`, `02_Slider_Extension.step` ×2, `3D_Printed_Parts/01_Finger.step` ×2 | linear, jaw open/close |

**Joint sense / order as the code assumes it:** The app never indexes
individual joints by number — it works purely in **Cartesian TCP space**.
`move_to(x,y,z,roll,pitch,yaw)` runs SDK IK internally (`rebot_arm.py:918-928`,
`rebot_actuator.py:332-361`); `get_tcp_pose()` runs SDK FK
(`rebot_arm.py:882-890`). The joint-vector `q` is opaque (`q,_,_ =
self._arm.get_state()`, `rebot_arm.py:885`, `:901`). **Per-joint axis
direction + rotation sense are NOT in app code — MEASURE FROM SDK URDF / CAD.**

The 6-DOF + revolute-base + parallel-jaw structure is corroborated by the
grasp transform expecting a full 6-DOF wrist that can achieve arbitrary
approach pitch (`config.yaml:316-323`, scan poses use pitch up to 1.57 rad).

---

## 2. Joint limits

**Per-joint angular limits are NOT in app code.** They live in the SDK
`arm.yaml` / URDF (not on this Mac). What *is* in code:

### Cartesian workspace envelope (the practical "limits" the app enforces)
Measured IK reachability (`tools/artifacts/ik_envelope_b601dm.csv`, 6125 rows).
Columns: `x, y, z, pitch, yaw, ok, err`. `ok=1` ⇒ SDK `solve_ik` converged.

Feasible band (where `ok==1`, 4976 of 6125 grid points):
- **x ∈ [0.25, 0.55] m** (grid stepped 0.05; full grid 0.25–0.55)
- **y ∈ [-0.20, 0.20] m**
- **z ∈ [0.08, 0.25] m**
- **pitch ∈ {0, 0.225, 0.45, 0.675, 0.9, 1.2, 1.57} rad** (down-tilt)
- **yaw ∈ {-0.6, -0.3, 0, 0.3, 0.6} rad**
- (roll not swept — held at 0 in this probe.)

> Source rows e.g. `ik_envelope_b601dm.csv:1` header, `:2-5` (x=0.25,y=-0.2,z=0.08
> infeasible at several yaw), feasible region computed by filtering col6==1.
> NOTE: this is a coarse 0.05 m / 0.225 rad grid, NOT joint limits. It tells you
> the *reachable Cartesian box*, useful as a URDF sanity check, not as
> `<limit lower upper>` values.

A tighter operational box the grasp code trusts (hand-tuned, narrower than the
raw envelope): **x∈[0.20,0.34], y∈[-0.14,0.14], z∈[0.12,0.34], |rpy|≤0.5**
(`config.yaml:318`). Place/reach bounds: `place_bounds [0.20, 0.60, -0.26, 0.40]`
(x_min/x_max/y_min/y_max base frame, `config.yaml:362`); x_max 0.60 = "caps at
reach", x_min 0.20 = "base sits on the table" (`config.yaml:361`).

**Per-joint `<limit lower="" upper="" velocity="" effort="">` → MEASURE FROM SDK
`arm.yaml`** (and J1 has a physical yaw hard-stop part `02_Arm_Yaw_Limit.step`,
`01_Lower_Arm_Limit.step`, `01_Upper_Arm_Limit.step` — MEASURE the angle FROM CAD).

---

## 3. Link dimensions / offsets

**No link lengths, DH parameters, or TCP offset exist anywhere in the app
code.** FK/IK are delegated to the SDK model. Recoverable facts:

- The SDK loads its model from a URDF: `load_robot_model(urdf_path=...)` or the
  SDK default `load_robot_model()` (`rebot_arm.py:280-283`). **That URDF already
  contains all link lengths + joint origins — PULL IT FROM `/opt/rebot/`.**
- TCP frame = the SDK's "end effector frame" `get_end_effector_frame_id(model)`
  (`rebot_arm.py:286`). Where exactly the TCP sits (jaw tip vs flange) is
  **defined in that URDF — MEASURE FROM SDK URDF.**
- TCP→approach offsets the app applies on top of the SDK TCP:
  - `insertion_depth_m = 0.025` m advance along approach axis past the grasp
    point before closing (`config.yaml:391`).
  - `pregrasp_offset_m` retreat along tool-X (`transforms.py:181-192`,
    `_offset_along_tool_x`, `transforms.py:175-178`).

**MEASURE FROM CAD (link lengths / joint origins) if SDK URDF unavailable:**
| Link length / offset | Part file(s) |
|---|---|
| Base height (table → J1 axis) | `3D_Printed_Parts/01_BASE_Link.step`, `01_BASE_Plate.step`, `Metal_Parts/02_Base_Reinforcement_Part.step` |
| J1→J2 offset | `Metal_Parts/03_Link1.step` |
| J2→J3 (upper-arm length) | `Metal_Parts/03_Link2.step` + covers `3D_Printed_Parts/01_Upper_Arm_Cover.step` |
| J3→J4 (forearm length) | `Metal_Parts/03_Link3_L.step`, `03_Link3_R.step`, `02_Lower_Upper_Link_L/R.step` |
| J4→J5 | `Metal_Parts/02_Lower_Wrist_Link_L/R.step` |
| J5→J6 (wrist) | `Metal_Parts/03_Link5.step`, `02_Wrist_Bracket.step`, `02_FLANGE.step` |
| J6→gripper base (flange→jaw) | `Metal_Parts/02_Gripper_Connector_A.step`, `02_Gripper_Connector_B.step`, `02_Gear_Connector.step` |
| Jaw/finger geometry | `3D_Printed_Parts/01_Finger.step`, `Soft_Gripper_Finger.step`, `Metal_Parts/02_Rack.step` |

Full assembly for measuring overall pose:
`reBot-DevArm/hardware/reBot_B601_DM/reBot_B601_DM_v1.1_20260425.step` (34 MB).

---

## 4. Gripper

- **Type: parallel jaw, rack-and-pinion linear drive** (`02_Rack.step` +
  `02_Slider_Bracket.step` + `02_Slider_Extension.step`; one DM4310 motor in
  MIT/torque mode, `rebot_arm.py:468-469`).
- **Open width range: 0 – 0.09 m mechanical full open.**
  `_G_MAX_DIST_M = 0.09` (`rebot_arm.py:145`). Encoder maps motor angle
  `_G_ANGLE_OPEN = -5.0 rad` (full open) → 0.09 m
  (`rebot_arm.py:146`, `:515` `gripper_opening_m`).
  - Soft open limit `_G_OPEN_SOFT_LIMIT = -4.9 rad` (`rebot_arm.py:147`).
  - **Real measured full open ≈ 0.0853 m** at the -4.9 rad soft limit
    (`rebot_arm.py:170`), i.e. commanded 0.09 m never fully reached.
- **Usable jaw limit for grasp geometry: 0.085 m.** Candidates wider than
  this are rejected as too wide (`perception/ordinary_grasp.py:161`, `:165`).
  Release verification only trustworthy for objects ≤ ~0.078 m
  (`rebot_arm.py:174-175`).
- **Config open width:** action-frame max `open_distance_m = 0.09`
  (`config.yaml:282`); grasp-pipeline pre-grasp open `0.06`
  (`config.yaml:366`, deliberately < full open for safety).
- **Grasp force units = N·m (motor torque feed-forward), NOT newtons.**
  - SDK hard ceiling `_G_TAU_MAX = 1.5` N·m (`rebot_arm.py:150`).
  - Default `_G_DEFAULT_FORCE = 0.30` N·m (`rebot_arm.py:160`).
  - Config clamp `grasp_force = 0.8` N·m (`config.yaml:372`); per-class box =
    0.8 (`config.yaml:381-385`); actuator-level clamp default 0.6
    (`config.yaml:279`).
  - Close torque `_G_CLOSE_TORQUE = 1.0` N·m (`rebot_arm.py:154`).
  - Frame `gripper` field is a **signed magnitude**: `+x` = open to x metres,
    `-x` = grasp at |x| N·m, `0` = hold (`rebot_actuator.py:14-19`, `:372-407`).
- **Finger part files:** `3D_Printed_Parts/01_Finger.step` (×2, ABS),
  `Soft_Gripper_Finger.step`, `Soft_Gripper_Mount.step`. Gripper mounts:
  `02_Gripper_Connector_A/B.step`.
- **MEASURE FROM CAD:** jaw stroke geometry (rack travel per radian — code uses
  a *linear* `pos/-5.0 → 0..0.09` model, `rebot_arm.py:515`, which is the
  calibration the URDF prismatic joint should match), finger contact-pad
  position/size, finger length.

---

## 5. Base / world frame + TCP frame convention

### Base frame (as the app assumes)
- Right-handed. **+X = forward / reach direction** (arm reaches out in +x;
  x∈[0.20,0.60], base sits at low x, `config.yaml:361`). **+Z = up** (z is
  height above table, `config.yaml:316-323`; place height 0.15 m). **+Y =
  lateral** (y∈[-0.26,0.40]). Origin at the J1 base.
- TCP pose read as a 4×4 homogeneous transform `T[:3,3]=position`,
  `T[:3,:3]=rotation` (`rebot_arm.py:887-890`, `rebot_actuator.py:233-236`).
- Orientation convention for `move_to` rpy: **ZYX intrinsic Euler**
  (roll=X, pitch=Y, yaw=Z), `R = Rz@Ry@Rx` (`transforms.py:35-75`,
  `pose6d_to_mat4`; decode `mat4_to_pose6d` / `rotation_matrix_to_euler_zyx`,
  `transforms.py:78-99`).

### TCP frame (tool frame, as grasp code maps it)
From `grasp_axes_to_rebot_tcp_rotation` (`transforms.py:117-158`):
- **TCP +X = tool-forward = approach direction (points INTO the object,
  downward in base)** — `tcp_x = -approach` (`transforms.py:144`).
  ⇒ **The APPROACH AXIS of the gripper is the tool +X axis.**
- **TCP +Y = jaw open/close direction** (`transforms.py:129-132, 145-146`).
- **TCP +Z = completes right-handed frame**, aligned with the grip axis
  (`transforms.py:147-154`).
- Symmetry: a 180° twist about tool-X is grasp-equivalent; the code picks the
  branch with smaller |roll| (`canonicalize_parallel_gripper_tcp_rotation`,
  `transforms.py:102-114`, using `_ROT_X_PI`, `transforms.py:8-15`).

> Cross-check `transform_grasp_pose_to_base` (`transforms.py:181-192`):
> `T_grasp_base = T_cam2base @ T_grasp_cam`, then pre-grasp/insertion offsets
> are applied **along tool-X** (`_offset_along_tool_x`, `transforms.py:175-178`),
> confirming tool-X = approach. (The app's `perception/transforms.py` is a
> trimmed vendored copy of the grasp repo's `utils/transforms.py:195-239`.)

---

## 6. Camera mount (D405, eye-in-hand)

- **Camera: Intel RealSense D405**, wrist-mounted (eye-in-hand). CAD mount part
  **`3D_Printed_Parts/D405_305_Mount.step`** (alt mounts also present:
  `D435_Gemini2_Mount.step`, `UVC32_mount.step`). The mount bolts to the wrist
  (J5/J6 region), so the camera moves with the TCP.
- **Runtime camera** in the deployed config is actually an **Orbbec Gemini2**
  (`config.yaml:305-309`, `type: orbbec_gemini2`, 1280×720@30) — i.e. the
  D405 mount is the CAD ground truth but the shipped sensor may differ; the
  D435/Gemini2 mount part matches the runtime sensor. **Confirm which sensor +
  mount is on the real arm — MEASURE FROM ARM.**
- **Hand-eye transform (camera ← TCP), 4×4, eye-in-hand:** loaded from
  `hand_eye.npz` (TSAI calibration, 16 samples), key `T_hand_eye` (falls back
  to first key) (`config.yaml:311-313`, `grasp_plugin.py:965-982`).
  Path: `${REBOT_HAND_EYE:-/opt/rebot-models/hand_eye.npz}` (`config.yaml:313`).
- **Camera→base composition:** `T_cam2base = arm.get_tcp_pose() @ T_hand_eye`
  (`grasp_service.py:8`, `:378-379`). So `T_hand_eye` is the **static TCP→camera
  mount transform** the URDF camera link must encode.
- **For the URDF camera link origin (camera optical frame relative to wrist
  flange): use `T_hand_eye` from `hand_eye.npz` as the ground-truth extrinsic**,
  OR **MEASURE FROM CAD** the `D405_305_Mount.step` mounting face + the D405
  datasheet optical-center offset. The `hand_eye.npz` artifact is NOT in this
  checkout (lives at `/opt/rebot-models/`, see `RUNBOOK.md:78-89`) — pull it
  from the device. Camera intrinsics exist at
  `reBot-DevArm-Grasp/config/calibration/{orbbec_gemini2,realsense_d405}/intrinsics.npz`.

---

## 7. Gap list — what must still be measured before the URDF is final

**Pull from the SDK (best source — these are already authored there):**
1. **The canonical URDF** `reBotArm_control_py`'s shipped `urdf/*.urdf` (whatever
   `load_robot_model()` loads) + its meshes — at `/opt/rebot/reBotArm_control_py/`.
   This single artifact resolves items 2–6 below. (`rebot_arm.py:280-286`.)
2. Per-joint **axis directions + rotation sense** (J1–J6).
3. Per-joint **limits** `lower / upper / velocity / effort`
   (from SDK `arm.yaml` + URDF).
4. **Link lengths / joint origins** (DH or URDF `<origin xyz rpy>` per link).
5. **TCP frame definition** (exactly where the end-effector frame sits —
   flange face vs jaw tip).
6. **Joint ↔ motor map** (which of the 4×DM4310 / 3×DM4340P is each joint;
   `arm.yaml` motor list).

**Measure from CAD (`reBot-DevArm/hardware/reBot_B601_DM/`) if SDK URDF
unavailable:**
7. All link lengths / inter-joint offsets (part files in §3 table).
8. Base height (table → J1 axis).
9. Gripper jaw stroke geometry, finger length, contact-pad position
   (`01_Finger.step`, `02_Rack.step`); validate the linear
   `angle/-5.0 rad → 0..0.09 m` map (`rebot_arm.py:515`).
10. J1 yaw hard-stop angle (`02_Arm_Yaw_Limit.step`) + lower/upper arm limit
    stops (`01_Lower_Arm_Limit.step`, `01_Upper_Arm_Limit.step`).
11. Camera optical-frame offset from wrist flange (`D405_305_Mount.step` + D405
    datasheet) — OR just use `T_hand_eye`.

**Measure from / pull off the real arm:**
12. `hand_eye.npz` (TCP→camera extrinsic) — `/opt/rebot-models/hand_eye.npz`.
13. Which camera + mount is physically installed (D405 vs Orbbec Gemini2).
14. Inertial properties (mass, COM, inertia tensor per link) — **nothing in
    code or required by the listed sources; needed for dynamics in Isaac Sim.**
    MEASURE FROM CAD (assign material densities) or the SDK URDF if it carries
    `<inertial>`.
15. Collision meshes — derive from the STEP parts (simplified convex hulls).

---

## EVIDENCE — raw excerpts

### STEP file inventory
`reBot-DevArm/hardware/reBot_B601_DM/`:
```
3D_Printed_Parts/   Assembly_Steps/   Metal_Parts/   Purchased_Parts/
performance_testing/  reBot_B601_DM_v1.1_20260425.step (34 MB full assembly)
readme.md  readme_zh/jp/fr/es.md
```
`Metal_Parts/` (link parts):
```
02_Arm_Yaw_Limit.step  02_Base_Motor_Shim.step  02_Base_Reinforcement_Part.step
02_FLANGE.step  02_Gear_Connector.step  02_Gripper_Connector_A.step
02_Gripper_Connector_B.step  02_Lower_Upper_Link_L.step  02_Lower_Upper_Link_R.step
02_Lower_Wrist_Link_L.step  02_Lower_Wrist_Link_R.step  02_Motor_Back_Spacer.step
02_Motor_Front_Spacer.step  02_Rack.step  02_Slider_Bracket.step
02_Slider_Extension.step  02_Wrist_Bracket.step  02-RACK.step
03_Link1.step  03_Link2.step  03_Link3_L.step  03_Link3_R.step  03_Link5.step
03-Link1.step  03-Link2.step  03-Link3_L.step  03-Link3_R.step  03-Link5.step
images/
```
`3D_Printed_Parts/` (covers, fingers, camera mounts):
```
01_Arm_Handle.step  01_BASE_Link.step  01_BASE_Plate.step  01_Finger.step
01_Joint5_Cable Restraint_A.step  01_Joint6_7_Cable Restraint_A/B.step
01_Lower_Arm_Cover.step  01_Lower_Arm_Filler_L/M/R.step  01_Lower_Arm_Limit.step
01_Motor_Cover.step  01_Upper_Arm_Cover.step  01_Upper_Arm_Fuller_L/M/R.step
01_Upper_Arm_Limit.step  01-Rail-Bracket.step
D405_305_Mount.step  D435_Gemini2_Mount.step  UVC32_mount.step
DM-power-Bottom Cover.STEP  DM-power-Top Cover(-Sliding Cover).STEP
Soft_Gripper_Finger.step  Soft_Gripper_Mount.step
```

### Motors (BOM) — `readme.md`
```
131: | Brushless motor | DM4310(V4)  | 4 | 120 $/unit |
132: | Brushless motor | DM4340P(V4) | 3 | 175 $/unit |
21:  ... Fixed Joint 1 model from 4310 to 4340P. Added Cable Restraint for the 3 end joint motors ...
```

### IK envelope — `tools/artifacts/ik_envelope_b601dm.csv`
```
1: x,y,z,pitch,yaw,ok,err
2: 0.25,-0.2,0.08,0.0,-0.6,0,0.1021660169175075
...
feasible (ok==1): 4976/6125 rows
 x∈[0.25,0.55]  y∈[-0.20,0.20]  z∈[0.08,0.25]
 pitch∈{0,0.225,0.45,0.675,0.9,1.2,1.57}  yaw∈{-0.6,-0.3,0,0.3,0.6}
```

### Gripper constants — `rebot_arm.py`
```
145: _G_MAX_DIST_M      = 0.09
146: _G_ANGLE_OPEN      = -5.0
147: _G_OPEN_SOFT_LIMIT = -4.9
150: _G_TAU_MAX         = 1.5
154: _G_CLOSE_TORQUE    = 1.0
160: _G_DEFAULT_FORCE   = 0.30
170: ... soft-limit shortfall ... ~0.26 rad (0.0853m measured vs 0.09m commanded) ...
515: return float(np.clip(self._g_pos / _G_ANGLE_OPEN, 0.0, 1.0) * _G_MAX_DIST_M)
```
Jaw width gate — `perception/ordinary_grasp.py`:
```
161: if top is not None and top[3] > 0.085:
165:     (c for c in side_cands if c[3] <= 0.085),
```

### FK/IK delegation to SDK (no app-level kinematics) — `rebot_arm.py`
```
249-258: from reBotArm_control_py.kinematics import (IKSolverParams, compute_fk,
         get_end_effector_frame_id, load_robot_model, pos_rot_to_se3)
         from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik
280-286: self._model = load_robot_model(urdf_path=...) / load_robot_model()
         self._ee_frame_id = get_end_effector_frame_id(self._model)
882-890: get_tcp_pose: q,_,_ = self._arm.get_state(); compute_fk(self._model, q)
918-928: move_to → self._endpos_ctrl.move_to_traj(x,y,z,roll,pitch,yaw,duration)
```
SDK location — `rebot_arm.py:42` `/opt/rebot`; `config.yaml:267`
`repo_root: ${REBOT_REPO_ROOT:-/opt/rebot}`.

### TCP / approach convention — `transforms.py`
```
144: tcp_x = -approach   # tool-forward = approach (into object)
145-146: tcp_y = open_vec ...      # jaw open/close direction
147:     tcp_z = np.cross(tcp_x, tcp_y)
69-71:  ZYX intrinsic: R = Rz @ Ry @ Rx
175-178: _offset_along_tool_x: T[:3,3] - T[:3,0]*offset  (along tool-X)
```

### Hand-eye / camera — `config.yaml`, `grasp_service.py`, `grasp_plugin.py`
```
config.yaml:305-309: camera type: orbbec_gemini2, 1280x720@30
config.yaml:311-313: hand_eye_path ${REBOT_HAND_EYE:-/opt/rebot-models/hand_eye.npz}  (TSAI, 16 samples, key T_result/first)
grasp_service.py:8:   T_cam2base = arm.get_tcp_pose() @ T_hand_eye
grasp_service.py:378-379: tcp = arm.get_tcp_pose(); r_cam2base = (tcp @ T_hand_eye)[:3,:3]
grasp_plugin.py:977: key = "T_hand_eye" if "T_hand_eye" in data else data.files[0]
```

### Deliverable path
`/Users/harvest/project/seeed-local-voice/docs/sim/rebot_b601dm_kinematics.md`
