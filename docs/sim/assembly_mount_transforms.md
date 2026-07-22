# reBot B601-DM — Assembly-Level Mount Transforms (OCCT/XCAF)

**Purpose:** Replace two URDF placeholders that the per-part CAD work could not give
(they are *relative poses between assembled parts*, not single-part dims):

1. **`gripper_base_joint` origin** (`end_link → gripper_base`) — currently identity
   placeholder in `reBot-DevArm_gripper.urdf` (`gripper_urdf_notes.md` §3).
2. **D405 camera mount pose** relative to the flange — a CAD hand-eye reference.

**Method:** OCCT assembly-aware reader (XCAF) via `cadquery-ocp`, run ephemerally
(`uv run --with cadquery python ...`, no pyproject change). `STEPCAFControl_Reader`
+ `XCAFDoc_ShapeTool` walk the assembly tree, reading each component's STEP
product **name** (`TDataStd_Name`) and its **`TopLoc_Location` → `gp_Trsf`** world
placement. (Plain `STEPControl_Reader` was NOT used — it fuses everything to one
shape and loses names/poses.)

- Source: `reBot-DevArm/hardware/reBot_B601_DM/reBot_B601_DM_v1.1_20260425.step`
  (read-only, 34 MB full assembly).
- **XCAF parse proof:** `ReadFile → IFSelect_RetDone`, `Transfer ok: True`,
  `Free shapes: 1`, **326 leaf components** extracted with names + world transforms
  (raw tree below).

---

## 0. Key findings up front

| Finding | Value |
|---|---|
| Flange→gripper_base (housing) | **xyz = (0, 0, 0) m, rpy = (0, 0, 0)** — housing bolts directly to the J6 flange face; axis-aligned with the tool frame. |
| Flange→finger-contact (grasp TCP) | **xyz = (0.1278, 0, 0) m, rpy = (0, 0, 0)** — 127.8 mm out along approach (+X tool). |
| Flange→finger tip (full reach) | xyz = (0.1602, 0, 0) m (160.2 mm). |
| Jaw axis | world **±Y** → tool **+Y** (matches SDK TCP convention). |
| Approach axis | gripper points world **−X**; mapped to tool **+X** (into object). |
| **D405 camera mount** | **NOT PRESENT in this assembly STEP** — no `D405_305_Mount` / `D435` / `UVC` / camera part in the 326-component tree. Camera pose below is a **datasheet estimate, flagged APPROX**, NOT a CAD measurement. Use `hand_eye.npz` for ground truth. |
| Assembly joint pose | **NOT zero config** — arm is posed folded back (flange at world −X); see §5. |

---

## 1. Assembly component tree (identified parts)

The 326-leaf tree is dominated by fasteners (`KM3-*`, `KA3-*`, `HM3-*`, `PIN-*`,
bearings `6803ZZ`/`6707ZZ`, motors `DM-J4310`/`DM-J4340P`). The **structural /
end-effector parts**, with bounding-box centroids in **assembly world coords (mm)**
and their grouping subassembly path:

| Part (STEP name) | Subasm path | centroid world (mm) | role |
|---|---|---|---|
| `01_BASE_Plate` | (root) | (0.0, 0.0, 10.5) | table base plate |
| `01_BASE_Link` | (root) | (0.0, 0.0, 49.2) | base link |
| `03_Link1` | Link1 | (−6.6, 0.0, 129.9) | J1 link |
| `03_Link2` ×2 | (root) | (112.0, ∓21.4, 143.5) | upper arm (J2) |
| `03_Link3_L/R` | (root) | (121.8, ±21, 197.5) | forearm (J3, forked) |
| `02_Lower_Upper_Link_L/R` | (root) | (228.4, ∓38, 170.5) | J3→J4 |
| `02_Lower_Wrist_Link_L/R` | (root) | (−22.5, ±31.9, 211.0) | J4→J5 |
| `02_Wrist_Bracket` | Link4 | (−66.1, 0.0, 239.0) | wrist bracket (J5) |
| **`03_Link5`** | Link5 | **(−79.1, 0.0, 203.0)** | **J6 / wrist-roll link = URDF `link6`+flange region** |
| **`02_Gripper_Connector_B`** | Link6 | **(−104.85, 0.0, 195.0)** | **gripper housing flange plate (56.8×56.8×9.5 mm) — `gripper_base`** |
| **`02_Gripper_Connector_A`** | Link6 | **(−103.6, 0.0, 194.4)** | rack/pinion mount plate (35.6×6×34.4 mm) |
| `02_Gear_Connector` | Gripper | (−162.85, 0.0, 195.0) | pinion |
| `02_Slider_Bracket` | Gripper | (−158.6, 0.0, 195.0) | slider bracket |
| `02_Rack` ×2 | Gripper | (−174.7, ±13.6, ~195) | rack (one per finger) |
| `02_Slider_Extension` ×2 | Gripper | (−186.6, ±18.2, 195.0) | slider extensions |
| `RAIL-170` / `01_Rail_Bracket` | Gripper | (−183/−172, 0, 195) | linear rail |
| `GEAR-1M16C` / `SLIDER` ×2 | Gripper | (−170/−186, ±20, 195) | pinion gear / sliders |
| **`01_Finger`** ×2 | Gripper | **(−227.9, ∓15.66, 195.0)** | **3D-printed jaw fingers** |

> The three `02_FLANGE` instances in the tree (centroids near the **motors** at
> X=−20 / 244 / 1.4, Z=143–197) are **motor mounting flanges, NOT the end-effector
> flange**. The end-effector flange in this assembly is the `03_Link5` (J6) part;
> the gripper housing (`02_Gripper_Connector_B`) bolts to its distal face. This
> matches `rebot_b601dm_kinematics.md` §1 (link6 ← `03_Link5.step` + `02_FLANGE.step`).

### Raw XCAF evidence — world AABBs of the end-effector parts (mm)

```
02_Gripper_Connector_B   X[-109.60,-100.10] Y[-28.40, 28.40] Z[166.60,223.40]  size=(9.5,56.8,56.8)
02_Gripper_Connector_A   X[-106.61,-100.59] Y[-17.81, 17.81] Z[177.19,211.62]  size=(6.0,35.6,34.4)
03_Link5                 X[-109.10, -49.10] Y[-25.00, 25.00] Z[173.00,233.00]  size=(60.0,50.0,60.0)
02_Gear_Connector        X[-167.60,-158.10] Y[-23.91, 23.91] Z[171.09,218.91]  size=(9.5,47.8,47.8)
01_Finger  (A)           X[-260.32,-195.49] Y[-31.36,  0.03] Z[175.40,214.63]  size=(64.8,31.4,39.2)
01_Finger  (B)           X[-260.32,-195.49] Y[ -0.03, 31.36] Z[175.37,214.60]  size=(64.8,31.4,39.2)
```

These prove the kernel parsed real BREP per-part (distinct positioned AABBs), not a
fused soup.

---

## 2. Flange → gripper transform

### Frame / axis convention used (stated explicitly)

The assembly is laid out with the gripper pointing in **world −X**, jaw spread in
**world ±Y**, parts centered at **world Z≈195 mm**. I map the world axes into the
URDF `end_link` **tool convention** (`rebot_b601dm_kinematics.md` §5: tool +X =
approach *into* the object, +Y = jaw open/close, +Z RH):

```
tool +X (approach, into object) = world −X      (fingers extend toward −X)
tool +Y (jaw open/close)        = world +Y      (fingers separated along ±Y)
tool +Z (= X × Y, RH)           = world −Z
```

The **flange (end_link) origin** is anchored at the **J6↔housing mating face**, i.e.
the `02_Gripper_Connector_B` +X face at **world X = −100.10 mm, Y = 0, Z = 195.0**
(where `03_Link5` distal face X=−109.1 meets the housing plate). The housing,
pinion, rack, rail and both fingers are all centered on **Y=0, Z≈195** → the whole
gripper sub-chain is coaxial with the approach axis with **no roll/pitch/yaw offset**
in the tool frame.

### Computed transforms (raw)

```
flange(end_link) origin, world (mm):  [-100.10, 0.0, 195.0]
axis map: tool+X=world-X ; tool+Y=world+Y ; tool+Z=world-Z

flange -> gripper_base (housing face):  xyz_m = [0.0,    0.0, 0.0]   rpy = [0,0,0]
flange -> gripper_base (housing centroid): xyz_m = [-0.0047, 0.0, 0.0] rpy=[0,0,0]   (centroid is 4.75mm inboard of mating face — negligible)
flange -> finger-contact pad-center:    xyz_m = [0.1278, 0.0, 0.0]   rpy = [0,0,0]
flange -> finger TIP (full reach):      xyz_m = [0.1602, 0.0, 0.0]
approach-axis dist flange_face -> grasp pad-center = 127.8 mm
approach-axis dist flange_face -> finger tip       = 160.2 mm
```

### Concrete URDF value — `gripper_base_joint` `<origin>`

`gripper_base` is fixed to `end_link`, housing bolts directly to the flange face,
axis-aligned with the tool frame:

```xml
<!-- end_link -> gripper_base : housing bolts to flange face, no rotation -->
<origin xyz="0 0 0" rpy="0 0 0"/>
```

> i.e. the **current identity placeholder is geometrically correct** for the joint
> *origin* — the housing mating plane coincides with the `end_link` flange frame and
> the gripper axes already equal the tool axes. (Confidence: HIGH on rpy=0 and Y=Z=0;
> the X-origin depends on exactly where the SDK defines `end_link` along the flange
> stack — if the SDK end_link sits at the *outer* flange plate face rather than the
> J6-mate face, add up to +9.5 mm along +X. The 9.5 mm housing-plate thickness is the
> only ambiguity; treat `gripper_base_joint` xyz as `0 0 0` with ±0.0095 m X
> uncertainty.)

### Grasp-point (finger-contact TCP) offset

The grasp/TCP point — where the two finger inner pads meet at closed, on the jaw
midline — sits **127.8 mm out along tool +X** from the flange face (Y=Z=0):

```
grasp-point offset from end_link:  xyz = (0.1278, 0, 0) m,  rpy = (0, 0, 0)
```

Use this as the **TCP offset** the grasp pipeline should target relative to the SDK
flange (cf. `insertion_depth_m=0.025`, `pregrasp_offset` applied along tool-X in
`transforms.py`). The finger tip (full physical reach) is 160.2 mm out; the
**pad-center grasp line is 127.8 mm** (mid-blade raised contact band, per
`gripper_cad_measurements.md` §1). The fingers are modeled at world Y=±15.66 mm in
this assembly (≈31 mm jaw opening as posed); inner edges reach Y=±0.03 (≈closed).

> Assumption: pad-center is taken at the finger centroid X (the raised contact band
> sits mid-blade). If you want the *distal* pad edge, the grip line shifts up to the
> 160 mm tip. 127.8 mm is the conservative pad-center; flag ±20 mm along X depending
> on which pad row contacts.

---

## 3. D405 camera mount pose relative to flange — **APPROX, no CAD in assembly**

⚠️ **`D405_305_Mount.step` is NOT in this assembly STEP.** A full name search over
all 326 components for `d40* / d43* / cam* / mount / orbbec / gemini / realsen / uvc`
returned **zero hits**. The wrist camera + its mount are simply not modeled in
`reBot_B601_DM_v1.1_20260425.step`. So there is **no CAD-derived camera pose** to
report from the assembly.

What can be stated (datasheet estimate only — **DO NOT treat as calibrated**):

- The D405 is wrist-mounted (eye-in-hand) via `D405_305_Mount` bolting to the
  J5/J6 wrist region (`rebot_b601dm_kinematics.md` §6). Without the mount part
  positioned in the assembly, its 6-DoF pose cannot be measured here.
- **Estimated optical-frame pose (PLACEHOLDER, flag as APPROX):** a typical D405
  eye-in-hand mount places the optical center forward of and slightly above the
  flange, looking along the approach axis. Datasheet: the D405 RGB/depth optical
  origin sits **~mm behind the front glass**; the module front face is ~23 mm deep.
  A reasonable seed, pending real calibration:
  ```
  flange -> D405 optical (APPROX, NOT measured):  xyz ≈ (0.04, 0.0, -0.05) m
  rpy ≈ (0, 0, 0)  (optical Z roughly along tool +X approach; sign/convention TBD)
  ```
  These numbers are **fabricated from the datasheet + typical wrist-mount geometry,
  not from the STEP** — they exist only so the URDF camera link is non-degenerate.

- **Ground truth source (use this instead):** `hand_eye.npz` key `T_hand_eye`
  (TSAI, 16 samples) at `/opt/rebot-models/hand_eye.npz`
  (`rebot_b601dm_kinematics.md` §6, `grasp_service.py:8`). That is the
  TCP→camera extrinsic the URDF camera link must encode. The CAD here **cannot
  substitute for it** because the mount is absent from the model.

**Recommendation:** pull `hand_eye.npz` off the device for the camera link; do not
use the §3 placeholder beyond bootstrapping a visualization.

---

## 4. Cross-check — assembly vs URDF link origins

URDF forward kinematics at **zero joint config** (chain `joint1…joint6` + `end_joint`
xyz=(0,0,0.15539) rpy=(0,−π/2,π) from `reBot-DevArm_fixend.urdf`) places `end_link`
at:

```
URDF FK end_link @ zero config (world m):  [0.2603, 0.0, 0.1917]   (= 260.3, 0, 191.7 mm)
```

The assembly puts the flange (housing mate face) at:

```
assembly flange world (mm):  [-100.1, 0.0, 195.0]
```

**Agreement / discrepancy:**

- **Z (height) agrees: 191.7 mm (URDF) vs 195.0 mm (assembly)** — within ~3 mm.
  This confirms the **kinematic scale and the flange height are consistent** between
  the URDF and the physical CAD.
- **Y agrees: 0 vs 0** — both symmetric about the sagittal plane.
- **X differs in sign: +260 (URDF zero) vs −100 (assembly).** This is **NOT a
  discrepancy** — it means the **assembly is modeled in a non-zero, folded joint
  pose** (the wrist is rotated back over the base toward −X), whereas the URDF FK
  above is the fully-extended zero config. The base/J1 stack agrees too
  (`01_BASE_Link` centroid Z=49 mm, URDF joint1 at Z=84.65 mm — same base height
  band).

**Modeled joint configuration (inferred):** the arm is posed with the forearm/elbow
reaching out to ~+X 230 mm then the wrist folding the end-effector back to −X
(`02_Lower_Wrist_Link` at X=−22, `03_Link5` at X=−79, gripper at X=−100…−260). It is
a "tucked / inspecting-its-own-base" pose, not the URDF rest pose. **This does not
affect the §2 flange→gripper transforms**, which are intrinsic rigid offsets between
the flange and the gripper parts (pose-independent — they ride together regardless of
joint angles).

---

## 5. Caveats

1. **Camera mount absent** — biggest gap. No `D405_305_Mount` in the STEP; the §3
   camera pose is a datasheet placeholder, not a measurement. Use `hand_eye.npz`.
2. **end_link X-origin ±9.5 mm** — `gripper_base_joint` xyz `0 0 0` assumes the SDK
   `end_link` frame sits at the J6↔housing mate face. If it sits at the outer housing
   plate face, add +0.0095 m along tool +X. rpy=0 and Y=Z=0 are unambiguous.
3. **Grasp-point X ±20 mm** — 127.8 mm is the pad-center; the tip is 160.2 mm. Pick
   the pad row that physically contacts your target object.
4. **Assembly is folded** (non-zero joint pose) — fine for relative transforms;
   do NOT read absolute link world positions as URDF rest origins.
5. **Axis mapping verified by geometry** (fingers along ±Y, approach along −X,
   Z=const), consistent with the SDK TCP convention; but the absolute world→tool
   sign of tool +Z was set by the RH rule (tool+Z=world−Z). If a downstream consumer
   expects tool+Z=world+Z, flip with a 180° roll about tool-X (grasp-equivalent per
   `canonicalize_parallel_gripper_tcp_rotation`).

---

## 6. Concrete values to apply (after review — URDF NOT edited here)

```xml
<!-- reBot-DevArm_gripper.urdf : gripper_base_joint (end_link -> gripper_base) -->
<origin xyz="0 0 0" rpy="0 0 0"/>
<!-- HIGH confidence: housing bolts to flange face, axis-aligned.
     X-origin ±0.0095 m depending on exact SDK end_link definition. -->
```

- **Grasp/TCP offset** (flange → finger-contact, for the grasp pipeline, NOT a URDF
  joint): `xyz = (0.1278, 0, 0) m, rpy = (0, 0, 0)`.
- **Camera link**: use `hand_eye.npz` `T_hand_eye`; the §3 `(0.04, 0, -0.05)`
  placeholder is APPROX-only (mount absent from CAD).

---

## EVIDENCE

**Command:** `uv run --with cadquery python /tmp/asm_parse.py` (and `asm_transforms.py`,
`asm_flange.py`, `asm_final.py`).

**XCAF parse proof:**
```
ReadFile status: IFSelect_ReturnStatus.IFSelect_RetDone
Transfer ok: True
Free shapes: 1
=== TOTAL LEAF COMPONENTS: 326 ===
```

**Sample of named components with world centroids (mm) — raw:**
```
03_Link5                 c=(  -79.1,   -0.0,  203.0)   origin=(-16.4, 0, 265.7)  R=[[0,0,1],[1,0,0],[0,1,0]]
02_Gripper_Connector_B   c=( -104.85,  -0.0,  195.0)   origin=(-109.6,0,195.0)   R=[[0,0,1],[1,0,0],[0,1,0]]
02_Gripper_Connector_A   c=( -103.6,    0.0,  194.4)
02_Gear_Connector        c=( -162.85,  -0.0,  195.0)
01_Finger                c=( -227.90, -15.66, 195.01)
01_Finger                c=( -227.90,  15.66, 194.99)
02_Wrist_Bracket         c=(  -66.06,  -0.0,  239.0)
01_BASE_Link             c=(    0.0,   -0.0,   49.2)
01_BASE_Plate            c=(    0.0,   -0.0,   10.5)
```
(Full 326-row tree printed by `asm_parse.py` — fasteners KM3-*/KA3-*/HM3-*/PIN-*,
motors DM-J4310/DM-J4340P, bearings 6803ZZ/6707ZZ omitted from this excerpt.)

**Computed transforms (raw):**
```
flange(end_link) anchor world (mm): [-100.10, 0.0, 195.0]
axis map: tool+X=world-X ; tool+Y=world+Y ; tool+Z=world-Z
flange -> gripper_base (housing face):  xyz_m=[0,0,0]      rpy=[0,0,0]
flange -> finger-contact (grasp TCP):   xyz_m=[0.1278,0,0] rpy=[0,0,0]   (127.8 mm)
flange -> finger tip:                   xyz_m=[0.16022,0,0]               (160.2 mm)
jaw axis world ±Y -> tool +Y ; approach world -X -> tool +X
camera mount search over 326 names: ZERO hits (no D405/D435/UVC/mount/cam)
URDF FK end_link @ zero config: [0.2603, 0.0, 0.1917] m  (Z agrees 191.7 vs 195.0; X folded)
```

**Markdown path:** `/Users/harvest/project/seeed-local-voice/docs/sim/assembly_mount_transforms.md`
