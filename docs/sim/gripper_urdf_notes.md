# reBot B601-DM Gripper URDF — Authoring Notes

**File produced:** `sim/rebot_b601dm_urdf/urdf/reBot-DevArm_gripper.urdf`
**Source:** copy of the pristine SDK rigid URDF
`sim/rebot_b601dm_urdf/urdf/reBot-DevArm_fixend.urdf` (6-DOF, `end_joint`
`fixed` joining `link6` → `end_link` flange), with an actuated parallel-jaw
gripper appended. The pristine fixend URDF and the meshes were **not modified**.

Validated with `yourdfpy` (ephemeral `uv run --with yourdfpy`, no pyproject
change — there is no root `pyproject.toml`, and adding the dep to `agent/` would
pollute production deps, so an ephemeral env was used instead). Raw parser
output is in the task EVIDENCE section.

---

## 1. What was added

### Links (3 new)
| Link | Role | Collision geometry | Mass (APPROX) |
|------|------|--------------------|---------------|
| `gripper_base` | rack/pinion housing bolted to flange | box `0.06 × 0.05 × 0.04 m` | 0.15 kg |
| `left_finger`  | left jaw + contact pad | box `0.05 × 0.012 × 0.024 m` (X=length along approach, Y=jaw thickness, Z=pad width) | 0.08 kg |
| `right_finger` | right jaw + contact pad (mirror) | box `0.05 × 0.012 × 0.024 m` | 0.08 kg |

All three links carry `<inertial>` (rough box-inertia tensors) so Isaac/PhysX
doesn't choke on zero-mass links.

### Joints (3 new)
| Joint | Type | Parent → Child | Axis | Limits | dynamics |
|-------|------|----------------|------|--------|----------|
| `gripper_base_joint` | fixed | `end_link` → `gripper_base` | — | — | — |
| `left_finger_joint`  | prismatic | `gripper_base` → `left_finger` | `0 -1 0` (−Y) | lower=0, upper=0.0425, effort=30, velocity=0.1 | damping=5.0, friction=1.0 |
| `right_finger_joint` | prismatic | `gripper_base` → `right_finger` | `0 1 0` (+Y) | lower=0, upper=0.0425, effort=30, velocity=0.1 | damping=5.0, friction=1.0 |

**Jaw span:** each finger travels 0 → 0.0425 m, symmetric about center →
**total jaw span = 0.085 m**, matching the SDK "usable jaw limit"
(`perception/ordinary_grasp.py:161`, `rebot_b601dm_kinematics.md §4`).

### Attachment frame
`gripper_base` is fixed to **`end_link`** (the SDK flange that bears the TCP),
not `link6`, because `end_link` is the frame the SDK FK reports the
end-effector at. The gripper-base local frame is defined so **+X = approach**
and **+Y = jaw open/close**, per the SDK TCP convention
(`rebot_b601dm_kinematics.md §5`: tool +X = approach into object, tool +Y =
jaw axis, +Z RH).

---

## 2. Parallel-jaw modeling choice: two independent prismatic joints

Chosen: **two independent prismatic joints** (`left_finger_joint`,
`right_finger_joint`), each 0 → 0.0425 m, to be **driven together by a coupled
position drive applied Isaac-side** — rather than one prismatic + a `<mimic>`
joint.

Rationale:
- Isaac Sim / PhysX *does* support URDF `<mimic>`, but mimic coupling is
  historically the more fragile import path (it is converted to a PhysX
  articulation tendon / coupled drive and has had import-version quirks).
- Two independent joints import as plain articulation DOFs every time; the
  symmetric coupling is trivially re-imposed in the USD/Isaac layer by sending
  the same target to both drives (or wiring a PhysX coupled joint / mimic in
  USD). This is the more robust, predictable path for a grasp sim.
- The geometry is already symmetric (axes `−Y` / `+Y`, same limits), so equal
  targets give a centered, symmetric jaw.

If you prefer a single-DOF URDF, replace `right_finger_joint` with a `<mimic
joint="left_finger_joint" multiplier="1.0"/>` (axis `0 1 0`) — kinematically
equivalent.

---

## 3. APPROX values — refine against CAD / real arm

Every number below is an approximation. **Flag list for refinement:**

| Value | Current (APPROX) | Source / basis | Refine against |
|-------|------------------|----------------|----------------|
| Finger length (along approach, X) | 0.05 m | `01_Finger.step` bbox Z-span ≈ 65 mm (includes mount features → trimmed to 50 mm pad) | CAD `01_Finger.step` |
| Finger jaw thickness (Y) | 0.012 m | `01_Finger.step` bbox X-span ≈ 31 mm (mostly mount) → pad ≈ 12 mm | CAD |
| Finger pad width (Z) | 0.024 m | `01_Finger.step` bbox Y-span ≈ 39 mm → pad ≈ 24 mm | CAD |
| Finger inner-face offset from center at closed | ±0.006 m | half jaw-thickness | CAD / real arm |
| Finger mass | 0.08 kg each | task-suggested 50–150 g range (ABS print + slider) | CAD density / weigh real finger |
| `gripper_base` box | 0.06 × 0.05 × 0.04 m | `02_Gripper_Connector_A.step` bbox ≈ 75 × 46 × 73 mm, trimmed | CAD |
| `gripper_base` mass | 0.15 kg | guess (metal connectors + motor housing) | CAD / weigh |
| `gripper_base` COM / placement | at +0.02 X from flange, rpy=0 0 0 | identity placeholder | **SDK TCP frame** (not on this Mac) |
| `gripper_base_joint` origin xyz/rpy | `0 0 0` / `0 0 0` | placeholder identity to end_link | **SDK URDF TCP frame** — exact rotation between `end_link` local axes and the SDK tool frame is unknown locally |
| Per-finger travel upper | 0.0425 m | = 0.085/2 usable jaw limit (`ordinary_grasp.py:161`) | matches code; OK but confirm full-open ≈ 0.0853 m real (`rebot_arm.py:170`) |
| Prismatic `effort` | 30 N | crude `1.5 N·m / ~0.05 m pinion-radius est` → ~30 N; the SDK commands **torque (N·m)** not linear force | rack pinion radius from CAD `02_Rack.step` pitch + measure real finger force |
| Prismatic `velocity` | 0.1 m/s | placeholder | real arm jaw speed |
| Prismatic `damping` / `friction` | 5.0 / 1.0 | placeholder | tune in sim |
| Inertia tensors (all 3 links) | box approximations | derived from box dims + mass | CAD inertia or measured |

CAD bbox measurements were extracted by parsing `CARTESIAN_POINT` coordinates
out of the STEP files (no CAD kernel available on this Mac) — they bound the
*whole part including mounting tabs*, so the pad dims were trimmed down by hand.
Treat as rough.

---

## 4. Mesh path handling

The arm-link mesh refs were **kept verbatim** as
`package://reBot-DevArm_description_fixend/meshes/<link>.STL` (unchanged from
the pristine URDF). The new gripper links use **primitive `<box>` geometry**, so
they need no mesh files.

For validation, the parser was given a `filename_handler` that maps the
`package://…/meshes/<file>` ref onto the local
`sim/rebot_b601dm_urdf/meshes/` directory. For Isaac Sim import, either:
- register the ROS package `reBot-DevArm_description_fixend` → `meshes/` parent,
  or
- run the URDF importer with the meshes dir on its search path.

(The relative-rewrite option `../meshes/<file>.STL` was considered but not used,
to keep the new URDF byte-consistent with the pristine arm mesh refs.)

---

## 5. Isaac-side physics TODO (applied in USD/Isaac, NOT in this URDF)

These belong in the USD stage / Isaac importer settings, not the URDF:

1. **Convex collision decomposition** of the arm-link *visual* STL meshes
   (base_link..end_link). The STLs are detailed visual meshes; for stable
   contact, run convex-decomposition (e.g. CoACD / V-HACD via the URDF importer
   "Convex Decomposition" collision approximation) on each arm link. The gripper
   links already use box collisions (cheap + stable) so they need no
   decomposition.
2. **PhysX friction / contact material on the finger pads** — assign a
   high-friction physics material (static/dynamic μ ≈ 1.0–1.5, low restitution)
   to the `left_finger` / `right_finger` pad collision faces so objects don't
   slip out of the grasp. The arm links can keep a default lower-friction
   material.
3. **Joint drive gains** — set PhysX position-drive `stiffness` / `damping` on
   the two prismatic finger drives for stable position control of jaw width
   (and on the 6 arm revolute drives). The URDF `<dynamics damping/friction>`
   are only seed values; real control gains live on the USD drive.
4. **Coupled jaw drive** — impose the symmetric coupling between
   `left_finger_joint` and `right_finger_joint` (equal position targets, or a
   PhysX mimic/coupled joint) so the jaw stays centered.
5. **`base_link` fixed to world** — add a fixed joint (or `Articulation Root` +
   fixed base) pinning `base_link` to the world so the arm is grounded.
6. **N·m grasp-force → PhysX finger drive force mapping** — the SDK commands the
   gripper motor in **torque (N·m)** (ceiling 1.5, default 0.8, close 1.0;
   `rebot_arm.py:150/154/160`). To reproduce grasp force in sim, convert that
   motor torque to a linear finger drive force via the rack pinion radius
   `F = τ / r` (measure `r` from `02_Rack.step` pinion pitch), then either cap
   the prismatic drive `maxForce` or apply it as a closing force target. The
   URDF `effort=30 N` is a placeholder for this — refine once `r` is known.

---

## 6. Mapping the gripper joint to the runtime gripper command

The runtime gripper command (`rebot_actuator.py:14-19`,
`rebot_b601dm_kinematics.md §4`) is a **signed magnitude** on a single
`gripper` frame field:

| Command `g` | Meaning | Sim mapping |
|-------------|---------|-------------|
| `g > 0` | **open** to `g` metres (jaw width) | set each finger prismatic target to `g/2` (clamped to [0, 0.0425]); total span = `g` |
| `g < 0` | **grasp** at `|g|` N·m motor torque | drive both fingers *closed* (toward 0) with drive force = `|g| / r` (rack pinion radius), i.e. a torque-mode close |
| `g = 0` | **hold** current state | freeze both finger drive targets at current position |

So:
- The prismatic joint **position** corresponds to **half the open width**
  (`finger_pos = open_width / 2`), matching the SDK linear map
  `pos/-5.0 rad → 0..0.09 m` (`rebot_arm.py:515`) — the URDF's per-finger upper
  0.0425 m gives full usable open width 0.085 m.
- The prismatic joint **drive force** (when closing) corresponds to the SDK
  **N·m torque** command via the pinion radius (see §5 item 6).
- Full mechanical open is ~0.09 m commanded but ~0.0853 m real
  (`rebot_arm.py:170`); the URDF caps at the 0.085 m *usable* limit, which is
  what the grasp pipeline trusts.

---

## 7. Validation summary

Parsed cleanly with `yourdfpy`. Confirmed:
- 11 links, 10 joints; tree is connected
  `base_link → link1…link6 → end_link → gripper_base → {left,right}_finger`.
- `left_finger_joint` / `right_finger_joint` are `prismatic`, lower=0,
  upper=0.0425; `gripper_base_joint` is `fixed`.
- All arm mesh `filename=` refs resolve (scene built with 11 geometries).

Raw joint/link dump is in the task EVIDENCE section.
