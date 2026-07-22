# reBot B601-DM Gripper — CAD-Kernel Measurements

**Purpose:** Replace the hand-trimmed `CARTESIAN_POINT`-scraped approximations in
`gripper_urdf_notes.md` §3 with real BREP measurements from the STEP files, using
a true OCCT CAD kernel (CadQuery / `cadquery-ocp`), run ephemerally on macOS via
`uv run --with cadquery python ...` (no pyproject changes).

All raw numbers are in **mm as stored in the STEP**; URDF-facing values converted
to **m**.

---

## 0. Kernel / provenance

- Kernel: `cadquery-ocp` (OpenCASCADE 7.x Python bindings), installed ephemerally:
  `uv run --with cadquery python cad_measure.py`
- Every part loaded as a single valid solid (proof below): `STEPControl_Reader`
  → `TransferRoots` → `OneShape`, then BREP face/solid traversal + `Bnd_Box`
  bounding boxes + `BRepAdaptor_Surface` cylinder/cone radius extraction.
- Mesh export: `BRepMesh_IncrementalMesh(lin=0.3mm, ang=0.5rad)` → binary STL.

Solid/face counts (proof the kernel parsed real BREP, not point soup):

| Part | solids | faces |
|------|--------|-------|
| `01_Finger.step` | 1 | 69 |
| `02_Rack.step` | 1 | 226 |
| `02_Gear_Connector.step` | 1 | 39 |
| `02_Gripper_Connector_A.step` | 1 | 248 |
| `02_Gripper_Connector_B.step` | 1 | 61 |
| `02_FLANGE.step` | 1 | 31 |

---

## 1. Finger (`01_Finger.step`)

Bounding box (mm): `X[-17.36, 14.03]  Y[-19.01, 20.22]  Z[-0.01, 64.82]`
→ **dx = 31.39, dy = 39.23, dz = 64.82 mm**

Axis interpretation (from the large planar-face normals):

| STEP axis | dim (mm) | meaning | evidence |
|-----------|----------|---------|----------|
| **Z** | **64.82** | finger **length along approach** | longest extent; finger blade runs +Z |
| **X** | **31.39** | jaw **open/close** direction | the two large `normal=±X` pad faces (area 1446 mm² at X=14.03; area 739 mm² at X=2.29) are the back & inner contact faces of the jaw |
| **Y** | **39.23** | finger / pad **width** | symmetric `normal=±Y` side faces at Y=±19 |

Contact-pad detail (large planar faces, `BRepGProp` area + centroid):
- **Inner contact face** (faces the opposing finger): planar `normal=+X`,
  area **739 mm²**, centroid X=2.29 mm — this is the gripping pad plane.
- Outer back face: planar `normal=+X`, area 1446 mm², centroid X=14.03 mm.
- → **jaw blade thickness (back→contact) ≈ 14.03 − 2.29 = 11.7 mm** in X.
- Pad band in Y: side relief faces at Y≈±10 (88 mm² each) indicate the raised
  contact pad is ~20 mm wide centered, within the 39.2 mm full finger width.

**Recommended URDF finger box** (`left_finger`/`right_finger` collision), using
URDF convention **X=length-along-approach, Y=jaw-thickness, Z=pad-width**:

| URDF axis | measured source | value (m) |
|-----------|-----------------|-----------|
| length (X) | STEP Z full = 64.82 mm (full blade) → usable pad span ~50 mm | **0.050** (pad) or **0.065** (full blade) |
| jaw thickness (Y) | STEP X back→contact = 11.7 mm | **0.012** |
| pad width (Z) | raised pad band ~20–24 mm (full finger 39 mm) | **0.024** |
| inner contact-face offset | STEP X contact plane at +2.3 mm from finger origin | **±0.006** (half thickness; OK) |

> Note: the previous APPROX values (0.05 / 0.012 / 0.024) happen to land close —
> they were hand-trimmed well. The **real full finger length is 64.8 mm** (not 50);
> 50 mm is a reasonable *pad* length, but if you want the full physical blade for
> collision, use **0.065 m** and reference the exported `finger.stl`.

---

## 2. Gripper housing (`02_Gripper_Connector_A/B`) → `gripper_base`

- `02_Gripper_Connector_A.step` bbox (mm): `dx=35.62  dy=6.02  dz=34.43`
  → a thin (6 mm) **35.6 × 34.4 mm plate** (the rack/pinion mounting plate), NOT a
  deep housing. (The earlier note's "75×46×73" was a point-scrape artifact / wrong
  part — the real connector-A plate is 35.6 × 6 × 34.4 mm.)
- `02_Gripper_Connector_B.step` bbox (mm): `dx=56.80  dy=56.80  dz=9.50`
  → a **56.8 × 56.8 mm square flange plate, 9.5 mm thick** (the larger mounting
  flange / cover).

**Recommended `gripper_base` box:** the housing is really two stacked plates
(34–57 mm in the jaw/approach plane, ~6–10 mm thick each, ~16 mm combined stack).
A reasonable collision box bounding the stack:
**0.057 × 0.016 × 0.057 m** (X≈approach-plane, Y≈stack thickness, Z≈jaw-plane),
or keep the simpler **0.06 × 0.05 × 0.04 m** APPROX if a chunkier proxy is fine.
The real part is **flatter** than the APPROX 50 mm Y depth suggested.

---

## 3. Pinion pitch radius `r` (rack-and-pinion) — **the key value for F = τ/r**

Source part: `02_Gear_Connector.step` (the pinion). Gear axis is **Y** (from the
hub cylinder `r=17.5, dir=(0,−1,0)`, axis at X=0, Z=−144.5).

**Distinct cylinder radii in `02_Gear_Connector.step` (mm → count):**
`{1.6: 7, 2.25: 3, 3.95: 1, 17.5: 1}`
- `1.6, 2.25, 3.95` = bolt holes / dowel / shaft bore (mounting features).
- `17.5` = the **gear outer (tip) cylinder** — the OD of the pinion = Ø35 mm.

**Pitch radius derivation (cross-checked two ways):**

1. **From the rack module** (`02_Rack.step`): the rack tooth-tip planar faces lie
   at X-centroids spaced **3.325 mm** (27 tips, 26 gaps, all 3.324–3.325 mm).
   → circular pitch **p = 3.325 mm** → **module m = p/π = 1.058 mm**.

2. **From the pinion tooth count**: the pinion's involute flanks are modeled as
   21 cone faces; tooth tips reach **r = 17.50 mm** (max vertex radius from the
   gear axis). For a standard gear, tip radius = pitch_r + addendum(=m):
   - z = 31 teeth → pitch_r = m·z/2 = **16.41 mm**, tip = 16.41 + 1.06 = **17.46 mm**
     ✓ matches the measured 17.50 mm tip to within 0.04 mm.
   - (z=30 → tip 16.93; z=32 → tip 17.99 — both miss; **z=31 is the fit.**)

> **Chosen pinion pitch radius: r ≈ 16.4 mm = 0.0164 m** (Ø32.8 mm pitch circle,
> module 1.06, z=31, OD Ø35 mm). Confidence: high — the rack-derived module and the
> measured tip radius independently agree on z=31 / pitch_r 16.4 mm.

> ⚠️ Uncertainty flag: this assumes a standard addendum = 1·m. If the pinion uses a
> short/long-addendum profile the pitch radius could shift ±0.5 mm (pitch_r in
> 15.9–16.9 mm). The **17.5 mm OD** is exact; the pitch radius is the OD minus one
> module. The bolt-hole cylinders (1.6/2.25/3.95 mm) are definitively NOT the gear.

**Force conversion for the URDF prismatic `effort`** (F = τ / r):

| τ (N·m) | r (m) | F = τ/r (N) |
|---------|-------|-------------|
| 0.8 (default) | 0.0164 | **48.8 N** |
| 1.0 (close)   | 0.0164 | **61.0 N** |
| 1.5 (ceiling) | 0.0164 | **91.5 N** |

→ **Recommended prismatic `effort` = ~91 N** (use the 1.5 N·m torque ceiling so the
drive can reproduce max grasp force). The current APPROX `effort=30 N` was based on
a guessed r≈0.05 m and is **~3× too low** — the real pinion is smaller (r=0.0164 m),
so the same torque yields a *larger* linear force.

---

## 4. Rack stroke (`02_Rack.step`)

- Rack bbox (mm): `dx=89.999  dy=6.000  dz=12.002` → **rack length = 90.0 mm**,
  6 mm wide, 12 mm tall.
- Toothed region: 27 tooth tips span X **−34.48 → +51.97 mm = 86.45 mm** of toothed
  length (tooth-tip fillet cylinders span 87.1 mm).
- `02_Rack.step` and `02-RACK.step` are the **same part** (identical bbox
  90.0×6.0×12.0; 226 vs 223 faces — trivial rebuild diff).

**Cross-check vs jaw stroke:** the full rack is 90 mm long; the *single rack*
drives one finger, and the jaw is symmetric (two fingers / two rack segments or a
center pinion driving both). The full mechanical open ≈ **85.3 mm** total jaw
(`rebot_arm.py:170`) and usable **85 mm** match the rack's ~86 mm toothed travel
envelope — consistent. The per-finger 0.0425 m (42.5 mm) upper limit is confirmed
(half of 85 mm). **No change to the travel limits — they are correct.**

---

## 5. Exported STL collision meshes

Written to `sim/rebot_b601dm_urdf/meshes/gripper/` (binary STL, lin=0.3 mm):

```
finger.stl           81 KB   (01_Finger.step — real finger blade for collision)
gripper_housing.stl 950 KB   (02_Gripper_Connector_A.step — optional housing mesh)
```

These let the URDF later swap the `left_finger`/`right_finger` `<box>` pads for the
real finger mesh collision. (`gripper_housing.stl` is larger/denser — decimate
further or keep the `gripper_base` as a box; mesh is optional.)

---

## 6. "Replace APPROX with measured" — diff against `gripper_urdf_notes.md` §3

| Value | APPROX (current) | **Measured (CAD)** | Source |
|-------|------------------|--------------------|--------|
| Finger length (X, approach) | 0.05 m | **0.065 m** (full blade) / 0.050 m (pad ok) | `01_Finger.step` Z=64.82 mm |
| Finger jaw thickness (Y) | 0.012 m | **0.012 m** (back→contact = 11.7 mm) ✓ | `01_Finger.step` X faces 14.03→2.29 |
| Finger pad width (Z) | 0.024 m | **0.024 m** (pad band ~20–24 of 39 mm) ✓ | `01_Finger.step` Y side faces |
| Inner-face offset at closed | ±0.006 m | **±0.006 m** ✓ | contact plane at X+2.3 mm |
| `gripper_base` box | 0.06×0.05×0.04 m | **0.057×0.016×0.057 m** (flatter plate stack) | `02_Gripper_Connector_A` 35.6×6.0×34.4 + `_B` 56.8×56.8×9.5 |
| **Pinion radius r** | ~0.05 m (guessed) | **0.0164 m** (Ø35 OD, m=1.06, z=31) | `02_Gear_Connector.step` tip Ø35 − module |
| **Prismatic `effort`** | 30 N | **~91 N** (= 1.5 N·m / 0.0164 m) | F = τ/r with measured r |
| Per-finger travel upper | 0.0425 m | **0.0425 m** ✓ (rack 86 mm toothed ÷2) | `02_Rack.step` 90 mm / 86 mm toothed |
| Rack length / stroke | (not measured) | **90.0 mm** total, **86.5 mm** toothed travel | `02_Rack.step` bbox |

**Key correction:** the pinion is **smaller** than the old guess (r=0.0164 m, not
~0.05 m) → the prismatic `effort` should be **raised to ~91 N**, not 30 N, to
faithfully convert the 1.5 N·m torque ceiling into linear finger force.

---

## 7. Do NOT auto-apply

These are recommendations only. Per the task, the URDF edit is deferred until the
device mount-transform agent also reports, so the gripper-base placement and the
finger dims/effort get applied together in one pass.
