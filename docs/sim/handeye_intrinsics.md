# Real Hand-Eye Calibration + Camera Intrinsics (production reBot arm)

Pulled READ-ONLY from the production arm `seeed-orin-nx`, container `voice-rebot-arm`
(image `voice-rebot-arm:v0.8.0-vision-20260613i`). No mutations, no git.

Source files on device (owned by uid 1000, untouched):
- `/opt/rebot-models/hand_eye.npz`
- `/opt/rebot-models/calibration/orbbec_gemini2/intrinsics.npz`

---

## 1. Hand-eye transform (`hand_eye.npz`)

Keys:
- `T_result` (4,4) float64 — the eye-in-hand transform
- `mode` = `eye_in_hand`
- `method` = `TSAI`
- `n_samples` = `16`

`T_result` (camera frame ← TCP frame, i.e. `T_hand_eye`):

```
[[-2.48258049e-03 -3.68924343e-01  9.29456113e-01 -6.79878262e-02]
 [-9.99980750e-01  6.20126876e-03 -2.09513609e-04  9.14317349e-03]
 [-5.68651248e-03 -9.29438741e-01 -3.68932636e-01  5.38366739e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
```

Translation (camera origin in TCP frame): `[-0.06799, 0.00914, 0.05384]` m.

### Sign convention
Per `grasp_service.py:479`:

```python
T_cam2base = tcp_pose @ T_hand_eye
```

So `T_hand_eye == T_result` is the **TCP→camera** mount transform (camera pose expressed
in the TCP frame). Composed left-with the live TCP pose (base←TCP) it yields the camera
pose in the base frame (base←camera). This is a classic eye-in-hand layout: the camera is
rigidly bolted to the wrist/TCP, the calibration solved by TSAI over 16 samples.

Config wiring (`apps/voice_rebot_arm/config.yaml:311-313`):
```yaml
# Eye-in-hand hand-eye calibration (TSAI, 16 samples). npz key is T_result;
# the loader falls back to the first key, so no rename needed.
hand_eye_path: "${REBOT_HAND_EYE:-/opt/rebot-models/hand_eye.npz}"
```

---

## 2. Camera model + intrinsics

Configured camera (`apps/voice_rebot_arm/config.yaml:306-310`):
```yaml
camera:
  type: "orbbec_gemini2"
  color_width: 1280
  color_height: 720
  fps: 30
```

**A saved intrinsics file EXISTS** (so we use the REAL K, not a datasheet default):
`/opt/rebot-models/calibration/orbbec_gemini2/intrinsics.npz`

Keys:
- `camera_matrix` (3,3) float64 — K
- `dist_coeffs` (5,) float64 — OpenCV [k1, k2, p1, p2, k3]
- `resolution` (2,) int32 — [width, height]

K (color stream @ 1280×720):
```
[[691.65454102   0.         639.18127441]
 [  0.         691.59686279 359.49066162]
 [  0.           0.           1.        ]]
```
- fx = 691.6545, fy = 691.5969
- cx = 639.1813, cy = 359.4907
- resolution = [1280, 720]

Distortion `dist_coeffs`:
```
[ 4.23667870e+01 -8.73914871e+01 -1.33444660e-03 -7.45896032e-05  9.74315414e+01]
```
(k1=42.37, k2=-87.39, p1=-0.00133, p2=-0.0000746, k3=97.43 — large radial terms; this is
the saved per-unit calibration, not a pinhole ideal. For the Isaac sim pinhole camera the
K above is what matters; distortion can be ignored for a synthetic pinhole render unless we
deliberately model lens distortion.)

> Note: the live driver normally reads K from the Orbbec SDK on open, but here a saved
> `intrinsics.npz` is present and is the authoritative per-device calibration — use it.

---

## 3. What landed on the Mac

Under `/Users/harvest/project/seeed-local-voice/sim/calib/`:
- `hand_eye.npz` (1200 B, md5 `b81eb3918c26fe8ebabcf6ad61d13661`)
- `intrinsics.npz` (906 B, md5 `0a44e531c5cdf0316886238e237c9ebb`)

Both verified to load on the Mac (`np.load` returns the keys/shapes above).

---

## 4. How to map this into the sim / harness

### (a) Isaac sim camera prim
1. **Intrinsics → camera prim.** Set the sim color camera to 1280×720 and apply the real K:
   - From `fx, fy, cx, cy = 691.6545, 691.5969, 639.1813, 359.4907`, the horizontal FOV is
     `2*atan(W / (2*fx)) = 2*atan(1280 / 1383.31) ≈ 85.0°`; vertical FOV
     `2*atan(720 / (2*fy)) ≈ 55.0°`.
   - In USD, set `focalLength` + `horizontalAperture` so that
     `fx = focalLength * (W / horizontalAperture)`. Pick e.g. `horizontalAperture = 36 mm`
     → `focalLength = 36 * 691.6545 / 1280 ≈ 19.45 mm`,
     `verticalAperture = 36 * 720 / 1280 = 20.25 mm`. cx/cy are near-centered
     (cx≈639.18 vs ideal 639.5, cy≈359.49 vs ideal 359.5) so principal-point offset is
     negligible — a centered pinhole is fine.
2. **Mount (T_hand_eye) on the wrist.** Parent the camera prim under the wrist/TCP link and
   set its local transform to `T_result` (TCP→camera). The camera then pans with the arm
   exactly as the real eye-in-hand rig. To get camera-in-base at any pose, the sim/grasp
   code does the same `T_cam2base = tcp_pose @ T_hand_eye`.
   - Mind the axis convention: `T_result` is OpenCV camera frame (x-right, y-down,
     z-forward/optical). Isaac/USD cameras use -z forward, +y up. Apply the standard
     OpenCV↔USD camera-axis flip (rotate 180° about camera x, i.e. negate y and z axes)
     when authoring the USD prim, OR keep the OpenCV convention in code and only convert at
     render time. Keep this consistent with whatever convention `grasp_service.py` already
     assumes for `tcp_pose`.

### (b) Replacing the SYNTHESIZED extrinsic/K in `tools/synthetic_grasp_harness.py`
(Do NOT edit the harness here — this just states the swap so the Tier-A sweep can be re-run
with real calibration.)

At the top of `tools/synthetic_grasp_harness.py` the harness currently synthesizes a camera
extrinsic and K. Swap them for the pulled real values:
- **K**: replace the synthesized intrinsic with
  `np.load('sim/calib/intrinsics.npz')['camera_matrix']` at resolution `[1280, 720]`
  (fx≈691.65, fy≈691.60, cx≈639.18, cy≈359.49). If the harness renders at a different
  resolution, scale K by the resolution ratio (`fx,cx *= W'/1280`, `fy,cy *= H'/720`).
- **Extrinsic**: replace the synthesized eye-in-hand mount with
  `T_hand_eye = np.load('sim/calib/hand_eye.npz')['T_result']` and compute the per-pose
  camera extrinsic as `T_cam2base = tcp_pose @ T_hand_eye` (matching
  `grasp_service.py:479`), instead of a hand-tuned/synthetic camera pose.

Then re-run the Tier-A sweep and check whether the **"flat box z<0.08 unreachable"** finding
still holds under the true intrinsics + true wrist-mounted extrinsic. The real K has a wider
horizontal FOV (~85°) and the real mount sits ~68 mm back / ~54 mm up along the TCP, which
changes how much of the near-tabletop the eye-in-hand camera actually sees at low z — that
is exactly the variable the synthetic values were guessing.

---

## EVIDENCE (raw)

### `ls -la /opt/rebot-models/` + find
```
===REBOT-MODELS===
total 65164
drwxrwxr-x 4 1000 1000     4096 Jun 12 13:56 .
drwxr-xr-x 1 root root     4096 Jun 14 00:15 ..
drwxrwxr-x 3 1000 1000     4096 Jun 12 02:55 calibration
-rw-r--r-- 1 1000 1000     1200 Jun 12 02:55 hand_eye.npz
drwxrwxrwx 2 root root     4096 Jun 12 13:40 trt-cache
-rw-rw-r-- 1 1000 1000 24906708 Jun 12 13:56 yoloe-26s-seg-box.engine
-rw-rw-r-- 1 1000 1000 41794017 Jun 11 10:20 yoloe-26s-seg-box.onnx
===FIND-HANDEYE===
/opt/rebot-models/hand_eye.npz
===FIND-INTRINSICS===
/opt/rebot-models/calibration/orbbec_gemini2/intrinsics.npz
```

### Printed npz keys + matrices
```
=====/opt/rebot-models/hand_eye.npz=====
-- T_result (4, 4) float64
[[-2.48258049e-03 -3.68924343e-01  9.29456113e-01 -6.79878262e-02]
 [-9.99980750e-01  6.20126876e-03 -2.09513609e-04  9.14317349e-03]
 [-5.68651248e-03 -9.29438741e-01 -3.68932636e-01  5.38366739e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
-- mode (1,) <U11
['eye_in_hand']
-- n_samples (1,) int64
[16]
-- method (1,) <U4
['TSAI']
=====/opt/rebot-models/calibration/orbbec_gemini2/intrinsics.npz=====
-- camera_matrix (3, 3) float64
[[691.65454102   0.         639.18127441]
 [  0.         691.59686279 359.49066162]
 [  0.           0.           1.        ]]
-- dist_coeffs (5,) float64
[ 4.23667870e+01 -8.73914871e+01 -1.33444660e-03 -7.45896032e-05  9.74315414e+01]
-- resolution (2,) int32
[1280  720]
```

### Camera config excerpt (apps/voice_rebot_arm/config.yaml)
```
306:    camera:
307-      type: "orbbec_gemini2"
308-      color_width: 1280
309-      color_height: 720
310-      fps: 30
311-    # Eye-in-hand hand-eye calibration (TSAI, 16 samples). npz key is T_result;
313:    hand_eye_path: "${REBOT_HAND_EYE:-/opt/rebot-models/hand_eye.npz}"
```

### Mac: `ls -la sim/calib/` + load-verify
```
-rw-r--r--@ 1 harvest  staff  1200 Jun 14 10:15 hand_eye.npz
-rw-r--r--@ 1 harvest  staff   906 Jun 14 10:15 intrinsics.npz

sim/calib/hand_eye.npz   ['T_result', 'mode', 'n_samples', 'method'] [(4, 4), (1,), (1,), (1,)]
sim/calib/intrinsics.npz ['camera_matrix', 'dist_coeffs', 'resolution'] [(3, 3), (5,), (2,)]
```
md5 (device == Mac): hand_eye `b81eb3918c26fe8ebabcf6ad61d13661`, intrinsics `0a44e531c5cdf0316886238e237c9ebb`.
