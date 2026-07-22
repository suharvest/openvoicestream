# GG-CNN2 ONNX Export Notes (Phase 3 prep — reBot grasp pipeline)

Exported and validated on wsl2-local, 2026-06-12.

## Artifacts

| File | Input | Outputs | md5 | Size |
|---|---|---|---|---|
| `ggcnn2-300.onnx` | `depth` [1,1,300,300] f32 | pos, cos, sin, width — each [1,1,300,300] f32 | `eafc4683a67f10cc1d4b12e27d6c9321` | 307,284 B |
| `ggcnn2-360.onnx` | `depth` [1,1,360,360] f32 | pos, cos, sin, width — each [1,1,360,360] f32 | `673fceaa9ee70605aae78286bcc66c76` | 307,284 B |

- Static shapes, no dynamic axes, opset 12 (torch 2.12 exporter), single-file (no external `.data`).
- `onnx.checker.check_model` PASS for both.
- Model is fully convolutional — the 360 variant is the same weights traced at a larger input.

## Source

- Repo: https://github.com/dougsm/ggcnn @ commit `0c50aa7600e8a30d44c5c85cebd6e3394a81f30e`
- License: BSD-3-Clause (repo LICENSE)
- Weights: https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn2_weights_cornell.zip
  - sha256 `f71e3575fe70bea6817239f9fad98264af5cfd95dd9bbe69ff4144696f34f972`
  - Loaded from `epoch_50_cornell_statedict.pt` into `models.ggcnn2.GGCNN2` (state_dict, Cornell-trained).

## Preprocessing contract (must match repo eval code)

From `utils/dataset_processing/image.py` `DepthImage`:

1. **Inpaint** missing depth (value 0 / NaN→0): `cv2.copyMakeBorder(img,1,1,1,1,BORDER_DEFAULT)`,
   mask = `(img == 0)`, scale by `abs(img).max()` into [-1,1] float32,
   `cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)`, strip the 1px border, rescale back.
2. **Crop/resize** to the network input size (300x300 centre crop in repo eval).
3. **Normalise** (`DepthImage.normalise`, image.py:202-206):
   `img = np.clip(img - img.mean(), -1, 1)` — depth in metres, zero-centred per-image, clipped to [-1,1].
4. Feed as float32 `[1,1,H,W]` tensor named `depth`.

## Output map order (from `GGCNN2.forward`)

`pos` (grasp quality, apply sigmoid? NO — raw conv output, repo post-processes with
gaussian filter only), `cos` = cos(2θ), `sin` = sin(2θ), `width` (grasp width, scale
×150px in repo post-processing). Grasp angle θ = 0.5 * atan2(sin, cos).

## Validation (wsl2-local, x86_64 WSL2)

- Versions: torch 2.12.0+cpu, onnx 1.21.0 (consolidation/checker), onnxruntime 1.26.0, Python 3.11.
- Parity torch vs onnxruntime CPU (same realistic depth input, per-output max-abs-diff):
  - 300: [4.13e-06, 5.90e-06, 1.37e-06, 2.19e-06] → MAX 5.90e-06 (< 1e-4 PASS)
  - 360: [9.64e-06, 1.42e-05, 3.34e-06, 5.19e-06] → MAX 1.42e-05 (< 1e-4 PASS)
- onnxruntime CPU **single-thread** latency (10 runs after 3 warmup, wsl2-local reference):
  - 300: mean 35.2 ms (min 34.0, max 40.3)
  - 360: mean 51.7 ms (min 49.7, max 55.1)
  - Budget is <300 ms on a weaker CPU → ~8x headroom at 300x300 on this reference box.
- Full numbers in `ggcnn2-export-report.json` (parity measured pre-consolidation; consolidation
  only embeds the identical external weight tensors into the .onnx, ORT re-run confirmed OK).

## Repro

Workspace on wsl2-local: `~/ggcnn-export` (uv project; `setup_and_run.sh`, `export_ggcnn2.py`, `consolidate.py`).
