# rkvoice-stream / rkvoice-engine — BOTTOM (Rockchip RK NPU layer)

**Analyzed refs:** `rkvoice-stream` main @ **76e9ded**; `rkvoice-engine` main @ **1f133f3**.
**Tool:** directory inspection (light depth per scope).

These are the RK3576/RK3588 counterparts of the Jetson stack. **rkvoice-stream** = the runtime/server;
**rkvoice-engine** = the model conversion (ONNX→RKNN / →RKLLM) pipeline (recently MIT-licensed / open-sourced).

---

## 1. rkvoice-stream (RK runtime + server)
```
rkvoice_stream/
  app/server.py            the RK voice server (health + runtime_info endpoints)
  backends/                RK ASR/TTS backends
  engine/  asr.py  tts.py  the engine entrypoints
  runtime/ rkllm_wrapper.py the RKLLM C-API ctypes wrapper + __init__
  platform/                platform detection
  vad/
configs/  docker/  models/{asr,tts,common}/  baseline/  tools/  tests/  docs/evidence/
```
- `runtime/rkllm_wrapper.py` is the RKLLM driver shim (ctypes over librkllmrt).
- `engine/{asr,tts}.py` are the RK equivalents of voxedge's jetson backends — but RK is a **separate runtime
  tree**, not under voxedge. seeed reaches RK via `server/core/rk_runtime.py` + `rk_artifacts.py` and the
  registry keys `rk.asr` / `rk.tts` (resolved through `build_rk_{asr,tts}_config`).
- Conversion/verification helpers under `models/`: `convert_rknn_fixed.py`, `convert_vocoder_rknn.py`,
  `verify_code2wav_stateful_parity.py`, `rk3576_tts_dump.py`, MOSS onnx→rkllm converters.

## 2. rkvoice-engine (RK model conversion pipeline)
```
addon/models/
  common/convert_rknn_fixed.py
  asr/qwen3/   export_qwen3_asr_weights.py  export_decode_{variants,stream}.py  export_fixed_shapes.py
               export_rkllm_talker.py  export_onnx.py  export_tokenizer_encode.py
  asr/paraformer/  convert_paraformer_rknn.py  export_paraformer_hybrid.py
  tts/moss/    convert_moss_onnx_to_rkllm_state.py  export_moss_rkllm_runtime_assets.py
               probe_moss_{rkllm,rknn}_runtime.py
  tts/convert_vocoder_rknn.py
manifests/
```
- This is the RK analogue of the fork's `tensorrt_edgellm/scripts/{export,quantize}.py` — the "build the
  device artifacts" side. Pure conversion/export drivers; no runtime serving.

## 3. Placement in the layering
- RK occupies the **same BOTTOM tier** as the TRT fork + jetson-voice-engine, but on a parallel rail:
  RKLLM/RKNN instead of TensorRT. It is **NOT** consumed through voxedge's `backends/jetson`; seeed has a
  dedicated RK path (`rk.asr`/`rk.tts` registry keys → `rk_runtime`/`rk_artifacts` + voxedge `backends/rk/*`).
  voxedge DOES carry a thin `voxedge/backends/rk/` (asr.py/tts.py/runtime.py/artifacts.py), so the RK runtime
  proper lives in rkvoice-stream while voxedge has the seeed-facing RK adapter — note this split for consolidation.
