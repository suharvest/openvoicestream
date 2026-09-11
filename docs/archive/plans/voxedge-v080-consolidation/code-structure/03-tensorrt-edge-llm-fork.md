# TensorRT-Edge-LLM fork — BOTTOM layer (source of truth for the C++ runtime)

**Analyzed ref:** `port/qwen3-tts-base-v080` @ **873ca22** — the v0.8.0 base port.
> NOTE: `wip/fp8-embedding` points to the **same commit** (873ca22) — the fp8 text-embedding
> work landed on the base-v080 branch head; there is no separate fp8 tree to diff.
> The repo's *checked-out* branch is `v071/customvoice-product` (LEGACY v0.7.1) — **NOT analyzed as canonical.**
**int4 drivers ref:** `suharvest/wip/native-int4-talker` @ **ff2318e** (branches off the 0.8.0 release 8ac8fd6).

AST/structural method: BSD `ctags` only (no C++ tags) and no libclang available on this host →
C++ class/method inventory extracted by careful header parsing (regex over declarations). Python tooling
(`tensorrt_edgellm/`) read directly. Stated per the guardrails.

Read at-ref via `git archive port/qwen3-tts-base-v080 | tar -x` (working tree never switched).

---

## 1. Top-level tree (v0.8.0)

```
cpp/            C++ runtime + kernels + plugins (compiles to edgellmCore / edgellmKernels / NvInfer_edgellm_plugin)
examples/omni/  qwen3_tts_inference.cpp + qwen3_tts_streaming_worker.cpp  (link edgellmCore)
tensorrt_edgellm/  Python export/quant tooling + model definitions (the "build the engines" side)
kernelSrcs/ cmake/ 3rdParty/ unittests/ tests/ experimental/pybind/
CMakeLists.txt  (top — declares commonLibraryExt, adds cpp/ + examples/, ENABLE_CUTE_DSL group)
```

## 2. CMake build-target graph (the load-bearing part for consolidation)

Top `CMakeLists.txt`:
- `commonLibraryExt` (INTERFACE) → `TensorRT::TensorRT`, `TensorRT::OnnxParser`.
- `add_subdirectory(cpp)` then `add_subdirectory(examples)`.
- `ENABLE_CUTE_DSL` is a semicolon group (`fmha;gdn`); `OFF` falls back to cuBLAS
  (the documented Jetson CUDA 12.6 build path — CuTe-DSL prebuilt needs CUDA 12.8 `cudaLibrary*`).
- `unitTest` exe links `edgellmCore commonLibraryExt`.

`cpp/CMakeLists.txt` — **the four artifacts**:

| Target | Type | Sources (GLOB_RECURSE) |
|---|---|---|
| `edgellmKernels` | STATIC | `kernels/*.{cpp,cu}` + `common/*.cpp` (MoE-Marlin `ops`/`sm80` excluded; rdc=true for marlin) |
| **`edgellmCore`** | STATIC | `common/ + kernels/ + sampler/ + multimodal/ + action/ + runtime/ + profiling/ + tokenizer/` — **this is the runtime everything links** |
| `edgellmBuilder` | STATIC | `builder/*.cpp` + common (+ `NV_ONNX_PARSER_LIB`) |
| `NvInfer_edgellm_plugin` | SHARED (so.1.0) | `plugins/*.cpp` |

`examples/omni/CMakeLists.txt` — **the two voice executables** (both `PRIVATE edgellmCore exampleUtils commonLibraryExt`):
- `qwen3_tts_inference` — one-shot TTS demo.
- **`qwen3_tts_streaming_worker`** — the JSON-line stdin/stdout streaming worker, **slot-pool N>1**
  (this is the binary the Python TTS backend spawns; see voxedge `trt_edge_llm_tts.py::_ensure_worker`).
- Both conditionally link `trt_edgellm_cutedsl_cudart_shim` with `-Wl,-u,cudaLibrary*` forced symbols
  (the CUDA 12.0–12.6 missing-export workaround).

> **Consolidation note:** cuBLAS is **dlopen'd at runtime inside the TTS runtime** ("no compile-time
> linking needed") — so the worker binary is portable across CUDA minor versions w.r.t. cuBLAS.

## 3. `cpp/runtime/` — the runtime classes

| File | Class / key API |
|---|---|
| `qwen3OmniTTSRuntime.{h,cpp}` | **`Qwen3OmniTTSRuntime`** — the TTS engine driver |
| `llmInferenceRuntime.{h,cpp}` | `LLMInferenceRuntime` (thinker/LLM) |
| `llmRuntimeUtils.{h,cpp}` | `Message`, `LLMGenerationRequest/Response`, `RopeConfig`, `collectRopeConfig`, rope/nope/longrope cache init, `buildBatchMapping`, `EmbeddingData` |
| `streaming.{h,cpp}` | **`StreamChannel`** (create/consume/tryPop/waitPop/cancel/setStreamInterval), `StreamChunk`, `decodePerSlot`, `emitChunks`, `applyStopStringMatch` |
| `slotPool.h` | **`SlotPool<TSlot>`** template — capacity/acquireFree/acquireOrExisting/release/bind/unbind, mutex-guarded. Namespace `tensorrt_edge_llm::runtime`. **This is the N>1 concurrency primitive** the worker reuses (`rt_slotpool = tensorrt_edge_llm::runtime`). |
| `kvCacheManager.{h,cpp}` | `KVCacheManager`, `KVLayerConfig`, `getSeparateKVCache` |
| `hybridCacheManager.{h,cpp}` | `HybridCacheManager` — KV + recurrent/conv state (GDN/Mamba hybrid): compact/capture/restore batch, `commitSequenceLength` |
| `mambaCacheManager.{h,cpp}` | `MambaCacheManager` |
| `audioUtils.h`, `imageUtils.{h,cpp}` | multimodal IO helpers |
| sub-dirs | `config/ decoding/ exec/ features/ legacy/ preprocess/ state/` |

### `Qwen3OmniTTSRuntime` (qwen3OmniTTSRuntime.h) — load-bearing surface
- ctor: `Qwen3OmniTTSRuntime(talkerEngineDir, codePredictorEngineDir, tokenizerDir, cudaStream_t)`
- Request structs:
  - `TalkerGenerationRequest` — maxAudioLength, talkerTemperature/TopK/TopP, repetitionPenalty,
    `speakerName`/`speakerId`/**`speakerEmbedding` (vector<float>)**, messages, applyChatTemplate,
    `codecChunkFrames`/`subsequentChunkFrames`, `shouldCancel`.
  - `OmniGenerationRequest` — fullText/textTokenIds, **`prefillLength`** (layer0/layer14 cover [0,prefillLength)), same sampling fields.
  - `ThinkerTalkerStreamingConfig` — `talkerPrefillThreshold`, `codecChunkFrames`.
  - `TalkerGenerationResponse` — `batchRvqCodes` (3-deep vector), numFramesPerSample, success.
- Generation entrypoints (batched + single overloads):
  `handleAudioGeneration(...)`, `handleAudioGenerationFromThinker(...)`,
  **`handleStreamingGeneration(LLMInferenceRuntime& thinker, ...)`** — the joint thinker→talker stream.
- Internals: `captureDecodingCUDAGraph`, `getSpeakerIdByName`, `initializeTTSEmbeddings`,
  `executeTalkerPrefillStep`, `runCodePredictorGenerationForFrame`, `computeResidualConnection`,
  `extractTalkerLastHidden`, `runTalkerGenerationLoop`, `runSingleTalkerDecodeFrame`,
  `buildTalkerPrefillFromSegments`.
- State structs: `PerBatchTalkerState` (codecToken/talkerFrames/finished/seenTokenSet/rvqCodes/lastChunkEnd),
  `PerBatchStreamingHooks` (codecChunkFrames/subsequentChunkFrames/shouldCancel), `SegmentInfo`.

> **Base vs CustomVoice (consolidation-critical):** the base v0.8.0 runtime takes `speakerName`/`speakerId`/
> `speakerEmbedding` directly (8-row fixed speaker prefix). The legacy v0.7.1 customvoice path's blockers
> (preload-talker-embeds / 9-row prefix / W8A16-EOS) are **NOT present here** — this branch is the clean base.

## 4. `cpp/kernels/` — kernel families

```
common/  contextAttentionKernels/  decodeAttentionKernels/  embeddingKernels/
gdnKernels/  int4GroupwiseGemmKernels/  kvCacheUtilKernels/  mamba/  moe/
posEncoding/  preprocessKernels/  speculative/  talkerMLPKernels/
```
TTS-relevant: `embeddingKernels` (text/speaker embedding), `posEncoding`, **`talkerMLPKernels`**,
`int4GroupwiseGemmKernels` (dequantize.cuh, int4WoQGemvCuda.cu, int4WoqGemmCuda.cu — **already present on base-v080**),
`kvCacheUtilKernels` (the historically-patched KV-sizing kernel). GDN/Mamba/MoE serve the thinker LLM.

## 5. `cpp/plugins/` — TRT plugins (compiled into `NvInfer_edgellm_plugin.so`)
```
attentionPlugin/ gatedDeltaNet/ int4GroupwiseGemmPlugin/ int4MoePlugin/ mamba/
nvfp4MoePlugin/ nvfp4MoePluginGeforce/ utils/ vitAttentionPlugin/
```
**`int4GroupwiseGemmPlugin` is present on base-v080** (kernels + plugin both). What the int4 branch adds
is the *export drivers*, not new runtime plugin code (see §7).

## 6. `tensorrt_edgellm/` — Python export/quant tooling (engine-build side)

```
scripts/   export.py  quantize.py  reduce_vocab.py  insert_lora.py  merge_lora.py
           process_lora_weights.py  preprocess_audio.py
quantization/  quantize.py  quantization_configs.py  qwen3_asr_loader.py
               nemotron_h_patch.py  models/qwen3_asr/...
models/qwen3_tts/   modeling_qwen3_tts_text.py  _talker.py  _audio.py  _code2wav.py
                    modeling_code_predictor.py
models/qwen3_asr/   modeling_qwen3_asr_{text,audio}.py
models/qwen3_omni/  ...   models/qwen3_5{,_moe}/ ... (thinker LLMs)
onnx/ checkpoint/ lora/ chat_templates/ vocab_reduction/
```
`scripts/export.py` + `scripts/quantize.py` are the canonical engine-build entrypoints; the per-model
`modeling_*.py` define the TRT graph builders. **qwen3_tts is split into text / talker / audio / code2wav / code_predictor** sub-models — mirrors the runtime's talker + code-predictor + code2wav stages.

## 7. int4 export drivers — `wip/native-int4-talker` @ ff2318e (the reproducibility risk)

These scripts live **at repo root on the int4 branch** (NOT under `tensorrt_edgellm/scripts/`):
- **`quantize_talker_stage1.py`** — `main()` argparse; `remap_talker_to_hf(src_sd)`; loads Qwen3 talker,
  runs `mtq.quantize(int4_awq)` (TensorRT-ModelOpt), checks NaN/Inf calib batches, `export_hf_checkpoint`.
- **`stage2_export.py`** — `reprefix_unified_to_talker(uni_sd)` + `main()` (build TRT engine from stage1 ckpt).
- **`cp_stage2_export.py`** — code-predictor stage-2 export.

The branch also re-adds (vs the 0.8.0 release base it forked from): `cpp/kernels/int4GroupwiseGemmKernels/*`,
`cpp/plugins/int4GroupwiseGemmPlugin/*`, `cpp/plugins/int4MoePlugin/*`, `unittests/woqInt4{Gemm,Gemv}Test.cu`.
On `port/qwen3-tts-base-v080` those kernel/plugin trees already exist — so the **distinguishing payload of the
int4 branch is the three root-level driver scripts** (stage1/stage2/cp_stage2). They are *not merged into the
v0.8.0 base port branch*, so int4-talker reproducibility currently depends on a separate WIP branch — flag for
the consolidation plan.

## 8. examples/omni worker (`qwen3_tts_streaming_worker.cpp`) — structure
- `struct Args` + `parseArgs` (`--max_slots`, engine dirs, tokenizer). `--max_slots=1` ⇒ single slot/thread (back-compat).
- `audioToFloatSamples(AudioData)`; `emitEvent(Json)` (coutMutex-serialized JSON-line output).
- Cancel registry: `registerCancel/unregisterCancel/tripCancel` (per-request atomic flag).
- `struct TtsSlot { runtime, cudaStream, std::thread worker, atomic inUse, atomic shutdown, queue/cv }`;
  `struct WorkItem { Json request }`; `gMaxSlots`.
- Architecture: 1 stdin reader thread routes each JSON line to a free slot; **N worker threads, each bound to
  one slot** (uses `tensorrt_edge_llm::runtime::SlotPool` for binding via `bindSession/unbindSession`).
- This is the C++ counterpart of the Python `WorkerIO` protocol in voxedge/seeed (`worker_io.py`).
