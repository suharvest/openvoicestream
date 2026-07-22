# Qwen3-TTS 0.6B BASE → TensorRT-Edge-LLM v0.8.0 streaming worker (N>1) — port spec

Status: WORKING SPEC (synthesized from 3 codex deep-reads + 1 voxedge interface map, 2026-06-19).
Goal: streaming low-latency Qwen3-TTS **0.6B base** (fixed speaker, NOT CustomVoice) on **v0.8.0**, N>1 parallel,
re-export from HF, integrated into voxedge `jetson.trt_edge_llm` backend, perf'd on orin-nano (fp16 + W8A16),
production-deployable. Do NOT touch live `seeed-orin-nx` arm stack.

## Repos / branches
- Fork: `/Users/harvest/project/TensorRT-Edge-LLM`
  - `origin/release/0.8.0` (f9cc746/f9c29..) = NVIDIA upstream v0.8.0 = PORT TARGET
  - `highperf/runtime-service` = v0.7.0 base adaptation; has `examples/omni/qwen3_tts_worker.cpp`
  - `v071/customvoice-product` (HEAD 893ba2a) = has `examples/omni/qwen3_tts_streaming_worker.cpp` w/ slot-pool + shared-engine ctor
  - push remote = `suharvest` (github.com/suharvest/TensorRT-Edge-LLM)
- voxedge: `/Users/harvest/project/voxedge` (backend layer — already speaks the worker JSON protocol)
- Official Qwen3-TTS impl: `/Users/harvest/project/Qwen3-TTS`
- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (12Hz codec). Weights cached on wsl2-local HF cache.
  Community ONNX ref: `elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX`.

## ⚠️ CORRECTION (2026-06-20, from codex deep-read) — Base = speaker-encoder/reference-audio path
Earlier assumption "base = clean 8-row fixed-speaker path" was WRONG. Reality:
- **Base** (`0.6B-Base`) conditions on a **speaker_encoder + reference audio** → external float embedding vector
  (Qwen3-TTS `modeling_qwen3_tts.py:1822-1825,1941-1954,2070-2073,2166-2172`). Effectively reference-driven voice.
- **CustomVoice** = discrete named-speaker token id → embedding-table lookup (`modeling_qwen3_tts.py:2091-2101`). 9 fixed voices.
- v0.8.0 **HARD-REJECTS base**: `tensorrt_edgellm/scripts/export.py:1668-1671` ("Only ... CustomVoice checkpoints are supported")
  AND stripped the embedding mechanism from runtime/kernel.
- v0.7.0 `highperf/runtime-service` HAD the full base path → we BACKPORT it. User confirmed: our own extension, ran on v0.7.0.

### BASE BACKPORT — 5 touch points (codex, file:line + v0.7 source)
1. Exporter: delete guard `export.py:1668-1671`; re-add speaker_encoder ONNX stage from `highperf:tensorrt_edgellm/onnx_export/audio_export.py:156-206` (skip when model.speaker_encoder is None).
2. Runtime header: add `std::vector<float> speakerEmbedding` to `TalkerGenerationRequest` (`qwen3OmniTTSRuntime.h:120-123`); v0.7 had it at `highperf:...h:152-156`.
3. Runtime cpp: copy embedding→GPU when present (`qwen3OmniTTSRuntime.cpp:753-777` is the customvoice-only path; v0.7 ref `highperf:...cpp:4208-4245`).
4. CUDA kernel: restore external-embedding path at row 6 — `talkerMLPKernels.cu:488-491` (v0.8 = embTable[speakerId] only); v0.7 accepted external embedding OR token OR fallback at `highperf:...talkerMLPKernels.cu:447-497`.
5. Runtime + worker: run reference audio through speaker_encoder TRT engine → embedding at inference; worker passes speaker_embedding_b64 (current `qwen3_tts_streaming_worker.cpp:297-300` only passes speaker_id).

HF config keys to confirm (Base vs CustomVoice): root `model_type,tts_model_type,tts_model_size,speaker_encoder_config`;
talker_config `spk_id,codec_language_id,num_code_groups,text_hidden_size,codec_* ids`; root `tts_pad/bos/eos_token_id`.

## KEY DE-RISK: base is clean
All three historical blockers are **CustomVoice-product-only — NONE apply to base**:
- `QWEN3_TTS_PRELOAD_TALKER_EMBEDS` / NVIDIA #87 IssueC (product cpp:1129-1151)
- 9-row vs 8-row prefix (product kernel :479/:509). Base = fixed **8-row** (`assistantPreambleKernel`, "8+textLen+2").
- W8A16-talker EOS breakage (product-only quantize script).

## v0.8.0 already has (REUSE, don't re-port)
- `cpp/runtime/qwen3OmniTTSRuntime.{h,cpp}` with fixed-speaker `TalkerGenerationRequest{speakerName,speakerId}` (no `language`).
- 8-row base `assistantPreambleKernel` (cu:438-564).
- Export tooling `tensorrt_edgellm/models/qwen3_tts/*.py` (talker, code_predictor, code2wav, audio, text).
- `examples/omni/qwen3_tts_inference.cpp` (OFFLINE only — no streaming worker).
- `ThinkerTalkerStreamingConfig{talkerPrefillThreshold,codecChunkFrames,onAudioChunkReady}` (h:250-257).
- Native batched `runTalkerGenerationLoop` (cpp:~1380-1456) — BUT see N>1 below: do NOT use batching for streaming.

## Missing on 0.8.0 → THE PORT
`examples/omni/qwen3_tts_streaming_worker.cpp` (JSON-line streaming worker w/ slot-pool) — port from product HEAD,
driving the base fixed-speaker path. (highperf's `qwen3_tts_worker.cpp` is the simpler non-slot variant.)

## N>1 concurrency DECISION = Option (A): N independent runtime instances
- One worker thread + CUDA stream + runtime per slot (`--max_slots`). Each runs batch=1 → per-instance
  `FrameCallback`, `batchIdx` always 0, no demux. This is what product `qwen3_tts_streaming_worker.cpp`
  (893ba2a:121-147 parse, :701-724 per-slot stream+runtime, :727-730 Code2Wav, :755-758 thread) already does.
- REJECT native batching (C): couples TTFA across independently-arriving streaming requests (one `globalFrame`
  loop, cpp:1403-1408). REJECT calling one runtime from N threads — `Qwen3OmniTTSRuntime` owns shared mutable
  tensors (h:512-589, written in runTalkerGenerationLoop cpp:1415-1480) → thread-unsafe.
- MOSS N=2 precedent = multi-slot worker-thread pool (commit d92a306, moss worker :512-515 parse, :769-779
  spawn, runtime acquire/release :363-405). Same shape as Option (A).

## 8GB OOM RISK (biggest risk) + mitigation
- N=2 independent fp16 instances ≈ 3-4GB (talker ~1.2GB each + CP + Code2Wav + KV + contexts). Tight on 8GB,
  worse if LLM service co-resident.
- MITIGATION = **shared-engine constructor** (share `ICudaEngine` across slots → halves weight memory):
  exists at product `893ba2a:cpp/runtime/qwen3OmniTTSRuntime.h:110-130` + `766470` "shared-engine ctor path".
  PORT this for N=2. Validate with `tegrastats` before committing N=2.

## TOUCH-LIST (port, option A + shared-engine)
1. `examples/omni/CMakeLists.txt` (0.8: ~6-20) — add `qwen3_tts_streaming_worker` target.
2. `examples/omni/qwen3_tts_streaming_worker.cpp` — `git checkout` from 893ba2a, then strip CustomVoice-only
   request fields (`language`, `speaker_embedding_b64`, predictor sampling not on 0.8), fix response field
   renames (`rvqCodes`→`batchRvqCodes`, `numFrames`→`numFramesPerSample`), fix includes to 0.8 layout
   (`runtime/legacy/llmEngineRunner.h`, `runtime/llmInferenceRuntime.h`), drop `RuntimeOptions` struct (0.8 has 4-arg ctor h:95-96).
3. `cpp/runtime/qwen3OmniTTSRuntime.h` (0.8:109-129) — add per-request `FrameCallback` field to `TalkerGenerationRequest`.
4. `cpp/runtime/qwen3OmniTTSRuntime.cpp` (0.8:~1455) — invoke callback after each frame append; (~1547-1553) final-flush callback.
5. `cpp/runtime/qwen3OmniTTSRuntime.h` (0.8:110-130) — port shared-engine ctor (N=2 memory).
6. code2WavRunner: USE 0.8's `cpp/multimodal/code2WavRunner.*` (detects waveform dtype cpp:169-180 — better than
   highperf's hardcoded FLOAT). statefulCode2WavRunner = product/highperf only; optional, `#ifdef` for N=1.
7. `cpp/kernels/qwen3TtsCpKernels/*` (W8A16 CP kernels) = product-only, NOT needed for base fp16 — skip.
- voxedge `trt_edge_llm_tts.py`: NO CHANGE (worker_concurrency already gates `--max_slots`).

## PORT STRATEGY = manual merge (NOT cherry-pick)
```
git switch -c port/qwen3-tts-base-v080 origin/release/0.8.0
git checkout v071/customvoice-product -- examples/omni/qwen3_tts_streaming_worker.cpp   # then edit per touch-list
# port CMakeLists target block by hand; add runtime FrameCallback by hand
```
Edit order for incremental compile: CMakeLists → runtime.h (callback field+typedef) → runtime.cpp (invoke) →
worker.cpp (strip/rename/includes). Highest C++ risk historically = FrameCallback into batched loop; with
Option (A) batch=1 this is trivial (batchIdx=0).

## BUILD (orin-nano sm_87) — VALIDATED ✅ (2026-06-19)
No build.sh on 0.8 — CMake direct. CONFIRMED recipe (qwen3_tts_inference built clean, libs resolved):
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 -DENABLE_CUTE_DSL=OFF
cmake --build . --target qwen3_tts_streaming_worker -j3   # -j3 max (8GB RAM), no OOM at j3
```
**MUST use `ENABLE_CUTE_DSL=OFF`** — `=ALL` pulls prebuilt libcutedsl_aarch64.a needing CUDA 12.8+ APIs
(`cudaLibraryLoadData` etc.) absent on JetPack CUDA 12.6 → link fails. OFF = cuBLAS GEMM fallback (intended Jetson
path). `-DCMAKE_CUDA_ARCHITECTURES=87` not needed (toolchain+EMBEDDED_TARGET set sm_87). Binary → `build/examples/omni/`
(custom RUNTIME_OUTPUT_DIRECTORY). Clone at orin-nano `~/project/edgellm-v080-build`. Reuse this build dir on the port
branch (off same release/0.8.0 commit) → only changed TUs recompile, worker link is fast.
See memory edgellm_v080_jetson_cutedsl_cuda126_2026_06_19.

## ENGINE EXPORT (Phase 1)
- ONNX export (PyTorch→ONNX, arch-independent): run on wsl2-local (has weights+GPU) OR orin, via
  `tensorrt_edgellm/models/qwen3_tts/*.py`. Components: talker, code_predictor, code2wav (+ audio/text).
- TRT engine build (.plan): MUST be on Orin (sm_87) using 0.8 builder (v0.8 `llmBuilder.cpp` external-weight-copy
  is mandatory — engines from v0.7 builder are incompatible).

## voxedge integration (Phase 3) — backend EXISTS
`jetson.trt_edge_llm` IS the qwen3-tts backend (`voxedge/backends/jetson/trt_edge_llm_tts.py`, talks
`qwen3_tts_worker`/`qwen3_tts_inference` JSON-line via WorkerIO). Env knobs: `EDGE_LLM_TTS_WORKER_BIN`,
`EDGE_LLM_TTS_TALKER_DIR`, `_CP_DIR`, `_CODE2WAV_DIR`, `_TOKENIZER_DIR`, `EDGE_LLM_QWEN3_PROFILE=highperf`,
`EDGE_LLM_TTS_STATEFUL_CODE2WAV`, `OVS_TTS_WORKER_CONCURRENCY`→`--max_slots`.
Just add: profile `configs/profiles/jetson-edgellm-v080-qwen3ttsbase.json` (model after ...-moss.json),
leaf `configs/leaves/qwen3-tts-base.yaml`, registry already maps `jetson.trt_edge_llm`.
- Worker JSON protocol: req fields incl `id,text,stream,first_chunk_frames,chunk_frames,adaptive_chunks,
  speaker,speaker_id`; events `ready{init_ms}`, `chunk{audio_b64,frames,samples,sample_rate,is_final,code2wav_ms}`,
  `done`, `error`, `cancelled`.

## FINAL (2026-06-20) — int4 + fp8, fully NATIVE, ASR-validated
Config: **int4-AWQ talker + int4-AWQ CodePredictor + fp8 text_embedding + fp16 code2wav** + v3 normalized ref_06 embedding + talker_temperature 0.9 + first_chunk_frames 4.
- Perf: **RTF 0.44, warm TTFA 0.21s** (fp16 was 0.69/0.54). VRAM: talker 907→245, CP 191→98, text_embedding 622→320 = **~−1.06GB/instance**, no speed/quality cost.
- Quality: EN WER 0%, ZH CER 1.7% (= fp16 baseline) — ASR/CER A/B validated, clean EOS CN+EN.
- All upstream-native: int4 = stock v0.8.0 quantize.py int4_awq (Int4GroupwiseGemmPlugin, sm_87 source-compiled, NO backport); fp8 embedding = existing portable fp8 kernel + small runtime patch (load fp8+scales, wire scales at TTS embedding call sites). code2wav INT8 NOT viable (TRT10.3 can't QDQ Conv1D).
- **Quant recipes** (reproducible): int4 talker/CP = wsl2 `~/project/edgellm-v080-export/{quantize_talker_stage1,stage2_export,quantize_cp_stage1,cp_stage2_export}.py` (rebuild as Qwen3ForCausalLM → INT4_AWQ_CFG → re-prefix → existing _export → int4 ONNX; exclude lm_head + CP down_proj). fp8 emb = `scripts/quantize_text_embedding_fp8.py` (e4m3 per-group128 scale=amax/448) or numpy/ml_dtypes equiv. Branch suharvest port/qwen3-tts-base-v080 @ 873ca22 (C++: cuBLAS GEMM fallback + M=1 GEMV + shim link + fp8 runtime patch).
- **GOTCHA (the lesson)**: always normalize CN script (opencc t2s) before CER — raw whisper traditional-vs-simplified inflated a phantom "15pt int4 degradation" to a real ~0. And: lowering talker_temperature breaks EOS (120s runaway); pitch consistency = longer/normalized ref embedding, NOT temperature.

## RESULT (2026-06-20) — SHIPPED fp16 (superseded by int4+fp8 above)
Working streaming Qwen3-TTS **0.6B base** on v0.8.0 / orin-nano sm_87: **RTF 0.69, warm TTFA 0.54s, N=2 parallel** (5.6GB free), intelligibility ASR-verified. CuTe-DSL GEMM not viable sm_87 → cuBLAS-free tiled+M=1 GEMV fallback. W8A16 concluded NOT viable (EOS) → fp16 ships.
Commits: fork `suharvest/port/qwen3-tts-base-v080` (worker + runtime speakerEmbedding + kernel fallback + shim link); voxedge `origin/main b783037` (base_speaker_embedding_b64 injection in jetson.trt_edge_llm); seeed-local-voice `7231f23` (profile+leaf+builder).

## DEPLOY RECIPE (production-deployable; do NOT touch live seeed-orin-nx)
Artifacts (regenerate via the BUILD + ENGINE-build + EXPORT sections above):
- worker `qwen3_tts_streaming_worker` + `libNvInfer_edgellm_plugin.so` (built ENABLE_CUTE_DSL=OFF on Orin)
- engines `talker/llm.engine`, `code_predictor/llm.engine`, `code2wav/code2wav.engine` (talker/CP built maxInputLen=1024 maxKVCacheCapacity=1536; code2wav maxCodeLen=128)
- `tokenizer` files (from exported onnx `llm/`), `ref_embedding.b64.txt` (fixed base voice)
Production layout (profile `jetson-edgellm-v080-qwen3ttsbase.json` references): worker→`/opt/jv-workers/qwen3_tts_streaming_worker`, plugin→same dir + `EDGELLM_PLUGIN_PATH`, engines→`/opt/models/qwen3-tts-base/engines/{talker,code_predictor,code2wav}`, tokenizer→`/opt/models/qwen3-tts-base/tokenizer`, embedding→`/opt/models/qwen3-tts-base/ref_embedding.b64.txt` (profile sets `EDGE_LLM_TTS_BASE_SPK_EMBED_PATH`).
Runtime env (profile-set): `EDGELLM_PLUGIN_PATH`, `LD_LIBRARY_PATH=/usr/local/cuda/lib64`, `EDGE_LLM_TTS_*` dirs, `OVS_TTS_WORKER_CONCURRENCY=1|2`, `ENABLE_CUTE_DSL` irrelevant at runtime (baked in worker). Activate: `OVS_PROFILE=jetson-edgellm-v080-qwen3ttsbase`.
Image bake = operator step: overlay the artifacts into the voice image at the above paths (engines ~1.3GB) + ship the profile; do NOT rebuild the canonical Dockerfile from scratch (overlay only). On-device speaker-encoder (live reference-audio→embedding) = future item; current = fixed precomputed embedding.

## Phases & gates
1. Export base 0.6B → ONNX → fp16 engines (on validated 0.8 builder).
2. Port streaming worker (option A) + shared-engine; build on orin-nano.
3. profile/leaf; reuse jetson.trt_edge_llm; /tts + /tts/stream path.
4. Perf orin-nano fp16: TTFA/RTF + ASR-accuracy gate (bytes≠speech — energy+ASR round-trip).
5. W8A16 quantize + fp16 vs W8A16 compare (VRAM+TTFA+CER). Decide ship precision.
6. Image bake + deploy profile + commit (split by repo/module) + Memory.
Incremental: N=1 correct+perf FIRST, then N=2 shared-engine + tegrastats VRAM validation.
