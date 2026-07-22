# Edge-Voice Code-Structure Reference (v0.8.0 consolidation)

Structural/AST reference for the canonical edge-voice projects. **READ-ONLY analysis** — no working tree was
switched (all at-ref reads via `git show`/`git archive`); no builds/deploys. Complements
`consolidation-plan.md` (this is the *structure*; the plan is the *intent*).

## Per-project files
| # | File | Project | Layer | Ref analyzed |
|---|---|---|---|---|
| 1 | [01-seeed-local-voice.md](01-seeed-local-voice.md) | seeed-local-voice (server + "OpenVoiceStream" agent) | **TOP** | main @ 4532ed0 |
| 2 | [02-voxedge.md](02-voxedge.md) | voxedge (engine + backend impls) | **MIDDLE** | main @ b783037 |
| 3 | [03-tensorrt-edge-llm-fork.md](03-tensorrt-edge-llm-fork.md) | TensorRT-Edge-LLM fork (C++ runtime + Python export) | **BOTTOM** | **port/qwen3-tts-base-v080 @ 873ca22** (+ int4 drivers wip/native-int4-talker @ ff2318e) |
| 4 | [04-jetson-voice-engine.md](04-jetson-voice-engine.md) | jetson-voice-engine (Jetson overlay/workers) | **BOTTOM** | main @ 3750ea9 (UPSTREAM_PIN=v0.7.1) |
| 5 | [05-rkvoice.md](05-rkvoice.md) | rkvoice-stream / rkvoice-engine (RK NPU) | **BOTTOM** | main @ 76e9ded / 1f133f3 |

## 3-layer dependency graph

```
 TOP     seeed-local-voice
         ├─ server/            (FastAPI WS/HTTP service)
         │    server/core/_ASR_REGISTRY, _TTS_REGISTRY  ──lazy import──┐
         │    server/core/voxedge_backend_config.build_*_config()       │
         │    server/main.py  ── run_turn, WebSocketTransport ──┐       │
         └─ agent/ovs_agent/   ("OpenVoiceStream" device agent) │       │
              tools/runner.py  ── run_turn ───────────────┐     │       │
                                                          ▼     ▼       ▼
 MIDDLE  voxedge
         ├─ engine/turn_driver.run_turn   (shared server-loop + agent pump)
         ├─ transport/base                (Transport / InProcessTransport / WebSocketTransport)
         ├─ capabilities/{punctuation,speaker_embedding}   (re-exported by seeed)
         └─ backends/
              base.py (ASRBackend/TTSBackend ABCs)
              jetson/trt_edge_llm_tts.py  ── spawns ──┐
              jetson/{matcha,kokoro,qwen3,moss_tts_nano,paraformer,sensevoice}_trt.py
              sherpa/{asr,tts}            rk/{asr,tts,runtime,artifacts}
                                                       │
                                                       ▼ (JSON-line worker protocol, worker_io.py)
 BOTTOM  TensorRT-Edge-LLM fork  @ port/qwen3-tts-base-v080  (873ca22)  ← SOURCE OF TRUTH
         ├─ cpp/ → edgellmCore.a (STATIC) + edgellmKernels + NvInfer_edgellm_plugin.so
         │    runtime/Qwen3OmniTTSRuntime, LLMInferenceRuntime, StreamChannel, SlotPool<TSlot>, HybridCacheManager
         ├─ examples/omni/qwen3_tts_streaming_worker  (links edgellmCore; the spawned binary)
         └─ tensorrt_edgellm/scripts/{export,quantize}.py  (+ int4 drivers on wip/native-int4-talker)

 BOTTOM  jetson-voice-engine (main, UPSTREAM_PIN=v0.7.1)  — parallel encoding of the same features:
         native/edgellm_voice_worker/{qwen3_tts,qwen3_asr}_worker.cpp  (IMPORT prebuilt edgellmCore.a)
         engine-overlay/patches/0001..0008  +  patches/product/fp8-embedding  (= native in fork v0.8.0)

 BOTTOM  rkvoice-stream / rkvoice-engine (RK rail; NOT through voxedge.jetson):
         seeed rk.asr/rk.tts → server/core/rk_runtime + voxedge/backends/rk  →  rkvoice-stream RKLLM runtime
```

## Where the layers physically meet
1. **TOP→MIDDLE (Python imports):** seeed `_ASR/_TTS_REGISTRY` values = `voxedge.backends.*` dotted paths,
   imported lazily; `agent/ovs_agent/tools/runner.py` + `server/main.py` import `voxedge.engine.turn_driver.run_turn`.
2. **MIDDLE→BOTTOM (process boundary):** voxedge `trt_edge_llm_tts.py`/`trt_edge_llm_asr.py` **spawn the C++
   worker binary** and speak the JSON-line protocol (`worker_io.py` ↔ `examples/omni/qwen3_tts_streaming_worker.cpp`).
   No shared address space — this is the clean cut for swapping in the v0.8.0 worker.
3. **BOTTOM build:** `edgellmCore` (STATIC) is the single linked artifact; both the fork's omni worker AND
   jetson-voice-engine's native workers link it (fork: as a target; jetson-voice-engine: as IMPORTED prebuilt).

## Findings that bear on the consolidation plan
- **`port/qwen3-tts-base-v080` == `wip/fp8-embedding`** (both 873ca22): fp8 text-embedding is already on the
  base port head — no separate fp8 branch to merge.
- **The int4-talker export drivers are isolated on `wip/native-int4-talker` (ff2318e)** — `quantize_talker_stage1.py`,
  `stage2_export.py`, `cp_stage2_export.py` at repo root, NOT in `tensorrt_edgellm/scripts/`, and NOT on the
  base-v080 branch. The int4 *kernels/plugins* (`int4GroupwiseGemm*`) ARE on base-v080. So int4-talker
  reproducibility depends on a side WIP branch → reproducibility risk flagged in the plan is real & located.
- **jetson-voice-engine is still pinned to upstream v0.7.1** while the v0.8.0 runtime is in the fork — its 8
  numbered patches + fp8 product patches are the v0.7.1 encoding of features that are now native in v0.8.0.
- **Three worker source copies** of the same concept: fork `examples/omni/qwen3_tts_streaming_worker.cpp`,
  jetson-voice-engine `native/edgellm_voice_worker/*`, and seeed `deploy/asr-worker-v080/qwen3_asr_worker.cpp`.
- **Two-layer ABCs**: seeed `server/core/{asr,tts}_backend.py` define facade ABCs that resolve into voxedge's
  impl-base ABCs in `backends/base.py` — intentional but a dedup candidate.
- **RK asymmetry**: jetson/cpu backends are static `(module,class)` tuples in the registry; RK is resolved only
  via `build_rk_*_config` + a dedicated `rk_runtime`/`voxedge.backends.rk` path — normalize during consolidation.

## Methods / tools used
- Python (projects 1,2): stdlib `ast` via a custom extractor (`/tmp/ast_extract.py`) — class bases + method
  signatures + import edges. (griffe/pydeps not needed; stdlib ast sufficed.)
- C++ (project 3): **no libclang and no universal-ctags on host** (only BSD ctags) → header declaration parsing
  (regex over class/struct/method decls) + CMake target reading. Stated per guardrails.
- Refs confirmed with `git branch -a` + `git log --oneline`; trees read at-ref via `git archive | tar -x`
  into `/tmp/edge-voice-src/` (working tree never switched).
