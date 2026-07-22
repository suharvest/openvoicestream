# jetson-voice-engine — BOTTOM (Jetson overlay / addon over upstream TRT-Edge-LLM)

**Analyzed ref:** `main` @ **3750ea9**.
**Tool:** directory + CMake/patch inspection.

This repo is the **Jetson-side overlay**: it does NOT vendor the full TRT-Edge-LLM source. It pins an
**upstream commit** and carries OUR features as patches + the native voice workers that link the built `edgellmCore`.

> **Critical consolidation fact:** `engine-overlay/UPSTREAM_PIN` = **`364769036fc83351d9d0aac4cc064a8e56a83178`**
> = NVIDIA TensorRT-Edge-LLM **v0.7.1** (Merge PR #90, dev-release/0.7.1).
> So this overlay is still anchored to v0.7.1. The **v0.8.0 runtime lives in the FORK's
> `port/qwen3-tts-base-v080` branch** (see 03-…), where these same features are now *native*. The overlay's
> our-features-as-patches and the fork's native-v0.8.0 are two parallel encodings of the same capabilities —
> the consolidation should collapse them onto the v0.8.0 fork.

---

## 1. Tree
```
native/edgellm_voice_worker/   the C++ voice workers (link the externally-built edgellmCore)
  qwen3_tts_worker.cpp  qwen3_asr_worker.cpp
  mel_extractor.{cpp,h}  audio_vad_split.{cpp,h}  kissfft/  tests/  CMakeLists.txt  README.md
engine-overlay/                the upstream-overlay machinery
  UPSTREAM_PIN  upstream.remote  build.sh  DIVERGENCE.md  README.md  .gitignore
  patches/   0001..0008 (our features as numbered patches — see §3)
  manifests/ customvoice-v071.toml  qwen3-asr-sm87.toml  qwen3-tts-highperf-sm87.toml
  addon/
patches/product/   edgellm-qwen3-tts-text-embedding-fp8*.patch (3 variants) + paraformer-eof-fix.patch
configs/profiles/  deploy/{artifacts,audio_preprocessing}/  models/{common,kokoro,matcha,moss-tts-nano,paraformer,qwen3}/
scripts/  tests/golden_mels/  docs/{performance,plans,issues,audio-evidence}/
```

## 2. `native/edgellm_voice_worker/CMakeLists.txt` — build-target graph
- `project(jetson_voice_edgellm_workers LANGUAGES C CXX CUDA)`.
- **Requires an externally-built TRT tree**: asserts `${EDGE_LLM_BUILD_DIR}/cpp/libedgellmCore.a` exists.
- `read_edgellm_cache_entry(...)` reads the upstream build's CMakeCache (`ENABLE_CUTE_DSL`, `CUTE_DSL_ARTIFACT_TAG`,
  `EMBEDDED_TARGET`) to stay ABI-consistent with how edgellmCore was built.
- `add_library(edgellmCore STATIC IMPORTED GLOBAL)` → links the prebuilt `libedgellmCore.a`.
- Builds `qwen3_tts_worker` + `qwen3_asr_worker` (+ unit tests for mel_extractor / audio_vad_split).
- Adds CUDA stub lib link options (`-L${CUDA_DIR}/lib/stubs`).

> So: **upstream TRT builds edgellmCore.a; this repo's workers IMPORT and link it.** This mirrors the fork's
> `examples/omni/qwen3_tts_streaming_worker` (which links edgellmCore as a regular target). The two worker
> source families (`deploy/asr-worker-v080/qwen3_asr_worker.cpp` in seeed, `native/edgellm_voice_worker/*` here,
> and `examples/omni/*` in the fork) are **three copies of the worker concept** — a key consolidation target.

## 3. `engine-overlay/patches/` — OUR features as numbered patches (the our-vs-upstream split)
| patch | feature |
|---|---|
| 0001-orin-tegra-build-compat | Orin/Tegra build fixes |
| 0002-weight-streaming-budget | weight-streaming memory budget |
| 0003-asr-streaming-session | streaming ASR session (the #15 prefix/chunk-confirm path) |
| 0004-tts-slotpool-concurrency | **N>1 TTS slot-pool** (native in fork v0.8.0 as `SlotPool`) |
| 0005-customvoice-language-conditioning | CustomVoice language conditioning (v0.7.1 product) |
| 0006-server-sse-disconnect-and-openai-api | SSE-disconnect fix + OpenAI API |
| 0007-server-openai-api-docs | API docs |
| 0008-build-misc-example-registration | example registration / build misc |

`patches/product/` additionally has the **fp8 text-embedding** patches in 3 forms
(`-cpp-only`, `-unified`, plain) + `paraformer-eof-fix`. The fp8 embedding work is **already merged natively**
on the fork's `port/qwen3-tts-base-v080`/`wip/fp8-embedding` (873ca22) — so these product patches are the
v0.7.1-overlay encoding of what is native in v0.8.0.

`engine-overlay/manifests/*.toml` describe the build variants (customvoice-v071, qwen3-asr-sm87, qwen3-tts-highperf-sm87);
`build.sh` is the only sanctioned build entry; `DIVERGENCE.md` documents the delta vs upstream.
