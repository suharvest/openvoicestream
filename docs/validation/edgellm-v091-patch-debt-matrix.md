# TensorRT Edge-LLM v0.9.1 patch-debt matrix

Date: 2026-07-24  
Scope: static work stream C only  
Official baseline: NVIDIA `v0.9.1`,
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`  
Previous baseline: NVIDIA `v0.9.0`,
`1ac0f2b99642045125e1c5ac7b109434ba3b36c7`

This report does not claim runtime validation. No Orin device, build wiring,
patch file, or upstream repository was modified.

## Executive result

- The active overlay is **not removable as a whole**. No complete active patch
  is semantically demonstrated to have been absorbed by v0.9.1.
- An independent check of each patch against pristine v0.9.1 plus `addon/`
  found **21/38 clean** and **17/38 not independently applicable**. This is
  intentionally a dependency-blind count: failures include patches that are
  incremental on earlier local commits.
- A dependency-aware ordered replay found **31/38 active patches clean** and a
  **seven-patch rebase cluster**:
  `0006`, `0014`, `0017`, `0019`, `0020`, `0023`, and `0031`.
  Five of these produce direct rejected hunks in a reject-assisted replay
  (`0006`, `0017`, `0019`, `0023`, `0031`); `0014` and `0020` are transitively
  coupled to the rejected config hunks and must be rebased with that cluster.
- `0001-orin-tegra-build-compat.patch` conflicts in
  `cmake/CuteDsl.cmake` and `cpp/CMakeLists.txt`. v0.9.1 supports explicit
  `-DEMBEDDED_TARGET=jetson-orin`, but does not absorb Tegra auto-detection or
  all static-library link propagation in the local patch.
- `0002-weight-streaming-budget-v090-OPTIN.patch` still applies cleanly.
  Official `builderUtils.cpp` still has no equivalent
  `kWEIGHT_STREAMING`/4-GiB workspace/tactic-source policy.
- **Pure official v0.9.1 is currently an upgrade blocker on the target
  platform.** A real Orin NX JP6.2 / TensorRT 10.3 compile fails at
  `cpp/common/trtUtils.cpp:155`: v0.9.1 derives a streamed deserializer from
  `nvinfer1::IStreamReaderV2`, but the installed TRT 10.3 headers expose only
  `nvinfer1::IStreamReader`. This is a newly confirmed upstream compatibility
  gap, not a textual overlay conflict. A minimal version-gated continuation is
  being tested separately and is not evaluated in this static report.
- The official packaged SM87 CuTe artifact is also incompatible with the
  target CUDA toolchain. It was produced with CUDA 13.2 / cutlass-dsl 4.6 and
  fails at 34% of a CUDA 12.6 build because `cudaLibrary_t` is unavailable.
  Regenerating the v0.9.1 SM87 artifact locally with cutlass-dsl 4.6 /
  CUDA 12.6 advances the build to 99% and links the plugin, but pybind and
  `llm_inference` still fail with unresolved `cudaLibraryLoadData`,
  `cudaLibraryUnload`, `cudaLibraryGetKernel`, and
  `cudaKernelSetAttributeForDevice`. This is not a successful official CuTe
  path and directly blocks retirement of local patches 0001–0003.
- TensorRT 10.3 also lacks the newer `nvinfer1::DataType::kFP4` API used by
  v0.9.1 source. A TensorRT-version guard is required before the official tree
  can compile on the production baseline.
- A v0.9.1-generated cutlass-dsl 4.5.1 SM87 artifact now provides decisive
  link-diagnosis evidence. It contains all 18 tree/split variants, and its
  4.5.1 metadata makes official CMake enable
  `trt_edgellm_cutedsl_cudart_shim`. The shim correctly defines weak
  `cudaLibrary*` to `cuLibrary*` mappings and `NvInfer_edgellm_plugin` links.
  However, the shim is omitted from the `_edgellm_runtime` and
  `llm_inference` link lines, which leaves their final CUDA library symbols
  unresolved. This confirms that the generic static-link propagation core of
  `0001-orin-tegra-build-compat.patch` remains required and is strongly
  upstreamable. A clean rebuild with the minimal CMake propagation patch is
  still pending.
- v0.9.1 also introduces an independent SM87 context-FMHA blocker. Compared
  with v0.9.0 it adds two head-size-256 `CUSTOM_MASK` cubins generated for
  CUDA 12.8. Qwen3.5 export correctly records `vision=false`, but
  `FMHAKernelList` eagerly loads every cubin matching dtype and SM while
  ignoring mask type. On JP6.2 this loads the unrelated CUDA-12.8-only custom
  mask images and fails with `INVALID_IMAGE`. A three-file patch that scopes
  cubin load/cache selection by mask compiled successfully, and a fresh Qwen
  engine build completed with `rc=0`. True custom-mask requests remain
  fail-loud on JP6.2. This is a P0 generic upstream candidate; Qwen runtime
  inference remains pending.
- Two active commits are sequence-internal debt:
  `0012` adds a development microbenchmark and `0022` deletes it. Their net
  product delta is zero; both should be omitted from a squashed minimal series.
- Official v0.9.1 now has a substantially rewritten OpenAI/tool-calling server,
  including `experimental/server/tool_calling.py` and upstream tests. Those
  portions of archival patch `0006` are absorbed. The historical client
  disconnect cancellation is **not** absorbed: `api_server.py` has no
  `is_disconnected` watcher and `engine.py` only calls `channel.cancel()` on a
  worker exception, not when the response consumer disconnects.
- The official release adds generic FP8 and Qwen3 CodePredictor support, but
  does **not** implement this overlay's FP8 `text_embedding.safetensors`
  table-plus-scales loader. Active patch `0028` therefore remains distinct.
- Static evidence supports upstream preparation for small generic fixes, but
  not submitting them before device tests. Best first candidates are `0035`
  (destination-dtype checkpoint load), `0025` (empty-prefill guard), the
  generic part of the Orin link fix, and an isolated SSE-disconnect reproducer
  plus cancellation fix.

## Method and evidence limits

The official tree was read from a local Git object store and verified with:

```text
git rev-parse HEAD
7f061f21f0a581ba234a1e233c9315b89d8e47d6

git tag --points-at HEAD
v0.9.1
```

The v0.9.0-to-v0.9.1 release delta is 471 files, 136,974 additions, and 7,661
deletions. NVIDIA published that delta as release commit `f98267f` followed by
merge `7f061f2`, so file/code-path evidence is more precise than trying to
attribute each behavior to a separate public commit.

Applicability was checked on a fresh `git archive` of the exact v0.9.1 SHA,
after copying the 33 `addon/` files, then replaying the 38 patches in numeric
order. A reject-assisted second pass preserved accepted hunks so later
dependencies could be inspected. Reverse-checking the complete patches did
not demonstrate any whole active patch as absorbed. The only reverse-clean
result was `0022`, because it is a deletion of the file introduced by `0012`;
that is net-zero-series evidence, not upstream absorption.

The two applicability views answer different questions:

```text
independent check (each patch vs pristine v0.9.1 + addon): 21 clean / 17 fail
ordered/dependency-aware classification:                  31 clean / 7 rebase
direct rejected-hunk patches in reject-assisted replay:   33 clean / 5 reject
```

The five direct rejects are `0006`, `0017`, `0019`, `0023`, and `0031`.
`0014` and `0020` appear clean only after rejected prerequisite hunks are
partially accepted, so the conservative ordered result counts both in the
seven-patch rebase cluster. Independent failures such as `0003`, `0021`,
`0026`–`0030`, `0032`, `0034`, and `0035` mostly express intended dependency
on earlier patches and are not classified as v0.9.1 conflicts.

Classification meanings are those in the audit specification. `CLEAN` only
means textual apply. `REBASE` means direct or transitive conflict.
`unverified/blocked` is used whenever runtime behavior or source provenance is
insufficient; conflict alone is never treated as invalid.

## Confirmed pure-official build blocker

Hardware evidence from work stream A changes the upgrade decision even though
many local patches apply textually:

```text
target:  Jetson Orin NX, JetPack 6.2, TensorRT 10.3
source:  pure official v0.9.1 @ 7f061f21f0a581ba234a1e233c9315b89d8e47d6
failure: cpp/common/trtUtils.cpp:155
cause:   nvinfer1::IStreamReaderV2 is absent from TRT 10.3 headers;
         TRT 10.3 provides nvinfer1::IStreamReader
```

Primary classification: `still-required-rebase` at the integration level.
There is no existing local patch in the audited 38-patch chain that owns this
new v0.9.1 compatibility gap. Until a version-gated compatibility change
builds and passes runtime deserialization on TRT 10.3, the recommendation
cannot be “upgrade to pure official v0.9.1.” A successful compile alone is not
enough: the compatibility path must deserialize the relevant engines,
propagate read/seek errors correctly, and pass clean startup/shutdown plus the
GDN/server and voice regression gates.

Upstream suitability: **P0**. The fix should be the smallest generic
TensorRT-version-gated adapter, retain `IStreamReaderV2` where the installed
TensorRT actually supports it, and use the TRT 10.3-compatible
`IStreamReader` contract otherwise. It must not encode Jetson-, Seeed-, model-,
or product-specific behavior. The minimal continuation and its hardware test
are being handled separately; this report does not claim that implementation
has passed.

The same hardware build exposed three additional, independent blockers:

```text
official packaged SM87 CuTe artifact:
  producer stack: CUDA 13.2 / cutlass-dsl 4.6
  consumer stack: CUDA 12.6 on Orin NX
  result:         compile fails at 34%; cudaLibrary_t is unavailable

locally regenerated v0.9.1 SM87 CuTe artifact:
  producer stack: CUDA 12.6 / cutlass-dsl 4.6
  result:         reaches 99%; NvInfer_edgellm_plugin links
  final failure:  pybind and llm_inference have unresolved symbols:
                  cudaLibraryLoadData
                  cudaLibraryUnload
                  cudaLibraryGetKernel
                  cudaKernelSetAttributeForDevice

TensorRT headers:
  baseline:       TensorRT 10.3
  failure:        nvinfer1::DataType::kFP4 is unavailable
  requirement:    compile-time TensorRT-version guard
```

Primary classification for both CuTe results: `still-required-rebase` at the
integration level and a **pure-official upgrade blocker**. Rebuilding the
artifact for CUDA 12.6 fixes the generated-code/header mismatch, but it does
not provide the CUDA library API symbols needed by final executables. The
plugin linking is useful localization evidence, not a pass. The unresolved
symbols must be resolved through a generic, version-correct shim/link strategy
and then exercised at runtime. This evidence raises the generic static-target
link-propagation portion of the local Orin compatibility patch to a P0 upstream
candidate.

Primary classification for the `kFP4` reference: `still-required-rebase` and a
pure-official compile blocker on TRT 10.3. This should be a small P1 upstream
candidate: guard FP4-only code with the TensorRT version/API capability while
preserving FP4 unchanged on supported TensorRT releases.

A v0.9.1-generated cutlass-dsl 4.5.1 artifact has now passed the artifact
structure and shim-selection part of the experiment:

- all 18 required tree/split variants are present;
- 4.5.1 metadata activates `trt_edgellm_cutedsl_cudart_shim`;
- the shim supplies weak `cudaLibrary* -> cuLibrary*` mappings;
- `NvInfer_edgellm_plugin` links successfully.

The final diagnosis is therefore no longer blocked: official CMake omits the
shim from `_edgellm_runtime` and `llm_inference`, and those final targets retain
the unresolved symbols. This is concrete static/link evidence that the local
compatibility patch's generic propagation hunk remains necessary.

The minimal CMake propagation continuation itself remains
`unverified/blocked` until a clean rebuild completes. After link success it
must still load and pass correctness/concurrency/performance gates. The 4.5.1
result does not justify pinning an old DSL as the permanent fix; upstream
should propagate the selected compatibility shim to every final consumer.

### Confirmed FMHA eager-load blocker and mask-scoped fix

The v0.9.1 release adds two SM87 context-FMHA cubins that were absent in
v0.9.0:

```text
head size:       256
mask type:       CUSTOM_MASK
producer CUDA:   12.8
target platform: JetPack 6.2 / CUDA 12.6 / SM87
failure:         INVALID_IMAGE
```

The affected Qwen3.5 export is not a vision/custom-mask model
(`vision=false`). The failure occurs because `FMHAKernelList` eagerly
materializes all cubins for a matching dtype and SM without first restricting
the list to the requested mask type. Thus an unsupported, unused custom-mask
image prevents a valid non-custom-mask engine from building.

A minimal three-file load/cache patch now includes mask type in cubin
selection. Confirmed evidence:

- the patch compiles;
- a fresh Qwen engine build finishes successfully with `rc=0`;
- unsupported true-custom-mask use remains fail-loud on JP6.2 rather than
  silently selecting an incorrect kernel.

Primary classification: `still-required-clean` at the v0.9.1 integration
level. This is a generic lazy/scoped kernel-registry correctness fix, not
Qwen-, Seeed-, or product-specific behavior. It is a **P0 upstream candidate**:
the loader should only load/cache cubins eligible for the requested
dtype/SM/mask contract, while preserving explicit failure when the requested
contract itself is unsupported.

The successful engine build now has partial runtime evidence. Direct Qwen3.5
GDN runs pass at 35.10-39.24 tok/s with 2,372-2,379 MiB peak unified memory;
20/20 sequential server requests and 50/50 abort -> immediate-next cycles pass
with a clean interval-log scan. However, two simultaneous clients reproduce
the historical `Myelin ... already loaded binary graph` error and one stream
returns no content. Abort recovery TTFT is also about 354 ms slower than the
identical idle-prompt baseline. GDN+MTP remains unbuilt because an isolated
full base+draft export and two-engine build cannot be safely bounded with
16 GiB free while preserving the 4 GiB restart floor. The matrix therefore
classifies v0.9.1 vanilla GDN direct/serial behavior as validated, but the
server concurrency and production-equivalent MTP upgrade gate as blocked.
Raw evidence is under
`/home/harvest/validation/edgellm-v091-official-20260724T0418Z`
(`117-*` through `140-*`).

## Active 38-patch matrix

Path names are relative to the TensorRT Edge-LLM source root. Confidence is
about the static classification, not runtime correctness.

| # | Feature and affected files | Apply | Primary classification | v0.9.1 evidence | Keep / upstream recommendation | Confidence |
|---|---|---|---|---|---|---|
| 0001 | CuTe-DSL cudart shim for omni executables; `examples/omni/CMakeLists.txt` | CLEAN | `still-required-clean` | A complete 18-variant cutlass-dsl 4.5.1 artifact enables the correct weak-symbol shim and links the plugin, but official CMake omits that shim from `_edgellm_runtime` and `llm_inference` | Retain. Generic final-target shim propagation is confirmed required and is a strong P0 upstream candidate; a clean minimal-patch rebuild and runtime/concurrency gates remain pending. | high |
| 0002 | cuBLAS-free tiled FP16 Talker GEMM; `cpp/kernels/talkerMLPKernels/talkerMLPKernels.cu` | CLEAN | `still-required-clean` | No equivalent non-CuTe fallback in official Talker kernels | Retain as the `ENABLE_CUTE_DSL=OFF` rollback. Retire only if the official SM87 CuTe path passes correctness/stability/perf/memory under full model concurrency. | high |
| 0003 | M=1 warp-per-column Talker GEMV; same file as 0002 | CLEAN | `still-required-clean` | No equivalent local kernel path in v0.9.1 | Same retirement gate as 0002; isolated microbench improvement is insufficient. | high |
| 0004 | ASR lane allocator; new `cpp/runtime/asrStreamingSessionRuntime.{cpp,h}` | CLEAN | `product-specific` | Files do not exist upstream; current consumer is the vendored N>1 worker | Retain only while that worker is supported; move with worker, not generic core. | high |
| 0005 | expose `maxSessionBatchSize()`; `cpp/runtime/llmInferenceRuntime.h` | CLEAN | `product-specific` | No symbol upstream | Retain with 0004/worker; not independently upstreamable. | high |
| 0006 | mixed-precision config switch; `tensorrt_edgellm/config.py` | REBASE | `product-specific` | v0.9.1 rewrote config parsing; no `mixed_precision`/`bf16_residual` key | Rebase as part of one SparkTTS precision patch, not as a historical incremental commit. | high |
| 0007 | `BF16Linear`; `tensorrt_edgellm/models/linear.py` | CLEAN | `product-specific` | Official linear code changed but contains no overlay class/contract | Retain with precision island only; consider generic upstream only with model-independent tests. | high |
| 0008 | BF16 residual / FP16 attention modeling; `models/default/modeling_default.py` | CLEAN | `product-specific` | No matching precision-island switch upstream | Retain for SparkTTS; unsuitable as-is for upstream default model. | high |
| 0009 | destination-declared half dtype loading; `checkpoint/loader.py` | CLEAN | `product-specific` | Official loader still unconditionally maps BF16 source to FP16 | Retain as prerequisite, but supersede its historical form with the more correct 0035 semantics. | medium |
| 0010 | match stray FP32 export weights to consumer dtype; `onnx/export.py` | CLEAN | `product-specific` | No equivalent consumer-aware cast found | Retain with mixed-precision export; upstream only after a generic exporter reproducer. | medium |
| 0011 | BF16-output INT4 GEMV/GEMM; three files under `cpp/kernels/int4GroupwiseGemmKernels/` | CLEAN | `product-specific` | v0.9.1 adds separate CuTe INT4 kernels but not this local fallback API | Retain for non-CuTe SparkTTS path; performance/precision tests required. | high |
| 0012 | development BF16 overflow microbenchmark; `.../bf16_overflow_microbench.cu` | CLEAN | `obsolete` | 0022 deletes the same newly-added file | Drop 0012 and 0022 together from minimal history. | high |
| 0013 | BF16 output dtype in INT4 plugin; `cpp/plugins/int4GroupwiseGemmPlugin/{.cpp,.h}` | CLEAN | `product-specific` | No equivalent plugin field in official source | Retain with 0011; upstream only as one tested INT4 output-dtype feature. | high |
| 0014 | `mixed_precision_with_quant` config key; `config.py` | REBASE | `product-specific` | No key upstream; transitively coupled to 0006 conflict | Fold into rebased SparkTTS config patch. | high |
| 0015 | `AWQLinear` BF16 bridge; `models/linear.py` | CLEAN | `product-specific` | Official AWQ path lacks this bridge | Retain with mixed-precision INT4 path. | high |
| 0016 | INT4 ONNX `output_dtype`; `models/ops.py`, `onnx/dynamo_translations.py`, `onnx/onnx_custom_schemas.py` | CLEAN | `product-specific` | v0.9.1 has many unrelated `output_dtype` uses, but not this W4A16 contract | Retain and fold with 0011/0013/0015; potential generic upstream feature after tests. | medium |
| 0017 | quantizer `exclude_attention`; `quantization/quantization_configs.py`, `quantization/quantize.py`, `scripts/quantize.py` | REBASE | `product-specific` | Quantization pipeline was rewritten and has no matching option | Re-implement only if a fresh SparkTTS calibration proves it necessary. | high |
| 0018 | mixed INT4/BF16 wiring test; `tests/test_mixed_precision_int4.py` | CLEAN | `product-specific` | No equivalent official test | Retain as regression evidence for the rebased feature. | high |
| 0019 | read mixed-precision-with-quant from checkpoint config; `config.py` | REBASE | `product-specific` | Config layout drift; no matching key | Fold with 0006/0014/0020/0023. | high |
| 0020 | ModelOpt `int4_awq` mixed precision; `config.py`, `models/linear.py` | REBASE | `product-specific` | Replays only after partially accepting rejected config hunks; therefore not independently clean | Rebase with the complete precision cluster and rerun export tests. | high |
| 0021 | correct ONNX BF16 enum (7, not UINT8 5); `onnx/dynamo_translations.py` | CLEAN | `still-required-clean` | Official INT4 translation does not contain this local mapping | Retain; good generic upstream candidate with a tiny ONNX type assertion. | high |
| 0022 | remove the 0012-only microbenchmark | CLEAN; reverse-clean | `obsolete` | Net effect with 0012 is zero | Drop both commits from minimal series. | high |
| 0023 | rename config to `bf16_residual`; six implementation files plus `tests/test_bf16_residual_int4.py` | REBASE | `product-specific` | Official has no `bf16_residual`; config file drift causes rejection | Replace 0006–0023 history with coherent feature commits and retained tests. | high |
| 0024 | native streaming TTS worker and `slotPool.h`; `examples/omni/CMakeLists.txt`, `qwen3_tts_streaming_worker.cpp`, `slotPool.h` | CLEAN | `product-specific` | No official worker/slot pool | Retain for product IPC/N=2. Keep out of core upstream proposal unless generalized. | high |
| 0025 | reject empty/N<=0 prefill; `cpp/runtime/qwen3OmniTTSRuntime.cpp` | CLEAN | `still-required-clean` | Official runtime lacks the guard | Retain; strong generic correctness candidate with malformed-request test. | high |
| 0026 | CustomVoice 9-row language conditioning; Talker kernels and `qwen3OmniTTSRuntime.{cpp,h}` | CLEAN | `still-required-clean` | Official supports CustomVoice export but has no `codec_language_id` runtime map | Retain. Upstream candidate because it completes an official model variant, after quality test. | high |
| 0027 | external speaker-embedding conditioning; same four runtime/kernel files | CLEAN | `still-required-clean` | No equivalent speaker embedding request path found | Retain for Base/clone behavior; upstream model-feature candidate after API cleanup. | medium |
| 0028 | FP8 text-embedding table + scales; `qwen3OmniTTSRuntime.{cpp,h}`, `scripts/quantize_text_embedding_fp8.py` | CLEAN | `still-required-clean` | Official loads the first text-embedding tensor directly and passes no scales. v0.9.1 CodePredictor FP8 is a different feature. | Retain; upstream optimization candidate with file-format and quality tests. | high |
| 0029 | JSON worker fields `language` and `speaker_embedding_b64`; streaming worker | CLEAN | `product-specific` | Official has no overlay worker IPC schema | Retain locally; do not upstream as-is. | high |
| 0030 | cooperative per-frame cancel; TTS runtime headers/source and streaming worker | CLEAN | `still-required-clean` | Official TTS runtime has no `shouldCancel` callback | Retain; split generic runtime cancellation from product JSON control before upstreaming. | high |
| 0031 | shared-engine constructors / slot reuse; `code2WavRunner.{cpp,h}`, `engineExecutor.{cpp,h}`, TTS runtime, worker | REBASE | `product-specific` | v0.9.1 changed executor/deserialization ownership; three direct rejected files | Rebase only after N=2 memory/byte-identity test; not safe to forward-port mechanically. | high |
| 0032 | differentiated worker chunking; streaming worker | CLEAN | `product-specific` | No upstream worker policy | Retain locally if latency gate still benefits. | high |
| 0033 | MOSS-TTS-Nano runtime/kernels/legacy worker/tests; eight new files | CLEAN | `still-required-clean` | Model/runtime absent upstream | Retain if MOSS remains a supported product model. Possible large upstream model-support proposal, not a bug-fix PR. | high |
| 0034 | MOSS worker CMake integration; legacy worker deletion plus `examples/omni/{CMakeLists.txt,moss_tts_nano_worker.cpp}` | CLEAN | `still-required-clean` | No official MOSS target | Retain with 0033; upstream only as the same model-support series. | high |
| 0035 | cast FP32 checkpoint tensors to declared half destination; `checkpoint/loader.py`, `tests/python-unittests/test_loader_dtype_cast.py` | CLEAN | `still-required-clean` | Official `_set_tensor` only casts BF16 to FP16 and raw-replaces parameters for FP32 | Retain. Highest-quality small upstream candidate; keep FP32-scale preservation tests. | high |
| 0036 | force Qwen3-ASR `rope_type=mrope`; `scripts/export.py` | CLEAN | `unverified/blocked` | v0.9.1 added broader rope normalization and Qwen3-ASR quantization paths, but the local export hook still applies and semantic equivalence is not proven | Do not delete. Export a fresh ASR checkpoint and inspect generated config/output before deciding absorbed vs required. | medium |
| 0037 | allow Qwen3-TTS Base export; `scripts/export.py` | CLEAN | `still-required-clean` | Official v0.9.1 still calls `p.error` for non-CustomVoice at lines 2584–2587 | Retain for Base. Strong upstream feature candidate with Base pipeline test. | high |
| 0038 | carry `codec_language_id`; `quantization/qwen3_omni.py`, `scripts/export.py` | CLEAN | `still-required-clean` | Official source contains no `codec_language_id` | Retain with 0026; upstream together as CustomVoice completion. | high |

## Non-series patches

| Patch | Apply | Primary classification | Evidence and action | Upstream suitability |
|---|---|---|---|---|
| `0001-orin-tegra-build-compat.patch` | REBASE (`CuteDsl.cmake`, `cpp/CMakeLists.txt`) | `still-required-rebase` | Official now documents explicit `EMBEDDED_TARGET=jetson-orin` and sets SM87, but has no Tegra auto-detect. More decisively, a complete 4.5.1 SM87 artifact activates the correct weak-symbol shim and links the plugin, while official final link lines omit the shim from `_edgellm_runtime`/`llm_inference`. The generic static-link propagation hunk is therefore confirmed load-bearing; clean rebuild with its minimal extraction is pending. Other hunks link cuBLAS/cuBLASLt and filter macOS metadata. | Static/final-target shim propagation: strong P0 upstream candidate. Split it from auto-detect, metadata filtering, and product registrations. |
| `0002-weight-streaming-budget-v090-OPTIN.patch` | CLEAN | `still-required-clean` | Official builder has no 4-GiB workspace cap, `kWEIGHT_STREAMING`, or cuBLAS-only tactic policy. This is build-time policy, not a runtime streaming-budget implementation. It was not used in the serving baseline. Keep opt-in only; validate that limiting tactics does not cause build failure/perf loss. | Potentially useful, but hard-coded policy should become explicit CLI/config knobs before upstream. |
| `0006-server-sse-disconnect-and-openai-api.patch` | REBASE | `still-required-rebase` | Official v0.9.1 absorbs/replaces tool rendering, parsing, tool response validation, streaming tool deltas, engine-dir routing, logprobs/logit-bias, and associated tests. The 50-cycle abort -> next-request stability gate passes, but official `generate_stream()` still does not call `channel.cancel()` when the response consumer disappears; it only joins the worker for up to five seconds. Mean next-request TTFT is 421.09 ms versus 66.60 ms for the identical idle prompt, and the server has no cancellation acknowledgement. Retire all absorbed/tool-specific hunks; re-create only a minimal disconnect watcher, channel cancellation, and server-lifecycle test. | SSE cancellation subset: generic candidate after a stronger long-generation cancellation/compute-release proof, explicitly not to be auto-submitted. Product early-name behavior: do not carry unless official behavior is insufficient. |
| `0007-server-openai-api-docs.patch` | REBASE | `obsolete` | Bound to the old server/cache API. Official experimental-server docs and server were rewritten; the old examples cannot be carried verbatim. | Drop; write new docs only for a retained SSE fix. |
| `0008-build-misc-example-registration.patch` | CLEAN | `product-specific` | Registers local spikes/workers and adds macOS ignore entries. Textual apply does not make absent/archival targets useful. | Keep target registration beside retained addon features; omit archival spikes and separate trivial `.gitignore` cleanup. |
| `spark-cutedsl-tag-from-arch.patch` | not part of v0.9.1 chain | `unverified/blocked` | Packaging/tag override for a prior CuTe artifact workflow; v0.9.1 substantially rewrote `build_cutedsl.py` and ships new aarch64 artifacts. | Re-evaluate only if on-device SM87 artifact generation still needs it. |
| `spark-llm-serialize-syscache-pybind.patch` | not part of v0.9.1 chain | `unverified/blocked` | Historical server/cache binding delta; official pybind/server/runtime cache APIs changed. | Do not forward-port without a current API-level reproducer. |

## `engine-overlay/addon/` inventory (33 files)

Every file below is additive relative to v0.9.1. Additive does not mean it
belongs in a generic upstream patch.

### W8A16 Talker quantization — `product-specific`

Retain while the high-performance Talker artifact is supported. The kernels,
plugin, and transformation scripts are one feature and should not be split
across unrelated patches.

```text
cpp/kernels/w8A16LinearKernels/w8A16Linear.cu
cpp/kernels/w8A16LinearKernels/w8A16Linear.h
cpp/plugins/w8A16LinearPlugin/w8A16LinearPlugin.cpp
cpp/plugins/w8A16LinearPlugin/w8A16LinearPlugin.h
scripts/create_qwen3_vocoder50_wrapper.py
scripts/quantize_onnx_matmul_w8a16.py
scripts/quantize_onnx_matmul_w8a16_awq.py
```

Upstream suitability: not as a Seeed artifact recipe. A generic W8A16 plugin
could be proposed only with independent ONNX/plugin correctness and Orin
performance tests. Confidence: high.

### Qwen3-TTS CodePredictor kernels — `product-specific`

```text
cpp/kernels/qwen3TtsCpKernels/qwen3TtsCpKernels.cu
cpp/kernels/qwen3TtsCpKernels/qwen3TtsCpKernels.h
```

Official v0.9.1 optimizes CodePredictor and adds FP8 paths, but textual/source
inspection does not prove these local kernels obsolete. Compare the generated
engines and per-frame synchronization before removal. Confidence: medium.

### Stateful Code2Wav — `product-specific`

```text
cpp/multimodal/statefulCode2WavRunner.cpp
cpp/multimodal/statefulCode2WavRunner.h
```

This is tied to product streaming and slot-pool behavior. Retain with 0024,
0030, and 0031; validate N=2 byte identity. Confidence: high.

### Deployment recipes and issue records — `product-specific`

```text
docs/deploy-container/README.md
docs/deploy-container/STATUS.md
docs/deploy-container/build_qwen3_awq_on_orin.sh
docs/deploy-container/build_server_bindings_on_orin.sh
docs/deploy-container/export_qwen3_awq_on_wsl.sh
docs/deploy-container/package_qwen3_awq_onnx.sh
docs/deploy-container/qwen_processed_chat_template.json
docs/deploy-container/run_qwen3_awq_inference.sh
docs/deploy-container/serve_qwen3_awq_http.sh
docs/known-issues/qwen35-orin-nx-oom.md
docs/known-issues/w8a16-talker-handoff.md
```

Keep as product documentation, but refresh commands against v0.9.1. Do not
copy deployment-specific scripts into a generic upstream PR. Confidence: high.

### ASR incremental-KV spikes — `unverified/blocked`

```text
examples/llm/spike_m1_append_prefill_embeds.cpp
examples/llm/spike_m2_session_lifecycle.cpp
examples/llm/spike_m35_audio_runner_split.cpp
examples/llm/spike_m36_empirical_lcs.cpp
examples/llm/spike_v080_PORT_NOTES.md
examples/llm/spike_v080_m1_append_prefill.cpp
examples/llm/spike_v080_m2_session_lifecycle.cpp
```

These are documented as non-serving experiments. Move them out of the default
build overlay or archive them after preserving results. They are not evidence
that a v0.9.1 incremental-ASR path works. Confidence: high on non-serving
status; low on algorithm validity.

### Archival server scaffold — mixed, primary `obsolete`

```text
experimental/server/launch.sh
experimental/server/tests/__init__.py
experimental/server/tests/test_cache_messages_branch.py
experimental/server/tests/test_tool_call_stream_parser.py
```

Official v0.9.1 now has its own server entry point, tool-calling module, and
tool/server tests. These old tests target symbols from archival patch 0006.
Drop or rewrite them. Preserve no test merely because it is additive; replace
the group with a focused disconnect/abort regression if the SSE fix is kept.
Confidence: high.

## Vendored `native/edgellm_voice_worker/` inventory (18 files)

Primary classification for the vendored worker as a whole:
`product-specific`. It implements the product process/JSON/PCM boundary and is
not part of NVIDIA v0.9.1. It should remain single-sourced outside the upstream
runtime unless individual workers are redesigned as generic official examples.

### Build and contract

```text
CMakeLists.txt
README.md
```

### Product workers

```text
qwen3_asr_worker.cpp
qwen3_tts_worker.cpp
spark_tts_worker.cpp
```

The ASR worker is the current N>1 integration consumer. The Qwen3 and Spark
workers encode product IPC and artifact layout. Retain, but compile all three
against v0.9.1 before changing the pin. Upstream suitability: low as-is.

### DSP/VAD implementation and vendored KissFFT

```text
audio_vad_split.cpp
audio_vad_split.h
mel_extractor.cpp
mel_extractor.h
kissfft/COPYING
kissfft/_kiss_fft_guts.h
kissfft/kiss_fft.c
kissfft/kiss_fft.h
kissfft/kiss_fft_log.h
kissfft/kiss_fftr.c
kissfft/kiss_fftr.h
```

Retain as a product component. Keep the KissFFT license and avoid proposing
this bundled third-party code as an Edge-LLM change. Official v0.9.1 online GPU
fbank is not automatically a replacement: it is gated by the CuTe build path
and has not been behaviorally compared with these product mel features.

### Native unit tests

```text
tests/test_audio_vad_split.cpp
tests/test_mel_extractor.cpp
```

Retain and run on the migration branch. Upstream suitability: only with a
generic worker/DSP proposal.

## Relevant archived Edge-LLM deltas

These files are not in the active build chain. Their classification answers
whether they should remain part of future migration debt, not whether deleting
the archival record is required.

| Archive | v0.9.1 apply | Primary classification | Recommendation |
|---|---|---|---|
| `v080-0001` audio chunk API | conflict | `unverified/blocked` | Non-serving incremental-ASR experiment; preserve only as design history. |
| `v080-0002` ASR runtime hooks | conflict | `unverified/blocked` | Superseded structurally by v0.9 runtime APIs; requires a new design, not rebase. |
| `v080-0003` streaming spikes | clean alone | `obsolete` | Addon already carries the spike sources; do not duplicate. |
| `v080-0004` single-token guard | conflict/missing prerequisite | `unverified/blocked` | Keep only with the deferred incremental-ASR experiment. |
| `v080-0005` per-lane reset/manager | conflict | `obsolete` | Serving N>1 uses vendored worker plus active 0004/0005, not this old runtime experiment. |
| `v080-0006` continuous batcher | clean alone | `unverified/blocked` | New files applying cleanly do not establish a working v0.9.1 path. |
| `v080-0007` old CustomVoice conditioning | conflict | `obsolete` | Superseded by active 0026/0029/0038. |
| `v080-0008` old TTS CuTe wrap | conflict | `obsolete` | Superseded by active 0001 plus Orin compatibility patch. |
| `v080-0010` batch-lane concurrency | conflict | `unverified/blocked` | Alternative to production slot-pool; no v0.9.1 behavior evidence. |
| `v080-0011` MOSS port | clean on pristine tree | `obsolete` | Superseded and single-sourced by active 0033/0034. |
| `v080-0012` streaming decode hook | conflict | `unverified/blocked` | Deferred incremental-ASR track only. |
| `v080-0024` incremental KV streaming | conflict | `unverified/blocked` | Working-copy experiment; no v0.9.1 evidence. |
| `v080-0025-prefix-cap-BROKEN` | missing non-root target path | `unverified/blocked` | Historical docs call it broken, but raw current behavior is unavailable; never forward-port. |
| `v080-0026-prefix-rollback-WORKS` | missing non-root target path | `unverified/blocked` | Preserve as experiment provenance only. |
| `v080-0027-prefix-rollback-byteexact-final` | missing non-root target path | `unverified/blocked` | Preserve as experiment provenance only. |
| `patches/v080-0021-asr-worker-n2-streaming.patch` | target is outside upstream | `obsolete` | Its product behavior is now represented in the vendored v0.9-adapted worker; do not replay onto NVIDIA source. |
| three `patches/product/edgellm-qwen3-tts-text-embedding-fp8*.patch` variants | old-context/unordered | `obsolete` | Superseded by active 0028, which is the v0.9.0 single source. Keep at most historical provenance. |
| `patches/product/paraformer-eof-fix.patch` | not an Edge-LLM target | `product-specific` | Separate sherpa-onnx/Paraformer issue. Keep out of the Edge-LLM migration series; upstream to sherpa-onnx only with its own EOF reproducer. |

## Special retirement gate for patches 0001–0003

Official CuTe support is adjacent to, but not static proof of replacement for,
the local hardcoded GEMM/GEMV path. Patches 0001–0003 must remain until an
unmodified v0.9.1 comparison on Orin NX demonstrates all of the following:

1. Build `ENABLE_CUTE_DSL=ALL` for SM87 from official v0.9.1 inputs, recording
   whether the shipped artifact or an on-device official
   `kernelSrcs/build_cutedsl.py --gpu_arch sm_87` build is used. No local
   0001/0002/0003 hunk or stale v0.8/v0.9.0 artifact may be loaded.
2. Link and load every relevant official CuTe object successfully on
   JetPack 6.2/CUDA 12.6/TRT 10.3. Patch 0001 cannot retire if the official
   static-target cudart shim/driver/`--wrap=_cudaLaunchKernelEx` propagation
   is insufficient, even if compilation alone succeeds.
   Current evidence fails this gate twice: the packaged CUDA 13.2 artifact
   fails the CUDA 12.6 compile at 34%, while a CUDA 12.6-regenerated artifact
   reaches 99% but leaves four `cudaLibrary*`/`cudaKernel*` symbols unresolved
   in pybind and `llm_inference`.
   The 4.5.1 follow-up localizes the latter failure: its complete 18-variant
   artifact selects a correct weak-symbol shim and the plugin links, but
   official CMake does not propagate that shim into `_edgellm_runtime` or
   `llm_inference`. A minimal propagation rebuild is pending.
3. Compare official CuTe against the current `ENABLE_CUTE_DSL=OFF` plus
   0002/0003 fallback using the same engines, seeds, prompts, TTS text,
   speakers/languages, and power/clock state.
4. Pass model correctness for every existing deployed model/variant:
   Qwen3.5 GDN and GDN+MTP, SenseVoice/Qwen3 ASR as applicable, Qwen3-TTS
   CustomVoice and Base, SparkTTS, and MOSS-TTS-Nano. Use existing golden audio,
   semantic/CER gates, and solo-output byte/quality references.
5. Pass concurrency, not just solo inference: N=2 shared-engine TTS 50-shot
   stress, overlapping ASR+TTS, overlapping Qwen3.5 server requests, and the
   full existing service/model co-residency scenario including `translator`.
   Require zero CUDA, TensorRT, Myelin, assertion, hang, and corruption errors.
6. Show no material regression in TTFT/TTFA, tokens/s or realtime factor,
   tail latency, peak unified memory, and remaining memory headroom. Report
   absolute numbers and deltas; do not retire the fallback based only on a
   kernel microbenchmark.

Only after that gate may 0002/0003 be classified `obsolete`. Patch 0001 has an
independent link-correctness purpose and must be evaluated separately even if
official CuTe outperforms the fallback.

## Proposed minimal-debt shape (subject to runtime gates)

This is a restructuring recommendation, not a new patch series:

1. Drop sequence-only `0012` and `0022`.
2. Replace historical incremental commits `0006`–`0023` with:
   - one SparkTTS BF16-residual model/export patch;
   - one INT4 BF16-output kernel/plugin/ONNX patch;
   - one quantization-policy patch;
   - focused unit tests.
3. Keep ASR N>1 worker integration together (`0004`, `0005`, vendored worker)
   and out of unrelated upstream runtime changes.
4. Keep TTS product worker/slot-pool changes together (`0024`, `0029`,
   `0031`, `0032`, addon stateful Code2Wav).
5. Extract generic candidates as independent commits:
   - `0035` checkpoint destination-dtype cast;
   - `0025` empty-prefill validation;
   - `0021` ONNX BF16 enum assertion/fix;
   - Orin static-link propagation from compatibility patch;
   - SSE disconnect cancellation with a deterministic abort-next-request test;
   - `0037` Base export and `0026`+`0038` CustomVoice language completion,
     each with official-model pipeline tests.
6. Keep W8A16, worker JSON protocol, differentiated chunking, device recipes,
   and bundled DSP code local.
7. Do not delete `0036` or the local CP kernel merely because v0.9.1 added
   adjacent official features. Resolve both through fresh export/engine/output
   comparisons.

## Upstream candidate queue

| Priority | Candidate | Required cleanup/evidence before PR |
|---|---|---|
| P0 | TensorRT 10.3 streamed-deserializer compatibility (`IStreamReaderV2` -> version-gated `IStreamReader`) | Reproduce on clean v0.9.1 with TRT 10.3 headers; add compile-time coverage for old/new TensorRT APIs; deserialize real engines; verify read/seek/error semantics and startup/shutdown. A minimal continuation is under separate test. |
| P0 | mask-scoped context-FMHA cubin load/cache | Upstream the minimal three-file fix with a regression where a non-custom-mask SM87 engine ignores incompatible CUDA-12.8-only CUSTOM_MASK cubins, plus a negative test proving a true unsupported custom-mask request still fails loudly. Compile, fresh Qwen engine build, direct inference, 20 sequential server requests, and 50 abort/recovery cycles pass. Simultaneous server clients still fail in a separate shared-runtime/context path. |
| P0 | CUDA 12.6 / SM87 CuTe artifact and final-target CUDA-library API compatibility | Cover both failures: reject or regenerate incompatible CUDA 13.2 packaged artifacts, then propagate the selected compatibility shim to every final consumer. The 4.5.1 artifact proves all 18 variants and weak mappings are present; `_edgellm_runtime`/`llm_inference` link lines are the confirmed gap. Require clean rebuild, load, runtime kernel launch, and full existing-model concurrency; plugin-only link is insufficient. |
| P0 | FP32 checkpoint tensor -> declared FP16/BF16 destination (`0035`) | Run CPU unit tests on clean v0.9.1; verify FP32 scale buffers remain FP32; reduce patch to loader + tests. |
| P0 | SSE disconnect cancellation (small subset of archival 0006) | The 50-cycle abort -> next-request stability test passes, but official code does not cancel the channel on consumer disconnect and recovery TTFT is +354 ms versus the identical idle prompt. Add a long-running generation reproducer with server-side cancellation acknowledgement and compute-release timing before proposing the minimal fix. Do not submit automatically. |
| P1 | empty TTS prefill guard (`0025`) | Add malformed/empty request unit or runtime test and document error contract. |
| P1 | Qwen3-TTS Base export (`0037`) | Official Base checkpoint export/build/synthesis test; remove product wording from warning. |
| P1 | CustomVoice language map (`0026` + `0038`) | Minimal exporter/runtime patch, language-conditioned audio/ASR semantic test. |
| P0 | Orin static/final-target CuTe shim propagation | Split the now-confirmed CMake propagation fix from auto-detect, metadata filters, and private registrations. Add a link test proving the selected shim reaches `_edgellm_runtime`, pybind, and `llm_inference`; then demonstrate clean JP6.2 load and runtime. This is the focused, strongly upstreamable source-level candidate within the broader artifact/toolchain blocker above. |
| P1 | TensorRT 10.3 `kFP4` capability/version guard | Compile cleanly with TRT 10.3 where `DataType::kFP4` is absent and with a newer TensorRT where it is present; ensure the guard hides only FP4-specific code and does not change supported FP4 behavior. |
| P2 | ONNX BF16 enum fix (`0021`) | Tiny exported-node test proving enum 7 and rejecting enum 5. |
| P2 | FP8 text embedding (`0028`) | Define stable tensor names/scale layout, quantizer test, memory and quality evidence. |
| P2 | cooperative TTS cancellation (`0030`) | Remove product JSON coupling; runtime-level callback test. |
| P3 | MOSS-TTS-Nano support (`0033`/`0034`) | Treat as model-support proposal with ownership, checkpoint, license, accuracy, and CI plan. |

## Qwen3-TTS Base follow-up (2026-07-25)

Fresh export evidence resolves the earlier exporter uncertainty:

- exact official base:
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6`;
- pinned Base model:
  `5d83992436eae1d760afd27aff78a71d676296fc`;
- preserved INT4 stage-2 driver:
  `wip/native-int4-talker@ff2318e66525365b2ed9f55811bf5d2381280ed8`;
- isolated WSL branch/worktree:
  `codex/v091-base-extension`;
- only the non-CustomVoice CLI guard patch was applied for export.

The fresh driver completed all three stages and loaded every external tensor
before running `onnx.checker`:

```text
Talker nodes:                1643
Int4GroupwiseGemmPlugin:      196
AttentionPlugin:               28
Talker model.onnx.data SHA: 0aa0695193a24f3d4a63f06bd50f4568540cb14372521c2bd3242f1b592e8fbe
CP model.onnx.data SHA:     829a59622ed1e521628f268147d34fdad5645c9ef5daebfe383f2f1de94c7603
Code2Wav data SHA:          2e8c88b6145a0e1ec2746e7d8cfbcb88af60547f2487be457866e092a50a34d4
```

This demonstrates that the official v0.9.1 Talker, CodePredictor, and
Code2Wav component exporters already handle the Base checkpoint. The public
CLI's unconditional CustomVoice guard is the sole exporter entry blocker.
The guard patch remains `still-required-clean`, with high confidence.
The current v0.9.1 series names this change `0035`; `0037` above refers to the
older archival numbering.

It is not sufficient by itself for an upstream Base claim: the official C++
runtime still lacks the external speaker-embedding conditioning path and the
release has no speaker-encoder exporter. Runtime patch `0027` in the previous
numbering remains required for Base, while the Base64 worker transport remains
product-specific until strict validation is added. The upstream submission
split is documented in
`docs/specs/edgellm-v091-qwen3-tts-base-upstream-plan.md`.

Fresh Orin NX evidence closes the Base device gate. The four versioned engines
built from the artifacts above, 3 Chinese and 2 English strict ASR roundtrips
passed, and independent Base N=2 completed 50/50 cancel/continue/recovery
rounds with byte-identical isolation and zero CUDA/TensorRT hits. The
speaker-encoder source required a TensorRT 10.3 compatibility transform whose
outputs were bit-exact to the source ONNX at mel lengths 10, 555, and 2000.
External runtime conditioning and strict worker transport therefore remain
real extension debt; the three component exporters do not.

The upstream-neutral guard patch now allows only `base`, retains a hard error
for unknown variants, and contains no local branch or historical version
wording. It is staged in the WSL extension worktree but has not been submitted.

The migration target is now validated as official v0.9.1 plus a smaller,
explicit extension/compatibility stack, not pure official v0.9.1. Fresh GDN
plus MTP, ASR, MOSS, SenseVoice, CustomVoice FP16/INT4, Base INT4, and
SparkTTS BF16/W4A16 device evidence exists. Both Spark variants passed fresh
CuTe builds, N=2, 50-round cancellation/recovery, clone, and strict semantic
roundtrips; the mixed-precision Spark patch remains product-specific debt
rather than an upstream-ready generic change. The remaining load-bearing debt
includes TRT 10.3 compile guards, SM87 CuTe/final-target shim propagation,
FMHA compatibility, voice runtime features, and the server's Myelin
concurrency limitation. MOSS remains product-extension debt, but its
worker-level concurrency gap is closed locally: the two-thread dispatcher and
cooperative cancellation path passed 50/50 Orin NX N=2 rounds. This extension
stays separate from NVIDIA bug-fix PRs because upstream has no MOSS model
target. Product scheduling must still cap Base at N=1 while GDN is active;
independent Base N=2 is validated. These remaining items should be split into
focused upstream changes rather than carried as one historical patch stack.
