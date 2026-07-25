# TensorRT Edge-LLM v0.9.1 upstream migration audit

## Objective

Validate whether the Jetson Orin NX deployment can move from the current
Qwen3.5 GDN runtime based on TensorRT Edge-LLM v0.8.0 to the official v0.9.1
release, then reduce the local voice-engine patch stack to the smallest set
that is still required.

The preferred result is:

1. use an unmodified official v0.9.1 checkout wherever possible;
2. retain only patches that fix a reproducible product requirement or upstream
   defect;
3. prepare generally useful retained fixes as upstream-ready changes;
4. keep a documented rollback path to the existing v0.8.0 GDN service.

## Baselines

- Official release: TensorRT Edge-LLM `v0.9.1`
  (`7f061f21f0a581ba234a1e233c9315b89d8e47d6`).
- Current voice overlay: v0.9.0
  (`1ac0f2b...`) plus 38 ordered functional patches and
  `0001-orin-tegra-build-compat.patch`.
- Current production GDN baseline on `orin-nx`: image
  `edge-llm-chat-service:v0.8.0-gdn-mtp-merged`.
- Device: Jetson Orin NX 16 GB, JetPack 6.2 / L4T R36.4.3,
  TensorRT 10.3.0.30.
- Reusable model source:
  `/home/harvest/edgellm-workspace/qwen35-4b-awq/hf_src`.

## Safety and rollback

- The user has authorized stopping these containers for validation:
  `edge-llm-chat-service` and `translator`.
- Record their image IDs, mounts, restart policies, ports, and health state
  before changing runtime state.
- Restore the two containers with
  `docker start edge-llm-chat-service translator` after validation, unless the
  user explicitly asks to keep the test runtime active.
- Do not modify, clean, reset, or reuse these dirty source trees:
  `/home/harvest/project/edgellm-v080` and
  `/home/harvest/project/tensorrt-edge-llm`.
- Do not overwrite the existing v0.8 engine directory, Hugging Face model
  source, Docker images, or production compose files.
- Do not run Docker prune, system prune, package upgrades, recursive deletion,
  or remove caches/artifacts without separate approval.
- Use a uniquely named v0.9.1 checkout and uniquely named build/export/engine
  directories. Check free space before every material build stage.
- Do not print credentials, tokens, private registry configuration, or complete
  environment dumps.
- Before each build/export stage, record an estimated artifact budget and free
  space. Keep enough space to restart the original containers and abort the
  stage if the estimate would leave less than 4 GB free. Any deletion, cache
  cleanup, or Docker pruning requires separate user approval.

## Work streams

### A. Pure official v0.9.1 validation

Build and test the exact official v0.9.1 commit without the local patch stack.
Reuse the existing Qwen3.5 model source read-only. Produce fresh export and
engine artifacts in isolated directories.

The GDN/server test must cover:

- record the production v0.8 engine shapes, cache settings, GDN/MTP draft
  configuration, request limits, and server options;
- distinguish vanilla GDN from the production-equivalent GDN+MTP path, and test
  both when v0.9.1 supports both;
- model export and TensorRT engine build;
- server startup and model load;
- one deterministic short request and one normal sampling request;
- at least 20 repeated sequential requests;
- at least two concurrent requests;
- a request long enough to exercise generation beyond initial warm-up;
- reproduce or cite preserved raw evidence for the historical failure:
  abort `/v1/chat/completions` during streaming, immediately issue the next
  request, then check for the Myelin `already loaded binary graph` crash;
- on v0.9.1, run at least 50 abort/immediate-next-request cycles and require the
  server to remain healthy with zero Myelin, CUDA, TensorRT, or assertion
  errors;
- clean server shutdown;
- CUDA/TensorRT errors, assertion failures, hangs, crashes, and unexpected
  synchronization failures in logs;
- startup time, TTFT, output tokens/second, peak device memory, and remaining
  memory headroom using the same prompts and engine settings as v0.8;
- restore and run `translator` concurrently, then require both services to stay
  healthy and functional without OOM.

Record exact commands, commit SHA, build flags, engine metadata, logs, and
pass/fail outcomes. Record request seeds when supported, response/status
validity, request timeouts, and evidence of actual concurrency overlap. Raw
evidence must live under a uniquely named v0.9.1 validation directory. A
successful engine build alone is not sufficient. If performance or memory
regresses versus the existing approximately 35 tok/s and 5.76 GB combined
reference, quantify the delta and make it an explicit upgrade decision item.

Official provenance requires a clean checkout at the exact SHA, initialized
submodules or LFS objects where required, clean build/export directories, and
recorded compiler, CUDA, TensorRT, CMake flags, plugin hash, and runtime library
paths. Logs must demonstrate that no v0.8/v0.9.0 plugin or stale generated
artifact was loaded.

### B. Official feature and regression check

Verify v0.9.1 changes that directly affect this project:

- Qwen3.5 GDN and server stability;
- official packaged sm_87 CuTe artifact behavior on JP6.2/CUDA 12.6, noting
  its recorded build-toolkit version;
- when the packaged artifact is incompatible, a clean v0.9.1 local sm_87 CuTe
  generation using the device-compatible CUDA/cutlass-dsl toolchain;
- official CuTe versus the local hard-coded FP16 GEMM/GEMV fallback, comparing
  output correctness, build/link/load behavior, single-request and concurrent
  stability, latency/throughput, and unified-memory use;
- CodePredictor runtime synchronization changes;
- CodePredictor sampling defaults;
- Qwen3-TTS/Omni CodePredictor FP8 paths where supported;
- online GPU fbank availability and its `ENABLE_CUTE_DSL` constraint;
- streamed engine deserialization and runtime teardown behavior;
- auxiliary-stream and implicit-stream synchronization fixes.

Unsupported features should be marked `not applicable` with the exact hardware,
JetPack, TensorRT, CUDA, or build-flag reason rather than treated as failures.
This exception only applies to optional features. Failure to build or run the
production GDN path on JP6.2/TRT 10.3 is an upgrade blocker.

### C. Full divergence audit

Inventory every local divergence, not only the active patch chain:

- all 38 ordered v0.9.0 functional patches;
- `0001-orin-tegra-build-compat.patch`;
- the opt-in weight-streaming patch;
- all files under `engine-overlay/addon/`;
- vendored `native/edgellm_voice_worker` code;
- archival server SSE/OpenAI patches and their test scaffolding;
- other archived patches that still represent an unresolved product behavior
  or upstream bug.

Group additive files into coherent features, but preserve a file-level inventory
so no delta disappears from the accounting. Each patch or feature group receives
exactly one primary classification:

- `absorbed`: upstream implements the same behavior;
- `still-required-clean`: applies cleanly and its behavior is still needed;
- `still-required-rebase`: needed, but conflicts with v0.9.1;
- `product-specific`: valid but unsuitable for generic upstream;
- `obsolete`: no longer needed or targets a removed path;
- `invalid`: behavior is demonstrated to be incorrect;
- `unverified/blocked`: evidence is insufficient or required artifacts are
  unavailable.

For each patch, record:

- affected feature and files;
- textual apply/reverse-apply result;
- upstream commits or code paths that supersede or conflict with it;
- a behavioral test or concrete static evidence;
- whether it should remain locally;
- upstream suitability, evidence confidence, and any cleanup needed before
  submission.

Textual applicability is evidence, not the final classification.
`invalid` must not be used merely because a test is unavailable. A divergence
may be removed only when `absorbed` or `obsolete` is supported by semantic or
behavioral evidence.

### D. Minimal v0.9.1 integration

Only after the pure-official baseline and classification are complete, create a
minimal patch series for v0.9.1. Preserve patch ordering and document every
retained delta.

Regression scope:

- Qwen3.5 GDN server tests from stream A;
- SenseVoiceSmall ASR build/runtime test against the existing golden clips and
  CER/semantic gate;
- Qwen3-ASR 0.6B int4 streaming and offline modes;
- SparkTTS 0.5B BF16/mixed-precision and W4A16 production paths;
- Qwen3-TTS CustomVoice int4 and Base voice-clone paths;
- MOSS-TTS-Nano;
- any other model found in the deployed Edge-LLM worker manifest, recorded
  explicitly rather than silently omitted;
- the N=2 shared-engine gate: concurrent PCM must match solo output, 50-shot
  stress must have zero CUDA errors, and memory saving should be compared with
  the existing approximately 1284 MB result;
- for every model whose product contract allows concurrency: N=1 baseline,
  maximum configured session count, simultaneous-start overlap, cancellation
  of one session while another continues, repeated 50-shot stress, output
  isolation/correctness, CUDA error scan, peak unified memory, and recovery for
  the next request;
- mixed-service concurrency: supported ASR+TTS pairs, Qwen3.5 server plus the
  voice service, and both together with `translator`, bounded by the production
  session ceilings;
- Jetson JP6.2 build compatibility;
- the existing non-CuTe build path, unless a CuTe-enabled build is proven clean
  on this device.

Local patches `v090-sparktts-0001..0003` may be retired only when the official
CuTe path builds, links, and loads without those patches and the complete
model/concurrency matrix above meets correctness, stability, performance, and
memory gates. A successful CuTe microbenchmark or GDN-only run is insufficient.

## Acceptance criteria

The audit is complete when:

1. pure official v0.9.1 has a reproducible GDN/server pass or a minimal
   reproducible failure with logs and exact commands;
2. every local patch, additive feature group, vendored worker, and relevant
   archived delta has a documented classification and evidence;
3. the minimal retained patch stack applies from a clean v0.9.1 checkout;
4. required GDN, ASR, TTS, N=2, and JP6.2 regression checks have explicit
   results;
5. retained generic fixes are separated into upstream submission candidates;
6. the report recommends one of:
   `upgrade to official v0.9.1`, `upgrade with minimal patches`, or
   `remain on v0.8.0`, with blockers and rollback steps;
7. the original `edge-llm-chat-service` and `translator` containers are
   restored unless the user requests otherwise, with the original container
   IDs/images, mounts, ports, restart policies, and configuration unchanged;
   health checks and ports 8000/9001 must be verified after restoration.

## Expected repository outputs

- This specification.
- A v0.9.1 patch-state matrix beside the existing overlay documentation.
- Updated upstream pin/build wiring only if the regression gate passes.
- Reproduction notes and evidence paths for GDN and voice tests.
- A short list of upstream-ready patches, each scoped to one independently
  reproducible issue.

“Upstream-ready” means a minimal generic commit on a clean official base, with
a reproducer or test, rationale, and no Seeed/product coupling. This task may
prepare candidate commits and PR text, but must not submit an upstream pull
request without separate user authorization.
