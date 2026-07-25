# TensorRT Edge-LLM v0.9.1 production migration

Status: implementation specification  
Date: 2026-07-24  
Branch: `codex/edgellm-v091-upstream-audit`

## Goal

Move the complete Jetson Edge-LLM product stack from the v0.9.0 baseline to
official NVIDIA v0.9.1 plus the smallest currently required local patch set,
build fresh v0.9.1 engines on `orin-nx`, pass the existing model/concurrency
gates, and switch the production services only after a rollback-safe release
candidate passes.

The target is debt reduction across releases. It is not a one-off audit and
does not require eliminating every local product feature in this release.

## Source and build-chain changes

1. `third_party/jetson-voice-engine/engine-overlay/UPSTREAM_PIN:1`
   - pin exact official v0.9.1 SHA
     `7f061f21f0a581ba234a1e233c9315b89d8e47d6`;
   - describe the v0.9.1 minimal series and retained rollback policy.
2. `third_party/jetson-voice-engine/engine-overlay/build.sh:5-155`
   - replace the v0.9.0 patch glob and trailing compatibility patch with the
     ordered `patches/v091-candidate/*.patch` series;
   - fail if the series is empty or non-contiguous;
   - keep exact pinned submodule initialization;
   - print v0.9.1 provenance rather than stale v0.9.0 claims.
3. `third_party/jetson-voice-engine/engine-overlay/README.md`,
   `DIVERGENCE.md`, and patch-state documentation
   - make v0.9.1 the active reproducible chain;
   - retain v0.9.0 material as rollback/history, not as the active build path;
   - distinguish generic upstream candidates from product-specific patches.
4. Deployment/profile files currently named `*-v090*`
   - add v0.9.1 equivalents rather than overwriting rollback profiles;
   - point only v0.9.1 profiles/images at `/opt/edgellm-v091`;
   - never mix v0.9.0 engine/plugin/worker artifacts into a v0.9.1 release
     image.
5. `deploy/docker/` and release scripts
   - introduce a v0.9.1 prefix/overlay image path;
   - bake fresh plugin, workers, engines, sidecars, manifests, and provenance;
   - preserve the current production image and container identities as the
     rollback target until cutover succeeds.
6. Additional active contracts that must migrate together:
   - keep `engine-overlay/upstream.remote` on the canonical NVIDIA remote;
   - replace the three stale v0.7.1 `engine-overlay/manifests/*.toml` manifests
     with exact v0.9.1 build contracts;
   - extend `engine-overlay/build-engines-for-device.sh:70-121` to produce
     explicit ASR b1/b2 and GDN base+MTP artifacts, and cover the Base speaker
     encoder;
   - update `server/tests/test_profile_loader.py:550-608` with v0.9.1 profile
     contracts while preserving v0.8/v0.9.0 rollback tests;
   - commit the inner `third_party/jetson-voice-engine` changes and then update
     the outer submodule pin. A dirty/uncommitted inner tree is not a
     reproducible production migration.
7. The separate `/Users/harvest/project/edge-llm-chat-service` wrapper is part
   of the stack:
   - migrate its runtime/API adaptation from v0.8.0 assumptions to v0.9.1;
   - preserve and retest `_SingleFlightASGIMiddleware` in
     `edge_llm_chat_service/guard.py:619-725`;
   - add readiness/capacity evidence and a regression that forbids HTTP 200
     with an empty SSE stream;
   - build its versioned v0.9.1 image without overwriting the rollback tag.

## Device cleanup boundary

Target: `orin-nx`, selected only through `fleet`.

May delete only from a fresh, explicit allowlist:

- individually named completed audit source/build trees when their hashes/logs
  are already preserved under `/home/harvest/validation/`;
- temporary ONNX/export/build workspaces that are reproducible and not mounted
  by a container;
- Docker build cache and unreferenced audit images/volumes;
- superseded temporary v0.9.1 engine directories after the accepted artifact
  is copied to its final versioned location.

Must preserve:

- production and rollback images, containers, named model volumes, and active
  compose/config files;
- existing v0.8/v0.9.0 engines until v0.9.1 cutover passes;
- dirty source trees;
- `/home/harvest/validation/` evidence;
- at least 4 GiB free at every build checkpoint.

Every deletion target must be resolved to an explicit absolute path and
checked with `realpath`, container mounts, open file descriptors, and an
artifact manifest/hash before removal. Paths containing `accepted`,
`release`, or `rollback` are permanently excluded. Wildcards are not valid
deletion targets.

## Fresh v0.9.1 artifacts

All release-acceptance artifacts must be exported/built from the exact v0.9.1
candidate source, not merely loaded by a v0.9.1 worker:

- Qwen3.5-4B AWQ GDN base;
- Qwen3.5 MTP draft/speculative engine;
- Qwen3-ASR 0.6B INT4, batch 1 and batch 2;
- Qwen3-TTS CustomVoice INT4, batch 2;
- Qwen3-TTS Base FP16, batch 2 plus speaker encoder;
- SparkTTS BF16 and W4A16, batch 2;
- MOSS-TTS-Nano when the missing codec plan can be generated from preserved
  ONNX sidecars;
- SenseVoiceSmall from the canonical scaled ONNX when the source can be
  recovered without changing model semantics.

Each final artifact directory must contain source SHA, patch-series hash,
TensorRT/CUDA/L4T/SM metadata, build command, engine hashes, and required
sidecar hashes.

### Device-preflight build contracts

- CustomVoice INT4 is not an upstream built-in path. The build must require an
  executable reviewed `EDGELLM_TTS_INT4_DRIVER`; its ONNX output and final
  talker engine must carry `DRIVER_REVISION` and `PROVENANCE.md`. Missing
  driver/output is fatal.
- Official v0.9.1 rejects Qwen3-TTS Base checkpoints
  (`tts_model_type='base'`) and has no `tensorrt-edgellm-export-audio`
  speaker-encoder command. Base must require a reviewed
  `EDGELLM_TTS_BASE_DRIVER` for talker/code-predictor/code2wav plus an explicit
  `QWEN3_TTS_BASE_SPEAKER_ENCODER_ONNX_DIR` and source revision. These remain
  local integrations until upstream supports them; historical ONNX cannot be
  relabeled as a v0.9.1 export.
- Initial ASR b1/b2 release engines use the production-equivalent
  `maxInputLen=1024` and `maxKVCacheCapacity=1536`. Any larger context requires
  explicit `ASR_LONG_CONTEXT=1` and remains a separate opt-in artifact.
- SparkTTS BF16 and W4A16 are independent build modes and artifact roots. Each
  requires a reviewed driver, `DRIVER_REVISION`, `PROVENANCE.md`, and a
  `token-gate.json` proving exactly 32 global semantic tokens.
- `audio_build` is a required executable build target, not an optional
  best-effort dependency.
- Before quantization/export, compare the torch CUDA major with the CUDA driver
  API version. A CUDA 13 torch runtime on the JP6.2 CUDA 12.6 driver must fail
  before model loading.

## Runtime gates

1. GDN:
   - deterministic, sampled, and 512-token direct inference;
   - 20 sequential server requests;
   - two overlapping clients with independent non-empty output;
   - 50 abort/immediate-recovery cycles;
   - MTP acceptance counters and performance/memory delta versus vanilla.
   - two-client pass means both clients receive at least one content token and
     `[DONE]`, with zero Myelin/CUDA/TRT/error hits;
   - MTP must accept at least one draft token and report its acceptance ratio;
     throughput must be at least 95% of vanilla and peak unified memory must
     leave at least 2 GiB device headroom. A slower result may remain available
     behind a disabled-by-default flag but cannot become the default.
2. ASR:
   - offline/streaming corpus;
   - distinct-language N=2 with N+1 rejection and slot recovery;
   - 50 paired sessions with zero cross-talk.
   - use `bench/parity/run_parity.py` and the SHA-locked checked-in corpus:
     aggregate CER may regress by at most the script's
     `CER_REGRESSION_PCT=0.05` relative to the checked-in/device v0.9.0
     baseline; streaming LCS versus one-shot must be at least 0.95; no
     transcript may match the other concurrent input.
3. TTS:
   - CustomVoice language/speaker rows and Base voice clone;
   - Spark BF16/W4A16 correctness;
   - N=2 overlap, output isolation, cancel-A/continue-B, immediate recovery,
     and 50-pair stress;
   - CuTe ON versus fallback OFF output quality, latency, memory, and error
     comparison before retiring the local GEMM/GEMV fallback.
   - every output must be non-silent, have a valid sample rate/duration, meet
     `models/common/verify_tts_asr_roundtrip.py` using its checked-in case
     thresholds (`min_similarity` 0.50-0.55), and contain no PCM from the other
     request;
   - PCM must have RMS above 0.01, clipping fraction below 1%, and duration
     within 10% of the same-prompt v0.9.0 baseline;
   - for voice-clone/speaker gates, CAM++ embedding cosine against the
     same-speaker v0.9.0 baseline must be at least 0.95 and at least 0.05 higher
     than the wrong-speaker reference. If the deterministic worker promises
     byte identity, its concurrent output must instead match the solo SHA-256
     exactly;
   - CuTe ON may replace fallback only if all model gates pass with no material
     quality regression: same-prompt ASR LCS may drop by at most 0.05, CAM++
     cosine by at most 0.02, duration ratio must stay in 0.90-1.10, latency may
     regress by no more than 10%, and peak unified memory by no more than
     512 MiB.
4. MOSS/SenseVoice:
   - build missing assets first; otherwise migration is incomplete rather than
     passed.
5. Co-residency:
   - overlap LLM, translator, ASR, and TTS requests;
   - record peak unified memory and maintain at least 2 GiB headroom;
   - zero CUDA, TensorRT, Myelin, assertion, hang, or worker-exit errors.

## Cutover and rollback

- Do not switch production merely because compilation or old-engine ABI smoke
  passes.
- Build a versioned v0.9.1 image and start it on isolated ports/names first.
- Save the exact previous container IDs, image IDs, mounts, env, restart
  policy, and health bodies.
- Before cleanup/cutover, give every rollback image an immutable local rollback
  tag and save `docker inspect`, compose, and effective env copies under the
  migration evidence root.
- Cut over only after all available model gates pass and all remaining blocked
  rows have an explicit user-approved disposition.
- Keep the v0.9.0 image and engine directories until a post-cutover soak passes.
- If GDN simultaneous concurrency remains broken, production must retain an
  explicit process-level singleflight guard. The existing wrapper's
  `_SingleFlightASGIMiddleware` is the required starting point: maximum active
  engine requests is one, wait timeout is bounded, timeout returns structured
  `503 engine_busy`, and the lock releases on completion, disconnect, and
  error. The readiness endpoint must expose worker/capacity state. A
  false-green `/health` response or HTTP 200 empty SSE stream fails cutover.

## Evidence and completion

Write the running result to
`docs/validation/edgellm-v091-production-migration.md`.

Completion requires:

- active v0.9.1 pin and deterministic patch chain;
- clean apply/build from exact official source;
- fresh version-matched release artifacts;
- full runnable model/concurrency gates;
- production service cutover with healthy endpoints;
- rollback evidence;
- local tests, shell syntax checks, and `git diff --check`;
- a separate upstream-PR queue prepared only after migration results are
  stable. No upstream PR is submitted as part of this task.

## Guardrails

- No `git reset --hard` outside the overlay's isolated generated worktree.
- No deletion of production/rollback assets or dirty source trees.
- No Docker volume/image pruning without resolving references first.
- No wildcard-based deletion; use only the reviewed absolute-path allowlist.
- No package/JetPack/TensorRT upgrade.
- No registry push or upstream PR without separate authorization.
- Stop and checkpoint before more than 25 remote shell steps or before any
  deletion outside the explicitly audited temporary paths.
