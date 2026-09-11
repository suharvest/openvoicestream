> Path note (post-restructure): the product service moved `app/`→`server/`
> (`app/main.py`→`server/main.py`, `app/core/`→`server/core/`). Backend
> implementations cited below as `app/backends/...` (jetson/rk/cpu) now live in the
> `voxedge` package (`voxedge.backends.*`); those `app/backends/...` paths
> are kept verbatim only to preserve the original line-anchored references — map
> them to the corresponding `voxedge` module when implementing.

# Week 3 Production Hardening Spec
Status: design only.
Date: 2026-05-24.
Scope: `seeed-local-voice` production hardening after Week 1 and Week 2.
Do not implement code from this document without a follow-up task.
This document is intentionally detailed enough to become an implementation brief.
It is not a patch plan for this turn.
Week 1 already shipped API-key auth, global concurrency cap, and `livez` / `readyz`.
Week 2 already shipped Prometheus `/metrics`, GPU watchdog, and JSON logging.
Main currently has 9 commits and 108 tests passing, per task context.
The `qwen3_trt` TTS N=2 path is the reference stability gate.
Reference qwen3 result from task context:
- 0 CUDA errors across 30+ sustained dual-client bursts.
- Audio MD5 byte-identical before and after stress.
- TTFA ratio N=2/N=1 <= 1.5x.
- Commit range `ff92458..8d6d1b2`, dated 2026-05-22.
Additional shipped context:
- `matcha_trt` unload VRAM release verified in `2967688`.
- `kokoro_trt` hot reload verified in `65d25f4` and `e651fce`.
- `trt_edge_llm_tts` module-level env staleness fix landed in `f009d9a`.
- `moss_tts_nano` TRT C++ path reached production on 2026-05-24 in `3c6c263`.
- `moss_tts_nano` current TTFA baseline from context is 157 ms.
- TTS cooperative cancel Part D disconnect watcher remains deferred.
- C++ cancel code remains dormant.
Explicit non-goals:
- Do not implement ASR worker pooling.
- Do not implement Part D disconnect watcher.
- Do not implement Grafana dashboards.
- Do not broaden this into a general performance rewrite.
- Do not add pyproject dependencies unless the implementation task proves they are unavoidable.
- Do not make automatic per-commit CI hardware gates.
Global performance constraint:
- The harness code must not add more than 5% runtime overhead to the backend path it measures.
- Measurement should run outside the hot request path where possible.
- Per-request instrumentation must be client-side timing and response inspection.
- Server-side changes for measurement are out of scope unless a later implementation task explicitly allows them.
Files read for this spec:
- `bench/perf/` top-level files were listed.
- The first 80 lines of each top-level file in `bench/perf/` were inspected.
- `bench/perf/.DS_Store` is present and binary; it is not a usable source file.
- `bench/perf/smoke_moss_tts_backend.py` exists and was read.
- `app/backends/jetson/moss_tts_nano.py` was read in full.
- `server/core/backend_manager.py` was read for lifecycle and request gating.
- `server/core/tts_backend.py` was read for backend registration and capability contracts.
- `server/main.py` was also inspected because `max_workers` env handling is there, not in `backend_manager.py`.
- `app/backends/jetson/kokoro_trt.py` was inspected for N=2 shared-state surfaces.
- `app/backends/jetson/matcha_trt.py` was inspected for N=2 shared-state surfaces.
- `bench/perf/corpus/manifest.json` and `bench/perf/corpus/tts_prompts.json` were inspected for corpus design.
- `configs/profiles/jetson-moss-tts-nano-trt.json` and `configs/profiles/jetson-moss-tts-nano.json` were inspected for MOSS slot settings.
Important source discovery:
- No top-level `bench/perf/corpus_zh.txt` exists in this checkout.
- No top-level `bench/perf/corpus_en.txt` exists in this checkout.
- The available prompt corpus is `bench/perf/corpus/tts_prompts.json`.
- The available ASR WAV corpus manifest is `bench/perf/corpus/manifest.json`.
- Baseline result files exist under `bench/perf/results/`.
- Existing results include device-specific directories such as `_from_orin-nano`, `_from_harvest-pi`, and `_from_cat-remote`.
## Deliverable 1: kokoro_trt + matcha_trt N=2 Stability Gate
### Overview
Deliverable 1 applies the qwen3 N=2 production gate to `kokoro_trt` and `matcha_trt`.
The gate is run before assuming any bug exists.
The implementation phase should create two focused scripts:
- `bench/perf/stability_kokoro_n2.py`
- `bench/perf/stability_matcha_n2.py`
Both scripts should reuse existing perf client patterns.
They should use the same semantics as qwen3's accepted gate:
- Capture a clean N=1 baseline.
- Capture a pre-stress audio MD5.
- Run 30 or more sustained dual-client bursts.
- Confirm 0 CUDA errors.
- Capture a post-stress audio MD5.
- Compare pre-stress and post-stress audio bytes.
- Compute TTFA ratio N=2/N=1.
- Pass only when TTFA ratio is <= 1.5x.
The gate is a measurement and stability gate first.
Only if the gate fails should an implementation task investigate backend fixes.
The scripts should be backend-specific mostly in naming and profile expectations.
They should not fork large amounts of client code.
They should call into `bench/perf/client.py` where possible.
They can reuse the simple request pattern from `bench/perf/load_2client_tts.py`.
The existing `load_2client_tts.py` states the acceptance ratio clearly: concurrent TTFA p50 must be <= 1.5x the single-client baseline at lines `bench/perf/load_2client_tts.py:3` through `bench/perf/load_2client_tts.py:7`.
The existing two-client harness already uses `ThreadPoolExecutor(max_workers=N)` at `bench/perf/load_2client_tts.py:46`.
It measures first audio past the 4-byte sample-rate header at `bench/perf/load_2client_tts.py:26` through `bench/perf/load_2client_tts.py:39`.
The new stability scripts should preserve that first-audio definition.
The new stability scripts should add sustained burst loops, MD5 checks, and CUDA error scanning.
They should emit one JSON result per run.
They should also emit a short Markdown summary for operator review.
Expected output fields:
- backend name.
- target base URL.
- profile name if discoverable.
- warmup count.
- N=1 TTFA p50 and p95.
- N=2 TTFA p50 and p95 per client.
- TTFA ratio N=2/N=1.
- burst count.
- HTTP error count.
- CUDA error count.
- pre-stress audio MD5.
- post-stress audio MD5.
- pass/fail with reasons.
Fallback strategy:
- For Kokoro, set `OVS_TTS_STREAM_MAX_WORKERS_KOKORO=1`.
- For Matcha, set `OVS_TTS_STREAM_MAX_WORKERS_MATCHA=1`.
- If backend-specific env names are not yet wired, use global `OVS_TTS_STREAM_MAX_WORKERS=1`.
- The spec intentionally names backend-specific env vars so production can force one backend to single-slot without muting qwen3's N=2 path.
The implementation task must decide whether the backend-specific env vars are added in `server/main.py` or in profile application.
The design preference is to resolve backend-specific override in one place near executor creation.
That keeps fallback behavior visible beside existing `OVS_TTS_STREAM_MAX_WORKERS` handling.
### Affected Files
Existing file: `server/main.py`.
- `_get_tts_stream_executor()` is where stream executor workers are configured, not `BackendManager`.
- Existing N=2 comments and fallback guidance live at `server/main.py:296` through `server/main.py:335`.
- The current env read is `int(os.environ.get("OVS_TTS_STREAM_MAX_WORKERS", "2"))` at `server/main.py:331` through `server/main.py:333`.
- Prefetch defaults mirror `executor._max_workers` at `server/main.py:1166` through `server/main.py:1190`.
- The N=2 investigation comment mentions CUDA illegal memory access at `server/main.py:299` through `server/main.py:305`.
- The qwen3 context comment says default `max_workers=2` and fallback `OVS_TTS_STREAM_MAX_WORKERS=1` at `server/main.py:318` through `server/main.py:330`.
Existing file: `server/core/backend_manager.py`.
- `BackendManager` lifecycle owner starts at `server/core/backend_manager.py:102`.
- The manager builds and preloads the backend at `server/core/backend_manager.py:151` through `server/core/backend_manager.py:176`.
- Request gating uses `acquire()` at `server/core/backend_manager.py:236` through `server/core/backend_manager.py:240`.
- Status includes `inflight_http` and `inflight_ws` at `server/core/backend_manager.py:225` through `server/core/backend_manager.py:232`.
- No `max_workers` env var handling was found in this file.
Existing file: `server/core/tts_backend.py`.
- TTS capability enum is defined at `server/core/tts_backend.py:19` through `server/core/tts_backend.py:27`.
- The base backend contract defines `synthesize()` at `server/core/tts_backend.py:85` through `server/core/tts_backend.py:96`.
- Streaming default contract is at `server/core/tts_backend.py:116` through `server/core/tts_backend.py:120`.
- Backend registry includes `jetson.matcha_trt` at `server/core/tts_backend.py:132`.
- Backend registry includes `jetson.kokoro_trt` at `server/core/tts_backend.py:133`.
- Factory lookup and lazy import happen at `server/core/tts_backend.py:146` through `server/core/tts_backend.py:161`.
Existing file: `bench/perf/client.py`.
- It is the shared perf instrumentation source of truth at `bench/perf/client.py:1` through `bench/perf/client.py:4`.
- It imports `requests` and `websocket-client` at `bench/perf/client.py:11` through `bench/perf/client.py:13`.
- `wav_duration_s()` is available at `bench/perf/client.py:42` through `bench/perf/client.py:44`.
- ASR result timing fields exist at `bench/perf/client.py:60` through `bench/perf/client.py:80`.
Existing file: `bench/perf/load_2client_tts.py`.
- Existing comments define two-client TTFA acceptance at `bench/perf/load_2client_tts.py:1` through `bench/perf/load_2client_tts.py:8`.
- It defines the target `/tts/stream` URL at `bench/perf/load_2client_tts.py:12` through `bench/perf/load_2client_tts.py:14`.
- It uses four simple Chinese texts at `bench/perf/load_2client_tts.py:15` through `bench/perf/load_2client_tts.py:20`.
- It records TTFA from request start to first audio at `bench/perf/load_2client_tts.py:22` through `bench/perf/load_2client_tts.py:43`.
Existing file: `bench/perf/corpus/tts_prompts.json`.
- Prompt corpus declares identical text as the comparison contract at `bench/perf/corpus/tts_prompts.json:1` through `bench/perf/corpus/tts_prompts.json:4`.
- Chinese short prompts are present at `bench/perf/corpus/tts_prompts.json:5` through `bench/perf/corpus/tts_prompts.json:9`.
- Chinese long prompts are present at `bench/perf/corpus/tts_prompts.json:10` through `bench/perf/corpus/tts_prompts.json:14`.
- English short prompts are present at `bench/perf/corpus/tts_prompts.json:15` through `bench/perf/corpus/tts_prompts.json:19`.
- English long prompts are present at `bench/perf/corpus/tts_prompts.json:20` through `bench/perf/corpus/tts_prompts.json:24`.
Existing file: `app/backends/jetson/kokoro_trt.py`.
- `KokoroTRTBackend` starts at `app/backends/jetson/kokoro_trt.py:215`.
- Hot reload is enabled at `app/backends/jetson/kokoro_trt.py:218` through `app/backends/jetson/kokoro_trt.py:224`.
- Mutable shared backend fields are initialized at `app/backends/jetson/kokoro_trt.py:226` through `app/backends/jetson/kokoro_trt.py:252`.
- Capabilities include `BASIC_TTS`, `STREAMING`, and `MULTI_SPEAKER` at `app/backends/jetson/kokoro_trt.py:258` through `app/backends/jetson/kokoro_trt.py:264`.
- `unload()` synchronization and teardown ordering starts at `app/backends/jetson/kokoro_trt.py:273`.
- CUDA pool sync occurs at `app/backends/jetson/kokoro_trt.py:320` through `app/backends/jetson/kokoro_trt.py:326`.
- Context dictionaries are destroyed at `app/backends/jetson/kokoro_trt.py:328` through `app/backends/jetson/kokoro_trt.py:348`.
- Engine dictionaries are destroyed at `app/backends/jetson/kokoro_trt.py:350` through `app/backends/jetson/kokoro_trt.py:370`.
- CUDA pool teardown occurs at `app/backends/jetson/kokoro_trt.py:379` through `app/backends/jetson/kokoro_trt.py:385`.
- Direct engine loading creates a single execution context and CUDA pool at `app/backends/jetson/kokoro_trt.py:453` through `app/backends/jetson/kokoro_trt.py:469`.
- Hybrid mode creates a single prefix context and CUDA pool at `app/backends/jetson/kokoro_trt.py:482` through `app/backends/jetson/kokoro_trt.py:491`.
- Split generator mode creates shared engine/context dictionaries and a shared CUDA pool at `app/backends/jetson/kokoro_trt.py:540` through `app/backends/jetson/kokoro_trt.py:575`.
- Streaming chunks are produced by calling `synthesize()` and slicing WAV PCM at `app/backends/jetson/kokoro_trt.py:779` through `app/backends/jetson/kokoro_trt.py:795`.
- Direct TRT execution binds tensors on the shared context and shared pool at `app/backends/jetson/kokoro_trt.py:915` through `app/backends/jetson/kokoro_trt.py:950`.
- Hybrid TRT execution routes through `_run_trt_engine()` at `app/backends/jetson/kokoro_trt.py:952` through `app/backends/jetson/kokoro_trt.py:964`.
- Split generator uses shared per-stage contexts at `app/backends/jetson/kokoro_trt.py:966` through `app/backends/jetson/kokoro_trt.py:1000`.
- Bucket selection returns shared engine/context dictionaries at `app/backends/jetson/kokoro_trt.py:1002` through `app/backends/jetson/kokoro_trt.py:1010`.
- `_run_trt_context()` uses shared pool allocation, shared context shape binding, and shared stream execution at `app/backends/jetson/kokoro_trt.py:1032` through `app/backends/jetson/kokoro_trt.py:1074`.
Existing file: `app/backends/jetson/matcha_trt.py`.
- `MatchaTRTBackend` starts at `app/backends/jetson/matcha_trt.py:171`.
- Hot reload support is enabled at `app/backends/jetson/matcha_trt.py:179` through `app/backends/jetson/matcha_trt.py:184`.
- Mutable shared backend fields are initialized at `app/backends/jetson/matcha_trt.py:186` through `app/backends/jetson/matcha_trt.py:210`.
- Capabilities include `BASIC_TTS`, `STREAMING`, and `MULTI_LANGUAGE` at `app/backends/jetson/matcha_trt.py:216` through `app/backends/jetson/matcha_trt.py:222`.
- `unload()` sync and teardown ordering starts at `app/backends/jetson/matcha_trt.py:231`.
- CUDA pool sync occurs at `app/backends/jetson/matcha_trt.py:261` through `app/backends/jetson/matcha_trt.py:267`.
- Estimator contexts are destroyed at `app/backends/jetson/matcha_trt.py:269` through `app/backends/jetson/matcha_trt.py:275`.
- Vocos context is destroyed at `app/backends/jetson/matcha_trt.py:277` through `app/backends/jetson/matcha_trt.py:282`.
- Engines are destroyed at `app/backends/jetson/matcha_trt.py:284` through `app/backends/jetson/matcha_trt.py:297`.
- CUDA pool teardown occurs at `app/backends/jetson/matcha_trt.py:303` through `app/backends/jetson/matcha_trt.py:310`.
- Engine loading creates one Vocos context at `app/backends/jetson/matcha_trt.py:426` through `app/backends/jetson/matcha_trt.py:443`.
- Split estimator mode creates shared estimator contexts at `app/backends/jetson/matcha_trt.py:445` through `app/backends/jetson/matcha_trt.py:456`.
- A single shared CUDA pool is created at `app/backends/jetson/matcha_trt.py:462`.
- `synthesize()` reads shared pool at `app/backends/jetson/matcha_trt.py:609`.
- `synthesize()` calls ORT or split TRT acoustic stages at `app/backends/jetson/matcha_trt.py:631` through `app/backends/jetson/matcha_trt.py:638`.
- Vocos binds tensors on a shared context and executes on a shared stream at `app/backends/jetson/matcha_trt.py:659` through `app/backends/jetson/matcha_trt.py:678`.
- Per-request CUDA buffers are freed through shared pool `free_all()` at `app/backends/jetson/matcha_trt.py:708`.
- Streaming delegates to synthesize at `app/backends/jetson/matcha_trt.py:711` through `app/backends/jetson/matcha_trt.py:720`.
- Split estimator TRT binds tensors on a shared context and shared pool at `app/backends/jetson/matcha_trt.py:782` through `app/backends/jetson/matcha_trt.py:809`.
- `CudaMemoryPool` stores a single stream and allocation list at `app/backends/jetson/matcha_trt.py:812` through `app/backends/jetson/matcha_trt.py:825`.
- `allocate()` appends into a shared allocation list at `app/backends/jetson/matcha_trt.py:843` through `app/backends/jetson/matcha_trt.py:851`.
- `free_all()` frees and clears the shared allocation list at `app/backends/jetson/matcha_trt.py:883` through `app/backends/jetson/matcha_trt.py:890`.
- `stream_handle()` returns the one shared stream at `app/backends/jetson/matcha_trt.py:892` through `app/backends/jetson/matcha_trt.py:895`.
### New Modules
New module: `bench/perf/stability_kokoro_n2.py`.
Responsibilities:
- Verify target profile reports or behaves like Kokoro.
- Read prompts from `bench/perf/corpus/tts_prompts.json`.
- Prefer English prompts for Kokoro because Kokoro backend is English-focused.
- Use short and long English prompts for MD5 and TTFA diversity.
- Run N=1 baseline.
- Run pre-stress full-audio capture.
- Run at least 30 N=2 bursts.
- Run post-stress full-audio capture.
- Hash raw full response audio.
- Write JSON and Markdown result files.
- Exit nonzero on gate failure.
New module: `bench/perf/stability_matcha_n2.py`.
Responsibilities:
- Verify target profile reports or behaves like Matcha.
- Read prompts from `bench/perf/corpus/tts_prompts.json`.
- Use both Chinese and English prompts because Matcha capability advertises `MULTI_LANGUAGE`.
- Run N=1 baseline.
- Run pre-stress full-audio capture.
- Run at least 30 N=2 bursts.
- Run post-stress full-audio capture.
- Hash raw full response audio.
- Write JSON and Markdown result files.
- Exit nonzero on gate failure.
Recommended shared helper path if implementation wants to avoid duplication:
- `bench/perf/stability_tts_n2_common.py`
This helper is optional.
If added, it must remain a small local helper and avoid new dependencies.
Suggested CLI options:
- `--base-url`, default `http://localhost:8621`.
- `--bursts`, default `30`.
- `--warmup`, default `3`.
- `--timeout`, default `60`.
- `--output-dir`, default `bench/perf/results`.
- `--prompt-id`, optional fixed prompt for MD5 capture.
- `--lang`, optional prompt filter.
- `--category`, optional prompt filter.
- `--scan-log`, optional path to container or service log for CUDA error scanning.
- `--container`, optional container name for `docker logs`.
- `--fail-on-ratio`, default `1.5`.
Output naming:
- `kokoro_n2_stability_<timestamp>.json`
- `kokoro_n2_stability_<timestamp>.md`
- `matcha_n2_stability_<timestamp>.json`
- `matcha_n2_stability_<timestamp>.md`
JSON schema:
- `backend`: string.
- `base_url`: string.
- `started_at`: ISO timestamp.
- `ended_at`: ISO timestamp.
- `bursts_requested`: integer.
- `bursts_completed`: integer.
- `n1`: object.
- `n2`: object.
- `md5`: object.
- `errors`: array.
- `cuda_error_count`: integer.
- `pass`: boolean.
- `failure_reasons`: array of strings.
N=1 object:
- `ttfa_ms`: array.
- `p50_ms`: number.
- `p95_ms`: number.
- `errors`: integer.
N=2 object:
- `pairs`: array.
- `client0_p50_ms`: number.
- `client1_p50_ms`: number.
- `combined_p50_ms`: number.
- `combined_p95_ms`: number.
- `ratio_vs_n1_p50`: number.
MD5 object:
- `prompt_id`: string.
- `pre`: string.
- `post`: string.
- `stable`: boolean.
- `bytes_pre`: integer.
- `bytes_post`: integer.
Markdown report sections:
- Summary.
- Gate result.
- N=1 TTFA.
- N=2 TTFA.
- MD5 stability.
- Errors and CUDA log scan.
- Reproduction command.
### Test Plan
Step 1: confirm the target service is using the intended backend.
Step 2: run a single smoke request through `/tts/stream`.
Step 3: run N=1 baseline with at least 10 measured requests.
Step 4: compute N=1 TTFA p50 after warmup.
Step 5: capture pre-stress full WAV or stream bytes for a fixed prompt.
Step 6: compute pre-stress MD5 on the audio payload.
Step 7: run 30 or more sustained N=2 bursts.
Step 8: every burst sends two concurrent `/tts/stream` requests.
Step 9: each client must receive audio payload beyond the 4-byte sample-rate header.
Step 10: record TTFA for each client in each burst.
Step 11: record wall time for each pair.
Step 12: close each stream cleanly after receiving sufficient audio if using TTFA-only path.
Step 13: periodically run a full-body request to catch post-burst corruption.
Step 14: capture post-stress full audio for the same fixed prompt.
Step 15: compute post-stress MD5.
Step 16: compare pre and post MD5.
Step 17: scan captured HTTP errors.
Step 18: scan captured service logs for CUDA signatures.
Step 19: fail if any CUDA error appears.
Step 20: fail if any full-body MD5 changes.
Step 21: fail if TTFA ratio N=2/N=1 exceeds 1.5x.
Step 22: fail if any request returns non-2xx.
Step 23: fail if any stream returns no PCM data.
Step 24: fail if the service becomes not-ready.
Step 25: print fallback command using backend-specific max-worker env var.
CUDA error patterns:
- `CUDA runtime error`.
- `illegal memory access`.
- `cudaMemcpy`.
- `cudaMemsetAsync`.
- `cudaStreamSynchronize failed`.
- `execute_async_v3 returned False`.
- `CUDNN_STATUS`.
- `TensorRT` errors containing `Error Code`.
The log scanner should count patterns.
It should include surrounding lines in the Markdown report.
It should not parse logs in the request hot path.
The test harness itself should not import TensorRT or CUDA packages.
All GPU observations should be black-box through HTTP and logs.
### Acceptance Criteria
Kokoro passes when:
- At least 30 dual-client bursts complete.
- N=2 combined TTFA p50 divided by N=1 TTFA p50 is <= 1.5.
- CUDA error count is 0.
- HTTP error count is 0.
- Every stream returns PCM data.
- Pre-stress audio MD5 equals post-stress audio MD5.
- Result JSON and Markdown are written.
Matcha passes when:
- At least 30 dual-client bursts complete.
- N=2 combined TTFA p50 divided by N=1 TTFA p50 is <= 1.5.
- CUDA error count is 0.
- HTTP error count is 0.
- Every stream returns PCM data.
- Pre-stress audio MD5 equals post-stress audio MD5.
- Result JSON and Markdown are written.
If either backend fails:
- Do not mark N=2 production-stable.
- Set backend-specific single-slot fallback.
- Preserve logs and result files.
- Open a follow-up implementation task with exact failure evidence.
Fallback acceptance:
- `OVS_TTS_STREAM_MAX_WORKERS_KOKORO=1` forces Kokoro single-slot behavior.
- `OVS_TTS_STREAM_MAX_WORKERS_MATCHA=1` forces Matcha single-slot behavior.
- If backend-specific env support is not implemented yet, document use of `OVS_TTS_STREAM_MAX_WORKERS=1`.
- Fallback must be reversible without rebuilding model artifacts.
### Edge Cases
Kokoro can run in direct engine mode, hybrid mode, split-generator mode, or CPU fallback.
The gate must record runtime mode from API metadata if available.
If runtime mode is unavailable over API, the run report should say so.
Kokoro's English-focused behavior means Chinese prompts may not be a fair stability corpus.
Kokoro's MD5 prompt should be English unless a product profile explicitly supports Chinese Kokoro.
Kokoro chunking depends on `KOKORO_STREAM_CHUNK_MS` at `app/backends/jetson/kokoro_trt.py:785` through `app/backends/jetson/kokoro_trt.py:790`.
Different chunk size can change TTFA measurement.
The script should record `KOKORO_STREAM_CHUNK_MS` when visible in environment or operator metadata.
Matcha supports multilingual prompts.
Matcha token truncation to 80 tokens occurs at `app/backends/jetson/matcha_trt.py:625` through `app/backends/jetson/matcha_trt.py:630`.
Long prompt MD5 should avoid accidental truncation if the goal is deterministic byte comparison.
Streaming endpoints prepend a 4-byte sample-rate header in the current client convention.
MD5 should define whether it hashes header plus PCM or PCM only.
The recommendation is to hash the full response bytes for byte-identical gate and record the choice.
If audio includes nondeterministic WAV metadata timestamps, hash PCM payload instead.
The existing stream path is PCM chunks, so response bytes should be deterministic if backend is deterministic.
HTTP keep-alive reuse can affect TTFA.
The scripts should either consistently reuse sessions or consistently avoid reuse.
The qwen3 gate should be mirrored as closely as possible.
First run after restart may include warmup noise.
Warmups must be discarded.
Thermal throttling can produce false TTFA failures.
The report should include device power mode and temperature if already available through existing commands.
Adding temperature collection must not require new dependencies.
Container logs may rotate.
The script should capture a start timestamp and only scan logs after that point when possible.
If Docker is unavailable, `--scan-log` should support a plain text log path.
If no log source is provided, the gate should warn and fail closed only if the acceptance policy requires log scanning.
Recommended production gate requires a log source.
### Regression Risks
The biggest Kokoro risk is shared TRT context use under concurrent calls.
Kokoro direct mode uses one `_ctx` and one `_pool` at `app/backends/jetson/kokoro_trt.py:915` through `app/backends/jetson/kokoro_trt.py:950`.
Kokoro hybrid mode uses one `_ctx`, one `_engine`, and one `_pool` through `_run_trt_engine()` at `app/backends/jetson/kokoro_trt.py:1026` through `app/backends/jetson/kokoro_trt.py:1030`.
Kokoro split mode shares context dictionaries across requests at `app/backends/jetson/kokoro_trt.py:1012` through `app/backends/jetson/kokoro_trt.py:1024`.
Kokoro `_run_trt_context()` sets tensor addresses on a shared context at `app/backends/jetson/kokoro_trt.py:1036` through `app/backends/jetson/kokoro_trt.py:1043`.
Kokoro `_run_trt_context()` executes on a shared stream at `app/backends/jetson/kokoro_trt.py:1065` through `app/backends/jetson/kokoro_trt.py:1073`.
The biggest Matcha risk is shared Vocos context use under concurrent calls.
Matcha Vocos uses `self._vocos_ctx` shared across calls at `app/backends/jetson/matcha_trt.py:663` through `app/backends/jetson/matcha_trt.py:678`.
Matcha split estimator uses shared contexts at `app/backends/jetson/matcha_trt.py:782` through `app/backends/jetson/matcha_trt.py:809`.
Matcha CudaMemoryPool uses a shared allocation list at `app/backends/jetson/matcha_trt.py:822` through `app/backends/jetson/matcha_trt.py:825`.
Concurrent calls could interleave `allocate()` and `free_all()` on the same list.
`allocate()` appends to the shared list at `app/backends/jetson/matcha_trt.py:843` through `app/backends/jetson/matcha_trt.py:851`.
`free_all()` clears that list at `app/backends/jetson/matcha_trt.py:883` through `app/backends/jetson/matcha_trt.py:890`.
Do not assume those are broken.
They are suspicious shared-state locations to inspect only if the gate fails.
Possible fixes, if failures are found:
- Per-slot execution context pools.
- Per-slot CUDA memory pools.
- Per-slot CUDA streams.
- Backend-local lock fallback.
- Executor-level single-slot fallback.
Do not implement these fixes as part of the initial gate task.
The gate must make the failure reproducible first.
## Deliverable 2: moss_tts_nano N=2 Design
### Overview
Deliverable 2 is design-only.
No code should be implemented for MOSS N=2 in Week 3 unless a separate decision task approves it.
The current MOSS backend is a Python wrapper around a subprocess worker.
The current profile config is single-slot.
`MossTtsNanoBackend.__init__()` reads `moss_max_slots` from the profile and defaults to 1 at `app/backends/jetson/moss_tts_nano.py:64`.
The C++ worker receives `--max-slots` from Python at `app/backends/jetson/moss_tts_nano.py:136` through `app/backends/jetson/moss_tts_nano.py:143`.
The TRT profile sets `MOSS_MAX_SLOTS` to `"1"` at `configs/profiles/jetson-moss-tts-nano-trt.json:21`.
The TRT profile sets `moss_max_slots` to `1` at `configs/profiles/jetson-moss-tts-nano-trt.json:28`.
The ORT fallback profile sets `MOSS_MAX_SLOTS` to `"1"` at `configs/profiles/jetson-moss-tts-nano.json:24`.
The ORT fallback profile sets `moss_max_slots` to `1` at `configs/profiles/jetson-moss-tts-nano.json:31`.
Current production context:
- C++ TRT path is production as of 2026-05-24.
- TTFA baseline is 157 ms.
- Current backend uses `_max_slots=1`.
- Current backend uses one subprocess worker.
- Current backend uses JSONL over stdio.
The design question:
- Is N=2 on Orin Nano 8GB worth doing?
- If yes, which architecture should be explored first?
The answer is cautious:
- N=2 is not obviously worth doing for MOSS on Orin Nano 8GB.
- TTFA 157 ms is already within interactive latency.
- The memory budget is the primary constraint.
- N=2 should be pursued only if customer workload requires simultaneous TTS streams or pipeline overlap.
- The first experiment should be a minimum viable POC, not production implementation.
Recommended approach:
- Approach C: Python-side multi-slot plus worker IPC demux is the recommended POC path.
- It should be treated as a POC only.
- It validates whether the existing single C++ worker can actually host two logical in-flight requests.
- It avoids doubling model weights immediately.
- It exposes protocol and state bugs before committing to a larger C++ redesign.
If Approach C proves the C++ worker cannot safely demux concurrent slots, stop.
Then either keep single-slot or consider Approach A only on larger devices.
Approach B is the cleanest long-term architecture but the highest implementation risk.
Approach A is operationally simple but likely too expensive for Orin Nano 8GB.
### Affected Files
Existing file: `app/backends/jetson/moss_tts_nano.py`.
- File risk notes describe JSONL over stdio at `app/backends/jetson/moss_tts_nano.py:3` through `app/backends/jetson/moss_tts_nano.py:8`.
- Risk notes say malformed worker output may surface as timeouts at `app/backends/jetson/moss_tts_nano.py:3` through `app/backends/jetson/moss_tts_nano.py:6`.
- Risk notes say streaming concurrency depends on worker request ids at `app/backends/jetson/moss_tts_nano.py:7` through `app/backends/jetson/moss_tts_nano.py:8`.
- Risk notes warn respawn is best effort after GPU/TensorRT init failures at `app/backends/jetson/moss_tts_nano.py:9` through `app/backends/jetson/moss_tts_nano.py:10`.
- Worker paths are resolved from env at `app/backends/jetson/moss_tts_nano.py:81` through `app/backends/jetson/moss_tts_nano.py:87`.
- Backend name is `jetson.moss_tts_nano` at `app/backends/jetson/moss_tts_nano.py:89` through `app/backends/jetson/moss_tts_nano.py:92`.
- Capabilities include `BASIC_TTS`, `STREAMING`, `VOICE_CLONE`, and `MULTI_LANGUAGE` at `app/backends/jetson/moss_tts_nano.py:93` through `app/backends/jetson/moss_tts_nano.py:100`.
- `is_ready()` checks a single process at `app/backends/jetson/moss_tts_nano.py:106` through `app/backends/jetson/moss_tts_nano.py:108`.
- `preload()` starts one worker process at `app/backends/jetson/moss_tts_nano.py:110` through `app/backends/jetson/moss_tts_nano.py:158`.
- Python worker path and C++ worker path diverge at `app/backends/jetson/moss_tts_nano.py:120` through `app/backends/jetson/moss_tts_nano.py:143`.
- C++ path passes `--max-slots` at `app/backends/jetson/moss_tts_nano.py:136` through `app/backends/jetson/moss_tts_nano.py:143`.
- One stdout reader and one stderr reader are started at `app/backends/jetson/moss_tts_nano.py:159` through `app/backends/jetson/moss_tts_nano.py:172`.
- Startup waits for `worker_ready` at `app/backends/jetson/moss_tts_nano.py:174` through `app/backends/jetson/moss_tts_nano.py:196`.
- `shutdown()` clears all request queues and stops the single process at `app/backends/jetson/moss_tts_nano.py:198` through `app/backends/jetson/moss_tts_nano.py:209`.
- `generate_streaming()` creates one UUID per request at `app/backends/jetson/moss_tts_nano.py:214` through `app/backends/jetson/moss_tts_nano.py:218`.
- `generate_streaming()` registers a queue and sends one request at `app/backends/jetson/moss_tts_nano.py:221` through `app/backends/jetson/moss_tts_nano.py:227`.
- Chunk events are decoded from base64 at `app/backends/jetson/moss_tts_nano.py:241` through `app/backends/jetson/moss_tts_nano.py:256`.
- Done metadata is stored in thread-local state at `app/backends/jetson/moss_tts_nano.py:258` through `app/backends/jetson/moss_tts_nano.py:261`.
- Worker errors and exits are surfaced at `app/backends/jetson/moss_tts_nano.py:262` through `app/backends/jetson/moss_tts_nano.py:267`.
- Retry respawns the worker once on process or timeout failures at `app/backends/jetson/moss_tts_nano.py:271` through `app/backends/jetson/moss_tts_nano.py:277`.
- `synthesize()` collects streaming PCM into WAV at `app/backends/jetson/moss_tts_nano.py:281` through `app/backends/jetson/moss_tts_nano.py:317`.
- Voice cloning passes base64 reference audio at `app/backends/jetson/moss_tts_nano.py:319` through `app/backends/jetson/moss_tts_nano.py:340`.
- `_build_request()` includes `id`, `text`, `stream`, transport, format, and chunk frames at `app/backends/jetson/moss_tts_nano.py:348` through `app/backends/jetson/moss_tts_nano.py:359`.
- `_build_request()` currently does not include a slot id at `app/backends/jetson/moss_tts_nano.py:352` through `app/backends/jetson/moss_tts_nano.py:367`.
- `_send_request()` serializes JSON and writes to one process stdin under `_proc_lock` at `app/backends/jetson/moss_tts_nano.py:369` through `app/backends/jetson/moss_tts_nano.py:379`.
- Request queue registration uses `_request_queues` keyed by request id at `app/backends/jetson/moss_tts_nano.py:384` through `app/backends/jetson/moss_tts_nano.py:391`.
- `_stdout_reader()` reads JSONL events from one process stdout at `app/backends/jetson/moss_tts_nano.py:403` through `app/backends/jetson/moss_tts_nano.py:428`.
- `_route_stdout_event()` routes by request id at `app/backends/jetson/moss_tts_nano.py:443` through `app/backends/jetson/moss_tts_nano.py:457`.
- `_publish_worker_exit()` broadcasts worker exit to all live request queues at `app/backends/jetson/moss_tts_nano.py:459` through `app/backends/jetson/moss_tts_nano.py:467`.
- Process termination is single-process at `app/backends/jetson/moss_tts_nano.py:468` through `app/backends/jetson/moss_tts_nano.py:493`.
Existing file: `bench/perf/smoke_moss_tts_backend.py`.
- Smoke test purpose is Python backend to C++ worker to WAV at `bench/perf/smoke_moss_tts_backend.py:2` through `bench/perf/smoke_moss_tts_backend.py:7`.
- Usage is documented at `bench/perf/smoke_moss_tts_backend.py:9` through `bench/perf/smoke_moss_tts_backend.py:13`.
- CLI options include worker, engine, tokenizer, and codec paths at `bench/perf/smoke_moss_tts_backend.py:26` through `bench/perf/smoke_moss_tts_backend.py:38`.
- Environment configuration maps args to MOSS env vars at `bench/perf/smoke_moss_tts_backend.py:41` through `bench/perf/smoke_moss_tts_backend.py:45`.
- The smoke creates `MossTtsNanoBackend({})` at `bench/perf/smoke_moss_tts_backend.py:64` through `bench/perf/smoke_moss_tts_backend.py:68`.
- It preloads backend and prints readiness at `bench/perf/smoke_moss_tts_backend.py:71` through `bench/perf/smoke_moss_tts_backend.py:74`.
- Streaming mode records first chunk TTFA at `bench/perf/smoke_moss_tts_backend.py:82` through `bench/perf/smoke_moss_tts_backend.py:98`.
- It shuts down backend at `bench/perf/smoke_moss_tts_backend.py:112` through `bench/perf/smoke_moss_tts_backend.py:113`.
Existing file: `configs/profiles/jetson-moss-tts-nano-trt.json`.
- Description says C++ worker subprocess handles inference and TTFA is about 157 ms on Orin NX at `configs/profiles/jetson-moss-tts-nano-trt.json:2` through `configs/profiles/jetson-moss-tts-nano-trt.json:3`.
- `tts_backend` is `jetson.moss_tts_nano` at `configs/profiles/jetson-moss-tts-nano-trt.json:5`.
- Execution policy is serialized GPU at `configs/profiles/jetson-moss-tts-nano-trt.json:7` through `configs/profiles/jetson-moss-tts-nano-trt.json:10`.
- C++ worker binary path is configured at `configs/profiles/jetson-moss-tts-nano-trt.json:17`.
- Engine, tokenizer, and codec paths are configured at `configs/profiles/jetson-moss-tts-nano-trt.json:18` through `configs/profiles/jetson-moss-tts-nano-trt.json:20`.
- Current max slots env is 1 at `configs/profiles/jetson-moss-tts-nano-trt.json:21`.
- Current `moss_max_slots` is 1 at `configs/profiles/jetson-moss-tts-nano-trt.json:28`.
### New Modules
No production module is required for this design deliverable.
If a POC is approved later, likely touched paths are:
- `app/backends/jetson/moss_tts_nano.py`
- the C++ worker source path for `moss_tts_nano_worker`
- `configs/profiles/jetson-moss-tts-nano-trt.json`
- `bench/perf/stability_moss_n2.py`
- `bench/perf/smoke_moss_tts_backend.py`
Minimum viable POC path for Approach C:
- Add slot acquisition in Python wrapper.
- Include `"slot": 0` or `"slot": 1` in `_build_request()`.
- Keep request id as primary routing key.
- Pass `--max-slots=2` to C++ worker.
- Update C++ worker protocol parser to accept optional `slot`.
- Ensure every worker event echoes both `id` and `slot`.
- Make C++ worker maintain per-slot request state.
- Preserve one stdout JSONL stream.
- Preserve one stdin JSONL stream.
- Reject a request if Python has no free slot.
- Keep timeout and respawn semantics unchanged for POC.
POC must not change:
- Voice clone semantics.
- WAV conversion.
- Public TTS API schema.
- Backend registration key.
- Default production profile.
### Test Plan
Design-only test plan for future POC:
Step 1: measure current N=1 baseline on Orin Nano 8GB.
Step 2: capture current memory at idle after preload.
Step 3: capture memory during one full synthesis.
Step 4: capture memory after synthesis completes.
Step 5: estimate free headroom under production container config.
Step 6: set experimental profile to `moss_max_slots=2`.
Step 7: run two concurrent short prompts.
Step 8: record TTFA for both clients.
Step 9: record total wall time.
Step 10: record VRAM and RSS peak.
Step 11: run 30 dual-client bursts only if the first five bursts have no worker errors.
Step 12: compare pre/post MD5 on fixed prompt.
Step 13: scan logs for CUDA and TensorRT errors.
Step 14: run voice clone smoke if voice clone is in the profile.
Step 15: confirm unload/shutdown returns worker process cleanly.
Step 16: confirm failed request does not poison the remaining slot.
Step 17: confirm timeout on one slot does not orphan queues.
Step 18: confirm worker exit wakes both live request queues.
Step 19: confirm respawn resets slot state.
Step 20: confirm service readiness after respawn.
Approach A test emphasis:
- Measure model duplication memory.
- Verify two worker processes can preload simultaneously.
- Verify process crash isolation.
- Verify both workers produce byte-identical output for the same prompt.
Approach B test emphasis:
- Verify true internal demux.
- Verify per-slot C++ state isolation.
- Verify cancellation and done events per slot.
- Verify no head-of-line blocking on stdout.
Approach C test emphasis:
- Verify Python slot allocation correctness.
- Verify C++ protocol compatibility.
- Verify event routing by id remains sufficient.
- Verify optional `slot` field does not break old worker.
### Acceptance Criteria
Because this deliverable is design-only, acceptance means:
- The user can decide whether to pursue MOSS N=2.
- The tradeoffs are explicit.
- The recommended POC path is clear.
- The memory risk on Orin Nano 8GB is called out.
- TTFA degradation budget is defined.
Suggested future MOSS N=2 performance acceptance:
- TTFA N=2/N=1 <= 1.5x for Orin NX.
- TTFA N=2/N=1 <= 1.75x for Orin Nano 8GB during POC.
- Absolute Orin Nano N=2 TTFA should stay below 300 ms for the first audio chunk if baseline is 157 ms.
- 0 CUDA errors over 30 dual-client bursts.
- Pre/post MD5 stable.
- Peak memory must leave at least 1.0 GB practical headroom on Orin Nano 8GB.
- If headroom is below 1.0 GB, do not ship N=2 on Orin Nano 8GB.
Why 300 ms for MOSS POC:
- Baseline context is 157 ms.
- A 1.5x ratio gives about 236 ms.
- A POC-only 1.75x ceiling gives about 275 ms.
- 300 ms is a pragmatic upper bound for POC noise on Orin Nano.
- Production should tighten back to <= 1.5x if the architecture is accepted.
Is N=2 worth doing on Orin Nano 8GB?
- Not by default.
- It is worth exploring only if simultaneous TTS users are a real requirement.
- It may be worth doing on Orin NX before Orin Nano.
- Orin Nano 8GB has limited memory headroom for duplicated TRT engines and per-slot buffers.
- A 157 ms TTFA baseline already meets most conversational needs.
- The product benefit of N=2 is throughput and overlap, not single-user latency.
### Edge Cases
Worker stdout is a single ordered JSONL stream.
Concurrent slots can interleave events.
Every event must include request id.
The Python wrapper already drops events without request id at `app/backends/jetson/moss_tts_nano.py:448` through `app/backends/jetson/moss_tts_nano.py:457`.
If C++ emits slot-only events without id, Python cannot route them safely.
The existing startup event `worker_ready` has no request id and routes to the control queue at `app/backends/jetson/moss_tts_nano.py:443` through `app/backends/jetson/moss_tts_nano.py:447`.
That behavior must remain unchanged.
Timeout on one request currently respawns the worker after retry conditions at `app/backends/jetson/moss_tts_nano.py:271` through `app/backends/jetson/moss_tts_nano.py:277`.
With multiple live slots, respawn kills all in-flight slots.
The POC must accept this as a coarse failure model.
Python `_proc_lock` serializes stdin writes at `app/backends/jetson/moss_tts_nano.py:371` through `app/backends/jetson/moss_tts_nano.py:379`.
That is fine for request submission.
It does not serialize worker execution once requests are accepted.
Thread-local metadata is used for the last stream at `app/backends/jetson/moss_tts_nano.py:224` and `app/backends/jetson/moss_tts_nano.py:259` through `app/backends/jetson/moss_tts_nano.py:260`.
That is safe only if each request consumes metadata in its own thread.
The POC must not read another request's `last_stream_metadata`.
Voice clone uses large base64 reference audio at `app/backends/jetson/moss_tts_nano.py:330` through `app/backends/jetson/moss_tts_nano.py:339`.
Two concurrent clone requests can increase stdin payload and memory pressure.
MOSS current worker returns raw PCM without side-channel format per file risk notes at `app/backends/jetson/moss_tts_nano.py:11` through `app/backends/jetson/moss_tts_nano.py:12`.
Slot demux must not change sample rate or channel assumptions.
### Regression Risks
Approach A: Multi-process.
- Complexity: medium.
- Python changes are conceptually straightforward.
- Supervisor logic must manage two subprocesses.
- Request routing must pick a worker.
- Health must aggregate two workers.
- Crash handling must avoid killing healthy worker unnecessarily.
- Latency impact: likely good under two clients if memory does not thrash.
- Memory overhead: high.
- Each process can duplicate TRT engines and worker memory.
- On Orin Nano 8GB, this may consume unacceptable headroom.
- Risk level: medium-high because memory pressure can cause system instability.
- Benefit: best process isolation.
- Drawback: model duplication.
Approach B: C++ worker internal multi-slot.
- Complexity: high.
- Requires C++ worker scheduler and per-slot state isolation.
- Requires IPC protocol changes.
- Requires careful TensorRT context and CUDA stream ownership.
- Requires demux and backpressure inside worker.
- Latency impact: best theoretical outcome if C++ scheduling is correct.
- Memory overhead: medium.
- Model weights can be shared.
- Per-slot contexts, KV buffers, codec state, and scratch buffers still add memory.
- Risk level: high.
- Benefit: clean production architecture.
- Drawback: largest implementation and debug surface.
Approach C: Python-side multi-slot plus worker IPC demux.
- Complexity: medium.
- Python owns slot ids and admission.
- C++ worker accepts slot id and keeps per-slot state.
- Request id remains routing key.
- Latency impact: potentially good, but single worker stdout and internal worker scheduling can create head-of-line blocking.
- Memory overhead: medium.
- Model weights can remain shared.
- Per-slot C++ state and buffers are still required.
- Risk level: medium-high.
- Benefit: fastest way to learn whether shared-worker N=2 is viable.
- Drawback: easy to produce a half-measure if C++ internals are not actually per-slot safe.
Recommendation rationale:
- Approach A is too memory-expensive as first choice for Orin Nano 8GB.
- Approach B is too large for a Week 3 hardening cycle.
- Approach C is the right POC because it gives the fastest signal with less memory duplication.
- Approach C must be gated hard before production.
- If Approach C cannot pass the qwen3-style gate, do not keep iterating blindly.
## Deliverable 3: Multi-Device E2E Parity Test Harness
### Overview
Deliverable 3 designs a parity harness across Jetson, RK3576/RK3588, and Raspberry Pi plus Hailo.
Goal:
- The same API calls should produce functionally equivalent results across device classes.
This is not a benchmark leaderboard.
It is a regression gate.
It answers:
- Does ASR still return the expected text quality?
- Does TTS return usable audio within device budget?
- Does V2V preserve key interaction behavior?
- Which backend or device exceeded budget?
Planned new paths:
- `bench/parity/run_parity.py`
- `bench/parity/results/<device>/<timestamp>.json`
- `scripts/parity_gate.sh`
The harness has two modes:
- Mac-runnable mock mode.
- Real-hardware remote mode.
Mock mode:
- Runs without fleet access.
- Reads local fixture JSON.
- Exercises comparison and report generation.
- Does not claim device pass/fail.
Remote mode:
- Uses `fleet match` to discover targets.
- Uses `fleet exec` to run commands remotely.
- Pulls or prints result JSON per device.
- Produces a cross-device comparison table.
CI hook:
- Manual trigger only.
- Not automatic per commit.
- Suitable for nightly or pre-release hardware validation.
Comparison dimensions:
- ASR.
- TTS.
- V2V.
ASR gates:
- CER gate <= baseline + 5%.
- Final text completeness.
- Partial count order-of-magnitude.
TTS gates:
- Synthesis success rate 100%.
- Audio duration vs RTF budget.
- PCM data present.
V2V gates:
- Barge-in response latency.
- Stop intent behavior.
- Empty-final reconnect behavior.
Device budgets from provided context:
- Orin Nano TTFA: 200-500 ms.
- Orin NX TTFA: 150-300 ms.
- RK3576 TTFA: 300-1500 ms.
These budgets are explicitly from task context.
They are not inferred from current result files.
### Affected Files
Existing file: `bench/perf/README.md`.
- The perf harness is cross-device and reproducible at `bench/perf/README.md:1` through `bench/perf/README.md:3`.
- Existing scenarios list ASR, TTS, V2V, concurrent, clone, noise, stability, boot, and matrix at `bench/perf/README.md:5` through `bench/perf/README.md:17`.
- Existing output convention writes `results/<scenario>_<timestamp>.{json,md}` at `bench/perf/README.md:19` through `bench/perf/README.md:22`.
- Setup mentions `websocket-client`, `requests`, and `numpy` at `bench/perf/README.md:35` through `bench/perf/README.md:37`.
- Quick-start commands for ASR, TTS, V2V, concurrent, and matrix are at `bench/perf/README.md:41` through `bench/perf/README.md:60`.
Existing file: `bench/perf/perf.py`.
- Unified harness scenarios and output pattern are documented at `bench/perf/perf.py:1` through `bench/perf/perf.py:10`.
- Shared clients and runners are imported at `bench/perf/perf.py:18` through `bench/perf/perf.py:26`.
- Results directory is `bench/perf/results` at `bench/perf/perf.py:28`.
- Common metadata includes scenario, mode, base URL, warmup, runs, container, client host, and platform at `bench/perf/perf.py:31` through `bench/perf/perf.py:43`.
- Mode inference treats localhost as local and other URLs as remote at `bench/perf/perf.py:46` through `bench/perf/perf.py:50`.
- ASR command loads corpus and writes results at `bench/perf/perf.py:59` through `bench/perf/perf.py:83`.
- TTS command loads prompts and writes results at `bench/perf/perf.py:86` through `bench/perf/perf.py:97`.
- V2V command loads corpus and creates clients at `bench/perf/perf.py:100` through `bench/perf/perf.py:119`.
- Common CLI args include base URL, container, warmup, runs, category, lang, and mode label at `bench/perf/perf.py:329` through `bench/perf/perf.py:345`.
- Concurrent CLI accepts parallel 1, 2, or 4 at `bench/perf/perf.py:383` through `bench/perf/perf.py:389`.
- Stability CLI exists at `bench/perf/perf.py:415` through `bench/perf/perf.py:420`.
Existing file: `bench/perf/runners.py`.
- Quality metric normalization starts at `bench/perf/runners.py:23`.
- Chinese number normalization is optional and no-ops if `cn2an` is missing at `bench/perf/runners.py:31` through `bench/perf/runners.py:58`.
- Error-rate calculation uses CER for zh and WER for other languages at `bench/perf/runners.py:69` through `bench/perf/runners.py:90`.
- Corpus root is `bench/perf/corpus` at `bench/perf/runners.py:97`.
- Corpus loader starts at `bench/perf/runners.py:115` through `bench/perf/runners.py:120`.
- Concurrent runner defines ASR-only, TTS-only, and ASR+TTS modes at `bench/perf/runners.py:320` through `bench/perf/runners.py:327`.
- TTS worker records audio duration, TFD, total, RTF, and wall time at `bench/perf/runners.py:347` through `bench/perf/runners.py:357`.
- Concurrent execution uses `ThreadPoolExecutor` at `bench/perf/runners.py:372` through `bench/perf/runners.py:376`.
- Records include parallel, concurrency mode, EOS mode, and scenario wall time at `bench/perf/runners.py:379` through `bench/perf/runners.py:383`.
Existing file: `bench/perf/corpus/manifest.json`.
- WAV corpus bytes must be bit-identical across devices at `bench/perf/corpus/manifest.json:1` through `bench/perf/corpus/manifest.json:3`.
- Audio spec is 16 kHz mono 16-bit WAV at `bench/perf/corpus/manifest.json:4` through `bench/perf/corpus/manifest.json:9`.
- Categories define short and long at `bench/perf/corpus/manifest.json:10` through `bench/perf/corpus/manifest.json:13`.
- Chinese short entries begin at `bench/perf/corpus/manifest.json:15` through `bench/perf/corpus/manifest.json:58`.
- Chinese long entries begin at `bench/perf/corpus/manifest.json:60` through `bench/perf/corpus/manifest.json:103`.
- English short entries begin at `bench/perf/corpus/manifest.json:105` through `bench/perf/corpus/manifest.json:120`.
Existing file: `bench/perf/corpus/tts_prompts.json`.
- TTS prompt text content is the cross-device contract at `bench/perf/corpus/tts_prompts.json:1` through `bench/perf/corpus/tts_prompts.json:4`.
- There are 10 Chinese prompts at `bench/perf/corpus/tts_prompts.json:5` through `bench/perf/corpus/tts_prompts.json:14`.
- There are 10 English prompts at `bench/perf/corpus/tts_prompts.json:15` through `bench/perf/corpus/tts_prompts.json:24`.
Existing file: `bench/perf/stats.py`.
- Stats rendering defines labels including RTF, TFD, total, CER/WER, and V2V latencies at `bench/perf/stats.py:14` through `bench/perf/stats.py:34`.
- Steady-state filtering is at `bench/perf/stats.py:37` through `bench/perf/stats.py:38`.
- Percentile helper is at `bench/perf/stats.py:41` through `bench/perf/stats.py:47`.
- Summary grouping is at `bench/perf/stats.py:50` through `bench/perf/stats.py:75`.
- Markdown rendering starts at `bench/perf/stats.py:78` through `bench/perf/stats.py:80`.
Existing files under `bench/perf/results/`.
- Baseline data exists under `bench/perf/results/`.
- Device-specific result snapshots exist under `_from_orin-nano`.
- Device-specific result snapshots exist under `_from_harvest-pi`.
- Device-specific result snapshots exist under `_from_cat-remote`.
- Named summary files include `BENCH_20260514.md`, `BENCH_20260515_FINAL.md`, and `V2V_LATENCY_ANALYSIS_20260514.md`.
- The parity harness should reference these as historical baselines when no newer baseline is supplied.
### New Modules
New module: `bench/parity/run_parity.py`.
Responsibilities:
- Parse CLI arguments.
- Load corpus selection.
- Run mock mode or remote mode.
- In remote mode, call `fleet match`.
- In remote mode, call `fleet exec` for each target.
- Store per-device JSON at `bench/parity/results/<device>/<timestamp>.json`.
- Load per-device JSON back into a comparator.
- Produce a cross-device comparison table.
- Flag budgets and regressions.
- Exit nonzero on gate failure.
New script: `scripts/parity_gate.sh`.
Responsibilities:
- Provide one-shot operator entry point.
- Set strict shell flags.
- Select device groups.
- Invoke `bench/parity/run_parity.py`.
- Pass `--mode remote` by default when fleet is available.
- Pass `--mode mock` for local dry run.
- Write report path to stdout.
Optional new path:
- `bench/parity/fixtures/mock/*.json`
Mock fixtures:
- One Jetson-like result.
- One RK3576-like result.
- One RPi+Hailo-like result.
- One intentionally failing fixture for comparator tests.
Output directory:
- `bench/parity/results/<device>/<timestamp>.json`
- `bench/parity/results/summary_<timestamp>.md`
- `bench/parity/results/summary_<timestamp>.json`
Suggested CLI:
- `--mode mock|remote`
- `--devices all|jetson|rk|rpi-hailo|<fleet-selector>`
- `--base-url http://localhost:8621`
- `--runs 10`
- `--warmup 2`
- `--out bench/parity/results`
- `--baseline bench/perf/results`
- `--fail-on-budget`
- `--fail-on-cer-regression`
- `--skip-v2v`
- `--skip-asr`
- `--skip-tts`
Remote command design:
- Use fleet discovery outside Python if the fleet CLI is easier to reason about in shell.
- Or keep all orchestration in Python for one JSON control plane.
- The spec recommends Python orchestration for result collation.
- `scripts/parity_gate.sh` should be a thin wrapper.
No new dependency is required.
Use standard library:
- `argparse`.
- `json`.
- `subprocess`.
- `datetime`.
- `pathlib`.
- `statistics`.
- `textwrap`.
Use existing local modules:
- `bench/perf/client.py`.
- `bench/perf/runners.py`.
- `bench/perf/stats.py`.
### Test Plan
Mock mode test checklist:
- Run `python bench/parity/run_parity.py --mode mock`.
- Verify fixtures load.
- Verify summary JSON is written.
- Verify summary Markdown is written.
- Verify comparison table includes all mock devices.
- Verify a mock budget failure exits nonzero when enabled.
- Verify a mock CER regression is flagged.
- Verify missing optional V2V data is reported as skipped, not crash.
Remote mode test checklist:
- Run `scripts/parity_gate.sh --mode remote`.
- Confirm `fleet match` returns target devices.
- Confirm each target has service URL or SSH command metadata.
- Confirm each target can run perf commands.
- Confirm per-device output is collected.
- Confirm all JSON files include device id, backend ids, and timestamps.
- Confirm comparison table is generated locally.
- Confirm nonzero exit when any hard gate fails.
ASR test checklist:
- Use 10 zh WAV entries and 10 en WAV entries from `bench/perf/corpus/manifest.json`.
- Compute CER for zh and WER for en using existing `compute_error_rate()`.
- Compare current device error rate to baseline.
- Fail when CER/WER exceeds baseline + 5%.
- Check final text completeness.
- Check partial count is within order-of-magnitude of baseline.
- Record missing final text as hard failure.
TTS test checklist:
- Use 10 zh prompts and 10 en prompts from `bench/perf/corpus/tts_prompts.json`.
- Require 100% synthesis success.
- Require PCM data present.
- Record TFD or TTFA.
- Record total wall time.
- Record audio duration.
- Compute RTF when duration is available.
- Compare TTFA against device budget.
- Flag but do not necessarily fail if audio duration differs unless duration is wildly invalid.
V2V test checklist:
- Run barge-in response latency scenario.
- Run stop intent scenario.
- Run empty-final reconnect scenario.
- Record response latency.
- Record whether reconnect was successful.
- Record whether stop intent suppressed further TTS.
- Fail on broken interaction behavior.
Comparison table columns:
- Device.
- Device class.
- Backend ASR.
- Backend TTS.
- ASR zh CER.
- ASR en WER.
- ASR regression flag.
- TTS success rate.
- TTS TTFA p50.
- TTS budget.
- TTS budget flag.
- V2V barge-in latency.
- V2V stop intent.
- V2V reconnect.
- Overall status.
Budget rules:
- Orin Nano TTFA budget: 200-500 ms.
- Orin NX TTFA budget: 150-300 ms.
- RK3576 TTFA budget: 300-1500 ms.
- Unknown device class should warn and not apply TTFA hard budget unless configured.
Baseline rules:
- Prefer an explicit `--baseline` JSON file.
- If absent, scan `bench/perf/results/` for latest matching device summary.
- If no baseline is found, create a baseline candidate but do not fail CER regression.
- Always fail functional missing-output errors.
### Acceptance Criteria
Parity harness accepted when:
- `bench/parity/run_parity.py` can run in mock mode on Mac.
- Mock mode requires no hardware.
- Remote mode can target at least one Jetson and one non-Jetson device through fleet.
- Per-device JSON is written under `bench/parity/results/<device>/`.
- Summary JSON is written.
- Summary Markdown is written.
- Cross-device table is present.
- ASR CER/WER regression flag works.
- TTS TTFA budget flag works.
- V2V behavioral flags work.
- Manual CI hook path is documented.
- No new dependency is added.
Manual CI hook accepted when:
- It is manually triggered.
- It does not run per commit.
- It preserves artifacts.
- It has a clear timeout.
- It prints fleet target list before running.
- It exits nonzero on hard gate failure.
### Edge Cases
Devices may be offline.
Fleet may return stale devices.
A device may be reachable over SSH but service may be down.
A service may be ready but running a fallback backend.
ASR may be absent on TTS-only profiles.
TTS may be absent on ASR-only profiles.
V2V may be unsupported on minimal profiles.
Unknown device classes should not silently pass budgets.
Corpus files may be missing.
`bench/perf/corpus/fetch.py` setup may not have been run.
The harness should fail with a clear corpus setup message.
Different devices may use different sample rates.
PCM-present checks should not assume 48 kHz for all TTS backends.
Duration checks should decode WAV when WAV is returned.
Streaming endpoints may return raw PCM with a sample-rate header.
The harness must record response type.
Partial count can vary by streaming VAD and chunk size.
The gate uses order-of-magnitude, not exact equality.
CER baseline may be absent for a new device.
In that case the harness should report baseline missing and mark regression as unknown.
Mac mock mode must not import fleet-only modules.
Remote mode must preserve command stdout and stderr for debugging.
Log size can be large.
Per-device JSON should include summary metrics, not full audio bytes.
Audio files should be optional artifacts, not embedded in JSON.
### Regression Risks
Fleet orchestration can become flaky if device selection is too broad.
Use explicit selectors for release gates.
Mock mode can drift from real JSON schema.
Keep mock fixtures generated from real result shape when possible.
Parsing old `bench/perf/results/` can be brittle.
Prefer explicit baseline files when a release gate matters.
CER/WER normalization differences can cause false regressions.
Use existing `bench/perf/runners.py` normalization at `bench/perf/runners.py:61` through `bench/perf/runners.py:90`.
TTS duration is not semantic quality.
Do not fail solely on small duration differences.
V2V tests can conflate network latency with device behavior.
Record local vs remote mode like perf.py does at `bench/perf/perf.py:46` through `bench/perf/perf.py:50`.
Manual CI can consume device time.
Keep it opt-in.
Do not make it part of every PR.
## Implementation Sequence
Deliverable 1 comes first.
Reason:
- kokoro/matcha N=2 validation has sufficient existing baseline patterns.
- `qwen3_trt` already defines the accepted gate.
- Existing `bench/perf/load_2client_tts.py` already provides a minimal two-client TTFA pattern.
- The risk is bounded to measurement scripts and optional fallback env wiring.
- It can produce immediate production confidence or immediate rollback guidance.
Deliverable 1 sequence:
1. Add `bench/perf/stability_kokoro_n2.py`.
2. Add `bench/perf/stability_matcha_n2.py`.
3. Reuse existing client and corpus loaders.
4. Run Kokoro N=1 baseline.
5. Run Kokoro N=2 30-burst gate.
6. Run Matcha N=1 baseline.
7. Run Matcha N=2 30-burst gate.
8. Preserve result artifacts.
9. If a backend fails, enable single-slot fallback for that backend.
10. Only then investigate suspicious shared-state locations.
Deliverable 2 is second and remains design-only.
Reason:
- MOSS already has 157 ms TTFA baseline.
- Orin Nano 8GB memory headroom is uncertain.
- N=2 may not justify complexity or memory risk.
- User should decide whether to proceed after seeing the tradeoffs.
Deliverable 2 sequence:
1. Review this design with product and device constraints.
2. Decide whether N=2 is required for MOSS on Orin Nano.
3. If yes, approve a POC task.
4. Start with Approach C POC.
5. Gate the POC using qwen3-style criteria.
6. Stop if memory headroom or stability is unacceptable.
Deliverable 3 can run in parallel with Deliverable 1.
Reason:
- It is a harness and comparator design.
- It does not depend on MOSS N=2.
- It can reuse existing perf corpus and result formats.
- It improves release confidence across device classes.
Deliverable 3 sequence:
1. Add mock-mode comparator first.
2. Add result schema.
3. Add Markdown table rendering.
4. Add fleet remote mode.
5. Add `scripts/parity_gate.sh`.
6. Run one-device remote smoke.
7. Run multi-device remote gate.
8. Add manual CI trigger path.
Final ordering:
1. Deliverable 1: kokoro/matcha N=2 gate.
2. Deliverable 3: parity harness in parallel after Deliverable 1 scripts begin.
3. Deliverable 2: MOSS N=2 decision, then optional POC only if approved.
Scope guard:
- Do not implement ASR worker pooling.
- Do not implement Part D disconnect watcher.
- Do not implement Grafana dashboards.
- Do not change public API payloads for this spec.
- Do not add dependencies without explicit justification.
- Do not make hardware parity automatic per commit.
