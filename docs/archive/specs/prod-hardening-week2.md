# Production Hardening Week 2 Spec

Status: design-only specification.
Scope: Week 2 production hardening.
Source date: 2026-05-24.
Repository: `seeed-local-voice`.

## Source Review Ledger

- Read `server/core/metrics.py`.
- Read `server/core/gpu_watchdog.py`.
- Read `server/core/api_auth.py`.
- Read `server/core/session_limiter.py`.
- Read `server/main.py`, with focus on logging setup, probes, middleware gap, WS entry points, and perf logging hooks.
- Read `server/core/backend_manager.py`, with focus on state machine, reload, rollback, drain, and manager status.
- Read `bench/perf/` tree, including client, runners, CLI, stress harnesses, corpus manifests, results, and timing docs.
- Read `app/requirements.txt`.
- Read `deploy/docker-compose.yml`.
- Read `deploy/docker-compose.radxa.yml`.
- Read `deploy/docker-compose.rk.yml`.
- Read `deploy/docker-compose.rpi.yml`.
- This spec does not re-spec Week 1 behavior already shipped.
- This spec covers exactly three deliverables.

## Week 1 Baseline Not Re-Specified

- Optional API key auth already exists in `server/core/api_auth.py`.
- Global concurrency cap already exists in `server/core/session_limiter.py`.
- In-process counter stub already exists in `server/core/metrics.py`.
- GPU watchdog stub already exists in `server/core/gpu_watchdog.py`.
- `/livez`, `/readyz`, and `/health` are already separated.
- Four Week 1 commits exist on main and are not pushed.
- Week 2 preserves Week 1 public function signatures where stated.
- Week 2 changes are design-only in this document.

## Deliverable 1: Prometheus Metrics Exporter

### 1. Affected Files And Anchors

- `server/core/metrics.py:1` currently identifies the file as a Week 1 in-process stub.
- `server/core/metrics.py:17` defines the module lock used by the stub.
- `server/core/metrics.py:20` stores `_sessions_active`.
- `server/core/metrics.py:23` stores `_sessions_rejected_total`.
- `server/core/metrics.py:24` stores `_auth_rejected_total`.
- `server/core/metrics.py:27` defines `inc_sessions_active()`.
- `server/core/metrics.py:35` defines `dec_sessions_active()`.
- `server/core/metrics.py:47` defines `get_sessions_active()`.
- `server/core/metrics.py:52` defines `inc_sessions_rejected(reason)`.
- `server/core/metrics.py:62` defines `get_sessions_rejected(reason=None)`.
- `server/core/metrics.py:69` defines `inc_auth_rejected(endpoint)`.
- `server/core/metrics.py:75` defines `get_auth_rejected(endpoint=None)`.
- `server/core/metrics.py:82` defines `snapshot()`.
- `server/core/metrics.py:92` defines `_reset_for_tests()`.
- `server/core/session_limiter.py:157` records active session acquire through `metrics.inc_sessions_active()`.
- `server/core/session_limiter.py:167` records active session release through `metrics.dec_sessions_active()`.
- `server/core/session_limiter.py:249` records HTTP session rejection.
- `server/core/session_limiter.py:287` records WS session rejection.
- `server/core/api_auth.py:119` imports metrics on HTTP auth rejection.
- `server/core/api_auth.py:120` records `inc_auth_rejected(request.url.path or "unknown")`.
- `server/core/api_auth.py:140` records manual unauthorized responses.
- `server/core/api_auth.py:184` imports metrics on WS auth rejection.
- `server/core/api_auth.py:185` records `inc_auth_rejected(ws.url.path or "unknown")`.
- `server/main.py:29` configures current text logging.
- `server/main.py:35` creates the FastAPI app.
- `server/main.py:326` registers startup with `@app.on_event("startup")`.
- `server/main.py:342` initializes the session limiter.
- `server/main.py:590` defines `/livez`.
- `server/main.py:601` defines `/readyz`.
- `server/main.py:672` defines `/health`.
- `server/main.py:682` exposes temporary `_WorkerIO` cancel count in `/health`.
- `server/main.py:839` defines `/tts`.
- `server/main.py:845` wraps `/tts` with `acquire_http("/tts")`.
- `server/main.py:863` calls `backend.synthesize()` for manager-backed TTS.
- `server/main.py:878` calls `tts_service.synthesize()` for legacy TTS.
- `server/main.py:887` exposes audio duration response header.
- `server/main.py:888` exposes inference time response header.
- `server/main.py:889` exposes `X-RTF`.
- `server/main.py:899` defines `/tts/stream`.
- `server/main.py:912` imports metrics in `/tts/stream`.
- `server/main.py:920` tries the streaming HTTP session acquire.
- `server/main.py:923` records HTTP streaming rejection.
- `server/main.py:972` starts the manager-backed streaming generator.
- `server/main.py:1004` starts the `/tts/stream` disconnect watcher.
- `server/main.py:1079` defines `_submit()` for sentence streaming.
- `server/main.py:1085` calls `backend.generate_streaming(...)`.
- `server/main.py:1117` dispatches streaming synthesis to the TTS stream executor.
- `server/main.py:1152` observes the first TTS stream chunk per sentence.
- `server/main.py:1193` starts the legacy streaming generator.
- `server/main.py:1245` calls legacy `backend.generate_streaming(...)`.
- `server/main.py:1271` dispatches legacy streaming synthesis to the TTS stream executor.
- `server/main.py:1475` calls clone streaming `backend.generate_streaming(...)`.
- `server/main.py:1511` calls legacy clone streaming `backend.generate_streaming(...)`.
- `server/main.py:1533` defines `/asr`.
- `server/main.py:1552` calls manager-backed `asr_be.transcribe(...)`.
- `server/main.py:1562` calls legacy `asr_be.transcribe(...)`.
- `server/main.py:1576` defines `/asr/stream`.
- `server/main.py:1605` checks WS auth before accept.
- `server/main.py:1608` accepts `/asr/stream`.
- `server/main.py:1611` acquires the WS session slot.
- `server/main.py:1619` creates the WS handle for backend manager tracking.
- `server/main.py:1680` logs ASR stream open.
- `server/main.py:1706` runs `force_endpoint` in ASR executor.
- `server/main.py:1708` runs `stream.prepare_finalize`.
- `server/main.py:1709` runs `stream.finalize`.
- `server/main.py:1728` runs `stream.prepare_finalize` on empty bytes EOS.
- `server/main.py:1729` runs `stream.finalize` on empty bytes EOS.
- `server/main.py:1741` accepts ASR waveform in executor.
- `server/main.py:1751` runs `stream.prepare_finalize` after VAD endpoint.
- `server/main.py:1752` runs `stream.finalize` after VAD endpoint.
- `server/main.py:1813` defines `/v2v/stream`.
- `server/main.py:1854` checks V2V WS auth.
- `server/main.py:1857` accepts V2V WS.
- `server/main.py:1860` acquires V2V WS session.
- `server/main.py:1869` creates the V2V WS handle.
- `server/main.py:1999` logs V2V stream open.
- `server/main.py:2037` creates `tts_q`.
- `server/main.py:2059` defines V2V dispatcher.
- `server/main.py:2080` runs VAD process.
- `server/main.py:2094` cancels current TTS task on speech start.
- `server/main.py:2097` sets the current TTS stop event.
- `server/main.py:2129` accepts V2V ASR audio.
- `server/main.py:2159` enqueues V2V TTS sentences.
- `server/main.py:2164` enqueues flushed V2V TTS sentences.
- `server/main.py:2174` cancels current TTS task on client abort.
- `server/main.py:2177` sets current TTS stop event on client abort.
- `server/main.py:2186` cancels ASR manager on barge-in.
- `server/main.py:2212` polls V2V ASR partials.
- `server/main.py:2257` finalizes V2V ASR with status.
- `server/main.py:2366` calls V2V `synth_be.generate_streaming(...)`.
- `server/main.py:2403` dispatches V2V TTS synth to the TTS stream executor.
- `server/main.py:2425` catches `asyncio.CancelledError` during V2V TTS.
- `server/main.py:2493` cancels ASR manager on WS close.
- `server/main.py:2505` releases the V2V session token.
- `server/core/backend_manager.py:49` defines `BackendState`.
- `server/core/backend_manager.py:81` defines `BackendManager`.
- `server/core/backend_manager.py:129` defines `start()`.
- `server/core/backend_manager.py:151` transitions to READY.
- `server/core/backend_manager.py:154` defines `shutdown()`.
- `server/core/backend_manager.py:157` transitions shutdown to DRAINING.
- `server/core/backend_manager.py:165` transitions shutdown to FAILED.
- `server/core/backend_manager.py:199` defines `status()`.
- `server/core/backend_manager.py:210` defines `acquire()`.
- `server/core/backend_manager.py:218` increments `_inflight_http`.
- `server/core/backend_manager.py:224` decrements `_inflight_http`.
- `server/core/backend_manager.py:229` registers WS handles.
- `server/core/backend_manager.py:232` unregisters WS handles.
- `server/core/backend_manager.py:237` defines `_wait_for_http_drain`.
- `server/core/backend_manager.py:264` defines `_force_close_ws_sessions`.
- `server/core/backend_manager.py:287` defines `reload()`.
- `server/core/backend_manager.py:301` rejects concurrent reloads.
- `server/core/backend_manager.py:309` rejects reload when not READY.
- `server/core/backend_manager.py:375` transitions to DRAINING.
- `server/core/backend_manager.py:378` force-closes WS sessions.
- `server/core/backend_manager.py:379` waits for HTTP drain.
- `server/core/backend_manager.py:387` transitions to RELOADING.
- `server/core/backend_manager.py:411` transitions reload success to READY.
- `server/core/backend_manager.py:421` returns `status: reloaded`.
- `server/core/backend_manager.py:433` catches reload failures.
- `server/core/backend_manager.py:463` transitions rollback success to READY.
- `server/core/backend_manager.py:465` returns `status: rolled_back`.
- `server/core/backend_manager.py:477` transitions rollback failure to FAILED.
- `server/core/backend_manager.py:497` defines `init_backend_managers(...)`.
- `app/requirements.txt:1` starts the dependency list.
- `bench/perf/README.md:9` documents ASR RTF, TFD, and WER/CER.
- `bench/perf/README.md:10` documents TTS RTF and TFD.
- `bench/perf/README.md:141` defines Wall RTF.
- `bench/perf/README.md:142` defines Finalize RTF.
- `bench/perf/README.md:143` defines TFD.
- `bench/perf/README.md:144` defines EOS to first audio.
- `bench/perf/client.py:60` defines `ASRResult`.
- `bench/perf/client.py:65` defines ASR `tfd_ms`.
- `bench/perf/client.py:66` defines ASR `eos_to_final_ms`.
- `bench/perf/client.py:67` defines ASR `rtf`.
- `bench/perf/client.py:70` defines ASR `finalize_rtf`.
- `bench/perf/client.py:76` defines `asr_finalize_compute_ms`.
- `bench/perf/client.py:83` defines `TTSResult`.
- `bench/perf/client.py:87` defines TTS `tfd_ms`.
- `bench/perf/client.py:88` defines TTS `total_ms`.
- `bench/perf/client.py:89` defines TTS `rtf`.
- `bench/perf/client.py:119` defines offline ASR timing.
- `bench/perf/client.py:121` starts offline ASR timer.
- `bench/perf/client.py:130` computes offline ASR processing ms.
- `bench/perf/client.py:133` computes offline ASR RTF.
- `bench/perf/client.py:137` defines streaming ASR timing.
- `bench/perf/client.py:161` starts streaming ASR first-send clock.
- `bench/perf/client.py:189` captures first partial time.
- `bench/perf/client.py:196` captures EOS time.
- `bench/perf/client.py:281` captures final time.
- `bench/perf/client.py:286` computes ASR finalize compute time.
- `bench/perf/client.py:293` computes ASR streaming RTF.
- `bench/perf/client.py:294` computes ASR finalize RTF.
- `bench/perf/client.py:319` starts TTS timer.
- `bench/perf/client.py:333` captures first TTS chunk time.
- `bench/perf/client.py:338` captures TTS end time.
- `bench/perf/client.py:351` computes TTS RTF.
- `bench/perf/client.py:390` composes V2V EOS-to-first-audio.
- `bench/perf/client.py:449` starts V2V stream ASR timing.
- `bench/perf/client.py:477` captures first V2V ASR partial.
- `bench/perf/client.py:483` captures V2V EOS time.
- `bench/perf/client.py:502` captures V2V endpoint time.
- `bench/perf/client.py:504` captures V2V final time.
- `bench/perf/client.py:546` computes endpoint latency.
- `bench/perf/client.py:547` computes ASR finalize latency.
- `bench/perf/load_2client_tts.py:3` documents TTFA measurement.
- `bench/perf/load_2client_tts.py:23` starts TTFA timer.
- `bench/perf/load_2client_tts.py:32` captures first audio time.
- `bench/perf/load_2client_tts.py:43` returns TTFA ms.
- `bench/perf/multi_sentence_pipeline.py:4` documents TTFA.
- `bench/perf/multi_sentence_pipeline.py:35` starts TTFA timer.
- `bench/perf/multi_sentence_pipeline.py:48` detects first PCM beyond header.
- `bench/perf/multi_sentence_pipeline.py:53` stores `ttfa_ms`.
- `bench/perf/stress_cancel_n1.py:50` defines one early-break TTFA iteration.
- `bench/perf/stress_cancel_n1.py:61` returns TTFA for first PCM chunk.
- `bench/perf/stress_cancel_n1.py:209` reports TTFA min.
- `bench/perf/stress_cancel_n1.py:210` reports TTFA p50.
- `bench/perf/stats.py:15` maps `rtf` to Wall RTF.
- `bench/perf/stats.py:16` maps `finalize_rtf`.
- `bench/perf/stats.py:21` maps EOS to first audio.
- `bench/perf/stats.py:24` maps ASR compute.
- `deploy/docker-compose.yml:14` defines the Jetson `speech` service.
- `deploy/docker-compose.yml:28` starts its environment block.
- `deploy/docker-compose.radxa.yml:13` defines the Radxa `speech` service.
- `deploy/docker-compose.radxa.yml:27` starts its environment block.
- `deploy/docker-compose.rk.yml:13` defines the RK3576 `speech` service.
- `deploy/docker-compose.rk.yml:29` starts its environment block.
- `deploy/docker-compose.rpi.yml:10` defines the RPi `speech` service.
- `deploy/docker-compose.rpi.yml:18` starts its environment block.

### 2. New Module Paths And Function Signatures

- Keep `server/core/metrics.py` as the only public metrics module.
- Do not add Prometheus imports to `server/main.py` outside endpoint glue.
- Add `prometheus_client>=0.20,<1` to `app/requirements.txt`.
- Preserve `inc_sessions_active() -> int`.
- Preserve `dec_sessions_active() -> int`.
- Preserve `get_sessions_active() -> int`.
- Preserve `inc_sessions_rejected(reason: str) -> int`.
- Preserve `get_sessions_rejected(reason: str | None = None) -> int | dict[str, int]`.
- Preserve `inc_auth_rejected(endpoint: str) -> int`.
- Preserve `get_auth_rejected(endpoint: str | None = None) -> int | dict[str, int]`.
- Preserve `snapshot() -> dict`.
- Preserve `_reset_for_tests() -> None`.
- Add pseudocode signature `record_session_acquired() -> int`.
- Add pseudocode signature `record_session_released() -> int`.
- Add pseudocode signature `record_session_rejected(reason: str) -> int`.
- Add pseudocode signature `record_auth_rejected(endpoint: str) -> int`.
- Add pseudocode signature `record_tts_ttfa(backend: str, seconds: float) -> None`.
- Add pseudocode signature `record_tts_rtf(backend: str, rtf: float) -> None`.
- Add pseudocode signature `record_asr_decode_duration(backend: str, seconds: float) -> None`.
- Add pseudocode signature `set_asr_cer(backend: str, cer: float) -> None`.
- Add pseudocode signature `set_backend_state(manager: str, state: str, value: int | float = 1) -> None`.
- Add pseudocode signature `record_backend_reload(result: str) -> None`.
- Add pseudocode signature `record_worker_cancel(backend: str, reason: str) -> None`.
- Add pseudocode signature `inc_active_ws_sessions() -> int`.
- Add pseudocode signature `dec_active_ws_sessions() -> int`.
- Add pseudocode signature `set_queue_depth(queue: str, depth: int) -> None`.
- Add pseudocode signature `render_prometheus() -> bytes`.
- Add pseudocode signature `prometheus_content_type() -> str`.
- Add pseudocode signature `_reset_for_tests() -> None`.
- Add FastAPI route pseudocode `@app.get("/metrics")`.
- Add route handler pseudocode `async def metrics_endpoint(request: Request) -> Response`.
- Add optional auth helper pseudocode `def _metrics_requires_key() -> bool`.
- Add optional auth helper pseudocode `def _check_metrics_key(request: Request) -> None`.
- Do not define custom collectors outside `server/core/metrics.py`.
- Do not expose raw prometheus registry globals from the module.

### 3. Metric Inventory

- `ovs_sessions_active` remains a Gauge.
- `ovs_sessions_rejected_total{reason}` remains a Counter.
- `ovs_auth_rejected_total{endpoint}` remains a Counter.
- `ovs_tts_ttfa_seconds{backend}` is a Histogram.
- `ovs_tts_ttfa_seconds` buckets are exactly `[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]`.
- `ovs_tts_rtf{backend}` is a Histogram.
- `ovs_tts_rtf` buckets are exactly `[0.1, 0.3, 0.5, 1.0, 2.0]`.
- `ovs_asr_decode_duration_seconds{backend}` is a Histogram.
- `ovs_asr_decode_duration_seconds` buckets are exactly `[0.01, 0.05, 0.1, 0.5, 1.0]`.
- `ovs_asr_cer{backend}` is a Gauge.
- `ovs_backend_state{manager,state}` is a Gauge.
- `ovs_backend_state` manager label values are `asr` and `tts`.
- `ovs_backend_state` state label values mirror `BackendState`: `init`, `ready`, `draining`, `reloading`, `failed`.
- `ovs_backend_reload_total{result}` is a Counter.
- `ovs_backend_reload_total` result labels are `success`, `fail`, and `rollback`.
- `ovs_worker_cancels_total{backend,reason}` is a Counter.
- `ovs_active_ws_sessions` is a Gauge.
- `ovs_queue_depth{queue}` is a Gauge.
- `ovs_queue_depth` queue labels are `tts_stream`, `asr`, and `tts_worker`.
- `ovs_gpu_watchdog_ok` belongs to Deliverable 2 but is exported by the same module.
- `ovs_gpu_watchdog_check_duration_seconds` belongs to Deliverable 2 but is exported by the same module.
- `ovs_gpu_watchdog_failures_total{platform,reason}` belongs to Deliverable 2 but is exported by the same module.
- Prometheus default process/platform collectors are allowed if `prometheus_client` registers them by default.
- If default collectors are too noisy in tests, isolate app metrics in a module-owned registry for test resets.

### 4. Data Flow

- Existing session limiter code continues to call current helper names.
- `SessionLimiter.try_acquire()` at `server/core/session_limiter.py:147` remains the acquire boundary.
- `metrics.inc_sessions_active()` at `server/core/session_limiter.py:157` maps to the Prometheus active session Gauge.
- `SessionLimiter._release()` at `server/core/session_limiter.py:160` remains the release boundary.
- `metrics.dec_sessions_active()` at `server/core/session_limiter.py:167` decrements the same Gauge.
- `acquire_http()` rejection at `server/core/session_limiter.py:247` records `ovs_sessions_rejected_total{reason="http"}`.
- `try_acquire_ws()` rejection at `server/core/session_limiter.py:285` records `ovs_sessions_rejected_total{reason="ws"}`.
- HTTP auth rejection at `server/core/api_auth.py:117` records `ovs_auth_rejected_total{endpoint=request.url.path}`.
- WS auth rejection at `server/core/api_auth.py:182` records `ovs_auth_rejected_total{endpoint=ws.url.path}`.
- `/metrics` handler emits `render_prometheus()` with the Prometheus content type.
- `/metrics` is default unprotected, matching `/livez` at `server/main.py:590`.
- `/metrics` is default unprotected, matching `/readyz` at `server/main.py:601`.
- `/metrics` is default unprotected, matching `/health` at `server/main.py:672`.
- If `OVS_METRICS_REQUIRE_KEY` is truthy, `/metrics` requires the same API key source as Week 1 public endpoints.
- Metrics auth must not call `_require_api_key` blindly if that would protect `/metrics` only when keys exist but ignore `OVS_METRICS_REQUIRE_KEY`.
- Metrics auth should reuse `api_auth.check_http(request)` only after confirming the env flag is enabled.
- If `OVS_METRICS_REQUIRE_KEY=true` and no `OVS_API_KEYS` are configured, `/metrics` returns 401 rather than silently opening.
- The 401 path should increment `ovs_auth_rejected_total{endpoint="/metrics"}` through existing auth code.
- Histogram observations occur on request/session completion or first chunk events, not during scrape.
- Gauge state observations are updated when state changes, not calculated during scrape.

### 5. Instrumentation Anchors

- Record TTS non-streaming RTF after `backend.synthesize(...)` returns at `server/main.py:863`.
- Record TTS non-streaming RTF after `tts_service.synthesize(...)` returns at `server/main.py:878`.
- Use `meta.get("rtf", 0)` already exposed at `server/main.py:889`.
- Use backend label from manager backend `backend.name` near `server/main.py:855`.
- Use backend label from legacy backend `tts_service.backend_name()` near `server/main.py:883`.
- Record TTS streaming TTFA in manager stream path when first chunk passes through `server/main.py:1152`.
- Start the TTFA timer immediately before `_submit(0, queues[0])` at `server/main.py:1126`.
- Observe TTFA when `first_chunk_seen` flips at `server/main.py:1152`.
- Record TTS streaming queue depth from `len(sentences)` at `server/main.py:955`.
- Record `ovs_queue_depth{queue="tts_stream"}` around queue creation at `server/main.py:1122`.
- Record TTS legacy streaming TTFA when the first chunk yields at `server/main.py:1277`.
- Start legacy TTFA before `loop.run_in_executor(...)` at `server/main.py:1271`.
- Record clone streaming TTFA in manager clone path around `server/main.py:1475`.
- Record clone streaming TTFA in legacy clone path around `server/main.py:1511`.
- Record ASR offline decode duration around `asr_be.transcribe(...)` at `server/main.py:1552`.
- Record ASR offline decode duration around legacy `asr_be.transcribe(...)` at `server/main.py:1562`.
- Record ASR streaming decode duration around `stream.finalize` at `server/main.py:1709`.
- Record ASR streaming decode duration around `stream.finalize` at `server/main.py:1729`.
- Record ASR streaming decode duration around `stream.finalize` at `server/main.py:1752`.
- Record ASR V2V decode duration around `asr_manager.finalize_with_status(...)` at `server/main.py:2257`.
- Record active WS session increment immediately after accept at `server/main.py:1608`.
- Record active WS session decrement in the `/asr/stream` finally block near `server/main.py:1655`.
- Record active WS session increment immediately after accept at `server/main.py:1857`.
- Record active WS session decrement in V2V cleanup near `server/main.py:2505`.
- Record V2V `tts_q` depth after `await tts_q.put(sentence)` at `server/main.py:2160`.
- Record V2V `tts_q` depth after flush puts at `server/main.py:2164`.
- Record V2V `tts_q` depth after `await tts_q.get()` at `server/main.py:2345`.
- Record worker cancel with `backend="tts"` and `reason="speech_start"` when task is cancelled at `server/main.py:2094`.
- Record worker cancel with `backend="tts"` and `reason="client_abort"` when task is cancelled at `server/main.py:2174`.
- Record worker cancel with `backend="tts"` and `reason="task_cancelled"` at `server/main.py:2425`.
- Record worker cancel with `backend="asr"` and `reason="bargein"` at `server/main.py:2186`.
- Record worker cancel with `backend="asr"` and `reason="ws_close"` at `server/main.py:2493`.
- Record backend state `init` during manager construction at `server/core/backend_manager.py:112`.
- Record backend state `failed` on start failure at `server/core/backend_manager.py:139`.
- Record backend state `ready` after start at `server/core/backend_manager.py:151`.
- Record backend state `draining` on shutdown at `server/core/backend_manager.py:157`.
- Record backend state `failed` after shutdown at `server/core/backend_manager.py:165`.
- Record backend state `draining` during reload at `server/core/backend_manager.py:375`.
- Record backend state `reloading` during reload at `server/core/backend_manager.py:387`.
- Record backend state `ready` on reload success at `server/core/backend_manager.py:411`.
- Increment reload total `success` before return at `server/core/backend_manager.py:421`.
- Increment reload total `fail` in the exception branch at `server/core/backend_manager.py:433`.
- Increment reload total `rollback` before return at `server/core/backend_manager.py:465`.
- Record backend state `failed` on rollback failure at `server/core/backend_manager.py:477`.

### 6. Reusing `bench/perf/` Hooks

- Treat `bench/perf/client.py` as the timing vocabulary source.
- `bench/perf/client.py:65` maps to server-side ASR first decoded timing but not a Week 2 Prometheus metric.
- `bench/perf/client.py:67` maps to `ovs_asr_decode_duration_seconds` derived RTF only when audio duration is available.
- `bench/perf/client.py:70` clarifies compute-bound streaming finalize RTF.
- `bench/perf/client.py:76` maps directly to server-side ASR decode duration.
- `bench/perf/client.py:87` maps to `ovs_tts_ttfa_seconds`.
- `bench/perf/client.py:89` maps to `ovs_tts_rtf`.
- `bench/perf/client.py:286` is the client-side equivalent of ASR finalize compute.
- `bench/perf/client.py:293` is the client-side RTF formula for streaming ASR.
- `bench/perf/client.py:294` is the compute-bound ASR finalize RTF formula.
- `bench/perf/client.py:333` captures first TTS chunk, matching server-side `server/main.py:1152`.
- `bench/perf/client.py:351` computes TTS RTF as total wall divided by synthesized duration.
- `bench/perf/load_2client_tts.py:23` and `bench/perf/load_2client_tts.py:32` show the TTFA clock boundaries.
- `bench/perf/multi_sentence_pipeline.py:48` shows the first PCM beyond sample-rate header rule.
- `bench/perf/stress_cancel_n1.py:61` shows early-break TTFA should still be real first PCM.
- Server instrumentation should not import `bench/perf`.
- Server instrumentation should reuse the same semantics, not the same module.
- For streaming TTS, sample-rate header bytes do not count as TTFA audio.
- For streaming TTS, first PCM chunk after the 4-byte header counts as TTFA.
- For streaming ASR, decode duration means finalize compute, not real-time audio ingestion.
- For offline ASR, decode duration means `transcribe(...)` wall time.
- For rolling CER, leave only a hook because server does not have reference transcripts.
- Optional CER hook can be fed by future eval jobs or perf harness post-processing.

### 7. Edge Cases And Failure Modes

- Prometheus client import failure should be impossible after dependency install but fail startup loudly if missing.
- Duplicate metric registration can happen in tests that reload modules.
- Use a module-local registry or guard construction to avoid duplicate registration.
- `_reset_for_tests()` must reset app metrics without breaking the default registry.
- Prometheus counters cannot decrement, so existing getters need a shadow value or direct sample extraction.
- Existing tests may assert integer return values from `inc_*` helpers.
- Preserve integer return values by maintaining lightweight shadow counters under the existing lock.
- Existing `snapshot()` should return the Week 1 keys and new keys only if tests are updated.
- Do not remove `snapshot()` because `/readyz` and tests may rely on it.
- Histogram labels must be bounded to avoid cardinality blowups.
- Backend label should use stable backend names like `asr_be.name` and `backend.name`.
- Endpoint labels for auth are existing behavior and can remain path strings.
- Queue labels must be fixed constants.
- State labels must be fixed enum values.
- `ovs_backend_state` must clear previous state for a manager when setting a new state.
- If previous state is not cleared, a manager can appear in multiple states.
- Use helper `set_backend_state(manager, state)` to set selected state to 1 and all other known states to 0.
- Unknown backend manager names are ignored or normalized only in tests.
- Metrics endpoint must not block on backend locks.
- Metrics endpoint must not acquire session limiter slots.
- Metrics endpoint must not call `gpu_watchdog.status()` if that may run probes; scrape should only read cached state.
- Metrics endpoint should be low allocation but does serialize all samples.
- Metrics endpoint should return 200 even while `/readyz` is 503.
- Metrics endpoint should return 401 only when `OVS_METRICS_REQUIRE_KEY` is enabled and auth fails.
- Metrics endpoint should return text exposition with correct `Content-Type`.
- Streaming TTFA must not be observed twice for a single request.
- Multi-sentence streaming should define TTFA as first audio for the request, not every sentence.
- Per-sentence TTFA is not in Week 2 scope.
- V2V TTS TTFA should observe first audio per V2V session or per sentence only if named as request-level first audio.
- Prefer request-level first audio for `ovs_tts_ttfa_seconds`.
- Streaming cancellation before first PCM should not observe TTFA.
- Streaming errors before first PCM should not observe TTFA.
- RTF should not observe when audio duration is missing or zero.
- Negative timing values must be discarded.
- NaN and infinite values must be discarded.
- Queue depth should be set to 0 on cleanup where possible.
- Worker cancel reasons must be constrained to a small set.
- `/health` temporary `_WorkerIO` cancel count remains for backward compatibility in Week 2.
- `ovs_worker_cancels_total` becomes the production replacement for that temporary health field.

### 8. Regression Risks

- Adding `prometheus_client` can change import time and memory baseline.
- Duplicate metric registration can make test suites flaky.
- A global default registry can preserve state across tests.
- Changing `metrics.py` return values can break Week 1 unit tests.
- Replacing text counters without shadow counters can break `snapshot()` tests.
- Adding `/metrics` without route ordering care should not affect existing routes.
- Protecting `/metrics` by default would break standard scraping assumptions.
- Leaving `/metrics` unprotected when `OVS_METRICS_REQUIRE_KEY=true` would violate this spec.
- Measuring TTFA inside executor threads can call Prometheus objects from non-main threads; prometheus_client supports thread-safe observation.
- High-frequency queue gauge updates can add overhead in hot V2V loops.
- Instrument only enqueue/dequeue boundaries, not every audio frame.
- Backend state gauges can go stale if state changes occur outside helper calls.
- Reload counters can double-count if both exception and rollback paths increment incorrectly.
- ASR decode duration may be over-counted if it includes VAD silence wait.
- TTS TTFA may be under-counted if the timer starts after admission or after queueing.
- Active WS gauge can leak if decrement is not in every finally path.
- Use try/finally around accepted WS sessions.

### 9. Test Checklist

- Unit: importing `server.core.metrics` works without starting FastAPI.
- Unit: `inc_sessions_active()` returns incremented integer.
- Unit: `dec_sessions_active()` clamps at zero.
- Unit: `inc_sessions_rejected("http")` returns expected count.
- Unit: `inc_sessions_rejected("ws")` returns expected count.
- Unit: `inc_auth_rejected("/tts")` returns expected count.
- Unit: `snapshot()` contains Week 1 keys.
- Unit: `_reset_for_tests()` clears shadow counters and gauges.
- Unit: `record_tts_ttfa("mock", 0.1)` emits histogram samples.
- Unit: `record_tts_rtf("mock", 0.5)` emits histogram samples.
- Unit: `record_asr_decode_duration("mock", 0.05)` emits histogram samples.
- Unit: `set_asr_cer("mock", 0.1)` emits gauge sample.
- Unit: `set_backend_state("tts", "ready")` sets ready to 1.
- Unit: `set_backend_state("tts", "ready")` sets non-ready states to 0.
- Unit: `record_backend_reload("success")` increments counter.
- Unit: `record_backend_reload("rollback")` increments counter.
- Unit: `record_worker_cancel("tts", "client_abort")` increments counter.
- Unit: `inc_active_ws_sessions()` increments active WS gauge.
- Unit: `dec_active_ws_sessions()` clamps active WS gauge at zero.
- Unit: `set_queue_depth("tts_stream", 2)` emits queue gauge.
- Integration: `GET /metrics` returns 200 without `OVS_API_KEYS` and without `OVS_METRICS_REQUIRE_KEY`.
- Integration: `GET /metrics` returns text exposition.
- Integration: `GET /metrics` includes `ovs_sessions_active`.
- Integration: `GET /metrics` includes `ovs_tts_ttfa_seconds_bucket`.
- Integration: `GET /metrics` includes `ovs_tts_rtf_bucket`.
- Integration: `GET /metrics` includes `ovs_asr_decode_duration_seconds_bucket`.
- Integration: `GET /metrics` includes `ovs_backend_state`.
- Integration: `GET /metrics` includes `ovs_active_ws_sessions`.
- Integration: `GET /metrics` with `OVS_METRICS_REQUIRE_KEY=true` and no token returns 401.
- Integration: `GET /metrics` with `OVS_METRICS_REQUIRE_KEY=true` and valid bearer returns 200.
- Integration: `/livez` remains unprotected.
- Integration: `/readyz` remains unprotected.
- Integration: `/health` remains unprotected and deprecated.
- Integration: `/tts` records `ovs_tts_rtf` when synthesis succeeds.
- Integration: `/tts/stream` records one TTFA observation on first PCM.
- Integration: `/tts/stream` cancelled before PCM does not record TTFA.
- Integration: `/asr` records `ovs_asr_decode_duration_seconds`.
- Integration: `/asr/stream` increments active WS on accept and decrements on close.
- Integration: `/v2v/stream` increments active WS on accept and decrements on close.
- Integration: `/admin/backend/reload` success increments reload `success`.
- Integration: failed reload with rollback increments `fail` and `rollback`.
- Smoke: `curl /metrics` works inside each target container.
- Smoke: Prometheus scrape does not affect `/readyz` state.
- Smoke: p95 `/metrics` response time stays below 50 ms on edge targets with normal series count.
- Smoke: run `bench/perf/load_2client_tts.py` and verify server TTFA histogram receives samples.
- Smoke: run `bench/perf/perf.py asr` and verify ASR decode histogram receives samples.

## Deliverable 2: GPU Watchdog Real Implementation

### 1. Affected Files And Anchors

- `server/core/gpu_watchdog.py:1` identifies the current watchdog stub.
- `server/core/gpu_watchdog.py:12` defines `is_ok()`.
- `server/core/gpu_watchdog.py:18` currently returns `True`.
- `server/main.py:326` registers startup where the background task should be launched.
- `server/main.py:337` initializes the session limiter before model downloads.
- `server/main.py:349` initializes coordinator after limiter.
- `server/main.py:355` checks RK runtime before backend imports.
- `server/main.py:396` starts ASR backend preload.
- `server/main.py:440` starts TTS service setup.
- `server/main.py:496` wires backend managers.
- `server/main.py:578` logs service ready.
- `server/main.py:601` defines `/readyz`.
- `server/main.py:615` imports backend manager in `/readyz`.
- `server/main.py:616` imports session limiter in `/readyz`.
- `server/main.py:617` imports `gpu_watchdog` in `/readyz`.
- `server/main.py:657` begins watchdog readiness check.
- `server/main.py:659` calls `_gw_mod.is_ok()`.
- `server/main.py:660` appends `gpu_watchdog_failed`.
- `server/main.py:664` returns 503 when reasons exist.
- `server/main.py:669` returns ready.
- `server/core/metrics.py` will export watchdog metrics from Deliverable 1.
- `app/requirements.txt` should not gain hard GPU dependencies.
- Compose files do not need watchdog env by default unless operators choose to set interval.

### 2. New Module Paths And Function Signatures

- Keep implementation in `server/core/gpu_watchdog.py`.
- Do not add platform-specific modules under `server/core` for Week 2.
- Preserve `is_ok() -> bool`.
- Add pseudocode `def status() -> dict`.
- Add pseudocode `async def start() -> None`.
- Add pseudocode `async def stop() -> None`.
- Add pseudocode `async def _run_loop() -> None`.
- Add pseudocode `async def _check_once() -> WatchdogCheck`.
- Add pseudocode `def _detect_platform() -> str`.
- Add pseudocode `def _check_jetson() -> tuple[bool, str]`.
- Add pseudocode `def _check_rk() -> tuple[bool, str]`.
- Add pseudocode `def _check_hailo() -> tuple[bool, str]`.
- Add pseudocode `def _check_desktop_cuda() -> tuple[bool, str]`.
- Add pseudocode `def _check_cpu_only() -> tuple[bool, str]`.
- Add pseudocode `def _parse_interval() -> float`.
- Add pseudocode `class WatchdogStatus`.
- `WatchdogStatus.ok: bool`.
- `WatchdogStatus.platform: str`.
- `WatchdogStatus.reason: str`.
- `WatchdogStatus.last_checked_at: float | None`.
- `WatchdogStatus.consecutive_failures: int`.
- `WatchdogStatus.consecutive_successes: int`.
- `WatchdogStatus.last_duration_s: float | None`.
- `WatchdogStatus.checks_total: int`.
- `WatchdogStatus.failures_total: int`.
- `status()` returns only JSON-serializable values.
- `status()` includes `reason`.
- `status()` includes `platform`.
- `status()` includes hysteresis counters.
- `status()` must not run hardware probes.
- `is_ok()` must not run hardware probes.
- `is_ok()` reads cached status only.
- `start()` launches one background asyncio task.
- `stop()` cancels and awaits that task.
- Startup integration uses `asyncio.create_task`.
- Shutdown integration should stop the task if a shutdown hook exists or is added.

### 3. Platform Auto-Detection

- Jetson detection should check `/etc/nv_tegra_release`.
- Jetson detection should check whether `tegrastats` exists in PATH.
- Jetson detection can check `nvidia-smi` existence as a fallback.
- Jetson detection can check Python import `pycuda.driver` as optional.
- RK detection should check `/sys/class/devfreq`.
- RK detection should check paths matching `/sys/class/devfreq/*/cur_freq`.
- RK detection should check `/sys/kernel/debug/rknpu/load` when readable.
- RK detection should honor env hints like `RK_PLATFORM`, `ASR_PLATFORM`, or `LANGUAGE_MODE=rk`.
- Hailo detection should check `hailortcli` in PATH.
- Hailo detection should remain best-effort.
- Desktop CUDA detection should check optional `pynvml`.
- Desktop CUDA detection should check `nvidia-smi`.
- CPU-only detection is the fallback.
- CPU-only platform is always OK.
- Optional imports must be inside probe functions.
- Optional import failure means the feature is unavailable.
- Optional import failure must not fail service startup.
- Probe command failures should record a reason.
- Probe command timeout should record a reason.
- Debugfs permission failure on RK should not be fatal if other RK health sources work.
- Missing all RK sources on an RK profile should fail the check after hysteresis.
- Missing all Jetson sources on a Jetson profile should fail the check after hysteresis.
- Missing Hailo CLI on a non-Hailo system should not fail.
- Hailo best-effort failure on a Hailo-detected system should fail after hysteresis only if no successful health signal exists.

### 4. Check Semantics

- Default interval is 5 seconds.
- Env `OVS_GPU_WATCHDOG_INTERVAL_S` overrides the interval.
- Invalid interval values log a warning and use 5 seconds.
- Minimum interval should clamp to 1 second.
- A single check should be lightweight.
- Jetson check via `tegrastats` should run with a short timeout.
- Jetson check via `nvidia-smi` should run with a short timeout.
- Jetson optional `pycuda.driver` light op should initialize context carefully.
- CUDA light op must not allocate large buffers.
- CUDA light op should be skipped if import fails.
- RK `cur_freq` check validates files are readable and numeric.
- RK `rknpu/load` check validates readable content and does not require a specific load value.
- Hailo `hailortcli` check validates CLI returns successfully.
- Desktop CUDA `pynvml` check validates device count and handle query.
- Desktop CUDA `nvidia-smi` check validates command success.
- CPU-only returns `(True, "cpu_only")`.
- Consecutive failure threshold N is 3.
- Consecutive recovery threshold M is 5.
- Before 3 failures, cached `ok` remains true if previously true.
- After 3 consecutive failures, cached `ok` becomes false.
- While false, a single success does not recover.
- After 5 consecutive successes, cached `ok` becomes true.
- Failure reason should preserve the latest failure.
- Recovery reason should become `ok`.
- Startup initial state should be OK for CPU-only.
- Startup initial state for GPU/NPU platforms can be optimistic until first 3 failed checks.
- First successful check should set platform and reason immediately.
- Hysteresis counters should reset opposite direction counters.

### 5. Data Flow

- `server/main.py` startup launches watchdog after profile/env application at `server/main.py:330`.
- Starting after profile load lets detector use `LANGUAGE_MODE`, `RK_PLATFORM`, and profile env.
- Starting before model preload catches hardware issues during warmup but should not block startup.
- The task should be launched near `server/main.py:342` to run alongside other startup work.
- `/readyz` reads cached watchdog state at `server/main.py:659`.
- If `is_ok()` returns false, `/readyz` returns 503 as it already does at `server/main.py:664`.
- Week 2 should enrich `/readyz` detail with `gpu_watchdog.status()`.
- Existing `reasons` array should retain `gpu_watchdog_failed`.
- Add optional `details.gpu_watchdog` object to readiness response.
- The readiness detail must include `reason`.
- The readiness detail must include `platform`.
- The readiness detail must include hysteresis counters.
- `/livez` must not depend on watchdog.
- `/health` may include watchdog status for backward-compatible operator detail, but readiness is the gating endpoint.
- Watchdog records `ovs_gpu_watchdog_ok` after every check.
- Watchdog observes `ovs_gpu_watchdog_check_duration_seconds` for every check.
- Watchdog increments `ovs_gpu_watchdog_failures_total{platform,reason}` for failed checks.
- The failure counter increments on raw check failures, not only after hysteresis fail state.
- Readiness uses hysteresis state, not raw check result.
- Watchdog metrics should be no-op safe if metrics module import fails during early startup.

### 6. Edge Cases And Failure Modes

- If the background task crashes, `is_ok()` should become false after setting reason `watchdog_task_crashed`.
- If the event loop is closing, cancellation should not log noisy tracebacks.
- If platform detection changes during runtime, keep the initial platform unless status explicitly supports update.
- If `tegrastats` hangs, command timeout prevents loop stall.
- If `nvidia-smi` hangs, command timeout prevents loop stall.
- If debugfs path requires root, RK fallback should try devfreq.
- If no RK frequency files are readable, fail with reason `rk_probe_unavailable`.
- If Hailo CLI is not installed but Hailo is not detected, do not fail.
- If Hailo CLI is installed but returns non-zero, fail with reason `hailo_cli_failed`.
- If `pynvml` is installed but NVML init fails, fallback to `nvidia-smi`.
- If all CUDA probes fail on CUDA-detected platform, fail with reason `cuda_probe_failed`.
- If CPU-only fallback is selected inside a GPU container by mistake, detection is too weak.
- To reduce false CPU-only fallback, env hints and device files must be checked before fallback.
- Watchdog must avoid importing heavy GPU modules at module import time.
- Watchdog must not create CUDA contexts on CPU-only systems.
- Watchdog must not block request path.
- `status()` should return stale `last_checked_at` if checks have not run yet.
- `/readyz` should not throw if `status()` raises unexpectedly.
- Invalid `OVS_GPU_WATCHDOG_INTERVAL_S` must not crash startup.
- Negative interval must not create a busy loop.
- Failure reason labels can create cardinality if raw exception strings are used.
- Normalize reasons to fixed tokens.
- Include detailed exception text in logs, not labels.
- Metrics failure reason labels should be fixed tokens such as `probe_timeout`, `probe_unavailable`, `probe_failed`, `task_crashed`.

### 7. Regression Risks

- Launching a background task without shutdown cleanup can leak tasks in tests.
- Running shell probes every 5 seconds can be too expensive on small CPU targets.
- Keep command timeouts low and prefer file reads.
- PyCUDA imports can allocate GPU resources unexpectedly.
- Keep PyCUDA light op optional and last for Jetson.
- NVML import can fail on Jetson images.
- Do not make NVML a hard dependency.
- Readiness 503 on transient probe failures could flap deployments.
- Hysteresis thresholds reduce flapping risk.
- If startup state is false before first checks, orchestrators may never route during slow boot.
- Keep startup optimistic until hysteresis proves failure.
- Existing tests may monkeypatch `gpu_watchdog.is_ok`; preserve that interface.
- Adding `status()` should not change old monkeypatches.
- `/readyz` response schema additions should be additive.
- Compose healthchecks already use `/readyz`; watchdog false will restart or mark unhealthy depending orchestrator policy.
- On RPi CPU-only, watchdog must always OK.

### 8. Test Checklist

- Unit: importing `server.core.gpu_watchdog` does not import pycuda.
- Unit: importing `server.core.gpu_watchdog` does not import pynvml.
- Unit: `is_ok()` returns a bool.
- Unit: `status()` returns a dict with `reason`.
- Unit: invalid interval returns default 5 seconds.
- Unit: interval below 1 second clamps to 1 second.
- Unit: CPU-only detection returns OK.
- Unit: simulated Jetson with working `tegrastats` returns OK.
- Unit: simulated Jetson with command timeout records failure.
- Unit: simulated RK with numeric `cur_freq` returns OK.
- Unit: simulated RK with unreadable files and profile hint fails.
- Unit: simulated Hailo with successful `hailortcli` returns OK.
- Unit: simulated desktop CUDA with working NVML returns OK.
- Unit: optional pycuda import failure is not fatal.
- Unit: optional pynvml import failure is not fatal.
- Unit: three consecutive failures flip `is_ok()` false.
- Unit: four successes after fail keep `is_ok()` false.
- Unit: five successes after fail flip `is_ok()` true.
- Unit: failed checks increment failure metric with normalized reason.
- Unit: check duration histogram observes all checks.
- Integration: startup creates exactly one watchdog task.
- Integration: shutdown cancels watchdog task cleanly.
- Integration: `/readyz` returns 200 when watchdog OK and other gates OK.
- Integration: `/readyz` returns 503 when watchdog cached state is false.
- Integration: `/readyz` not-ready body includes `gpu_watchdog_failed`.
- Integration: `/readyz` not-ready body includes watchdog detail with reason.
- Smoke: RPi CPU-only profile remains ready after watchdog starts.
- Smoke: RK profile can read at least one RK health signal.
- Smoke: Jetson profile can read at least one Jetson health signal.
- Smoke: watchdog loop CPU overhead is negligible over 60 seconds idle.
- Smoke: `curl /metrics` shows `ovs_gpu_watchdog_ok`.
- Smoke: killing or blocking a probe path produces failures without crashing the app.

## Deliverable 3: Structured JSON Logging

### 1. Affected Files And Anchors

- `server/main.py:29` currently calls `logging.basicConfig(...)`.
- `server/main.py:30` sets current level to INFO.
- `server/main.py:31` sets current text format.
- `server/main.py:33` creates `logger`.
- `server/main.py:35` creates the FastAPI app before any middleware is declared.
- `server/main.py` currently has no `@app.middleware("http")` match in the reviewed source.
- `server/core/api_auth.py:86` defines `mask_key(value)`.
- `server/core/api_auth.py:124` logs HTTP auth rejection.
- `server/core/api_auth.py:128` logs masked supplied token.
- `server/core/api_auth.py:189` logs WS auth rejection.
- `server/core/api_auth.py:193` logs masked supplied token.
- `server/main.py:590` defines `/livez`.
- `server/main.py:601` defines `/readyz`.
- `server/main.py:672` defines `/health`.
- `server/main.py:839` defines `/tts`.
- `server/main.py:899` defines `/tts/stream`.
- `server/main.py:1533` defines `/asr`.
- `server/main.py:1576` defines `/asr/stream`.
- `server/main.py:1605` runs WS auth before accept.
- `server/main.py:1608` accepts ASR WS.
- `server/main.py:1813` defines `/v2v/stream`.
- `server/main.py:1854` runs V2V WS auth before accept.
- `server/main.py:1857` accepts V2V WS.
- `server/main.py:1999` logs V2V stream open with backend context in message text.
- `server/main.py:2506` logs V2V stream closed.
- `server/core/backend_manager.py:140` logs start failure.
- `server/core/backend_manager.py:152` logs manager ready.
- `server/core/backend_manager.py:257` logs drain timeout.
- `server/core/backend_manager.py:274` logs WS close failure.
- `server/core/backend_manager.py:414` logs reload success.
- `server/core/backend_manager.py:433` logs reload failure.
- `server/core/backend_manager.py:472` logs rollback failure.
- `app/requirements.txt:1` starts dependency list.
- `deploy/docker-compose.yml:28` starts Jetson speech environment block.
- `deploy/docker-compose.radxa.yml:27` starts Radxa speech environment block.
- `deploy/docker-compose.rk.yml:29` starts RK3576 speech environment block.
- `deploy/docker-compose.rpi.yml:18` starts RPi speech environment block.

### 2. New Module Paths And Function Signatures

- Add `server/core/logging_config.py`.
- Add pseudocode `request_id_var: ContextVar[str | None]`.
- Add pseudocode `session_id_var: ContextVar[str | None]`.
- Add pseudocode `backend_var: ContextVar[str | None]`.
- Add pseudocode `def setup_logging() -> None`.
- Add pseudocode `def configure_root_logger(format_name: str) -> None`.
- Add pseudocode `class OVSJsonFormatter(JsonFormatter)`.
- Add pseudocode `def mask_sensitive_value(value: str | None) -> str`.
- Add pseudocode `def mask_url_query(url: str) -> str`.
- Add pseudocode `def get_request_id() -> str | None`.
- Add pseudocode `def set_request_context(request_id: str | None = None, session_id: str | None = None, backend: str | None = None) -> TokenBundle`.
- Add pseudocode `def reset_request_context(tokens: TokenBundle) -> None`.
- Add pseudocode `def generate_request_id() -> str`.
- Add pseudocode `def request_id_from_headers(headers) -> str | None`.
- Add pseudocode `def sanitize_headers_for_log(headers) -> dict`.
- Add HTTP middleware in `server/main.py`.
- Middleware pseudocode signature `@app.middleware("http")`.
- Middleware pseudocode signature `async def request_context_middleware(request: Request, call_next)`.
- Add WS helper pseudocode `def init_ws_context(ws: WebSocket) -> TokenBundle`.
- Add WS helper pseudocode `def clear_ws_context(tokens: TokenBundle) -> None`.
- Add dependency `python-json-logger>=2.0,<4` to `app/requirements.txt`.
- Alternative minimal hand-written formatter is allowed only if dependency is rejected during implementation.
- Prefer `pythonjsonlogger` package import path from `python-json-logger`.

### 3. JSON Fields

- `ts` is required.
- `ts` format is ISO 8601 with milliseconds.
- `ts` should include timezone or UTC `Z`.
- `level` is required.
- `logger` is required.
- `msg` is required.
- `session_id` is required when context has it.
- `session_id` should be null or absent when unavailable; choose one style and test it.
- `backend` is required when context has it.
- `backend` should be null or absent when unavailable; choose one style and test it.
- `request_id` is required when context has it.
- `request_id` should be present for every HTTP request.
- `request_id` should be present for every accepted WS connection.
- `exc_info` should still render exception details for `logger.exception`.
- Preserve existing message interpolation semantics.
- Existing `logger.info("foo")` calls should work unchanged.
- Existing `logger.warning("x=%s", x)` calls should work unchanged.
- Existing `logger.exception(...)` calls should include stack traces.
- Text format remains current behavior by default.
- JSON format is enabled with `OVS_LOG_FORMAT=json`.
- Text format is enabled with `OVS_LOG_FORMAT=text`.
- Unknown `OVS_LOG_FORMAT` values fall back to text and log a warning.
- Production compose sets `OVS_LOG_FORMAT=json`.

### 4. Request ID Flow

- HTTP middleware runs after app creation at `server/main.py:35`.
- HTTP middleware reads inbound `X-Request-ID`.
- If inbound `X-Request-ID` exists and is sane, propagate it.
- If missing, generate a new request id.
- Generated request id should be collision-resistant and short enough for logs.
- Use UUID4 hex or similar.
- Store request id in `request_id_var`.
- Add `X-Request-ID` response header on every HTTP response.
- Reset contextvars in a finally block.
- Middleware must mask sensitive data if it logs request metadata.
- Middleware must not read request body.
- Middleware must not acquire session slots.
- Middleware must not protect `/livez`, `/readyz`, `/health`, or `/metrics`.
- WS handlers read `X-Request-ID` from headers before or after auth.
- WS handlers generate request id if header missing.
- WS handlers store request id in contextvars before first log in handler.
- For `/asr/stream`, set context near `server/main.py:1576` before `check_ws` at `server/main.py:1605`.
- For `/v2v/stream`, set context near `server/main.py:1813` before `check_ws` at `server/main.py:1854`.
- WS context resets in final cleanup.
- Contextvars propagate across `await` boundaries.
- Contextvars do not automatically propagate into `run_in_executor` threads.
- Executor worker logs may not include request id unless wrapped.
- Week 2 should wrap executor functions only where logs are expected and cheap.
- For `/tts/stream`, `_run()` at `server/main.py:1082` can copy context before executor dispatch.
- For legacy `/tts/stream`, `_run()` at `server/main.py:1242` can copy context before executor dispatch.
- For V2V TTS, `_run_synth()` at `server/main.py:2356` can copy context before executor dispatch.
- If executor context wrapping is too invasive, document executor log context as best-effort.
- The request id must pass across normal async awaits.

### 5. Session And Backend Context Flow

- HTTP `/tts` can set backend context after manager acquisition near `server/main.py:855`.
- HTTP `/tts` legacy path can set backend context near `server/main.py:878`.
- HTTP `/tts/stream` can set backend context after `backend = await acquire_cm.__aenter__()` at `server/main.py:962`.
- HTTP `/tts/stream` legacy path can set backend context after `backend = tts_service.get_backend()` at `server/main.py:1176`.
- HTTP `/asr` can set backend context after manager acquisition at `server/main.py:1550`.
- HTTP `/asr` legacy path can set backend context near `server/main.py:1560`.
- WS `/asr/stream` can set backend context after backend selection at `server/main.py:1626`.
- WS `/v2v/stream` can set backend context after ASR/TTS selection at `server/main.py:1948` and `server/main.py:1987`.
- Session id can be generated at session admission.
- `/asr/stream` session id can be created after accept at `server/main.py:1608`.
- `/v2v/stream` session id can be created after accept at `server/main.py:1857`.
- `/tts/stream` session id can be created when `_session_token` is acquired at `server/main.py:920`.
- `/tts` non-streaming session id can be created inside `acquire_http` context at `server/main.py:845`.
- Do not change external API payloads to include session id in Week 2.
- Session id exists for logs only.
- Backend context can be overwritten inside nested ASR/TTS phases.
- Backend context reset should use tokens to avoid leaking values across requests.

### 6. Security Masking

- Reuse the principle from `api_auth.mask_key` at `server/core/api_auth.py:86`.
- Authorization header values must never be logged raw.
- Masked Authorization should show scheme plus masked token prefix only if useful.
- Query param `token` must never be logged raw.
- Any URL logged with query string should pass through `mask_url_query`.
- `token=<value>` becomes `token=<masked>`.
- Empty token becomes `token=<missing>`.
- Multiple token params are all masked.
- Header maps logged for debugging must pass through `sanitize_headers_for_log`.
- Existing auth logs already call `mask_key` at `server/core/api_auth.py:128`.
- Existing WS auth logs already call `mask_key` at `server/core/api_auth.py:193`.
- New middleware access logs must not log raw headers.
- New middleware access logs must not log raw query strings containing token.
- Request IDs must be validated to avoid log injection.
- Strip control characters from inbound request id.
- Cap inbound request id length, for example 128 chars.
- Generate a fresh id if inbound id is empty after stripping.
- JSON formatter must encode safely through the logging library.

### 7. Docker Compose Changes

- `deploy/docker-compose.yml` has `speech` service at `deploy/docker-compose.yml:14`.
- Add `OVS_LOG_FORMAT=${OVS_LOG_FORMAT:-json}` to the environment block starting at `deploy/docker-compose.yml:28`.
- `deploy/docker-compose.radxa.yml` has `speech` service at `deploy/docker-compose.radxa.yml:13`.
- Add `OVS_LOG_FORMAT=${OVS_LOG_FORMAT:-json}` to the environment block starting at `deploy/docker-compose.radxa.yml:27`.
- `deploy/docker-compose.rk.yml` has `speech` service at `deploy/docker-compose.rk.yml:13`.
- Add `OVS_LOG_FORMAT=${OVS_LOG_FORMAT:-json}` to the environment block starting at `deploy/docker-compose.rk.yml:29`.
- `deploy/docker-compose.rpi.yml` has `speech` service at `deploy/docker-compose.rpi.yml:10`.
- Add `OVS_LOG_FORMAT=${OVS_LOG_FORMAT:-json}` to the environment block starting at `deploy/docker-compose.rpi.yml:18`.
- Do not add the env var to the Jetson `translator` service at `deploy/docker-compose.yml:81`.
- Do not change healthcheck commands.
- Do not rename services.
- Do not require JSON logs outside compose production paths.

### 8. Edge Cases And Failure Modes

- `logging.basicConfig` can be a no-op if handlers already exist.
- `setup_logging()` should explicitly configure root handlers.
- Tests may configure logging first; avoid duplicate handlers.
- Uvicorn access logs may use separate loggers.
- Decide whether to configure `uvicorn`, `uvicorn.error`, and `uvicorn.access`.
- JSON formatting app logs only is acceptable if uvicorn logs remain text in dev.
- Production should prefer JSON for root and uvicorn loggers.
- Existing Unicode log messages should encode correctly in JSON.
- Text mode should preserve current format from `server/main.py:31`.
- Contextvars can leak if tokens are not reset in finally blocks.
- WS early returns before main try/finally can leak context.
- Wrap WS handler body with outer try/finally for context cleanup.
- Request id response header must be present on error responses.
- Middleware exceptions must reset context before propagating.
- Streaming responses continue after middleware returns; context may reset too early for generator logs.
- For streaming responses, the generator executes after middleware returns in some ASGI paths.
- Set context again inside streaming generator where logs and metrics happen.
- `/tts/stream` generator at `server/main.py:972` should capture context values.
- Legacy `/tts/stream` generator at `server/main.py:1193` should capture context values.
- Clone stream generator at `server/main.py:1466` should capture context values.
- V2V tasks created at `server/main.py:2454` should inherit context at create time.
- Python task context propagation handles async tasks by default.
- Executor thread logs need explicit `contextvars.copy_context()`.
- If JSON formatter fails, logging must not crash request handling.
- Use defensive formatter code for missing fields.
- Do not serialize raw request objects.
- Do not serialize bytes audio.
- Do not include API keys in exception messages.

### 9. Regression Risks

- Replacing logging setup may hide logs if handler configuration is wrong.
- JSON logs may break local grep workflows if enabled by default.
- Default remains text to avoid that.
- Compose JSON default changes production behavior intentionally.
- Middleware can add overhead to every HTTP request.
- Keep request-id generation cheap.
- Middleware can interfere with streaming responses if it consumes body.
- Do not consume request body.
- Context wrapping executor calls can add complexity.
- Apply executor context wrapping only at log-heavy boundaries.
- Header propagation can fail if response construction bypasses middleware due to exceptions.
- Test exception paths.
- Masking query strings can accidentally alter actual request URLs if applied before routing.
- Only mask copies used for logging.
- Do not mutate `request.url`.
- Inbound request id can contain user-controlled strings.
- Sanitize before storing.

### 10. Test Checklist

- Unit: `setup_logging()` with `OVS_LOG_FORMAT=text` preserves text format.
- Unit: `setup_logging()` with `OVS_LOG_FORMAT=json` emits valid JSON.
- Unit: JSON log has `ts`.
- Unit: JSON log has `level`.
- Unit: JSON log has `logger`.
- Unit: JSON log has `msg`.
- Unit: JSON log includes `request_id` when contextvar set.
- Unit: JSON log includes `session_id` when contextvar set.
- Unit: JSON log includes `backend` when contextvar set.
- Unit: JSON log handles `logger.info("x=%s", "y")`.
- Unit: JSON log handles `logger.exception`.
- Unit: mask function masks Authorization token values.
- Unit: mask function masks query param `token`.
- Unit: mask function masks repeated `token` params.
- Unit: inbound request id strips control characters.
- Unit: overlong request id is rejected or truncated.
- Integration: HTTP request without `X-Request-ID` gets response `X-Request-ID`.
- Integration: HTTP request with `X-Request-ID` propagates response header.
- Integration: HTTP logs include propagated request id.
- Integration: `/livez` still returns 200.
- Integration: `/readyz` still returns expected status.
- Integration: `/health` still returns deprecated headers.
- Integration: `/tts` logs include request id in JSON mode.
- Integration: `/asr` logs include request id in JSON mode.
- Integration: `/tts/stream` generator logs include request id where feasible.
- Integration: `/asr/stream` accepted WS logs include request id.
- Integration: `/v2v/stream` accepted WS logs include request id.
- Integration: WS with `X-Request-ID` propagates into handler logs.
- Integration: WS without `X-Request-ID` generates one.
- Integration: auth rejection logs do not expose raw Authorization.
- Integration: WS token query rejection logs do not expose raw token.
- Smoke: production compose env includes `OVS_LOG_FORMAT=json` in all four speech services.
- Smoke: `docker compose config` for each file renders `OVS_LOG_FORMAT`.
- Smoke: JSON logs parse with `jq`.
- Smoke: text mode still human-readable for local dev.

## Cross-Cutting Dependency Diff

- Add `prometheus_client>=0.20,<1`.
- Add `python-json-logger>=2.0,<4`.
- Do not add `pycuda`.
- Do not add `pynvml`.
- Do not add Hailo Python packages.
- Do not add RK Python packages.
- Do not add OTLP exporters.
- Do not add OpenTelemetry packages.
- Keep `fastapi>=0.115.0`.
- Keep `uvicorn[standard]>=0.32.0`.
- Keep `soundfile>=0.12.0`.
- Keep `numpy>=1.24.0`.
- Keep `python-multipart>=0.0.9`.
- Keep `webrtcvad-wheels>=2.0.14`.
- Dependency diff is limited to two required packages.

## Cross-Cutting Env Var Diff

- Add `OVS_METRICS_REQUIRE_KEY`.
- `OVS_METRICS_REQUIRE_KEY` default is unset or false.
- Truthy values are `1`, `true`, `yes`, and `on`.
- Add `OVS_GPU_WATCHDOG_INTERVAL_S`.
- `OVS_GPU_WATCHDOG_INTERVAL_S` default is `5`.
- Add `OVS_LOG_FORMAT`.
- `OVS_LOG_FORMAT` values are `json` and `text`.
- `OVS_LOG_FORMAT` default is `text`.
- Production compose sets `OVS_LOG_FORMAT=json`.
- No new env var should be required to boot.
- Invalid `OVS_GPU_WATCHDOG_INTERVAL_S` falls back to default.
- Invalid `OVS_LOG_FORMAT` falls back to text.
- Invalid `OVS_METRICS_REQUIRE_KEY` values are treated as false unless truthy.

## Profile Schema Diff

- No profile schema fields are required for Week 2.
- No profile JSON changes are required.
- `OVS_GPU_WATCHDOG_INTERVAL_S` is process env only.
- `OVS_LOG_FORMAT` is process env only.
- `OVS_METRICS_REQUIRE_KEY` is process env only.
- Existing profile env blocks may optionally include these values later.
- Profile loader should not validate these values in Week 2.
- BackendManager profile reload should not change these runtime settings.
- Hot reload should not restart the logging subsystem.
- Hot reload should not restart the watchdog task.
- Hot reload may change backend labels through live backend names.
- Hot reload may update backend state metrics.

## Performance Budget Analysis

- Metrics exporter import budget: less than 50 ms process startup overhead on edge devices.
- Metrics exporter memory budget: less than 5 MiB additional RSS for normal series count.
- Metrics observation budget: less than 10 microseconds per counter or gauge update.
- Metrics histogram budget: less than 25 microseconds per observation.
- Metrics scrape budget: less than 50 ms p95 for `/metrics` under normal labels.
- Metrics label budget: all labels are bounded except existing endpoint path labels.
- Metrics hot path budget: avoid queue-depth update on every audio chunk.
- Metrics hot path budget: observe TTS TTFA once per stream.
- Metrics hot path budget: observe ASR decode once per finalize.
- Watchdog loop budget: less than 1 percent CPU on idle edge devices.
- Watchdog shell command timeout budget: each command should timeout below interval.
- Watchdog default interval budget: one check every 5 seconds.
- Watchdog memory budget: less than 2 MiB excluding optional platform library side effects.
- Watchdog request path budget: zero hardware probes in request path.
- Watchdog readiness budget: cached status lookup only.
- JSON logging HTTP middleware budget: less than 1 ms per request.
- JSON formatter budget: log-time only, not request body time.
- Request id generation budget: negligible UUID cost.
- Contextvar set/reset budget: negligible relative to model inference.
- Executor context wrapping budget: only for selected functions.
- Compose env changes have no runtime overhead beyond JSON logging.

## Implementation Sequencing

- Step 1: implement Prometheus metrics primitives in `server/core/metrics.py`.
- Step 2: preserve Week 1 metrics tests and add Prometheus exposition tests.
- Step 3: add `/metrics` endpoint in `server/main.py`.
- Step 4: add optional `/metrics` API key protection with `OVS_METRICS_REQUIRE_KEY`.
- Step 5: instrument BackendManager state and reload counters.
- Step 6: instrument request/session metrics at HTTP and WS anchors.
- Step 7: instrument TTFA, RTF, ASR decode, worker cancel, and queue depth anchors.
- Step 8: verify with unit tests and targeted `/metrics` smoke test.
- Step 9: implement `gpu_watchdog.py` cached background checker.
- Step 10: launch watchdog task during startup and stop it during shutdown.
- Step 11: add `/readyz` detail integration.
- Step 12: add watchdog metrics.
- Step 13: verify CPU-only, mocked Jetson, mocked RK, mocked Hailo, and mocked CUDA paths.
- Step 14: implement logging configuration module.
- Step 15: wire logging setup before current app loggers emit startup logs.
- Step 16: add HTTP request-id middleware.
- Step 17: add WS request context helpers at `/asr/stream` and `/v2v/stream`.
- Step 18: add masking helpers and tests.
- Step 19: add dependency lines.
- Step 20: add `OVS_LOG_FORMAT=json` to all four speech compose environment blocks.
- Step 21: run full targeted regression suite.
- Step 22: run smoke checks against local app if feasible.
- Step 23: compare `bench/perf` client timing with server histograms for sanity.
- Each step should be independently verifiable.
- Metrics can ship before watchdog.
- Watchdog can ship before JSON logging.
- JSON logging can ship last because it changes operator-facing output.

## Regression Test Matrix

- Metrics unit tests must run without GPU hardware.
- Watchdog unit tests must run without GPU hardware.
- Logging unit tests must run without GPU hardware.
- `/metrics` integration tests must run without model preload when possible.
- `/readyz` integration tests should monkeypatch backend manager and watchdog state.
- `/tts/stream` TTFA tests can use fake backend generators.
- `/asr/stream` decode tests can use fake stream objects.
- BackendManager metrics tests can use fake factories and preloaders.
- Compose tests can parse YAML or use `docker compose config` when available.
- Perf smoke tests can be manual on target hardware.

## Out Of Scope

- No OTLP.
- No tracing.
- No OpenTelemetry.
- No distributed trace propagation.
- No per-IP rate limiting.
- No new admin auth model.
- No BackendManager refactor.
- No change to BackendManager state machine semantics.
- No change to profile schema.
- No change to API response payloads except additive `/readyz` detail.
- No change to `/health` deprecation policy.
- No removal of temporary `/health` worker cancel count in Week 2.
- No per-sentence TTS histogram.
- No Prometheus Pushgateway.
- No Grafana dashboard in this spec.
- No alert rules in this spec.
- No persistent metric storage in this service.
- No hard dependency on CUDA Python packages.
- No hard dependency on Hailo packages.
- No hard dependency on RK packages.
- No new deployment service.

## Acceptance Summary

- `/metrics` exists and exposes Prometheus text exposition.
- Existing Week 1 metrics helper signatures still work.
- New TTS, ASR, backend, worker, WS, queue, and watchdog metrics exist.
- `/metrics` is open by default.
- `OVS_METRICS_REQUIRE_KEY` can protect `/metrics`.
- GPU watchdog runs in the background every 5 seconds by default.
- GPU watchdog supports Jetson, RK, Hailo, desktop CUDA, and CPU-only paths.
- GPU watchdog uses hysteresis: 3 failures to fail, 5 successes to recover.
- `is_ok()` remains a bool cached read.
- `status()` returns a dict with `reason`.
- `/readyz` returns 503 while watchdog is failed.
- JSON logging is opt-in through `OVS_LOG_FORMAT=json`.
- Text logging remains default.
- Production compose files set `OVS_LOG_FORMAT=json`.
- Request IDs are generated or propagated for HTTP and WS.
- Sensitive Authorization and `token` values are masked.
- No implementation code changes are included in this spec file.
