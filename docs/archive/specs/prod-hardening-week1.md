# Production Hardening Week 1 Spec

## Overview

This project is an edge V2V voice pipeline for ASR, LLM-driven turn handling, and TTS on Jetson Orin Nano/NX, RK3576/RK3588, and Raspberry Pi plus Hailo.
Week 1 hardening protects the public API boundary, rejects overload immediately, and separates process liveness from backend readiness.
This is design only; it specifies behavior, modules, touch points, and verification, with no implementation code.

| Deliverable | Summary | Primary files |
| --- | --- | --- |
| Optional API-Key Auth | Default-off API-key protection for public voice endpoints. | `server/core/api_auth.py`, `server/main.py` |
| Global Concurrent Session Limit | Non-queueing admission control with HTTP `429` and WS `4429`. | `server/core/session_limiter.py`, `server/core/metrics.py`, `server/main.py` |
| `/livez` + `/readyz` | Production probes while preserving `/health`. | `server/main.py`, `server/core/gpu_watchdog.py`, docker-compose healthchecks |

Guiding principles:

- Default-off auth: unset or empty auth configuration preserves current behavior.
- Reject-not-queue: edge devices should fail fast under pressure instead of accumulating latency.
- Probe-not-guess: orchestrators should rely on explicit liveness/readiness signals.

## Source Anchors Read

Files read:

- `server/core/admin_auth.py`
- `server/core/profile_loader.py`
- `server/core/backend_manager.py`
- `server/main.py`
- `deploy/docker/` listing

Admin auth anchors:

- `server/core/admin_auth.py:1` documents the existing admin route policy.
- `server/core/admin_auth.py:5` allows loopback clients without a key.
- `server/core/admin_auth.py:8` defines `OVS_ADMIN_KEY`.
- `server/core/admin_auth.py:10` intentionally ignores `X-Forwarded-For`.
- `server/core/admin_auth.py:36` reads the admin key from env.
- `server/core/admin_auth.py:40` strips empty key values.
- `server/core/admin_auth.py:44` defines `require_admin`.
- `server/core/admin_auth.py:55` implements loopback bypass.
- `server/core/admin_auth.py:58` reloads `OVS_ADMIN_KEY` per dependency call.
- `server/core/admin_auth.py:70` uses `hmac.compare_digest`.

Profile loader anchors:

- `server/core/profile_loader.py:3` says profiles set env defaults before backend imports.
- `server/core/profile_loader.py:26` defines operator-owned env prefixes.
- `server/core/profile_loader.py:40` treats declared-but-empty docker env values specially.
- `server/core/profile_loader.py:70` defines `_env`.
- `server/core/profile_loader.py:92` resolves profile references.
- `server/core/profile_loader.py:94` checks `OVS_PROFILE_JSON`, `OVS_PROFILE`, and `OVS_PROFILE_DEFAULT`.
- `server/core/profile_loader.py:143` exposes `current_profile()`.
- `server/core/profile_loader.py:155` defines `apply_profile()`.
- `server/core/profile_loader.py:192` reads the profile `env` block.
- `server/core/profile_loader.py:219` writes profile env unless operator-owned.
- `server/core/profile_loader.py:243` defines `apply_profile_from_env()`.

BackendManager anchors:

- `server/core/backend_manager.py:49` defines `BackendState`.
- `server/core/backend_manager.py:51` defines `BackendState.READY`.
- `server/core/backend_manager.py:112` stores `_state`.
- `server/core/backend_manager.py:150` transitions to READY during `start()`.
- `server/core/backend_manager.py:169` exposes `state`.
- `server/core/backend_manager.py:183` exposes `is_ready()`.
- `server/core/backend_manager.py:199` exposes `status()`.
- `server/core/backend_manager.py:210` defines `acquire()`.
- `server/core/backend_manager.py:213` rejects acquire when not READY.
- `server/core/backend_manager.py:218` increments `_inflight_http`.
- `server/core/backend_manager.py:223` decrements `_inflight_http`.
- `server/core/backend_manager.py:229` registers WebSocket handles.
- `server/core/backend_manager.py:232` unregisters WebSocket handles.
- `server/core/backend_manager.py:497` initializes manager singletons.
- `server/core/backend_manager.py:541` exposes `tts_manager()`.
- `server/core/backend_manager.py:547` exposes `asr_manager()`.

Main app anchors:

- `server/main.py:35` creates the FastAPI app.
- `server/main.py:71` declares `_tts_stream_executor`.
- `server/main.py:251` defines `_get_tts_stream_executor()`.
- `server/main.py:286` constructs `_tts_stream_executor`.
- `server/main.py:287` reads `OVS_TTS_STREAM_MAX_WORKERS`, default `2`.
- `server/main.py:318` defines startup.
- `server/main.py:323` imports profile loader helpers.
- `server/main.py:324` applies the active profile from env.
- `server/main.py:476` starts BackendManager wiring.
- `server/main.py:538` calls `init_backend_managers`.
- `server/main.py:552` starts ASR manager.
- `server/main.py:554` starts TTS manager when configured and ready.
- `server/main.py:563` defines `/health`.
- `server/main.py:599` defines `/asr/capabilities`.
- `server/main.py:615` defines `/tts/capabilities`.
- `server/main.py:641` defines `/tts/speakers`.
- `server/main.py:656` defines `/tts/speakers/register`.
- `server/main.py:701` defines `/tts/speakers/{speaker_id}` delete.
- `server/main.py:722` defines `/tts`.
- `server/main.py:768` defines `/tts/stream` OPTIONS.
- `server/main.py:773` defines `/tts/stream` POST.
- `server/main.py:803` enters manager path for `/tts/stream`.
- `server/main.py:1011` releases `/tts/stream` manager acquire in generator finalization.
- `server/main.py:1130` defines `/tts/clone`.
- `server/main.py:1179` defines `/tts/clone/embedding`.
- `server/main.py:1208` defines `/tts/clone/stream`.
- `server/main.py:1258` acquires manager for `/tts/clone/stream`.
- `server/main.py:1286` releases clone-stream manager acquire in generator finalization.
- `server/main.py:1325` defines `/asr`.
- `server/main.py:1361` defines `/asr/stream`.
- `server/main.py:1386` accepts `/asr/stream`.
- `server/main.py:1394` registers `/asr/stream` with BackendManager.
- `server/main.py:1430` unregisters `/asr/stream`.
- `server/main.py:1584` defines `/v2v/stream`.
- `server/main.py:1621` accepts `/v2v/stream`.
- `server/main.py:1630` registers V2V with ASR manager.
- `server/main.py:1632` registers V2V with TTS manager.
- `server/main.py:2152` runs V2V TTS synthesis on `_tts_stream_executor`.
- `server/main.py:2250` unregisters V2V from ASR manager.
- `server/main.py:2252` unregisters V2V from TTS manager.
- `server/main.py:2299` defines admin TTS runtime GET.
- `server/main.py:2316` defines admin TTS runtime PATCH.
- `server/main.py:2349` defines admin speakers reload.
- `server/main.py:2373` defines admin backend reload.
- `server/main.py:2391` defines admin backend status.

Docker discovery:

- `deploy/docker/` contains Dockerfiles and entrypoint scripts.
- `deploy/docker/docker-compose*.yml`: not found — verify before implementing.
- `deploy/docker/docker-compose*.yaml`: not found — verify before implementing.
- Repo-level compose files outside that requested directory exist: `deploy/docker-compose.yml`, `deploy/docker-compose.radxa.yml`, `deploy/docker-compose.rk.yml`, `deploy/docker-compose.rpi.yml`.

## Deliverable 1: Optional API-Key Auth

### Goal

Add optional API-key protection for public voice endpoints.
The feature is disabled when `OVS_API_KEYS` is unset or empty.
When disabled, all requests pass as they do today.
When enabled, public voice endpoints require a configured key.
Admin endpoints keep existing loopback/shared-secret auth.
Health and probe endpoints stay open.

### New Module

Proposed path:

- `server/core/api_auth.py`

Responsibilities:

- Parse `OVS_API_KEYS`.
- Detect whether auth is enabled.
- Extract HTTP `Authorization: Bearer <key>`.
- Extract WS bearer header or `?token=<key>`.
- Compare candidate keys in constant time.
- Build HTTP `401` responses.
- Build WS `4401` close reasons.
- Mask key values for logs.
- Expose a small check API for `server/main.py`.

Non-responsibilities:

- Admin route auth.
- Session limiting.
- Route registration.
- Prometheus export.

### Env Var

Name:

- `OVS_API_KEYS`

Format:

- Comma-separated string.
- Strip whitespace around each comma-separated entry.
- Drop empty entries.
- Unset means disabled.
- Empty string means disabled.
- Whitespace-only means disabled.
- Comma-only means disabled.

Examples:

- unset: disabled.
- `""`: disabled.
- `" "`: disabled.
- `"abc"`: one key.
- `"abc,def"`: two keys.
- `"abc, def"`: two keys after stripping.
- `"abc,,def"`: two keys after dropping the empty entry.

Hot update:

- Read env on every request check.
- Do not cache parsed keys across requests.
- This mirrors `admin_auth.py`: `_admin_key()` reads env at `server/core/admin_auth.py:36` and is called at `server/core/admin_auth.py:58`.
- Active WS sessions continue after a key change.
- New HTTP requests and new WS sessions use the new env value.

Comparison:

- Use constant-time comparison.
- Follow the `hmac.compare_digest` pattern at `server/core/admin_auth.py:70`.

### Protected Endpoints

When auth is enabled, protect these public voice endpoints:

| Endpoint | Method | Anchor | Credential |
| --- | --- | --- | --- |
| `/asr/capabilities` | GET | `server/main.py:599` | HTTP bearer |
| `/tts/capabilities` | GET | `server/main.py:615` | HTTP bearer |
| `/tts/speakers` | GET | `server/main.py:641` | HTTP bearer |
| `/tts/speakers/register` | POST | `server/main.py:656` | HTTP bearer |
| `/tts/speakers/{speaker_id}` | DELETE | `server/main.py:701` | HTTP bearer |
| `/tts` | POST | `server/main.py:722` | HTTP bearer |
| `/tts/stream` | POST | `server/main.py:773` | HTTP bearer |
| `/tts/clone` | POST | `server/main.py:1130` | HTTP bearer |
| `/tts/clone/embedding` | POST | `server/main.py:1179` | HTTP bearer |
| `/tts/clone/stream` | POST | `server/main.py:1208` | HTTP bearer |
| `/asr` | POST | `server/main.py:1325` | HTTP bearer |
| `/asr/stream` | WS | `server/main.py:1361` | bearer or query token |
| `/v2v/stream` | WS | `server/main.py:1584` | bearer or query token |

The task explicitly calls out `/asr/stream` and `/tts`.
Other endpoints in the table are public voice endpoints found in `server/main.py`.
`/tts/stream` OPTIONS at `server/main.py:768` should remain open for preflight unless a separate CORS policy says otherwise.

### Unprotected Endpoints

Do not protect with `OVS_API_KEYS`:

- `/health`, existing route at `server/main.py:563`.
- `/livez`, new route.
- `/readyz`, new route.
- `/admin/tts/runtime`, GET at `server/main.py:2299`.
- `/admin/tts/runtime`, PATCH at `server/main.py:2316`.
- `/admin/tts/speakers/reload` at `server/main.py:2349`.
- `/admin/backend/reload` at `server/main.py:2373`.
- `/admin/backend/status` at `server/main.py:2391`.

Admin invariants:

- Keep loopback bypass at `server/core/admin_auth.py:55`.
- Keep `OVS_ADMIN_KEY` separate from `OVS_API_KEYS`.
- `OVS_API_KEYS` must not satisfy admin auth.
- `OVS_ADMIN_KEY` must not satisfy public API auth.

### HTTP Auth Contract

Supported credential:

- `Authorization: Bearer <key>`

Unsupported for HTTP:

- Query token.
- Cookie token.
- Custom token header.
- Basic auth.

Disabled behavior:

- Missing Authorization passes.
- Invalid Authorization passes.
- Malformed Authorization passes.

Enabled behavior:

- Missing Authorization fails.
- Non-Bearer scheme fails.
- Empty bearer token fails.
- Unknown token fails.
- Matching token passes.

Failure:

- Status `401`.
- JSON body `{"error":"unauthorized"}`.
- Optional detail `missing_or_invalid_api_key`.
- Header `WWW-Authenticate: Bearer`.
- Never echo token values.

### WebSocket Auth Contract

Supported credentials:

- `Authorization: Bearer <key>`.
- `?token=<key>`.

Header precedence:

- Header wins if both header and query token exist.
- Invalid header fails even if query token is valid.

Enabled failure:

- Accept only if needed for deterministic application close.
- Close code `4401`.
- JSON reason `{"error":"unauthorized"}`.
- Optional detail `missing_or_invalid_api_key`.
- Never include supplied token or configured key in reason.

WebSocket upgrade note:

- HTTP `401` is only possible before the WebSocket upgrade completes.
- After `accept()`, Starlette/FastAPI can only send WS frames and close frames.
- Existing `/asr/stream` accepts at `server/main.py:1386`.
- Existing `/v2v/stream` accepts at `server/main.py:1621`.
- Week 1 requires deterministic close: check before backend allocation, accept, close `4401`, and return.

### Data Flow

Required diagram:

`request -> FastAPI middleware/dependency -> api_auth.check() -> endpoint OR 401/4429`

HTTP flow:

`HTTP request -> api_auth.check_http() -> endpoint OR 401`

WS flow:

`WS request -> api_auth.check_ws() -> endpoint OR accept+close(4401)`

Ordering:

- Auth runs before session limiter.
- Unauthorized requests do not consume session slots.
- Session limiter owns `429` and `4429`.

### Affected Files

Add:

- `server/core/api_auth.py`

Modify:

- `server/main.py`

Primary anchors:

- HTTP guards near `server/main.py:599`, `:615`, `:641`, `:656`, `:701`, `:722`, `:773`, `:1130`, `:1179`, `:1208`, `:1325`.
- WS guard near `server/main.py:1361`, before `server/main.py:1394`.
- WS guard near `server/main.py:1584`, before `server/main.py:1630`.

### Logging and Masking

Never log full keys.
Allowed mask is first 8 chars plus `...`.
For shorter keys, log the short prefix plus `...`.
Missing supplied key logs as `<missing>`.
Log endpoint, client host if available, reason category, and masked supplied key prefix.
Do not log raw query strings containing `token`.
Do not log Authorization header values.

### Edge Cases

- Unset `OVS_API_KEYS`: disabled.
- Empty `OVS_API_KEYS`: disabled.
- Spaces after comma split: strip.
- Duplicate keys: allowed.
- Reload during active WS session: active session continues; new sessions use new env.
- WS query param URL encoding: rely on Starlette decoding; clients must encode reserved characters.
- Bearer scheme: case-insensitive scheme, stripped token edges, no internal stripping.
- Admin loopback must remain unaffected.
- `/health`, `/livez`, and `/readyz` must remain open.

### Regression Risk

- Global middleware may accidentally gate `/admin/*`.
- Global middleware may accidentally gate probes.
- Query tokens may leak through access logs.
- Full token values may leak through exception logs.
- Capability endpoints may have clients that assume anonymous discovery; if so, carveout must be explicit.

### Test Checklist

- Unit: unset env disables auth.
- Unit: empty, whitespace, and comma-only env disable auth.
- Unit: single and multiple keys match.
- Unit: spaces around keys are stripped.
- Unit: missing HTTP bearer fails when enabled.
- Unit: wrong HTTP bearer fails.
- Unit: valid HTTP bearer passes.
- Unit: missing WS credential fails when enabled.
- Unit: valid WS header passes.
- Unit: valid WS query token passes.
- Unit: header precedence is enforced.
- Unit: env hot-update changes next check.
- Unit: mask helper never returns raw long key.
- Integration: `POST /tts` passes when disabled.
- Integration: `POST /tts` returns `401` without key when enabled.
- Integration: `POST /tts` passes with valid bearer.
- Integration: `WS /asr/stream` closes `4401` without key.
- Integration: `WS /asr/stream?token=<key>` passes.
- Integration: `WS /v2v/stream` closes `4401` without key.
- Integration: `/health`, `/livez`, `/readyz`, `/admin/*` remain outside API-key auth.

## Deliverable 2: Global Concurrent Session Limit + 429 Reject

### Goal

Add one process-wide concurrent voice session limiter.
The limiter protects memory, GPU/NPU runtime stability, executor pressure, and latency.
It must reject immediately when full.
It must not queue.

### New Modules

Proposed limiter:

- `server/core/session_limiter.py`

Metrics stub:

- `server/core/metrics.py`

`session_limiter.py` owns limit derivation, process-wide initialization, non-blocking acquire, release, and read-only snapshots.
`metrics.py` owns Week 1 in-process counters and gauges that can be wired to Prometheus in Week 2.

### Configuration

Profile field:

- `max_concurrent_sessions`

Env override:

- `OVS_MAX_CONCURRENT_SESSIONS`

Validation:

- Must parse as integer.
- Must be greater than `0`.
- `0` is startup error.
- Negative values are startup error.
- Non-integer values are startup error.

Defaults:

| Target | Default | Rationale |
| --- | ---: | --- |
| `orin-nx` | 2 | Orin NX has 16GB and can pair with two TTS workers. |
| `orin-nano` | 1 | Orin Nano has 8GB; smaller limit reflects single TTS worker stability. |
| `rk` | 1 | RK NPU deployments prioritize stability. |
| `desktop` | 4 | Development and CI target. |
| unknown | 1 | Conservative fallback. |

Precedence:

- Env override wins.
- Profile JSON field wins over inferred target default.
- Target default wins over unknown fallback.

### Startup and Profile Integration

Relevant anchors:

- Startup begins at `server/main.py:318`.
- Profile is applied at `server/main.py:324`.
- Model download starts at `server/main.py:357`.
- `current_profile()` is at `server/core/profile_loader.py:143`.
- Profile `env` handling starts at `server/core/profile_loader.py:192`.

Recommended placement:

- Initialize limiter after profile application at `server/main.py:324`.
- Initialize before model downloads and backend preload.
- Fail startup early if configuration is invalid.

Profile schema:

- Add top-level integer `max_concurrent_sessions`.
- Do not place it under profile `env`.
- Read it from `current_profile()`.
- Let `OVS_MAX_CONCURRENT_SESSIONS` override it.

Target inference may inspect profile name, `LANGUAGE_MODE`, `RK_PLATFORM`, and env equivalents.

### Slot Accounting

Counts as one slot:

- One `/asr/stream` WS session, including its ASR lifetime.
- One `/v2v/stream` WS session, including ASR and TTS turns.
- One `/tts` HTTP request.
- One `/tts/stream` HTTP request until streaming ends.
- One `/tts/clone` HTTP request.
- One `/tts/clone/stream` HTTP request until streaming ends.
- One `/asr` HTTP request.
- Recommended: one `/tts/clone/embedding` request.

Does not count:

- Capability endpoints.
- Speaker list endpoint.
- Admin endpoints.
- `/health`, `/livez`, `/readyz`.

Session-limited anchors:

- `/tts`: `server/main.py:722`.
- `/tts/stream`: `server/main.py:773`.
- `/tts/clone`: `server/main.py:1130`.
- `/tts/clone/embedding`: `server/main.py:1179`.
- `/tts/clone/stream`: `server/main.py:1208`.
- `/asr`: `server/main.py:1325`.
- `/asr/stream`: `server/main.py:1361`.
- `/v2v/stream`: `server/main.py:1584`.

### Acquire Semantics

Use `asyncio.Semaphore` at app startup, sized from resolved limit, or implement equivalent non-blocking behavior with an `asyncio.Lock` plus integer active count.
Required behavior is `acquire_nowait()`.
If no slot is available, reject immediately.
Do not await capacity.
Do not enqueue.
Verify `asyncio.Semaphore.acquire_nowait()` support before implementation; if unavailable, use the lock/counter approach.

Limiter interface should expose:

- `limit`.
- `active`.
- `available`.
- `try_acquire()`.
- Release token or async context manager.

Release rules:

- Release on all normal exits.
- Release on exceptions.
- Release on client disconnect.
- Guard against double release.
- Never let active count go negative.

### HTTP Reject Contract

When full:

- Status `429`.
- Body `{"error":"too_many_sessions","current":N,"limit":M}`.
- Header `Retry-After: 5`.
- Increment `ovs_sessions_rejected_total` with `reason="http"`.
- Do not acquire BackendManager.
- Do not start ASR/TTS work.
- Do not enter `_tts_stream_executor`.

`current` is active holders at rejection time.
`limit` is configured maximum.

### WebSocket Reject Contract

When full:

- Accept the WebSocket.
- Close immediately with code `4429`.
- Reason `{"error":"too_many_sessions","current":N,"limit":M}`.
- Increment `ovs_sessions_rejected_total` with `reason="ws"`.
- Do not register with BackendManager.
- Do not initialize VAD.
- Do not create ASR session state.
- Do not create TTS queues.

Slot lifetime:

- Acquire at WS accept.
- Release at WS close, including error close.
- For `/asr/stream`, existing accept is `server/main.py:1386` and unregister is `server/main.py:1430`.
- For `/v2v/stream`, existing accept is `server/main.py:1621` and unregisters are `server/main.py:2250` and `server/main.py:2252`.

### Data Flow

HTTP admitted:

`HTTP request -> api_auth.check_http() -> session_limiter.try_acquire() -> endpoint -> response/generator finally -> release`

HTTP rejected:

`HTTP request -> api_auth.check_http() -> session_limiter.try_acquire() fails -> 429 JSON`

WS admitted:

`WS request -> api_auth.check_ws() -> accept -> session_limiter.try_acquire() -> BackendManager.register_ws() -> handler -> unregister -> release`

WS rejected:

`WS request -> api_auth.check_ws() -> accept -> session_limiter.try_acquire() fails -> close(4429)`

Streaming HTTP:

- Acquire before returning `StreamingResponse`.
- Release in generator `finally`.
- If setup fails after acquire, release before returning or raising.

Streaming anchors:

- `/tts/stream` manager acquire starts at `server/main.py:807`.
- `/tts/stream` manager release is at `server/main.py:1011`.
- `/tts/clone/stream` manager acquire starts at `server/main.py:1258`.
- `/tts/clone/stream` manager release is at `server/main.py:1286`.

### Relationship to `_tts_stream_executor`

The session limiter is decoupled from `_tts_stream_executor`.
The session semaphore is the outer gate for voice sessions.
The TTS thread pool is the inner execution resource.
BackendManager inflight counters are for reload drain, not admission control.

Executor anchors:

- `_tts_stream_executor` declared at `server/main.py:71`.
- `_get_tts_stream_executor()` starts at `server/main.py:251`.
- Executor constructed at `server/main.py:286`.
- Default workers read at `server/main.py:287`.
- `/tts/stream` uses executor at `server/main.py:915` and `server/main.py:962`.
- `/tts/clone/stream` uses executor at `server/main.py:1279`.
- `/v2v/stream` uses executor at `server/main.py:2152`.

Recommendation:

- `max_concurrent_sessions` SHOULD be less than or equal to `_tts_stream_executor.max_workers` for Orin targets.
- Orin Nano: both should normally be `1`.
- Orin NX: both can be `2`.
- Existing comments at `server/main.py:253` through `server/main.py:285` document CUDA and sustained N=2 caveats.

### Metrics Stub

Create `server/core/metrics.py`.
Week 1 has no Prometheus dependency.
Use thread-safe or asyncio-safe counters.
Expose snapshots for tests and readiness.

Metrics:

- `ovs_sessions_active`: gauge-style active holder count.
- `ovs_sessions_rejected_total`: counter with Week 1 reason values `http` and `ws`.

Implementation notes:

- Use integer counters.
- Use `threading.Lock` or equivalent.
- Never label by token, key prefix, client IP, or raw URL.

### Edge Cases

- Session released on unhandled exception through `try/finally`.
- Session released on HTTP streaming disconnect through generator `finally`.
- Session released on WS disconnect through outer handler `finally`.
- BackendManager acquire failure after session acquire must release the session slot.
- Semaphore size change on profile hot-swap requires restart in Week 1.
- Future resize may drain sessions, reinitialize limiter, and resume readiness.
- `OVS_MAX_CONCURRENT_SESSIONS=0` is startup error, not reject-all mode.
- Missing limiter after startup is fatal; do not silently run unlimited.
- `/readyz` reads limiter state without acquiring.

### Regression Risk

- Too-high session limit can expose `_tts_stream_executor` and CUDA shared-state issues.
- BackendManager `acquire()` still must wrap backend work so reload drain sees inflight HTTP at `server/core/backend_manager.py:218`.
- BackendManager registered sockets may be force-closed during reload; limiter release still must run.
- `/readyz` will report not-ready when slots are full; this is intentional and must not be used as liveness.

### Test Checklist

- Unit: env override wins over profile.
- Unit: profile value wins over target default.
- Unit: Orin NX default is 2.
- Unit: Orin Nano default is 1.
- Unit: RK default is 1.
- Unit: desktop default is 4.
- Unit: unknown default is 1.
- Unit: zero, negative, and non-integer values fail validation.
- Unit: acquire succeeds below limit.
- Unit: acquire fails at limit.
- Unit: release decrements active.
- Unit: double release cannot make active negative.
- Unit: rejection increments counter.
- Integration: with limit 1, first `/tts` enters and concurrent second `/tts` returns `429`.
- Integration: `429` body includes `error`, `current`, `limit`.
- Integration: `429` includes `Retry-After: 5`.
- Integration: with limit 1, second `/asr/stream` closes `4429`.
- Integration: `4429` reason is JSON.
- Integration: slot releases after WS close, streaming disconnect, and exception.
- Integration: `/readyz` returns 503 when active equals limit.

## Deliverable 3: `/livez` + `/readyz`

### Goal

Add explicit production probes.
`/livez` indicates the process is alive enough to route.
`/readyz` indicates the service should receive new voice traffic.
`/health` remains compatible but is deprecated.

### Existing `/health`

Existing route:

- `server/main.py:563` defines `/health`.
- `server/main.py:596` returns the current result.

Current body reports:

- TTS readiness.
- TTS backend.
- TTS capabilities.
- ASR readiness.
- ASR backend.
- ASR capabilities.
- Worker cancel count when available.

Week 1 behavior:

- Preserve body and status compatibility.
- Add headers.
- Do not require API key.
- Do not consume a session slot.

### `/livez`

Route handler description:

- Method GET.
- Path `/livez`.
- Handler name `livez`.
- Signature has no request body, no auth dependency, and no session dependency.
- Response `200 {"status":"ok"}`.

Requirements:

- Always 200 while process can serve the route.
- No backend dependency.
- No GPU dependency.
- No model dependency.
- No profile dependency.
- Target latency below 1ms under normal local conditions.

Placement:

- Add near health section beginning at `server/main.py:561`.
- Place adjacent to existing `/health` at `server/main.py:563`.

### `/readyz`

Route handler description:

- Method GET.
- Path `/readyz`.
- Handler name `readyz`.
- Signature has no request body, no auth dependency, and read-only state checks.
- Ready response `200 {"status":"ready"}`.
- Not-ready response `503 {"status":"not_ready","reasons":[...]}`.

Ready only when all are true:

- Required BackendManager state is READY.
- Session semaphore has capacity.
- `gpu_watchdog.is_ok()` is true.

Reason strings:

- `backend_not_ready`.
- `backend_manager_unavailable`.
- `sessions_full`.
- `session_limiter_unavailable`.
- `gpu_watchdog_failed`.

Reasons must be stable and machine-readable.

### BackendManager Check

State enum:

- `BackendState.READY` is at `server/core/backend_manager.py:51`.

Accessors:

- `tts_manager()` at `server/core/backend_manager.py:541`.
- `asr_manager()` at `server/core/backend_manager.py:547`.
- `BackendManager.state` at `server/core/backend_manager.py:169`.
- `BackendManager.is_ready()` at `server/core/backend_manager.py:183`.

Interpretation:

- All required managers for the active profile must be READY.
- ASR manager is required when ASR is configured.
- TTS manager is required when TTS is configured and not intentionally absent.
- ASR-only profiles should not fail because TTS is absent.
- Existing TTS configured check appears at `server/main.py:421`.

Lazy TTS:

- Existing startup supports `LAZY_TTS` at `server/main.py:423`.
- Week 1 must choose readiness semantics before implementation.
- Recommended production posture: avoid `LAZY_TTS` in readiness-managed deployments.

### Session Capacity Check

Ready condition:

- `active_sessions < limit`.

Not-ready condition:

- `active_sessions >= limit`.

This is a soft signal that the next request would be rejected.
It is not the hard gate; session limiter admission remains the hard gate.
`/readyz` must not acquire a session slot.
`/readyz` must not mutate limiter metrics.

### GPU Watchdog Stub

New module:

- `server/core/gpu_watchdog.py`

Function:

- `is_ok() -> bool`

Week 1 behavior:

- Always returns true.
- Include TODO for Week 2 hardware checks.
- Do not import CUDA, RKNN, Hailo, NVML, or platform-specific libraries.

Week 2 candidates:

- GPU memory pressure.
- NPU runtime status.
- Thermal throttling.
- CUDA context health.
- Device fault state.

### `/health` Deprecation Headers

Add:

- `Deprecation: true`
- `Link: </readyz>; rel="successor-version"`

Reference:

- RFC 8594 for the `Deprecation` response header.

Implementation note:

- Preserve current body.
- Preserve current status behavior.
- Add headers via response object or explicit JSON response.

### Docker Compose Healthcheck

Requirement:

- Change healthcheck from `/health` to `/readyz` in all `deploy/docker/docker-compose*.yml` files.

Files found under requested path:

- `deploy/docker/docker-compose*.yml`: not found — verify before implementing.
- `deploy/docker/docker-compose*.yaml`: not found — verify before implementing.

No before/after healthcheck diff block can be shown for that path because no matching file exists.

Files found outside requested path:

- `deploy/docker-compose.yml`.
- `deploy/docker-compose.radxa.yml`.
- `deploy/docker-compose.rk.yml`.
- `deploy/docker-compose.rpi.yml`.

Before implementation:

- Confirm whether the intended path is `deploy/` rather than `deploy/docker/`.
- If repo-level files are in scope, preserve interval, timeout, retries, start period, and command style.
- Change only `/health` to `/readyz`.

Current diff status:

- not found — verify before implementing.

### Edge Cases

- During backend DRAINING, RELOADING, or FAILED, `/readyz` returns 503.
- Backend state machine is documented at `server/core/backend_manager.py:11` through `server/core/backend_manager.py:22`.
- After successful reload or rollback to READY, `/readyz` returns 200 again.
- Concurrent `/readyz` probes are read-only and do not allocate sessions.
- When sessions are full, `/readyz` returns 503.
- Existing CI callers may still use `/health`; keep it working and warn maintainers to move to `/readyz`.
- Docker healthcheck timing values must be preserved.

### Regression Risk

- Using `/readyz` as liveness can restart healthy but saturated containers; `/livez` is liveness.
- Strict readiness can mark lazy TTS deployments unready; decide policy before implementation.
- Startup catches BackendManager wiring exceptions at `server/main.py:555`; `/readyz` should return `backend_manager_unavailable`.
- Changing `/health` body can break callers; preserve it.

### Test Checklist

- `/livez` returns `200 {"status":"ok"}`.
- `/livez` does not depend on backend readiness.
- `/readyz` returns 200 when managers READY, capacity exists, and watchdog OK.
- `/readyz` returns 503 when manager INIT, DRAINING, RELOADING, or FAILED.
- `/readyz` returns 503 when sessions are full.
- `/readyz` returns 503 when watchdog is patched false.
- `/readyz` returns 200 for ASR-only profile without TTS requirement.
- `/health` body preserves existing keys.
- `/health` includes `Deprecation: true`.
- `/health` includes `Link: </readyz>; rel="successor-version"`.
- `/health`, `/livez`, and `/readyz` remain open when API keys are enabled.

## Profile Schema Diff

| Name | Type | Default | Profile field | docker-compose env passthrough | Description |
| --- | --- | --- | --- | --- | --- |
| `OVS_API_KEYS` | comma-separated string | unset, disabled | none | yes | Enables public voice API-key auth when non-empty. |
| `OVS_MAX_CONCURRENT_SESSIONS` | integer string | target-derived | overrides `max_concurrent_sessions` | yes | Overrides global voice session limit; must be greater than 0. |
| `max_concurrent_sessions` | integer | target-derived | top-level JSON field | no | Profile-declared session limit. |

Target defaults:

| Target | Default |
| --- | ---: |
| `orin-nx` | 2 |
| `orin-nano` | 1 |
| `rk` | 1 |
| `desktop` | 4 |
| unknown | 1 |

Profile placement:

- Add `max_concurrent_sessions` as a top-level JSON integer.
- Do not add it to profile `env`.
- Keep env passthrough for `OVS_API_KEYS` and `OVS_MAX_CONCURRENT_SESSIONS`.
- Do not hard-code API secrets in compose.
- `deploy/docker/docker-compose*.yml`: not found — verify before implementing.

## Metrics Naming Convention

Week 2 metric names:

- `ovs_sessions_active`
- `ovs_sessions_rejected_total`
- `ovs_auth_rejected_total`
- `ovs_readyz_check_duration_seconds`

Metric definitions:

- `ovs_sessions_active`: gauge, no labels, current session holders.
- `ovs_sessions_rejected_total`: counter, label `reason=[ws|http]`, admission rejections.
- `ovs_auth_rejected_total`: counter, label `endpoint`, API-key rejections.
- `ovs_readyz_check_duration_seconds`: histogram, readiness check duration for Week 2.

Naming rules:

- Use `ovs_` prefix.
- Use snake_case.
- Use `_total` for counters.
- Use `_seconds` for durations.
- Keep labels low-cardinality.
- Never label by API key, token prefix, client IP, or raw URL.

Week 1:

- Implement only lightweight in-process counters needed by limiter tests and readiness.
- Avoid Prometheus dependency.
- Keep the API narrow for Week 2 replacement.

## Implementation Sequence

### 1. Create `server/core/metrics.py` Stub

Files:

- `server/core/metrics.py`

Verify:

- `pytest tests -k metrics`
- Confirm active increments and decrements.
- Confirm rejected counters increment by reason.

### 2. Create `server/core/api_auth.py`

Files:

- `server/core/api_auth.py`

Verify:

- `pytest tests -k api_auth`
- `curl -i -X POST http://127.0.0.1:8000/tts`
- `curl -i -H 'Authorization: Bearer test-key' -X POST http://127.0.0.1:8000/tts`

### 3. Create `server/core/session_limiter.py`

Files:

- `server/core/session_limiter.py`
- `server/core/metrics.py`

Verify:

- `pytest tests -k session_limiter`
- Configure limit `1`, hold one slot, and confirm second acquire rejects immediately.

### 4. Create `server/core/gpu_watchdog.py` Stub

Files:

- `server/core/gpu_watchdog.py`

Verify:

- `pytest tests -k gpu_watchdog`
- Import module and assert `is_ok()` is true.

### 5. Wire Auth Middleware or Guards into `server/main.py`

Files:

- `server/main.py`

Anchors:

- `server/main.py:599`, `:615`, `:641`, `:656`, `:701`, `:722`, `:773`, `:1130`, `:1179`, `:1208`, `:1325`, `:1361`, `:1584`.

Verify:

- Auth disabled: protected endpoints behave as before.
- Auth enabled: missing HTTP key returns `401`.
- Auth enabled: valid bearer passes.
- Auth enabled: missing WS key closes `4401`.
- `/health`, `/livez`, `/readyz`, and `/admin/*` remain outside API-key auth.

### 6. Wire Session Limiter into WS Accept and HTTP POST Handlers

Files:

- `server/main.py`
- `server/core/session_limiter.py`
- `server/core/metrics.py`

Anchors:

- Startup near `server/main.py:318`.
- Profile applied at `server/main.py:324`.
- HTTP handlers at `server/main.py:722`, `:773`, `:1130`, `:1179`, `:1208`, `:1325`.
- WS accepts at `server/main.py:1386` and `server/main.py:1621`.

Verify:

- `OVS_MAX_CONCURRENT_SESSIONS=1`.
- First long WS holds slot.
- Second WS closes `4429`.
- Concurrent HTTP voice request returns `429`.
- Slot releases after close.

### 7. Add `/livez` and `/readyz` Routes to `server/main.py`

Files:

- `server/main.py`
- `server/core/gpu_watchdog.py`
- `server/core/session_limiter.py`

Anchors:

- Health section at `server/main.py:561`.
- Existing `/health` at `server/main.py:563`.
- `BackendState.READY` at `server/core/backend_manager.py:51`.
- Manager `state` at `server/core/backend_manager.py:169`.

Verify:

- `curl -i http://127.0.0.1:8000/livez`
- `curl -i http://127.0.0.1:8000/readyz`
- Ready path returns 200.
- Not-ready path returns 503 with reasons.

### 8. Update `/health` Route with Deprecation Headers

Files:

- `server/main.py`

Anchors:

- Route starts at `server/main.py:563`.
- Existing return is at `server/main.py:596`.

Verify:

- `curl -i http://127.0.0.1:8000/health`
- Existing JSON keys remain.
- `Deprecation: true` exists.
- `Link: </readyz>; rel="successor-version"` exists.

### 9. Update Profile Schema in `profile_loader.py`

Files:

- `server/core/profile_loader.py`
- selected profile JSON files if product owners approve defaults.
- profile validation tests if present.

Anchors:

- `current_profile()` at `server/core/profile_loader.py:143`.
- `apply_profile()` at `server/core/profile_loader.py:155`.
- Profile env handling at `server/core/profile_loader.py:192`.

Verify:

- Load profile with `max_concurrent_sessions`.
- Confirm `current_profile()` exposes it.
- Confirm env override wins in limiter tests.
- `pytest tests -k 'profile and max_concurrent_sessions'`

### 10. Update Docker-Compose Healthcheck Stanzas

Files:

- `deploy/docker/docker-compose*.yml`: not found — verify before implementing.
- `deploy/docker/docker-compose*.yaml`: not found — verify before implementing.

Likely files if scope is corrected:

- `deploy/docker-compose.yml`
- `deploy/docker-compose.radxa.yml`
- `deploy/docker-compose.rk.yml`
- `deploy/docker-compose.rpi.yml`

Verify:

- `rg -n "/health|/readyz|healthcheck" deploy/docker`
- If scope is corrected: `rg -n "/health|/readyz|healthcheck" deploy`
- Preserve healthcheck interval, timeout, retries, start period, and command style.

## Acceptance Criteria

Deliverable 1:

- `OVS_API_KEYS` unset or empty leaves public voice endpoints open.
- HTTP voice endpoints require bearer token when enabled.
- WS voice endpoints require bearer or query token when enabled.
- Admin endpoints remain on loopback/admin-key auth.
- Probe endpoints remain open.
- Full keys never appear in logs.

Deliverable 2:

- Limit derives from env, profile, or target default.
- HTTP overload returns `429` with required body and `Retry-After`.
- WS overload closes `4429` with required JSON reason.
- No overload path queues.
- Slots release on completion, disconnect, and exceptions.
- Metrics stub tracks active and rejected sessions.

Deliverable 3:

- `/livez` returns 200 independently of backend state.
- `/readyz` returns 200 only when managers READY, capacity exists, and watchdog OK.
- `/readyz` returns 503 with stable reasons otherwise.
- `/health` body stays compatible and includes deprecation headers.
- Compose path mismatch is resolved before implementation.

## Non-Goals

- No OAuth, JWT, user identity model, or IP rate limiting.
- No Prometheus dependency or real GPU/NPU telemetry in Week 1.
- No semaphore resize during hot profile reload.
- No BackendManager reload redesign or `_tts_stream_executor` default change.
- No overload queue, priority scheduling, or per-endpoint limits.
- No API-key protection for admin routes or probes.

## Open Verification Notes

`deploy/docker/docker-compose*.yml` was not found.
Repo-level compose files exist under `deploy/`.
Confirm path intent before implementation.
If repo-level compose files are in scope, inspect actual healthcheck stanzas and produce exact per-file before/after diffs then.
`asyncio.Semaphore.acquire_nowait()` support must be verified against the project Python version.
If unavailable, implement non-blocking admission with an `asyncio.Lock` plus integer active count.
`LAZY_TTS` readiness semantics need a product decision before implementation.
Recommended production posture is to avoid lazy TTS when readiness probes control traffic.

## Final Guardrails

Auth before session admission; session admission before backend work.
Readiness is read-only; health probes stay unauthenticated.
Admin auth stays independent; do not log secrets.
Do not queue overload or guess docker-compose paths.
