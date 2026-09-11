> Path note (post-restructure): the product service moved `app/`→`server/`
> (`app/main.py`→`server/main.py`, `app/core/`→`server/core/`). Backend
> implementations cited below as `app/backends/...` (jetson/rk/cpu) now live in the
> `voxedge` package (`voxedge.backends.*`); those `app/backends/...` paths
> are kept verbatim only to preserve the original line-anchored references — map
> them to the corresponding `voxedge` module when implementing.

# MOSS-TTS-Nano N=2 Design (Deliverable 2, design-only)

Status: design only. No code lands from this document. Extracted verbatim
from `docs/archive/specs/prod-hardening-week3.md` §Deliverable 2 for easier
standalone review.

Date: 2026-05-24.
Reference spec: `docs/archive/specs/prod-hardening-week3.md` (full Week 3 brief).

## Overview

Deliverable 2 is design-only.
No code should be implemented for MOSS N=2 in Week 3 unless a separate
decision task approves it.

The current MOSS backend is a Python wrapper around a subprocess worker.
The current profile config is single-slot.

- `MossTtsNanoBackend.__init__()` reads `moss_max_slots` from the profile
  and defaults to 1 at `app/backends/jetson/moss_tts_nano.py:64`.
- The C++ worker receives `--max-slots` from Python at
  `app/backends/jetson/moss_tts_nano.py:136` through
  `app/backends/jetson/moss_tts_nano.py:143`.
- The TRT profile sets `MOSS_MAX_SLOTS` to `"1"` at
  `configs/profiles/jetson-moss-tts-nano-trt.json:21`.
- The TRT profile sets `moss_max_slots` to `1` at
  `configs/profiles/jetson-moss-tts-nano-trt.json:28`.
- The ORT fallback profile sets `MOSS_MAX_SLOTS` to `"1"` at
  `configs/profiles/jetson-moss-tts-nano.json:24`.
- The ORT fallback profile sets `moss_max_slots` to `1` at
  `configs/profiles/jetson-moss-tts-nano.json:31`.

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
- N=2 should be pursued only if customer workload requires simultaneous
  TTS streams or pipeline overlap.
- The first experiment should be a minimum viable POC, not production
  implementation.

Recommended approach:
- Approach C: Python-side multi-slot plus worker IPC demux is the
  recommended POC path.
- It should be treated as a POC only.
- It validates whether the existing single C++ worker can actually host
  two logical in-flight requests.
- It avoids doubling model weights immediately.
- It exposes protocol and state bugs before committing to a larger C++
  redesign.

If Approach C proves the C++ worker cannot safely demux concurrent
slots, stop. Then either keep single-slot or consider Approach A only on
larger devices.

Approach B is the cleanest long-term architecture but the highest
implementation risk.
Approach A is operationally simple but likely too expensive for Orin
Nano 8GB.

## Affected Files

Existing file: `app/backends/jetson/moss_tts_nano.py`.
- File risk notes describe JSONL over stdio at
  `app/backends/jetson/moss_tts_nano.py:3` through
  `app/backends/jetson/moss_tts_nano.py:8`.
- Risk notes say malformed worker output may surface as timeouts at
  `app/backends/jetson/moss_tts_nano.py:3` through
  `app/backends/jetson/moss_tts_nano.py:6`.
- Risk notes say streaming concurrency depends on worker request ids at
  `app/backends/jetson/moss_tts_nano.py:7` through
  `app/backends/jetson/moss_tts_nano.py:8`.
- Risk notes warn respawn is best effort after GPU/TensorRT init
  failures at `app/backends/jetson/moss_tts_nano.py:9` through
  `app/backends/jetson/moss_tts_nano.py:10`.
- Worker paths are resolved from env at
  `app/backends/jetson/moss_tts_nano.py:81` through
  `app/backends/jetson/moss_tts_nano.py:87`.
- Backend name is `jetson.moss_tts_nano` at
  `app/backends/jetson/moss_tts_nano.py:89` through
  `app/backends/jetson/moss_tts_nano.py:92`.
- Capabilities include `BASIC_TTS`, `STREAMING`, `VOICE_CLONE`, and
  `MULTI_LANGUAGE` at `app/backends/jetson/moss_tts_nano.py:93` through
  `app/backends/jetson/moss_tts_nano.py:100`.
- `is_ready()` checks a single process at
  `app/backends/jetson/moss_tts_nano.py:106` through
  `app/backends/jetson/moss_tts_nano.py:108`.
- `preload()` starts one worker process at
  `app/backends/jetson/moss_tts_nano.py:110` through
  `app/backends/jetson/moss_tts_nano.py:158`.
- Python worker path and C++ worker path diverge at
  `app/backends/jetson/moss_tts_nano.py:120` through
  `app/backends/jetson/moss_tts_nano.py:143`.
- C++ path passes `--max-slots` at
  `app/backends/jetson/moss_tts_nano.py:136` through
  `app/backends/jetson/moss_tts_nano.py:143`.
- One stdout reader and one stderr reader are started at
  `app/backends/jetson/moss_tts_nano.py:159` through
  `app/backends/jetson/moss_tts_nano.py:172`.
- Startup waits for `worker_ready` at
  `app/backends/jetson/moss_tts_nano.py:174` through
  `app/backends/jetson/moss_tts_nano.py:196`.
- `shutdown()` clears all request queues and stops the single process at
  `app/backends/jetson/moss_tts_nano.py:198` through
  `app/backends/jetson/moss_tts_nano.py:209`.
- `generate_streaming()` creates one UUID per request at
  `app/backends/jetson/moss_tts_nano.py:214` through
  `app/backends/jetson/moss_tts_nano.py:218`.
- `generate_streaming()` registers a queue and sends one request at
  `app/backends/jetson/moss_tts_nano.py:221` through
  `app/backends/jetson/moss_tts_nano.py:227`.
- Chunk events are decoded from base64 at
  `app/backends/jetson/moss_tts_nano.py:241` through
  `app/backends/jetson/moss_tts_nano.py:256`.
- Done metadata is stored in thread-local state at
  `app/backends/jetson/moss_tts_nano.py:258` through
  `app/backends/jetson/moss_tts_nano.py:261`.
- Worker errors and exits are surfaced at
  `app/backends/jetson/moss_tts_nano.py:262` through
  `app/backends/jetson/moss_tts_nano.py:267`.
- Retry respawns the worker once on process or timeout failures at
  `app/backends/jetson/moss_tts_nano.py:271` through
  `app/backends/jetson/moss_tts_nano.py:277`.
- `synthesize()` collects streaming PCM into WAV at
  `app/backends/jetson/moss_tts_nano.py:281` through
  `app/backends/jetson/moss_tts_nano.py:317`.
- Voice cloning passes base64 reference audio at
  `app/backends/jetson/moss_tts_nano.py:319` through
  `app/backends/jetson/moss_tts_nano.py:340`.
- `_build_request()` includes `id`, `text`, `stream`, transport, format,
  and chunk frames at `app/backends/jetson/moss_tts_nano.py:348` through
  `app/backends/jetson/moss_tts_nano.py:359`.
- `_build_request()` currently does not include a slot id at
  `app/backends/jetson/moss_tts_nano.py:352` through
  `app/backends/jetson/moss_tts_nano.py:367`.
- `_send_request()` serializes JSON and writes to one process stdin
  under `_proc_lock` at `app/backends/jetson/moss_tts_nano.py:369`
  through `app/backends/jetson/moss_tts_nano.py:379`.
- Request queue registration uses `_request_queues` keyed by request id
  at `app/backends/jetson/moss_tts_nano.py:384` through
  `app/backends/jetson/moss_tts_nano.py:391`.
- `_stdout_reader()` reads JSONL events from one process stdout at
  `app/backends/jetson/moss_tts_nano.py:403` through
  `app/backends/jetson/moss_tts_nano.py:428`.
- `_route_stdout_event()` routes by request id at
  `app/backends/jetson/moss_tts_nano.py:443` through
  `app/backends/jetson/moss_tts_nano.py:457`.
- `_publish_worker_exit()` broadcasts worker exit to all live request
  queues at `app/backends/jetson/moss_tts_nano.py:459` through
  `app/backends/jetson/moss_tts_nano.py:467`.
- Process termination is single-process at
  `app/backends/jetson/moss_tts_nano.py:468` through
  `app/backends/jetson/moss_tts_nano.py:493`.

Existing file: `bench/perf/smoke_moss_tts_backend.py`.
- Smoke test purpose is Python backend to C++ worker to WAV at
  `bench/perf/smoke_moss_tts_backend.py:2` through
  `bench/perf/smoke_moss_tts_backend.py:7`.
- Usage is documented at `bench/perf/smoke_moss_tts_backend.py:9`
  through `bench/perf/smoke_moss_tts_backend.py:13`.
- CLI options include worker, engine, tokenizer, and codec paths at
  `bench/perf/smoke_moss_tts_backend.py:26` through
  `bench/perf/smoke_moss_tts_backend.py:38`.
- Environment configuration maps args to MOSS env vars at
  `bench/perf/smoke_moss_tts_backend.py:41` through
  `bench/perf/smoke_moss_tts_backend.py:45`.
- The smoke creates `MossTtsNanoBackend({})` at
  `bench/perf/smoke_moss_tts_backend.py:64` through
  `bench/perf/smoke_moss_tts_backend.py:68`.
- It preloads backend and prints readiness at
  `bench/perf/smoke_moss_tts_backend.py:71` through
  `bench/perf/smoke_moss_tts_backend.py:74`.
- Streaming mode records first chunk TTFA at
  `bench/perf/smoke_moss_tts_backend.py:82` through
  `bench/perf/smoke_moss_tts_backend.py:98`.
- It shuts down backend at `bench/perf/smoke_moss_tts_backend.py:112`
  through `bench/perf/smoke_moss_tts_backend.py:113`.

Existing file: `configs/profiles/jetson-moss-tts-nano-trt.json`.
- Description says C++ worker subprocess handles inference and TTFA is
  about 157 ms on Orin NX at
  `configs/profiles/jetson-moss-tts-nano-trt.json:2` through
  `configs/profiles/jetson-moss-tts-nano-trt.json:3`.
- `tts_backend` is `jetson.moss_tts_nano` at
  `configs/profiles/jetson-moss-tts-nano-trt.json:5`.
- Execution policy is serialized GPU at
  `configs/profiles/jetson-moss-tts-nano-trt.json:7` through
  `configs/profiles/jetson-moss-tts-nano-trt.json:10`.
- C++ worker binary path is configured at
  `configs/profiles/jetson-moss-tts-nano-trt.json:17`.
- Engine, tokenizer, and codec paths are configured at
  `configs/profiles/jetson-moss-tts-nano-trt.json:18` through
  `configs/profiles/jetson-moss-tts-nano-trt.json:20`.
- Current max slots env is 1 at
  `configs/profiles/jetson-moss-tts-nano-trt.json:21`.
- Current `moss_max_slots` is 1 at
  `configs/profiles/jetson-moss-tts-nano-trt.json:28`.

## New Modules (deferred)

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

## Test Plan (deferred POC)

Design-only test plan for future POC:

1. measure current N=1 baseline on Orin Nano 8GB.
2. capture current memory at idle after preload.
3. capture memory during one full synthesis.
4. capture memory after synthesis completes.
5. estimate free headroom under production container config.
6. set experimental profile to `moss_max_slots=2`.
7. run two concurrent short prompts.
8. record TTFA for both clients.
9. record total wall time.
10. record VRAM and RSS peak.
11. run 30 dual-client bursts only if the first five bursts have no
    worker errors.
12. compare pre/post MD5 on fixed prompt.
13. scan logs for CUDA and TensorRT errors.
14. run voice clone smoke if voice clone is in the profile.
15. confirm unload/shutdown returns worker process cleanly.
16. confirm failed request does not poison the remaining slot.
17. confirm timeout on one slot does not orphan queues.
18. confirm worker exit wakes both live request queues.
19. confirm respawn resets slot state.
20. confirm service readiness after respawn.

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

## Acceptance Criteria

Because this deliverable is design-only, acceptance means:
- The user can decide whether to pursue MOSS N=2.
- The tradeoffs are explicit.
- The recommended POC path is clear.
- The memory risk on Orin Nano 8GB is called out.
- TTFA degradation budget is defined.

Suggested future MOSS N=2 performance acceptance:
- TTFA N=2/N=1 <= 1.5x for Orin NX.
- TTFA N=2/N=1 <= 1.75x for Orin Nano 8GB during POC.
- Absolute Orin Nano N=2 TTFA should stay below 300 ms for the first
  audio chunk if baseline is 157 ms.
- 0 CUDA errors over 30 dual-client bursts.
- Pre/post MD5 stable.
- Peak memory must leave at least 1.0 GB practical headroom on Orin Nano
  8GB.
- If headroom is below 1.0 GB, do not ship N=2 on Orin Nano 8GB.

Why 300 ms for MOSS POC:
- Baseline context is 157 ms.
- A 1.5x ratio gives about 236 ms.
- A POC-only 1.75x ceiling gives about 275 ms.
- 300 ms is a pragmatic upper bound for POC noise on Orin Nano.
- Production should tighten back to <= 1.5x if the architecture is
  accepted.

Is N=2 worth doing on Orin Nano 8GB?
- Not by default.
- It is worth exploring only if simultaneous TTS users are a real
  requirement.
- It may be worth doing on Orin NX before Orin Nano.
- Orin Nano 8GB has limited memory headroom for duplicated TRT engines
  and per-slot buffers.
- A 157 ms TTFA baseline already meets most conversational needs.
- The product benefit of N=2 is throughput and overlap, not single-user
  latency.

## Edge Cases

- Worker stdout is a single ordered JSONL stream.
- Concurrent slots can interleave events.
- Every event must include request id.
- The Python wrapper already drops events without request id at
  `app/backends/jetson/moss_tts_nano.py:448` through
  `app/backends/jetson/moss_tts_nano.py:457`.
- If C++ emits slot-only events without id, Python cannot route them
  safely.
- The existing startup event `worker_ready` has no request id and routes
  to the control queue at `app/backends/jetson/moss_tts_nano.py:443`
  through `app/backends/jetson/moss_tts_nano.py:447`.
- That behavior must remain unchanged.
- Timeout on one request currently respawns the worker after retry
  conditions at `app/backends/jetson/moss_tts_nano.py:271` through
  `app/backends/jetson/moss_tts_nano.py:277`.
- With multiple live slots, respawn kills all in-flight slots.
- The POC must accept this as a coarse failure model.
- Python `_proc_lock` serializes stdin writes at
  `app/backends/jetson/moss_tts_nano.py:371` through
  `app/backends/jetson/moss_tts_nano.py:379`.
- That is fine for request submission.
- It does not serialize worker execution once requests are accepted.
- Thread-local metadata is used for the last stream at
  `app/backends/jetson/moss_tts_nano.py:224` and
  `app/backends/jetson/moss_tts_nano.py:259` through
  `app/backends/jetson/moss_tts_nano.py:260`.
- That is safe only if each request consumes metadata in its own thread.
- The POC must not read another request's `last_stream_metadata`.
- Voice clone uses large base64 reference audio at
  `app/backends/jetson/moss_tts_nano.py:330` through
  `app/backends/jetson/moss_tts_nano.py:339`.
- Two concurrent clone requests can increase stdin payload and memory
  pressure.
- MOSS current worker returns raw PCM without side-channel format per
  file risk notes at `app/backends/jetson/moss_tts_nano.py:11` through
  `app/backends/jetson/moss_tts_nano.py:12`.
- Slot demux must not change sample rate or channel assumptions.

## Regression Risks

Approach A: Multi-process.
- Complexity: medium.
- Python changes are conceptually straightforward.
- Supervisor logic must manage two subprocesses.
- Request routing must pick a worker.
- Health must aggregate two workers.
- Crash handling must avoid killing healthy worker unnecessarily.
- Latency impact: likely good under two clients if memory does not
  thrash.
- Memory overhead: high.
- Each process can duplicate TRT engines and worker memory.
- On Orin Nano 8GB, this may consume unacceptable headroom.
- Risk level: medium-high because memory pressure can cause system
  instability.
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
- Per-slot contexts, KV buffers, codec state, and scratch buffers still
  add memory.
- Risk level: high.
- Benefit: clean production architecture.
- Drawback: largest implementation and debug surface.

Approach C: Python-side multi-slot plus worker IPC demux.
- Complexity: medium.
- Python owns slot ids and admission.
- C++ worker accepts slot id and keeps per-slot state.
- Request id remains routing key.
- Latency impact: potentially good, but single worker stdout and
  internal worker scheduling can create head-of-line blocking.
- Memory overhead: medium.
- Model weights can remain shared.
- Per-slot C++ state and buffers are still required.
- Risk level: medium-high.
- Benefit: fastest way to learn whether shared-worker N=2 is viable.
- Drawback: easy to produce a half-measure if C++ internals are not
  actually per-slot safe.

Recommendation rationale:
- Approach A is too memory-expensive as first choice for Orin Nano 8GB.
- Approach B is too large for a Week 3 hardening cycle.
- Approach C is the right POC because it gives the fastest signal with
  less memory duplication.
- Approach C must be gated hard before production.
- If Approach C cannot pass the qwen3-style gate, do not keep iterating
  blindly.

## Decision Trigger

Open this design only when:
- A real customer workload requires two simultaneous TTS streams.
- Or pipeline overlap of generation with playback is needed to hit a
  product latency target that 157 ms TTFA cannot meet alone.

Otherwise: keep MOSS at `moss_max_slots=1` and treat this document as a
parked design.
