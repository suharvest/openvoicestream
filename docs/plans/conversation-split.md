# Plan: split `voxedge/engine/conversation.py`

**Status: substantially LANDED (2026-06).** The staged split has shipped:
`session_state.py`, `asr_loop.py`, `audio_dispatcher.py`, `client_events.py`,
`tts_sequencer.py`, and `llm_turn.py` are extracted, and the LLM↔tool pump that
lived in `_llm_turn_with_tools` is now the provider-agnostic
`turn_driver.run_turn` shared by both the server-loop and client-loop
(see [turn-driver-unification.md](turn-driver-unification.md)). `conversation.py`
is now an ~810-LOC `Session` coordinator (down from ~1600). Each step shipped
byte-equivalent through the perf/behavioral bench gate. The original design below
is retained for reference / provenance.

## Problem

`voxedge/engine/conversation.py` is ~1,600 LOC / 81 KB. It is not a god-object
(responsibilities are orthogonal and well-commented) but it is the single
highest-churn, highest-risk file in the stack: barge-in, ASR stale-final
suppression, TTS sequencing, server-loop tool dispatch, and slot-leak fixes all
live here and interlock through one shared `Session` state dict. New developers
have to read the whole file to safely touch any part of it.

## Current structure (what's inside)

```
_ToolCallAcc          tool-delta accumulator        (~small)
_SentenceBuffer       CJK+EN sentence splitter       (~small)
ConversationEngine    backend holder / factory       (~110 LOC)
Session               the orchestration loop         (~1,300 LOC)
  __init__            per-connection state dict
  _audio_loop         VAD → speech_start/end → barge-in
  _event_loop         CLIENT_* frame multiplexing
  _asr_out_task       partial polling + M1 deadline + finalize
  _on_asr_final       route to LLM or TTS
  _llm_turn_with_tools  multi-turn LLM↔tool pump (server-loop)
  _tts_out_task       sentence dequeue + generate_streaming + M3 deadline
  helpers             _bargein_tts, _open_asr_turn, _enqueue_tts_text, …
```

## Target structure

Keep `Session` as a thin coordinator that owns the state dict and spawns tasks;
extract the four loops into focused collaborators in `voxedge/engine/`:

| New module | Moves out | Surface |
|---|---|---|
| `audio_dispatcher.py` | `_audio_loop` + VAD/barge-in trigger | `run(session_state, transport, on_speech_start, on_speech_end)` |
| `client_events.py` | `_event_loop` (CLIENT_TEXT/ASR_EOS/ABORT/TOOL_*) | `run(session_state, transport, handlers)` |
| `asr_loop.py` | `_asr_out_task` + `_on_asr_final` + M1 watchdog | wraps `ASRSessionManager` |
| `tts_sequencer.py` | `_tts_out_task` + sentence buffer + M3 watchdog | `enqueue(text)`, `run()` |
| `llm_turn.py` | `_llm_turn_with_tools` + tool dispatch | `run(messages, tool_registry, ctx)` |
| `conversation.py` | `Session` (state + task spawn), `ConversationEngine` | unchanged public API |

The shared `state` dict is the coupling to break carefully: replace ad-hoc dict
keys with a typed `SessionState` dataclass so each collaborator declares exactly
which fields it reads/writes (this is where most latent bugs hide).

> **Codex review (2026-06-13) — incorporated below.** The decomposition is
> directionally sound, but moving loops into files is **not** the first move:
> several pieces of state are written from multiple loops, so the real first
> phase is a **state/transition + facade API**, not file extraction. The
> "each step independently shippable" framing in the original draft was too
> optimistic and is corrected in *Sequencing*. Key evidence below cites
> `conversation.py` line numbers.

### Prerequisite phase — encapsulate state BEFORE moving any file

These must exist before any loop is extracted, or the result is just "typed
shared globals" still mutated from four directions:

1. **`SessionState` with grouped transition methods, not a plain dataclass.**
   Several updates are atomic protocol operations spanning several fields:
   - open-ASR-turn clears endpoint, sets active+generation, anchors timeout
     together (line ~691);
   - VAD endpoint stamps `endpoint_pending` with the current ASR generation
     (line ~410);
   - ASR output drops stale endpoint markers by comparing
     `endpoint_pending_gen` vs `asr_active_gen` (line ~792);
   - finalize only clears `asr_active` if the finalized generation is still
     current (line ~830).
   Expose these as methods (`open_asr_generation`, `stamp_endpoint`,
   `clear_stale_endpoint`, `close_input`, `begin/end_llm_turn`,
   `begin/end_tts_sentence`).

2. **A TTS facade owning the TTS queue/buffer/flush — created in-place first.**
   TTS state is written from **four** places, not just `_tts_out_task`: client
   text/flush in `_event_loop` (line ~432), LLM plain path (line ~931), tool-loop
   path (line ~1028), barge-in cleanup (line ~605). The facade owns `_tts_q`,
   `_tts_buffer`, `tts_flush`, `current_tts_task`, `current_tts_stop` with
   explicit `enqueue_text` / `flush_text` / `mark_flush` / `interrupt_and_clear`
   / `run`. Route `_event_loop`, `_on_asr_final`, `_llm_turn_with_tools`,
   `_bargein_tts` through it **before** the file is split out.

3. **An `InterruptionController` (or barge-in state methods) around
   `llm_barged`.** Barge-in sets the flag, cancels pending remote tool futures,
   waits boundedly for `current_llm_task`, then drains TTS + resets buffer
   (line ~617). The tool loop polls the flag at lines ~1014, ~1022, ~1064,
   ~1142, ~1172. This cross-cut must be a named abstraction, not a raw flag
   threaded through every extracted module.

## Constraints that make this risky

- **The public API must not change.** `ConversationEngine(backends=…)` and the
  `Transport` contract are consumed by `server/main.py` and the tests; the split
  is pure-internal refactor.
- **`llm_barged` cooperative-cancel semantics must be preserved exactly.**
  `task.cancel()` alone is unsafe here (a cancel can be swallowed mid-dispatch);
  the flag-based barge-in is load-bearing. Any extraction must keep the same
  checkpoints.
- **Generation-ID guards** (stale-final suppression) cross the audio/asr
  boundary — splitting audio and asr loops must not reorder those checks.
- **Close-input is a hidden cross-loop transition — must-test.** `_watch_input_end`
  sets **both** `asr_session_closed` and `tts_flush` (line ~1414); that exact
  pair drives ASR terminal-final behaviour (line ~838) and TTS final `tts_done`
  emission (line ~1206). Any split must preserve this as one named
  `close_input` transition, not two independent field writes.
- **`_on_asr_final` is a bridge, not ASR logic — keep it in `Session`.** It must
  not move wholesale into `asr_loop.py` or that module inherits LLM/TTS
  dependencies. The ASR loop emits accepted finals through a callback; the
  close-out duplicate-trigger suppression lives at line ~837.
- **`_event_loop` spans three unrelated domains.** Text-to-TTS frames, ASR
  EOS/abort, and remote tool advertise/result routing. Tool-advertise mutates
  engine-level LLM state and warms the LLM prefix (lines ~529, ~546); remote
  tool-results resolve pending futures (line ~456). Those belong with
  `llm_turn`/tool-session support, **not** a generic `client_events` module —
  so `client_events.py` as originally scoped is too broad.

## Why not now (the deferral is the decision)

This file is **shared with the production robot-arm stack** (`seeed-orin-nx`,
the one device we must not destabilise). A behavioural regression in barge-in or
slot cleanup would pass a two-device ASR/TTS smoke test and only surface as a
field hang. This refactor therefore needs its **own** validation cycle:

1. Full `voxedge/tests/` green (engine, watchdogs, server-loop, slot-leak).
2. `bench/` server-loop timing + barge-in stress, before/after parity.
3. A dedicated real-machine soak on a **non-production** device with repeated
   barge-in / multi-turn / tool-call rounds — not just "ASR works, TTS works".

Bundling it into the architecture-cleanup pass (docs + packaging) would couple a
safe change to a risky one under insufficient validation. Do it as a standalone,
bench-gated change.

## Sequencing when picked up (revised per Codex review)

Note the first **two** steps move **no files** — they build the state/transition
API that makes later extraction safe. "TTS first" is only safe *after* the
facade step; done before it, `tts_sequencer` is still mutated from the LLM and
event paths.

1. **[DONE — voxedge `3368c20`]** `SessionState` + grouped transition methods
   (open-ASR-generation, stamp endpoint, clear-stale-endpoint, close-input,
   deactivate-ASR). Migrated the dict in place, no behaviour change. Mac 221
   passed; orin-nano orchestration subset 81/0.
2. **[DONE — voxedge `b3feeba`]** In-place TTS facade `_TTSChannel` (owns
   queue+buffer; `enqueue_text`/`flush_and_signal`/`interrupt_synth`/
   `drain_and_reset`). Routed `_event_loop`, `_on_asr_final`,
   `_llm_turn_with_tools`, `_bargein_tts` through it; no file move. The consumer
   loop stays `Session._tts_out_task` (reads `self._tts.q`) and relocates with
   the channel in step 3. Mac 221 passed; orin-nano 81/0.
3. **[DONE — voxedge `23384ff`]** Extracted `tts_sequencer.py` (`_TTSChannel` +
   consumer loop `_tts_out_task` → `run()`) and `protocol.py` (shared wire
   constants + `_is_pool_saturated`, cycle-free). Mac 221; orin-nano 83/0.
4. **[DONE — voxedge `b982752`]** Extracted `asr_loop.py` (`_open_asr_turn` →
   `open_turn()`, `_asr_out_task` → `run()`); `_on_asr_final` /
   `_emit_pool_saturated` / `ASRSessionManager` stay on Session. Mac 221.
5. **[DONE — voxedge `f4b0e98`]** Extracted `audio_dispatcher.py` (`_audio_loop`
   → `run()`). Mac 221.
6. **[DONE — voxedge `3f58cb3`]** Extracted `client_events.py` (`_event_loop` →
   `run()` demux); handlers stay on Session. Mac 221.
7. **[DONE — voxedge `307191c`]** Extracted `llm_turn.py` (`_llm_turn_with_tools`
   → `_LLMTurn.run()` + `_emit_preamble` + `_ToolCallAcc`); `_on_asr_final`
   stays on Session and drives it. Mac 221; orin-nano final soak 105/0.

**SPLIT COMPLETE.** conversation.py 1610 → 801 lines; split into session_state
(102) + protocol (60) + tts_sequencer (252) + asr_loop (251) + audio_dispatcher
(89) + client_events (90) + llm_turn (280). Steps 3–7 are each independently
revertible; 1–2 were prerequisites. Each step gated on Mac suite green
(221 passed, only 2 pre-existing Jetson `trt_edge_llm_tts` failures) + on-device
orchestration bench on orin-nano.

> Remaining (out of this task's scope): the modules still hold a Session
> back-ref (`self._sess`); fully severing into explicit constructor deps is a
> later cleanup. And before this ships to the **production arm** it still wants
> the dedicated barge-in / multi-turn / tool-round real-machine soak (the unit +
> orchestration-bench gate used here is strong but not that full soak).
