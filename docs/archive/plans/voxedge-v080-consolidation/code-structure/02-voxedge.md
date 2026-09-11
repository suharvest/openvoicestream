# voxedge — MIDDLE layer (engine + backend abstraction library)

**Analyzed ref:** `main` @ **b783037** ("feat(trt-edge-llm-tts): fixed base-model speaker-embedding injection").
**AST tool:** Python `ast` module (custom extractor `/tmp/ast_extract.py`) over the 94 real `.py` files.

voxedge is the **provider-agnostic engine + backend implementations** that seeed-local-voice consumes.
It contains zero device-specific server glue; it exposes backend classes + a turn driver + transports.

---

## 1. Package tree

```
voxedge/
  __init__.py            (only exports __version__ — public surface is the submodules below)
  backends/
    base.py              ABCs: ASRBackend, ASRStream, TTSBackend, + capability enums + ConcurrencyCapability
    jetson/              the TRT/Jetson backends (see §3)
    rk/                  Rockchip backends (asr.py, tts.py, runtime.py, artifacts.py)
    sherpa/              CPU sherpa-onnx backends (asr.py, tts.py)
  engine/                the turn driver + conversation/session machinery (see §4)
  capabilities/          punctuation.py, speaker_embedding.py  (opt-in stateless add-ons)
  transport/             base.py  (Transport ABC + InProcessTransport)
  artifacts/  audio/  tests/
```

## 2. Backend abstraction — `voxedge/backends/base.py`

The contract every backend implements (this is what seeed's `_ASR_REGISTRY`/`_TTS_REGISTRY` resolve to):

- `class ASRStream(ABC)`: `accept_waveform(sr, samples)`, `finalize()->(text,lang)`, `get_partial()->(text,final)`,
  `prepare_finalize()`, `cancel_and_finalize()`, `cancel()`, `close()`.
- `class OfflineAccumulateStream(ASRStream)`: buffers audio, finalize calls backend.transcribe_array (sherpa/offline path).
- `class ASRBackend(ABC)`: `name`, `capabilities()->set[ASRCapability]`, `sample_rate`, `is_ready`, `preload`,
  `transcribe(...)`, `transcribe_array(...)`, `create_stream(language)->ASRStream`, `has_capability`, `unload`,
  `concurrency_capability()->ConcurrencyCapability`.
- `class TTSBackend(ABC)`: parallel surface (name/model_id/capabilities/sample_rate/is_ready/preload/unload,
  `_synthesize_impl`, `_generate_streaming_impl`, `rate_pitch_caps`, optional `clone_voice`/`extract_speaker_embedding`).
- Enums: `ASRCapability`, `TTSCapability`; `TranscriptionResult`.

> **Note:** seeed-local-voice ALSO defines its own `ASRBackend`/`TTSBackend` ABCs in `server/core/{asr,tts}_backend.py`.
> The voxedge ones are the implementation base; the seeed ones are the server-facing facade that the registry
> resolves into. This is a deliberate two-layer ABC (facade in seeed, impl base in voxedge) — see consolidation note.

## 3. `backends/jetson/` — the TRT backend family

```
trt_edge_llm_tts.py   TRTEdgeLLMTTSConfig + TRTEdgeLLMTTSBackend  (Qwen3-TTS via C++ worker)
trt_edge_llm_asr.py   TRTEdgeLLMASRBackend                        (Qwen3-ASR via C++ worker)
trt_edge_llm_ipc.py   IPC/worker-spawn plumbing shared by the two above
_trt_edge_llm_util.py shared helpers
worker_io.py          Python side of the JSON-line worker protocol (WorkerIO)
matcha_trt.py         MatchaTRTBackend  (+ CudaMemoryPool)
kokoro_trt.py         KokoroTRTBackend  (+ _KokoroCtxSlot, _OrtIoNames, _run_cpu_onnx)
qwen3_trt.py          Qwen3TRTBackend   (legacy Qwen3-TTS path)
moss_tts_nano.py      MossTtsNanoBackend
paraformer_trt.py     ParaformerTRTBackend (+ decode_ids, _ParaformerCtxBundle)
sensevoice_trt.py     SenseVoiceTRTBackend
_util.py
```

### `TRTEdgeLLMTTSBackend` (trt_edge_llm_tts.py) — load-bearing methods
- `class TRTEdgeLLMTTSConfig` (frozen-ish dataclass): `__post_init__`, `highperf_enabled`, `stateful_code2wav_enabled`, `fast_perf_profile`.
- `PoolSaturatedError(RuntimeError)` (N>1 backpressure → maps to seeed 4429).
- Backend: `concurrency_capability`, `supports_hot_reload`, `preload`/`unload`,
  `_use_worker`, `_worker_env`, `_ensure_worker`, `_restart_worker_locked`,
  `_synthesize_worker`, `_synthesize_worker_via_stream`, `_generate_streaming_{impl,single}`,
  `_synthesize_impl`, `rate_pitch_caps`, `clone_voice`, `extract_speaker_embedding`,
  `_load_product_explicit_kv_backend`, `_explicit_kv_flags`.
- **It spawns the C++ `qwen3_tts_streaming_worker` binary** (from the TRT fork examples/omni) and speaks the
  JSON-line protocol via `worker_io.py`. This is the seam where the Python (voxedge) layer meets the C++
  (TRT fork) layer. `_worker_env`/`_explicit_kv_flags` build the worker argv/env.

## 4. `engine/` — turn driver + conversation machinery (the unification target)

```
turn_driver.py        run_turn(...) — provider-agnostic, NO I/O. The unified server/client pump.
conversation.py       higher-level conversation orchestration
coordinator.py        coordination glue
asr_loop.py  asr_session_manager.py  audio_dispatcher.py   ASR side
tts_buffer.py  tts_sequencer.py                            TTS side
llm_turn.py  builtin_tools.py  tool_registry.py            LLM/tool-calling
capability_resolver.py  concurrency_capability.py          capability negotiation
client_events.py  protocol.py  session_state.py
```

### `turn_driver.run_turn` (turn_driver.py) — the unified pump
- Protocols: `TextSink` (`text`/`preamble`/`flush`), `MessageSink`
  (`add_assistant_tool_calls`/`add_assistant_text`/`add_tool_result`/`working_messages`).
- `async def run_turn(...)` — the single shared turn loop. `_ToolCallAcc`, `_template_fires`, `_template_completion`.
- **Consumed directly by BOTH:** seeed's server (`server/main.py` / `server/core/coordinator`) AND the agent
  (`agent/ovs_agent/tools/runner.py` imports `from voxedge.engine.turn_driver import run_turn`). This is the
  server-loop/client-loop unification recorded in memory (turn_driver_unification).

## 5. `transport/base.py`
- `class Transport(ABC)`: `recv_audio`/`send_audio`/`recv_event`/`send_event`/`close`.
- `class InProcessTransport(Transport)`: feed_audio/feed_event/end_input + audio_out/events_out +
  drain_*_nowait. Used by the in-process e2e harness and the agent's local pump.
- `transport/base.py::_pump` is load-bearing for the production `OVS_V2V_ENGINE=voxedge` path (per memory).

## 6. `capabilities/`
- `punctuation.py` — CT-Transformer punctuation (opt-in, default-off, stateless; byte-level no-op when off).
- `speaker_embedding.py` — CAM++ speaker embedding (opt-in). Re-exported by seeed `server/core/{punctuation,speaker_embedding}.py`.

## 7. Public API surface actually imported by seeed-local-voice
(see also 01-seeed-local-voice.md §dependency edges)
- `voxedge.backends.base.{LLMBackend, LLMEvent}` ← `server/core/edge_llm_backend.py`
- `voxedge.backends.jetson.{trt_edge_llm_tts, trt_edge_llm_asr, matcha_trt, kokoro_trt, qwen3_trt, moss_tts_nano, paraformer_trt}`
  ← resolved lazily via seeed's registry + `voxedge_backend_config.py`
- `voxedge.backends.sherpa.{asr,tts}` ← sherpa registry entries
- `voxedge.engine.turn_driver.run_turn` ← agent runner + server coordinator
- `voxedge.engine.asr_session_manager` ← `server/core/asr_session_manager.py` (re-export)
- `voxedge.transport.base.WebSocketTransport` ← server
- `voxedge.capabilities.{punctuation,speaker_embedding}` ← server capability shims
