# voxedge

> Edge-native, local-first real-time voice conversation library — "Pipecat for the edge".

**Status: Phase 1a — pure-Python foundation (additive scaffolding only).**

This package is the open-core extraction of the seeed-local-voice product stack.
See `docs/specs/edge-voice-library-architecture.md` for the full architecture.

## What's here (Phase 1a)

- `voxedge/backends/base.py` — clean backend ABCs (`ASRBackend`/`ASRStream`,
  `TTSBackend`, `VADBackend`/`VADSession`, `LLMBackend`/`LLMEvent`) with **no
  env / profile coupling** — constructors take explicit params only.
- `voxedge/transport/base.py` — `Transport` ABC + `InProcessTransport`
  (zero-IPC asyncio queues, the default) + `WebSocketTransport` (duck-typed
  ws adapter, no FastAPI dependency).
- `voxedge/engine/conversation.py` — `ConversationEngine` + `Session`, the
  VAD-segmentation / barge-in / multi-turn / sentence-buffer / ASR→(LLM)→TTS
  orchestration loop ported from `app/main.py`.
- `voxedge/backends/mock.py` — Mock backends so the whole engine runs
  end-to-end on a laptop with no CUDA.
- `voxedge/tests/` — proves the architecture runs end-to-end on Mac.

## Design constraints

- **Pure Python.** No CUDA / torch / tensorrt in the core. Heavy adapters live
  behind optional extras (`voxedge[trt]`, `voxedge[rknn]`) — placeholders for now.
- **No env reads in the library.** All config is injected as explicit params.

## Quickstart (Phase 1a, mock backends)

```python
import asyncio
from voxedge.engine import ConversationEngine
from voxedge.transport import InProcessTransport
from voxedge.backends.mock import MockASR, MockTTS, MockVAD

engine = ConversationEngine(
    backends={"asr": MockASR(), "tts": MockTTS(), "vad": MockVAD()},
    multi_utterance=True,
)
transport = InProcessTransport()
asyncio.run(engine.run(transport))
```
