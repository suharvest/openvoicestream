# SLV /v2v/stream WAV-injection harness — #37 parity tool

`v2v_wav_inject.py` is a deterministic, repeatable WAV-injection client for
SLV's `/v2v/stream` WebSocket. It exists to produce **before/after baselines
for #37** (tool-calling engine migration) without a microphone.

## Architecture clarification (READ FIRST)

VoiceArm @ seeed-orin-nx is **3 containers**:

| Container     | Port  | Role                                                      |
|---------------|-------|-----------------------------------------------------------|
| `seeed-voice` (SLV) | 8621 | ASR (`trt_edgellm`) + TTS (`matcha_trt`) **service**. NO LLM, NO tools. Accepts streaming audio in, text in, emits asr_final + TTS audio out. |
| `edge-llm`    | 8000  | Qwen3-4B-AWQ OpenAI-compatible LLM server.                |
| `voice-arm` (agent) | 8765 | The **client** that owns the LLM + tool loop. Connects to SLV `/v2v/stream`, receives `asr_final`, calls edge-llm with its 10 arm tools, streams the reply text back into SLV (`CLIENT_TEXT`), then `tts_flush`. The 10 robot-arm tools are registered HERE, not in SLV. |

So **the LLM + tool loop runs in the agent client, not in SLV.** Evidence
(`agent/openvoicestream_agent/`):
- `slv_client.py:113` — "One persistent WS to /v2v/stream for the entire App lifetime."
- `slv_client.py:400-403` `send_text` (CLIENT_TEXT), `:405-407` `flush_tts` —
  the agent pushes LLM output text into SLV for synthesis.
- `app_base.py:578` `on_user_utterance` → `app_mode` runs the LLM + tool loop;
  edge-llm is reached at `app_base.py:122-129` (`EdgeLLMBackend`, base_url :8000).
- `app/core/v2v.py:34` `CLIENT_TEXT = "text"  # streaming text input for TTS` —
  SLV consumes text, it never produces it.

**Parity implication:** asr_final / tool-preamble / tts_done parity lives in the
**voice-arm agent log**. SLV only owns the ASR text and TTS audio bytes/TTFA.

## What the harness does

Impersonates the agent at the SLV protocol level but injects a **fixed
deterministic reply text** for the TTS leg instead of calling the LLM:

1. open WS → send `config` (multi_utterance, vad)
2. stream WAV PCM (realtime-paced) + trailing silence
3. send `asr_eos` (force finalize, deterministic regardless of VAD)
4. receive `asr_final` → record text + start TTFA clock
5. send `{"type":"text","text": <fixed reply>}` + `{"type":"tts_flush"}`
6. collect `tts_started` / binary PCM / `tts_sentence_done` / `tts_done`
   → first PCM byte = TTFA endpoint

Because it never calls the agent LLM, it **structurally cannot trigger a
robot-arm action**. It also lints `asr_final` + the injected reply against the
actions.yaml trigger phrases and flags any hit loudly (`arm_trigger_hits`).

## Usage

```bash
# From the repo root on the dev machine (Mac), via Tailscale to the device.
# proxy is bypassed in-code (proxy=None) for direct LAN/Tailscale.
uv run python bench/parity/v2v_wav_inject.py \
  --host 100.111.134.124 --port 8621 \
  --wav bench/perf/corpus/short/zh_short_04.wav \
  --reply "今天是星期五，天气晴朗。" \
  --vad none \
  --out /tmp/voicearm-baseline/no-arm/run1.json
```

Output JSON fields: `asr_final`, `asr_language`, `reply_injected`,
`tts_started`, `tts_sentence_done_count`, `tts_done`, `tts_pcm_bytes`,
`tts_sample_rate`, `ttfa_ms` (asr_final → first TTS PCM byte),
`asr_eos_to_final_ms`, `arm_trigger_hits`, `events` (raw timeline with parity
grep keys).

Suggested no-arm WAVs (corpus, all neutral news text, zero trigger words):
`zh_short_04.wav` (周二，他在大阪去世), `zh_short_03.wav`, `zh_short_05.wav`.

## KNOWN BLOCKER (must resolve before live run)

SLV's `session_limiter` is **limit=1** (single ASR worker, no env override).
The voice-arm agent holds the one `/v2v/stream` session slot **continuously**
for its entire lifetime (verified: only 1 `connection open` on SLV in 30 min).
ALL SLV inference endpoints (`/asr`, `/asr/stream`, `/v2v/stream`) share that
one slot. So while the agent runs, this harness is rejected:

```
WS close 4429 {"error":"too_many_sessions","current":1,"limit":1}   (/v2v/stream)
HTTP 429    {"detail":{"error":"too_many_sessions","current":1,"limit":1}}  (/asr)
```

To run the harness for a real TTFA baseline, the agent must not hold the slot:

- **Option A (recommended):** stop `voice-arm` briefly → run harness against the
  freed SLV slot → restart `voice-arm`. Cleanest direct-SLV baseline.
- **Option B:** raise SLV session limit to 2 so harness + agent coexist (note:
  2nd concurrent ASR session contends the single ASR worker).

Both need explicit approval (container stop / config change), which is out of
scope for the read-only capture run.

## #37 parity plan

The parity grep keys are all present in the **voice-arm agent log** and
confirmed live:

| key | log line |
|-----|----------|
| `asr_final` | `slv_client: SLV evt: {'type': 'asr_final', 'text': ...}` |
| `tool preamble ... tool=` | `tools.runner: tool preamble (early): text='好的。' tool=dance` |
| `tts_done` / `tts_sentence_done` | `slv_client: SLV evt: {'type': 'tts_done', ...}` |
| `cache_warmed=... warmup_ms` | `app_base: LLM backend warmup result: {'cache_warmed': True, 'cache_warmup_ms': 50, 'graph_warmup_ms': 391, ...}` |

**Before/after diff procedure for #37:**
1. (no-arm chat) run this harness with a fixed WAV + fixed reply → record
   `asr_final`, `ttfa_ms`, `tts_pcm_bytes`. Same inputs ⇒ same outputs ⇒
   regression = any drift in asr_final text or TTFA.
2. (LLM/cache) capture the agent startup `LLM backend warmup result` line →
   compare `cache_warmed`, `cache_warmup_ms`, `graph_warmup_ms`, `prompt_chars`
   before vs after the migration (prompt_chars changes if tool schema changes).
3. (arm turns — separate run, after on-site safety confirmation) drive a real
   trigger phrase through the agent and grep `tool preamble ... tool=<name>` +
   `tts_done` to confirm the tool loop parity.
