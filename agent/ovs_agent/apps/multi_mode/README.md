# Multi-mode app (`multi_mode`)

`MultiModeApp` is the standard voice app: one persistent SLV session whose
per-turn behavior is delegated to a runtime-switchable `AppMode`. It replaces
the old `DialogueApp` and is the base class the arm apps build on.

Source: `app.py` (`MultiModeApp` / `App`) · `config.yaml`

## Modes

| Mode | Behavior | LLM |
|---|---|---|
| `chat` | voice dialogue — ASR partials stream in, LLM tokens stream straight to TTS | yes |
| `interpreter` | translate each finalized utterance (sentence-level) | no |
| `monologue` | long-form speech → text capture | no |
| `transcribe` | pure transcription of the mic stream | no |

INVARIANT: LLM tokens stream directly to SLV — there is no client-side
sentence batching anywhere in the loop.

Modes switch at runtime (wake source / HTTP / MQTT / serial keyword), so one
deployment can be a chatbot by day and a transcriber for meetings without a
restart.

## Runtime dependencies

```text
multi_mode app
  ├── ovs-agent shared runtime
  ├── SLV speech service :8621 (/v2v/stream: ASR + VAD + TTS)
  ├── OpenAI-compatible LLM (chat mode only; others run LLM-free)
  └── optional: translator service :9001 (interpreter mode)
```

## Run

```bash
uv run ovs-agent run multi_mode --config agent/ovs_agent/apps/multi_mode/config.yaml
```

Key configuration (see `config.yaml` for the full list):

| Key | Default | Notes |
|---|---|---|
| `default_mode` | `chat` | initial AppMode |
| `client_vad_backend` | `auto` | silero preferred, energy fallback; drives `asr_eos` |
| `llm_backend` | `edge_llm` | swap to any OpenAI-compatible endpoint |

## Deployment matrix

TBD — no SenseCraft solution ships this app directly today; solutions pick a
specialized app (`conversation`, `voice_arm`, …) built on it.

## Measured results

Not measured at app level. Backend-level numbers (ASR CER, V2V latency per
profile) are in the top-level [Performance](../../../../README.md#performance)
section and must not be quoted as app-level measurements.

## Known limitations

- Client-side VAD thresholds in the default config are tuned for a quiet
  desk mic; reSpeaker arrays should follow the mic-pump calibration notes in
  `voice_arm/config.yaml`.
