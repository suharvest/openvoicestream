# Translator app (`translator`)

Sentence-level voice translation with no LLM in the loop: you speak one
language, the device speaks back the translation.

Source: `app.py` (`TranslatorApp` / `App`) · `config.yaml`

## Pipeline

```text
mic → SLV /v2v/stream (ASR + silero VAD)
     → ASRFinal (utterance finalized by VAD silence)
     → CTranslate2 translator (:9001)
     → SLV TTS (translated text spoken aloud)
```

No LLM backend (`llm_backend: "noop"`) — translation quality and latency are
the MT service's, and the voice path stays fully local.

## Run

```bash
uv run ovs-agent run translator --config agent/ovs_agent/apps/translator/config.yaml
```

Key configuration:

| Key | Default | Notes |
|---|---|---|
| `translator_src_lang` / `tgt_lang` | `zho_Hans` → `eng_Latn` | FLORES-200 codes |
| `translator_url` | `http://localhost:9001` | CTranslate2-compatible MT service |
| `slv_config.vad_silence_ms` | `400` | utterance boundary; lower = snappier, higher = safer for slow speakers |

## Dependencies

- SLV speech service (`/v2v/stream`, ASR + TTS)
- A translator service speaking the CTranslate2 HTTP contract (default port
  9001). Not shipped in this repo's compose set yet.

## Deployment matrix

TBD — no SenseCraft solution record yet.

## Measured results

Not measured. In particular there is no end-to-end
speech→translation→speech latency figure for any target; do not quote ASR/TTS
benchmarks as one.

## Known limitations

- Sentence-level only: the translation starts after the utterance finalizes
  (VAD silence). For clause-lag and overlap styles see `simul_interpret`.
- One direction per process (`src`/`tgt` are fixed at config load).
