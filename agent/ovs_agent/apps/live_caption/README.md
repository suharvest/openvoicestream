# Live caption app (`live_caption`)

Real-time bilingual captions from streaming ASR, rendered side-by-side
(original + translation) on the dashboard. No TTS, no LLM.

Source: `app.py` (`LiveCaptionApp` / `App`) · `config.yaml`

With `translator_backend: "noop"` the "translation" is a pass-through, so the
same app doubles as a **pure transcription** view (original == translated).

## Pipeline

```text
mic → SLV /v2v/stream ASR partials
     → SegmentCommitter (retranslation strategy)
     → on_translation broadcast → dashboard WebSocket (/caption page)
```

The committed prefix is locked; the volatile tail is re-translated (debounced,
default 250 ms) and re-emitted, so the caption refreshes while the user keeps
speaking. On `ASRFinal` the remaining tail is force-committed with
full-utterance context.

## Run

```bash
uv run ovs-agent run live_caption --config agent/ovs_agent/apps/live_caption/config.yaml
# open the dashboard /caption page (DebugDashboardPlugin)
```

Key configuration:

| Key | Default | Notes |
|---|---|---|
| `translator_backend` | `ctranslate2` | set `noop` for pure transcription |
| `translate_debounce_ms` | `250` | tail re-translation cadence |
| `barge_in_enabled` | `false` | no assistant audio exists to barge into |

## Dependencies

- SLV speech service (`/v2v/stream`, ASR only in practice)
- CTranslate2-compatible translator service when bilingual (`:9001`)
- Mic in only — no speaker output path

## Deployment matrix

TBD — no SenseCraft solution record yet.

## Measured results

Not measured. Caption latency (partial→render) is unrecorded for every target.

## Known limitations

- Re-translation strategy means the tail flickers by design until it commits;
  the committed prefix never changes.
