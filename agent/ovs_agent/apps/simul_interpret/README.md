# Simultaneous interpret app (`simul_interpret`)

Simultaneous(-ish) speech interpretation: ASR partials are committed clause by
clause (monotonic — a committed clause is never retracted, because spoken
audio can't be un-said), translated incrementally, and spoken. No LLM.

Source: `app.py` (`SimulInterpretApp` / `App`) · `config.yaml`

## Modes

| `overlap_mode` | Behavior | Hardware needs |
|---|---|---|
| `"off"` (default, clause-lag) | translate each clause incrementally during partials, but speak the whole utterance once on `ASRFinal` — the robust TTS path. The listener is ~one sentence behind. | any |
| `"on"` (experimental, full-duplex) | speak each clause as it commits while the user keeps talking. The EchoFilter drops self-echo partials the hardware AEC leaks. | AEC mic array or headphones |

Barge-in is disabled by design — translation must never interrupt itself.

## Pipeline

```text
mic → SLV /v2v/stream ASR partials
     → SegmentCommitter (monotonic, agreement_n=2)
     → CTranslate2 translator (:9001)
     → SLV TTS   (clause-lag: on ASRFinal · overlap: per committed clause)
```

## Run

```bash
uv run ovs-agent run simul_interpret --config agent/ovs_agent/apps/simul_interpret/config.yaml
```

Key configuration:

| Key | Default | Notes |
|---|---|---|
| `overlap_mode` | `off` | see the table above |
| `committer_agreement_n` | `2` | partials that must agree before a clause commits |
| `slv_config.asr_language` | `auto` | source language detection |

## Dependencies

- SLV speech service (`/v2v/stream`, ASR + TTS)
- CTranslate2-compatible translator service (`:9001` by default)
- For `overlap_mode: "on"`: an AEC-capable mic (e.g. reSpeaker) or headphones

## Deployment matrix

TBD — no SenseCraft solution record yet.

## Measured results

Not measured. Clause-commit latency, translation lag, and un-translated tail
rates are all unrecorded for every target.

## Known limitations

- `overlap_mode: "on"` is experimental and depends on hardware AEC quality;
  without it the interpreter hears its own TTS.
- Monotonic commitment means a mistranslated clause is spoken and final —
  there is no retraction, by design.
