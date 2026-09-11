# Voice arm app (`voice_arm`)

Voice-controlled SO-ARM100 actuator: wake-word → streaming ASR → LLM
tool-calling → arm motion, with spoken acknowledgements while the arm moves.

Source: `app.py` (`VoiceArmApp` / `App`) · `config.yaml` (+ `actions.yaml`
recording workflow)

Built on `MultiModeApp`; the actuator is pluggable via
`metadata.actuator.backend` (currently `so_arm`; `rebot_arm` lives in
`voice_rebot_arm`).

## Pipeline

```text
reSpeaker mic ──► TappedAudioIO (capture taps feed the wake-word detector)
   │
   ├── OpenWakeWord ("hey jarvis") ──► app wakes
   └── SLV /v2v/stream ──► streaming ASR
                              │
                              ▼
                 LLM with actuator tool-calling
                 (ack "好的" → tool call → confirm "已挥手")
                              │
                              ▼
                 SLV streaming TTS + serial → SO-ARM
```

The system prompt enforces literal trigger matching: the LLM may only call a
tool when the user's transcription contains the exact trigger phrase listed
in that tool's description — ASR-garbled phrases must be refused, not guessed.

## Run

```bash
uv run ovs-agent run voice_arm --config agent/ovs_agent/apps/voice_arm/config.yaml
```

Key configuration (see `config.yaml` for the full annotated set):

| Key | Default | Notes |
|---|---|---|
| `metadata.actuator.backend` | `so_arm` | actuator driver |
| `metadata.wakeword.model` | `hey jarvis` | OpenWakeWord model |
| `reconnect_on_wake` | `true` | fresh ASR worker per wake (Qwen3-ASR degrades on a long-lived session) |
| energy gate / mic gain | tuned | calibrated for reSpeaker Flex XVF3800 ch1 on seeed-orin-nx |

## Dependencies

- SLV speech service (`/v2v/stream`)
- OpenAI-compatible LLM with tool-calling (edge-llm or external)
- SO-ARM on a serial port (`ARM_PORT`, `auto` resolves)
- reSpeaker-class mic (channel layout auto-detects the firmware profile)

## Deployment matrix

TBD — the mic-pump calibration notes reference seeed-orin-nx hardware; no
solution compose record exists yet. The rebot variant's matrix is in
`voice_rebot_arm/README.md`.

## Measured results

Not measured at app level (wake→motion latency, command accuracy under ASR
garbling are unrecorded). The 2026-09 rebot-arm robustness fixes
(late attach, disconnect generation, dead-gripper surfacing) landed with
agent-level tests in `agent/tests/`.

## Known limitations

- Wake-word + VAD tuning is hardware-specific; the shipped values assume the
  reSpeaker Flex on Orin. See the calibration comments in `config.yaml`
  before porting to another mic.
- Single-turn by default (`clear_history_on_turn_end: true`) — it is a command
  interface, not a chatbot.
