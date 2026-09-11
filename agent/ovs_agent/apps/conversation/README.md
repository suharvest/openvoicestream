# Conversation app

`ConversationApp` is the minimal full-duplex conversational voice app. It
provides one `ChatMode` on top of the shared `ovs_agent` runtime.

Source:

- `app.py` — `ConversationApp` / `App`
- `config.yaml` — local/default configuration

The app owns conversation behavior and optional runtime wake-word
registration. The shared runtime owns audio I/O, the persistent SLV WebSocket,
VAD, barge-in, playback draining, LLM streaming, and session lifecycle.

## Runtime dependencies

```text
conversation app
  ├── ovs-agent shared runtime
  ├── SLV speech service :8621
  │     └── /v2v/stream: ASR + VAD + TTS
  └── OpenAI-compatible LLM
```

The app defaults to `openai_compat` and `qwen3.5-flash` in its standalone
configuration. A SenseCraft deployment may override these values through the
mounted config and environment.

## SenseCraft deployment matrix

These rows describe the current files under
`sensecraft-solutions/solutions/conversational_voice_ai/assets/docker`.
They are configured deployment paths, not claims that every row has completed
an end-to-end performance run.

| Target | Compose file | Speech service | Agent service | LLM service | Agent command |
|---|---|---|---|---|---|
| Jetson general | `docker-compose.jetson.yml` | `seeed-local-voice:v0.9.0-ondemand-20260815-7330af9` | `ovs-agent:voiceagent-20260819-4480743-runtimekws` | External OpenAI-compatible endpoint by default | `ovs-agent run conversation` |
| Jetson Orin NX local LLM | `docker-compose.orin-nx-local.yml` | `seeed-local-voice:jetson-jp62-trt103-edgellm-v091-vox080a0-7330af9` | `ovs-agent:voiceagent-20260819-4480743-runtimekws` | `edge-llm-chat-service:v0.9.1-gdn-mtp-runtime-20260804-v13` | `ovs-agent run conversation` |
| RK3576 | `docker-compose.rk3576.yml` | `openvoicestream:rk-qwen3asr-20260816-ea4e258-coldstart` | `ovs-agent:voiceagent-20260819-4480743-runtimekws` | External OpenAI-compatible endpoint by default | `ovs-agent run conversation` |
| RK3588 | `docker-compose.rk3588.yml` | `openvoicestream:rk-qwen3asr-20260816-ea4e258-coldstart` | `ovs-agent:voiceagent-20260819-4480743-runtimekws` | External OpenAI-compatible endpoint by default | `ovs-agent run conversation` |
| RK3588 + RK1828 | `docker-compose.rk3588-rk1828.yml` | extends the RK3588 speech service | extends the RK3588 agent service | `edge-llm-rk1828:20260731-kvreuse` | `ovs-agent run conversation` |

All listed compose files use host networking. The Agent connects to
`ws://127.0.0.1:8621/v2v/stream`, mounts `/dev/snd`, the deployment config,
audio profiles, and persistent Agent state.

## Recommended model/profile matrix

| Target | Recommended speech path | Recommended LLM path | Reason | App-level measurement |
|---|---|---|---|---|
| Jetson Orin NX local | `jetson-edgellm-v091-matcha` — Qwen3 ASR + Matcha TTS | Qwen3.5-4B GDN/MTP, 8K | Local, multilingual, compose-defined path | Not measured here |
| Jetson general | `jetson-qwen3asr-matcha` | `qwen3.5-flash` via the configured OpenAI-compatible endpoint | Default solution configuration; external LLM is replaceable | Not measured here |
| RK3576 | `rk3576-default` — Qwen3 ASR RKNN W8A8 + Matcha RKNN | Configured OpenAI-compatible endpoint | Default solution profile for the RK3576 NPU | Not measured here |
| RK3588 | `rk3588-default` — Qwen3 ASR RKNN W8A8 + Matcha RKNN | Configured OpenAI-compatible endpoint | Default solution profile for the RK3588 NPU | Not measured here |
| RK3588 + RK1828 | RK3588 speech path | Local Qwen3-4B on RK1828 | Keeps the LLM on the device | Not measured here |

The speech-server repository contains backend benchmarks, but those values are
not automatically end-to-end `ConversationApp` results. Use them only when
the app configuration, compose image, model revisions, audio device, endpoint
definition, and test method match.

## Functional acceptance

```bash
COMPOSE=docker-compose.orin-nx-local.yml
docker compose -f "$COMPOSE" config
docker compose -f "$COMPOSE" up -d
docker compose -f "$COMPOSE" ps
curl -fsS http://127.0.0.1:8621/health
curl -fsS http://127.0.0.1:8000/health
```

For external-LLM variants, replace the second health check with a supported
check for the configured LLM endpoint.

Then verify:

1. Speak one Chinese or English utterance into the configured microphone.
2. Confirm one assistant response is produced.
3. Interrupt playback with a second utterance and confirm barge-in drains or
   aborts the prior response.
4. Confirm the configured LLM model receives the request.
5. If `PIPELINE_MODE=wake_word`, verify the configured wake phrase and tone.
6. Record `docker compose ps`, health responses, relevant logs, and the exact
   configuration used.

## Performance status

No app-specific end-to-end performance numbers are claimed here yet. Measure
and record these per target in a dated deployment record:

- startup to ready;
- EOS-to-first-audio;
- ASR finalization latency;
- TTS TTFA and RTF;
- end-to-end LLM turn latency;
- CER/WER on a named corpus;
- resident memory and VRAM;
- wake-word false accept/false reject rate, if enabled;
- barge-in success rate;
- repeated-turn and concurrent-session stability.

Use the deployment record template in
[`../README.md`](../README.md#performance-record-template). Keep raw JSON and
logs outside this README and link them from the record.

## Known limits

- External-LLM variants are not fully local unless the configured endpoint is
  local.
- The app's default `config.yaml` is not the authoritative SenseCraft config;
  the mounted solution config is authoritative.
- The image tags above identify configured artifacts. A release record should
  additionally capture immutable image digests.
- Existing SLV backend benchmarks are not `ConversationApp` measurements
  without an equivalent end-to-end test.
