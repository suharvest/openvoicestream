# Agent applications

Each directory under this directory is a product/application layer built on
the shared `ovs_agent` runtime.

## Catalog

| App | One-liner | Pipeline | README |
|---|---|---|---|
| `conversation` | Minimal full-duplex voice dialogue (barge-in supported) | ASR → LLM → TTS | [yes](conversation/README.md) |
| `home_assistant` | Voice-control an existing Home Assistant | ASR → HA intents | [yes](home_assistant/README.md) |
| `companion_robot` | Voice entry point for embodied robots (Reachy Mini, etc.) | ASR → LLM + robot tools → TTS | [yes](companion_robot/README.md) |
| `voice_rebot_arm` | Voice-controlled robot arm: force-control gripper + IK, wake-word, camera-guided grasp | wake-word → ASR → LLM tool-calls → arm | [yes](voice_rebot_arm/README.md) |
| `voice_arm` | Voice-controlled SO-ARM100 actuator, wake-word | wake-word → ASR → LLM tools → TTS | [yes](voice_arm/README.md) |
| `multi_mode` | Standard voice app with runtime-switchable modes | ASR → LLM → TTS | [yes](multi_mode/README.md) |
| `translator` | Sentence-level voice translation, no LLM | ASR → MT → TTS | [yes](translator/README.md) |
| `simul_interpret` | Simultaneous interpretation with monotonic commitment | ASR → MT → TTS | [yes](simul_interpret/README.md) |
| `live_caption` | Real-time bilingual live captions | ASR → MT → broadcast | [yes](live_caption/README.md) |

Every app now has a README; deployment matrices and measured results are
filled in only where a record exists — rows without one say `TBD` /
`not measured`, and that must stay honest.


An app is not a standalone speech engine. The normal dependency chain is:

```text
SenseCraft solution
  └── ovs-agent run <app>
        ├── shared ovs_agent runtime
        ├── SLV /v2v/stream
        │     ├── ASR
        │     ├── VAD
        │     └── TTS
        ├── LLM or translation service
        └── optional device plugins/actions
```

`app.py` contains application behavior. `config.yaml` contains local defaults
and environment substitutions. Speech backends, model loading, and
device-specific container wiring remain in the shared SLV and deployment
layers.

## App README contract

An app used by a deployable solution should have a README next to its
`app.py`. Keep the README specific to that app and link here for shared
runtime behavior.

Each app README should contain:

1. Purpose and user-facing behavior.
2. Source entrypoint and configuration file.
3. SenseCraft solution IDs that use the app.
4. Shared dependencies and app-specific dependencies.
5. Deployment matrix.
6. Recommended model/profile matrix.
7. Functional acceptance procedure.
8. Performance records and raw evidence.
9. Known limitations and unmeasured claims.

## Deployment matrix template

| Solution | Target | Compose/config | Services | Agent command | Audio/device requirements | Status |
|---|---|---|---|---|---|---|
| `<solution-id>` | `<board>` | `<path>` | `<speech>`, `<llm>`, `<agent>` | `ovs-agent run <app>` | `<USB/ALSA/actuator>` | `verified / pending` |

Record image tags and immutable digests separately:

| Service | Image tag | Image digest | Runtime/profile | Model/artifact revision |
|---|---|---|---|---|
| speech | `<tag>` | `<sha256:...>` | `<profile>` | `<revision/SHA>` |
| llm | `<tag>` | `<sha256:...>` | `<profile>` | `<revision/SHA>` |
| agent | `<tag>` | `<sha256:...>` | `<config>` | `<commit>` |

## Recommended model template

| Target/profile | ASR | TTS | LLM | Recommendation reason | Evidence |
|---|---|---|---|---|---|
| `<target/profile>` | `<model>` | `<model>` | `<model>` | `<latency/features/resource>` | `<link>` |

A recommendation is not a measurement. A model may be listed as the default
from the deployment configuration while its app-level end-to-end performance
remains `not measured`.

## Functional acceptance template

```bash
docker compose -f <compose-file> config
docker compose -f <compose-file> up -d
curl -fsS http://127.0.0.1:8621/health
curl -fsS http://127.0.0.1:<llm-port>/health   # if a local LLM exists
docker compose -f <compose-file> ps
```

Then verify the actual app behavior:

- microphone input is detected;
- one complete user utterance produces one assistant response;
- ASR language and TTS language match the configuration;
- barge-in stops or drains the current playback as configured;
- the configured LLM/translation path is reachable;
- optional wake-word, tools, actuator actions, or dashboard behavior works;
- no repeated reconnect, audio-device, model-load, or healthcheck errors occur.

Record the command, date, hardware, and raw logs for every acceptance run.

## Performance record template

Create one record per deployment/test run, for example:

`docs/perf/deployments/<date>-<target>-<app>.md`

```md
# <app> deployment record — <date>

- Solution:
- App:
- Device / RAM:
- JetPack/BSP/kernel:
- Power or performance mode:
- Host audio device:
- Compose/config:
- Git commit:
- Speech image tag + digest:
- Agent image tag + digest:
- LLM image/tag or endpoint:
- ASR/TTS/LLM model revisions and SHA-256:
- Relevant environment variables:
- Test command:
- Corpus/fixture and sample count:
- Warm or cold:
- Concurrency:

## Functional result

- Speech health:
- LLM health:
- App startup:
- Single-turn conversation:
- Barge-in:
- Wake word/tools/device action:
- Errors/warnings:

## Measurements

| Metric | Value | Definition | Raw evidence |
|---|---:|---|---|
| startup to ready | TBD | process start to health/ready | `<path>` |
| EOS to first audio | TBD | audio end to first assistant PCM | `<path>` |
| ASR latency | TBD | audio end to final text | `<path>` |
| TTS TTFA | TBD | request/prefill to first audio | `<path>` |
| end-to-end turn latency | TBD | user EOS to assistant first audio/end | `<path>` |
| ASR CER/WER | TBD | named corpus and scorer | `<path>` |
| TTS RTF | TBD | wall time / output duration | `<path>` |
| resident memory | TBD | measurement command and scope | `<path>` |
| VRAM | TBD | measurement command and scope | `<path>` |
| concurrent sessions | TBD | N and pass/fail gate | `<path>` |

## Reproduction

- exact commands:
- raw JSON/log paths:
- artifact manifest:
- reviewer/gate:
```

## Evidence rules

- A number without hardware, image/model pins, configuration, test method, and
  raw output is an observation, not a comparable benchmark.
- Do not copy speech-server benchmark values into an app README unless the app,
  LLM path, audio path, and endpoint definition are the same.
- Mark values as `historical`, `current`, `profile-specific`, or `not measured`.
- Do not report a single deployment result as a guarantee for all boards.
- Never record API keys, passwords, or private endpoints in the repository.
