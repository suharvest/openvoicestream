# Deploy: v0.9.0 voice stack (one-command `docker run`)

Self-contained v0.9.0 ASR + TTS image for Jetson Orin NX (sm_87 / JetPack 6.2 /
CUDA 12.6 / TensorRT 10.3). All engines, workers, and the plugin are **baked in**
(`/opt/edgellm-v090`) — no host engine mount, no runtime download. Two profiles:
**N=1** (single session, lowest RAM) and **N=2** (two concurrent sessions).

- **Image**: `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.9.0-n1n2-baked-20260707`
- **Digest**: `sha256:4d54b37a57d4c48d1b00648bf677063ee83174c5bc3da9e846755d38e5228059`
- **Size**: ~2.5 GB pull. Dominated by the baked TensorRT engines; the base
  runtime (~353 MB) host-mounts the CUDA/TRT libs so they are not in the image.
  A single ASR thinker engine (`asr-b2`, batch-2) serves both N=1 and N=2 — the
  redundant batch-1 engine is dropped (~670 MB saved vs the 20260706 build,
  which is retained for rollback).

## What's in it

| Piece | Version / detail |
|---|---|
| ASR | Qwen3-ASR 0.6B int4 (streaming; batch-2 `asr-b2` serves both N=1 and N=2) |
| TTS | Qwen3-TTS-12Hz-0.6B CustomVoice int4 (native streaming, lean code2wav) |
| Runtime | TensorRT-Edge-LLM v0.9.0 (`integration/v090-sparktts`), voxedge `0.0.4a0` |
| Concurrency | N=1 (`-customvoice` profile) or N=2 (`-n2` profile, shared-engine TTS + `asr-b2`) |

## Prerequisite: the LLM backend

This image is the **voice stack (ASR + TTS)** only. It calls an OpenAI-compatible
LLM at `EDGE_LLM_BASE_URL`. The Qwen3.5-4B GDN LLM stays on **v0.8.0** (decode
parity — see BENCHMARKS) and runs as a separate `edge-llm-chat-service` container.
Point `EDGE_LLM_BASE_URL` at it (default `http://172.17.0.1:8000/v1` reaches a
service published on the host).

## N=1 — single session (recommended default)

```bash
docker run -d --name seeed-voice-v090 \
  --runtime nvidia --gpus all \
  -p 8621:8000 \
  -e OVS_PROFILE=jetson-edgellm-v090-customvoice \
  -e EDGE_LLM_BASE_URL=http://172.17.0.1:8000/v1 \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.9.0-n1n2-baked-20260707
```

## N=2 — two concurrent sessions

Flips the session-gate triple (`LAZY_TTS=1`, `OVS_TTS_WORKER_CONCURRENCY=2`,
`OVS_MAX_CONCURRENT_SESSIONS=2`) and selects `asr-b2` (batch-2) so ASR opens two
lanes. A 3rd concurrent session is rejected with `4429 too_many_sessions`.
Fits 16 GB Orin NX (~12 GB RAM at N=2, shared-engine TTS).

```bash
docker run -d --name seeed-voice-v090-n2 \
  --runtime nvidia --gpus all \
  -p 8621:8000 \
  -e OVS_PROFILE=jetson-edgellm-v090-n2 \
  -e EDGE_LLM_BASE_URL=http://172.17.0.1:8000/v1 \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.9.0-n1n2-baked-20260707
```

The `-n2` profile carries the session-gate env itself (profile_owned_env), so no
extra `-e` flags are needed — selecting the profile is enough.

## Verify

```bash
curl -s http://127.0.0.1:8621/health          # -> {"asr":true,"tts":true,...}
# offline ASR sanity (a 16 kHz wav):
curl -s -F audio=@sample_zh.wav http://127.0.0.1:8621/asr
# N=2: two concurrent /asr should both transcribe with no cross-talk;
#      a 3rd concurrent session -> HTTP 4429 too_many_sessions.
```

On-device gate (orin-nx, 2026-07-06): N=1 ASR CER 0 + TTS energy OK; N=2 two
concurrent `/asr` CER 0 with no cross-talk, 3rd → 4429, 0 CUDA errors, RAM
11.9 / 15.6 GB.

## Rollback

Stop this container and start the prior N=1 profile / v0.8.0 image. The v0.8.0
profiles (`jetson-edgellm-v080-*`) are retained for rollback.

## Notes

- `docker run` (not compose) on purpose — matches the baseline run command;
  `compose up` can drift the image tag.
- Isolated by design: no named-volume mount, engines are baked. Never mount the
  production `speech-models` volume into a test container.
