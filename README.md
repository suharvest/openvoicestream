# OpenVoiceStream

> **English** | [中文](README.zh-CN.md)

**Native-engine streaming ASR + TTS for edge dialogue.** One container, stable HTTP/WebSocket APIs, and validated paths across Jetson, Rockchip, and Raspberry Pi ecosystems.

<p align="center">
  <a href="https://github.com/suharvest/openvoicestream"><img src="https://img.shields.io/github/stars/suharvest/openvoicestream?style=social" alt="GitHub stars" /></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/ASR-Paraformer%20%7C%20Qwen3--ASR%20%7C%20SenseVoice-2f80ed.svg" alt="ASR: Paraformer, Qwen3-ASR, SenseVoice" /></a>
  <a href="#tts-model-comparison"><img src="https://img.shields.io/badge/TTS-Matcha--TTS%20%7C%20Qwen3--TTS%20%7C%20SparkTTS%20%7C%20Kokoro%20%7C%20MOSS--TTS--Nano-f97316.svg" alt="TTS: Matcha-TTS, Qwen3-TTS, SparkTTS, Kokoro, MOSS-TTS-Nano" /></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/engines-TensorRT--EdgeLLM%20%7C%20RKNN%20%7C%20sherpa--onnx-16a34a.svg" alt="Engines: TensorRT-EdgeLLM, RKNN, sherpa-onnx" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/deploy-Docker-2563eb.svg" alt="Deploy with Docker" /></a>
  <a href="#supported-devices"><img src="https://img.shields.io/badge/ecosystems-Jetson%20%7C%20Rockchip%20%7C%20Raspberry%20Pi-65a30d.svg" alt="Supported ecosystems: Jetson, Rockchip, Raspberry Pi" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-facc15.svg" alt="MIT license" /></a>
</p>

<p align="center">
  <img src="docs/media/hero.png" alt="OpenVoiceStream - streaming ASR and TTS for edge dialogue" width="760" />
</p>

**OpenVoiceStream is the deployable voice product** — the FastAPI/WebSocket server, device profiles, install/deploy machinery, and the agent gallery (voice-controlled robot arm, live captioning, simultaneous interpretation, translation). It runs fully on-device, avoids heavyweight ML frameworks in the hot path, and keeps the client API stable while you switch between sherpa-onnx, TensorRT-EdgeLLM, RKNN, and CPU ONNX backends.

**The speech engine underneath is [`voxedge`](https://github.com/suharvest/voxedge)** — a standalone, pip-installable (`pip install voxedge`), pure-Python/numpy library that does the real-time ASR + TTS + conversation loop. This repo *consumes* voxedge (as a wheel) and adds everything needed to ship it as a product. Want to embed edge voice in your own app? Use voxedge directly. Want a turnkey on-device voice server with prebuilt images and agents? You're in the right place.

## Why This Matters

OpenVoiceStream is meant to make local voice practical at product scale: start
with low-cost real-time voice I/O, then move up to human-like speech or a fully
local voice + LLM loop without changing the client API.

<p align="center">
  <img src="docs/media/solution-lineup.png" alt="OpenVoiceStream solution lineup: recommended hardware paths for real-time voice I/O, production edge voice, human-like local speech, and voice plus local LLM" width="900" />
</p>

Board prices vary by region and kit contents. The point is the order of
magnitude: simple Raspberry Pi-class boards can handle real-time voice input and
output, while Jetson-class edge AI boards can run expressive speech and local LLM
dialogue without a per-call speech API bill.

## Quick Start

Clone once on the target device. The installer validates the host, selects the
right compose file, pulls the image, starts the service, and can run health,
capability, TTS smoke, and TTS-to-ASR round-trip checks.

```bash
git clone --recurse-submodules https://github.com/suharvest/openvoicestream.git
cd openvoicestream

# Auto-detect Jetson, Rockchip, or Raspberry Pi.
deploy/install.sh --pull --verify
```

Choose explicitly when auto-detect is not enough:

```bash
deploy/install.sh --target jetson --pull --verify
deploy/install.sh --target rk3588 --pull --verify
deploy/install.sh --target rk3576 --pull --verify
deploy/install.sh --target rpi --pull --verify
```

> **New to this repo?** [`docs/REPRODUCE.md`](docs/REPRODUCE.md) is the
> end-to-end, from-zero reproduction guide: run a prebuilt image (Path A),
> rebuild the engines from scratch (Path B), or build the images (Path C).

After startup, the service listens on `http://device:8621`:

| Target | URL | Compose file | Image |
|---|---|---|---|
| Jetson | `http://device:8621` | `deploy/docker-compose.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.14-hotswap` |
| RK3576 | `http://device:8621` | `deploy/docker-compose.rk.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-qwen3asr-opt-20260610` |
| RK3588 | `http://device:8621` | `deploy/docker-compose.radxa.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-qwen3asr-opt-20260610` |
| Raspberry Pi | `http://device:8621` | `deploy/docker-compose.rpi.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rpi-v1.0-onnx` |

The published Docker images currently keep the previous registry namespace so
existing deployments can pull the same artifacts during the rename.

Manual verification:

```bash
# Same default URL on Jetson, RK3576, RK3588, and Raspberry Pi.
deploy/verify.sh --url http://device:8621 --tts-smoke --roundtrip
curl http://device:8621/health
```

Client examples live in [`examples/`](examples/):

```bash
python3 examples/stream_tts_to_wav.py \
  --url http://device:8621 \
  --text "你好，欢迎使用 OpenVoiceStream。" \
  --out /tmp/ovs-tts.wav
```

**Deploy with compose** when you want to manage profiles yourself:

```bash
# Chinese + English on Jetson, using the lightweight Paraformer + Matcha path.
docker compose -f deploy/docker-compose.yml up -d

# English only on Jetson.
LANGUAGE_MODE=en docker compose -f deploy/docker-compose.yml up -d

# Kokoro TensorRT TTS on Jetson Orin (TTS only, English, 53 speakers).
OVS_PROFILE=jetson-kokoro-trt docker compose -f deploy/docker-compose.yml up -d

# Paraformer ASR + Kokoro TTS on Jetson Orin (bilingual ASR, English TTS).
OVS_PROFILE=jetson-paraformer-kokoro docker compose -f deploy/docker-compose.yml up -d

# Qwen3 multilingual ASR/TTS on Jetson Orin NX.
OVS_PROFILE=jetson-multilang-highperf-nx \
docker compose -f deploy/docker-compose.yml up -d

# MOSS-TTS-Nano multilingual TTS on Jetson Orin (TTS only, 48kHz stereo, C++ TRT path).
OVS_PROFILE=jetson-moss-tts-nano-trt docker compose -f deploy/docker-compose.yml up -d

# Paraformer RKNN ASR + Matcha RKNN TTS on Rockchip RK3588.
# This profile name is the stable Paraformer alias; it uses the current
# hybrid encoder + RKNN decoder artifact set.
OVS_PROFILE=rk3588-paraformer-matcha \
docker compose -f deploy/docker-compose.radxa.yml up -d

# Qwen3 RKNN ASR + Kokoro RKNN TTS on Rockchip RK3588 (multilingual, NPU-accelerated).
OVS_PROFILE=rk3588-kokoro-rknn \
docker compose -f deploy/docker-compose.radxa.yml up -d

# Paraformer RKNN ASR + Matcha RKNN TTS on Rockchip RK3576.
# This profile name is the stable Paraformer alias; it uses the current
# hybrid encoder + RKNN decoder artifact set.
OVS_PROFILE=rk3576-paraformer-matcha \
docker compose -f deploy/docker-compose.rk.yml up -d
```

`deploy/install.sh --pull --verify` auto-detects Jetson/RK/RPi when run on the
target device. The Jetson default stays on the lightweight `zh_en` path (Paraformer +
Matcha) because it is the fastest path to reproduce. Use `jetson-paraformer-kokoro`
for bilingual ASR with expressive English TTS, `jetson-kokoro-trt` for TTS-only,
`jetson-moss-tts-nano-trt` for lightweight multilingual TTS-only (48kHz stereo),
or a `jetson-multilang-*` profile for the Qwen3 TensorRT-EdgeLLM route. On Rockchip,
use `rk3588-paraformer-matcha` or `rk3576-paraformer-matcha` for the current
validated Paraformer RKNN ASR path (hybrid encoder + RKNN decoder) with Matcha
TTS, or `rk3588-kokoro-rknn` for Qwen3 RKNN ASR with higher-quality multilingual
Kokoro RKNN TTS.

## Demo Gallery

Browser-based demo portal served from the device itself: live device status, one
card per capability (live captions, TTS playground, voice chat with barge-in,
voice clone, speaker diarization), runtime ASR/TTS model hot-switching, and a
kiosk mode for trade shows (`DEMO_KIOSK=1`).

```bash
docker compose -f demos/docker-compose.demos.yml --profile all up -d
# open http://<device>:8700
```

See [`demos/README.md`](demos/README.md) for deployment and server
prerequisites, and [`docs/DEMOS.md`](docs/DEMOS.md) for the full index of demo
assets (gallery cards, API examples, agent examples, bench showpieces).

## Table of Contents

- [Why This Matters](#why-this-matters)
- [Quick Start](#quick-start)
- [Demo Gallery](#demo-gallery)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Qwen3 Multilingual Path](#qwen3-multilingual-path)
- [Performance](#performance)
- [Configuration](#configuration)
- [Models](#models)
- [Supported Devices](#supported-devices)
- [Patched sherpa-onnx](#patched-sherpa-onnx)
- [Project Structure](#project-structure)
- [Changelog](#changelog)
- [Acknowledgements](#acknowledgements)

## Key Features

- **Streaming-first API** — WebSocket ASR with partial/final results and HTTP streaming TTS with sentence-level audio chunks.
- **Native engine runtime** — TensorRT-EdgeLLM on Jetson, RKNN/RKLLM on Rockchip, sherpa-onnx and ONNX Runtime on CPU/CUDA paths.
- **Reusable edge voice library** — the backends ship as the standalone, pip-installable [`voxedge`](https://github.com/suharvest/voxedge) package (`pip install --pre voxedge`); this repo is the product server + deploy on top of it.
- **Stable backend contract** — clients keep the same `/asr/stream`, `/tts`, `/tts/stream`, and `/health` calls when profiles change.
- **Measured low latency** — 58 ms EOS-to-first-audio on Jetson Orin NX with Paraformer + Matcha; 157 ms with Qwen3 ASR/TTS voice clone.
- **TensorRT-Edge-LLM v0.9.0 voice stack** — six models re-verified on Orin NX; **SparkTTS W4A16 is the standout** (RTF 0.50, TTFA 0.41–0.46 s, zero quality loss). The LLM service stays on v0.8.0 by design. See [BENCHMARKS.md](BENCHMARKS.md).
- **Concurrent N=2** — verified 2-session ASR streaming (zh/en, no cross-talk) and N=2 Qwen3-TTS Base (int4 talker, ~4 GB RAM; or shared-engine with only +1.6 GB for the 2nd slot; re-verified on v0.9.0). See [BENCHMARKS.md](BENCHMARKS.md).
- **Multilingual options** — Chinese+English, English-only, and 52-language Qwen3 paths are exposed through the same service.
- **Container-first deploy** — prebuilt images, target-specific compose files, host checks, model downloads, and verification scripts are included.
- **LLM-ready agent layer** — `agent/` streams ASR results into an OpenAI-compatible or EdgeLLM backend, then streams LLM tokens directly back to TTS.
- **Fully local economics** — no speech API key, no per-call ASR/TTS bill, no runtime internet dependency after artifacts are cached, and no PyTorch/Transformers in the voice hot path.

## Architecture

```text
┌───────────────────────────────────────────────────────────┐
│  Edge device (Jetson Orin / RK3576 / RK3588 / RPi 4–5)    │
│                                                           │
│  FastAPI service (container :8000; host default :8621)     │
│  ├── WS /asr/stream    Streaming ASR                      │
│  │     └─ zh_en: Paraformer  │  en: Zipformer  │  multi: Qwen3-ASR  │  rk: Paraformer RKNN · Qwen3-ASR │
│  ├── POST /asr          SenseVoice offline ASR (zh+en)    │
│  ├── POST /tts          Batch TTS                         │
│  └── POST /tts/stream   Streaming TTS                     │
│        └─ zh_en: Matcha-TTS  │  en: Kokoro v1.0  │  multi: Qwen3-TTS │
│                                                           │
│  Inference: sherpa-onnx · TRT-EdgeLLM · RKNN              │
└───────────────────────────────────────────────────────────┘
         ▲ HTTP / WebSocket
         │
   Any client (SBC, laptop, robot, kiosk, ...)
```

Models are selected automatically based on `LANGUAGE_MODE`:

| Service | Endpoint | zh_en (default) | en | multilingual | Protocol |
|---------|----------|-----------------|-----|---------------|----------|
| **Streaming ASR** | `WS /asr/stream` | Paraformer bilingual | Zipformer English | Qwen3-ASR (52 langs) | WebSocket: int16 PCM in, JSON out |
| **Streaming TTS** | `POST /tts/stream` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS (voice clone) | HTTP: JSON in, raw PCM stream |
| **Batch TTS** | `POST /tts` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS (voice clone) | HTTP: JSON in, WAV out |
| Offline ASR | `POST /asr` | SenseVoice (zh+en+ja+ko+yue) | SenseVoice (same) | Qwen3-ASR (52 langs) | HTTP: WAV upload, JSON out |

**Backend capabilities differ:**

| Backend | Speed control | Pitch shift | Voice clone | Languages | Streaming |
|---------|--------------|-------------|-------------|-----------|-----------|
| Sherpa (zh_en/en) | ✅ | ✅ | ❌ | 2 (zh+en) | ✅ |
| Paraformer RKNN (RK) | ❌ | ❌ | ❌ | 2 (zh+en) | ✅ |
| Kokoro TRT (Jetson) | ❌ | ❌ | ❌ | 1 (en) | ✅ |
| Kokoro RKNN (RK3588) | ❌ | ❌ | ❌ | multi | ✅ |
| Qwen3 (multilingual) | ❌ | ❌ | ✅ (x-vector) | 52 | ✅ |
| Qwen3-CustomVoice | ❌ | ❌ | ❌ (9 presets + instruct) | 52 | ✅ |
| MOSS-TTS-Nano (Jetson) | ❌ | ❌ | ❌ | multi | ✅ |
| RKNN (Rockchip) | ✅ | ✅ | ❌ | 2 (zh+en) | ✅ |

The service is model-agnostic at the API level — clients send audio/text, get audio/text back. Swap engines without changing client code. Unsupported parameters return `501` with `{"required_capability": "..."}`.

## API Reference

### Streaming ASR (WebSocket)

```
WS /asr/stream?sample_rate=16000&language=auto
```

- Client sends: raw **int16 PCM bytes** (audio chunks, e.g. 100ms each)
- Client sends: **empty bytes** `b""` to signal end of audio
- Server sends: JSON `{"text": "...", "is_final": bool, "is_stable": bool}`

```python
import asyncio, websockets

async def transcribe():
    async with websockets.connect("ws://device:8621/asr/stream?sample_rate=16000") as ws:
        for chunk in audio_chunks:  # np.int16 arrays
            await ws.send(chunk.tobytes())
            result = await ws.recv()  # partial results
        await ws.send(b"")  # signal end
        final = await ws.recv()  # {"text": "...", "is_final": true}
```

### Offline ASR (HTTP)

```bash
curl -X POST http://device:8621/asr \
  -F "file=@recording.wav" -F "language=auto"
# {"text": "transcribed text"}
```

### TTS (HTTP)

```bash
curl -X POST http://device:8621/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "sid": 52, "speed": 1.0}' \
  --output output.wav
```

Parameters: `text` (required), `sid` (speaker ID, default 52), `speed` (rate, default 1.0)

**Note:** `speed` works only on backends that advertise speed control
(Sherpa/Matcha/RKNN). Qwen3-TTS (`multilanguage` profiles) does not currently
support reliable speed or pitch adjustment, so clients should treat those
parameters as unsupported on Qwen3.

### Speaker Management

Endpoints for listing, registering, and deleting TTS speakers. Speaker IDs are
scoped to the active TTS model.

```bash
# List all speakers for the active TTS model
curl http://device:8621/tts/speakers
# {"model_id": "kokoro-multi-lang-v1_0", "default_speaker_id": 52, "speakers": [...]}

# Register a voice-clone embedding (requires VOICE_CLONE capability)
curl -X POST http://device:8621/tts/speakers/register \
  -H "Content-Type: application/json" \
  -d '{"speaker_embedding_b64": "...", "label": "my-voice"}'

# Delete a registered speaker (preset speakers cannot be deleted)
curl -X DELETE http://device:8621/tts/speakers/42
```

Kokoro exposes 53 preset speakers (ids 0-52) with per-language voice labels
(`af_heart`, `bm_george`, `zf_xiaobei`, etc.). Qwen3-TTS exposes voice-clone
capability via `/tts/clone/embedding` plus persistent registration.

### TTS Streaming (HTTP)

Returns raw PCM: first 4 bytes = sample rate (uint32 LE), then int16 samples.

```
POST /tts/stream
Content-Type: application/json
{"text": "Hello world", "sid": 52}
```

### Health Check

```
GET /health  →  {"asr": bool, "tts": bool, "streaming_asr": bool}
```

## Qwen3 Multilingual Path

`OVS_PROFILE=jetson-multilang-highperf*` enables Qwen3-ASR + Qwen3-TTS — 52 languages plus voice cloning. The integration code lives in this repo; Qwen-specific export, engine builds, and worker glue are maintained in the standalone companion repo [`suharvest/jetson-voice-engine`](https://github.com/suharvest/jetson-voice-engine) (pinned here as a submodule at `third_party/jetson-voice-engine/`). Large model artifacts live in [`harvestsu/qwen3-edgellm-jetson-artifacts`](https://huggingface.co/harvestsu/qwen3-edgellm-jetson-artifacts) on Hugging Face.

**Quickest path on a fresh Orin NX:**

```bash
git clone https://github.com/suharvest/jetson-voice-engine.git
bash jetson-voice-engine/scripts/reproduce_qwen3_highperf.sh \
  --reference /path/to/24kHz_mono.wav   # optional: gates the voice-clone path
```

The orchestrator builds the runtime, downloads + SHA-256-verifies the HF artifacts, builds the slim docker image, starts the service, and runs the verifier (`scripts/verify_reproduction.sh`). Exit 0 means the slim container on port 18092 is healthy and serving the validated stack.

**Two runtime profiles** under the same API surface:

| Profile | Goal | Default behavior |
|---------|------|------------------|
| `official` | Minimal-diff EdgeLLM example. Close enough to upstream that it can be reviewed or upstreamed as a Qwen3 ASR/TTS example. | Semantic/correctness fixes only — tokenizer layout, sampling, runtime contract, stream callback. Regular exported Talker/CodePredictor/Code2Wav directories. |
| `highperf` (default) | Product low-latency dual-resident path for Orin. | Full vocab, ASR FP8 embedding, FP16 CustomVoice Talker on Orin NX with 1024-token Talker KV cap, CP BF16 I/O + `lm_head` pretranspose, stateful Code2Wav, CP decode CUDA graph, `ACTIVE_CP_GROUPS=13`. |

Use `jetson-multilang-highperf-nx` on Orin NX when consuming the NX-native engine set; the default `jetson-multilang-highperf` profile targets the Nano artifact set. Profiles in [`configs/profiles`](configs/profiles) set env defaults only; explicit env vars still override them.

**CustomVoice variant.** Setting `QWEN3_TTS_VARIANT=customvoice` (or an `OVS_TTS_MODEL_ID` containing `customvoice`) selects the Qwen3-TTS-12Hz-0.6B-CustomVoice talker. It ships **9 built-in speakers** (vivian, ryan, aiden, serena, dylan, eric, uncle_fu, ono_anna, sohee) driven by natural-language instructions instead of x-vector voice cloning — so the `VOICE_CLONE` capability is off and `/speakers/register` is rejected. Current CustomVoice production precision is FP16 on Orin NX; the default NX engine uses a 1024-token Talker KV cap to reduce resident memory. W8A16 is rejected until a no-preload EOS-valid quant exists.

For detailed branch ownership, engine env vars, frozen-baseline numbers, and artifact handling, see [`docs/plans/qwen3-current-frozen-baseline-2026-05-10.md`](docs/plans/qwen3-current-frozen-baseline-2026-05-10.md).

Current release status, image digests, artifact repositories, and known gaps are
tracked in [`docs/productization-status.md`](docs/productization-status.md).

## Performance

### Cross-Device Benchmarks (measured 2026-05-18)

Jetson/RPi rows are from the original local forced-EOS gate against
`http://127.0.0.1:8621`. RK rows were rerun after the true-streaming fix with
`QWEN3_ASR_CHUNK_CONFIRM=0`, `--eos vad`, and `--vad-silence-ms 800`; their V2V
column is split `/asr/stream` plus `/tts/stream`.

| Target / profile | Image | TTS backend | ASR backend | TTS RTF p50 | ASR fRTF p50 | ASR CER p50 | V2V EOS→audio p50 |
|---|---|---|---|---:|---:|---:|---:|
| Orin Nano `jetson-multilang-highperf` | `jetson-v1.12-highperf` | `trt_edgellm` | `trt_edgellm` | 0.470 | 0.076 | 5.3% | 251 ms |
| Orin NX `jetson-multilang-highperf-nx` | `jetson-v1.12-highperf` | `trt_edgellm` | `trt_edgellm` | 0.417 | 0.042 | 5.3% | 157 ms |
| Orin Nano `jetson-qwen3asr-matcha` | `jetson-v1.12-highperf` | `matcha_trt` | `trt_edgellm` | 0.024 | 0.075 | 5.3% | 286 ms |
| Orin NX `jetson-qwen3asr-matcha-nx` | `jetson-v1.12-highperf` | `matcha_trt` | `trt_edgellm` | 0.018 | 0.042 | 5.3% | 162 ms |
| Orin Nano `jetson-zh-en` | `jetson-v1.12-highperf` | `matcha_trt` | `paraformer_trt` | 0.023 | 0.077 | 13.3% | 327 ms |
| Orin NX `jetson-zh-en` | `jetson-v1.12-highperf` | `matcha_trt` | `paraformer_trt` | 0.018 | 0.015 | 10.5% | 58 ms |
| RK3588 `rk3588-default` | `rk-qwen3asr-opt-20260610` | `rk:matcha_rknn` | `rk:qwen3_asr_rk` | 0.124 | 0.318 | 10.1% long avg | 528 ms |
| RK3576 `rk3576-default` | `rk-qwen3asr-opt-20260610` | `rk:matcha_rknn` | `rk:qwen3_asr_rk` | 0.290 | 0.265 | 9.8% long avg | 1020 ms |
| Raspberry Pi 5 `rpi5-default` | `rpi-v1.0-onnx` | `sherpa` | `sherpa_asr` | 0.078 | 0.000 | 20.0% | 3 ms |

The RK rows use the 2026-06-10 high-performance Qwen3 ASR W8A8 + Matcha
recheck. Forced client-EOS V2V p50 is 528 ms on RK3588 and 1020 ms on RK3576;
long-dictation average error is 10.1% / 9.8%. The real `/v2v/stream` path still
depends on the configured VAD endpointing delay.

Deployment footprint from the same run:

| Target | Image size | Model / engine volume | Resident memory | Startup to ready |
|---|---:|---:|---:|---:|
| Orin Nano | 2.02 GB | 5.14 GB | 2.14 GiB | 14 s |
| Orin NX | 2.02 GB | 5.45 GB | 1.02 GiB | 13 s |
| RK3588 | 767 MB | 3.31 GB ASR + 301 MB TTS | 4.09 GiB | 9 s |
| RK3576 | 767 MB | 2.21 GB ASR + 351 MB TTS | 2.71 GiB | 15 s |
| Raspberry Pi 5 | 568 MB | 2.19 GB | n/a from Docker stats | 9 s |

Concurrency smoke (`parallel=2`, `asr_tts_simul`) passed on Jetson Nano/NX
Paraformer+Matcha, RK3588, RK3576, and Raspberry Pi 5. Jetson p=2 is
functional but TTS becomes throughput-bound (RTF ~1.3-1.4), so use Orin NX or a
Qwen3 ASR + Matcha split when low-latency concurrent dialogue matters. Full raw
JSON paths and methodology are in
[`docs/benchmarks/streaming-release-gate-2026-05-18.md`](docs/benchmarks/streaming-release-gate-2026-05-18.md).

### v0.8.0 Concurrency (N>1) — verified 2026-06-21

The TensorRT-Edge-LLM v0.8.0 stack adds **validated 2-session concurrency** on
Jetson, with a byte-identical audio/transcript gate (concurrent output ==
solo output) and zero CUDA/race errors. N=2 is the validated ceiling.

- **ASR N=2 streaming** (Orin NX, gate v080-0023) — two concurrent sessions
  (e.g. one Chinese + one English) with no cross-talk; a 3rd concurrent session
  is rejected with `4429 too_many_sessions`. Streaming final CER 0.105 (offline
  ~0.05 on the same clip); 0 CUDA errors.
- **TTS N=2, int4 talker** (Orin Nano) — slot-pool concurrency (independent,
  staggered-friendly lanes). ~4 GB system RAM at N=2 (fits 8 GB and 16 GB), no
  OOM. int4-AWQ+fp8 talker engine is **245.9 MB vs 903 MB fp16 (−73%)**.
- **TTS N=2, shared-engine** (Orin Nano) — the 2nd slot reuses resident weights,
  so it adds **only +1.6 GB** (context/KV, not a 2nd weight copy) — ~436 MB
  saved vs two independent instances. Concurrent output byte-identical to solo.
- **Zero regression vs v0.7.1** (Orin NX) — ASR `--check` 17/20 PASS; English and
  clean Chinese all pass, several clips improved (e.g. `zh_long_01` CER
  0.080 → 0.043). The 3 FAILs are abs-tolerance gate brittleness on high-baseline
  hard-clip clips, not a regression.

Full tables, gates, and reproduction artifacts are in [BENCHMARKS.md](BENCHMARKS.md);
the deployment runbook is [docs/deploy-v080-n1n2.md](docs/deploy-v080-n1n2.md).

### v0.9.0 Upgrade — voice stack on TensorRT-Edge-LLM 0.9.0 (verified 2026-07-04)

The **voice stack (ASR + TTS)** now runs on **TensorRT-Edge-LLM v0.9.0**
(re-verified across six models on a real Orin NX). The **LLM service**
(Qwen3.5-4B GDN) deliberately **stays on v0.8.0** — its decode parity vs v0.9.0
is within ≲2% with no gain, and the v0.9.0 `experimental/server` + GDN combo
crashes.

- **SparkTTS-0.5B — headline win.** On v0.9.0 the **W4A16** INT4-AWQ engine
  becomes the all-round pick: **RTF 0.50** (v0.8.0 baseline 0.74), **TTFA
  0.41–0.46 s** (v0.8.0 bf16 0.64–0.71 s; earlier baseline 0.92 s), with **zero
  quality loss** (ZH CER 0 / EN WER 0). Both bf16 and W4A16 engines ship.
- **Qwen3-ASR 0.6B int4** — streaming + offline transcription **CER 0**, no
  regression vs the v0.8.0 golden set.
- **Qwen3-TTS CustomVoice int4** — 9-row language conditioning, cancel, and EN
  frame counts correct; **RTF 0.61**. N=1 by design (`min(asr 2, tts 1) = 1`).
- **Qwen3-TTS Base** — voice-clone works; the Base embedding controls timbre
  (CAM++ cross-reference cos 0.366 vs same-reference 0.66–0.70).
- **MOSS-TTS-Nano** — TTFA **95–157 ms** (on par with the prior baseline).
- **N=2 shared-engine** re-verified on Base and SparkTTS: ~1284 MB VRAM saved,
  PCM byte-identical, 0 CUDA errors over 50 shots. A production N=2 on v0.9.0
  needs the lean engines (`code2wav optCodeLen=48` + `max_position_embeddings=4096`)
  to absorb the larger init transient.

Pins: fork `integration/v090-sparktts` (v0.9.0 tag `1ac0f2b` + patches), submodule
overlay `repin/v090-overlay`, voxedge wheel `0.0.4a0`. v0.9.0 also retires the mel
front-end (WAV-ingest, `EDGELLM_REQUEST_AUDIO_WAV=1`), adds a native streaming API,
and requires an absolute `EDGELLM_PLUGIN_PATH`. See [BENCHMARKS.md](BENCHMARKS.md)
and the re-port spec
[`docs/specs/edgellm-v090-tts-re-port.md`](docs/specs/edgellm-v090-tts-re-port.md).

### TTS Model Comparison

The current release uses Matcha/Vocos for the bilingual path, Kokoro for
English-only deployments, Qwen3-TTS when voice cloning or 52-language TTS is
required, MOSS-TTS-Nano for a lightweight multilingual TTS-only path, and
SparkTTS for attribute-controllable timbres plus zero-shot voice clone. The RTF
numbers below are from the 2026-05-18 benchmark run where available; the unused
research models are kept as historical context.

| Model | Current role | Streaming RTF p50 | First audio p50 | Notes |
|-------|--------------|------------------:|----------------:|-------|
| **Matcha-TTS + Vocos** | Default bilingual TTS | 0.018 on Orin NX, 0.075 on RK3588, 0.078 on RPi5 | 2.6-7.5 ms | Fastest practical TTS path; no voice clone. |
| **Qwen3-TTS** | Multilingual voice clone | 0.417 on Orin NX, 0.470 on Orin Nano | 4.4-7.3 ms | Higher quality/features, much heavier than Matcha. x-vector clone, or `customvoice` variant (9 instruction-controlled presets). |
| **SparkTTS** | Controllable + voice clone (Jetson) | **0.50 (v0.9.0 W4A16)**, 0.74 on v0.8.0 | **0.41–0.46 s (v0.9.0 W4A16)**, ~0.25 s clone / ~0.9 s controllable on v0.8.0 | Qwen2.5-0.5B + BiCodec single-codebook. **50 controllable timbres** (gender × 5 pitch × 5 speed, no reference audio) **and** zero-shot voice clone (cos ~0.90). On **v0.9.0 W4A16 is the all-round pick** — faster and lighter with zero quality loss; bf16 also ships. W4A16 INT4-AWQ engine 645 MB (−58%), bf16/fp16 mixed-precision (Qwen2.5 fp16-overflow fix). ZH CER 0 / EN WER ≤0.02; N=2 byte-identical. |
| **MOSS-TTS-Nano** | Multilingual TTS-only (Jetson) | — | ~157 ms TTFA on Orin NX | 0.1B model, 48kHz stereo via C++ TRT (19× faster than ORT CPU fallback). No voice clone. |
| **Kokoro v1.0** | English-only TTS | Not in this benchmark run | Historical ~130 ms TTFT | Kept for English-only deployments. On RK3588 a hybrid CPU+NPU RKNN path serves multilingual TTS (`rk3588-kokoro-rknn`). |
| CosyVoice3 | Research only | Not shipped | Historical ~800 ms TTFT | Higher quality, too heavy for this release. |
| F5-TTS | Research only | Not shipped | Historical ~2.5 s TTFT | Not suitable for low-latency edge dialogue. |

Current streaming benchmark scripts live in `bench/perf/`.

### Performance Tuning

Run once after boot on Jetson to lock clocks to max:

```bash
sudo ./scripts/setup-performance.sh
```

This sets MAXN power mode, locks CPU/GPU clocks, and disables dynamic frequency scaling. Critical for consistent inference latency.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OVS_PROFILE` | unset | Preferred OpenVoiceStream profile selector, e.g. `jetson-zh-en`, `jetson-multilang-highperf-nx`, `rk3588-default`, `rpi5-default` |
| `LANGUAGE_MODE` | `zh_en` | `zh_en` (Chinese+English), `en` (English only), or `multilanguage` (Qwen3, 52 langs; profiles usually set this for you) |
| `TTS_PROVIDER` | `cuda` | ONNX execution provider |
| `TTS_DEFAULT_SID` | `52` | Default TTS speaker ID (52=af_cute, 3=af_heart) — Sherpa only |
| `TTS_DEFAULT_SPEED` | `1.0` | TTS playback speed for backends that support it; Qwen3-TTS does not |
| `TTS_NUM_THREADS` | `4` | TTS inference threads |
| `TTS_PITCH_SHIFT` | `0` | Pitch shift in semitones — **Sherpa only** |
| `SENSEVOICE_LANGUAGE` | `auto` | SenseVoice language hint |
| `STREAMING_ASR_PROVIDER` | `cuda` | Streaming ASR execution provider |
| `MODEL_DIR` | `/opt/models` | Model storage directory |

Copy `.env.example` to `.env` to customize.

### Jetson Kokoro TensorRT Profile

`OVS_PROFILE=jetson-kokoro-trt` enables the validated Kokoro split-generator
runtime on Jetson Orin. The path is:

```text
TRT encoder prefix -> CPU length regulator -> TRT decoder backbone FP16
-> TRT source BF16 -> TRT generator rest FP16 -> CPU post/ISTFT
```

The profile declares its TensorRT engines in `required_engines`, so startup
uses the normal artifact resolver: cache hit, then prebuilt artifact bundle,
then local Jetson build fallback via `scripts/build_kokoro_split_generator_trt.sh`.
It ships two generator buckets: `64-256` frames and `256-512` frames. Streaming
requests also have a backend-level phoneme-token splitter
(`KOKORO_STREAM_MAX_SEGMENT_TOKENS`, default `64`) so long unpunctuated text is
bounded before it reaches TensorRT; non-streaming `/tts` uses the same guard
instead of silently truncating long input.

Additional Kokoro profiles share the same artifact set:

| Profile | Segment tokens | Use |
|---|---:|---|
| `jetson-kokoro-trt` / `jetson-kokoro-trt-perf` | 64 | Default performance path (TTS only). |
| `jetson-kokoro-trt-quality` | 48 | Conservative long-text quality gate. |
| `jetson-kokoro-trt-long` | 96 | Longer segments, more 256-512 bucket coverage. |
| `jetson-paraformer-kokoro` | 64 | Paraformer ASR + Kokoro TTS combined (bilingual input, English output). |

The corresponding artifact layout is produced with:

```bash
python3 scripts/build_engine_bundle.py \
  --profile configs/profiles/jetson-kokoro-trt.json \
  --out /tmp/seeed-local-voice-kokoro-artifacts \
  --skip-build
```

The frozen artifact record is
[`deploy/artifacts/kokoro_trt_manifest.json`](deploy/artifacts/kokoro_trt_manifest.json);
the reproduction and TTS-to-ASR gate are documented in
[`docs/kokoro-trt-reproduction.md`](docs/kokoro-trt-reproduction.md).
Use `scripts/verify_tts_asr_roundtrip.py` when Kokoro TTS and the local ASR
service are exposed on separate ports.

## Models

Auto-downloaded on first start and cached in a Docker volume:

| Model | Size | Mode | Purpose |
|-------|------|------|---------|
| Paraformer streaming zh-en | ~230 MB | `zh_en` | Streaming ASR (bilingual) |
| Matcha-TTS + Vocos zh-en | ~125 MB | `zh_en` | TTS synthesis |
| Zipformer streaming en | ~65 MB | `en` | Streaming ASR (English only) |
| Kokoro TTS v1.0 | ~719 MB | `en` | TTS synthesis (English, 53 speakers) |
| SenseVoice zh-en-ja-ko-yue | ~500 MB | both | Offline ASR (5 languages) |
| Qwen3-TTS 0.6B + TRT engines | ~2.5 GB | `multilanguage` | TTS + voice clone (52 languages); `customvoice` variant swaps cloning for 9 instruction-controlled preset voices |
| Qwen3-ASR encoder + decoder | ~1.5 GB | `multilanguage` | ASR (52 languages, streaming) |
| MOSS-TTS-Nano 0.1B + TRT engines | ~0.5 GB | `multilanguage` | TTS synthesis only (multilingual, 48kHz stereo); Jetson `jetson-moss-tts-nano-trt` |
| Kokoro RKNN (hybrid) | ~719 MB | RK3588 | Multilingual TTS via CPU+NPU hybrid; `rk3588-kokoro-rknn` |

Measured Docker volume sizes in the current release are larger than individual
model tarballs because they include compiled engines and profile-specific
artifacts: 5.14-5.45 GB on Jetson, 2.56-3.61 GB on RK, and 2.19 GB on
Raspberry Pi 5.

## Supported Devices

OpenVoiceStream is validated on the following hardware. Any device in the same class should work; these are the ones we measure against.

| Device class | Validated on | Notes |
|---|---|---|
| **NVIDIA Jetson Orin** | Jetson Orin Nano 8GB, Orin NX 16GB, AGX Orin | CUDA 12.6 / JetPack 6.2. Full feature set including Qwen3 multilingual + voice clone. |
| **Rockchip NPU** | Radxa ROCK 5T (RK3588), Banana Pi BPI-M5 Pro (RK3576) | RKNN runtime. Qwen3-ASR works; release TTS uses the validated hybrid Matcha path. |
| **Raspberry Pi (CPU)** | Raspberry Pi 5 8GB, Raspberry Pi 4 4GB | CPU inference. Lowest BOM (~$80). Real-time zh+en commands. |

Requirements: Docker plus enough disk for the image and model volume. Current
measured footprints are about 7.5 GB total for Jetson, 3.2-4.4 GB for RK, and
2.8 GB for Raspberry Pi 5. Runtime memory depends on the profile: about 1.0-2.1
GiB on Jetson, 2.7-4.1 GiB on RK, and CPU-only on Raspberry Pi. On Jetson,
NVIDIA Container Runtime is required; on Rockchip, the host NPU driver
(`rknpu`) must be loaded.

## Patched sherpa-onnx

OpenVoiceStream ships a patched sherpa-onnx that fixes Paraformer streaming tail truncation (the stock version drops the last 1–3 characters). The patch:

1. **IsReady()** — forces decode of remaining frames after `InputFinished()`
2. **DecodeStream()** — zero-pads partial final chunks
3. **CIF force-fire** — emits residual tokens at end-of-stream

Pre-built `.so` files live in `patches/sherpa-onnx-lib/` (aarch64, Python 3.10, CUDA 12.6). See `patches/README.md` for rebuild instructions.

## Project Structure

> **New here?** Read [ARCHITECTURE.md](ARCHITECTURE.md) first — it maps the three
> repos (this product + `voxedge` library + `voxedge-engine`), the two processes,
> and how to run the whole thing locally with no GPU. [DEVELOP.md](DEVELOP.md) is
> the dev-box checklist; [docs/CONFIGURATION.md](docs/CONFIGURATION.md) covers
> profiles and env vars.

```text
openvoicestream/
├── server/                  # FastAPI voice service (the product server)
│   ├── main.py              # Endpoints and startup
│   ├── core/                # VAD, ASR/TTS contracts, streaming, HF artifact download
│   └── utils/               # numpy mel + helpers
├── agent/                   # the voice agent — a SEPARATE package + container
│   └── ovs_agent/           #   framework + apps/ (voice_arm = SO-ARM app)
├── voices/                  # Custom voice embeddings (auto-patched into model)
├── bench/                   # Streaming + V2V latency benchmarks (perf harness)
├── patches/                 # Paraformer EOF truncation fix
├── scripts/                 # Engine build, model download, diagnostics
│   └── kokoro_experiments/  # Archived Kokoro graph-surgery investigations
├── examples/                # API usage examples (TTS streaming, V2V client)
├── tests/                   # Integration and E2E tests
├── deploy/
│   ├── docker-compose.yml   # Production deploy (pre-built image)
│   ├── artifacts/           # Deployment manifests
│   └── docker/
│       ├── Dockerfile.jetson  # Jetson Orin Nano/NX/AGX (zh_en or multilingual)
│       ├── Dockerfile.rk      # Rockchip RK3576/RK3588 NPU
│       └── Dockerfile.rpi     # Raspberry Pi 4/5 (CPU)
├── configs/                 # Device profiles (Jetson, RK, RPi)
├── third_party/             # Submodules (independently maintained)
│   ├── jetson-voice-engine  # Qwen3 export + engine build for Jetson
│   └── rkvoice-stream       # Rockchip NPU streaming voice runtime
└── docs/                    # Guides, runbooks, comparison reports
```

The **per-engine ASR/TTS backends live in the sibling [`voxedge`](https://github.com/suharvest/voxedge) library** (`pip install --pre voxedge`), not in this repo. The product's backend registry (`server/core/asr_backend.py` / `tts_backend.py`) points at `voxedge.backends.*`; install `voxedge[rk]` on Rockchip for the NPU runtime.

Clone with `--recurse-submodules` to pull `third_party/*`, or run `git submodule update --init --recursive` after cloning.

### Unified backend structure (self-serve reproduce & publish)

Every backend — Jetson (TensorRT-Edge-LLM), Rockchip (RKNN), and Raspberry Pi
(sherpa-onnx) — follows the **same layout**, so any one of them can be
reproduced, rebuilt, and published without insider knowledge:

| Per-backend asset | Purpose |
|---|---|
| `recipes/` | the engine/model build + export steps (pin the upstream commit, run the export API) |
| `HF_ARTIFACTS` | the published Hugging Face bundles end users pull (e.g. `harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8`) |
| `docs/` (runbook) | deploy + verify steps for that backend (e.g. [docs/deploy-v080-n1n2.md](docs/deploy-v080-n1n2.md)) |
| `AGENTS` | the agent/dispatch guardrails for working on that backend |

Jetson, RK, and RPi are **first-class peers** — none is the "main" backend, and
the same `recipes → HF_ARTIFACTS → docs → AGENTS` contract holds for each, so
anyone can self-serve a reproduction or a release.

> **DIVERGENCE — fork vs self-authored runtime.** The one structural difference
> is the *source* of the runtime: the Jetson backend's runtime extensions live
> in our **fork of TensorRT-Edge-LLM** (upstream-bug fixes + local runtime
> extensions land in the fork; `jetson-voice-engine` only carries overlay /
> recipes and regenerates patches from it). The RK and RPi runtimes are
> **self-authored** (`rkvoice-stream`, patched sherpa-onnx). This is a deliberate
> ownership boundary, not an inconsistency — every backend still exposes the same
> recipes/artifacts/docs/agents surface above.

## Changelog

### 2026-07 — TensorRT-Edge-LLM v0.9.0 voice-stack upgrade

- **Voice stack (ASR + TTS) upgraded to v0.9.0**, re-verified across six models
  on a real Orin NX (2026-07-04). The **LLM service (Qwen3.5-4B GDN) stays on
  v0.8.0** — v0.9.0 decode parity is within ≲2% with no gain, and the v0.9.0
  `experimental/server` + GDN combo crashes.
- **SparkTTS W4A16 is the headline win** — on v0.9.0 it becomes the all-round
  pick: **RTF 0.50** (was 0.74) and **TTFA 0.41–0.46 s** (was 0.64–0.71 s bf16 /
  0.92 s earlier) with **zero quality loss**. bf16 and W4A16 engines both ship.
- Qwen3-ASR int4 CER 0 (no regression); CustomVoice int4 RTF 0.61 (N=1 by
  design); Base voice-clone works; MOSS-TTS-Nano TTFA 95–157 ms. N=2 shared-engine
  re-verified on Base/SparkTTS (~1284 MB VRAM saved, PCM byte-identical, 0 CUDA
  errors / 50 shots).
- Pins: fork `integration/v090-sparktts` (tag `1ac0f2b` + patches), submodule
  overlay `repin/v090-overlay`, voxedge wheel `0.0.4a0`. v0.9.0 retires the mel
  front-end (WAV-ingest), adds a native streaming API, and needs an absolute
  `EDGELLM_PLUGIN_PATH`. See [BENCHMARKS.md](BENCHMARKS.md) and
  [`docs/specs/edgellm-v090-tts-re-port.md`](docs/specs/edgellm-v090-tts-re-port.md).

### 2026-06 — v0.8.0 N>1 concurrency verified

- **N=2 ASR streaming + N=2 Qwen3-TTS Base verified on Jetson** (2026-06-21).
  Byte-identical concurrent==solo gate, 0 CUDA errors. int4 talker 245.9 MB
  (−73% vs fp16); shared-engine 2nd slot only +1.6 GB. Zero regression vs v0.7.1
  (ASR 17/20, several clips improved). See [BENCHMARKS.md](BENCHMARKS.md) and the
  [deploy runbook](docs/deploy-v080-n1n2.md).

### 2026-06 — Open source & edge voice library split

- **Open source.** OpenVoiceStream is now public (MIT). The repo split into a focused
  product plus independently published libraries.
- **Voice library extracted to `voxedge`.** The per-engine ASR/TTS backends moved out
  of the product into a standalone, pip-installable library — `pip install --pre voxedge`
  (the product depends on it; `voxedge[rk]` also pulls the Rockchip runtime). Engine-build
  and model-conversion tooling split into companion repos:
  [`jetson-voice-engine`](https://github.com/suharvest/jetson-voice-engine) (Qwen3 export +
  TensorRT build), [`rkvoice-stream`](https://github.com/suharvest/rkvoice-stream)
  (Rockchip NPU streaming runtime, on PyPI), and
  [`rkvoice-engine`](https://github.com/suharvest/rkvoice-engine) (RK model conversion).
- **Product package renamed** `app/` → `server/` (imports are `server.core.*`; entrypoint
  `server.main:app`).
- **Slim images self-provision from Hugging Face.** New slim image variants ship without
  baked model engines and pull the host-matched artifact set from HF on first boot (the
  thick images still bake them). Current published builds: Jetson `prod-unified-v8`
  (unified slim) and Rockchip `rk-slim-2026-06-01`. The `deploy/docker-compose*.yml`
  defaults still pin the stable baked tags listed below — set the image explicitly to run
  a slim build.
- **Actionable provisioning + agent hardening.** Engine resolution now reports per-engine
  failures with stable codes (F1–F7) and copy-pasteable fixes instead of a bare crash; the
  voice agent gained server-loop tool-calling, barge-in, and reconnect robustness.

### Stable baked images (compose defaults)

- **Jetson** — `jetson-v1.14-hotswap`, ~2 GB, host CUDA/TensorRT mounted from JetPack and
  models/engines cached in `speech-models`. Ships the BackendManager hot-reload state
  machine (`POST /admin/backend/reload`, `GET /admin/backend/status`) for live profile
  swaps without container recreate. Tags are immutable once published; compose files
  reference them explicitly so upgrades are a deliberate commit, not a floating tag.
- **Rockchip** — `rk-v1.4-closedloop`, 767 MB, runtime-pinned RKNN dependencies and
  validated hybrid Matcha TTS.
- **Raspberry Pi** — `rpi-v1.0-onnx`, 568 MB, CPU-only ONNX path.

See the 2026-05-18 benchmark report for image size, model volume,
resident memory, startup time, and concurrency results.

### v2.3

- **Paraformer + Kokoro combined profile** — new `jetson-paraformer-kokoro` profile pairs bilingual Paraformer ASR with Kokoro TensorRT TTS (53 English speakers) on Jetson Orin.
- **Paraformer RKNN on Rockchip** — NPU-accelerated Paraformer ASR via RKNN (hybrid encoder on NPU + RKNN decoder) with dedicated `rk3588-paraformer-matcha` and `rk3576-paraformer-matcha` profiles. The older CPU-decoder Paraformer path is deprecated.
- **Model-scoped speaker registry** — speaker tables are now per-TTS-model; Kokoro exposes all 53 labeled voices (`af_heart`, `bm_george`, `zf_xiaobei`, etc.).
- **Speaker management API** — `GET /tts/speakers`, `POST /tts/speakers/register`, `DELETE /tts/speakers/{id}` for listing, registering, and deleting speakers.
- **Profile loader hardening** — operator-set env keys are preserved across profile reloads; stale keys are cleaned on profile switch.
- **TTS speaker resolution** — `speaker_kwargs_for_id()` resolves speakers against the active model, unifying the code path across Kokoro, Qwen3, Matcha, and sherpa backends.

### v2.2

- **Endpoint detection** — server proactively sends `is_final` when the speaker pauses (0.6s trailing silence), reducing response latency
- **Fix WebSocket lifecycle** — properly close connections after finalize to prevent stale connection reuse
- **Production deploy compose** — `deploy/docker-compose.yml` with pre-built image (no build step needed)

### v2.1

- Streaming TTS with sentence-level callback
- Custom voice embedding support via pitch shift

### v2.0

- Initial release: Paraformer + Matcha (zh_en), Zipformer + Kokoro (en)
- Patched sherpa-onnx for Paraformer streaming EOF fix

## Contributing

Issues and PRs are welcome. The most useful contributions:

- New backend integrations (other NPUs, other inference engines)
- Streaming benchmarks on additional hardware
- Bug reports with reproducible audio samples and `LANGUAGE_MODE` / profile info
- Documentation improvements, especially deployment recipes for new devices

If you're working on a larger change, open an issue first to align on the approach. Sub-project changes (Qwen3 export, Rockchip runtime) belong in their own repos: [`jetson-voice-engine`](https://github.com/suharvest/jetson-voice-engine), [`rkvoice-stream`](https://github.com/suharvest/rkvoice-stream).

## Acknowledgements

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — speech inference engine powering the bilingual ASR and TTS paths
- [next-gen Kaldi](https://github.com/k2-fsa) — research foundation behind sherpa-onnx
- [Paraformer](https://github.com/modelscope/FunASR) — streaming bilingual ASR model
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — fast flow-matching TTS (zh+en mode)
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — high-quality English TTS with 53 speakers (en mode)
- [Zipformer](https://github.com/k2-fsa/icefall) — efficient transducer ASR (en mode)
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — multilingual offline ASR
- [Qwen3](https://huggingface.co/Qwen) — multilingual ASR + TTS foundation model (52-language path)
- [TensorRT-EdgeLLM](https://github.com/NVIDIA/TensorRT-LLM) — Jetson inference runtime for the Qwen3 path
- [RKNN Toolkit](https://github.com/rockchip-linux/rknn-toolkit2) — Rockchip NPU runtime for the RK3576/RK3588 path
