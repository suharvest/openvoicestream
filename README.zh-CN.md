# OpenVoiceStream

> [English](README.md) | **中文**

**面向边缘对话的原生引擎流式 ASR + TTS。** 单一容器，稳定的 HTTP/WebSocket API，并在 Jetson、Rockchip 与 Raspberry Pi 生态上经过验证的运行路径。

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

**OpenVoiceStream 是可直接部署的语音产品** —— 包含 FastAPI/WebSocket 服务、设备 profile、安装/部署工具链，以及 agent 应用集（语音控制机械臂、实时字幕、同声传译、翻译）。它完全在设备本地运行，在热路径上避免使用重量级 ML 框架，并在你于 sherpa-onnx、TensorRT-EdgeLLM、RKNN 和 CPU ONNX 后端之间切换时，保持客户端 API 稳定不变。

**底层的语音引擎是 [`voxedge`](https://github.com/suharvest/voxedge)** —— 一个独立的、可通过 pip 安装（`pip install voxedge`）的纯 Python/numpy 库，负责实时 ASR + TTS + 对话循环。本仓库以 wheel 形式 *使用* voxedge，并在其之上补齐将其作为产品交付所需的一切。想在自己的应用里嵌入边缘语音？直接使用 voxedge。想要一套开箱即用、带预构建镜像和 agent 的设备端语音服务？那你来对地方了。

## Why This Matters

OpenVoiceStream 的目标是让本地语音在产品规模上变得可行：从低成本的实时语音输入/输出起步，然后在不改动客户端 API 的前提下，进阶到拟人化语音，或完全本地的语音 + LLM 对话循环。

<p align="center">
  <img src="docs/media/solution-lineup.png" alt="OpenVoiceStream solution lineup: recommended hardware paths for real-time voice I/O, production edge voice, human-like local speech, and voice plus local LLM" width="900" />
</p>

主板价格因地区和套件内容而异。重点在于量级：简单的 Raspberry Pi 级别主板就能处理实时语音输入和输出，而 Jetson 级别的边缘 AI 主板则可以运行富有表现力的语音以及本地 LLM 对话，无需为每次调用支付语音 API 费用。

## Quick Start

在目标设备上克隆一次即可。安装器会校验主机、选择正确的 compose 文件、拉取镜像、启动服务，并可运行健康检查、能力检查、TTS 冒烟测试以及 TTS-到-ASR 往返测试。

```bash
git clone --recurse-submodules https://github.com/suharvest/openvoicestream.git
cd openvoicestream

# Auto-detect Jetson, Rockchip, or Raspberry Pi.
deploy/install.sh --pull --verify
```

当自动检测不足以满足需求时，可显式指定：

```bash
deploy/install.sh --target jetson --pull --verify
deploy/install.sh --target rk3588 --pull --verify
deploy/install.sh --target rk3576 --pull --verify
deploy/install.sh --target rpi --pull --verify
```

> **初次接触本仓库？** [`docs/REPRODUCE.md`](docs/REPRODUCE.md) 是端到端、从零开始的复现指南：运行预构建镜像（路径 A）、从零重建引擎（路径 B），或构建镜像（路径 C）。

启动后，服务监听在 `http://device:8621`：

| Target | URL | Compose file | Image |
|---|---|---|---|
| Jetson | `http://device:8621` | `deploy/docker-compose.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.14-hotswap` |
| RK3576 | `http://device:8621` | `deploy/docker-compose.rk.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-qwen3asr-opt-20260610` |
| RK3588 | `http://device:8621` | `deploy/docker-compose.radxa.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-qwen3asr-opt-20260610` |
| Raspberry Pi | `http://device:8621` | `deploy/docker-compose.rpi.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rpi-v1.0-onnx` |

目前已发布的 Docker 镜像仍沿用先前的 registry 命名空间，以便现有部署在改名期间仍能拉取相同的产物。

手动验证：

```bash
# Same default URL on Jetson, RK3576, RK3588, and Raspberry Pi.
deploy/verify.sh --url http://device:8621 --tts-smoke --roundtrip
curl http://device:8621/health
```

客户端示例位于 [`examples/`](examples/)：

```bash
python3 examples/stream_tts_to_wav.py \
  --url http://device:8621 \
  --text "你好，欢迎使用 OpenVoiceStream。" \
  --out /tmp/ovs-tts.wav
```

**使用 compose 部署**，当你希望自己管理 profile 时：

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

在目标设备上运行时，`deploy/install.sh --pull --verify` 会自动检测 Jetson/RK/RPi。Jetson 默认保持在轻量级的 `zh_en` 路径（Paraformer + Matcha），因为它是最快的复现路径。需要双语 ASR 搭配富有表现力的英文 TTS 时使用 `jetson-paraformer-kokoro`，仅需 TTS 时使用 `jetson-kokoro-trt`，需要轻量级多语言纯 TTS（48kHz 立体声）时使用 `jetson-moss-tts-nano-trt`，需要 Qwen3 TensorRT-EdgeLLM 路线时使用 `jetson-multilang-*` profile。在 Rockchip 上，使用 `rk3588-paraformer-matcha` 或 `rk3576-paraformer-matcha` 走当前已验证的 Paraformer RKNN ASR 路径（hybrid encoder + RKNN decoder）搭配 Matcha TTS，或使用 `rk3588-kokoro-rknn` 走 Qwen3 RKNN ASR 搭配更高质量的多语言 Kokoro RKNN TTS。

## Demo Gallery

设备本机提供的浏览器演示门户：实时设备状态、每个能力一张演示卡（实时字幕、语音合成体验、带打断的语音对话、声音克隆、说话人分离）、运行时 ASR/TTS 模型热切换，以及面向展会的 kiosk 模式（`DEMO_KIOSK=1`）。

```bash
docker compose -f demos/docker-compose.demos.yml --profile all up -d
# 打开 http://<device>:8700
```

部署与服务端前置条件见 [`demos/README.md`](demos/README.md)；全部演示资产（gallery 卡片、API 示例、agent 示例、bench 演示脚本）的总索引见 [`docs/DEMOS.md`](docs/DEMOS.md)。

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

- **流式优先 API** —— 带 partial/final 结果的 WebSocket ASR，以及带句级音频块的 HTTP 流式 TTS。
- **原生引擎运行时** —— Jetson 上的 TensorRT-EdgeLLM、Rockchip 上的 RKNN/RKLLM、CPU/CUDA 路径上的 sherpa-onnx 和 ONNX Runtime。
- **可复用的边缘语音库** —— 各后端以独立的、可通过 pip 安装的 [`voxedge`](https://github.com/suharvest/voxedge) 包形式发布（`pip install --pre voxedge`）；本仓库是构建在其之上的产品服务 + 部署。
- **稳定的后端契约** —— 在 profile 切换时，客户端仍保持相同的 `/asr/stream`、`/tts`、`/tts/stream` 和 `/health` 调用。
- **实测低延迟** —— 在 Jetson Orin NX 上使用 Paraformer + Matcha 时，EOS-到-首音频为 58 ms；使用 Qwen3 ASR/TTS 声音克隆时为 157 ms。
- **TensorRT-Edge-LLM v0.9.0 语音栈** —— 六个模型在 Orin NX 上重新验证；**SparkTTS W4A16 是最大亮点**（RTF 0.50、TTFA 0.41–0.46 s、质量零损）。LLM 服务按设计留在 v0.8.0。详见 [BENCHMARKS.md](BENCHMARKS.md)。
- **N=2 并发** —— 已验证 2 会话 ASR 流式（中/英，无串扰）以及 N=2 Qwen3-TTS Base（int4 talker，约 4 GB RAM；或采用 shared-engine 时第二个槽仅多占用 +1.6 GB；已在 v0.9.0 上重新验证）。详见 [BENCHMARKS.md](BENCHMARKS.md)。
- **多语言选项** —— 中英双语、仅英文，以及 52 语言的 Qwen3 路径，均通过同一个服务暴露。
- **容器优先部署** —— 预构建镜像、针对目标的 compose 文件、主机检查、模型下载和验证脚本均已包含在内。
- **面向 LLM 的 agent 层** —— `agent/` 将 ASR 结果流式送入 OpenAI 兼容或 EdgeLLM 后端，再把 LLM token 直接流式回送到 TTS。
- **完全本地的经济性** —— 无语音 API key、无按次 ASR/TTS 费用、产物缓存后运行时无需联网，且语音热路径中没有 PyTorch/Transformers。

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

模型根据 `LANGUAGE_MODE` 自动选择：

| Service | Endpoint | zh_en (default) | en | multilingual | Protocol |
|---------|----------|-----------------|-----|---------------|----------|
| **流式 ASR** | `WS /asr/stream` | Paraformer 双语 | Zipformer 英文 | Qwen3-ASR（52 语言） | WebSocket：输入 int16 PCM，输出 JSON |
| **流式 TTS** | `POST /tts/stream` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS（声音克隆） | HTTP：输入 JSON，输出原始 PCM 流 |
| **批量 TTS** | `POST /tts` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS（声音克隆） | HTTP：输入 JSON，输出 WAV |
| 离线 ASR | `POST /asr` | SenseVoice（zh+en+ja+ko+yue） | SenseVoice（同上） | Qwen3-ASR（52 语言） | HTTP：上传 WAV，输出 JSON |

**各后端能力不同：**

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

本服务在 API 层面与模型无关 —— 客户端发送音频/文本，得到音频/文本返回。在不改动客户端代码的情况下即可更换引擎。不受支持的参数会返回 `501` 并附带 `{"required_capability": "..."}`。

## API Reference

### 流式 ASR（WebSocket）

```
WS /asr/stream?sample_rate=16000&language=auto
```

- 客户端发送：原始 **int16 PCM 字节**（音频块，例如每块 100ms）
- 客户端发送：**空字节** `b""` 以表示音频结束
- 服务端发送：JSON `{"text": "...", "is_final": bool, "is_stable": bool}`

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

### 离线 ASR（HTTP）

```bash
curl -X POST http://device:8621/asr \
  -F "file=@recording.wav" -F "language=auto"
# {"text": "transcribed text"}
```

### TTS（HTTP）

```bash
curl -X POST http://device:8621/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "sid": 52, "speed": 1.0}' \
  --output output.wav
```

参数：`text`（必填）、`sid`（说话人 ID，默认 52）、`speed`（语速，默认 1.0）

**注意：** `speed` 仅在声明支持语速控制的后端（Sherpa/Matcha/RKNN）上生效。Qwen3-TTS（`multilanguage` profile）目前不支持可靠的语速或音高调整，因此客户端应将这些参数视为在 Qwen3 上不受支持。

### Speaker Management

用于列出、注册和删除 TTS 说话人的接口。说话人 ID 的作用域限定于当前激活的 TTS 模型。

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

Kokoro 暴露 53 个预设说话人（id 0-52），带有各语言的语音标签（`af_heart`、`bm_george`、`zf_xiaobei` 等）。Qwen3-TTS 通过 `/tts/clone/embedding` 暴露声音克隆能力，并支持持久化注册。

### TTS 流式（HTTP）

返回原始 PCM：前 4 字节 = 采样率（uint32 LE），随后是 int16 采样点。

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

`OVS_PROFILE=jetson-multilang-highperf*` 启用 Qwen3-ASR + Qwen3-TTS —— 52 语言外加声音克隆。集成代码位于本仓库；Qwen 专属的导出、引擎构建和 worker 胶水代码维护在独立的配套仓库 [`suharvest/jetson-voice-engine`](https://github.com/suharvest/jetson-voice-engine)（在此以 submodule 形式固定在 `third_party/jetson-voice-engine/`）。大型模型产物位于 Hugging Face 上的 [`harvestsu/qwen3-edgellm-jetson-artifacts`](https://huggingface.co/harvestsu/qwen3-edgellm-jetson-artifacts)。

**在全新 Orin NX 上最快的路径：**

```bash
git clone https://github.com/suharvest/jetson-voice-engine.git
bash jetson-voice-engine/scripts/reproduce_qwen3_highperf.sh \
  --reference /path/to/24kHz_mono.wav   # optional: gates the voice-clone path
```

该编排脚本会构建运行时、下载并以 SHA-256 校验 HF 产物、构建 slim docker 镜像、启动服务，并运行验证器（`scripts/verify_reproduction.sh`）。退出码 0 表示端口 18092 上的 slim 容器健康，并正在提供经过验证的整套栈服务。

在同一 API 表面下提供 **两套运行时 profile**：

| Profile | Goal | Default behavior |
|---------|------|------------------|
| `official` | 最小改动的 EdgeLLM 示例。足够贴近上游，可作为 Qwen3 ASR/TTS 示例被审阅或上游合并。 | 仅做语义/正确性修复 —— tokenizer 布局、采样、运行时契约、stream callback。常规导出的 Talker/CodePredictor/Code2Wav 目录。 |
| `highperf`（默认） | 面向 Orin 的产品级低延迟双驻留路径。 | 完整 vocab、ASR FP8 embedding、Orin NX 上的 FP16 CustomVoice Talker（1024-token Talker KV 上限）、CP BF16 I/O + `lm_head` 预转置、有状态 Code2Wav、CP decode CUDA graph、`ACTIVE_CP_GROUPS=13`。 |

在 Orin NX 上消费 NX 原生引擎集时使用 `jetson-multilang-highperf-nx`；默认的 `jetson-multilang-highperf` profile 面向 Nano 产物集。[`configs/profiles`](configs/profiles) 中的 profile 仅设置 env 默认值；显式 env 变量仍会覆盖它们。

**CustomVoice 变体。** 设置 `QWEN3_TTS_VARIANT=customvoice`（或在 `OVS_TTS_MODEL_ID` 中包含 `customvoice`）会选择 Qwen3-TTS-12Hz-0.6B-CustomVoice talker。它内置 **9 个内建说话人**（vivian、ryan、aiden、serena、dylan、eric、uncle_fu、ono_anna、sohee），由自然语言指令驱动，而非 x-vector 声音克隆 —— 因此 `VOICE_CLONE` 能力关闭，`/speakers/register` 会被拒绝。当前 CustomVoice 生产精度在 Orin NX 上为 FP16；默认 NX 引擎使用 1024-token Talker KV 上限以降低驻留内存。在不存在 no-preload 且 EOS 有效的量化版本之前，W8A16 被拒绝。

关于详细的分支归属、引擎 env 变量、冻结基线数字和产物处理，参见 [`docs/plans/qwen3-current-frozen-baseline-2026-05-10.md`](docs/plans/qwen3-current-frozen-baseline-2026-05-10.md)。

当前发布状态、镜像 digest、产物仓库和已知缺口跟踪在 [`docs/productization-status.md`](docs/productization-status.md)。

## Performance

### 跨设备基准测试（2026-05-18 实测）

Jetson/RPi 行来自最初针对 `http://127.0.0.1:8621` 的本地 forced-EOS gate。RK 行在 true-streaming 修复后以 `QWEN3_ASR_CHUNK_CONFIRM=0`、`--eos vad` 和 `--vad-silence-ms 800` 重新运行；其 V2V 列拆分为 `/asr/stream` 加 `/tts/stream`。

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

RK 行使用 2026-06-10 的高性能 Qwen3 ASR W8A8 + Matcha 复检。强制客户端-EOS 的 V2V p50 在 RK3588 上为 528 ms，在 RK3576 上为 1020 ms；长篇听写平均错误率为 10.1% / 9.8%。真实的 `/v2v/stream` 路径仍取决于所配置的 VAD endpointing 延迟。

同一次运行得到的部署占用：

| Target | Image size | Model / engine volume | Resident memory | Startup to ready |
|---|---:|---:|---:|---:|
| Orin Nano | 2.02 GB | 5.14 GB | 2.14 GiB | 14 s |
| Orin NX | 2.02 GB | 5.45 GB | 1.02 GiB | 13 s |
| RK3588 | 767 MB | 3.31 GB ASR + 301 MB TTS | 4.09 GiB | 9 s |
| RK3576 | 767 MB | 2.21 GB ASR + 351 MB TTS | 2.71 GiB | 15 s |
| Raspberry Pi 5 | 568 MB | 2.19 GB | n/a from Docker stats | 9 s |

并发冒烟测试（`parallel=2`，`asr_tts_simul`）在 Jetson Nano/NX Paraformer+Matcha、RK3588、RK3576 和 Raspberry Pi 5 上均通过。Jetson p=2 功能可用，但 TTS 会变为吞吐受限（RTF 约 1.3-1.4），因此当低延迟并发对话重要时，请使用 Orin NX 或 Qwen3 ASR + Matcha 的拆分方案。完整的原始 JSON 路径和方法学见 [`docs/benchmarks/streaming-release-gate-2026-05-18.md`](docs/benchmarks/streaming-release-gate-2026-05-18.md)。

### v0.8.0 并发（N>1）—— 2026-06-21 验证

TensorRT-Edge-LLM v0.8.0 栈在 Jetson 上新增了 **经过验证的 2 会话并发**，带有逐字节一致的音频/转录 gate（并发输出 == 单独输出），且零 CUDA/race 错误。N=2 是已验证的上限。

- **ASR N=2 流式**（Orin NX，gate v080-0023）—— 两个并发会话（例如一个中文 + 一个英文）无串扰；第 3 个并发会话被拒绝并返回 `4429 too_many_sessions`。流式 final CER 0.105（同一片段离线约 0.05）；0 CUDA 错误。
- **TTS N=2，int4 talker**（Orin Nano）—— 槽池并发（独立、对错峰友好的通道）。N=2 时约 4 GB 系统 RAM（可装入 8 GB 和 16 GB），无 OOM。int4-AWQ+fp8 talker 引擎为 **245.9 MB，对比 fp16 的 903 MB（−73%）**。
- **TTS N=2，shared-engine**（Orin Nano）—— 第二个槽复用驻留权重，因此仅额外增加 **+1.6 GB**（context/KV，而非第二份权重拷贝）—— 相比两个独立实例节省约 436 MB。并发输出与单独输出逐字节一致。
- **相对 v0.7.1 零回归**（Orin NX）—— ASR `--check` 17/20 通过；英文和干净中文全部通过，多个片段有所改善（例如 `zh_long_01` CER 0.080 → 0.043）。3 个失败是高基线 hard-clip 片段上的绝对容差 gate 脆性，并非回归。

完整的表格、gate 和复现产物见 [BENCHMARKS.md](BENCHMARKS.md)；部署 runbook 见 [docs/deploy-v080-n1n2.md](docs/deploy-v080-n1n2.md)。

### v0.9.0 升级 —— 语音栈迁移到 TensorRT-Edge-LLM 0.9.0（2026-07-04 验证）

**语音栈（ASR + TTS）** 现已运行在 **TensorRT-Edge-LLM v0.9.0** 上（六个模型在真实 Orin NX 上重新验证）。**LLM 服务**（Qwen3.5-4B GDN）按设计**留在 v0.8.0** —— 其 decode 相对 v0.9.0 的 parity 在 ≲2% 以内且无收益，且 v0.9.0 `experimental/server` + GDN 组合会崩溃。

- **SparkTTS-0.5B —— 最大亮点。** 在 v0.9.0 上 **W4A16** INT4-AWQ 引擎成为全面优选：**RTF 0.50**（v0.8.0 基线 0.74）、**TTFA 0.41–0.46 s**（v0.8.0 bf16 0.64–0.71 s；更早基线 0.92 s），且**质量零损**（中文 CER 0 / 英文 WER 0）。bf16 与 W4A16 双引擎均发布。
- **Qwen3-ASR 0.6B int4** —— 流式 + 离线转写 **CER 0**，相对 v0.8.0 金标准无回归。
- **Qwen3-TTS CustomVoice int4** —— 9-row 语言条件、cancel、英文帧数均正确；**RTF 0.61**。N=1 by design（`min(asr 2, tts 1) = 1`）。
- **Qwen3-TTS Base** —— 声音克隆生效；Base embedding 控制音色（CAM++ 跨参考 cos 0.366 vs 同参考 0.66–0.70）。
- **MOSS-TTS-Nano** —— TTFA **95–157 ms**（与旧基线持平）。
- **N=2 shared-engine** 在 Base 和 SparkTTS 上重新验证：省约 1284 MB 显存、PCM 逐字节一致、50 连发 0 CUDA 错误。v0.9.0 上生产 N=2 需要 lean 引擎（`code2wav optCodeLen=48` + `max_position_embeddings=4096`）以吸收更大的 init 瞬态。

Pin：fork `integration/v090-sparktts`（v0.9.0 tag `1ac0f2b` + patch）、submodule overlay `repin/v090-overlay`、voxedge wheel `0.0.4a0`。v0.9.0 还退役了 mel 前端（改 WAV-ingest，`EDGELLM_REQUEST_AUDIO_WAV=1`）、新增原生流式 API，并要求 `EDGELLM_PLUGIN_PATH` 使用绝对路径。详见 [BENCHMARKS.md](BENCHMARKS.md) 和 re-port spec [`docs/specs/edgellm-v090-tts-re-port.md`](docs/specs/edgellm-v090-tts-re-port.md)。

### TTS Model Comparison

当前发布版本在双语路径使用 Matcha/Vocos，在仅英文部署使用 Kokoro，在需要声音克隆或 52 语言 TTS 时使用 Qwen3-TTS，为轻量级多语言纯 TTS 路径使用 MOSS-TTS-Nano，以及用 SparkTTS 提供属性可控音色 + zero-shot 声音克隆。下表中的 RTF 数字在可获得处取自 2026-05-18 的基准测试运行；未使用的研究模型作为历史背景保留。

| Model | Current role | Streaming RTF p50 | First audio p50 | Notes |
|-------|--------------|------------------:|----------------:|-------|
| **Matcha-TTS + Vocos** | 默认双语 TTS | Orin NX 上 0.018，RK3588 上 0.075，RPi5 上 0.078 | 2.6-7.5 ms | 实践中最快的 TTS 路径；无声音克隆。 |
| **Qwen3-TTS** | 多语言声音克隆 | Orin NX 上 0.417，Orin Nano 上 0.470 | 4.4-7.3 ms | 质量/特性更高，但比 Matcha 重得多。x-vector 克隆，或 `customvoice` 变体（9 个指令控制的预设）。 |
| **SparkTTS** | 可控音色 + 声音克隆（Jetson） | **0.50（v0.9.0 W4A16）**，v0.8.0 上 0.74 | **0.41–0.46 s（v0.9.0 W4A16）**，v0.8.0 上克隆约 0.25 s / 可控约 0.9 s | Qwen2.5-0.5B + BiCodec 单码本。**50 种可控音色**（性别 × 5 音高 × 5 语速，无需参考音频）**且**支持 zero-shot 声音克隆（音色 cos ~0.90）。在 **v0.9.0 上 W4A16 成为全面优选** —— 更快更轻且质量零损；bf16 也发布。W4A16 INT4-AWQ 引擎 645 MB（−58%），bf16/fp16 混合精度（修复 Qwen2.5 fp16 溢出）。中文 CER 0 / 英文 WER ≤0.02；N=2 字节级一致。 |
| **MOSS-TTS-Nano** | 多语言纯 TTS（Jetson） | — | Orin NX 上约 157 ms TTFA | 0.1B 模型，通过 C++ TRT 输出 48kHz 立体声（比 ORT CPU 兜底快 19×）。无声音克隆。 |
| **Kokoro v1.0** | 仅英文 TTS | 不在本次基准运行中 | 历史值约 130 ms TTFT | 为仅英文部署保留。在 RK3588 上，hybrid CPU+NPU RKNN 路径提供多语言 TTS（`rk3588-kokoro-rknn`）。 |
| CosyVoice3 | 仅研究 | 未发布 | 历史值约 800 ms TTFT | 质量更高，但对本次发布过重。 |
| F5-TTS | 仅研究 | 未发布 | 历史值约 2.5 s TTFT | 不适合低延迟边缘对话。 |

当前的流式基准脚本位于 `bench/perf/`。

### Performance Tuning

在 Jetson 上启动后运行一次，将时钟锁定到最高：

```bash
sudo ./scripts/setup-performance.sh
```

这会设置 MAXN 功耗模式、锁定 CPU/GPU 时钟，并禁用动态频率调节。这对于一致的推理延迟至关重要。

## Configuration

### 环境变量

| Variable | Default | Description |
|----------|---------|-------------|
| `OVS_PROFILE` | unset | 首选的 OpenVoiceStream profile 选择器，例如 `jetson-zh-en`、`jetson-multilang-highperf-nx`、`rk3588-default`、`rpi5-default` |
| `LANGUAGE_MODE` | `zh_en` | `zh_en`（中文+英文）、`en`（仅英文），或 `multilanguage`（Qwen3，52 语言；profile 通常会为你设置此项） |
| `TTS_PROVIDER` | `cuda` | ONNX 执行 provider |
| `TTS_DEFAULT_SID` | `52` | 默认 TTS 说话人 ID（52=af_cute，3=af_heart）—— 仅 Sherpa |
| `TTS_DEFAULT_SPEED` | `1.0` | 支持该功能的后端的 TTS 播放语速；Qwen3-TTS 不支持 |
| `TTS_NUM_THREADS` | `4` | TTS 推理线程数 |
| `TTS_PITCH_SHIFT` | `0` | 音高偏移（半音）—— **仅 Sherpa** |
| `SENSEVOICE_LANGUAGE` | `auto` | SenseVoice 语言提示 |
| `STREAMING_ASR_PROVIDER` | `cuda` | 流式 ASR 执行 provider |
| `MODEL_DIR` | `/opt/models` | 模型存储目录 |

将 `.env.example` 复制为 `.env` 即可自定义。

### Jetson Kokoro TensorRT Profile

`OVS_PROFILE=jetson-kokoro-trt` 在 Jetson Orin 上启用经过验证的 Kokoro split-generator 运行时。其路径为：

```text
TRT encoder prefix -> CPU length regulator -> TRT decoder backbone FP16
-> TRT source BF16 -> TRT generator rest FP16 -> CPU post/ISTFT
```

该 profile 在 `required_engines` 中声明其 TensorRT 引擎，因此启动时使用常规的产物解析器：先命中缓存，然后是预构建产物 bundle，最后通过 `scripts/build_kokoro_split_generator_trt.sh` 走本地 Jetson 构建兜底。它提供两个 generator bucket：`64-256` 帧和 `256-512` 帧。流式请求还有一个后端级别的 phoneme-token 切分器（`KOKORO_STREAM_MAX_SEGMENT_TOKENS`，默认 `64`），使得长的无标点文本在到达 TensorRT 之前就被限定边界；非流式 `/tts` 使用相同的保护机制，而非静默截断长输入。

额外的 Kokoro profile 共享同一产物集：

| Profile | Segment tokens | Use |
|---|---:|---|
| `jetson-kokoro-trt` / `jetson-kokoro-trt-perf` | 64 | 默认性能路径（仅 TTS）。 |
| `jetson-kokoro-trt-quality` | 48 | 保守的长文本质量 gate。 |
| `jetson-kokoro-trt-long` | 96 | 更长的分段，更多 256-512 bucket 覆盖。 |
| `jetson-paraformer-kokoro` | 64 | Paraformer ASR + Kokoro TTS 组合（双语输入，英文输出）。 |

对应的产物布局通过以下命令生成：

```bash
python3 scripts/build_engine_bundle.py \
  --profile configs/profiles/jetson-kokoro-trt.json \
  --out /tmp/seeed-local-voice-kokoro-artifacts \
  --skip-build
```

冻结的产物记录为 [`deploy/artifacts/kokoro_trt_manifest.json`](deploy/artifacts/kokoro_trt_manifest.json)；复现和 TTS-到-ASR gate 记录在 [`docs/kokoro-trt-reproduction.md`](docs/kokoro-trt-reproduction.md)。当 Kokoro TTS 和本地 ASR 服务暴露在不同端口上时，使用 `scripts/verify_tts_asr_roundtrip.py`。

## Models

首次启动时自动下载并缓存在 Docker volume 中：

| Model | Size | Mode | Purpose |
|-------|------|------|---------|
| Paraformer streaming zh-en | ~230 MB | `zh_en` | 流式 ASR（双语） |
| Matcha-TTS + Vocos zh-en | ~125 MB | `zh_en` | TTS 合成 |
| Zipformer streaming en | ~65 MB | `en` | 流式 ASR（仅英文） |
| Kokoro TTS v1.0 | ~719 MB | `en` | TTS 合成（英文，53 说话人） |
| SenseVoice zh-en-ja-ko-yue | ~500 MB | both | 离线 ASR（5 语言） |
| Qwen3-TTS 0.6B + TRT engines | ~2.5 GB | `multilanguage` | TTS + 声音克隆（52 语言）；`customvoice` 变体以 9 个指令控制的预设语音替换克隆 |
| Qwen3-ASR encoder + decoder | ~1.5 GB | `multilanguage` | ASR（52 语言，流式） |
| MOSS-TTS-Nano 0.1B + TRT engines | ~0.5 GB | `multilanguage` | 仅 TTS 合成（多语言，48kHz 立体声）；Jetson `jetson-moss-tts-nano-trt` |
| Kokoro RKNN (hybrid) | ~719 MB | RK3588 | 通过 CPU+NPU hybrid 的多语言 TTS；`rk3588-kokoro-rknn` |

当前发布版本中实测的 Docker volume 大小比单个模型 tarball 更大，因为它们包含已编译的引擎和 profile 专属产物：Jetson 上 5.14-5.45 GB，RK 上 2.56-3.61 GB，Raspberry Pi 5 上 2.19 GB。

## Supported Devices

OpenVoiceStream 在以下硬件上经过验证。任何同类设备应当都能工作；这些是我们用于实测的设备。

| Device class | Validated on | Notes |
|---|---|---|
| **NVIDIA Jetson Orin** | Jetson Orin Nano 8GB、Orin NX 16GB、AGX Orin | CUDA 12.6 / JetPack 6.2。完整特性集，包括 Qwen3 多语言 + 声音克隆。 |
| **Rockchip NPU** | Radxa ROCK 5T (RK3588)、Banana Pi BPI-M5 Pro (RK3576) | RKNN 运行时。Qwen3-ASR 可用；发布版 TTS 使用经过验证的 hybrid Matcha 路径。 |
| **Raspberry Pi (CPU)** | Raspberry Pi 5 8GB、Raspberry Pi 4 4GB | CPU 推理。最低 BOM（约 $80）。实时中英文命令。 |

要求：Docker 加上足以容纳镜像和模型 volume 的磁盘空间。当前实测占用约为 Jetson 总计 7.5 GB、RK 3.2-4.4 GB、Raspberry Pi 5 2.8 GB。运行时内存取决于 profile：Jetson 约 1.0-2.1 GiB，RK 2.7-4.1 GiB，Raspberry Pi 上为纯 CPU。在 Jetson 上，需要 NVIDIA Container Runtime；在 Rockchip 上，必须加载主机 NPU 驱动（`rknpu`）。

## Patched sherpa-onnx

OpenVoiceStream 附带一个打过补丁的 sherpa-onnx，修复了 Paraformer 流式尾部截断问题（原版会丢掉最后 1–3 个字符）。该补丁：

1. **IsReady()** —— 在 `InputFinished()` 后强制解码剩余帧
2. **DecodeStream()** —— 对不完整的最终块进行零填充
3. **CIF force-fire** —— 在流结束时输出残余 token

预构建的 `.so` 文件位于 `patches/sherpa-onnx-lib/`（aarch64、Python 3.10、CUDA 12.6）。重建说明见 `patches/README.md`。

## Project Structure

> **初次接触？** 先阅读 [ARCHITECTURE.md](ARCHITECTURE.md) —— 它梳理了三个仓库（本产品 + `voxedge` 库 + `voxedge-engine`）、两个进程，以及如何在无 GPU 的情况下在本地运行整套系统。[DEVELOP.md](DEVELOP.md) 是开发机检查清单；[docs/CONFIGURATION.md](docs/CONFIGURATION.md) 涵盖 profile 和 env 变量。

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

**各引擎的 ASR/TTS 后端位于同级的 [`voxedge`](https://github.com/suharvest/voxedge) 库中**（`pip install --pre voxedge`），而非本仓库。产品的后端注册表（`server/core/asr_backend.py` / `tts_backend.py`）指向 `voxedge.backends.*`；在 Rockchip 上安装 `voxedge[rk]` 以获得 NPU 运行时。

使用 `--recurse-submodules` 克隆以拉取 `third_party/*`，或在克隆后运行 `git submodule update --init --recursive`。

### 统一的后端结构（自助复现与发布）

每个后端 —— Jetson（TensorRT-Edge-LLM）、Rockchip（RKNN）和 Raspberry Pi（sherpa-onnx）—— 都遵循 **相同的布局**，因此其中任何一个都可以在无内部知识的情况下被复现、重建和发布：

| Per-backend asset | Purpose |
|---|---|
| `recipes/` | 引擎/模型的构建 + 导出步骤（固定上游 commit，运行导出 API） |
| `HF_ARTIFACTS` | 终端用户拉取的已发布 Hugging Face bundle（例如 `harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8`） |
| `docs/`（runbook） | 该后端的部署 + 验证步骤（例如 [docs/deploy-v080-n1n2.md](docs/deploy-v080-n1n2.md)） |
| `AGENTS` | 在该后端上工作的 agent/dispatch 护栏 |

Jetson、RK 和 RPi 是 **一等同侪** —— 没有哪个是“主”后端，且相同的 `recipes → HF_ARTIFACTS → docs → AGENTS` 契约对每个后端都成立，因此任何人都可以自助完成复现或发布。

> **差异 —— fork 与自研运行时。** 唯一的结构性差异在于运行时的 *来源*：Jetson 后端的运行时扩展位于我们 **fork 的 TensorRT-Edge-LLM** 中（上游 bug 修复 + 本地运行时扩展落在 fork 里；`jetson-voice-engine` 只承载 overlay/recipes 并从中重新生成补丁）。RK 和 RPi 运行时是 **自研的**（`rkvoice-stream`、打补丁的 sherpa-onnx）。这是有意为之的归属边界，而非不一致 —— 每个后端仍暴露上述相同的 recipes/artifacts/docs/agents 表面。

## Changelog

### 2026-07 — TensorRT-Edge-LLM v0.9.0 语音栈升级

- **语音栈（ASR + TTS）升级到 v0.9.0**，六个模型在真实 Orin NX 上重新验证（2026-07-04）。**LLM 服务（Qwen3.5-4B GDN）留在 v0.8.0** —— v0.9.0 decode parity 在 ≲2% 以内且无收益，且 v0.9.0 `experimental/server` + GDN 组合会崩溃。
- **SparkTTS W4A16 是最大亮点** —— 在 v0.9.0 上成为全面优选：**RTF 0.50**（原 0.74）、**TTFA 0.41–0.46 s**（原 bf16 0.64–0.71 s / 更早 0.92 s），且**质量零损**。bf16 与 W4A16 双引擎均发布。
- Qwen3-ASR int4 CER 0（无回归）；CustomVoice int4 RTF 0.61（N=1 by design）；Base 声音克隆生效；MOSS-TTS-Nano TTFA 95–157 ms。N=2 shared-engine 在 Base/SparkTTS 上重新验证（省约 1284 MB 显存、PCM 逐字节一致、50 连发 0 CUDA 错误）。
- Pin：fork `integration/v090-sparktts`（tag `1ac0f2b` + patch）、submodule overlay `repin/v090-overlay`、voxedge wheel `0.0.4a0`。v0.9.0 退役 mel 前端（改 WAV-ingest）、新增原生流式 API，并需要绝对路径的 `EDGELLM_PLUGIN_PATH`。详见 [BENCHMARKS.md](BENCHMARKS.md) 和 [`docs/specs/edgellm-v090-tts-re-port.md`](docs/specs/edgellm-v090-tts-re-port.md)。

### 2026-06 — v0.8.0 N>1 并发已验证

- **N=2 ASR 流式 + N=2 Qwen3-TTS Base 在 Jetson 上已验证**（2026-06-21）。逐字节一致的并发==单独 gate，0 CUDA 错误。int4 talker 245.9 MB（相对 fp16 −73%）；shared-engine 第二个槽仅 +1.6 GB。相对 v0.7.1 零回归（ASR 17/20，多个片段有改善）。详见 [BENCHMARKS.md](BENCHMARKS.md) 和 [部署 runbook](docs/deploy-v080-n1n2.md)。

### 2026-06 — 开源 & 边缘语音库拆分

- **开源。** OpenVoiceStream 现已公开（MIT）。仓库拆分为一个聚焦的产品加上独立发布的库。
- **语音库提取为 `voxedge`。** 各引擎的 ASR/TTS 后端从产品中迁出，成为独立的、可通过 pip 安装的库 —— `pip install --pre voxedge`（产品依赖它；`voxedge[rk]` 还会拉取 Rockchip 运行时）。引擎构建和模型转换工具拆分到配套仓库：[`jetson-voice-engine`](https://github.com/suharvest/jetson-voice-engine)（Qwen3 导出 + TensorRT 构建）、[`rkvoice-stream`](https://github.com/suharvest/rkvoice-stream)（Rockchip NPU 流式运行时，在 PyPI 上），以及 [`rkvoice-engine`](https://github.com/suharvest/rkvoice-engine)（RK 模型转换）。
- **产品包重命名** `app/` → `server/`（导入为 `server.core.*`；入口点 `server.main:app`）。
- **slim 镜像从 Hugging Face 自助置备。** 新的 slim 镜像变体不烘焙模型引擎，而在首次启动时从 HF 拉取与主机匹配的产物集（thick 镜像仍会烘焙它们）。当前已发布的构建：Jetson `prod-unified-v8`（统一 slim）和 Rockchip `rk-slim-2026-06-01`。`deploy/docker-compose*.yml` 默认仍固定下方列出的稳定烘焙 tag —— 需显式设置镜像才能运行 slim 构建。
- **可操作的置备 + agent 加固。** 引擎解析现在以稳定的错误码（F1–F7）报告各引擎的失败并给出可复制粘贴的修复方法，而非裸崩溃；语音 agent 获得了 server-loop 工具调用、barge-in 和重连健壮性。

### 稳定的烘焙镜像（compose 默认）

- **Jetson** —— `jetson-v1.14-hotswap`，约 2 GB，主机 CUDA/TensorRT 从 JetPack 挂载，模型/引擎缓存在 `speech-models` 中。附带 BackendManager 热重载状态机（`POST /admin/backend/reload`、`GET /admin/backend/status`），可在不重建容器的情况下进行实时 profile 切换。tag 一经发布即不可变；compose 文件显式引用它们，因此升级是一次有意的 commit，而非浮动 tag。
- **Rockchip** —— `rk-v1.4-closedloop`，767 MB，运行时固定的 RKNN 依赖和经过验证的 hybrid Matcha TTS。
- **Raspberry Pi** —— `rpi-v1.0-onnx`，568 MB，纯 CPU ONNX 路径。

镜像大小、模型 volume、驻留内存、启动时间和并发结果见 2026-05-18 的基准报告。

### v2.3

- **Paraformer + Kokoro 组合 profile** —— 新的 `jetson-paraformer-kokoro` profile 在 Jetson Orin 上将双语 Paraformer ASR 与 Kokoro TensorRT TTS（53 个英文说话人）配对。
- **Rockchip 上的 Paraformer RKNN** —— 通过 RKNN 实现 NPU 加速的 Paraformer ASR（NPU 上的 hybrid encoder + RKNN decoder），并配有专用的 `rk3588-paraformer-matcha` 和 `rk3576-paraformer-matcha` profile。较旧的 CPU-decoder Paraformer 路径已弃用。
- **模型作用域的说话人注册表** —— 说话人表现在按 TTS 模型划分；Kokoro 暴露全部 53 个带标签的语音（`af_heart`、`bm_george`、`zf_xiaobei` 等）。
- **Speaker management API** —— `GET /tts/speakers`、`POST /tts/speakers/register`、`DELETE /tts/speakers/{id}`，用于列出、注册和删除说话人。
- **Profile loader 加固** —— operator 设置的 env key 在 profile 重载间得以保留；陈旧的 key 在 profile 切换时被清理。
- **TTS 说话人解析** —— `speaker_kwargs_for_id()` 针对当前激活的模型解析说话人，统一了 Kokoro、Qwen3、Matcha 和 sherpa 后端之间的代码路径。

### v2.2

- **端点检测** —— 当说话人停顿（0.6s 尾部静音）时，服务端主动发送 `is_final`，降低响应延迟
- **修复 WebSocket 生命周期** —— 在 finalize 后正确关闭连接，防止复用陈旧连接
- **生产部署 compose** —— `deploy/docker-compose.yml`，使用预构建镜像（无需构建步骤）

### v2.1

- 带句级 callback 的流式 TTS
- 通过音高偏移支持自定义语音 embedding

### v2.0

- 首次发布：Paraformer + Matcha（zh_en）、Zipformer + Kokoro（en）
- 打补丁的 sherpa-onnx，修复 Paraformer 流式 EOF 问题

## Contributing

欢迎提交 Issue 和 PR。最有价值的贡献：

- 新的后端集成（其他 NPU、其他推理引擎）
- 在更多硬件上的流式基准测试
- 带可复现音频样本及 `LANGUAGE_MODE` / profile 信息的 bug 报告
- 文档改进，尤其是针对新设备的部署配方

如果你在进行较大的改动，请先开 Issue 以对齐方案。子项目改动（Qwen3 导出、Rockchip 运行时）应归入它们各自的仓库：[`jetson-voice-engine`](https://github.com/suharvest/jetson-voice-engine)、[`rkvoice-stream`](https://github.com/suharvest/rkvoice-stream)。

## Acknowledgements

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) —— 驱动双语 ASR 和 TTS 路径的语音推理引擎
- [next-gen Kaldi](https://github.com/k2-fsa) —— sherpa-onnx 背后的研究基础
- [Paraformer](https://github.com/modelscope/FunASR) —— 流式双语 ASR 模型
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) —— 快速的 flow-matching TTS（zh+en 模式）
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) —— 高质量英文 TTS，带 53 个说话人（en 模式）
- [Zipformer](https://github.com/k2-fsa/icefall) —— 高效的 transducer ASR（en 模式）
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) —— 多语言离线 ASR
- [Qwen3](https://huggingface.co/Qwen) —— 多语言 ASR + TTS 基础模型（52 语言路径）
- [TensorRT-EdgeLLM](https://github.com/NVIDIA/TensorRT-LLM) —— Qwen3 路径的 Jetson 推理运行时
- [RKNN Toolkit](https://github.com/rockchip-linux/rknn-toolkit2) —— RK3576/RK3588 路径的 Rockchip NPU 运行时
