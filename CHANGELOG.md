# Changelog

History of OpenVoiceStream releases and milestones. The README carries only
the current state; this file carries the record.

### 2026-08 — v0.9.1 Orin NX migration

- Migrated the qualified Orin NX deployment to the v0.9.1 Qwen3-ASR +
  Matcha-TTS speech service and Qwen3.5-4B GDN/MTP LLM, with 8K context by
  default and a qualified optional 4K engine.
- Published immutable, SHA-locked model-level LLM artifacts and documented
  empty-cache installation and the independently tested v0.8 rollback path in
  the [deployment guide](docs/deploy/jetson-orin-nx-v091.md).
- Added the OpenAI-compatible audio/discovery surface: chunked streaming on
  `POST /v1/audio/speech`, transcription on `POST /v1/audio/transcriptions`,
  and model/capability discovery through `GET /v1/models` and
  `GET /v1/capabilities`. Voice and speed support are discovered per model.

### 2026-07 — Historical TensorRT-Edge-LLM v0.9.0 voice-stack upgrade

- **Voice stack (ASR + TTS) upgraded to v0.9.0**, re-verified across six models
  on a real Orin NX (2026-07-04). **For that historical release, the LLM
  service (Qwen3.5-4B GDN) remained on v0.8.0** — v0.9.0 decode parity was
  within ≲2% with no gain, and the v0.9.0
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
  [`jetson-voice-engine`](https://github.com/Seeed-Solution/jetson-voice-engine) (Qwen3 export +
  TensorRT build), [`rkvoice-stream`](https://github.com/Seeed-Solution/rkvoice-stream)
  (Rockchip NPU streaming runtime, on PyPI), and
  [`rkvoice-engine`](https://github.com/Seeed-Solution/rkvoice-engine) (RK model conversion).
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
