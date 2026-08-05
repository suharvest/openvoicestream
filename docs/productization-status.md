# Productization Status

Last updated: 2026-08-05.

This is the release checklist for making OpenVoiceStream reproducible,
high-performance, and usable as a streaming edge voice library.

## Current Release Artifacts

| Target | Image | Artifact source | Release-gate result |
|---|---|---|---|
| Jetson Orin NX 16 GB, v0.9.1 local-LLM profile | Speech: `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-jp62-trt103-edgellm-v091-20260804-r5`; LLM: `sensecraft-missionpack.seeed.cn/solution/edge-llm-chat-service:v0.9.1-gdn-mtp-runtime-20260804-v13` | Model-level immutable HF revisions in `deploy/artifacts/v091-release-lock.json`: Qwen3-ASR, Matcha, and Qwen3.5-4B GDN/MTP 4K/8K | PASS: empty-cache install, ASR + Matcha + 8K LLM overlap, OpenAI-compatible HTTP, and v0.8 rollback |
| Legacy/general Jetson voice profiles | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.12-highperf` | `harvestsu/qwen3-edgellm-jetson-artifacts` for Qwen3; `harvestsu/seeed-local-voice-artifacts` for Paraformer/Matcha TRT `zh_en` engines | Historical profile-specific gates remain documented in `BENCHMARKS.md`; this row is not the v0.9.1 local-LLM deployment |
| RK3576/RK3588 | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-qwen3asr-opt-20260610` | `harvestsu/seeed-local-voice-rk-artifacts` plus `deploy/artifacts/rk_manifest.json` | Runtime and service PASS; Qwen3 ASR W8A8 + hybrid Matcha TTS validated on RK3576 and RK3588 |
| Raspberry Pi 4/5 / CM4/CM5 | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rpi-v1.0-onnx` | Official ONNX assets downloaded at first boot | PASS for CPU TTS to ASR round-trip on RPi5 |

## Reproduction Path

| Requirement | Status | Evidence |
|---|---|---|
| One-command deployment | Done | `deploy/install.sh --pull --verify` auto-detects Jetson/RK/RPi on-device; `--target jetson|rk3576|rk3588|rpi` remains available for explicit deploys. |
| Target-specific runtime checks | Done | `deploy/install.sh` checks Docker, compose, disk, Jetson NVIDIA runtime, RK `/dev/rknpu`, and RPi architecture. |
| Runtime/artifact compatibility | Done | Jetson/RK use manifest/version checks and prebuilt runtime sidecars; RPi uses ONNX directly. |
| Artifact download and verification | Done | v0.9.1 model-level payloads are locked by immutable revision, size, and SHA-256; RK manifests lock generated RKNN/RKLLM files; older Qwen3 high-performance flows also verify HF artifacts. |
| Stable API across backends | Done | Legacy `/asr/stream`, `/asr`, `/tts`, `/tts/stream`, `/health`, and `/capabilities` remain; OpenAI-compatible `/v1/audio/transcriptions`, chunked `/v1/audio/speech`, `/v1/models`, and `/v1/capabilities` expose the selected deployment. |
| Model and voice discovery | Done | `/v1/models` returns canonical model IDs and aliases; `/v1/capabilities` returns readiness, voices, speed/pitch/cloning controls, streaming formats, and concurrency ceilings for the active profile. |
| Copy-paste client examples | Done | `examples/stream_tts_to_wav.py` covers zero-dependency HTTP TTS streaming; `examples/v2v_tts_only.py` covers `/v2v/stream` TTS token forwarding. |
| Matcha zh/en automatic language handling | Done | Matcha/Sherpa/RK TTS paths normalize `language=auto` to Chinese/English based on text. |
| Agent multi-mode shell | Done | `ovs-agent run` defaults to `MultiModeApp`; chat, interpreter, monologue, and transcribe modes share the same streaming pipeline. |
| Agent prompt tuning UX | Done | Debug dashboard edits the active mode `system_prompt` and `temperature`, with YAML persistence when started from a config file. |
| Agent runtime robustness | Done | Wake/sleep gates drop late ASR events, PTT/VAD `asr_eos` is deduped per turn, silent modes restore IDLE, and shutdown cancels pending sleep timers. |
| Robot product scaffold | Done | `ovs-agent run companion_robot` provides a dedicated App shell for embodied assistants while reusing the same streaming SLV pipeline. |
| Streaming cache hit metrics | Implemented | Agent parses streamed `cache_metrics`; TensorRT Edge LLM companion repo commit `18a955c` emits cache metrics on the final SSE chunk. |
| Local non-hardware test gate | Done | `.github/workflows/ci.yml` runs shell syntax, compose config, Python compile, language tests, and agent unit tests. |
| Hardware release gate | Done for current release set | Orin NX v0.9.1 empty-cache/co-residency/rollback, Jetson voice, RK3588/RK3576, and RPi closed-loop gates PASS within their documented profile scopes. |

## Latest Measured Gate

Raw benchmark reports live in `bench/product_results/` where applicable. The
final v0.9.1 device checkpoint is the linked validation document below.

| Target | Report | TTS short zh RTF / TTFA | ASR error / latency | TTS to ASR |
|---|---|---:|---:|---|
| Jetson Orin NX v0.9.1 final profile | `docs/validation/edgellm-v091-release-checkpoint-20260803.md` | Matcha first streamed WAV bytes 88 ms | Qwen3-ASR request 105 ms on the short fixture | ASR + LLM + TTS overlap PASS; 0.358 / 1.592 / 0.380 s |
| Jetson Orin Nano | `manual-closed-loop-20260517` | smoke PASS | provider TRT/TRT | PASS, similarity 1.00 |
| RK3588 | `product_eval_20260517-152334` | 0.161 | 30.8% | PASS, similarity 1.00 |
| Raspberry Pi 5 | `product_eval_20260517-152334` | 0.172 | 7.7% | PASS, similarity 0.80 |

Jetson zh_en was reverified on 2026-05-17 after the Paraformer TRT fix using
the production compose/image path: Matcha TRT TTS -> Paraformer TRT ASR returned
`你好今天天气真不错` for `你好，今天天气真不错。` with similarity `1.0`.
The older `product_eval_20260517-135525` row remains a Qwen3 high-performance
benchmark snapshot, not the current zh_en closed-loop gate.

## Remaining Enhancements

1. Full RKNN Matcha/Vocos TTS is experimental for closed-loop V2V. The release
   profile uses the validated hybrid NPU path: Matcha acoustic on ORT and Vocos
   on RKNN/NPU.
2. Hardware tests are not suitable for public CI yet. They should stay as
   explicit device release gates because they require Jetson/RK/RPi runners and
   large model volumes.

## Next High-Value Work

1. Re-export or repair the full RKNN Matcha/Vocos TTS artifacts so
   `MATCHA_USE_ORT=0`, `MATCHA_MODEL_SEQ_LEN=96`, and
   `MATCHA_MODEL_FRAMES=256` can pass the same closed-loop gate as the hybrid
   release path.
2. Add a hardware release runner that executes `bench/product_eval.py` on the
   device fleet and publishes the JSON/Markdown reports automatically.
3. Promote TensorRT Edge LLM streaming `cache_metrics` support upstream, or keep
   the companion runtime pinned until that patch is released.

The Orin NX v0.9.1 migration itself has no remaining release blocker. Native
simultaneous GDN contexts remain an upstream/runtime optimization issue, so
the qualified production service advertises and enforces N=1 for that model.
