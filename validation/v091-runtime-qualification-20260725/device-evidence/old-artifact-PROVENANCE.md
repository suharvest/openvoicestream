# TensorRT-Edge-LLM v0.9.1 Orin NX release

- Artifact set: `orin-nx-edgellm-v091-jp62-trt103-sm87-20260725`
- NVIDIA upstream: `https://github.com/NVIDIA/TensorRT-Edge-LLM.git`
- Upstream tag/SHA: `v0.9.1` / `7f061f21f0a581ba234a1e233c9315b89d8e47d6`
- Local overlay: 41 ordered patches, series SHA-256 `aea49a0efc1acafd8ac18e0a4e6606fb4f7398be3b28fd9bee912bf613b90610`
- Platform: Jetson Orin NX 16GB, SM87, JetPack 6.2, L4T R36.4.3, CUDA 12.6, TensorRT 10.3.0.30, aarch64
- Production LLM: Qwen3.5-4B GDN+MTP, native runtime N=1 guarded by service singleflight
- ASR: Qwen3-ASR INT4 b1/b2, validated N=2
- TTS: CustomVoice INT4/FP16 N=2; Base INT4 isolated N=2 and N=1 while GDN is active; SparkTTS BF16/W4A16 N=2; MOSS-TTS-Nano N=2 with local concurrent dispatcher/cooperative cancel worker
- Auxiliary ASR: SenseVoiceSmall TRT

`manifest.json` is the machine-readable inventory. `SHA256SUMS` covers every payload file present before those two envelope files were generated. Engines are device/runtime-bound and must not be used on a different TensorRT/CUDA/SM target.

MOSS is a local model extension and is intentionally excluded from NVIDIA bug-fix PRs. Upstream PR candidates are prepared locally only and must not be submitted without explicit owner confirmation.
