"""voxedge Jetson TensorRT backend adapters.

adapted from app/backends/jetson/* (2026-05-30), dedup after registry switch.

Decoupled, additive copies of the production Jetson TRT backends:

  * :mod:`voxedge.backends.jetson.matcha_trt`     — Matcha TTS (TRT vocos + estimator)
  * :mod:`voxedge.backends.jetson.kokoro_trt`     — Kokoro TTS (engine / hybrid / split)
  * :mod:`voxedge.backends.jetson.qwen3_trt`      — Qwen3-TTS (C++ pybind pipeline)
  * :mod:`voxedge.backends.jetson.moss_tts_nano`  — MOSS-TTS-Nano (JSONL subprocess worker)
  * :mod:`voxedge.backends.jetson.paraformer_trt` — Paraformer streaming ASR (TRT enc/dec)

Every backend takes an explicit ``XxxConfig`` dataclass at construction time;
no module-scope ``os.environ`` reads, ABCs sourced from
:mod:`voxedge.backends.base` and :mod:`voxedge.engine.concurrency_capability`,
and all heavy runtime imports (tensorrt / cuda / onnxruntime / piper_phonemize /
tokenizers / pybind engine) are deferred into methods so the package imports
cleanly on a CUDA-less dev box (e.g. macOS).
"""
