"""Matcha TTS backend via TensorRT (Jetson iGPU).

adapted from app/backends/jetson/matcha_trt.py (2026-05-30), dedup after
registry switch.

Supports: BASIC_TTS, STREAMING (chunked PCM from synthesized audio),
MULTI_LANGUAGE. Models: ORT encoder + estimator (N=3) TRT + vocos TRT in
split mode.

Decoupling from the production copy:
  * ABCs from ``voxedge.backends.base`` (TTSBackend / TTSCapability) and
    ``ConcurrencyCapability`` from ``voxedge.engine.concurrency_capability``.
  * ALL module-scope + ``__init__`` ``os.environ.get(...)`` reads (MATCHA_*,
    VOCOS_ENGINE, ACOUSTIC_ONNX, LEXICON/TOKENS, OVS_TTS_STREAM_MAX_WORKERS,
    OVS_*_ARENA_SIZE_MB, MATCHA_MIN_MEL_FRAMES, OVS_TTS_MODEL_ID, ...) are
    replaced by an explicit :class:`MatchaTRTConfig` injected at construction.
    voxedge has ZERO module-scope env reads. Remaining method-internal reads
    (MATCHA_ACOUSTIC_EP, ACOUSTIC_ONNX during split-onnx generation,
    MATCHA_STREAM_CHUNK_MS) are lifted to config fields too.
  * ``CudaMemoryPool`` + arena sizing come from the sibling ``._util`` (was a
    matcha→kokoro cross-module import + ``_read_arena_size_bytes`` env read).
  * ``resolve_speaker_kwargs`` / ``detect_zh_en`` from ``._util`` (was
    ``app.core.tts_speakers`` / ``app.core.language``).
  * ``model_id`` is a config field (the voxedge ``TTSBackend`` base has no
    ``OVS_TTS_MODEL_ID``-reading ``model_id`` property).
  * Heavy runtime (tensorrt / cuda / onnxruntime / piper_phonemize) stays
    method-local so the module imports on a CUDA-less box.
  * The split-ONNX auto-generation path imported ``scripts.split_matcha_trt``;
    voxedge ships no such generator, so a missing split ONNX raises instead of
    auto-generating (the engines are expected to already exist).
"""

from __future__ import annotations

import io
import logging
import queue
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from voxedge.backends.base import TTSBackend, TTSCapability
from voxedge.engine.concurrency_capability import ConcurrencyCapability

from ._util import (
    CudaMemoryPool,
    arena_size_bytes,
    detect_zh_en,
    resolve_speaker_kwargs,
)

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256

# Model constants
MAX_MEL_FRAMES = 600
MEL_DIM = 80
ODE_DT = 1.0 / 3.0
N_ODE_STEPS = 3
MEL_SIGMA = 5.446792
MEL_MEAN = -2.9521978


# ── env → config mapping (defaults byte-equal to production env defaults) ────
#   MATCHA_MODEL_BASE                   → model_base ("/opt/models/matcha-icefall-zh-en")
#   LANGUAGE_MODE                       → language_mode ("zh_en")
#   VOCOS_ENGINE                        → vocos_engine (<base>/engines/vocos_fp16.engine)
#   ACOUSTIC_ONNX                       → acoustic_onnx (<base>/model-steps-3.onnx)
#   MATCHA_SPLIT_ENCODER_ONNX           → split_encoder_onnx (<base>/onnx/matcha_encoder_trt.onnx)
#   MATCHA_SPLIT_ESTIMATOR_ENGINE       → split_estimator_engine (<base>/engines/matcha_estimator_step0_bf16.engine)
#   LEXICON_PATH                        → lexicon_path (<base>/lexicon.txt)
#   TOKENS_PATH                         → tokens_path (<base>/tokens.txt)
#   MATCHA_MIN_MEL_FRAMES               → min_mel_frames (72)
#   MATCHA_ACOUSTIC_EP                  → acoustic_ep ("")
#   OVS_TTS_STREAM_MAX_WORKERS          → stream_max_workers (2)  ← K, gates parallel
#   OVS_MATCHA_ARENA_SIZE_MB / OVS_CUDA_ARENA_SIZE_MB → arena_size_mb (16)
#   MATCHA_STREAM_CHUNK_MS              → stream_chunk_ms (40)
#   OVS_TTS_MODEL_ID                    → model_id ("matcha_trt")


@dataclass
class MatchaTRTConfig:
    """Explicit construction-time config for :class:`MatchaTRTBackend`.

    Path fields default to the production layout under ``model_base`` (resolved
    in ``__post_init__``). Nothing reads ``os.environ``.
    """

    model_base: str = "/opt/models/matcha-icefall-zh-en"
    language_mode: str = "zh_en"
    vocos_engine: Optional[str] = None
    acoustic_onnx: Optional[str] = None
    split_encoder_onnx: Optional[str] = None
    split_estimator_engine: Optional[str] = None
    lexicon_path: Optional[str] = None
    tokens_path: Optional[str] = None

    min_mel_frames: int = 72
    # "" → full acoustic ONNX on ORT CPU; "SPLIT_TRT"/"TRT_SPLIT"/"HYBRID_TRT"
    # → ORT encoder + TRT estimator ODE loop.
    acoustic_ep: str = ""

    # K concurrent slots (was OVS_TTS_STREAM_MAX_WORKERS). N>1 enables parallel.
    stream_max_workers: int = 2
    # Per-slot CUDA arena size in MB (was OVS_MATCHA_ARENA_SIZE_MB).
    arena_size_mb: int = 16
    # Streaming PCM chunk size in ms (was MATCHA_STREAM_CHUNK_MS).
    stream_chunk_ms: int = 40

    model_id: str = "matcha_trt"

    def __post_init__(self) -> None:
        import os.path as _p
        base = self.model_base
        if self.vocos_engine is None:
            self.vocos_engine = _p.join(base, "engines", "vocos_fp16.engine")
        if self.acoustic_onnx is None:
            self.acoustic_onnx = _p.join(base, "model-steps-3.onnx")
        if self.split_encoder_onnx is None:
            self.split_encoder_onnx = _p.join(base, "onnx", "matcha_encoder_trt.onnx")
        if self.split_estimator_engine is None:
            self.split_estimator_engine = _p.join(
                base, "engines", "matcha_estimator_step0_bf16.engine"
            )
        if self.lexicon_path is None:
            self.lexicon_path = _p.join(base, "lexicon.txt")
        if self.tokens_path is None:
            self.tokens_path = _p.join(base, "tokens.txt")
        self.stream_max_workers = max(1, int(self.stream_max_workers))


def _pad_mel_axis(arr: np.ndarray, min_frames: int) -> np.ndarray:
    """Pad mel-time tensors to the TensorRT profile minimum."""
    frames = int(arr.shape[2])
    if frames >= min_frames:
        return arr
    return np.pad(arr, ((0, 0), (0, 0), (0, min_frames - frames)), mode="constant")


def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 samples to WAV bytes."""
    buf = io.BytesIO()
    num_samples = len(samples)
    data_size = num_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    arr = np.clip(samples, -1.0, 1.0)
    buf.write((arr * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


_HANN_PERIODIC = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann


def _istft(mag: np.ndarray, x: np.ndarray, y: np.ndarray, length: Optional[int] = None) -> np.ndarray:
    """ISTFT matching sherpa-onnx vocos pipeline (knf::StftConfig center=1)."""
    complex_spec = (mag * (x + 1j * y)).astype(np.complex64)  # [F, T]
    n_frames = complex_spec.shape[1]
    output_len = (n_frames - 1) * HOP_LENGTH + N_FFT

    audio = np.zeros(output_len, dtype=np.float32)
    win_sum = np.zeros(output_len, dtype=np.float32)
    sq_window = (_HANN_PERIODIC ** 2).astype(np.float32)
    for i in range(n_frames):
        frame = np.fft.irfft(complex_spec[:, i], n=N_FFT).astype(np.float32) * _HANN_PERIODIC
        start = i * HOP_LENGTH
        audio[start:start + N_FFT] += frame
        win_sum[start:start + N_FFT] += sq_window
    audio = audio / np.maximum(win_sum, 1e-8)

    pad = N_FFT // 2
    audio = audio[pad:-pad] if pad > 0 and len(audio) > 2 * pad else audio

    if length is not None:
        if len(audio) > length:
            audio = audio[:length]
        elif len(audio) < length:
            audio = np.pad(audio, (0, length - len(audio)))
    return audio


class _MatchaCtxSlot:
    """One pre-allocated set of TRT contexts + persistent CudaMemoryPool."""

    def __init__(self, vocos_engine, split_estimator_engines, arena_bytes: int):
        self.pool = CudaMemoryPool(arena_size_bytes=arena_bytes)
        self.vocos_ctx = (
            vocos_engine.create_execution_context() if vocos_engine is not None else None
        )
        self.split_estimator_ctxs = [
            eng.create_execution_context() for eng in split_estimator_engines
        ]

    def reset_per_request(self):
        try:
            self.pool.free_all()
        except Exception:
            logger.exception("MatchaCtxSlot.reset_per_request: pool.free_all raised")

    def destroy(self):
        try:
            self.pool.synchronize()
        except Exception:
            pass
        try:
            self.pool.destroy()
        except Exception:
            pass
        self.vocos_ctx = None
        self.split_estimator_ctxs = []


class MatchaTRTBackend(TTSBackend):
    """Matcha TTS backend (full acoustic ONNX + TRT Vocos, or split-TRT)."""

    supports_hot_reload: bool = True

    def concurrency_capability(self) -> ConcurrencyCapability:
        # K = config.stream_max_workers (was OVS_TTS_STREAM_MAX_WORKERS / profile
        # tts_stream_max_workers). Engines (weights) shared; each slot holds its
        # own TRT execution contexts so K concurrent synthesize() calls never
        # share a context.
        k = max(1, int(self._config.stream_max_workers))
        return ConcurrencyCapability(
            supports_parallel=k > 1,
            max_concurrent=k,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="single_runtime_multiplex",
        )

    def __init__(self, config: Optional[MatchaTRTConfig] = None):
        self._config = config or MatchaTRTConfig()
        self._acoustic_ort = None
        self._split_encoder_ort = None
        self._split_estimator_engines = []
        self._split_estimator_ctxs = []
        self._acoustic_mode = "full_ort"
        self._vocos_engine = None
        self._vocos_ctx = None
        self._cuda_pool = None
        self._lexicon = None
        self._token_to_id = None
        self._ready = False
        self._ctx_pool: "queue.Queue[_MatchaCtxSlot]" = queue.Queue()
        self._slots: list[_MatchaCtxSlot] = []
        self._pool_size: int = 0
        self._min_mel_frames = int(self._config.min_mel_frames)

    @property
    def name(self) -> str:
        return "matcha_trt"

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def capabilities(self) -> set[TTSCapability]:
        return {
            TTSCapability.BASIC_TTS,
            TTSCapability.STREAMING,
            TTSCapability.MULTI_LANGUAGE,
        }

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def is_ready(self) -> bool:
        return self._ready

    def unload(self) -> None:
        """Release TRT engines + execution contexts + ORT sessions + CUDA pool.

        Release ordering (sync stream → ctxs before engines → engines → ORT →
        pool → gc x2) is identical to the production copy. Idempotent.
        """
        if (
            not self._ready
            and self._acoustic_ort is None
            and self._split_encoder_ort is None
            and not self._split_estimator_engines
            and self._vocos_engine is None
            and self._cuda_pool is None
            and not self._slots
        ):
            return

        try:
            if self._cuda_pool is not None:
                try:
                    self._cuda_pool.synchronize()
                except Exception:
                    logger.exception("Matcha unload: pool.synchronize failed; continuing")

            while True:
                try:
                    self._ctx_pool.get_nowait()
                except queue.Empty:
                    break
            for i, slot in enumerate(self._slots):
                try:
                    slot.destroy()
                except Exception:
                    logger.exception("Matcha unload: slot[%d] destroy raised", i)
            self._slots = []
            self._pool_size = 0

            for i, ctx in enumerate(self._split_estimator_ctxs):
                try:
                    del ctx
                except Exception:
                    logger.exception("Matcha unload: estimator ctx[%d] del raised", i)
            self._split_estimator_ctxs = []

            if self._vocos_ctx is not None:
                try:
                    del self._vocos_ctx
                except Exception:
                    logger.exception("Matcha unload: vocos ctx del raised")
                self._vocos_ctx = None

            for i, eng in enumerate(self._split_estimator_engines):
                try:
                    del eng
                except Exception:
                    logger.exception("Matcha unload: estimator engine[%d] del raised", i)
            self._split_estimator_engines = []

            if self._vocos_engine is not None:
                try:
                    del self._vocos_engine
                except Exception:
                    logger.exception("Matcha unload: vocos engine del raised")
                self._vocos_engine = None

            self._acoustic_ort = None
            self._split_encoder_ort = None

            if self._cuda_pool is not None:
                try:
                    self._cuda_pool.destroy()
                except Exception:
                    logger.exception("Matcha unload: pool.destroy failed; continuing")
                self._cuda_pool = None

            import gc
            gc.collect()
            gc.collect()
        except Exception:
            logger.exception("MatchaTRTBackend.unload outer-try failed; continuing")
        finally:
            self._lexicon = None
            self._token_to_id = None
            self._ready = False

    def preload(self) -> None:
        self._load_lexicon()
        self._load_acoustic_ort()
        self._load_engines()
        self._build_ctx_pool()
        self._warmup()
        self._ready = True

    def _build_ctx_pool(self) -> None:
        """Pre-allocate K context slots and seed the queue."""
        k = max(1, int(self._config.stream_max_workers))
        self._pool_size = k
        self._slots = []
        arena_bytes = arena_size_bytes(self._config.arena_size_mb)
        t0 = time.time()
        for _ in range(k):
            slot = _MatchaCtxSlot(
                self._vocos_engine, self._split_estimator_engines, arena_bytes
            )
            self._slots.append(slot)
            self._ctx_pool.put(slot)
        logger.info(
            "Matcha ctx pool: %d slots pre-allocated (%.2fs)", k, time.time() - t0
        )

    def _load_acoustic_ort(self):
        import os
        import onnxruntime as ort
        ep_override = (self._config.acoustic_ep or "").upper()
        if ep_override in ("SPLIT_TRT", "TRT_SPLIT", "HYBRID_TRT"):
            self._acoustic_mode = "split_trt"
            self._ensure_split_onnx()
            if not os.path.exists(self._config.split_encoder_onnx):
                raise FileNotFoundError(
                    f"Split Matcha encoder ONNX not found: {self._config.split_encoder_onnx}."
                )
            sess_opt = ort.SessionOptions()
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._split_encoder_ort = ort.InferenceSession(
                self._config.split_encoder_onnx, sess_opt, providers=["CPUExecutionProvider"]
            )
            logger.info("Split Matcha encoder ORT loaded: %s", self._config.split_encoder_onnx)
            return

        path = self._config.acoustic_onnx
        providers = ["CPUExecutionProvider"]
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._acoustic_ort = ort.InferenceSession(path, sess_opt, providers=providers)
        logger.info("Acoustic ORT loaded (%s): %s",
                     self._acoustic_ort.get_providers()[0], path)

    def _ensure_split_onnx(self) -> None:
        """Verify split ONNX artifacts exist.

        The production copy auto-generated them from the full model via
        ``scripts.split_matcha_trt``; voxedge ships no such generator, so the
        artifacts are expected to already exist (raised by the caller if not).
        """
        import os
        estimator0 = os.path.join(
            os.path.dirname(self._config.split_encoder_onnx),
            "matcha_estimator_step0_trt.onnx",
        )
        if os.path.exists(self._config.split_encoder_onnx) and os.path.exists(estimator0):
            return
        logger.warning(
            "Split Matcha ONNX missing (%s); voxedge ships no generator — "
            "build the split artifacts offline.",
            self._config.split_encoder_onnx,
        )

    def _load_lexicon(self):
        import os
        self._lexicon = {}
        if os.path.exists(self._config.lexicon_path):
            with open(self._config.lexicon_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self._lexicon[parts[0]] = parts[1:]
            logger.info("Loaded %d lexicon entries from %s", len(self._lexicon), self._config.lexicon_path)

        self._token_to_id = {}
        if os.path.exists(self._config.tokens_path):
            with open(self._config.tokens_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.rstrip("\n").rstrip("\r")
                    if not raw:
                        continue
                    rsep = max(raw.rfind(" "), raw.rfind("\t"))
                    if rsep < 0:
                        continue
                    tok = raw[:rsep] or " "
                    try:
                        tid = int(raw[rsep + 1:])
                    except ValueError:
                        continue
                    self._token_to_id[tok] = tid
            logger.info("Loaded %d tokens from %s", len(self._token_to_id), self._config.tokens_path)

    def _load_engines(self):
        """Load TRT engines FIRST, then initialize CUDA memory pool."""
        import os
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)

        def load_engine(path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Engine not found: {path}")
            with open(path, "rb") as f:
                runtime = trt.Runtime(trt_logger)
                engine = runtime.deserialize_cuda_engine(f.read())
            return engine

        t0 = time.time()
        self._vocos_engine = load_engine(self._config.vocos_engine)
        logger.info("Vocos loaded: %s (%.1fs)", self._config.vocos_engine, time.time() - t0)

        if self._acoustic_mode == "split_trt":
            t0 = time.time()
            base_dir = os.path.dirname(self._config.split_estimator_engine)
            self._split_estimator_engines = []
            names = []
            for step in range(N_ODE_STEPS):
                path = os.path.join(base_dir, f"matcha_estimator_step{step}_bf16.engine")
                engine = load_engine(path)
                self._split_estimator_engines.append(engine)
                names.append([engine.get_tensor_name(i) for i in range(engine.num_io_tensors)])
            logger.info(
                "Split Matcha estimator TRT loaded from %s (%.1fs, tensors=%s)",
                base_dir, time.time() - t0, names,
            )

    def _warmup(self):
        texts = ["你好", "你好世界"]
        start = time.time()
        for t in texts:
            self.synthesize(t)
        logger.info("Warmup: %.1fs", time.time() - start)

    _IPA_REPLACEMENTS = [
        ("eɪ", "A"), ("aɪ", "I"), ("ɔɪ", "Y"),
        ("oʊ", "O"), ("əʊ", "O"), ("aʊ", "W"),
        ("tʃ", "ʧ"), ("dʒ", "ʤ"),
        ("ɝ", "ɜɹ"), ("ɚ", "əɹ"),
        ("g", "ɡ"), ("r", "ɹ"), ("e", "ɛ"),
        ("ː", ""),
    ]

    def _phonemize_english(self, text: str) -> list[str]:
        import piper_phonemize
        sentences = piper_phonemize.phonemize_espeak(text, "en-us")
        if not sentences:
            logger.warning("piper-phonemize returned empty for: %r", text)
            return []
        out = []
        for sent_idx, phoneme_list in enumerate(sentences):
            if sent_idx > 0 and " " in self._token_to_id:
                out.append(" ")
            joined = "".join(p for p in phoneme_list if p)
            for src, dst in self._IPA_REPLACEMENTS:
                joined = joined.replace(src, dst)
            for cp in joined:
                if cp in self._token_to_id:
                    out.append(cp)
        return out

    def _text_to_tokens(self, text: str) -> list[int]:
        import re
        tokens: list[int] = []
        space_id = self._token_to_id.get(" ")
        prev_was_english = False

        _FW_PUNCT = {
            "，": ",", "。": ".", "！": "!", "？": "?",
            "、": ",", "；": ";", "：": ":",
            "（": "(", "）": ")", "［": "[", "］": "]",
            "【": "[", "】": "]", "〈": "<", "〉": ">",
            "《": "<", "》": ">",
        }

        segments = re.findall(
            r'[一-鿿]+|[A-Za-z][A-Za-z\' ]*[A-Za-z]|[A-Za-z]|[^一-鿿A-Za-z]+',
            text,
        )

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            if re.match(r'^[一-鿿]+$', seg):
                tokens.extend(self._chinese_to_tokens(seg))
                prev_was_english = False
            elif re.match(r'^[A-Za-z]', seg):
                if prev_was_english and space_id is not None:
                    tokens.append(space_id)
                phonemes = self._phonemize_english(seg)
                for p in phonemes:
                    tid = self._token_to_id.get(p)
                    if tid is not None:
                        tokens.append(tid)
                if not phonemes:
                    logger.warning("Empty phonemes for English seg %r", seg)
                prev_was_english = True
            else:
                for ch in seg:
                    mapped = _FW_PUNCT.get(ch, ch)
                    tid = self._token_to_id.get(mapped)
                    if tid is not None:
                        tokens.append(tid)

        return tokens

    def _chinese_to_tokens(self, text: str) -> list[int]:
        tokens = []
        i = 0
        while i < len(text):
            found = False
            for length in range(min(4, len(text) - i), 0, -1):
                word = text[i:i+length]
                if word in self._lexicon:
                    phonemes = self._lexicon[word]
                    for p in phonemes:
                        if p in self._token_to_id:
                            tokens.append(self._token_to_id[p])
                    i += length
                    found = True
                    break
            if not found:
                char = text[i]
                if char in self._lexicon:
                    phonemes = self._lexicon[char]
                    for p in phonemes:
                        if p in self._token_to_id:
                            tokens.append(self._token_to_id[p])
                i += 1
        return tokens

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        resolve_speaker_kwargs(self.model_id, allow_embedding=False, speaker_id=speaker_id, **kwargs)
        if speed is None:
            speed = 1.0
        detected_language = detect_zh_en(text, language)

        if self._slots:
            slot = self._ctx_pool.get()
            pool = slot.pool
            vocos_ctx = slot.vocos_ctx
            split_estimator_ctxs = slot.split_estimator_ctxs
        else:
            slot = None
            pool = CudaMemoryPool()
            vocos_ctx = (
                self._vocos_engine.create_execution_context()
                if self._vocos_engine is not None else None
            )
            split_estimator_ctxs = [
                eng.create_execution_context()
                for eng in self._split_estimator_engines
            ]

        try:
            t_start = time.time()

            t0 = time.time()
            tokens = self._text_to_tokens(text)
            text_ms = (time.time() - t0) * 1000
            if len(tokens) == 0:
                logger.warning("No tokens for text: %r", text)
                silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
                return _samples_to_wav(silence, SAMPLE_RATE), {
                    "duration": 0.1,
                    "inference_time": 0.0,
                    "language": detected_language,
                }

            num_tokens = min(len(tokens), 80)
            t0 = time.time()
            x = np.array([tokens[:num_tokens]], dtype=np.int64)
            x_length = np.array([num_tokens], dtype=np.int64)
            noise_scale = np.array([1.0], dtype=np.float32)
            length_scale = np.array([1.0 / speed], dtype=np.float32)
            if self._acoustic_mode == "split_trt":
                mel = self._run_split_acoustic(
                    x, x_length, noise_scale, length_scale,
                    pool=pool, estimator_ctxs=split_estimator_ctxs,
                )
            else:
                ao = self._acoustic_ort.run(None, {
                    "x": x, "x_length": x_length,
                    "noise_scale": noise_scale, "length_scale": length_scale,
                })
                mel = ao[0]
            encoder_ms = (time.time() - t0) * 1000
            estimator_ms = 0.0
            mel_frames = mel.shape[2]
            if mel.shape[2] > MAX_MEL_FRAMES:
                mel = mel[:, :, :MAX_MEL_FRAMES]
                mel_frames = MAX_MEL_FRAMES
            valid_mel_frames = mel_frames
            mel = _pad_mel_axis(mel, self._min_mel_frames)
            mel_frames = mel.shape[2]
            mask_valid = valid_mel_frames

            def alloc(arr):
                ptr = pool.allocate(arr.nbytes)
                pool.copy_htod(arr, ptr)
                return ptr
            logger.debug("matcha frames: tokens=%d mask=%d mel_frames=%d (~%.2fs)",
                         num_tokens, mask_valid, mel_frames, mel_frames * HOP_LENGTH / SAMPLE_RATE)

            t0 = time.time()
            mel_input = mel[:, :, :mel_frames].astype(np.float32)
            d_mel = alloc(mel_input)
            vocos_ctx.set_tensor_address("mels", d_mel)
            vocos_ctx.set_input_shape("mels", (1, MEL_DIM, mel_frames))

            mag = np.zeros((1, 513, mel_frames), dtype=np.float32)
            out_x = np.zeros((1, 513, mel_frames), dtype=np.float32)
            out_y = np.zeros((1, 513, mel_frames), dtype=np.float32)

            d_mag = pool.allocate(mag.nbytes)
            d_x_out = pool.allocate(out_x.nbytes)
            d_y_out = pool.allocate(out_y.nbytes)

            vocos_ctx.set_tensor_address("mag", d_mag)
            vocos_ctx.set_tensor_address("x", d_x_out)
            vocos_ctx.set_tensor_address("y", d_y_out)
            vocos_ctx.execute_async_v3(pool.stream_handle())
            pool.synchronize()

            pool.copy_dtoh(d_mag, mag)
            pool.copy_dtoh(d_x_out, out_x)
            pool.copy_dtoh(d_y_out, out_y)
            vocos_ms = (time.time() - t0) * 1000

            audio = _istft(mag[0], out_x[0], out_y[0], length=valid_mel_frames * HOP_LENGTH)
            audio = np.clip(audio, -1.0, 1.0)

            elapsed = time.time() - t_start
            duration = len(audio) / SAMPLE_RATE
            wav_bytes = _samples_to_wav(audio.astype(np.float32), SAMPLE_RATE)

            meta = {
                "duration": round(duration, 3),
                "inference_time": round(elapsed, 3),
                "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
                "sample_rate": SAMPLE_RATE,
                "num_tokens": num_tokens,
                "text_ms": round(text_ms, 1),
                "encoder_ms": round(encoder_ms, 1),
                "estimator_ms": round(estimator_ms, 1),
                "vocos_ms": round(vocos_ms, 1),
                "language": detected_language,
            }
            return wav_bytes, meta
        finally:
            if slot is not None:
                try:
                    slot.reset_per_request()
                finally:
                    self._ctx_pool.put(slot)
            else:
                try:
                    pool.free_all()
                except Exception:
                    pass
                try:
                    pool.destroy()
                except Exception:
                    pass

    def generate_streaming(self, text: str, **kwargs):
        """Yield raw PCM int16 chunks (chunk-level streaming)."""
        synth_kwargs = {
            "speaker_id": kwargs.get("speaker_id", kwargs.get("sid")),
            "speed": kwargs.get("speed"),
            "pitch_shift": kwargs.get("pitch_shift", kwargs.get("pitch")),
            "language": kwargs.get("language"),
        }
        wav_bytes, _meta = self.synthesize(text, **synth_kwargs)
        if len(wav_bytes) <= 44:
            return

        chunk_ms = max(10, min(200, int(self._config.stream_chunk_ms)))
        bytes_per_sample = 2
        chunk_bytes = max(
            bytes_per_sample,
            int(SAMPLE_RATE * chunk_ms / 1000) * bytes_per_sample,
        )

        pcm = wav_bytes[44:]
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset:offset + chunk_bytes]
            if chunk:
                yield chunk

    def _run_split_acoustic(
        self,
        x: np.ndarray,
        x_length: np.ndarray,
        noise_scale: np.ndarray,
        length_scale: np.ndarray,
        *,
        pool: "CudaMemoryPool",
        estimator_ctxs: list,
    ) -> np.ndarray:
        """Run Matcha acoustic as ORT encoder + TRT estimator ODE loop."""
        mu, mask, z = self._split_encoder_ort.run(None, {
            "x": x,
            "x_length": x_length,
            "noise_scale": noise_scale,
            "length_scale": length_scale,
        })
        mu = np.ascontiguousarray(mu.astype(np.float32))
        mask = np.ascontiguousarray(mask.astype(np.float32))
        z = np.ascontiguousarray(z.astype(np.float32))
        if z.shape[2] > MAX_MEL_FRAMES:
            mu = mu[:, :, :MAX_MEL_FRAMES]
            mask = mask[:, :, :MAX_MEL_FRAMES]
            z = z[:, :, :MAX_MEL_FRAMES]

        valid_frames = int(np.clip(np.rint(mask.sum()), 1, MAX_MEL_FRAMES))
        mu = _pad_mel_axis(mu, self._min_mel_frames)
        mask = _pad_mel_axis(mask, self._min_mel_frames)
        z = _pad_mel_axis(z, self._min_mel_frames)
        for step in range(N_ODE_STEPS):
            feeds = {"z": z, "mu": mu, "mask": mask}
            velocity = self._run_estimator_trt(step, feeds, pool=pool, ctx=estimator_ctxs[step])
            z = z + ODE_DT * velocity

        mel = z[:, :, :valid_frames] * MEL_SIGMA + MEL_MEAN
        return mel.astype(np.float32)

    def _run_estimator_trt(
        self,
        step: int,
        feeds: dict[str, np.ndarray],
        *,
        pool: "CudaMemoryPool",
        ctx,
    ) -> np.ndarray:

        def alloc_input(name: str, arr: np.ndarray) -> int:
            arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
            ptr = pool.allocate(arr.nbytes)
            pool.copy_htod(arr, ptr)
            ctx.set_tensor_address(name, ptr)
            try:
                ctx.set_input_shape(name, tuple(arr.shape))
            except Exception:
                pass
            return ptr

        for name, arr in feeds.items():
            alloc_input(name, arr)

        frames = int(feeds["z"].shape[2])
        velocity = np.empty((1, MEL_DIM, frames), dtype=np.float32)
        d_velocity = pool.allocate(velocity.nbytes)
        ctx.set_tensor_address("velocity", d_velocity)
        ok = ctx.execute_async_v3(pool.stream_handle())
        if not ok:
            raise RuntimeError("Matcha estimator TRT execute_async_v3 returned False")
        pool.synchronize()
        pool.copy_dtoh(d_velocity, velocity)
        return velocity
