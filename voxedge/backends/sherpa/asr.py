"""Sherpa-onnx ASR backend — voxedge adapter.

adapted from app/backends/cpu/sherpa_asr.py (2026-05-30), dedup after registry
switch.

Streaming Paraformer (zh_en) / Zipformer (en) + offline SenseVoice.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ABCs imported from ``voxedge.backends.base`` (not ``app.core.*``).
  * ALL module-scope ``os.environ.get(...)`` reads replaced by an explicit
    ``SherpaASRConfig`` dataclass injected at construction time. voxedge has
    no module-scope or hardcoded env reads.
  * ``import sherpa_onnx`` / ``import soundfile`` stay lazy so this module
    imports cleanly without the optional ``voxedge[sherpa]`` extra.

Supports: OFFLINE, STREAMING
"""

from __future__ import annotations

import glob
import io
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from voxedge.backends.base import (
    ASRBackend,
    ASRCapability,
    ASRStream,
    TranscriptionResult,
)
from voxedge.engine.concurrency_capability import ConcurrencyCapability

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var                  → SherpaASRConfig field
#   LANGUAGE_MODE                   → language_mode       (default "zh_en")
#   STREAMING_MODEL_DIR             → streaming_model_dir (default per language_mode)
#   STREAMING_ASR_PROVIDER          → streaming_provider  (default "cuda")
#   OFFLINE_ASR_PROVIDER/ASR_PROVIDER → offline_provider  (default = streaming_provider)
#   STREAMING_ASR_NUM_THREADS       → num_threads         (default 4)
#   MODEL_DIR                       → model_root          (default "/opt/models")

_DEFAULT_ASR_DIRS = {
    "zh_en": "/opt/models/paraformer-streaming",
    "en": "/opt/models/zipformer-en",
}


@dataclass
class SherpaASRConfig:
    """Explicit construction-time config for :class:`SherpaASRBackend`.

    Every field default is identical to the production env default; nothing
    reads ``os.environ``. ``streaming_model_dir`` / ``offline_provider``
    default to ``None`` and are resolved in ``__post_init__`` to preserve the
    original language-conditional / fallback semantics exactly.
    """

    language_mode: str = "zh_en"  # "zh_en" or "en"
    streaming_model_dir: Optional[str] = None
    streaming_provider: str = "cuda"
    offline_provider: Optional[str] = None
    num_threads: int = 4
    model_root: str = "/opt/models"

    def __post_init__(self) -> None:
        if self.streaming_model_dir is None:
            self.streaming_model_dir = _DEFAULT_ASR_DIRS.get(
                self.language_mode, _DEFAULT_ASR_DIRS["zh_en"]
            )
        if self.offline_provider is None:
            self.offline_provider = self.streaming_provider


# ---------------------------------------------------------------------------
# BPE merge table (Zipformer / "en" mode only)
# ---------------------------------------------------------------------------

_MERGE_WORDS = {
    "TO DAY": "TODAY",
    "TO NIGHT": "TONIGHT",
    "TO MORROW": "TOMORROW",
    "TO GETHER": "TOGETHER",
    "TO WARD": "TOWARD",
    "TO WARDS": "TOWARDS",
    "SOME THING": "SOMETHING",
    "SOME ONE": "SOMEONE",
    "SOME WHERE": "SOMEWHERE",
    "SOME HOW": "SOMEHOW",
    "SOME TIMES": "SOMETIMES",
    "SOME TIME": "SOMETIME",
    "ANY THING": "ANYTHING",
    "ANY ONE": "ANYONE",
    "ANY WHERE": "ANYWHERE",
    "ANY WAY": "ANYWAY",
    "EVERY THING": "EVERYTHING",
    "EVERY ONE": "EVERYONE",
    "EVERY WHERE": "EVERYWHERE",
    "EVERY BODY": "EVERYBODY",
    "NO THING": "NOTHING",
    "NO WHERE": "NOWHERE",
    "NO BODY": "NOBODY",
    "MY SELF": "MYSELF",
    "YOUR SELF": "YOURSELF",
    "HIM SELF": "HIMSELF",
    "HER SELF": "HERSELF",
    "IT SELF": "ITSELF",
    "OUR SELVES": "OURSELVES",
    "THEM SELVES": "THEMSELVES",
    "MEAN WHILE": "MEANWHILE",
    "AL READY": "ALREADY",
    "AL THOUGH": "ALTHOUGH",
    "AL WAYS": "ALWAYS",
    "AL MOST": "ALMOST",
    "AL TOGETHER": "ALTOGETHER",
    "BREAK FAST": "BREAKFAST",
    "UNDER STAND": "UNDERSTAND",
    "OUT SIDE": "OUTSIDE",
    "IN SIDE": "INSIDE",
    "WITH OUT": "WITHOUT",
    "BE CAUSE": "BECAUSE",
    "BE COME": "BECOME",
    "BE FORE": "BEFORE",
    "BE TWEEN": "BETWEEN",
    "BE HIND": "BEHIND",
}


def _fix_bpe_splits(text: str) -> str:
    """Merge BPE-split words back together (en mode only)."""
    for split, merged in _MERGE_WORDS.items():
        text = text.replace(split, merged)
    return text


# ---------------------------------------------------------------------------
# SherpaASRStream
# ---------------------------------------------------------------------------


class SherpaASRStream(ASRStream):
    """Streaming ASR session backed by a sherpa_onnx OnlineRecognizer."""

    def __init__(self, recognizer, language_mode: str = "zh_en"):
        self._recognizer = recognizer
        self._language_mode = language_mode
        self._stream = recognizer.create_stream()
        self._last_text = ""
        self._is_endpoint = False
        self._cancelled = False
        self._final_text_cache = ""

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if self._cancelled:
            return
        recognizer = self._recognizer

        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if np.abs(samples).max() > 1.0:
            samples = samples / 32768.0

        self._stream.accept_waveform(sample_rate, samples)

        while recognizer.is_ready(self._stream):
            recognizer.decode_stream(self._stream)

        text = recognizer.get_result(self._stream).strip()
        if self._language_mode == "en":
            text = _fix_bpe_splits(text)

        is_endpoint = recognizer.is_endpoint(self._stream)

        self._last_text = text
        self._is_endpoint = is_endpoint

        if is_endpoint:
            # Reset stream for next utterance
            self._stream = self._recognizer.create_stream()

    def finalize(self) -> tuple[str, Optional[str]]:
        if self._cancelled:
            return self._final_text_cache, None
        recognizer = self._recognizer
        stream = self._stream

        if self._language_mode == "en":
            silence = np.zeros(int(16000 * 0.8), dtype=np.float32)
            stream.accept_waveform(16000, silence)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        text = recognizer.get_result(stream).strip()
        if self._language_mode == "en":
            text = _fix_bpe_splits(text)
        # Sherpa backends are single-language (language_mode configured);
        # no per-utterance language detection.
        return text, None

    def get_partial(self) -> tuple[str, bool]:
        text = self._last_text
        is_endpoint = self._is_endpoint
        if is_endpoint:
            self._is_endpoint = False
            self._last_text = ""
        return text, is_endpoint

    def cancel_and_finalize(self) -> None:
        if self._cancelled:
            return
        # Cache current partial as the final text — no extra decode pass,
        # no silence-pad, no native input_finished() (avoids any decode
        # loop in libsherpa).
        self._final_text_cache = self._last_text
        self._cancelled = True


# ---------------------------------------------------------------------------
# SherpaASRBackend
# ---------------------------------------------------------------------------


class SherpaASRBackend(ASRBackend):

    # CPU / ORT model — releasable in-process via del + gc.
    supports_hot_reload = True

    @classmethod
    def concurrency_capability(cls, profile=None):
        """Declare concurrency for desktop/CPU ASR.

        CPU/ORT recognizer objects are independent across streams; the soft
        cap of 4 matches the historical desktop default and bounds CPU thread
        contention.
        """
        return ConcurrencyCapability(
            supports_parallel=True,
            max_concurrent=4,
            is_stateful=True,
            requires_exclusive_device=False,
            scaling_mode="external_managed",
        )

    def __init__(self, config: Optional[SherpaASRConfig] = None):
        self._config = config or SherpaASRConfig()
        self._online_recognizer = None
        self._offline_recognizer = None

    @property
    def name(self) -> str:
        return "sherpa_asr"

    @property
    def capabilities(self) -> set[ASRCapability]:
        caps = set()
        if self._offline_recognizer is not None:
            caps.add(ASRCapability.OFFLINE)
        if self._online_recognizer is not None:
            caps.add(ASRCapability.STREAMING)
        return caps

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._offline_recognizer is not None or self._online_recognizer is not None

    def preload(self) -> None:
        try:
            self._online_recognizer = self._load_online_recognizer()
            logger.info("Sherpa streaming ASR loaded")
        except Exception as e:
            logger.info("Streaming ASR not available: %s", e)
        try:
            self._offline_recognizer = self._load_offline_recognizer()
            logger.info("Sherpa offline ASR loaded")
        except Exception as e:
            logger.info("Offline ASR not available: %s", e)

    def unload(self) -> None:
        """Release online/offline recognizers. Idempotent."""
        if self._online_recognizer is None and self._offline_recognizer is None:
            return
        try:
            self._online_recognizer = None
            self._offline_recognizer = None
            import gc
            gc.collect()
        except Exception:
            logger.exception("SherpaASRBackend.unload failed; continuing")

    def create_stream(self, language: str = "auto") -> ASRStream:
        if self._online_recognizer is None:
            raise RuntimeError("Online recognizer not loaded; call preload() first")
        return SherpaASRStream(self._online_recognizer, self._config.language_mode)

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if self._offline_recognizer is None:
            raise RuntimeError("Offline recognizer not loaded; call preload() first")

        import soundfile as sf

        data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Convert to mono float32
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        # Resample to 16 kHz using numpy linear interpolation (no subprocess/sox)
        if sample_rate != 16000:
            target_len = int(len(data) * 16000 / sample_rate)
            data = np.interp(
                np.linspace(0, len(data) - 1, target_len),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
            sample_rate = 16000

        recognizer = self._offline_recognizer
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, data)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        return TranscriptionResult(text=text, language=language)

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_online_recognizer(self):
        """Load Paraformer (zh_en) or Zipformer (en) streaming recognizer."""
        import sherpa_onnx

        cfg = self._config
        model_dir = cfg.streaming_model_dir
        tokens = os.path.join(model_dir, "tokens.txt")

        if cfg.language_mode == "en":
            encoder = os.path.join(model_dir, "encoder-epoch-99-avg-1-chunk-16-left-128.onnx")
            decoder = os.path.join(model_dir, "decoder-epoch-99-avg-1-chunk-16-left-128.onnx")
            joiner = os.path.join(model_dir, "joiner-epoch-99-avg-1-chunk-16-left-128.onnx")

            logger.info("Loading streaming Zipformer (en) from %s (provider=%s)", model_dir, cfg.streaming_provider)
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                provider=cfg.streaming_provider,
                num_threads=cfg.num_threads,
                enable_endpoint_detection=True,
                rule2_min_trailing_silence=0.6,
            )
            logger.info("Streaming Zipformer (en) loaded.")
        else:
            encoder = os.path.join(model_dir, "encoder.onnx")
            decoder = os.path.join(model_dir, "decoder.onnx")

            logger.info("Loading streaming Paraformer from %s (provider=%s)", model_dir, cfg.streaming_provider)
            recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
                encoder=encoder,
                decoder=decoder,
                tokens=tokens,
                provider=cfg.streaming_provider,
                num_threads=cfg.num_threads,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=0.6,
                rule3_min_utterance_length=20,
            )
            logger.info("Streaming Paraformer loaded.")

        return recognizer

    def _load_offline_recognizer(self):
        """Load SenseVoice offline recognizer."""
        import sherpa_onnx

        cfg = self._config
        model_root = cfg.model_root
        base = os.path.join(model_root, "sensevoice")
        dirs = glob.glob(os.path.join(base, "sherpa-onnx-sense-voice-*"))
        if not dirs:
            dirs = glob.glob(os.path.join(model_root, "sherpa-onnx-sense-voice-*"))
        model_dir = dirs[0] if dirs else base

        model_path = os.path.join(model_dir, "model.int8.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        logger.info("Loading SenseVoice model from %s (provider=%s)", model_dir, cfg.offline_provider)
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            use_itn=True,
            provider=cfg.offline_provider,
        )
        logger.info("SenseVoice model loaded.")
        return recognizer
