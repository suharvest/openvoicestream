"""Sherpa-onnx TTS backend (Matcha / Kokoro) — voxedge adapter.

adapted from app/backends/cpu/sherpa.py (2026-05-30), dedup after registry
switch.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ABCs imported from ``voxedge.backends.base`` (not ``app.core.*``).
  * Helper functions reproduced in ``._util`` (no ``app.*`` import).
  * ALL ~8 module-scope ``os.environ.get(...)`` reads replaced by an explicit
    ``SherpaTTSConfig`` dataclass injected at construction time. voxedge has
    no module-scope or hardcoded env reads (memory
    trt_edge_llm_tts_env_staleness: module-scope env breaks hot reload).
  * ``import sherpa_onnx`` stays lazy (inside methods) so this module imports
    cleanly on a machine without the optional ``voxedge[sherpa]`` extra.

Supports: BASIC_TTS, STREAMING, MULTI_LANGUAGE (zh_en) / MULTI_SPEAKER (en)
"""

from __future__ import annotations

import io
import logging
import os
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from voxedge.backends.base import TTSBackend, TTSCapability
from voxedge.engine.concurrency_capability import ConcurrencyCapability

from ._util import detect_zh_en, resolve_speaker_kwargs

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var                  → SherpaTTSConfig field
#   LANGUAGE_MODE                   → language_mode      (default "zh_en")
#   SHERPA_TTS_MODEL_DIR/TTS_MODEL_DIR → model_dir       (default per language_mode)
#   TTS_PROVIDER                    → provider           (default "cuda")
#   TTS_NUM_THREADS                 → num_threads        (default 4)
#   TTS_DEFAULT_SID                 → default_speaker_id (default per language_mode)
#   TTS_DEFAULT_SPEED               → default_speed      (default 1.0)
#   TTS_PITCH_SHIFT                 → pitch_shift        (default 0.0)
#   OVS_TTS_MODEL_ID                → model_id           (default "sherpa")

_DEFAULT_TTS_DIRS = {
    "zh_en": "/opt/models/matcha-icefall-zh-en",
    "en": "/opt/models/kokoro-multi-lang-v1_0",
}
_DEFAULT_SIDS = {"zh_en": 0, "en": 52}


@dataclass
class SherpaTTSConfig:
    """Explicit construction-time config for :class:`SherpaTTSBackend`.

    Every field has a default identical to the production env default; nothing
    here reads ``os.environ``. ``model_dir`` / ``default_speaker_id`` default
    to ``None`` and are resolved from ``language_mode`` in ``__post_init__`` so
    that the language-conditional defaults match the old code exactly.
    """

    language_mode: str = "zh_en"  # "zh_en" or "en"
    model_dir: Optional[str] = None
    provider: str = "cuda"
    num_threads: int = 4
    default_speaker_id: Optional[int] = None
    default_speed: float = 1.0
    pitch_shift: float = 0.0
    model_id: str = "sherpa"

    def __post_init__(self) -> None:
        if self.model_dir is None:
            self.model_dir = _DEFAULT_TTS_DIRS.get(
                self.language_mode, _DEFAULT_TTS_DIRS["zh_en"]
            )
        if self.default_speaker_id is None:
            self.default_speaker_id = _DEFAULT_SIDS.get(self.language_mode, 0)


def _pitch_shift_samples(samples: list, semitones: float) -> list:
    if semitones == 0:
        return samples
    ratio = 2 ** (semitones / 12)
    arr = np.array(samples, dtype=np.float32)
    new_len = int(len(arr) / ratio)
    indices = np.linspace(0, len(arr) - 1, new_len)
    return np.interp(indices, np.arange(len(arr)), arr).tolist()


def _samples_to_wav(samples: list, sample_rate: int) -> bytes:
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
    arr = np.array(samples, dtype=np.float32)
    np.clip(arr, -1.0, 1.0, out=arr)
    buf.write((arr * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class SherpaTTSBackend(TTSBackend):
    """Sherpa-onnx TTS (Matcha Chinese+English or Kokoro English)."""

    # CPU / ORT model — Python del + gc actually frees memory here, so this
    # backend can be hot-reloaded in-process.
    supports_hot_reload = True

    @classmethod
    def concurrency_capability(cls, profile=None):
        """Declare concurrency for desktop/CPU TTS.

        CPU/ORT is stateless at the device level (no GPU/NPU lock), so multiple
        synthesize() calls can run in parallel up to a soft cap that bounds CPU
        thread contention. ``max_concurrent=4`` mirrors the historical desktop
        default.
        """
        return ConcurrencyCapability(
            supports_parallel=True,
            max_concurrent=4,
            is_stateful=True,
            requires_exclusive_device=False,
            scaling_mode="external_managed",
        )

    def __init__(self, config: Optional[SherpaTTSConfig] = None):
        self._config = config or SherpaTTSConfig()
        self._tts = None
        self._ready = False

    @property
    def name(self) -> str:
        return "sherpa"

    @property
    def model_id(self) -> str:
        """Model-scope key — injected via config (was ``OVS_TTS_MODEL_ID``)."""
        return self._config.model_id

    @property
    def capabilities(self) -> set[TTSCapability]:
        caps = {TTSCapability.BASIC_TTS, TTSCapability.STREAMING}
        if self._config.language_mode == "zh_en":
            caps.add(TTSCapability.MULTI_LANGUAGE)
        if self._config.language_mode == "en":
            caps.add(TTSCapability.MULTI_SPEAKER)
        return caps

    @property
    def sample_rate(self) -> int:
        if self._tts:
            return self._tts.sample_rate
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        self._load_model()
        self._warmup()
        self._ready = True

    def _load_model(self):
        import sherpa_onnx

        cfg = self._config
        model_dir = cfg.model_dir
        if cfg.language_mode == "en":
            logger.info("Loading Kokoro TTS from %s", model_dir)
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=os.path.join(model_dir, "model.onnx"),
                        voices=os.path.join(model_dir, "voices.bin"),
                        tokens=os.path.join(model_dir, "tokens.txt"),
                        lexicon=os.path.join(model_dir, "lexicon-us-en.txt"),
                        data_dir=os.path.join(model_dir, "espeak-ng-data"),
                        dict_dir=model_dir,
                    ),
                    provider=cfg.provider,
                    num_threads=cfg.num_threads,
                ),
            )
        else:
            logger.info("Loading Matcha TTS from %s", model_dir)
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                        acoustic_model=os.path.join(model_dir, "model-steps-3.onnx"),
                        vocoder=os.path.join(model_dir, "vocos-16khz-univ.onnx"),
                        lexicon=os.path.join(model_dir, "lexicon.txt"),
                        tokens=os.path.join(model_dir, "tokens.txt"),
                        data_dir=os.path.join(model_dir, "espeak-ng-data"),
                        dict_dir=model_dir,
                    ),
                    provider=cfg.provider,
                    num_threads=cfg.num_threads,
                ),
            )

        self._tts = sherpa_onnx.OfflineTts(config)
        logger.info("TTS loaded (sample_rate=%d).", self._tts.sample_rate)

    def _warmup(self):
        cfg = self._config
        if cfg.language_mode == "en":
            texts = ["OK", "Sure.", "Hello, nice to meet you."]
        else:
            texts = ["好", "你好", "今天天气不错", "OK", "Hello."]
        n_rounds = 5 if cfg.provider == "cuda" else 1
        start = time.time()
        for _ in range(n_rounds):
            for t in texts:
                self._tts.generate(t, sid=cfg.default_speaker_id, speed=1.0)
        logger.info("TTS warmup: %.1fs", time.time() - start)

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        cfg = self._config
        voice = resolve_speaker_kwargs(allow_embedding=False, speaker_id=speaker_id, **kwargs)
        speaker_id = voice.get("speaker_id", cfg.default_speaker_id)
        if speed is None:
            speed = cfg.default_speed
        if pitch_shift is None:
            pitch_shift = cfg.pitch_shift
        detected_language = detect_zh_en(text, language)

        start = time.time()
        audio = self._tts.generate(text, sid=speaker_id, speed=speed)
        if not audio.samples or len(audio.samples) == 0:
            logger.warning("Speaker %d empty, fallback to %d", speaker_id, cfg.default_speaker_id)
            audio = self._tts.generate(text, sid=cfg.default_speaker_id, speed=speed)
        elapsed = time.time() - start

        samples = _pitch_shift_samples(audio.samples, pitch_shift)
        duration = len(samples) / audio.sample_rate
        wav_bytes = _samples_to_wav(samples, audio.sample_rate)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
            "sample_rate": audio.sample_rate,
            "language": detected_language,
        }
        return wav_bytes, meta

    def unload(self) -> None:
        """Release the sherpa-onnx OfflineTts handle. Idempotent."""
        if not self._ready and self._tts is None:
            return
        try:
            self._tts = None
            import gc
            gc.collect()
        except Exception:
            logger.exception("SherpaTTSBackend.unload failed; continuing")
        finally:
            self._ready = False

    def generate_streaming(self, text: str, **kwargs):
        """Yield PCM int16 chunks as the vocoder produces them (true streaming)."""
        import queue
        import threading

        cfg = self._config
        voice = resolve_speaker_kwargs(allow_embedding=False, **kwargs)
        sid = voice.get("speaker_id", cfg.default_speaker_id)
        speed = kwargs.get("speed")
        if speed is None:
            speed = cfg.default_speed
        pitch = kwargs.get("pitch_shift")
        if pitch is None:
            pitch = cfg.pitch_shift
        language = detect_zh_en(text, kwargs.get("language"))

        audio_queue: queue.Queue[bytes | None] = queue.Queue()

        def callback(samples, progress):
            shifted = _pitch_shift_samples(samples, pitch)
            arr = np.array(shifted, dtype=np.float32)
            np.clip(arr, -1.0, 1.0, out=arr)
            pcm = (arr * 32767).astype(np.int16).tobytes()
            audio_queue.put(pcm)
            return 1

        def runner():
            try:
                self._tts.generate(text, sid=sid, speed=speed, callback=callback)
            except Exception as e:
                logger.exception("TTS streaming generate failed: %s", e)
            finally:
                audio_queue.put(None)

        threading.Thread(target=runner, daemon=True).start()

        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            yield chunk
