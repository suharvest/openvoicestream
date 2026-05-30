"""RK TTS adapter — voxedge adapter.

adapted from app/backends/rk/tts.py + app/core/rk_*.py (2026-05-30), dedup
after registry switch.

Wraps ``rkvoice_stream.create_tts()`` output. rkvoice-stream's TTSBackend ABC
is smaller than ours (no ``capabilities``, no ``language`` arg, ``speaker_id``
is int with default 0); the adapter forwards everything the voxedge contract
requires and exposes a conservative default capability set.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ABCs imported from ``voxedge.backends.base`` (not ``app.core.*``);
    ``ConcurrencyCapability`` from ``voxedge.engine.concurrency_capability``.
  * The ``model_id`` (was ``OVS_TTS_MODEL_ID`` env, read by the production
    base ``TTSBackend.model_id`` property) is injected via ``RKTTSConfig``.
    voxedge has no module-scope or hardcoded env reads.
  * ``detect_zh_en`` / ``resolve_speaker_kwargs`` reproduced in ``._util``
    (no ``app.*`` import).
  * ``import rkvoice_stream`` stays inside ``__init__`` (lazy) so this module
    imports cleanly without the optional ``voxedge[rk]`` extra / rknn runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from voxedge.backends.base import TTSBackend, TTSCapability
from voxedge.engine.concurrency_capability import ConcurrencyCapability

from ._util import detect_zh_en, resolve_speaker_kwargs

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var                  → RKTTSConfig field
#   OVS_TTS_MODEL_ID                → model_id   (default "rk"; production base
#                                                 fell back to backend name)

# rkvoice-stream's TTSBackend doesn't expose a capability set. The shipped
# backends (matcha_rknn, piper_rknn, qwen3_rknn) all do basic + streaming TTS,
# so declare that as the floor. The wire layer feature-detects optional things
# (voice clone, etc.) via has_capability().
_DEFAULT_RK_TTS_CAPS = {
    TTSCapability.BASIC_TTS,
    TTSCapability.STREAMING,
    TTSCapability.MULTI_LANGUAGE,
}


@dataclass
class RKTTSConfig:
    """Explicit construction-time config for :class:`RKTTSBackend`.

    ``model_id`` was the production ``OVS_TTS_MODEL_ID`` env (used to key the
    speaker registry). The RK adapter is single-speaker so the value is only
    passed to ``resolve_speaker_kwargs`` and otherwise unused; default ``"rk"``
    matches the production fallback to the backend name.
    """

    model_id: str = "rk"


class RKTTSBackend(TTSBackend):
    """Adapter around rkvoice_stream.create_tts(). Backend selection is
    delegated to rkvoice-stream via the ``TTS_BACKEND`` env var (set in the
    rk3576/rk3588 profile / process env); that env is read by rkvoice-stream,
    not by this adapter."""

    @classmethod
    def concurrency_capability(cls, profile=None):
        """Declare concurrency for RK NPU TTS.

        rkvoice-stream owns the NPU lifecycle, serializes through one NPU
        device, and cannot be safely multiplexed across slots. Single-session
        only.
        """
        return ConcurrencyCapability(
            supports_parallel=False,
            max_concurrent=1,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="external_managed",
        )

    def __init__(self, config: Optional[RKTTSConfig] = None):
        from rkvoice_stream import create_tts

        self._config = config or RKTTSConfig()
        self._inner = create_tts()
        # Cache metadata at construction time so post-unload status queries
        # (manager.status() / health checks) don't crash on
        # ``self._inner is None``.
        try:
            self._cached_name = f"rk:{self._inner.name}"
        except Exception:
            self._cached_name = "rk:unknown"
        try:
            self._cached_sample_rate = int(self._inner.get_sample_rate())
        except Exception:
            self._cached_sample_rate = 0

    @property
    def name(self) -> str:
        if self._inner is None:
            return self._cached_name
        return f"rk:{self._inner.name}"

    @property
    def model_id(self) -> str:
        """Model-scope key for speaker tables — injected via config (was
        ``OVS_TTS_MODEL_ID``)."""
        return self._config.model_id

    @property
    def capabilities(self) -> set[TTSCapability]:
        return set(_DEFAULT_RK_TTS_CAPS)

    @property
    def sample_rate(self) -> int:
        if self._inner is None:
            return self._cached_sample_rate
        return self._inner.get_sample_rate()

    def is_ready(self) -> bool:
        if self._inner is None:
            return False
        return self._inner.is_ready()

    def preload(self) -> None:
        if self._inner is None:
            raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
        self._inner.preload()

    def unload(self) -> None:
        """Drop the rkvoice-stream inner backend handle. Idempotent.

        ``supports_hot_reload`` stays False — the NPU is held by the
        rkvoice-stream backend and a deeper teardown contract belongs to that
        repo. Provide a best-effort release here so future support can plug in
        without touching the manager.
        """
        if self._inner is None:
            return
        try:
            self._inner = None
            import gc
            gc.collect()
        except Exception:
            logger.exception("RKTTSBackend.unload failed; continuing")

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if self._inner is None:
            raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
        voice = resolve_speaker_kwargs(
            self.model_id, allow_embedding=False, speaker_id=speaker_id, **kwargs
        )
        sid = voice.get("speaker_id", 0)
        # rkvoice-stream's synthesize() doesn't take `language`; pass it
        # through kwargs only when explicitly set so backends that ignore it
        # are unaffected.
        language = detect_zh_en(text, language)
        kwargs.setdefault("language", language)
        return self._inner.synthesize(
            text=text,
            speaker_id=sid,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        )

    def generate_streaming(self, text: str, **kwargs):
        """Bridge our base-class generate_streaming() to rkvoice-stream's
        synthesize_stream().

        rkvoice-stream yields ``(audio, metadata)`` tuples where ``audio`` is
        either float32 [-1,1], int16 PCM, or raw bytes. The wire layer
        (`/tts/stream`) expects int16 PCM bytes per chunk, so coerce here —
        starlette's StreamingResponse calls ``.encode()`` on non-bytes items
        and explodes on tuples (`'tuple' object has no attribute 'encode'`).
        """
        if self._inner is None:
            raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
        voice = resolve_speaker_kwargs(self.model_id, allow_embedding=False, **kwargs)
        speaker_id = voice.get("speaker_id", 0)
        kwargs.pop("speaker_id", None)
        speed = kwargs.pop("speed", None)
        pitch_shift = kwargs.pop("pitch_shift", None)
        language = detect_zh_en(text, kwargs.pop("language", None))
        kwargs.setdefault("language", language)
        for item in self._inner.synthesize_stream(
            text=text,
            speaker_id=speaker_id,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        ):
            audio = item[0] if isinstance(item, tuple) else item
            if audio is None:
                continue
            if isinstance(audio, (bytes, bytearray)):
                if len(audio) == 0:
                    continue
                yield bytes(audio)
                continue
            if isinstance(audio, np.ndarray):
                if audio.size == 0:
                    continue
                if audio.dtype == np.int16:
                    yield audio.tobytes()
                else:
                    a = np.asarray(audio, dtype=np.float32)
                    a = np.clip(a, -1.0, 1.0)
                    yield (a * 32767.0).astype(np.int16).tobytes()
                continue
            # Unknown payload — skip rather than poison the stream.
            continue

    def synthesize_stream(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        if self._inner is None:
            raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
        language = detect_zh_en(text, language)
        kwargs.setdefault("language", language)
        yield from self._inner.synthesize_stream(
            text=text,
            speaker_id=speaker_id if speaker_id is not None else 0,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        )
