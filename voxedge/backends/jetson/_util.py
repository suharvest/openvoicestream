"""Shared, env-free helpers for the voxedge Jetson TRT backends.

adapted from app/backends/jetson/matcha_trt.py + app/core/language.py +
app/core/tts_speakers.py (2026-05-30), dedup after registry switch.

Holds the pieces matcha/kokoro shared via cross-module import in production
(``CudaMemoryPool`` + ``_read_arena_size_bytes``) plus the two tiny app.core
helpers the TTS backends pulled in (``detect_zh_en`` / ``resolve_speaker_kwargs``).

Decoupling notes:
  * ``_read_arena_size_bytes`` is GONE — production read ``OVS_MATCHA_ARENA_SIZE_MB``
    / ``OVS_KOKORO_ARENA_SIZE_MB`` / ``OVS_CUDA_ARENA_SIZE_MB`` from env. voxedge
    passes ``arena_size_mb`` through the backend config dataclass and converts to
    bytes at slot-construction time (see ``arena_size_bytes`` below).
  * ``resolve_speaker_kwargs`` keeps the production ``model_id``-positional
    signature (matcha/kokoro/qwen3 call ``resolve_speaker_kwargs(self.model_id,
    allow_embedding=..., ...)``). ``model_id`` is accepted for compat but unused
    (no registry in voxedge — preset-speaker-name lookups are dropped).
  * The ``cuda`` / ``cudart`` import inside ``CudaMemoryPool`` stays method-local
    so this module imports on a CUDA-less box.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def arena_size_bytes(arena_size_mb: Optional[int], default_mb: int = 16) -> int:
    """Convert a per-backend CUDA arena size (MB) to bytes.

    Env-free replacement for production ``_read_arena_size_bytes`` (which read
    ``OVS_*_ARENA_SIZE_MB`` from os.environ). ``None`` falls back to
    ``default_mb``; sub-1 MB requests clamp to 1 MB. Returns bytes.
    """
    mb = default_mb if arena_size_mb is None else int(arena_size_mb)
    return max(1, mb) * 1024 * 1024


# ── language detection (← app/core/language.py:21-45) ────────────────────────

_AUTO_VALUES = {"", "auto", "detect", "default"}


def normalize_auto_language(language: Optional[str]) -> Optional[str]:
    """Return None when the caller asked the backend to auto-detect."""
    if language is None:
        return None
    lang = str(language).strip()
    if lang.lower() in _AUTO_VALUES:
        return None
    return lang


def detect_zh_en(text: str, language: Optional[str] = None) -> str:
    """Detect the TTS language used by bilingual Matcha-style backends.

    Byte-equivalent to app/core/language.py:detect_zh_en. The zh-en Matcha
    model handles embedded English in Chinese text, so mixed input anchors to
    ``zh`` when any CJK char is present, and returns ``en`` only for pure
    non-CJK input.
    """
    explicit = normalize_auto_language(language)
    if explicit:
        lowered = explicit.lower()
        if lowered in {"chinese", "mandarin", "cn", "zh-cn", "zh_hans"}:
            return "zh"
        if lowered in {"english", "en-us", "en_us", "us"}:
            return "en"
        return explicit

    for ch in text:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
        ):
            return "zh"
    return "en"


def resolve_speaker_kwargs(
    model_id: str,
    *,
    allow_embedding: bool = True,
    **kwargs: object,
) -> dict[str, object]:
    """Env-free, registry-free speaker kwargs resolver.

    Mirrors app/core/tts_speakers.py priority (first wins):
      1. ``speaker_embedding`` — raw float32 bytes (direct voice clone).
      2. ``speaker_id`` — numeric id passed straight through.
      3. ``sid`` — deprecated alias for speaker_id.

    Returns ``{"speaker_embedding": bytes}``, ``{"speaker_id": int}`` or ``{}``.
    ``model_id`` is accepted for signature-compat with the production helper
    but unused (no registry in voxedge — preset speaker-name lookups dropped).
    """
    emb = kwargs.get("speaker_embedding")
    if emb is not None:
        if not allow_embedding:
            raise ValueError(
                f"Model {model_id!r} does not support voice clone embeddings"
            )
        return {"speaker_embedding": emb}

    sid = kwargs.get("speaker_id", kwargs.get("sid"))
    if sid is not None:
        return {"speaker_id": int(sid)}

    return {}


class CudaMemoryPool:
    """CUDA memory pool with optional per-slot arena (sub-allocator).

    Byte-equivalent to app/backends/jetson/matcha_trt.py:CudaMemoryPool — a
    bump-pointer sub-allocator over a single big ``cudaMalloc`` arena, driven
    by one CUDA stream, with a per-call cudaMalloc overflow path and a legacy
    free-list mode when ``arena_size_bytes`` is None.

    The ``cuda`` import stays method-local so this module imports without CUDA.
    """

    @staticmethod
    def _cuda_err(result):
        """Normalize cuda-python return value to cudaError_t."""
        if isinstance(result, tuple):
            return result[0]
        return result

    def __init__(self, arena_size_bytes: int | None = None):
        self._stream = None
        self._allocations: list[int] = []
        self._initialized = False
        self._arena_size: int | None = arena_size_bytes
        self._arena_ptr: int | None = None
        self._arena_offset: int = 0
        self._overflow_allocs: list[int] = []
        self._peak_offset: int = 0
        self._overflow_count: int = 0
        self._overflow_bytes: int = 0

    def _init_cuda(self):
        if self._initialized:
            return
        from cuda import cudart

        err, self._stream = cudart.cudaStreamCreate()
        if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")
        self._initialized = True
        logger.info("CudaMemoryPool initialized with stream %d", int(self._stream))

    def _ensure_arena(self) -> None:
        if self._arena_size is None or self._arena_ptr is not None:
            return
        from cuda import cudart
        err, ptr = cudart.cudaMalloc(self._arena_size)
        if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc(arena {self._arena_size}) failed: {err}")
        self._arena_ptr = int(ptr)
        logger.info(
            "CudaMemoryPool arena ready: ptr=0x%x size=%d (%.1f MB)",
            self._arena_ptr, self._arena_size, self._arena_size / (1024 * 1024),
        )

    def allocate(self, size_bytes: int) -> int:
        self._init_cuda()
        from cuda import cudart
        if self._arena_size is not None:
            self._ensure_arena()
            assert self._arena_ptr is not None
            n_aligned = (size_bytes + 255) & ~255
            if self._arena_offset + n_aligned <= self._arena_size:
                ptr = self._arena_ptr + self._arena_offset
                self._arena_offset += n_aligned
                if self._arena_offset > self._peak_offset:
                    self._peak_offset = self._arena_offset
                return int(ptr)
            err, ptr = cudart.cudaMalloc(size_bytes)
            if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc(overflow {size_bytes}) failed: {err}")
            self._overflow_allocs.append(int(ptr))
            self._overflow_count += 1
            self._overflow_bytes += size_bytes
            return int(ptr)
        err, ptr = cudart.cudaMalloc(size_bytes)
        if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc({size_bytes}) failed: {err}")
        self._allocations.append(ptr)
        return int(ptr)

    def copy_htod(self, host_arr: np.ndarray, dev_ptr: int):
        self._init_cuda()
        from cuda import cudart
        err = cudart.cudaMemcpy(
            dev_ptr, host_arr.ctypes.data, host_arr.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpy H2D failed: {err}")

    def copy_dtoh(self, dev_ptr: int, host_arr: np.ndarray):
        self._init_cuda()
        from cuda import cudart
        err = cudart.cudaMemcpy(
            host_arr.ctypes.data, dev_ptr, host_arr.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpy D2H failed: {err}")

    def synchronize(self):
        if self._stream is not None:
            from cuda import cudart
            err = cudart.cudaStreamSynchronize(self._stream)
            if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaStreamSynchronize failed: {err}")

    def free_all(self):
        if self._arena_size is not None:
            self._arena_offset = 0
            if self._overflow_allocs:
                from cuda import cudart
                for ptr in self._overflow_allocs:
                    cudart.cudaFree(ptr)
                self._overflow_allocs.clear()
            return
        if not self._allocations:
            return
        from cuda import cudart
        for ptr in self._allocations:
            cudart.cudaFree(ptr)
        self._allocations.clear()

    def stream_handle(self) -> int:
        self._init_cuda()
        return int(self._stream)

    def destroy(self) -> None:
        if self._arena_size is not None and (
            self._peak_offset > 0 or self._overflow_count > 0
        ):
            logger.info(
                "CudaMemoryPool destroy: arena=%d B (%.1f MB) peak_used=%d B (%.1f MB %.1f%%) "
                "overflow_count=%d overflow_bytes=%d",
                self._arena_size, self._arena_size / (1024 * 1024),
                self._peak_offset, self._peak_offset / (1024 * 1024),
                100.0 * self._peak_offset / self._arena_size if self._arena_size else 0.0,
                self._overflow_count, self._overflow_bytes,
            )
        self.free_all()
        if self._arena_ptr is not None:
            try:
                from cuda import cudart
                cudart.cudaFree(self._arena_ptr)
            except Exception:
                logger.exception("CudaMemoryPool.destroy arena cudaFree raised; continuing")
            self._arena_ptr = None
            self._arena_offset = 0
        if self._stream is not None:
            try:
                from cuda import cudart
                err = cudart.cudaStreamDestroy(self._stream)
                if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
                    logger.warning("cudaStreamDestroy returned err=%s", err)
            except Exception:
                logger.exception("CudaMemoryPool.destroy stream destroy raised; continuing")
            self._stream = None
        self._initialized = False
