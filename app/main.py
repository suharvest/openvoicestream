"""FastAPI speech service: ASR + TTS with pluggable backends."""

from __future__ import annotations

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel
class _WSHandle:
    """Lightweight WS-session handle for BackendManager.register_ws().

    Replaces ``types.SimpleNamespace`` here because Python 3.10's
    SimpleNamespace lacks ``__weakref__`` (added in 3.11), and
    BackendManager._ws_handles is a WeakSet. The Jetson image still
    ships Python 3.10.12, so any handle stored in a WeakSet must be a
    plain class.
    """
    __slots__ = ("websocket", "task", "__weakref__")

    def __init__(self, websocket, task):
        self.websocket = websocket
        self.task = task
from typing import Literal, Optional

# Week 2: configure logging (JSON or text) from OVS_LOG_FORMAT before
# any other module emits a startup log. Falls back gracefully to the
# legacy text format if the env var is unset/invalid.
from app.core.logging_config import (  # noqa: E402  (must precede app creation)
    setup_logging,
    set_request_context,
    reset_request_context,
    request_id_from_headers,
    generate_request_id,
    mask_url_query,
)

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Jetson Speech Service", version="2.0.0")


# Week 2: HTTP middleware injects/propagates X-Request-ID and stores it
# in the request_id contextvar so every log line from the handler can
# include it. Never reads request body. Probes (/livez /readyz /health
# /metrics) are NOT skipped because we still want the response header.
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    inbound = request_id_from_headers(request.headers)
    request_id = inbound or generate_request_id()
    tokens = set_request_context(request_id=request_id)
    try:
        try:
            response = await call_next(request)
        except Exception:
            # Make sure the request_id is visible in the exception log
            # before we propagate so operators can correlate.
            logger.exception(
                "unhandled exception in request: %s",
                mask_url_query(str(request.url)),
            )
            raise
        # Add the response header. Streaming responses are passed through
        # unchanged; the generator captures its own context.
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        reset_request_context(tokens)


# Week 1 production hardening: optional API-key auth for public voice
# endpoints. Disabled when OVS_API_KEYS is unset/empty. See
# docs/specs/prod-hardening-week1.md Deliverable 1.
def _require_api_key(request: Request) -> None:
    from app.core.api_auth import check_http
    check_http(request)


class TTSRequest(BaseModel):
    text: str
    sid: int | None = None
    speaker_id: int | None = None
    speaker_embedding_b64: str | None = None
    speed: float | None = None
    pitch: float | None = None
    language: str | None = None


class CloneRequest(BaseModel):
    text: str
    speaker_embedding_b64: str  # base64-encoded speaker embedding
    language: str | None = None


class CloneStreamRequest(BaseModel):
    text: str
    speaker_embedding_b64: str  # base64-encoded speaker embedding
    language: str | None = None
    streaming_profile: str | None = None
    first_chunk_frames: int | None = None
    chunk_frames: int | None = None


_asr_backend = None

# Dedicated single-thread executor for streaming TTS (T3 fix).
# Default asyncio executor spawns multiple worker threads; each new thread
# observes a cold CUDA per-thread context for the C++ TRT engine, which
# inflates streaming prefill from ~16ms (warm) to 33-122ms (cold) under
# any concurrency. Pinning streaming TTS to a single worker keeps the
# CUDA context warm across all requests.
_tts_stream_executor: ThreadPoolExecutor | None = None

# Dedicated single-thread executor for streaming ASR.  Without this,
# concurrent WS connections dispatch ASR work to separate IO threads,
# each racing on _ASR_CUDA_STREAM (process-global singleton) leading to
# CUDA Graph capture failures.  One worker serialises all ASR ops on a
# consistent thread with a warm CUDA context.
_asr_executor: ThreadPoolExecutor | None = None


class _VoiceCloneUnsupportedError(Exception):
    """Raised by ``_request_voice_kwargs`` when the request carries a
    ``speaker_embedding_b64`` but the active backend explicitly disables
    voice cloning. Callers translate this into a 400 JSON response with
    the unified capability payload (Bug 3 fix — /tts and /tts/stream).
    """

    def __init__(self, backend) -> None:  # type: ignore[no-untyped-def]
        self.backend = backend
        backend_name = getattr(backend, "name", "tts")
        super().__init__(
            f"Current TTS backend ({backend_name}) does not support voice cloning"
        )


def _voice_clone_unsupported_payload(backend) -> dict:  # type: ignore[no-untyped-def]
    backend_name = getattr(backend, "name", "tts")
    msg = (
        f"Current TTS backend ({backend_name}) does not support voice "
        "cloning. Use a built-in speaker_id via /tts, or switch to a "
        "clone-capable backend."
    )
    return {
        "error": msg,
        "required_capability": "voice_clone",
        "backend": backend_name,
        "supports_voice_cloning": False,
    }


def _request_voice_kwargs(req: TTSRequest, *, backend=None) -> dict:
    """Resolve TTS kwargs for one synth call.

    Mixes (in priority order):
      request payload > runtime overrides > speaker-table default

    Returns a dict combining the backend-specific speaker kwargs (from
    :func:`speaker_kwargs_for_id`) plus ``speed`` / ``pitch_shift`` when the
    merge (request payload + runtime overrides) yields a value. Callers
    should ``**``-spread the result into ``synthesize`` / ``generate_streaming``
    and must NOT additionally pass ``speed`` / ``pitch_shift`` from the raw
    request, or runtime overrides will be silently discarded (FIX_2).

    ``backend`` is the live TTS backend (from BackendManager.acquire()); when
    omitted we fall back to ``tts_service`` / env so the helper still works
    if called outside an acquire() scope.
    """
    from app.core.tts_speakers import speaker_kwargs_for_id
    from app.core.tts_runtime import merge_tts_request_kwargs

    speaker_id = req.speaker_id if req.speaker_id is not None else req.sid
    if speaker_id is not None and req.speaker_embedding_b64:
        raise ValueError("speaker_id and speaker_embedding_b64 cannot be used together")
    if req.speaker_embedding_b64:
        # Bug 3 fix: pre-response capability gate for /tts and /tts/stream.
        # Without this, /tts on a non-clone backend (CustomVoice) raises
        # NotImplementedError → FastAPI 500; /tts/stream is worse because
        # response headers are already on the wire when the worker thread
        # raises, leaving the client with a half-written stream.
        if backend is not None and getattr(backend, "supports_voice_cloning", True) is False:
            from app.core.tts_backend import TTSCapability
            if not backend.has_capability(TTSCapability.VOICE_CLONE):
                raise _VoiceCloneUnsupportedError(backend)
        try:
            return {"speaker_embedding": base64.b64decode(req.speaker_embedding_b64)}
        except Exception as exc:
            raise ValueError("Invalid base64 speaker_embedding_b64") from exc

    if backend is not None:
        model_id = backend.model_id
    else:
        from app.core import tts_service
        if tts_service.is_ready():
            model_id = tts_service.get_backend().model_id
        else:
            model_id = os.environ.get("OVS_TTS_MODEL_ID") or "qwen3-tts"

    merged = merge_tts_request_kwargs(
        request_speaker_id=speaker_id,
        request_speed=req.speed,
        request_pitch_shift=getattr(req, "pitch", None),
        model_id=model_id,
    )
    # Translate merged speaker_id into backend-specific kwargs (speaker_id
    # for preset, speaker_embedding for an embedding-typed entry, etc.).
    out: dict = speaker_kwargs_for_id(merged["speaker_id"], model_id)
    # FIX_2: thread merged speed / pitch_shift through so PATCH /admin/tts/runtime
    # actually takes effect. Only include keys that resolved to a non-None value
    # so backends keep using their intrinsic defaults when nothing was set.
    if merged.get("speed") is not None:
        out["speed"] = merged["speed"]
    if merged.get("pitch_shift") is not None:
        out["pitch_shift"] = merged["pitch_shift"]
    return out


def _get_asr_backend():
    return _asr_backend


_tts_lazy_start_lock = None  # asyncio.Lock; created on first use


def _try_tts_manager():
    """Return the TTS BackendManager if it is initialised+ready, else None.

    Kept for ASR-only profiles where TTS isn't wired at all. For LAZY_TTS the
    ``_ensure_tts_manager_started`` coroutine should be awaited first so the
    manager is in READY state before this is consulted.
    """
    try:
        from app.core.backend_manager import tts_manager  # local import; PR3 module
        mgr = tts_manager()
    except RuntimeError:
        return None
    return mgr if mgr.is_ready() else None


async def _ensure_tts_manager_started():
    """FIX_3 / FIX_3_completion: drive the TTS BackendManager to READY.

    Return values:
      * ``mgr`` — manager exists and is READY (caller must use ``acquire``).
      * ``None`` — manager was never installed (ASR-only profile, or
        ``init_backend_managers`` wasn't called). Caller may fall back to the
        legacy ``tts_service.synthesize`` path.

    Raises ``HTTPException(503)`` when the manager exists but is *not*
    serviceable — FAILED, DRAINING, RELOADING, or when ``start()`` fails to
    bring an INIT-state manager to READY. This is intentional: a FAILED
    manager indicates a configuration / resource problem and silently
    falling back to legacy ``tts_service`` would bypass the drain contract
    and mask the failure from operators.
    """
    import asyncio as _asyncio
    global _tts_lazy_start_lock
    try:
        from app.core.backend_manager import tts_manager, BackendState
        mgr = tts_manager()
    except RuntimeError:
        # Manager singleton never installed → legacy fallback is OK.
        return None

    if mgr.is_ready():
        return mgr

    # FAILED is non-recoverable here — surface as 503, never fall through to
    # legacy tts_service (which would skip drain / hide the failure).
    if mgr.state == BackendState.FAILED:
        raise HTTPException(
            status_code=503,
            detail={"error": "tts_manager_failed", "state": "failed"},
        )

    # DRAINING / RELOADING are transient — surface 503 so the client retries.
    if mgr.state != BackendState.INIT:
        raise HTTPException(
            status_code=503,
            detail={"error": "tts_manager_unavailable", "state": mgr.state.value},
        )

    if _tts_lazy_start_lock is None:
        _tts_lazy_start_lock = _asyncio.Lock()
    async with _tts_lazy_start_lock:
        if mgr.is_ready():
            return mgr
        if mgr.state == BackendState.FAILED:
            raise HTTPException(
                status_code=503,
                detail={"error": "tts_manager_failed", "state": "failed"},
            )
        if mgr.state != BackendState.INIT:
            raise HTTPException(
                status_code=503,
                detail={"error": "tts_manager_unavailable", "state": mgr.state.value},
            )
        try:
            await mgr.start()
        except Exception as exc:
            logger.exception("lazy TTS manager.start() failed")
            # start() failure flips state to FAILED. Surface 503 instead of
            # silently falling back to legacy tts_service.
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "tts_manager_start_failed",
                    "state": mgr.state.value,
                    "message": str(exc),
                },
            ) from exc
    if mgr.is_ready():
        return mgr
    # Defensive: start() returned without exception but state isn't READY.
    raise HTTPException(
        status_code=503,
        detail={"error": "tts_manager_unavailable", "state": mgr.state.value},
    )


def _try_asr_manager():
    """Return the ASR BackendManager if it is initialised+ready, else None."""
    try:
        from app.core.backend_manager import asr_manager
        mgr = asr_manager()
    except RuntimeError:
        return None
    return mgr if mgr.is_ready() else None


def _resolve_tts_stream_max_workers() -> tuple[int, str | None, str]:
    """Resolve the TTS stream executor `max_workers` from env + the
    currently-loaded backend's concurrency_capability. Returns
    `(workers, backend_name_or_None, source_label)`.

    Spec docs/specs/concurrency-capability-framework.md §5: executor cap
    and the WorkerIO semaphore must derive from the same capability
    source. Resolution precedence:

      1. backend-specific env (OVS_TTS_STREAM_MAX_WORKERS_{KOKORO,MATCHA,
         QWEN3,MOSS}) if matching, then global OVS_TTS_STREAM_MAX_WORKERS
         — env values are CLAMPED to the backend ceiling.
      2. backend.concurrency_capability(profile).max_concurrent (None ->
         legacy global default of 2 since the executor needs a finite N).
      3. legacy default ``2``.

    Codex Week 3 BLOCKER 4 lazy-startup workflow is preserved: the
    one-shot post-ready refresh still applies because we re-read both
    backend name and capability each call until the flag flips.
    """
    try:
        from app.core import tts_service as _tts_svc
        backend_name = (
            (_tts_svc.backend_name() or "").lower()
            if _tts_svc.is_ready()
            else ""
        )
    except Exception:
        backend_name = ""

    # Delegate to the shared resolver (spec §5). Profile lookup +
    # capability fallback + env clamp now live in one place — see
    # ``app.core.capability_resolver``.
    try:
        from app.core.profile_loader import current_profile
        from app.core.capability_resolver import resolve_executor_for_tts
        prof = current_profile()
    except Exception:
        prof = {}
        from app.core.capability_resolver import resolve_executor_for_tts

    workers, name, src = resolve_executor_for_tts(
        profile=prof, tts_backend_name=backend_name or None,
    )
    # Surface the legacy WARNING for env > ceiling clamps. The resolver
    # already produced the warning text; we reproduce it here at WARNING
    # level to preserve the prior log surface for ops dashboards.
    try:
        from app.core.capability_resolver import resolve as _resolve_cap
        snapshot = _resolve_cap(
            profile=prof, tts_backend_name=backend_name or None,
        )
        for w in snapshot.clamp_warnings:
            if w.startswith("TTS executor:"):
                logger.warning(w)
    except Exception:
        pass
    return workers, name, src


# Tracks whether the cached executor was created BEFORE the TTS backend
# name could be resolved. If True, the first /tts/stream call that lands
# with a ready backend will refresh the executor so backend-specific
# OVS_TTS_STREAM_MAX_WORKERS_* envs actually take effect.
_tts_stream_executor_resolved_backend: bool = False


def _get_tts_stream_executor() -> ThreadPoolExecutor:
    global _tts_stream_executor, _tts_stream_executor_resolved_backend
    # Codex Week 3 BLOCKER 4: if the cached executor was built before the
    # TTS backend was identifiable, try once more now that backend_name()
    # may resolve. This lets backend-specific env overrides apply even
    # when the executor was lazily touched during early startup.
    if _tts_stream_executor is not None and not _tts_stream_executor_resolved_backend:
        try:
            from app.core import tts_service as _tts_svc
            backend_ready_now = _tts_svc.is_ready() and bool(_tts_svc.backend_name())
        except Exception:
            backend_ready_now = False
        if backend_ready_now:
            new_workers, backend_name, env_used = _resolve_tts_stream_max_workers()
            if new_workers != _tts_stream_executor._max_workers:
                logger.info(
                    "TTS executor: refreshing max_workers %d → %d "
                    "(backend=%s, env=%s) after TTS service became ready",
                    _tts_stream_executor._max_workers,
                    new_workers, backend_name, env_used,
                )
                old = _tts_stream_executor
                _tts_stream_executor = ThreadPoolExecutor(
                    max_workers=new_workers,
                    thread_name_prefix="tts-stream",
                )
                # Best-effort shutdown of the old executor without
                # blocking; in-flight tasks finish naturally.
                try:
                    old.shutdown(wait=False, cancel_futures=False)
                except Exception:
                    pass
            else:
                logger.info(
                    "TTS executor: backend=%s resolved post-init "
                    "(env=%s, max_workers=%d, no change needed)",
                    backend_name, env_used, new_workers,
                )
            _tts_stream_executor_resolved_backend = True
    if _tts_stream_executor is None:
        # Phase 3b-B-4 part-4 INVESTIGATION RESULT: lifting max_workers above
        # 1 exposes a deeper bug in the C++ stateful Code2WavRunner reset
        # path. Two concurrent /tts/stream requests cause:
        #
        #   CUDA runtime error in cudaMemsetAsync(state.read.rawPointer(), ...)
        #   an illegal memory access was encountered
        #
        # The C++ engine slot pools (Phase 3b-B-1) + worker thread-dispatch
        # (Phase 3b-B-2) + per-slot Code2Wav (Phase 3b-B-4 part-2 commit
        # `5e1323f`) all carry the right per-slot data, but per-slot
        # StatefulCode2WavRunner state buffer initialization isn't actually
        # multi-slot safe yet — that's the next bottleneck to fix. Until
        # that's resolved, keep this serializing at the HTTP layer so the
        # worker never sees two in-flight requests simultaneously. The
        # `OVS_TTS_WORKER_CONCURRENCY` env is wired all the way down (engine
        # pools sized to it, worker dispatcher uses it, _WorkerIO semaphore
        # picks it up) but its only practical effect today is making the
        # cold-start eager-init less of a spike when the cap eventually
        # rises.
        # Phase B C1+C2+C3+C5 landed (fork commits e1abd90, fff8a38,
        # 99cf14a) — per-request locals for the talker + CP scratch
        # tensors plus C5 Code2Wav worker mutex. Real N=2 throughput
        # IS achievable on Orin NX: empirically 1.3-1.5× single-client
        # TTFA on the first N=2 request-pair after restart (within the
        # ≤ 1.5× spec gate). Audio MD5 byte-identical baseline at N=1.
        # Caveat: sustained N=2 (3+ consecutive bursts) still shows
        # cumulative state corruption from residual shared state
        # (mSamplingWorkspace and/or TRT context sharing inside the
        # CodePredictor engine slot pool, not yet traced). Default
        # max_workers=2 lets the optimization apply; if you observe
        # CUDA errors in production, set OVS_TTS_STREAM_MAX_WORKERS=1
        # to fall back to the C5b runtime-mutex stability gate.
        # Week 3 spec §D1: backend-specific override env > global > default.
        # Lets ops force one backend single-slot without muting the others.
        max_workers, backend_name, env_used = _resolve_tts_stream_max_workers()
        if backend_name:
            logger.info(
                "TTS executor: backend=%s using %s=%d",
                backend_name, env_used, max_workers,
            )
            _tts_stream_executor_resolved_backend = True
        else:
            logger.info(
                "TTS executor: backend not yet resolved at init; using %s=%d "
                "(will refresh once TTS service is ready)",
                env_used, max_workers,
            )
        _tts_stream_executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tts-stream",
        )
    return _tts_stream_executor


def _resolve_asr_max_workers() -> tuple[int, str]:
    """Resolve the ASR executor ``max_workers`` from the active ASR backend's
    ``concurrency_capability`` (slot-pool ceiling N). Returns
    ``(workers, source_label)``.

    Same capability source as the C++ worker's ``--max_slots N`` + the
    backend's ``WorkerIO`` Semaphore(N) (see
    ``trt_edge_llm_asr.concurrency_capability``): precedence is env
    ``EDGE_LLM_ASR_MAX_CONCURRENT`` → profile ``asr_max_slots`` → default 1.
    Reading it through the unified ``capability_resolver`` keeps the executor
    cap, the WorkerIO semaphore, and the session-limiter ceiling derived from
    one place (spec docs/specs/concurrency-capability-framework.md §5).

    Default 1 → byte-equivalent to the legacy single-thread executor, so an
    unset ``asr_max_slots`` is fully backward compatible (no behavior change).
    """
    try:
        from app.core.profile_loader import current_profile
        from app.core.capability_resolver import resolve as _resolve_cap
        prof = current_profile()
    except Exception:
        return 1, "default(no-profile)"
    try:
        snapshot = _resolve_cap(profile=prof)
        n = snapshot.asr_cap.max_concurrent
    except Exception:
        return 1, "default(resolve-failed)"
    if not isinstance(n, int) or n < 1:
        # max_concurrent=None (capability undeclared) → legacy single slot.
        return 1, "default(asr_cap=None)"
    return n, "asr_cap.max_concurrent"


# Tracks whether the cached ASR executor was sized BEFORE the ASR backend /
# profile could be resolved. If True, the first /asr/stream (or /v2v) call
# that lands with a resolvable capability re-sizes the executor once so
# ``asr_max_slots`` / ``EDGE_LLM_ASR_MAX_CONCURRENT`` actually take effect even
# when the executor was lazily touched during early startup (mirrors the TTS
# executor's BLOCKER-4 post-ready refresh).
_asr_executor_resolved_capability: bool = False


def _get_asr_executor() -> ThreadPoolExecutor:
    global _asr_executor, _asr_executor_resolved_capability
    # One-shot post-ready refresh: if the cached executor was built before the
    # ASR capability was resolvable, re-size it now (the profile / backend may
    # have become available since). Idempotent — flips the flag on first
    # successful resolution.
    if _asr_executor is not None and not _asr_executor_resolved_capability:
        new_workers, src = _resolve_asr_max_workers()
        if not src.startswith("default("):
            _asr_executor_resolved_capability = True
            if new_workers != _asr_executor._max_workers:
                logger.info(
                    "ASR executor: refreshing max_workers %d → %d (source=%s) "
                    "after capability became resolvable",
                    _asr_executor._max_workers, new_workers, src,
                )
                old = _asr_executor
                _asr_executor = ThreadPoolExecutor(
                    max_workers=new_workers, thread_name_prefix="asr-stream"
                )
                # In-flight ASR ops finish on the old executor naturally.
                try:
                    old.shutdown(wait=False, cancel_futures=False)
                except Exception:
                    pass
    if _asr_executor is None:
        # max_workers=N from the ASR slot-pool ceiling. Lifting this above 1
        # is SAFE under the slot-pool worker: each decoder slot's CUDA graph
        # is captured once at ``initSlotPool`` time (per-slot, NOT per-request),
        # so concurrent run_in_executor dispatch never races on graph capture.
        # The single-thread serialization that this executor historically
        # enforced (process-global _ASR_CUDA_STREAM capture race) is obsolete
        # for slot-pool backends — keeping max_workers=1 here would otherwise
        # serialize accept_waveform/finalize and silently defeat the
        # worker-side slot-pool + WorkerIO Semaphore(N) concurrency.
        # Default N=1 (asr_max_slots unset) → byte-equivalent to the legacy
        # single-thread executor, so this is backward compatible.
        max_workers, src = _resolve_asr_max_workers()
        if not src.startswith("default("):
            _asr_executor_resolved_capability = True
            logger.info(
                "ASR executor: max_workers=%d (source=%s)", max_workers, src,
            )
        else:
            logger.info(
                "ASR executor: max_workers=%d (source=%s; will refresh once "
                "ASR capability is resolvable)", max_workers, src,
            )
        _asr_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="asr-stream"
        )
    return _asr_executor


def _is_pool_saturated(exc: BaseException) -> tuple[bool, int | None]:
    """Duck-type a backend ``PoolSaturatedError`` (ASR or TTS — two distinct
    classes in different modules) without importing either.

    Both backends define ``PoolSaturatedError(RuntimeError)`` with a class
    attribute ``status = 4429`` and an instance attribute ``max_slots``. They
    are deliberately NOT ``WorkerProtocolError`` subclasses, so the session
    manager does not treat them as worker faults and does NOT trigger a
    destructive kill+respawn — a saturation is "backend busy", not a protocol
    error. We mirror that here: recognize by ``status == 4429`` (+ class name
    as a belt-and-braces check) and surface a clean 4429 reject, never a
    worker restart.

    Returns ``(is_saturated, max_slots_or_None)``.
    """
    if getattr(exc, "status", None) == 4429 or type(exc).__name__ == "PoolSaturatedError":
        ms = getattr(exc, "max_slots", None)
        return True, ms if isinstance(ms, int) else None
    return False, None


def _unpack_finalize_result(raw):
    """Normalise ``ASRStream.finalize()`` return to ``(text, language)``.

    Per the ASRStream ABC contract, backends MUST return ``(text, language)``.
    Missed migrations fail loudly here (TypeError at unpack) rather than
    silently degrading to ``language=None``.
    """
    text, lang = raw
    return text or "", lang


def _default_vad_backend() -> str:
    return (
        os.environ.get("OVS_VAD_BACKEND")
        or "silero"
    ).strip() or "silero"


def _default_vad_silence_ms() -> int:
    raw = os.environ.get("OVS_VAD_SILENCE_MS") or "400"
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid OVS_VAD_SILENCE_MS=%r; using 400", raw)
        return 400
    return max(0, value)

@app.on_event("shutdown")
async def shutdown_watchdog():
    """Cancel the GPU watchdog background task on app shutdown.

    Best-effort: any errors here are swallowed because the process is
    going away regardless.
    """
    try:
        from app.core import gpu_watchdog as _gw
        await _gw.stop()
    except Exception:
        logger.debug("gpu_watchdog stop raised during shutdown", exc_info=True)


@app.on_event("startup")
async def startup():
    global _asr_backend

    try:
        from app.core.profile_loader import apply_profile_from_env, current_profile
        apply_profile_from_env()
    except Exception as exc:
        logger.error("Failed to apply OpenVoiceStream profile: %s", exc)
        raise

    # Week 1 production hardening: initialise the global session limiter
    # immediately after profile application, BEFORE model downloads and
    # backend preload. A bad limit value (zero/negative/non-int env) MUST
    # fail startup early. See docs/specs/prod-hardening-week1.md
    # Deliverable 2.
    try:
        from app.core.session_limiter import init_limiter
        init_limiter(current_profile())
    except Exception as exc:
        logger.error("SessionLimiter init failed: %s", exc)
        raise

    # Initialise the execution coordinator from the loaded profile's
    # execution_policy block. Default to concurrent (no lock) when the
    # profile does not declare one — matches the previous behaviour.
    from app.core.coordinator import init_coordinator, get_coordinator
    _prof = current_profile()
    init_coordinator(
        _prof.get("execution_policy", {"mode": "concurrent"}),
        profile=_prof,
    )

    # Week 2: launch the GPU/NPU watchdog background task. Failures here
    # never block startup — the task is purely diagnostic.
    try:
        from app.core import gpu_watchdog as _gw
        await _gw.start()
    except Exception:
        logger.exception("gpu_watchdog: start() failed; continuing without watchdog")

    # Rockchip userspace runtime is vendored in the RK image. Validate it
    # before importing rkvoice-stream backends so version/hash mismatches fail
    # with a clear operator action instead of opaque native runtime errors.
    if (current_profile().get("env") or {}).get("LANGUAGE_MODE") == "rk" or os.environ.get("LANGUAGE_MODE") == "rk":
        from app.core.rk_runtime import check_rk_runtime
        check_rk_runtime(current_profile())

    # Log language mode configuration
    language_mode = os.environ.get("LANGUAGE_MODE", "zh_en")
    logger.info("=" * 60)
    logger.info("LANGUAGE_MODE: %s", language_mode)
    logger.info(
        "VAD default: backend=%s silence_ms=%d",
        _default_vad_backend(),
        _default_vad_silence_ms(),
    )
    if language_mode == "multilanguage":
        logger.info("  → Using Qwen3 TTS + ASR (52 languages, voice cloning)")
    else:
        logger.info("  → Using Sherpa TTS + ASR (zh/en mode)")
    logger.info("=" * 60)

    from app.core import model_downloader
    model_dir = os.environ.get("MODEL_DIR", "/opt/models")
    model_downloader.ensure_models(language_mode, model_dir)

    # Resolve any TRT engines declared by the active profile. Must run
    # AFTER model_downloader (ONNX inputs may be needed for fallback
    # compile) and BEFORE any backend module is imported by the factories
    # (backends read engine paths from env vars at module import time).
    try:
        from app.core.engine_resolver import resolve_all
        resolved = resolve_all(current_profile())
        if resolved:
            logger.info("engine_resolver: resolved %d engine(s)", len(resolved))
            for env_var, path in resolved.items():
                logger.info("  %s → %s", env_var, path)
    except Exception as exc:
        logger.error("engine_resolver failed: %s", exc)
        raise

    # ASR backend (load before TTS to avoid ORT session conflicts)
    # Note: create_asr_backend() will auto-select based on LANGUAGE_MODE
    try:
        from app.core.asr_backend import create_asr_backend
        _asr_backend = create_asr_backend()  # Let it auto-detect from LANGUAGE_MODE
        logger.info("Pre-loading ASR (%s)...", _asr_backend.name)
        _asr_backend.preload()
        logger.info("ASR backend: %s (capabilities: %s)",
                     _asr_backend.name, [c.value for c in _asr_backend.capabilities])

        # Warm up ASR executor thread so its CUDA per-thread context is
        # initialised before the first streaming request.  Without this the
        # very first accept_waveform pays a cold-context tax on encoder.
        # SKIP_ASR_WARMUP=1 skips this on memory-constrained devices (Nano 8GB):
        # saves ~300-400 MB at startup, costs ~100ms one-time cold-context tax
        # on the very first ASR request.
        if os.environ.get("SKIP_ASR_WARMUP", "").lower() in ("1", "true", "yes"):
            logger.info("ASR streaming warmup skipped (SKIP_ASR_WARMUP set).")
        else:
            _asyncio = __import__("asyncio")
            _executor = _get_asr_executor()

            def _warm_asr():
                # Some backends (e.g. SherpaASRBackend) don't expose a
                # transcribe_audio convenience method; their warmup is
                # implicit in preload(). Skip silently to avoid log noise.
                if not hasattr(_asr_backend, "transcribe_audio"):
                    logger.info(
                        "ASR warmup skipped: %s has no transcribe_audio (preload already warmed).",
                        type(_asr_backend).__name__,
                    )
                    return
                try:
                    import numpy as _np
                    silence = _np.zeros(16000, dtype=_np.float32)
                    _asr_backend.transcribe_audio(silence)
                    # Note: warmup primes ONE executor thread's CUDA context.
                    # With a slot-pool backend (max_workers=N) per-slot CUDA
                    # graphs are captured at initSlotPool, not per-thread, so a
                    # single warm thread is sufficient.
                    logger.info("ASR streaming executor warmed up (CUDA primed).")
                except Exception as exc:
                    logger.warning("ASR warm-up failed: %s", exc)

            await _asyncio.get_event_loop().run_in_executor(_executor, _warm_asr)
    except Exception as e:
        logger.warning("ASR backend failed: %s", e)

    from app.core import tts_service
    if not tts_service.is_configured():
        logger.info("ASR-only mode: profile declares no tts_backend; TTS endpoints will return 503.")
    elif os.environ.get("LAZY_TTS", "").lower() in ("1", "true", "yes"):
        logger.info("TTS preload skipped (LAZY_TTS set); will load on first request.")
    else:
        logger.info("Pre-loading TTS model...")
        tts_service.preload()

    # Warm up the dedicated streaming-TTS executor thread so its CUDA
    # per-thread context is initialized before the first /tts/stream
    # request lands. Without this, the very first streaming request
    # pays a ~30ms cold-context tax on prefill.
    # Skip when LAZY_TTS or ASR-only — TTS not loaded yet, can't warm what isn't there.
    if not tts_service.is_configured():
        pass  # ASR-only mode, no TTS warmup
    elif os.environ.get("LAZY_TTS", "").lower() in ("1", "true", "yes"):
        logger.info("TTS streaming warmup skipped (LAZY_TTS).")
    else:
      try:
        from app.core.tts_backend import TTSCapability
        if tts_service.has_capability(TTSCapability.STREAMING):
            backend = tts_service.get_backend()
            executor = _get_tts_stream_executor()

            def _warm_stream():
                try:
                    # Run one tiny streaming synthesis on the executor
                    # thread to materialize CUDA context state.
                    stream_kwargs = {}
                    profile = os.environ.get("EDGE_LLM_TTS_WARMUP_STREAMING_PROFILE")
                    if profile:
                        stream_kwargs["streaming_profile"] = profile
                    warmup_text = os.environ.get("EDGE_LLM_TTS_WARMUP_TEXT", "你好")
                    for _ in backend.generate_streaming(warmup_text, **stream_kwargs):
                        pass
                except Exception as exc:  # pragma: no cover
                    logger.warning("TTS streaming warm-up failed: %s", exc)

            import asyncio as _asyncio
            await _asyncio.get_event_loop().run_in_executor(executor, _warm_stream)
            logger.info("TTS streaming executor warmed up (1 thread, CUDA primed).")
      except Exception as exc:  # pragma: no cover
        logger.warning("TTS streaming executor warm-up skipped: %s", exc)

    # Register backend getters with the coordinator so 'exclusive' policy can
    # call unload() on the dormant slot. Lambdas resolve lazily so they cope
    # with backends loaded after this point (LAZY_TTS).
    try:
        from app.core import tts_service as _tts_service_mod
        coord = get_coordinator()
        coord.register_backend("asr", lambda: _asr_backend)
        coord.register_backend("tts", lambda: _tts_service_mod._backend)
    except Exception as exc:  # pragma: no cover
        logger.warning("Coordinator backend registration skipped: %s", exc)

    # ── BackendManager wiring (PR4) ─────────────────────────────────────
    # Wrap the already-preloaded ASR/TTS instances in lifecycle managers so
    # /admin/backend/reload + acquire()-based request gating can drain
    # inflight work and hot-swap backends. The factories below return the
    # *current* singleton; on a reload the manager will call them again
    # after unloading the previous one, by which point tts_service /
    # _asr_backend have been re-bound to fresh instances. preloader is a
    # no-op on initial start (already loaded above); on reload the factory
    # invokes the real backend factory which performs its own preload.
    try:
        from app.core import backend_manager as _bm
        from app.core import tts_service as _tts_service_mod
        from app.core.asr_backend import create_asr_backend as _create_asr
        from app.core.tts_backend import create_tts_backend as _create_tts

        # On reload, build a fresh instance and rebind the legacy module
        # globals so downstream code (which still reads tts_service /
        # _asr_backend directly) sees the new backend.
        def _asr_factory():
            global _asr_backend
            if _asr_backend is None:
                _asr_backend = _create_asr()
            return _asr_backend

        def _asr_preloader(b):
            # Initial start: already preloaded above. On reload, the
            # factory returns a freshly constructed (un-preloaded)
            # instance, so we call preload() here.
            if not b.is_ready():
                b.preload()

        def _asr_unloader(b):
            global _asr_backend
            try:
                b.unload()
            finally:
                _asr_backend = None

        def _tts_factory():
            if _tts_service_mod._backend is None:
                _tts_service_mod._backend = _create_tts()
            return _tts_service_mod._backend

        def _tts_preloader(b):
            if not b.is_ready():
                b.preload()

        def _tts_unloader(b):
            try:
                b.unload()
            finally:
                _tts_service_mod._backend = None

        # FIX_4_completion: seed both managers with the profile ref used at
        # startup (OVS_PROFILE_JSON / OVS_PROFILE / OVS_PROFILE_DEFAULT). Same
        # precedence as profile_loader.apply_profile_from_env so rollback
        # re-applies via the identical source.
        _initial_profile_ref = (
            os.environ.get("OVS_PROFILE_JSON")
            or os.environ.get("OVS_PROFILE")
            or os.environ.get("OVS_PROFILE_DEFAULT")
        )
        _bm.init_backend_managers(
            tts_factory=_tts_factory,
            tts_preloader=_tts_preloader,
            tts_unloader=_tts_unloader,
            asr_factory=_asr_factory,
            asr_preloader=_asr_preloader,
            asr_unloader=_asr_unloader,
            initial_profile_ref=_initial_profile_ref,
        )

        # Bring up managers. ASR is always started if a backend exists;
        # TTS respects ASR-only profiles and LAZY_TTS env (matches the
        # legacy preload skip above).
        if _asr_backend is not None:
            await _bm.asr_manager().start()
        if tts_service.is_configured() and tts_service.is_ready():
            await _bm.tts_manager().start()
    except Exception as exc:  # pragma: no cover
        logger.warning("BackendManager wiring skipped: %s", exc)

    logger.info("Speech service ready.")


# ── Health & Capabilities ────────────────────────────────────────

# RFC 8594 deprecation hint pointing /health users at /readyz.
_HEALTH_DEPRECATION_HEADERS = {
    "Deprecation": "true",
    "Link": '</readyz>; rel="successor-version"',
}


def _metrics_requires_key() -> bool:
    """Return True when ``OVS_METRICS_REQUIRE_KEY`` opts into API-key
    protection for ``/metrics``. Default-off so standard Prometheus
    scrapes work without auth."""
    raw = os.environ.get("OVS_METRICS_REQUIRE_KEY", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


@app.get("/metrics")
async def metrics_endpoint(request: Request) -> Response:
    """Prometheus text exposition.

    Default unprotected (standard Prometheus scrape pattern). Set
    ``OVS_METRICS_REQUIRE_KEY=true`` to require the same API key used
    by public voice endpoints — ``Authorization: Bearer <key>``.

    Read-only: never blocks on backend locks, never acquires a session
    slot, never runs GPU probes. Returns 200 even while ``/readyz`` is
    503 so operators can scrape during incidents.
    """
    if _metrics_requires_key():
        # Reuse the existing HTTP auth path; it raises 401 (with a
        # ``ovs_auth_rejected_total{endpoint="/metrics"}`` bump) when
        # the token is missing or invalid.
        from app.core.api_auth import check_http
        check_http(request)

    from app.core import metrics as _metrics_mod
    body = _metrics_mod.render_prometheus()
    return Response(content=body, media_type=_metrics_mod.prometheus_content_type())


@app.get("/livez")
async def livez():
    """Process-liveness probe (always 200 while the route is reachable).

    No backend / GPU / model / profile dependency. Use this for
    orchestrator liveness restart policy; ``/readyz`` controls traffic
    admission instead.
    """
    return JSONResponse({"status": "ok"})


@app.get("/readyz")
async def readyz():
    """Readiness probe: 200 only when the service should receive traffic.

    Ready iff:
      * Required BackendManager(s) report READY (ASR always; TTS unless
        the profile is ASR-only or ``LAZY_TTS=1``).
      * The global session limiter has free capacity.
      * ``gpu_watchdog.is_ok()`` returns True.

    Read-only: never acquires a session slot, never mutates limiter
    state. Returns 503 with stable ``reasons[]`` otherwise (see spec
    Deliverable 3).
    """
    from app.core import backend_manager as _bm_mod
    from app.core import session_limiter as _sl_mod
    from app.core import gpu_watchdog as _gw_mod
    from app.core import tts_service

    reasons: list[str] = []

    # BackendManager readiness — only managers that are *required* for
    # the active profile.
    try:
        asr_mgr = _bm_mod.asr_manager()
    except Exception:
        asr_mgr = None
    try:
        tts_mgr = _bm_mod.tts_manager()
    except Exception:
        tts_mgr = None

    asr_required = _get_asr_backend() is not None
    lazy_tts = os.environ.get("LAZY_TTS", "").lower() in ("1", "true", "yes")
    tts_required = tts_service.is_configured() and not lazy_tts

    if asr_required:
        if asr_mgr is None:
            reasons.append("backend_manager_unavailable")
        elif not asr_mgr.is_ready():
            reasons.append("backend_not_ready")
    if tts_required:
        if tts_mgr is None:
            if "backend_manager_unavailable" not in reasons:
                reasons.append("backend_manager_unavailable")
        elif not tts_mgr.is_ready():
            if "backend_not_ready" not in reasons:
                reasons.append("backend_not_ready")

    # Session capacity.
    limiter = _sl_mod.get_limiter()
    if limiter is None:
        reasons.append("session_limiter_unavailable")
    elif limiter.available <= 0:
        reasons.append("sessions_full")

    # GPU/NPU watchdog (Week 2: real background-checked status).
    wd_detail = None
    try:
        if not _gw_mod.is_ok():
            reasons.append("gpu_watchdog_failed")
        try:
            wd_detail = _gw_mod.status()
        except Exception:
            wd_detail = None
    except Exception:
        reasons.append("gpu_watchdog_failed")

    if reasons:
        body = {"status": "not_ready", "reasons": reasons}
        if wd_detail is not None:
            body["details"] = {"gpu_watchdog": wd_detail}
        return JSONResponse(body, status_code=503)
    return JSONResponse({"status": "ready"})


@app.get("/health")
async def health():
    from app.core import tts_service

    result = {
        "tts": tts_service.is_ready(),
        "tts_backend": tts_service.backend_name() if tts_service.is_ready() else None,
        "tts_capabilities": [c.value for c in tts_service.capabilities()] if tts_service.is_ready() else [],
    }

    # Part D disconnect-watcher instrumentation: expose WorkerIO cancel
    # counter so stress harness can read it. Temporary; remove once stable.
    try:
        from app.core.worker_io import WorkerIO
        with WorkerIO._cancel_count_lock:
            result["tts_worker_cancel_count"] = WorkerIO._cancel_count
    except Exception:
        pass

    # ASR
    try:
        from app.core.asr_backend import create_asr_backend
        asr_be = _get_asr_backend()
        result["asr"] = asr_be.is_ready() if asr_be else False
        result["asr_backend"] = asr_be.name if asr_be and asr_be.is_ready() else None
        result["asr_capabilities"] = [c.value for c in asr_be.capabilities] if asr_be and asr_be.is_ready() else []
        if asr_be and asr_be.is_ready() and hasattr(asr_be, "providers"):
            result["asr_providers"] = asr_be.providers
    except Exception:
        result["asr"] = False
        result["asr_backend"] = None
        result["asr_capabilities"] = []

    # /health is preserved for backward-compat but deprecated; orchestrators
    # should migrate to /readyz (RFC 8594 Deprecation hint).
    return JSONResponse(result, headers=_HEALTH_DEPRECATION_HEADERS)


@app.get("/asr/capabilities")
async def asr_capabilities(_: None = Depends(_require_api_key)):
    """Return ASR backend info and supported capabilities."""
    asr_be = _get_asr_backend()
    if not asr_be or not asr_be.is_ready():
        return JSONResponse({"error": "ASR not ready"}, status_code=503)
    caps = {
        "backend": asr_be.name,
        "capabilities": [c.value for c in asr_be.capabilities],
        "sample_rate": asr_be.sample_rate,
    }
    if hasattr(asr_be, "providers"):
        caps["providers"] = asr_be.providers
    return caps


@app.get("/tts/capabilities")
async def tts_capabilities(_: None = Depends(_require_api_key)):
    """Return TTS backend info and supported capabilities."""
    from app.core import tts_service
    from app.core.tts_speakers import available_speakers
    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)
    backend = tts_service.get_backend()
    caps = [c.value for c in tts_service.capabilities()]
    return {
        "backend": tts_service.backend_name(),
        "model_id": backend.model_id,
        "capabilities": caps,
        "supports_voice_cloning": getattr(backend, "supports_voice_cloning", "voice_clone" in caps),
        "sample_rate": tts_service.get_sample_rate(),
        "speakers": available_speakers(backend.model_id),
    }


# ── Speaker Management ─────────────────────────────────────────────


class RegisterSpeakerRequest(BaseModel):
    speaker_embedding_b64: str
    label: str | None = None
    speaker_id: int | None = None


@app.get("/tts/speakers")
async def tts_speakers_list(_: None = Depends(_require_api_key)):
    """List all speakers registered for the active TTS model."""
    from app.core import tts_service
    from app.core.tts_speakers import available_speakers, default_speaker_id
    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)
    backend = tts_service.get_backend()
    from app.core.tts_backend import TTSCapability
    return {
        "model_id": backend.model_id,
        "default_speaker_id": default_speaker_id(backend.model_id),
        "speakers": available_speakers(backend.model_id),
        "supports_voice_cloning": getattr(
            backend,
            "supports_voice_cloning",
            TTSCapability.VOICE_CLONE in backend.capabilities,
        ),
    }


@app.post("/tts/speakers/register")
async def tts_speakers_register(
    req: RegisterSpeakerRequest,
    _: None = Depends(_require_api_key),
):
    """Register a voice-clone embedding as a persistent speaker.

    Accepts a base64-encoded speaker embedding (from /tts/clone/embedding)
    and assigns it a permanent speaker_id for subsequent /tts calls.
    """
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability
    from app.core.tts_speakers import register_speaker

    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)
    if not tts_service.has_capability(TTSCapability.VOICE_CLONE):
        # MEDIUM: use the unified capability-aware response (400 when the
        # backend *explicitly* disables cloning, 501 when it merely lacks
        # the capability) so /tts/speakers/register matches /tts/clone*.
        unsupported, _ok = _voice_clone_unsupported_response()
        if unsupported is not None:
            return unsupported

    try:
        emb = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    backend = tts_service.get_backend()
    try:
        spec = register_speaker(
            model_id=backend.model_id,
            payload=req.speaker_embedding_b64,
            label=req.label or "",
            meta={"dim": len(emb) // 4, "dtype": "float32"},
            speaker_id=req.speaker_id,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {
        "speaker_id": spec.id,
        "type": spec.type,
        "label": spec.label,
        "model_id": backend.model_id,
    }


@app.delete("/tts/speakers/{speaker_id}")
async def tts_speakers_delete(
    speaker_id: int,
    _: None = Depends(_require_api_key),
):
    """Delete a registered embedding speaker. Preset speakers cannot be deleted."""
    from app.core import tts_service
    from app.core.tts_speakers import unregister_speaker
    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)

    backend = tts_service.get_backend()
    try:
        ok = unregister_speaker(backend.model_id, speaker_id)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    if not ok:
        return JSONResponse({"error": f"Speaker {speaker_id} not found"}, status_code=404)
    return {"deleted": True, "speaker_id": speaker_id}


# ── TTS ──────────────────────────────────────────────────────────

@app.post("/tts")
async def tts(req: TTSRequest, _: None = Depends(_require_api_key)):
    from app.core import tts_service
    from app.core.coordinator import get_coordinator
    from app.core.session_limiter import acquire_http

    async with acquire_http("/tts"):
        return await _tts_synthesize(req)


async def _tts_synthesize(req: TTSRequest):
    from app.core import tts_service
    from app.core.coordinator import get_coordinator

    mgr = await _ensure_tts_manager_started()
    if mgr is not None:
        async with mgr.acquire() as backend:
            try:
                voice_kwargs = _request_voice_kwargs(req, backend=backend)
            except _VoiceCloneUnsupportedError as exc:
                return JSONResponse(
                    _voice_clone_unsupported_payload(exc.backend),
                    status_code=400,
                )
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            async with get_coordinator().acquire("tts"):
                # FIX_2: speed/pitch_shift come from voice_kwargs (merged with
                # runtime overrides). Do NOT pass req.speed/req.pitch directly.
                wav_bytes, meta = backend.synthesize(
                    text=req.text,
                    language=req.language,
                    **voice_kwargs,
                )
        # Week 2: record server-side TTS RTF for /metrics.
        try:
            from app.core import metrics as _m
            _m.record_tts_rtf(getattr(backend, "name", "tts"), float(meta.get("rtf", 0) or 0))
        except Exception:
            pass
    else:
        # Manager not initialised (ASR-only or wiring failed at startup) —
        # legacy tts_service path. Kept for ASR-only profiles where the
        # TTS manager is intentionally never started; LAZY_TTS is now handled
        # by _ensure_tts_manager_started above.
        try:
            legacy_backend = tts_service.get_backend() if tts_service.is_ready() else None
            voice_kwargs = _request_voice_kwargs(req, backend=legacy_backend)
        except _VoiceCloneUnsupportedError as exc:
            return JSONResponse(
                _voice_clone_unsupported_payload(exc.backend),
                status_code=400,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        async with get_coordinator().acquire("tts"):
            wav_bytes, meta = tts_service.synthesize(
                text=req.text,
                language=req.language,
                **voice_kwargs,
            )
        try:
            from app.core import metrics as _m
            _m.record_tts_rtf(tts_service.backend_name() or "tts", float(meta.get("rtf", 0) or 0))
        except Exception:
            pass
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": str(meta.get("duration", meta.get("duration_s", 0))),
            "X-Inference-Time": str(meta.get("inference_time", meta.get("inference_time_s", 0))),
            "X-RTF": str(meta.get("rtf", 0)),
        },
    )


async def _safe_cleanup_acquire_and_session(acquire_cm, release_session_fn):
    """Codex round-4 GAP B: best-effort serial cleanup helper for TTS
    streaming endpoints.

    Both ``acquire_cm.__aexit__()`` and ``release_session_fn()`` must run
    even if one of them raises. The previous pattern wrote them on
    consecutive lines without protection — if ``__aexit__`` raised
    (BackendManager bug, GeneratorExit re-raised, etc.) the slot release
    was silently skipped. Using two independent try/except blocks
    guarantees both run.
    """
    try:
        await acquire_cm.__aexit__(None, None, None)
    except BaseException:
        pass
    try:
        release_session_fn()
    except BaseException:
        pass


@app.options("/tts/stream")
async def tts_stream_options():
    return Response(status_code=200)


@app.post("/tts/stream")
async def tts_stream(
    req: TTSRequest,
    request: Request,
    _: None = Depends(_require_api_key),
):
    """Stream TTS as raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks."""
    import asyncio
    import struct
    import time
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability
    from app.core.coordinator import get_coordinator
    from app.core.session_limiter import get_limiter
    from app.core import metrics as _metrics

    # Reject-not-queue: acquire session slot BEFORE setup work. Slot
    # ownership is handed to the StreamingResponse generator's finally
    # block so it releases on disconnect / exception / normal end.
    _sl = get_limiter()
    _session_token = None
    if _sl is not None:
        _session_token = _sl.try_acquire()
        if _session_token is None:
            snap = _sl.snapshot()
            _metrics.inc_sessions_rejected("http")
            return JSONResponse(
                {"error": "too_many_sessions",
                 "current": snap["active"], "limit": snap["limit"]},
                status_code=429,
                headers={"Retry-After": "5"},
            )

    def _release_session():
        if _session_token is not None:
            _session_token.release()

    # FIX_1+FIX_3: prefer the BackendManager path so /admin/backend/reload's
    # drain logic sees streaming requests in flight. Fall back to the legacy
    # tts_service path only when the manager isn't initialised (ASR-only).
    #
    # Codex MUST-FIX 1 (Week 4 round 2): catch BaseException so CancelledError
    # also releases the slot. Python 3.8+ CancelledError is a BaseException
    # subclass, not Exception, so `except Exception` would silently leak the
    # slot on client cancel mid-setup.
    try:
        mgr = await _ensure_tts_manager_started()

        # Capability gate uses tts_service so the response shape stays
        # consistent — both paths read the same underlying backend.
        if not tts_service.has_capability(TTSCapability.STREAMING):
            _release_session()
            return JSONResponse(
                {"error": "Streaming not supported by current backend",
                 "required_capability": "streaming"},
                status_code=501,
            )
    except BaseException:
        _release_session()
        raise

    # Sentence-level streaming: split the request text into sentences (via
    # pysbd when the language is supported, regex fallback otherwise) and
    # call the TTS backend per sentence.
    from app.core.v2v import SentenceBuffer
    sbuf = SentenceBuffer(language=req.language)
    sentences = list(sbuf.add(req.text or "")) + list(sbuf.flush())

    if mgr is not None:
        # Acquire OUTSIDE the generator so inflight_http is bumped synchronously
        # at endpoint entry — otherwise it would only increment when the client
        # starts iterating the StreamingResponse, and reload drain could miss it.
        # Codex MUST-FIX 1 (Week 4 round 2): wrap acquire_cm.__aenter__() so
        # if it raises (FAILED/DRAINING manager) the session slot is released.
        # Previously this await sat outside the try block.
        acquire_cm = mgr.acquire()
        try:
            backend = await acquire_cm.__aenter__()
        except BaseException:
            _release_session()
            raise
        try:
            try:
                voice_kwargs = _request_voice_kwargs(req, backend=backend)
            except _VoiceCloneUnsupportedError as exc:
                # Bug 3 fix: pre-response 400 — better than 500 mid-stream
                # after StreamingResponse already wrote the sample-rate
                # header. Capability gate fires before any bytes go out.
                await _safe_cleanup_acquire_and_session(acquire_cm, _release_session)
                return JSONResponse(
                    _voice_clone_unsupported_payload(exc.backend),
                    status_code=400,
                )
            except ValueError as exc:
                # Codex round-4 GAP B: best-effort serial cleanup so
                # __aexit__ raising cannot skip _release_session().
                await _safe_cleanup_acquire_and_session(acquire_cm, _release_session)
                return JSONResponse({"error": str(exc)}, status_code=400)
            sr = backend.sample_rate

            async def stream():
                # Part D disconnect watcher (spec §3): Starlette cancellation
                # does not reliably close the inner sync generator running in
                # _tts_stream_executor — so poll request.is_disconnected()
                # every 100 ms and explicitly close the generator on
                # disconnect. The for-loop break path in _run calls .close()
                # on the wrapped generator, which raises GeneratorExit into
                # _generate_streaming_single() and triggers
                # _WorkerIO.cancel(req_id) (trt_edge_llm_tts.py:1255-1269).
                #
                # [Sentence pipeline parallelism] Single-user multi-sentence
                # streaming used to be strictly serial: sentence N drains
                # before sentence N+1 is even submitted. Slot 1 of the worker
                # pool (sized to OVS_TTS_WORKER_CONCURRENCY=2) sat idle in
                # the typical single-client case. Now: submit a sliding
                # window of `prefetch` sentences and drain their chunk queues
                # in order. Chunk order on the wire is unchanged (sentence
                # N's chunks are yielded before sentence N+1's), so audio
                # MD5 is byte-identical to the serial baseline. The win is
                # wall-clock: while sentence N's audio is being yielded to
                # the client, sentence N+1's prefill + early decode is
                # already running on the second slot.
                import threading as _threading
                cancel_flag = _threading.Event()
                # Active sync generators (one per in-flight sentence). The
                # disconnect watcher must close ALL of them so each
                # underlying _generate_streaming_single() receives
                # GeneratorExit and emits worker_io.cancel(req_id).
                active_gens: list = []
                gen_lock = _threading.Lock()
                watcher_task: asyncio.Task | None = None

                async def _disconnect_watcher():
                    # Directly drain the ASGI receive channel; Starlette's
                    # is_disconnected() uses a tight cancel-scope that often
                    # misses uvicorn's http.disconnect events under
                    # StreamingResponse on Python 3.10. Blocking on raw
                    # request.receive() is reliable: uvicorn pushes
                    # http.disconnect there as soon as the socket closes.
                    logger.info("tts/stream: disconnect watcher started")
                    try:
                        while not cancel_flag.is_set():
                            try:
                                message = await request.receive()
                            except Exception:
                                logger.debug(
                                    "disconnect watcher receive() failed",
                                    exc_info=True,
                                )
                                return
                            if message.get("type") == "http.disconnect":
                                cancel_flag.set()
                                # Snapshot the set under lock then close
                                # outside to avoid holding it during slow
                                # close() calls.
                                with gen_lock:
                                    gens = list(active_gens)
                                for g in gens:
                                    try:
                                        g.close()
                                    except Exception:
                                        logger.debug(
                                            "disconnect watcher gen.close() raised",
                                            exc_info=True,
                                        )
                                logger.info(
                                    "tts/stream: client disconnected — cancel flag raised (%d gens closed)",
                                    len(gens),
                                )
                                return
                    except asyncio.CancelledError:
                        pass

                try:
                    async with get_coordinator().acquire("tts"):
                        yield struct.pack("<I", sr)
                        if not sentences:
                            return
                        loop = asyncio.get_event_loop()
                        watcher_task = asyncio.create_task(_disconnect_watcher())
                        # Week 2: TTFA timer starts after admission (post sr
                        # header), observed once when the first real PCM
                        # chunk passes the boundary.
                        _ttfa_t0 = time.perf_counter()
                        _ttfa_recorded = False

                        # Pipeline window: max sentences in flight at once.
                        # Capped by the TTS stream executor size so we never
                        # block waiting for an executor slot.
                        # OVS_TTS_STREAM_PREFETCH overrides; default mirrors
                        # max_workers (2).
                        #
                        # CRITICAL: we do NOT pre-submit sentence 1 alongside
                        # sentence 0. If both prefills run simultaneously the
                        # GPU contention also hits sentence 0, so the TTFA
                        # of the very first chunk regresses (~520ms → ~920ms
                        # in early tests). Instead, sentence i+1 is submitted
                        # the moment sentence i emits its FIRST chunk — i.e.
                        # sentence i has cleared prefill and is in decode/
                        # Code2Wav, so its first audio is already on the way.
                        # This keeps sentence 0's TTFA at the single-sentence
                        # baseline while still overlapping sentence i+1's
                        # prefill with sentence i's decode.
                        executor = _get_tts_stream_executor()
                        prefetch_max = min(
                            int(os.environ.get(
                                "OVS_TTS_STREAM_PREFETCH",
                                str(executor._max_workers),
                            )),
                            len(sentences),
                        )

                        def _submit(idx: int, q: "asyncio.Queue[bytes | None]"):
                            text = sentences[idx]

                            def _run():
                                gen = None
                                try:
                                    gen = backend.generate_streaming(
                                        text,
                                        language=req.language,
                                        **voice_kwargs,
                                    )
                                    with gen_lock:
                                        active_gens.append(gen)
                                    for chunk in gen:
                                        if cancel_flag.is_set():
                                            break
                                        loop.call_soon_threadsafe(q.put_nowait, chunk)
                                except Exception as _e:
                                    # Slot-pool saturation is "backend busy",
                                    # not a synth fault: log at warning (no
                                    # stacktrace) and do NOT trigger a worker
                                    # restart. The HTTP response headers are
                                    # already in flight here, so the stream
                                    # ends gracefully (empty/short audio) rather
                                    # than a 429 — the WS paths surface 4429.
                                    _sat, _ms = _is_pool_saturated(_e)
                                    if _sat:
                                        logger.warning(
                                            "tts/stream slot-pool saturated for "
                                            "sentence=%r (max_slots=%s)",
                                            text[:80], _ms,
                                        )
                                    else:
                                        logger.exception(
                                            "tts/stream synthesis failed for sentence=%r",
                                            text,
                                        )
                                finally:
                                    if gen is not None:
                                        try:
                                            gen.close()
                                        except Exception:
                                            logger.debug(
                                                "gen.close() in _run raised",
                                                exc_info=True,
                                            )
                                        with gen_lock:
                                            try:
                                                active_gens.remove(gen)
                                            except ValueError:
                                                pass
                                    loop.call_soon_threadsafe(q.put_nowait, None)

                            loop.run_in_executor(executor, _run)

                        # Allocate queues. Submit ONLY sentence 0 to start —
                        # sentence 1+ will be submitted as sentence i emits
                        # its first chunk (see comment above for rationale).
                        queues: list[asyncio.Queue[bytes | None]] = [
                            asyncio.Queue() for _ in range(len(sentences))
                        ]
                        next_to_submit = 1
                        _submit(0, queues[0])

                        def _maybe_prefetch():
                            nonlocal next_to_submit
                            if (
                                next_to_submit < len(sentences)
                                and not cancel_flag.is_set()
                                # Keep the in-flight window bounded by
                                # prefetch_max — if it's 1, never prefetch
                                # (effectively serial, byte-equiv to old).
                                and (next_to_submit - current_idx) < prefetch_max
                            ):
                                _submit(next_to_submit, queues[next_to_submit])
                                next_to_submit += 1

                        # Drain in order. Submit sentence i+1 as soon as
                        # sentence i emits its first audio chunk.
                        for current_idx in range(len(sentences)):
                            if cancel_flag.is_set():
                                break
                            q = queues[current_idx]
                            first_chunk_seen = False
                            while True:
                                chunk = await q.get()
                                if chunk is None:
                                    break
                                if not first_chunk_seen:
                                    _maybe_prefetch()
                                    first_chunk_seen = True
                                    if not _ttfa_recorded:
                                        try:
                                            from app.core import metrics as _m2
                                            _m2.record_tts_ttfa(
                                                getattr(backend, "name", "tts"),
                                                time.perf_counter() - _ttfa_t0,
                                            )
                                        except Exception:
                                            pass
                                        _ttfa_recorded = True
                                yield chunk
                            # Also try after sentence completes, in case it
                            # produced zero chunks (degenerate path).
                            _maybe_prefetch()
                finally:
                    if watcher_task is not None:
                        watcher_task.cancel()
                        try:
                            await watcher_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    # Codex round-4 GAP B: best-effort serial cleanup so
                    # __aexit__ raising cannot skip _release_session().
                    await _safe_cleanup_acquire_and_session(acquire_cm, _release_session)

            return StreamingResponse(stream(), media_type="application/octet-stream")
        except BaseException:
            # MUST-FIX 1 round 2: cover CancelledError (BaseException) too.
            # MUST-FIX 1 round 3: each cleanup must be best-effort so a
            # failing __aexit__ / release cannot mask the original
            # exception or short-circuit subsequent cleanups.
            try:
                await acquire_cm.__aexit__(None, None, None)
            except BaseException:
                pass
            try:
                _release_session()
            except BaseException:
                pass
            raise

    # Manager not initialised — legacy direct-backend path.
    backend = tts_service.get_backend()
    sr = tts_service.get_sample_rate()
    try:
        voice_kwargs = _request_voice_kwargs(req, backend=backend)
    except _VoiceCloneUnsupportedError as exc:
        # Bug 3 fix: pre-response capability gate on the legacy path too.
        _release_session()
        return JSONResponse(
            _voice_clone_unsupported_payload(exc.backend),
            status_code=400,
        )
    except ValueError as exc:
        _release_session()
        return JSONResponse({"error": str(exc)}, status_code=400)

    if not sentences:
        async def empty():
            try:
                async with get_coordinator().acquire("tts"):
                    yield struct.pack("<I", sr)
            finally:
                _release_session()
        return StreamingResponse(empty(), media_type="application/octet-stream")

    async def stream_legacy():
        # Part D disconnect watcher — mirrors the manager-branch logic above.
        import threading as _threading
        cancel_flag = _threading.Event()
        gen_holder: list = [None]
        gen_lock = _threading.Lock()
        watcher_task: asyncio.Task | None = None

        async def _disconnect_watcher():
            logger.info("tts/stream (legacy): disconnect watcher started")
            try:
                while not cancel_flag.is_set():
                    try:
                        message = await request.receive()
                    except Exception:
                        logger.debug(
                            "legacy disconnect watcher receive() failed",
                            exc_info=True,
                        )
                        return
                    if message.get("type") == "http.disconnect":
                        cancel_flag.set()
                        with gen_lock:
                            g = gen_holder[0]
                        if g is not None:
                            try:
                                g.close()
                            except Exception:
                                logger.debug(
                                    "legacy disconnect watcher gen.close() raised",
                                    exc_info=True,
                                )
                        logger.info(
                            "tts/stream (legacy): client disconnected — cancel flag raised"
                        )
                        return
            except asyncio.CancelledError:
                pass

        try:
            async with get_coordinator().acquire("tts"):
                yield struct.pack("<I", sr)
                loop = asyncio.get_event_loop()
                watcher_task = asyncio.create_task(_disconnect_watcher())
                for sentence in sentences:
                    if cancel_flag.is_set():
                        break
                    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

                    def _run(text=sentence):
                        gen = None
                        try:
                            gen = backend.generate_streaming(
                                text,
                                language=req.language,
                                **voice_kwargs,
                            )
                            with gen_lock:
                                gen_holder[0] = gen
                            for chunk in gen:
                                if cancel_flag.is_set():
                                    break
                                loop.call_soon_threadsafe(queue.put_nowait, chunk)
                        except Exception:
                            logger.exception("tts/stream synthesis failed for sentence=%r", text)
                        finally:
                            if gen is not None:
                                try:
                                    gen.close()
                                except Exception:
                                    logger.debug(
                                        "legacy gen.close() in _run raised",
                                        exc_info=True,
                                    )
                            with gen_lock:
                                gen_holder[0] = None
                            loop.call_soon_threadsafe(queue.put_nowait, None)

                    loop.run_in_executor(_get_tts_stream_executor(), _run)

                    while True:
                        chunk = await queue.get()
                        if chunk is None:
                            break
                        yield chunk
        finally:
            if watcher_task is not None:
                watcher_task.cancel()
                try:
                    await watcher_task
                except (asyncio.CancelledError, Exception):
                    pass
            _release_session()

    return StreamingResponse(stream_legacy(), media_type="application/octet-stream")


# ── Voice Clone ───��──────────────────────────────────────────────


def _voice_clone_unsupported_response():
    """Build the unified 400/501 JSON response for backends without voice clone.

    Returns a tuple ``(response, supports_clone)`` where ``supports_clone`` is
    True iff the active backend advertises VOICE_CLONE capability. Callers
    early-return the response when supports_clone is False.
    """
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability

    if tts_service.has_capability(TTSCapability.VOICE_CLONE):
        return None, True

    backend = tts_service.get_backend() if tts_service.is_ready() else None
    supports_clone = getattr(backend, "supports_voice_cloning", None)
    if supports_clone is False:
        msg = (
            f"Current TTS backend ({tts_service.backend_name()}) does not support voice "
            "cloning. Switch to MOSS or another clone-capable backend, or use a built-in "
            "speaker_id via /tts."
        )
        return JSONResponse(
            {"error": msg,
             "required_capability": "voice_clone",
             "backend": tts_service.backend_name(),
             "supports_voice_cloning": False},
            status_code=400,
        ), False
    return JSONResponse(
        {"error": "Voice cloning not supported by current backend",
         "required_capability": "voice_clone",
         "backend": tts_service.backend_name()},
        status_code=501,
    ), False


@app.post("/tts/clone")
async def tts_clone(req: CloneRequest, _: None = Depends(_require_api_key)):
    """Synthesize with voice cloning. Requires voice_clone capability."""
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability
    from app.core.coordinator import get_coordinator
    from app.core.session_limiter import acquire_http

    async with acquire_http("/tts/clone"):
        return await _tts_clone_impl(req)


async def _tts_clone_impl(req: CloneRequest):
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability
    from app.core.coordinator import get_coordinator

    unsupported, _ok = _voice_clone_unsupported_response()
    if unsupported is not None:
        return unsupported

    try:
        speaker_embedding = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    # FIX_1: route through manager.acquire() so reload drain sees this request.
    mgr = await _ensure_tts_manager_started()
    if mgr is not None:
        async with mgr.acquire() as backend:
            async with get_coordinator().acquire("tts"):
                wav_bytes, meta = backend.clone_voice(
                    text=req.text,
                    speaker_embedding=speaker_embedding,
                    language=req.language,
                )
    else:
        async with get_coordinator().acquire("tts"):
            wav_bytes, meta = tts_service.clone_voice(
                text=req.text,
                speaker_embedding=speaker_embedding,
                language=req.language,
            )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": str(meta.get("duration", meta.get("duration_s", 0))),
            "X-Inference-Time": str(meta.get("inference_time", meta.get("inference_time_s", 0))),
            "X-RTF": str(meta.get("rtf", 0)),
        },
    )


@app.post("/tts/clone/embedding")
async def tts_extract_embedding(
    file: UploadFile = File(...),
    _: None = Depends(_require_api_key),
):
    """Extract speaker embedding from reference audio WAV.

    Returns base64-encoded speaker embedding that can be reused
    across multiple /tts/clone calls.
    """
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability
    from app.core.session_limiter import acquire_http

    unsupported, _ok = _voice_clone_unsupported_response()
    if unsupported is not None:
        return unsupported

    async with acquire_http("/tts/clone/embedding"):
        audio_bytes = await file.read()
        from app.core.coordinator import get_coordinator
        async with get_coordinator().acquire("tts"):
            embedding = tts_service.extract_speaker_embedding(audio_bytes)
        return {
            "speaker_embedding_b64": base64.b64encode(embedding).decode(),
            "embedding_size": len(embedding),
        }


@app.post("/tts/clone/stream")
async def tts_clone_stream(
    req: CloneStreamRequest,
    _: None = Depends(_require_api_key),
):
    """Stream TTS with voice cloning.

    Returns raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks.
    Requires voice_clone capability.
    """
    import asyncio
    import struct
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability
    from app.core.session_limiter import get_limiter
    from app.core import metrics as _metrics

    unsupported, _ok = _voice_clone_unsupported_response()
    if unsupported is not None:
        return unsupported

    if not tts_service.has_capability(TTSCapability.STREAMING):
        return JSONResponse(
            {"error": "Streaming not supported by current backend",
             "required_capability": "streaming"},
            status_code=501,
        )

    try:
        speaker_embedding = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    # Reject-not-queue admission gate. Slot lifetime spans the entire
    # streaming response — release happens in the generator finally.
    _sl = get_limiter()
    _session_token = None
    if _sl is not None:
        _session_token = _sl.try_acquire()
        if _session_token is None:
            snap = _sl.snapshot()
            _metrics.inc_sessions_rejected("http")
            return JSONResponse(
                {"error": "too_many_sessions",
                 "current": snap["active"], "limit": snap["limit"]},
                status_code=429,
                headers={"Retry-After": "5"},
            )

    def _release_session():
        if _session_token is not None:
            _session_token.release()

    from app.core.coordinator import get_coordinator

    stream_kwargs: dict = {
        "speaker_embedding": speaker_embedding,
        "language": req.language,
    }
    if req.first_chunk_frames is not None:
        stream_kwargs["first_chunk_frames"] = req.first_chunk_frames
    if req.chunk_frames is not None:
        stream_kwargs["chunk_frames"] = req.chunk_frames
    if req.streaming_profile is not None:
        stream_kwargs["streaming_profile"] = req.streaming_profile

    # FIX_1: enter manager.acquire() at endpoint scope so reload drain
    # observes the inflight streaming request immediately.
    #
    # Codex MUST-FIX 1: lazy-start can raise — release the just-acquired
    # session slot rather than leaking it on FAILED/DRAINING manager.
    try:
        mgr = await _ensure_tts_manager_started()
    except BaseException:
        # MUST-FIX 1 round 2: cover CancelledError (BaseException) too.
        _release_session()
        raise
    if mgr is not None:
        acquire_cm = mgr.acquire()
        try:
            backend = await acquire_cm.__aenter__()
        except BaseException:
            _release_session()
            raise
        try:
            sr = backend.sample_rate

            async def stream():
                try:
                    async with get_coordinator().acquire("tts"):
                        yield struct.pack("<I", sr)
                        loop = asyncio.get_event_loop()
                        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

                        def _run():
                            try:
                                for chunk in backend.generate_streaming(req.text, **stream_kwargs):
                                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                            except Exception:
                                logger.exception("tts/clone/stream synthesis failed")
                            finally:
                                loop.call_soon_threadsafe(queue.put_nowait, None)

                        loop.run_in_executor(_get_tts_stream_executor(), _run)
                        while True:
                            chunk = await queue.get()
                            if chunk is None:
                                break
                            yield chunk
                finally:
                    # Codex round-4 GAP B: best-effort serial cleanup so
                    # __aexit__ raising cannot skip _release_session().
                    await _safe_cleanup_acquire_and_session(acquire_cm, _release_session)

            return StreamingResponse(stream(), media_type="application/octet-stream")
        except BaseException:
            # MUST-FIX 1 round 2: cover CancelledError (BaseException) too.
            # MUST-FIX 1 round 3: best-effort cleanups so neither
            # __aexit__ nor _release_session can mask the original
            # exception or skip the other release path.
            try:
                await acquire_cm.__aexit__(None, None, None)
            except BaseException:
                pass
            try:
                _release_session()
            except BaseException:
                pass
            raise

    # Legacy fallback (manager not initialised).
    sr = tts_service.get_sample_rate()
    backend = tts_service.get_backend()

    async def stream_legacy():
        try:
            async with get_coordinator().acquire("tts"):
                yield struct.pack("<I", sr)
                loop = asyncio.get_event_loop()
                queue: asyncio.Queue[bytes | None] = asyncio.Queue()

                def _run():
                    try:
                        for chunk in backend.generate_streaming(req.text, **stream_kwargs):
                            loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    except Exception:
                        logger.exception("tts/clone/stream synthesis failed")
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                loop.run_in_executor(_get_tts_stream_executor(), _run)

                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
        finally:
            _release_session()

    return StreamingResponse(stream_legacy(), media_type="application/octet-stream")


# ── ASR ──────────────────────────────────────────────────────────

@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Query("auto"),
    _: None = Depends(_require_api_key),
):
    from app.core.session_limiter import acquire_http
    async with acquire_http("/asr"):
        return await _asr_impl(file, language)


async def _asr_impl(file: UploadFile, language: str):
    import time as _time
    audio_bytes = await file.read()

    from app.core.coordinator import get_coordinator
    mgr = _try_asr_manager()
    if mgr is not None:
        async with mgr.acquire() as asr_be:
            async with get_coordinator().acquire("asr"):
                _t0 = _time.perf_counter()
                result = asr_be.transcribe(audio_bytes, language=language)
                try:
                    from app.core import metrics as _m
                    _m.record_asr_decode_duration(asr_be.name, _time.perf_counter() - _t0)
                except Exception:
                    pass
            return {
                "text": result.text,
                "language": result.language,
                "backend": asr_be.name,
                **result.meta,
            }
    asr_be = _get_asr_backend()
    if asr_be and asr_be.is_ready():
        async with get_coordinator().acquire("asr"):
            _t0 = _time.perf_counter()
            result = asr_be.transcribe(audio_bytes, language=language)
            try:
                from app.core import metrics as _m
                _m.record_asr_decode_duration(asr_be.name, _time.perf_counter() - _t0)
            except Exception:
                pass
        return {
            "text": result.text,
            "language": result.language,
            "backend": asr_be.name,
            **result.meta,
        }
    else:
        return JSONResponse(
            status_code=503,
            content={"error": "ASR backend not available"},
        )


@app.websocket("/asr/stream")
async def asr_stream(
    ws: WebSocket,
    language: str = "auto",
    sample_rate: int = 16000,
    vad: Optional[str] = None,           # default from OVS_VAD_BACKEND
    vad_silence_ms: Optional[int] = None,
):
    """Streaming ASR via WebSocket.

    Client sends: raw int16 PCM bytes
    Client sends: empty bytes b"" to signal end
    Server sends: JSON {"text": "...", "is_final": bool, "is_stable": bool}

    Server-side VAD auto-finalizes on silence — client no longer needs to
    send the empty b"" frame in open-mic dialogue mode. ``vad_silence_ms``
    controls the silence threshold (default 400 ms). Pass ?vad=none for
    legacy forced-EOS-only behavior.

    Requires an ASR backend with STREAMING capability.
    """
    import asyncio
    import numpy as np
    from app.core.asr_backend import ASRCapability
    from app.core.api_auth import check_ws
    from app.core.session_limiter import try_acquire_ws

    # Auth runs BEFORE accept (when possible). check_ws() accepts+closes
    # 4401 on failure so the WS hand-off is deterministic.
    if not await check_ws(ws):
        return

    # Week 2: capture/generate request id before accept so the very
    # first WS log line carries the same correlator as later logs.
    _ws_request_id = request_id_from_headers(ws.headers) or generate_request_id()
    _ws_ctx_tokens = set_request_context(request_id=_ws_request_id)

    await ws.accept()

    # Reject-not-queue admission gate.
    _session_token = await try_acquire_ws(ws, "/asr/stream")
    if _session_token is None:
        reset_request_context(_ws_ctx_tokens)
        return

    # Week 2: track active streaming WS for /metrics. Paired decrement
    # lives in the finally block at the bottom of this handler.
    try:
        from app.core import metrics as _m_ws
        _m_ws.inc_active_ws_sessions()
        _ws_metric_taken = True
    except Exception:
        _ws_metric_taken = False

    # Register this WS session with the BackendManager (if available) so a
    # subsequent /admin/backend/reload can force-close it (code 1012) and
    # cancel the handler task instead of waiting forever for drain.
    _asr_mgr = _try_asr_manager()
    _ws_handle = _WSHandle(websocket=ws, task=asyncio.current_task())
    if _asr_mgr is not None:
        _asr_mgr.register_ws(_ws_handle)
    vad_backend = vad if vad is not None else _default_vad_backend()
    vad_silence = _default_vad_silence_ms() if vad_silence_ms is None else max(0, int(vad_silence_ms))

    # Choose backend: prefer ASR backend with STREAMING, fall back to sherpa
    asr_be = _get_asr_backend()
    use_backend_stream = (
        asr_be is not None
        and asr_be.is_ready()
        and asr_be.has_capability(ASRCapability.STREAMING)
    )

    # Lazy-init server-side VAD only if requested. Reuses the shared
    # singleton model from app.core.vad — no extra model load per
    # connection.
    vad_session = None
    if vad_backend and vad_backend not in ("none", "off", "disabled"):
        try:
            from app.core import vad as vad_mod
            vad_session = vad_mod.create_vad(
                vad_backend, sample_rate=sample_rate, silence_ms=vad_silence
            )
        except Exception as e:
            logger.warning("VAD '%s' init failed (%s); falling back to forced-EOS", vad_backend, e)
            vad_session = None

    try:
        if use_backend_stream:
            from app.core.coordinator import get_coordinator
            async with get_coordinator().acquire("asr"):
                await _asr_stream_backend(ws, asr_be, language, sample_rate, vad_session)
        else:
            await ws.send_json({"error": "no streaming ASR available"})
            await ws.close()
    finally:
        # best-effort cleanup: if unregister_ws raises, _session_token.release()
        # must still run (slot leak guard symmetric to /tts/stream pattern).
        if _asr_mgr is not None:
            try:
                _asr_mgr.unregister_ws(_ws_handle)
            except BaseException:
                pass
        if _session_token is not None:
            try:
                _session_token.release()
            except BaseException:
                pass
        if _ws_metric_taken:
            try:
                from app.core import metrics as _m_ws
                _m_ws.dec_active_ws_sessions()
            except Exception:
                pass
        reset_request_context(_ws_ctx_tokens)


async def _asr_stream_backend(
    ws: WebSocket,
    asr_be,
    language: str,
    sample_rate: int,
    vad_session=None,
):
    """Streaming ASR using ASR backend (accumulate-then-transcribe).

    Supports a ``reset`` control command: the client may send a JSON text
    message ``{"command": "reset"}`` at any time.  This discards the
    current stream and creates a fresh one without closing the WebSocket.
    """
    import asyncio
    import json as _json
    import numpy as np

    stream = asr_be.create_stream(language=language)
    logger.info("ASR stream opened (backend=%s)", asr_be.name)

    # Close-frame override: set to 4429 on slot-pool saturation so the finally
    # block closes with a reject-not-queue code instead of the default 1000.
    _asr_close_code: int | None = None
    _asr_close_reason: str | None = None

    try:
        while True:
            msg = await ws.receive()

            # ── Text message: control command ──
            if "text" in msg and msg["text"]:
                try:
                    cmd = _json.loads(msg["text"])
                except (ValueError, TypeError):
                    continue
                if cmd.get("command") == "reset":
                    # Release per-stream GPU resources from the old stream
                    # before swapping (no-op for backends without close()).
                    try:
                        _old_close = getattr(stream, "close", None)
                        if _old_close is not None:
                            _old_close()
                    except Exception:
                        logger.exception("ASR reset: old stream close raised")
                    stream = asr_be.create_stream(language=language)
                    await ws.send_json({
                        "type": "reset",
                        "text": "",
                        "is_final": True,
                        "is_stable": True,
                        "reset": True,
                    })
                    logger.debug("ASR stream reset by client command (backend=%s)", asr_be.name)
                elif cmd.get("command") == "end_utterance" or (cmd.get("type") or "").lower() == "eou":
                    _loop = asyncio.get_event_loop()
                    force_endpoint = getattr(stream, "force_endpoint", None)
                    if force_endpoint is not None:
                        final_text = await _loop.run_in_executor(_get_asr_executor(), force_endpoint)
                        detected_language = None
                    else:
                        await _loop.run_in_executor(_get_asr_executor(), stream.prepare_finalize)
                        raw_final = await _loop.run_in_executor(_get_asr_executor(), stream.finalize)
                        final_text, detected_language = _unpack_finalize_result(raw_final)
                    payload = {
                        "type": "final",
                        "text": final_text,
                        "is_final": True,
                        "is_stable": True,
                    }
                    if detected_language:
                        payload["language"] = detected_language
                    await ws.send_json(payload)
                    logger.debug("ASR utterance endpoint forced (backend=%s)", asr_be.name)
                continue

            # ── Binary message: audio data ──
            data = msg.get("bytes", b"")
            if data is None:
                # WebSocket disconnect frame — no bytes key
                break

            if len(data) == 0:
                # End of audio — pre-encode tail, then decode
                _loop = asyncio.get_event_loop()
                await _loop.run_in_executor(_get_asr_executor(), stream.prepare_finalize)
                raw_final = await _loop.run_in_executor(_get_asr_executor(), stream.finalize)
                final_text, detected_language = _unpack_finalize_result(raw_final)
                payload = {
                    "type": "final",
                    "text": final_text,
                    "is_final": True,
                    "is_stable": True,
                }
                if detected_language:
                    payload["language"] = detected_language
                await ws.send_json(payload)
                break

            # Buffer audio (run in thread to avoid blocking event loop)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            _loop = asyncio.get_event_loop()
            await _loop.run_in_executor(_get_asr_executor(), stream.accept_waveform, sample_rate, samples)

            # Server-side VAD endpoint detection (opt-in via ?vad=)
            if vad_session is not None:
                from app.core.vad import VADSession
                event = vad_session.process(samples)
                if event == VADSession.SPEECH_END:
                    # Emit vad_endpoint BEFORE finalize so the client can split
                    # VAD silence-wait from ASR compute time.
                    await ws.send_json({"type": "vad_endpoint"})
                    await _loop.run_in_executor(_get_asr_executor(), stream.prepare_finalize)
                    raw_final = await _loop.run_in_executor(_get_asr_executor(), stream.finalize)
                    final_text, detected_language = _unpack_finalize_result(raw_final)
                    try:
                        payload = {
                            "type": "final",
                            "text": final_text,
                            "is_final": True,
                            "is_stable": True,
                            "endpoint": "vad",
                        }
                        if detected_language:
                            payload["language"] = detected_language
                        await ws.send_json(payload)
                    except Exception:
                        # Client may have disconnected during a slow finalize
                        # (e.g. TRT-EdgeLLM on Jetson). Nothing to send to.
                        pass
                    break

            # Check for partial results
            partial_text, is_endpoint = stream.get_partial()
            if partial_text:
                if is_endpoint:
                    await ws.send_json({
                        "type": "final",
                        "text": partial_text,
                        "is_final": True,
                        "is_stable": True,
                    })
                else:
                    await ws.send_json({
                        "type": "partial",
                        "text": partial_text,
                        "is_final": False,
                        "is_stable": False,
                    })

    except WebSocketDisconnect:
        logger.debug("ASR stream client disconnected (backend=%s)", asr_be.name)
    except Exception as e:
        # Slot-pool saturation: the backend rejected a begin because every
        # decoder slot is busy (PoolSaturatedError, status 4429). This is a
        # "backend busy" condition, NOT a worker fault — surface it as a clean
        # 4429 close (matching SessionLimiter's reject-not-queue semantics) and
        # do NOT trigger a destructive worker restart (the PoolSaturatedError
        # class is intentionally off the WorkerProtocolError lineage so the
        # session manager already avoids the rebuild path).
        _saturated, _max_slots = _is_pool_saturated(e)
        if _saturated:
            logger.warning(
                "ASR stream slot-pool saturated (backend=%s, max_slots=%s); "
                "rejecting with 4429",
                asr_be.name, _max_slots,
            )
            try:
                from app.core import metrics as _m_sat
                _m_sat.inc_sessions_rejected("ws")
            except Exception:
                pass
            _reason = _json.dumps({"error": "pool_saturated", "max_slots": _max_slots})
            _asr_close_code = 4429
            _asr_close_reason = _reason
        else:
            logger.error("ASR stream error (backend=%s): %s", asr_be.name, e, exc_info=True)
            # Surface backend failures to clients as structured terminal frames.
            # Otherwise benchmark clients only observe a socket close and lose the
            # actual failure reason.
            try:
                await ws.send_json({
                    "type": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "backend": asr_be.name,
                    "is_final": True,
                    "is_stable": True,
                })
            except Exception:
                pass
    finally:
        # Release per-stream TRT contexts and device buffers (no-op for
        # backends without per-stream GPU resources).
        try:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        except Exception:
            logger.exception("ASR stream close raised")
        try:
            if _asr_close_code is not None:
                await ws.close(code=_asr_close_code, reason=_asr_close_reason)
            else:
                await ws.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────
# Unified V2V WebSocket: ASR + TTS + VAD + barge-in
# ──────────────────────────────────────────────────────────────────────

@app.websocket("/v2v/stream")
async def v2v_stream(ws: WebSocket):
    """Unified bi-directional WebSocket: speech in, partials + audio out.

    Client may enable any subset of features via the first ``config``
    JSON frame. See ``docs/api/v2v-stream.md`` for the protocol spec.
    Minimum viable patterns:

      TTS-only (LLM token stream → audio):
        send {"type":"config", "tts_language":"zh"}
        send {"type":"text", "text":"..."} repeatedly
        send {"type":"tts_flush"}; await binary chunks + tts_done

      ASR-only (mic → text, with auto VAD endpoint):
        send {"type":"config", "asr_language":"zh"}
        send PCM binary chunks
        await asr_partial / asr_endpoint / asr_final

      V2V (full duplex):
        config with both asr_language + tts_language
        interleave binary (mic) with text (LLM tokens)
        receive partials, endpoints, audio
        send {"type":"abort"} to barge-in
    """
    import asyncio
    import json as _json
    import struct
    import numpy as np

    from app.core import tts_service, v2v as v2v_proto
    from app.core.asr_backend import ASRCapability
    from app.core.tts_backend import TTSCapability
    from app.core import vad as vad_mod
    from app.core.coordinator import get_coordinator

    coord = get_coordinator()

    from app.core.api_auth import check_ws
    from app.core.session_limiter import try_acquire_ws

    # Auth before accept; deterministic 4401 close on failure.
    if not await check_ws(ws):
        return

    # Week 2: request id context for V2V WS.
    _v2v_request_id = request_id_from_headers(ws.headers) or generate_request_id()
    _v2v_ctx_tokens = set_request_context(request_id=_v2v_request_id)

    await ws.accept()

    # Reject-not-queue admission gate.
    _v2v_session_token = await try_acquire_ws(ws, "/v2v/stream")
    if _v2v_session_token is None:
        try:
            reset_request_context(_v2v_ctx_tokens)
        except BaseException:
            pass
        return

    # Week 2: active WS gauge increment. Paired decrement is in both the
    # early-exit helper (_v2v_release_early) and the final cleanup block.
    try:
        from app.core import metrics as _m_v2v
        _m_v2v.inc_active_ws_sessions()
        _v2v_ws_metric_taken = True
    except Exception:
        _v2v_ws_metric_taken = False

    # Register this v2v session with whichever BackendManager(s) are
    # available so /admin/backend/reload of either kind can hard-close
    # the WS (code 1012) instead of letting the connection linger.
    _v2v_asr_mgr = _try_asr_manager()
    _v2v_tts_mgr = _try_tts_manager()
    _v2v_handle = _WSHandle(websocket=ws, task=asyncio.current_task())
    if _v2v_asr_mgr is not None:
        _v2v_asr_mgr.register_ws(_v2v_handle)
    if _v2v_tts_mgr is not None:
        _v2v_tts_mgr.register_ws(_v2v_handle)

    # MUST-FIX 1 round 2: make release idempotent + a nonlocal flag so the
    # outer setup try/except BaseException can safely cover CancelledError
    # without double-releasing on the normal path.
    _v2v_released = {"done": False}

    def _v2v_release_early():
        # Release admission resources for early-exit paths that bypass
        # the main try/finally below. Idempotent: safe to call multiple
        # times (e.g. once from an inner branch, once from outer cancel
        # guard).
        if _v2v_released["done"]:
            return
        _v2v_released["done"] = True
        if _v2v_asr_mgr is not None:
            try:
                _v2v_asr_mgr.unregister_ws(_v2v_handle)
            except Exception:
                pass
        if _v2v_tts_mgr is not None:
            try:
                _v2v_tts_mgr.unregister_ws(_v2v_handle)
            except Exception:
                pass
        if _v2v_session_token is not None:
            try:
                _v2v_session_token.release()
            except Exception:
                pass
        if _v2v_ws_metric_taken:
            try:
                from app.core import metrics as _m_v2v
                _m_v2v.dec_active_ws_sessions()
            except Exception:
                pass
        # Note: do not reset_request_context here because the early-exit
        # helper may be called inside try/finally that itself resets;
        # caller is responsible for context cleanup.

    # MUST-FIX 1 round 2: wrap setup + main loop so any CancelledError
    # mid-setup (BaseException) still triggers admission cleanup via the
    # idempotent _v2v_release_early() helper in the outer finally.
    try:
        # ── Stage 1: receive initial config ─────────────────────────────
        try:
            first_msg = await ws.receive()
        except WebSocketDisconnect:
            _v2v_release_early(); return
        cfg_text = first_msg.get("text", "")
        if not cfg_text:
            await ws.close(code=1003); _v2v_release_early(); return
        try:
            cfg = _json.loads(cfg_text)
        except (ValueError, TypeError):
            await ws.close(code=1003); _v2v_release_early(); return
        if cfg.get("type") != v2v_proto.CLIENT_CONFIG:
            await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                "error": "first message must be a config frame"})
            await ws.close(code=1003); _v2v_release_early(); return
    
        # Codex MUST-FIX 1: config parsing below can raise ValueError/TypeError
        # on bad client input (e.g. non-int sample_rate). Without this guard,
        # the exception escapes the slot-acquired region without releasing the
        # session token / decrementing the active-WS gauge / unregistering from
        # BackendManagers.
        try:
            asr_language    = cfg.get("asr_language")  # e.g. "zh" / "Chinese" / "en" / "auto" / None
            tts_language    = cfg.get("tts_language")  # truthy = enable TTS; "auto" = let backend detect
            # "auto" enables TTS but tells downstream (sentence buffer + backend) to
            # not assume a language — backends with auto-detect (e.g. qwen3) will pick
            # one from the text content; SentenceBuffer falls back to a regex splitter.
            tts_language_norm = None if tts_language == "auto" else tts_language
            # Normalize common client-supplied aliases to the lowercase full names
            # qwen3 TTS expects ("chinese"/"english"/...). Sherpa TTS ignores the
            # value entirely, so this is a no-op there.
            if tts_language_norm:
                _TTS_LANG_ALIAS = {
                    "zh": "chinese", "zh-cn": "chinese", "zh-hans": "chinese",
                    "en": "english", "en-us": "english", "en-gb": "english",
                    "ja": "japanese", "jp": "japanese",
                    "ko": "korean", "kr": "korean",
                }
                key = tts_language_norm.strip().lower()
                tts_language_norm = _TTS_LANG_ALIAS.get(key, key)
            tts_voice       = cfg.get("tts_voice")
            tts_speaker_id = cfg.get("tts_speaker_id")
            tts_speed       = cfg.get("tts_speed")
            # Resolve speaker once at config time — avoids mid-session changes
            # (e.g. unregister) affecting later sentences in the same session.
            tts_speaker_kwargs: dict = {}
            if tts_speaker_id is not None and tts_language:
                from app.core.tts_speakers import speaker_kwargs_for_id
                if tts_service.is_ready():
                    tts_speaker_kwargs = speaker_kwargs_for_id(
                        int(tts_speaker_id), tts_service.get_backend().model_id
                    )
            sample_rate     = int(cfg.get("sample_rate", 16000))
            vad_backend     = cfg.get("vad", _default_vad_backend() if asr_language else "none")
            vad_silence_ms  = int(cfg.get("vad_silence_ms", _default_vad_silence_ms()))
            multi_utterance = bool(cfg.get("multi_utterance", False))
        except (ValueError, TypeError) as _cfg_exc:
            try:
                await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                    "error": f"invalid config field: {_cfg_exc}"})
            except Exception:
                pass
            try:
                await ws.close(code=1003)
            except Exception:
                pass
            _v2v_release_early()
            try:
                reset_request_context(_v2v_ctx_tokens)
            except BaseException:
                pass
            return
    
        if not asr_language and not tts_language:
            await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                "error": "config must enable asr_language and/or tts_language"})
            await ws.close(code=1003); _v2v_release_early(); return
    
        # ── Stage 2: bring up the backends ──────────────────────────────
        asr_be = None
        asr_manager = None  # ASRSessionManager — owns per-utterance lifecycle.
        asr_enabled = False
        vad = None
        if asr_language:
            asr_be = _get_asr_backend()
            if asr_be is None or not asr_be.is_ready() or not asr_be.has_capability(ASRCapability.STREAMING):
                await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                    "error": "asr_language requested but no streaming ASR backend ready"})
                # Codex round-4 GAP A: ws.close() itself can raise (e.g. socket
                # already torn down) — must not skip _v2v_release_early() or
                # the session slot leaks. Wrap close, then release unconditionally.
                try:
                    await ws.close(code=1011)
                except BaseException:
                    pass
                _v2v_release_early()
                return
            # Defer stream creation until first speech-start (or first audio
            # without VAD) — the manager creates a fresh stream per utterance.
            from app.core.asr_session_manager import (
                ASRSessionManager,
                ASRSessionUnavailable,
            )
            asr_manager = ASRSessionManager(
                backend=asr_be,
                language=asr_language,
                coord=coord,
                executor=_get_asr_executor(),
            )
            asr_enabled = True
            # VAD init runs in executor: silero ONNX first-load takes ~500ms and
            # would otherwise stall the event loop. ValueError (e.g. unsupported
            # sample rate) is a hard config error → reject and close. Other init
            # failures fall back to no-VAD with a warning.
            try:
                _loop_init = asyncio.get_event_loop()
                vad = await _loop_init.run_in_executor(
                    None,
                    lambda: vad_mod.create_vad(vad_backend, sample_rate=sample_rate, silence_ms=vad_silence_ms),
                )
            except ValueError as e:
                await ws.send_json({"type": v2v_proto.SERVER_ERROR, "error": f"VAD config: {e}"})
                await ws.close(code=1003); _v2v_release_early(); return
            except Exception as e:
                logger.warning("v2v VAD init (%s) failed: %s — running without VAD", vad_backend, e)
                vad = None
    
        tts_be = None
        tts_buffer = None
        if tts_language:
            if not tts_service.is_ready() or not tts_service.has_capability(TTSCapability.STREAMING):
                await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                    "error": "tts_language requested but no streaming TTS backend ready"})
                # Codex round-4 GAP A: same guard as ASR backend-not-ready
                # above — ws.close() raising must not skip slot release.
                try:
                    await ws.close(code=1011)
                except BaseException:
                    pass
                _v2v_release_early()
                return
            tts_be = tts_service.get_backend()
            low_latency_tts = os.environ.get("OVS_TTS_LOW_LATENCY_CHUNKING", "1").lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
            if low_latency_tts:
                tts_buffer = v2v_proto.LowLatencyTTSBuffer(language=tts_language_norm)
            else:
                tts_buffer = v2v_proto.SentenceBuffer(language=tts_language_norm)
    
        logger.info("v2v stream opened (asr=%s tts=%s vad=%s spk_id=%s spk_kwargs=%s)",
                    asr_language or "off", tts_language or "off", vad_backend if asr_language else "off",
                    tts_speaker_id, list(tts_speaker_kwargs.keys()) if tts_speaker_kwargs else None)
    
        # ── Stage 3: per-connection state + write serialization ─────────
        send_lock = asyncio.Lock()
        state = {
            # Per-utterance ASR endpoint signalling. Replaces the old
            # asr_eos / vad_endpoint / vad_endpoint_pending flags now that
            # ASRSessionManager owns the stream lifecycle.
            "asr_session_closed": False,   # client explicitly ended ASR (asr_eos or ws close)
            "endpoint_pending":   None,    # ("vad" | "client_eos"), set by dispatcher,
                                           # consumed by asr_out_task
            "endpoint_pending_gen": None,  # generation tag for endpoint_pending;
                                           # if it no longer matches asr_active_gen
                                           # by the time asr_out_task observes it,
                                           # the endpoint belongs to a preempted
                                           # utterance and must NOT fire finalize
                                           # against the new one (gen-race fix,
                                           # codex root-cause 2026-05-19).
            "tts_flush":        False,   # set when client tts_flush
            "current_tts_task": None,    # running TTS synth task (cancellable)
            "current_tts_stop": None,    # threading.Event to signal synth thread
                                         # to stop on barge-in (avoids orphan
                                         # synth blocking the TTS executor)
            "tts_started":      False,   # tts_started frame sent for current sentence
            "client_closed":    False,
            "asr_active":       False,   # tracks whether manager.on_speech_start
                                         # has been called for the current utterance
            "asr_active_gen":   0,       # generation tagged onto asr_active so a
                                         # stale finalize doesn't clear a fresh
                                         # utterance's asr_active flag (BUG 2)
            "endpoint_finalize_pending": False,  # set when dispatcher already
                                         # accepted the speech-end chunk and the
                                         # asr_out_task should finalize next tick
                                         # (BUG 3: avoids the flag-set/audio-accept
                                         # race that lost the tail of utterances)
            # SLV #2 (2026-05-27): per-ASR-turn wall-clock deadline. Set when a
            # turn starts (on_speech_start), cleared when it cleanly finalizes
            # / cancels. asr_out_task polls this every tick; on expiry it forces
            # a cancel + restart_worker so the WorkerIO semaphore slot is freed
            # and the v2v handler can unwind (releases the SessionLimiter slot,
            # preventing 4429 mute from a stuck Qwen3 inference).
            "asr_turn_started_at": None,
        }
        tts_q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
    
        async def send_json(payload):
            async with send_lock:
                try:
                    await ws.send_json(payload)
                except Exception:
                    state["client_closed"] = True
    
        async def send_bytes(data):
            async with send_lock:
                try:
                    await ws.send_bytes(data)
                except Exception:
                    state["client_closed"] = True
    
        async def send_error(msg):
            await send_json({"type": v2v_proto.SERVER_ERROR, "error": msg})
    
        # ── Stage 4: tasks ──────────────────────────────────────────────
    
        async def dispatcher():
            """Receive incoming binary (audio) + text (control) frames."""
            try:
                while not state["client_closed"]:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        state["client_closed"] = True
                        # Fast slot release: cancel the (possibly worker-blocked)
                        # work tasks so the SessionLimiter slot frees within ~1s
                        # instead of waiting up to OVS_ASR_TURN_TIMEOUT_S (45s)
                        # for the per-turn deadline. asr_out_task's blocking
                        # awaits (finalize / get_partial via run_in_executor) are
                        # not shielded, so cancellation propagates immediately.
                        for _wt in work_tasks:
                            if not _wt.done():
                                _wt.cancel()
                        break
                    # binary → ASR input
                    data = msg.get("bytes")
                    if data:
                        if not asr_enabled:
                            continue  # ignored in TTS-only mode
                        # After session close, drop further audio. Spec: client
                        # must open a new WebSocket to start another session.
                        if state["asr_session_closed"]:
                            continue
                        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        speech_started_now = False
                        speech_ended_now = False
                        if vad is not None:
                            event = vad.process(samples)
                            if event == vad_mod.VADSession.SPEECH_START:
                                # Notify client FIRST so it can stop buffering /
                                # playing TTS audio, then perform the server-side
                                # barge-in (cancel in-flight TTS, open fresh ASR).
                                await send_json({
                                    "type": v2v_proto.SERVER_VAD_EVENT,
                                    "event": v2v_proto.VAD_EVENT_SPEECH_START,
                                })
                                # Auto barge-in: cancel any in-flight TTS, then
                                # open a fresh ASR utterance (pre-empts any
                                # still-active session per spec).
                                t = state["current_tts_task"]
                                if t is not None and not t.done():
                                    t.cancel()
                                stop = state["current_tts_stop"]
                                if stop is not None:
                                    stop.set()
                                try:
                                    async with coord.acquire("asr"):
                                        new_gen = await asr_manager.on_speech_start()
                                except ASRSessionUnavailable as e:
                                    # Race #1: ASR worker rebuild ladder
                                    # exhausted — surface to client + skip
                                    # this turn rather than silently
                                    # accepting audio with no transcript.
                                    logger.warning(
                                        "v2v: on_speech_start failed (VAD): %s", e
                                    )
                                    await send_json({
                                        "type": v2v_proto.SERVER_ERROR,
                                        "error": "asr_unavailable",
                                    })
                                    state["asr_active"] = False
                                    state["endpoint_pending"] = None
                                    state["endpoint_pending_gen"] = None
                                    continue
                                # Clear any stale endpoint from the previous
                                # utterance — a VAD speech-end that was pending
                                # finalize while this new speech-start preempted
                                # it must NOT cause asr_out_task to call
                                # finalize() against the fresh generation
                                # (codex root-cause 2026-05-19: stale endpoint
                                # firing on the wrong generation).
                                state["endpoint_pending"] = None
                                state["endpoint_pending_gen"] = None
                                state["asr_active"] = True
                                state["asr_active_gen"] = new_gen
                                state["asr_turn_started_at"] = loop.time()
                                speech_started_now = True
                            elif event == vad_mod.VADSession.SPEECH_END:
                                # Defer setting endpoint_pending until AFTER we
                                # accept this final chunk below — otherwise the
                                # asr_out_task observes the flag and calls
                                # finalize() while the tail audio is still
                                # in-flight, silently dropping it (BUG 3).
                                speech_ended_now = True
                        # No-VAD mode: open the session lazily on first audio.
                        if vad is None and not state["asr_active"]:
                            try:
                                async with coord.acquire("asr"):
                                    new_gen = await asr_manager.on_speech_start()
                            except ASRSessionUnavailable as e:
                                # Race #1: same as VAD path above.
                                logger.warning(
                                    "v2v: on_speech_start failed (no-VAD): %s", e
                                )
                                await send_json({
                                    "type": v2v_proto.SERVER_ERROR,
                                    "error": "asr_unavailable",
                                })
                                state["asr_active"] = False
                                continue
                            state["endpoint_pending"] = None
                            state["endpoint_pending_gen"] = None
                            state["asr_active"] = True
                            state["asr_active_gen"] = new_gen
                            state["asr_turn_started_at"] = loop.time()
                        if state["asr_active"]:
                            async with coord.acquire("asr"):
                                await asr_manager.accept_audio(samples)
                        # Now safe to flag the endpoint — audio chunk that
                        # carried the speech-end has been delivered to the
                        # stream. asr_out_task will pick this up on the next
                        # poll and call finalize().
                        if speech_ended_now:
                            state["endpoint_pending"] = "vad"
                            state["endpoint_pending_gen"] = state["asr_active_gen"]
                            if not multi_utterance:
                                state["asr_session_closed"] = True
                            # Notify client of VAD speech_end so it can update
                            # its state machine (e.g. show "thinking" indicator,
                            # await asr_final). Sent AFTER endpoint_pending is
                            # latched to keep ordering deterministic w.r.t. the
                            # asr_final that follows from asr_out_task.
                            await send_json({
                                "type": v2v_proto.SERVER_VAD_EVENT,
                                "event": v2v_proto.VAD_EVENT_SPEECH_END,
                            })
                        continue
                    # text → JSON control
                    text = msg.get("text", "")
                    if not text:
                        continue
                    try:
                        payload = _json.loads(text)
                    except (ValueError, TypeError):
                        continue
                    typ = payload.get("type")
                    if typ == v2v_proto.CLIENT_TEXT and tts_buffer is not None:
                        for sentence in tts_buffer.add(payload.get("text", "")):
                            await tts_q.put(sentence)
                    elif typ == v2v_proto.CLIENT_TTS_FLUSH:
                        if tts_buffer is not None:
                            for sentence in tts_buffer.flush():
                                await tts_q.put(sentence)
                        state["tts_flush"] = True
                    elif typ == v2v_proto.CLIENT_ASR_EOS:
                        state["endpoint_pending"] = "client_eos"
                        state["endpoint_pending_gen"] = state["asr_active_gen"]
                        if not multi_utterance:
                            state["asr_session_closed"] = True
                    elif typ == v2v_proto.CLIENT_ABORT:
                        t = state["current_tts_task"]
                        if t is not None and not t.done():
                            t.cancel()
                        stop = state["current_tts_stop"]
                        if stop is not None:
                            stop.set()
                        # Drain queue so flush doesn't replay queued sentences
                        while not tts_q.empty():
                            try: tts_q.get_nowait()
                            except asyncio.QueueEmpty: break
                        # Cancel any in-flight ASR utterance too — spec: barge-in
                        # discards pending finals and resets to IDLE.
                        if asr_manager is not None and state["asr_active"]:
                            async with coord.acquire("asr"):
                                await asr_manager.cancel("bargein")
                            state["asr_active"] = False
                            state["asr_turn_started_at"] = None
            except WebSocketDisconnect:
                state["client_closed"] = True
                # See websocket.disconnect branch above: cancel work tasks so
                # the SessionLimiter slot releases fast on client disconnect.
                for _wt in work_tasks:
                    if not _wt.done():
                        _wt.cancel()

        async def asr_out_task():
            """Drive partial polling + per-utterance finalize via the manager.
    
            Each utterance is its own ``ASRSessionManager`` stream. We poll
            the *active* stream (manager.stream) for partials, then on an
            endpoint trigger (VAD speech-end, client asr_eos, or backend
            is_endpoint) we call ``manager.finalize()`` which destroys the
            stream and returns the final text.
            """
            last_streamed_final = None
            # SLV #2 (2026-05-27): wall-clock per-turn deadline. Env-driven so
            # we can dial up on slow Jetsons. Covers the gap where the ASR
            # backend gets wedged inside WorkerIO (qwen3_asr_worker stuck on
            # an inference, GPU OOM deadlock, or stdout reader hung) — none of
            # the existing per-stream timeouts (WorkerIO q.get 60s) fire when
            # the worker is *producing* events that never lead to a final.
            asr_turn_timeout_s = float(
                os.getenv("OVS_ASR_TURN_TIMEOUT_S", "45.0")
            )
            while not state["client_closed"]:
                # ── Wall-clock turn deadline ────────────────────────────
                # Active turn started but hasn't produced a final within
                # the deadline → force cancel + worker restart so the
                # WorkerIO semaphore slot is freed and this handler can
                # unwind cleanly (releases the SessionLimiter slot).
                turn_started = state.get("asr_turn_started_at")
                if (
                    state.get("asr_active")
                    and turn_started is not None
                    and (loop.time() - turn_started) > asr_turn_timeout_s
                ):
                    elapsed = loop.time() - turn_started
                    logger.warning(
                        "v2v ASR turn exceeded %.1fs wall-clock (elapsed=%.1fs); "
                        "aborting turn + force-cancel ASR session",
                        asr_turn_timeout_s, elapsed,
                    )
                    # Step 1: try cooperative cancel with a tight budget.
                    if asr_manager is not None:
                        try:
                            await asyncio.wait_for(
                                asr_manager.cancel("turn_timeout"),
                                timeout=2.0,
                            )
                        except (asyncio.TimeoutError, Exception) as _exc:
                            # cancel itself jammed → escalate to worker
                            # restart directly. restart_worker uses the
                            # default executor (not the wedged ASR slot),
                            # so it cannot deadlock here.
                            logger.error(
                                "v2v ASR cancel timed out / failed (%s); "
                                "force-restarting worker",
                                _exc,
                            )
                            try:
                                fn = getattr(asr_be, "restart_worker", None)
                                if fn is not None:
                                    loop2 = asyncio.get_event_loop()
                                    await loop2.run_in_executor(None, fn)
                            except Exception:
                                logger.exception(
                                    "v2v ASR restart_worker after turn timeout failed"
                                )
                    # Step 2: clear state + emit a final so the client side
                    # cancels its turn promptly (instead of waiting for
                    # its own thinking watchdog). Treat as empty final.
                    state["asr_active"] = False
                    state["asr_turn_started_at"] = None
                    state["endpoint_pending"] = None
                    state["endpoint_pending_gen"] = None
                    try:
                        await send_error(
                            f"asr: per-turn deadline {asr_turn_timeout_s:.0f}s "
                            f"exceeded"
                        )
                    except Exception:
                        logger.exception("send_error after asr turn timeout failed")
                    if multi_utterance and not state["asr_session_closed"]:
                        # Multi-utterance: keep running; the next speech_start
                        # will issue a new generation.
                        await asyncio.sleep(0.05)
                        continue
                    else:
                        # Single-utterance: terminate the task. The outer
                        # try/finally will release the slot.
                        return

                # Pull a stream snapshot under the manager's lock so we can
                # tag any partial with the generation it came from. If the
                # generation has advanced by emit-time, drop the partial —
                # it belongs to an utterance that's already been replaced
                # (BUG 4: stale-stream partial leak).
                partial, is_endpoint, partial_gen = "", False, 0
                if state["asr_active"]:
                    try:
                        async with coord.acquire("asr"):
                            partial_gen, partial, is_endpoint = (
                                await asr_manager.get_partial_for_generation()
                            )
                    except Exception:
                        partial, is_endpoint, partial_gen = "", False, 0
                    if partial and partial_gen == asr_manager.current_generation \
                            and partial_gen == state["asr_active_gen"]:
                        await send_json({"type": v2v_proto.SERVER_ASR_PARTIAL,
                                         "text": partial, "is_stable": bool(is_endpoint)})
    
                endpoint_reason = state["endpoint_pending"]
                # Gen-race gate: if endpoint_pending was stamped against a
                # generation that has since been preempted (VAD speech-start
                # of a new utterance, or post-worker-restart on_speech_start),
                # drop it on the floor instead of firing finalize against the
                # *new* active utterance. Without this gate the new utterance
                # gets finalized too early and the manager rejects the result
                # with "finalize result discarded (state=ACTIVE)"
                # (codex root-cause 2026-05-19).
                if (
                    endpoint_reason
                    and state.get("endpoint_pending_gen") is not None
                    and state.get("endpoint_pending_gen") != state["asr_active_gen"]
                ):
                    state["endpoint_pending"] = None
                    state["endpoint_pending_gen"] = None
                    endpoint_reason = None
    
                endpoint_fired = (
                    bool(endpoint_reason)
                    or (is_endpoint and state["asr_active"])
                )
    
                if endpoint_fired:
                    # Drain pending flag now to avoid double-firing.
                    state["endpoint_pending"] = None
                    state["endpoint_pending_gen"] = None
                    # Emit asr_endpoint only for VAD / backend endpoints,
                    # not client-driven eos.
                    if endpoint_reason != "client_eos":
                        await send_json({"type": v2v_proto.SERVER_ASR_ENDPOINT})
    
                    if state["asr_active"]:
                        finalize_gen = state["asr_active_gen"]
                        async with coord.acquire("asr"):
                            ran_gen, final_text, finalize_accepted, detected_language = (
                                await asr_manager.finalize_with_status(
                                    endpoint_reason or "backend_endpoint"
                                )
                            )
                        # Only clear asr_active if the generation we finalized
                        # is still the active one. If a new speech_start
                        # bumped the generation while finalize was in flight,
                        # leaving asr_active=True is correct — audio for the
                        # new utterance must continue to flow (BUG 2).
                        if finalize_accepted and state["asr_active_gen"] == finalize_gen:
                            state["asr_active"] = False
                            state["asr_turn_started_at"] = None
                    else:
                        final_text = ""
                        ran_gen = state["asr_active_gen"]
                        finalize_accepted = True
                        detected_language = None

                    if not finalize_accepted:
                        logger.info(
                            "suppressing discarded asr_final from gen=%s current_gen=%s reason=%s",
                            ran_gen,
                            state["asr_active_gen"],
                            endpoint_reason or "backend_endpoint",
                        )
                        continue

                    # Multi-utterance: mid-session finals carry
                    # session_complete=False; close-out final on
                    # asr_session_closed carries True.
                    if multi_utterance:
                        is_closing = state["asr_session_closed"]
                        if is_closing:
                            duplicate = (final_text or "") == (last_streamed_final or "")
                            final_payload = {
                                "type": v2v_proto.SERVER_ASR_FINAL,
                                "text": final_text or "",
                                "session_complete": True,
                                "duplicate_of_streamed": duplicate,
                            }
                            if detected_language:
                                final_payload["language"] = detected_language
                            await send_json(final_payload)
                            return
                        else:
                            final_payload = {
                                "type": v2v_proto.SERVER_ASR_FINAL,
                                "text": final_text or "",
                                "session_complete": False,
                            }
                            if detected_language:
                                final_payload["language"] = detected_language
                            await send_json(final_payload)
                            last_streamed_final = final_text or ""
                            # keep the loop running for the next utterance
                    else:
                        final_payload = {
                            "type": v2v_proto.SERVER_ASR_FINAL,
                            "text": final_text or "",
                        }
                        if detected_language:
                            final_payload["language"] = detected_language
                        await send_json(final_payload)
                        return
    
                # Exit only when the session is closed and there's nothing
                # left to finalize — single-utterance terminates above on
                # the endpoint; multi-utterance terminates on close-out.
                if state["asr_session_closed"] and not state["asr_active"]:
                    return
    
                await asyncio.sleep(0.05)
    
        async def tts_out_task():
            """Drain sentence queue → synthesize → emit audio.
    
            Sends the 4-byte sample-rate header on first successful synth
            (NOT first attempted synth — so a cancelled-mid-flight first
            sentence doesn't leave the client with a header but no audio).
            """
            sr_header_sent = False
            while not state["client_closed"]:
                # Exit when client said flush and the queue is drained.
                if state["tts_flush"] and tts_q.empty():
                    # Multi-utterance: per-turn flush ends one turn but not
                    # the SESSION. Reset the sticky flag, emit a per-turn
                    # tts_done (session_complete=False mirroring ASR's
                    # mid-session final at :1763-1779), then loop back to
                    # wait for the next turn. Without this the task returns
                    # after round 1 → asyncio.gather() unblocks → WS closes,
                    # which is the "TTS stuck after round 1" bug.
                    if multi_utterance and not state.get("asr_session_closed", False):
                        state["tts_flush"] = False
                        if not state["client_closed"]:
                            await send_json({
                                "type": v2v_proto.SERVER_TTS_DONE,
                                "session_complete": False,
                            })
                        continue
                    break
                try:
                    sentence = await asyncio.wait_for(tts_q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                audio_queue: asyncio.Queue = asyncio.Queue()
                # Likely #1 fix: signal the synth thread to stop mid-iteration
                # on barge-in (single-thread TTS executor would otherwise be
                # blocked by the orphaned generator until it completes).
                import threading as _threading
                stop_event = _threading.Event()
                state["current_tts_stop"] = stop_event
    
                def _run_synth(s, synth_be):
                    try:
                        stream_kwargs = {"language": tts_language_norm}
                        if tts_speaker_kwargs:
                            stream_kwargs.update(tts_speaker_kwargs)
                        elif tts_voice is not None:
                            stream_kwargs["voice"] = tts_voice  # deprecated
                        if tts_speed is not None:    stream_kwargs["speed"] = tts_speed
                        # FIX_C: use the manager-acquired backend so a reload
                        # waiting for drain sees this request as inflight.
                        for chunk in synth_be.generate_streaming(s, **stream_kwargs):
                            if stop_event.is_set():
                                break
                            loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)
                    except Exception as e:
                        # TTS slot-pool saturation is "backend busy", NOT a
                        # synth failure — log at warning (no stacktrace) and tag
                        # the queue item so the drain loop emits a clean 4429
                        # signal rather than a destructive error. The
                        # PoolSaturatedError class is off the WorkerProtocolError
                        # lineage, so no worker restart is triggered here.
                        _sat, _ms = _is_pool_saturated(e)
                        if _sat:
                            logger.warning(
                                "v2v tts slot-pool saturated for sentence=%r "
                                "(max_slots=%s)", s[:80], _ms,
                            )
                            loop.call_soon_threadsafe(
                                audio_queue.put_nowait,
                                ("__saturated__", _ms),
                            )
                        else:
                            logger.exception("v2v tts synthesis failed for sentence=%r", s)
                            loop.call_soon_threadsafe(audio_queue.put_nowait, ("__error__", str(e)))
                    finally:
                        loop.call_soon_threadsafe(audio_queue.put_nowait, None)
    
                async def drain():
                    nonlocal sr_header_sent
                    # PR5 / FIX_C: take BackendManager.acquire() *per utterance*
                    # so admin reload's drain logic sees this synth as inflight.
                    # Per-utterance (vs per-session) is intentional: v2v sessions
                    # can run for minutes; holding acquire across the whole
                    # session would block every reload until the user hangs up.
                    # _v2v_tts_mgr is captured from the enclosing scope (set
                    # earlier from _try_tts_manager()); fall back to the
                    # already-bound tts_be when manager wiring is absent (partial
                    # config / tests).
                    tts_mgr_local = _v2v_tts_mgr
                    if tts_mgr_local is not None:
                        acquire_cm = tts_mgr_local.acquire()
                        synth_backend = await acquire_cm.__aenter__()
                    else:
                        acquire_cm = None
                        synth_backend = tts_be
                    try:
                        # Coord lock per-sentence: cheap on concurrent profiles;
                        # serializes sentences against ASR on serialized profiles.
                        async with coord.acquire("tts"):
                            if not sr_header_sent:
                                sr = tts_service.get_sample_rate() if hasattr(tts_service, "get_sample_rate") else 16000
                                await send_bytes(struct.pack("<I", sr))
                                sr_header_sent = True
                            await send_json({"type": v2v_proto.SERVER_TTS_STARTED, "sentence": sentence})
                            loop.run_in_executor(_get_tts_stream_executor(), _run_synth, sentence, synth_backend)
                            state["tts_started"] = True
                            # Watchdog: if the synth thread doesn't produce
                            # a chunk within this many seconds, treat the
                            # sentence as failed and continue. Without this
                            # a wedged TTS backend (model load issue, GPU
                            # OOM, etc.) leaves the client (and any
                            # downstream agent) waiting forever on a
                            # promise the server can never fulfil.
                            tts_chunk_timeout_s = float(
                                os.getenv("OVS_TTS_CHUNK_TIMEOUT_S", "10.0")
                            )
                            while True:
                                try:
                                    item = await asyncio.wait_for(
                                        audio_queue.get(),
                                        timeout=tts_chunk_timeout_s,
                                    )
                                except asyncio.TimeoutError:
                                    logger.warning(
                                        "v2v tts watchdog: no chunk within %.1fs for "
                                        "sentence=%r — aborting synth and emitting error",
                                        tts_chunk_timeout_s, sentence[:80],
                                    )
                                    stop_event.set()
                                    await send_error(
                                        f"tts: synth produced no chunks within "
                                        f"{tts_chunk_timeout_s:.0f}s"
                                    )
                                    break
                                if item is None:
                                    break
                                if isinstance(item, tuple) and item[0] == "__saturated__":
                                    # Backend busy (slot-pool saturated). Surface
                                    # a typed reject-not-queue signal; do NOT
                                    # tear down the worker. The session stays
                                    # alive so the client can retry.
                                    try:
                                        await send_json({
                                            "type": v2v_proto.SERVER_ERROR,
                                            "error": "pool_saturated",
                                            "status": 4429,
                                            "max_slots": item[1],
                                        })
                                    except Exception:
                                        pass
                                    break
                                if isinstance(item, tuple) and item[0] == "__error__":
                                    await send_error(f"tts: {item[1]}")
                                    break
                                await send_bytes(item)
                            await send_json({"type": v2v_proto.SERVER_TTS_SENTENCE_DONE, "sentence": sentence})
                    finally:
                        if acquire_cm is not None:
                            try:
                                await acquire_cm.__aexit__(None, None, None)
                            except Exception:
                                logger.exception("v2v tts acquire exit failed")
    
                task = asyncio.create_task(drain())
                state["current_tts_task"] = task
                # Outer per-sentence deadline. Codex review 2026-05-26
                # caught the gap: the inner ``audio_queue.get()`` watchdog
                # only fires AFTER ``tts_started`` is emitted (drain() has
                # already passed backend acquire + coord acquire +
                # send_json). If the wedge is in any of those earlier
                # steps — backend-manager acquire blocked by a stuck
                # reload, ``coord.acquire`` deadlock, or Matcha's
                # pre-yield ORT/TRT setup — the client never sees
                # ``tts_started`` and waits forever. Wrap the whole drain
                # in a deadline that covers ALL of the above; tuning via
                # env so we can dial up on slow Jetsons.
                tts_sentence_timeout_s = float(
                    os.getenv("OVS_TTS_SENTENCE_TIMEOUT_S", "15.0")
                )
                try:
                    await asyncio.wait_for(task, timeout=tts_sentence_timeout_s)
                except asyncio.TimeoutError:
                    logger.warning(
                        "v2v tts: per-sentence deadline %.1fs exceeded for "
                        "sentence=%r — cancelling drain and continuing",
                        tts_sentence_timeout_s, sentence[:80],
                    )
                    stop_event.set()
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    # Try to send an error event so the client side
                    # cancels its turn promptly (instead of waiting for
                    # its own thinking watchdog at 20s).
                    if not state["client_closed"]:
                        try:
                            await send_error(
                                f"tts: per-sentence deadline {tts_sentence_timeout_s:.0f}s "
                                f"exceeded"
                            )
                        except Exception:
                            logger.exception("send_error after tts deadline failed")
                except asyncio.CancelledError:
                    # Barge-in: tell the synth thread to break out of the
                    # generator loop, then drain any chunks it produced
                    # before noticing the flag.
                    stop_event.set()
                    try:
                        while True:
                            item = audio_queue.get_nowait()
                            if item is None: break
                    except asyncio.QueueEmpty:
                        pass
                finally:
                    state["current_tts_task"] = None
                    state["current_tts_stop"] = None
            if not state["client_closed"]:
                # Session-final tts_done. In multi-utterance mode tag it as
                # session_complete=True so the client can distinguish it from
                # the per-turn dones emitted above. Single-utterance mode
                # omits the field for backward compatibility.
                payload = {"type": v2v_proto.SERVER_TTS_DONE}
                if multi_utterance:
                    payload["session_complete"] = True
                await send_json(payload)
    
        # ── Stage 5: orchestrate ────────────────────────────────────────
        # Bug #3 fix: dispatcher loops on ws.receive() forever (only exits
        # on disconnect). If we asyncio.gather all three, the server hangs
        # after asr_final / tts_done. Spawn work tasks separately, wait for
        # them, then cancel the dispatcher.
        dispatcher_task = asyncio.create_task(dispatcher())
        work_tasks = []
        if asr_enabled:
            work_tasks.append(asyncio.create_task(asr_out_task()))
        if tts_be is not None:
            work_tasks.append(asyncio.create_task(tts_out_task()))
    
        # NIT 3 round 3: track whether the V2V loop exited via a server
        # error so the WebSocket close frame carries the standard 1011
        # "internal error" code rather than the default 1005/1000.
        _v2v_server_error = False
        # Override the close code: a slot-pool saturation that bubbles up
        # through the ASR/TTS work tasks should close 4429 (reject-not-queue),
        # not 1011 (server error). None = use the default close logic below.
        _v2v_close_code: int | None = None
        try:
            if work_tasks:
                try:
                    await asyncio.gather(*work_tasks, return_exceptions=False)
                except asyncio.CancelledError:
                    # Work tasks were cancelled by the dispatcher on client
                    # disconnect (fast slot release) — not a server error and
                    # not a cancellation of this handler. The finally below
                    # still releases the SessionLimiter slot. Re-raise only if
                    # THIS handler was genuinely cancelled (no client close).
                    if not state["client_closed"]:
                        raise
            else:
                # No work tasks (shouldn't happen — config rejected earlier),
                # just keep the dispatcher running until the client closes.
                await dispatcher_task
        except Exception as e:
            # Slot-pool saturation bubbling up from ASR finalize / TTS synth:
            # treat as "backend busy", NOT a server error. Emit a typed 4429
            # reject and close 4429 (reject-not-queue) — do NOT trigger the
            # 1011 path or a destructive worker restart.
            _sat, _ms = _is_pool_saturated(e)
            if _sat:
                _v2v_close_code = 4429
                logger.warning(
                    "v2v stream slot-pool saturated (max_slots=%s); rejecting "
                    "with 4429", _ms,
                )
                try:
                    from app.core import metrics as _m_v2v_sat
                    _m_v2v_sat.inc_sessions_rejected("ws")
                except Exception:
                    pass
                try:
                    await send_json({
                        "type": v2v_proto.SERVER_ERROR,
                        "error": "pool_saturated",
                        "status": 4429,
                        "max_slots": _ms,
                    })
                except Exception:
                    pass
            else:
                _v2v_server_error = True
                logger.error("v2v stream error: %s", e, exc_info=True)
                try:
                    await send_error(f"{type(e).__name__}: {e}")
                except Exception:
                    pass
        finally:
            if not dispatcher_task.done():
                dispatcher_task.cancel()
                try:
                    await dispatcher_task
                except (asyncio.CancelledError, Exception):
                    pass
            for t in work_tasks:
                if not t.done():
                    t.cancel()
            # Tell the synth thread to bail (if running) so the TTS executor
            # frees up for the next connection.
            stop = state["current_tts_stop"]
            if stop is not None:
                stop.set()
            # Race #6: await the cancelled work tasks before releasing the
            # ASR/TTS slot. Otherwise the next WS connection on this
            # SessionLimiter slot can grab the limited ASR executor while
            # the previous worker thread is still running, producing
            # spurious "ASR busy" or worker-protocol errors on the very
            # first turn of the new connection.
            if work_tasks:
                try:
                    await asyncio.gather(*work_tasks, return_exceptions=True)
                except Exception:
                    pass
            # Cancel any in-flight ASR utterance before closing the socket
            # so the worker doesn't leak the session.
            if asr_manager is not None:
                try:
                    await asr_manager.cancel("ws_close")
                except Exception:
                    pass
            try:
                if _v2v_close_code is not None:
                    # Slot-pool saturation → reject-not-queue close.
                    await ws.close(
                        code=_v2v_close_code,
                        reason='{"error":"pool_saturated"}',
                    )
                elif _v2v_server_error:
                    await ws.close(code=1011)
                else:
                    await ws.close()
            except Exception:
                pass
            if _v2v_asr_mgr is not None:
                try:
                    _v2v_asr_mgr.unregister_ws(_v2v_handle)
                except BaseException:
                    pass
            if _v2v_tts_mgr is not None:
                try:
                    _v2v_tts_mgr.unregister_ws(_v2v_handle)
                except BaseException:
                    pass
            if _v2v_session_token is not None:
                try:
                    _v2v_session_token.release()
                except BaseException:
                    pass
            if _v2v_ws_metric_taken:
                try:
                    from app.core import metrics as _m_v2v
                    _m_v2v.dec_active_ws_sessions()
                    _v2v_ws_metric_taken = False
                except Exception:
                    pass
            try:
                reset_request_context(_v2v_ctx_tokens)
            except BaseException:
                pass
            logger.info("v2v stream closed")
    except BaseException:
        # MUST-FIX 1 round 2: covers CancelledError (BaseException) raised
        # mid-setup before the inner main try/finally is established. The
        # release helper is idempotent so this is safe even on the normal
        # path where the inner finally already released.
        # MUST-FIX 1 round 3: wrap each cleanup in best-effort try/except
        # so a failing helper cannot mask the original exception or
        # short-circuit subsequent cleanups.
        try:
            _v2v_release_early()
        except BaseException:
            pass
        try:
            reset_request_context(_v2v_ctx_tokens)
        except BaseException:
            pass
        raise


# ── Admin: TTS runtime overrides ────────────────────────────────────────────

class TTSRuntimePatch(BaseModel):
    speaker_id: Optional[int] = None
    speed: Optional[float] = None
    pitch_shift: Optional[float] = None


def _current_tts_model_id() -> Optional[str]:
    from app.core import tts_service
    if not tts_service.is_ready():
        return None
    try:
        return tts_service.get_backend().model_id
    except Exception:
        return None


def _effective_tts_values(model_id: Optional[str]) -> dict:
    from app.core import tts_runtime
    from app.core.tts_speakers import default_speaker_id
    snap = tts_runtime.get_overrides()
    if snap.default_speaker_id is not None:
        eff_speaker = snap.default_speaker_id
    elif model_id is not None:
        try:
            eff_speaker = default_speaker_id(model_id)
        except Exception:
            eff_speaker = None
    else:
        eff_speaker = None
    return {
        "speaker_id": eff_speaker,
        "speed": snap.default_speed,
        "pitch_shift": snap.default_pitch_shift,
    }


def _admin_dep():
    from app.core.admin_auth import require_admin
    return require_admin


@app.get("/admin/tts/runtime")
async def admin_tts_runtime_get(_: None = Depends(_admin_dep())):
    from app.core import tts_runtime
    snap = tts_runtime.get_overrides()
    model_id = _current_tts_model_id()
    return {
        "model_id": model_id,
        "overrides": {
            "speaker_id": snap.default_speaker_id,
            "speed": snap.default_speed,
            "pitch_shift": snap.default_pitch_shift,
            "updated_at": snap.updated_at,
        },
        "effective": _effective_tts_values(model_id),
    }


@app.patch("/admin/tts/runtime")
async def admin_tts_runtime_patch(
    req: TTSRuntimePatch,
    _: None = Depends(_admin_dep()),
):
    from app.core import tts_runtime
    fields = req.model_fields_set
    kwargs: dict = {}
    if "speaker_id" in fields:
        kwargs["speaker_id"] = req.speaker_id
    if "speed" in fields:
        kwargs["speed"] = req.speed
    if "pitch_shift" in fields:
        kwargs["pitch_shift"] = req.pitch_shift
    model_id = _current_tts_model_id()
    if model_id is not None:
        kwargs["model_id"] = model_id
    try:
        snap = tts_runtime.update_overrides(**kwargs)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    return {
        "model_id": model_id,
        "overrides": {
            "speaker_id": snap.default_speaker_id,
            "speed": snap.default_speed,
            "pitch_shift": snap.default_pitch_shift,
            "updated_at": snap.updated_at,
        },
        "effective": _effective_tts_values(model_id),
    }


@app.post("/admin/tts/speakers/reload")
async def admin_tts_speakers_reload(
    _: None = Depends(_admin_dep()),
):
    from app.core.tts_speakers import reload_speakers, available_speakers
    reload_speakers()
    model_id = _current_tts_model_id()
    count = 0
    if model_id is not None:
        try:
            count = len(available_speakers(model_id))
        except Exception:
            count = 0
    return {"reloaded": True, "model_id": model_id, "count": count}


# ── Admin: Backend hot-reload ───────────────────────────────────────────────

class BackendReloadRequest(BaseModel):
    kind: Literal["tts", "asr"]
    profile: str
    drain_timeout_s: Optional[float] = None


@app.post("/admin/backend/reload")
async def admin_backend_reload(
    payload: BackendReloadRequest,
    _: None = Depends(_admin_dep()),
):
    from app.core.backend_manager import tts_manager, asr_manager

    if payload.kind == "tts":
        mgr = tts_manager()
    else:  # "asr"  (Literal already constrains the values)
        mgr = asr_manager()
    # drain_timeout_s override is plumbed into the request schema for
    # forward compatibility; the manager does not yet expose a setter,
    # so we ignore it for now (TODO: surface a per-call drain timeout
    # on BackendManager.reload).
    return await mgr.reload(payload.profile, reason="admin")


@app.get("/admin/backend/status")
async def admin_backend_status(_: None = Depends(_admin_dep())):
    from app.core.backend_manager import tts_manager, asr_manager
    return {"tts": tts_manager().status(), "asr": asr_manager().status()}
