"""Per-utterance ASR session manager.

Owns the lifecycle of streaming ASR sessions for a single WebSocket connection:
fresh ``ASRStream`` per utterance, generation tokens guarding against stale
finals, bounded cancellation with worker-restart fallback, and ERROR_REBUILD
recovery on worker protocol errors.

State machine
-------------

    IDLE ──speech_start──► ACTIVE ──speech_end / asr_eos──► FINALIZING ──ack──► IDLE
                              │                                  │
                              └────────── cancel ────────────────┴─► CANCELLING ──► IDLE
                                                                          │
                                                                          ▼
                                                              (waits ≤500ms for end-ack;
                                                               on timeout calls restart_worker())

    Any ──worker error──► ERROR_REBUILD ──(retry ≤3 / backoff 50,150,400ms)──► IDLE
                                       └─ exhausted ──► restart_worker() ──► IDLE

Each transition into ``ACTIVE`` issues a fresh ``generation_id``; finals tagged
with a stale generation are silently dropped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class ASRSessionUnavailable(RuntimeError):
    """Raised when on_speech_start cannot produce a working ASR stream.

    Signals to the caller that the ASR worker is unrecoverable for this
    turn — caller MUST NOT flag the session as active (race #1: silent
    no-op accept_audio loop with client stuck THINKING).
    """


class SessionState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    FINALIZING = "finalizing"
    CANCELLING = "cancelling"
    ERROR_REBUILD = "error_rebuild"


# Worker-protocol error types (mirrored on trt_edge_llm_asr backend);
# duck-typed via class name so tests / non-jetson backends don't need to
# import the jetson module.
_WORKER_ERROR_NAMES = {
    "NoActiveSessionError",
    "SessionAlreadyActiveError",
    "WorkerExitError",
    "WorkerProtocolError",
}


def _is_worker_protocol_error(exc: BaseException) -> bool:
    if exc is None:
        return False
    for cls in type(exc).__mro__:
        if cls.__name__ in _WORKER_ERROR_NAMES:
            return True
    return False


def _safe_close_stream(stream: Any) -> None:
    """Release per-stream backend resources (TRT contexts, device buffers).

    Default ASRStream.close() is a no-op; backends like paraformer_trt
    override it to drop per-stream TRT IExecutionContext + cudaMalloc'd
    buffers (added 2026-05-25 for N>=2 concurrency safety). We swallow any
    exception so close() never breaks lifecycle teardown.
    """
    if stream is None:
        return
    close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        logger.exception("ASRSessionManager: stream.close raised; ignoring")


class ASRSessionManager:
    """Async-safe per-utterance ASR session orchestrator.

    Backends are synchronous; all calls into them are hopped through
    ``loop.run_in_executor`` to avoid blocking the event loop. A single
    instance-level ``asyncio.Lock`` serializes state transitions.
    """

    # Retry/backoff schedule for ERROR_REBUILD (≤3 attempts before
    # falling back to a full worker restart).
    _REBUILD_BACKOFF_S = (0.05, 0.15, 0.40)
    _CANCEL_ACK_TIMEOUT_S = 0.5

    def __init__(
        self,
        backend: Any,
        language: str = "auto",
        coord: Any = None,
        *,
        sample_rate: int = 16000,
        executor: Any = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._backend = backend
        self._language = language
        self._coord = coord  # BackendCoordinator (optional)
        # B3 fix: probe the backend's sample rate instead of hardcoding 16000 in
        # accept_audio (which fed non-16kHz ASR variants at the wrong rate).
        # Mirrors voxedge.engine.asr_session_manager (keeps the two copies in
        # sync on this fix until the dedup re-export lands).
        self._sample_rate = int(getattr(backend, "sample_rate", sample_rate) or sample_rate)
        self._executor = executor  # asr executor (optional)
        self._loop = loop  # late-bound if None
        self._lock = asyncio.Lock()
        self._state: SessionState = SessionState.IDLE
        self._stream: Any = None
        self._generation: int = 0
        self._last_error: Optional[BaseException] = None
        # Coalesce concurrent error notifications: while a recovery is
        # in progress, further mark_error / _handle_error_locked calls
        # for the same generation collapse to a no-op so we don't kick
        # off N restart_worker() calls for one stuck worker.
        self._recovery_in_progress: bool = False
        # Shared future for coalescing concurrent _async_mark_error calls
        # within one error window. While this is non-None, additional
        # callers await it instead of triggering their own rebuild.
        self._recovery_future: Optional[asyncio.Future] = None

    # ── public introspection ───────────────────────────────────────────
    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def current_generation(self) -> int:
        return self._generation

    @property
    def stream(self) -> Any:
        return self._stream

    # ── helpers ────────────────────────────────────────────────────────
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        return asyncio.get_event_loop()

    async def _run_sync(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        loop = self._get_loop()
        if kwargs:
            def _bound():
                return fn(*args, **kwargs)
            return await loop.run_in_executor(self._executor, _bound)
        return await loop.run_in_executor(self._executor, fn, *args)

    def _new_stream_sync(self) -> Any:
        return self._backend.create_stream(language=self._language)

    async def _create_stream(self) -> Any:
        return await self._run_sync(self._new_stream_sync)

    # ── public API ─────────────────────────────────────────────────────
    async def on_speech_start(self) -> int:
        """Transition IDLE→ACTIVE (cancelling any prior session first).

        Returns the new generation id. Stale finals from a previous
        generation must be ignored by the caller.
        """
        async with self._lock:
            # If we are still mid-session (typically because TTS was
            # playing and the user started speaking again), end the old
            # session before issuing a new generation. Discarded final.
            if self._state in (SessionState.ACTIVE, SessionState.FINALIZING):
                await self._inner_cancel(reason="speech_start_preempt")
            elif self._state == SessionState.CANCELLING:
                # Another cancel is in flight; wait for it to settle
                # (cheap: cancel runs under our own lock already).
                pass

            # Now (re)create stream with bounded retry on worker errors.
            self._generation += 1
            try:
                self._stream = await self._create_stream()
            except Exception as exc:  # noqa: BLE001
                logger.warning("ASRSessionManager: create_stream failed: %s", exc)
                await self._handle_error_locked(exc)
                # After _handle_error_locked runs the full rebuild ladder
                # (incl. worker restart). If even that failed, state will
                # have been set to IDLE and stream remains None — in which
                # case we MUST surface the failure to the caller rather
                # than silently flagging ACTIVE (race #1: WS would stay
                # alive with a dead ASR, audio dropped, no final, client
                # stuck THINKING forever). If a stream was rebuilt during
                # recovery, state is already ACTIVE and we can proceed.
                if self._stream is None:
                    self._state = SessionState.IDLE
                    raise ASRSessionUnavailable(
                        "ASR worker unavailable after rebuild"
                    ) from exc
            self._state = SessionState.ACTIVE
            return self._generation

    async def accept_audio(self, samples) -> None:
        """Push a chunk of audio at the current stream.

        No-op outside ACTIVE. Failures route to ERROR_REBUILD.

        Recovery note: the audio chunk that triggers ERROR_REBUILD is
        dropped, not replayed. Callers are responsible for retransmit if
        needed; in this codebase none do, which is acceptable because
        VAD will re-trigger on the next speech burst and ASR backends
        only need recent audio context, not the lost frame.
        """
        # Take the snapshot under the lock, then release before the long
        # IO call so other transitions (cancel/finalize) aren't blocked.
        async with self._lock:
            if self._state != SessionState.ACTIVE:
                return
            stream = self._stream
            if stream is None:
                # ERROR_REBUILD recovery — try to (re)create lazily.
                try:
                    stream = await self._create_stream()
                    self._stream = stream
                except Exception as exc:  # noqa: BLE001
                    await self._handle_error_locked(exc)
                    return
        try:
            await self._run_sync(stream.accept_waveform, self._sample_rate, samples)
        except Exception as exc:  # noqa: BLE001
            async with self._lock:
                await self._handle_error_locked(exc)

    async def finalize(self, reason: str = "vad_end") -> str:
        """Transition ACTIVE→FINALIZING→IDLE; return final text.

        Returns "" if not in a finalizable state (defensive — caller may
        race with cancel). For callers that emit protocol frames, prefer
        :meth:`finalize_with_status` so discarded finalize completions can
        be distinguished from accepted empty transcripts.
        """
        _gen, text = await self.finalize_with_generation(reason)
        return text

    async def finalize_with_generation(self, reason: str = "vad_end") -> tuple[int, str]:
        """Like :meth:`finalize` but returns ``(generation_id, text)``.

        ``generation_id`` is the generation the finalize ran against.
        Callers that only need to avoid clobbering shared state can compare
        the returned generation against their own active-generation snapshot.
        Callers that emit protocol frames should use
        :meth:`finalize_with_status` instead.

        Returns ``(gen, "")`` on no-op / discarded paths.
        """
        gen, text, _accepted, _lang = await self.finalize_with_status(reason)
        return gen, text

    async def finalize_with_status(
        self, reason: str = "vad_end"
    ) -> tuple[int, str, bool, Optional[str]]:
        """Like :meth:`finalize_with_generation`, plus an accepted flag and
        detected language.

        ``accepted`` is False when the manager discarded the finalize
        result because the stream was cancelled, no longer finalizable, or
        superseded by another generation. Callers that emit protocol frames
        should use this flag instead of comparing against mutable outer
        connection state.

        ``detected_language`` is the per-utterance language reported by
        the ASR backend (e.g. ``"Chinese"``), or ``None`` if the backend
        does not perform language ID. Always ``None`` on discarded /
        no-op paths.
        """
        async with self._lock:
            if self._state not in (SessionState.ACTIVE,):
                return self._generation, "", False, None
            gen = self._generation
            self._state = SessionState.FINALIZING
            stream = self._stream
        if stream is None:
            async with self._lock:
                self._state = SessionState.IDLE
            return gen, "", False, None
        try:
            raw = await self._run_sync(stream.finalize)
        except Exception as exc:  # noqa: BLE001
            async with self._lock:
                await self._handle_error_locked(exc)
            return gen, "", False, None
        # Backends MUST return ``(text, language)``. Per the ASRStream
        # ABC contract; missed migrations should fail loudly (TypeError
        # at unpack) rather than silently degrade to ``language=None``.
        final_text, detected_language = raw
        async with self._lock:
            # If a cancel raced past us (e.g. abort-during-FINALIZING),
            # the cancel routine will have moved us into CANCELLING and
            # already discarded the result. Honor that.
            if self._state != SessionState.FINALIZING:
                logger.info("ASRSessionManager: finalize result discarded (state=%s)", self._state)
                return gen, "", False, None
            # Also drop if the generation advanced (a new speech_start
            # raced past us via _inner_cancel/preempt).
            if self._generation != gen:
                logger.info(
                    "ASRSessionManager: finalize result discarded (stale gen %d != current %d)",
                    gen, self._generation,
                )
                return gen, "", False, None
            _safe_close_stream(self._stream)
            self._stream = None
            self._state = SessionState.IDLE
            return gen, final_text or "", True, detected_language

    async def get_partial_for_generation(self) -> tuple[int, str, bool]:
        """Snapshot ``(generation, partial_text, is_endpoint)`` atomically.

        Returns ``(generation, "", False)`` if there's no active stream.
        Callers should compare ``generation`` against
        :attr:`current_generation` before emitting partials downstream
        to drop partials that leaked from a now-replaced stream.
        """
        async with self._lock:
            gen = self._generation
            stream = self._stream
            if stream is None or self._state != SessionState.ACTIVE:
                return gen, "", False
        try:
            partial, is_endpoint = await self._run_sync(stream.get_partial)
        except Exception:  # noqa: BLE001
            return gen, "", False
        return gen, partial or "", bool(is_endpoint)

    async def cancel(self, reason: str = "bargein") -> None:
        async with self._lock:
            await self._inner_cancel(reason=reason)

    async def _inner_cancel(self, *, reason: str) -> None:
        """Lock must be held by caller."""
        if self._state in (SessionState.IDLE,):
            return
        prev_state = self._state
        self._state = SessionState.CANCELLING
        stream = self._stream
        self._stream = None
        # close() is idempotent w.r.t. later cancel()/cancel_and_finalize() —
        # for paraformer_trt, close() only releases per-stream TRT contexts
        # and buffers, which are not needed for the cancel ack itself.
        # We defer close() until *after* the cancel call below so the cancel
        # path can still touch stream state if it needs to. See finally.
        if stream is None:
            self._state = SessionState.IDLE
            return

        # Run cancel_and_finalize() in executor with a hard timeout. If
        # the worker doesn't ack the end event within the budget, fall
        # back to a worker restart.
        def _cancel_call():
            try:
                # Prefer explicit cancel() if provided, else fall back to
                # cancel_and_finalize() (always present on the base class).
                if hasattr(stream, "cancel"):
                    stream.cancel()
                else:
                    stream.cancel_and_finalize()
            except Exception as exc:  # noqa: BLE001
                # Bubble up to async layer so we can decide restart vs swallow.
                raise exc

        loop = self._get_loop()
        fut = loop.run_in_executor(self._executor, _cancel_call)
        try:
            await asyncio.wait_for(fut, timeout=self._CANCEL_ACK_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "ASRSessionManager: cancel(%s) timed out from state=%s; restarting worker",
                reason, prev_state,
            )
            await self._maybe_restart_worker()
        except Exception as exc:  # noqa: BLE001
            if _is_worker_protocol_error(exc):
                logger.warning(
                    "ASRSessionManager: cancel(%s) raised worker error %s; restarting",
                    reason, type(exc).__name__,
                )
                await self._maybe_restart_worker()
            else:
                logger.info("ASRSessionManager: cancel(%s) swallowed exc=%s", reason, exc)
        _safe_close_stream(stream)
        self._state = SessionState.IDLE

    def mark_error(self, exc: BaseException) -> None:
        """Synchronous shim so accept_waveform threads / partial pollers
        can flag the manager. Defers to the next async tick to actually
        run the recovery."""
        self._last_error = exc
        # Schedule async handler if possible.
        try:
            loop = self._get_loop()
        except Exception:
            return
        if loop.is_running():
            asyncio.ensure_future(self._async_mark_error(exc), loop=loop)

    async def _async_mark_error(self, exc: BaseException) -> None:
        # Coalesce concurrent error notifications onto a single shared
        # recovery future so 5 simultaneous mark_error calls produce one
        # restart, not five. The lock check is cheap.
        fut: Optional[asyncio.Future] = None
        own_recovery = False
        # Critical section to claim/observe the shared future. We can't
        # use self._lock here because the recovery itself holds it; use
        # a small synchronous check on a dedicated coalescing flag set
        # transactionally with the future.
        if self._recovery_future is not None and not self._recovery_future.done():
            fut = self._recovery_future
        else:
            loop = self._get_loop()
            self._recovery_future = loop.create_future()
            own_recovery = True
            fut = self._recovery_future

        if not own_recovery:
            try:
                await fut
            except Exception:
                pass
            return

        try:
            async with self._lock:
                await self._handle_error_locked(exc)
        finally:
            if not fut.done():
                fut.set_result(None)
            self._recovery_future = None

    async def _handle_error_locked(self, exc: BaseException) -> None:
        self._last_error = exc
        # Coalesce: if another recovery is already running for this
        # session (e.g. 5 partial pollers all saw the same worker exit
        # and called mark_error), let the first one drive recovery
        # alone. Without this guard, each call would fire its own
        # restart_worker() pass on top of the 3-attempt retry schedule.
        if self._recovery_in_progress:
            return
        self._recovery_in_progress = True
        _safe_close_stream(self._stream)
        self._stream = None
        self._state = SessionState.ERROR_REBUILD
        if not _is_worker_protocol_error(exc):
            logger.info("ASRSessionManager: non-protocol error during ASR: %s", exc)
        try:
            await self._do_rebuild_locked()
        finally:
            self._recovery_in_progress = False

    async def _do_rebuild_locked(self) -> None:
        for attempt, delay in enumerate(self._REBUILD_BACKOFF_S):
            await asyncio.sleep(delay)
            try:
                self._stream = await self._create_stream()
                self._state = SessionState.ACTIVE
                logger.info(
                    "ASRSessionManager: ERROR_REBUILD recovered on attempt %d",
                    attempt + 1,
                )
                return
            except Exception as inner:
                logger.warning(
                    "ASRSessionManager: ERROR_REBUILD attempt %d failed: %s",
                    attempt + 1, inner,
                )
                self._last_error = inner
        # Exhausted retries → fall back to a full worker restart.
        await self._maybe_restart_worker()
        # One last attempt after restart; if it also fails, give up to IDLE.
        try:
            self._stream = await self._create_stream()
            self._state = SessionState.ACTIVE
        except Exception as inner:
            logger.warning("ASRSessionManager: post-restart create_stream failed: %s", inner)
            _safe_close_stream(self._stream)
            self._stream = None
            self._state = SessionState.IDLE

    async def _maybe_restart_worker(self) -> None:
        backend = self._backend
        fn = getattr(backend, "restart_worker", None)
        if fn is None:
            return
        # IMPORTANT: do NOT submit to ``self._executor``. That executor is
        # the single-thread ASR slot, which is exactly the thread that's
        # currently wedged on a stuck worker request. Queuing restart
        # behind it would deadlock. The default executor (``None``) is a
        # multi-thread pool and is always free, which is what we need to
        # forcibly kill the worker subprocess from outside the wedged
        # thread. See trt_edge_llm_asr.restart_worker for why it must
        # NOT acquire ``_worker_lock``.
        loop = self._get_loop()
        try:
            await loop.run_in_executor(None, fn)
            logger.info("ASRSessionManager: backend.restart_worker() completed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ASRSessionManager: restart_worker failed: %s", exc)
