"""Generic subprocess-worker IO multiplexer.

Extracted from ``app/backends/jetson/trt_edge_llm_tts.py`` (commit `64185fa`
cooperative cancel protocol; spec ``docs/specs/concurrency-capability-framework.md``
Section 5).

This is the framework-layer abstraction that demuxes a single JSON-line
subprocess (one stdin / one stdout) into N in-flight per-request streams,
keyed by ``request_id`` / ``id``. Used today by the TRT-Edge-LLM TTS
backend. Future ASR worker backends (spec P1+) will reuse this.

Public API per spec Section 5:

    wio = WorkerIO(proc, concurrency)

    # Async (spec target — preferred for new code, e.g. future ASR worker)
    async for event in wio.send_request(rid, payload):
        ...
    wio.cancel(rid)
    wio.close()

    # Sync (legacy shim retained for existing TTS path that runs inside a
    # ThreadPoolExecutor; the surrounding generator-of-PCM-chunks pipeline
    # cannot trivially flip to async without restructuring app/main.py's
    # streaming response wiring. Will be removed once that migration lands).
    for event in wio.request(payload):
        ...

Both APIs share the same underlying ``_inflight`` map, ``_stdin_lock``,
reader thread, and semaphore, so they coexist safely on the same instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import subprocess
import threading
from typing import AsyncIterator, Iterator

logger = logging.getLogger(__name__)


class WorkerExitError(RuntimeError):
    """Raised when the worker subprocess dies while a request is in flight."""


class PoolSaturatedError(RuntimeError):
    """Raised when a non-queueing WorkerIO has no free request slot."""

    status = 4429

    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        super().__init__(f"worker request pool saturated (max_slots={max_slots})")


class WorkerIO:
    """Per-worker stdin writer + stdout reader thread, multiplexing N in-flight requests.

    Replaces a coarse per-request lock (which would serialize full
    request→response cycles end-to-end) with:

      * a single ``_stdin_lock`` that only protects the single-line JSON write,
      * a daemon reader thread that demuxes stdout events to per-request
        ``queue.Queue`` instances keyed by ``request_id``/``id``,
      * a ``threading.Semaphore`` bounding in-flight requests to ``concurrency``.

    When the worker subprocess EOFs (crash / restart), the reader thread wakes
    every in-flight caller with a sentinel ``{"event": "_worker_exit"}`` so
    they raise ``WorkerExitError`` instead of hanging on ``q.get(timeout=...)``.

    A given ``WorkerIO`` instance is bound to ONE subprocess. To restart the
    worker, discard the old instance and create a new one (handled by the
    owning backend, e.g. ``_ensure_worker`` / ``_restart_worker``).
    """

    # Class-level temporary instrumentation for Part D disconnect-watcher
    # validation (internal throughput spec §3). Counts every
    # cancel() invocation across all WorkerIO instances since process
    # start. Surfaced via the /health endpoint and via debug log. Remove
    # once Part D is validated and stable.
    _cancel_count: int = 0
    _cancel_count_lock = threading.Lock()

    def __init__(self, proc: subprocess.Popen, concurrency: int):
        self._proc = proc
        self._stdin_lock = threading.Lock()
        self._inflight: dict[str, "queue.Queue"] = {}
        self._inflight_lock = threading.Lock()
        self._max_slots = max(1, int(concurrency))
        self._sem = threading.Semaphore(self._max_slots)
        self._queue_when_saturated = (
            os.environ.get("OVS_WORKER_IO_QUEUE_WHEN_SATURATED", "1")
            .strip()
            .lower()
            not in {"0", "false", "no", "off"}
        )
        # Set by close(); requests that acquire the semaphore after this
        # is True must abort instead of writing to a dead worker stdin.
        self._closed = False
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="worker-io-stdout",
            daemon=True,
        )
        self._reader_thread.start()

    async def send_request(
        self, request_id: str, payload: dict
    ) -> AsyncIterator[dict]:
        """Async-iterate worker events for one request.

        Spec section 5 target API. The semaphore + per-request queue +
        sentinel-on-worker-exit semantics match ``request()`` exactly; the
        only difference is awaitable ``q.get`` (offloaded to the default
        thread executor) so callers integrate into an asyncio event loop
        without blocking it.

        If the consumer breaks out of the ``async for`` (or ``aclose()`` is
        invoked), the ``finally`` arm fires ``cancel(request_id)`` so the
        worker emits a terminal ``cancelled`` event. This mirrors the
        ``GeneratorExit`` arm in the sync streaming caller at
        ``trt_edge_llm_tts.py``.
        """
        # Ensure payload carries the request_id the caller passed in.
        payload = dict(payload)
        payload.setdefault("id", request_id)
        if payload.get("id") != request_id:
            raise ValueError(
                f"payload['id']={payload.get('id')!r} != request_id={request_id!r}"
            )

        loop = asyncio.get_running_loop()
        # Run the blocking semaphore acquire off the loop. The semaphore is
        # a back-pressure gate; it can block when concurrency is saturated.
        # If the awaiting task is cancelled while the executor thread is
        # still blocked in self._sem.acquire(), the thread cannot be
        # interrupted — it may eventually grab the token after we're gone.
        # Attach a done-callback so the late-acquired token is released
        # back to the pool instead of leaking the slot.
        if not self._queue_when_saturated:
            if not self._sem.acquire(blocking=False):
                raise PoolSaturatedError(self._max_slots)
        else:
            _fut = loop.run_in_executor(None, self._sem.acquire)
            try:
                await _fut
            except BaseException:
                def _release_if_late_acquire(f: "asyncio.Future") -> None:
                    # f.result() can raise CancelledError if the wrapper
                    # future itself was cancelled before the executor thread
                    # ran. Catch BaseException (not Exception) so we don't
                    # propagate a cancel-trace out of a done-callback.
                    try:
                        if f.result() is True:
                            self._sem.release()
                    except BaseException:
                        pass
                _fut.add_done_callback(_release_if_late_acquire)
                raise

        # Hold semaphore from here. If close() ran while we were waiting,
        # abort immediately and release the slot back; do not register
        # inflight and do not write to dead worker stdin.
        # NB: this is a fast-path check. The authoritative gate is the
        # `_closed` re-check inside `_stdin_lock` below — close() also
        # sets `_closed` under `_stdin_lock`, so any stdin write and
        # any close() call are strictly ordered.
        if self._closed:
            self._sem.release()
            raise WorkerExitError("WorkerIO closed before request could start")

        q: "queue.Queue" = queue.Queue()
        with self._inflight_lock:
            self._inflight[request_id] = q

        # Track whether the request reached a natural terminal event
        # (``done``/``cancelled``). Any other exit path — task cancel while
        # awaiting q.get, GeneratorExit at yield, TimeoutError, WorkerExitError,
        # arbitrary exception — must send a cancel JSON to the worker so it
        # stops producing chunks for an abandoned slot. The cancel call itself
        # is best-effort (worker may already be dead).
        finished_naturally = False
        try:
            assert self._proc.stdin is not None
            try:
                with self._stdin_lock:
                    # Re-check under the same lock that close() uses to set
                    # the flag; this closes the TOCTOU window between the
                    # fast-path check above and the actual write.
                    if self._closed:
                        raise WorkerExitError(
                            "WorkerIO closed between acquire and stdin write"
                        )
                    self._proc.stdin.write(
                        json.dumps(payload, ensure_ascii=False) + "\n"
                    )
                    self._proc.stdin.flush()
            except Exception:
                with self._inflight_lock:
                    self._inflight.pop(request_id, None)
                raise

            while True:
                try:
                    event = await loop.run_in_executor(None, q.get, True, 60.0)
                except queue.Empty:
                    raise TimeoutError(
                        f"WorkerIO.send_request: no event for {request_id} in 60s"
                    )
                if event.get("event") == "_worker_exit":
                    raise WorkerExitError("worker subprocess died mid-request")
                yield event
                if event.get("event") in ("done", "cancelled"):
                    finished_naturally = True
                    return
        finally:
            with self._inflight_lock:
                self._inflight.pop(request_id, None)
            self._sem.release()
            if not finished_naturally:
                try:
                    self.cancel(request_id)
                except Exception:
                    logger.debug(
                        "cancel() during async cleanup failed", exc_info=True
                    )

    def close(self) -> None:
        """Tear down: wake every in-flight caller and stop the reader.

        After ``close()``, in-flight ``send_request``/``request`` generators
        observe a ``_worker_exit`` sentinel and surface ``WorkerExitError``.
        Sets a closed flag so that any request currently blocked in
        ``self._sem.acquire()`` (saturation back-pressure) will abort
        on wake-up instead of writing to a dead worker's stdin. The
        reader thread itself is daemon and will exit naturally when
        the subprocess stdout EOFs; ``close()`` does not kill the
        subprocess — that is the owning backend's responsibility (proc
        lifecycle stays where it is today).
        """
        # Set _closed under _stdin_lock so any in-flight stdin write that
        # holds the lock observes the flag before the next acquirer can.
        # Combined with the re-check inside send_request/request under
        # the same lock, this guarantees no write happens after close().
        with self._stdin_lock:
            self._closed = True
        with self._inflight_lock:
            queues = list(self._inflight.values())
            self._inflight.clear()
        for q in queues:
            q.put({"event": "_worker_exit"})

    def request(self, payload: dict) -> Iterator[dict]:
        """Send ``payload`` to the worker and yield response events until ``done``.

        Caller must include a unique ``id`` field in ``payload``. The generator
        terminates when an ``event=="done"`` or ``event=="cancelled"`` is
        received, or raises ``WorkerExitError`` if the worker dies mid-request.
        """
        # Wrap the entire body in try/finally so the semaphore is always
        # released regardless of where we exit (including failures in
        # setup between acquire and the inner try).
        if not self._queue_when_saturated:
            if not self._sem.acquire(blocking=False):
                raise PoolSaturatedError(self._max_slots)
        else:
            self._sem.acquire()
        req_id: str | None = None
        try:
            # Mirror the async path: if close() ran while we waited on
            # the semaphore, abort instead of writing to a dead worker.
            if self._closed:
                raise WorkerExitError("WorkerIO closed before request could start")
            req_id = payload["id"]
            q: "queue.Queue" = queue.Queue()
            # CRITICAL ordering: insert the queue BEFORE writing stdin so
            # the reader thread can never observe an event for ``req_id``
            # before the queue exists (would otherwise be dropped as "stale").
            with self._inflight_lock:
                self._inflight[req_id] = q
            assert self._proc.stdin is not None
            with self._stdin_lock:
                # Re-check _closed under stdin_lock — close() also sets
                # the flag under the same lock, so this closes the TOCTOU
                # gap between the fast-path check above and the write.
                if self._closed:
                    raise WorkerExitError(
                        "WorkerIO closed between acquire and stdin write"
                    )
                self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            while True:
                event = q.get(timeout=60.0)
                if event.get("event") == "_worker_exit":
                    raise WorkerExitError("worker subprocess died mid-request")
                yield event
                if event.get("event") in ("done", "cancelled"):
                    return
        finally:
            if req_id is not None:
                with self._inflight_lock:
                    self._inflight.pop(req_id, None)
            self._sem.release()

    def cancel(self, req_id: str) -> None:
        """Best-effort cancel for an in-flight request.

        Writes a cancel JSON to the worker's stdin. The worker will check
        its per-request atomic flag at the next chunk boundary and emit
        a ``{"event":"cancelled", ...}`` terminal event in lieu of ``done``.

        Safe to call from any thread. Safe to call after the request has
        naturally completed (worker silently drops unknown cancels).
        """
        with WorkerIO._cancel_count_lock:
            WorkerIO._cancel_count += 1
            count_snapshot = WorkerIO._cancel_count
        logger.info(
            "WorkerIO.cancel: req_id=%s total_cancel_count=%d",
            req_id,
            count_snapshot,
        )
        try:
            assert self._proc.stdin is not None
            with self._stdin_lock:
                # Skip writing if close() has set _closed under the same
                # lock — the worker stdin is already considered dead.
                # Avoids a guaranteed-fail write and the noisy "broken
                # pipe" debug trace that the catch below would emit.
                if self._closed:
                    return
                self._proc.stdin.write(
                    json.dumps({"type": "cancel", "id": req_id}) + "\n"
                )
                self._proc.stdin.flush()
        except Exception:
            # Worker may have already exited; the reader-loop sentinel
            # will surface that via WorkerExitError on the next q.get().
            logger.debug(
                "cancel() write failed; worker may be exiting",
                exc_info=True,
            )

    def _reader_loop(self) -> None:
        """Drain worker stdout, dispatching events to per-request queues."""
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    logger.debug("worker emitted non-JSON line: %r", line[:200])
                    continue
                rid = event.get("request_id") or event.get("id")
                with self._inflight_lock:
                    q = self._inflight.get(rid) if rid else None
                if q is not None:
                    q.put(event)
                # else: stale / unsolicited (e.g. spurious "ready" replays
                # or events after a restart). Drop silently.
        except Exception:
            logger.exception("worker stdout reader crashed")
        finally:
            # Worker died (or stdout closed). Wake every in-flight caller
            # with the sentinel so they raise WorkerExitError instead of
            # hanging on q.get(timeout=60).
            with self._inflight_lock:
                for q in self._inflight.values():
                    q.put({"event": "_worker_exit"})
                self._inflight.clear()


# Backwards-compat alias for callers still referencing the old private name.
# Will be removed once all internal references migrate to ``WorkerIO``.
_WorkerIO = WorkerIO
