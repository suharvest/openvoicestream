"""Unit tests for WorkerIO (spec docs/specs/concurrency-capability-framework.md §5).

Covers:
  * sync ``request()`` legacy shim still works (commit 1a contract)
  * async ``send_request()`` demuxes events keyed by request_id
  * EOF on stdout -> in-flight callers get WorkerExitError
  * semaphore release on success AND on exception path
  * cancel() writes a cancel JSON to worker stdin
  * close() wakes in-flight callers with the _worker_exit sentinel
  * async iterator cancellation (consumer breaks early -> cancel() fired)
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
from typing import Iterable

import pytest

from server.core.worker_io import PoolSaturatedError, WorkerIO, WorkerExitError


def asynctest(fn):
    """Decorator turning an async test fn into a sync one (no pytest-asyncio)."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(fn(*args, **kwargs))
        finally:
            loop.close()
    wrapper.__name__ = fn.__name__
    return wrapper


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.closed = False
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            self.writes.append(s)
        return len(s)

    def flush(self) -> None:
        pass


class _FakeStdoutQueue:
    """Iterable stdout fed by a queue.Queue; closes on None sentinel."""

    def __init__(self) -> None:
        import queue as _q
        self._q: "_q.Queue[str | None]" = _q.Queue()

    def feed(self, line: str) -> None:
        self._q.put(line if line.endswith("\n") else line + "\n")

    def eof(self) -> None:
        self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdoutQueue()


def test_sync_nonqueueing_pool_rejects_immediately(monkeypatch):
    monkeypatch.setenv("OVS_WORKER_IO_QUEUE_WHEN_SATURATED", "0")
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)
    assert wio._sem.acquire(blocking=False)
    with pytest.raises(PoolSaturatedError) as caught:
        list(wio.request({"id": "overflow"}))
    assert caught.value.status == 4429
    assert caught.value.max_slots == 1
    assert proc.stdin.writes == []
    wio._sem.release()


@asynctest
async def test_async_nonqueueing_pool_rejects_immediately():
    previous = os.environ.get("OVS_WORKER_IO_QUEUE_WHEN_SATURATED")
    os.environ["OVS_WORKER_IO_QUEUE_WHEN_SATURATED"] = "false"
    try:
        proc = _FakeProc()
        wio = WorkerIO(proc, concurrency=1)
        assert wio._sem.acquire(blocking=False)
        with pytest.raises(PoolSaturatedError) as caught:
            await anext(wio.send_request("overflow", {"text": "busy"}))
        assert caught.value.status == 4429
        assert caught.value.max_slots == 1
        assert proc.stdin.writes == []
        wio._sem.release()
    finally:
        if previous is None:
            os.environ.pop("OVS_WORKER_IO_QUEUE_WHEN_SATURATED", None)
        else:
            os.environ["OVS_WORKER_IO_QUEUE_WHEN_SATURATED"] = previous


# ---------------------------------------------------------------------------
# Sync request() — preserved legacy shim
# ---------------------------------------------------------------------------

def test_sync_request_demuxes_by_request_id():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=2)
    payload = {"id": "req-A", "text": "hi"}

    def _feeder():
        time.sleep(0.05)
        proc.stdout.feed(json.dumps({"id": "req-B", "event": "chunk", "audio": "ignore"}))
        proc.stdout.feed(json.dumps({"id": "req-A", "event": "chunk", "audio": "a1"}))
        proc.stdout.feed(json.dumps({"id": "req-A", "event": "done", "ok": True}))

    threading.Thread(target=_feeder, daemon=True).start()
    events = list(wio.request(payload))
    kinds = [e["event"] for e in events]
    assert kinds == ["chunk", "done"]
    assert events[0]["audio"] == "a1"
    # stdin saw the JSON line for req-A
    assert any("req-A" in w for w in proc.stdin.writes)


def test_sync_request_worker_exit_sentinel_raises():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)

    def _kill():
        time.sleep(0.05)
        proc.stdout.eof()

    threading.Thread(target=_kill, daemon=True).start()
    with pytest.raises(WorkerExitError):
        for _ in wio.request({"id": "req-1"}):
            pass


# ---------------------------------------------------------------------------
# Async send_request()
# ---------------------------------------------------------------------------

@asynctest
async def test_async_send_request_basic():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=2)

    def _feeder():
        time.sleep(0.02)
        proc.stdout.feed(json.dumps({"id": "rid-1", "event": "chunk", "audio": "x"}))
        proc.stdout.feed(json.dumps({"id": "rid-1", "event": "done", "ok": True}))

    threading.Thread(target=_feeder, daemon=True).start()
    events = []
    async for ev in wio.send_request("rid-1", {"id": "rid-1", "text": "hello"}):
        events.append(ev)
    assert [e["event"] for e in events] == ["chunk", "done"]


@asynctest
async def test_async_send_request_id_mismatch_rejected():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)
    with pytest.raises(ValueError):
        async for _ in wio.send_request("rid-A", {"id": "rid-B"}):
            break


@asynctest
async def test_async_send_request_consumer_break_fires_cancel():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)

    def _feeder():
        time.sleep(0.02)
        proc.stdout.feed(json.dumps({"id": "rid-X", "event": "chunk", "audio": "a"}))
        # NOTE: never send done — simulates ongoing stream.

    threading.Thread(target=_feeder, daemon=True).start()
    gen = wio.send_request("rid-X", {"id": "rid-X"})
    first = await gen.__anext__()
    assert first["event"] == "chunk"
    await gen.aclose()
    # cancel() should have written a {"type":"cancel","id":"rid-X"} to stdin.
    cancel_writes = [w for w in proc.stdin.writes if '"type": "cancel"' in w or '"type":"cancel"' in w]
    assert cancel_writes, f"expected a cancel write, got {proc.stdin.writes!r}"
    assert "rid-X" in cancel_writes[-1]


@asynctest
async def test_async_send_request_cancel_while_awaiting_fires_cancel():
    """Task cancellation during q.get() must also send cancel JSON to worker.

    Regression: prior `cancelled_by_consumer` flag was only set inside the
    yield, so cancellation while blocked on q.get (no event ready) would
    leak the worker slot — worker keeps producing chunks for an abandoned
    request. Per Codex P1 review MUST_FIX.
    """
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)

    async def consume():
        # Worker never feeds any event for this rid — generator stays
        # blocked on q.get. We cancel the surrounding task externally.
        async for _ in wio.send_request("rid-W", {"id": "rid-W"}):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)  # let task enter q.get
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    cancel_writes = [
        w for w in proc.stdin.writes
        if '"type": "cancel"' in w or '"type":"cancel"' in w
    ]
    assert cancel_writes, f"expected a cancel write, got {proc.stdin.writes!r}"
    assert "rid-W" in cancel_writes[-1]
    # Semaphore must also be released.
    assert wio._sem._value == 1, "semaphore leaked after await-cancel"


@asynctest
async def test_async_two_concurrent_streams_aclose_simultaneously():
    """N=2 concurrent streams both aclose at the same time release both slots.

    Models the production case where two WS clients drop mid-stream. Both
    cancel JSONs must reach worker stdin and the semaphore must return to
    its full count so a third request can proceed without deadlock.
    """
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=2)

    # Use longer feeder delay so both generators have time to register
    # in _inflight before chunks arrive — reader drops events for unknown
    # rids silently (realistic behavior: worker only emits for known
    # requests).
    def _feeder(rid: str):
        time.sleep(0.15)
        proc.stdout.feed(json.dumps({"id": rid, "event": "chunk", "audio": "a"}))

    gen_a = wio.send_request("rid-A", {"id": "rid-A"})
    gen_b = wio.send_request("rid-B", {"id": "rid-B"})

    # Kick off both anext concurrently so both register in _inflight
    # before either feeder fires.
    task_a = asyncio.create_task(gen_a.__anext__())
    task_b = asyncio.create_task(gen_b.__anext__())
    await asyncio.sleep(0.05)  # let both register inflight

    threading.Thread(target=_feeder, args=("rid-A",), daemon=True).start()
    threading.Thread(target=_feeder, args=("rid-B",), daemon=True).start()

    first_a = await asyncio.wait_for(task_a, timeout=2.0)
    first_b = await asyncio.wait_for(task_b, timeout=2.0)
    assert first_a["event"] == "chunk" and first_b["event"] == "chunk"

    await asyncio.gather(gen_a.aclose(), gen_b.aclose())

    cancel_writes = [
        w for w in proc.stdin.writes
        if '"type": "cancel"' in w or '"type":"cancel"' in w
    ]
    cancelled_rids = {rid for rid in ("rid-A", "rid-B")
                      if any(rid in w for w in cancel_writes)}
    assert cancelled_rids == {"rid-A", "rid-B"}, (
        f"expected cancel for both, got {cancel_writes!r}"
    )
    assert wio._sem._value == 2, "semaphore did not fully release after dual aclose"


@asynctest
async def test_async_cancel_during_saturated_acquire_does_not_leak_semaphore():
    """Cancellation during sem acquire (saturated) must not leak the slot.

    Regression: prior code put `await loop.run_in_executor(None, sem.acquire)`
    outside the try/finally. When N slots are saturated and a new
    send_request awaits acquire, then is cancelled, the executor thread
    cannot be interrupted and may eventually grab the token from a
    coroutine that no longer exists -> permanent slot leak.

    Per Codex P1 final-review MUST_FIX. Fix attaches a done-callback to
    the acquire future that releases the late-acquired token.
    """
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)

    # Hold the only slot with a long-running first request.
    def _feeder_long():
        time.sleep(0.5)
        proc.stdout.feed(json.dumps({"id": "rid-hold", "event": "done", "ok": True}))

    threading.Thread(target=_feeder_long, daemon=True).start()
    gen1 = wio.send_request("rid-hold", {"id": "rid-hold"})
    task1 = asyncio.create_task(gen1.__anext__())
    await asyncio.sleep(0.05)  # let it acquire + register inflight
    assert wio._sem._value == 0, "slot 1 should be taken"

    # Second request blocks on acquire (saturated).
    gen2 = wio.send_request("rid-wait", {"id": "rid-wait"})
    task2 = asyncio.create_task(gen2.__anext__())
    await asyncio.sleep(0.05)  # let it start waiting on acquire
    task2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task2

    # Drain task1 to release its slot.
    await asyncio.wait_for(task1, timeout=2.0)
    try:
        await gen1.__anext__()
    except StopAsyncIteration:
        pass

    # Now wait for the executor thread to actually wake up + late-release.
    # The blocked acquire() call can only return once a token becomes
    # available, which is the moment task1's finally releases it.
    deadline = time.time() + 2.0
    while wio._sem._value < 1 and time.time() < deadline:
        await asyncio.sleep(0.05)

    assert wio._sem._value == 1, (
        f"semaphore leaked after cancel-during-acquire (sem={wio._sem._value})"
    )


@asynctest
async def test_close_while_request_blocked_on_acquire_aborts_cleanly():
    """close() while a request is waiting on saturated acquire must abort
    the request with WorkerExitError, not write to dead worker stdin.

    Per Codex P1 third-pass MUST_FIX: previously close() did not set a
    closed flag, so the blocked request would wake up, register inflight,
    and write payload JSON to a worker stdin that may already be closed.
    """
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)

    # Saturate the only slot with a request that never completes.
    gen_hold = wio.send_request("rid-hold", {"id": "rid-hold"})
    task_hold = asyncio.create_task(gen_hold.__anext__())
    await asyncio.sleep(0.05)

    # Second request will block on acquire.
    gen_wait = wio.send_request("rid-wait", {"id": "rid-wait"})

    async def consume_wait():
        with pytest.raises(WorkerExitError):
            async for _ in gen_wait:
                pass

    task_wait = asyncio.create_task(consume_wait())
    await asyncio.sleep(0.05)

    # close() should cancel task_hold (worker_exit) AND signal task_wait
    # to abort when its acquire wakes up.
    wio.close()

    # task_hold gets _worker_exit
    with pytest.raises(WorkerExitError):
        await asyncio.wait_for(task_hold, timeout=2.0)
    # task_wait must also raise WorkerExitError (closed flag honored)
    await asyncio.wait_for(task_wait, timeout=2.0)

    # No stdin writes should reference rid-wait — closed flag must prevent
    # the abandoned request from posting its payload.
    rid_wait_writes = [w for w in proc.stdin.writes if "rid-wait" in w and "cancel" not in w]
    assert not rid_wait_writes, (
        f"closed WorkerIO still wrote payload for blocked request: {rid_wait_writes!r}"
    )


@asynctest
async def test_async_send_request_worker_exit():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)

    def _kill():
        time.sleep(0.02)
        proc.stdout.eof()

    threading.Thread(target=_kill, daemon=True).start()
    with pytest.raises(WorkerExitError):
        async for _ in wio.send_request("rid-K", {"id": "rid-K"}):
            pass


# ---------------------------------------------------------------------------
# Semaphore release semantics
# ---------------------------------------------------------------------------

@asynctest
async def test_semaphore_released_on_success():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)

    async def run_one(rid: str):
        def _feeder():
            time.sleep(0.01)
            proc.stdout.feed(json.dumps({"id": rid, "event": "done", "ok": True}))
        threading.Thread(target=_feeder, daemon=True).start()
        async for _ in wio.send_request(rid, {"id": rid}):
            pass

    # Sequential successes must not deadlock — sem must be released.
    await asyncio.wait_for(run_one("a"), timeout=2.0)
    await asyncio.wait_for(run_one("b"), timeout=2.0)
    await asyncio.wait_for(run_one("c"), timeout=2.0)


@asynctest
async def test_semaphore_released_on_worker_exit():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)

    def _kill():
        time.sleep(0.02)
        proc.stdout.eof()
    threading.Thread(target=_kill, daemon=True).start()

    with pytest.raises(WorkerExitError):
        async for _ in wio.send_request("rid-1", {"id": "rid-1"}):
            pass

    # If semaphore had leaked, a second request would hang. Use a tight
    # timeout to assert no leak; we expect the second request to ALSO
    # raise WorkerExitError quickly (sentinel already drained).
    # Fresh inflight insertion + write would race with closed pipe; we
    # simply confirm the acquire returns quickly.
    acquired = await asyncio.wait_for(
        asyncio.get_running_loop().run_in_executor(None, wio._sem.acquire),
        timeout=1.0,
    )
    assert acquired is True


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

@asynctest
async def test_close_wakes_inflight_with_worker_exit():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=2)

    async def run():
        with pytest.raises(WorkerExitError):
            async for _ in wio.send_request("rid-close", {"id": "rid-close"}):
                pass

    task = asyncio.create_task(run())
    # give it time to register inflight
    await asyncio.sleep(0.05)
    wio.close()
    await asyncio.wait_for(task, timeout=2.0)


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------

def test_cancel_writes_cancel_json():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)
    wio.cancel("rid-Z")
    matching = [w for w in proc.stdin.writes if "cancel" in w and "rid-Z" in w]
    assert matching, proc.stdin.writes


def test_cancel_after_close_is_silent_noop():
    """cancel() after close() must not attempt stdin write.

    Codex P1 final-review NIT: post-close cancel previously tried the
    write and relied on the broken-pipe catch to swallow it, emitting a
    noisy debug trace. Now cancel() returns silently if _closed is set.
    """
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)
    wio.close()
    writes_before = list(proc.stdin.writes)
    wio.cancel("rid-after-close")
    # No new stdin activity should appear from cancel().
    assert proc.stdin.writes == writes_before, (
        f"cancel() wrote after close(): new writes = "
        f"{proc.stdin.writes[len(writes_before):]!r}"
    )
