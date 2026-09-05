import asyncio
import sys
import threading
import types

import pytest

from server.core.api_execution import execute_tts


class Lease:
    def __init__(self, events, value=None): self.events, self.value = events, value
    async def __aenter__(self): self.events.append("enter"); return self.value if self.value is not None else self
    async def __aexit__(self, *exc): self.events.append("exit")


class Coordinator:
    def __init__(self, events): self.events = events
    def acquire(self, _slot): return Lease(self.events)


class Manager:
    def __init__(self, backend, events): self.backend, self.events = backend, events
    def acquire(self): return Lease(self.events, self.backend)


class SlowBackend:
    name = "kokoro_convonly"
    sample_rate = 24000
    def __init__(self): self.started = threading.Event(); self.release = threading.Event(); self.thread = None
    def synthesize(self, **_):
        self.thread = threading.get_ident(); self.started.set(); self.release.wait(2); return b"wav", {"ok": True}

class FailingBackend(SlowBackend):
    def synthesize(self, **_):
        self.thread = threading.get_ident(); self.started.set(); self.release.wait(2); raise RuntimeError("native failed")


async def run_call(backend, manager=None, legacy=None):
    return await execute_tts(text="x", language="en", voice_kwargs={}, manager=manager, legacy_service=legacy, coordinator=Coordinator([]))


def test_kokoro_sync_synthesis_runs_off_loop_and_heartbeat_progresses():
    return asyncio.run(_test_kokoro_sync_synthesis_runs_off_loop_and_heartbeat_progresses())
async def _test_kokoro_sync_synthesis_runs_off_loop_and_heartbeat_progresses():
    backend = SlowBackend(); events=[]; manager=Manager(backend, events)
    task=asyncio.create_task(execute_tts(text="x", language="en", voice_kwargs={}, manager=manager, legacy_service=None, coordinator=Coordinator(events)))
    await asyncio.to_thread(backend.started.wait, 1)
    heartbeat=asyncio.Event(); asyncio.get_running_loop().call_soon(heartbeat.set)
    await asyncio.wait_for(heartbeat.wait(), 1)
    backend.release.set(); result=await task
    assert result.audio == b"wav" and backend.thread != threading.get_ident()


def test_cancellation_twice_waits_for_native_future_before_releasing_leases():
    return asyncio.run(_test_cancellation_twice_waits_for_native_future_before_releasing_leases())
async def _test_cancellation_twice_waits_for_native_future_before_releasing_leases():
    backend=SlowBackend(); events=[]; task=asyncio.create_task(execute_tts(text="x", language="en", voice_kwargs={}, manager=Manager(backend,events), legacy_service=None, coordinator=Coordinator(events)))
    await asyncio.to_thread(backend.started.wait, 1)
    task.cancel(); task.cancel(); await asyncio.sleep(0)
    assert events == ["enter", "enter"]
    backend.release.set()
    with pytest.raises(asyncio.CancelledError): await task
    assert events == ["enter", "enter", "exit", "exit"]


def test_cancel_drain_preserves_cancelled_error_when_native_fails():
    return asyncio.run(_test_cancel_drain_preserves_cancelled_error_when_native_fails())
async def _test_cancel_drain_preserves_cancelled_error_when_native_fails():
    backend=FailingBackend(); events=[]; task=asyncio.create_task(execute_tts(text="x", language="en", voice_kwargs={}, manager=Manager(backend,events), legacy_service=None, coordinator=Coordinator(events)))
    await asyncio.to_thread(backend.started.wait, 1)
    task.cancel(); await asyncio.sleep(0); task.cancel(); backend.release.set()
    with pytest.raises(asyncio.CancelledError): await task
    assert events == ["enter", "enter", "exit", "exit"]


def test_legacy_backend_keeps_inline_behavior():
    return asyncio.run(_test_legacy_backend_keeps_inline_behavior())
async def _test_legacy_backend_keeps_inline_behavior():
    class Legacy:
        def is_ready(self): return True
        def get_backend(self): return self
        name="legacy"
        def synthesize(self, **_): return b"wav", {}
    result=await run_call(Legacy(), legacy=Legacy())
    assert result.backend == "legacy"


@pytest.mark.parametrize("name", ["rk.tts", "rk:matcha", "legacy"])
def test_other_backends_never_receive_kokoro_cancel_event_or_start_watcher(name):
    async def run():
        loop_thread = threading.get_ident()

        class Backend:
            def synthesize(self, **kwargs):
                assert "cancel_event" not in kwargs
                assert threading.get_ident() == loop_thread
                return b"wav", {}

        backend = Backend()
        backend.name = name

        async def disconnect():
            pytest.fail("non-Kokoro backend started disconnect watcher")

        result = await execute_tts(
            text="x", language="en", voice_kwargs={}, manager=Manager(backend, []),
            legacy_service=None, coordinator=Coordinator([]), disconnect_awaitable=disconnect,
        )
        assert result.audio == b"wav"

    asyncio.run(run())


@pytest.mark.parametrize("trigger", ["disconnect", "watcher_error", "watcher_error_cancel_twice", "cancel_twice", "anyio_cancel", "normal", "false"])
def test_watcher_and_native_children_are_joined_before_leases_exit(trigger):
    import anyio
    from server.core.api_execution import _TransportDisconnected

    async def run():
        initial_tasks = asyncio.all_tasks()
        events = []
        native_cancel = threading.Event()
        watcher_started = asyncio.Event()
        watcher_signal = asyncio.Event()
        watcher_stopped = asyncio.Event()
        child_stopped = asyncio.Event()
        backend = SlowBackend()

        async def receive_child():
            try:
                await asyncio.Event().wait()
            finally:
                child_stopped.set()

        async def disconnect():
            try:
                async with anyio.create_task_group() as children:
                    children.start_soon(receive_child)
                    watcher_started.set()
                    await watcher_signal.wait()
                    children.cancel_scope.cancel()
                if trigger.startswith("watcher_error"):
                    raise RuntimeError("receive failed")
                return trigger != "false"
            finally:
                watcher_stopped.set()

        scope = anyio.CancelScope()

        async def call():
            with scope:
                return await execute_tts(
                    text="x", language="en", voice_kwargs={}, manager=Manager(backend, events),
                    legacy_service=None, coordinator=Coordinator(events), cancel_event=native_cancel,
                    disconnect_awaitable=disconnect,
                )

        task = asyncio.create_task(call())
        try:
            await asyncio.wait_for(watcher_started.wait(), 1)
            assert await asyncio.to_thread(backend.started.wait, 1)
            if trigger in {"disconnect", "watcher_error", "watcher_error_cancel_twice", "false"}:
                watcher_signal.set()
            elif trigger == "cancel_twice":
                task.cancel()
            elif trigger == "anyio_cancel":
                scope.cancel()
            if trigger not in {"normal", "false"}:
                assert await asyncio.to_thread(native_cancel.wait, 1)
                assert events == ["enter", "enter"]
                if trigger in {"cancel_twice", "watcher_error_cancel_twice"}:
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                assert not task.done()
            backend.release.set()
            if trigger == "disconnect":
                with pytest.raises(_TransportDisconnected):
                    await task
            elif trigger == "watcher_error":
                with pytest.raises(RuntimeError, match="receive failed"):
                    await task
            elif trigger in {"cancel_twice", "watcher_error_cancel_twice"}:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
            assert watcher_stopped.is_set() and child_stopped.is_set()
            assert events == ["enter", "enter", "exit", "exit"]
            await asyncio.sleep(0)
            assert asyncio.all_tasks() == initial_tasks
        finally:
            backend.release.set()
            watcher_signal.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_repeated_cancellation_during_watcher_cleanup_does_not_interrupt_its_owner():
    import anyio

    async def run():
        initial_tasks = asyncio.all_tasks()
        backend = SlowBackend()
        events = []
        watcher_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()
        cleanup_done = asyncio.Event()
        native_cancel = threading.Event()

        async def disconnect():
            try:
                watcher_started.set()
                await asyncio.Event().wait()
            finally:
                with anyio.CancelScope(shield=True):
                    cleanup_started.set()
                    await cleanup_release.wait()
                    cleanup_done.set()

        task = asyncio.create_task(execute_tts(
            text="x", language="en", voice_kwargs={}, manager=Manager(backend, events),
            legacy_service=None, coordinator=Coordinator(events), cancel_event=native_cancel,
            disconnect_awaitable=disconnect,
        ))
        try:
            await asyncio.wait_for(watcher_started.wait(), 1)
            backend.release.set()
            await asyncio.wait_for(cleanup_started.wait(), 1)
            task.cancel()
            assert await asyncio.to_thread(native_cancel.wait, 1)
            task.cancel()
            await asyncio.sleep(0)
            assert events == ["enter", "enter"]
            assert not task.done() and not cleanup_done.is_set()
            cleanup_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cleanup_done.is_set()
            assert events == ["enter", "enter", "exit", "exit"]
            await asyncio.sleep(0)
            assert asyncio.all_tasks() == initial_tasks
        finally:
            backend.release.set()
            cleanup_release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_simultaneous_native_failure_wins_over_disconnect(monkeypatch):
    from server.core import api_execution

    async def run():
        real_wait = asyncio.wait

        async def wait_for_both(tasks, **_kwargs):
            await asyncio.gather(*tasks, return_exceptions=True)
            return set(tasks), set()

        monkeypatch.setattr(asyncio, "wait", wait_for_both)

        async def disconnect():
            return True

        try:
            with pytest.raises(RuntimeError, match="native failed"):
                await api_execution._execute_kokoro_call(
                    lambda: (_ for _ in ()).throw(RuntimeError("native failed")),
                    threading.Event(),
                    disconnect,
                )
        finally:
            monkeypatch.setattr(asyncio, "wait", real_wait)

    asyncio.run(run())


@pytest.mark.parametrize("expected_cancel", [True, False])
def test_disconnect_joins_backend_and_classifies_late_failure(
    monkeypatch, caplog, expected_cancel
):
    from server.core.api_execution import _TransportDisconnected

    module_name = "rkvoice_stream.backends.tts.kokoro_convonly"
    cancelled_type = type(
        "ConvOnlyCancelled", (RuntimeError,), {"__module__": module_name}
    )
    monkeypatch.setitem(
        sys.modules,
        module_name,
        types.SimpleNamespace(ConvOnlyCancelled=cancelled_type),
    )

    async def run():
        cancel_event = threading.Event()

        def backend_call():
            assert cancel_event.wait(1)
            if expected_cancel:
                raise cancelled_type("submitted work drained")
            raise RuntimeError("late native failure")

        async def disconnect():
            return True

        with pytest.raises(_TransportDisconnected):
            await execute_tts(
                text="x",
                language="en",
                voice_kwargs={},
                manager=Manager(
                    type(
                        "Backend",
                        (),
                        {
                            "name": "rk:kokoro_convonly",
                            "sample_rate": 24000,
                            "synthesize": staticmethod(lambda **_: backend_call()),
                        },
                    )(),
                    [],
                ),
                legacy_service=None,
                coordinator=Coordinator([]),
                cancel_event=cancel_event,
                disconnect_awaitable=disconnect,
            )

    asyncio.run(run())
    failures = [
        record
        for record in caplog.records
        if record.getMessage() == "finite TTS backend failed after client disconnect"
    ]
    assert len(failures) == (0 if expected_cancel else 1)
    if failures:
        assert failures[0].exc_info[0] is RuntimeError


def test_convonly_cancel_match_requires_exact_registered_class(monkeypatch):
    from server.core.api_execution import _is_kokoro_convonly_cancelled

    module_name = "rkvoice_stream.backends.tts.kokoro_convonly"
    exact = type("ConvOnlyCancelled", (RuntimeError,), {"__module__": module_name})
    subclass = type("ConvOnlyCancelled", (exact,), {"__module__": module_name})
    same_name = type("ConvOnlyCancelled", (RuntimeError,), {"__module__": module_name})
    monkeypatch.setitem(
        sys.modules,
        module_name,
        types.SimpleNamespace(ConvOnlyCancelled=exact),
    )

    assert _is_kokoro_convonly_cancelled(exact("cancelled"))
    assert not _is_kokoro_convonly_cancelled(subclass("cancelled"))
    assert not _is_kokoro_convonly_cancelled(same_name("cancelled"))
    assert not _is_kokoro_convonly_cancelled(RuntimeError("cancelled"))
