"""Finite cancellation through real Starlette BaseHTTPMiddleware receive chains."""

import asyncio
import json
import logging
import time

from fastapi import FastAPI, Request, Response
import pytest

from server.core.api_execution import _TransportDisconnected, execute_tts
from server.tests.test_api_execution_offload import Coordinator, Manager, SlowBackend


@pytest.mark.parametrize("middleware_layers", [1, 3])
def test_normal_finish_through_http_middleware_does_not_wait_for_disconnect(middleware_layers):
    async def run():
        events = []
        received = asyncio.Queue()
        await received.put({"type": "http.request", "body": b"{}", "more_body": False})
        sent = []
        watcher_stopped = asyncio.Event()

        class Backend(SlowBackend):
            def synthesize(self, **kwargs):
                time.sleep(0.02)
                return b"wav", {}

        app = FastAPI()

        async def middleware(request, call_next):
            return await call_next(request)

        for _ in range(middleware_layers):
            app.middleware("http")(middleware)

        @app.post("/tts")
        async def endpoint(request: Request):
            await request.body()

            async def disconnect():
                try:
                    while True:
                        if (await request.receive())["type"] == "http.disconnect":
                            return True
                finally:
                    watcher_stopped.set()

            result = await execute_tts(
                text="x", language="en", voice_kwargs={},
                manager=Manager(Backend(), events), legacy_service=None,
                coordinator=Coordinator(events), disconnect_awaitable=disconnect,
            )
            return Response(result.audio)

        async def send(message):
            sent.append(message)

        task = asyncio.create_task(app(
            {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
             "method": "POST", "path": "/tts", "raw_path": b"/tts",
             "root_path": "", "scheme": "http", "query_string": b"",
             "headers": [(b"content-type", b"application/json")],
             "client": ("127.0.0.1", 123), "server": ("localhost", 80),
             "http_version": "1.1"}, received.get, send,
        ))
        try:
            done, _ = await asyncio.wait([task], timeout=0.3)
            assert task in done, f"response deadlocked: sent={sent}, leases={events}"
            await task
            assert watcher_stopped.is_set()
            assert events == ["enter", "enter", "exit", "exit"]
            assert any(item.get("body") == b"wav" for item in sent)
        finally:
            await received.put({"type": "http.disconnect"})
            await asyncio.wait_for(task, 2)

    asyncio.run(run())


@pytest.fixture
def finite_app(monkeypatch):
    from server import main
    from server.core import coordinator, session_limiter

    events = []

    class Backend(SlowBackend):
        name = "rk:kokoro_convonly"
        model_id = "kokoro-test"
        capabilities = set()
        supports_voice_cloning = False

        def __init__(self):
            super().__init__()
            self.cancel_event = None

        def rate_pitch_caps(self):
            return False, False

        def synthesize(self, **kwargs):
            self.cancel_event = kwargs["cancel_event"]
            self.started.set()
            assert self.release.wait(2), "test did not release native work"
            return b"wav", {"backend": "kokoro_convonly", "engine": "mixed", "fallback": True,
                            "segments": [{"engine": "npu"}, {"engine": "cpu"}],
                            "segment_count": 2, "cpu_generator_ms": 7.5}

    backend = Backend()

    async def ensure():
        return Manager(backend, events)

    monkeypatch.setattr(main, "_ensure_tts_manager_started", ensure)
    monkeypatch.setattr(main, "_request_voice_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(main, "_v1_resolve_voice_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(coordinator, "get_coordinator", lambda: Coordinator(events))
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"max_concurrent_sessions": 4})
    try:
        yield main, backend, events
    finally:
        session_limiter._reset_for_tests()


@pytest.mark.parametrize("path", ["/tts", "/v1/tts"])
@pytest.mark.parametrize("trigger", ["normal", "disconnect", "cancel_twice", "completion_race"])
def test_actual_finite_routes_with_middleware_own_native_and_receive_tasks(
    finite_app, path, trigger, caplog
):
    main, backend, events = finite_app
    caplog.set_level(logging.INFO, logger=main.__name__)

    async def run():
        initial_tasks = asyncio.all_tasks()
        received = asyncio.Queue()
        payload = {"text": "hello"}
        if path == "/v1/tts":
            payload["model"] = backend.model_id
        await received.put({"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False})
        sent = []
        receive_started = asyncio.Event()
        receive_active = 0

        async def receive():
            nonlocal receive_active
            receive_active += 1
            if received.empty():
                receive_started.set()
            try:
                return await received.get()
            finally:
                receive_active -= 1

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "method": "POST", "path": path, "raw_path": path.encode(),
                 "root_path": "", "scheme": "http", "query_string": b"",
                 "headers": [(b"content-type", b"application/json")],
                 "client": ("127.0.0.1", 123), "server": ("localhost", 80), "http_version": "1.1"}
        task = asyncio.create_task(main.app(scope, receive, send))
        try:
            assert await asyncio.to_thread(backend.started.wait, 1)
            await asyncio.wait_for(receive_started.wait(), 1)
            if trigger == "disconnect":
                await received.put({"type": "http.disconnect"})
            elif trigger == "cancel_twice":
                task.cancel()
            if trigger in {"disconnect", "cancel_twice"}:
                assert await asyncio.to_thread(backend.cancel_event.wait, 1)
                assert events == ["enter", "enter"]
                if trigger == "cancel_twice":
                    task.cancel()
                    await asyncio.sleep(0)
                assert not task.done()
            backend.release.set()
            if trigger == "completion_race":
                await received.put({"type": "http.disconnect"})
            done, _ = await asyncio.wait([task], timeout=1)
            assert task in done, f"stuck {trigger}: {events}, {sent}"
            result = (await asyncio.gather(task, return_exceptions=True))[0]
            disconnect_message = (
                f"route={path} client disconnected; finite cancellation drained"
            )
            disconnect_records = [
                record
                for record in caplog.records
                if record.levelno == logging.INFO
                and record.getMessage() == disconnect_message
            ]
            if trigger == "normal":
                assert result is None
                assert disconnect_records == []
                response = next(item for item in sent if item["type"] == "http.response.start")
                assert response["status"] == 200
                headers = dict(response["headers"])
                assert headers[b"x-kokoro-engine"] == b"mixed"
                assert headers[b"x-kokoro-segments"] == b"2"
                assert headers[b"x-kokoro-cpu-generator-ms"] == b"7.5"
                assert any(item.get("body") == b"wav" for item in sent)
            elif trigger == "disconnect":
                assert result is None, repr(result)
                response = next(
                    item for item in sent if item["type"] == "http.response.start"
                )
                assert response["status"] == 204
                assert len(disconnect_records) == 1
                assert not any(
                    marker in record.getMessage()
                    for record in caplog.records
                    for marker in ("Exception in ASGI application", "exception group")
                )
                assert not any(
                    record.levelno >= logging.ERROR for record in caplog.records
                )
            elif trigger == "cancel_twice":
                assert isinstance(result, asyncio.CancelledError), repr(result)
                assert disconnect_records == []
            else:
                assert result is None, repr(result)
                response = next(
                    item for item in sent if item["type"] == "http.response.start"
                )
                if response["status"] == 204:
                    assert len(disconnect_records) == 1
                else:
                    assert response["status"] == 200
                    assert disconnect_records == []
            assert events == ["enter", "enter", "exit", "exit"]
            assert receive_active == 0
            await asyncio.sleep(0)
            assert asyncio.all_tasks() == initial_tasks
        finally:
            backend.release.set()
            await received.put({"type": "http.disconnect"})
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("path", ["/tts", "/v1/tts"])
def test_direct_finite_handler_call_can_omit_request(finite_app, path):
    main, backend, events = finite_app
    backend.release.set()

    async def run():
        if path == "/tts":
            return await main.tts(main.TTSRequest(text="hello"))
        return await main.v1_tts(main.NativeTTSRequest(model=backend.model_id, text="hello"))

    response = asyncio.run(run())
    assert response.status_code == 200
    assert response.body == b"wav"
    assert events == ["enter", "enter", "exit", "exit"]


@pytest.mark.parametrize("segments,count,valid", [
    ([{}, {}], 2, "2"), ([{}], None, "1"), (None, 256, "256"),
    ([{}], 2, None), ([], 0, None), (None, True, None),
    (None, 257, None), (None, "2", None), (None, -1, None),
])
def test_finite_segment_header_validates_backend_aggregate(segments, count, valid):
    from server.main import _tts_response_headers

    metadata = {"backend": "kokoro_convonly", "segment_count": count}
    if segments is not None:
        metadata["segments"] = segments
    headers = _tts_response_headers(metadata, backend="rk:kokoro_convonly")
    assert headers.get("X-Kokoro-Segments") == valid


@pytest.mark.parametrize("engine", ["npu", "cpu", "mixed", "unknown", "cpu\r\ninjected", 1, ["cpu"]])
def test_finite_engine_header_is_bounded(engine):
    from server.main import _tts_response_headers

    headers = _tts_response_headers({"backend": "kokoro_convonly", "engine": engine}, backend="rk:kokoro_convonly")
    expected = engine if isinstance(engine, str) and engine in {"npu", "cpu", "mixed"} else None
    assert headers.get("X-Kokoro-Engine") == expected


@pytest.mark.parametrize("value,valid", [(0, "0"), (7.5, "7.5"), (float("inf"), None),
                                         (-1, None), (86_400_001, None), (True, None)])
def test_finite_cpu_generator_time_header_is_bounded(value, valid):
    from server.main import _tts_response_headers

    headers = _tts_response_headers({"backend": "kokoro_convonly", "cpu_generator_ms": value}, backend="rk:kokoro_convonly")
    assert headers.get("X-Kokoro-CPU-Generator-Ms") == valid
