"""Contract tests for the strict native v1 HTTP aliases.

The route tests are skipped in the minimal source checkout when the optional
Prometheus dependency is absent; the transport-neutral execution and bounded
reader tests remain runnable without a web stack or GPU backend.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest


class _Upload:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self, size: int = -1):
        if not self._data:
            return b""
        if size < 0:
            size = len(self._data)
        chunk, self._data = self._data[:size], self._data[size:]
        return chunk


def test_bounded_upload_counts_decoded_bytes():
    from server.core.api_execution import APIExecutionError, read_bounded_upload

    async def run():
        assert await read_bounded_upload(_Upload(b"abcdef"), max_bytes=6, chunk_size=2) == b"abcdef"
        with pytest.raises(APIExecutionError) as caught:
            await read_bounded_upload(_Upload(b"abcdefg"), max_bytes=6, chunk_size=2)
        assert caught.value.status_code == 413
        assert caught.value.code == "payload_too_large"
        assert caught.value.param == "file"

    asyncio.run(run())


def test_execution_prepare_and_backend_lease_share_one_owner():
    from server.core.api_execution import execute_tts

    events: list[str] = []

    class Backend:
        name = "fake"
        sample_rate = 16000

        def synthesize(self, text, language=None, **kwargs):
            events.append("synthesize")
            assert events == ["manager_enter", "prepare", "coordinator_enter", "synthesize"]
            assert kwargs == {"speaker_id": 7}
            return b"wav", {"duration": 1}

    backend = Backend()

    class Manager:
        @asynccontextmanager
        async def acquire(self):
            events.append("manager_enter")
            try:
                yield backend
            finally:
                events.append("manager_exit")

    class Coordinator:
        @asynccontextmanager
        async def acquire(self, kind):
            assert kind == "tts"
            events.append("coordinator_enter")
            try:
                yield
            finally:
                events.append("coordinator_exit")

    async def run():
        result = await execute_tts(
            text="x",
            language="auto",
            voice_kwargs={},
            manager=Manager(),
            legacy_service=None,
            coordinator=Coordinator(),
            prepare=lambda be: (events.append("prepare") or {"speaker_id": 7}),
        )
        assert result.audio == b"wav"

    asyncio.run(run())
    assert events == [
        "manager_enter",
        "prepare",
        "coordinator_enter",
        "synthesize",
        "coordinator_exit",
        "manager_exit",
    ]


def test_execution_asr_model_validation_callback_runs_inside_manager_lease():
    from server.core.api_execution import execute_asr
    from server.core.asr_backend import TranscriptionResult

    events: list[str] = []

    class Backend:
        name = "fake-asr"

        def transcribe(self, audio, language="auto"):
            events.append("transcribe")
            assert events == ["manager_enter", "prepare", "coordinator_enter", "transcribe"]
            return TranscriptionResult("ok", language)

    backend = Backend()

    class Manager:
        @asynccontextmanager
        async def acquire(self):
            events.append("manager_enter")
            try:
                yield backend
            finally:
                events.append("manager_exit")

    class Coordinator:
        @asynccontextmanager
        async def acquire(self, kind):
            assert kind == "asr"
            events.append("coordinator_enter")
            try:
                yield
            finally:
                events.append("coordinator_exit")

    async def run():
        result = await execute_asr(
            audio=b"pcm",
            language="auto",
            manager=Manager(),
            legacy_backend=None,
            coordinator=Coordinator(),
            prepare=lambda be: events.append("prepare"),
        )
        assert result.text == "ok"

    asyncio.run(run())
    assert events == [
        "manager_enter",
        "prepare",
        "coordinator_enter",
        "transcribe",
        "coordinator_exit",
        "manager_exit",
    ]


def test_execution_cancel_releases_coordinator_and_manager_leases():
    from server.core.api_execution import execute_tts

    events: list[str] = []

    class Backend:
        name = "cancelled"

        def synthesize(self, **_kwargs):
            events.append("synthesize")
            raise asyncio.CancelledError()

    class Manager:
        @asynccontextmanager
        async def acquire(self):
            events.append("manager_enter")
            try:
                yield Backend()
            finally:
                events.append("manager_exit")

    class Coordinator:
        @asynccontextmanager
        async def acquire(self, _kind):
            events.append("coordinator_enter")
            try:
                yield
            finally:
                events.append("coordinator_exit")

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await execute_tts(
                text="x",
                language=None,
                voice_kwargs={},
                manager=Manager(),
                legacy_service=None,
                coordinator=Coordinator(),
            )

    asyncio.run(run())
    assert events == [
        "manager_enter", "coordinator_enter", "synthesize",
        "coordinator_exit", "manager_exit",
    ]


@pytest.fixture()
def native_client(monkeypatch):
    pytest.importorskip("prometheus_client")
    from fastapi.testclient import TestClient
    from server import main
    from server.core import coordinator, session_limiter, tts_runtime

    class TTS:
        name = "fake-tts"
        model_id = "qwen3-tts-customvoice"
        sample_rate = 16000
        capabilities = set()
        supports_voice_cloning = False

        def is_ready(self):
            return True

        def rate_pitch_caps(self):
            return True, True

        def synthesize(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return b"wav", {"duration": 1}

        def __init__(self):
            self.calls = []

    class Manager:
        def __init__(self, backend):
            self.backend = backend
            self.state = "ready"

        def get_backend_unsafe(self):
            return self.backend

        @asynccontextmanager
        async def acquire(self):
            yield self.backend

        def is_ready(self):
            return True

    tts = TTS()
    manager = Manager(tts)
    tts_runtime.reset_overrides()
    monkeypatch.setattr(main, "_ensure_tts_manager_started", lambda: None)

    async def _ensure():
        return manager

    monkeypatch.setattr(main, "_ensure_tts_manager_started", _ensure)
    coordinator._coordinator = None
    coordinator.init_coordinator({"mode": "concurrent"})
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"max_concurrent_sessions": 4})
    client = TestClient(main.app)
    try:
        yield client, tts, manager
    finally:
        tts_runtime.reset_overrides()
        session_limiter._reset_for_tests()


def test_v1_tts_strict_model_voice_and_controls(native_client):
    client, tts, _manager = native_client

    ok = client.post(
        "/v1/tts",
        json={"model": "qwen3-tts-customvoice", "text": "hello", "voice": "vivian", "speed": 1.0},
    )
    assert ok.status_code == 200, ok.text
    assert ok.content == b"wav"
    assert tts.calls[-1][1]["speaker_id"] == 3065

    # Legacy and native aliases must serialize the same backend result and
    # preserve the historical response headers.
    legacy = client.post("/tts", json={"text": "hello", "speaker_id": 3065})
    assert legacy.status_code == 200
    assert legacy.content == ok.content
    assert legacy.headers["content-type"] == ok.headers["content-type"]
    assert legacy.headers["x-audio-duration"] == ok.headers["x-audio-duration"]

    unknown_model = client.post("/v1/tts", json={"model": "other", "text": "x"})
    assert unknown_model.status_code == 404
    assert unknown_model.json()["error"]["code"] == "unknown_model"

    unknown_voice = client.post(
        "/v1/tts",
        json={"model": "qwen3-tts-customvoice", "text": "x", "voice": "not-a-voice"},
    )
    assert unknown_voice.status_code == 404
    assert unknown_voice.json()["error"]["code"] == "unsupported_voice"

    too_fast = client.post(
        "/v1/tts",
        json={"model": "qwen3-tts-customvoice", "text": "x", "speed": 9},
    )
    assert too_fast.status_code == 400
    assert too_fast.json()["error"]["code"] == "unsupported_control"


def test_v1_tts_accepts_model_scoped_base_embedding_profile(
    native_client, monkeypatch, tmp_path
):
    import numpy as np
    from server.core import sparktts_voices

    client, tts, _manager = native_client
    monkeypatch.setenv("SPARKTTS_VOICES_DIR", str(tmp_path))
    tts.model_id = "qwen3-tts-0.6b-base"
    tts.supports_voice_cloning = True
    embedding = np.arange(16, dtype=np.float32).tobytes()
    sparktts_voices.register_embedding_voice(
        "clone:base-v1",
        embedding,
        model_id=tts.model_id,
    )

    response = client.post(
        "/v1/tts",
        json={"model": tts.model_id, "text": "hello", "voice": "clone:base-v1"},
    )
    assert response.status_code == 200, response.text
    kwargs = tts.calls[-1][1]
    assert kwargs["speaker_embedding"] == embedding
    assert "voice" not in kwargs
    assert "speaker_id" not in kwargs


def test_v1_tts_accepts_every_declared_spark_style_shape(native_client):
    client, tts, _manager = native_client
    tts.model_id = "sparktts-0p5b"

    response = client.post(
        "/v1/tts",
        json={"model": tts.model_id, "text": "hello", "voice": "female_very_high_low"},
    )
    assert response.status_code == 200, response.text
    kwargs = tts.calls[-1][1]
    assert kwargs["voice"] == "female_very_high_low"
    assert "speaker" not in kwargs
    assert "speaker_id" not in kwargs

    invalid = client.post(
        "/v1/tts",
        json={"model": tts.model_id, "text": "hello", "voice": "female_extreme_low"},
    )
    assert invalid.status_code == 404
    assert invalid.json()["error"]["code"] == "unsupported_voice"


@pytest.mark.parametrize("voice", [None, ""])
def test_v1_tts_omitted_voice_honors_runtime_default(native_client, voice):
    from server.core import tts_runtime

    client, tts, _manager = native_client
    tts_runtime.update_overrides(
        speaker_id=3066,
        model_id="qwen3-tts-customvoice",
    )
    body = {"model": tts.model_id, "text": "hello"}
    if voice is not None:
        body["voice"] = voice
    response = client.post("/v1/tts", json=body)
    assert response.status_code == 200, response.text
    assert tts.calls[-1][1]["speaker_id"] == 3066


@pytest.mark.parametrize(
    ("model_id", "profile_type"),
    [
        ("qwen3-tts-0.6b-base", "voice_profile"),
        ("sparktts-0p5b", "speaker_embedding"),
        ("moss-tts-nano-v1", "speaker_embedding"),
    ],
)
def test_v1_tts_rejects_wrong_reusable_profile_type(
    native_client, monkeypatch, model_id, profile_type
):
    from server.core import sparktts_voices

    client, tts, _manager = native_client
    tts.model_id = model_id
    tts.supports_voice_cloning = True
    monkeypatch.setattr(
        sparktts_voices,
        "list_voices",
        lambda **_kwargs: [{
            "voice_id": "clone:wrong-type",
            "model_id": model_id,
            "compatible_models": [model_id],
            "profile_type": profile_type,
        }],
    )
    response = client.post(
        "/v1/tts",
        json={"model": model_id, "text": "hello", "voice": "clone:wrong-type"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unsupported_voice"


def test_v1_tts_text_limit_and_429(native_client, monkeypatch):
    client, _tts, _manager = native_client
    monkeypatch.setenv("OVS_API_MAX_TEXT_BYTES", "4")
    too_long = client.post("/v1/tts", json={"model": "qwen3-tts-customvoice", "text": "12345"})
    assert too_long.status_code == 413
    assert too_long.json()["error"]["code"] == "payload_too_large"

    from server.core import session_limiter
    token = session_limiter.get_limiter().try_acquire()
    try:
        busy = client.post("/v1/tts", json={"model": "qwen3-tts-customvoice", "text": "x"})
        assert busy.status_code == 429
        assert busy.json()["error"]["code"] == "too_many_sessions"
    finally:
        token.release()


def test_native_v1_auth_matches_legacy(native_client, monkeypatch):
    client, _tts, _manager = native_client
    monkeypatch.setenv("OVS_API_KEYS", "native-secret")

    legacy_missing = client.post("/tts", json={"text": "x"})
    native_missing = client.post(
        "/v1/tts",
        json={"model": "qwen3-tts-customvoice", "text": "x"},
    )
    assert legacy_missing.status_code == native_missing.status_code == 401
    assert legacy_missing.headers["www-authenticate"] == native_missing.headers["www-authenticate"] == "Bearer"

    headers = {"Authorization": "Bearer native-secret"}
    legacy_ok = client.post("/tts", json={"text": "x"}, headers=headers)
    native_ok = client.post(
        "/v1/tts",
        json={"model": "qwen3-tts-customvoice", "text": "x"},
        headers=headers,
    )
    assert legacy_ok.status_code == native_ok.status_code == 200


def test_v1_backend_pool_saturation_and_internal_error_are_safe(native_client):
    client, tts, _manager = native_client

    class PoolSaturatedError(RuntimeError):
        status = 4429

    def saturated(*_args, **_kwargs):
        raise PoolSaturatedError("worker full")

    tts.synthesize = saturated
    busy = client.post(
        "/v1/tts",
        json={"model": tts.model_id, "text": "x"},
    )
    assert busy.status_code == 429
    assert busy.json()["error"]["code"] == "backend_busy"
    assert busy.headers["retry-after"] == "1"

    def broken(*_args, **_kwargs):
        raise RuntimeError("secret engine path: /opt/private/model.engine")

    tts.synthesize = broken
    failed = client.post(
        "/v1/tts",
        json={"model": tts.model_id, "text": "x"},
    )
    assert failed.status_code == 503
    assert failed.json()["error"] == {
        "code": "backend_error",
        "message": "backend execution failed",
    }
    assert "/opt/private" not in failed.text

    from fastapi import HTTPException
    from server import main
    serialized = main._native_error_response(HTTPException(
        status_code=503,
        detail={
            "error": "tts_manager_start_failed",
            "message": "engine missing at /opt/private/model.engine",
        },
    ))
    assert b"/opt/private" not in serialized.body
    assert b"TTS backend failed to start" in serialized.body


def test_v1_tts_backend_503_and_capability_aliases(native_client, monkeypatch):
    client, _tts, manager = native_client
    from fastapi import HTTPException
    from server import main

    async def failed_start():
        raise HTTPException(status_code=503, detail={"error": "tts_manager_failed"})

    monkeypatch.setattr(main, "_ensure_tts_manager_started", failed_start)
    failed = client.post("/v1/tts", json={"model": "qwen3-tts-customvoice", "text": "x"})
    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "tts_manager_failed"

    # These aliases must dispatch successfully to their legacy builders (and
    # therefore must not call a dependency-injected handler without its
    # required argument).  This fixture has no installed global manager, so a
    # truthful not-ready response is expected.
    monkeypatch.setattr(main, "_get_tts_manager", lambda: None)
    caps = client.get("/v1/tts/capabilities")
    speakers = client.get("/v1/tts/speakers")
    assert caps.status_code in (200, 503)
    assert speakers.status_code in (200, 503)
    assert caps.status_code != 500
    assert speakers.status_code != 500


def test_v1_asr_model_and_bounded_audio(monkeypatch):
    pytest.importorskip("prometheus_client")
    from fastapi.testclient import TestClient
    from server import main
    from server.core import coordinator, session_limiter
    from server.core.asr_backend import TranscriptionResult

    class ASR:
        name = "fake-asr"

        def is_ready(self):
            return True

        def transcribe(self, audio, language="auto"):
            return TranscriptionResult("decoded", language)

    class Manager:
        def __init__(self):
            self.backend = ASR()

        def is_ready(self):
            return True

        def get_backend_unsafe(self):
            return self.backend

        @asynccontextmanager
        async def acquire(self):
            yield self.backend

    manager = Manager()
    monkeypatch.setattr(main, "_get_asr_manager", lambda: manager)
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {"asr_backend": "jetson.trt_edge_llm", "asr_model_id": "qwen3-asr"},
    )
    coordinator._coordinator = None
    coordinator.init_coordinator({"mode": "concurrent"})
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"max_concurrent_sessions": 4})
    client = TestClient(main.app)
    try:
        good = client.post(
            "/v1/asr?model=qwen3-asr",
            files={"file": ("a.wav", b"pcm", "audio/wav")},
        )
        assert good.status_code == 200, good.text
        assert good.json()["text"] == "decoded"

        legacy = client.post(
            "/asr?language=auto",
            files={"file": ("a.wav", b"pcm", "audio/wav")},
        )
        assert legacy.status_code == 200, legacy.text
        assert legacy.json() == good.json()

        wrong = client.post(
            "/v1/asr?model=other",
            files={"file": ("a.wav", b"pcm", "audio/wav")},
        )
        assert wrong.status_code == 404
        assert wrong.json()["error"]["code"] == "unknown_model"

        monkeypatch.setenv("OVS_API_MAX_AUDIO_BYTES", "3")
        too_large = client.post(
            "/v1/asr?model=qwen3-asr",
            files={"file": ("a.wav", b"pcmx", "audio/wav")},
        )
        assert too_large.status_code == 413
        assert too_large.json()["error"]["code"] == "payload_too_large"
    finally:
        session_limiter._reset_for_tests()
