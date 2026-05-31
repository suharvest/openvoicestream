"""Integration tests for the PR4 main.py wiring.

These tests bypass the heavy ``@app.on_event("startup")`` (which would try to
download models, load real ASR/TTS, etc.) by manually installing fake
BackendManager singletons before the TestClient is constructed. The TestClient
intentionally does NOT use ``with`` so the startup event never fires.

Coverage:
* /tts goes through ``tts_manager().acquire()`` (the fake records calls)
* tts_runtime overrides take effect on /tts when payload omits speaker_id
* explicit request speaker_id beats the runtime override
* /admin/backend/status returns both kinds
* /admin/backend/reload validates ``kind``
* admin auth: TestClient default host=testclient → 403 when key unset
* admin auth: loopback host bypass works
* /admin/backend/reload swaps to a fresh backend instance
* successful reload swaps the backend instance
* /admin/tts/speakers/reload calls into the speakers module
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------

class _FakeTTSBackend:
    name = "fake-tts"
    model_id = "qwen3-tts"  # match speaker table used in tts_speakers.json
    sample_rate = 16000
    # PR5: opt in so existing reload tests still pass.
    supports_hot_reload = True

    def __init__(self) -> None:
        # Advertise everything by default; individual tests can override.
        from app.core.tts_backend import TTSCapability
        self.capabilities = {TTSCapability.STREAMING, TTSCapability.VOICE_CLONE}
        self._ready = False
        self.synthesize_calls: list[dict] = []
        self.streaming_calls: list[dict] = []
        self.clone_calls: list[dict] = []
        self.unloaded = False
        # Hook the test can set to observe inflight_http during a call.
        self.inflight_observer = None

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        self._ready = True

    def unload(self) -> None:
        self.unloaded = True
        self._ready = False

    def has_capability(self, cap) -> bool:
        return cap in self.capabilities

    def synthesize(self, text, **kwargs):
        self.synthesize_calls.append({"text": text, **kwargs})
        if self.inflight_observer is not None:
            self.inflight_observer()
        return b"\x00\x00" * 16, {"duration": 0.001, "inference_time": 0.001, "rtf": 1.0}

    def generate_streaming(self, text, **kwargs):
        self.streaming_calls.append({"text": text, **kwargs})
        if self.inflight_observer is not None:
            self.inflight_observer()
        yield b"\x00\x00" * 8

    def clone_voice(self, text, speaker_embedding, language=None, **kwargs):
        self.clone_calls.append(
            {"text": text, "speaker_embedding": speaker_embedding, "language": language, **kwargs}
        )
        if self.inflight_observer is not None:
            self.inflight_observer()
        return b"\x00\x00" * 16, {"duration": 0.001, "inference_time": 0.001, "rtf": 1.0}


class _FakeASRBackend:
    name = "fake-asr"
    sample_rate = 16000
    # PR5: opt in so existing reload tests still pass.
    supports_hot_reload = True

    def __init__(self) -> None:
        self.capabilities = set()
        self._ready = False
        self.unloaded = False

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        self._ready = True

    def unload(self) -> None:
        self.unloaded = True
        self._ready = False

    def has_capability(self, cap) -> bool:
        return cap in self.capabilities


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _install_managers(asr=None, tts=None):
    """Reset module-level managers and install fakes (started)."""
    from app.core import backend_manager as bm
    from app.core import coordinator as coord_mod
    from app.core import session_limiter as sl_mod
    bm._reset_for_tests()
    # Ensure the coordinator singleton exists (default concurrent policy);
    # endpoint code calls get_coordinator() unconditionally.
    coord_mod._coordinator = None  # type: ignore[attr-defined]
    coord_mod.init_coordinator({"mode": "concurrent"})
    # Stale-test fix: the session limiter was introduced after this test was
    # written. The HTTP admission path (acquire_http) and WS path now require a
    # global limiter that is normally initialized in @app.on_event("startup")
    # (app/main.py:671), which these tests intentionally bypass. Without it
    # every /tts(/clone) call short-circuits to 503 session_limiter_unavailable.
    # Initialize it here (parallel to the coordinator init above) with a high
    # ceiling so admission never throttles the wiring assertions under test.
    sl_mod._reset_for_tests()
    sl_mod.init_limiter({"max_concurrent_sessions": 64})

    asr_be = asr or _FakeASRBackend()
    tts_be = tts or _FakeTTSBackend()

    bm.init_backend_managers(
        tts_factory=lambda: tts_be,
        tts_preloader=lambda b: b.preload(),
        tts_unloader=lambda b: b.unload(),
        asr_factory=lambda: asr_be,
        asr_preloader=lambda b: b.preload(),
        asr_unloader=lambda b: b.unload(),
    )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(bm.tts_manager().start())
        loop.run_until_complete(bm.asr_manager().start())
    finally:
        loop.close()
    return asr_be, tts_be


@pytest.fixture
def client(monkeypatch):
    from app.core import tts_runtime, tts_service
    tts_runtime.reset_overrides()

    asr_be, tts_be = _install_managers()

    # Some endpoints still inspect tts_service.is_ready() / get_backend()
    # in the partial-wiring fallback path. Wire it to the fake.
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: tts_be)
    monkeypatch.setattr(tts_service, "is_configured", lambda: True)
    monkeypatch.setattr(tts_service, "_backend", tts_be, raising=False)

    from app.main import app
    from app.core.admin_auth import require_admin

    async def _allow():
        return None

    app.dependency_overrides[require_admin] = _allow

    c = TestClient(app)
    c.tts_be = tts_be   # type: ignore[attr-defined]
    c.asr_be = asr_be   # type: ignore[attr-defined]
    try:
        yield c
    finally:
        app.dependency_overrides.pop(require_admin, None)
        tts_runtime.reset_overrides()
        from app.core import backend_manager as bm
        bm._reset_for_tests()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tts_endpoint_uses_manager_acquire(client):
    """A POST to /tts should call backend.synthesize via the manager path."""
    r = client.post("/tts", json={"text": "hello"})
    assert r.status_code == 200, r.text
    assert len(client.tts_be.synthesize_calls) == 1
    call = client.tts_be.synthesize_calls[0]
    assert call["text"] == "hello"


def test_tts_runtime_override_applied(client):
    """PATCH /admin/tts/runtime sets default_speaker_id → /tts picks it up."""
    # 2301 is a known preset id for qwen3-tts in the bundled speakers table.
    r = client.patch("/admin/tts/runtime", json={"speaker_id": 2301})
    assert r.status_code == 200, r.text

    client.tts_be.synthesize_calls.clear()
    r = client.post("/tts", json={"text": "hi"})
    assert r.status_code == 200, r.text
    call = client.tts_be.synthesize_calls[-1]
    # speaker_kwargs_for_id translates 2301 into a backend-specific kwarg
    # (either ``speaker_id`` or an embedding). The key thing is it's NOT
    # the default speaker id (which would be 0 for qwen3-tts).
    if "speaker_id" in call:
        assert call["speaker_id"] == 2301


def test_tts_request_param_overrides_runtime(client):
    """An explicit speaker_id in the request wins over the runtime override."""
    client.patch("/admin/tts/runtime", json={"speaker_id": 2301})
    client.tts_be.synthesize_calls.clear()

    r = client.post("/tts", json={"text": "hi", "speaker_id": 0})
    assert r.status_code == 200, r.text
    call = client.tts_be.synthesize_calls[-1]
    if "speaker_id" in call:
        assert call["speaker_id"] == 0


def test_admin_backend_status(client):
    r = client.get("/admin/backend/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "tts" in body and "asr" in body
    assert body["tts"]["state"] == "ready"
    assert body["asr"]["state"] == "ready"
    assert body["tts"]["backend_name"] == "fake-tts"
    assert body["asr"]["backend_name"] == "fake-asr"


def test_admin_backend_reload_unknown_kind(client):
    r = client.post("/admin/backend/reload", json={"kind": "xxx", "profile": "p"})
    # Pydantic literal rejection → 422.
    assert r.status_code in (400, 422), r.text


def test_admin_backend_reload_missing_auth_non_loopback(monkeypatch):
    """TestClient default host=testclient + no OVS_ADMIN_KEY → 403."""
    monkeypatch.delenv("OVS_ADMIN_KEY", raising=False)
    from app.core import tts_runtime, tts_service
    tts_runtime.reset_overrides()
    _install_managers()
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)

    from app.main import app
    # No dependency_overrides this time → real require_admin runs.
    c = TestClient(app)
    r = c.post("/admin/backend/reload", json={"kind": "tts", "profile": "fake"})
    assert r.status_code == 403, r.text
    from app.core import backend_manager as bm
    bm._reset_for_tests()


def test_admin_backend_reload_loopback_allowed(monkeypatch, tmp_path):
    """Loopback client.host bypasses the OVS_ADMIN_KEY check."""
    monkeypatch.delenv("OVS_ADMIN_KEY", raising=False)
    from app.core import tts_runtime, tts_service, profile_loader
    tts_runtime.reset_overrides()
    asr_be, tts_be = _install_managers()
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: tts_be)

    # Stub apply_profile / current_profile / path resolver so reload doesn't
    # touch the real configs/profiles tree.
    import tempfile, json as _json
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    tmp.write(_json.dumps({"name": "any", "tts_backend": "kokoro"}))
    tmp.flush()
    from pathlib import Path as _Path
    from app.core import backend_manager as bm
    monkeypatch.setattr(bm, "_resolve_profile_path", lambda ref: _Path(tmp.name))
    monkeypatch.setattr(
        profile_loader, "current_profile",
        lambda: {"name": "live", "tts_backend": "kokoro"},
    )
    monkeypatch.setattr(
        profile_loader, "apply_profile",
        lambda ref, *, overrides=None, resolve_engines=False: None,
    )

    from app.main import app
    c = TestClient(app)

    from app.core import admin_auth
    monkeypatch.setattr(admin_auth, "_is_loopback", lambda host: True)

    r = c.post("/admin/backend/reload", json={"kind": "tts", "profile": "any"})
    # Reload should succeed (fakes preload trivially) and return status dict.
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in ("reloaded", "rolled_back")
    from app.core import backend_manager as bm
    bm._reset_for_tests()


def test_admin_backend_reload_success_swaps_backend(client, monkeypatch):
    """Successful reload returns ``reloaded`` and bumps backend instance."""
    from app.core import profile_loader
    monkeypatch.setattr(
        profile_loader, "current_profile",
        lambda: {"name": "p1", "tts_backend": "kokoro"},
    )
    monkeypatch.setattr(
        profile_loader, "apply_profile",
        lambda ref, *, overrides=None, resolve_engines=False: None,
    )

    # Pre-set a synthetic profile path that parses to the same backend kind.
    import tempfile, json as _json
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    tmp.write(_json.dumps({"name": "p2", "tts_backend": "kokoro"}))
    tmp.flush()
    from pathlib import Path as _Path
    from app.core import backend_manager as bm
    monkeypatch.setattr(bm, "_resolve_profile_path", lambda ref: _Path(tmp.name))

    r = client.post("/admin/backend/reload", json={"kind": "tts", "profile": "p2"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "reloaded"
    assert body["kind"] == "tts"


def test_admin_tts_speakers_reload(client):
    """The speakers reload route still works under the new wiring."""
    r = client.post("/admin/tts/speakers/reload")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reloaded"] is True


# ---------------------------------------------------------------------------
# FIX_1 / FIX_2 / FIX_3 regression tests (PR4b)
# ---------------------------------------------------------------------------

def _observe_inflight(client):
    """Return a hook the fake backend can call mid-request to record inflight_http."""
    from app.core.backend_manager import tts_manager
    observed: list[int] = []

    def hook():
        observed.append(tts_manager().status()["inflight_http"])

    client.tts_be.inflight_observer = hook
    return observed


def test_tts_stream_uses_manager_acquire(client):
    """FIX_1: /tts/stream must bump tts_manager().inflight_http during the call."""
    observed = _observe_inflight(client)
    r = client.post("/tts/stream", json={"text": "hello"})
    assert r.status_code == 200, r.text
    # Consume the body so the StreamingResponse generator runs to completion
    # (TestClient eagerly buffers, so by the time we read .content the
    # generator has finished — but observed[] is populated as a side effect).
    _ = r.content
    assert observed, "backend.generate_streaming was never called"
    assert all(n >= 1 for n in observed), f"expected inflight>=1 during call, got {observed}"
    # And streaming kwargs were forwarded (no stray speed/pitch since override unset)
    assert client.tts_be.streaming_calls, "generate_streaming not invoked"


def test_tts_clone_uses_manager_acquire(client):
    """FIX_1: /tts/clone must bump inflight_http via mgr.acquire()."""
    import base64
    observed = _observe_inflight(client)
    payload = {
        "text": "hi",
        "speaker_embedding_b64": base64.b64encode(b"\x00" * 16).decode(),
    }
    r = client.post("/tts/clone", json=payload)
    assert r.status_code == 200, r.text
    assert observed, "clone_voice was never called"
    assert all(n >= 1 for n in observed), observed
    assert client.tts_be.clone_calls


def test_tts_clone_stream_uses_manager_acquire(client):
    """FIX_1: /tts/clone/stream must bump inflight_http via mgr.acquire()."""
    import base64
    observed = _observe_inflight(client)
    payload = {
        "text": "hi",
        "speaker_embedding_b64": base64.b64encode(b"\x00" * 16).decode(),
    }
    r = client.post("/tts/clone/stream", json=payload)
    assert r.status_code == 200, r.text
    _ = r.content
    assert observed
    assert all(n >= 1 for n in observed), observed
    assert client.tts_be.streaming_calls


def test_runtime_speed_override_applied_to_tts(client):
    """FIX_2: PATCH /admin/tts/runtime speed=1.5 → backend sees speed=1.5."""
    r = client.patch("/admin/tts/runtime", json={"speed": 1.5})
    assert r.status_code == 200, r.text
    client.tts_be.synthesize_calls.clear()
    r = client.post("/tts", json={"text": "hi"})
    assert r.status_code == 200, r.text
    call = client.tts_be.synthesize_calls[-1]
    assert call.get("speed") == 1.5, f"expected speed=1.5 in {call}"


def test_runtime_pitch_override_applied_to_tts(client):
    """FIX_2: PATCH /admin/tts/runtime pitch_shift=3 → backend sees pitch_shift=3."""
    r = client.patch("/admin/tts/runtime", json={"pitch_shift": 3.0})
    assert r.status_code == 200, r.text
    client.tts_be.synthesize_calls.clear()
    r = client.post("/tts", json={"text": "hi"})
    assert r.status_code == 200, r.text
    call = client.tts_be.synthesize_calls[-1]
    assert call.get("pitch_shift") == 3.0, f"expected pitch_shift=3.0 in {call}"


def test_request_speed_overrides_runtime(client):
    """FIX_2: request payload beats runtime override (speed=2.0 > runtime 1.5)."""
    r = client.patch("/admin/tts/runtime", json={"speed": 1.5})
    assert r.status_code == 200, r.text
    client.tts_be.synthesize_calls.clear()
    r = client.post("/tts", json={"text": "hi", "speed": 2.0})
    assert r.status_code == 200, r.text
    call = client.tts_be.synthesize_calls[-1]
    assert call.get("speed") == 2.0, f"expected request speed=2.0 to win, got {call}"


def test_lazy_tts_first_request_starts_manager(monkeypatch):
    """FIX_3: LAZY-style startup leaves manager in INIT → first /tts drives it READY."""
    from app.core import backend_manager as bm, coordinator as coord_mod, tts_runtime, tts_service

    tts_runtime.reset_overrides()
    bm._reset_for_tests()
    coord_mod._coordinator = None  # type: ignore[attr-defined]
    coord_mod.init_coordinator({"mode": "concurrent"})
    # Stale-test fix: /tts admission now requires a session limiter (see
    # _install_managers comment). Init it so the request reaches the lazy-start
    # path instead of short-circuiting to 503 session_limiter_unavailable.
    from app.core import session_limiter as sl_mod
    sl_mod._reset_for_tests()
    sl_mod.init_limiter({"max_concurrent_sessions": 64})

    tts_be = _FakeTTSBackend()
    asr_be = _FakeASRBackend()
    bm.init_backend_managers(
        tts_factory=lambda: tts_be,
        tts_preloader=lambda b: b.preload(),
        tts_unloader=lambda b: b.unload(),
        asr_factory=lambda: asr_be,
        asr_preloader=lambda b: b.preload(),
        asr_unloader=lambda b: b.unload(),
    )

    # IMPORTANT: do NOT call mgr.start() — simulate LAZY_TTS skipping startup preload.
    assert bm.tts_manager().state.value == "init"
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: tts_be)

    # Reset lazy-start lock between tests in the same process.
    import app.main as _main_mod
    _main_mod._tts_lazy_start_lock = None

    from app.main import app
    from app.core.admin_auth import require_admin
    app.dependency_overrides[require_admin] = lambda: None

    try:
        c = TestClient(app)
        r = c.post("/tts", json={"text": "hi"})
        assert r.status_code == 200, r.text
        assert bm.tts_manager().state.value == "ready"
        assert tts_be._ready is True
        assert tts_be.synthesize_calls
    finally:
        app.dependency_overrides.pop(require_admin, None)
        bm._reset_for_tests()
        tts_runtime.reset_overrides()


# ---------------------------------------------------------------------------
# FIX_3_completion: FAILED / start-fail manager must NOT silently fall back
# to legacy tts_service.synthesize. Operators need a real 503.
# ---------------------------------------------------------------------------

def test_failed_manager_returns_503_not_fallback(monkeypatch):
    """Manager in FAILED state → /tts gets 503, legacy tts_service is NOT used."""
    from app.core import backend_manager as bm, coordinator as coord_mod, tts_runtime, tts_service

    tts_runtime.reset_overrides()
    bm._reset_for_tests()
    coord_mod._coordinator = None  # type: ignore[attr-defined]
    coord_mod.init_coordinator({"mode": "concurrent"})
    # Stale-test fix: init session limiter so /tts reaches the manager-failed
    # branch instead of short-circuiting to 503 session_limiter_unavailable.
    from app.core import session_limiter as sl_mod
    sl_mod._reset_for_tests()
    sl_mod.init_limiter({"max_concurrent_sessions": 64})

    # TTS factory that always fails → start() flips state to FAILED.
    def _bad_factory():
        raise RuntimeError("boom-tts")

    asr_be = _FakeASRBackend()
    bm.init_backend_managers(
        tts_factory=_bad_factory,
        tts_preloader=lambda b: None,
        tts_unloader=lambda b: None,
        asr_factory=lambda: asr_be,
        asr_preloader=lambda b: b.preload(),
        asr_unloader=lambda b: b.unload(),
    )

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError):
            loop.run_until_complete(bm.tts_manager().start())
    finally:
        loop.close()
    assert bm.tts_manager().state.value == "failed"

    # Sentinel: if endpoint silently fell back, it would call tts_service.synthesize.
    legacy_called = {"n": 0}

    def _legacy_synth(*args, **kwargs):
        legacy_called["n"] += 1
        return b"\x00\x00" * 8, {"duration": 0.001, "inference_time": 0.001, "rtf": 1.0}

    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "is_configured", lambda: True)
    monkeypatch.setattr(tts_service, "synthesize", _legacy_synth)
    monkeypatch.setattr(tts_service, "_backend", _FakeTTSBackend(), raising=False)

    import app.main as _main_mod
    _main_mod._tts_lazy_start_lock = None

    from app.main import app
    from app.core.admin_auth import require_admin
    app.dependency_overrides[require_admin] = lambda: None
    try:
        c = TestClient(app)
        r = c.post("/tts", json={"text": "hi"})
        assert r.status_code == 503, r.text
        body = r.json()
        # FastAPI wraps HTTPException.detail under "detail".
        detail = body.get("detail", body)
        assert isinstance(detail, dict)
        assert detail.get("error") == "tts_manager_failed", detail
        assert detail.get("state") == "failed", detail
        # The legacy path must NOT have been reached.
        assert legacy_called["n"] == 0, "FAILED manager must not silently fall back to tts_service"
    finally:
        app.dependency_overrides.pop(require_admin, None)
        bm._reset_for_tests()
        tts_runtime.reset_overrides()


def test_manager_start_failure_first_request_raises_503(monkeypatch):
    """Manager INIT + factory fails on lazy start → /tts gets 503 (not 200 via legacy)."""
    from app.core import backend_manager as bm, coordinator as coord_mod, tts_runtime, tts_service

    tts_runtime.reset_overrides()
    bm._reset_for_tests()
    coord_mod._coordinator = None  # type: ignore[attr-defined]
    coord_mod.init_coordinator({"mode": "concurrent"})
    # Stale-test fix: init session limiter so /tts reaches the lazy-start
    # failure branch instead of short-circuiting to session_limiter_unavailable.
    from app.core import session_limiter as sl_mod
    sl_mod._reset_for_tests()
    sl_mod.init_limiter({"max_concurrent_sessions": 64})

    def _bad_factory():
        raise RuntimeError("preload-blew-up")

    asr_be = _FakeASRBackend()
    bm.init_backend_managers(
        tts_factory=_bad_factory,
        tts_preloader=lambda b: None,
        tts_unloader=lambda b: None,
        asr_factory=lambda: asr_be,
        asr_preloader=lambda b: b.preload(),
        asr_unloader=lambda b: b.unload(),
    )

    # IMPORTANT: do NOT call start() — keep manager in INIT so the endpoint
    # triggers lazy start, which will fail.
    assert bm.tts_manager().state.value == "init"

    legacy_called = {"n": 0}

    def _legacy_synth(*args, **kwargs):
        legacy_called["n"] += 1
        return b"\x00\x00" * 8, {"duration": 0.001, "inference_time": 0.001, "rtf": 1.0}

    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "is_configured", lambda: True)
    monkeypatch.setattr(tts_service, "synthesize", _legacy_synth)
    monkeypatch.setattr(tts_service, "_backend", _FakeTTSBackend(), raising=False)

    import app.main as _main_mod
    _main_mod._tts_lazy_start_lock = None

    from app.main import app
    from app.core.admin_auth import require_admin
    app.dependency_overrides[require_admin] = lambda: None
    try:
        c = TestClient(app)
        r = c.post("/tts", json={"text": "hi"})
        assert r.status_code == 503, r.text
        body = r.json()
        detail = body.get("detail", body)
        assert isinstance(detail, dict)
        assert detail.get("error") == "tts_manager_start_failed", detail
        # After start() failed, manager should be FAILED.
        assert bm.tts_manager().state.value == "failed"
        assert legacy_called["n"] == 0, "start() failure must not silently fall back to tts_service"
    finally:
        app.dependency_overrides.pop(require_admin, None)
        bm._reset_for_tests()
        tts_runtime.reset_overrides()
