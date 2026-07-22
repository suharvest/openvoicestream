"""Device-side voice enrollment via CPU-ONNX (torch-less) — endpoint coverage.

Verifies the acceptance contract for the ONNX enroll path:

  1. POST /tts/voices/enroll succeeds by calling the active backend's
     ``extract_speaker_embedding`` (CPU-ONNX, float32[1024]) and persisting an
     *embedding-profile* — even when the PyTorch SparkTTS enroller is absent.
  2. POST /tts (voice="<id>") resolves that embedding-profile server-side and
     forwards the raw ``speaker_embedding`` bytes to the backend (closed loop).
  3. /tts/capabilities exposes ``supports_voice_enrollment``.
  4. When the backend cannot enroll AND torch is absent → 501 (honest fallback).

Reuses the fake-backend manager harness from ``test_main_hot_swap`` (no CUDA).
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.tests.test_main_hot_swap import (
    _FakeASRBackend,
    _FakeTTSBackend,
    _install_managers,
)

# The float32[1024] vector the fake ONNX encoder "extracts".
_ENROLL_EMB = np.zeros(1024, dtype=np.float32).tobytes()


class _EnrollTTSBackend(_FakeTTSBackend):
    """Fake Qwen3 BASE backend: voice-clone + a usable CPU-ONNX enroller."""

    name = "fake-enroll"
    # supports_voice_enrollment gated True (encoder "present").
    supports_voice_enrollment = True

    def __init__(self) -> None:
        super().__init__()
        self.extract_calls: list[bytes] = []

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        self.extract_calls.append(audio_wav_bytes)
        return _ENROLL_EMB


class _NoEnrollTTSBackend(_FakeTTSBackend):
    """Fake backend with no CPU-ONNX enroller (Jetson without speaker_encoder)."""

    name = "fake-no-enroll"
    supports_voice_enrollment = False


def _make_client(monkeypatch, tmp_path, tts_be):
    from server.core import tts_runtime, tts_service, session_limiter

    monkeypatch.setenv("SPARKTTS_VOICES_DIR", str(tmp_path))
    tts_runtime.reset_overrides()
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"max_concurrent_sessions": 64})

    _install_managers(asr=_FakeASRBackend(), tts=tts_be)

    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: tts_be)
    monkeypatch.setattr(tts_service, "is_configured", lambda: True)
    monkeypatch.setattr(tts_service, "_backend", tts_be, raising=False)
    monkeypatch.setattr(tts_service, "backend_name", lambda: tts_be.name)

    from server.main import app
    from server.core.admin_auth import require_admin

    async def _allow():
        return None

    app.dependency_overrides[require_admin] = _allow
    c = TestClient(app)
    c.tts_be = tts_be  # type: ignore[attr-defined]
    return c


@pytest.fixture
def enroll_client(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path, _EnrollTTSBackend())
    try:
        yield c
    finally:
        from server.main import app
        from server.core.admin_auth import require_admin
        from server.core import backend_manager as bm, session_limiter, tts_runtime
        app.dependency_overrides.pop(require_admin, None)
        tts_runtime.reset_overrides()
        session_limiter._reset_for_tests()
        bm._reset_for_tests()


def test_enroll_via_onnx_persists_embedding_profile(enroll_client, tmp_path):
    # torch enroller absent → the ONNX path is the only way this can succeed.
    from server.core import sparktts_voices
    wav = b"RIFFxxxxWAVE" + b"\x00" * 64

    r = enroll_client.post(
        "/tts/voices/enroll",
        data={"voice_id": "clone:onnx"},
        files={"file": ("ref.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["voice_id"] == "clone:onnx"
    assert body["method"] == "onnx_speaker_encoder"
    assert body["profile_type"] == "speaker_embedding"

    # Backend extractor actually ran on the uploaded audio.
    assert enroll_client.tts_be.extract_calls == [wav]

    # Persisted profile round-trips to the exact float32 embedding.
    raw = sparktts_voices.load_embedding_voice("clone:onnx")
    assert raw == _ENROLL_EMB
    assert (tmp_path / "clone_onnx.json").exists()
    assert (tmp_path / "clone_onnx.npz").exists()


def test_stream_resolves_enrolled_voice_to_speaker_embedding(enroll_client):
    """enroll → /tts voice="<id>" closes the loop: backend gets speaker_embedding."""
    enroll_client.post(
        "/tts/voices/enroll",
        data={"voice_id": "clone:loop"},
        files={"file": ("ref.wav", b"RIFFxxxxWAVE" + b"\x00" * 64, "audio/wav")},
    )
    enroll_client.tts_be.synthesize_calls.clear()

    r = enroll_client.post("/tts", json={"text": "hi", "voice": "clone:loop"})
    assert r.status_code == 200, r.text

    call = enroll_client.tts_be.synthesize_calls[-1]
    # Server-side resolution: raw float32 bytes forwarded as speaker_embedding,
    # and the opaque `voice` selector dropped (BASE backend has no registry).
    assert call.get("speaker_embedding") == _ENROLL_EMB
    assert "voice" not in call


def test_capabilities_exposes_supports_voice_enrollment(enroll_client):
    r = enroll_client.get("/tts/capabilities")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["supports_voice_enrollment"] is True
    # existing signal untouched (zero-regression)
    assert body["supports_voice_cloning"] is True


def test_enroll_501_when_no_onnx_and_no_torch(monkeypatch, tmp_path):
    """No CPU-ONNX enroller + no torch stack → honest 501 (not a crash)."""
    from server.core import sparktts_voices
    monkeypatch.setattr(sparktts_voices, "_load_enroller", lambda md: None)

    c = _make_client(monkeypatch, tmp_path, _NoEnrollTTSBackend())
    try:
        r = c.post(
            "/tts/voices/enroll",
            data={"voice_id": "clone:x"},
            files={"file": ("ref.wav", b"\x00\x00", "audio/wav")},
        )
        assert r.status_code == 501, r.text
        assert "hint" in r.json()
    finally:
        from server.main import app
        from server.core.admin_auth import require_admin
        from server.core import backend_manager as bm, session_limiter, tts_runtime
        app.dependency_overrides.pop(require_admin, None)
        tts_runtime.reset_overrides()
        session_limiter._reset_for_tests()
        bm._reset_for_tests()


def test_capabilities_false_when_backend_cannot_enroll(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path, _NoEnrollTTSBackend())
    try:
        r = c.get("/tts/capabilities")
        assert r.status_code == 200, r.text
        assert r.json()["supports_voice_enrollment"] is False
    finally:
        from server.main import app
        from server.core.admin_auth import require_admin
        from server.core import backend_manager as bm, session_limiter, tts_runtime
        app.dependency_overrides.pop(require_admin, None)
        tts_runtime.reset_overrides()
        session_limiter._reset_for_tests()
        bm._reset_for_tests()
