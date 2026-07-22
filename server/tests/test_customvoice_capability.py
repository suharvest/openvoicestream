"""Regression tests for the Qwen3-CustomVoice capability gating.

Covers the three HIGH-severity bugs flagged in CTO review of
`feat(tts): Qwen3-CustomVoice backend + capability-aware clone fallback`:

1. ``QWEN3_TTS_VARIANT=customvoice`` flips ``supports_voice_cloning`` to False
   AND pins the backend ``model_id`` to ``qwen3-tts-customvoice`` (Bug 2 —
   without the auto-switch the 9 built-in speakers in that table become
   invisible because ``self.model_id`` would resolve to base ``qwen3-tts``).
2. ``/tts/clone`` returns 400 with the unified capability-aware payload when
   the active backend explicitly disables cloning.
3. ``/tts`` with ``speaker_embedding_b64`` returns 400 (not 500) — pre-response
   capability gate, mirrors /tts/clone semantics.
4. ``/tts/stream`` with ``speaker_embedding_b64`` returns 400 BEFORE any audio
   bytes / sample-rate header is on the wire (Bug 3 — without this fix the
   error surfaced mid-stream as a worker exception with the response already
   committed).

The tests reuse the fake-backend harness from ``test_main_hot_swap`` instead
of standing up the real Qwen3 TRT pipeline (which needs CUDA + engines).
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from server.tests.test_main_hot_swap import (
    _FakeASRBackend,
    _FakeTTSBackend,
    _install_managers,
)


# ---------------------------------------------------------------------------
# Bug 3 — HTTP endpoint capability gates
# ---------------------------------------------------------------------------


class _NoCloneTTSBackend(_FakeTTSBackend):
    """Fake backend that mirrors Qwen3-CustomVoice: streaming OK,
    cloning explicitly disabled."""

    def __init__(self) -> None:
        super().__init__()
        from server.core.tts_backend import TTSCapability
        self.capabilities = {TTSCapability.STREAMING, TTSCapability.BASIC_TTS}
        self.supports_voice_cloning = False

    name = "fake-customvoice"
    model_id = "qwen3-tts-customvoice"


@pytest.fixture
def no_clone_client(monkeypatch):
    """TestClient wired to a backend that disables voice cloning."""
    from server.core import tts_runtime, tts_service, session_limiter
    tts_runtime.reset_overrides()
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"limits": {"max_concurrent_sessions": 8}})

    tts_be = _NoCloneTTSBackend()
    _install_managers(asr=_FakeASRBackend(), tts=tts_be)

    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: tts_be)
    monkeypatch.setattr(tts_service, "is_configured", lambda: True)
    monkeypatch.setattr(tts_service, "_backend", tts_be, raising=False)
    monkeypatch.setattr(tts_service, "backend_name", lambda: tts_be.name)
    from server.core.tts_backend import TTSCapability
    monkeypatch.setattr(
        tts_service,
        "has_capability",
        lambda cap: cap in tts_be.capabilities,
    )

    from server.main import app
    from server.core.admin_auth import require_admin

    async def _allow():
        return None

    app.dependency_overrides[require_admin] = _allow
    c = TestClient(app)
    c.tts_be = tts_be  # type: ignore[attr-defined]
    try:
        yield c
    finally:
        app.dependency_overrides.pop(require_admin, None)
        tts_runtime.reset_overrides()
        session_limiter._reset_for_tests()
        from server.core import backend_manager as bm
        bm._reset_for_tests()


def test_tts_clone_returns_400_capability_payload(no_clone_client):
    """/tts/clone on a no-clone backend → 400 with capability payload."""
    r = no_clone_client.post(
        "/tts/clone",
        json={"text": "hi", "speaker_embedding_b64": "AAAAAA=="},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["required_capability"] == "voice_clone"
    assert body["supports_voice_cloning"] is False
    assert body["backend"] == "fake-customvoice"


def test_tts_with_embedding_returns_400(no_clone_client):
    """/tts (non-clone endpoint) carrying speaker_embedding_b64 → 400.

    Before the Bug 3 fix this leaked through to backend.synthesize() and
    raised NotImplementedError → FastAPI 500.
    """
    r = no_clone_client.post(
        "/tts",
        json={"text": "hi", "speaker_embedding_b64": "AAAAAA=="},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["required_capability"] == "voice_clone"
    assert body["supports_voice_cloning"] is False
    # backend.synthesize must NOT have been called — the gate fires
    # BEFORE the coordinator/synthesize boundary.
    assert no_clone_client.tts_be.synthesize_calls == []


def test_tts_stream_with_embedding_returns_400_pre_response(no_clone_client):
    """/tts/stream carrying speaker_embedding_b64 → 400 BEFORE the streaming
    body is opened. The pre-Bug-3 path opened a StreamingResponse, emitted
    the 4-byte sample-rate header, and only THEN raised in the worker
    thread, leaving the client with a half-written stream.
    """
    r = no_clone_client.post(
        "/tts/stream",
        json={"text": "hi", "speaker_embedding_b64": "AAAAAA=="},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["required_capability"] == "voice_clone"
    assert body["supports_voice_cloning"] is False
    # No streaming kicked off.
    assert no_clone_client.tts_be.streaming_calls == []
