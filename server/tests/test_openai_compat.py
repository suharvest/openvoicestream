"""OpenAI Audio Transcriptions compatibility — POST /v1/audio/transcriptions.

Contract coverage for the OpenAI-compatible shim over ``_asr_impl``:

  1. Default (``json``) response is ``{"text": ...}``.
  2. ``response_format=text`` returns plain text.
  3. ``response_format=verbose_json`` carries task/language/duration/text/segments.
  4. ``model`` is accepted-and-ignored; ``language`` is forwarded to the backend.
  5. ``OVS_API_KEYS`` gating: missing key → 401, correct key → 200.
  6. ``response_format=srt`` (and ``vtt``) → 400 unsupported.
  7. 100 back-to-back requests all succeed with a stable response shape.

Reuses the fake-backend manager harness from ``test_main_hot_swap`` (no GPU,
no real models).
"""
from __future__ import annotations

import io
import wave

import pytest
from fastapi.testclient import TestClient

from server.tests.test_main_hot_swap import _FakeASRBackend, _install_managers


class _FakeResult:
    def __init__(self, text: str, language: str):
        self.text = text
        self.language = language
        self.meta: dict = {}


class _TranscribingASRBackend(_FakeASRBackend):
    """Fake ASR backend returning a fixed transcript and recording calls."""

    name = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> _FakeResult:
        self.calls.append({"language": language, "n_bytes": len(audio_bytes)})
        return _FakeResult(text="你好世界", language="zh")


def _make_wav(seconds: float = 3.0, sample_rate: int = 16000) -> bytes:
    """PCM16 mono silence WAV of the given duration (stdlib only)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return buf.getvalue()


@pytest.fixture
def harness(monkeypatch):
    """(client, fake_backend) with managers/limiter/coordinator installed."""
    monkeypatch.delenv("OVS_API_KEYS", raising=False)
    asr_be = _TranscribingASRBackend()
    _install_managers(asr=asr_be)
    from server.main import app
    return TestClient(app), asr_be


def _post(client: TestClient, data: dict | None = None, headers: dict | None = None):
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_wav(), "audio/wav")},
        data=data or {},
        headers=headers or {},
    )


def test_default_json_format(harness):
    client, _ = harness
    r = _post(client)
    assert r.status_code == 200
    assert r.json() == {"text": "你好世界"}


def test_text_format_returns_plain_text(harness):
    client, _ = harness
    r = _post(client, data={"response_format": "text"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text == "你好世界"


def test_verbose_json_format(harness):
    client, _ = harness
    r = _post(client, data={"response_format": "verbose_json"})
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "zh"
    assert body["text"] == "你好世界"
    assert body["segments"] == []
    # 3 s of 16 kHz mono PCM16 — duration derived from the WAV header.
    assert body["duration"] == pytest.approx(3.0, abs=0.01)


def test_model_ignored_and_language_forwarded(harness):
    client, asr_be = harness
    r = _post(client, data={"model": "whisper-1", "language": "zh"})
    assert r.status_code == 200
    assert r.json() == {"text": "你好世界"}
    assert asr_be.calls[-1]["language"] == "zh"


def test_missing_language_defaults_to_auto(harness):
    client, asr_be = harness
    r = _post(client)
    assert r.status_code == 200
    assert asr_be.calls[-1]["language"] == "auto"


def test_api_key_enforcement(harness, monkeypatch):
    client, _ = harness
    monkeypatch.setenv("OVS_API_KEYS", "test-key-123")

    r = _post(client)
    assert r.status_code == 401

    r = _post(client, headers={"Authorization": "Bearer test-key-123"})
    assert r.status_code == 200
    assert r.json() == {"text": "你好世界"}


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_unsupported_formats_rejected(harness, fmt):
    client, asr_be = harness
    before = len(asr_be.calls)
    r = _post(client, data={"response_format": fmt})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "not supported" in detail["error"]
    # Rejected before the backend is invoked.
    assert len(asr_be.calls) == before


def test_100_sequential_requests_stable(harness):
    client, asr_be = harness
    for i in range(100):
        r = _post(client, data={"language": "zh"})
        assert r.status_code == 200, f"request {i} failed: {r.status_code} {r.text}"
        assert r.json() == {"text": "你好世界"}
    assert len(asr_be.calls) >= 100
