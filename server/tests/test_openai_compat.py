"""Contract tests for the small OpenAI-compatible audio adapter."""

from __future__ import annotations

import io
import struct
from contextlib import asynccontextmanager

import pytest


def _wav(payload: bytes = b"\x01\x00\x02\x00", *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    block_align = channels * 2
    byte_rate = sample_rate * block_align
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, 16)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


@pytest.fixture()
def openai_client(monkeypatch):
    pytest.importorskip("prometheus_client")
    from fastapi.testclient import TestClient
    from server import main
    from server.core import coordinator, profile_loader, session_limiter, tts_runtime
    from server.core.asr_backend import TranscriptionResult

    class TTS:
        name = "fake-tts"
        model_id = "qwen3-tts-customvoice"
        sample_rate = 16000
        capabilities = set()
        supports_voice_cloning = False

        def __init__(self):
            self.calls = []
            self.output = _wav()

        def is_ready(self):
            return True

        def rate_pitch_caps(self):
            return True, True

        def synthesize(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return self.output, {"duration": 1, "inference_time": 0.01, "rtf": 0.1}

    class ASR:
        name = "fake-asr"
        model_id = "qwen3-asr-0.6b"

        def is_ready(self):
            return True

        def transcribe(self, audio, language="auto"):
            self.calls.append((audio, language))
            return TranscriptionResult("decoded text", language)

        def __init__(self):
            self.calls = []

    class Manager:
        def __init__(self, backend):
            self.backend = backend
            self.state = "ready"

        def is_ready(self):
            return True

        def get_backend_unsafe(self):
            return self.backend

        @asynccontextmanager
        async def acquire(self):
            yield self.backend

    tts = TTS()
    asr = ASR()
    tts_manager = Manager(tts)
    asr_manager = Manager(asr)
    tts_runtime.reset_overrides()
    monkeypatch.setattr(main, "_get_tts_manager", lambda: tts_manager)
    monkeypatch.setattr(main, "_get_asr_manager", lambda: asr_manager)

    async def _ensure_tts():
        return tts_manager

    monkeypatch.setattr(main, "_ensure_tts_manager_started", _ensure_tts)
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "tts_backend": "fake.tts",
            "tts_model_id": "qwen3-tts-customvoice",
            "asr_backend": "fake.asr",
            "asr_model_id": "qwen3-asr-0.6b",
        },
    )
    coordinator._coordinator = None
    coordinator.init_coordinator({"mode": "concurrent"})
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"max_concurrent_sessions": 4})
    client = TestClient(main.app)
    try:
        yield client, tts, asr, tts_manager, asr_manager
    finally:
        tts_runtime.reset_overrides()
        session_limiter._reset_for_tests()


def test_speech_wav_and_pcm_strip_only_wav_header(openai_client):
    client, tts, _asr, _tm, _am = openai_client
    wav = client.post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-customvoice", "input": "hello", "voice": "vivian"},
    )
    assert wav.status_code == 200, wav.text
    assert wav.headers["content-type"].startswith("audio/wav")
    assert wav.content.startswith(b"RIFF")
    assert tts.calls[-1][1]["speaker_id"] == 3065

    pcm = client.post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-customvoice", "input": "hello", "response_format": "pcm"},
    )
    assert pcm.status_code == 200, pcm.text
    assert pcm.headers["content-type"].startswith("audio/pcm")
    assert pcm.content == b"\x01\x00\x02\x00"
    assert pcm.headers["x-sample-rate"] == "16000"


@pytest.mark.parametrize(
    ("body", "code", "status"),
    [
        ({"input": "x"}, "missing_required_parameter", 400),
        ({"model": "other", "input": "x"}, "unknown_model", 404),
        ({"model": "qwen3-tts-customvoice", "input": "x", "response_format": "mp3"}, "unsupported_format", 400),
        ({"model": "qwen3-tts-customvoice", "input": "x", "speed": 5}, "unsupported_control", 400),
    ],
)
def test_speech_strict_errors(openai_client, body, code, status):
    response = openai_client[0].post("/v1/audio/speech", json=body)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "type" in response.json()["error"]


def test_speech_rejects_malformed_backend_wav_for_pcm(openai_client):
    client, tts, _asr, _tm, _am = openai_client
    tts.output = b"not wav"
    response = client.post(
        "/v1/audio/speech",
        json={"model": tts.model_id, "input": "x", "response_format": "pcm"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "invalid_backend_audio"
    assert response.json()["error"]["type"] == "server_error"


def test_pcm_parser_rejects_chunks_outside_declared_riff_boundary():
    from server.api.openai_compat import _wav_pcm_payload

    valid_chunks = _wav()[12:]
    forged = b"RIFF" + struct.pack("<I", 4) + b"WAVE" + valid_chunks
    with pytest.raises(ValueError, match="missing fmt or data"):
        _wav_pcm_payload(forged)


@pytest.mark.parametrize(
    ("channels", "block_align"),
    [(3, 6), (1, 4)],
)
def test_pcm_parser_rejects_unsupported_channels_and_bad_alignment(channels, block_align):
    from server.api.openai_compat import _wav_pcm_payload

    sample_rate = 16000
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        16,
    )
    payload = b"\x00" * max(block_align, 2)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    wav = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
    with pytest.raises(ValueError):
        _wav_pcm_payload(wav)


def test_transcriptions_json_text_and_default_language(openai_client):
    client, _tts, asr, _tm, _am = openai_client
    fields = {"model": "qwen3-asr", "language": "", "response_format": "json", "temperature": "0"}
    response = client.post(
        "/v1/audio/transcriptions",
        data=fields,
        files={"file": ("sample.wav", b"pcm", "audio/wav")},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"text": "decoded text"}
    assert asr.calls[-1] == (b"pcm", "auto")

    text = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-asr", "response_format": "text"},
        files={"file": ("sample.wav", b"pcm", "audio/wav")},
    )
    assert text.status_code == 200
    assert text.headers["content-type"].startswith("text/plain")
    assert text.text == "decoded text"


def test_transcriptions_use_http_admission_lease(openai_client, monkeypatch):
    from server.core import session_limiter

    entered: list[str] = []

    @asynccontextmanager
    async def fake_acquire(endpoint: str):
        entered.append(endpoint)
        yield

    monkeypatch.setattr(session_limiter, "acquire_http", fake_acquire)
    response = openai_client[0].post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-asr"},
        files={"file": ("sample.wav", b"pcm", "audio/wav")},
    )
    assert response.status_code == 200, response.text
    assert entered == ["/v1/audio/transcriptions"]


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("prompt", "hello", "unsupported_control"),
        ("temperature", "0.2", "unsupported_control"),
        ("timestamp_granularities[]", "segment", "unsupported_control"),
        ("verbose_json", "true", "unsupported_format"),
        ("response_format", "srt", "unsupported_format"),
    ],
)
def test_transcriptions_reject_unsupported_controls(openai_client, field, value, code):
    response = openai_client[0].post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-asr", field: value},
        files={"file": ("sample.wav", b"pcm", "audio/wav")},
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == code


def test_transcriptions_bounded_decoded_audio(openai_client, monkeypatch):
    client = openai_client[0]
    monkeypatch.setenv("OVS_API_MAX_AUDIO_BYTES", "3")
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-asr"},
        files={"file": ("sample.wav", b"1234", "audio/wav")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_models_is_openai_list_deduped_and_canonical(openai_client):
    response = openai_client[0].get("/v1/models")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "list"
    rows = {item["id"]: item for item in body["data"]}
    assert set(rows) == {"qwen3-tts-customvoice", "qwen3-asr"}
    assert rows["qwen3-tts-customvoice"]["modalities"] == ["tts"]
    assert rows["qwen3-asr"]["modalities"] == ["asr"]
    assert "qwen3-asr-0.6b" in rows["qwen3-asr"]["aliases"]
    assert "qwen3-asr" not in rows["qwen3-asr"]["aliases"]
    assert rows["qwen3-tts-customvoice"]["ready"] is True
    assert rows["qwen3-asr"]["ready"] is True
    assert rows["qwen3-tts-customvoice"]["created"] == 0


def test_audio_auth_is_normalized_and_preserves_www_authenticate(openai_client, monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", "secret")
    response = openai_client[0].post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-customvoice", "input": "x"},
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["type"] == "authentication_error"


def test_audio_non_multipart_and_missing_fields_are_enveloped(openai_client):
    client = openai_client[0]
    bad_type = client.post("/v1/audio/transcriptions", json={"model": "qwen3-asr"})
    assert bad_type.status_code == 400
    assert bad_type.json()["error"]["code"] == "invalid_multipart"
    missing = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-asr"},
        files={"other": ("sample.wav", b"pcm", "audio/wav")},
    )
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == "missing_required_parameter"
