import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.asr_backend import ASRCapability, TranscriptionResult
from voxedge.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRBackend,
    _float_audio_to_wav_bytes,
)


def test_float_audio_to_wav_bytes_roundtrip_header():
    wav_bytes = _float_audio_to_wav_bytes(np.zeros(16000, dtype=np.float32), 16000)

    assert wav_bytes[:4] == b"RIFF"
    assert b"WAVE" in wav_bytes[:16]


def test_trt_edgellm_asr_stream_accumulates_and_finalizes(monkeypatch):
    backend = TRTEdgeLLMASRBackend()
    backend._ready = True
    calls = []

    def fake_transcribe(wav_bytes, language="auto"):
        calls.append((wav_bytes, language))
        return type("Result", (), {"text": "你好"})()

    monkeypatch.setattr(backend, "transcribe", fake_transcribe)
    stream = backend.create_stream(language="Chinese")
    stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))
    stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))

    assert stream.get_partial() == ("", False)
    # finalize() now returns (text, detected_language) per the
    # language-pipeline migration. Backend's fake_transcribe returns a
    # plain Result without language → detected stays None.
    assert stream.finalize() == ("你好", None)
    assert calls[0][1] == "Chinese"
    assert calls[0][0][:4] == b"RIFF"


def test_trt_edgellm_asr_advertises_streaming_capability():
    backend = TRTEdgeLLMASRBackend()

    assert ASRCapability.STREAMING in backend.capabilities


def test_trt_edgellm_asr_offline_transcribe_segments_long_audio(monkeypatch):
    import voxedge.backends.jetson.trt_edge_llm_asr as asr_mod

    monkeypatch.setenv("OVS_VAD_BACKEND", "none")
    monkeypatch.setenv("EDGE_LLM_ASR_OFFLINE_SEGMENT_SEC", "2.0")
    audio = np.ones(16000 * 5, dtype=np.float32) * 0.02
    wav_bytes = _float_audio_to_wav_bytes(audio, 16000)

    def fake_split(samples, sample_rate=16000):
        assert sample_rate == 16000
        return [
            samples[: sample_rate * 2],
            samples[sample_rate * 2 : sample_rate * 4],
            samples[sample_rate * 4 :],
        ]

    calls = []

    def fake_mel(segment_wav):
        calls.append(segment_wav)
        return np.zeros((1, 128, 100), dtype=np.float32)

    def fake_worker(mel_path, elapsed_mel_s):
        idx = len(calls)
        return TranscriptionResult(
            text=f"第{idx}段。",
            language="Chinese",
            inference_time_s=0.1,
            mel_time_s=0.01,
            worker_time_s=0.09,
        )

    monkeypatch.setattr(asr_mod, "_split_at_silence_vad", fake_split)
    monkeypatch.setattr(asr_mod, "audio_bytes_to_mel", fake_mel)

    backend = TRTEdgeLLMASRBackend()
    backend._ready = True
    monkeypatch.setattr(backend, "_use_worker", lambda: True)
    monkeypatch.setattr(backend, "_transcribe_worker", fake_worker)

    result = backend.transcribe(wav_bytes, language="Chinese")

    assert result.text == "第1段第2段第3段。"
    assert result.language == "Chinese"
    assert result.meta["segmented"] is True
    assert result.meta["segment_count"] == 3
    assert len(calls) == 3


def test_trt_edgellm_asr_offline_split_uses_configured_vad(monkeypatch):
    import app.core.vad as vad_mod
    import voxedge.backends.jetson.trt_edge_llm_asr as asr_mod

    class ScriptedVAD:
        def __init__(self):
            self.calls = 0

        def process(self, samples):
            self.calls += 1
            return vad_mod.VADSession.SPEECH_END if self.calls == 25 else None

        def reset(self):
            self.calls = 0

    created = {}

    def fake_create_vad(backend, sample_rate, silence_ms, **kwargs):
        created["backend"] = backend
        created["sample_rate"] = sample_rate
        created["silence_ms"] = silence_ms
        return ScriptedVAD()

    monkeypatch.setenv("OVS_VAD_BACKEND", "silero")
    monkeypatch.setenv("OVS_VAD_SILENCE_MS", "320")
    monkeypatch.setenv("EDGE_LLM_ASR_OFFLINE_SEGMENT_SEC", "10")
    monkeypatch.setattr(vad_mod, "create_vad", fake_create_vad)

    audio = np.ones(16000, dtype=np.float32) * 0.02
    segments = asr_mod._split_offline_audio(audio, 16000, max_segment_s=10)

    assert created == {"backend": "silero", "sample_rate": 16000, "silence_ms": 320}
    assert len(segments) == 2


# ── worker-error classification ─────────────────────────────────────────


from voxedge.backends.jetson.trt_edge_llm_asr import (  # noqa: E402
    NoActiveSessionError,
    SessionAlreadyActiveError,
    WorkerExitError,
    WorkerProtocolError,
    _classify_worker_response,
)


def test_classify_no_active_session():
    err = _classify_worker_response({"event": "error", "error": "no active session for id=abc"})
    assert isinstance(err, NoActiveSessionError)


def test_classify_session_already_active():
    err = _classify_worker_response({"event": "error", "error": "session already active"})
    assert isinstance(err, SessionAlreadyActiveError)


def test_classify_worker_exit():
    err = _classify_worker_response({"ok": False, "error": "worker terminated unexpectedly"})
    assert isinstance(err, WorkerExitError)


def test_classify_unknown_error_returns_none():
    # Returning None means "not a typed protocol error" — _worker_request
    # will still raise a generic WorkerProtocolError.
    assert _classify_worker_response({"event": "error", "error": "decoder failed"}) is None
    assert _classify_worker_response({"ok": True}) is None


def test_worker_request_injects_typed_no_active_session(monkeypatch):
    backend = TRTEdgeLLMASRBackend()

    def fake_request(input_data):
        # Simulate the real path's behaviour using the same parser:
        # bypass actual subprocess by re-implementing the protocol shim.
        line = '{"event":"error","error":"no active session"}\n'
        import json as _j
        out = _j.loads(line)
        typed = _classify_worker_response(out, request_event=input_data.get("event"))
        if typed is not None:
            raise typed
        raise RuntimeError(out)

    monkeypatch.setattr(backend, "_worker_request", fake_request)
    import pytest
    with pytest.raises(NoActiveSessionError):
        backend._worker_request({"event": "chunk", "id": "x"})


def test_worker_request_injects_typed_session_already_active(monkeypatch):
    backend = TRTEdgeLLMASRBackend()

    def fake_request(input_data):
        import json as _j
        out = _j.loads('{"event":"error","error":"session already active for id=x"}')
        typed = _classify_worker_response(out)
        if typed is not None:
            raise typed
        raise RuntimeError(out)

    monkeypatch.setattr(backend, "_worker_request", fake_request)
    import pytest
    with pytest.raises(SessionAlreadyActiveError):
        backend._worker_request({"event": "begin", "id": "x"})


def test_worker_request_injects_worker_exit_on_empty_line():
    # Direct unit check that the bare exit path raises WorkerExitError.
    err = _classify_worker_response({"event": "error", "error": "worker exited"})
    assert isinstance(err, WorkerExitError)


def test_worker_request_broken_pipe_raises_worker_exit():
    """SIGKILL of the worker causes BrokenPipeError on stdin.write. Without
    classification, that escapes as a raw IOError and the session manager
    cannot route it to ERROR_REBUILD. After the WorkerIO migration,
    _worker_request must still surface this as WorkerExitError so
    restart_worker fires.
    """
    import pytest
    from app.core.worker_io import WorkerIO
    backend = TRTEdgeLLMASRBackend()

    class _DeadStdin:
        def write(self, _payload):
            raise BrokenPipeError("worker dead")
        def flush(self):
            raise BrokenPipeError("worker dead")

    class _Stdout:
        # WorkerIO's reader thread iterates stdout; an empty iterator
        # just causes it to EOF immediately (we never get there because
        # the stdin.write above fires first).
        def __iter__(self):
            return iter(())

    class _Worker:
        stdin = _DeadStdin()
        stdout = _Stdout()

    proc = _Worker()
    backend._worker = proc
    backend._wio = WorkerIO(proc, concurrency=1)
    backend._ensure_worker = lambda: None  # don't try to spawn

    with pytest.raises(WorkerExitError):
        backend._worker_request({"event": "begin", "id": "x"})
    assert backend._worker is None  # cleared so next call rebuilds


def test_restart_worker_is_idempotent_with_no_running_worker():
    """restart_worker() must be safe to call when nothing is running."""
    backend = TRTEdgeLLMASRBackend()
    assert backend._worker is None
    backend.restart_worker()  # no-op, must not raise
    assert backend._worker is None


def test_typed_errors_subclass_worker_protocol_error():
    assert issubclass(NoActiveSessionError, WorkerProtocolError)
    assert issubclass(SessionAlreadyActiveError, WorkerProtocolError)
    assert issubclass(WorkerExitError, WorkerProtocolError)


def test_supports_hot_reload_true_when_worker_mode():
    backend = TRTEdgeLLMASRBackend()
    backend._config["use_worker"] = True
    assert backend.supports_hot_reload is True


def test_supports_hot_reload_false_when_inprocess():
    backend = TRTEdgeLLMASRBackend()
    backend._config["use_worker"] = False
    assert backend.supports_hot_reload is False


def test_unload_idempotent_when_not_ready():
    backend = TRTEdgeLLMASRBackend()
    # Fresh: _ready=False, _worker=None — must early return without raising.
    backend.unload()
    assert backend._ready is False
    assert backend._worker is None


def test_unload_kills_worker_and_marks_not_ready(monkeypatch):
    backend = TRTEdgeLLMASRBackend()
    backend._ready = True
    called = {"restart": 0}

    def fake_restart():
        called["restart"] += 1
        backend._worker = None

    monkeypatch.setattr(backend, "restart_worker", fake_restart)

    class _Dummy:
        def poll(self):
            return None

    backend._worker = _Dummy()
    backend.unload()
    assert called["restart"] == 1
    assert backend._ready is False
    assert backend._worker is None
