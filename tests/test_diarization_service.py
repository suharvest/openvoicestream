"""Unit tests for the diarization product shim (server/core/diarization.py).

CPU-only, no real CAM++ model: ``compute_embedding`` is monkeypatched to return
synthetic prototype vectors so we exercise the session orchestration, offline
segmentation + clustering, and the feature-flag default-off behaviour without a
profile or model download. The clustering kernel itself lives in voxedge and is
tested there; here we verify the product layer wires it correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from server.core import diarization as diar
from server.core import speaker_embedding as spk

DIM = 192


def _proto(idx: int) -> np.ndarray:
    """A unit basis vector — distinct prototypes are orthogonal (cosine 0)."""
    v = np.zeros(DIM, dtype=np.float32)
    v[idx] = 1.0
    return v


PROTO_A = _proto(0)
PROTO_B = _proto(1)


# ── feature flag (default OFF) ───────────────────────────────────────────────

def test_diarize_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OVS_DIARIZE", raising=False)
    assert diar.diarize_enabled() is False


def test_diarize_enabled_via_env(monkeypatch):
    monkeypatch.setenv("OVS_DIARIZE", "true")
    assert diar.diarize_enabled() is True
    monkeypatch.setenv("OVS_DIARIZE", "0")
    assert diar.diarize_enabled() is False


# ── online session orchestration ─────────────────────────────────────────────

def test_session_diarizer_assigns_multiple_speakers(monkeypatch):
    monkeypatch.delenv("OVS_DIARIZE", raising=False)  # params from defaults
    d = diar.make_session_diarizer()
    assert d is not None

    # A, B, A → two distinct speakers, A re-identified on its second turn.
    s0 = d.assign(PROTO_A, 0.0, 1.0)
    s1 = d.assign(PROTO_B, 1.5, 2.5)
    s2 = d.assign(PROTO_A, 3.0, 4.0)

    assert s0.speaker == "spk_0"
    assert s1.speaker == "spk_1"
    assert s2.speaker == "spk_0"            # same speaker as s0
    assert d.num_speakers == 2
    # Confidence is the cosine to the cluster centroid (orthogonal protos → ~1).
    assert s2.confidence >= 0.9


def test_summary_payload_relabels(monkeypatch):
    d = diar.make_session_diarizer()
    d.assign(PROTO_A, 0.0, 1.0)
    d.assign(PROTO_B, 1.5, 2.5)
    d.assign(PROTO_A, 3.0, 4.0)
    summary = diar.summary_payload(d)
    assert summary is not None
    assert summary["type"] == "diarization_summary"
    assert summary["num_speakers"] == 2
    assert len(summary["segments"]) == 3
    # Segments are time-ordered with the contract fields.
    for seg in summary["segments"]:
        assert set(seg) >= {"start", "end", "speaker", "confidence"}


def test_summary_payload_none_for_empty():
    d = diar.make_session_diarizer()
    assert diar.summary_payload(d) is None
    assert diar.summary_payload(None) is None


# ── offline diarize_audio (segmentation + clustering) ────────────────────────

def _make_two_speaker_audio(sr: int = 16000):
    """Three 0.6 s speech spans (A, B, A) separated by 0.5 s of silence.

    Speaker identity is encoded in amplitude so the monkeypatched embedder can
    return the right prototype from the audio slice alone.
    """
    def tone(amp: float, dur: float):
        t = np.arange(int(sr * dur)) / sr
        return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    a1 = tone(0.5, 0.6)   # speaker A (loud)
    b = tone(0.2, 0.6)    # speaker B (quiet)
    a2 = tone(0.5, 0.6)   # speaker A again
    return np.concatenate([a1, silence, b, silence, a2]), sr


def _fake_embed(samples, sample_rate):
    rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float32) ** 2)))
    return PROTO_A if rms > 0.25 else PROTO_B


def test_diarize_audio_segments_and_clusters(monkeypatch):
    monkeypatch.setattr(spk, "compute_embedding", _fake_embed)
    audio, sr = _make_two_speaker_audio()

    segs = diar.diarize_audio(audio, sr)
    # Three speech spans detected, two distinct speakers.
    assert len(segs) == 3
    speakers = [s.speaker for s in segs]
    assert len(set(speakers)) == 2
    # First and last span are the same (loud) speaker; middle is the other.
    assert speakers[0] == speakers[2]
    assert speakers[1] != speakers[0]
    # Time-ordered, monotonic, within clip bounds.
    assert segs[0].start < segs[1].start < segs[2].start
    assert segs[-1].end <= len(audio) / sr + 1e-3


def test_diarize_audio_respects_num_speakers(monkeypatch):
    monkeypatch.setattr(spk, "compute_embedding", _fake_embed)
    audio, sr = _make_two_speaker_audio()
    segs = diar.diarize_audio(audio, sr, num_speakers=1)
    assert len({s.speaker for s in segs}) == 1   # forced into one cluster


def test_diarize_response_envelope(monkeypatch):
    monkeypatch.setattr(spk, "compute_embedding", _fake_embed)
    audio, sr = _make_two_speaker_audio()
    segs = diar.diarize_audio(audio, sr)

    resp = diar.diarize_response(segs, return_embeddings=False)
    assert resp["num_speakers"] == 2
    assert resp["embedding_model"]
    assert all("embedding_b64" not in s for s in resp["segments"])

    resp_emb = diar.diarize_response(segs, return_embeddings=True)
    assert resp_emb["dim"] == DIM
    assert all("embedding_b64" in s for s in resp_emb["segments"])


def test_diarize_audio_empty_when_no_embeddings(monkeypatch):
    # Model unavailable → compute_embedding returns None → empty result.
    monkeypatch.setattr(spk, "compute_embedding", lambda s, sr: None)
    audio, sr = _make_two_speaker_audio()
    assert diar.diarize_audio(audio, sr) == []


def test_diarize_audio_silent_input(monkeypatch):
    monkeypatch.setattr(spk, "compute_embedding", _fake_embed)
    silent = np.zeros(16000, dtype=np.float32)
    assert diar.diarize_audio(silent, 16000) == []
