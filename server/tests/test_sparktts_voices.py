"""SparkTTS clone voice registry/enrollment server layer (P3).

Exercises register-from-profile / list / delete against a temp voices dir,
plus the EnrollmentUnavailable fallback when the torch stack is absent.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from server.core import sparktts_voices


@pytest.fixture
def voices_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKTTS_VOICES_DIR", str(tmp_path))
    # Make sure no live backend registry interferes.
    monkeypatch.setattr(sparktts_voices, "_live_registry", lambda: None)
    return tmp_path


def _profile_bytes(voice_id="clone:t", *, ref_semantic=False):
    g = np.arange(32, dtype=np.int32)
    rs = np.array([5, 6, 7] if ref_semantic else [], dtype=np.int32)
    buf = io.BytesIO()
    np.savez(buf, global_ids=g, ref_semantic_ids=rs, d_vector=np.zeros(1024, np.float32))
    npz_bytes = buf.getvalue()
    j = {"voice_id": voice_id, "ref_text": "x" if ref_semantic else None,
         "sample_rate": 16000, "ref_semantic_len": len(rs)}
    return json.dumps(j).encode("utf-8"), npz_bytes


def test_register_list_delete(voices_env):
    jb, nb = _profile_bytes("clone:alice")
    res = sparktts_voices.register_from_profile_files(jb, nb)
    assert res["voice_id"] == "clone:alice"
    # on-disk pair exists with canonical names
    assert (voices_env / "clone_alice.json").exists()
    assert (voices_env / "clone_alice.npz").exists()

    listed = sparktts_voices.list_voices()
    assert any(v["voice_id"] == "clone:alice" for v in listed)

    assert sparktts_voices.delete_voice("clone:alice") is True
    assert not (voices_env / "clone_alice.json").exists()
    assert sparktts_voices.delete_voice("clone:alice") is False


def test_register_rejects_wrong_global_count(voices_env):
    g = np.arange(10, dtype=np.int32)  # not 32
    buf = io.BytesIO()
    np.savez(buf, global_ids=g)
    jb = json.dumps({"voice_id": "clone:bad"}).encode()
    with pytest.raises(ValueError):
        sparktts_voices.register_from_profile_files(jb, buf.getvalue())


def test_register_requires_voice_id(voices_env):
    g = np.arange(32, dtype=np.int32)
    buf = io.BytesIO()
    np.savez(buf, global_ids=g)
    jb = json.dumps({"sample_rate": 16000}).encode()  # no voice_id
    with pytest.raises(ValueError):
        sparktts_voices.register_from_profile_files(jb, buf.getvalue())


def test_register_voice_id_override(voices_env):
    jb, nb = _profile_bytes("clone:orig")
    res = sparktts_voices.register_from_profile_files(jb, nb, voice_id="clone:override")
    assert res["voice_id"] == "clone:override"
    assert (voices_env / "clone_override.json").exists()


def test_enroll_from_audio_unavailable_without_torch(voices_env, monkeypatch):
    monkeypatch.setattr(sparktts_voices, "_load_enroller", lambda md: None)
    with pytest.raises(sparktts_voices.EnrollmentUnavailable):
        sparktts_voices.enroll_from_audio(b"\x00\x00", "clone:x")


# --------------------------------------------------------------- embedding-profile


def test_register_embedding_voice_writes_float32_profile(voices_env):
    emb = np.arange(1024, dtype=np.float32)
    res = sparktts_voices.register_embedding_voice(
        "clone:emb", emb.tobytes(), sample_rate=24000, ref_text="hello",
    )
    assert res["voice_id"] == "clone:emb"
    assert res["profile_type"] == "speaker_embedding"
    assert res["embedding_dim"] == 1024

    jpath = voices_env / "clone_emb.json"
    npath = voices_env / "clone_emb.npz"
    assert jpath.exists() and npath.exists()

    j = json.loads(jpath.read_text())
    assert j["profile_type"] == "speaker_embedding"
    assert j["embedding_dim"] == 1024
    assert j["embedding_dtype"] == "float32"
    assert j["npz_file"] == "clone_emb.npz"
    assert j["sample_rate"] == 24000

    with np.load(npath) as npz:
        assert "speaker_embedding" in npz
        stored = npz["speaker_embedding"]
        assert stored.dtype == np.float32
        assert stored.shape == (1024,)
        np.testing.assert_array_equal(stored, emb)


def test_register_embedding_voice_rejects_non_float32_aligned(voices_env):
    with pytest.raises(ValueError):
        sparktts_voices.register_embedding_voice("clone:bad", b"\x00\x00\x00")  # 3 bytes


def test_load_embedding_voice_roundtrip(voices_env):
    emb = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    sparktts_voices.register_embedding_voice("clone:rt", emb.tobytes())
    raw = sparktts_voices.load_embedding_voice("clone:rt")
    assert raw is not None
    back = np.frombuffer(raw, dtype=np.float32)
    assert back.shape == (1024,)
    np.testing.assert_array_equal(back, emb)


def test_load_embedding_voice_none_for_unknown(voices_env):
    assert sparktts_voices.load_embedding_voice("clone:nope") is None


def test_load_embedding_voice_ignores_global_ids_profile(voices_env):
    """A SparkTTS global_ids clone is NOT an embedding-profile → returns None."""
    jb, nb = _profile_bytes("clone:sparky")
    sparktts_voices.register_from_profile_files(jb, nb)
    assert sparktts_voices.load_embedding_voice("clone:sparky") is None
