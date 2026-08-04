"""SparkTTS clone voice registry/enrollment server layer (P3).

Exercises register-from-profile / list / delete against a temp voices dir,
plus the EnrollmentUnavailable fallback when the torch stack is absent.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

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


def test_profile_metadata_is_canonical_and_model_filter_isolated(voices_env):
    jb, nb = _profile_bytes("clone:spark")
    sparktts_voices.register_from_profile_files(jb, nb)
    sparktts_voices.register_embedding_voice("clone:qwen", np.arange(8, dtype=np.float32).tobytes())

    spark = json.loads((voices_env / "clone_spark.json").read_text())
    assert spark["model_id"] == sparktts_voices.SPARK_MODEL_ID
    assert spark["profile_type"] == sparktts_voices.SPARK_PROFILE_TYPE
    assert spark["compatible_models"] == [sparktts_voices.SPARK_MODEL_ID]

    qwen = json.loads((voices_env / "clone_qwen.json").read_text())
    assert qwen["model_id"] == sparktts_voices.QWEN_BASE_MODEL_ID
    assert qwen["profile_type"] == sparktts_voices.EMBEDDING_PROFILE_TYPE
    assert qwen["compatible_models"] == [sparktts_voices.QWEN_BASE_MODEL_ID]

    assert [v["voice_id"] for v in sparktts_voices.list_voices(model_id="sparktts-0p5b")] == [
        "clone:spark"
    ]
    assert [v["voice_id"] for v in sparktts_voices.list_voices(model_id="qwen3-tts-0.6b-base")] == [
        "clone:qwen"
    ]
    assert sparktts_voices.load_embedding_voice("clone:spark") is None
    assert sparktts_voices.load_embedding_voice("clone:qwen", model_id="sparktts-0p5b") is None


def test_legacy_spark_profile_is_read_only_and_legacy_embedding_fails_closed(voices_env):
    # Profiles generated before the metadata contract can be migrated only
    # when the NPZ shape unambiguously identifies Spark global ids.
    (voices_env / "legacy_spark.json").write_text(json.dumps({"voice_id": "legacy:spark"}))
    buf = io.BytesIO()
    np.savez(buf, global_ids=np.arange(32, dtype=np.int32))
    (voices_env / "legacy_spark.npz").write_bytes(buf.getvalue())
    assert sparktts_voices.list_voices(model_id=sparktts_voices.SPARK_MODEL_ID)[0]["legacy_metadata"]

    # An old embedding profile has no safe model inference and must not leak
    # into either backend's capabilities.
    (voices_env / "legacy_embedding.json").write_text(json.dumps({"voice_id": "legacy:embedding"}))
    buf = io.BytesIO()
    np.savez(buf, speaker_embedding=np.zeros(8, dtype=np.float32))
    (voices_env / "legacy_embedding.npz").write_bytes(buf.getvalue())
    assert all(v["voice_id"] != "legacy:embedding" for v in sparktts_voices.list_voices())

    with pytest.raises(ValueError, match="legacy profile lacks canonical metadata"):
        sparktts_voices.register_from_profile_files(*_profile_bytes("legacy:spark"))


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


def test_enroll_from_audio_stages_then_uses_atomic_registry_commit(voices_env, monkeypatch):
    class FakeEnroller:
        def __init__(self):
            self.stage_dirs = []

        def enroll(self, wav_path, voice_id, ref_text):
            assert os.path.exists(wav_path)
            return {"voice_id": voice_id, "ref_text": ref_text}

        def write(self, profile, out_dir):
            self.stage_dirs.append(out_dir)
            jb, nb = _profile_bytes(profile["voice_id"])
            (Path(out_dir) / "profile.json").write_bytes(jb)
            (Path(out_dir) / "profile.npz").write_bytes(nb)
            return str(Path(out_dir) / "profile.npz"), str(Path(out_dir) / "profile.json")

    fake = FakeEnroller()
    monkeypatch.setattr(sparktts_voices, "_load_enroller", lambda md: fake)
    result = sparktts_voices.enroll_from_audio(b"wav", "clone:staged", ref_text="hello")
    assert result["voice_id"] == "clone:staged"
    assert fake.stage_dirs
    assert str(voices_env) not in fake.stage_dirs[0]
    assert (voices_env / "clone_staged.json").exists()
    assert (voices_env / "clone_staged.npz").exists()


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


def test_safe_id_collision_is_rejected(voices_env):
    """The legacy ':'/'/' sanitiser is lossy and must not overwrite another id."""
    jb, nb = _profile_bytes("clone:a:b")
    sparktts_voices.register_from_profile_files(jb, nb)
    jb2, nb2 = _profile_bytes("clone:a/b")
    with pytest.raises(ValueError, match="safe-id collision"):
        sparktts_voices.register_from_profile_files(jb2, nb2)
    assert json.loads((voices_env / "clone_a_b.json").read_text())["voice_id"] == "clone:a:b"


def test_half_written_pair_is_not_listed_or_replaced(voices_env):
    """A JSON-only pair is treated as an interrupted write, never as a voice."""
    (voices_env / "clone_half.json").write_text(
        json.dumps({"voice_id": "clone:half", "npz_file": "clone_half.npz"})
    )
    assert sparktts_voices.list_voices() == []
    jb, nb = _profile_bytes("clone:half")
    with pytest.raises(ValueError, match="incomplete voice profile"):
        sparktts_voices.register_from_profile_files(jb, nb)


def test_atomic_pair_replacement_rolls_back_on_second_replace_failure(voices_env, monkeypatch):
    jb, nb = _profile_bytes("clone:atomic")
    sparktts_voices.register_from_profile_files(jb, nb)
    jpath = voices_env / "clone_atomic.json"
    npath = voices_env / "clone_atomic.npz"
    old_json = jpath.read_bytes()
    old_npz = npath.read_bytes()
    real_replace = os.replace

    def fail_json_replace(src, dst):
        if os.path.abspath(dst) == os.path.abspath(jpath):
            raise OSError("simulated json rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_json_replace)
    with pytest.raises(OSError, match="simulated json rename failure"):
        sparktts_voices.register_from_profile_files(jb, nb)
    assert jpath.read_bytes() == old_json
    assert npath.read_bytes() == old_npz
    assert [v["voice_id"] for v in sparktts_voices.list_voices()] == ["clone:atomic"]


def test_restart_recovers_write_journal_after_hard_exit_between_renames(voices_env, monkeypatch):
    jb, nb = _profile_bytes("clone:restart")
    sparktts_voices.register_from_profile_files(jb, nb)
    jpath = voices_env / "clone_restart.json"
    real_replace = os.replace

    def fail_json_replace(src, dst):
        if os.path.abspath(dst) == os.path.abspath(jpath):
            raise OSError("simulated power loss before metadata rename")
        return real_replace(src, dst)

    # Suppress the in-process rollback to model a process dying before its
    # finally block.  A fresh list/read must recover the old complete pair.
    real_recover = sparktts_voices._recover_transaction
    monkeypatch.setattr(sparktts_voices, "_recover_transaction", lambda _journal: None)
    monkeypatch.setattr(os, "replace", fail_json_replace)
    with pytest.raises(OSError, match="simulated power loss"):
        sparktts_voices.register_from_profile_files(jb, nb)
    assert list(voices_env.glob(".sparktts-txn-*.json"))

    monkeypatch.setattr(os, "replace", real_replace)
    monkeypatch.setattr(sparktts_voices, "_recover_transaction", real_recover)
    listed = sparktts_voices.list_voices(model_id=sparktts_voices.SPARK_MODEL_ID)
    assert [v["voice_id"] for v in listed] == ["clone:restart"]
    assert not list(voices_env.glob(".sparktts-txn-*.json"))


def test_delete_unlink_failure_is_not_reported_as_success(voices_env, monkeypatch):
    jb, nb = _profile_bytes("clone:delete-fail")
    sparktts_voices.register_from_profile_files(jb, nb)
    jpath = voices_env / "clone_delete-fail.json"
    npath = voices_env / "clone_delete-fail.npz"
    real_unlink = os.unlink

    def fail_npz(path):
        if os.path.abspath(path) == os.path.abspath(npath):
            raise OSError("simulated unlink failure")
        return real_unlink(path)

    monkeypatch.setattr(os, "unlink", fail_npz)
    with pytest.raises(OSError, match="simulated unlink failure"):
        sparktts_voices.delete_voice("clone:delete-fail")
    # The first unlink was rolled back, so neither component is lost.
    assert jpath.exists() and npath.exists()


def test_restart_recovers_delete_journal_after_hard_exit(voices_env, monkeypatch):
    jb, nb = _profile_bytes("clone:delete-restart")
    sparktts_voices.register_from_profile_files(jb, nb)
    npath = voices_env / "clone_delete-restart.npz"
    real_unlink = os.unlink

    def fail_npz(path):
        if os.path.abspath(path) == os.path.abspath(npath):
            raise OSError("simulated power loss before second unlink")
        return real_unlink(path)

    real_recover = sparktts_voices._recover_transaction
    monkeypatch.setattr(sparktts_voices, "_recover_transaction", lambda _journal: None)
    monkeypatch.setattr(os, "unlink", fail_npz)
    with pytest.raises(OSError, match="simulated power loss"):
        sparktts_voices.delete_voice("clone:delete-restart")
    assert list(voices_env.glob(".sparktts-txn-*.json"))

    monkeypatch.setattr(os, "unlink", real_unlink)
    # Restore the recovery hook to model a fresh process rather than relying on
    # the failed delete's in-memory state.
    monkeypatch.setattr(sparktts_voices, "_recover_transaction", real_recover)
    listed = sparktts_voices._list_from_disk(model_id=sparktts_voices.SPARK_MODEL_ID)
    assert [v["voice_id"] for v in listed] == ["clone:delete-restart"]
    assert not list(voices_env.glob(".sparktts-txn-*.json"))


def test_profile_survives_reload_in_a_second_process(voices_env, monkeypatch):
    """A child process can register into the shared data root and a fresh
    interpreter can discover the complete pair after the parent reloads."""
    monkeypatch.setenv("SPARKTTS_VOICES_DIR", str(voices_env))
    code = """
import numpy as np
from server.core.sparktts_voices import register_embedding_voice
register_embedding_voice('clone:subprocess', np.arange(16, dtype=np.float32).tobytes())
"""
    env = dict(os.environ)
    env["SPARKTTS_VOICES_DIR"] = str(voices_env)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    listed = sparktts_voices._list_from_disk()
    assert any(v["voice_id"] == "clone:subprocess" for v in listed)
