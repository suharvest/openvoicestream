import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


# Stale-test update: the speaker registry was refactored to be model-scoped.
# `speaker_kwargs_for_id` / `speaker_spec_for_id` / `available_speakers` now all
# require a `model_id` argument, and the env override table (`OVS_TTS_SPEAKERS_JSON`)
# is parsed once at module import into `_ENV_OVERRIDES`. These tests inject the env
# JSON by rebuilding that table and invalidating the per-model cache, then assert
# against the current output shapes (preset → "speaker"/"payload"; embedding →
# truncated "speaker_embedding_b64").
_MODEL = "qwen3-tts"


def _install_env_speakers(monkeypatch, json_str):
    from server.core import tts_speakers
    monkeypatch.setenv("OVS_TTS_SPEAKERS_JSON", json_str)
    # Rebuild the module-level env override table (normally frozen at import)
    # and drop any cached per-model map so the new overrides take effect.
    monkeypatch.setattr(tts_speakers, "_ENV_OVERRIDES", tts_speakers._env_overrides())
    tts_speakers._invalidate_cache()


def test_tts_speaker_registry_supports_preset_and_embedding(monkeypatch):
    from server.core import tts_speakers

    emb = b"\x00\x00\x80?" * 1024
    _install_env_speakers(
        monkeypatch,
        '{"2301":{"type":"preset","speaker":"2301"},'
        '"10001":{"type":"embedding","speaker_embedding_b64":"%s"}}'
        % base64.b64encode(emb).decode("ascii"),
    )

    assert tts_speakers.speaker_kwargs_for_id(2301, _MODEL) == {
        "speaker_id": 2301,
        "speaker": "2301",
    }
    assert tts_speakers.speaker_kwargs_for_id(10001, _MODEL) == {
        "speaker_id": 10001,
        "speaker_embedding": emb,
    }

    speakers = tts_speakers.available_speakers(_MODEL)
    by_id = {s["id"]: s for s in speakers}
    assert by_id[2301]["type"] == "preset"
    assert by_id[2301]["payload"] == "2301"
    assert by_id[10001]["type"] == "embedding"
    # Embedding payloads are truncated for display; just confirm the field exists.
    assert "speaker_embedding_b64" in by_id[10001]


def test_tts_speaker_registry_rejects_unknown_when_configured(monkeypatch):
    from server.core import tts_speakers

    monkeypatch.setenv("OVS_TTS_ALLOW_UNMAPPED_SPEAKER_ID", "0")

    try:
        tts_speakers.speaker_kwargs_for_id(9999, _MODEL)
    except ValueError as exc:
        assert "Unknown TTS speaker_id" in str(exc)
    else:
        raise AssertionError("expected unknown speaker_id to be rejected")


def test_sparktts_default_speaker_maps_to_intrinsic_style():
    from server.core import tts_speakers

    assert tts_speakers.speaker_kwargs_for_id(0, "sparktts-0p5b") == {
        "speaker_id": 0,
        "speaker": "female_moderate_moderate",
    }


def test_speaker_mutation_fails_closed_on_corrupt_registry(monkeypatch, tmp_path):
    from server.core import tts_speakers

    path = tmp_path / "speakers.json"
    original = b"{not-json"
    path.write_bytes(original)
    monkeypatch.setenv("OVS_TTS_SPEAKERS_FILE", str(path))
    with pytest.raises(RuntimeError, match="cannot safely update speaker registry"):
        tts_speakers.register_speaker_from_embedding("qwen3-tts", b"\x00\x00\x80?" * 4)
    assert path.read_bytes() == original

    with pytest.raises(RuntimeError, match="cannot safely update speaker registry"):
        tts_speakers.unregister_speaker("qwen3-tts", 10000)
    assert path.read_bytes() == original


def test_speaker_registration_is_cross_process_read_modify_write_safe(tmp_path):
    path = tmp_path / "speakers.json"
    code = """
from server.core.tts_speakers import register_speaker_from_embedding
register_speaker_from_embedding('qwen3-tts', b'\\x00\\x00\\x80?' * 4)
"""
    env = dict(os.environ)
    env["OVS_TTS_SPEAKERS_FILE"] = str(path)
    root = Path(__file__).resolve().parents[2]
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = [proc.communicate(timeout=30) for proc in procs]
    assert all(proc.returncode == 0 for proc in procs), outputs
    data = json.loads(path.read_text())
    assert sorted(data["qwen3-tts"]) == ["10000", "10001"]
