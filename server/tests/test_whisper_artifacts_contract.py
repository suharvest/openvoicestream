"""The Whisper deployment chain: leaf, profiles, provisioning.

The backend being registered in code is not the same as being deployable. This
covers the three places that have to agree — the leaf's env, the profile's env,
and the file list `_ensure_whisper_artifacts` will actually fetch — because a
mismatch between them fails at model load on a device, not here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from server.core.asr_backend import _ASR_REGISTRY
from server.core.model_downloader import (
    _WHISPER_ENCODER_FILES,
    _WHISPER_SHARED,
    _ensure_whisper_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
LEAF = ROOT / "configs/leaves/whisper-asr.yaml"
PROFILES = sorted(ROOT.glob("configs/profiles/*whisper*.json"))
SPECS = ("hailo.whisper", "rk.whisper", "jetson.whisper_trt")


def _leaves() -> dict:
    return yaml.safe_load(LEAF.read_text(encoding="utf-8"))["leaves"]


def test_every_whisper_spec_is_registered_and_provisioned():
    for spec in SPECS:
        assert spec in _ASR_REGISTRY, f"{spec} missing from the ASR registry"
        assert spec in _WHISPER_ENCODER_FILES, (
            f"{spec} has no artifact list; a profile selecting it would start "
            f"and then fail at model load with a missing file"
        )


def test_profiles_exist_for_every_spec():
    assert PROFILES, "no whisper profiles found"
    covered = {json.loads(p.read_text())["asr_backend"] for p in PROFILES}
    assert covered == set(SPECS)


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_profile_env_is_self_consistent(path):
    profile = json.loads(path.read_text(encoding="utf-8"))
    env = profile["env"]
    spec = profile["asr_backend"]
    variant = env["WHISPER_VARIANT"]

    encoder, decoders = _WHISPER_ENCODER_FILES[spec][variant]
    # The encoder path must name the file the downloader fetches — except on
    # Jetson, where the .plan is BUILT on-device from the shipped .onnx and the
    # two therefore differ by design.
    if spec == "jetson.whisper_trt":
        assert env["WHISPER_ENCODER_PATH"].endswith(".plan")
        assert encoder.endswith(".onnx")
    else:
        assert env["WHISPER_ENCODER_PATH"].endswith(Path(encoder).name), (
            f"{path.name}: WHISPER_ENCODER_PATH does not match the file "
            f"WHISPER_VARIANT={variant!r} provisions"
        )
    # tiny and base decoders are not interchangeable (4 layers / d384 against
    # 6 / d512) and crossing them yields fluent nonsense rather than an error.
    family = "tiny" if "tiny" in variant else "base"
    assert env["WHISPER_DECODER_DIR"].endswith(f"/decoder/{family}")
    assert all(f"/{family}/" in d for d in decoders)


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_window_matches_the_compiled_encoder(path):
    """WHISPER_WINDOW_S is not a knob — rknn-lite silently reinterprets a
    mismatched buffer and the transcript comes back as plausible nonsense."""
    env = json.loads(path.read_text(encoding="utf-8"))["env"]
    spec = json.loads(path.read_text(encoding="utf-8"))["asr_backend"]
    encoder, _ = _WHISPER_ENCODER_FILES[spec][env["WHISPER_VARIANT"]]
    window = env["WHISPER_WINDOW_S"]
    assert f"{window}s" in Path(encoder).name, (
        f"{path.name}: window {window}s is not the window {Path(encoder).name} "
        f"was compiled at"
    )


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_only_the_hailo_profiles_cap_tokens(path):
    """The cap exists because the Hailo pairing never emits EOS. Setting it
    elsewhere would truncate correct output."""
    profile = json.loads(path.read_text(encoding="utf-8"))
    capped = "WHISPER_MAX_NEW_TOKENS" in profile["env"]
    assert capped == (profile["asr_backend"] == "hailo.whisper")


def test_leaf_env_matches_the_profile_env():
    leaves = _leaves()
    by_encoder = {
        leaf["runtime_env"]["WHISPER_ENCODER_PATH"]: leaf["runtime_env"]
        for leaf in leaves.values() if "runtime_env" in leaf
    }
    for path in PROFILES:
        env = json.loads(path.read_text(encoding="utf-8"))["env"]
        leaf_env = by_encoder.get(env["WHISPER_ENCODER_PATH"])
        assert leaf_env is not None, f"{path.name}: no leaf declares this encoder"
        for key in ("WHISPER_WINDOW_S", "WHISPER_DECODER_DIR", "WHISPER_VARIANT"):
            assert leaf_env[key] == env[key], f"{path.name}: {key} disagrees with the leaf"


def test_leaf_artifact_lists_match_the_downloader():
    for leaf in _leaves().values():
        files = set(leaf.get("artifacts", {}).get("files", ()))
        if leaf["backend"] == "whisper":       # the shared sub-leaf
            assert files == set(_WHISPER_SHARED)
            continue
        variant = leaf["runtime_env"]["WHISPER_VARIANT"]
        encoder, decoders = _WHISPER_ENCODER_FILES[leaf["backend"]][variant]
        assert files == {encoder, *decoders}


def test_an_unknown_variant_is_refused_rather_than_downloading_the_default(monkeypatch, tmp_path):
    """A typo must fail loudly. Silently falling back to the default variant
    would provision a 10 s encoder for a profile asking for 20 s, which
    rknn-lite does not validate."""
    monkeypatch.setenv("WHISPER_VARIANT", "base15")
    monkeypatch.setenv("WHISPER_MODEL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="WHISPER_VARIANT"):
        _ensure_whisper_artifacts("rk.whisper")


def test_a_non_whisper_spec_is_a_no_op():
    _ensure_whisper_artifacts("cpu.sherpa_asr")   # must not raise
