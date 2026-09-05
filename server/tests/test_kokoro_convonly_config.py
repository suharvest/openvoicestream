"""Host-only profile wiring checks; no native/NPU qualification."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from server.core import profile_loader

ROOT = Path(__file__).resolve().parents[2]


def _path(platform):
    return ROOT / "configs" / "profiles" / f"{platform}-kokoro-convonly.json"


@pytest.fixture(autouse=True)
def isolate_profile_state(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(profile_loader, "_OPERATOR_KEYS", frozenset())
    monkeypatch.setattr(profile_loader, "_OPERATOR_SNAPSHOT", {})
    monkeypatch.setattr(profile_loader, "_OWNED_OVERRIDES", set())
    monkeypatch.setattr(profile_loader, "_APPLIED_KEYS", set())
    monkeypatch.setattr(profile_loader, "_CURRENT_PROFILE", {})


@pytest.mark.parametrize("platform,intra", [("rk3576", "6"), ("rk3588", "4")])
def test_profile_wiring_uses_explicit_bundle_and_no_legacy_fallback(platform, intra):
    profile = json.loads(_path(platform).read_text())
    env = profile["env"]
    assert profile["tts_backend"] == "rk.tts"
    assert profile["asr_backend"] is None
    assert profile["execution_policy"] == {"mode": "serialized", "shared_resource": "npu"}
    assert env == {
        "LANGUAGE_MODE": "rk", "RK_ENSURE_MATCHA_RESOURCES": "0",
        "RK_PLATFORM": platform, "ASR_BACKEND": "disabled",
        "TTS_BACKEND": "kokoro_convonly",
        "KOKORO_CONVONLY_ROOT": f"/opt/kokoro-convonly/{platform}",
        "KOKORO_JA_DICDIR": "/opt/resources/ja/unidic-lite-1.0.8",
        "KOKORO_CONVONLY_MANIFEST_SHA256": {
            "rk3576": "24244b7054bc3626fc22f4ee9bc013ef63aaa5cf409675cafbc10e1c53957ed9",
            "rk3588": "83733c717e0ce5b76ac1295e4827cf3ad2e111955259e9d670897e100fabeb6e",
        }[platform],
        "KOKORO_FRONTEND_INTRA_OP_THREADS": intra,
        "KOKORO_FRONTEND_INTER_OP_THREADS": "1", "RK_ARTIFACT_AUTO_DOWNLOAD": "0",
    }
    assert "Production" in profile["description"]
    assert not profile.get("profile_owned_env")


@pytest.mark.parametrize("platform", ["rk3576", "rk3588"])
def test_actual_profile_loader_applies_fixed_manifest_digest(platform):
    profile_loader.apply_profile(str(_path(platform)))
    assert os.environ["KOKORO_CONVONLY_MANIFEST_SHA256"] == {
        "rk3576": "24244b7054bc3626fc22f4ee9bc013ef63aaa5cf409675cafbc10e1c53957ed9",
        "rk3588": "83733c717e0ce5b76ac1295e4827cf3ad2e111955259e9d670897e100fabeb6e",
    }[platform]
    assert os.environ["TTS_BACKEND"] == "kokoro_convonly"
    assert os.environ["RK_PLATFORM"] == platform
    assert os.environ["OVS_TTS_MODEL_ID"] == f"kokoro-convonly-v1_0-{platform}"


def test_actual_profile_loader_preserves_operator_receipt(monkeypatch):
    # Sentinel only: this is not a real qualified manifest digest.
    key = "KOKORO_CONVONLY_MANIFEST_SHA256"
    monkeypatch.setenv(key, "operator-receipt-sentinel")
    monkeypatch.setattr(profile_loader, "_OPERATOR_KEYS", frozenset({key}))
    profile_loader.apply_profile(str(_path("rk3576")))
    assert os.environ[key] == "operator-receipt-sentinel"


def test_profile_exposes_external_japanese_dictionary_mount(monkeypatch):
    profile_loader.apply_profile(str(_path("rk3576")))
    assert os.environ["KOKORO_JA_DICDIR"] == "/opt/resources/ja/unidic-lite-1.0.8"

