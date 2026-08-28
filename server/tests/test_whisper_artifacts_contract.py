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

from server.core import voxedge_backend_config as vbc
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


def test_whisper_env_is_operator_overridable():
    """Every WHISPER_* key a profile sets must be operator-overridable.

    `_OPERATOR_KEY_PREFIXES` is a hand-kept list and a missing prefix fails
    silently: the profile overwrites the operator's value and logs nothing.
    Measured on RK3588 — `-e WHISPER_LANGUAGE=zh` on the container read back as
    the profile's `en`, and the only symptom was Mandarin decoded into English.
    """
    from server.core.profile_loader import _OPERATOR_KEY_PREFIXES

    keys = set()
    for path in PROFILES:
        keys |= set(json.loads(path.read_text(encoding="utf-8"))["env"])
    whisper_keys = {k for k in keys if k.startswith("WHISPER_")}
    assert whisper_keys, "no WHISPER_* keys found in the profiles"
    for key in sorted(whisper_keys):
        assert key.startswith(_OPERATOR_KEY_PREFIXES), (
            f"{key} matches no operator prefix, so a profile silently wins over "
            f"an operator setting it"
        )


# ── 独立审查（codex）报出的部署侧缺陷，逐条钉死 ────────────────────────────

def test_a_bad_shape_critical_number_raises_instead_of_defaulting():
    """窗口和 boundary guard 选的是编译期形状，静默退回默认比崩溃更糟。

    rknn-lite 不校验窗口：喂错形状它不报错，重解释缓冲区后返回像话的胡话，
    而日志里没有任何东西指向那个拼写错误。
    """
    env = {"WHISPER_ENCODER_PATH": "/m/e.rknn", "WHISPER_WINDOW_S": "twenty"}
    with pytest.raises(ValueError, match="WHISPER_WINDOW_S"):
        vbc.build_whisper_asr_config("rknn", env=env)
    env = {"WHISPER_ENCODER_PATH": "/m/e.hef", "WHISPER_PADDING_CUTOFF_S": "one"}
    with pytest.raises(ValueError, match="WHISPER_PADDING_CUTOFF_S"):
        vbc.build_whisper_asr_config("hailo", env=env)


def test_a_non_critical_number_still_falls_back():
    # Thread count does not select a graph dimension; a typo there should not
    # take the service down.
    cfg = vbc.build_whisper_asr_config("rknn", env={
        "WHISPER_ENCODER_PATH": "/m/e.rknn", "WHISPER_DECODER_THREADS": "many",
    })
    assert cfg.decoder_threads == 0


@pytest.mark.parametrize("variant,family", [("base10", "base"), ("tiny", "tiny")])
def test_default_paths_follow_the_root_the_downloader_writes_to(variant, family):
    """One root for download AND load.

    Deriving these from MODEL_DIR meant an operator who moved
    WHISPER_MODEL_DIR downloaded to one place and loaded from another, and the
    old `decoder_onnx` default named a directory the downloader never creates.
    """
    cfg = vbc.build_whisper_asr_config("rknn", env={
        "WHISPER_ENCODER_PATH": "/m/e.rknn",
        "WHISPER_MODEL_DIR": "/data/w",
        "WHISPER_VARIANT": variant,
    })
    assert cfg.vocab_dir == "/data/w"
    assert cfg.decoder_dir == f"/data/w/decoder/{family}"
    # and that is a path the downloader actually populates
    spec = "rk.whisper" if variant == "base10" else "hailo.whisper"
    _, decoders = _WHISPER_ENCODER_FILES[spec][variant]
    assert all(d.startswith(f"decoder/{family}/") for d in decoders)


def test_the_jetson_plan_is_actually_built():
    """The Orin profile points at a .plan that only the build step creates.

    Provisioning fetches ONNX; without the build the profile starts and then
    fails at model load on a file nothing ever writes.
    """
    import inspect

    from server.core.model_downloader import (
        _build_whisper_trt_engine,
        _ensure_whisper_artifacts,
    )

    assert "_build_whisper_trt_engine" in inspect.getsource(_ensure_whisper_artifacts)
    src = inspect.getsource(_build_whisper_trt_engine)
    # BF16 is not a preference here: the fp16 build of this graph fails silently.
    assert "BuilderFlag.BF16" in src
    assert "BuilderFlag.FP16" not in src
