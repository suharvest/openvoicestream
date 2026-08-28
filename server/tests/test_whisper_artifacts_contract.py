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
def test_the_derived_paths_are_the_ones_provisioning_creates(path):
    """Profiles pin no paths; they pin a variant. This checks the derivation.

    Hardcoding absolute paths in the profile is what made WHISPER_MODEL_DIR
    relocate the download without relocating the load, so the invariant to hold
    is not "the profile names the right file" but "the config builder derives a
    path the downloader actually writes".
    """
    profile = json.loads(path.read_text(encoding="utf-8"))
    env = profile["env"]
    spec = profile["asr_backend"]
    variant = env["WHISPER_VARIANT"]
    assert not any(v.startswith("/opt/models") for v in env.values()), (
        f"{path.name} pins an absolute path; that defeats WHISPER_MODEL_DIR"
    )

    kind = {"hailo.whisper": "hailo", "rk.whisper": "rknn",
            "jetson.whisper_trt": "tensorrt"}[spec]
    cfg = vbc.build_whisper_asr_config(
        kind, env={**env, "WHISPER_MODEL_DIR": "/models/w"}
    )
    encoder, decoders = _WHISPER_ENCODER_FILES[spec][variant]

    if spec == "jetson.whisper_trt":
        # What ships is ONNX; what loads is the plan built from it on-device —
        # and it must be the EXACT path provisioning writes, not merely
        # something ending in .plan.
        from server.core.model_downloader import _WHISPER_TRT_PLAN

        assert encoder.endswith(".onnx")
        assert cfg.encoder_path == f"/models/w/{_WHISPER_TRT_PLAN}"
    else:
        assert cfg.encoder_path == f"/models/w/{encoder}"
    # tiny and base decoders are not interchangeable (4 layers / d384 against
    # 6 / d512) and crossing them yields fluent nonsense rather than an error.
    family = "tiny" if "tiny" in variant else "base"
    assert cfg.decoder_dir == f"/models/w/decoder/{family}"
    assert cfg.vocab_dir == "/models/w"
    assert all(d.startswith(f"decoder/{family}/") for d in decoders)


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
    """Every profile's (variant, window) pair must be declared by some leaf."""
    # Keyed on (backend, variant): "base" names a different artifact on Hailo
    # than on Jetson, so the variant alone collides.
    declared = {
        (leaf["backend"], leaf["runtime_env"]["WHISPER_VARIANT"]): leaf["runtime_env"]
        for leaf in _leaves().values() if "runtime_env" in leaf
    }
    for path in PROFILES:
        profile = json.loads(path.read_text(encoding="utf-8"))
        env = profile["env"]
        leaf_env = declared.get((profile["asr_backend"], env["WHISPER_VARIANT"]))
        assert leaf_env is not None, (
            f"{path.name}: no leaf declares "
            f"{profile['asr_backend']}/{env['WHISPER_VARIANT']}"
        )
        assert leaf_env["WHISPER_WINDOW_S"] == env["WHISPER_WINDOW_S"], (
            f"{path.name}: window disagrees with the leaf"
        )


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
    env = {"WHISPER_VARIANT": "base10", "WHISPER_WINDOW_S": "twenty"}
    with pytest.raises(ValueError, match="WHISPER_WINDOW_S"):
        vbc.build_whisper_asr_config("rknn", env=env)
    env = {"WHISPER_VARIANT": "tiny", "WHISPER_PADDING_CUTOFF_S": "one"}
    with pytest.raises(ValueError, match="WHISPER_PADDING_CUTOFF_S"):
        vbc.build_whisper_asr_config("hailo", env=env)


def test_a_non_critical_number_still_falls_back():
    # Thread count does not select a graph dimension; a typo there should not
    # take the service down.
    cfg = vbc.build_whisper_asr_config("rknn", env={
        "WHISPER_VARIANT": "base10", "WHISPER_DECODER_THREADS": "many",
    })
    assert cfg.decoder_threads == 0


@pytest.mark.parametrize("variant,family", [("base10", "base"), ("tiny", "tiny")])
def test_default_paths_follow_the_root_the_downloader_writes_to(variant, family):
    """One root for download AND load.

    Deriving these from MODEL_DIR meant an operator who moved
    WHISPER_MODEL_DIR downloaded to one place and loaded from another, and the
    old `decoder_onnx` default named a directory the downloader never creates.
    """
    kind = "rknn" if variant == "base10" else "hailo"
    cfg = vbc.build_whisper_asr_config(kind, env={
        "WHISPER_MODEL_DIR": "/data/w", "WHISPER_VARIANT": variant,
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


# ── 复审（codex 第二轮）报出的缺陷 ──────────────────────────────────────

def test_the_trt_build_pins_the_dynamic_input(monkeypatch, tmp_path):
    """The encoder ONNX declares `batch_size` dynamic.

    TensorRT refuses to build a network with a dynamic input unless an
    optimization profile pins it — without one the build produces no engine and
    the Orin profile fails at model load on a file that was never written.

    This EXECUTES the builder against a fake TensorRT rather than grepping the
    source. The previous version only searched for method names, so it passed
    while the code raised AttributeError on `IOptimizationProfile.num_inputs` —
    an attribute the real API does not have (verified against TRT 10.3 on
    device: set_shape / get_shape / set_shape_input / get_shape_input /
    extra_memory_target, and nothing else).
    """
    import sys
    import types

    calls = {"shapes": [], "profiles": 0}

    class _Profile:                     # only what the real API exposes
        def set_shape(self, name, mn, opt, mx):
            calls["shapes"].append((name, mn, opt, mx))

    class _Tensor:
        name, shape = "input_features", (-1, 80, 3000)

    class _Network:
        num_inputs = 1
        def get_input(self, i): return _Tensor()

    class _Config:
        def add_optimization_profile(self, p): calls["profiles"] += 1
        def set_flag(self, f): pass
        def set_memory_pool_limit(self, *a): pass

    class _Builder:
        def __init__(self, logger): pass
        def create_network(self, flags): return _Network()
        def create_optimization_profile(self): return _Profile()
        def create_builder_config(self): return _Config()
        def build_serialized_network(self, n, c): return b"ENGINE"

    class _Parser:
        num_errors = 0
        def __init__(self, n, l): pass
        def parse(self, data): return True

    fake = types.ModuleType("tensorrt")
    fake.__version__ = "10.3.0"
    fake.Logger = lambda *a: None
    fake.Logger.WARNING = fake.Logger.ERROR = 0
    fake.Builder, fake.OnnxParser = _Builder, _Parser
    fake.NetworkDefinitionCreationFlag = types.SimpleNamespace(EXPLICIT_BATCH=0)
    fake.BuilderFlag = types.SimpleNamespace(BF16=1)
    fake.MemoryPoolType = types.SimpleNamespace(WORKSPACE=0)
    monkeypatch.setitem(sys.modules, "tensorrt", fake)

    from server.core.model_downloader import _build_whisper_trt_engine

    onnx = tmp_path / "enc.onnx"; onnx.write_bytes(b"x")
    plan = tmp_path / "enc.plan"
    _build_whisper_trt_engine(str(onnx), str(plan))

    assert calls["shapes"] == [("input_features", (1, 80, 3000)) * 1
                               + ((1, 80, 3000), (1, 80, 3000))]
    assert calls["profiles"] == 1, "the profile was never added to the config"
    assert plan.read_bytes() == b"ENGINE"
    sidecar = json.loads((tmp_path / "enc.plan.buildinfo.json").read_text())
    assert sidecar["plan_bytes"] == len(b"ENGINE")


def test_a_hand_supplied_plan_is_usable_without_a_sidecar():
    """The documented escape hatch for a TensorRT that cannot build BF16.

    Treating a sidecar-less plan as stale would rebuild it, and on such a
    TensorRT the rebuild raises — making the escape hatch unusable.
    """
    import inspect

    from server.core.model_downloader import _build_whisper_trt_engine

    src = inspect.getsource(_build_whisper_trt_engine)
    assert "not os.path.exists(info_path)" in src
    # and the sidecar records the artifact's size, so a truncated plan with an
    # intact sidecar is rebuilt rather than trusted until deserialization fails
    assert "plan_bytes" in src


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "NaN"])
def test_non_finite_shape_values_are_rejected(value):
    """`float()` accepts nan and inf, and for nan BOTH `x <= 0` and `x > 0` are
    False — so no downstream range check catches them either."""
    with pytest.raises(ValueError):
        vbc.build_whisper_asr_config("rknn", env={
            "WHISPER_MODEL_DIR": "/w", "WHISPER_VARIANT": "base10",
            "WHISPER_WINDOW_S": value,
        })


def test_a_negative_cutoff_is_rejected():
    """A negative cutoff makes the usable window longer than the compiled graph,
    and the front end then truncates the excess silently."""
    from voxedge.backends.whisper import WhisperASRConfig

    with pytest.raises(ValueError, match="padding_cutoff_s"):
        WhisperASRConfig(encoder_kind="hailo", encoder_path="x", decoder_dir="y",
                         vocab_dir="z", window_s=10.0, padding_cutoff_s=-1.0)


def test_the_two_artifact_tables_cover_the_same_pairs():
    """`_WHISPER_ENCODER_FILES` and `_WHISPER_GEOMETRY` are keyed the same way
    and must stay in step.

    Nothing structural links them: adding a variant to one and forgetting the
    other surfaces as a bare KeyError deep inside config construction, at
    service start, on whichever board declares that variant.
    """
    from server.core.model_downloader import _WHISPER_ENCODER_FILES, _WHISPER_GEOMETRY

    files = {(spec, v) for spec, d in _WHISPER_ENCODER_FILES.items() for v in d}
    assert files == set(_WHISPER_GEOMETRY), (
        f"only in files: {sorted(files - set(_WHISPER_GEOMETRY))}; "
        f"only in geometry: {sorted(set(_WHISPER_GEOMETRY) - files)}"
    )


def test_every_declared_geometry_matches_its_filename():
    """The window in the table must be the window the artifact was compiled at.

    The filenames carry it, so this is checkable rather than a matter of trust —
    and getting it wrong is exactly the mismatch rknn-lite does not validate.
    """
    from server.core.model_downloader import _WHISPER_ENCODER_FILES, _WHISPER_GEOMETRY

    for (spec, variant), (window, _cutoff) in _WHISPER_GEOMETRY.items():
        encoder = _WHISPER_ENCODER_FILES[spec][variant][0]
        assert f"{int(window)}s" in Path(encoder).name, (
            f"{spec}/{variant}: geometry says {window}s but the artifact is "
            f"{Path(encoder).name}"
        )


def test_only_hailo_declares_a_boundary_cutoff():
    """The cutoff is Hailo's boundary-hallucination guard; a non-zero value
    elsewhere would shorten the usable window for no reason."""
    from server.core.model_downloader import _WHISPER_GEOMETRY

    for (spec, _variant), (_window, cutoff) in _WHISPER_GEOMETRY.items():
        assert (cutoff > 0) == (spec == "hailo.whisper"), f"{spec}: cutoff {cutoff}"
