"""Tests for server.core.composition_boot (TRACK 1 SLICE 2 — gated seam).

Covers:
  * flat profile (no `composition`) → strict no-op: os.environ untouched, no
    extra pull files, validator never invoked;
  * synthetic composition profile (orin-nx, asr n2 + tts n2 + override) →
    validator passes, resolve_env sets EDGE_LLM_TTS_TALKER_* (b2 paths),
    union-pull returns the de-duped ASR+TTS+shared file list, and a
    pre-existing os.environ key is NOT overwritten (env-wins precedence);
  * bad composition (tts n2 on nano = over-budget, and an unknown leaf id) →
    CompositionError, boot aborts.
"""

from __future__ import annotations

import os

import pytest

from server.core import composition_boot as cb
from server.core import leaf_composition as lc


ASR_N2 = "asr.qwen3_asr.orin-nx.n2"
TTS_N2 = "tts.qwen3_tts.orin-nx.n2"
TTS_SHARED = "tts.qwen3_tts.shared"


@pytest.fixture(autouse=True)
def _clean_env():
    """Snapshot os.environ and restore it after each test (no leakage)."""
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


# ---------------------------------------------------------------------------
# Flat profile → strict no-op
# ---------------------------------------------------------------------------

def test_flat_profile_is_noop(monkeypatch):
    # Guard: if load_registry is touched at all on the flat path, fail loudly.
    def _boom(*a, **k):  # pragma: no cover - asserts it's never called
        raise AssertionError("load_registry must NOT run on the flat path")

    monkeypatch.setattr(lc, "load_registry", _boom)

    before = dict(os.environ)
    for profile in ({}, None, {"asr_backend": "jetson.trt_edge_llm"}):
        assert cb.apply_composition(profile) == []
    # Byte-identical: nothing added, removed, or changed.
    assert dict(os.environ) == before
    assert cb.PULL_FILES_ENV not in os.environ


def test_falsy_composition_is_noop():
    # An explicitly empty/falsy composition block is also skipped.
    for falsy in (None, {}, "", 0, []):
        assert cb.apply_composition({"composition": falsy}) == []
    assert cb.PULL_FILES_ENV not in os.environ


# ---------------------------------------------------------------------------
# Active composition (orin-nx, asr n2 + tts n2)
# ---------------------------------------------------------------------------

def _nx_n2_profile(overrides=None):
    return {
        "composition": {
            "device": "orin-nx",
            "asr": ASR_N2,
            "tts": TTS_N2,
            "overrides": overrides or {},
        }
    }


def test_active_sets_talker_b2_env():
    os.environ["QWEN3_ARTIFACT_ROOT"] = "/opt/models/qwen3-edgellm"
    pulls = cb.apply_composition(_nx_n2_profile())

    # b2 talker paths emitted (n2 = batch-lane b2).
    assert os.environ["EDGE_LLM_TTS_TALKER_DIR"] == (
        "/opt/models/qwen3-edgellm/tts/talker_b2"
    )
    assert os.environ["EDGE_LLM_TTS_TALKER_ENGINE"] == (
        "/opt/models/qwen3-edgellm/engines/orin-nx/highperf/"
        "talker_fp16_b2/talker_decode.engine"
    )
    # shared sub-leaf env also applied.
    assert os.environ["EDGE_LLM_TTS_TALKER_BACKEND"] == "qwen3_tts_explicit_kv"
    # ASR engine env applied (v0.8.0 engine set: highperf-v080, per
    # configs/leaves/qwen3-asr-nx.yaml — the old highperf-v2 path is retired).
    assert os.environ["EDGE_LLM_ASR_ENGINE_DIR"] == (
        "/opt/models/qwen3-edgellm/engines/orin-nx/highperf-v080/"
        "asr_thinker_full_fp8embed"
    )
    assert pulls  # non-empty


def test_active_union_pull_dedups_asr_tts_shared():
    pulls = cb.apply_composition(_nx_n2_profile())
    registry = lc.load_registry()
    expected = lc.resolve_pull([ASR_N2, TTS_N2], registry)

    assert pulls == expected
    assert len(pulls) == len(set(pulls))  # de-duped

    # ASR files present.
    asr_files = lc.load_registry().get_leaf(ASR_N2).artifacts.files
    for f in asr_files:
        assert f in pulls
    # TTS n2 + shared sub-leaf files present (shared appears once).
    tts_files = registry.get_leaf(TTS_N2).artifacts.files
    shared_files = registry.get_leaf(TTS_SHARED).artifacts.files
    for f in (*tts_files, *shared_files):
        assert pulls.count(f) == 1

    # Published additively to os.environ for observability.
    assert cb.PULL_FILES_ENV in os.environ
    assert os.environ[cb.PULL_FILES_ENV].split(os.pathsep) == pulls


def test_env_wins_over_leaf_precedence():
    # Pre-set a key the TTS leaf would otherwise emit → it must NOT be
    # overwritten (operator/compose/.env/shell owns it).
    os.environ["EDGE_LLM_TTS_TALKER_DIR"] = "/operator/owned/talker"
    cb.apply_composition(_nx_n2_profile())
    assert os.environ["EDGE_LLM_TTS_TALKER_DIR"] == "/operator/owned/talker"
    # A key NOT pre-set is still filled from the leaf.
    assert "EDGE_LLM_TTS_TALKER_ENGINE" in os.environ


def test_override_applied_when_not_in_env():
    pulls = cb.apply_composition(
        _nx_n2_profile(overrides={"EDGE_LLM_TTS_TALKER_BACKEND": "custom_ovr"})
    )
    assert os.environ["EDGE_LLM_TTS_TALKER_BACKEND"] == "custom_ovr"
    assert pulls


# ---------------------------------------------------------------------------
# Bad composition → CompositionError (boot aborts)
# ---------------------------------------------------------------------------

def test_over_budget_on_nano_raises():
    profile = {
        "composition": {
            "device": "orin-nano",
            "asr": ASR_N2,
            "tts": TTS_N2,  # FP16 b2 9057MB → over nano headroom
        }
    }
    before = dict(os.environ)
    with pytest.raises(lc.CompositionError) as exc:
        cb.apply_composition(profile)
    assert "memory budget" in str(exc.value)
    assert "orin-nano" in str(exc.value)
    # Validation runs before any env mutation → os.environ untouched on abort.
    assert dict(os.environ) == before


def test_unknown_leaf_id_raises():
    profile = {
        "composition": {
            "device": "orin-nx",
            "asr": "asr.does.not.exist",
            "tts": TTS_N2,
        }
    }
    with pytest.raises(lc.CompositionError) as exc:
        cb.apply_composition(profile)
    assert "not built" in str(exc.value)


def test_missing_device_raises():
    with pytest.raises(lc.CompositionError) as exc:
        cb.apply_composition({"composition": {"asr": ASR_N2}})
    assert "device" in str(exc.value)


def test_no_leaves_selected_raises():
    with pytest.raises(lc.CompositionError) as exc:
        cb.apply_composition({"composition": {"device": "orin-nx"}})
    assert "no leaves" in str(exc.value)


# ---------------------------------------------------------------------------
# v0.9.0 composition (orin-nx, v090 asr n2 + v090 tts n2). The v080 cases
# above stay green — the v090 leaves are NEW ids in configs/leaves/*-v090.yaml.
# ---------------------------------------------------------------------------

ASR_V090_N2 = "asr.qwen3_asr_v090.orin-nx.n2"
TTS_V090_N2 = "tts.qwen3_tts_v090.orin-nx.n2"


def test_v090_active_sets_env_and_omits_mel_keys():
    os.environ["QWEN3_ARTIFACT_ROOT"] = "/opt/models/qwen3-edgellm"
    pulls = cb.apply_composition({
        "composition": {
            "device": "orin-nx",
            "asr": ASR_V090_N2,
            "tts": TTS_V090_N2,
        }
    })

    # v0.9.0 engine set: highperf-v090, int4 thinker. N=2 (and now N=1) use the
    # batch-2 thinker (asr-b2) — a batch-2 engine serves batch-1 too, so the
    # redundant batch-1 engine is dropped.
    assert os.environ["EDGE_LLM_ASR_ENGINE_DIR"] == (
        "/opt/models/qwen3-edgellm/engines/orin-nx/highperf-v090/"
        "asr_thinker_full_int4_b2"
    )
    # v0.9.0 requires an ABSOLUTE plugin path (cwd-resolution fix).
    assert os.environ["EDGELLM_PLUGIN_PATH"] == (
        "/opt/edgellm-v090/libNvInfer_edgellm_plugin.so"
    )
    # Lean non-stateful code2wav on the v090 TTS path.
    assert os.environ["EDGE_LLM_TTS_STATEFUL_CODE2WAV"] == "0"
    # v0.9.0 WAV-ingest: the leaf emits EDGELLM_REQUEST_AUDIO_WAV=1 (read by the
    # voxedge ASR preload guard to skip the mel-asset check); the host-side mel
    # front-end keys are retired on the v090 path and must not be emitted.
    assert os.environ["EDGELLM_REQUEST_AUDIO_WAV"] == "1"
    assert "EDGE_LLM_ASR_MEL_SETTINGS" not in os.environ
    assert "EDGE_LLM_ASR_MEL_FILTERS" not in os.environ
    assert pulls  # non-empty union pull
    assert any("highperf-v090" in f for f in pulls)
    assert not any("highperf-v080" in f for f in pulls)
