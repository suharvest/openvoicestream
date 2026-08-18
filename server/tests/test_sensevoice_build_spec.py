"""Changing a SenseVoice engine build knob must force a rebuild.

The engine is cached on disk and building it takes minutes on a Jetson, so the
provisioning path skips the build when a ``.plan`` is already there. Once the
build is parameterised that shortcut becomes a trap: flip ``SENSEVOICE_TRT_FP16``
or ``SENSEVOICE_TRT_ARGMAX`` and you would keep serving an engine built with the
old settings, with nothing in the logs to say so. These tests pin the cache key
so a "just change a parameter" pipeline stays trustworthy.

No TensorRT here — only the spec/staleness logic is exercised.
"""
from __future__ import annotations

import json

import pytest

from server.core import model_downloader as md


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "SENSEVOICE_TRT_FP16",
        "SENSEVOICE_TRT_ARGMAX",
        "SENSEVOICE_TRT_WORKSPACE_GIB",
        "SENSEVOICE_TRT_OPT_LEVEL",
    ):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def engine(tmp_path):
    onnx = tmp_path / "sense-voice-encoder.scaled.fixed.onnx"
    onnx.write_bytes(b"x" * 1024)
    plan = tmp_path / "sensevoice.plan"
    plan.write_bytes(b"p" * 64)
    return onnx, plan


def _write_sidecar(plan, spec, trt="10.3.0"):
    with open(str(plan) + ".buildinfo.json", "w", encoding="utf-8") as fh:
        json.dump({"trt": trt, "spec": spec}, fh)


def _trt_version():
    return md._trt_version_or_none() or "10.3.0"


# ── Defaults ─────────────────────────────────────────────────────────

def test_default_spec(engine):
    onnx, _ = engine
    spec = md._sensevoice_build_spec(str(onnx))
    assert spec["fp16"] is True
    assert spec["argmax"] is False
    assert spec["workspace_gib"] == 3
    assert spec["onnx_size"] == 1024


def test_missing_onnx_does_not_raise(tmp_path):
    spec = md._sensevoice_build_spec(str(tmp_path / "nope.onnx"))
    assert spec["onnx_size"] == -1


# ── Cache key: a changed knob must invalidate ────────────────────────

@pytest.mark.parametrize("env,value", [
    ("SENSEVOICE_TRT_FP16", "0"),
    ("SENSEVOICE_TRT_ARGMAX", "1"),
    ("SENSEVOICE_TRT_WORKSPACE_GIB", "2"),
    ("SENSEVOICE_TRT_OPT_LEVEL", "5"),
])
def test_changed_knob_marks_engine_stale(engine, monkeypatch, env, value):
    onnx, plan = engine
    _write_sidecar(plan, md._sensevoice_build_spec(str(onnx)), trt=_trt_version())

    monkeypatch.setenv(env, value)
    stale = md._sensevoice_engine_staleness(
        str(plan), md._sensevoice_build_spec(str(onnx))
    )
    assert stale is not None, f"{env}={value} must force a rebuild"
    assert "build spec changed" in stale


def test_unchanged_spec_keeps_engine(engine):
    onnx, plan = engine
    spec = md._sensevoice_build_spec(str(onnx))
    _write_sidecar(plan, spec, trt=_trt_version())
    assert md._sensevoice_engine_staleness(str(plan), spec) is None


def test_redownloaded_onnx_marks_engine_stale(engine):
    onnx, plan = engine
    _write_sidecar(plan, md._sensevoice_build_spec(str(onnx)), trt=_trt_version())

    onnx.write_bytes(b"y" * 2048)  # different artifact, same name
    stale = md._sensevoice_engine_staleness(
        str(plan), md._sensevoice_build_spec(str(onnx))
    )
    assert stale is not None and "onnx_size" in stale


def test_trt_upgrade_marks_engine_stale(engine, monkeypatch):
    """A .plan is version-specific — a TRT upgrade must not reuse it."""
    onnx, plan = engine
    _write_sidecar(plan, md._sensevoice_build_spec(str(onnx)), trt="9.9.9-old")
    monkeypatch.setattr(md, "_trt_version_or_none", lambda: "10.3.0")
    stale = md._sensevoice_engine_staleness(
        str(plan), md._sensevoice_build_spec(str(onnx))
    )
    assert stale is not None and "TensorRT changed" in stale


def test_trt_unavailable_skips_version_check(engine, monkeypatch):
    """Without TensorRT the rebuild could not run either — leave the engine."""
    onnx, plan = engine
    _write_sidecar(plan, md._sensevoice_build_spec(str(onnx)), trt="9.9.9-old")
    monkeypatch.setattr(md, "_trt_version_or_none", lambda: None)
    assert md._sensevoice_engine_staleness(
        str(plan), md._sensevoice_build_spec(str(onnx))
    ) is None


# ── Missing engine / legacy engines ──────────────────────────────────

def test_absent_engine_is_stale(tmp_path):
    spec = md._sensevoice_build_spec(str(tmp_path / "m.onnx"))
    assert md._sensevoice_engine_staleness(str(tmp_path / "none.plan"), spec) == (
        "no engine on disk"
    )


def test_empty_engine_is_stale(engine):
    onnx, plan = engine
    plan.write_bytes(b"")
    spec = md._sensevoice_build_spec(str(onnx))
    assert md._sensevoice_engine_staleness(str(plan), spec) == "no engine on disk"


def test_legacy_engine_without_sidecar_is_kept_when_spec_is_default(engine):
    """Upgrading the code must not trigger a surprise multi-minute rebuild."""
    onnx, plan = engine  # no sidecar written
    spec = md._sensevoice_build_spec(str(onnx))
    assert md._sensevoice_engine_staleness(str(plan), spec) is None


def test_legacy_engine_without_sidecar_rebuilds_when_spec_is_non_default(
    engine, monkeypatch
):
    """...but a legacy engine cannot be assumed to match a non-default spec."""
    onnx, plan = engine  # no sidecar
    monkeypatch.setenv("SENSEVOICE_TRT_ARGMAX", "1")
    stale = md._sensevoice_engine_staleness(
        str(plan), md._sensevoice_build_spec(str(onnx))
    )
    assert stale is not None and "predates build info" in stale


# ── Flag parsing ─────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("no", False),
])
def test_env_flag_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("SENSEVOICE_TRT_ARGMAX", raw)
    assert md._env_flag("SENSEVOICE_TRT_ARGMAX", False) is expected


def test_env_flag_blank_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SENSEVOICE_TRT_FP16", "   ")
    assert md._env_flag("SENSEVOICE_TRT_FP16", True) is True


def test_env_int_rejects_garbage(monkeypatch):
    monkeypatch.setenv("SENSEVOICE_TRT_WORKSPACE_GIB", "big")
    assert md._env_int("SENSEVOICE_TRT_WORKSPACE_GIB", 3) == 3
