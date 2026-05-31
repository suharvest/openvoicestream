"""Regression tests for the TRT-Edge-LLM TTS artifact path resolution.

Background: before this fix, `trt_edge_llm_ipc` exported TTS_TALKER_DIR /
TTS_CODE_PREDICTOR_DIR / TTS_TOKENIZER_DIR as module-level constants captured
from os.environ at import time. Hot reload via BackendManager.apply_profile()
mutates os.environ but cannot reach those frozen constants, so a fresh backend
instance built post-reload would consult stale paths and fail preload().

Fix: add resolve_tts_*_dir() helpers that re-read os.environ each call, and
have TRTEdgeLLMTTSBackend.__init__ capture the resolved paths as instance
attributes (BackendManager builds a fresh instance after every apply_profile,
so __init__ always sees the latest env).
"""

from __future__ import annotations

import os


def test_resolvers_read_env_fresh(monkeypatch):
    """resolve_tts_talker_dir must reflect the *current* os.environ, never a
    snapshot taken at import time."""
    import app.core.deploy_paths as ipc

    monkeypatch.setenv("EDGE_LLM_TTS_TALKER_DIR", "/path/A")
    assert ipc.resolve_tts_talker_dir() == "/path/A"

    monkeypatch.setenv("EDGE_LLM_TTS_TALKER_DIR", "/path/B")
    assert ipc.resolve_tts_talker_dir() == "/path/B"

    monkeypatch.setenv("EDGE_LLM_TTS_TOKENIZER_DIR", "/tok/X")
    assert ipc.resolve_tts_tokenizer_dir() == "/tok/X"
    monkeypatch.setenv("EDGE_LLM_TTS_TOKENIZER_DIR", "/tok/Y")
    assert ipc.resolve_tts_tokenizer_dir() == "/tok/Y"

    monkeypatch.setenv("EDGE_LLM_TTS_CP_DIR", "/cp/1")
    assert ipc.resolve_tts_code_predictor_dir() == "/cp/1"
    monkeypatch.setenv("EDGE_LLM_TTS_CP_DIR", "/cp/2")
    assert ipc.resolve_tts_code_predictor_dir() == "/cp/2"


def test_resolver_code_predictor_defaults_off_talker(monkeypatch):
    """When no explicit CP dir, the default sits next to the talker dir
    (mirrors module-level cold-boot logic)."""
    import app.core.deploy_paths as ipc

    monkeypatch.setenv("EDGE_LLM_TTS_TALKER_DIR", "/root/engines/talker")
    monkeypatch.delenv("EDGE_LLM_TTS_CP_DIR", raising=False)
    # Also disable highperf bf16-io probe to keep result deterministic:
    monkeypatch.setenv("QWEN3_RUNTIME_PROFILE", "balanced")

    cp = ipc.resolve_tts_code_predictor_dir()
    assert cp == "/root/engines/code_predictor"


def test_backend_init_captures_current_env(monkeypatch):
    """Each TRTEdgeLLMTTSBackend instance must snapshot env at __init__ time,
    so a fresh instance built after apply_profile() sees the new paths."""
    import voxedge.backends.jetson.trt_edge_llm_tts as tts_mod

    monkeypatch.setenv("EDGE_LLM_TTS_TALKER_DIR", "/instance/A")
    monkeypatch.setenv("EDGE_LLM_TTS_TOKENIZER_DIR", "/instance/A_tok")
    monkeypatch.setenv("EDGE_LLM_TTS_CP_DIR", "/instance/A_cp")

    b = tts_mod.TRTEdgeLLMTTSBackend()
    assert b._talker_dir == "/instance/A"
    assert b._tokenizer_dir == "/instance/A_tok"
    assert b._code_predictor_dir == "/instance/A_cp"

    # Simulate apply_profile() rewriting env, then BackendManager building a
    # NEW backend. The new instance must see the new values; the old one must
    # keep its original snapshot (per-instance state, not shared module state).
    monkeypatch.setenv("EDGE_LLM_TTS_TALKER_DIR", "/instance/B")
    monkeypatch.setenv("EDGE_LLM_TTS_TOKENIZER_DIR", "/instance/B_tok")
    monkeypatch.setenv("EDGE_LLM_TTS_CP_DIR", "/instance/B_cp")

    b2 = tts_mod.TRTEdgeLLMTTSBackend()
    assert b2._talker_dir == "/instance/B"
    assert b2._tokenizer_dir == "/instance/B_tok"
    assert b2._code_predictor_dir == "/instance/B_cp"

    # Old instance unchanged — proves we don't share state via module-level
    # mutable globals.
    assert b._talker_dir == "/instance/A"
