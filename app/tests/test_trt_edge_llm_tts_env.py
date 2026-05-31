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


# NOTE: TRTEdgeLLMTTSBackend used to snapshot talker/tokenizer/cp dirs from env
# in __init__. Those now come from the injected env-free config
# (talker_dir/tokenizer_dir/code_predictor_dir); env→config is covered in
# test_voxedge_backend_config.py, so the per-instance env-snapshot test is
# dropped (immutable injected config makes the regression impossible). The
# app.core.deploy_paths resolver tests above still read env fresh and stay.
