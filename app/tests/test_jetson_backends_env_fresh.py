"""Regression tests for jetson backends honoring env at __init__ time.

Background: backends originally captured artifact paths at module import via
``os.environ.get(...)`` module constants. After hot-reload via
BackendManager.apply_profile() rewrites os.environ, a fresh backend
instance built from the same module would still see the *import-time* path
because the module constants were frozen.

Fix: each backend's ``__init__`` now reads the current env via a
``_resolve_*_paths()`` helper (or instance attrs filled from the resolver),
and BackendManager rebuilds the backend after every apply_profile() — so
every new instance sees the latest profile-applied env.

These tests verify that two sequentially constructed instances pick up
different env values (the previous instance keeps its snapshot — no shared
module-level mutable state).
"""

from __future__ import annotations

import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_native_deps() -> None:
    """Stub TRT / CUDA modules so the jetson backends import on Mac/CI."""
    for mod_name in ("tensorrt", "cuda", "cuda.bindings"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)


# NOTE: matcha/kokoro/qwen3/asr backends used to capture artifact paths +
# sampling defaults into backend-internal attrs (``_model_base`` / ``_paths`` /
# dict ``_config[...]``) by reading env in __init__. With the env-free voxedge
# migration those values come from an immutable config injected by the product
# config-builder; env → config (incl. ASR sampling + TTS code2wav/worker paths)
# is covered in test_voxedge_backend_config.py. The per-instance env-snapshot
# regression those tests guarded is now structurally impossible (config is
# immutable + injected per build), so they are dropped. The product-side
# ``app.core.deploy_paths`` resolver tests below still read env fresh and stay.


# ---------------------------------------------------------------------------
# trt_edge_llm_ipc — lazy resolvers read env fresh
# ---------------------------------------------------------------------------

def test_qwen3_runtime_profile_resolver_reads_env_fresh(monkeypatch):
    _stub_native_deps()
    import app.core.deploy_paths as ipc

    monkeypatch.setenv("EDGE_LLM_QWEN3_PROFILE", "official")
    assert ipc.qwen3_runtime_profile() == "official"
    assert ipc.qwen3_highperf_enabled() is False

    monkeypatch.setenv("EDGE_LLM_QWEN3_PROFILE", "high-perf")
    assert ipc.qwen3_runtime_profile() == "high_perf"

    monkeypatch.setenv("EDGE_LLM_QWEN3_PROFILE", "highperf")
    assert ipc.qwen3_highperf_enabled() is True


def test_tts_code2wav_dir_resolver_reads_env_fresh(monkeypatch):
    _stub_native_deps()
    import app.core.deploy_paths as ipc

    monkeypatch.setenv("EDGE_LLM_TTS_CODE2WAV_DIR", "/c2w/A")
    assert ipc.resolve_tts_code2wav_dir() == "/c2w/A"

    monkeypatch.setenv("EDGE_LLM_TTS_CODE2WAV_DIR", "/c2w/B")
    assert ipc.resolve_tts_code2wav_dir() == "/c2w/B"


def test_tts_worker_binary_resolver_reads_env_fresh(monkeypatch):
    _stub_native_deps()
    import app.core.deploy_paths as ipc

    monkeypatch.setenv("EDGE_LLM_TTS_WORKER_BIN", "/bin/A_worker")
    assert ipc.resolve_tts_worker_binary() == "/bin/A_worker"

    monkeypatch.setenv("EDGE_LLM_TTS_WORKER_BIN", "/bin/B_worker")
    assert ipc.resolve_tts_worker_binary() == "/bin/B_worker"


def test_asr_worker_binary_resolver_reads_env_fresh(monkeypatch):
    _stub_native_deps()
    import app.core.deploy_paths as ipc

    monkeypatch.setenv("EDGE_LLM_ASR_WORKER_BIN", "/bin/A_asr_worker")
    assert ipc.resolve_asr_worker_binary() == "/bin/A_asr_worker"

    monkeypatch.setenv("EDGE_LLM_ASR_WORKER_BIN", "/bin/B_asr_worker")
    assert ipc.resolve_asr_worker_binary() == "/bin/B_asr_worker"


# NOTE: TRTEdgeLLMTTSBackend used to snapshot code2wav/worker/profile from env in
# __init__. Those now come from the injected env-free config (code2wav_dir,
# worker_binary, qwen3_runtime_profile); env → config is covered in
# test_voxedge_backend_config.py, so the per-instance env-snapshot test is
# dropped (immutable injected config makes the regression impossible).
