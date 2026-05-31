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


# ---------------------------------------------------------------------------
# matcha_trt
# ---------------------------------------------------------------------------

def test_matcha_init_reads_current_env(monkeypatch):
    _stub_native_deps()
    import voxedge.backends.jetson.matcha_trt as matcha_mod

    monkeypatch.setenv("MATCHA_MODEL_BASE", "/instance/A_matcha")
    monkeypatch.setenv("VOCOS_ENGINE", "/instance/A_matcha/vocos.engine")
    monkeypatch.setenv("LEXICON_PATH", "/instance/A_matcha/lex.txt")
    monkeypatch.setenv("TOKENS_PATH", "/instance/A_matcha/tok.txt")

    b1 = matcha_mod.MatchaTRTBackend()
    assert b1._model_base == "/instance/A_matcha"
    assert b1._vocos_engine_path == "/instance/A_matcha/vocos.engine"
    assert b1._lexicon_path == "/instance/A_matcha/lex.txt"
    assert b1._tokens_path == "/instance/A_matcha/tok.txt"

    monkeypatch.setenv("MATCHA_MODEL_BASE", "/instance/B_matcha")
    monkeypatch.setenv("VOCOS_ENGINE", "/instance/B_matcha/vocos.engine")
    monkeypatch.setenv("LEXICON_PATH", "/instance/B_matcha/lex.txt")
    monkeypatch.setenv("TOKENS_PATH", "/instance/B_matcha/tok.txt")

    b2 = matcha_mod.MatchaTRTBackend()
    assert b2._model_base == "/instance/B_matcha"
    assert b2._vocos_engine_path == "/instance/B_matcha/vocos.engine"

    # Old instance unchanged — proves per-instance snapshot, not shared state.
    assert b1._model_base == "/instance/A_matcha"


def test_matcha_init_defaults_derived_from_model_base(monkeypatch):
    _stub_native_deps()
    import voxedge.backends.jetson.matcha_trt as matcha_mod

    monkeypatch.setenv("MATCHA_MODEL_BASE", "/m/base")
    monkeypatch.delenv("VOCOS_ENGINE", raising=False)
    monkeypatch.delenv("LEXICON_PATH", raising=False)
    monkeypatch.delenv("TOKENS_PATH", raising=False)

    b = matcha_mod.MatchaTRTBackend()
    assert b._vocos_engine_path == "/m/base/engines/vocos_fp16.engine"
    assert b._lexicon_path == "/m/base/lexicon.txt"
    assert b._tokens_path == "/m/base/tokens.txt"


# ---------------------------------------------------------------------------
# kokoro_trt
# ---------------------------------------------------------------------------

def test_kokoro_init_reads_current_env(monkeypatch):
    _stub_native_deps()
    import voxedge.backends.jetson.kokoro_trt as kokoro_mod

    monkeypatch.setenv("KOKORO_MODEL_BASE", "/instance/A_kokoro")
    monkeypatch.setenv("KOKORO_VOICES", "/instance/A_kokoro/voices.bin")
    monkeypatch.setenv("KOKORO_TOKENS", "/instance/A_kokoro/tokens.txt")
    monkeypatch.setenv(
        "KOKORO_SPLIT_DECODER_ENGINE_LONG",
        "/instance/A_kokoro/engines/decoder_long.engine",
    )

    b1 = kokoro_mod.KokoroTRTBackend()
    assert b1._paths["model_base"] == "/instance/A_kokoro"
    assert b1._paths["voices_bin"] == "/instance/A_kokoro/voices.bin"
    assert b1._paths["tokens_path"] == "/instance/A_kokoro/tokens.txt"
    assert b1._paths["split_decoder_engine_long"] == \
        "/instance/A_kokoro/engines/decoder_long.engine"

    monkeypatch.setenv("KOKORO_MODEL_BASE", "/instance/B_kokoro")
    monkeypatch.setenv("KOKORO_VOICES", "/instance/B_kokoro/voices.bin")

    b2 = kokoro_mod.KokoroTRTBackend()
    assert b2._paths["model_base"] == "/instance/B_kokoro"
    assert b2._paths["voices_bin"] == "/instance/B_kokoro/voices.bin"
    # Old instance unchanged.
    assert b1._paths["model_base"] == "/instance/A_kokoro"


def test_kokoro_init_defaults_derived_from_model_base(monkeypatch):
    _stub_native_deps()
    import voxedge.backends.jetson.kokoro_trt as kokoro_mod

    monkeypatch.setenv("KOKORO_MODEL_BASE", "/k/base")
    for key in (
        "KOKORO_VOICES", "KOKORO_TOKENS", "KOKORO_TRT_ENGINE",
        "KOKORO_SPLIT_DECODER_ENGINE_LONG", "KOKORO_HYBRID_DIR",
    ):
        monkeypatch.delenv(key, raising=False)

    b = kokoro_mod.KokoroTRTBackend()
    assert b._paths["voices_bin"] == "/k/base/voices.bin"
    assert b._paths["tokens_path"] == "/k/base/tokens.txt"
    assert b._paths["engine_path"] == "/k/base/engines/kokoro_fp16.engine"
    assert b._paths["hybrid_dir"] == "/k/base/hybrid"


# ---------------------------------------------------------------------------
# qwen3_trt
# ---------------------------------------------------------------------------

def test_qwen3_init_reads_current_env(monkeypatch):
    _stub_native_deps()
    import voxedge.backends.jetson.qwen3_trt as qwen3_mod

    monkeypatch.setenv("QWEN3_MODEL_BASE", "/instance/A_qwen3")
    monkeypatch.setenv(
        "QWEN3_TALKER_ENGINE", "/instance/A_qwen3/engines/talker.engine"
    )
    monkeypatch.setenv("QWEN3_TOKENIZER_DIR", "/instance/A_qwen3/tok")

    b1 = qwen3_mod.Qwen3TRTBackend()
    assert b1._paths["base"] == "/instance/A_qwen3"
    assert b1._paths["talker_engine"] == "/instance/A_qwen3/engines/talker.engine"
    assert b1._paths["tokenizer_dir"] == "/instance/A_qwen3/tok"

    monkeypatch.setenv("QWEN3_MODEL_BASE", "/instance/B_qwen3")
    monkeypatch.setenv("QWEN3_TALKER_ENGINE", "/instance/B_qwen3/engines/talker.engine")
    b2 = qwen3_mod.Qwen3TRTBackend()
    assert b2._paths["base"] == "/instance/B_qwen3"
    assert b2._paths["talker_engine"] == "/instance/B_qwen3/engines/talker.engine"
    # Old instance unchanged.
    assert b1._paths["base"] == "/instance/A_qwen3"


# ---------------------------------------------------------------------------
# trt_edge_llm_asr — sampling defaults captured per-instance via _load_config
# ---------------------------------------------------------------------------

def test_asr_sampling_defaults_captured_per_instance(monkeypatch):
    _stub_native_deps()
    import voxedge.backends.jetson.trt_edge_llm_asr as asr_mod

    monkeypatch.setenv("ASR_TEMPERATURE", "0.5")
    monkeypatch.setenv("ASR_TOP_P", "0.9")
    monkeypatch.setenv("ASR_TOP_K", "10")
    monkeypatch.setenv("ASR_MAX_GENERATE_LENGTH", "300")
    monkeypatch.delenv("EDGE_LLM_ASR_MANIFEST", raising=False)

    b1 = asr_mod.TRTEdgeLLMASRBackend()
    assert b1._config["temperature"] == pytest.approx(0.5)
    assert b1._config["top_p"] == pytest.approx(0.9)
    assert b1._config["top_k"] == 10
    assert b1._config["max_generate_length"] == 300

    monkeypatch.setenv("ASR_TEMPERATURE", "1.5")
    monkeypatch.setenv("ASR_TOP_K", "1")
    b2 = asr_mod.TRTEdgeLLMASRBackend()
    assert b2._config["temperature"] == pytest.approx(1.5)
    assert b2._config["top_k"] == 1
    # Old instance unchanged.
    assert b1._config["temperature"] == pytest.approx(0.5)


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


# ---------------------------------------------------------------------------
# trt_edge_llm_tts — code2wav + worker binary captured per-instance
# ---------------------------------------------------------------------------

def test_tts_backend_init_captures_code2wav_and_worker(monkeypatch):
    _stub_native_deps()
    import voxedge.backends.jetson.trt_edge_llm_tts as tts_mod

    monkeypatch.setenv("EDGE_LLM_TTS_TALKER_DIR", "/inst/A/talker")
    monkeypatch.setenv("EDGE_LLM_TTS_TOKENIZER_DIR", "/inst/A/tok")
    monkeypatch.setenv("EDGE_LLM_TTS_CP_DIR", "/inst/A/cp")
    monkeypatch.setenv("EDGE_LLM_TTS_CODE2WAV_DIR", "/inst/A/c2w")
    monkeypatch.setenv("EDGE_LLM_TTS_WORKER_BIN", "/inst/A/worker")
    monkeypatch.setenv("EDGE_LLM_QWEN3_PROFILE", "highperf")

    b1 = tts_mod.TRTEdgeLLMTTSBackend()
    assert b1._code2wav_dir == "/inst/A/c2w"
    assert b1._worker_binary == "/inst/A/worker"
    assert b1._qwen3_runtime_profile == "highperf"

    monkeypatch.setenv("EDGE_LLM_TTS_CODE2WAV_DIR", "/inst/B/c2w")
    monkeypatch.setenv("EDGE_LLM_TTS_WORKER_BIN", "/inst/B/worker")
    monkeypatch.setenv("EDGE_LLM_QWEN3_PROFILE", "official")

    b2 = tts_mod.TRTEdgeLLMTTSBackend()
    assert b2._code2wav_dir == "/inst/B/c2w"
    assert b2._worker_binary == "/inst/B/worker"
    assert b2._qwen3_runtime_profile == "official"
    # Old instance unchanged.
    assert b1._code2wav_dir == "/inst/A/c2w"
