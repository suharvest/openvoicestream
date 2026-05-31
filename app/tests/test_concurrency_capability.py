"""Unit tests for ConcurrencyCapability and backend declarations.

Covers spec docs/specs/concurrency-capability-framework.md P0:
- dataclass shape + conservative defaults
- ABC default returns conservative capability
- 4 N>=2-safe backends declare correctly
- env / profile pool-size overrides are honored
"""

from __future__ import annotations

import importlib
import os

import pytest


def _reload(modpath: str):
    mod = importlib.import_module(modpath)
    return importlib.reload(mod)


# ---------- Dataclass ---------------------------------------------------------


def test_default_is_conservative():
    from app.core.concurrency_capability import ConcurrencyCapability

    cap = ConcurrencyCapability.default()
    assert cap.supports_parallel is False
    assert cap.max_concurrent == 1
    assert cap.is_stateful is True
    assert cap.requires_exclusive_device is True
    assert cap.scaling_mode == "single_runtime_multiplex"
    assert cap.vram_mb_per_slot is None


def test_dataclass_is_frozen():
    from app.core.concurrency_capability import ConcurrencyCapability

    cap = ConcurrencyCapability.default()
    with pytest.raises(Exception):
        cap.max_concurrent = 99  # type: ignore[misc]


def test_dataclass_max_concurrent_none_allowed():
    from app.core.concurrency_capability import ConcurrencyCapability

    cap = ConcurrencyCapability(
        supports_parallel=True,
        max_concurrent=None,
        scaling_mode="multi_runtime_per_slot",
    )
    assert cap.max_concurrent is None


# ---------- ABC defaults ------------------------------------------------------


def test_abc_asr_default_capability():
    from app.core.asr_backend import ASRBackend
    from app.core.concurrency_capability import ConcurrencyCapability

    cap = ASRBackend.concurrency_capability()
    assert cap == ConcurrencyCapability.default()


def test_abc_tts_default_capability():
    from app.core.tts_backend import TTSBackend
    from app.core.concurrency_capability import ConcurrencyCapability

    cap = TTSBackend.concurrency_capability()
    assert cap == ConcurrencyCapability.default()


# ---------- Backend declarations ---------------------------------------------


def test_trt_edge_llm_tts_default_n1(monkeypatch):
    monkeypatch.delenv("OVS_TTS_WORKER_CONCURRENCY", raising=False)
    from voxedge.backends.jetson.trt_edge_llm_tts import TRTEdgeLLMTTSBackend

    cap = TRTEdgeLLMTTSBackend.concurrency_capability()
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is False
    assert cap.scaling_mode == "single_runtime_multiplex"
    assert cap.requires_exclusive_device is True


def test_trt_edge_llm_tts_env_override(monkeypatch):
    monkeypatch.setenv("OVS_TTS_WORKER_CONCURRENCY", "3")
    from voxedge.backends.jetson.trt_edge_llm_tts import TRTEdgeLLMTTSBackend

    cap = TRTEdgeLLMTTSBackend.concurrency_capability()
    assert cap.max_concurrent == 3
    assert cap.supports_parallel is True


def test_trt_edge_llm_tts_profile_override(monkeypatch):
    monkeypatch.delenv("OVS_TTS_WORKER_CONCURRENCY", raising=False)
    from voxedge.backends.jetson.trt_edge_llm_tts import TRTEdgeLLMTTSBackend

    cap = TRTEdgeLLMTTSBackend.concurrency_capability(
        profile={"tts_worker_concurrency": 4}
    )
    assert cap.max_concurrent == 4
    assert cap.supports_parallel is True


def test_trt_edge_llm_tts_env_beats_profile(monkeypatch):
    monkeypatch.setenv("OVS_TTS_WORKER_CONCURRENCY", "2")
    from voxedge.backends.jetson.trt_edge_llm_tts import TRTEdgeLLMTTSBackend

    cap = TRTEdgeLLMTTSBackend.concurrency_capability(
        profile={"tts_worker_concurrency": 8}
    )
    # env takes precedence (matches existing __init__ behavior)
    assert cap.max_concurrent == 2


def test_matcha_default_k2(monkeypatch):
    monkeypatch.delenv("OVS_TTS_STREAM_MAX_WORKERS", raising=False)
    from voxedge.backends.jetson.matcha_trt import MatchaTRTBackend

    cap = MatchaTRTBackend.concurrency_capability()
    assert cap.max_concurrent == 2
    assert cap.supports_parallel is True
    assert cap.scaling_mode == "single_runtime_multiplex"


def test_matcha_env_override(monkeypatch):
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS", "4")
    from voxedge.backends.jetson.matcha_trt import MatchaTRTBackend

    cap = MatchaTRTBackend.concurrency_capability()
    assert cap.max_concurrent == 4


def test_matcha_profile_override(monkeypatch):
    monkeypatch.delenv("OVS_TTS_STREAM_MAX_WORKERS", raising=False)
    from voxedge.backends.jetson.matcha_trt import MatchaTRTBackend

    cap = MatchaTRTBackend.concurrency_capability(
        profile={"tts_stream_max_workers": 3}
    )
    assert cap.max_concurrent == 3


def test_matcha_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS", "not-an-int")
    from voxedge.backends.jetson.matcha_trt import MatchaTRTBackend

    cap = MatchaTRTBackend.concurrency_capability()
    assert cap.max_concurrent == 2  # fallback


def test_kokoro_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS", "not-an-int")
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend

    cap = KokoroTRTBackend.concurrency_capability()
    assert cap.max_concurrent == 2  # fallback


def test_trt_edge_llm_tts_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("OVS_TTS_WORKER_CONCURRENCY", "not-an-int")
    from voxedge.backends.jetson.trt_edge_llm_tts import TRTEdgeLLMTTSBackend

    cap = TRTEdgeLLMTTSBackend.concurrency_capability()
    assert cap.max_concurrent == 1  # conservative fallback


def test_matcha_k1_serialized(monkeypatch):
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS", "1")
    from voxedge.backends.jetson.matcha_trt import MatchaTRTBackend

    cap = MatchaTRTBackend.concurrency_capability()
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is False


def test_kokoro_default_k2(monkeypatch):
    monkeypatch.delenv("OVS_TTS_STREAM_MAX_WORKERS", raising=False)
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend

    cap = KokoroTRTBackend.concurrency_capability()
    assert cap.max_concurrent == 2
    assert cap.supports_parallel is True


def test_kokoro_env_override(monkeypatch):
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS", "6")
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend

    cap = KokoroTRTBackend.concurrency_capability()
    assert cap.max_concurrent == 6


def test_kokoro_profile_override(monkeypatch):
    monkeypatch.delenv("OVS_TTS_STREAM_MAX_WORKERS", raising=False)
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend

    cap = KokoroTRTBackend.concurrency_capability(
        profile={"tts_backend_config": {"stream_max_workers": 5}}
    )
    assert cap.max_concurrent == 5


def test_paraformer_unbounded():
    from voxedge.backends.jetson.paraformer_trt import ParaformerTRTBackend

    cap = ParaformerTRTBackend.concurrency_capability()
    assert cap.supports_parallel is True
    assert cap.max_concurrent is None
    assert cap.scaling_mode == "multi_runtime_per_slot"
    assert cap.requires_exclusive_device is True


# ---------- Non-declared backends fall back to conservative default ----------


def test_qwen3_trt_falls_back_to_default():
    """qwen3_trt does not override concurrency_capability (per spec, kept N=1)."""
    from voxedge.backends.jetson.qwen3_trt import Qwen3TRTBackend
    from app.core.concurrency_capability import ConcurrencyCapability

    cap = Qwen3TRTBackend.concurrency_capability()
    assert cap == ConcurrencyCapability.default()
    assert cap.max_concurrent == 1


# ---------- CPU / desktop backends: parallel 4, non-exclusive ----------


def test_sherpa_tts_cpu_capability():
    from app.backends.cpu.sherpa import SherpaBackend
    cap = SherpaBackend.concurrency_capability()
    assert cap.supports_parallel is True
    assert cap.max_concurrent == 4
    assert cap.requires_exclusive_device is False
    assert cap.scaling_mode == "external_managed"


def test_sherpa_asr_cpu_capability():
    from app.backends.cpu.sherpa_asr import SherpaASRBackend
    cap = SherpaASRBackend.concurrency_capability()
    assert cap.supports_parallel is True
    assert cap.max_concurrent == 4
    assert cap.requires_exclusive_device is False
    assert cap.scaling_mode == "external_managed"


# ---------- RK NPU backends: serial, exclusive, max 1 ----------


def test_rk_asr_capability():
    from app.backends.rk.asr import RKASRBackend
    cap = RKASRBackend.concurrency_capability()
    assert cap.supports_parallel is False
    assert cap.max_concurrent == 1
    assert cap.requires_exclusive_device is True
    assert cap.scaling_mode == "external_managed"


def test_rk_tts_capability():
    from app.backends.rk.tts import RKTTSBackend
    cap = RKTTSBackend.concurrency_capability()
    assert cap.supports_parallel is False
    assert cap.max_concurrent == 1
    assert cap.requires_exclusive_device is True
    assert cap.scaling_mode == "external_managed"
