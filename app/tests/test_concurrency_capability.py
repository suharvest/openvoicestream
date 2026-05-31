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


# NOTE: jetson backend config→capability declarations (trt_edge_llm_tts, matcha,
# kokoro, paraformer, qwen3) moved to voxedge with the env-free migration. The
# env/profile → config precedence is covered in test_voxedge_backend_config.py;
# the config → ConcurrencyCapability half is covered in voxedge:
#   voxedge/tests/test_{trt_edge_llm_tts,matcha,kokoro,paraformer}_concurrency_cap.py.
# The product-only ABC defaults + CPU/RK declarations stay below.


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
