"""Coordinator mode-resolution tests (spec §4).

`concurrent` is permitted only when both ASR + TTS declare
``supports_parallel=True`` with effective max > 1. Otherwise downgrade
to ``serialized``. Profile ``exclusive`` is always honored as-is.
"""

from __future__ import annotations

import asyncio

from server.core.coordinator import BackendCoordinator, _resolve_mode


# ---------- _resolve_mode unit table ----------


def test_concurrent_jetson_pair_stays_concurrent():
    """matcha (parallel, K=2) + paraformer (parallel, max=None) -> concurrent."""
    policy = {"mode": "concurrent"}
    profile = {
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
    }
    assert _resolve_mode(policy, profile) == "concurrent"


def test_concurrent_with_rk_downgrades_to_serialized():
    """rk asr/tts declare supports_parallel=False -> downgrade."""
    policy = {"mode": "concurrent"}
    profile = {
        "asr_backend": "rk.asr",
        "tts_backend": "rk.tts",
    }
    assert _resolve_mode(policy, profile) == "serialized"


def test_concurrent_with_cpu_pair_stays_concurrent():
    """sherpa CPU (parallel, 4) on both sides -> concurrent."""
    policy = {"mode": "concurrent"}
    profile = {
        "asr_backend": "cpu.sherpa_asr",
        "tts_backend": "cpu.sherpa",
    }
    assert _resolve_mode(policy, profile) == "concurrent"


def test_concurrent_mixed_rk_asr_jetson_tts_downgrades():
    """Any single non-parallel backend downgrades the pair."""
    policy = {"mode": "concurrent"}
    profile = {
        "asr_backend": "rk.asr",
        "tts_backend": "jetson.matcha_trt",
    }
    assert _resolve_mode(policy, profile) == "serialized"


def test_profile_serialized_stays_serialized():
    policy = {"mode": "serialized"}
    profile = {
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
    }
    assert _resolve_mode(policy, profile) == "serialized"


def test_profile_exclusive_always_honored():
    """``exclusive`` is never auto-resolved away."""
    policy = {"mode": "exclusive"}
    # even with backends that could run concurrent
    profile = {
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
    }
    assert _resolve_mode(policy, profile) == "exclusive"


def test_no_profile_uses_raw_policy():
    """Legacy callers without profile argument retain old behavior."""
    assert _resolve_mode({"mode": "concurrent"}, None) == "concurrent"
    assert _resolve_mode({"mode": "serialized"}, None) == "serialized"
    assert _resolve_mode({}, None) == "concurrent"  # default


# ---------- Integration with BackendCoordinator ----------


def test_coordinator_downgrade_creates_lock():
    policy = {"mode": "concurrent"}
    profile = {"asr_backend": "rk.asr", "tts_backend": "rk.tts"}
    c = BackendCoordinator(policy, profile=profile)
    assert c.mode == "serialized"
    assert c._lock is not None


def test_coordinator_concurrent_jetson_no_lock():
    policy = {"mode": "concurrent"}
    profile = {
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
    }
    c = BackendCoordinator(policy, profile=profile)
    assert c.mode == "concurrent"
    assert c._lock is None


def test_coordinator_back_compat_no_profile():
    """Old call signature (no profile) still works."""
    c = BackendCoordinator({"mode": "serialized"})
    assert c.mode == "serialized"
    assert c._lock is not None


def test_exclusive_first_request_evicts_preloaded_opposite_backend():
    events = []

    class Backend:
        def __init__(self, name):
            self.name = name
            self.ready = True

        def is_ready(self):
            return self.ready

        def unload(self):
            events.append(("unload", self.name))
            self.ready = False

        def preload(self):
            events.append(("preload", self.name))
            self.ready = True

    asr = Backend("asr")
    tts = Backend("tts")
    c = BackendCoordinator({"mode": "exclusive"})
    c.register_backend("asr", lambda: asr)
    c.register_backend("tts", lambda: tts)

    async def exercise():
        async with c.acquire("tts"):
            assert tts.is_ready()
            assert not asr.is_ready()

    asyncio.run(exercise())

    assert events == [("unload", "asr")]


def test_exclusive_switch_reloads_target_after_previous_eviction():
    events = []

    class Backend:
        def __init__(self, name):
            self.name = name
            self.ready = True

        def is_ready(self):
            return self.ready

        def unload(self):
            events.append(("unload", self.name))
            self.ready = False

        def preload(self):
            events.append(("preload", self.name))
            self.ready = True

    asr = Backend("asr")
    tts = Backend("tts")
    c = BackendCoordinator({"mode": "exclusive"})
    c.register_backend("asr", lambda: asr)
    c.register_backend("tts", lambda: tts)

    async def exercise():
        async with c.acquire("tts"):
            pass
        async with c.acquire("asr"):
            assert asr.is_ready()
            assert not tts.is_ready()

    asyncio.run(exercise())

    assert events == [
        ("unload", "asr"),
        ("unload", "tts"),
        ("preload", "asr"),
    ]
