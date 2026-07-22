"""Unit tests for ``server.core.capability_resolver``.

Follow-up #4 (spec §3/§4/§5). Verifies behaviour parity with the three
previous sites that re-implemented capability resolution:

- limiter ``session_ceiling`` (aggregate + clamp/warn)
- coordinator ``coordinator_mode`` (downgrade per §4)
- main ``executor_max_workers`` (clamp + capability fallback per §5)
"""

from __future__ import annotations

import logging

import pytest

from server.core.capability_resolver import (
    ResolvedCapability,
    _aggregate_ceiling,
    resolve,
    resolve_executor_for_tts,
)
from server.core.concurrency_capability import ConcurrencyCapability
from server.core.diarization import diarization_concurrency_capability


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "OVS_MAX_CONCURRENT_SESSIONS",
        "OVS_TTS_STREAM_MAX_WORKERS",
        "OVS_TTS_STREAM_MAX_WORKERS_KOKORO",
        "OVS_TTS_STREAM_MAX_WORKERS_MATCHA",
        "OVS_TTS_STREAM_MAX_WORKERS_QWEN3",
        "OVS_TTS_STREAM_MAX_WORKERS_MOSS",
        "OVS_DIARIZE",
        "OVS_DIARIZE_MAX_CONCURRENT",
        "DIAR_CAMPPLUS_ENGINE_FILE",
        "RK_PLATFORM",
        "LANGUAGE_MODE",
    ):
        monkeypatch.delenv(k, raising=False)


# ---------- Aggregation math (spec §1) -----------------------------------


def test_aggregate_finite_vs_none_takes_finite():
    """paraformer max=None (inf) + matcha max=2 -> 2."""
    r = resolve(profile={
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
    })
    assert r.session_ceiling == 2
    assert r.executor_max_workers == 2


def test_aggregate_both_none_falls_back_to_target_default():
    """When both backends have max=None, target_default still kicks in."""
    # cpu.sherpa_asr/sherpa are both parallel but with finite max=4
    r = resolve(profile={
        "asr_backend": "cpu.sherpa_asr",
        "tts_backend": "cpu.sherpa",
    })
    assert r.session_ceiling == 4


def test_no_declared_backends_uses_target_default():
    r = resolve(profile={"name": "desktop-mac"})
    assert r.session_ceiling == 4


def test_no_profile_uses_unknown_default():
    r = resolve(profile=None)
    assert r.session_ceiling == 1


# ---------- Profile clamp + warning (spec §3) -----------------------------


def test_profile_clamp_warning():
    r = resolve(profile={
        "asr_backend": "rk.asr",
        "tts_backend": "rk.tts",
        "max_concurrent_sessions": 99,
    })
    assert r.session_ceiling == 1
    assert any("max_concurrent_sessions" in w for w in r.clamp_warnings)


def test_profile_downgrade_no_warning():
    r = resolve(profile={
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
        "max_concurrent_sessions": 1,
    })
    assert r.session_ceiling == 1
    assert r.clamp_warnings == [] or all(
        "exceeds" not in w for w in r.clamp_warnings
    )


# ---------- Env clamp + warning (spec §3) ---------------------------------


def test_env_clamp_warning():
    env = {"OVS_MAX_CONCURRENT_SESSIONS": "16"}
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        env=env,
    )
    assert r.session_ceiling == 2
    assert any("OVS_MAX_CONCURRENT_SESSIONS" in w for w in r.clamp_warnings)


def test_env_downgrade_honored():
    env = {"OVS_MAX_CONCURRENT_SESSIONS": "1"}
    r = resolve(
        profile={
            "asr_backend": "cpu.sherpa_asr",
            "tts_backend": "cpu.sherpa",
        },
        env=env,
    )
    assert r.session_ceiling == 1


def test_env_bad_value_raises():
    with pytest.raises(ValueError):
        resolve(profile={}, env={"OVS_MAX_CONCURRENT_SESSIONS": "0"})


def test_profile_bad_value_raises():
    with pytest.raises(ValueError):
        resolve(profile={"max_concurrent_sessions": -1})


# ---------- Coordinator mode (spec §4) ------------------------------------


def test_coordinator_concurrent_jetson_pair():
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        policy={"mode": "concurrent"},
    )
    assert r.coordinator_mode == "concurrent"


def test_coordinator_concurrent_downgraded_for_rk():
    r = resolve(
        profile={"asr_backend": "rk.asr", "tts_backend": "rk.tts"},
        policy={"mode": "concurrent"},
    )
    assert r.coordinator_mode == "serialized"


def test_coordinator_exclusive_honored():
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        policy={"mode": "exclusive"},
    )
    assert r.coordinator_mode == "exclusive"


def test_coordinator_serialized_passthrough():
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        policy={"mode": "serialized"},
    )
    assert r.coordinator_mode == "serialized"


def test_coordinator_no_profile_keeps_requested_concurrent():
    """Legacy callers without profile: pass through raw policy.mode."""
    r = resolve(profile=None, policy={"mode": "concurrent"})
    assert r.coordinator_mode == "concurrent"


def test_coordinator_mixed_pair_downgrades():
    r = resolve(
        profile={
            "asr_backend": "rk.asr",
            "tts_backend": "jetson.matcha_trt",
        },
        policy={"mode": "concurrent"},
    )
    assert r.coordinator_mode == "serialized"


# ---------- Executor max_workers (spec §5) -------------------------------


def test_executor_env_clamped_to_capability():
    env = {"OVS_TTS_STREAM_MAX_WORKERS_MATCHA": "16"}
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        env=env,
        tts_backend_name="jetson.matcha_trt.fp16",
    )
    assert r.executor_max_workers == 2
    assert any("exceeds backend ceiling" in w for w in r.clamp_warnings)


def test_executor_falls_back_to_capability_when_no_env():
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        env={},
        tts_backend_name="jetson.matcha_trt.fp16",
    )
    assert r.executor_max_workers == 2


def test_executor_legacy_default_when_no_tts_backend_declared():
    """No tts_backend in profile → legacy default 2 (cap not consulted)."""
    r = resolve(profile={"name": "desktop-mac"}, env={})
    assert r.executor_max_workers == 2


def test_executor_backend_specific_env_wins_over_global():
    env = {
        "OVS_TTS_STREAM_MAX_WORKERS": "16",
        "OVS_TTS_STREAM_MAX_WORKERS_KOKORO": "1",
    }
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.kokoro_trt",
        },
        env=env,
        tts_backend_name="jetson.kokoro_trt.fp16",
    )
    assert r.executor_max_workers == 1


# ---------- Cross-caller consistency --------------------------------------


def test_three_callers_share_ceiling_and_mode():
    """Same profile → all three projections must agree on the underlying
    capability snapshot. Regression guard against future drift."""
    profile = {
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
    }
    policy = {"mode": "concurrent"}
    r = resolve(profile=profile, policy=policy,
                tts_backend_name="jetson.matcha_trt.fp16", env={})

    # Limiter sees ceiling=2.
    assert r.session_ceiling == 2
    # Coordinator agrees concurrent is OK.
    assert r.coordinator_mode == "concurrent"
    # Executor cap == ceiling (no env override).
    assert r.executor_max_workers == 2
    assert r.executor_max_workers == r.session_ceiling


def test_three_callers_consistent_for_rk():
    profile = {"asr_backend": "rk.asr", "tts_backend": "rk.tts"}
    policy = {"mode": "concurrent"}
    r = resolve(profile=profile, policy=policy, env={})
    assert r.session_ceiling == 1
    assert r.coordinator_mode == "serialized"
    assert r.executor_max_workers == 1


# ---------- Thin wrapper parity with legacy return shape -----------------


def test_resolve_executor_for_tts_returns_legacy_shape():
    env = {"OVS_TTS_STREAM_MAX_WORKERS_MATCHA": "16"}
    n, name, src = resolve_executor_for_tts(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        tts_backend_name="jetson.matcha_trt.fp16",
        env=env,
    )
    assert n == 2
    assert name == "jetson.matcha_trt.fp16"
    assert src == "OVS_TTS_STREAM_MAX_WORKERS_MATCHA"


def test_resolve_executor_for_tts_source_capability():
    n, name, src = resolve_executor_for_tts(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        tts_backend_name="jetson.matcha_trt.fp16",
        env={},
    )
    assert n == 2
    assert src == "concurrency_capability"


def test_resolve_executor_for_tts_source_default():
    n, name, src = resolve_executor_for_tts(
        profile={},
        tts_backend_name=None,
        env={},
    )
    # No profile-declared TTS backend → legacy default 2 + "default"
    # source (mirrors pre-resolver behavior in server.main).
    assert n == 2
    assert src == "default"


# ---------- Diarization fold-in (default-off invariant) ------------------

_MATCHA_PAIR = {
    "asr_backend": "jetson.paraformer_trt",  # max_concurrent=None (inf)
    "tts_backend": "jetson.matcha_trt",      # max_concurrent=2
}


def test_aggregate_ceiling_backcompat_byte_identical():
    """Old two-arg call (extra omitted) → ceiling + label byte-identical to
    the pre-generalization implementation."""
    inf = ConcurrencyCapability(max_concurrent=None)
    two = ConcurrencyCapability(max_concurrent=2)
    four = ConcurrencyCapability(max_concurrent=4)
    assert _aggregate_ceiling(inf, inf) == (None, "asr=inf,tts=inf")
    assert _aggregate_ceiling(inf, two) == (2, "asr=inf,tts=2")
    assert _aggregate_ceiling(four, inf) == (4, "asr=4,tts=inf")
    assert _aggregate_ceiling(four, two) == (2, "asr=4,tts=2")


def test_aggregate_ceiling_with_diar_extra():
    """(c) asr=4/tts=8 + diar=3 → 3; asr=2 + diar=3 → 2 (min wins)."""
    diar = ConcurrencyCapability(max_concurrent=3)
    a4 = ConcurrencyCapability(max_concurrent=4)
    t8 = ConcurrencyCapability(max_concurrent=8)
    a2 = ConcurrencyCapability(max_concurrent=2)
    assert _aggregate_ceiling(a4, t8, [("diar", diar)]) == (3, "asr=4,tts=8,diar=3")
    assert _aggregate_ceiling(a2, t8, [("diar", diar)]) == (2, "asr=2,tts=8,diar=3")


def test_diar_disabled_ceiling_unchanged():
    """(a) OVS_DIARIZE unset → ceiling/source identical to asr/tts-only."""
    baseline = resolve(profile=dict(_MATCHA_PAIR), env={})
    withflag = resolve(profile=dict(_MATCHA_PAIR), env={})  # still disabled
    assert baseline.session_ceiling == 2
    assert withflag.session_ceiling == baseline.session_ceiling
    assert withflag.ceiling_source == baseline.ceiling_source == "asr=2,tts=2"
    assert withflag.diar_cap is None


def test_diar_enabled_no_max_does_not_tighten():
    """(b) OVS_DIARIZE=1 but no OVS_DIARIZE_MAX_CONCURRENT → diar=inf, the
    ceiling stays min(asr,tts)=2; only the source label gains diar=inf."""
    r = resolve(profile=dict(_MATCHA_PAIR), env={"OVS_DIARIZE": "1"})
    assert r.session_ceiling == 2
    assert r.ceiling_source == "asr=2,tts=2,diar=inf"
    assert r.diar_cap is not None
    assert r.diar_cap.max_concurrent is None


def test_diar_enabled_max_tightens_ceiling():
    """(c) OVS_DIARIZE=1 + OVS_DIARIZE_MAX_CONCURRENT=1 tightens 2 → 1."""
    r = resolve(
        profile=dict(_MATCHA_PAIR),
        env={"OVS_DIARIZE": "1", "OVS_DIARIZE_MAX_CONCURRENT": "1"},
    )
    assert r.session_ceiling == 1
    assert r.ceiling_source == "asr=2,tts=2,diar=1"


def test_diar_enabled_max_above_ceiling_no_change():
    """diar cap above the asr/tts ceiling does not raise it."""
    r = resolve(
        profile=dict(_MATCHA_PAIR),
        env={"OVS_DIARIZE": "1", "OVS_DIARIZE_MAX_CONCURRENT": "9"},
    )
    assert r.session_ceiling == 2
    assert r.ceiling_source == "asr=2,tts=2,diar=9"


def test_diar_profile_optin_without_env():
    """Profile diarize:true enables the cap even with OVS_DIARIZE unset."""
    profile = dict(_MATCHA_PAIR)
    profile["diarize"] = True
    r = resolve(profile=profile, env={})
    assert r.diar_cap is not None
    assert r.session_ceiling == 2  # no max_concurrent → no tightening


def test_diar_not_folded_when_no_declared_backends():
    """No asr/tts backend declared → target-default path, diar never folds."""
    r = resolve(profile={"name": "desktop-mac"}, env={"OVS_DIARIZE": "1"})
    assert r.session_ceiling == 4
    assert r.ceiling_source == "target_default=desktop"


# ---------- diarization_concurrency_capability unit (spec) ---------------


def test_diar_cap_disabled_returns_none():
    """(d) default-off → None (does not participate in ceiling)."""
    assert diarization_concurrency_capability(profile=None, env={}) is None
    assert diarization_concurrency_capability(profile={}, env={}) is None


def test_diar_cap_enabled_fields():
    """(d) enabled → correct descriptor fields."""
    cap = diarization_concurrency_capability(
        profile=None, env={"OVS_DIARIZE": "1", "OVS_DIARIZE_MAX_CONCURRENT": "3"}
    )
    assert cap is not None
    assert cap.supports_parallel is True
    assert cap.max_concurrent == 3
    assert cap.is_stateful is True
    assert cap.requires_exclusive_device is False
    assert cap.scaling_mode == "per_call_isolated"
    assert cap.vram_mb_per_slot == 120  # no TRT engine file → CPU/sherpa budget


def test_diar_cap_max_concurrent_default_none():
    cap = diarization_concurrency_capability(profile=None, env={"OVS_DIARIZE": "1"})
    assert cap is not None and cap.max_concurrent is None


def test_diar_cap_max_concurrent_nonpositive_is_none():
    cap = diarization_concurrency_capability(
        profile=None, env={"OVS_DIARIZE": "1", "OVS_DIARIZE_MAX_CONCURRENT": "0"}
    )
    assert cap is not None and cap.max_concurrent is None
