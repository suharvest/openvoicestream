"""An undeclared ASR/TTS kind must not constrain the kind that IS declared.

Before this rule, ``capability_resolver`` gave an absent kind
``ConcurrencyCapability.default()`` (max=1, supports_parallel=False). Since the
session ceiling is ``min(asr, tts)`` and the coordinator downgrade needs *both*
kinds to be parallel-capable, every ASR-only profile was pinned to one session
and forced to ``serialized`` by a TTS backend that does not exist — observed in
the field as::

    coordinator: downgrading concurrent -> serialized
        (asr.supports_parallel=False/max=1, tts.supports_parallel=False/max=1)

on a profile carrying no ``tts_backend`` key at all.

"Declared" means *in the profile and known to the registry*. The two other
cases ``_resolve_backend_class`` also answers ``None`` for — unknown spec and
failed import — are faults, not absences, and must keep constraining the gate.
That distinction is what these tests pin; the registries are monkeypatched with
local fakes so nothing here needs the optional ``voxedge`` extra installed.
"""
from __future__ import annotations

import pytest

from server.core import capability_resolver as cr
from server.core.concurrency_capability import ConcurrencyCapability


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in ("OVS_MAX_CONCURRENT_SESSIONS", "OVS_DIARIZE", "LANGUAGE_MODE"):
        monkeypatch.delenv(k, raising=False)


def _backend(cap: ConcurrencyCapability):
    class _Fake:
        @classmethod
        def concurrency_capability(cls, profile=None):
            return cap

    return _Fake


def _install(monkeypatch, *, asr=None, tts=None):
    """Point both registries at local fakes, bypassing voxedge imports."""
    import server.core.asr_backend as asr_mod
    import server.core.tts_backend as tts_mod

    asr_reg = {"fake.asr": ("server.tests.test_absent_kind_capability", "_unused")}
    tts_reg = {"fake.tts": ("server.tests.test_absent_kind_capability", "_unused")}
    monkeypatch.setattr(asr_mod, "_ASR_REGISTRY", asr_reg, raising=False)
    monkeypatch.setattr(tts_mod, "_TTS_REGISTRY", tts_reg, raising=False)

    def _resolve_cls(profile, key, registry):
        spec = (profile or {}).get(key)
        if key == "asr_backend" and spec == "fake.asr":
            return _backend(asr) if asr is not None else None
        if key == "tts_backend" and spec == "fake.tts":
            return _backend(tts) if tts is not None else None
        return None

    monkeypatch.setattr(cr, "_resolve_backend_class", _resolve_cls)
    # The voxedge-specific config path is irrelevant for fakes; force the
    # classmethod fallback in ``_capability_for``.
    monkeypatch.setattr(
        cr, "_capability_for",
        lambda cls, profile, spec=None, kind=None: (
            cls.concurrency_capability(profile)
            if cls is not None
            else ConcurrencyCapability.default()
        ),
    )


PARALLEL_4 = ConcurrencyCapability(supports_parallel=True, max_concurrent=4)
SERIAL_1 = ConcurrencyCapability(supports_parallel=False, max_concurrent=1)


# ── Absence must not constrain ───────────────────────────────────────

def test_asr_only_profile_keeps_its_own_ceiling(monkeypatch):
    """ASR declares 4, no tts_backend key → ceiling 4, not min(4, 1)."""
    _install(monkeypatch, asr=PARALLEL_4)
    r = cr.resolve(profile={"asr_backend": "fake.asr"})
    assert r.session_ceiling == 4


def test_asr_only_profile_is_not_downgraded_by_absent_tts(monkeypatch):
    _install(monkeypatch, asr=PARALLEL_4)
    r = cr.resolve(profile={
        "asr_backend": "fake.asr",
        "execution_policy": {"mode": "concurrent"},
    })
    assert r.coordinator_mode == "concurrent"


def test_tts_only_profile_keeps_its_own_ceiling(monkeypatch):
    """Symmetry: an absent ASR must not clamp a TTS-only profile."""
    _install(monkeypatch, tts=PARALLEL_4)
    r = cr.resolve(profile={"tts_backend": "fake.tts"})
    assert r.session_ceiling == 4


# ── A declared kind still constrains ─────────────────────────────────

def test_declared_serial_tts_still_clamps(monkeypatch):
    _install(monkeypatch, asr=PARALLEL_4, tts=SERIAL_1)
    r = cr.resolve(profile={
        "asr_backend": "fake.asr",
        "tts_backend": "fake.tts",
        "execution_policy": {"mode": "concurrent"},
    })
    assert r.session_ceiling == 1
    assert r.coordinator_mode == "serialized"


def test_declared_but_unimportable_stays_conservative(monkeypatch):
    """Declared + in registry + import fails → fault, not absence.

    This is the live case on a host without the ``voxedge`` extra: the gate
    must NOT relax just because the class could not be loaded.
    """
    _install(monkeypatch, asr=PARALLEL_4, tts=None)  # tts class unresolvable
    r = cr.resolve(profile={
        "asr_backend": "fake.asr",
        "tts_backend": "fake.tts",  # declared and in registry
        "execution_policy": {"mode": "concurrent"},
    })
    assert r.session_ceiling == 1, "unimportable declared backend must still clamp"
    assert r.coordinator_mode == "serialized"


def test_declared_unknown_spec_stays_conservative(monkeypatch):
    """Declared but absent from the registry is a misconfiguration, not absence."""
    _install(monkeypatch, asr=PARALLEL_4)
    r = cr.resolve(profile={
        "asr_backend": "fake.asr",
        "tts_backend": "nope.not_a_backend",
    })
    assert r.session_ceiling == 1


# ── Operator overrides keep their precedence ─────────────────────────

def test_profile_pin_below_ceiling_wins(monkeypatch):
    """The rpi guard: profile pin < raised ceiling must be honoured."""
    _install(monkeypatch, asr=PARALLEL_4)
    r = cr.resolve(profile={
        "asr_backend": "fake.asr",
        "max_concurrent_sessions": 1,
    })
    assert r.session_ceiling == 1


def test_env_above_raised_ceiling_still_clamps(monkeypatch):
    _install(monkeypatch, asr=PARALLEL_4)
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "8")
    r = cr.resolve(profile={"asr_backend": "fake.asr"})
    assert r.session_ceiling == 4
    assert any("clamping to 4" in w for w in r.clamp_warnings)


def test_no_declared_backends_still_uses_target_default(monkeypatch):
    """Neither kind declared → unchanged target-table path."""
    _install(monkeypatch)
    r = cr.resolve(profile={"name": "desktop-mac"})
    assert r.session_ceiling == cr._TARGET_DEFAULTS["desktop"]
