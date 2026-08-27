"""``asr_max_slots`` must actually reach the SenseVoice backend's capability.

This chain was broken in two independent places and nothing caught it, because
every failure degraded to ``ConcurrencyCapability.default()`` (max_concurrent=1)
behind a debug log — so the knob looked wired up and did nothing. On device the
symptom was ``SessionLimiter initialized: effective_limit=1`` no matter what the
profile said.

The two breaks:

1. ``_ASR_CONFIG_BUILDERS`` had no ``jetson.sensevoice_trt`` entry, so
   ``build_config_for_spec`` returned ``None`` and the voxedge path bailed out
   before ever building a config.
2. ``concurrency_capability_for_spec`` set ``stub._config``, but the SenseVoice
   backend reads ``self._cfg``. Even with (1) fixed the stub raised
   ``AttributeError``.

These tests pin the whole chain — profile value in, capability out — using a
local fake backend, so they run without the voxedge extra installed.
"""
from __future__ import annotations

import pytest

from server.core import voxedge_backend_config as vbc
from server.core.concurrency_capability import ConcurrencyCapability


class _CfgStyleBackend:
    """Reads its config from ``_cfg`` — the SenseVoice convention."""

    def concurrency_capability(self, profile=None):
        return ConcurrencyCapability(
            supports_parallel=False,
            max_concurrent=max(1, int(self._cfg.max_concurrent)),
        )


class _ConfigStyleBackend:
    """Reads its config from ``_config`` — the paraformer convention."""

    def concurrency_capability(self, profile=None):
        return ConcurrencyCapability(
            supports_parallel=True,
            max_concurrent=max(1, int(self._config.max_concurrent)),
        )


class _FakeCfg:
    def __init__(self, max_concurrent):
        self.max_concurrent = max_concurrent


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("SENSEVOICE_MAX_CONCURRENT", raising=False)


@pytest.fixture(autouse=True)
def _stub_voxedge(monkeypatch):
    """Stand in for the optional voxedge extra.

    ``build_sensevoice_trt_config`` imports the real dataclass, which is absent
    on a dev host. Without this the resolver tests would fail — or worse, pass
    for the wrong reason: a failed import degrades to ``default()``, whose
    max_concurrent is 1, so a "defaults to 1 slot" assertion would hold while
    proving nothing.
    """
    import sys
    import types
    from dataclasses import dataclass, field

    @dataclass
    class _StubConfig:
        engine: str = ""
        model_dir: str = ""
        bpe_model: object = None
        max_concurrent: int = 1

        def __post_init__(self):
            self.max_concurrent = max(1, int(self.max_concurrent))

    mods = {}
    for name in ("voxedge", "voxedge.backends", "voxedge.backends.jetson",
                 "voxedge.backends.jetson.sensevoice_trt"):
        mod = types.ModuleType(name)
        mods[name] = mod
        monkeypatch.setitem(sys.modules, name, mod)
    mods["voxedge.backends.jetson.sensevoice_trt"].SenseVoiceTRTConfig = _StubConfig
    return _StubConfig


# ── The registry must know the spec at all ───────────────────────────

def test_sensevoice_spec_is_registered():
    """Break #1: an unregistered spec makes every knob a no-op."""
    assert "jetson.sensevoice_trt" in vbc._ASR_CONFIG_BUILDERS


def _registered_asr_specs():
    from server.core.asr_backend import _ASR_REGISTRY

    return sorted(_ASR_REGISTRY)


@pytest.mark.parametrize("spec", _registered_asr_specs())
def test_every_registered_asr_spec_has_a_builder(spec):
    """Derived from the registry, not a hand-kept list.

    This used to be a literal list of specs, which is the same maintenance trap
    as break #1 above: adding a spec to ``_ASR_REGISTRY`` and forgetting the
    builder left the test green and every concurrency knob a silent no-op.
    """
    assert spec in vbc._ASR_CONFIG_BUILDERS, (
        f"{spec} is in _ASR_REGISTRY but has no config builder; its "
        f"concurrency capability would silently fall back to N=1"
    )
    assert callable(vbc._ASR_CONFIG_BUILDERS[spec])


# ── The stub must satisfy either attribute convention ────────────────

@pytest.mark.parametrize("cls,expected_parallel", [
    (_CfgStyleBackend, False),      # reads _cfg
    (_ConfigStyleBackend, True),    # reads _config
])
def test_stub_feeds_both_attribute_conventions(monkeypatch, cls, expected_parallel):
    """Break #2: setting only one name raised AttributeError, silently."""
    monkeypatch.setitem(
        vbc._ASR_CONFIG_BUILDERS, "fake.spec",
        lambda profile=None, env=None: _FakeCfg(4),
    )
    cap = vbc.concurrency_capability_for_spec("fake.spec", cls, "asr", {})
    assert cap is not None, "stub construction must not fail"
    assert cap.max_concurrent == 4
    assert cap.supports_parallel is expected_parallel


# ── End to end: profile value reaches the resolver ───────────────────

def _resolve_with(monkeypatch, profile, cls=_CfgStyleBackend):
    """Resolve against the REAL registry.

    Deliberately does not monkeypatch ``_ASR_CONFIG_BUILDERS`` — an earlier
    version did, and that re-registered the very entry whose absence was break
    #1, so these tests passed with the bug present.
    """
    from server.core import capability_resolver as cr

    monkeypatch.setattr(
        cr, "_resolve_backend_class",
        lambda prof, key, reg: cls if key == "asr_backend" else None,
    )
    return cr.resolve(profile=profile)


def test_asr_max_slots_reaches_the_session_ceiling(monkeypatch):
    """The on-device symptom: effective_limit stayed 1 whatever the profile said."""
    r = _resolve_with(monkeypatch, {
        "asr_backend": "jetson.sensevoice_trt",
        "asr_max_slots": 4,
    })
    assert r.session_ceiling == 4
    assert r.asr_cap.max_concurrent == 4
    # Admission rises, execution stays serialized — the deliberate pairing.
    assert r.asr_cap.supports_parallel is False


def test_env_override_reaches_the_session_ceiling(monkeypatch):
    monkeypatch.setenv("SENSEVOICE_MAX_CONCURRENT", "3")
    r = _resolve_with(monkeypatch, {"asr_backend": "jetson.sensevoice_trt"})
    assert r.session_ceiling == 3


def test_default_is_one_slot(monkeypatch):
    r = _resolve_with(monkeypatch, {"asr_backend": "jetson.sensevoice_trt"})
    assert r.session_ceiling == 1


def test_builder_alone_was_never_the_broken_part():
    """`build_sensevoice_trt_config` was always correct — the resolver wasn't."""
    cfg = vbc.build_sensevoice_trt_config(profile={"asr_max_slots": 4}, env={})
    assert cfg.max_concurrent == 4
