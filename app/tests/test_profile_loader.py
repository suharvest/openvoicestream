"""Tests for app.core.profile_loader hot-reload semantics (PR1)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from app.core import profile_loader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_module_state(monkeypatch):
    """Reset profile_loader module-level state between tests."""
    # Start each test with empty operator set + empty applied set + no profile.
    monkeypatch.setattr(profile_loader, "_OPERATOR_KEYS", frozenset())
    monkeypatch.setattr(profile_loader, "_APPLIED_KEYS", set())
    monkeypatch.setattr(profile_loader, "_CURRENT_PROFILE", {})
    yield


def _write_profile(tmp_path: Path, name: str, body: dict) -> Path:
    body = {"name": name, **body}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_first_apply_writes_env_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("FOO_BAR", raising=False)
    p = _write_profile(tmp_path, "pA", {"env": {"FOO_BAR": "x"}})

    profile = profile_loader.apply_profile(str(p))

    assert profile["name"] == "pA"
    import os
    assert os.environ["FOO_BAR"] == "x"
    assert "FOO_BAR" in profile_loader.get_applied_keys()
    assert "OVS_PROFILE_NAME" in profile_loader.get_applied_keys()


def test_second_apply_overwrites_previous_profile_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("A_KEY", raising=False)
    import os

    a = _write_profile(tmp_path, "A", {"env": {"A_KEY": "1"}})
    profile_loader.apply_profile(str(a))
    assert os.environ["A_KEY"] == "1"

    b = _write_profile(tmp_path, "B", {"env": {"A_KEY": "2"}})
    profile_loader.apply_profile(str(b))
    assert os.environ["A_KEY"] == "2"  # bug #1: previously stuck at "1"


def test_second_apply_clears_keys_only_in_old_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("OVS_X", raising=False)
    monkeypatch.delenv("OVS_Y", raising=False)
    import os

    a = _write_profile(tmp_path, "A", {"env": {"OVS_X": "1"}})
    profile_loader.apply_profile(str(a))
    assert os.environ["OVS_X"] == "1"

    b = _write_profile(tmp_path, "B", {"env": {"OVS_Y": "2"}})
    profile_loader.apply_profile(str(b))
    assert "OVS_X" not in os.environ  # bug #5: stale key cleared
    assert os.environ["OVS_Y"] == "2"


def test_operator_env_never_overwritten(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("OVS_OPERATOR_TEST", "operator-value")
    monkeypatch.setattr(
        profile_loader, "_OPERATOR_KEYS", frozenset({"OVS_OPERATOR_TEST"})
    )

    p = _write_profile(
        tmp_path, "P", {"env": {"OVS_OPERATOR_TEST": "profile-value"}}
    )
    profile_loader.apply_profile(str(p))

    assert os.environ["OVS_OPERATOR_TEST"] == "operator-value"
    assert "OVS_OPERATOR_TEST" not in profile_loader.get_applied_keys()


def test_snapshot_operator_keys_excludes_empty_values(monkeypatch):
    """docker-compose passes declared-but-unset vars as empty strings,
    not unset; these must not be treated as operator-owned (otherwise
    profile defaults silently fail to apply — orin-nx regression
    2026-05-20 with QWEN3_ARTIFACT_MANIFEST="")."""
    monkeypatch.setenv("QWEN3_ARTIFACT_MANIFEST", "")
    monkeypatch.setenv("QWEN3_ARTIFACT_SET", "")
    monkeypatch.setenv("QWEN3_ARTIFACT_ROOT", "/opt/models/qwen3-edgellm")

    snapshot = profile_loader._snapshot_operator_keys()

    assert "QWEN3_ARTIFACT_MANIFEST" not in snapshot
    assert "QWEN3_ARTIFACT_SET" not in snapshot
    assert "QWEN3_ARTIFACT_ROOT" in snapshot


def test_operator_env_not_cleared_on_reapply(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("OVS_OPERATOR_TEST", "operator-value")
    monkeypatch.setattr(
        profile_loader, "_OPERATOR_KEYS", frozenset({"OVS_OPERATOR_TEST"})
    )

    a = _write_profile(
        tmp_path, "A", {"env": {"OVS_OPERATOR_TEST": "p1", "OTHER": "o1"}}
    )
    profile_loader.apply_profile(str(a))
    assert os.environ["OVS_OPERATOR_TEST"] == "operator-value"

    b = _write_profile(tmp_path, "B", {"env": {"OTHER": "o2"}})
    profile_loader.apply_profile(str(b))
    assert os.environ["OVS_OPERATOR_TEST"] == "operator-value"


def test_tts_model_id_recomputed_on_reload(tmp_path, monkeypatch):
    monkeypatch.delenv("OVS_TTS_MODEL_ID", raising=False)
    import os

    a = _write_profile(tmp_path, "A", {"tts_model_id": "kokoro-en", "env": {}})
    profile_loader.apply_profile(str(a))
    assert os.environ["OVS_TTS_MODEL_ID"] == "kokoro-en"

    b = _write_profile(tmp_path, "B", {"tts_model_id": "matcha-zh", "env": {}})
    profile_loader.apply_profile(str(b))
    assert os.environ["OVS_TTS_MODEL_ID"] == "matcha-zh"  # bug #3


def test_apply_profile_with_explicit_ref_param(tmp_path, monkeypatch):
    """Explicit profile_ref bypasses env resolution (bug #4 fix)."""
    monkeypatch.delenv("OVS_PROFILE", raising=False)
    monkeypatch.delenv("OVS_PROFILE_JSON", raising=False)
    monkeypatch.delenv("LANGUAGE_MODE", raising=False)

    p = _write_profile(tmp_path, "explicit", {"env": {"K": "v"}})

    profile = profile_loader.apply_profile(str(p))
    assert profile["name"] == "explicit"
    import os
    assert os.environ["K"] == "v"


def test_concurrent_apply_thread_safe(tmp_path, monkeypatch):
    import os

    profiles = []
    for i in range(4):
        p = _write_profile(
            tmp_path, f"P{i}", {"env": {f"KEY_{i}": f"v{i}"}}
        )
        profiles.append(str(p))

    errors: list[BaseException] = []

    def worker(path: str) -> None:
        try:
            for _ in range(20):
                profile_loader.apply_profile(path)
        except BaseException as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(p,)) for p in profiles]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []

    # Final state must be self-consistent: the current profile's name must
    # match one of the inputs, and _APPLIED_KEYS must reflect that profile
    # (i.e. exactly one KEY_i is present in env among the four).
    current = profile_loader.current_profile()
    assert current.get("name", "").startswith("P")
    final_idx = int(current["name"][1:])

    applied = profile_loader.get_applied_keys()
    assert f"KEY_{final_idx}" in applied
    assert os.environ.get(f"KEY_{final_idx}") == f"v{final_idx}"
    for i in range(4):
        if i == final_idx:
            continue
        assert f"KEY_{i}" not in os.environ, (
            f"stale KEY_{i} leaked; final profile was P{final_idx}"
        )


def test_apply_profile_from_env_still_works(tmp_path, monkeypatch):
    """apply_profile_from_env() honors OVS_PROFILE (compat path)."""
    import os

    monkeypatch.delenv("OVS_PROFILE_JSON", raising=False)
    monkeypatch.delenv("LANGUAGE_MODE", raising=False)
    monkeypatch.delenv("COMPAT_KEY", raising=False)

    p = _write_profile(tmp_path, "compat", {"env": {"COMPAT_KEY": "ok"}})
    # OVS_PROFILE resolves via _profile_path which expects either a filename
    # under configs/profiles or an absolute path. Use absolute path here.
    monkeypatch.setenv("OVS_PROFILE", str(p))

    profile = profile_loader.apply_profile_from_env()
    assert profile["name"] == "compat"
    assert os.environ["COMPAT_KEY"] == "ok"


# ---------------------------------------------------------------------------
# Artifact pre-flight helpers
# ---------------------------------------------------------------------------

def test_expected_artifact_paths_heuristic(monkeypatch):
    """Only path-like suffixes + absolute expanded values are reported."""
    monkeypatch.delenv("SOMEVAR", raising=False)
    profile = {
        "name": "p",
        "env": {
            "FOO_ENGINE": "/a/b.engine",
            "BAR_DIR": "/c/d",
            "BAZ_VALUE": "/should/be/skipped",
            "REL_PATH": "deploy/x",
            "ABS_PATH": "/p/q",
            "OTHER_JSON": "/m/n.json",
            "NON_STR_PATH": 42,
        },
    }
    out = profile_loader.expected_artifact_paths(profile)
    assert out == {
        "FOO_ENGINE": "/a/b.engine",
        "BAR_DIR": "/c/d",
        "ABS_PATH": "/p/q",
        "OTHER_JSON": "/m/n.json",
    }


def test_expected_artifact_paths_expands_vars(monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", "/opt/models")
    profile = {"env": {"X_DIR": "$ARTIFACT_ROOT/foo"}}
    out = profile_loader.expected_artifact_paths(profile)
    assert out == {"X_DIR": "/opt/models/foo"}


def test_find_missing_artifacts_all_present(tmp_path):
    f = tmp_path / "model.engine"
    f.write_text("blob", encoding="utf-8")
    d = tmp_path / "subdir"
    d.mkdir()
    profile = {
        "env": {
            "MY_ENGINE": str(f),
            "MY_DIR": str(d),
        },
    }
    assert profile_loader.find_missing_artifacts(profile) == []


def test_find_missing_artifacts_some_missing(tmp_path):
    f = tmp_path / "exists.engine"
    f.write_text("blob", encoding="utf-8")
    profile = {
        "env": {
            "MY_ENGINE": str(f),
            "MISSING_DIR": "/definitely/not/here/xyz",
            "ALSO_MISSING_PATH": "/nope/zzz",
        },
    }
    missing = profile_loader.find_missing_artifacts(profile)
    keys = {m["env_var"] for m in missing}
    assert keys == {"MISSING_DIR", "ALSO_MISSING_PATH"}
    for m in missing:
        assert m["path"].startswith("/")


def test_expected_artifact_paths_two_pass_expansion(monkeypatch):
    """Profile-self-referential ${VAR} must resolve against the profile's
    own env block, not whatever value (if any) is in the current process env."""
    # Crucially, do NOT set QWEN3_ARTIFACT_ROOT in os.environ — the profile
    # defines it itself; the second key references it.
    monkeypatch.delenv("QWEN3_ARTIFACT_ROOT", raising=False)
    profile = {
        "env": {
            "QWEN3_ARTIFACT_ROOT": "/opt/X",
            "EDGE_LLM_ASR_ENGINE_DIR": "${QWEN3_ARTIFACT_ROOT}/engines/orin-nano/asr",
        }
    }
    out = profile_loader.expected_artifact_paths(profile)
    assert out["QWEN3_ARTIFACT_ROOT"] == "/opt/X"
    assert out["EDGE_LLM_ASR_ENGINE_DIR"] == "/opt/X/engines/orin-nano/asr"


def test_expected_artifact_paths_profile_overrides_env(monkeypatch):
    """If both os.environ and the profile define the same key, the profile
    value wins (otherwise dry-run validates the wrong paths)."""
    monkeypatch.setenv("QWEN3_ARTIFACT_ROOT", "/wrong/from/env")
    profile = {
        "env": {
            "QWEN3_ARTIFACT_ROOT": "/right/from/profile",
            "X_DIR": "${QWEN3_ARTIFACT_ROOT}/sub",
        }
    }
    out = profile_loader.expected_artifact_paths(profile)
    assert out["X_DIR"] == "/right/from/profile/sub"


def test_expected_artifact_paths_new_suffixes(monkeypatch):
    """New path-suffix heuristics added 2026-05-21: _ONNX, _ROOT, _BASE,
    _VOICES, _TOKENS, _LONG."""
    monkeypatch.delenv("ANYTHING_ROOT", raising=False)
    profile = {
        "env": {
            "FOO_ONNX": "/m/foo.onnx",
            "BAR_ROOT": "/r/bar",
            "BAZ_BASE": "/b/baz",
            "VOX_VOICES": "/v/voices.bin",
            "TOK_TOKENS": "/t/tokens.txt",
            "ENG_LONG": "/e/engine_long.engine",
            # Non-path _TOKENS value (scalar) should be filtered by startswith("/")
            "SEG_TOKENS": "64",
        }
    }
    out = profile_loader.expected_artifact_paths(profile)
    assert out == {
        "FOO_ONNX": "/m/foo.onnx",
        "BAR_ROOT": "/r/bar",
        "BAZ_BASE": "/b/baz",
        "VOX_VOICES": "/v/voices.bin",
        "TOK_TOKENS": "/t/tokens.txt",
        "ENG_LONG": "/e/engine_long.engine",
    }


def test_json_blob_not_treated_as_path(monkeypatch):
    """_JSON suffix matches both a path and a JSON blob; the startswith("/")
    filter is the safety net that keeps blobs out of the artifact list."""
    profile = {
        "env": {
            "OVS_TTS_SPEAKERS_JSON": '{"0":"","2301":"2301"}',
            "MY_CONFIG_JSON": "/etc/cfg.json",
        }
    }
    out = profile_loader.expected_artifact_paths(profile)
    assert "OVS_TTS_SPEAKERS_JSON" not in out
    assert out["MY_CONFIG_JSON"] == "/etc/cfg.json"


def test_apply_profile_two_pass_expansion_writes_env(tmp_path, monkeypatch):
    """apply_profile must write the fully-resolved paths into os.environ,
    using the profile's own env block to resolve self-references."""
    import os

    monkeypatch.delenv("QWEN3_ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("EDGE_LLM_ASR_ENGINE_DIR", raising=False)

    p = _write_profile(tmp_path, "P", {
        "env": {
            "QWEN3_ARTIFACT_ROOT": "/opt/X",
            "EDGE_LLM_ASR_ENGINE_DIR": "${QWEN3_ARTIFACT_ROOT}/engines/asr",
        }
    })
    profile_loader.apply_profile(str(p))
    assert os.environ["QWEN3_ARTIFACT_ROOT"] == "/opt/X"
    assert os.environ["EDGE_LLM_ASR_ENGINE_DIR"] == "/opt/X/engines/asr"


def test_critical_key_conflict_raises(tmp_path, monkeypatch):
    """LANGUAGE_MODE pre-set to a different value than the profile must
    hard-fail (root cause of orin-nano Qwen3 silent-skip bug 2026-05-25)."""
    monkeypatch.setenv("LANGUAGE_MODE", "zh_en")
    monkeypatch.setattr(
        profile_loader, "_OPERATOR_KEYS", frozenset({"LANGUAGE_MODE"})
    )

    p = _write_profile(
        tmp_path, "needs-multilang",
        {"env": {"LANGUAGE_MODE": "multilanguage"}},
    )
    with pytest.raises(RuntimeError, match="Remove LANGUAGE_MODE"):
        profile_loader.apply_profile(str(p))


def test_critical_key_matching_value_does_not_raise(tmp_path, monkeypatch):
    """If the operator env matches the profile-declared value, no conflict."""
    import os

    monkeypatch.setenv("LANGUAGE_MODE", "multilanguage")
    monkeypatch.setattr(
        profile_loader, "_OPERATOR_KEYS", frozenset({"LANGUAGE_MODE"})
    )

    p = _write_profile(
        tmp_path, "agrees",
        {"env": {"LANGUAGE_MODE": "multilanguage"}},
    )
    # Must not raise.
    profile = profile_loader.apply_profile(str(p))
    assert profile["name"] == "agrees"
    assert os.environ["LANGUAGE_MODE"] == "multilanguage"


def test_apply_profile_returns_empty_when_no_ref(monkeypatch):
    """No env hints + no explicit ref → returns {} without touching state."""
    for k in ("OVS_PROFILE_JSON", "OVS_PROFILE", "OVS_PROFILE_DEFAULT",
              "LANGUAGE_MODE", "OVS_PRESET"):
        monkeypatch.delenv(k, raising=False)

    result = profile_loader.apply_profile()
    assert result == {}
    assert profile_loader.current_profile() == {}


def test_resolve_engines_injects_and_reconciles_engine_keys(tmp_path, monkeypatch):
    """B1: resolve_engines=True injects engine env keys AND folds them into
    _APPLIED_KEYS so the next apply_profile's stale-clear rotates them — no
    engine-env pollution across reloads/rollbacks. resolve_engines=False does
    not inject new keys but still clears the previous profile's engine keys."""
    import os
    from app.core import engine_resolver

    monkeypatch.delenv("ENGINE_A", raising=False)
    monkeypatch.delenv("ENGINE_B", raising=False)

    def fake_resolve_all(profile):
        injected = {}
        for e in profile.get("required_engines", []):
            os.environ[e["env_var"]] = e["path"]
            injected[e["env_var"]] = e["path"]
        return injected

    monkeypatch.setattr(engine_resolver, "resolve_all", fake_resolve_all)

    a = _write_profile(
        tmp_path, "A", {"required_engines": [{"env_var": "ENGINE_A", "path": "/eng/a"}]}
    )
    profile_loader.apply_profile(str(a), resolve_engines=True)
    assert os.environ.get("ENGINE_A") == "/eng/a"
    assert "ENGINE_A" in profile_loader._APPLIED_KEYS

    # Reload to profile B: ENGINE_A is stale -> cleared, ENGINE_B injected.
    b = _write_profile(
        tmp_path, "B", {"required_engines": [{"env_var": "ENGINE_B", "path": "/eng/b"}]}
    )
    profile_loader.apply_profile(str(b), resolve_engines=True)
    assert os.environ.get("ENGINE_B") == "/eng/b"
    assert "ENGINE_A" not in os.environ                       # no pollution
    assert "ENGINE_B" in profile_loader._APPLIED_KEYS
    assert "ENGINE_A" not in profile_loader._APPLIED_KEYS

    # resolve_engines=False: no new injection, but B's engine key is still
    # cleared by the stale-clear (it's tracked in _APPLIED_KEYS).
    c = _write_profile(tmp_path, "C", {"env": {"PLAIN": "1"}})
    profile_loader.apply_profile(str(c), resolve_engines=False)
    assert "ENGINE_B" not in os.environ
