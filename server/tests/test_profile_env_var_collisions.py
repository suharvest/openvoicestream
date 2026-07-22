"""Guard against a profile clobbering a directory env var with a file path.

engine_resolver injects every ``required_engines`` entry's ``env_var`` with
that entry's ``engine_path`` -- a path to one file (engine_resolver module
docstring, "Backends read engine paths from env vars at import time"). Two
profiles used to point five entries at ``MOSS_ENGINE_DIR`` and one at
``MOSS_CODEC_ONNX_DIR``. Those are the names the MOSS backend reads as
DIRECTORIES (voxedge_backend_config._moss_config), and the profiles also set
them correctly in their ``env`` block -- so the last entry to be injected
silently replaced the directory with ``.../moss_tts_local_fixed_sampled_frame.plan``
and the worker got ``--engine-dir=<a .plan file>``.

Two independent rules fall out of that, and both are checked here because
either one alone would have missed the bug.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

PROFILE_DIR = Path(__file__).resolve().parents[2] / "configs" / "profiles"
PROFILES = sorted(PROFILE_DIR.glob("*.json"))


def _entries(profile_path: Path) -> list[dict]:
    return json.loads(profile_path.read_text()).get("required_engines") or []


def _env_block(profile_path: Path) -> dict:
    return json.loads(profile_path.read_text()).get("env") or {}


assert PROFILES, f"no profiles found under {PROFILE_DIR}"


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.stem)
def test_env_var_is_unique_within_a_profile(profile: Path) -> None:
    """Two entries sharing an env_var means one of them silently wins."""
    names = [e["env_var"] for e in _entries(profile) if e.get("env_var")]
    dupes = [name for name, n in Counter(names).items() if n > 1]
    assert not dupes, (
        f"{profile.name}: {dupes} used by more than one required_engines entry; "
        f"engine_resolver injects each entry in order, so only the last path survives"
    )


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.stem)
def test_entry_env_var_does_not_overwrite_a_different_env_block_value(profile: Path) -> None:
    """An entry must not replace an env-block value with a *different* path.

    Sharing a name is fine when both sides say the same thing -- several kokoro
    and matcha profiles list an engine's own path in the env block as a default
    and the resolver just re-injects it. The bug is when the two disagree: the
    env block holds a directory the backend reads, and the entry silently
    replaces it with one file inside that directory.
    """
    env = _env_block(profile)
    conflicts = [
        (e["env_var"], env[e["env_var"]], e.get("engine_path"))
        for e in _entries(profile)
        if e.get("env_var") in env and env[e["env_var"]] != e.get("engine_path")
    ]
    assert not conflicts, (
        f"{profile.name}: {[c[0] for c in conflicts]} are set in the env block "
        f"but a required_engines entry resolves them to a different path, "
        f"replacing what the backend reads: {conflicts}"
    )
