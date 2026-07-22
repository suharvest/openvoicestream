"""Keep profile ``engine_path`` in step with where the manifest puts the file.

An engine's on-disk location is declared twice: the artifact manifest says
where provisioning writes it, and the profile's ``required_engines`` entry says
where ``engine_resolver`` looks for it. Nothing ties the two together, so the
codec plan spent a release pointing at ``codec_onnx/`` in the profile while the
manifest had been moved to ``engines/``. The resolver found nothing, fell back
to an HF manifest for a model id that has never been published, and aborted
startup:

    ✗ codec_decode_step.plan [F1]: no published artifact manifest for model
      'moss-audio-tokenizer-nano'
    EngineResolutionError: 1 required engine(s) could not be provisioned.

The development Jetson had symlinks bridging both halves, so this only ever
appeared on a clean deployment -- and it survived one round of fixing precisely
because the manifest was corrected and the profile was not.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO_ROOT / "configs" / "profiles"
MOSS_MANIFEST = REPO_ROOT / "deploy" / "artifacts" / "moss_manifest.json"

MOSS_MODEL_ROOT = "/opt/models/moss-tts-nano"

PROFILES = sorted(PROFILE_DIR.glob("*.json"))
assert PROFILES, f"no profiles under {PROFILE_DIR}"


def _manifest_destinations() -> set[str]:
    """Absolute paths the MOSS manifest provisions under model_root."""
    manifest = json.loads(MOSS_MANIFEST.read_text())
    root = (manifest.get("targets") or {}).get("model_root", MOSS_MODEL_ROOT)
    return {
        f"{root}/{entry['path'].lstrip('/')}"
        for entry in manifest["files"]
        if entry.get("dest", "model_root") == "model_root"
    }


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.stem)
def test_moss_engine_paths_are_provisioned_by_the_manifest(profile: Path) -> None:
    """Every MOSS engine a profile requires must be one the manifest writes.

    Only entries under the MOSS model root are checked -- other backends have
    their own provisioning and are out of scope here.
    """
    provisioned = _manifest_destinations()
    entries = json.loads(profile.read_text()).get("required_engines") or []

    orphans = [
        f"{e.get('model_id')}: {e['engine_path']}"
        for e in entries
        if e.get("engine_path", "").startswith(MOSS_MODEL_ROOT + "/")
        and e["engine_path"] not in provisioned
    ]

    assert not orphans, (
        f"{profile.name} requires MOSS engines the manifest never writes there. "
        f"engine_resolver will miss them, fall back to a per-model HF manifest "
        f"that does not exist, and abort startup:\n  " + "\n  ".join(orphans)
    )
