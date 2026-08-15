from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "deploy/artifacts/v010-candidate-release-lock.json"
TEMPLATES = ROOT / "configs/profiles"
SCRIPT = ROOT / "scripts/render_edgellm_v010_production_profiles.py"


def _published_lock() -> dict:
    lock = json.loads(LOCK_PATH.read_text())
    lock["artifact_set"] = "orin-sm87-edgellm-v010-20260815-v1"
    lock["release_state"] = "artifacts_qualified_image_pending"
    lock["source"]["status"] = "published"
    for target in lock["targets"].values():
        target["qualification_status"] = "passed"
    for artifact in lock["model_artifacts"].values():
        identities = artifact.get("target_variants") or {"shared": artifact}
        for identity in identities.values():
            if identity["status"] != "published_retained_v091":
                identity.update(
                    revision="a" * 40,
                    payload_sha256="b" * 64,
                    payload_size=123,
                    status="published",
                )
    return lock


def _run(lock: dict, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    lock_path = tmp_path / "lock.json"
    output = tmp_path / "profiles"
    lock_path.write_text(json.dumps(lock))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lock",
            str(lock_path),
            "--templates-dir",
            str(TEMPLATES),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, output


def test_renders_target_scoped_profiles_with_immutable_identities(tmp_path: Path) -> None:
    result, output = _run(_published_lock(), tmp_path)
    assert result.returncode == 0, result.stderr
    paths = sorted(output.glob("*.json"))
    assert len(paths) == 10
    assert len([path for path in paths if "orin-nx" in path.name]) == 6
    assert len([path for path in paths if "orin-nano" in path.name]) == 4
    assert not list(output.glob("*orin-nano-moss.json"))
    assert not list(output.glob("*orin-nano-sparktts.json"))
    for path in paths:
        profile = json.loads(path.read_text())
        serialized = json.dumps(profile)
        assert profile["name"] == path.stem
        assert profile["deployment_scope"] == "production-published"
        assert profile["artifact_set"] == "orin-sm87-edgellm-v010-20260815-v1"
        assert "candidate" not in serialized
        assert "/opt/models-v010-candidate/" not in serialized
        for artifact in profile["model_artifacts"]:
            assert len(artifact["revision"]) == 40
            assert len(artifact["payload_sha256"]) == 64
            assert artifact["payload_size"] > 0


def test_rejects_candidate_lock(tmp_path: Path) -> None:
    lock = _published_lock()
    lock["release_state"] = "candidate_unpublished"
    result, _ = _run(lock, tmp_path)
    assert result.returncode != 0
    assert "build-ready or published" in result.stderr


def test_rejects_unqualified_target(tmp_path: Path) -> None:
    lock = _published_lock()
    lock["targets"]["orin-nano"]["qualification_status"] = "pending"
    result, _ = _run(lock, tmp_path)
    assert result.returncode != 0
    assert "orin-nano is not qualified" in result.stderr


def test_rejects_unpublished_target_variant(tmp_path: Path) -> None:
    lock = _published_lock()
    lock["model_artifacts"]["qwen3-asr-0.6b"]["target_variants"]["orin-nano"].update(
        revision=None,
        status="candidate_packaged_not_published",
    )
    result, _ = _run(lock, tmp_path)
    assert result.returncode != 0
    assert "qwen3-asr-0.6b.orin-nano is not published" in result.stderr
