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


def _run(
    lock: dict,
    tmp_path: Path,
    templates_dir: Path | None = None,
    expected_count: int | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    lock_path = tmp_path / "lock.json"
    output = tmp_path / "profiles"
    lock_path.write_text(json.dumps(lock))
    argv = [
        sys.executable,
        str(SCRIPT),
        "--lock",
        str(lock_path),
        "--templates-dir",
        str(templates_dir or TEMPLATES),
        "--output-dir",
        str(output),
    ]
    if expected_count is not None:
        argv += ["--expected-count", str(expected_count)]
    result = subprocess.run(
        argv,
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
        assert "target_allowlist" not in profile
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


def test_target_allowlist_restricts_a_template_to_one_target(tmp_path: Path) -> None:
    """A concurrency variant qualified only on the 16GB NX must not reach the Nano.

    The 3-lane profile needs an ASR engine that physically holds three lanes;
    publishing it for the 8GB Nano would ship a profile that silently degrades
    (qwen3_asr_worker clamps lanes to the engine batch). target_allowlist is
    how a template says "this target only", and the key must be consumed by
    the renderer rather than published.
    """
    templates = tmp_path / "templates"
    templates.mkdir()
    base = json.loads((TEMPLATES / "jetson-edgellm-v010-candidate-matcha.json").read_text())
    variant = json.loads(json.dumps(base))
    variant["name"] = "jetson-edgellm-v010-candidate-matcha-n3"
    variant["target_allowlist"] = ["orin-nx-16gb"]
    (templates / "jetson-edgellm-v010-candidate-matcha.json").write_text(json.dumps(base))
    (templates / "jetson-edgellm-v010-candidate-matcha-n3.json").write_text(json.dumps(variant))

    # matcha renders for both targets; the n3 variant only for the NX => 3
    result, output = _run(_published_lock(), tmp_path, templates_dir=templates, expected_count=3)
    assert result.returncode == 0, result.stderr
    names = sorted(path.name for path in output.glob("*.json"))
    assert names == [
        "jetson-edgellm-v010-orin-nano-matcha.json",
        "jetson-edgellm-v010-orin-nx-matcha-n3.json",
        "jetson-edgellm-v010-orin-nx-matcha.json",
    ]
    rendered = json.loads((output / "jetson-edgellm-v010-orin-nx-matcha-n3.json").read_text())
    assert "target_allowlist" not in rendered


def test_target_allowlist_rejects_unknown_target(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    base = json.loads((TEMPLATES / "jetson-edgellm-v010-candidate-matcha.json").read_text())
    base["target_allowlist"] = ["orin-agx-64gb"]
    (templates / "jetson-edgellm-v010-candidate-matcha.json").write_text(json.dumps(base))
    result, _ = _run(_published_lock(), tmp_path, templates_dir=templates, expected_count=1)
    assert result.returncode != 0
    assert "target_allowlist has unknown targets" in result.stderr
