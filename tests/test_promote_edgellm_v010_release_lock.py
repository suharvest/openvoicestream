from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/promote_edgellm_v010_release_lock.py"
CANDIDATE = ROOT / "deploy/artifacts/v010-candidate-release-lock.json"
PLAN = ROOT / "deploy/artifacts/v010-publication-plan.json"
SERVICE_REVISION = "a" * 40
FINAL_ARTIFACT_SET = "orin-sm87-edgellm-v010-20260815-v1"


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    lock = json.loads(CANDIDATE.read_text())
    plan = json.loads(PLAN.read_text())
    plan["external_upload_authorized"] = True
    for target in lock["targets"].values():
        target["qualification_status"] = "passed"
    for index, (key, package) in enumerate(plan["packages"].items()):
        model_id, target_id = key.rsplit("/", 1)
        artifact = lock["model_artifacts"][model_id]
        identity = (
            artifact["target_variants"][target_id]
            if "target_variants" in artifact
            else artifact
        )
        if identity.get("payload_sha256") is None:
            identity["payload_sha256"] = f"{index + 1:x}" * 64
            identity["payload_size"] = index + 1
        package.update(
            staging_uri=f"fleet://spark/verified/package-{index}",
            payload_sha256=identity["payload_sha256"],
            payload_size=identity["payload_size"],
            manifest_sha256="c" * 64,
            sums_sha256="d" * 64,
            published_revision=f"{index + 1:x}" * 40,
            status="published",
        )
    lock_path = tmp_path / "candidate.json"
    plan_path = tmp_path / "plan.json"
    lock_path.write_text(json.dumps(lock))
    plan_path.write_text(json.dumps(plan))
    return lock_path, plan_path


def _build_ready(
    tmp_path: Path,
    lock_path: Path,
    plan_path: Path,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    output = tmp_path / "build-ready.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "build-ready",
            "--candidate-lock",
            str(lock_path),
            "--publication-plan",
            str(plan_path),
            "--output",
            str(output),
            "--repo-root",
            str(ROOT),
            "--artifact-set",
            FINAL_ARTIFACT_SET,
            "--service-revision",
            SERVICE_REVISION,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, output


def test_promotes_complete_publication_plan_to_build_ready_lock(tmp_path: Path) -> None:
    lock_path, plan_path = _inputs(tmp_path)
    result, output = _build_ready(tmp_path, lock_path, plan_path)
    assert result.returncode == 0, result.stderr
    lock = json.loads(output.read_text())
    assert lock["artifact_set"] == FINAL_ARTIFACT_SET
    assert lock["release_state"] == "artifacts_qualified_image_pending"
    assert lock["deployable"] is False
    assert lock["source"]["status"] == "published"
    assert all(
        artifact["status"] == "qualified_for_image"
        for artifact in lock["runtime_artifacts"].values()
    )
    assert all(
        image["status"] == "build_pending"
        and image["service_revision"] == SERVICE_REVISION
        and image["ref"] is None
        for image in lock["runtime_images"].values()
    )


def test_rejects_plan_without_immediate_upload_authorization(tmp_path: Path) -> None:
    lock_path, plan_path = _inputs(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["external_upload_authorized"] = False
    plan_path.write_text(json.dumps(plan))
    result, output = _build_ready(tmp_path, lock_path, plan_path)
    assert result.returncode != 0
    assert "external upload is not authorized" in result.stderr
    assert not output.exists()


def test_rejects_payload_identity_drift_without_writing_output(tmp_path: Path) -> None:
    lock_path, plan_path = _inputs(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["packages"]["qwen3-asr-0.6b/orin-nano"]["payload_size"] += 1
    plan_path.write_text(json.dumps(plan))
    result, output = _build_ready(tmp_path, lock_path, plan_path)
    assert result.returncode != 0
    assert "package payload differs" in result.stderr
    assert not output.exists()


def test_promotes_build_ready_lock_to_digest_pinned_final_lock(tmp_path: Path) -> None:
    lock_path, plan_path = _inputs(tmp_path)
    build_result, build_output = _build_ready(tmp_path, lock_path, plan_path)
    assert build_result.returncode == 0, build_result.stderr
    image_identities = {}
    for index, image_key in enumerate(("speech", "llm"), start=1):
        digest = f"{index:x}" * 64
        image_identities[image_key] = {
            "ref": f"registry.example/v010/{image_key}@sha256:{digest}",
            "image_id": "sha256:" + "e" * 64,
            "registry_digest": "sha256:" + digest,
            "size_bytes": index * 1024,
        }
    images_path = tmp_path / "images.json"
    images_path.write_text(json.dumps(image_identities))
    output = tmp_path / "published.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "published",
            "--build-ready-lock",
            str(build_output),
            "--image-identities",
            str(images_path),
            "--output",
            str(output),
            "--repo-root",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lock = json.loads(output.read_text())
    assert lock["release_state"] == "published_and_qualified"
    assert lock["deployable"] is True
    assert all(
        artifact["status"] == "published_in_image"
        for artifact in lock["runtime_artifacts"].values()
    )
    assert all(
        image["status"] == "published" and "@sha256:" in image["ref"]
        for image in lock["runtime_images"].values()
    )


def test_final_promotion_rejects_image_identity_field_injection(tmp_path: Path) -> None:
    lock_path, plan_path = _inputs(tmp_path)
    build_result, build_output = _build_ready(tmp_path, lock_path, plan_path)
    assert build_result.returncode == 0, build_result.stderr
    identities = {}
    for index, image_key in enumerate(("speech", "llm"), start=1):
        digest = f"{index:x}" * 64
        identities[image_key] = {
            "ref": f"registry.example/v010/{image_key}@sha256:{digest}",
            "image_id": "sha256:" + "e" * 64,
            "registry_digest": "sha256:" + digest,
            "size_bytes": 1024,
        }
    identities["speech"]["service_revision"] = "f" * 40
    images_path = tmp_path / "injected-images.json"
    images_path.write_text(json.dumps(identities))
    output = tmp_path / "must-not-exist.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "published",
            "--build-ready-lock",
            str(build_output),
            "--image-identities",
            str(images_path),
            "--output",
            str(output),
            "--repo-root",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unexpected fields: speech" in result.stderr
    assert not output.exists()
