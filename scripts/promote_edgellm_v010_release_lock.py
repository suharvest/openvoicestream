#!/usr/bin/env python3
"""Promote the v0.10 candidate lock through build-ready and published phases."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from check_edgellm_v010_release_lock import validate


GIT_SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
IMAGE_KEYS = {"speech", "llm"}
IMAGE_IDENTITY_FIELDS = {"ref", "image_id", "registry_digest", "size_bytes"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _validate_value(
    value: dict[str, Any],
    repo_root: Path,
    *,
    require_image_build_ready: bool = False,
    require_published: bool = False,
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="v010-release-lock.",
            suffix=".json",
            delete=False,
        ) as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temporary_path = Path(stream.name)
        validate(
            temporary_path,
            repo_root,
            require_published,
            require_image_build_ready=require_image_build_ready,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _package_key(model_id: str, target_id: str) -> str:
    return f"{model_id}/{target_id}"


def build_ready(
    candidate_path: Path,
    plan_path: Path,
    output_path: Path,
    repo_root: Path,
    artifact_set: str,
    service_revision: str,
) -> dict[str, Any]:
    lock = _read(candidate_path)
    plan = _read(plan_path)
    _require(lock.get("release_state") == "candidate_unpublished", "input is not candidate")
    _require(lock.get("deployable") is False, "candidate input must be non-deployable")
    _require(
        plan.get("artifact_set") == lock.get("artifact_set"),
        "publication plan does not belong to the candidate lock",
    )
    _require(plan.get("external_upload_authorized") is True, "external upload is not authorized")
    _require("candidate" not in artifact_set and artifact_set, "final artifact_set is invalid")
    _require(bool(GIT_SHA.fullmatch(service_revision)), "service revision is not immutable")
    _require(
        all(target.get("qualification_status") == "passed" for target in lock["targets"].values()),
        "all release targets must be qualified before image build",
    )

    packages = plan.get("packages") or {}
    expected_package_keys: set[str] = set()
    for target_id, target in lock["targets"].items():
        for lane in target["supported_lanes"]:
            model_id = lane.split("/", 1)[1]
            artifact = lock["model_artifacts"][model_id]
            variants = artifact.get("target_variants")
            identity = variants[target_id] if variants is not None else artifact
            if identity.get("status") == "published_retained_v091":
                continue
            key = _package_key(model_id, target_id)
            expected_package_keys.add(key)
            package = packages.get(key) or {}
            _require(package.get("repo") == artifact.get("repo"), f"repo drift: {key}")
            _require(package.get("status") == "published", f"package is not published: {key}")
            _require(
                bool(GIT_SHA.fullmatch(str(package.get("published_revision", "")))),
                f"package revision is not immutable: {key}",
            )
            _require(
                package.get("payload_sha256") == identity.get("payload_sha256")
                and package.get("payload_size") == identity.get("payload_size"),
                f"package payload differs from candidate lock: {key}",
            )
            _require(
                bool(SHA256.fullmatch(str(package.get("manifest_sha256", ""))))
                and bool(SHA256.fullmatch(str(package.get("sums_sha256", "")))),
                f"package metadata hashes are incomplete: {key}",
            )
            _require(
                str(package.get("staging_uri", "")).startswith("fleet://spark/"),
                f"verified staging URI is absent: {key}",
            )
            identity.update(
                revision=package["published_revision"],
                status="published",
            )

    _require(set(packages) == expected_package_keys, "publication plan package coverage drift")
    for key, artifact in lock["runtime_artifacts"].items():
        _require(
            artifact.get("status") == "candidate_staged_qualified",
            f"runtime artifact is not candidate-qualified: {key}",
        )
        artifact["status"] = "qualified_for_image"
    for image in lock["runtime_images"].values():
        image.update(
            ref=None,
            image_id=None,
            registry_digest=None,
            size_bytes=None,
            service_revision=service_revision,
            status="build_pending",
        )
    lock.update(
        artifact_set=artifact_set,
        release_state="artifacts_qualified_image_pending",
        deployable=False,
    )
    lock["source"]["status"] = "published"
    lock.pop("candidate_notice", None)
    lock["release_notice"] = (
        "All source, model, runtime, and target inputs are qualified. This embedded "
        "pre-image lock is non-deployable until external image identities are frozen."
    )

    _validate_value(lock, repo_root, require_image_build_ready=True)
    _write_new(output_path, lock)
    return lock


def published(
    build_ready_path: Path,
    images_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    lock = _read(build_ready_path)
    images = _read(images_path)
    validate(
        build_ready_path,
        repo_root,
        False,
        require_image_build_ready=True,
    )
    _require(set(images) == IMAGE_KEYS, "image identity file must contain speech and llm")
    for image_key, identity in images.items():
        _require(isinstance(identity, dict), f"invalid image identity: {image_key}")
        _require(
            set(identity) == IMAGE_IDENTITY_FIELDS,
            f"image identity has unexpected fields: {image_key}",
        )
        lock["runtime_images"][image_key].update(identity, status="published")
    for artifact in lock["runtime_artifacts"].values():
        artifact["status"] = "published_in_image"
    lock.update(release_state="published_and_qualified", deployable=True)
    lock["release_notice"] = (
        "Published and qualified TensorRT-Edge-LLM v0.10 release. The v0.9.1 "
        "release lock remains the explicit rollback identity."
    )
    _validate_value(lock, repo_root, require_published=True)
    _write_new(output_path, lock)
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    # Some supported environments expose the PyPI argparse backport ahead of
    # the stdlib module.  Its add_subparsers() does not accept required= even
    # though the returned action supports the attribute.
    subparsers = parser.add_subparsers(dest="phase")
    subparsers.required = True

    build = subparsers.add_parser("build-ready")
    build.add_argument("--candidate-lock", type=Path, required=True)
    build.add_argument("--publication-plan", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--artifact-set", required=True)
    build.add_argument("--service-revision", required=True)

    final = subparsers.add_parser("published")
    final.add_argument("--build-ready-lock", type=Path, required=True)
    final.add_argument("--image-identities", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--repo-root", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.phase == "build-ready":
            result = build_ready(
                args.candidate_lock.resolve(),
                args.publication_plan.resolve(),
                args.output.resolve(),
                args.repo_root.resolve(),
                args.artifact_set,
                args.service_revision,
            )
        else:
            result = published(
                args.build_ready_lock.resolve(),
                args.image_identities.resolve(),
                args.output.resolve(),
                args.repo_root.resolve(),
            )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"artifact_set": result["artifact_set"], "release_state": result["release_state"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
