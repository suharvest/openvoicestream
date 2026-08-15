#!/usr/bin/env python3
"""Validate the additive TensorRT-Edge-LLM v0.10 release identity.

Candidate locks may contain explicit null publication fields.  Production
builds pass ``--require-published`` and fail closed until every immutable
identity is present.  The checker also binds the lock to the pinned submodule
and overlay manifests, preventing a release lock from silently outliving its
source contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(
    lock_path: Path,
    repo_root: Path,
    require_published: bool,
    *,
    check_gitlink: bool = True,
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(lock.get("schema_version") == 1, "schema_version must be 1")
    _require(lock.get("rollback_release_lock") == "deploy/artifacts/v091-release-lock.json", "v0.9.1 rollback lock must remain explicit")

    source = lock.get("source") or {}
    _require(bool(GIT_SHA.fullmatch(str(source.get("upstream_sha", "")))), "source.upstream_sha must be immutable")
    _require(bool(GIT_SHA.fullmatch(str(source.get("submodule_sha", "")))), "source.submodule_sha must be immutable")
    submodule = repo_root / str(source.get("submodule_path", ""))
    gitlink = (repo_root / ".gitmodules").is_file() and submodule.is_dir()
    if check_gitlink:
        _require(bool(gitlink), f"pinned submodule is absent: {submodule}")
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--stage", source["submodule_path"]],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = result.stdout.split()
        _require(len(fields) >= 2 and fields[0] == "160000", "source path is not a gitlink")
        _require(fields[1] == source["submodule_sha"], "release lock submodule_sha differs from the root gitlink")

    overlay = submodule / str(source.get("overlay", "")) / "manifests"
    contracts = source.get("manifest_contracts") or {}
    _require(bool(contracts), "source.manifest_contracts must not be empty")
    for name, expected in contracts.items():
        path = overlay / name
        _require(path.is_file(), f"manifest contract is absent: {path}")
        _require(bool(SHA256.fullmatch(str(expected))), f"invalid manifest sha256: {name}")
        _require(_sha256(path) == expected, f"manifest contract drift: {name}")

    baselines = lock.get("qualification_baselines") or {}
    _require(set(baselines) == {"orin-nx", "orin-nano"}, "qualification baselines must bind both Orin targets")
    for target, identity in baselines.items():
        path = repo_root / str(identity.get("path", ""))
        expected = str(identity.get("sha256", ""))
        _require(path.is_file(), f"qualification baseline is absent: {target}")
        _require(bool(SHA256.fullmatch(expected)), f"invalid qualification baseline sha256: {target}")
        _require(_sha256(path) == expected, f"qualification baseline drift: {target}")

    candidate = lock.get("release_state") == "candidate_unpublished"
    if candidate:
        _require(lock.get("deployable") is False, "candidate lock must be non-deployable")
    elif lock.get("release_state") == "published_and_qualified":
        _require(lock.get("deployable") is True, "published lock must be deployable")
    else:
        raise ValueError("unknown release_state")

    if require_published:
        _require(not candidate, "v0.10 release is still candidate_unpublished")
        images = lock.get("runtime_images") or {}
        _require(set(images) == {"speech", "llm"}, "runtime_images must bind speech and llm independently")
        for image_name, image in images.items():
            _require(bool(image.get("ref")), f"runtime_images.{image_name}.ref is unpublished")
            _require(
                bool(
                    SHA256.fullmatch(
                        str(image.get("registry_digest", "")).removeprefix("sha256:")
                    )
                ),
                f"runtime_images.{image_name}.registry_digest is unpublished",
            )
            _require(
                isinstance(image.get("size_bytes"), int)
                and image["size_bytes"] > 0,
                f"runtime_images.{image_name}.size_bytes is unpublished",
            )

        for model_id, artifact in (lock.get("model_artifacts") or {}).items():
            _require(bool(GIT_SHA.fullmatch(str(artifact.get("revision", "")))), f"{model_id}.revision is unpublished")
            _require(bool(SHA256.fullmatch(str(artifact.get("payload_sha256", "")))), f"{model_id}.payload_sha256 is unpublished")
            _require(isinstance(artifact.get("payload_size"), int) and artifact["payload_size"] > 0, f"{model_id}.payload_size is unpublished")
            _require(artifact.get("status") in {"published", "published_retained_v091"}, f"{model_id}.status is not published")

        for path, artifact in (lock.get("runtime_artifacts") or {}).items():
            _require(bool(SHA256.fullmatch(str(artifact.get("sha256", "")))), f"{path}.sha256 is unpublished")
            _require(isinstance(artifact.get("size"), int) and artifact["size"] > 0, f"{path}.size is unpublished")
            _require(artifact.get("status") == "published", f"{path}.status is not published")

        for target, identity in (lock.get("targets") or {}).items():
            _require(identity.get("qualification_status") == "passed", f"{target} is not qualified")

    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-published", action="store_true")
    parser.add_argument("--skip-gitlink-check", action="store_true")
    args = parser.parse_args()
    try:
        lock = validate(
            args.lock.resolve(),
            args.repo_root.resolve(),
            args.require_published,
            check_gitlink=not args.skip_gitlink_check,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps({"artifact_set": lock["artifact_set"], "release_state": lock["release_state"], "deployable": lock["deployable"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
