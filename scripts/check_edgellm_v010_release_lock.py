#!/usr/bin/env python3
"""Validate the additive TensorRT-Edge-LLM v0.10 release identity.

Candidate locks may contain explicit null publication fields. Production image
builds and entrypoints use ``--require-image-build-ready``: all source, model,
runtime and target qualification identities must be final, while the image's
own output identity remains null. After push, an external release job fills the
image identity and uses ``--require-published``. This avoids an impossible
self-referential image digest without weakening either gate.
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

SPEECH_LANES = frozenset(
    {
        "speech/qwen3-asr-0.6b",
        "speech/qwen3-tts-0.6b-base",
        "speech/qwen3-tts-0.6b-customvoice",
        "speech/moss-tts-nano",
        "speech/matcha-icefall-zh-en",
        "speech/sparktts-0.5b",
    }
)
LLM_LANES = frozenset(
    {
        "llm/qwen3.5-4b-gdn-mtp-4k",
        "llm/qwen3.5-4b-gdn-mtp-8k",
    }
)
NANO_SPEECH_LANES = frozenset(
    {
        "speech/qwen3-asr-0.6b",
        "speech/qwen3-tts-0.6b-base",
        "speech/qwen3-tts-0.6b-customvoice",
        "speech/matcha-icefall-zh-en",
    }
)
IMAGE_SELF_IDENTITY_FIELDS = ("ref", "image_id", "registry_digest", "size_bytes")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_target_scope(lock: dict[str, Any]) -> None:
    targets = lock.get("targets") or {}
    _require(
        set(targets) == {"orin-nx-16gb", "orin-nano"},
        "targets must bind Orin NX 16GB and Orin Nano independently",
    )

    parsed: dict[str, tuple[set[str], dict[str, str]]] = {}
    for target_id, target in targets.items():
        supported_raw = target.get("supported_lanes")
        excluded_raw = target.get("excluded_lanes")
        _require(
            isinstance(supported_raw, list)
            and all(isinstance(lane, str) and "/" in lane for lane in supported_raw)
            and len(supported_raw) == len(set(supported_raw)),
            f"{target_id}.supported_lanes must be unique lane identifiers",
        )
        _require(
            isinstance(excluded_raw, list),
            f"{target_id}.excluded_lanes must be a list",
        )
        excluded: dict[str, str] = {}
        for item in excluded_raw:
            _require(
                isinstance(item, dict)
                and isinstance(item.get("lane"), str)
                and "/" in item["lane"]
                and isinstance(item.get("reason"), str)
                and bool(item["reason"].strip()),
                f"{target_id}.excluded_lanes entries require lane and reason",
            )
            _require(
                item["lane"] not in excluded,
                f"{target_id}.excluded_lanes contains duplicate {item['lane']}",
            )
            excluded[item["lane"]] = item["reason"]
        supported = set(supported_raw)
        _require(
            supported.isdisjoint(excluded),
            f"{target_id} cannot both support and exclude the same lane",
        )
        parsed[target_id] = supported, excluded

    nx = targets["orin-nx-16gb"]
    nx_supported, nx_excluded = parsed["orin-nx-16gb"]
    _require(
        nx.get("deployment_scope") == "full_speech_and_llm",
        "orin-nx-16gb must declare full_speech_and_llm",
    )
    _require(
        nx_supported == SPEECH_LANES | LLM_LANES,
        "orin-nx-16gb must support all locked speech and Qwen3.5 lanes",
    )
    _require(not nx_excluded, "orin-nx-16gb has unexpected excluded lanes")

    nano = targets["orin-nano"]
    nano_supported, nano_excluded = parsed["orin-nano"]
    _require(
        nano.get("deployment_scope") == "speech_only",
        "orin-nano must declare speech_only",
    )
    _require(
        nano_supported == NANO_SPEECH_LANES,
        "orin-nano supported_lanes must contain only Nano-qualified speech lanes",
    )
    _require(
        set(nano_excluded) == (SPEECH_LANES - NANO_SPEECH_LANES) | LLM_LANES,
        "orin-nano must explicitly exclude MOSS, Spark, and both Qwen3.5 lanes",
    )

    model_ids = set(lock.get("model_artifacts") or {})
    scoped_model_ids = {lane.split("/", 1)[1] for lane in SPEECH_LANES | LLM_LANES}
    _require(
        model_ids == scoped_model_ids,
        "model_artifacts and supported/excluded lane identities differ",
    )
    supported_targets_by_model: dict[str, set[str]] = {
        model_id: set() for model_id in model_ids
    }
    for target_id, (supported, _) in parsed.items():
        for lane in supported:
            supported_targets_by_model[lane.split("/", 1)[1]].add(target_id)
    for model_id, artifact in lock["model_artifacts"].items():
        variants = artifact.get("target_variants")
        expected_targets = supported_targets_by_model[model_id]
        if variants is None:
            _require(
                len(expected_targets) == 1
                or artifact.get("portability") == "onnx-runtime-cross-target",
                f"{model_id} requires target_variants for device-specific TensorRT plans",
            )
            continue
        _require(
            isinstance(variants, dict) and set(variants) == expected_targets,
            f"{model_id}.target_variants must match supported target scope",
        )
        _require(
            not any(
                field in artifact
                for field in ("revision", "payload_sha256", "payload_size", "status")
            ),
            f"{model_id} must not mix aggregate and target-variant identities",
        )

    images = lock.get("runtime_images") or {}
    _require(
        set(images) == {"speech", "llm"},
        "runtime_images must bind speech and llm independently",
    )
    _require(
        images["speech"].get("target_ids") == ["orin-nx-16gb", "orin-nano"],
        "speech image must target Orin NX and Orin Nano",
    )
    _require(
        images["llm"].get("target_ids") == ["orin-nx-16gb"],
        "llm image must target Orin NX only",
    )
    runtime_keys = set(lock.get("runtime_artifacts") or {})
    speech_keys = set(images["speech"].get("runtime_artifact_keys") or [])
    llm_keys = set(images["llm"].get("runtime_artifact_keys") or [])
    _require(
        speech_keys == {key for key in runtime_keys if key.startswith("speech/")},
        "speech image must bind exactly its namespaced runtime artifacts",
    )
    _require(
        llm_keys == {key for key in runtime_keys if key.startswith("llm/")},
        "llm image must bind exactly its namespaced runtime artifacts",
    )
    _require(
        speech_keys.isdisjoint(llm_keys) and speech_keys | llm_keys == runtime_keys,
        "runtime artifacts must be partitioned between speech and llm images",
    )


def _validate_gate_policy(lock: dict[str, Any]) -> None:
    _require(
        lock.get("gate_policy")
        == {
            "candidate_gate": "require-candidate",
            "embedded_image_gate": "require-image-build-ready",
            "external_publication_gate": "require-published",
            "image_build_state": "artifacts_qualified_image_pending",
            "image_self_identity_fields": list(IMAGE_SELF_IDENTITY_FIELDS),
        },
        "gate_policy must preserve the candidate/build-ready/external phases",
    )


def _validate_release_prerequisites(
    lock: dict[str, Any],
    runtime_root: Path | None,
    image_key: str | None,
    expected_runtime_status: str,
) -> None:
    _require(
        (lock.get("source") or {}).get("status") == "published",
        "source.status is not published",
    )
    for model_id, artifact in (lock.get("model_artifacts") or {}).items():
        identities = artifact.get("target_variants") or {"shared": artifact}
        for variant_name, identity in identities.items():
            label = f"{model_id}.{variant_name}"
            _require(
                bool(GIT_SHA.fullmatch(str(identity.get("revision", "")))),
                f"{label}.revision is unpublished",
            )
            _require(
                bool(SHA256.fullmatch(str(identity.get("payload_sha256", "")))),
                f"{label}.payload_sha256 is unpublished",
            )
            _require(
                isinstance(identity.get("payload_size"), int)
                and identity["payload_size"] > 0,
                f"{label}.payload_size is unpublished",
            )
            _require(
                identity.get("status") in {"published", "published_retained_v091"},
                f"{label}.status is not published",
            )

    runtime_artifacts = lock.get("runtime_artifacts") or {}
    _require(bool(runtime_artifacts), "runtime_artifacts must not be empty")
    for artifact_key, artifact in runtime_artifacts.items():
        relative = artifact.get("path")
        _require(
            isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            f"{artifact_key}.path must be a safe relative path",
        )
        _require(
            ".." not in Path(relative).parts,
            f"{artifact_key}.path must be a safe relative path",
        )
        _require(
            bool(SHA256.fullmatch(str(artifact.get("sha256", "")))),
            f"{artifact_key}.sha256 is unpublished",
        )
        _require(
            isinstance(artifact.get("size"), int) and artifact["size"] > 0,
            f"{artifact_key}.size is unpublished",
        )
        _require(
            artifact.get("status") == expected_runtime_status,
            f"{artifact_key}.status is not {expected_runtime_status}",
        )
    if runtime_root is not None:
        _require(image_key in {"speech", "llm"}, "--runtime-root requires --image-key")
        resolved_root = runtime_root.resolve()
        expected_keys = lock["runtime_images"][image_key]["runtime_artifact_keys"]
        for artifact_key in expected_keys:
            artifact = runtime_artifacts[artifact_key]
            relative = artifact["path"]
            path = (resolved_root / relative).resolve()
            _require(
                path.is_relative_to(resolved_root),
                f"runtime artifact escapes runtime root: {relative}",
            )
            _require(
                path.is_file() and not path.is_symlink(),
                f"runtime artifact is absent or not a regular file: {relative}",
            )
            _require(
                path.stat().st_size == artifact["size"],
                f"runtime artifact size mismatch: {relative}",
            )
            _require(
                _sha256(path) == artifact["sha256"],
                f"runtime artifact sha256 mismatch: {relative}",
            )
            if relative.startswith("bin/"):
                _require(
                    bool(path.stat().st_mode & 0o111),
                    f"runtime worker is not executable: {relative}",
                )

    for target, identity in (lock.get("targets") or {}).items():
        _require(
            identity.get("qualification_status") == "passed",
            f"{target} is not qualified",
        )


def _validate_build_pending_images(lock: dict[str, Any]) -> None:
    for image_name, image in lock["runtime_images"].items():
        _require(
            image.get("status") == "build_pending",
            f"runtime_images.{image_name}.status must be build_pending",
        )
        _require(
            bool(GIT_SHA.fullmatch(str(image.get("service_revision", "")))),
            f"runtime_images.{image_name}.service_revision is unpublished",
        )
        for field in IMAGE_SELF_IDENTITY_FIELDS:
            _require(
                image.get(field) is None,
                f"runtime_images.{image_name}.{field} must remain null at image build",
            )


def _validate_candidate_images(lock: dict[str, Any]) -> None:
    for image_name, image in lock["runtime_images"].items():
        _require(
            image.get("status") == "unpublished",
            f"runtime_images.{image_name}.status must be unpublished for candidate",
        )
        _require(
            image.get("service_revision") is None,
            f"runtime_images.{image_name}.service_revision must be null for candidate",
        )
        for field in IMAGE_SELF_IDENTITY_FIELDS:
            _require(
                image.get(field) is None,
                f"runtime_images.{image_name}.{field} must be null for candidate",
            )


def _validate_published_images(lock: dict[str, Any]) -> None:
    for image_name, image in lock["runtime_images"].items():
        _require(bool(image.get("ref")), f"runtime_images.{image_name}.ref is unpublished")
        _require(
            bool(
                SHA256.fullmatch(
                    str(image.get("image_id", "")).removeprefix("sha256:")
                )
            ),
            f"runtime_images.{image_name}.image_id is unpublished",
        )
        digest = str(image.get("registry_digest", "")).removeprefix("sha256:")
        _require(
            bool(SHA256.fullmatch(digest)),
            f"runtime_images.{image_name}.registry_digest is unpublished",
        )
        _require(
            image["ref"].endswith(f"@sha256:{digest}"),
            f"runtime_images.{image_name}.ref must be pinned to registry_digest",
        )
        _require(
            isinstance(image.get("size_bytes"), int) and image["size_bytes"] > 0,
            f"runtime_images.{image_name}.size_bytes is unpublished",
        )
        _require(
            bool(GIT_SHA.fullmatch(str(image.get("service_revision", "")))),
            f"runtime_images.{image_name}.service_revision is unpublished",
        )
        _require(
            image.get("status") == "published",
            f"runtime_images.{image_name}.status is not published",
        )


def validate(
    lock_path: Path,
    repo_root: Path,
    require_published: bool,
    *,
    check_gitlink: bool = True,
    require_candidate: bool = False,
    require_image_build_ready: bool = False,
    runtime_root: Path | None = None,
    image_key: str | None = None,
    expected_service_revision: str | None = None,
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(lock.get("schema_version") == 1, "schema_version must be 1")
    _require(lock.get("rollback_release_lock") == "deploy/artifacts/v091-release-lock.json", "v0.9.1 rollback lock must remain explicit")
    _validate_target_scope(lock)
    _validate_gate_policy(lock)

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

    release_state = lock.get("release_state")
    candidate = release_state == "candidate_unpublished"
    build_pending = release_state == "artifacts_qualified_image_pending"
    published = release_state == "published_and_qualified"
    if candidate or build_pending:
        _require(lock.get("deployable") is False, "candidate lock must be non-deployable")
    elif published:
        _require(lock.get("deployable") is True, "published lock must be deployable")
    else:
        raise ValueError("unknown release_state")

    if runtime_root is not None:
        _require(
            require_image_build_ready or require_published,
            "--runtime-root requires a build-ready or published gate",
        )
    if image_key is not None:
        _require(runtime_root is not None, "--image-key requires --runtime-root")

    if require_candidate:
        _require(candidate, "v0.10 gray image requires candidate_unpublished")
        _require(
            source.get("status") == "candidate_pinned",
            "candidate source.status must be candidate_pinned",
        )
        _validate_candidate_images(lock)

    if require_image_build_ready:
        _require(
            build_pending,
            "v0.10 image requires artifacts_qualified_image_pending",
        )
        _validate_release_prerequisites(
            lock, runtime_root, image_key, "qualified_for_image"
        )
        _validate_build_pending_images(lock)

    if require_published:
        _require(published, "v0.10 release is not published_and_qualified")
        _validate_release_prerequisites(
            lock, runtime_root, image_key, "published_in_image"
        )
        _validate_published_images(lock)

    if expected_service_revision is not None:
        _require(
            bool(GIT_SHA.fullmatch(expected_service_revision)),
            "expected service revision must be an immutable git SHA",
        )
        for image_name, image in lock["runtime_images"].items():
            _require(
                image.get("service_revision") == expected_service_revision,
                f"runtime_images.{image_name}.service_revision differs from image build",
            )

    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument("--require-candidate", action="store_true")
    gate.add_argument("--require-image-build-ready", action="store_true")
    gate.add_argument("--require-published", action="store_true")
    parser.add_argument("--skip-gitlink-check", action="store_true")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--image-key", choices=("speech", "llm"))
    parser.add_argument("--expected-service-revision")
    args = parser.parse_args()
    try:
        lock = validate(
            args.lock.resolve(),
            args.repo_root.resolve(),
            args.require_published,
            check_gitlink=not args.skip_gitlink_check,
            require_candidate=args.require_candidate,
            require_image_build_ready=args.require_image_build_ready,
            runtime_root=args.runtime_root,
            image_key=args.image_key,
            expected_service_revision=args.expected_service_revision,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps({"artifact_set": lock["artifact_set"], "release_state": lock["release_state"], "deployable": lock["deployable"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
