#!/usr/bin/env python3
"""Render target-specific v0.10 production profiles from a qualified lock."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


GIT_SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
TEMPLATE_GLOB = "jetson-edgellm-v010-candidate-*.json"
TARGET_SLUGS = {
    "orin-nx-16gb": "orin-nx",
    "orin-nano": "orin-nano",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _identity(lock: dict[str, Any], model_id: str, target_id: str) -> dict[str, Any]:
    artifact = lock["model_artifacts"][model_id]
    variants = artifact.get("target_variants")
    identity = variants[target_id] if variants is not None else artifact
    _require(
        identity.get("status") in {"published", "published_retained_v091"},
        f"{model_id}.{target_id} is not published",
    )
    _require(
        bool(GIT_SHA.fullmatch(str(identity.get("revision", "")))),
        f"{model_id}.{target_id} has no immutable revision",
    )
    _require(
        bool(SHA256.fullmatch(str(identity.get("payload_sha256", "")))),
        f"{model_id}.{target_id} has no immutable payload sha256",
    )
    _require(
        isinstance(identity.get("payload_size"), int) and identity["payload_size"] > 0,
        f"{model_id}.{target_id} has no payload size",
    )
    return identity


def _replace_paths(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("/opt/models-v010-candidate/", "/opt/models-v010/")
    if isinstance(value, list):
        return [_replace_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_paths(item) for key, item in value.items()}
    return value


def render(
    lock_path: Path,
    templates_dir: Path,
    output_dir: Path,
) -> list[Path]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(
        lock.get("release_state")
        in {"artifacts_qualified_image_pending", "published_and_qualified"},
        "production profiles require a build-ready or published v0.10 lock",
    )
    _require(
        lock.get("source", {}).get("status") == "published",
        "source is unpublished",
    )
    _require(
        "candidate" not in str(lock.get("artifact_set", "")),
        "production artifact_set must not retain the candidate identity",
    )
    _require(
        set(lock.get("targets", {})) == set(TARGET_SLUGS),
        "lock must contain the exact NX and Nano targets",
    )
    templates = sorted(templates_dir.glob(TEMPLATE_GLOB))
    _require(bool(templates), f"no candidate templates found in {templates_dir}")
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")

    rendered: list[tuple[str, dict[str, Any]]] = []
    for target_id, target_slug in TARGET_SLUGS.items():
        target = lock["targets"][target_id]
        _require(
            target.get("qualification_status") == "passed",
            f"{target_id} is not qualified",
        )
        supported = set(target.get("supported_lanes") or [])
        for template_path in templates:
            template = json.loads(template_path.read_text(encoding="utf-8"))
            model_ids = [
                str(item["canonical_model_id"])
                for item in template.get("model_artifacts") or []
            ]
            _require(bool(model_ids), f"{template_path.name} has no model artifacts")
            lanes = {f"speech/{model_id}" for model_id in model_ids}
            if not lanes.issubset(supported):
                continue

            profile = _replace_paths(template)
            candidate_name = str(profile.get("name", ""))
            _require(
                candidate_name.startswith("jetson-edgellm-v010-candidate-"),
                f"invalid candidate profile name: {candidate_name}",
            )
            suffix = candidate_name.removeprefix("jetson-edgellm-v010-candidate-")
            name = f"jetson-edgellm-v010-{target_slug}-{suffix}"
            profile.update(
                name=name,
                description=(
                    f"Published TensorRT-Edge-LLM v0.10 production profile for "
                    f"{target_id}, rendered from {lock['artifact_set']}."
                ),
                deployment_scope="production-published",
                artifact_set=lock["artifact_set"],
                target_id=target_id,
                release_lock_artifact_set=lock["artifact_set"],
            )
            for model in profile["model_artifacts"]:
                model_id = str(model["canonical_model_id"])
                artifact = lock["model_artifacts"][model_id]
                identity = _identity(lock, model_id, target_id)
                _require(
                    model.get("repo") == artifact.get("repo"),
                    f"repo drift: {model_id}",
                )
                model.update(
                    revision=identity["revision"],
                    payload_sha256=identity["payload_sha256"],
                    payload_size=identity["payload_size"],
                )
            serialized = json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
            _require("candidate" not in serialized, f"candidate sentinel leaked into {name}")
            rendered.append((name, profile))

    _require(len(rendered) == 10, "expected six NX and four Nano production profiles")
    output_dir.mkdir(parents=True)
    paths: list[Path] = []
    for name, profile in rendered:
        path = output_dir / f"{name}.json"
        path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--templates-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = render(
            args.lock.resolve(),
            args.templates_dir.resolve(),
            args.output_dir.resolve(),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"profiles": [path.name for path in paths]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
