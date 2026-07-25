#!/usr/bin/env python3
"""Regenerate and verify a TensorRT-Edge-LLM v0.9.1 release manifest.

The existing manifest supplies immutable release metadata. This tool replaces
only its payload inventory and publication flag, then writes a matching
SHA256SUMS. Engine/plan metadata sidecars are validated against their target
artifact so a read-only deployment cannot accept stale compatibility data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


CONTROL_FILES = {"manifest.json", "SHA256SUMS"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_paths(root: Path) -> list[Path]:
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in CONTROL_FILES
    ]
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def validate_sidecars(root: Path, entries: list[dict[str, Any]]) -> int:
    digest_by_path = {entry["path"]: entry["sha256"] for entry in entries}
    checked = 0
    for sidecar in sorted(root.rglob("*.meta.json")):
        relative = sidecar.relative_to(root).as_posix()
        target_relative = relative[: -len(".meta.json")]
        if target_relative not in digest_by_path:
            raise ValueError(f"sidecar target is absent from payload: {relative}")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        expected = data.get("engine_sha256")
        actual = digest_by_path[target_relative]
        if expected != actual:
            raise ValueError(
                f"sidecar digest mismatch for {target_relative}: "
                f"expected={expected!r} actual={actual}"
            )
        host = data.get("host")
        if not isinstance(host, dict):
            raise ValueError(f"sidecar has no host compatibility block: {relative}")
        for key in ("sm", "trt_version", "jp_version", "cuda_version", "platform"):
            if not host.get(key):
                raise ValueError(f"sidecar host.{key} is missing: {relative}")
        checked += 1
    return checked


def validate_required_files(
    manifest: dict[str, Any], entries: list[dict[str, Any]]
) -> int:
    required = manifest.get("required_files")
    if required is None:
        return 0
    if (
        not isinstance(required, list)
        or not all(isinstance(path, str) and path for path in required)
        or len(required) != len(set(required))
    ):
        raise ValueError("required_files must be a unique list of non-empty paths")
    available = CONTROL_FILES | {entry["path"] for entry in entries}
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"required release payload is missing: {missing}")
    return len(required)


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--published-to-hf",
        choices=("preserve", "true", "false"),
        default="preserve",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest is missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    for path in payload_paths(root):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )

    sidecar_count = validate_sidecars(root, entries)
    required_count = validate_required_files(manifest, entries)
    manifest["files"] = entries
    if args.published_to_hf != "preserve":
        manifest["published_to_hf"] = args.published_to_hf == "true"

    manifest_text = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    sums_text = "".join(
        f"{entry['sha256']}  {entry['path']}\n" for entry in entries
    )
    atomic_write(manifest_path, manifest_text)
    atomic_write(root / "SHA256SUMS", sums_text)

    print(
        json.dumps(
            {
                "root": str(root),
                "files": len(entries),
                "bytes": sum(entry["size"] for entry in entries),
                "sidecars_verified": sidecar_count,
                "required_files_verified": required_count,
                "published_to_hf": manifest.get("published_to_hf"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
