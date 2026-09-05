#!/usr/bin/env python3
"""Resolve (LANGUAGE, DEVICE) to exactly one OpenVoiceStream profile + env.

Reads configs/matrix/language_device.yaml. Prints a shell-sourceable env block
(or JSON) and exits:

    0  cell status is `measured`
    0  cell status is `untested` — a WARN goes to stderr, the env is still emitted
    2  cell status is `unsupported` — nothing is emitted; Chinese never falls
       back to Whisper
    3  unknown language or device
    4  the resolved profile has no JSON in configs/profiles/

This selects a profile. It does not touch inference code, and it never invents
a profile name: every `ovs_profile` in the matrix must exist on disk.

Usage:
    resolve_profile.py --language zh --device rk3576
    resolve_profile.py --language en --device orin_nx --format json
    resolve_profile.py --write-env /opt/voice/resolved/ovs.env   # LANGUAGE/DEVICE from env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_UNSUPPORTED = 2
EXIT_UNKNOWN = 3
EXIT_MISSING_PROFILE = 4

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = REPO_ROOT / "configs" / "matrix" / "language_device.yaml"
DEFAULT_PROFILES_DIR = REPO_ROOT / "configs" / "profiles"


class ResolveError(Exception):
    """Resolution failed. `code` is the process exit code to use."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def load_matrix(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ResolveError(
            "PyYAML is required to read the language/device matrix "
            f"({path}); install pyyaml in this image.",
            EXIT_UNKNOWN,
        ) from exc
    if not path.is_file():
        raise ResolveError(f"matrix file not found: {path}", EXIT_UNKNOWN)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "cells" not in data:
        raise ResolveError(f"matrix file is not a language/device matrix: {path}", EXIT_UNKNOWN)
    return data


def normalise_language(raw: str) -> str:
    """Accept `zh`, `ZH`, `zh-CN`, `zh_CN`, ` zh ` as the same language."""
    value = (raw or "").strip().replace("_", "-").lower()
    return value.split("-", 1)[0] if value else value


def normalise_device(raw: str) -> str:
    return (raw or "").strip().replace("-", "_").lower()


def find_cell(matrix: dict[str, Any], language: str, device: str) -> dict[str, Any]:
    languages = matrix.get("languages") or {}
    devices = matrix.get("devices") or {}

    if language not in languages:
        raise ResolveError(
            f"unknown language {language!r}; matrix knows: {', '.join(sorted(languages))}",
            EXIT_UNKNOWN,
        )
    if device not in devices:
        raise ResolveError(
            f"unknown device {device!r}; matrix knows: {', '.join(sorted(devices))}",
            EXIT_UNKNOWN,
        )

    group = languages[language]["group"]
    for cell in matrix["cells"]:
        if cell.get("device") == device and cell.get("group") == group:
            return cell
    raise ResolveError(
        f"matrix has no cell for device={device} group={group} (language={language})",
        EXIT_UNKNOWN,
    )


def build_env(
    matrix: dict[str, Any], cell: dict[str, Any], language: str, device: str
) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in (matrix.get("common_env") or {}).items():
        env[str(key)] = str(value)
    for key, value in (cell.get("env") or {}).items():
        env[str(key)] = str(value)
    env["OVS_PROFILE"] = str(cell["ovs_profile"])
    env["OVS_MATRIX_STATUS"] = str(cell["status"])
    env["OVS_MATRIX_DEVICE"] = device
    env["LANGUAGE"] = language
    env["OVS_LANGUAGE"] = language
    env["ASR_LANGUAGE"] = language
    env["TTS_LANGUAGE"] = language
    return env


def resolve(
    language: str,
    device: str,
    matrix_path: Path = DEFAULT_MATRIX,
    profiles_dir: Path | None = DEFAULT_PROFILES_DIR,
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    """Return (env, cell, warnings). Raises ResolveError on any refusal."""
    matrix = load_matrix(matrix_path)
    language = normalise_language(language)
    device = normalise_device(device)
    cell = find_cell(matrix, language, device)
    status = cell.get("status")

    if status == "unsupported":
        reason = " ".join(str(cell.get("reason") or "no reason recorded").split())
        raise ResolveError(
            f"{language} on {device} is not supported: {reason}",
            EXIT_UNSUPPORTED,
        )
    if status not in ("measured", "untested"):
        raise ResolveError(
            f"cell device={device} group={cell.get('group')} has unknown status {status!r}",
            EXIT_UNKNOWN,
        )

    profile = cell.get("ovs_profile")
    if not profile:
        raise ResolveError(
            f"cell device={device} group={cell.get('group')} is {status} but names no profile",
            EXIT_UNKNOWN,
        )
    if profiles_dir is not None and not (Path(profiles_dir) / f"{profile}.json").is_file():
        raise ResolveError(
            f"profile {profile!r} is not present in {profiles_dir}",
            EXIT_MISSING_PROFILE,
        )

    warnings: list[str] = []
    if status == "untested":
        warnings.append(
            f"WARN: {language} on {device} resolves to profile {profile}, whose end-to-end "
            "combination has no measured number yet (status: untested). Deploy is allowed; "
            "do not publish latency or accuracy figures for it."
        )
        planned = cell.get("planned_alternative")
        if planned:
            warnings.append(
                f"WARN: the pairing planned for this cell is {planned.get('description')}"
                f" — blocked on: {planned.get('blocked_on')}"
            )

    return build_env(matrix, cell, language, device), cell, warnings


def format_env(env: dict[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in sorted(env.items())) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", default=os.environ.get("LANGUAGE", ""))
    parser.add_argument("--device", default=os.environ.get("DEVICE", ""))
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument(
        "--no-verify-profile",
        action="store_true",
        help="skip the check that the resolved profile JSON exists on disk",
    )
    parser.add_argument("--format", choices=("env", "json"), default="env")
    parser.add_argument(
        "--write-env",
        type=Path,
        default=None,
        help="write the env block to this path instead of stdout",
    )
    args = parser.parse_args(argv)

    if not args.language or not args.device:
        print(
            "ERROR: both --language/LANGUAGE and --device/DEVICE are required",
            file=sys.stderr,
        )
        return EXIT_UNKNOWN

    try:
        env, cell, warnings = resolve(
            args.language,
            args.device,
            matrix_path=args.matrix,
            profiles_dir=None if args.no_verify_profile else args.profiles_dir,
        )
    except ResolveError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.code

    for line in warnings:
        print(line, file=sys.stderr)

    if args.format == "json":
        payload = json.dumps(
            {"env": env, "status": cell["status"], "profile": cell["ovs_profile"]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        rendered = payload + "\n"
    else:
        rendered = format_env(env)

    if args.write_env:
        args.write_env.parent.mkdir(parents=True, exist_ok=True)
        args.write_env.write_text(rendered, encoding="utf-8")
        print(f"resolved {env['LANGUAGE']}/{env['OVS_MATRIX_DEVICE']} -> "
              f"{env['OVS_PROFILE']} ({env['OVS_MATRIX_STATUS']}) -> {args.write_env}",
              file=sys.stderr)
    else:
        sys.stdout.write(rendered)

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
