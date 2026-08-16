from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts" / "package_edgellm_v010_artifact.py"
PROMOTE = ROOT / "scripts" / "promote_edgellm_v010_runtime_manifest.py"


def test_promotes_without_mutating_or_copying_payload(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "spec_base.engine").write_bytes(b"base")
    (payload / "spec_draft.engine").write_bytes(b"draft")
    source = tmp_path / "source"
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE),
            str(payload),
            str(source),
            "--artifact", "qwen-v010",
            "--model-id", "qwen3.5-4b-gdn-mtp-8k",
            "--repo", "org/repo",
            "--source", "model-revision",
            "--profile", "full-build-recipe",
            "--provenance", '{"precision":"W4A16"}',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    source_manifest = (source / "manifest.json").read_bytes()
    output = tmp_path / "promoted"

    result = subprocess.run(
        [
            sys.executable,
            str(PROMOTE),
            str(source),
            str(output),
            "--engine-profile", "8k",
            "--max-input-len", "8192",
            "--max-kv-cache-capacity", "8192",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (source / "manifest.json").read_bytes() == source_manifest
    assert os.stat(source / "payload.tar").st_ino == os.stat(output / "payload.tar").st_ino
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["engine_profile"] == "8k"
    assert manifest["engine_contract"] == {
        "max_input_len": 8192,
        "max_kv_cache_capacity": 8192,
    }
    assert manifest["provenance"]["build_profile"] == "full-build-recipe"
    assert (output / "SHA256SUMS").read_text().splitlines()[0].endswith(
        "  manifest.json"
    )


def test_rejects_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(PROMOTE),
            str(tmp_path / "missing"),
            str(output),
            "--engine-profile", "4k",
            "--max-input-len", "4096",
            "--max-kv-cache-capacity", "4096",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
