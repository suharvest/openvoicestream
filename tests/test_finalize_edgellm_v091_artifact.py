from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "finalize_edgellm_v091_artifact.py"
)


def test_finalizer_inventories_payload_and_validates_sidecar(tmp_path: Path):
    engine = tmp_path / "engines" / "model.engine"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"engine")
    digest = hashlib.sha256(b"engine").hexdigest()
    sidecar = engine.with_name("model.engine.meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "engine_sha256": digest,
                "host": {
                    "sm": "87",
                    "trt_version": "10.3",
                    "jp_version": "6.2",
                    "cuda_version": "12.6",
                    "platform": "tegra",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_set": "test",
                "files": [],
                "published_to_hf": True,
                "upstream_sha": "abc",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "SHA256SUMS").write_text("stale\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            "--published-to-hf",
            "false",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["files"] == 2
    assert summary["sidecars_verified"] == 1
    assert summary["published_to_hf"] is False

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["upstream_sha"] == "abc"
    assert manifest["published_to_hf"] is False
    assert [entry["path"] for entry in manifest["files"]] == [
        "engines/model.engine",
        "engines/model.engine.meta.json",
    ]
    assert len((tmp_path / "SHA256SUMS").read_text().splitlines()) == 2


def test_finalizer_rejects_stale_sidecar(tmp_path: Path):
    engine = tmp_path / "bad.plan"
    engine.write_bytes(b"plan")
    (tmp_path / "bad.plan.meta.json").write_text(
        json.dumps(
            {
                "engine_sha256": "0" * 64,
                "host": {
                    "sm": "87",
                    "trt_version": "10.3",
                    "jp_version": "6.2",
                    "cuda_version": "12.6",
                    "platform": "tegra",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": [], "published_to_hf": False}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "sidecar digest mismatch" in result.stderr
