"""Profile-scoped download contract for external v0.9.1 engines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _manifest(path: Path, *, digest: str) -> None:
    path.write_text(
        json.dumps(
            {
                "hf_repo_id": "example/engines",
                "revision": "locked",
                "artifact_sets": {
                    "orin-r6": {
                        "root": "/legacy/root",
                        "hf_prefix": "sets/orin-r6/v091",
                        "required_files": [
                            "engines/asr/llm.engine",
                            "engines/tts/llm.engine",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_download_uses_hf_prefix_profile_scope_and_stable_symlink(
    tmp_path, monkeypatch
):
    from server.core import qwen3_artifact_downloader as downloader
    import huggingface_hub

    artifact_root = tmp_path / "models" / "edgellm-v091"
    repo_root = tmp_path / "models"
    manifest = tmp_path / "manifest.json"
    engine = b"qualified-engine"
    digest = hashlib.sha256(engine).hexdigest()
    _manifest(manifest, digest=digest)

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        root = Path(kwargs["local_dir"]) / "sets/orin-r6/v091"
        engine_dir = root / "engines/asr"
        engine_dir.mkdir(parents=True)
        (engine_dir / "llm.engine").write_bytes(engine)
        (engine_dir / "config.json").write_text("{}", encoding="utf-8")
        config_digest = hashlib.sha256(b"{}").hexdigest()
        (root / "SHA256SUMS").write_text(
            f"{digest}  engines/asr/llm.engine\n"
            f"{config_digest}  engines/asr/config.json\n",
            encoding="utf-8",
        )
        return str(root)

    monkeypatch.setattr(downloader, "_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setenv("OVS_PROFILE", "jetson-edgellm-v091-matcha")
    monkeypatch.setenv("QWEN3_ARTIFACT_SET", "orin-r6")
    monkeypatch.setenv("QWEN3_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("QWEN3_REPO_CACHE_ROOT", str(repo_root))

    assert downloader.ensure_artifacts([str(artifact_root / "engines/asr/llm.engine")])
    assert artifact_root.is_symlink()
    assert (artifact_root / "engines/asr/config.json").is_file()
    assert calls[0]["allow_patterns"] == [
        "sets/orin-r6/v091/PROVENANCE.md",
        "sets/orin-r6/v091/SHA256SUMS",
        "sets/orin-r6/v091/engines/asr/**",
        "sets/orin-r6/v091/manifest.json",
    ]
    assert all("engines/tts" not in item for item in calls[0]["allow_patterns"])


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    from server.core import qwen3_artifact_downloader as downloader
    import huggingface_hub

    artifact_root = tmp_path / "models" / "edgellm-v091"
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, digest="0" * 64)

    def fake_snapshot_download(**kwargs):
        root = Path(kwargs["local_dir"]) / "sets/orin-r6/v091"
        target = root / "engines/asr/llm.engine"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"corrupt")
        (root / "SHA256SUMS").write_text(
            f"{'0' * 64}  engines/asr/llm.engine\n", encoding="utf-8"
        )

    monkeypatch.setattr(downloader, "_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setenv("OVS_PROFILE", "jetson-edgellm-v091-matcha")
    monkeypatch.setenv("QWEN3_ARTIFACT_SET", "orin-r6")
    monkeypatch.setenv("QWEN3_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("QWEN3_REPO_CACHE_ROOT", str(tmp_path / "models"))

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        downloader.ensure_artifacts(
            [str(artifact_root / "engines/asr/llm.engine")]
        )


def test_download_does_not_hide_nonempty_operator_artifact_root(
    tmp_path, monkeypatch
):
    from server.core import qwen3_artifact_downloader as downloader
    import huggingface_hub

    artifact_root = tmp_path / "models" / "edgellm-v091"
    artifact_root.mkdir(parents=True)
    (artifact_root / "operator-file").write_text("keep", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, digest="0" * 64)

    def fake_snapshot_download(**kwargs):
        root = Path(kwargs["local_dir"]) / "sets/orin-r6/v091"
        target = root / "engines/asr/llm.engine"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"engine")
        digest = hashlib.sha256(b"engine").hexdigest()
        (root / "SHA256SUMS").write_text(
            f"{digest}  engines/asr/llm.engine\n", encoding="utf-8"
        )

    monkeypatch.setattr(downloader, "_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setenv("OVS_PROFILE", "jetson-edgellm-v091-matcha")
    monkeypatch.setenv("QWEN3_ARTIFACT_SET", "orin-r6")
    monkeypatch.setenv("QWEN3_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("QWEN3_REPO_CACHE_ROOT", str(tmp_path / "models"))

    with pytest.raises(RuntimeError, match="exists and is not the HF set root"):
        downloader.ensure_artifacts(
            [str(artifact_root / "engines/asr/llm.engine")]
        )
    assert (artifact_root / "operator-file").read_text() == "keep"
