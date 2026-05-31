"""Tests for app.core.moss_artifacts MOSS-TTS-Nano runtime provisioner (#47).

Covers:
  * pulls each manifest file to the correct on-device target (engines/codec
    under model_root, worker binary under worker_dir);
  * idempotent — a present + hash-matching file is not re-downloaded and not
    deleted;
  * optional files (worker) are skipped on download failure, not fatal;
  * required-file hash mismatch is fatal.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from app.core import moss_artifacts


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def manifest(tmp_path):
    """A minimal MOSS manifest with one engine file, one codec file, and the
    worker — pointing model_root/worker_dir at tmp_path subdirs."""
    model_root = tmp_path / "models" / "moss-tts-nano"
    worker_dir = tmp_path / "jv-workers"
    return {
        "hf_repo": "test/repo",
        "hf_prefix": "models/moss-tts-nano",
        "revision": "main",
        "targets": {
            "model_root": str(model_root),
            "worker_dir": str(worker_dir),
        },
        "files": [
            {"path": "engines/moss_tts_prefill.plan", "dest": "model_root"},
            {"path": "codec_onnx/codec_decode_step.plan", "dest": "model_root"},
            {
                "path": "moss_tts_nano_worker",
                "dest": "worker_dir",
                "executable": True,
                "optional": True,
            },
        ],
    }


def _patch_manifest(monkeypatch, manifest):
    monkeypatch.setattr(moss_artifacts, "_load_manifest", lambda: manifest)
    monkeypatch.setenv("MOSS_ARTIFACT_AUTO_DOWNLOAD", "1")


def test_provisions_files_to_correct_targets(monkeypatch, manifest):
    """Each manifest file lands at the right path: engines/codec keep their
    subpath under model_root, worker is flat under worker_dir."""
    _patch_manifest(monkeypatch, manifest)
    contents = {
        "engines/moss_tts_prefill.plan": b"PREFILL",
        "codec_onnx/codec_decode_step.plan": b"CODEC",
        "moss_tts_nano_worker": b"WORKERBIN",
    }
    downloaded: list[str] = []

    def fake_download(url, dest):
        # url ends with prefix/source_path — recover which file.
        for rel, payload in contents.items():
            if url.endswith(rel):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
                downloaded.append(rel)
                return
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(moss_artifacts, "_download", fake_download)
    moss_artifacts.ensure_moss_artifacts()

    model_root = manifest["targets"]["model_root"]
    worker_dir = manifest["targets"]["worker_dir"]
    from pathlib import Path

    assert (Path(model_root) / "engines" / "moss_tts_prefill.plan").read_bytes() == b"PREFILL"
    assert (Path(model_root) / "codec_onnx" / "codec_decode_step.plan").read_bytes() == b"CODEC"
    # worker is flattened (no models/ prefix) under worker_dir
    assert (Path(worker_dir) / "moss_tts_nano_worker").read_bytes() == b"WORKERBIN"
    assert set(downloaded) == set(contents)


def test_idempotent_skips_present_hash_matching_files(monkeypatch, manifest):
    """A file already present with a matching hash is NOT re-downloaded and
    NOT deleted (idempotent re-run)."""
    from pathlib import Path

    payload = b"PREFILL"
    manifest["files"] = [
        {
            "path": "engines/moss_tts_prefill.plan",
            "dest": "model_root",
            "sha256": _sha256(payload),
        }
    ]
    _patch_manifest(monkeypatch, manifest)

    model_root = Path(manifest["targets"]["model_root"])
    existing = model_root / "engines" / "moss_tts_prefill.plan"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(payload)

    def boom(url, dest):
        raise AssertionError("must not download a hash-matching file")

    monkeypatch.setattr(moss_artifacts, "_download", boom)
    moss_artifacts.ensure_moss_artifacts()
    assert existing.read_bytes() == payload  # untouched


def test_md5_verified_file_skipped(monkeypatch, manifest):
    """HF manifest ships md5 (not sha256) for bundled files — md5 match counts."""
    from pathlib import Path

    payload = b"CODEC"
    manifest["files"] = [
        {
            "path": "codec_onnx/codec_decode_step.plan",
            "dest": "model_root",
            "md5": _md5(payload),
        }
    ]
    _patch_manifest(monkeypatch, manifest)
    p = Path(manifest["targets"]["model_root"]) / "codec_onnx" / "codec_decode_step.plan"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)

    monkeypatch.setattr(
        moss_artifacts, "_download",
        lambda url, dest: (_ for _ in ()).throw(AssertionError("no download")),
    )
    moss_artifacts.ensure_moss_artifacts()


def test_optional_file_download_failure_not_fatal(monkeypatch, manifest):
    """The worker is optional (baked in fat image); its download failure must
    not abort provisioning of the required engines."""
    from pathlib import Path

    _patch_manifest(monkeypatch, manifest)

    def fake_download(url, dest):
        if url.endswith("moss_tts_nano_worker"):
            raise moss_artifacts.MossArtifactError("simulated 404")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ok")

    monkeypatch.setattr(moss_artifacts, "_download", fake_download)
    # Must not raise despite the worker failing.
    moss_artifacts.ensure_moss_artifacts()
    model_root = Path(manifest["targets"]["model_root"])
    assert (model_root / "engines" / "moss_tts_prefill.plan").exists()
    assert not (Path(manifest["targets"]["worker_dir"]) / "moss_tts_nano_worker").exists()


def test_required_hash_mismatch_is_fatal(monkeypatch, manifest):
    """A required file whose post-download hash mismatches aborts hard and the
    bad file is removed."""
    from pathlib import Path

    manifest["files"] = [
        {
            "path": "engines/moss_tts_prefill.plan",
            "dest": "model_root",
            "sha256": _sha256(b"EXPECTED"),
        }
    ]
    _patch_manifest(monkeypatch, manifest)

    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"WRONG")

    monkeypatch.setattr(moss_artifacts, "_download", fake_download)
    with pytest.raises(moss_artifacts.MossArtifactError, match="sha256 mismatch"):
        moss_artifacts.ensure_moss_artifacts()
    bad = Path(manifest["targets"]["model_root"]) / "engines" / "moss_tts_prefill.plan"
    assert not bad.exists()


def test_auto_download_disabled_is_noop(monkeypatch, manifest):
    monkeypatch.setenv("MOSS_ARTIFACT_AUTO_DOWNLOAD", "0")
    monkeypatch.setattr(
        moss_artifacts, "_load_manifest",
        lambda: (_ for _ in ()).throw(AssertionError("manifest must not load")),
    )
    moss_artifacts.ensure_moss_artifacts()  # returns silently


def test_bundled_manifest_loads_and_is_consistent():
    """The shipped deploy/artifacts/moss_manifest.json parses, declares the
    worker + the 6 plan files, and the worker carries the #48 sha256."""
    manifest = moss_artifacts._load_manifest()
    files = {f["path"]: f for f in manifest["files"]}
    assert "moss_tts_nano_worker" in files
    assert files["moss_tts_nano_worker"]["sha256"] == (
        "b09f7f8d183f1e1dda8d261a5fbf22e4f429e25b84dfa125de5f8a820dc706f7"
    )
    assert files["moss_tts_nano_worker"]["dest"] == "worker_dir"
    plans = [p for p in files if p.endswith(".plan")]
    # 5 TTS plans + 1 codec plan
    assert len(plans) == 6
    # codec plan lives under codec_onnx, dest model_root
    assert files["codec_onnx/codec_decode_step.plan"]["dest"] == "model_root"
