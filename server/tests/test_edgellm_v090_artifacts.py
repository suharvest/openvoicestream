"""Tests for server.core.edgellm_v090_artifacts (v090 ASR on-demand provisioning).

Covers:
  * files land under engine_root keeping their manifest subpath (incl. the
    engines/asr_audio_encoder/audio/ nesting) and the worker gets +x;
  * idempotent — a present + hash-matching file is not re-downloaded;
  * a STALE manifest does not destroy a good local file (the regression that
    motivated the install-ordering fix in artifact_provision);
  * the shipped bundled manifest is the ASR-only subset.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from server.core import artifact_provision, edgellm_v090_artifacts as ev090, moss_artifacts


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


@pytest.fixture
def manifest(tmp_path):
    """A minimal v090 manifest: one thinker file, the nested audio encoder
    engine, and the worker binary — rooted at a tmp_path engine_root."""
    return {
        "hf_repo": "test/repo",
        "hf_prefix": "models/edgellm-v090-asr",
        "revision": "main",
        "targets": {"engine_root": str(tmp_path / "edgellm-v090")},
        "files": [
            {"path": "engines/asr_thinker_full_int4_b2/llm.engine"},
            {"path": "engines/asr_audio_encoder/audio/audio_encoder.engine"},
            {"path": "bin/qwen3_asr_worker", "executable": True},
        ],
    }


def _patch_manifest(monkeypatch, manifest):
    monkeypatch.setattr(ev090, "_load_manifest", lambda manifest_path=None: manifest)
    monkeypatch.setenv("EDGELLM_V090_ARTIFACT_AUTO_DOWNLOAD", "1")
    monkeypatch.delenv("EDGELLM_V090_ENGINE_ROOT", raising=False)


def test_provisions_files_under_engine_root_preserving_nesting(monkeypatch, manifest):
    """Every file keeps its manifest subpath under engine_root — notably the
    audio encoder's extra audio/ level, which the runtime appends itself."""
    _patch_manifest(monkeypatch, manifest)
    contents = {
        "engines/asr_thinker_full_int4_b2/llm.engine": b"THINKER",
        "engines/asr_audio_encoder/audio/audio_encoder.engine": b"AUDIOENC",
        "bin/qwen3_asr_worker": b"WORKERBIN",
    }
    downloaded: list[str] = []

    def fake_download(url, dest):
        for rel, payload in contents.items():
            # dest is the .staged sibling; match on the url instead.
            if url.endswith(rel):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
                downloaded.append(rel)
                return
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(ev090, "_download", fake_download)
    ev090.ensure_edgellm_v090_artifacts()

    root = Path(manifest["targets"]["engine_root"])
    for rel, payload in contents.items():
        assert (root / rel).read_bytes() == payload
    assert set(downloaded) == set(contents)
    # worker must be executable
    assert os.access(root / "bin" / "qwen3_asr_worker", os.X_OK)
    # and no .staged leftovers
    assert not list(root.rglob("*.staged"))


def test_idempotent_skips_present_hash_matching_files(monkeypatch, manifest):
    payload = b"THINKER"
    manifest["files"] = [
        {"path": "engines/asr_thinker_full_int4_b2/llm.engine", "md5": _md5(payload)}
    ]
    _patch_manifest(monkeypatch, manifest)
    existing = Path(manifest["targets"]["engine_root"]) / manifest["files"][0]["path"]
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(payload)

    monkeypatch.setattr(
        ev090, "_download",
        lambda url, dest: (_ for _ in ()).throw(AssertionError("must not download")),
    )
    ev090.ensure_edgellm_v090_artifacts()
    assert existing.read_bytes() == payload


def test_stale_manifest_does_not_destroy_good_local_file(monkeypatch, manifest):
    """Regression: a manifest whose md5 no longer matches what HF serves used to
    (a) overwrite the local file and then (b) delete it on the failed check,
    leaving nothing. Now the check happens on the .staged copy, so the existing
    file survives intact and the error still propagates."""
    good = b"GOOD-ENGINE-ON-DEVICE"
    manifest["files"] = [
        {
            "path": "engines/asr_thinker_full_int4_b2/llm.engine",
            "md5": _md5(b"STALE-EXPECTATION"),
        }
    ]
    _patch_manifest(monkeypatch, manifest)
    dest = Path(manifest["targets"]["engine_root"]) / manifest["files"][0]["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(good)

    def fake_download(url, staged):
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"WHAT-HF-ACTUALLY-SERVES")

    monkeypatch.setattr(ev090, "_download", fake_download)
    with pytest.raises(ev090.EdgellmV090ArtifactError, match="md5 mismatch"):
        ev090.ensure_edgellm_v090_artifacts()

    assert dest.read_bytes() == good, "pre-existing good artifact must survive"
    assert not (dest.parent / (dest.name + ".staged")).exists()


def test_moss_stale_manifest_also_preserves_local_file(monkeypatch, tmp_path):
    """Same guarantee for the MOSS provisioner (shared install path)."""
    good = b"GOOD-PLAN"
    manifest = {
        "targets": {
            "model_root": str(tmp_path / "moss"),
            "worker_dir": str(tmp_path / "workers"),
        },
        "files": [
            {
                "path": "engines/moss_tts_prefill.plan",
                "dest": "model_root",
                "md5": _md5(b"STALE"),
            }
        ],
    }
    monkeypatch.setattr(moss_artifacts, "_load_manifest", lambda: manifest)
    monkeypatch.setenv("MOSS_ARTIFACT_AUTO_DOWNLOAD", "1")
    dest = tmp_path / "moss" / "engines" / "moss_tts_prefill.plan"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(good)

    def fake_download(url, staged):
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"DIFFERENT")

    monkeypatch.setattr(moss_artifacts, "_download", fake_download)
    with pytest.raises(moss_artifacts.MossArtifactError, match="md5 mismatch"):
        moss_artifacts.ensure_moss_artifacts()
    assert dest.read_bytes() == good


def test_download_failure_leaves_existing_file_intact(monkeypatch, manifest):
    """A network failure (not just a hash mismatch) must also be non-destructive."""
    good = b"GOOD"
    manifest["files"] = [
        {"path": "engines/asr_thinker_full_int4_b2/llm.engine", "md5": _md5(b"OTHER")}
    ]
    _patch_manifest(monkeypatch, manifest)
    dest = Path(manifest["targets"]["engine_root"]) / manifest["files"][0]["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(good)

    def boom(url, staged):
        raise ev090.EdgellmV090ArtifactError("simulated 404")

    monkeypatch.setattr(ev090, "_download", boom)
    with pytest.raises(ev090.EdgellmV090ArtifactError, match="simulated 404"):
        ev090.ensure_edgellm_v090_artifacts()
    assert dest.read_bytes() == good


def test_auto_download_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("EDGELLM_V090_ARTIFACT_AUTO_DOWNLOAD", "0")
    monkeypatch.setattr(
        ev090, "_load_manifest",
        lambda manifest_path=None: (_ for _ in ()).throw(AssertionError("must not load")),
    )
    ev090.ensure_edgellm_v090_artifacts()


def test_bundled_manifest_is_the_asr_only_subset():
    """The shipped manifest parses, targets /opt/edgellm-v090, declares the 10
    ASR files with md5+size, marks the worker executable, and carries NONE of
    the v090 TTS engines (that 1.39GB is what this whole change avoids)."""
    manifest = ev090._load_manifest()
    assert manifest["targets"]["engine_root"] == "/opt/edgellm-v090"
    files = {f["path"]: f for f in manifest["files"]}
    assert len(files) == 10
    assert all(f.get("md5") and f.get("size") for f in files.values())
    assert files["bin/qwen3_asr_worker"]["executable"] is True
    # exact nesting of the audio encoder (EDGE_LLM_ASR_AUDIO_ENC_DIR is the parent)
    assert "engines/asr_audio_encoder/audio/audio_encoder.engine" in files
    assert files["libNvInfer_edgellm_plugin.so"]["md5"] == "12e4ff753431d33ce3d2a2bab212acf6"
    assert not [p for p in files if "tts_" in p]


def test_bundled_manifest_matches_profile_field():
    """The v090 profile's asr_artifact_manifest must point at the shipped file
    and resolve through _load_manifest's repo-relative candidate."""
    import json

    profile = json.loads(
        (Path(ev090._repo_root()) / "configs" / "profiles" / "jetson-edgellm-v090-moss.json").read_text()
    )
    rel = profile["asr_artifact_manifest"]
    assert rel == ev090.BUNDLED_MANIFEST_REL
    assert ev090._load_manifest(rel)["name"] == "edgellm-v090-asr"


def test_install_verified_is_atomic_no_partial_dest(tmp_path):
    """artifact_provision.install_verified never leaves a partial dest behind."""
    dest = tmp_path / "sub" / "artifact.bin"
    item = {"md5": _md5(b"OK")}

    def dl(url, staged):
        staged.write_bytes(b"OK")

    artifact_provision.install_verified(
        "http://x/artifact.bin", dest, item, dl,
        lambda p, i: artifact_provision.check_hashes(p, i, RuntimeError),
    )
    assert dest.read_bytes() == b"OK"
    assert not (dest.parent / "artifact.bin.staged").exists()
