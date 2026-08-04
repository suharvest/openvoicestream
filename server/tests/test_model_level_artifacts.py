"""Model-level v0.9.1 artifact source and downloader contracts."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import types
from pathlib import Path

import pytest

from server.core import leaf_composition as lc
from server.core import qwen3_artifact_downloader as qad


ASR_REPO = "harvestsu/qwen3-asr-0.6b-jetson-artifacts"
ASR_REV = "9a82e1ae0fd8dce3ab090e66ae72e3b99ec9c9bf"


def _archive(files: dict[str, bytes]) -> bytes:
    """Build a tiny tar.gz payload with exact payload-relative paths."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tf:
        for rel, data in files.items():
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def _schema_v2(model_id: str, files: dict[str, bytes], archive_name: str = "payload.tar.gz") -> tuple[dict, bytes]:
    payload = _archive(files)
    manifest = {
        "schema_version": 2,
        "model_id": model_id,
        "files": {
            rel: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            for rel, data in files.items()
        },
        "payload": {
            "path": archive_name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        },
    }
    return manifest, payload


def _required_payload(required: list[str]) -> dict[str, bytes]:
    """Materialize file-or-directory requirements into a small payload."""
    result: dict[str, bytes] = {}
    for index, rel in enumerate(required):
        # Profile entries name directory roots for engine sets. A marker file
        # beneath the root exercises directory-presence verification while
        # extension-bearing entries remain regular files.
        if Path(rel).suffix:
            result[rel] = f"artifact-{index}".encode()
        else:
            result[f"{rel}/required.bin"] = f"artifact-{index}".encode()
    return result


def _install_mocks(monkeypatch, manifests: dict[tuple[str, str], tuple[dict, bytes]]):
    """Patch network calls and return fetch/download event lists."""
    fetches: list[tuple[str, str, str, str]] = []
    downloads: list[tuple[str, str, str]] = []

    def fake_fetch(model_id, *, repo=None, revision=None, manifest_path=None):
        key = (str(repo), str(revision))
        fetches.append((str(model_id), key[0], key[1], str(manifest_path)))
        try:
            manifest, _ = manifests[key]
        except KeyError as exc:  # pragma: no cover - makes fixture failures clear
            raise AssertionError(f"unexpected manifest source {key}") from exc
        return json.loads(json.dumps(manifest))

    def fake_download(rel_path, dest, expected_sha256=None, expected_size=None, *, repo=None, revision=None):
        key = (str(repo), str(revision))
        downloads.append((str(rel_path), key[0], key[1]))
        manifest, payload = manifests[key]
        assert rel_path == manifest["payload"]["path"]
        assert expected_sha256 == manifest["payload"]["sha256"]
        assert expected_size == manifest["payload"]["size"]
        Path(dest).write_bytes(payload)
        return Path(dest)

    from server.core import hf_artifacts

    monkeypatch.setattr(hf_artifacts, "fetch_manifest", fake_fetch)
    monkeypatch.setattr(hf_artifacts, "download_file", fake_download)
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")
    return fetches, downloads


def test_leaf_model_sources_keep_repositories_independent():
    registry = lc.load_registry()
    sources = lc.resolve_model_sources(
        [
            "asr.qwen3_asr_v091.orin-nx.n1",
            "tts.qwen3_tts_v091.orin-nx.n1",
        ],
        registry,
    )
    by_model = {source.model_id: source for source in sources}
    assert by_model["qwen3-asr"].repo == ASR_REPO
    assert by_model["qwen3-asr"].canonical_id == "qwen3-asr-0.6b"
    assert by_model["qwen3-tts-customvoice"].repo == (
        "harvestsu/qwen3-tts-0.6b-customvoice-jetson-artifacts"
    )
    assert by_model["qwen3-asr"].files != by_model["qwen3-tts-customvoice"].files


def test_schema_v2_payload_install_cache_hit_and_independent_repos(tmp_path, monkeypatch):
    requests = [
        {
            "model_id": "asr",
            "canonical_model_id": "asr-canonical",
            "repo": "org/asr-artifacts",
            "revision": "asr-rev",
            "required_files": ["engines/asr"],
        },
        {
            "model_id": "tts",
            "canonical_model_id": "tts-canonical",
            "repo": "org/tts-artifacts",
            "revision": "tts-rev",
            "required_files": ["engines/tts", "models/ref.bin"],
        },
    ]
    manifests = {}
    for request in requests:
        files = _required_payload(request["required_files"])
        manifests[(request["repo"], request["revision"])] = _schema_v2(
            request["canonical_model_id"], files
        )
    fetches, downloads = _install_mocks(monkeypatch, manifests)
    cache_root = tmp_path / "cache"
    requests = [
        {
            **request,
            "cache_root": str(cache_root),
            "root": str(tmp_path / request["canonical_model_id"]),
        }
        for request in requests
    ]

    assert qad.ensure_model_requests(requests)
    assert {event[1] for event in downloads} == {"org/asr-artifacts", "org/tts-artifacts"}
    assert (cache_root / "asr-canonical" / "manifest.json").is_file()
    assert (cache_root / "tts-canonical" / "manifest.json").is_file()
    first_fetches = len(fetches)
    first_downloads = len(downloads)

    # A valid manifest and all SHA-256 locks make the second invocation a
    # cache hit; no repository is touched again.
    assert qad.ensure_model_requests(requests)
    assert len(fetches) == first_fetches
    assert len(downloads) == first_downloads


def test_schema_v2_hash_drift_redownload_failure_leaves_cache_uninstalled(tmp_path, monkeypatch):
    required = ["engines/asr"]
    files = _required_payload(required)
    manifest, payload = _schema_v2("asr-canonical", files)
    key = ("org/asr-artifacts", "locked")
    manifests = {key: (manifest, payload)}
    _install_mocks(monkeypatch, manifests)
    cache_root = tmp_path / "cache"
    request = {
        "model_id": "asr",
        "canonical_model_id": "asr-canonical",
        "repo": key[0],
        "revision": key[1],
        "required_files": required,
        "cache_root": str(cache_root),
        "root": str(tmp_path / "runtime"),
    }
    qad.ensure_model_requests([request])
    installed = cache_root / "asr-canonical" / "engines/asr/required.bin"
    installed.write_bytes(b"tampered")

    from server.core import hf_artifacts

    def corrupt_download(rel_path, dest, **kwargs):
        Path(dest).write_bytes(b"not-the-payload")
        return Path(dest)

    monkeypatch.setattr(hf_artifacts, "download_file", corrupt_download)
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        qad.ensure_model_requests([request])
    # The old cache is retained until a complete replacement is verified.
    assert installed.read_bytes() == b"tampered"


def test_legacy_nested_manifest_falls_back_to_snapshot_download(tmp_path, monkeypatch):
    from server.core import hf_artifacts

    required = ["engines/legacy"]
    files = _required_payload(required)
    manifest, _ = _schema_v2("legacy", files)
    # Legacy repositories have no root manifest; the compatibility path asks
    # for models/<model_id>/manifest.json and then retries the root form.
    calls: list[str] = []

    def fake_fetch(model_id, *, repo=None, revision=None, manifest_path=None):
        calls.append(str(manifest_path or f"models/{model_id}/manifest.json"))
        if len(calls) == 1:
            raise hf_artifacts.ArtifactError("404")
        return {"model_id": "legacy", "files": manifest["files"]}

    def fake_snapshot_download(**kwargs):
        root = Path(kwargs["local_dir"])
        for rel, data in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return str(root)

    monkeypatch.setattr(hf_artifacts, "fetch_manifest", fake_fetch)
    # The runtime image installs huggingface_hub for the legacy snapshot path;
    # keep this unit test hermetic when only the stdlib HF resolver is present.
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")
    request = {
        "model_id": "legacy",
        "repo": "org/legacy-artifacts",
        "revision": "main",
        "required_files": required,
        "cache_root": str(tmp_path / "cache"),
        "root": str(tmp_path / "runtime"),
    }
    assert qad.ensure_model_requests([request])
    assert calls == ["manifest.json", "models/legacy/manifest.json"]
    assert (tmp_path / "cache" / "legacy" / "engines/legacy/required.bin").is_file()


@pytest.mark.parametrize(
    "profile_path",
    sorted(Path(__file__).resolve().parents[2].glob("configs/profiles/jetson-edgellm-v091-*.json")),
    ids=lambda path: Path(path).stem,
)
def test_every_formal_v091_profile_downloads_each_model_source_separately(
    profile_path: Path, tmp_path, monkeypatch
):
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    entries = profile.get("model_artifacts")
    assert isinstance(entries, list) and entries
    manifests: dict[tuple[str, str], tuple[dict, bytes]] = {}
    requests = []
    for index, entry in enumerate(entries):
        assert {
            "model_id",
            "repo",
            "revision",
            "canonical_model_id",
            "root",
            "required_files",
        } <= entry.keys()
        required = list(entry["required_files"])
        files = _required_payload(required)
        key = (entry["repo"], entry["revision"])
        manifests[key] = _schema_v2(entry["canonical_model_id"], files)
        requests.append(
            {
                **entry,
                "cache_root": str(tmp_path / "cache"),
                "root": str(tmp_path / "runtime" / entry["canonical_model_id"]),
            }
        )
    fetches, downloads = _install_mocks(monkeypatch, manifests)
    assert qad.ensure_model_requests(requests)
    repos = {entry["repo"] for entry in entries}
    assert {event[1] for event in downloads} == repos
    assert {event[1] for event in fetches} == repos
    for entry in entries:
        cache = tmp_path / "cache" / entry["canonical_model_id"]
        assert (cache / "manifest.json").is_file()
        # Every profile-declared required path is materialized under its own
        # canonical cache; no shared aggregate directory is used.
        for rel in entry["required_files"]:
            path = cache / rel
            assert path.exists(), (profile_path, rel)
