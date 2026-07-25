from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "check_moss_worker_runtime.py"
)
DOCKERFILE = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "docker"
    / "Dockerfile.jetson.edgellm-v091-runtime"
)
RELEASE_LOCK = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "artifacts"
    / "v091-release-lock.json"
)
SPEC = importlib.util.spec_from_file_location("check_moss_worker_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_accepts_resolved_onnxruntime_without_undefined_symbols():
    output = """
        libonnxruntime.so.1 => /usr/local/lib/onnxruntime/libonnxruntime.so.1 (0x1)
        libstdc++.so.6 => /lib/aarch64-linux-gnu/libstdc++.so.6 (0x2)
    """

    assert MODULE.validate_ldd_output(output) == []


def test_rejects_missing_onnxruntime_soname():
    output = """
        libonnxruntime.so.1 => not found
        undefined symbol: OrtGetApiBase, version VERS_1.20.0 (/worker)
    """

    errors = MODULE.validate_ldd_output(output)
    assert "libonnxruntime.so.1 did not resolve to an absolute path" in errors
    assert "one or more shared libraries are not found" in errors
    assert "one or more versioned symbols are undefined" in errors


def test_rejects_versioned_symbol_mismatch_even_when_library_resolves():
    output = """
        libonnxruntime.so.1 => /usr/local/lib/onnxruntime/libonnxruntime.so.1 (0x1)
        /worker: /usr/local/lib/onnxruntime/libonnxruntime.so.1:
            version `VERS_1.20.0' not found (required by /worker)
        undefined symbol: OrtGetApiBase, version VERS_1.20.0 (/worker)
    """

    assert MODULE.validate_ldd_output(output) == [
        "the worker requires an unavailable symbol version",
        "one or more versioned symbols are undefined",
    ]


def test_release_lock_supplies_immutable_worker_sha(tmp_path: Path):
    expected = "a" * 64
    lock = tmp_path / "release-lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": {
                    "bin/moss_tts_nano_worker": {"sha256": expected}
                },
            }
        ),
        encoding="utf-8",
    )

    assert MODULE.expected_sha256_from_release_lock(
        lock, "bin/moss_tts_nano_worker"
    ) == expected


def test_worker_hash_mismatch_fails_even_when_ldd_is_clean(
    tmp_path: Path, monkeypatch
):
    worker = tmp_path / "moss_tts_nano_worker"
    worker.write_bytes(b"candidate")
    worker.chmod(0o755)
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "libonnxruntime.so.1 => "
                "/usr/local/lib/onnxruntime/libonnxruntime.so.1 (0x1)\n"
            ),
            stderr="",
        ),
    )

    _, errors = MODULE.check_worker(worker, expected_sha256="0" * 64)

    assert len(errors) == 1
    assert "SHA256 mismatch" in errors[0]


def test_v091_runtime_image_wires_soname_and_semantic_worker_gate():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "ln -sf libonnxruntime.so.1.23.2" in dockerfile
    assert "scripts/check_moss_worker_runtime.py" in dockerfile
    assert "scripts/start_edgellm_v091_runtime.py" in dockerfile
    assert "deploy/artifacts/v091-release-lock.json" in dockerfile
    assert "COPY deploy/artifacts/v091-release-gate/moss_tts_nano_worker" in dockerfile
    assert "--worker /opt/edgellm-v091/bin/moss_tts_nano_worker" in dockerfile
    assert "--release-lock /opt/speech/deploy/v091-release-lock.json" in dockerfile
    assert 'CMD ["python3", "/opt/speech/scripts/start_edgellm_v091_runtime.py"]' in dockerfile
    assert "MOSS_WORKER_SHA256" not in dockerfile
    assert "/opt/jv-workers/moss_tts_nano_worker" not in dockerfile


def test_v091_release_lock_pins_the_clean_formal_worker():
    lock = json.loads(RELEASE_LOCK.read_text(encoding="utf-8"))

    assert lock["schema_version"] == 1
    assert lock["artifact_set"].endswith("-20260725-r2")
    assert lock["source"]["upstream_sha"] == (
        "7f061f21f0a581ba234a1e233c9315b89d8e47d6"
    )
    assert lock["source"]["engine_overlay_sha"] == (
        "4b28dd24bcf5c5240f604b7d6be78ad39d9c5e2f"
    )
    assert lock["artifacts"]["bin/moss_tts_nano_worker"]["sha256"] == (
        "9d114d8390e684c8876e2ef9e20e28ee6d4ec6ce18b81df5da3ba64c8f057deb"
    )
