from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


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


def test_release_lock_supplies_complete_immutable_worker_record(tmp_path: Path):
    expected = "a" * 64
    lock = tmp_path / "release-lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": {
                    "bin/moss_tts_nano_worker": {
                        "sha256": expected,
                        "size": 123,
                        "mode": "0755",
                        "required_onnxruntime_symbol_version": "VERS_1.23.2",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = MODULE.artifact_from_release_lock(
        lock, "bin/moss_tts_nano_worker"
    )
    assert artifact.sha256 == expected
    assert artifact.size == 123
    assert artifact.mode == 0o755
    assert artifact.required_onnxruntime_symbol_version == "VERS_1.23.2"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "0000"),
        ("required_onnxruntime_symbol_version", "VERS_BOGUS"),
    ],
)
def test_release_lock_rejects_tampered_worker_contract(
    tmp_path: Path, field: str, value
):
    record = {
        "sha256": "a" * 64,
        "size": 123,
        "mode": "0755",
        "required_onnxruntime_symbol_version": "VERS_1.23.2",
    }
    record[field] = value
    lock = tmp_path / "release-lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": {"bin/moss_tts_nano_worker": record},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        MODULE.artifact_from_release_lock(
            lock, "bin/moss_tts_nano_worker"
        )


def test_worker_size_must_match_release_lock(
    tmp_path: Path, monkeypatch
):
    worker = tmp_path / "moss_tts_nano_worker"
    worker.write_bytes(b"candidate")
    worker.chmod(0o755)

    def _run(command, **kwargs):
        del kwargs
        output = (
            "libonnxruntime.so.1 => /opt/ort/libonnxruntime.so.1 (0x1)\n"
            if command[0] == "ldd"
            else "                 U OrtGetApiBase@VERS_1.23.2\n"
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", _run)
    lock = tmp_path / "release-lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": {
                    "bin/moss_tts_nano_worker": {
                        "sha256": MODULE.sha256_file(worker),
                        "size": 1,
                        "mode": "0755",
                        "required_onnxruntime_symbol_version": "VERS_1.23.2",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    artifact = MODULE.artifact_from_release_lock(
        lock, "bin/moss_tts_nano_worker"
    )

    _, errors = MODULE.check_worker(worker, release_artifact=artifact)

    assert any("size mismatch" in error for error in errors)


def test_worker_mode_must_match_release_lock(tmp_path: Path, monkeypatch):
    worker = tmp_path / "moss_tts_nano_worker"
    worker.write_bytes(b"candidate")
    worker.chmod(0o700)

    def _run(command, **kwargs):
        del kwargs
        output = (
            "libonnxruntime.so.1 => /opt/ort/libonnxruntime.so.1 (0x1)\n"
            if command[0] == "ldd"
            else "                 U OrtGetApiBase@VERS_1.23.2\n"
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", _run)
    artifact = MODULE.ReleaseArtifact(
        sha256=MODULE.sha256_file(worker),
        size=worker.stat().st_size,
        mode=0o755,
        required_onnxruntime_symbol_version="VERS_1.23.2",
    )

    _, errors = MODULE.check_worker(worker, release_artifact=artifact)

    assert any("mode mismatch" in error for error in errors)


def test_worker_hash_mismatch_fails_even_when_ldd_is_clean(
    tmp_path: Path, monkeypatch
):
    worker = tmp_path / "moss_tts_nano_worker"
    worker.write_bytes(b"candidate")
    worker.chmod(0o755)
    def _run(command, **kwargs):
        del kwargs
        if command[0] == "ldd":
            output = (
                "libonnxruntime.so.1 => "
                "/usr/local/lib/onnxruntime/libonnxruntime.so.1 (0x1)\n"
            )
        else:
            output = "                 U OrtGetApiBase@VERS_1.23.2\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", _run)
    artifact = MODULE.ReleaseArtifact(
        sha256="0" * 64,
        size=len(b"candidate"),
        mode=0o755,
        required_onnxruntime_symbol_version="VERS_1.23.2",
    )

    _, errors = MODULE.check_worker(worker, release_artifact=artifact)

    assert len(errors) == 1
    assert "SHA256 mismatch" in errors[0]


def test_build_static_gate_skips_ldd_but_still_checks_nm(
    tmp_path: Path, monkeypatch
):
    worker = tmp_path / "moss_tts_nano_worker"
    worker.write_bytes(b"candidate")
    worker.chmod(0o755)
    commands = []

    def _run(command, **kwargs):
        del kwargs
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="                 U OrtGetApiBase@VERS_1.23.2\n",
            stderr="",
        )

    monkeypatch.setattr(MODULE.subprocess, "run", _run)
    artifact = MODULE.ReleaseArtifact(
        sha256=MODULE.sha256_file(worker),
        size=worker.stat().st_size,
        mode=0o755,
        required_onnxruntime_symbol_version="VERS_1.23.2",
    )

    _, errors = MODULE.check_worker(
        worker,
        release_artifact=artifact,
        run_ldd=False,
    )

    assert errors == []
    assert len(commands) == 1
    assert commands[0][0] == "nm"


def test_rejects_wrong_imported_onnxruntime_symbol_version():
    assert MODULE.validate_nm_output(
        "                 U OrtGetApiBase@VERS_1.99.0\n",
        "VERS_1.23.2",
    ) == [
        "the worker imports the wrong OrtGetApiBase symbol version: "
        "expected VERS_1.23.2, found ['VERS_1.99.0']"
    ]


def test_accepts_exact_imported_onnxruntime_symbol_version():
    assert MODULE.validate_nm_output(
        "                 U OrtGetApiBase@VERS_1.23.2\n",
        "VERS_1.23.2",
    ) == []


def test_v091_runtime_image_wires_soname_and_semantic_worker_gate():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "ln -sf libonnxruntime.so.1.23.2" in dockerfile
    assert (
        'LD_LIBRARY_PATH="${ort_dir}'
        '${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"'
    ) in dockerfile
    assert "ENV LD_LIBRARY_PATH=" not in dockerfile
    assert "scripts/check_moss_worker_runtime.py" in dockerfile
    assert "scripts/start_edgellm_v091_runtime.py" in dockerfile
    assert "deploy/artifacts/v091-release-lock.json" in dockerfile
    assert "COPY deploy/artifacts/v091-release-gate/moss_tts_nano_worker" in dockerfile
    assert "--worker /opt/edgellm-v091/bin/moss_tts_nano_worker" in dockerfile
    assert "--release-lock /opt/speech/deploy/v091-release-lock.json" in dockerfile
    assert "--skip-ldd" in dockerfile
    assert 'CMD ["python3", "/opt/speech/scripts/start_edgellm_v091_runtime.py"]' in dockerfile
    assert "MOSS_WORKER_SHA256" not in dockerfile
    assert "/opt/jv-workers/moss_tts_nano_worker" not in dockerfile


def test_v091_release_lock_pins_the_clean_formal_worker():
    lock = json.loads(RELEASE_LOCK.read_text(encoding="utf-8"))

    assert lock["schema_version"] == 1
    assert lock["artifact_set"].endswith("-20260803-r3")
    assert lock["source"]["upstream_sha"] == (
        "7f061f21f0a581ba234a1e233c9315b89d8e47d6"
    )
    assert lock["source"]["engine_overlay_sha"] == (
        "6bb19de346f468d1fc9ed1108fa94817fc42be7c"
    )
    assert lock["artifacts"]["bin/moss_tts_nano_worker"]["sha256"] == (
        "9d114d8390e684c8876e2ef9e20e28ee6d4ec6ce18b81df5da3ba64c8f057deb"
    )
    assert lock["artifacts"]["bin/moss_tts_nano_worker"]["size"] == 449864
    assert lock["artifacts"]["bin/moss_tts_nano_worker"]["mode"] == "0755"
    assert lock["artifacts"]["bin/moss_tts_nano_worker"][
        "required_onnxruntime_symbol_version"
    ] == "VERS_1.23.2"
