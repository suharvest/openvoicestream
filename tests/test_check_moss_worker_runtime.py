from __future__ import annotations

import importlib.util
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


def test_v091_runtime_image_wires_soname_and_semantic_worker_gate():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "ln -sf libonnxruntime.so.1.23.2" in dockerfile
    assert "scripts/check_moss_worker_runtime.py" in dockerfile
    assert "--worker /opt/jv-workers/moss_tts_nano_worker" in dockerfile
