from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "start_edgellm_v091_runtime.py"
)


def _load_start_wrapper():
    spec = importlib.util.spec_from_file_location(
        "start_edgellm_v091_runtime", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_failure_never_starts_uvicorn(monkeypatch):
    wrapper = _load_start_wrapper()
    command = []
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda args, **kwargs: (
            command.extend(args) or SimpleNamespace(returncode=7)
        ),
    )
    monkeypatch.setattr(
        wrapper.os,
        "execvp",
        lambda *args: pytest.fail(f"unexpected execvp: {args}"),
    )

    assert wrapper.main() == 7
    assert wrapper.IMAGE_WORKER in command
    assert wrapper.RELEASE_LOCK in command
    assert "--skip-ldd" not in command


def test_ort_path_is_prepended_without_discarding_inherited_paths(
    monkeypatch,
):
    wrapper = _load_start_wrapper()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/base/lib:/host/lib")

    wrapper.prepend_ort_library_path()

    assert wrapper.os.environ["LD_LIBRARY_PATH"] == (
        f"{wrapper.ORT_LIBRARY_DIR}:/base/lib:/host/lib"
    )


def test_ort_path_has_no_trailing_colon_when_inherited_path_is_empty(
    monkeypatch,
):
    wrapper = _load_start_wrapper()
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    wrapper.prepend_ort_library_path()

    assert wrapper.os.environ["LD_LIBRARY_PATH"] == wrapper.ORT_LIBRARY_DIR


def test_success_execs_uvicorn_only_after_preflight(monkeypatch):
    wrapper = _load_start_wrapper()
    calls = []
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda args, **kwargs: (
            calls.append(("preflight", args)) or SimpleNamespace(returncode=0)
        ),
    )

    class ExecCalled(Exception):
        pass

    def _execvp(executable, args):
        calls.append(("exec", executable, args))
        raise ExecCalled

    monkeypatch.setattr(wrapper.os, "execvp", _execvp)

    with pytest.raises(ExecCalled):
        wrapper.main()
    assert calls[0][0] == "preflight"
    assert calls[1][0] == "exec"
    assert calls[1][2][1:3] == ["-m", "uvicorn"]
