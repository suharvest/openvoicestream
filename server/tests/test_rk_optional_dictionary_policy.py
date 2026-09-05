"""Bounded optional-dictionary policy for the default RK image."""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from importlib import metadata
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "deploy" / "docker" / "Dockerfile.rk"
DOCKERIGNORE = ROOT / ".dockerignore"
JAPANESE_FACTORY = ROOT / "third_party" / "rkvoice-stream" / "rkvoice_stream" / "backends" / "tts" / "kokoro_long32_frontend.py"


def _guard_function():
    source = DOCKERFILE.read_text()
    start = source.index("def reject_optional_unidic():")
    end = source.index("\nreject_optional_unidic()", start)
    tree = ast.parse(source[start:end])
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    namespace = {"importlib": importlib, "metadata": metadata}
    exec(compile(ast.Module([function], type_ignores=[]), str(DOCKERFILE), "exec"), namespace)
    return namespace["reject_optional_unidic"]


def test_default_image_guard_allows_absent_optional_dictionary(monkeypatch):
    guard = _guard_function()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    guard()


def test_dockerignore_excludes_submodule_experiment_venv():
    assert "third_party/rkvoice-stream/.venv/" in DOCKERIGNORE.read_text().splitlines()


@pytest.mark.parametrize("present", ["module", "distribution"])
def test_default_image_guard_rejects_optional_dictionary(monkeypatch, present):
    guard = _guard_function()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: types.SimpleNamespace(name=name) if present == "module" else None)

    def version(name):
        if present == "distribution":
            return "0.1.0"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", version)
    with pytest.raises(SystemExit, match="not allowed in the default RK image"):
        guard()


def test_japanese_factory_uses_jag2p_without_unidic_request(monkeypatch):
    spec = importlib.util.spec_from_file_location("kokoro_long32_frontend_policy", JAPANESE_FACTORY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    calls = []

    class FakeJAG2P:
        def __init__(self):
            calls.append("JAG2P")

        def __call__(self, text):
            return "ja", None

    real_import = module.importlib.import_module

    def fake_import(name):
        if name == "misaki.ja":
            return types.SimpleNamespace(JAG2P=FakeJAG2P)
        if name == "misaki.espeak":
            return types.SimpleNamespace(EspeakFallback=lambda **kwargs: object())
        if name == "unidic_lite":
            raise AssertionError("Japanese Kokoro factory must not request UniDic")
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    assert module.KokoroLong32Frontend._official_g2p("j")("text") == "ja"
    assert calls == ["JAG2P"]
