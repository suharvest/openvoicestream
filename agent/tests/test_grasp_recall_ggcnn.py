"""Tests for GG-CNN enable (Item B) + detection-recall levers (Item D) wiring.

SDK-free: we mock ``YoloOnnxSegmenter`` and the GG-CNN loader so the plugin's
``_ensure_perception`` runs on a Mac without onnxruntime / camera / CAN bus.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.grasp_plugin import GraspPlugin


class _FakeApp:
    def __init__(self) -> None:
        self.tool_registry = None
        self.session = None
        self.plugins = []


class _RecordingSegmenter:
    """Captures the ctor args so tests can assert names / input_size / kwargs."""

    last: dict = {}

    def __init__(self, model_path, names, **kwargs):
        type(self).last = {
            "model_path": model_path,
            "names": list(names),
            "kwargs": dict(kwargs),
        }


class _FakeCamera:
    def __init__(self, *a, **k):
        self.opened = False

    def open(self):
        self.opened = True


def _install_fakes(monkeypatch, *, ggcnn_loaded=None):
    """Patch the lazily-imported segmenter / camera / hand-eye / ggcnn so
    _ensure_perception runs without real deps. ``ggcnn_loaded`` (a list) records
    each GgcnnRefiner(model_path) construction."""
    import ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx as yo
    import ovs_agent.apps.voice_rebot_arm.perception.camera as cam_mod

    _RecordingSegmenter.last = {}
    monkeypatch.setattr(yo, "YoloOnnxSegmenter", _RecordingSegmenter)
    monkeypatch.setattr(cam_mod, "make_camera", lambda *a, **k: _FakeCamera())

    # Mock the ggcnn_refiner module (only imported when enabled).
    mod = types.ModuleType(
        "ovs_agent.apps.voice_rebot_arm.perception.ggcnn_refiner"
    )

    class _FakeGgcnn:
        def __init__(self, model_path):
            self.model_path = model_path
            if ggcnn_loaded is not None:
                ggcnn_loaded.append(model_path)

    mod.GgcnnRefiner = _FakeGgcnn
    monkeypatch.setitem(
        sys.modules,
        "ovs_agent.apps.voice_rebot_arm.perception.ggcnn_refiner",
        mod,
    )


def _make_plugin(cfg) -> GraspPlugin:
    p = GraspPlugin(_FakeApp(), config=cfg)
    # Skip hand-eye file load.
    p._load_hand_eye = lambda: np.eye(4)  # type: ignore[assignment]
    return p


_BASE = {
    "yolo_model_path": "/tmp/box.engine",
    "yolo_classes": ["box", "carton", "package"],
    "hand_eye_path": "/tmp/he.npz",
}


# ── Item B: GG-CNN enable + env override ─────────────────────────────────────
def test_ggcnn_enabled_truthy_string():
    p = _make_plugin({**_BASE, "ggcnn_refiner": "true"})
    assert p._ggcnn_enabled() is True
    p2 = _make_plugin({**_BASE, "ggcnn_refiner": "false"})
    assert p2._ggcnn_enabled() is False
    p3 = _make_plugin({**_BASE, "ggcnn_refiner": True})
    assert p3._ggcnn_enabled() is True


def test_ggcnn_loaded_when_enabled(monkeypatch):
    loaded: list = []
    _install_fakes(monkeypatch, ggcnn_loaded=loaded)
    p = _make_plugin({**_BASE, "ggcnn_refiner": True})
    p._ensure_perception()
    assert p._ggcnn is not None
    assert len(loaded) == 1  # the refiner was constructed


def test_ggcnn_not_loaded_when_disabled(monkeypatch):
    loaded: list = []
    _install_fakes(monkeypatch, ggcnn_loaded=loaded)
    p = _make_plugin({**_BASE, "ggcnn_refiner": False})
    p._ensure_perception()
    assert p._ggcnn is None
    assert loaded == []


def test_ggcnn_custom_model_path(monkeypatch):
    loaded: list = []
    _install_fakes(monkeypatch, ggcnn_loaded=loaded)
    p = _make_plugin(
        {**_BASE, "ggcnn_refiner": True, "ggcnn_model_path": "/tmp/custom.onnx"}
    )
    p._ensure_perception()
    assert loaded == ["/tmp/custom.onnx"]


# ── Item D #2: REBOT_GRASP_CLASSES env override ──────────────────────────────
def test_classes_env_override(monkeypatch):
    _install_fakes(monkeypatch)
    monkeypatch.setenv("REBOT_GRASP_CLASSES", '["banana", "orange", "cup"]')
    p = _make_plugin({**_BASE, "ggcnn_refiner": False})
    p._ensure_perception()
    assert _RecordingSegmenter.last["names"] == ["banana", "orange", "cup"]


def test_classes_env_malformed_falls_back_to_config(monkeypatch):
    _install_fakes(monkeypatch)
    monkeypatch.setenv("REBOT_GRASP_CLASSES", "not-json{")
    p = _make_plugin({**_BASE, "ggcnn_refiner": False})
    p._ensure_perception()
    assert _RecordingSegmenter.last["names"] == ["box", "carton", "package"]


def test_classes_env_unset_uses_config(monkeypatch):
    _install_fakes(monkeypatch)
    monkeypatch.delenv("REBOT_GRASP_CLASSES", raising=False)
    p = _make_plugin({**_BASE, "ggcnn_refiner": False})
    p._ensure_perception()
    assert _RecordingSegmenter.last["names"] == ["box", "carton", "package"]


def test_classes_env_non_list_json_falls_back(monkeypatch):
    _install_fakes(monkeypatch)
    monkeypatch.setenv("REBOT_GRASP_CLASSES", '{"a": 1}')  # valid JSON, not list
    p = _make_plugin({**_BASE, "ggcnn_refiner": False})
    p._ensure_perception()
    assert _RecordingSegmenter.last["names"] == ["box", "carton", "package"]


# ── Item D #3: yolo_input_size passthrough ───────────────────────────────────
def test_input_size_unset_keeps_default(monkeypatch):
    _install_fakes(monkeypatch)
    p = _make_plugin({**_BASE, "ggcnn_refiner": False})
    p._ensure_perception()
    # Unset → NOT passed → segmenter keeps its own 640 default.
    assert "input_size" not in _RecordingSegmenter.last["kwargs"]


def test_input_size_int_passes_square(monkeypatch):
    _install_fakes(monkeypatch)
    p = _make_plugin({**_BASE, "ggcnn_refiner": False, "yolo_input_size": 800})
    p._ensure_perception()
    assert _RecordingSegmenter.last["kwargs"]["input_size"] == (800, 800)


def test_input_size_pair_passes_through(monkeypatch):
    _install_fakes(monkeypatch)
    p = _make_plugin({**_BASE, "ggcnn_refiner": False, "yolo_input_size": [960, 540]})
    p._ensure_perception()
    assert _RecordingSegmenter.last["kwargs"]["input_size"] == (960, 540)


def test_input_size_malformed_ignored(monkeypatch):
    _install_fakes(monkeypatch)
    p = _make_plugin({**_BASE, "ggcnn_refiner": False, "yolo_input_size": "big"})
    p._ensure_perception()
    assert "input_size" not in _RecordingSegmenter.last["kwargs"]
