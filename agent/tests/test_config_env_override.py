"""Tests for env-overridable table-calibrated grasp config values.

`place_bounds` and `scan_poses` are baked into the image's config.yaml as YAML
lists, which can't use the ``${VAR:-default}`` scalar substitution. The plugin's
``_list_override`` helper lets a JSON env var (``REBOT_PLACE_BOUNDS`` /
``REBOT_SCAN_POSES``) override the baked default at container-recreate time so a
table re-calibration needs no image rebuild.

SDK-free / Mac-verifiable: we build a GraspPlugin with a plain config dict and
exercise the helper plus the param-build paths that read these keys. We never
touch onnxruntime / camera / CAN bus. Style mirrors test_grasp_plugin.py.
"""
from __future__ import annotations

import pytest

from ovs_agent.apps.voice_rebot_arm.grasp_plugin import GraspPlugin


# ── fakes ───────────────────────────────────────────────────────────


class _FakeApp:
    def __init__(self) -> None:
        self.tool_registry = None
        self.session = None
        self.plugins = []


_DEFAULT_BOUNDS = [0.20, 0.60, -0.26, 0.40]
_DEFAULT_POSES = [
    [0.27, 0.00, 0.30, 0.0, 0.30, 0.0],
    [0.25, 0.10, 0.30, 0.0, 0.30, 0.35],
]


def _make_plugin() -> GraspPlugin:
    plugin = GraspPlugin(
        _FakeApp(),
        config={"place_bounds": _DEFAULT_BOUNDS, "scan_poses": _DEFAULT_POSES},
    )
    return plugin


# ── _list_override helper ───────────────────────────────────────────


def test_list_override_unset_returns_config(monkeypatch) -> None:
    monkeypatch.delenv("REBOT_PLACE_BOUNDS", raising=False)
    plugin = _make_plugin()
    assert plugin._list_override("REBOT_PLACE_BOUNDS", "place_bounds") == _DEFAULT_BOUNDS  # noqa: SLF001


def test_list_override_empty_returns_config(monkeypatch) -> None:
    monkeypatch.setenv("REBOT_PLACE_BOUNDS", "")
    plugin = _make_plugin()
    assert plugin._list_override("REBOT_PLACE_BOUNDS", "place_bounds") == _DEFAULT_BOUNDS  # noqa: SLF001
    monkeypatch.setenv("REBOT_PLACE_BOUNDS", "   ")
    assert plugin._list_override("REBOT_PLACE_BOUNDS", "place_bounds") == _DEFAULT_BOUNDS  # noqa: SLF001


def test_list_override_valid_json_used(monkeypatch) -> None:
    monkeypatch.setenv("REBOT_PLACE_BOUNDS", "[0.1,0.7,-0.3,0.5]")
    plugin = _make_plugin()
    assert plugin._list_override("REBOT_PLACE_BOUNDS", "place_bounds") == [0.1, 0.7, -0.3, 0.5]  # noqa: SLF001


def test_list_override_bad_json_falls_back_and_warns(monkeypatch, caplog) -> None:
    monkeypatch.setenv("REBOT_PLACE_BOUNDS", "not json")
    plugin = _make_plugin()
    with caplog.at_level("WARNING"):
        out = plugin._list_override("REBOT_PLACE_BOUNDS", "place_bounds")  # noqa: SLF001
    assert out == _DEFAULT_BOUNDS
    assert any("REBOT_PLACE_BOUNDS" in r.message for r in caplog.records)


def test_list_override_non_list_falls_back_and_warns(monkeypatch, caplog) -> None:
    monkeypatch.setenv("REBOT_PLACE_BOUNDS", '{"a": 1}')
    plugin = _make_plugin()
    with caplog.at_level("WARNING"):
        out = plugin._list_override("REBOT_PLACE_BOUNDS", "place_bounds")  # noqa: SLF001
    assert out == _DEFAULT_BOUNDS
    assert any("REBOT_PLACE_BOUNDS" in r.message for r in caplog.records)


# ── place_bounds via the put_down kwargs build ──────────────────────


def _build_place_bounds(plugin: GraspPlugin):
    """Mirror the bounds-selection + validation in _dispatch_put_down."""
    pb = plugin._list_override("REBOT_PLACE_BOUNDS", "place_bounds")  # noqa: SLF001
    if not pb:
        return None
    bounds = [float(v) for v in pb]
    return bounds if len(bounds) == 4 else None


def test_place_bounds_env_used(monkeypatch) -> None:
    monkeypatch.setenv("REBOT_PLACE_BOUNDS", "[0.1,0.7,-0.3,0.5]")
    plugin = _make_plugin()
    assert _build_place_bounds(plugin) == [0.1, 0.7, -0.3, 0.5]


def test_place_bounds_unset_uses_config(monkeypatch) -> None:
    monkeypatch.delenv("REBOT_PLACE_BOUNDS", raising=False)
    plugin = _make_plugin()
    assert _build_place_bounds(plugin) == _DEFAULT_BOUNDS


def test_place_bounds_malformed_not_json_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("REBOT_PLACE_BOUNDS", "not json")
    plugin = _make_plugin()
    # Helper falls back to config default → downstream length check passes.
    assert _build_place_bounds(plugin) == _DEFAULT_BOUNDS


def test_place_bounds_wrong_length_rejected_downstream(monkeypatch) -> None:
    # Valid JSON list but wrong length: helper returns it, downstream 4-value
    # check rejects → no place_bounds applied (no crash).
    monkeypatch.setenv("REBOT_PLACE_BOUNDS", "[1,2,3]")
    plugin = _make_plugin()
    assert _build_place_bounds(plugin) is None


# ── scan_poses via _search_params and _grasp_params ─────────────────


def test_scan_poses_env_used_in_search_params(monkeypatch) -> None:
    monkeypatch.setenv("REBOT_SCAN_POSES", "[[0.3,0,0.3,0,0.3,0]]")
    plugin = _make_plugin()
    params = plugin._search_params()  # noqa: SLF001
    assert params["scan_poses"] == [(0.3, 0.0, 0.3, 0.0, 0.3, 0.0)]


def test_scan_poses_unset_uses_config_in_search_params(monkeypatch) -> None:
    monkeypatch.delenv("REBOT_SCAN_POSES", raising=False)
    plugin = _make_plugin()
    params = plugin._search_params()  # noqa: SLF001
    assert params["scan_poses"] == [tuple(p) for p in _DEFAULT_POSES]


def test_scan_poses_malformed_falls_back_in_search_params(monkeypatch) -> None:
    monkeypatch.setenv("REBOT_SCAN_POSES", "not json")
    plugin = _make_plugin()
    params = plugin._search_params()  # noqa: SLF001
    assert params["scan_poses"] == [tuple(p) for p in _DEFAULT_POSES]


def test_scan_poses_env_used_in_grasp_params(monkeypatch) -> None:
    monkeypatch.setenv("REBOT_SCAN_POSES", "[[0.3,0,0.3,0,0.3,0]]")
    plugin = _make_plugin()
    out = plugin._grasp_params()  # noqa: SLF001
    assert out["scan_poses"] == [(0.3, 0.0, 0.3, 0.0, 0.3, 0.0)]


def test_scan_poses_unset_uses_config_in_grasp_params(monkeypatch) -> None:
    monkeypatch.delenv("REBOT_SCAN_POSES", raising=False)
    plugin = _make_plugin()
    out = plugin._grasp_params()  # noqa: SLF001
    assert out["scan_poses"] == [tuple(p) for p in _DEFAULT_POSES]


# ── yolo_classes via REBOT_GRASP_CLASSES (detection recall, Item D) ──────────
# Same _list_override helper, exercised at the level the segmenter build reads.

_DEFAULT_CLASSES = ["box", "cardboard box", "carton", "package"]


def _make_classes_plugin() -> GraspPlugin:
    return GraspPlugin(_FakeApp(), config={"yolo_classes": _DEFAULT_CLASSES})


def test_grasp_classes_env_override(monkeypatch) -> None:
    monkeypatch.setenv("REBOT_GRASP_CLASSES", '["banana", "orange"]')
    plugin = _make_classes_plugin()
    assert plugin._list_override("REBOT_GRASP_CLASSES", "yolo_classes") == [  # noqa: SLF001
        "banana", "orange",
    ]


def test_grasp_classes_unset_returns_config(monkeypatch) -> None:
    monkeypatch.delenv("REBOT_GRASP_CLASSES", raising=False)
    plugin = _make_classes_plugin()
    assert plugin._list_override("REBOT_GRASP_CLASSES", "yolo_classes") == _DEFAULT_CLASSES  # noqa: SLF001


def test_grasp_classes_malformed_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("REBOT_GRASP_CLASSES", "not-json{")
    plugin = _make_classes_plugin()
    assert plugin._list_override("REBOT_GRASP_CLASSES", "yolo_classes") == _DEFAULT_CLASSES  # noqa: SLF001
