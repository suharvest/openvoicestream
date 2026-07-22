"""Tests for unknown-object rejection (YOLOE vocab-decoupling step 7).

SDK-free / device-free: we stub the ArmPlugin / actuator / arm and the
perception init, and monkeypatch run_grasp_once / run_search_once so the
heavy pipeline never runs. We exercise the resolver helper and the two
dispatch branches (unknown_object: first vs reject).
"""
from __future__ import annotations

import asyncio

import pytest

from ovs_agent.apps.voice_rebot_arm.grasp_plugin import GraspPlugin


_CATALOG = ["box", "cardboard box", "carton", "package"]


# ── fakes (mirror tests/test_grasp_plugin.py) ───────────────────────


class _FakeActuator:
    def __init__(self, torque: bool = True) -> None:
        self.torque_enabled = torque
        self.robot = object()  # non-None → "connected"


class _FakeArmPlugin:
    def __init__(self, actuator) -> None:
        self.arm = actuator


class _FakeApp:
    def __init__(self) -> None:
        self.tool_registry = None
        self.session = None
        self.plugins = []


def _make_plugin(mode: str = "first") -> GraspPlugin:
    app = _FakeApp()
    plugin = GraspPlugin(app, {"yolo_classes": list(_CATALOG), "unknown_object": mode})
    actuator = _FakeActuator(torque=True)
    plugin._arm_plugin = _FakeArmPlugin(actuator)  # noqa: SLF001
    plugin._ensure_perception = lambda: None  # type: ignore[assignment]
    return plugin


# ── resolver helper unit tests ──────────────────────────────────────


def test_resolve_exact() -> None:
    assert GraspPlugin._resolve_catalog_label("box", _CATALOG) == "box"


def test_resolve_case_insensitive() -> None:
    assert GraspPlugin._resolve_catalog_label("BOX", _CATALOG) == "box"
    assert GraspPlugin._resolve_catalog_label("Carton", _CATALOG) == "carton"


def test_resolve_substring_target_contains_label() -> None:
    # spoken "the box please" contains the catalog label "box"
    assert GraspPlugin._resolve_catalog_label("the box please", _CATALOG) == "box"


def test_resolve_substring_label_contains_target() -> None:
    # catalog label "cardboard box" contains spoken "cardboard"
    assert GraspPlugin._resolve_catalog_label("cardboard", _CATALOG) == "cardboard box"


def test_resolve_multi_word() -> None:
    assert GraspPlugin._resolve_catalog_label("cardboard box", _CATALOG) == "cardboard box"


def test_resolve_no_match_returns_none() -> None:
    assert GraspPlugin._resolve_catalog_label("apple", _CATALOG) is None


def test_resolve_empty_inputs_return_none() -> None:
    assert GraspPlugin._resolve_catalog_label("", _CATALOG) is None
    assert GraspPlugin._resolve_catalog_label("box", []) is None


# ── grasp dispatch: 'first' mode (default, preserves behaviour) ─────


def _capture_grasp(plugin, target):
    """Dispatch grasp with run_grasp_once stubbed; return (result, seen-kwargs)."""
    import ovs_agent.apps.voice_rebot_arm.grasp_service as gs

    seen = {}

    def _fake_run(tgt, **kwargs):
        seen["target"] = tgt
        seen.update(kwargs)
        return {"success": True, "grasp_pose": [0.4, 0, 0.1, 0, 0, 0]}

    orig = gs.run_grasp_once
    gs.run_grasp_once = _fake_run
    try:
        async def _drive():
            res = await plugin._dispatch_grasp(target)  # noqa: SLF001
            if plugin._grasp_task is not None:  # noqa: SLF001
                await plugin._grasp_task  # noqa: SLF001
            return res

        res = asyncio.run(_drive())
    finally:
        gs.run_grasp_once = orig
    return res, seen


def test_first_mode_unknown_falls_back_to_catalog0_and_starts() -> None:
    plugin = _make_plugin("first")
    res, seen = _capture_grasp(plugin, "apple")
    # unknown 'apple' resolves to catalog[0] and a grasp task WOULD start
    assert res["started"] is True
    assert res["target"] == "box"           # catalog[0]
    assert seen["target"] == "box"          # the pipeline got the resolved label
    assert plugin._grasp_task is not None    # noqa: SLF001


def test_first_mode_is_the_default() -> None:
    # No unknown_object key at all → behaves like 'first'.
    app = _FakeApp()
    plugin = GraspPlugin(app, {"yolo_classes": list(_CATALOG)})
    actuator = _FakeActuator(torque=True)
    plugin._arm_plugin = _FakeArmPlugin(actuator)  # noqa: SLF001
    plugin._ensure_perception = lambda: None  # type: ignore[assignment]
    assert plugin._unknown_object_mode() == "first"  # noqa: SLF001
    res, _ = _capture_grasp(plugin, "apple")
    assert res["started"] is True
    assert res["target"] == "box"


# ── grasp dispatch: 'reject' mode ───────────────────────────────────


def test_reject_mode_unknown_refuses_and_starts_no_task() -> None:
    plugin = _make_plugin("reject")

    import ovs_agent.apps.voice_rebot_arm.grasp_service as gs

    called = {"n": 0}

    def _fake_run(tgt, **kwargs):
        called["n"] += 1
        return {"success": True}

    orig = gs.run_grasp_once
    gs.run_grasp_once = _fake_run
    try:
        res = asyncio.run(plugin._dispatch_grasp("apple"))  # noqa: SLF001
    finally:
        gs.run_grasp_once = orig

    assert res["started"] is False
    assert res["unknown_object"] == "apple"
    assert "not in the graspable catalog" in res["error"]
    assert res["catalog"] == _CATALOG
    assert called["n"] == 0                  # pipeline never invoked
    assert plugin._grasp_task is None         # noqa: SLF001 — no task created


def test_reject_mode_known_exact_still_starts() -> None:
    plugin = _make_plugin("reject")
    res, seen = _capture_grasp(plugin, "box")
    assert res["started"] is True
    assert res["target"] == "box"
    assert seen["target"] == "box"


def test_reject_mode_known_substring_still_starts() -> None:
    plugin = _make_plugin("reject")
    res, seen = _capture_grasp(plugin, "the box")
    assert res["started"] is True
    assert res["target"] == "box"
    assert seen["target"] == "box"


# ── search dispatch: 'reject' mode ──────────────────────────────────


def test_search_reject_mode_unknown_refuses_and_starts_no_task() -> None:
    plugin = _make_plugin("reject")

    import ovs_agent.apps.voice_rebot_arm.grasp_service as gs

    called = {"n": 0}

    def _fake_search(tgt, **kwargs):
        called["n"] += 1
        return {"found": True}

    orig = gs.run_search_once
    gs.run_search_once = _fake_search
    try:
        res = asyncio.run(plugin._dispatch_search("apple"))  # noqa: SLF001
    finally:
        gs.run_search_once = orig

    assert res["started"] is False
    assert res["unknown_object"] == "apple"
    assert "not in the searchable catalog" in res["error"]
    assert res["catalog"] == _CATALOG
    assert called["n"] == 0
    assert plugin._grasp_task is None  # noqa: SLF001


def test_search_first_mode_unknown_falls_back_to_catalog0() -> None:
    plugin = _make_plugin("first")

    import ovs_agent.apps.voice_rebot_arm.grasp_service as gs

    seen = {}

    def _fake_search(tgt, **kwargs):
        seen["target"] = tgt
        return {"found": True}

    orig = gs.run_search_once
    gs.run_search_once = _fake_search
    try:
        async def _drive():
            res = await plugin._dispatch_search("apple")  # noqa: SLF001
            await plugin._grasp_task  # noqa: SLF001
            return res

        res = asyncio.run(_drive())
    finally:
        gs.run_search_once = orig

    assert res["started"] is True
    assert res["target"] == "box"
    assert seen["target"] == "box"
