"""Tests for the adaptive-by-default grasp FORCE policy.

The dispatch chooses, per grasp, whether to ramp (adaptive) and what ceiling to
pass. Two modes, switched by ``adaptive_force_default``:

  • TRUE (the default): EVERY grasp ramps; the per-class value is the CEILING
    the ramp may reach (soft fruit low, box high), unlisted → global ceiling.
  • FALSE (legacy): a listed class uses its value as a FIXED force (no ramp);
    an unlisted class ramps to the global ceiling. This is byte-identical to
    the validated box demo.

SDK-free: we stub the ArmPlugin / actuator / arm and run_grasp_once so we can
inspect the (grasp_force, adaptive_force) the dispatch passes — no camera, no
CAN bus, no onnxruntime.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from ovs_agent.apps.voice_rebot_arm.grasp_plugin import GraspPlugin


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


# config close to production: box-family at the 0.8 ceiling, fruit/cup low.
_BY_CLASS = {
    "box": 0.8,
    "carton": 0.8,
    "yellow banana": 0.35,
    "orange": 0.35,
    "cup": 0.30,
    "water bottle": 0.5,
}


def _make_plugin(cfg: dict) -> tuple[GraspPlugin, _FakeActuator]:
    app = _FakeApp()
    plugin = GraspPlugin(app, cfg)
    actuator = _FakeActuator(torque=True)
    plugin._arm_plugin = _FakeArmPlugin(actuator)  # noqa: SLF001
    plugin._ensure_perception = lambda: None  # type: ignore[assignment]
    return plugin, actuator


def _capture_run_kwargs(plugin, target):
    """Dispatch a grasp with run_grasp_once stubbed; return its kwargs."""
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
            await plugin._grasp_task  # noqa: SLF001
            return res

        res = asyncio.run(_drive())
    finally:
        gs.run_grasp_once = orig
    return res, seen


# ── adaptive_force_default = TRUE (the new default) ─────────────────


def test_default_is_adaptive_when_unset() -> None:
    # No adaptive_force_default key at all → defaults to adaptive.
    plugin, _ = _make_plugin({
        "yolo_classes": ["box"],
        "grasp_force": 0.8,
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "box")
    assert seen["adaptive_force"] is True


def test_listed_soft_class_ramps_to_low_ceiling() -> None:
    # A LISTED soft class (yellow banana) → adaptive=True with the class value
    # as the ceiling (0.35), NOT a fixed force.
    plugin, _ = _make_plugin({
        "yolo_classes": ["yellow banana", "box"],
        "grasp_force": 0.8,
        "adaptive_force_default": True,
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "yellow banana")
    assert seen["adaptive_force"] is True
    assert seen["grasp_force"] == 0.35


def test_listed_box_ramps_to_high_ceiling() -> None:
    # A box → adaptive=True with the full 0.8 ceiling, so a rigid object can
    # still reach a firm grip when it keeps creeping.
    plugin, _ = _make_plugin({
        "yolo_classes": ["box"],
        "grasp_force": 0.8,
        "adaptive_force_default": True,
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "box")
    assert seen["adaptive_force"] is True
    assert seen["grasp_force"] == 0.8


def test_fixed_class_overrides_ramp_with_configured_force() -> None:
    # A rigid class in grasp_force_fixed_classes uses its configured force as a
    # FIXED grip (adaptive=False) EVEN with adaptive_force_default=True — the
    # ramp under-grips a rigid box (no compression → settles at the ~0.2 start →
    # the flush-aligned jaw slips). 2026-06-16.
    plugin, _ = _make_plugin({
        "yolo_classes": ["box"],
        "grasp_force": 0.8,
        "adaptive_force_default": True,
        "grasp_force_by_class": _BY_CLASS,
        "grasp_force_fixed_classes": ["box", "cardboard box", "carton", "package"],
    })
    _, seen = _capture_run_kwargs(plugin, "box")
    assert seen["adaptive_force"] is False
    assert seen["grasp_force"] == 0.8


def test_fixed_class_list_does_not_affect_soft_classes() -> None:
    # A soft class (banana) NOT in grasp_force_fixed_classes still ramps to its
    # low ceiling even when the box family is pinned to fixed force.
    plugin, _ = _make_plugin({
        "yolo_classes": ["yellow banana", "box"],
        "grasp_force": 0.8,
        "adaptive_force_default": True,
        "grasp_force_by_class": _BY_CLASS,
        "grasp_force_fixed_classes": ["box", "cardboard box", "carton", "package"],
    })
    _, seen = _capture_run_kwargs(plugin, "yellow banana")
    assert seen["adaptive_force"] is True
    assert seen["grasp_force"] == 0.35


def test_unlisted_class_ramps_to_global_ceiling() -> None:
    # An UNLISTED class → adaptive with the GLOBAL grasp_force as the ceiling.
    plugin, _ = _make_plugin({
        "yolo_classes": ["apple", "box"],
        "grasp_force": 0.8,
        "adaptive_force_default": True,
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "apple")
    assert seen["adaptive_force"] is True
    assert seen["grasp_force"] == 0.8


# ── adaptive_force_default = FALSE (legacy, byte-identical) ──────────


def test_legacy_box_uses_fixed_force_no_ramp() -> None:
    # adaptive_force_default=False: listed class → FIXED force, no ramp.
    # This is the validated box demo behaviour, byte-identical to today.
    plugin, _ = _make_plugin({
        "yolo_classes": ["box"],
        "grasp_force": 0.8,
        "adaptive_force_default": False,
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "box")
    assert seen["grasp_force"] == 0.8
    assert seen["adaptive_force"] is False


def test_legacy_unlisted_ramps_to_global() -> None:
    # adaptive_force_default=False: unlisted class → adaptive global ceiling.
    plugin, _ = _make_plugin({
        "yolo_classes": ["apple", "box"],
        "grasp_force": 0.8,
        "adaptive_force_default": False,
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "apple")
    assert seen["adaptive_force"] is True
    assert seen["grasp_force"] == 0.8


def test_legacy_soft_class_uses_fixed_low_force() -> None:
    # adaptive_force_default=False: a listed soft class is a FIXED low force,
    # not a ramp (the old per-class semantics).
    plugin, _ = _make_plugin({
        "yolo_classes": ["cup", "box"],
        "grasp_force": 0.8,
        "adaptive_force_default": False,
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "cup")
    assert seen["adaptive_force"] is False
    assert seen["grasp_force"] == 0.30


# ── string / env-style values flip the default ──────────────────────


@pytest.mark.parametrize("falsey", ["false", "0", "no", "off", "False"])
def test_string_falsey_disables_adaptive_default(falsey) -> None:
    # ${REBOT_ADAPTIVE_FORCE:-true} renders to a STRING in config; a falsey
    # string must flip to legacy mode (listed class → fixed force).
    plugin, _ = _make_plugin({
        "yolo_classes": ["box"],
        "grasp_force": 0.8,
        "adaptive_force_default": falsey,
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "box")
    assert seen["adaptive_force"] is False
    assert seen["grasp_force"] == 0.8


@pytest.mark.parametrize("truthy", ["true", "1", "yes", "on", "True"])
def test_string_truthy_keeps_adaptive_default(truthy) -> None:
    plugin, _ = _make_plugin({
        "yolo_classes": ["box"],
        "grasp_force": 0.8,
        "adaptive_force_default": truthy,
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "box")
    assert seen["adaptive_force"] is True
    assert seen["grasp_force"] == 0.8


# ── env override (REBOT_ADAPTIVE_FORCE) flips the rendered default ───
#
# The config renders ${REBOT_ADAPTIVE_FORCE:-true} at LOAD time, so the env var
# reaches the plugin as the adaptive_force_default value. We emulate that render
# here (the loader is exercised by the app's own config tests) to prove the env
# value, once rendered, selects the right policy.


def _render_adaptive_default() -> str:
    return os.environ.get("REBOT_ADAPTIVE_FORCE", "true")


def test_env_override_false(monkeypatch) -> None:
    monkeypatch.setenv("REBOT_ADAPTIVE_FORCE", "false")
    plugin, _ = _make_plugin({
        "yolo_classes": ["box"],
        "grasp_force": 0.8,
        "adaptive_force_default": _render_adaptive_default(),
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "box")
    assert seen["adaptive_force"] is False  # legacy: fixed force
    assert seen["grasp_force"] == 0.8


def test_env_override_default_true(monkeypatch) -> None:
    monkeypatch.delenv("REBOT_ADAPTIVE_FORCE", raising=False)
    plugin, _ = _make_plugin({
        "yolo_classes": ["box"],
        "grasp_force": 0.8,
        "adaptive_force_default": _render_adaptive_default(),
        "grasp_force_by_class": _BY_CLASS,
    })
    _, seen = _capture_run_kwargs(plugin, "box")
    assert seen["adaptive_force"] is True  # default: ramp to ceiling
