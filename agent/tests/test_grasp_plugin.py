"""Tests for GraspPlugin dispatch guards (Phase B safety).

SDK-free: we stub the ArmPlugin / actuator / arm and the perception init so we
can exercise the torque gate, the re-entry guard, and the stop() lifecycle on
a developer Mac. We never touch onnxruntime / camera / CAN bus.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from ovs_agent.apps.voice_rebot_arm.grasp_plugin import GraspPlugin
from ovs_agent.tools import ToolRegistry


# ── fakes ───────────────────────────────────────────────────────────


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


def _make_plugin(torque: bool = True) -> tuple[GraspPlugin, _FakeActuator]:
    app = _FakeApp()
    plugin = GraspPlugin(app)
    actuator = _FakeActuator(torque=torque)
    plugin._arm_plugin = _FakeArmPlugin(actuator)  # noqa: SLF001
    # Skip the real perception init (onnxruntime / camera).
    plugin._ensure_perception = lambda: None  # type: ignore[assignment]
    return plugin, actuator


# ── torque gate (item 2) ────────────────────────────────────────────


def test_dispatch_refused_when_torque_off() -> None:
    plugin, _ = _make_plugin(torque=False)
    res = asyncio.run(plugin._dispatch_grasp("banana"))  # noqa: SLF001
    assert res["started"] is False
    assert res["error"] == "torque disabled"


# ── re-entry guard (item 8) ─────────────────────────────────────────


def test_dispatch_rejects_reentry_while_running() -> None:
    plugin, _ = _make_plugin(torque=True)

    # First dispatch: stub the runner so the task stays "in flight".
    gate = threading.Event()

    async def _slow_runner() -> dict:
        # Block until the test releases it.
        while not gate.is_set():
            await asyncio.sleep(0.01)
        return {"success": True}

    async def _drive() -> tuple[dict, dict]:
        # Patch run via monkeypatching the _runner path: easiest is to replace
        # _dispatch_grasp's worker by pre-seeding a live task.
        plugin._grasp_task = asyncio.create_task(_slow_runner())  # noqa: SLF001
        # Second dispatch must be rejected because one is in flight.
        second = await plugin._dispatch_grasp("bottle")  # noqa: SLF001
        gate.set()
        await plugin._grasp_task  # noqa: SLF001
        return second, {}

    second, _ = asyncio.run(_drive())
    assert second["started"] is False
    assert second["error"] == "already_running"


def test_dispatch_allows_after_previous_done() -> None:
    # A COMPLETED previous task must NOT trigger the re-entry guard. We make
    # _ensure_perception raise a sentinel so dispatch returns right AFTER the
    # guard — proving the guard let us through (error is the perception error,
    # not "already_running").
    plugin, _ = _make_plugin(torque=True)

    async def _already_done() -> dict:
        return {"success": True}

    def _boom_perception() -> None:
        raise RuntimeError("perception-sentinel")

    plugin._ensure_perception = _boom_perception  # type: ignore[assignment]

    async def _drive() -> dict:
        done = asyncio.create_task(_already_done())
        await done
        plugin._grasp_task = done  # noqa: SLF001 — completed task
        return await plugin._dispatch_grasp("banana")  # noqa: SLF001

    res = asyncio.run(_drive())
    assert res["started"] is False
    assert res["error"] == "perception-sentinel"   # passed the re-entry guard


# ── empty / unavailable arm ─────────────────────────────────────────


def test_dispatch_empty_object_rejected() -> None:
    plugin, _ = _make_plugin(torque=True)
    res = asyncio.run(plugin._dispatch_grasp("   "))  # noqa: SLF001
    assert res["started"] is False
    assert "empty" in res["error"]


def test_dispatch_arm_not_connected() -> None:
    plugin, actuator = _make_plugin(torque=True)
    actuator.robot = None  # not connected
    res = asyncio.run(plugin._dispatch_grasp("banana"))  # noqa: SLF001
    assert res["started"] is False
    assert res["error"] == "arm not connected"


def test_setup_does_not_register_tool_when_disabled() -> None:
    app = _FakeApp()
    app.tool_registry = ToolRegistry()
    plugin = GraspPlugin(app, {"enabled": False})

    assert plugin.setup() is True
    assert not app.tool_registry.has("grasp_object")


# ── stop() lifecycle (item 9) ───────────────────────────────────────


# ── _load_hand_eye key variants ─────────────────────────────────────


def test_load_hand_eye_reads_T_hand_eye_key(tmp_path) -> None:
    import numpy as np
    npz = tmp_path / "he.npz"
    mat = np.eye(4, dtype=np.float64)
    np.savez(str(npz), T_hand_eye=mat)
    plugin = GraspPlugin(_FakeApp(), {"hand_eye_path": str(npz)})
    result = plugin._load_hand_eye()  # noqa: SLF001
    assert result is not None
    np.testing.assert_array_equal(result, mat)


def test_load_hand_eye_reads_T_result_key(tmp_path) -> None:
    """collect_handeye_eih.py saves key 'T_result'; loader must find it."""
    import numpy as np
    npz = tmp_path / "hand_eye.npz"
    mat = np.eye(4, dtype=np.float64) * 2
    np.savez(str(npz), T_result=mat)
    plugin = GraspPlugin(_FakeApp(), {"hand_eye_path": str(npz)})
    result = plugin._load_hand_eye()  # noqa: SLF001
    assert result is not None
    np.testing.assert_array_equal(result, mat)


def test_load_hand_eye_prefers_T_hand_eye_over_T_result(tmp_path) -> None:
    import numpy as np
    npz = tmp_path / "he.npz"
    mat_a = np.eye(4, dtype=np.float64)
    mat_b = np.eye(4, dtype=np.float64) * 3
    np.savez(str(npz), T_result=mat_b, T_hand_eye=mat_a)
    plugin = GraspPlugin(_FakeApp(), {"hand_eye_path": str(npz)})
    result = plugin._load_hand_eye()  # noqa: SLF001
    np.testing.assert_array_equal(result, mat_a)


def test_load_hand_eye_missing_path_returns_none() -> None:
    plugin = GraspPlugin(_FakeApp(), {})
    assert plugin._load_hand_eye() is None  # noqa: SLF001


def test_load_hand_eye_nonexistent_file_returns_none(tmp_path) -> None:
    plugin = GraspPlugin(_FakeApp(), {"hand_eye_path": str(tmp_path / "nope.npz")})
    assert plugin._load_hand_eye() is None  # noqa: SLF001


# ── stop() lifecycle (item 9) ───────────────────────────────────────


def test_stop_sets_cancel_waits_and_closes_camera() -> None:
    plugin, _ = _make_plugin(torque=True)

    closed = {"n": 0}

    class _FakeCam:
        def close(self) -> None:
            closed["n"] += 1

    plugin._camera = _FakeCam()  # noqa: SLF001

    observed_cancel = {"set": False}

    async def _worker() -> dict:
        # Simulate a grasp that watches the cancel event and returns promptly.
        for _ in range(200):
            if plugin._cancel_event.is_set():  # noqa: SLF001
                observed_cancel["set"] = True
                return {"cancelled": True}
            await asyncio.sleep(0.01)
        return {"cancelled": False}

    async def _drive() -> None:
        plugin._grasp_task = asyncio.create_task(_worker())  # noqa: SLF001
        await asyncio.sleep(0.05)
        await plugin.stop()

    asyncio.run(_drive())
    assert observed_cancel["set"] is True       # cancel event was set
    assert plugin._grasp_task is None           # noqa: SLF001 — cleared
    assert closed["n"] == 1                      # camera closed exactly once
    assert plugin._camera is None                # noqa: SLF001
