"""A dead gripper must be visible, and must refuse the motion.

2026-09-03, on device: a loose CAN lead left the jaw unable to enter MIT mode.
init_gripper() is best-effort by design — the arm is still worth driving — but
the failure was only ever printed, so every surface said the arm was healthy:

    [RebotArm] connected, motors enabled
    [RebotArmActuator] init_gripper failed (continuing): ...register 10...
    actuator connected (backend=rebot_arm port=None attempt=1)

/api/state reported nothing, and grasp_object ran the full
scan-plan-descend-close-lift before returning "nothing held". The operator
found it by watching the jaw not open.
"""
from __future__ import annotations

from typing import Any

import pytest

import ovs_agent.apps.voice_rebot_arm.rebot_actuator as mod
from ovs_agent.apps.voice_rebot_arm.rebot_actuator import _make_rebot_arm


class _FakeArm:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def connect(self, *a: Any, **k: Any) -> None: ...
    def init_gripper(self, *a: Any, **k: Any) -> None: ...
    def disconnect(self) -> None: ...
    def get_tcp_pose(self):
        return [0.0] * 6


class _NoGripperArm(_FakeArm):
    def init_gripper(self, *a: Any, **k: Any) -> None:
        raise RuntimeError(
            "gripper MIT mode switch failed: ensure_mode failed: "
            "register 10 not received within 1s"
        )


def _actuator(monkeypatch, arm_cls) -> Any:
    monkeypatch.setattr(mod, "resolve_serial_port", lambda *a, **k: "/dev/ttyACM1")
    monkeypatch.setattr(mod, "RebotArm", arm_cls)
    return _make_rebot_arm({"channel": "auto"})


def test_a_failed_gripper_init_still_connects_the_arm(monkeypatch) -> None:
    """Best-effort stays best-effort: the arm must still be usable."""
    act = _actuator(monkeypatch, _NoGripperArm)
    act.connect()
    assert act.robot is not None
    assert act.torque_enabled is True


def test_the_reason_survives_instead_of_only_being_printed(monkeypatch) -> None:
    act = _actuator(monkeypatch, _NoGripperArm)
    act.connect()
    assert act.gripper_ready is False
    assert "register 10" in (act.gripper_error or "")


def test_a_healthy_gripper_reports_ready(monkeypatch) -> None:
    act = _actuator(monkeypatch, _FakeArm)
    act.connect()
    assert act.gripper_ready is True
    assert act.gripper_error is None


def test_reconnect_clears_a_stale_reason(monkeypatch) -> None:
    """The reason belongs to the connection it came from, not the object."""
    act = _actuator(monkeypatch, _NoGripperArm)
    act.connect()
    assert act.gripper_error is not None
    act.disconnect()
    assert act.gripper_error is None
    monkeypatch.setattr(mod, "RebotArm", _FakeArm)   # lead re-seated
    act.connect()
    assert act.gripper_ready is True


def test_disconnected_arm_is_not_reported_gripper_ready(monkeypatch) -> None:
    act = _actuator(monkeypatch, _FakeArm)
    assert act.gripper_ready is False, "never connected"


# ── the tools must refuse rather than perform the motion blind ──────


@pytest.mark.parametrize("dispatch", ("_dispatch_grasp", "_dispatch_put_down"))
def test_grasp_and_put_down_refuse_a_dead_gripper(dispatch: str) -> None:
    """Both need the jaw. Refusing costs a sentence; not refusing costs a
    full arm cycle ending in a misleading "nothing held"."""
    import inspect

    from ovs_agent.apps.voice_rebot_arm import grasp_plugin

    src = inspect.getsource(grasp_plugin)
    assert src.count('gripper_err = getattr(actuator, "gripper_error", None)') >= 2, (
        "the gripper guard is missing from a dispatch path"
    )
    fn = getattr(grasp_plugin.GraspPlugin, dispatch, None)
    assert fn is not None, f"{dispatch} moved; re-check the guard"
    body = inspect.getsource(fn)
    assert "gripper_error" in body, f"{dispatch} does not check the gripper"
    assert "gripper unavailable" in body


def test_the_refusal_carries_success_false() -> None:
    """Otherwise the template fast-path speaks the success line on a refusal."""
    import inspect

    from ovs_agent.apps.voice_rebot_arm import grasp_plugin

    src = inspect.getsource(grasp_plugin)
    idx = 0
    while True:
        idx = src.find("gripper unavailable", idx + 1)
        if idx == -1:
            break
        window = src[max(0, idx - 300):idx]
        assert '"success": False' in window, (
            "a gripper refusal lacks success:False and would speak the "
            "success line"
        )


# ── the health endpoint ─────────────────────────────────────────────


def test_observation_server_exposes_gripper_health() -> None:
    import inspect

    from ovs_agent.plugins import actuator_observation_server as srv

    src = inspect.getsource(srv._build_app)  # noqa: SLF001
    assert '@app.get("/gripper")' in src
    # Health must NOT ride on /observation: that payload is the joint schema
    # ActionsManager validates action frames against.
    obs = src.split('@app.get("/observation")', 1)[1].split("@app.get", 1)[0]
    assert "gripper_error" not in obs


def test_a_backend_without_a_gripper_is_not_reported_broken() -> None:
    """SO-ARM and friends have no gripper_ready; absence != fault."""
    import inspect

    from ovs_agent.plugins import actuator_observation_server as srv

    src = inspect.getsource(srv._build_app)  # noqa: SLF001
    block = src.split('@app.get("/gripper")', 1)[1].split("@app.get", 1)[0]
    assert '"supported": False' in block
    assert '"ready": True' in block
