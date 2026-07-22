"""Cross-plugin / cross-action motion mutex + completion tone.

The arm has ONE serial bus and (on the rebot actuator) a PER-OP motion lock,
so two concurrent waypoint sequences interleave frame-by-frame into garbage
motion. These tests pin the race fixes:

  * ArmPlugin.dispatch_action refuses a DIFFERENT action while one runs
    (the old guard was per-name only — "wave" + "go_home" ran concurrently).
  * ArmPlugin.dispatch_action refuses while an external motion source (the
    grasp pipeline) reports busy, and vice versa: GraspPlugin dispatches
    refuse while a static action runs.
  * Same-name re-dispatch keeps its historical fast-ack shape
    ({"started": True, "already_running": True}).
  * GraspPlugin plays a completion tone when a motion finishes (success vs
    failure pitch) and stays silent on user-cancelled motions.
"""
from __future__ import annotations

import asyncio
import threading

from ovs_agent.apps.voice_rebot_arm.grasp_plugin import GraspPlugin
from ovs_agent.plugins.actuator_actions import ArmPlugin


# ── fakes ───────────────────────────────────────────────────────────


class _FakeActions:
    def get_sequence(self, name):
        return [{"joints": {"gripper": 0.0}, "delay": 0.01}]


class _SlowArm:
    """Actuator stub whose execute_sequence blocks until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls: list[str] = []

    def execute_sequence(self, frames) -> bool:
        self.calls.append("seq")
        self.release.wait(timeout=5.0)
        return True


def _make_arm_plugin() -> tuple[ArmPlugin, _SlowArm]:
    plugin = ArmPlugin.__new__(ArmPlugin)  # skip setup-heavy __init__ deps
    ArmPlugin.__init__(plugin, app=object(), config={"actions_yaml_path": "/dev/null"})
    arm = _SlowArm()
    plugin.arm = arm
    plugin.actions = _FakeActions()
    return plugin, arm


class _FakeApp:
    def __init__(self) -> None:
        self.tool_registry = None
        self.session = None
        self.plugins = []


def _make_grasp_plugin(arm_plugin=None) -> GraspPlugin:
    plugin = GraspPlugin(_FakeApp())
    plugin._arm_plugin = arm_plugin  # noqa: SLF001
    plugin._ensure_perception = lambda: None  # type: ignore[assignment]
    return plugin


# ── ArmPlugin: cross-name + external mutex ──────────────────────────


def test_dispatch_action_refuses_different_action_while_running() -> None:
    plugin, arm = _make_arm_plugin()

    async def _drive() -> tuple[dict, dict]:
        first = await plugin.dispatch_action("wave")
        second = await plugin.dispatch_action("go_home")
        arm.release.set()
        task = plugin._inflight_tasks.get("wave")  # noqa: SLF001
        if task is not None:
            await task
        return first, second

    first, second = asyncio.run(_drive())
    assert first == {"started": True, "action": "wave"}
    assert second["started"] is False
    assert "wave" in second["error"]
    # only ONE sequence ever reached the bus.
    assert arm.calls == ["seq"]


def test_dispatch_action_same_name_keeps_fast_ack_shape() -> None:
    plugin, arm = _make_arm_plugin()

    async def _drive() -> tuple[dict, dict]:
        first = await plugin.dispatch_action("wave")
        second = await plugin.dispatch_action("wave")
        arm.release.set()
        task = plugin._inflight_tasks.get("wave")  # noqa: SLF001
        if task is not None:
            await task
        return first, second

    _, second = asyncio.run(_drive())
    assert second == {"started": True, "action": "wave", "already_running": True}


def test_dispatch_action_refuses_while_external_source_busy() -> None:
    plugin, arm = _make_arm_plugin()
    plugin.register_motion_source(lambda: "grasp-box")

    async def _drive() -> dict:
        return await plugin.dispatch_action("wave")

    res = asyncio.run(_drive())
    assert res["started"] is False
    assert "grasp-box" in res["error"]
    assert arm.calls == []


def test_busy_action_reports_own_then_external() -> None:
    plugin, arm = _make_arm_plugin()
    assert plugin.busy_action() is None
    flag = {"busy": None}
    plugin.register_motion_source(lambda: flag["busy"])
    assert plugin.busy_action() is None
    flag["busy"] = "grasp-box"
    assert plugin.busy_action() == "grasp-box"


# ── GraspPlugin: refuses while a static action runs ─────────────────


class _BusyArmPlugin:
    """ArmPlugin stub: connected, torque on, busy_action configurable."""

    def __init__(self, busy=None) -> None:
        self._busy = busy
        self.arm = type(
            "_A", (), {"robot": object(), "torque_enabled": True}
        )()
        self.registered = []

    def busy_action(self):
        return self._busy

    def register_motion_source(self, check) -> None:
        self.registered.append(check)


def test_grasp_refused_while_static_action_runs() -> None:
    plugin = _make_grasp_plugin(_BusyArmPlugin(busy="wave"))
    res = asyncio.run(plugin._dispatch_grasp("box"))  # noqa: SLF001
    assert res["started"] is False
    assert res["error"].startswith("arm_busy")
    assert res["current"] == "wave"


def test_put_down_refused_while_static_action_runs() -> None:
    plugin = _make_grasp_plugin(_BusyArmPlugin(busy="point_at"))
    res = asyncio.run(plugin._dispatch_put_down())  # noqa: SLF001
    assert res["started"] is False
    assert res["error"].startswith("arm_busy")
    assert res["current"] == "point_at"


def test_grasp_own_slot_busy_reports_motion_name() -> None:
    plugin = _make_grasp_plugin(_BusyArmPlugin(busy=None))

    async def _slow() -> dict:
        await asyncio.sleep(0.2)
        return {"success": True}

    async def _drive() -> dict:
        plugin._grasp_task = asyncio.create_task(_slow(), name="grasp-box")  # noqa: SLF001
        res = await plugin._dispatch_put_down()  # noqa: SLF001
        await plugin._grasp_task  # noqa: SLF001
        return res

    res = asyncio.run(_drive())
    assert res["started"] is False
    assert res["error"].startswith("already_running")
    assert res["current"] == "grasp-box"


def test_start_registers_motion_source_with_arm_plugin() -> None:
    arm_plugin = _BusyArmPlugin()
    arm_plugin.__class__.__name__ = "ArmPlugin"  # found by start()'s scan
    app = _FakeApp()
    app.plugins = [arm_plugin]
    plugin = GraspPlugin(app)
    asyncio.run(plugin.start())
    assert len(arm_plugin.registered) == 1
    # the registered checker reflects the in-flight grasp task name.
    check = arm_plugin.registered[0]
    assert check() is None

    async def _drive() -> None:
        async def _slow() -> dict:
            await asyncio.sleep(0.05)
            return {}

        plugin._grasp_task = asyncio.create_task(_slow(), name="search-box")  # noqa: SLF001
        assert check() == "search-box"
        await plugin._grasp_task  # noqa: SLF001

    asyncio.run(_drive())


# ── completion tone ─────────────────────────────────────────────────


class _ToneApp(_FakeApp):
    def __init__(self) -> None:
        super().__init__()
        self.played: list[bytes] = []
        self._local_output_mic_suppress_until = 0.0

        class _Audio:
            output_sr = 16000

            def play_notification(_self, pcm: bytes) -> None:
                self.played.append(pcm)

        self.audio = _Audio()


def _done_task(result: dict) -> asyncio.Task:
    async def _r() -> dict:
        return result

    async def _make() -> asyncio.Task:
        t = asyncio.create_task(_r())
        await t
        return t

    return asyncio.run(_make())


def test_done_tone_played_on_success_and_failure_not_on_cancel() -> None:
    app = _ToneApp()
    plugin = GraspPlugin(app)

    plugin._on_grasp_done(_done_task({"success": True}))  # noqa: SLF001
    assert len(app.played) == 1
    plugin._on_grasp_done(_done_task({"success": False, "error": "x"}))  # noqa: SLF001
    assert len(app.played) == 2
    # success tone (higher pitch) differs from the failure tone.
    assert app.played[0] != app.played[1]
    # user-cancelled → silent (the user stopped it; they know).
    plugin._on_grasp_done(_done_task({"success": False, "cancelled": True}))  # noqa: SLF001
    assert len(app.played) == 2
    # mic suppressed while the tone plays.
    assert app._local_output_mic_suppress_until > 0.0  # noqa: SLF001


def test_done_tone_disabled_by_config() -> None:
    app = _ToneApp()
    plugin = GraspPlugin(app, {"done_tone": {"ms": 0}})
    plugin._on_grasp_done(_done_task({"success": True}))  # noqa: SLF001
    assert app.played == []


def test_done_tone_kinds_distinct():
    app = _ToneApp()
    plugin = GraspPlugin(app)
    plugin._on_grasp_done(_done_task({"success": True}))                                  # ok
    plugin._on_grasp_done(_done_task({"success": False, "error": "gripper closed but nothing held", "stage": "grasp"}))   # fail
    plugin._on_grasp_done(_done_task({"success": False, "error": "no valid grasp for target 'box'", "stage": "detect"}))  # not_found
    plugin._on_grasp_done(_done_task({"success": False, "error": "implausible jaw width 0.166m", "stage": "plausibility"}))  # out_of_range
    assert len(app.played) == 4
    # all four waveforms differ (length and/or content)
    assert len({(len(p), bytes(p[:64])) for p in app.played}) == 4
    # not_found (double beep with gap) is the longest of the fixed-freq ones
    assert len(app.played[2]) > len(app.played[1])


# ── deferred "your turn" tone (失败播报后再响, 2026-06-13) ──────────────


def test_failure_tone_deferred_until_assistant_done() -> None:
    """A failure WITH a spoken reason defers the completion tone until the
    reply finishes (on_assistant_done), so the tone is always the LAST output
    — the reliable 'your turn to speak' signal."""

    async def _drive() -> None:
        app = _ToneApp()  # no slv → _announce no-ops, but the defer still runs
        plugin = GraspPlugin(app)

        async def _r() -> dict:
            return {"success": False, "error": "no valid grasp",
                    "found": False, "target": "盒子"}

        task = asyncio.create_task(_r())
        await task
        plugin._on_grasp_done(task)  # noqa: SLF001

        # Tone NOT played yet — deferred behind the spoken reason.
        assert app.played == []
        assert plugin._pending_ready_tone == "not_found"  # noqa: SLF001

        # SLV finished the spoken reason → tone plays now.
        await plugin.on_assistant_done()
        assert len(app.played) == 1
        assert plugin._pending_ready_tone is None  # noqa: SLF001

        for t in asyncio.all_tasks() - {asyncio.current_task()}:
            t.cancel()

    asyncio.run(_drive())


def test_success_tone_is_immediate_not_deferred() -> None:
    """Success has no spoken reason, so the tone plays immediately (it already
    IS the last output) and nothing is left pending."""

    async def _drive() -> None:
        app = _ToneApp()
        plugin = GraspPlugin(app)

        async def _r() -> dict:
            return {"success": True, "grasp_pose": [0.0] * 6}

        task = asyncio.create_task(_r())
        await task
        plugin._on_grasp_done(task)  # noqa: SLF001

        assert len(app.played) == 1
        assert plugin._pending_ready_tone is None  # noqa: SLF001

    asyncio.run(_drive())


def test_failure_tone_fallback_fires_without_assistant_done() -> None:
    """If no tts_done ever arrives, the bounded fallback still plays the
    deferred tone so the user is never left without the signal."""

    async def _drive() -> None:
        import ovs_agent.apps.voice_rebot_arm.grasp_plugin as gp

        app = _ToneApp()
        plugin = GraspPlugin(app)
        plugin._pending_ready_tone = "fail"  # noqa: SLF001

        orig = gp.asyncio.sleep

        async def _fast(_s: float) -> None:
            return None

        gp.asyncio.sleep = _fast  # type: ignore[assignment]
        try:
            await plugin._ready_tone_fallback("fail")  # noqa: SLF001
        finally:
            gp.asyncio.sleep = orig  # type: ignore[assignment]

        assert len(app.played) == 1
        assert plugin._pending_ready_tone is None  # noqa: SLF001

    asyncio.run(_drive())


def test_assistant_done_noop_when_no_pending_tone() -> None:
    """A normal TTS reply (e.g. the success preamble) must not trigger a stray
    tone when nothing is pending."""

    async def _drive() -> None:
        app = _ToneApp()
        plugin = GraspPlugin(app)
        await plugin.on_assistant_done()
        assert app.played == []

    asyncio.run(_drive())
