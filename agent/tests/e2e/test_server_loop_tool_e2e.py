"""Server-loop tool chain on the LIVE dev orin-nx SLV (TC-001/002 + MT-008).

Closes the production blind spot: prod reBot runs **server-loop** (the SLV's LLM
selects tools and proxies execution back via SERVER_TOOL_CALL), but every other
e2e is client-loop, so the full tool path had zero real-SLV coverage. These
drive it end to end against the real SLV + real edge-llm LLM:

    in-process agent advertises tools (CLIENT_TOOL_ADVERTISE)
      → real SLV /v2v server-loop runs the real edge-llm LLM
      → the LLM picks the advertised tool
      → SLV sends SERVER_TOOL_CALL back
      → agent._handle_server_tool_call dispatches it against the local registry
         (== the stubs below record the call)

The assertion face is the stub tool being invoked — the dashboard /ws does NOT
expose SERVER_TOOL_CALL, but the in-process agent's own registry does. Fresh
registries holding only stubs are swapped in so the advertise payload is
controlled and the command has an unambiguous target.

Scenarios:
  * TC-001  test_server_loop_tool_chain            — single tool advertised, a
            command dispatches it.
  * TC-002  test_server_loop_selects_correct_tool  — TWO tools advertised; the
            command must select the right one and NOT the other.
  * TC-003  test_server_loop_sequential_tools      — two command turns dispatch
            two different tools in order on one session.
  * MT-008  test_server_loop_chat_then_command     — a chitchat turn fires NO
            tool (false-trigger guard); a following command turn fires it, on
            one persistent server-loop session.
  * MT-013  test_server_loop_mixed_chat_command_one_sentence — one utterance
            mixing greeting + command still dispatches the tool.
  * ER     test_server_loop_tool_failure_is_graceful — a refusing tool
            ({"started": False}) resolves to an assistant reply, no hang/crash.
  * ER     test_server_loop_server_side_tool_timeout — a hung tool (never
            returns) is timed out server-side by the SLV; the turn doesn't wedge.
  * MT-011 test_server_loop_coreference_put_back — grab X, then "put it back"
            resolves to put_down (object carried across turns).
  * TC     test_server_loop_tool_failure_retry_bounded — an always-failing tool
            retries within the iteration cap and ends (no infinite loop).

────────────────────────────────────────────────────────────────────────────
GATED: skipped unless ``OVS_E2E_SERVER_LOOP=1``. The dev orin-nx SLV runs
client-loop by default; these only pass when it has been flipped to server-loop.
Setup (verified 2026-06-14):

  1. On orin-nx, clone the relaunch script and add the server-loop env. The SLV
     is on a bridge network, so the edge-llm LLM must be reached via the host
     gateway, NOT localhost (edge_llm_base_url() defaults to the container's own
     127.0.0.1:8000 — server/core/edge_llm_backend.py:51):
       cp ~/relaunch_seeed_voice.sh ~/relaunch_serverloop.sh
       # in the edgellm branch's `docker run`, add:
       #   -e OVS_V2V_ENGINE=voxedge -e OVS_V2V_SERVER_LOOP=1 \
       #   -e EDGE_LLM_BASE_URL=http://172.17.0.1:8000/v1
       bash ~/relaunch_serverloop.sh edgellm jetson-qwen3asr-matcha-nx
  2. Run:
       OVS_E2E_SERVER_LOOP=1 \
       OVS_E2E_SLV_URL='ws://<device>:8621/v2v/stream' \
       env -u http_proxy -u https_proxy \
         uv run pytest tests/e2e/test_server_loop_tool_e2e.py -v -s
  3. ALWAYS restore client-loop afterwards (shared dev box):
       bash ~/relaunch_seeed_voice.sh edgellm jetson-qwen3asr-matcha-nx
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import replace

import aiohttp
import pytest

from .conftest import WAV_DIR
from .fake_audio import ScriptedAudioIO
from .probe import AgentProbe
from .test_rebot_voice_capture import _rebot_voice_config

pytestmark = pytest.mark.skipif(
    os.environ.get("OVS_E2E_SERVER_LOOP") != "1",
    reason=(
        "requires the dev orin-nx SLV flipped to server-loop "
        "(OVS_V2V_ENGINE=voxedge + OVS_V2V_SERVER_LOOP=1 + EDGE_LLM_BASE_URL "
        "→ host gateway); see module docstring. Set OVS_E2E_SERVER_LOOP=1 to run."
    ),
)


def _stub_registry(
    sink: list[dict],
    tools: tuple[str, ...] = ("grasp_object",),
    *,
    fail_tools: tuple[str, ...] = (),
    hang_tools: tuple[str, ...] = (),
):
    """Build a registry of recording stubs. Each dispatch appends
    ``{"tool": name, "args": {...}}`` to ``sink`` (the assertion surface).
    ``fail_tools`` → returns ``{"started": False, ...}`` (ok=False). ``hang_tools``
    → the handler awaits 60s (registered timeout_s=60 so the agent-side registry
    does NOT short it) — used to exercise the SLV's own server-side tool timeout."""
    from ovs_agent.tools.registry import ToolRegistry

    reg = ToolRegistry()

    if "grasp_object" in tools:
        @reg.tool(
            name="grasp_object",
            description=(
                "抓取/拿起指定的物体。Grab/pick up a named object. "
                "Triggers: 抓/拿/抓起, grab, pick up, grasp."
            ),
            preamble_text="好的，正在抓取。",
            timeout_s=60.0,
        )
        async def grasp_object(object_name: str = "box") -> dict:  # noqa: ANN001
            sink.append({"tool": "grasp_object", "args": {"object_name": object_name}})
            if "grasp_object" in hang_tools:
                await asyncio.sleep(60)  # never returns in time → SLV must time out
            if "grasp_object" in fail_tools:
                return {"started": False, "error": "perception: object not found"}
            return {"started": True, "object_name": object_name}

    if "put_down" in tools:
        @reg.tool(
            name="put_down",
            description=(
                "把当前拿着的物体放下/放回。Put down / put back the held object. "
                "Triggers: 放下/放回/放回去, put it back, put down, release it."
            ),
            preamble_text="好的，正在放回。",
        )
        def put_down() -> dict:
            sink.append({"tool": "put_down", "args": {}})
            if "put_down" in fail_tools:
                return {"started": False, "error": "nothing held"}
            return {"started": True, "action": "put_down"}

    if "go_home" in tools:
        @reg.tool(
            name="go_home",
            description="让机械臂回到原位/home 姿态。当用户要求回家、回原位时调用。",
            preamble_text="好的，正在回原位。",
        )
        def go_home() -> dict:
            sink.append({"tool": "go_home", "args": {}})
            if "go_home" in fail_tools:
                return {"started": False, "error": "arm offline"}
            return {"started": True, "action": "go_home"}

    return reg


async def _wake(port: int) -> None:
    async with aiohttp.ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{port}/api/control/wake") as r:
            assert r.status == 200


async def _wait_for(pred, timeout: float) -> bool:
    """Poll ``pred()`` until truthy or timeout. Returns whether it fired."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.3)
    return False


@asynccontextmanager
async def _server_loop_agent(test_config, registry):
    """Spawn an in-process reBot agent in server-loop mode with ``registry``
    advertised, wake it, and yield ``(app, audio, probe)``. Mirrors the proven
    single-turn harness; handles teardown."""
    cfg = replace(_rebot_voice_config(test_config), server_loop=True)

    from ovs_agent.apps.multi_mode.app import MultiModeApp

    app = MultiModeApp(cfg)
    app.tool_registry = registry  # advertise our stubs at boot

    audio = ScriptedAudioIO([])
    app.audio = audio
    ready = asyncio.Event()
    audio.ready_event = ready

    async def _wait_advertise_ready() -> None:
        while True:
            ev = getattr(app, "_advertise_ready", None)
            if ev is not None:
                await ev.wait()
                break
            await asyncio.sleep(0.02)
        ready.set()

    run_task = asyncio.create_task(app.run(), name="server-loop-e2e-run")
    ready_task = asyncio.create_task(_wait_advertise_ready())
    probe = AgentProbe(port=cfg.metadata["dashboard_port"])
    try:
        await probe.connect()
        await _wake(cfg.metadata["dashboard_port"])
        await probe.wait_event("on_wake", timeout=10)
        await asyncio.sleep(0.6)  # clear wake-tone mic suppression
        yield app, audio, probe
    finally:
        ready_task.cancel()
        try:
            await ready_task
        except BaseException:
            pass
        try:
            app.request_shutdown()
        except Exception:
            pass
        for closer in (audio.close, probe.close):
            try:
                await closer()
            except Exception:
                pass
        try:
            await asyncio.wait_for(run_task, timeout=8)
        except BaseException:
            run_task.cancel()
            try:
                await run_task
            except BaseException:
                pass


def _assistant_done_count(probe) -> int:
    return sum(1 for e in probe.events if e.get("event") == "on_assistant_done")


# ── TC-001: single advertised tool → command dispatches it ────────────


@pytest.mark.asyncio
async def test_server_loop_tool_chain(test_config):
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object",))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        audio.inject(WAV_DIR / "tts_q_grab_box.wav")  # "抓盒子"
        fired = await _wait_for(lambda: bool(calls), timeout=45)
        evts = [e.get("event") for e in probe.events][-30:]
        assert fired, f"server-loop did not dispatch the tool. calls={calls} events={evts}"
        assert calls[0]["tool"] == "grasp_object", calls


# ── TC-002: two tools advertised → the RIGHT one is selected ──────────


@pytest.mark.asyncio
async def test_server_loop_selects_correct_tool(test_config):
    """TC-002: with both ``grasp_object`` and ``go_home`` advertised, "抓盒子"
    must select grasp_object and NOT go_home. Guards against the LLM firing the
    wrong tool when several are available."""
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object", "go_home"))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        audio.inject(WAV_DIR / "tts_q_grab_box.wav")  # "抓盒子" → grasp, not home
        fired = await _wait_for(lambda: bool(calls), timeout=45)
        assert fired, f"no tool dispatched. calls={calls}"
        names = [c["tool"] for c in calls]
        assert "grasp_object" in names, f"expected grasp_object, got {names}"
        assert "go_home" not in names, f"go_home wrongly fired: {names}"


# ── MT-008: chitchat fires no tool; a following command does ──────────


@pytest.mark.asyncio
async def test_server_loop_chat_then_command(test_config):
    """MT-008: on one persistent server-loop session, a chitchat turn ("你好")
    must NOT fire a tool (false-trigger guard), while a following command turn
    ("抓盒子") must. Both the dialogue path and the tool path share the WS."""
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object",))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        # Turn 1 — chitchat. Expect an assistant reply, NO tool call.
        audio.inject(WAV_DIR / "hello.wav")  # "你好"
        got_reply = await _wait_for(lambda: _assistant_done_count(probe) >= 1, timeout=45)
        assert got_reply, "chitchat turn produced no assistant reply"
        assert calls == [], f"chitchat must not fire a tool, but did: {calls}"

        # Turn 2 — command. Expect grasp_object to fire.
        audio.inject(WAV_DIR / "tts_q_grab_box.wav")  # "抓盒子"
        fired = await _wait_for(lambda: bool(calls), timeout=45)
        evts = [e.get("event") for e in probe.events][-30:]
        assert fired, f"command turn did not dispatch the tool. calls={calls} events={evts}"
        assert calls[0]["tool"] == "grasp_object", calls


# ── TC-003: two commands in sequence → two tools, in order ────────────


@pytest.mark.asyncio
async def test_server_loop_sequential_tools(test_config):
    """TC-003: two command turns on one server-loop session dispatch two
    different tools in order — "抓盒子" → grasp_object, then "回家" → go_home.
    Proves sequential tool dispatch over a persistent session (each turn's
    SERVER_TOOL_CALL resolved before the next), no cross-talk."""
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object", "go_home"))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        audio.inject(WAV_DIR / "tts_q_grab_box.wav")  # "抓盒子"
        assert await _wait_for(lambda: any(c["tool"] == "grasp_object" for c in calls),
                               timeout=45), f"turn 1 grasp not dispatched. calls={calls}"
        # Turn 1's tool fires mid-turn, BEFORE its TTS reply plays. Injecting
        # turn 2 now would land it in turn-1's echo-gated playback window and it
        # would never finalize. Wait for turn 1 to fully complete (assistant
        # reply done) + a drain margin before the next utterance.
        assert await _wait_for(lambda: _assistant_done_count(probe) >= 1, timeout=45), (
            "turn 1 never completed its assistant reply"
        )
        await asyncio.sleep(1.0)  # clear the post-TTS echo-gate / mic-suppress tail

        audio.inject(WAV_DIR / "tts_q_go_home.wav")  # "回家"
        assert await _wait_for(lambda: any(c["tool"] == "go_home" for c in calls),
                               timeout=45), f"turn 2 go_home not dispatched. calls={calls}"

        order = [c["tool"] for c in calls]
        assert order.index("grasp_object") < order.index("go_home"), (
            f"tools fired out of order: {order}"
        )


# ── MT-013: one sentence mixing greeting + command still fires the tool ─


@pytest.mark.asyncio
async def test_server_loop_mixed_chat_command_one_sentence(test_config):
    """MT-013: a single utterance mixing chitchat and a command
    ("你好，帮我把盒子抓起来") must still dispatch the command's tool. Lenient on
    the greeting (LLM may or may not verbalize it); the load-bearing assertion
    is that the embedded command is honored, not swallowed by the chat framing."""
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object",))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        audio.inject(WAV_DIR / "cmd_hello_grab.wav")  # "你好，帮我把盒子抓起来"
        fired = await _wait_for(lambda: bool(calls), timeout=45)
        utt = [e.get("data") for e in probe.events if e.get("event") == "on_user_utterance"]
        assert fired, f"mixed chat+command did not dispatch the tool. utt={utt} calls={calls}"
        assert calls[0]["tool"] == "grasp_object", calls


# ── ER: a failing tool surfaces gracefully (no hang/crash) ────────────


@pytest.mark.asyncio
async def test_server_loop_tool_failure_is_graceful(test_config):
    """A tool that refuses ({"started": False}) must surface as ok=False to the
    server-loop without hanging or crashing the turn: the tool fires, and the
    LLM loop still terminates with an assistant reply (it tells the user). We
    assert graceful termination, NOT a specific retry count (model-dependent) —
    the retry *mechanics* are pinned deterministically by TC-006."""
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object",), fail_tools=("grasp_object",))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        audio.inject(WAV_DIR / "tts_q_grab_box.wav")  # "抓盒子" → stub refuses
        fired = await _wait_for(lambda: bool(calls), timeout=45)
        assert fired, f"tool never dispatched. calls={calls}"
        # The turn must still resolve (assistant reply), not hang on the failure.
        replied = await _wait_for(lambda: _assistant_done_count(probe) >= 1, timeout=45)
        assert replied, (
            f"failing tool did not resolve into an assistant reply (possible "
            f"hang). calls={calls} errors={probe.errors}"
        )


# ── ER: server-side tool timeout — a hung tool must not wedge the SLV turn ─


@pytest.mark.asyncio
async def test_server_loop_server_side_tool_timeout(test_config):
    """Server-side tool timeout: the tool never returns (awaits 60s, registered
    timeout_s=60 so the AGENT-side registry does NOT short it). The SLV's own
    server-side tool_result budget (~15s) must fire and end the turn, rather than
    wedging forever. Asserts the turn terminates (assistant reply OR FSM back to
    a resting state) within the budget."""
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object",), hang_tools=("grasp_object",))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        audio.inject(WAV_DIR / "cmd_en_grab_box.wav")  # "grab the box"
        assert await _wait_for(lambda: bool(calls), timeout=30), (
            f"tool never dispatched: {calls}"
        )
        # Tool is now hung. The SLV must time it out server-side and resolve the
        # turn — NOT hang. Recovery = an assistant reply OR FSM back to rest.
        def _recovered() -> bool:
            if _assistant_done_count(probe) >= 1:
                return True
            sh = probe.state_history
            return bool(sh) and sh[-1][1] in ("idle", "listening", "sleeping")
        recovered = await _wait_for(_recovered, timeout=35)
        evts = [e.get("event") for e in probe.events][-30:]
        assert recovered, (
            f"SLV did not recover from a hung tool within the server budget "
            f"(possible server-side timeout gap). state_history={probe.state_history[-5:]} "
            f"events={evts}"
        )


# ── MT-011: coreference — grab X, then 'put it back' → put_down ───────


@pytest.mark.asyncio
async def test_server_loop_coreference_put_back(test_config):
    """MT-011: grab an object, then 'put it back' must resolve to put_down — the
    LLM carries the just-grasped object across turns on one server-loop session.
    Real-LLM semantic; generous timeout."""
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object", "put_down"))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        audio.inject(WAV_DIR / "cmd_en_grab_box.wav")  # "grab the box"
        assert await _wait_for(
            lambda: any(c["tool"] == "grasp_object" for c in calls), timeout=45
        ), f"turn 1 grasp not dispatched: {calls}"
        assert await _wait_for(lambda: _assistant_done_count(probe) >= 1, timeout=45), (
            "turn 1 produced no assistant reply"
        )
        await asyncio.sleep(1.0)  # let turn-1 TTS finish (echo-gate clear)
        audio.inject(WAV_DIR / "cmd_en_put_back.wav")  # "put it back"
        assert await _wait_for(
            lambda: any(c["tool"] == "put_down" for c in calls), timeout=45
        ), f"'put it back' did not resolve to put_down (coreference miss): {calls}"


# ── TC: tool-failure retry is bounded (no infinite loop) ──────────────


@pytest.mark.asyncio
async def test_server_loop_tool_failure_retry_bounded(test_config):
    """An ALWAYS-failing tool ({"started": False}): the server-loop LLM may retry,
    but must give up within the iteration cap and end the turn with an assistant
    reply — never an infinite retry loop. Asserts the tool fired ≥1, bounded, and
    the turn terminated. Records the observed retry count (model behavior)."""
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object",), fail_tools=("grasp_object",))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        audio.inject(WAV_DIR / "cmd_en_grab_box.wav")  # "grab the box" → always fails
        assert await _wait_for(lambda: bool(calls), timeout=45), "tool never dispatched"
        assert await _wait_for(lambda: _assistant_done_count(probe) >= 1, timeout=60), (
            f"failing tool did not terminate into a reply (possible retry loop). "
            f"calls so far={len(calls)}"
        )
        await asyncio.sleep(1.0)  # let any in-flight retry settle
        print(f"\n[server-loop-e2e] always-failing tool retry count = {len(calls)}")
        assert 1 <= len(calls) <= 8, f"retry count out of bounds (cap regression?): {len(calls)}"
