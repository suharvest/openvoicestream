"""Server-loop client mode (#37 Phase 2-product, behind OVS_AGENT_SERVER_LOOP).

Covers the three contracts of the flag-gated server-loop mode:

  1. Advertise: on session open the agent uploads its tool schemas +
     system prompt + llm params via CLIENT_TOOL_ADVERTISE.
  2. Remote dispatch: a SERVER_TOOL_CALL frame runs the local handler and
     a CLIENT_TOOL_RESULT (correct call_id + result) is returned.
  3. Off (default): the flag resolves False, advertise is a no-op, and the
     local LLM/tool loop still runs (zero behaviour change).
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from openvoicestream_agent import Config, Session
from openvoicestream_agent.app_base import BaseApp
from openvoicestream_agent.app_mode import ModeContext, ModeManager
from openvoicestream_agent.modes import ChatMode
from openvoicestream_agent.llm.base import LLMBackend
from openvoicestream_agent.slv_client import ServerToolCall
from openvoicestream_agent.tools import ToolRegistry


# ── fakes ────────────────────────────────────────────────────────────


class FakeSLV:
    """Captures every frame the agent sends in server-loop mode."""

    def __init__(self) -> None:
        self.advertised: list[dict] = []
        self.tool_results: list[dict] = []
        self.text_frames: list[str] = []
        self.flushed = 0
        self.aborted = 0

    async def advertise_tools(self, tools, *, system_prompt=None, llm_params=None):
        self.advertised.append(
            {"tools": tools, "system_prompt": system_prompt, "llm_params": llm_params}
        )

    async def send_tool_result(self, call_id, name, *, ok, result=None, error=None):
        self.tool_results.append(
            {"id": call_id, "name": name, "ok": ok, "result": result, "error": error}
        )

    async def send_text(self, text: str) -> None:
        self.text_frames.append(text)

    async def flush_tts(self) -> None:
        self.flushed += 1

    async def abort(self) -> None:
        self.aborted += 1


class FakeLLM(LLMBackend):
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.stream_called = False

    async def stream(self, messages, **kw):  # type: ignore[override]
        self.stream_called = True
        for t in self.tokens:
            yield t


async def _noop_broadcast(*args, **kwargs):
    return None


def _make_app(server_loop: bool, registry: ToolRegistry) -> BaseApp:
    """Build a BaseApp without running __init__ (avoids audio/WS setup)."""
    cfg = Config(
        system_prompt="SYS",
        server_loop=server_loop,
        tools_max_iterations=4,
        llm_model="qwen-test",
    )
    app = BaseApp.__new__(BaseApp)
    app.config = cfg
    app.tool_registry = registry
    app.session = Session()
    app.events = type("E", (), {"emit": lambda *a, **k: None})()
    app.modes = None
    app.slv = FakeSLV()
    return app


def _ten_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for i in range(10):
        # Distinct closure per tool name.
        def _make(name):
            @reg.tool(name=name, description=f"tool {name}")
            def _h(value: str = "x") -> dict:
                return {"ran": name, "value": value}

        _make(f"tool_{i}")
    return reg


# ── 1. advertise ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_server_loop_advertises_tools_with_system_prompt():
    reg = _ten_tool_registry()
    app = _make_app(server_loop=True, registry=reg)

    await app._advertise_tools_if_server_loop()

    assert len(app.slv.advertised) == 1
    payload = app.slv.advertised[0]
    # All 10 tool schemas present, OpenAI shape.
    assert len(payload["tools"]) == 10
    names = {t["function"]["name"] for t in payload["tools"]}
    assert names == {f"tool_{i}" for i in range(10)}
    assert all(t["type"] == "function" for t in payload["tools"])
    # System prompt carried.
    assert payload["system_prompt"] == "SYS"
    # LLM params bundle carries the model + max iterations.
    assert payload["llm_params"]["model"] == "qwen-test"
    assert payload["llm_params"]["max_tool_iterations"] == 4


@pytest.mark.asyncio
async def test_off_mode_does_not_advertise():
    reg = _ten_tool_registry()
    app = _make_app(server_loop=False, registry=reg)

    await app._advertise_tools_if_server_loop()

    # Flag off → nothing sent over the WS.
    assert app.slv.advertised == []


@pytest.mark.asyncio
async def test_env_override_enables_advertise(monkeypatch):
    reg = _ten_tool_registry()
    app = _make_app(server_loop=False, registry=reg)  # config says off
    monkeypatch.setenv("OVS_AGENT_SERVER_LOOP", "1")  # env forces on

    await app._advertise_tools_if_server_loop()

    assert len(app.slv.advertised) == 1


# ── 2. remote SERVER_TOOL_CALL → CLIENT_TOOL_RESULT ──────────────────


@pytest.mark.asyncio
async def test_server_tool_call_dispatches_local_handler_and_returns_result():
    calls: list[dict] = []
    reg = ToolRegistry()

    @reg.tool(name="wave", description="wave the arm")
    def wave(side: str = "left") -> dict:
        calls.append({"side": side})
        return {"started": True, "action": "wave", "side": side}

    app = _make_app(server_loop=True, registry=reg)

    evt = ServerToolCall(id="call_x", name="wave", arguments={"side": "right"})
    # Route through the real dispatch entry point.
    await app._dispatch_one(evt)

    # Local handler actually ran with the server-supplied args.
    assert calls == [{"side": "right"}]
    # Exactly one CLIENT_TOOL_RESULT, correct call_id + result.
    assert len(app.slv.tool_results) == 1
    res = app.slv.tool_results[0]
    assert res["id"] == "call_x"
    assert res["name"] == "wave"
    assert res["ok"] is True
    assert res["result"] == {"started": True, "action": "wave", "side": "right"}


@pytest.mark.asyncio
async def test_server_tool_call_unknown_tool_returns_error_result():
    reg = ToolRegistry()
    app = _make_app(server_loop=True, registry=reg)

    evt = ServerToolCall(id="c2", name="does_not_exist", arguments={})
    await app._dispatch_one(evt)

    assert len(app.slv.tool_results) == 1
    res = app.slv.tool_results[0]
    assert res["id"] == "c2"
    assert res["ok"] is False
    assert "unknown tool" in (res["error"] or "")


@pytest.mark.asyncio
async def test_server_tool_call_handler_failure_returns_error_result():
    reg = ToolRegistry()

    @reg.tool(name="grip")
    def grip() -> dict:
        return {"success": False, "error": "serial bus unavailable"}

    app = _make_app(server_loop=True, registry=reg)

    evt = ServerToolCall(id="c3", name="grip", arguments={})
    await app._dispatch_one(evt)

    res = app.slv.tool_results[0]
    assert res["ok"] is False
    assert res["error"] == "serial bus unavailable"


# ── 3. off mode: local LLM loop still runs (zero behaviour change) ────


@pytest.mark.asyncio
async def test_off_mode_runs_local_llm_loop():
    cfg = Config(system_prompt="SYS", server_loop=False)
    slv = FakeSLV()
    llm = FakeLLM(["你", "好"])
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()

    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")

    await mgr.current.on_user_utterance(ctx, "hi")

    # Local LLM was invoked, tokens streamed, history grew — legacy path.
    assert llm.stream_called is True
    assert slv.text_frames == ["你", "好"]
    assert slv.flushed == 1
    assert session.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "你好"},
    ]


@pytest.mark.asyncio
async def test_server_loop_skips_local_llm_loop():
    cfg = Config(system_prompt="SYS", server_loop=True)
    slv = FakeSLV()
    llm = FakeLLM(["你", "好"])
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()

    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")

    await mgr.current.on_user_utterance(ctx, "hi")

    # Server owns the loop now: agent did NOT call its LLM, did NOT push
    # text to TTS, and did NOT touch local history.
    assert llm.stream_called is False
    assert slv.text_frames == []
    assert slv.flushed == 0
    assert session.history == []


# ── config flag resolution ───────────────────────────────────────────


def test_config_server_loop_default_off():
    assert Config().server_loop is False
    assert Config().server_loop_enabled() is False


def test_config_env_overrides(monkeypatch):
    cfg = Config(server_loop=False)
    monkeypatch.setenv("OVS_AGENT_SERVER_LOOP", "true")
    assert cfg.server_loop_enabled() is True
    monkeypatch.setenv("OVS_AGENT_SERVER_LOOP", "off")
    assert cfg.server_loop_enabled() is False
    # Field still wins when env unset.
    monkeypatch.delenv("OVS_AGENT_SERVER_LOOP", raising=False)
    assert Config(server_loop=True).server_loop_enabled() is True
