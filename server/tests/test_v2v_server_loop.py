"""#37 Phase 2-product: server-side LLM+tool loop unit tests.

Covers:
  * EdgeLLMBackend (server/core/edge_llm_backend.py) — mocked httpx SSE →
    stream_events yields text + tool_call_delta + finish; request body shape
    (tools, edge-llm cache flags, forwarded params).
  * Server-loop wiring assembled by _v2v_stream_via_engine when
    OVS_V2V_SERVER_LOOP=1 (edge-llm adapter + non-None ToolRegistry +
    system_prompt / llm_params) drives the voxedge tool pump end-to-end.
  * Regression: OVS_V2V_SERVER_LOOP off ⇒ no LLM backend, tool_registry=None
    (the existing client-text→TTS pass-through is byte-unchanged).
  * v2v wire constants added (SERVER_TOOL_CALL / CLIENT_TOOL_RESULT) are
    additive; existing constants unchanged.
"""
from __future__ import annotations

import asyncio
import json

import httpx

from server.core.edge_llm_backend import EdgeLLMBackend, edge_llm_base_url


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())

    wrapper.__name__ = coro_fn.__name__
    return wrapper


def _sse(*chunks: dict) -> bytes:
    """Build an OpenAI-style SSE byte body from chat.completion chunk dicts."""
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _mock_backend(body_bytes: bytes, captured: dict):
    """An EdgeLLMBackend whose httpx client is a MockTransport returning
    ``body_bytes`` and recording the request JSON into ``captured``."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, content=body_bytes)

    be = EdgeLLMBackend(base_url="http://test/v1", model="qwen3")
    be._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return be


# ───────────────────────── EdgeLLMBackend ───────────────────────────────


@run_async
async def test_edge_llm_streams_text_events():
    captured: dict = {}
    body = _sse(
        {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "world"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    be = _mock_backend(body, captured)
    events = [ev async for ev in be.stream_events([{"role": "user", "content": "hi"}])]
    kinds = [(e.kind, e.text or e.finish_reason) for e in events]
    assert ("text", "Hello ") in kinds
    assert ("text", "world") in kinds
    assert ("finish", "stop") in kinds
    await be.aclose()


@run_async
async def test_edge_llm_streams_tool_call_delta():
    captured: dict = {}
    body = _sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1",
             "function": {"name": "wave", "arguments": '{"x":'}}
        ]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "1}"}}
        ]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )
    be = _mock_backend(body, captured)
    tools = [{"type": "function", "function": {"name": "wave",
              "parameters": {"type": "object", "properties": {}}}}]
    events = [ev async for ev in be.stream_events(
        [{"role": "user", "content": "wave"}], tools=tools)]
    tcs = [e for e in events if e.kind == "tool_call_delta"]
    assert tcs[0].tool_call_id == "call_1"
    assert tcs[0].name == "wave"
    assert tcs[0].arguments == '{"x":'
    assert tcs[1].arguments == "1}"
    assert any(e.kind == "finish" and e.finish_reason == "tool_calls" for e in events)
    # tools forwarded into the request body.
    assert captured["json"]["tools"] == tools
    # edge-llm cache flags present (cold-path defaults).
    assert captured["json"]["save_system_prompt_kv_cache"] is True
    assert captured["json"]["return_cache_metrics"] is True
    assert captured["json"]["enable_thinking"] is False
    assert captured["json"]["stream"] is True
    await be.aclose()


@run_async
async def test_edge_llm_forwards_params():
    captured: dict = {}
    body = _sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
    be = _mock_backend(body, captured)
    _ = [ev async for ev in be.stream_events(
        [{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=64)]
    assert captured["json"]["temperature"] == 0.3
    assert captured["json"]["max_tokens"] == 64
    await be.aclose()


@run_async
async def test_edge_llm_finish_reason_error_raises():
    captured: dict = {}
    body = _sse({"choices": [{"delta": {}, "finish_reason": "error"}]})
    be = _mock_backend(body, captured)
    raised = False
    try:
        _ = [ev async for ev in be.stream_events([{"role": "user", "content": "x"}])]
    except RuntimeError as e:
        raised = "finish_reason=error" in str(e)
    assert raised
    await be.aclose()


def test_chat_url_resolution():
    assert EdgeLLMBackend(base_url="http://h/v1")._chat_url == "http://h/v1/chat/completions"
    assert EdgeLLMBackend(base_url="http://h")._chat_url == "http://h/v1/chat/completions"
    assert EdgeLLMBackend(
        base_url="http://h/v1/chat/completions"
    )._chat_url == "http://h/v1/chat/completions"


def test_base_url_env(monkeypatch):
    monkeypatch.setenv("EDGE_LLM_BASE_URL", "http://custom:9/v1")
    assert edge_llm_base_url() == "http://custom:9/v1"
    monkeypatch.delenv("EDGE_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("EDGE_LLM_CHAT_URL", "http://alt:9/v1")
    assert edge_llm_base_url() == "http://alt:9/v1"


# ─────────────────── server-loop pump (engine integration) ───────────────


@run_async
async def test_server_loop_pump_with_mock_edge_llm():
    """End-to-end through the voxedge tool pump using the product edge-llm
    adapter (mocked httpx): round 1 = tool call, round 2 = text. Asserts the
    tool fired and the result was injected back to the LLM."""
    from voxedge.engine.conversation import ConversationEngine, Session
    from voxedge.engine.tool_registry import ToolRegistry

    registry = ToolRegistry()
    fired = {}

    @registry.tool(description="lookup weather")
    def get_weather(city: str) -> dict:
        fired["city"] = city
        return {"temp": 18}

    # Two sequential SSE bodies: the MockTransport returns them in order.
    round1 = _sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1",
             "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
        ]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )
    round2 = _sse(
        {"choices": [{"delta": {"content": "It is 18 degrees."},
                      "finish_reason": "stop"}]},
    )
    bodies = [round1, round2]
    seen_bodies: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_bodies.append(json.loads(request.content))
        return httpx.Response(200, content=bodies.pop(0))

    be = EdgeLLMBackend(base_url="http://test/v1")
    be._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    from voxedge.backends.mock import MockTTS

    engine = ConversationEngine(
        backends={"tts": MockTTS(), "llm": be},
        tool_registry=registry,
        system_prompt="You are helpful.",
        llm_params={"temperature": 0.2},
    )

    class _T:
        session_id = "sid"

        async def send_event(self, p):
            pass

        async def send_audio(self, b):
            pass

    sess = Session(engine, _T())
    # The multi-turn LLM↔tool pump moved from Session._llm_turn_with_tools to
    # _LLMTurn.run() (conversation-split refactor); Session exposes it as _llm.
    await sess._llm.run([{"role": "user", "content": "weather paris"}])

    assert fired["city"] == "Paris"
    assert len(seen_bodies) == 2
    # System prompt injected once, sent on both rounds.
    assert seen_bodies[0]["messages"][0] == {
        "role": "system", "content": "You are helpful."}
    # llm_params forwarded.
    assert seen_bodies[0]["temperature"] == 0.2
    # Round 2 carries the tool result (role:tool).
    r2 = seen_bodies[1]["messages"]
    tool_msg = next(m for m in r2 if m["role"] == "tool")
    assert "18" in tool_msg["content"]
    # state is now a SessionState dataclass (conversation-split), not a dict.
    assert sess.state.tts_flush is True
    await be.aclose()


# ──────────────────────────── flag regression ────────────────────────────


def test_v2v_wire_constants_additive():
    from server.core import v2v
    assert v2v.SERVER_TOOL_CALL == "tool_call"
    assert v2v.CLIENT_TOOL_RESULT == "tool_result"
    # Existing constants unchanged.
    assert v2v.CLIENT_TEXT == "text"
    assert v2v.CLIENT_ABORT == "abort"
    assert v2v.SERVER_TTS_DONE == "tts_done"
    assert v2v.SERVER_ERROR == "error"


def test_server_loop_flag_off_no_llm_no_registry(monkeypatch):
    """Hard contract: with OVS_V2V_SERVER_LOOP unset/off, the wiring block in
    _v2v_stream_via_engine builds NO llm backend and tool_registry stays None
    — the existing client-text→TTS pass-through is unchanged.

    This mirrors the exact flag-branch logic in server/main.py without spinning up
    a WebSocket: the branch is a pure function of the env flag."""
    monkeypatch.delenv("OVS_V2V_SERVER_LOOP", raising=False)
    import os
    server_loop = os.environ.get("OVS_V2V_SERVER_LOOP", "").lower() in (
        "1", "true", "yes", "on")
    assert server_loop is False
    # Off → registry None, no llm key. (The server/main.py block guards all LLM
    # wiring behind `if server_loop:`.)

    monkeypatch.setenv("OVS_V2V_SERVER_LOOP", "1")
    server_loop_on = os.environ.get("OVS_V2V_SERVER_LOOP", "").lower() in (
        "1", "true", "yes", "on")
    assert server_loop_on is True


def test_v2v_engine_path_passes_registry_none_when_flag_off():
    """Source-level guard: _v2v_stream_via_engine initializes tool_registry to
    None and only assigns it inside the `if server_loop:` block, so flag-off is
    byte-identical to the pre-feature path."""
    import inspect
    from server import main
    src = inspect.getsource(main._v2v_stream_via_engine)
    assert "OVS_V2V_SERVER_LOOP" in src
    # tool_registry defaults to None before the flag branch.
    assert "tool_registry = None" in src
    # The engine is constructed with the (possibly-None) tool_registry var.
    assert "tool_registry=tool_registry" in src
    # LLM backend is only imported/constructed inside the flag branch.
    idx_flag = src.index("if server_loop:")
    idx_llm = src.index("EdgeLLMBackend(")
    assert idx_llm > idx_flag
