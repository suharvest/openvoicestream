"""Tests for the LLMEvent stream_events channel + back-compat .stream()."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ovs_agent.llm import LLMEvent
from ovs_agent.llm.openai_compat import (
    LLMStreamError,
    OpenAICompatBackend,
)


# ── upstream chunk fakes (mirrors test_llm_retry.py shape) ────────────


def _req() -> httpx.Request:
    return httpx.Request("POST", "http://example.invalid/v1/chat/completions")


class _Fn:
    def __init__(self, name: str | None = None, arguments: str | None = None):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(
        self,
        index: int = 0,
        id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ):
        self.index = index
        self.id = id
        self.function = _Fn(name=name, arguments=arguments)


class _Delta:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[_TC] | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[_TC] | None = None,
        finish_reason: str | None = None,
    ):
        self.delta = _Delta(content=content, tool_calls=tool_calls)
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[_TC] | None = None,
        finish_reason: str | None = None,
    ):
        self.choices = [_Choice(content, tool_calls, finish_reason)]
        self.model_extra: dict[str, Any] = {}


class _AsyncChunks:
    def __init__(self, script: list[Any]):
        self._script = list(script)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._script):
            raise StopAsyncIteration
        item = self._script[self._i]
        self._i += 1
        if isinstance(item, BaseException):
            raise item
        return item


class _ScriptedCompletions:
    def __init__(self, plan: list[Any]):
        self._plan = list(plan)
        self.calls = 0
        self.kwargs_history: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.kwargs_history.append(kwargs)
        if not self._plan:
            raise RuntimeError("scripted plan exhausted")
        nxt = self._plan.pop(0)
        self.calls += 1
        if isinstance(nxt, BaseException):
            raise nxt
        if isinstance(nxt, list):
            return _AsyncChunks(nxt)
        return nxt


class _FakeClient:
    def __init__(self, plan: list[Any]):
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(plan))

    async def close(self):
        return None


def _backend(plan: list[Any], **kw: Any) -> OpenAICompatBackend:
    b = OpenAICompatBackend(
        base_url="http://example.invalid/v1",
        api_key="EMPTY",
        model="fake",
        retry_on_transient=kw.pop("retry_on_transient", 0),
        retry_backoff_s=0.0,
        **kw,
    )
    b.client = _FakeClient(plan)  # type: ignore[assignment]
    return b


# ── tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_only_stream_yields_text_then_finish():
    plan = [[_Chunk(content="hello"), _Chunk(content=" world", finish_reason="stop")]]
    b = _backend(plan)
    events = [ev async for ev in b.stream_events([{"role": "user", "content": "hi"}])]
    kinds = [ev.kind for ev in events]
    assert kinds == ["text", "text", "finish"]
    assert events[0].text == "hello"
    assert events[1].text == " world"
    assert events[2].finish_reason == "stop"


@pytest.mark.asyncio
async def test_agent_internal_session_is_not_forwarded_to_openai_sdk():
    plan = [[_Chunk(content="ready"), _Chunk(finish_reason="stop")]]
    b = _backend(plan)
    session = object()

    events = [
        ev
        async for ev in b.stream_events(
            [{"role": "user", "content": "hi"}], session=session
        )
    ]

    assert [ev.kind for ev in events] == ["text", "finish"]
    call = b.client.chat.completions.kwargs_history[0]  # type: ignore[union-attr]
    assert "session" not in call


@pytest.mark.asyncio
async def test_tool_only_stream_accumulation_by_index():
    """Tool-only response: name+id come on first chunk, args fragmented."""
    plan = [[
        _Chunk(tool_calls=[_TC(index=0, id="call_1", name="get_time", arguments="")]),
        _Chunk(tool_calls=[_TC(index=0, arguments='{"tz":')]),
        _Chunk(tool_calls=[_TC(index=0, arguments='"UTC"}')]),
        _Chunk(finish_reason="tool_calls"),
    ]]
    b = _backend(plan)
    events = [ev async for ev in b.stream_events([{"role": "user", "content": "hi"}])]
    deltas = [ev for ev in events if ev.kind == "tool_call_delta"]
    assert len(deltas) == 3
    assert deltas[0].tool_call_id == "call_1"
    assert deltas[0].name == "get_time"
    args_concat = "".join(d.arguments or "" for d in deltas)
    assert args_concat == '{"tz":"UTC"}'
    finish = events[-1]
    assert finish.kind == "finish" and finish.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_mixed_text_and_tool_stream_in_order():
    plan = [[
        _Chunk(content="thinking..."),
        _Chunk(tool_calls=[_TC(index=0, id="c1", name="f", arguments="{}")]),
        _Chunk(finish_reason="tool_calls"),
    ]]
    b = _backend(plan)
    events = [ev async for ev in b.stream_events([{"role": "user", "content": "hi"}])]
    kinds = [ev.kind for ev in events]
    assert kinds == ["text", "tool_call_delta", "finish"]


@pytest.mark.asyncio
async def test_finish_reason_error_raises_llm_stream_error():
    plan = [[_Chunk(content="partial"), _Chunk(content=None, finish_reason="error")]]
    b = _backend(plan)
    yielded: list[LLMEvent] = []
    with pytest.raises(LLMStreamError):
        async for ev in b.stream_events([{"role": "user", "content": "hi"}]):
            yielded.append(ev)
    # The partial text was delivered before the error frame.
    assert any(ev.kind == "text" and ev.text == "partial" for ev in yielded)


@pytest.mark.asyncio
async def test_back_compat_stream_yields_text_only():
    plan = [[
        _Chunk(content="hi"),
        _Chunk(tool_calls=[_TC(index=0, id="c1", name="f", arguments="{}")]),
        _Chunk(content=" there", finish_reason="stop"),
    ]]
    b = _backend(plan)
    toks = [t async for t in b.stream([{"role": "user", "content": "x"}])]
    assert toks == ["hi", " there"]


@pytest.mark.asyncio
async def test_tools_kwarg_forwarded_to_request():
    plan = [[_Chunk(content="ok", finish_reason="stop")]]
    b = _backend(plan)
    tools = [{"type": "function", "function": {"name": "get_time", "parameters": {}}}]
    _ = [ev async for ev in b.stream_events(
        [{"role": "user", "content": "x"}], tools=tools
    )]
    completions = b.client.chat.completions  # type: ignore[attr-defined]
    assert completions.kwargs_history[0].get("tools") == tools


@pytest.mark.asyncio
async def test_tools_none_kwarg_not_forwarded():
    plan = [[_Chunk(content="ok", finish_reason="stop")]]
    b = _backend(plan)
    _ = [ev async for ev in b.stream_events(
        [{"role": "user", "content": "x"}], tools=None
    )]
    completions = b.client.chat.completions  # type: ignore[attr-defined]
    assert "tools" not in completions.kwargs_history[0]


@pytest.mark.asyncio
async def test_edge_llm_back_compat_stream_supports_session_kwarg():
    """The session= kwarg in EdgeLLMBackend.stream must keep working."""
    from ovs_agent.llm import EdgeLLMBackend
    from ovs_agent.session import Session

    b = EdgeLLMBackend(
        base_url="http://example.invalid/v1",
        api_key="EMPTY",
        model="fake",
        retry_on_transient=0,
        retry_backoff_s=0.0,
    )
    b.client = _FakeClient([[_Chunk(content="hi", finish_reason="stop")]])  # type: ignore[assignment]
    session = Session()
    toks = [t async for t in b.stream([{"role": "user", "content": "x"}], session=session)]
    assert toks == ["hi"]
    assert session.cache_warmed is True
