"""Tests for LLMBackend.warmup() — P9/P10 architecture down-shift.

Covers:
 1. Base default: ``LLMBackend.warmup()`` returns ``{}`` and never raises.
 2. ``OpenAICompatBackend`` drains one hidden one-token completion to prime
    its persistent cloud connection without publishing output.
 3. ``EdgeLLMBackend.warmup()`` posts to ``/v1/cache/system_prompt`` AND
    ``/v1/chat/completions`` with the real-shape body (prefix_cache=True,
    stream=True, max_tokens=1, tools included), and the response is fully
    drained.
 4. Fail-open: cache POST raising still returns a dict (cache_warmed=False).
 5. Empty system_prompt short-circuits with no HTTP calls.
"""
from __future__ import annotations

import json
import pytest

from ovs_agent.llm import (
    EdgeLLMBackend,
    LLMBackend,
    OpenAICompatBackend,
)


# ── Base default ────────────────────────────────────────────────────


class _DummyBackend(LLMBackend):
    """Concrete LLMBackend just so we can instantiate base for the test."""

    async def stream(self, messages, **kw):  # pragma: no cover - unused
        if False:
            yield ""


@pytest.mark.asyncio
async def test_base_warmup_default_is_noop():
    b = _DummyBackend()
    result = await b.warmup(system_prompt="hi", tools=[{"x": 1}])
    assert result == {}


@pytest.mark.asyncio
async def test_openai_compat_warmup_drains_hidden_one_token_completion(monkeypatch):
    b = OpenAICompatBackend(
        base_url="http://unreachable.invalid/v1",
        api_key="sk-test",
        model="x",
        default_params={"extra_body": {"request_timeout": 5}},
    )
    observed = {}

    async def fake_do_stream(messages, **kwargs):
        observed["messages"] = messages
        observed["kwargs"] = kwargs
        yield object()

    monkeypatch.setattr(b, "_do_stream", fake_do_stream)
    result = await b.warmup(
        system_prompt="hello world",
        tools=[{"type": "function", "function": {"name": "wave"}}],
    )
    assert result["graph_warmed"] is True
    assert observed["messages"] == [
        {"role": "system", "content": "hello world"},
        {"role": "user", "content": "."},
    ]
    assert observed["kwargs"]["max_tokens"] == 1
    assert observed["kwargs"]["temperature"] == 0.0
    assert observed["kwargs"]["extra_body"] == {
        "request_timeout": 5,
        "enable_thinking": False,
    }


@pytest.mark.asyncio
async def test_openai_compat_warmup_is_fail_open(monkeypatch):
    b = OpenAICompatBackend(
        base_url="http://unreachable.invalid/v1",
        api_key="sk-test",
        model="x",
    )

    async def failing_stream(messages, **kwargs):
        raise RuntimeError("offline")
        yield  # pragma: no cover

    monkeypatch.setattr(b, "_do_stream", failing_stream)
    result = await b.warmup(system_prompt="hello")
    assert result["graph_warmed"] is False


# ── EdgeLLMBackend.warmup() ────────────────────────────────────────


class _FakeStreamResponse:
    """Async context manager that mimics httpx.AsyncClient.stream(...)."""

    def __init__(self, lines):
        self._lines = list(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Records every POST + stream call for assertion."""

    calls: list[dict]

    def __init__(self, *a, **kw):
        # Reuse the class-level recorder so tests can inspect across
        # multiple context-manager instances.
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        type(self).calls.append({"method": "POST", "url": url, "body": json})
        # Mimic the server saying "messages branch accepted".
        return _FakeResponse({"messages_branch": True, "prompt_chars": 123})

    def stream(self, method, url, json=None):
        type(self).calls.append(
            {"method": method, "url": url, "body": json, "stream": True}
        )
        # Two SSE lines + an empty terminator.
        return _FakeStreamResponse(
            ['data: {"choices":[{"delta":{"content":"x"}}]}', "data: [DONE]", ""]
        )


@pytest.mark.asyncio
async def test_edge_llm_warmup_posts_cache_then_completion(monkeypatch):
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(
        "ovs_agent.llm.edge_llm.httpx.AsyncClient",
        _FakeAsyncClient,
    )

    b = EdgeLLMBackend(
        base_url="http://edge-llm:8000/v1",
        api_key="sk-test",
        model="qwen3-4b-awq",
    )
    tools = [{"type": "function", "function": {"name": "wave"}}]
    result = await b.warmup(
        system_prompt="You are a robot.",
        tools=tools,
        enable_thinking=False,
    )

    assert result["cache_warmed"] is True
    assert result["graph_warmed"] is True
    assert result["cache_warmup_ms"] >= 0
    assert result["graph_warmup_ms"] >= 0

    # Exactly 2 calls expected: cache POST, then chat completions stream.
    assert len(_FakeAsyncClient.calls) == 2

    cache_call = _FakeAsyncClient.calls[0]
    assert cache_call["url"].endswith("/v1/cache/system_prompt")
    assert cache_call["body"]["prefix_cache"] is True
    assert cache_call["body"]["enable_thinking"] is False
    assert cache_call["body"]["tools"] == tools
    assert cache_call["body"]["messages"][0]["role"] == "system"

    chat_call = _FakeAsyncClient.calls[1]
    assert chat_call["url"].endswith("/v1/chat/completions")
    assert chat_call.get("stream") is True
    assert chat_call["body"]["stream"] is True
    assert chat_call["body"]["max_tokens"] == 1
    assert chat_call["body"]["prefix_cache"] is True
    assert chat_call["body"]["return_cache_metrics"] is True
    assert chat_call["body"]["enable_thinking"] is False
    assert chat_call["body"]["tools"] == tools
    # Real-shape: a non-empty user message so the engine actually runs
    # a forward pass instead of short-circuiting on empty input.
    assert chat_call["body"]["messages"][-1]["role"] == "user"
    assert chat_call["body"]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_edge_llm_warmup_empty_prompt_short_circuits(monkeypatch):
    called = []

    class _NoClient:
        def __init__(self, *a, **kw):
            called.append("constructed")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(
        "ovs_agent.llm.edge_llm.httpx.AsyncClient", _NoClient
    )

    b = EdgeLLMBackend(
        base_url="http://edge-llm:8000/v1",
        api_key="sk-test",
        model="qwen3-4b-awq",
    )
    result = await b.warmup(system_prompt="", tools=None)
    assert result == {
        "cache_warmed": False,
        "graph_warmed": False,
        "cache_warmup_ms": 0,
        "graph_warmup_ms": 0,
    }
    assert called == []


@pytest.mark.asyncio
async def test_edge_llm_warmup_fail_open_on_cache_error(monkeypatch):
    class _BoomClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            raise RuntimeError("cache endpoint down")

        def stream(self, method, url, json=None):  # pragma: no cover
            raise AssertionError("graph warmup must not run if cache failed")

    monkeypatch.setattr(
        "ovs_agent.llm.edge_llm.httpx.AsyncClient", _BoomClient
    )

    b = EdgeLLMBackend(
        base_url="http://edge-llm:8000/v1",
        api_key="sk-test",
        model="qwen3-4b-awq",
    )
    result = await b.warmup(system_prompt="hi", tools=None)
    assert result["cache_warmed"] is False
    # graph warmup gated on cache_warmed → never ran.
    assert result["graph_warmed"] is False
