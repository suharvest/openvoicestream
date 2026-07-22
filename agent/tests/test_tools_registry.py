"""Tests for the @tool decorator + ToolRegistry dispatch."""
from __future__ import annotations

import asyncio
from typing import Literal, Optional

import pytest

from ovs_agent.tools import ToolRegistry


# ── schema generation per supported type ──────────────────────────────


def test_schema_for_basic_types():
    r = ToolRegistry()

    @r.tool(description="desc")
    def f(
        a_str: str,
        a_int: int,
        a_float: float,
        a_bool: bool,
        a_list: list,
        a_dict: dict,
    ) -> dict:
        return {}

    schemas = r.list_openai_tools()
    params = schemas[0]["function"]["parameters"]
    props = params["properties"]
    assert props["a_str"] == {"type": "string"}
    assert props["a_int"] == {"type": "integer"}
    assert props["a_float"] == {"type": "number"}
    assert props["a_bool"] == {"type": "boolean"}
    assert props["a_list"] == {"type": "array"}
    assert props["a_dict"] == {"type": "object"}
    assert set(params["required"]) == {
        "a_str", "a_int", "a_float", "a_bool", "a_list", "a_dict"
    }


def test_schema_for_literal_enum():
    r = ToolRegistry()

    @r.tool()
    def f(mode: Literal["a", "b", "c"]) -> dict:
        return {}

    props = r.list_openai_tools()[0]["function"]["parameters"]["properties"]
    assert props["mode"] == {"type": "string", "enum": ["a", "b", "c"]}


def test_schema_for_optional_unwraps():
    r = ToolRegistry()

    @r.tool()
    def f(s: Optional[str] = None, n: int | None = None) -> dict:
        return {}

    props = r.list_openai_tools()[0]["function"]["parameters"]["properties"]
    assert props["s"] == {"type": "string"}
    assert props["n"] == {"type": "integer"}
    # Optional params with defaults → not required
    params = r.list_openai_tools()[0]["function"]["parameters"]
    assert params.get("required", []) == []


def test_schema_for_parameterized_list():
    r = ToolRegistry()

    @r.tool()
    def f(xs: list[int]) -> dict:
        return {}

    props = r.list_openai_tools()[0]["function"]["parameters"]["properties"]
    assert props["xs"] == {"type": "array", "items": {"type": "integer"}}


def test_ctx_param_not_in_schema():
    r = ToolRegistry()

    @r.tool()
    def f(x: str, ctx) -> dict:
        return {}

    props = r.list_openai_tools()[0]["function"]["parameters"]["properties"]
    assert "ctx" not in props
    assert "x" in props


def test_description_defaults_to_docstring():
    r = ToolRegistry()

    @r.tool()
    def my_tool() -> dict:
        """does a useful thing."""
        return {}

    assert r.list_openai_tools()[0]["function"]["description"] == "does a useful thing."


# ── dispatch ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_sync_returns_dict():
    r = ToolRegistry()

    @r.tool()
    def f(x: int) -> dict:
        return {"x2": x * 2}

    result = await r.dispatch("f", {"x": 3}, ctx=None)
    assert result == {"x2": 6}


@pytest.mark.asyncio
async def test_dispatch_async_function():
    r = ToolRegistry()

    @r.tool()
    async def f(x: int) -> dict:
        await asyncio.sleep(0)
        return {"x": x}

    result = await r.dispatch("f", {"x": 5}, ctx=None)
    assert result == {"x": 5}


@pytest.mark.asyncio
async def test_dispatch_unknown_tool():
    r = ToolRegistry()
    out = await r.dispatch("nope", {}, ctx=None)
    assert out == {"success": False, "error": "unknown tool: nope"}


@pytest.mark.asyncio
async def test_dispatch_sanitizes_extra_kwargs():
    r = ToolRegistry()

    @r.tool()
    def f(x: int) -> dict:
        return {"x": x}

    # Extra "y" must be dropped, not raise TypeError.
    out = await r.dispatch("f", {"x": 1, "y": "ignored"}, ctx=None)
    assert out == {"x": 1}


@pytest.mark.asyncio
async def test_dispatch_injects_ctx_when_declared():
    r = ToolRegistry()

    @r.tool()
    def f(x: int, ctx) -> dict:
        return {"x": x, "ctx_seen": ctx is not None}

    sentinel = object()
    out = await r.dispatch("f", {"x": 1}, ctx=sentinel)
    assert out == {"x": 1, "ctx_seen": True}


@pytest.mark.asyncio
async def test_dispatch_wraps_non_dict_in_value():
    r = ToolRegistry()

    @r.tool()
    def f() -> int:
        return 42

    out = await r.dispatch("f", {}, ctx=None)
    assert out == {"value": 42}


@pytest.mark.asyncio
async def test_dispatch_exception_returns_error():
    r = ToolRegistry()

    @r.tool()
    def f() -> dict:
        raise ValueError("oops")

    out = await r.dispatch("f", {}, ctx=None)
    assert out == {"success": False, "error": "oops"}


@pytest.mark.asyncio
async def test_dispatch_timeout():
    r = ToolRegistry()

    @r.tool(timeout_s=0.05)
    async def f() -> dict:
        await asyncio.sleep(10)
        return {}

    out = await r.dispatch("f", {}, ctx=None)
    assert out["success"] is False
    assert "timed out" in out["error"]


# ── allow filtering ───────────────────────────────────────────────────


def test_list_openai_tools_allow_filter():
    r = ToolRegistry()

    @r.tool()
    def a() -> dict:
        return {}

    @r.tool()
    def b() -> dict:
        return {}

    names = {t["function"]["name"] for t in r.list_openai_tools({"a"})}
    assert names == {"a"}
    # None → expose all
    assert {t["function"]["name"] for t in r.list_openai_tools(None)} == {"a", "b"}


def test_register_arm_tools_skips_disabled_actions():
    from ovs_agent.tools.action_tools import register_arm_tools

    r = ToolRegistry()

    class _Arm:
        pass

    count = register_arm_tools(
        registry=r,
        arm_plugin=_Arm(),
        actions=[
            {"name": "wave", "description": "Wave"},
            {"name": "open_gripper", "description": "Open gripper"},
            {"name": "close_gripper", "description": "Close gripper"},
        ],
        disabled_actions={"open_gripper", "close_gripper"},
    )

    assert count == 1
    assert r.list_names() == ["wave"]


def test_register_arm_tools_defaults_to_parallel_with_preamble():
    # We deliberately KEEP the parallel + spoken-preamble default for arm tools
    # (an immediate "好的。" then LLM round 2 overlapping the motion). The
    # template/no-preamble variant was rejected: it changes the spoken
    # behaviour of every arm app (incl. SO-ARM) that relies on this default.
    from ovs_agent.tools.action_tools import register_arm_tools

    r = ToolRegistry()

    class _Arm:
        pass

    count = register_arm_tools(
        registry=r,
        arm_plugin=_Arm(),
        actions=[
            {"name": "wave", "description": "Wave"},
        ],
    )

    assert count == 1
    meta = r._tools["wave"]
    # Per-tool action-name preamble (wrong-tool audible detection): known
    # actions speak "好的，正在<动作>。", unknown ones fall back to "好的。".
    assert meta.preamble_text == "好的，正在挥手。"
    assert meta.response_mode == "parallel"


def test_tool_response_mode_and_completion_text_defaults():
    """New metadata fields default to ``await`` + empty string so existing
    callers behave exactly as before."""
    r = ToolRegistry()

    @r.tool()
    def t() -> dict:
        return {}

    meta = r._tools["t"]
    assert meta.response_mode == "await"
    assert meta.completion_text == ""


def test_tool_response_mode_and_completion_text_kwargs_threaded():
    """`response_mode` + `completion_text` kwargs on ``@tool`` are stored
    on the Tool dataclass verbatim."""
    r = ToolRegistry()

    @r.tool(response_mode="parallel", completion_text="挥完了")
    def wave() -> dict:
        return {"started": True}

    meta = r._tools["wave"]
    assert meta.response_mode == "parallel"
    assert meta.completion_text == "挥完了"


def test_unregister_returns_true_when_present_false_when_absent():
    from ovs_agent.tools import ToolRegistry
    reg = ToolRegistry()

    @reg.tool()
    def alpha() -> dict:
        return {"ok": True}

    @reg.tool()
    def beta() -> dict:
        return {"ok": True}

    assert reg.has("alpha")
    assert reg.has("beta")
    assert set(reg.list_names()) == {"alpha", "beta"}

    # First unregister removes; second returns False; idempotent.
    assert reg.unregister("alpha") is True
    assert reg.unregister("alpha") is False
    assert not reg.has("alpha")
    assert reg.has("beta")
    assert reg.list_names() == ["beta"]

    # list_openai_tools reflects the new state.
    schemas = reg.list_openai_tools()
    names = {t["function"]["name"] for t in schemas}
    assert names == {"beta"}
