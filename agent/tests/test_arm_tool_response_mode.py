"""The arm tools must not run an LLM round on their dispatch result.

Regression guard for the 2026-09-03 device report: one "please pick up the
box" spoke "Okay, grasping." twice, ran three LLM rounds, and closed with a
20-chunk narration, so the user was still listening 8.6s after the arm had
stopped moving.

Cause: the tools were registered ``response_mode="parallel"``, which runs an
LLM round on the fast-dispatch result. ``{"started": True}`` carries nothing
that reads as finished, so the model re-called the tool (refused by the
already_running guard, hence one physical grasp) and then narrated. The
preamble dedup set is per-round, so the repeat call spoke the preamble again.
"""
from __future__ import annotations

from typing import Any

import pytest

from ovs_agent.apps.voice_rebot_arm import grasp_plugin as gp


ARM_TOOLS = ("grasp_object", "search_object", "put_down")


def _tool_sources() -> str:
    import inspect

    return inspect.getsource(gp)


@pytest.mark.parametrize("name", ARM_TOOLS)
def test_arm_tools_skip_the_post_dispatch_llm_round(name: str) -> None:
    """response_mode must be template, not parallel."""
    src = _tool_sources()
    block = src.split(f'name="{name}"', 1)[1].split(")", 1)[0]
    assert 'response_mode="template"' in block, f"{name} still runs an LLM round"
    assert 'response_mode="parallel"' not in block


@pytest.mark.parametrize("name", ARM_TOOLS)
def test_arm_tools_declare_a_completion_text(name: str) -> None:
    """template with an empty completion_text silently falls back to await."""
    src = _tool_sources()
    block = src.split(f'name="{name}"', 1)[1].split(")", 1)[0]
    assert "completion_text=" in block
    assert 'completion_text=""' not in block


def test_every_refusal_carries_success_false() -> None:
    """The driver's template predicate keys off `success`, not `started`.

    _template_fires() short-circuits unless a result dict has
    `success is False`. A refusal returning only {"started": False} therefore
    reads as a success and speaks "Okay, grasping." with the torque off or the
    arm disconnected. Same mismatch as the 2026-06-12 server-loop ok-check.
    """
    import ast

    tree = ast.parse(_tool_sources())
    missing: list[int] = []
    seen = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            k.value for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        started = next(
            (
                v for k, v in zip(node.keys, node.values)
                if isinstance(k, ast.Constant) and k.value == "started"
            ),
            None,
        )
        if not (isinstance(started, ast.Constant) and started.value is False):
            continue
        seen += 1
        if "success" not in keys:
            missing.append(node.lineno)

    assert seen > 0, "no refusal returns found — did the tools move?"
    assert not missing, (
        f"{len(missing)} of {seen} refusal returns lack a success key "
        f"(lines {missing}); those speak the success line on failure"
    )


def test_the_driver_predicate_still_keys_off_success() -> None:
    """Pin the coupling this file depends on, so a voxedge change is visible.

    If _template_fires ever switches to another key, the success:False markers
    above stop protecting anything and the tools go back to lying on failure.
    """
    from voxedge.engine import turn_driver

    import inspect

    src = inspect.getsource(turn_driver._template_fires)  # noqa: SLF001
    assert 'res.get("success") is False' in src, (
        "voxedge changed the template success predicate; re-check every arm "
        "tool refusal path in grasp_plugin.py"
    )
