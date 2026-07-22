"""arm_tools — register one ovs-agent @tool per action in actions.yaml.

Each action becomes a no-argument tool whose description is the
``description:`` field from actions.yaml. The LLM picks an action by
function-calling; the tool dispatches into ArmPlugin.dispatch_action,
which returns as soon as the serial bus has accepted the first frame
(~200ms) — the remaining frames continue on a worker thread while the
LLM streams its acknowledgement and TTS plays out.

Response mode model (framework-level, see ovs-agent registry):
  * ``parallel`` (DEFAULT for arm actions): tool body returns fast
    after dispatch, LLM round 2 runs in parallel with the physical
    motion. Best for "wave" / "go home" style commands.
  * ``template``: tool body returns fast, runner SKIPS LLM round 2
    and synthesises a fixed ``completion_text`` instead — zero LLM
    latency between preamble and reply. Use when the action has a
    canonical verbal confirmation ("挥完了。") and creative LLM
    output adds nothing.
  * ``await``: tool body blocks until the motion completes, then LLM
    round 2 runs. Legacy behaviour — only useful if the LLM should
    reason over the post-motion observation cache.

actions.yaml schema (additions on top of the existing description):
  sequences:
    <name>:
      description: "..."
      response_mode: parallel | template | await   # optional
      completion_text: "挥完了。"                  # optional, used by template

Why closures-with-default-args: ``for entry in actions: def _tool(): ...``
captures ``entry`` by reference, so EVERY registered tool would dispatch
the last action. Wrapping in ``_make(name=...)`` binds the name at
closure-creation time, which is the canonical Python idiom for this.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default response mode for arm actions when actions.yaml is silent.
# We override the framework-level default of "await" because every
# pre-recorded SO-ARM motion is a "fire-and-forget" physical gesture —
# the user wants to hear "挥完了。" while the arm is still moving, not
# 1.8s of dead air waiting for the LLM to compose a fresh sentence.
_DEFAULT_ARM_RESPONSE_MODE = "parallel"


def register_arm_tools(
    registry,
    arm_plugin: Any,
    actions: list[dict],
    timeout_s: float = 15.0,
    disabled_actions: set[str] | None = None,
) -> int:
    """Register one tool per action; return count registered.

    Each entry in ``actions`` is the dict from
    ``ActionsManager.list_with_descriptions()`` — at minimum
    ``{"name", "description"}``, optionally with ``"response_mode"``
    and ``"completion_text"`` overrides parsed from actions.yaml.

    ``disabled_actions`` names actions to skip registering entirely (e.g. a
    motion the LLM should never be able to trigger on this deployment). Absent /
    empty → register everything (backward compatible).
    """
    disabled = disabled_actions or set()
    count = 0
    for entry in actions:
        action_name = entry.get("name")
        if not isinstance(action_name, str) or not action_name:
            continue
        if action_name in disabled:
            logger.info("skipping disabled arm tool name=%r", action_name)
            continue
        description = (entry.get("description") or "").strip() or (
            f"Execute the pre-recorded arm motion named {action_name!r}."
        )
        # Per-action response mode override, else "parallel" (arm default).
        rmode = entry.get("response_mode") or _DEFAULT_ARM_RESPONSE_MODE
        # completion_text only meaningful in template mode, but harmless
        # to thread through for parallel too (registry stores it; the
        # runner just ignores it when not in template mode).
        ctext = entry.get("completion_text") or ""

        pre = (entry.get("preamble") or "").strip()

        def _make(
            name: str = action_name,
            desc: str = description,
            mode: str = rmode,
            comp: str = ctext,
            pre_cfg: str = pre,
        ) -> None:
            if mode == "await":
                async def _tool() -> dict:
                    # Block-until-done semantics: the LLM will see the
                    # final {"success": bool, "action": name} dict and
                    # compose its reply with full knowledge of outcome.
                    return await arm_plugin.execute_action(name)
            else:
                # parallel + template both want fast return.
                async def _tool() -> dict:  # type: ignore[no-redef]
                    # Returns ~200ms after the serial bus accepts frame
                    # #1. The rest of the motion continues on a worker
                    # thread inside ArmPlugin; LLM round 2 / template
                    # reply overlap that worker.
                    return await arm_plugin.dispatch_action(name)

            _tool.__name__ = name
            _tool.__doc__ = desc
            # Per-tool preamble: speak the ACTION NAME instead of a generic
            # "好的" — the operator hears WHAT the model understood the
            # moment the tool fires and can shout 停 before the 2-5s motion
            # commits (instant wrong-tool detection). Optional per-action
            # ``preamble`` field in actions.yaml wins; generic fallback kept.
            # ``preamble_text="好的。"`` — period terminates the SLV
            # sentence buffer so synthesis fires immediately while the
            # 2-5s arm motion is still mid-dispatch. Qwen3-4B in
            # no-think/function-calling mode emits ``content=None +
            # tool_calls`` atomically, ignoring any "say OK first then
            # call the tool" prompt rule; this metadata is the
            # structural workaround.
            _DEFAULT_PREAMBLES = {
                "wave": "好的，正在挥手。",
                "go_home": "好的，正在回原位。",
                "open_gripper": "好的，正在张开夹爪。",
                "close_gripper": "好的，正在闭合夹爪。",
                "point_at": "好的，正在指向。",
            }
            preamble = pre_cfg or _DEFAULT_PREAMBLES.get(name, "好的。")
            registry.tool(
                name=name,
                description=desc,
                timeout_s=timeout_s,
                preamble_text=preamble,
                completion_text=comp,
                response_mode=mode,
            )(_tool)

        _make()
        count += 1
        logger.info(
            "registered arm tool name=%r response_mode=%r completion_text=%r",
            action_name, rmode, ctext,
        )
    logger.info("registered %d arm tools", count)
    return count


__all__ = ["register_arm_tools"]
