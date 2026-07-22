"""Built-in tools that ship with OpenVoiceStream Agent.

Kept deliberately minimal so the surface stays trustworthy. Apps that
need more should register additional ``@tool``-decorated functions
against :data:`default_registry` from this module."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .registry import default_registry as _r


@_r.tool(description="Return the current local time as ISO 8601.")
def time_now() -> dict[str, Any]:
    return {"now": datetime.now().isoformat()}


@_r.tool(description="Switch the agent to a different mode.")
async def set_mode(mode_name: str, ctx: Any) -> dict[str, Any]:
    """``mode_name`` is the target mode key from app config."""
    mm = getattr(ctx, "mode_manager", None)
    if mm is None:
        return {"success": False, "error": "mode_manager unavailable"}
    if mm.get(mode_name) is None:
        names = [m.get("name") for m in mm.list_all()]
        return {
            "success": False,
            "error": f"unknown mode: {mode_name}",
            "available": names,
        }
    try:
        await mm.switch(mode_name)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"switch failed: {e}"}
    return {"success": True, "mode": mode_name}
