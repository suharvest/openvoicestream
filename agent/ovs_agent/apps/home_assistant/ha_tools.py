"""Home Assistant control tools, exposed to the server-side LLM loop.

Importing this module registers the tools on the shared ``default_registry``;
the package ``__init__`` imports it so they exist before the agent's
``_advertise_tools_if_server_loop()`` runs at session open.

HOW THIS REACHES THE LLM
------------------------
These are ordinary local ``@tool`` handlers, but in server-loop deployments the
agent *advertises* their schemas over ``/v2v/stream`` and the server-side LLM
calls them back over the wire. So the code runs here, on the device, next to the
LAN the Home Assistant instance is on — which is the whole point: the LLM never
needs credentials or network access to the user's home.

CONTRACT NOTES THAT COST TIME TO REDISCOVER
-------------------------------------------
* Every handler returns a dict and NEVER raises. A raised exception becomes an
  opaque failure; a returned ``{"ok": False, "error": ...}`` lets the LLM say
  something useful, and lets it retry with a better device name.
* Keep the whole call under ~15 s. The per-tool ``timeout_s`` is not carried
  over the advertise wire, so the SERVER's default (~15 s) is what actually
  bounds a slow tool, regardless of what is set here.
* ``preamble_text`` is spoken the moment a tool starts, before its result
  exists. Worth it for anything with a physical effect: without it the user
  hears silence while a light turns on and assumes they were not heard.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from ovs_agent.tools import default_registry as _r

from .ha_client import HAClient, ResolveError

logger = logging.getLogger(__name__)

# Module-level client so every tool call reuses one connection pool and one
# device cache. Built lazily: importing this module must not require HA to be
# reachable (the agent imports it at startup, possibly before the network is up).
_client: Optional[HAClient] = None


def configure(client: HAClient) -> None:
    """Install the client the tools should use (called by the app on start)."""
    global _client
    _client = client


def _ha() -> HAClient:
    """Return the configured client, building one from env as a fallback."""
    global _client
    if _client is None:
        base = os.environ.get("HA_BASE_URL", "").strip()
        token = os.environ.get("HA_TOKEN", "").strip()
        if not base or not token:
            raise RuntimeError(
                "Home Assistant not configured: set HA_BASE_URL and HA_TOKEN "
                "(or call ha_tools.configure())"
            )
        _client = HAClient(base, token)
    return _client


def _fail(msg: str, candidates: Optional[list[str]] = None) -> dict[str, Any]:
    """Uniform failure dict. ``candidates`` gives the LLM a way to recover."""
    out: dict[str, Any] = {"ok": False, "error": msg}
    if candidates:
        # Cap the list: a large house would otherwise push a hundred names into
        # the LLM context on every miss.
        out["candidates"] = candidates[:20]
    return out


# Which service turns a given domain on/off. cover and lock do not use
# turn_on/turn_off at all, which is exactly the kind of detail we do not want
# the LLM to have to know.
_ON_OFF = {
    "light": ("turn_on", "turn_off"),
    "switch": ("turn_on", "turn_off"),
    "fan": ("turn_on", "turn_off"),
    "media_player": ("turn_on", "turn_off"),
    "input_boolean": ("turn_on", "turn_off"),
    "scene": ("turn_on", "turn_on"),      # a scene can only be activated
    "script": ("turn_on", "turn_off"),
    "climate": ("turn_on", "turn_off"),
    "cover": ("open_cover", "close_cover"),
    "lock": ("unlock", "lock"),
}


@_r.tool(
    description=(
        "List the smart-home devices that can be controlled, with their current "
        "state. Call this first when the user refers to a device you have not "
        "seen yet, or asks what can be controlled. Use the returned names "
        "verbatim when calling the other tools."
    ),
)
def list_devices() -> dict[str, Any]:
    """Compact device inventory: name, kind and state only.

    Deliberately NOT the full HA attribute dump — a normal house has enough
    entities that the raw payload would crowd out the conversation in the LLM's
    context window and make it likelier to pick the wrong one.
    """
    try:
        devs = _ha().devices(refresh=True)
    except Exception as e:  # noqa: BLE001 — network/auth/config all end up here
        logger.warning("[ha] list_devices failed: %r", e)
        return _fail(f"无法连接 Home Assistant: {e}")
    return {
        "ok": True,
        "count": len(devs),
        "devices": [{"name": d.name, "type": d.domain, "state": d.state}
                    for d in devs],
    }


@_r.tool(
    description=(
        "Turn a smart-home device ON (lights, switches, fans, media players); "
        "for a curtain or blind this OPENS it, and for a lock it UNLOCKS it. "
        "`device` is the spoken device name, e.g. 客厅灯."
    ),
    preamble_text="好的。",
    response_mode="template",
    completion_text="已经打开了。",
)
def turn_on(device: str) -> dict[str, Any]:
    """Domain-aware ON. Resolves the spoken name, then picks the right service."""
    return _switch(device, on=True)


@_r.tool(
    description=(
        "Turn a smart-home device OFF; for a curtain or blind this CLOSES it, "
        "and for a lock it LOCKS it. `device` is the spoken device name."
    ),
    preamble_text="好的。",
    response_mode="template",
    completion_text="已经关掉了。",
)
def turn_off(device: str) -> dict[str, Any]:
    """Domain-aware OFF."""
    return _switch(device, on=False)


def _switch(device: str, on: bool) -> dict[str, Any]:
    try:
        ha = _ha()
        d = ha.resolve(device)
    except ResolveError as e:
        return _fail(str(e), e.candidates)
    except Exception as e:  # noqa: BLE001
        return _fail(f"无法连接 Home Assistant: {e}")
    pair = _ON_OFF.get(d.domain)
    if pair is None:
        return _fail(f"{d.name} 不支持开关操作")
    service = pair[0] if on else pair[1]
    try:
        ha.call_service(d.domain, service, {"entity_id": d.entity_id})
    except Exception as e:  # noqa: BLE001
        return _fail(f"操作 {d.name} 失败: {e}")
    return {"ok": True, "device": d.name, "entity_id": d.entity_id,
            "action": service}


@_r.tool(
    description=(
        "Set a light's brightness as a percentage from 0 to 100. Also turns the "
        "light on. `device` is the spoken light name."
    ),
    preamble_text="好的。",
    response_mode="template",
    completion_text="已经调好了。",
)
def set_brightness(device: str, percent: int) -> dict[str, Any]:
    """Brightness in percent — HA's brightness_pct, not the 0-255 raw value."""
    if not 0 <= percent <= 100:
        return _fail("亮度必须在 0 到 100 之间")
    try:
        ha = _ha()
        # Narrow to lights: "客厅" is ambiguous across all domains but
        # unambiguous once we know the user is talking about brightness.
        d = ha.resolve(device, domains=("light",))
    except ResolveError as e:
        return _fail(str(e), e.candidates)
    except Exception as e:  # noqa: BLE001
        return _fail(f"无法连接 Home Assistant: {e}")
    try:
        ha.call_service("light", "turn_on",
                        {"entity_id": d.entity_id, "brightness_pct": percent})
    except Exception as e:  # noqa: BLE001
        return _fail(f"调节 {d.name} 亮度失败: {e}")
    return {"ok": True, "device": d.name, "brightness_pct": percent}


@_r.tool(
    description=(
        "Set how far open a curtain, blind or other cover is, 0 (fully closed) "
        "to 100 (fully open). `device` is the spoken cover name, e.g. 客厅窗帘."
    ),
    preamble_text="好的。",
    response_mode="template",
    completion_text="已经调好了。",
)
def set_cover_position(device: str, percent: int) -> dict[str, Any]:
    """Cover position in percent."""
    if not 0 <= percent <= 100:
        return _fail("位置必须在 0 到 100 之间")
    try:
        ha = _ha()
        d = ha.resolve(device, domains=("cover",))
    except ResolveError as e:
        return _fail(str(e), e.candidates)
    except Exception as e:  # noqa: BLE001
        return _fail(f"无法连接 Home Assistant: {e}")
    try:
        ha.call_service("cover", "set_cover_position",
                        {"entity_id": d.entity_id, "position": percent})
    except Exception as e:  # noqa: BLE001
        return _fail(f"调节 {d.name} 失败: {e}")
    return {"ok": True, "device": d.name, "position": percent}


@_r.tool(
    description=(
        "Read the current state of one smart-home device — whether it is on or "
        "off, a light's brightness, a curtain's position, and so on."
    ),
)
def get_state(device: str) -> dict[str, Any]:
    """Current state plus the few attributes worth speaking aloud."""
    try:
        ha = _ha()
        d = ha.resolve(device)
        raw = ha.raw_state(d.entity_id)
    except ResolveError as e:
        return _fail(str(e), e.candidates)
    except Exception as e:  # noqa: BLE001
        return _fail(f"无法读取 {device} 的状态: {e}")
    attrs = raw.get("attributes") or {}
    out: dict[str, Any] = {"ok": True, "device": d.name,
                           "state": raw.get("state", "")}
    # Only surface attributes a person would actually ask about. brightness is
    # reported 0-255 by HA; convert so the LLM does not read out "153".
    if attrs.get("brightness") is not None:
        out["brightness_pct"] = round(int(attrs["brightness"]) / 255 * 100)
    for k in ("current_position", "current_temperature", "temperature",
              "volume_level"):
        if attrs.get(k) is not None:
            out[k] = attrs[k]
    return out


@_r.tool(
    description=(
        "Escape hatch: call an arbitrary Home Assistant service directly, for "
        "anything the other tools do not cover (climate modes, media control, "
        "scripts, scenes). Prefer the specific tools when one fits."
    ),
    preamble_text="好的。",
)
def call_service(domain: str, service: str, entity_id: str = "",
                 data_json: str = "") -> dict[str, Any]:
    """Raw ``POST /api/services/<domain>/<service>``.

    ``data_json`` is an optional JSON object as a string — a string rather than
    a dict because the generated schema for a free-form object gives the LLM no
    guidance, and a JSON string round-trips through tool-call arguments cleanly.
    """
    import json
    payload: dict[str, Any] = {}
    if data_json.strip():
        try:
            parsed = json.loads(data_json)
        except (ValueError, TypeError) as e:
            return _fail(f"data_json 不是合法 JSON: {e}")
        if not isinstance(parsed, dict):
            return _fail("data_json 必须是一个 JSON 对象")
        payload.update(parsed)
    if entity_id:
        payload["entity_id"] = entity_id
    try:
        result = _ha().call_service(domain, service, payload)
    except Exception as e:  # noqa: BLE001
        return _fail(f"调用 {domain}.{service} 失败: {e}")
    return {"ok": True, "domain": domain, "service": service,
            "changed_entities": [e.get("entity_id") for e in (result or [])
                                 if isinstance(e, dict)]}


__all__ = [
    "configure", "list_devices", "turn_on", "turn_off",
    "set_brightness", "set_cover_position", "get_state", "call_service",
]
