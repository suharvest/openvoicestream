"""Shared model-switch API for demo backends.

Registers the three endpoints the shared switch panel
(``common/frontend/ui.js`` → ``createModelSwitchPanel``) talks to:

    GET  {api}/profiles  profiles from ``profiles_dir`` filtered by device
                         platform + SLV loadable pre-flight, then collapsed to
                         one entry per distinct model (Qwen3-ASR / Matcha / …).
    GET  {api}/status    aggregated SLV health + backend status (drives the
                         "current model" line and the settle poll).
    POST {api}/switch    validated proxy to SLV /admin/backend/reload.

The logic here was lifted verbatim from the gallery backend so every consumer
(gallery portal + each demo page) offers an identical switch experience. Do not
change the model-identity / dedupe behavior — only its home moved.

Usage (in a demo's ``create_app``)::

    slv = SLVProxy()  # reads SLV_URL / SLV_ADMIN_KEY
    register_switch_routes(app, slv, profiles_dir,
                           asr_label=os.environ.get("DEMO_ASR_MODEL_ID"))
    # ... then mount the static frontend LAST so /api/* wins.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .slv_proxy import ProbeResult, SLVProxy


# ── profile listing / platform filtering ────────────────────────────────────


def _platform_tokens(probe: ProbeResult) -> set[str]:
    """Derive platform tokens from SLV-reported backend names.

    Backend names follow ``<platform>.<engine>`` (e.g. ``jetson.trt_edge_llm``);
    the profile files follow ``<platform>-<combo>.json``. We keep this
    deliberately simple: collect the dotted prefixes; a profile matches when
    its filename prefix equals a token or starts with one (``rk`` matches
    ``rk3576``). No tokens derivable => no filtering.
    """
    tokens: set[str] = set()
    sources = []
    if probe.health:
        sources += [probe.health.get("asr_backend"), probe.health.get("tts_backend")]
    if probe.backend_status:
        for kind in ("asr", "tts"):
            entry = probe.backend_status.get(kind) or {}
            sources.append(entry.get("backend_name"))
            # Live SLV reports undotted backend names (e.g. ``matcha_trt``);
            # the loaded profile name (``jetson-qwen3asr-matcha-nx``) is the
            # reliable platform carrier, so derive a token from its prefix too.
            profile_name = entry.get("profile_name")
            if isinstance(profile_name, str) and "-" in profile_name:
                prefix = profile_name.split("-", 1)[0].strip().lower()
                if prefix:
                    tokens.add(prefix)
    for name in sources:
        if isinstance(name, str) and "." in name:
            prefix = name.split(".", 1)[0].strip().lower()
            if prefix:
                tokens.add(prefix)
    return tokens


# Friendly model names for the switch dropdown. The list shows ENGINES/MODELS,
# not bundle-profile stems: several profiles share one ASR engine (e.g. every
# trt_edge_llm profile = the same Qwen3-ASR) and must collapse to one entry.
# ASR keys off the backend (no per-profile asr_model_id exists); TTS keys off
# tts_model_id (customvoice vs base are the SAME backend but DIFFERENT models).
_ASR_MODEL_LABELS = {
    "jetson.trt_edge_llm": "Qwen3-ASR",
    "jetson.paraformer_trt": "Paraformer",
    "jetson.sensevoice_trt": "SenseVoice",
    "rk.sensevoice": "SenseVoice",
    "sherpa.sensevoice": "SenseVoice",
}
_TTS_MODEL_LABELS = {
    "qwen3-tts-0.6b-base": "Qwen3-TTS",
    "qwen3-tts-customvoice": "CustomVoice",
    "matcha-icefall-zh-en": "Matcha",
    "kokoro-multi-lang-v1_0": "Kokoro",
    "moss-tts-nano-v1": "MOSS",
}


def _prettify_key(key: str) -> str:
    """Fallback label for an unmapped engine/model id."""
    tail = str(key).split(".")[-1]
    return tail.replace("_", " ").replace("-", " ").strip().title() or str(key)


def _model_identity(entry: dict, kind: str) -> Optional[tuple[str, str]]:
    """(model_key, display_label) for a profile in ``kind``, or None when the
    profile declares no backend for that kind (so it never pollutes the list)."""
    if kind == "asr":
        be = entry.get("asr_backend")
        if not be:
            return None
        return be, _ASR_MODEL_LABELS.get(be, _prettify_key(be))
    be = entry.get("tts_backend")
    if not be:
        return None
    mid = entry.get("tts_model_id") or be
    return mid, _TTS_MODEL_LABELS.get(mid, _prettify_key(mid))


def _dedupe_by_model(profiles: list[dict], kind: str, current_profile: Optional[str]) -> list[dict]:
    """Collapse bundle profiles to one entry per distinct model, labelled by
    model name. ``name`` stays a representative profile stem to reload; ``label``
    is what the dropdown shows; ``current`` marks the loaded model."""
    cur_key = None
    if current_profile:
        cur_key_pair = _model_identity(
            next((p for p in profiles if (p.get("name") == current_profile
                  or _profile_stem(p) == current_profile)), {}),
            kind,
        )
        cur_key = cur_key_pair[0] if cur_key_pair else None
    out: list[dict] = []
    seen: dict[str, dict] = {}
    for p in profiles:
        ident = _model_identity(p, kind)
        if ident is None:
            continue  # profile provides no backend for this kind
        key, label = ident
        if key in seen:
            continue
        entry = {
            "name": _profile_stem(p) or p.get("name"),  # representative profile to reload
            "label": label,
            "model_key": key,
            "description": p.get("description", ""),
            "current": key == cur_key,
        }
        seen[key] = entry
        out.append(entry)
    return out


def _list_profiles(profiles_dir: Path, probe: ProbeResult) -> dict:
    if not profiles_dir.is_dir():
        return {"profiles": [], "filtered": False, "platforms": [],
                "error": f"profiles dir not found: {profiles_dir}"}

    tokens = _platform_tokens(probe)
    # Optional operator allowlist: DEMO_SWITCH_PROFILES="a,b,c" restricts the
    # switch panel to exactly these profile stems (e.g. only the profiles whose
    # engines are actually loadable on this SLV). When unset, fall back to the
    # device-platform filter. Matches by filename stem or logical name.
    allow = {s.strip() for s in (os.environ.get("DEMO_SWITCH_PROFILES") or "").split(",") if s.strip()}
    profiles = []
    for path in sorted(profiles_dir.glob("*.json")):
        prefix = path.stem.split("-", 1)[0].lower()
        if allow:
            if path.stem not in allow:
                continue
        elif tokens and not any(prefix == t or prefix.startswith(t) for t in tokens):
            continue
        entry = {"name": path.stem, "file": path.name}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry["name"] = data.get("name", path.stem)
            entry["description"] = data.get("description", "")
            entry["asr_backend"] = data.get("asr_backend")
            entry["tts_backend"] = data.get("tts_backend")
            entry["tts_model_id"] = data.get("tts_model_id")
        except Exception as exc:  # noqa: BLE001 — a broken profile shouldn't hide the rest
            entry["error"] = f"unreadable: {type(exc).__name__}"
        profiles.append(entry)
    return {"profiles": profiles, "filtered": bool(tokens), "platforms": sorted(tokens)}


async def _fetch_loadable_names(slv: SLVProxy, kind: str) -> Optional[set[str]]:
    """Ask SLV which profiles it can actually load for ``kind``.

    Returns the set of loadable profile names on success, or ``None`` when the
    SLV doesn't expose ``/admin/backend/loadable`` (older image), is
    unreachable, rejects admin auth, or returns a malformed body — in every
    such case the caller must fall back to the unfiltered listing.
    """
    try:
        resp = await slv.admin_get("/admin/backend/loadable")
    except Exception:  # noqa: BLE001 — SLV down / transport error → graceful fallback
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    section = data.get(kind)
    if not isinstance(section, dict):
        return None
    loadable = section.get("loadable")
    if not isinstance(loadable, list):
        return None
    return {str(n) for n in loadable}


def _profile_stem(entry: dict) -> str:
    file = entry.get("file") or ""
    return file[: -len(".json")] if file.endswith(".json") else ""


def _allowed_profile_names(profiles_dir: Path) -> set[str]:
    """Names accepted by /api/switch: every profile JSON in the directory,
    by logical name and by filename stem (SLV resolves both)."""
    allowed: set[str] = set()
    if not profiles_dir.is_dir():
        return allowed
    for path in profiles_dir.glob("*.json"):
        allowed.add(path.stem)
        try:
            name = json.loads(path.read_text(encoding="utf-8")).get("name")
            if isinstance(name, str) and name:
                allowed.add(name)
        except Exception:  # noqa: BLE001
            pass
    return allowed


# ── route registration ───────────────────────────────────────────────────────


class SwitchRequest(BaseModel):
    kind: Literal["tts", "asr"]
    profile: str
    drain_timeout_s: Optional[float] = None


def register_switch_routes(
    app: FastAPI,
    slv: SLVProxy,
    profiles_dir: Path,
    *,
    asr_label: Optional[str] = None,
    kiosk: bool = False,
) -> None:
    """Register ``/api/status``, ``/api/profiles`` and ``/api/switch`` on ``app``.

    ``slv`` is the shared :class:`SLVProxy`; ``profiles_dir`` the directory of
    profile JSONs offered in the switch panel. ``asr_label`` (typically
    ``DEMO_ASR_MODEL_ID``) is surfaced as the ASR status pill model when SLV
    reports no ``model_id``. ``kiosk`` is echoed on ``/api/status`` for parity
    with the gallery.

    MUST be called BEFORE mounting the static frontend at ``/`` so the /api/*
    routes take precedence over the ``html=True`` catch-all.
    """
    profiles_dir = Path(profiles_dir)

    @app.get("/api/status")
    async def api_status() -> dict:
        probe = await slv.probe()
        slv_dict = probe.to_dict()
        # Presentation label: the SLV doesn't expose an ASR model_id (only the
        # engine name). If the operator sets DEMO_ASR_MODEL_ID, surface it so the
        # ASR status pill shows the model instead of the raw engine. (The proper
        # fix is server-side /asr/capabilities model_id; this is the demo-layer
        # fallback until images ship that.)
        label = (asr_label or "").strip()
        if label:
            ac = slv_dict.get("asr_capabilities")
            if isinstance(ac, dict) and not ac.get("model_id"):
                ac["model_id"] = label
        return {
            "kiosk": kiosk,
            "slv_url": slv.base_url,
            "degraded": bool(probe.errors),
            "slv": slv_dict,
        }

    @app.get("/api/profiles")
    async def api_profiles(kind: Optional[str] = None) -> dict:
        probe = await slv.probe()
        result = _list_profiles(profiles_dir, probe)
        result["slv_reachable"] = probe.reachable
        result["loadable_filtered"] = False
        # When a kind is requested, narrow the listing to profiles the SLV can
        # actually load for that kind (real artifact pre-flight). If the SLV
        # can't answer (old image / down / no admin key), keep the existing
        # platform-filtered listing untouched.
        if kind in ("tts", "asr"):
            loadable = await _fetch_loadable_names(slv, kind)
            if loadable is not None:
                result["profiles"] = [
                    p for p in result.get("profiles", [])
                    if (p.get("name") or "") in loadable
                    or (_profile_stem(p) and _profile_stem(p) in loadable)
                ]
                result["loadable_filtered"] = True
            # Collapse bundle profiles into one entry per distinct model, shown
            # by model name (Qwen3-ASR / Paraformer / Matcha / …) rather than
            # bundle stem. Also drops profiles that don't provide this kind.
            cur = ((probe.backend_status or {}).get(kind) or {}).get("profile_name")
            result["profiles"] = _dedupe_by_model(result.get("profiles", []), kind, cur)
        return result

    @app.post("/api/switch")
    async def api_switch(req: SwitchRequest):
        # Allow only what /api/profiles offers: platform-filtered when the
        # device platform is derivable, so a jetson-* profile can't be sent
        # to an RK box (and vice versa) just by crafting the POST.
        probe = await slv.probe()
        listing = _list_profiles(profiles_dir, probe)
        allowed: set[str] = set()
        for p in listing.get("profiles", []):
            allowed.add(p.get("name") or "")
            file = p.get("file") or ""
            if file.endswith(".json"):
                allowed.add(file[: -len(".json")])
        allowed.discard("")
        if not listing.get("filtered"):
            allowed |= _allowed_profile_names(profiles_dir)
        if req.profile not in allowed:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "profile_not_allowed",
                    "profile": req.profile,
                    "hint": "profile must be one of GET /api/profiles",
                },
            )
        try:
            resp = await slv.reload_backend(
                req.kind, req.profile, drain_timeout_s=req.drain_timeout_s
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error": "slv_unreachable", "message": str(exc)},
            ) from exc
        # Pass SLV's verdict (reloaded / rolled_back / 4xx detail) through as-is.
        try:
            body = resp.json()
        except ValueError:
            body = {"error": "invalid_slv_response", "text": resp.text[:500]}
        return JSONResponse(body, status_code=resp.status_code)
