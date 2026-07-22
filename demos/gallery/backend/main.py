"""Gallery portal backend.

Endpoints:
    GET  /healthz       liveness of the gallery app itself
    GET  /api/status    aggregated SLV health + backend status (+ kiosk flag)
    GET  /api/catalog   demo cards from registry.json annotated with SLV capabilities
    GET  /api/profiles  profiles from DEMO_PROFILES_DIR filtered by device platform
    POST /api/switch    validated proxy to SLV /admin/backend/reload

``/api/status``, ``/api/profiles`` and ``/api/switch`` are the shared model-switch
API (``common.backend.switch_api``) — the exact same routes every demo page now
mounts. Only ``/api/catalog`` + card annotation is gallery-specific.

Environment:
    SLV_URL            SLV server base URL (default http://127.0.0.1:8621)
    SLV_ADMIN_KEY      forwarded as X-Admin-Key on admin calls (optional)
    DEMO_PROFILES_DIR  directory of profile JSONs offered in the switch panel
                       (default: <repo>/configs/profiles when run from the repo)
    DEMO_ASR_MODEL_ID  presentation label for the ASR status pill (optional)
    DEMO_KIOSK         truthy => kiosk mode (frontend hides debug details)

Run locally:  uv run uvicorn gallery.backend.main:app --port 8700
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from common.backend.slv_proxy import SLVProxy, ProbeResult
from common.backend.switch_api import register_switch_routes

_HERE = Path(__file__).resolve().parent          # demos/gallery/backend
_GALLERY_DIR = _HERE.parent                       # demos/gallery
_DEMOS_DIR = _GALLERY_DIR.parent                  # demos/
_REGISTRY_PATH = _DEMOS_DIR / "registry.json"
_DEFAULT_PROFILES_DIR = _DEMOS_DIR.parent / "configs" / "profiles"

_TRUTHY = {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY


# ── catalog ─────────────────────────────────────────────────────────────────


def _load_registry(registry_path: Path) -> list[dict]:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return list(data.get("demos", []))


def _check_need(need: str, probe: ProbeResult) -> tuple[Optional[bool], str]:
    """Return (satisfied, reason_key). satisfied=None => unknown (SLV down)."""
    health = probe.health or {}
    if need in ("asr", "tts"):
        if probe.health is None:
            return None, "slv_unreachable"
        return bool(health.get(need)), f"{need}_not_ready"

    if "." in need:
        kind, cap = need.split(".", 1)
        if probe.health is None:
            return None, "slv_unreachable"
        caps = health.get(f"{kind}_capabilities") or []
        if kind == "tts" and cap == "voice_clone":
            tc = probe.tts_capabilities or {}
            if tc.get("supports_voice_cloning") or cap in caps:
                return True, ""
            return False, "no_voice_clone"
        return cap in caps, f"missing_{kind}_{cap}"

    return None, "unknown_need"


_REASON_TEXT = {
    "slv_unreachable": {"zh": "语音服务不可达", "en": "voice server unreachable"},
    "asr_not_ready": {"zh": "ASR 后端未就绪", "en": "ASR backend not ready"},
    "tts_not_ready": {"zh": "TTS 后端未就绪", "en": "TTS backend not ready"},
    "no_voice_clone": {"zh": "当前 TTS 引擎不支持声音克隆", "en": "current TTS engine has no voice cloning"},
}


def _reason_text(key: str) -> dict:
    if key in _REASON_TEXT:
        return _REASON_TEXT[key]
    if key.startswith("missing_"):
        _, kind, cap = key.split("_", 2)
        return {
            "zh": f"当前 {kind.upper()} 引擎缺少能力：{cap}",
            "en": f"current {kind.upper()} engine lacks capability: {cap}",
        }
    return {"zh": "设备能力未知", "en": "device capability unknown"}


def _annotate_card(demo: dict, probe: ProbeResult) -> dict:
    card = dict(demo)
    needs = demo.get("needs") or []
    unmet_reason: Optional[str] = None
    unknown = False
    for need in needs:
        ok, reason = _check_need(need, probe)
        if ok is False:
            unmet_reason = reason
            break
        if ok is None:
            unknown = True

    if unmet_reason:
        card["state"] = "unsupported"
        card["reason"] = _reason_text(unmet_reason)
    elif demo.get("status") == "coming-soon":
        card["state"] = "coming-soon"
        card["reason"] = {"zh": "即将上线", "en": "coming soon"}
    elif unknown:
        card["state"] = "unknown"
        card["reason"] = _reason_text("slv_unreachable")
    else:
        card["state"] = "available"
        card["reason"] = {"zh": "", "en": ""}
    card["available"] = card["state"] == "available"
    return card


# ── app factory ──────────────────────────────────────────────────────────────


def create_app(
    proxy: SLVProxy | None = None,
    registry_path: Path | None = None,
    profiles_dir: Path | None = None,
    kiosk: bool | None = None,
) -> FastAPI:
    slv = proxy or SLVProxy()

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        await slv.aclose()

    app = FastAPI(title="slv-demo-gallery", docs_url=None, redoc_url=None,
                  lifespan=_lifespan)
    registry_path = registry_path or _REGISTRY_PATH
    profiles_dir = Path(
        profiles_dir
        or os.environ.get("DEMO_PROFILES_DIR")
        or _DEFAULT_PROFILES_DIR
    )
    kiosk_mode = _env_truthy("DEMO_KIOSK") if kiosk is None else kiosk

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "service": "slv-demo-gallery", "kiosk": kiosk_mode}

    @app.get("/api/catalog")
    async def api_catalog() -> dict:
        try:
            demos = _load_registry(registry_path)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                {"error": "registry_unreadable", "message": str(exc)}, status_code=500
            )
        probe = await slv.probe()
        return {
            "kiosk": kiosk_mode,
            "slv_reachable": probe.reachable,
            "degraded": bool(probe.errors),
            "errors": probe.errors,
            "demos": [_annotate_card(d, probe) for d in demos],
        }

    # Shared model-switch API: /api/status, /api/profiles, /api/switch. Same
    # routes every demo page mounts — registered BEFORE the static mount so
    # /api/* wins over the html=True catch-all.
    register_switch_routes(
        app, slv, profiles_dir,
        asr_label=os.environ.get("DEMO_ASR_MODEL_ID"),
        kiosk=kiosk_mode,
    )

    # Static frontend (mounted last so /api/* wins).
    common_frontend = _DEMOS_DIR / "common" / "frontend"
    if common_frontend.is_dir():  # pragma: no branch
        app.mount("/common", StaticFiles(directory=common_frontend), name="common")
    frontend = _GALLERY_DIR / "frontend"
    if frontend.is_dir():  # pragma: no branch
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8700")))
