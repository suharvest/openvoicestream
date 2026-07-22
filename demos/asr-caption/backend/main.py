"""asr-caption demo backend — live captions on /asr/stream.

Endpoints:
    GET /healthz      liveness of the demo app itself
    GET /api/config   where SLV's WebSocket lives, for the browser to connect
    /                 static frontend (this demo)
    /common/*         shared frontend assets (ui.css / ui.js / mic-capture.js ...)

Environment:
    SLV_URL   SLV server base URL (default http://127.0.0.1:8621)
    PORT      listen port for direct execution (default 8701)

The browser talks to SLV's ``/asr/stream`` WebSocket DIRECTLY (WebSocket is
not subject to CORS); this backend only serves static files and tells the
frontend where SLV lives. When SLV_URL points at loopback, ``/api/config``
flags it so the frontend substitutes ``location.hostname`` — a browser on
another machine then reaches the device instead of itself.

Run locally (module path contains a hyphen, so run the file directly):
    uv run python asr-caption/backend/main.py
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_HERE = Path(__file__).resolve().parent          # demos/asr-caption/backend
_APP_DIR = _HERE.parent                           # demos/asr-caption
_DEMOS_DIR = _APP_DIR.parent                      # demos/

# Allow running as a plain script (`uv run python asr-caption/backend/main.py`):
# the demos/ dir must be importable for common.backend.* .
if str(_DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMOS_DIR))

from common.backend.slv_proxy import SLVProxy  # noqa: E402
from common.backend.switch_api import register_switch_routes  # noqa: E402

DEFAULT_SLV_URL = "http://127.0.0.1:8621"
_DEFAULT_PROFILES_DIR = _DEMOS_DIR.parent / "configs" / "profiles"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _ws_endpoint(slv_url: str) -> dict:
    """Derive the browser-facing WebSocket endpoint from SLV_URL.

    ``loopback=True`` tells the frontend to replace ``host`` with
    ``location.hostname`` so external browsers hit the device, not themselves.
    """
    u = urlsplit(slv_url)
    scheme = "wss" if u.scheme == "https" else "ws"
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    return {
        "scheme": scheme,
        "host": host,
        "port": port,
        "loopback": host in _LOOPBACK_HOSTS,
    }


def create_app(slv_url: str | None = None, proxy: SLVProxy | None = None) -> FastAPI:
    slv_url = (slv_url or os.environ.get("SLV_URL") or DEFAULT_SLV_URL).rstrip("/")
    # The model-switch panel proxies SLV admin routes through this backend; the
    # WS mic/caption path still hits SLV directly from the browser.
    slv = proxy or SLVProxy(base_url=slv_url)

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        await slv.aclose()

    app = FastAPI(title="slv-demo-asr-caption", docs_url=None, redoc_url=None,
                  lifespan=_lifespan)
    profiles_dir = Path(os.environ.get("DEMO_PROFILES_DIR") or _DEFAULT_PROFILES_DIR)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "service": "slv-demo-asr-caption"}

    @app.get("/api/config")
    async def api_config() -> dict:
        return {
            "slv_url": slv_url,
            "ws": _ws_endpoint(slv_url),
            "asr_path": "/asr/stream",
        }

    # Shared model-switch API (/api/status, /api/profiles, /api/switch) — this
    # demo only exercises ASR, but the routes serve every kind; the frontend
    # panel is pinned to ["asr"]. Registered BEFORE the static mount.
    register_switch_routes(
        app, slv, profiles_dir, asr_label=os.environ.get("DEMO_ASR_MODEL_ID")
    )

    # Static frontend (mounted last so /api/* and /healthz win).
    common_frontend = _DEMOS_DIR / "common" / "frontend"
    if common_frontend.is_dir():  # pragma: no branch
        app.mount("/common", StaticFiles(directory=common_frontend), name="common")
    frontend = _APP_DIR / "frontend"
    if frontend.is_dir():  # pragma: no branch
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8701")))
