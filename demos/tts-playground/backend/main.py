"""TTS Playground demo backend.

Thin proxy in front of the SLV server (which has no CORS — browsers must go
through this same-origin backend):

    GET  /healthz         liveness of the demo app itself
    GET  /api/speakers    passthrough of SLV GET /tts/speakers
    POST /api/tts/stream  streaming passthrough of SLV POST /tts/stream —
                          forwarded chunk-by-chunk (no buffering) so the
                          browser-measured TTFA survives the proxy hop
    POST /tts/stream      alias of /api/tts/stream, so the shared
                          ``TTSStreamPlayer`` (common/frontend/slv-client.js),
                          which always POSTs ``{origin}/tts/stream``, works
                          against this backend with its default baseUrl

Wire format of the stream (verified against server/main.py): first 4 bytes =
sample_rate (uint32 LE), then raw int16 PCM chunks. This backend never parses
it — bytes in, bytes out.

Environment:
    SLV_URL     SLV server base URL (default http://127.0.0.1:8621)
    PORT        listen port (default 8702)
    DEMO_KIOSK  truthy => kiosk flag surfaced to the frontend

Run locally:  uv run python tts-playground/backend/main.py
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent          # demos/tts-playground/backend
_DEMO_DIR = _HERE.parent                          # demos/tts-playground
_DEMOS_DIR = _DEMO_DIR.parent                     # demos/

# Allow running as a plain script (`uv run python tts-playground/backend/main.py`):
# the demos/ dir must be importable for common.backend.slv_proxy.
if str(_DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMOS_DIR))

from common.backend.slv_proxy import SLVProxy  # noqa: E402
from common.backend.switch_api import register_switch_routes  # noqa: E402

_TRUTHY = {"1", "true", "yes", "on"}
_DEFAULT_PROFILES_DIR = _DEMOS_DIR.parent / "configs" / "profiles"


class TTSProxyRequest(BaseModel):
    """Mirror of the SLV TTSRequest fields this demo uses (server/main.py:174).

    Unset fields are omitted from the forwarded body so SLV applies its own
    defaults (and runtime overrides) instead of receiving explicit nulls.
    """

    text: str
    speaker_id: Optional[int] = None
    speed: Optional[float] = None
    pitch: Optional[float] = None
    language: Optional[str] = None
    voice: Optional[str] = None


def create_app(proxy: SLVProxy | None = None, kiosk: bool | None = None) -> FastAPI:
    slv = proxy or SLVProxy()

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        await slv.aclose()

    app = FastAPI(title="slv-demo-tts-playground", docs_url=None, redoc_url=None,
                  lifespan=_lifespan)
    kiosk_mode = (
        (os.environ.get("DEMO_KIOSK") or "").strip().lower() in _TRUTHY
        if kiosk is None
        else kiosk
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "service": "slv-demo-tts-playground", "kiosk": kiosk_mode}

    @app.get("/api/speakers")
    async def api_speakers():
        """Proxy SLV /tts/speakers, passing its verdict through as-is."""
        try:
            resp = await slv.get("/tts/speakers")
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"error": "slv_unreachable", "message": str(exc)}, status_code=502
            )
        try:
            body = resp.json()
        except ValueError:
            body = {"error": "invalid_slv_response", "text": resp.text[:500]}
        return JSONResponse(body, status_code=resp.status_code)

    async def _tts_stream(req: TTSProxyRequest):
        """Streaming proxy of SLV POST /tts/stream.

        The upstream response is opened *before* returning so error statuses
        can be passed through with their bodies; on 200 every received chunk
        is yielded immediately (aiter_raw — no decoding, no buffering).
        """
        cm = slv.stream_post("/tts/stream", json=req.model_dump(exclude_none=True))
        try:
            upstream = await cm.__aenter__()
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"error": "slv_unreachable", "message": str(exc)}, status_code=502
            )

        if upstream.status_code != 200:
            try:
                body = await upstream.aread()
                return Response(
                    content=body,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            finally:
                await cm.__aexit__(None, None, None)

        async def forward():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await cm.__aexit__(None, None, None)

        return StreamingResponse(
            forward(),
            media_type=upstream.headers.get("content-type", "application/octet-stream"),
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    app.post("/api/tts/stream")(_tts_stream)
    # Alias for the shared TTSStreamPlayer, which POSTs {origin}/tts/stream.
    app.post("/tts/stream")(_tts_stream)

    # Shared model-switch API (/api/status, /api/profiles, /api/switch). This
    # demo's frontend panel is pinned to ["tts"]. Registered BEFORE the mount.
    profiles_dir = Path(os.environ.get("DEMO_PROFILES_DIR") or _DEFAULT_PROFILES_DIR)
    register_switch_routes(
        app, slv, profiles_dir, asr_label=os.environ.get("DEMO_ASR_MODEL_ID")
    )

    # Static frontend (mounted last so /api/* wins).
    common_frontend = _DEMOS_DIR / "common" / "frontend"
    if common_frontend.is_dir():  # pragma: no branch
        app.mount("/common", StaticFiles(directory=common_frontend), name="common")
    frontend = _DEMO_DIR / "frontend"
    if frontend.is_dir():  # pragma: no branch
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8702")))
