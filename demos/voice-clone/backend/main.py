"""Voice Clone demo backend.

Thin proxy in front of the SLV server (which has no CORS — browsers must go
through this same-origin backend):

    GET  /healthz            liveness of the demo app itself
    GET  /api/capabilities   voice-clone capability probe of SLV /tts/capabilities
    GET  /api/voices         passthrough of SLV GET /tts/voices (registered clones)
    POST /api/enroll         raw WAV body (+ ?voice_id=&ref_text=) → forwarded to
                             SLV POST /tts/voices/enroll as multipart form-data
    POST /api/clone/stream   streaming synthesis with an enrolled clone voice —
                             forwarded chunk-by-chunk (no buffering) so the
                             browser-measured TTFA survives the proxy hop
    POST /tts/stream         alias of /api/clone/stream, so the shared
                             ``TTSStreamPlayer`` (common/frontend/slv-client.js),
                             which always POSTs ``{origin}/tts/stream``, works
                             against this backend with its default baseUrl

Server contract (verified against server/main.py):

* ``POST /tts/voices/enroll`` (server/main.py:1547) takes **multipart form-data**:
  ``file`` (UploadFile, "reference wav (3-15s, single speaker)"), ``voice_id``
  (Form, required), ``ref_text`` (Form, optional). It runs the SparkTTS analysis
  chain in-process and returns ``{"voice_id", "json", "npz", "registry_count"}``;
  on a host without the PyTorch stack (Jetson) it returns **501** with a hint to
  use ``/tts/voices/profile`` instead — passed through verbatim to the browser.
* An enrolled voice is a *VoiceProfile* addressed by ``voice_id`` and selected at
  synth time via the ``voice`` field of ``POST /tts/stream`` (TTSRequest.voice,
  server/main.py:174-185 → _request_voice_kwargs, server/main.py:314-320).
  ``POST /tts/clone/stream`` (server/main.py:2278) is a *different* clone path:
  its CloneStreamRequest requires a raw ``speaker_embedding_b64`` (CustomVoice
  embedding cloning) which the enroll flow never produces — so this demo's
  clone synthesis goes through ``/tts/stream`` + ``voice``.
* Wire format of the stream: first 4 bytes = sample_rate (uint32 LE), then raw
  int16 PCM chunks. This backend never parses it — bytes in, bytes out.
* Capability flag: GET /tts/capabilities → ``supports_voice_cloning`` (bool,
  server/main.py:1392-1408); 503 while TTS is not ready.

The browser sends the enroll recording as a **raw WAV body** (audio/wav) rather
than multipart because the demos environment deliberately has no
``python-multipart`` (FastAPI's multipart parser dependency); httpx *encodes*
multipart natively, so forwarding to SLV needs no extra dependency.

Environment:
    SLV_URL     SLV server base URL (default http://127.0.0.1:8621)
    PORT        listen port (default 8705)
    DEMO_KIOSK  truthy => kiosk flag surfaced to the frontend

Run locally:  uv run python voice-clone/backend/main.py
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent          # demos/voice-clone/backend
_DEMO_DIR = _HERE.parent                          # demos/voice-clone
_DEMOS_DIR = _DEMO_DIR.parent                     # demos/

# Allow running as a plain script (`uv run python voice-clone/backend/main.py`):
# the demos/ dir must be importable for common.backend.slv_proxy.
if str(_DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMOS_DIR))

from common.backend.slv_proxy import SLVProxy  # noqa: E402
from common.backend.switch_api import register_switch_routes  # noqa: E402

_TRUTHY = {"1", "true", "yes", "on"}
_DEFAULT_PROFILES_DIR = _DEMOS_DIR.parent / "configs" / "profiles"

# Server-side enroll contract: "reference wav (3-15s, single speaker)".
# 44-byte WAV header + 1 s of 16 kHz mono int16 is a generous lower bound for
# "obviously not a recording" — real validation stays on the SLV side.
_MIN_ENROLL_BYTES = 44 + 16000 * 2


class CloneStreamProxyRequest(BaseModel):
    """What the browser sends for clone synthesis. ``voice`` is the enrolled
    voice_id (VoiceProfile selector); forwarded to SLV /tts/stream verbatim.
    Unset optional fields are omitted so SLV applies its own defaults."""

    text: str
    voice: str
    language: Optional[str] = None
    speed: Optional[float] = None
    pitch: Optional[float] = None


def create_app(proxy: SLVProxy | None = None, kiosk: bool | None = None) -> FastAPI:
    slv = proxy or SLVProxy()

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        await slv.aclose()

    app = FastAPI(title="slv-demo-voice-clone", docs_url=None, redoc_url=None,
                  lifespan=_lifespan)
    kiosk_mode = (
        (os.environ.get("DEMO_KIOSK") or "").strip().lower() in _TRUTHY
        if kiosk is None
        else kiosk
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "service": "slv-demo-voice-clone", "kiosk": kiosk_mode}

    @app.get("/api/capabilities")
    async def api_capabilities():
        """Probe SLV /tts/capabilities for the voice_clone capability.

        Degrades gracefully (the frontend must be able to show a friendly
        full-screen notice instead of letting the user record for nothing):
        * SLV unreachable      → 502 {"reachable": false, supports: null}
        * SLV 503 (not ready)  → 200 {"reachable": true, supports: false,
                                       "reason": "tts_not_ready"}
        * SLV 200              → 200 with the boolean verdict
        """
        try:
            resp = await slv.get("/tts/capabilities")
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"reachable": False, "supports_voice_cloning": None,
                 "error": "slv_unreachable", "message": str(exc)},
                status_code=502,
            )
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if resp.status_code != 200:
            return {
                "reachable": True,
                "supports_voice_cloning": False,
                "reason": "tts_not_ready",
                "detail": body.get("error") or f"HTTP {resp.status_code}",
            }
        caps = body.get("capabilities") or []
        supports = bool(body.get("supports_voice_cloning", "voice_clone" in caps))
        return {
            "reachable": True,
            "supports_voice_cloning": supports,
            "backend": body.get("backend"),
            "model_id": body.get("model_id"),
            "capabilities": caps,
        }

    @app.get("/api/voices")
    async def api_voices():
        """Proxy SLV /tts/voices (registered clone voices), verdict as-is."""
        try:
            resp = await slv.get("/tts/voices")
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"error": "slv_unreachable", "message": str(exc)}, status_code=502
            )
        try:
            body = resp.json()
        except ValueError:
            body = {"error": "invalid_slv_response", "text": resp.text[:500]}
        return JSONResponse(body, status_code=resp.status_code)

    @app.post("/api/enroll")
    async def api_enroll(request: Request, voice_id: str = "", ref_text: str = ""):
        """Forward a browser recording to SLV POST /tts/voices/enroll.

        Body: the raw WAV bytes (frontend assembles 16 kHz mono PCM16 into a
        WAV blob). Repacked here into the multipart form SLV expects
        (file + voice_id + optional ref_text). SLV's verdict — including the
        Jetson 501 "enroll on a GPU host" case — passes through verbatim.
        """
        wav = await request.body()
        if len(wav) < _MIN_ENROLL_BYTES:
            return JSONResponse(
                {"error": "audio_too_short",
                 "message": "recording is empty or shorter than ~1 s; "
                            "the enroll reference needs 3-15 s of speech"},
                status_code=400,
            )
        vid = voice_id.strip() or time.strftime("web-%Y%m%d-%H%M%S")
        data = {"voice_id": vid}
        if ref_text.strip():
            data["ref_text"] = ref_text.strip()
        try:
            # SLVProxy has no multipart POST helper (json-only stream_post /
            # admin_post) and this card must not modify shared files, so reach
            # for its underlying AsyncClient; httpx encodes multipart natively.
            resp = await slv._client.post(  # noqa: SLF001
                "/tts/voices/enroll",
                files={"file": ("enroll.wav", wav, "audio/wav")},
                data=data,
            )
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"error": "slv_unreachable", "message": str(exc)}, status_code=502
            )
        try:
            body = resp.json()
        except ValueError:
            body = {"error": "invalid_slv_response", "text": resp.text[:500]}
        return JSONResponse(body, status_code=resp.status_code)

    async def _clone_stream(req: CloneStreamProxyRequest):
        """Streaming proxy of clone synthesis via SLV POST /tts/stream.

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

    app.post("/api/clone/stream")(_clone_stream)
    # Alias for the shared TTSStreamPlayer, which POSTs {origin}/tts/stream.
    # CloneStreamProxyRequest still applies: this demo always synthesizes with
    # an enrolled voice, so `voice` stays required on the alias too.
    app.post("/tts/stream")(_clone_stream)

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

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8705")))
