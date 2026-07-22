"""Controllable fake SLV server for tests and local development.

Mirrors the subset of the real SLV API the gallery talks to:
    GET  /health
    GET  /asr/capabilities
    GET  /tts/capabilities
    GET  /admin/backend/status
    POST /admin/backend/reload

State is a plain dict so tests can mutate it between requests:
    state["fail"]            set of endpoint keys forced to HTTP 500
                             ({"health","asr_capabilities","tts_capabilities",
                               "backend_status","reload"})
    state["admin_key"]       when set, /admin/* require a matching X-Admin-Key
    state["reload_response"] JSON body returned by /admin/backend/reload
    state["reload_status"]   HTTP status for the reload response (default 200)
    state["received"]        appended per reload call: {"headers", "body"}

Standalone (for curl verification of the gallery):
    uv run python tests/mock_slv.py            # port 8629, admin key "test-key"
    MOCK_SLV_PORT=9000 MOCK_ADMIN_KEY=k uv run python tests/mock_slv.py
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def default_state() -> dict:
    """Healthy Jetson-flavored device."""
    return {
        "fail": set(),
        "admin_key": None,
        "received": [],
        "reload_status": 200,
        # Loadable pre-flight: None simulates an SLV image without the
        # /admin/backend/loadable endpoint (→ 404). A dict is returned as-is.
        "loadable": None,
        "reload_response": {
            "status": "reloaded",
            "kind": "tts",
            "profile": None,  # filled from request
        },
        "health": {
            "tts": True,
            "tts_backend": "jetson.moss_tts_nano",
            "tts_capabilities": ["streaming", "speed", "pitch"],
            "asr": True,
            "asr_backend": "jetson.trt_edge_llm",
            "asr_capabilities": ["streaming", "multilingual"],
        },
        "asr_capabilities": {
            "backend": "jetson.trt_edge_llm",
            "capabilities": ["streaming", "multilingual"],
            "sample_rate": 16000,
        },
        "tts_capabilities": {
            "backend": "jetson.moss_tts_nano",
            "model_id": "moss-tts-nano-v1",
            "capabilities": ["streaming", "speed", "pitch"],
            "supports_voice_cloning": False,
            "sample_rate": 24000,
            "speakers": [{"id": 0, "name": "default"}],
        },
        "backend_status": {
            "tts": {
                "state": "ready",
                "profile_name": "jetson-qwen3asr-moss-nx",
                "backend_name": "jetson.moss_tts_nano",
                "inflight_http": 0,
                "inflight_ws": 0,
            },
            "asr": {
                "state": "ready",
                "profile_name": "jetson-qwen3asr-moss-nx",
                "backend_name": "jetson.trt_edge_llm",
                "inflight_http": 0,
                "inflight_ws": 0,
            },
        },
    }


def create_mock_slv(state: dict | None = None) -> tuple[FastAPI, dict]:
    state = state if state is not None else default_state()
    app = FastAPI(title="mock-slv")

    def _maybe_fail(key: str) -> JSONResponse | None:
        if key in state.get("fail", set()):
            return JSONResponse({"error": f"forced failure: {key}"}, status_code=500)
        return None

    def _check_admin(request: Request) -> JSONResponse | None:
        key = state.get("admin_key")
        if key and request.headers.get("X-Admin-Key") != key:
            return JSONResponse({"detail": "invalid X-Admin-Key"}, status_code=401)
        return None

    @app.get("/health")
    async def health() -> Any:
        return _maybe_fail("health") or JSONResponse(state["health"])

    @app.get("/asr/capabilities")
    async def asr_capabilities() -> Any:
        return _maybe_fail("asr_capabilities") or JSONResponse(state["asr_capabilities"])

    @app.get("/tts/capabilities")
    async def tts_capabilities() -> Any:
        return _maybe_fail("tts_capabilities") or JSONResponse(state["tts_capabilities"])

    @app.get("/admin/backend/status")
    async def backend_status(request: Request) -> Any:
        return (
            _maybe_fail("backend_status")
            or _check_admin(request)
            or JSONResponse(state["backend_status"])
        )

    @app.get("/admin/backend/loadable")
    async def backend_loadable(request: Request) -> Any:
        failed = _maybe_fail("loadable") or _check_admin(request)
        if failed:
            return failed
        loadable = state.get("loadable")
        if loadable is None:
            # Old SLV image without this endpoint.
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return JSONResponse(loadable)

    @app.post("/admin/backend/reload")
    async def backend_reload(request: Request) -> Any:
        body = await request.json()
        state["received"].append(
            {"headers": dict(request.headers), "body": body}
        )
        failed = _maybe_fail("reload") or _check_admin(request)
        if failed:
            return failed
        resp = dict(state["reload_response"])
        if resp.get("profile") is None:
            resp["profile"] = body.get("profile")
        return JSONResponse(resp, status_code=state.get("reload_status", 200))

    return app, state


if __name__ == "__main__":
    import uvicorn

    st = default_state()
    st["admin_key"] = os.environ.get("MOCK_ADMIN_KEY", "test-key")
    application, _ = create_mock_slv(st)
    uvicorn.run(
        application,
        host="127.0.0.1",
        port=int(os.environ.get("MOCK_SLV_PORT", "8629")),
    )
