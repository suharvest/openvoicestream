"""ArmDashboardPlugin — read-only web view of the arm + vision pipeline.

Serves a single static page plus a polling JSON API:

  GET /                 the dashboard page (plain html/js, no build step)
  GET /api/state        {arm, frame_seq, frame_meta, frame_history, events,
                         place_bounds, busy}
  GET /api/frame.jpg    latest annotated decision/idle frame
  GET /api/depth.jpg    matching depth colormap

Frame/event content comes from :mod:`dashboard_bus` (fed by GraspPlugin's
frame_sink tee + idle observer). Arm state is proxied server-side from the
existing observation server (FastAPI, :8775) so the browser never needs CORS.

Read-only by design — no control endpoints — so binding beyond loopback is
safe; default bind is 0.0.0.0 for LAN demo viewing (override with
OVS_ARM_DASHBOARD_BIND).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from ovs_agent.plugin import Plugin

logger = logging.getLogger(__name__)

_PAGE = Path(__file__).with_name("static_dashboard.html")

# Short TTL so the ~1Hz dashboard poll doesn't hammer the SLV / edge-llm
# /health endpoints; results are cached on the plugin instance and only
# re-fetched when stale.
_HEALTH_TTL_S = 10.0
_HEALTH_TIMEOUT_S = 2.0


# Re-exported from the shared helper: the arm was where the mic-less inject
# was first worked out, but every app needs it, so the implementation now lives
# in ovs_agent.debug_inject. Kept importable from here for existing tests.
from ovs_agent.debug_inject import wav_bytes_to_pcm16_mono as _wav_bytes_to_pcm16_mono



class ArmDashboardPlugin(Plugin):
    name = "arm_dashboard"

    def __init__(self, app: Any, config: Optional[dict] = None) -> None:
        super().__init__(app)
        self.cfg = dict(config or {})
        self._runner = None
        self._site = None
        self._started = False
        # Cached /health probe results (TTL-gated). Each entry:
        #   {"ts": <monotonic>, "ok": bool, "data": <dict|None>}
        self._health_cache: dict[str, dict] = {}

    async def start(self) -> None:
        if self.cfg.get("enabled", True) is False or self._started:
            return
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp not installed — arm_dashboard disabled")
            return
        self._started = True
        web_app = web.Application()
        web_app.router.add_get("/", self._handle_index)
        web_app.router.add_get("/api/state", self._api_state)
        web_app.router.add_get("/api/frame.jpg", self._api_frame)
        web_app.router.add_get("/api/depth.jpg", self._api_depth)
        # DEBUG-ONLY remote audio inject (mic-less e2e). Always registered, but
        # the handler refuses unless OVS_REBOT_DEBUG_INJECT=1 — so production is
        # inert until explicitly enabled (+ restart). Works in server-loop: the
        # injected PCM is forwarded to SLV by the normal mic pump.
        web_app.router.add_post("/api/control/inject_wav", self._api_inject_wav)
        self._runner = web.AppRunner(web_app)
        await self._runner.setup()
        bind = os.environ.get("OVS_ARM_DASHBOARD_BIND", "0.0.0.0").strip() or "0.0.0.0"
        port = int(self.cfg.get("port", 8776))
        self._site = web.TCPSite(self._runner, bind, port)
        await self._site.start()
        logger.info("arm_dashboard listening on http://%s:%d (read-only)", bind, port)
        await super().start()

    async def stop(self) -> None:
        await super().stop()
        if not self._started:
            return
        self._started = False
        try:
            if self._runner is not None:
                await self._runner.cleanup()
        except Exception:
            logger.debug("arm_dashboard cleanup failed", exc_info=True)
        self._runner = None
        self._site = None

    # ── handlers ─────────────────────────────────────────────────────
    async def _handle_index(self, request):  # noqa: ANN001
        from aiohttp import web

        if _PAGE.exists():
            return web.FileResponse(path=str(_PAGE))
        return web.Response(text="dashboard page missing", status=500)

    async def _api_inject_wav(self, request):  # noqa: ANN001
        """DEBUG-ONLY: POST a WAV body → fed STRAIGHT to the SLV as a spoken
        utterance, mic-less (bypasses the energy-gated mic pump, which otherwise
        drops low-energy syllables / the onset and delivered truncated or empty
        clips). In server-loop: SLV → ASR → LLM → tool_call → arm. Gated behind
        OVS_REBOT_DEBUG_INJECT=1 (default OFF) because a remote caller could
        otherwise move the physical arm. Forces asr_eos after feeding so the SLV
        finalizes regardless of VAD/endpoint config."""
        from aiohttp import web

        if os.environ.get("OVS_REBOT_DEBUG_INJECT") != "1":
            return web.json_response(
                {"ok": False, "error": "inject disabled; set "
                 "OVS_REBOT_DEBUG_INJECT=1 and restart the agent"},
                status=403,
            )
        from ovs_agent.debug_inject import inject_wav

        try:
            result = await inject_wav(self.app, await request.read())
        except ValueError as e:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except RuntimeError as e:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(e)}, status=409)
        return web.json_response(result)

    async def _api_state(self, request):  # noqa: ANN001
        from aiohttp import web

        from .dashboard_bus import BUS

        state = BUS.snapshot()
        state["arm"] = await asyncio.to_thread(self._fetch_observation)
        state["busy"] = self._busy_motion()
        state["gripper"] = await asyncio.to_thread(self._gripper_health)
        state["place_bounds"] = self._place_bounds()
        # Best-effort models/status block; a dead SLV / edge-llm must never
        # break /api/state (arm + frame data must still return).
        try:
            state["models"] = await self._models_block()
        except Exception:
            logger.debug("models block failed", exc_info=True)
            state["models"] = None
        return web.json_response(state)

    def _gripper_health(self) -> Optional[dict]:
        """Jaw availability, proxied from the observation server.

        init_gripper() is best-effort — the arm connects and drives fine
        without it — which used to make a failed jaw indistinguishable from a
        healthy one on every surface: connect logged "actuator connected",
        /api/state said nothing, and a grasp ran end to end before reporting
        "nothing held".
        """
        port = int(self.cfg.get("observation_port", 8775))
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/gripper", timeout=1.5
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    async def _api_frame(self, request):  # noqa: ANN001
        return self._jpg_response((await self._bus()).latest_jpg())

    async def _api_depth(self, request):  # noqa: ANN001
        return self._jpg_response((await self._bus()).latest_depth_jpg())

    @staticmethod
    async def _bus():
        from .dashboard_bus import BUS

        return BUS

    @staticmethod
    def _jpg_response(data: Optional[bytes]):
        from aiohttp import web

        if not data:
            return web.Response(status=404, text="no frame yet")
        return web.Response(
            body=data, content_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    # ── data sources ─────────────────────────────────────────────────
    def _fetch_observation(self) -> Optional[dict]:
        """Server-side proxy to the observation server (loopback, no CORS)."""
        port = int(self.cfg.get("observation_port", 8775))
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/observation", timeout=1.5
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def _busy_motion(self) -> Optional[str]:
        for plugin in getattr(self.app, "plugins", []) or []:
            if plugin.__class__.__name__ == "GraspPlugin":
                try:
                    return plugin._busy_motion_name()  # noqa: SLF001
                except Exception:
                    return None
        return None

    def _place_bounds(self) -> Optional[list]:
        pb = self.cfg.get("place_bounds")
        if pb:
            try:
                vals = [float(v) for v in pb]
                if len(vals) == 4:
                    return vals
            except (TypeError, ValueError):
                pass
        return None

    # ── models / live-status aggregation ─────────────────────────────
    def _llm_health_url(self) -> Optional[str]:
        """Derive the edge-llm /health URL from config.llm_base_url
        (http://host:port/v1 → http://host:port/health)."""
        base = getattr(getattr(self.app, "config", None), "llm_base_url", None)
        if not base:
            return None
        from urllib.parse import urlparse, urlunparse

        try:
            u = urlparse(base)
            if not u.scheme or not u.netloc:
                return None
            return urlunparse((u.scheme, u.netloc, "/health", "", "", ""))
        except Exception:
            return None

    def _slv_health_url(self) -> Optional[str]:
        """SLV /health URL derived from config.slv_http_base (ws→http already
        handled by the Config property)."""
        cfg = getattr(self.app, "config", None)
        base = getattr(cfg, "slv_http_base", None)
        if not base:
            return None
        return base.rstrip("/") + "/health"

    async def _cached_health(self, key: str, url: Optional[str]) -> dict:
        """Return a TTL-cached {"ok", "data"} for ``url``; only re-fetch when
        the cached entry is older than ``_HEALTH_TTL_S``. Network errors →
        ok=false (with the last-known data preserved). Never raises."""
        if not url:
            return {"ok": False, "data": None}
        now = time.monotonic()
        cached = self._health_cache.get(key)
        if cached is not None and (now - cached["ts"]) < _HEALTH_TTL_S:
            return {"ok": cached["ok"], "data": cached.get("data")}
        ok, data = await self._fetch_health(url)
        if not ok and cached is not None:
            # Preserve last-known body (for name continuity) but mark not-ok.
            data = cached.get("data")
        self._health_cache[key] = {"ts": now, "ok": ok, "data": data}
        return {"ok": ok, "data": data}

    @staticmethod
    async def _fetch_health(url: str) -> tuple[bool, Optional[dict]]:
        """Short-timeout, non-blocking GET of a /health endpoint. Proxy env is
        bypassed (agent talks to SLV / edge-llm on the docker network)."""
        try:
            import aiohttp
        except ImportError:
            return False, None
        try:
            timeout = aiohttp.ClientTimeout(total=_HEALTH_TIMEOUT_S)
            # trust_env=False → ignore HTTP(S)_PROXY/NO_PROXY so the on-network
            # health GET isn't routed through an external proxy.
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=False
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return False, None
                    try:
                        return True, await resp.json(content_type=None)
                    except Exception:
                        return True, None
        except Exception:
            return False, None

    async def _models_block(self) -> dict:
        cfg = getattr(self.app, "config", None)
        slv = getattr(self.app, "slv", None)

        llm = await self._cached_health("llm", self._llm_health_url())
        slv_h = await self._cached_health("slv", self._slv_health_url())
        slv_data = slv_h.get("data") or {}

        # edge-llm endpoint (host:port) for display.
        llm_endpoint = None
        llm_url = self._llm_health_url()
        if llm_url:
            from urllib.parse import urlparse

            try:
                llm_endpoint = urlparse(llm_url).netloc or None
            except Exception:
                llm_endpoint = None

        asr_backend = slv_data.get("asr_backend")
        tts_backend = slv_data.get("tts_backend")
        asr_ok = bool(slv_data.get("asr")) and slv_h.get("ok", False)
        tts_ok = bool(slv_data.get("tts")) and slv_h.get("ok", False)

        # SLV client connection state (guarded — absent on some builds).
        slv_healthy = None
        slv_reconnecting = None
        if slv is not None:
            try:
                fn = getattr(slv, "is_healthy", None)
                if callable(fn):
                    slv_healthy = bool(fn())
            except Exception:
                slv_healthy = None
            try:
                fn = getattr(slv, "is_reconnecting", None)
                if callable(fn):
                    slv_reconnecting = bool(fn())
            except Exception:
                slv_reconnecting = None

        return {
            "llm": {
                "name": getattr(cfg, "llm_model", None),
                "endpoint": llm_endpoint,
                "ok": llm.get("ok", False),
            },
            "asr": {"name": asr_backend, "ok": asr_ok},
            "tts": {"name": tts_backend, "ok": tts_ok},
            "slv": {"ok": slv_healthy, "reconnecting": slv_reconnecting},
        }
