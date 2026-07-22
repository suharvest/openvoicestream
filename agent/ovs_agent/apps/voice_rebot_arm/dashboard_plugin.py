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


def _wav_bytes_to_pcm16_mono(data: bytes, target_sr: int = 16000) -> bytes:
    """Decode WAV bytes → mono int16 PCM at ``target_sr`` (linear resample).
    Used by the debug inject endpoint; mirrors tests/e2e/fake_audio."""
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8
    elif sw == 4:
        arr = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
    else:
        arr = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1).astype(np.int16)
    if sr != target_sr and arr.size:
        n_out = int(len(arr) * target_sr / sr)
        x0 = np.linspace(0, 1, len(arr), endpoint=False)
        x1 = np.linspace(0, 1, n_out, endpoint=False)
        arr = np.interp(x1, x0, arr.astype(np.float32)).astype(np.int16)
    return arr.tobytes()


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
        data = await request.read()
        if not data:
            return web.json_response(
                {"ok": False, "error": "empty body; POST raw WAV bytes"}, status=400
            )
        audio = getattr(self.app, "audio", None)
        if audio is None or getattr(audio, "_in_queue", None) is None:
            return web.json_response(
                {"ok": False, "error": "mic capture not active"}, status=409
            )
        try:
            sr = int(getattr(audio, "input_sr", 16000))
            pcm = _wav_bytes_to_pcm16_mono(data, target_sr=sr)
        except Exception as e:  # noqa: BLE001
            return web.json_response(
                {"ok": False, "error": f"bad wav: {e}"}, status=400
            )
        slv = getattr(self.app, "slv", None)
        if slv is None or not hasattr(slv, "send_audio"):
            return web.json_response(
                {"ok": False, "error": "agent has no SLV audio path"}, status=409
            )
        logger.warning(
            "inject_wav: feeding %d PCM bytes (%.2fs @ %dHz) straight to SLV "
            "(bypassing energy gate + mic pump)", len(pcm), len(pcm) / 2 / sr, sr,
        )
        # Wake so the agent is connected/advertised + listening, then feed the PCM
        # STRAIGHT to the SLV WS — three things had to be bypassed to make injected
        # clips arrive intact (all observed real-machine 2026-06-14):
        #  1. the energy-gated mic pump discarded low-energy syllables / the onset
        #     → send direct (slv.send_audio), not via the mic queue;
        #  2. wake() can trigger an SLV WS reconnect (idle>30s) and PCM fed before
        #     the new /v2v stream accepts is lost → wait for the WS to be ready;
        #  3. the REAL mic pump runs concurrently on the SAME WS and its frames
        #     interleave with / drown the injection (SLV transcribed ambient room
        #     audio instead of the clip) → set app._injecting to suppress real-mic
        #     forwarding (gated in _send_audio_nonblocking) for the inject window.
        # A forced asr_eos finalizes; short-English empty-final is handled SLV-side
        # by the voxedge offline-transcribe fallback. Real voice hits none of this.
        try:
            await self.app.wake(source="inject_wav")
        except Exception:
            logger.debug("inject_wav: wake failed", exc_info=True)
        await asyncio.sleep(0.5)  # let the wake tone finish (drop_while_speaking)
        for _ in range(60):  # wait up to ~6s for the (possibly reconnected) WS
            try:
                if not slv.is_reconnecting() and slv.is_healthy():
                    break
            except Exception:
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)  # settle margin after the stream is ready
        step = max(2, int(sr * 0.064) * 2)  # ~64ms frames, even-aligned
        self.app._injecting = True  # suppress real-mic forwarding during the inject
        try:
            for i in range(0, len(pcm), step):
                await slv.send_audio(pcm[i:i + step])
                await asyncio.sleep(0.064)
            await slv.send_audio(b"\x00\x00" * int(sr * 0.3))  # trailing silence
            await asyncio.sleep(0.2)
            try:
                await self.app.send_asr_eos_once()
            except Exception:
                logger.debug("inject_wav: asr_eos failed", exc_info=True)
        finally:
            self.app._injecting = False
        return web.json_response(
            {"ok": True, "pcm_bytes": len(pcm), "sr": sr, "via": "slv_direct"}
        )

    async def _api_state(self, request):  # noqa: ANN001
        from aiohttp import web

        from .dashboard_bus import BUS

        state = BUS.snapshot()
        state["arm"] = await asyncio.to_thread(self._fetch_observation)
        state["busy"] = self._busy_motion()
        state["place_bounds"] = self._place_bounds()
        # Best-effort models/status block; a dead SLV / edge-llm must never
        # break /api/state (arm + frame data must still return).
        try:
            state["models"] = await self._models_block()
        except Exception:
            logger.debug("models block failed", exc_info=True)
            state["models"] = None
        return web.json_response(state)

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
