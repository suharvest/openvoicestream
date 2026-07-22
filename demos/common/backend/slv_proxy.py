"""Shared SLV client for demo backends.

Wraps a single ``httpx.AsyncClient`` and provides:

* :meth:`SLVProxy.probe` — best-effort aggregation of ``/health``,
  ``/asr/capabilities``, ``/tts/capabilities`` and ``/admin/backend/status``.
  Each sub-request is independently guarded so one failing endpoint never
  takes down the whole snapshot (partial results + per-endpoint errors).
* :meth:`SLVProxy.admin_get` / :meth:`SLVProxy.admin_post` — admin proxy that
  attaches the ``X-Admin-Key`` header (value from env ``SLV_ADMIN_KEY``).
  The admin key never leaves the demo backend; browsers talk to the demo
  backend only.

Environment:
    SLV_URL        base URL of the SLV server (default ``http://127.0.0.1:8621``)
    SLV_ADMIN_KEY  shared secret forwarded as ``X-Admin-Key`` (optional; on a
                   loopback deployment SLV allows admin routes without a key)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

DEFAULT_SLV_URL = "http://127.0.0.1:8621"
DEFAULT_TIMEOUT_S = 5.0
# Engine reloads load TRT engines from disk — allow minutes, not seconds.
RELOAD_TIMEOUT_S = float(os.environ.get("SLV_RELOAD_TIMEOUT_S") or 300.0)


@dataclass
class ProbeResult:
    """Aggregated SLV snapshot. ``reachable`` is True when at least /health answered."""

    reachable: bool = False
    health: Optional[dict] = None
    asr_capabilities: Optional[dict] = None
    tts_capabilities: Optional[dict] = None
    backend_status: Optional[dict] = None
    errors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reachable": self.reachable,
            "health": self.health,
            "asr_capabilities": self.asr_capabilities,
            "tts_capabilities": self.tts_capabilities,
            "backend_status": self.backend_status,
            "errors": self.errors,
        }


class SLVProxy:
    """Async client for one SLV server instance."""

    def __init__(
        self,
        base_url: str | None = None,
        admin_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SLV_URL") or DEFAULT_SLV_URL).rstrip("/")
        # Explicit empty string means "no key" (loopback deployments).
        if admin_key is None:
            admin_key = os.environ.get("SLV_ADMIN_KEY") or None
        self.admin_key = admin_key or None
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_s,
            transport=transport,
            # SLV may sit behind Tailscale; never let a system proxy intercept.
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── plain (non-admin) helpers ───────────────────────────────────────────

    async def _get_json(self, path: str) -> dict:
        resp = await self._client.get(path)
        resp.raise_for_status()
        return resp.json()

    async def get(self, path: str, params: Any = None) -> httpx.Response:
        """Plain (non-admin) GET passthrough — caller inspects status/body."""
        return await self._client.get(path, params=params)

    def stream_post(self, path: str, json: Any = None):
        """Streaming POST — returns httpx's async streaming context manager.

        Usage::

            cm = proxy.stream_post("/tts/stream", json=payload)
            async with cm as resp:
                async for chunk in resp.aiter_raw():
                    ...

        ``aiter_raw`` yields bytes exactly as they arrive from SLV, so a
        proxying demo backend can forward chunk-by-chunk without buffering
        (TTFA must survive the proxy hop).
        """
        return self._client.stream("POST", path, json=json)

    # ── admin proxy ─────────────────────────────────────────────────────────

    def _admin_headers(self) -> dict:
        return {"X-Admin-Key": self.admin_key} if self.admin_key else {}

    async def admin_get(self, path: str) -> httpx.Response:
        return await self._client.get(path, headers=self._admin_headers())

    async def admin_post(
        self, path: str, json: Any = None, timeout_s: float | None = None
    ) -> httpx.Response:
        kwargs: dict = {"json": json, "headers": self._admin_headers()}
        if timeout_s is not None:
            kwargs["timeout"] = httpx.Timeout(timeout_s, connect=10.0)
        return await self._client.post(path, **kwargs)

    # ── aggregation ─────────────────────────────────────────────────────────

    async def probe(self) -> ProbeResult:
        """Fetch health + capabilities + admin backend status, tolerating
        individual endpoint failures (result carries per-endpoint errors)."""
        result = ProbeResult()

        try:
            result.health = await self._get_json("/health")
            result.reachable = True
        except Exception as exc:  # noqa: BLE001 — degrade, never raise
            result.errors["health"] = _err_str(exc)

        for attr, path in (
            ("asr_capabilities", "/asr/capabilities"),
            ("tts_capabilities", "/tts/capabilities"),
        ):
            try:
                setattr(result, attr, await self._get_json(path))
                result.reachable = True
            except Exception as exc:  # noqa: BLE001
                result.errors[path.lstrip("/")] = _err_str(exc)

        try:
            resp = await self.admin_get("/admin/backend/status")
            resp.raise_for_status()
            result.backend_status = resp.json()
            result.reachable = True
        except Exception as exc:  # noqa: BLE001
            result.errors["admin/backend/status"] = _err_str(exc)

        return result

    async def reload_backend(
        self, kind: str, profile: str, drain_timeout_s: float | None = None
    ) -> httpx.Response:
        """Proxy ``POST /admin/backend/reload``. Caller validates ``profile``."""
        payload: dict = {"kind": kind, "profile": profile}
        if drain_timeout_s is not None:
            payload["drain_timeout_s"] = drain_timeout_s
        return await self.admin_post(
            "/admin/backend/reload", json=payload, timeout_s=RELOAD_TIMEOUT_S
        )


def _err_str(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return f"{type(exc).__name__}: {exc}"


def proxy_from_env(**overrides: Any) -> SLVProxy:
    """Build an SLVProxy from SLV_URL / SLV_ADMIN_KEY env vars."""
    return SLVProxy(**overrides)
