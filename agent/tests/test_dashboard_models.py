"""reBot ArmDashboardPlugin /api/state "models" block (ASR/TTS/LLM status).

Covers:
  - models block has the right names + ok flags from mocked /health responses
  - a health-fetch failure → ok=false but /api/state still returns arm/frame
    keys (no exception)
  - the 10s cache TTL prevents a re-fetch within the window (call count gated)

The SLV and edge-llm /health GETs are mocked at the ``_fetch_health`` boundary
so the test never touches the real network.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from ovs_agent.apps.voice_rebot_arm.dashboard_plugin import ArmDashboardPlugin


def _mk_app():
    app = MagicMock()
    app.config = SimpleNamespace(
        llm_model="Qwen/Qwen3-4B-AWQ",
        llm_base_url="http://edge-llm:8000/v1",
        slv_url="ws://slv-host:8621/v2v/stream",
        slv_http_base="http://slv-host:8621",
    )
    app.slv = SimpleNamespace(
        is_healthy=lambda: True,
        is_reconnecting=lambda: False,
    )
    app.plugins = []
    return app


def _mk_plugin(app):
    plugin = ArmDashboardPlugin(app, {"port": 0})
    return plugin


async def _get_state(plugin, unused_tcp_port):
    """Start the plugin HTTP server on a real port and GET /api/state."""
    plugin.cfg["port"] = unused_tcp_port
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.get(base + "/api/state")
            assert r.status == 200
            return await r.json()
    finally:
        await plugin.stop()


# ── names + ok flags ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_models_block_names_and_ok(monkeypatch, unused_tcp_port):
    app = _mk_app()
    plugin = _mk_plugin(app)

    async def fake_fetch(url):
        if "edge-llm" in url:
            return True, {}  # edge-llm /health 200
        if "slv-host" in url:
            return True, {
                "tts": True, "tts_backend": "matcha_trt",
                "asr": True, "asr_backend": "trt_edgellm",
            }
        return False, None

    monkeypatch.setattr(ArmDashboardPlugin, "_fetch_health", staticmethod(fake_fetch))

    st = await _get_state(plugin, unused_tcp_port)
    m = st["models"]
    assert m["llm"]["name"] == "Qwen/Qwen3-4B-AWQ"
    assert m["llm"]["endpoint"] == "edge-llm:8000"
    assert m["llm"]["ok"] is True
    assert m["asr"]["name"] == "trt_edgellm"
    assert m["asr"]["ok"] is True
    assert m["tts"]["name"] == "matcha_trt"
    assert m["tts"]["ok"] is True
    assert m["slv"]["ok"] is True
    assert m["slv"]["reconnecting"] is False


@pytest.mark.asyncio
async def test_asr_tts_ok_false_when_slv_health_down(monkeypatch, unused_tcp_port):
    """SLV /health booleans false → asr/tts ok=false even if names are present."""
    app = _mk_app()
    plugin = _mk_plugin(app)

    async def fake_fetch(url):
        if "edge-llm" in url:
            return True, {}
        if "slv-host" in url:
            return True, {
                "tts": False, "tts_backend": "matcha_trt",
                "asr": False, "asr_backend": "trt_edgellm",
            }
        return False, None

    monkeypatch.setattr(ArmDashboardPlugin, "_fetch_health", staticmethod(fake_fetch))

    st = await _get_state(plugin, unused_tcp_port)
    m = st["models"]
    assert m["asr"]["name"] == "trt_edgellm"
    assert m["asr"]["ok"] is False
    assert m["tts"]["ok"] is False


# ── health failure must not break /api/state ─────────────────────────


@pytest.mark.asyncio
async def test_health_failure_degrades_gracefully(monkeypatch, unused_tcp_port):
    """A dead SLV + edge-llm → ok=false / null names, but /api/state still
    returns the arm + frame keys without raising."""
    app = _mk_app()
    plugin = _mk_plugin(app)

    # Simulate a network failure the way _fetch_health reports it: (False, None).
    async def failing_fetch(url):
        return False, None

    monkeypatch.setattr(
        ArmDashboardPlugin, "_fetch_health", staticmethod(failing_fetch)
    )

    st = await _get_state(plugin, unused_tcp_port)
    # /api/state must still carry the core keys.
    assert "arm" in st
    assert "frame_seq" in st
    assert "events" in st
    m = st["models"]
    assert m["llm"]["ok"] is False
    assert m["asr"]["name"] is None
    assert m["asr"]["ok"] is False
    assert m["tts"]["name"] is None
    assert m["tts"]["ok"] is False
    # LLM name still comes from config even when the health probe fails.
    assert m["llm"]["name"] == "Qwen/Qwen3-4B-AWQ"


@pytest.mark.asyncio
async def test_slv_client_methods_absent(monkeypatch, unused_tcp_port):
    """is_healthy / is_reconnecting absent on the SLV client → slv.ok=None,
    no exception."""
    app = _mk_app()
    app.slv = SimpleNamespace()  # no is_healthy / is_reconnecting
    plugin = _mk_plugin(app)

    async def fake_fetch(url):
        if "slv-host" in url:
            return True, {"asr": True, "asr_backend": "x", "tts": True,
                          "tts_backend": "y"}
        return True, {}

    monkeypatch.setattr(ArmDashboardPlugin, "_fetch_health", staticmethod(fake_fetch))

    st = await _get_state(plugin, unused_tcp_port)
    assert st["models"]["slv"]["ok"] is None
    assert st["models"]["slv"]["reconnecting"] is None


# ── cache TTL gates re-fetch ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_ttl_prevents_refetch(monkeypatch):
    """Two _models_block() calls within the TTL window must fetch each /health
    URL only once (cached)."""
    app = _mk_app()
    plugin = _mk_plugin(app)

    calls: list[str] = []

    async def counting_fetch(url):
        calls.append(url)
        if "slv-host" in url:
            return True, {"asr": True, "asr_backend": "a", "tts": True,
                          "tts_backend": "t"}
        return True, {}

    monkeypatch.setattr(
        ArmDashboardPlugin, "_fetch_health", staticmethod(counting_fetch)
    )

    await plugin._models_block()
    n_after_first = len(calls)
    assert n_after_first == 2  # one llm + one slv

    # Second call within the TTL window → served from cache, no new fetch.
    await plugin._models_block()
    assert len(calls) == n_after_first, "cache TTL should have prevented re-fetch"


@pytest.mark.asyncio
async def test_cache_refetches_after_ttl(monkeypatch):
    """After the TTL expires (monotonic advanced) the next call re-fetches."""
    import ovs_agent.apps.voice_rebot_arm.dashboard_plugin as mod

    app = _mk_app()
    plugin = _mk_plugin(app)

    calls: list[str] = []

    async def counting_fetch(url):
        calls.append(url)
        if "slv-host" in url:
            return True, {"asr": True, "asr_backend": "a", "tts": True,
                          "tts_backend": "t"}
        return True, {}

    monkeypatch.setattr(
        ArmDashboardPlugin, "_fetch_health", staticmethod(counting_fetch)
    )

    fake_now = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: fake_now[0])

    await plugin._models_block()
    assert len(calls) == 2
    # Advance past the TTL.
    fake_now[0] += mod._HEALTH_TTL_S + 1.0
    await plugin._models_block()
    assert len(calls) == 4  # re-fetched both
