"""v2v-chat demo backend tests.

Lightweight by design: the interesting parts (mic capture, WS /v2v/stream
protocol, barge-in, playback) run in the browser against SLV directly; the
backend only serves statics and /api/config. The directory name contains a
hyphen, so the module is loaded from its file path (conftest helper).
"""

from __future__ import annotations

import json

import httpx

from tests.conftest import DEMOS_DIR, load_demo_backend

_APP_DIR = DEMOS_DIR / "v2v-chat"

v2v_chat = load_demo_backend("v2v-chat")


def client(slv_url: str = "http://127.0.0.1:8621") -> httpx.AsyncClient:
    app = v2v_chat.create_app(slv_url=slv_url)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://v2v-chat"
    )


# ── /healthz ─────────────────────────────────────────────────────────────────


async def test_healthz():
    async with client() as c:
        resp = await c.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["service"] == "slv-demo-v2v-chat"


# ── /api/config ──────────────────────────────────────────────────────────────


async def test_config_loopback_slv():
    async with client("http://127.0.0.1:8621") as c:
        resp = await c.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slv_url"] == "http://127.0.0.1:8621"
    assert body["v2v_path"] == "/v2v/stream"
    ws = body["ws"]
    assert ws == {"scheme": "ws", "host": "127.0.0.1", "port": 8621, "loopback": True}


async def test_config_remote_slv_not_loopback():
    async with client("http://192.168.3.7:8621") as c:
        resp = await c.get("/api/config")
    ws = resp.json()["ws"]
    assert ws["host"] == "192.168.3.7"
    assert ws["port"] == 8621
    assert ws["loopback"] is False
    assert ws["scheme"] == "ws"


async def test_config_https_slv_maps_to_wss_default_port():
    async with client("https://slv.example.com") as c:
        resp = await c.get("/api/config")
    ws = resp.json()["ws"]
    assert ws == {"scheme": "wss", "host": "slv.example.com", "port": 443,
                  "loopback": False}


async def test_config_reads_env_when_unset(monkeypatch):
    monkeypatch.setenv("SLV_URL", "http://10.0.0.5:9000/")
    app = v2v_chat.create_app()  # no explicit slv_url → env
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://v2v-chat"
    ) as c:
        resp = await c.get("/api/config")
    body = resp.json()
    assert body["slv_url"] == "http://10.0.0.5:9000"  # trailing slash stripped
    assert body["ws"]["host"] == "10.0.0.5"
    assert body["ws"]["loopback"] is False


# ── static frontend ──────────────────────────────────────────────────────────


async def test_index_served():
    async with client() as c:
        resp = await c.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "app.js" in resp.text  # page wires the demo module


async def test_app_js_served_and_uses_v2v_client():
    async with client() as c:
        resp = await c.get("/app.js")
    assert resp.status_code == 200
    assert "V2VStreamClient" in resp.text
    assert "/common/v2v-client.js" in resp.text


async def test_common_assets_mounted():
    async with client() as c:
        for path in ("/common/ui.css", "/common/ui.js", "/common/v2v-client.js",
                     "/common/mic-capture.js", "/common/mic-worklet.js"):
            resp = await c.get(path)
            assert resp.status_code == 200, path


# ── demo.json ────────────────────────────────────────────────────────────────


def test_demo_json_schema():
    data = json.loads((_APP_DIR / "demo.json").read_text(encoding="utf-8"))
    assert data["id"] == "v2v-chat"
    assert data["port"] == 8703
    assert data["needs"] == ["asr", "tts"]
    assert data["name"]["zh"] and data["name"]["en"]
    assert data["description"]["zh"] and data["description"]["en"]
