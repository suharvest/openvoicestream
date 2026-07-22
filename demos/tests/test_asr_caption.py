"""asr-caption demo backend tests.

Lightweight by design: the interesting parts (mic capture, WS streaming)
run in the browser against SLV directly; the backend only serves statics
and /api/config. The directory name contains a hyphen, so the module is
loaded from its file path instead of an import.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx

DEMOS_DIR = Path(__file__).resolve().parent.parent
_APP_DIR = DEMOS_DIR / "asr-caption"
_MAIN_PATH = _APP_DIR / "backend" / "main.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("asr_caption_backend_main", _MAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


asr_caption = _load_module()


def client(slv_url: str = "http://127.0.0.1:8621") -> httpx.AsyncClient:
    app = asr_caption.create_app(slv_url=slv_url)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://asr-caption"
    )


# ── /healthz ─────────────────────────────────────────────────────────────────


async def test_healthz():
    async with client() as c:
        resp = await c.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["service"] == "slv-demo-asr-caption"


# ── /api/config ──────────────────────────────────────────────────────────────


async def test_config_loopback_slv():
    async with client("http://127.0.0.1:8621") as c:
        resp = await c.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slv_url"] == "http://127.0.0.1:8621"
    assert body["asr_path"] == "/asr/stream"
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
    app = asr_caption.create_app()  # no explicit slv_url → env
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://asr-caption"
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
    assert "mic-capture.js" in resp.text  # page wires the shared mic module


async def test_common_assets_mounted():
    async with client() as c:
        for path in ("/common/ui.css", "/common/ui.js", "/common/slv-client.js",
                     "/common/mic-capture.js", "/common/mic-worklet.js"):
            resp = await c.get(path)
            assert resp.status_code == 200, path


# ── demo.json ────────────────────────────────────────────────────────────────


def test_demo_json_schema():
    data = json.loads((_APP_DIR / "demo.json").read_text(encoding="utf-8"))
    assert data["id"] == "asr-caption"
    assert data["port"] == 8701
    assert data["needs"] == ["asr"]
    assert data["name"]["zh"] and data["name"]["en"]
    assert data["description"]["zh"] and data["description"]["en"]
