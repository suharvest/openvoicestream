"""Gallery backend tests against the controllable mock SLV."""

from __future__ import annotations

import httpx
import pytest

from common.backend.slv_proxy import SLVProxy
from gallery.backend.main import create_app
from tests.conftest import gallery_client


# ── /healthz ─────────────────────────────────────────────────────────────────


async def test_healthz(mock_slv, profiles_dir):
    app, _ = mock_slv
    async with gallery_client(app, profiles_dir) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["service"] == "slv-demo-gallery"


# ── /api/catalog ─────────────────────────────────────────────────────────────


async def test_catalog_aggregation(mock_slv, profiles_dir):
    app, _ = mock_slv
    async with gallery_client(app, profiles_dir) as client:
        resp = await client.get("/api/catalog")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slv_reachable"] is True
    assert body["degraded"] is False

    cards = {c["id"]: c for c in body["demos"]}
    assert set(cards) == {
        "gallery", "asr-caption", "tts-playground", "v2v-chat",
        "diarization", "voice-clone",
    }
    # gallery itself is implemented and has no needs
    assert cards["gallery"]["state"] == "available"
    # P1 cards shipped: needs satisfied by the mock → available
    assert cards["asr-caption"]["state"] == "available"
    assert cards["tts-playground"]["state"] == "available"
    assert cards["v2v-chat"]["state"] == "available"
    # diarization shipped: needs (asr.streaming) satisfied by the mock
    assert cards["diarization"]["state"] == "available"
    # mock TTS has supports_voice_cloning=False → unsupported with reason
    assert cards["voice-clone"]["state"] == "unsupported"
    assert cards["voice-clone"]["reason"]["zh"]
    assert cards["voice-clone"]["reason"]["en"]
    # every card carries bilingual name/description
    for card in cards.values():
        assert card["name"]["zh"] and card["name"]["en"]
        assert card["description"]["zh"] and card["description"]["en"]


async def test_catalog_survives_partial_slv_failure(mock_slv, profiles_dir):
    app, state = mock_slv
    state["fail"] = {"backend_status", "tts_capabilities"}
    async with gallery_client(app, profiles_dir) as client:
        resp = await client.get("/api/catalog")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slv_reachable"] is True          # /health still up
    assert body["degraded"] is True
    assert "admin/backend/status" in body["errors"]
    assert "tts/capabilities" in body["errors"]
    assert len(body["demos"]) == 6                # catalog still fully rendered


async def test_catalog_slv_totally_down(profiles_dir):
    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    proxy = SLVProxy(base_url="http://mock-slv",
                     transport=httpx.MockTransport(_raise))
    gallery = create_app(proxy=proxy, profiles_dir=profiles_dir, kiosk=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gallery), base_url="http://gallery"
    ) as client:
        cat = await client.get("/api/catalog")
        status = await client.get("/api/status")

    assert cat.status_code == 200
    body = cat.json()
    assert body["slv_reachable"] is False
    assert body["degraded"] is True
    # cards degrade gracefully, no crash: shipped cards with unverifiable needs
    # become "unknown"; unshipped cards stay coming-soon; nothing flips to
    # unsupported without positive evidence.
    states = {c["id"]: c["state"] for c in body["demos"]}
    assert states["asr-caption"] == "unknown"
    assert states["diarization"] == "unknown"
    assert states["voice-clone"] == "unknown"      # not "unsupported" — no evidence
    assert states["gallery"] == "available"        # no needs → still available

    assert status.status_code == 200
    assert status.json()["slv"]["reachable"] is False


# ── /api/profiles ────────────────────────────────────────────────────────────


async def test_profiles_platform_filter(mock_slv, profiles_dir):
    app, _ = mock_slv  # health reports jetson.* backends
    async with gallery_client(app, profiles_dir) as client:
        resp = await client.get("/api/profiles")
    assert resp.status_code == 200
    body = resp.json()
    assert body["filtered"] is True
    assert body["platforms"] == ["jetson"]
    names = {p["name"] for p in body["profiles"]}
    assert names == {"jetson-qwen3asr-moss-nx", "jetson-kokoro-trt"}


async def test_profiles_unfiltered_when_slv_down(profiles_dir):
    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    proxy = SLVProxy(base_url="http://mock-slv",
                     transport=httpx.MockTransport(_raise))
    gallery = create_app(proxy=proxy, profiles_dir=profiles_dir, kiosk=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gallery), base_url="http://gallery"
    ) as client:
        resp = await client.get("/api/profiles")
    body = resp.json()
    assert body["filtered"] is False              # no platform info → list all
    assert len(body["profiles"]) == 3


# ── /api/switch ──────────────────────────────────────────────────────────────


async def test_switch_forwards_admin_key(mock_slv, profiles_dir):
    app, state = mock_slv
    state["admin_key"] = "sekrit"
    async with gallery_client(app, profiles_dir, admin_key="sekrit") as client:
        resp = await client.post(
            "/api/switch",
            json={"kind": "tts", "profile": "jetson-kokoro-trt"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "reloaded"
    # the mock recorded exactly one reload with the forwarded key
    assert len(state["received"]) == 1
    received = state["received"][0]
    assert received["headers"]["x-admin-key"] == "sekrit"
    assert received["body"] == {"kind": "tts", "profile": "jetson-kokoro-trt"}


async def test_switch_wrong_admin_key_passes_through_401(mock_slv, profiles_dir):
    app, state = mock_slv
    state["admin_key"] = "sekrit"
    async with gallery_client(app, profiles_dir, admin_key="wrong") as client:
        resp = await client.post(
            "/api/switch",
            json={"kind": "tts", "profile": "jetson-kokoro-trt"},
        )
    assert resp.status_code == 401                # SLV verdict passed through


async def test_switch_rejects_unknown_profile(mock_slv, profiles_dir):
    app, state = mock_slv
    async with gallery_client(app, profiles_dir) as client:
        resp = await client.post(
            "/api/switch",
            json={"kind": "tts", "profile": "../../etc/passwd"},
        )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "profile_not_allowed"
    assert state["received"] == []                # never reached SLV


async def test_switch_rejects_other_platform_profile(mock_slv, profiles_dir):
    """Mock SLV reports a jetson platform → rk3588-* profiles must be
    rejected even though the JSON exists in the profiles dir."""
    app, state = mock_slv
    async with gallery_client(app, profiles_dir) as client:
        resp = await client.post(
            "/api/switch",
            json={"kind": "tts", "profile": "rk3588-default"},
        )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "profile_not_allowed"
    assert state["received"] == []                # never reached SLV


async def test_switch_rejects_bad_kind(mock_slv, profiles_dir):
    app, _ = mock_slv
    async with gallery_client(app, profiles_dir) as client:
        resp = await client.post(
            "/api/switch",
            json={"kind": "llm", "profile": "jetson-kokoro-trt"},
        )
    assert resp.status_code == 422                # pydantic Literal["tts","asr"]


async def test_switch_passes_through_rollback(mock_slv, profiles_dir):
    app, state = mock_slv
    state["reload_response"] = {"status": "rolled_back", "error": "preload failed"}
    async with gallery_client(app, profiles_dir) as client:
        resp = await client.post(
            "/api/switch",
            json={"kind": "asr", "profile": "jetson-qwen3asr-moss-nx"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "rolled_back"


# ── kiosk env ────────────────────────────────────────────────────────────────


async def test_kiosk_env_enables_kiosk(mock_slv, profiles_dir, monkeypatch):
    monkeypatch.setenv("DEMO_KIOSK", "1")
    app, _ = mock_slv
    # kiosk=None → create_app reads DEMO_KIOSK from env
    async with gallery_client(app, profiles_dir, kiosk=None) as client:
        healthz = await client.get("/healthz")
        status = await client.get("/api/status")
    assert healthz.json()["kiosk"] is True
    assert status.json()["kiosk"] is True


async def test_kiosk_env_off_by_default(mock_slv, profiles_dir, monkeypatch):
    monkeypatch.delenv("DEMO_KIOSK", raising=False)
    app, _ = mock_slv
    async with gallery_client(app, profiles_dir, kiosk=None) as client:
        healthz = await client.get("/healthz")
    assert healthz.json()["kiosk"] is False
