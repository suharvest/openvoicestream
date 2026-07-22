"""Gallery /api/profiles?kind= loadable-filtering + graceful fallback.

The gallery narrows the switch listing to profiles the SLV reports it can
actually load for the requested kind (via /admin/backend/loadable). When the
SLV can't answer (missing endpoint / down / bad admin key), the gallery must
fall back to the existing platform-filtered listing untouched.
"""

from __future__ import annotations

from tests.conftest import gallery_client


def _loadable_body(tts, asr):
    return {
        "tts": {"loadable": tts, "unloadable": [], "invalid": []},
        "asr": {"loadable": asr, "unloadable": [], "invalid": []},
    }


async def test_filters_by_kind_when_loadable_present(mock_slv, profiles_dir):
    app, state = mock_slv
    # Platform filter keeps the two jetson-* profiles; loadable narrows further.
    state["loadable"] = _loadable_body(
        tts=["jetson-qwen3asr-moss-nx"],
        asr=["jetson-qwen3asr-moss-nx", "jetson-kokoro-trt"],
    )
    async with gallery_client(app, profiles_dir) as client:
        tts = (await client.get("/api/profiles?kind=tts")).json()
        asr = (await client.get("/api/profiles?kind=asr")).json()

    assert tts["loadable_filtered"] is True
    assert [p["name"] for p in tts["profiles"]] == ["jetson-qwen3asr-moss-nx"]

    assert asr["loadable_filtered"] is True
    # kokoro-trt is TTS-only (asr_backend=None) so the model-dedup layer drops
    # it from the ASR list even though the SLV vacuously reports it asr-loadable
    # — a TTS profile is not a meaningful ASR choice.
    assert {p["name"] for p in asr["profiles"]} == {"jetson-qwen3asr-moss-nx"}


async def test_empty_loadable_for_kind_yields_empty_filtered(mock_slv, profiles_dir):
    app, state = mock_slv
    state["loadable"] = _loadable_body(tts=[], asr=["jetson-kokoro-trt"])
    async with gallery_client(app, profiles_dir) as client:
        tts = (await client.get("/api/profiles?kind=tts")).json()

    # Device can't load any TTS profile → empty list, but the pre-flight DID run.
    assert tts["loadable_filtered"] is True
    assert tts["profiles"] == []


async def test_no_kind_param_does_not_filter(mock_slv, profiles_dir):
    app, state = mock_slv
    state["loadable"] = _loadable_body(tts=["jetson-qwen3asr-moss-nx"], asr=[])
    async with gallery_client(app, profiles_dir) as client:
        body = (await client.get("/api/profiles")).json()

    # No kind → unfiltered platform listing (both jetson profiles), no pre-flight.
    assert body["loadable_filtered"] is False
    assert {p["name"] for p in body["profiles"]} == {
        "jetson-qwen3asr-moss-nx", "jetson-kokoro-trt"
    }


async def test_fallback_when_loadable_endpoint_absent(mock_slv, profiles_dir):
    app, state = mock_slv
    state["loadable"] = None  # old SLV image → 404
    async with gallery_client(app, profiles_dir) as client:
        body = (await client.get("/api/profiles?kind=tts")).json()

    assert body["loadable_filtered"] is False
    assert {p["name"] for p in body["profiles"]} == {
        "jetson-qwen3asr-moss-nx", "jetson-kokoro-trt"
    }


async def test_fallback_when_slv_errors(mock_slv, profiles_dir):
    app, state = mock_slv
    state["loadable"] = _loadable_body(tts=["jetson-qwen3asr-moss-nx"], asr=[])
    state["fail"].add("loadable")  # SLV returns 500 for the pre-flight
    async with gallery_client(app, profiles_dir) as client:
        body = (await client.get("/api/profiles?kind=tts")).json()

    assert body["loadable_filtered"] is False
    assert {p["name"] for p in body["profiles"]} == {
        "jetson-qwen3asr-moss-nx", "jetson-kokoro-trt"
    }


async def test_fallback_when_admin_key_rejected(mock_slv, profiles_dir):
    app, state = mock_slv
    state["admin_key"] = "needed"  # SLV requires admin key
    state["loadable"] = _loadable_body(tts=["jetson-qwen3asr-moss-nx"], asr=[])
    # gallery proxy has NO admin key → /admin/backend/loadable → 401 → fallback.
    async with gallery_client(app, profiles_dir, admin_key=None) as client:
        body = (await client.get("/api/profiles?kind=tts")).json()

    assert body["loadable_filtered"] is False
    assert {p["name"] for p in body["profiles"]} == {
        "jetson-qwen3asr-moss-nx", "jetson-kokoro-trt"
    }
