"""Voice Clone backend tests.

Covers:
* /healthz liveness
* /api/capabilities probe — clone-capable vs clone-less backend, TTS-not-ready
  503 degradation, SLV-down 502
* /api/voices passthrough (body + SLV-down 502)
* /api/enroll — raw-WAV-body → multipart repack (file bytes + voice_id +
  optional ref_text land in SLV's form), verdict passthrough incl. the Jetson
  501 case, too-short guard, SLV-down 502
* /api/clone/stream — field forwarding (text + voice → SLV /tts/stream),
  byte-exact passthrough, and the load-bearing property: chunks are forwarded
  as they arrive, NOT buffered (asserted by driving the ASGI app directly and
  timestamping each http.response.body message, same method as
  test_tts_playground — httpx.ASGITransport aggregates bodies and would hide
  buffering)
* /tts/stream alias for the shared TTSStreamPlayer
* static frontend served at /
"""

from __future__ import annotations

import asyncio
import json
import struct
import time

import httpx

from common.backend.slv_proxy import SLVProxy
from tests.conftest import load_demo_backend

vc_main = load_demo_backend("voice-clone")

SAMPLE_RATE_HEADER = struct.pack("<I", 16000)

CAPS_CLONE = {
    "backend": "jetson.spark_tts",
    "model_id": "spark-tts-0.5b",
    "capabilities": ["streaming", "voice_clone"],
    "supports_voice_cloning": True,
    "sample_rate": 16000,
    "speakers": [],
}

CAPS_NO_CLONE = {
    "backend": "jetson.moss_tts_nano",
    "model_id": "moss-tts-nano-v1",
    "capabilities": ["streaming", "speed", "pitch"],
    "supports_voice_cloning": False,
    "sample_rate": 24000,
    "speakers": [{"id": 0, "name": "default"}],
}

VOICES = {
    "voices": [
        {"voice_id": "web-20260702-101500", "type": "clone", "sample_rate": 16000},
        {"voice_id": "alice", "type": "clone", "sample_rate": 16000},
    ],
    "voices_dir": "/opt/seeed-local-voice/data/sparktts_voices",
}

ENROLL_OK = {"voice_id": "web-1", "json": "/v/web-1.json", "npz": "/v/web-1.npz",
             "registry_count": 3}

# > backend's _MIN_ENROLL_BYTES (44 + 32000): ~1.5 s of fake 16 kHz PCM16 WAV
FAKE_WAV = b"RIFF" + b"\x00" * 40 + b"\x01\x02" * 24000


class SlowPCMStream(httpx.AsyncByteStream):
    """Fake SLV /tts/stream body: yields chunks with a delay between them,
    like a real-time synthesizer producing audio progressively."""

    def __init__(self, chunks: list[bytes], delay_s: float = 0.0) -> None:
        self._chunks = chunks
        self._delay_s = delay_s

    async def __aiter__(self):
        for i, chunk in enumerate(self._chunks):
            if i and self._delay_s:
                await asyncio.sleep(self._delay_s)
            yield chunk

    async def aclose(self) -> None:  # pragma: no cover — httpx cleanup hook
        pass


def make_mock_slv_transport(
    *,
    capabilities: dict | None = None,
    capabilities_status: int = 200,
    voices_status: int = 200,
    enroll_status: int = 200,
    enroll_response: dict | None = None,
    chunks: list[bytes] | None = None,
    delay_s: float = 0.0,
    stream_status: int = 200,
    received: list | None = None,
) -> httpx.MockTransport:
    """MockTransport speaking the SLV subset this demo proxies."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tts/capabilities":
            if capabilities_status != 200:
                return httpx.Response(
                    capabilities_status, json={"error": "TTS not ready"}
                )
            return httpx.Response(200, json=capabilities or CAPS_CLONE)
        if request.url.path == "/tts/voices":
            if voices_status != 200:
                return httpx.Response(voices_status, json={"error": "boom"})
            return httpx.Response(200, json=VOICES)
        if request.url.path == "/tts/voices/enroll":
            if received is not None:
                received.append(
                    {
                        "path": "/tts/voices/enroll",
                        "content_type": request.headers.get("content-type", ""),
                        "content": request.content,
                    }
                )
            if enroll_status != 200:
                return httpx.Response(
                    enroll_status,
                    json=enroll_response
                    or {"error": "enroll unavailable",
                        "hint": "POST /tts/voices/profile"},
                )
            return httpx.Response(200, json=enroll_response or ENROLL_OK)
        if request.url.path == "/tts/stream":
            if received is not None:
                received.append(
                    {"path": "/tts/stream", "body": json.loads(request.content)}
                )
            if stream_status != 200:
                return httpx.Response(
                    stream_status,
                    content=b'{"error": "TTS not ready"}',
                    headers={"content-type": "application/json"},
                )
            return httpx.Response(
                200,
                stream=SlowPCMStream(chunks or [], delay_s),
                headers={"content-type": "application/octet-stream"},
            )
        return httpx.Response(404, json={"error": "unknown path"})

    return httpx.MockTransport(handler)


def clone_app(transport: httpx.MockTransport):
    proxy = SLVProxy(base_url="http://mock-slv", transport=transport)
    return vc_main.create_app(proxy=proxy, kiosk=False)


def clone_client(transport: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=clone_app(transport)),
        base_url="http://voice-clone",
    )


def down_transport() -> httpx.MockTransport:
    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.MockTransport(_raise)


# ── /healthz ─────────────────────────────────────────────────────────────────


async def test_healthz():
    async with clone_client(make_mock_slv_transport()) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["service"] == "slv-demo-voice-clone"


# ── /api/capabilities ────────────────────────────────────────────────────────


async def test_capabilities_clone_supported():
    async with clone_client(make_mock_slv_transport(capabilities=CAPS_CLONE)) as client:
        resp = await client.get("/api/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["supports_voice_cloning"] is True
    assert body["backend"] == "jetson.spark_tts"
    assert body["model_id"] == "spark-tts-0.5b"


async def test_capabilities_clone_unsupported():
    async with clone_client(
        make_mock_slv_transport(capabilities=CAPS_NO_CLONE)
    ) as client:
        resp = await client.get("/api/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["supports_voice_cloning"] is False
    assert body["backend"] == "jetson.moss_tts_nano"


async def test_capabilities_tts_not_ready_degrades():
    async with clone_client(
        make_mock_slv_transport(capabilities_status=503)
    ) as client:
        resp = await client.get("/api/capabilities")
    assert resp.status_code == 200  # degraded verdict, not an opaque 5xx
    body = resp.json()
    assert body["reachable"] is True
    assert body["supports_voice_cloning"] is False
    assert body["reason"] == "tts_not_ready"


async def test_capabilities_slv_down_returns_502():
    async with clone_client(down_transport()) as client:
        resp = await client.get("/api/capabilities")
    assert resp.status_code == 502
    body = resp.json()
    assert body["reachable"] is False
    assert body["supports_voice_cloning"] is None
    assert body["error"] == "slv_unreachable"


# ── /api/voices ──────────────────────────────────────────────────────────────


async def test_voices_passthrough():
    async with clone_client(make_mock_slv_transport()) as client:
        resp = await client.get("/api/voices")
    assert resp.status_code == 200
    assert resp.json() == VOICES  # byte-for-byte SLV verdict, no reshaping


async def test_voices_slv_down_returns_502():
    async with clone_client(down_transport()) as client:
        resp = await client.get("/api/voices")
    assert resp.status_code == 502
    assert resp.json()["error"] == "slv_unreachable"


# ── /api/enroll ──────────────────────────────────────────────────────────────


async def test_enroll_repacks_wav_into_multipart():
    received: list = []
    async with clone_client(make_mock_slv_transport(received=received)) as client:
        resp = await client.post(
            "/api/enroll",
            params={"voice_id": "alice", "ref_text": "hello there"},
            content=FAKE_WAV,
            headers={"Content-Type": "audio/wav"},
        )
    assert resp.status_code == 200
    assert resp.json() == ENROLL_OK  # SLV verdict passes through untouched

    assert len(received) == 1
    fwd = received[0]
    assert fwd["path"] == "/tts/voices/enroll"
    # httpx-encoded multipart: form fields + the WAV bytes all present
    assert fwd["content_type"].startswith("multipart/form-data")
    assert b'name="voice_id"' in fwd["content"]
    assert b"alice" in fwd["content"]
    assert b'name="ref_text"' in fwd["content"]
    assert b"hello there" in fwd["content"]
    assert b'name="file"' in fwd["content"]
    assert b'filename="enroll.wav"' in fwd["content"]
    assert FAKE_WAV in fwd["content"]  # audio bytes forwarded verbatim


async def test_enroll_omits_ref_text_and_autogenerates_voice_id():
    received: list = []
    async with clone_client(make_mock_slv_transport(received=received)) as client:
        resp = await client.post("/api/enroll", content=FAKE_WAV)
    assert resp.status_code == 200
    fwd = received[0]
    assert b'name="ref_text"' not in fwd["content"]
    assert b'name="voice_id"' in fwd["content"]
    assert b"web-" in fwd["content"]  # auto-generated web-<timestamp> id


async def test_enroll_passes_through_jetson_501():
    # On a Jetson SLV answers 501 (in-process enroll needs the GPU-host
    # PyTorch stack) — the browser must see that verdict, not a masked 500.
    transport = make_mock_slv_transport(
        enroll_status=501,
        enroll_response={
            "error": "In-process SparkTTS enrollment is unavailable on this host",
            "hint": "POST /tts/voices/profile with a host-enrolled .json+.npz",
        },
    )
    async with clone_client(transport) as client:
        resp = await client.post("/api/enroll", content=FAKE_WAV)
    assert resp.status_code == 501
    assert "hint" in resp.json()


async def test_enroll_rejects_too_short_audio_without_calling_slv():
    received: list = []
    async with clone_client(make_mock_slv_transport(received=received)) as client:
        resp = await client.post("/api/enroll", content=b"RIFF tiny")
    assert resp.status_code == 400
    assert resp.json()["error"] == "audio_too_short"
    assert received == []  # never bothered SLV


async def test_enroll_slv_down_returns_502():
    async with clone_client(down_transport()) as client:
        resp = await client.post("/api/enroll", content=FAKE_WAV)
    assert resp.status_code == 502
    assert resp.json()["error"] == "slv_unreachable"


# ── /api/clone/stream ────────────────────────────────────────────────────────


async def test_clone_stream_forwards_voice_and_bytes():
    chunks = [SAMPLE_RATE_HEADER + b"\x01\x02" * 160, b"\x03\x04" * 160]
    received: list = []
    transport = make_mock_slv_transport(chunks=chunks, received=received)
    async with clone_client(transport) as client:
        resp = await client.post(
            "/api/clone/stream",
            json={"text": "你好，这是克隆声音", "voice": "web-20260702-101500"},
        )
    assert resp.status_code == 200
    # forwarded to SLV /tts/stream with the VoiceProfile `voice` selector;
    # unset optional fields (language/speed/pitch) omitted, not null
    assert received == [
        {
            "path": "/tts/stream",
            "body": {"text": "你好，这是克隆声音", "voice": "web-20260702-101500"},
        }
    ]
    assert resp.content == b"".join(chunks)  # byte-exact incl. 4-byte SR header


async def test_clone_stream_requires_voice():
    async with clone_client(make_mock_slv_transport()) as client:
        resp = await client.post("/api/clone/stream", json={"text": "hi"})
    assert resp.status_code == 422  # this demo always synthesizes with a clone


async def test_clone_stream_alias_route_for_shared_player():
    # common/frontend/slv-client.js TTSStreamPlayer always POSTs
    # {origin}/tts/stream — the demo backend must answer there too.
    chunks = [SAMPLE_RATE_HEADER + b"\x01\x02" * 8]
    transport = make_mock_slv_transport(chunks=chunks)
    async with clone_client(transport) as client:
        resp = await client.post("/tts/stream", json={"text": "hi", "voice": "alice"})
    assert resp.status_code == 200
    assert resp.content == chunks[0]


async def test_clone_stream_passes_through_upstream_error():
    transport = make_mock_slv_transport(stream_status=400)
    async with clone_client(transport) as client:
        resp = await client.post(
            "/api/clone/stream", json={"text": "hi", "voice": "ghost"}
        )
    assert resp.status_code == 400
    assert resp.json() == {"error": "TTS not ready"}


async def test_clone_stream_slv_down_returns_502():
    async with clone_client(down_transport()) as client:
        resp = await client.post(
            "/api/clone/stream", json={"text": "hi", "voice": "alice"}
        )
    assert resp.status_code == 502
    assert resp.json()["error"] == "slv_unreachable"


async def test_clone_stream_is_not_buffered():
    """Load-bearing TTFA property: the proxy must forward the first chunk long
    before the upstream finishes producing the last one.

    The mock SLV emits 4 chunks with 0.2 s pauses (total ≥ 0.6 s). We drive
    the demo ASGI app directly and timestamp every http.response.body message:
    if the proxy buffered, the first body byte would only arrive after ~0.6 s.
    """
    delay = 0.2
    chunks = [
        SAMPLE_RATE_HEADER + b"\x00\x01" * 400,
        b"\x02\x03" * 400,
        b"\x04\x05" * 400,
        b"\x06\x07" * 400,
    ]
    transport = make_mock_slv_transport(chunks=chunks, delay_s=delay)
    app = clone_app(transport)

    payload = json.dumps({"text": "streaming check", "voice": "alice"}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/clone/stream",
        "raw_path": b"/api/clone/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"voice-clone"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8705),
    }

    request_messages = [{"type": "http.request", "body": payload, "more_body": False}]

    async def receive():
        if request_messages:
            return request_messages.pop(0)
        # No disconnect: block until the response task group cancels us.
        await asyncio.Event().wait()

    events: list[tuple[float, str, int]] = []  # (t_since_start, type, nbytes)
    t0 = time.perf_counter()

    async def send(message):
        body = message.get("body") or b""
        events.append((time.perf_counter() - t0, message["type"], len(body)))

    await app(scope, receive, send)

    starts = [e for e in events if e[1] == "http.response.start"]
    bodies = [e for e in events if e[1] == "http.response.body" and e[2] > 0]
    assert starts, f"no response start: {events}"
    assert len(bodies) >= len(chunks), (
        f"expected >= {len(chunks)} separate body messages (chunked forwarding), "
        f"got {len(bodies)}: {events}"
    )
    total_bytes = sum(e[2] for e in bodies)
    assert total_bytes == sum(len(c) for c in chunks)

    first_at = bodies[0][0]
    last_at = bodies[-1][0]
    # First chunk must beat even the SECOND upstream chunk (produced at ~0.2 s);
    # the tail lands after all upstream delays (~0.6 s). Wide margins for CI.
    assert first_at < delay * 0.75, (
        f"first chunk arrived at {first_at:.3f}s — proxy is buffering "
        f"(upstream total ≈ {delay * (len(chunks) - 1):.1f}s)"
    )
    assert last_at > delay * (len(chunks) - 1) * 0.8, (
        f"last chunk at {last_at:.3f}s — mock produced audio too fast for "
        f"the assertion to be meaningful: {events}"
    )
    assert first_at < last_at / 3, (
        f"first ({first_at:.3f}s) not far enough ahead of last ({last_at:.3f}s)"
    )


# ── static frontend ──────────────────────────────────────────────────────────


async def test_index_html_served():
    async with clone_client(make_mock_slv_transport()) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    text = resp.text
    assert "Voice" in text and "Clone" in text
    # frontend rides on the shared design system + shared mic/player modules
    assert "/common/ui.css" in text
    assert "/app.js" in text


async def test_app_js_served_and_uses_shared_modules():
    async with clone_client(make_mock_slv_transport()) as client:
        resp = await client.get("/app.js")
    assert resp.status_code == 200
    text = resp.text
    assert "/common/mic-capture.js" in text
    assert "/common/slv-client.js" in text
    assert "/common/ui.js" in text
