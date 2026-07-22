"""TTS Playground backend tests.

Covers:
* /healthz liveness
* /api/speakers passthrough (body + non-200 verdicts + SLV-down 502)
* /api/tts/stream streaming proxy — request-field forwarding, byte-exact
  passthrough, upstream error passthrough, and the load-bearing property:
  chunks are forwarded as they arrive, NOT buffered (asserted by driving the
  ASGI app directly and timestamping each http.response.body message, because
  httpx.ASGITransport aggregates response bodies and would hide buffering).
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

tts_main = load_demo_backend("tts-playground")

SPEAKERS = {
    "model_id": "moss-tts-nano-v1",
    "default_speaker_id": 0,
    "speakers": [{"id": 0, "name": "default"}, {"id": 1, "name": "warm"}],
    "supports_voice_cloning": False,
}

SAMPLE_RATE_HEADER = struct.pack("<I", 24000)


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
    chunks: list[bytes] | None = None,
    delay_s: float = 0.0,
    stream_status: int = 200,
    stream_error_body: bytes = b'{"error": "TTS not ready"}',
    speakers_status: int = 200,
    received: list | None = None,
) -> httpx.MockTransport:
    """MockTransport speaking the SLV subset this demo proxies."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tts/speakers":
            if speakers_status != 200:
                return httpx.Response(
                    speakers_status, json={"error": "TTS not ready"}
                )
            return httpx.Response(200, json=SPEAKERS)
        if request.url.path == "/tts/stream":
            if received is not None:
                received.append(json.loads(request.content))
            if stream_status != 200:
                return httpx.Response(
                    stream_status,
                    content=stream_error_body,
                    headers={"content-type": "application/json"},
                )
            return httpx.Response(
                200,
                stream=SlowPCMStream(chunks or [], delay_s),
                headers={"content-type": "application/octet-stream"},
            )
        return httpx.Response(404, json={"error": "unknown path"})

    return httpx.MockTransport(handler)


def playground_app(transport: httpx.MockTransport):
    proxy = SLVProxy(base_url="http://mock-slv", transport=transport)
    return tts_main.create_app(proxy=proxy, kiosk=False)


def playground_client(transport: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=playground_app(transport)),
        base_url="http://tts-playground",
    )


# ── /healthz ─────────────────────────────────────────────────────────────────


async def test_healthz():
    async with playground_client(make_mock_slv_transport()) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["service"] == "slv-demo-tts-playground"


# ── /api/speakers ────────────────────────────────────────────────────────────


async def test_speakers_passthrough():
    async with playground_client(make_mock_slv_transport()) as client:
        resp = await client.get("/api/speakers")
    assert resp.status_code == 200
    assert resp.json() == SPEAKERS  # byte-for-byte SLV verdict, no reshaping


async def test_speakers_passes_through_slv_503():
    transport = make_mock_slv_transport(speakers_status=503)
    async with playground_client(transport) as client:
        resp = await client.get("/api/speakers")
    assert resp.status_code == 503
    assert resp.json() == {"error": "TTS not ready"}


async def test_speakers_slv_down_returns_502():
    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with playground_client(httpx.MockTransport(_raise)) as client:
        resp = await client.get("/api/speakers")
    assert resp.status_code == 502
    assert resp.json()["error"] == "slv_unreachable"


# ── /api/tts/stream ──────────────────────────────────────────────────────────


async def test_tts_stream_forwards_fields_and_bytes():
    chunks = [SAMPLE_RATE_HEADER + b"\x01\x02" * 160, b"\x03\x04" * 160]
    received: list = []
    transport = make_mock_slv_transport(chunks=chunks, received=received)
    async with playground_client(transport) as client:
        resp = await client.post(
            "/api/tts/stream",
            json={"text": "你好", "speaker_id": 1, "speed": 1.25, "pitch": -2.0},
        )
    assert resp.status_code == 200
    # unset optional fields (language, voice) are omitted, not sent as null
    assert received == [
        {"text": "你好", "speaker_id": 1, "speed": 1.25, "pitch": -2.0}
    ]
    assert resp.content == b"".join(chunks)  # byte-exact incl. 4-byte SR header


async def test_tts_stream_alias_route_for_shared_player():
    # common/frontend/slv-client.js TTSStreamPlayer always POSTs
    # {origin}/tts/stream — the demo backend must answer there too.
    chunks = [SAMPLE_RATE_HEADER + b"\x01\x02" * 8]
    transport = make_mock_slv_transport(chunks=chunks)
    async with playground_client(transport) as client:
        resp = await client.post("/tts/stream", json={"text": "hi"})
    assert resp.status_code == 200
    assert resp.content == chunks[0]


async def test_tts_stream_passes_through_upstream_error():
    transport = make_mock_slv_transport(stream_status=503)
    async with playground_client(transport) as client:
        resp = await client.post("/api/tts/stream", json={"text": "hi"})
    assert resp.status_code == 503
    assert resp.json() == {"error": "TTS not ready"}


async def test_tts_stream_slv_down_returns_502():
    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with playground_client(httpx.MockTransport(_raise)) as client:
        resp = await client.post("/api/tts/stream", json={"text": "hi"})
    assert resp.status_code == 502
    assert resp.json()["error"] == "slv_unreachable"


async def test_tts_stream_is_not_buffered():
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
    app = playground_app(transport)

    payload = json.dumps({"text": "streaming check"}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/tts/stream",
        "raw_path": b"/api/tts/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"tts-playground"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8702),
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
    async with playground_client(make_mock_slv_transport()) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    text = resp.text
    assert "TTS" in text and "Playground" in text
    # frontend rides on the shared design system + shared SLV client
    assert "/common/ui.css" in text
    assert "/common/slv-client.js" in text
