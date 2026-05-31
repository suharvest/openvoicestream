"""SLVClient against a mock /v2v/stream WS server."""
from __future__ import annotations

import asyncio
import json
import struct

import pytest
import websockets
from websockets.asyncio.server import serve

from openvoicestream_agent.slv_client import (
    ASREndpoint,
    ASRFinal,
    ASRPartial,
    SLVClient,
    TTSAudio,
    TTSDone,
    TTSSentenceDone,
    TTSStarted,
)


async def _mock_server(received: list, ready: asyncio.Event):
    async def handler(ws):
        ready.set()
        # 1. read config frame
        cfg_msg = await ws.recv()
        received.append(("config", cfg_msg))
        # 2. push canonical event sequence
        await ws.send(json.dumps({"type": "asr_partial", "text": "你", "is_stable": False}))
        await ws.send(json.dumps({"type": "asr_endpoint"}))
        await ws.send(json.dumps({
            "type": "asr_final", "text": "你好", "session_complete": False,
        }))
        await ws.send(json.dumps({"type": "tts_started", "sentence": "嗨"}))
        # First binary: 4-byte LE sample-rate header + dummy PCM
        sr = 24000
        pcm = b"\x01\x00" * 8
        await ws.send(struct.pack("<I", sr) + pcm)
        # Second binary: pure PCM
        await ws.send(b"\x02\x00" * 4)
        await ws.send(json.dumps({"type": "tts_sentence_done", "sentence": "嗨"}))
        await ws.send(json.dumps({"type": "tts_done"}))
        # 3. wait for any client frames that arrive during the test
        try:
            async for msg in ws:
                received.append(("from_client", msg))
        except websockets.ConnectionClosed:
            return

    server = await serve(handler, "127.0.0.1", 0)
    return server


@pytest.mark.asyncio
async def test_slv_client_decodes_event_sequence():
    received: list = []
    ready = asyncio.Event()
    server = await _mock_server(received, ready)
    host, port = server.sockets[0].getsockname()[:2]

    client = SLVClient(f"ws://{host}:{port}", {"asr_language": "zh", "tts_language": "zh"})
    await client.connect()

    collected = []

    async def collect():
        async for evt in client.events():
            collected.append(evt)
            if isinstance(evt, TTSDone):
                return

    try:
        await asyncio.wait_for(collect(), timeout=3.0)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()

    # Config frame received by server
    assert received[0][0] == "config"
    cfg = json.loads(received[0][1])
    assert cfg["type"] == "config"
    assert cfg["multi_utterance"] is True  # invariant 1 enforced

    types = [type(e) for e in collected]
    assert ASRPartial in types
    assert ASREndpoint in types
    assert ASRFinal in types
    assert TTSStarted in types
    assert TTSAudio in types
    assert TTSSentenceDone in types
    assert TTSDone in types

    # First TTSAudio: 4-byte SR stripped, sample_rate = 24000
    audios = [e for e in collected if isinstance(e, TTSAudio)]
    assert len(audios) == 2
    assert audios[0].sample_rate == 24000
    assert audios[0].pcm == b"\x01\x00" * 8
    assert audios[1].sample_rate == 24000  # cached for subsequent frames
    assert audios[1].pcm == b"\x02\x00" * 4

    # asr_final session_complete=False survived
    final = next(e for e in collected if isinstance(e, ASRFinal))
    assert final.text == "你好"
    assert final.session_complete is False


@pytest.mark.asyncio
async def test_slv_client_send_methods_emit_correct_payloads():
    received: list = []
    ready = asyncio.Event()
    server = await _mock_server(received, ready)
    host, port = server.sockets[0].getsockname()[:2]

    client = SLVClient(f"ws://{host}:{port}", {"asr_language": "zh"})
    await client.connect()

    await client.send_audio(b"\xab\xcd" * 16)
    await client.send_text("hello")
    await client.flush_tts()
    await client.abort()
    await client.asr_eos()

    # Give the server a moment to receive
    await asyncio.sleep(0.2)
    await client.close()
    server.close()
    await server.wait_closed()

    client_frames = [m for tag, m in received if tag == "from_client"]
    # binary frame
    assert any(isinstance(m, (bytes, bytearray)) and bytes(m) == b"\xab\xcd" * 16 for m in client_frames)
    # JSON frames
    json_frames = [json.loads(m) for m in client_frames if isinstance(m, str)]
    types = [j["type"] for j in json_frames]
    assert "text" in types
    assert "tts_flush" in types
    assert "abort" in types
    assert "asr_eos" in types
    text_frame = next(j for j in json_frames if j["type"] == "text")
    assert text_frame["text"] == "hello"


# ── server-loop frames (#37 Phase 2-product) ───────────────────────────


async def _tool_call_server(received: list):
    """Mock server that emits a SERVER_TOOL_CALL then echoes client frames."""
    async def handler(ws):
        received.append(("config", await ws.recv()))
        await ws.send(json.dumps({
            "type": "tool_call",
            "id": "call_x",
            "name": "wave",
            "arguments": {"side": "left"},
            "timeout_s": 12.0,
        }))
        try:
            async for msg in ws:
                received.append(("from_client", msg))
        except websockets.ConnectionClosed:
            return

    return await serve(handler, "127.0.0.1", 0)


@pytest.mark.asyncio
async def test_advertise_and_tool_result_wire_frames():
    from openvoicestream_agent.slv_client import ServerToolCall

    received: list = []
    server = await _tool_call_server(received)
    host, port = server.sockets[0].getsockname()[:2]

    client = SLVClient(f"ws://{host}:{port}", {"asr_language": "zh"})
    await client.connect()

    # advertise tools (CLIENT_TOOL_ADVERTISE)
    await client.advertise_tools(
        [{"type": "function", "function": {"name": "wave", "description": "", "parameters": {}}}],
        system_prompt="SP",
        llm_params={"model": "m"},
    )

    # Receive the SERVER_TOOL_CALL the mock server pushed.
    got: list = []

    async def collect():
        async for evt in client.events():
            got.append(evt)
            if isinstance(evt, ServerToolCall):
                return

    await asyncio.wait_for(collect(), timeout=3.0)

    # reply (CLIENT_TOOL_RESULT)
    tc = next(e for e in got if isinstance(e, ServerToolCall))
    await client.send_tool_result(
        tc.id, tc.name, ok=True, result={"started": True}
    )

    await asyncio.sleep(0.2)
    await client.close()
    server.close()
    await server.wait_closed()

    # Parsed SERVER_TOOL_CALL event shape.
    assert tc.id == "call_x"
    assert tc.name == "wave"
    assert tc.arguments == {"side": "left"}
    assert tc.timeout_s == 12.0

    client_frames = [json.loads(m) for tag, m in received
                     if tag == "from_client" and isinstance(m, str)]
    adv = next(j for j in client_frames if j["type"] == "tool_advertise")
    assert adv["tools"][0]["function"]["name"] == "wave"
    assert adv["system_prompt"] == "SP"
    assert adv["llm_params"] == {"model": "m"}
    res = next(j for j in client_frames if j["type"] == "tool_result")
    assert res["id"] == "call_x"
    assert res["name"] == "wave"
    assert res["ok"] is True
    assert res["result"] == {"started": True}


# ── is_healthy / SLVReconnectError ─────────────────────────────────────

@pytest.mark.asyncio
async def test_is_healthy_false_before_connect():
    from openvoicestream_agent.slv_client import SLVClient

    client = SLVClient("ws://127.0.0.1:1", {"foo": "bar"})
    assert client.is_healthy() is False


@pytest.mark.asyncio
async def test_is_healthy_false_after_close():
    received: list = []
    ready = asyncio.Event()
    server = await _mock_server(received, ready)
    try:
        port = server.sockets[0].getsockname()[1]
        client = SLVClient(f"ws://127.0.0.1:{port}", {"foo": "bar"})
        await client.connect()
        await asyncio.wait_for(ready.wait(), timeout=2.0)
        assert client.is_healthy() is True
        await client.close()
        assert client.is_healthy() is False
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_is_healthy_false_when_reader_dies():
    """Server closes the WS → reader exits → is_healthy() returns False."""
    from openvoicestream_agent.slv_client import SLVClient

    closed = asyncio.Event()

    async def handler(ws):
        await ws.recv()  # read config
        # Wait past the limiter-race grace window so connect() returns
        # healthy first; THEN close so we exercise is_healthy() detecting
        # a post-connect reader exit.
        await asyncio.sleep(0.15)
        await ws.close()
        closed.set()

    server = await serve(handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = SLVClient(f"ws://127.0.0.1:{port}", {"foo": "bar"})
        await client.connect()
        assert client.is_healthy() is True  # immediately after connect
        # Drain the queued SLVError so reader has fully exited
        await asyncio.wait_for(closed.wait(), timeout=2.0)
        # Give reader a beat to finish its finally block
        for _ in range(20):
            if not client.is_healthy():
                break
            await asyncio.sleep(0.05)
        assert client.is_healthy() is False
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_reconnect_error_when_server_keeps_closing():
    """All attempts rejected inside grace window → raises SLVReconnectError."""
    from openvoicestream_agent.slv_client import SLVClient, SLVReconnectError

    async def reject_handler(ws):
        # Accept, immediately close — triggers reader-done inside grace.
        try:
            await asyncio.wait_for(ws.recv(), timeout=0.1)
        except Exception:
            pass
        await ws.close()

    server = await serve(reject_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = SLVClient(f"ws://127.0.0.1:{port}", {"foo": "bar"})
        # Speed up the test — shrink backoffs.
        client._RECONNECT_BACKOFFS = (0.01, 0.01, 0.01)
        with pytest.raises(SLVReconnectError):
            await client.connect()
    finally:
        await client.close()
        server.close()
        await server.wait_closed()
