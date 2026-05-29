"""Transport abstraction (spec §5).

Two modes:
  * ``InProcessTransport`` (default) — brain ↔ engine over in-memory
    asyncio.Queues, zero IPC, lowest latency on a single device.
  * ``WebSocketTransport`` — wraps a ws-like object (duck-typed; no FastAPI
    import) and maps recv/send onto ws.receive / send_json / send_bytes /
    close, mirroring the production app/main.py mapping:
      ws.receive()      → recv_audio / recv_event   (app/main.py:2817-2833, 2944-2952)
      ws.send_json()    → send_event                (app/main.py:2795-2801)
      ws.send_bytes()   → send_audio                (app/main.py:2802-2807)
      ws.close()        → close                     (app/main.py:3537-3547)
"""
from __future__ import annotations

import asyncio
import json as _json
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class Transport(ABC):
    """Bidirectional audio + event channel between engine and client.

    Audio frames are raw bytes (int16 PCM). Events are JSON-serialisable
    dicts. Per spec §5.
    """

    @abstractmethod
    def recv_audio(self) -> AsyncIterator[bytes]:
        """Async-iterate inbound audio (int16 PCM) chunks."""
        ...

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Send one outbound audio chunk (raw PCM or sample-rate header)."""
        ...

    @abstractmethod
    def recv_event(self) -> AsyncIterator[dict]:
        """Async-iterate inbound control events (JSON dicts)."""
        ...

    @abstractmethod
    async def send_event(self, event: dict) -> None:
        """Send one outbound control event (JSON dict)."""
        ...

    @abstractmethod
    async def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> None:
        """Close the channel."""
        ...


# A sentinel pushed onto the in-process queues to signal end-of-stream.
_CLOSE = object()


class InProcessTransport(Transport):
    """Zero-IPC transport backed by asyncio.Queues.

    Layout (from the engine's point of view):
      * inbound audio  : client → :meth:`feed_audio` → engine ``recv_audio``
      * inbound events : client → :meth:`feed_event` → engine ``recv_event``
      * outbound audio : engine ``send_audio`` → :meth:`audio_out` queue
      * outbound events: engine ``send_event`` → :meth:`events_out` queue

    The ``feed_*`` / ``*_out`` helpers are the client-side handle. Tests and
    a local mic/speaker driver push audio + control in and read PCM + events
    out without any serialization. ``multi_utterance`` callers reuse one
    transport across turns.
    """

    def __init__(self) -> None:
        self._in_audio: asyncio.Queue = asyncio.Queue()
        self._in_event: asyncio.Queue = asyncio.Queue()
        self._out_audio: asyncio.Queue = asyncio.Queue()
        self._out_event: asyncio.Queue = asyncio.Queue()
        self._closed = False

    # ── client-side feed (input) ───────────────────────────────────────

    async def feed_audio(self, chunk: bytes) -> None:
        await self._in_audio.put(chunk)

    async def feed_event(self, event: dict) -> None:
        await self._in_event.put(event)

    def end_input(self) -> None:
        """Signal no more inbound audio/events (closes the recv iterators)."""
        self._in_audio.put_nowait(_CLOSE)
        self._in_event.put_nowait(_CLOSE)

    # ── client-side drain (output) ─────────────────────────────────────

    async def audio_out(self) -> bytes:
        """Await one outbound audio chunk (client side)."""
        return await self._out_audio.get()

    async def events_out(self) -> dict:
        """Await one outbound event (client side)."""
        return await self._out_event.get()

    def drain_events_nowait(self) -> list[dict]:
        """Non-blocking: collect all currently-queued outbound events."""
        out: list[dict] = []
        while not self._out_event.empty():
            out.append(self._out_event.get_nowait())
        return out

    def drain_audio_nowait(self) -> list[bytes]:
        """Non-blocking: collect all currently-queued outbound audio chunks."""
        out: list[bytes] = []
        while not self._out_audio.empty():
            out.append(self._out_audio.get_nowait())
        return out

    # ── engine-side Transport API ──────────────────────────────────────

    async def recv_audio(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._in_audio.get()
            if item is _CLOSE or self._closed:
                return
            yield item

    async def send_audio(self, chunk: bytes) -> None:
        await self._out_audio.put(chunk)

    async def recv_event(self) -> AsyncIterator[dict]:
        while True:
            item = await self._in_event.get()
            if item is _CLOSE or self._closed:
                return
            yield item

    async def send_event(self, event: dict) -> None:
        await self._out_event.put(event)

    async def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> None:
        self._closed = True
        # Unblock any pending recv_* iterators.
        self._in_audio.put_nowait(_CLOSE)
        self._in_event.put_nowait(_CLOSE)


class WebSocketTransport(Transport):
    """Adapter over a Starlette/FastAPI-style WebSocket (duck-typed).

    Expects ``ws`` to provide async ``receive()`` (returning a dict with
    ``type`` / ``bytes`` / ``text`` keys, like Starlette), ``send_json``,
    ``send_bytes``, and ``close``. We do NOT import FastAPI — anything with
    that shape works (mirrors app/main.py:2795-2807, 2817-2833, 3537-3547).

    receive() yields one frame at a time; this transport demultiplexes binary
    (→ audio) from text (→ event) into the two recv iterators. Because a
    single ``ws.receive()`` cannot be fanned out twice, the two iterators
    share a single underlying receive loop via an internal pump.
    """

    def __init__(self, ws) -> None:
        self._ws = ws
        self._audio_q: asyncio.Queue = asyncio.Queue()
        self._event_q: asyncio.Queue = asyncio.Queue()
        self._pump_task: Optional[asyncio.Task] = None
        self._closed = False

    def _ensure_pump(self) -> None:
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        """Single receive loop: demux ws frames into audio / event queues."""
        try:
            while not self._closed:
                msg = await self._ws.receive()
                mtype = msg.get("type")
                if mtype == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data:
                    await self._audio_q.put(data)
                    continue
                text = msg.get("text")
                if text:
                    try:
                        payload = _json.loads(text)
                    except (ValueError, TypeError):
                        continue
                    await self._event_q.put(payload)
        except Exception:
            pass
        finally:
            self._audio_q.put_nowait(_CLOSE)
            self._event_q.put_nowait(_CLOSE)

    async def recv_audio(self) -> AsyncIterator[bytes]:
        self._ensure_pump()
        while True:
            item = await self._audio_q.get()
            if item is _CLOSE:
                return
            yield item

    async def recv_event(self) -> AsyncIterator[dict]:
        self._ensure_pump()
        while True:
            item = await self._event_q.get()
            if item is _CLOSE:
                return
            yield item

    async def send_audio(self, chunk: bytes) -> None:
        await self._ws.send_bytes(chunk)

    async def send_event(self, event: dict) -> None:
        await self._ws.send_json(event)

    async def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> None:
        self._closed = True
        try:
            if code is not None:
                await self._ws.close(code=code)
            else:
                await self._ws.close()
        except Exception:
            pass
        if self._pump_task is not None and not self._pump_task.done():
            self._pump_task.cancel()


__all__ = ["Transport", "InProcessTransport", "WebSocketTransport"]
