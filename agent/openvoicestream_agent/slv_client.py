"""Async client for SLV's /v2v/stream WebSocket.

Single persistent connection per App lifetime (invariant 1). All public
send methods are serialized through an internal lock so frames never
interleave. Reader task decodes JSON vs binary frames and pushes typed
V2VEvent values onto a queue exposed via `events()`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import dataclass
from typing import Any, AsyncIterator

import websockets
from websockets.asyncio.client import connect as ws_connect

# Re-use SLV's protocol constants (invariant 6: never redeclare).
from app.core.v2v import (  # type: ignore[import-not-found]
    CLIENT_ABORT,
    CLIENT_ASR_EOS,
    CLIENT_CONFIG,
    CLIENT_TEXT,
    CLIENT_TOOL_ADVERTISE,
    CLIENT_TOOL_RESULT,
    CLIENT_TTS_FLUSH,
    SERVER_ASR_ENDPOINT,
    SERVER_ASR_FINAL,
    SERVER_ASR_PARTIAL,
    SERVER_ERROR,
    SERVER_TOOL_CALL,
    SERVER_TTS_DONE,
    SERVER_TTS_SENTENCE_DONE,
    SERVER_TTS_STARTED,
)

logger = logging.getLogger(__name__)


# ── Typed events ──────────────────────────────────────────────────────


class V2VEvent:
    """Base class for events emitted by SLVClient."""


@dataclass
class ASRPartial(V2VEvent):
    text: str
    is_stable: bool = False


@dataclass
class ASREndpoint(V2VEvent):
    pass


@dataclass
class ASRFinal(V2VEvent):
    text: str
    session_complete: bool = True
    duplicate_of_streamed: bool = False
    # Per-utterance detected language as reported by the ASR backend
    # (e.g. "Chinese", "English"). None when the backend doesn't perform
    # language ID. Modes (e.g. InterpreterMode) consume this to pick a
    # translator src language at runtime.
    language: "str | None" = None


@dataclass
class TTSStarted(V2VEvent):
    sentence: str


@dataclass
class TTSSentenceDone(V2VEvent):
    sentence: str


@dataclass
class TTSDone(V2VEvent):
    # SLV signals session_complete=True when the TTS turn ended at a
    # session boundary (slot will release, WS may close), False when
    # the turn ended cleanly but the slot is still held (multi-utterance
    # continuation expected). Race #4: previously this field was dropped
    # by the dataclass and the app treated every TTSDone as session-end,
    # producing spurious reconnects.
    session_complete: bool = True


@dataclass
class TTSAudio(V2VEvent):
    pcm: bytes
    sample_rate: int


@dataclass
class SLVError(V2VEvent):
    message: str


@dataclass
class ServerToolCall(V2VEvent):
    """Server-loop mode (#37 Phase 2-product): SLV's server-side LLM loop
    selected one of the tools this client advertised and asks us to run it
    locally (the resource — e.g. the arm — lives here). The dispatch loop
    executes it against the local registry and replies with a
    CLIENT_TOOL_RESULT frame (see ``SLVClient.send_tool_result``)."""

    id: str
    name: str
    arguments: dict
    timeout_s: float = 15.0


class SLVReconnectError(Exception):
    """Raised when ``SLVClient.reconnect()`` cannot establish a working WS.

    Callers (e.g. ``App.wake()``) catch this to decide policy — most often,
    refuse the wake and stay SLEEPING so the user notices something is
    wrong rather than experiencing a silent mute.
    """


# ── Client ───────────────────────────────────────────────────────────


class SLVClient:
    """One persistent WS to /v2v/stream for the entire App lifetime."""

    def __init__(self, url: str, config: dict[str, Any]) -> None:
        self.url = url
        self.config = dict(config)
        # Make sure multi_utterance is on (invariant 1).
        self.config["multi_utterance"] = True

        self._ws: Any | None = None
        self._reader_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._queue: asyncio.Queue[V2VEvent] = asyncio.Queue()
        self._tts_sample_rate: int | None = None
        self._closed = False
        # Set when reader exits for any reason; events() uses this to
        # break out of `await queue.get()` instead of hanging forever.
        self._reader_done: asyncio.Event = asyncio.Event()
        # Track wall-clock of last WS activity (recv from server OR send
        # audio/json from us). is_healthy() only checks TCP layer; the
        # server-side ASR session may be GC'd after long idle while the
        # TCP socket stays open. wake() consults seconds_since_activity()
        # to decide whether to force a reconnect (refresh ASR session)
        # even when is_healthy() reports True. See app_base.wake().
        self._last_activity_ts: float = 0.0
        # Race #3: set True while reconnect() is tearing down + reopening.
        # mic_pump consults is_reconnecting() to skip forwarding audio
        # during the outage (otherwise chunks queue up on _send_lock,
        # 0.5s send_audio timeout starves the mic pump, dropped-chunk
        # logs flood, and post-reconnect first utterance still carries
        # pre-reconnect preroll).
        self._reconnecting: bool = False

    def _touch_activity(self) -> None:
        try:
            self._last_activity_ts = asyncio.get_event_loop().time()
        except RuntimeError:
            # No running loop (e.g. unit test using time module). Fall back
            # gracefully — treat as if we just had activity.
            import time as _time
            self._last_activity_ts = _time.monotonic()

    def seconds_since_activity(self) -> float:
        """Wall-clock seconds since last observed WS activity.

        Returns inf if no activity has ever been recorded (fresh client
        with no traffic yet — caller should treat as "stale" / reconnect
        candidate after the first turn).
        """
        if self._last_activity_ts == 0:
            return float("inf")
        try:
            now = asyncio.get_event_loop().time()
        except RuntimeError:
            import time as _time
            now = _time.monotonic()
        return now - self._last_activity_ts

    # ── lifecycle ───────────────────────────────────────────────────

    # Time we wait after sending CLIENT_CONFIG before declaring a fresh
    # WS healthy. If the server closes within this window the limiter
    # almost certainly rejected us (SLV's per-client WS slot is still
    # busy with the previous session's teardown); back off and retry.
    _RECONNECT_GRACE_S = 0.05
    # Backoff schedule for the limiter race. Empirically the slot
    # releases inside ~40ms, but Jetson under thermal throttle can be
    # slower; cover up to ~1.75s of contention before giving up.
    _RECONNECT_BACKOFFS = (0.25, 0.5, 1.0)

    async def connect(self) -> None:
        if self._ws is not None:
            return  # idempotent
        await self._open_with_retry()

    def is_healthy(self) -> bool:
        """Best-effort liveness check for the SLV WebSocket.

        Returns False when:
        - the client is closed
        - the WS handle is missing (last send observed ``ConnectionClosed``
          and nulled it)
        - the reader task is missing or has already exited (server closed
          the stream, or ``_open_with_retry`` aborted before launching it)

        Cheap to call — no I/O. Used by ``App.wake()`` to decide whether
        to skip a wake when the previous turn's reconnect fail-silently
        left us with a dead stream.
        """
        if self._closed:
            return False
        if self._ws is None:
            return False
        if self._reader_task is None or self._reader_task.done():
            return False
        return True

    async def reconnect(self) -> None:
        """Tear down current WS and open a fresh one, replaying config.

        Self-healing against SLV's session-limiter race (server `app/main.py`
        ``try_acquire_ws`` admission): the previous session's WS slot is
        held until a teardown chain (dispatcher cancel → ASR cancel →
        ws.close → manager unregister → token release) completes. A
        reconnect that arrives inside that 3–40 ms window is closed with
        WS code 4429. We detect the immediate close by waiting on the
        reader-done signal for ``_RECONNECT_GRACE_S`` after sending the
        config frame; if reader fires inside the grace, we back off and
        retry.
        """
        if self._closed:
            return
        self._reconnecting = True
        try:
            async with self._send_lock:
                old_reader = self._reader_task
                old_ws = self._ws
                self._ws = None
                self._reader_task = None
                if old_reader is not None and not old_reader.done():
                    old_reader.cancel()
                    try:
                        await old_reader
                    except (asyncio.CancelledError, Exception):
                        pass
                if old_ws is not None:
                    try:
                        await old_ws.close()
                    except Exception:
                        pass
                # Grace: let server SessionLimiter (limit=1, Qwen3-ASR
                # worker is single-concurrent) observe the close + release
                # the slot before we open a fresh WS. With proactive
                # reconnects on tts_done / wake reverted (SLV server
                # v1.15+ ASR turn timeout handles stuck workers),
                # reconnect now only fires on genuine WS death — close-
                # before-open races are no longer densely triggered, so
                # 50ms of defensive grace is sufficient (was 150ms when
                # proactive reconnect was active).
                await asyncio.sleep(0.05)
                self._reader_done.clear()
                self._tts_sample_rate = None
                await self._open_with_retry()
        finally:
            self._reconnecting = False

    def is_reconnecting(self) -> bool:
        """True while reconnect() is mid-flight (race #3 gate)."""
        return self._reconnecting

    async def _open_with_retry(self) -> None:
        """Open a WS and verify it survived the limiter grace window.

        Must be called with ``self._send_lock`` already held by the
        caller (or in a single-threaded init path like ``connect``).
        """
        attempts = list(self._RECONNECT_BACKOFFS) + [None]
        for attempt_idx, backoff in enumerate(attempts):
            self._reader_done.clear()
            self._tts_sample_rate = None
            self._ws = await ws_connect(self.url, max_size=None)
            try:
                await self._ws.send(json.dumps({"type": CLIENT_CONFIG, **self.config}))
            except websockets.ConnectionClosed as e:
                # Server slammed the door before we could send config
                # (4429 limiter race, etc.). Treat as failed attempt and
                # fall through to backoff. Without this catch the
                # exception escapes wake() as "unexpected error".
                logger.info(
                    "SLV reconnect attempt %d: send(config) closed by server (%s)",
                    attempt_idx + 1, e,
                )
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
                if backoff is None:
                    raise SLVReconnectError(
                        f"SLV reconnect: server closed CLIENT_CONFIG send on all "
                        f"{len(attempts)} attempts ({e})"
                    )
                await asyncio.sleep(backoff)
                continue
            self._reader_task = asyncio.create_task(
                self._reader_loop(), name="slv-reader"
            )
            try:
                # If the server rejects this connection (limiter race,
                # bad config, etc.) the reader sees ConnectionClosed
                # almost immediately and sets _reader_done. Survive the
                # grace window → connection is healthy, return.
                await asyncio.wait_for(
                    self._reader_done.wait(), timeout=self._RECONNECT_GRACE_S
                )
            except asyncio.TimeoutError:
                # Healthy: reader is still running after grace window.
                # Mark fresh WS as just-active so the next wake doesn't
                # see "infinite idle" and immediately re-reconnect on top
                # of our brand new session (and trip the limiter).
                self._touch_activity()
                return
            # Reader fired → connection died inside grace window.
            # Tear down what we just built and back off.
            dead_reader = self._reader_task
            dead_ws = self._ws
            self._ws = None
            self._reader_task = None
            if dead_reader is not None and not dead_reader.done():
                dead_reader.cancel()
                try:
                    await dead_reader
                except (asyncio.CancelledError, Exception):
                    pass
            if dead_ws is not None:
                try:
                    await dead_ws.close()
                except Exception:
                    pass
            # Drain any SLVError the reader may have queued so subsequent
            # events() consumers don't see this rejection.
            try:
                while not self._queue.empty():
                    _ = self._queue.get_nowait()
            except Exception:
                pass
            if backoff is None:
                raise SLVReconnectError(
                    f"SLV reconnect: server closed within {self._RECONNECT_GRACE_S}s "
                    f"on all {len(attempts)} attempts (limiter race?)"
                )
            logger.warning(
                "SLV reconnect attempt %d rejected within %.0fms; retrying in %.2fs",
                attempt_idx + 1, self._RECONNECT_GRACE_S * 1000, backoff,
            )
            await asyncio.sleep(backoff)

    async def close(self) -> None:
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._ws = None

    # ── send helpers ────────────────────────────────────────────────

    async def _send_json(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            if self._ws is None:
                if self._closed:
                    return
                await self.connect()
            try:
                await self._ws.send(json.dumps(payload))
            except websockets.ConnectionClosed:
                logger.info("send_json: WS closed mid-send, dropping %s", payload.get("type"))
                # Null the dead handle so the next caller triggers
                # connect() instead of replaying onto the closed WS.
                self._ws = None

    async def send_audio(self, pcm: bytes) -> None:
        async with self._send_lock:
            if self._ws is None:
                if self._closed:
                    return
                await self.connect()
            try:
                await self._ws.send(pcm)
                # Do NOT touch activity on outgoing audio — the mute bug
                # is exactly "we keep sending into a dead session". Only
                # server-originated frames (handled in _handle_json /
                # _handle_binary) signal that the SLV session is alive.
            except websockets.ConnectionClosed:
                # Reader will notice and signal _reader_done; dispatch can
                # decide to reconnect. Audio chunks during reconnect are
                # naturally dropped — first ASR utterance after reconnect
                # picks up from the new chunks.
                self._ws = None

    async def send_text(self, text: str) -> None:
        if text:
            logger.info("SLV send text chunk len=%d", len(text))
        await self._send_json({"type": CLIENT_TEXT, "text": text})

    async def flush_tts(self) -> None:
        logger.info("SLV send tts_flush")
        await self._send_json({"type": CLIENT_TTS_FLUSH})

    async def abort(self) -> None:
        await self._send_json({"type": CLIENT_ABORT})

    async def asr_eos(self) -> None:
        await self._send_json({"type": CLIENT_ASR_EOS})

    # ── server-loop mode (#37 Phase 2-product) ───────────────────────

    async def advertise_tools(
        self,
        tools: list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        llm_params: dict[str, Any] | None = None,
    ) -> None:
        """Upload local tool schemas to SLV so the server-side LLM loop can
        select them (and proxy execution back via SERVER_TOOL_CALL).

        ``tools`` is the OpenAI-style ``tools[]`` list from
        ``ToolRegistry.list_openai_tools(...)``. ``system_prompt`` and
        ``llm_params`` are optional and only included when provided so the
        server can run the loop with the same prompt / sampling the client
        would have used locally. Only sent in server-loop mode — a legacy
        client never calls this.
        """
        payload: dict[str, Any] = {
            "type": CLIENT_TOOL_ADVERTISE,
            "tools": list(tools or []),
        }
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        if llm_params:
            payload["llm_params"] = dict(llm_params)
        logger.info(
            "SLV advertise %d tool(s) (sp_len=%d, llm_params=%s)",
            len(payload["tools"]),
            len(system_prompt or ""),
            bool(llm_params),
        )
        await self._send_json(payload)

    async def send_tool_result(
        self,
        call_id: str,
        name: str,
        *,
        ok: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Return the outcome of a SERVER_TOOL_CALL to SLV.

        On success pass ``ok=True`` + ``result`` (the handler's dict). On
        failure pass ``ok=False`` + ``error`` (a human-readable string).
        Mirrors the wire shape in spec §4.
        """
        payload: dict[str, Any] = {
            "type": CLIENT_TOOL_RESULT,
            "id": call_id,
            "name": name,
            "ok": bool(ok),
        }
        if ok:
            payload["result"] = result if isinstance(result, dict) else {}
        else:
            payload["error"] = error or "tool execution failed"
        logger.info("SLV tool_result id=%s name=%s ok=%s", call_id, name, ok)
        await self._send_json(payload)

    # ── reader ──────────────────────────────────────────────────────

    async def events(self) -> AsyncIterator[V2VEvent]:
        while True:
            # Drain anything already queued first so SLVError emitted on
            # reader exit is still surfaced to the consumer.
            if not self._queue.empty():
                yield self._queue.get_nowait()
                continue
            if self._reader_done.is_set() or self._closed:
                if (
                    not self._closed
                    and self._reader_task is not None
                    and not self._reader_task.done()
                ):
                    # A manual reconnect can cancel an old reader and start a
                    # new one while this iterator is still alive. The old
                    # reader's finally may set the shared event after the new
                    # reader is already running; don't make dispatch exit and
                    # reconnect a second time in that case.
                    self._reader_done.clear()
                    continue
                return
            get_task = asyncio.create_task(self._queue.get())
            done_task = asyncio.create_task(self._reader_done.wait())
            try:
                done, pending = await asyncio.wait(
                    {get_task, done_task}, return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                get_task.cancel()
                done_task.cancel()
                raise
            for t in pending:
                t.cancel()
            if get_task in done:
                yield get_task.result()
            else:
                if (
                    not self._closed
                    and self._reader_task is not None
                    and not self._reader_task.done()
                ):
                    self._reader_done.clear()
                    continue
                # Reader finished; flush any final items it pushed.
                while not self._queue.empty():
                    yield self._queue.get_nowait()
                return

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if isinstance(msg, (bytes, bytearray)):
                    await self._handle_binary(bytes(msg))
                else:
                    await self._handle_json(msg)
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as e:
            logger.info("SLV WS closed: %s", e)
            await self._queue.put(SLVError(f"connection closed: {e}"))
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("SLV reader crashed")
            await self._queue.put(SLVError(str(e)))
        finally:
            # Wake any consumer blocked on events().
            self._reader_done.set()

    async def _handle_binary(self, data: bytes) -> None:
        self._touch_activity()
        if self._tts_sample_rate is None:
            if len(data) < 4:
                await self._queue.put(SLVError("first binary frame < 4 bytes"))
                return
            (sr,) = struct.unpack("<I", data[:4])
            self._tts_sample_rate = sr
            pcm = data[4:]
            logger.info("SLV tts sample_rate=%d first_pcm=%d", sr, len(pcm))
            if pcm:
                await self._queue.put(TTSAudio(pcm=pcm, sample_rate=sr))
            return
        logger.info("SLV tts audio bytes=%d", len(data))
        await self._queue.put(TTSAudio(pcm=data, sample_rate=self._tts_sample_rate))

    async def _handle_json(self, raw: str) -> None:
        self._touch_activity()
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError as e:
            await self._queue.put(SLVError(f"bad json: {e}"))
            return
        t = evt.get("type")
        # TEMP DEBUG: log every event type so we can see what SLV emits
        # between turns. Cheap because events are sparse compared to PCM.
        logger.info("SLV evt: %s", {k: v for k, v in evt.items() if k != "text" or len(str(v)) < 100})
        if t == SERVER_ASR_PARTIAL:
            await self._queue.put(
                ASRPartial(text=evt.get("text", ""), is_stable=bool(evt.get("is_stable", False)))
            )
        elif t == SERVER_ASR_ENDPOINT:
            await self._queue.put(ASREndpoint())
        elif t == SERVER_ASR_FINAL:
            await self._queue.put(
                ASRFinal(
                    text=evt.get("text", ""),
                    session_complete=bool(evt.get("session_complete", True)),
                    duplicate_of_streamed=bool(evt.get("duplicate_of_streamed", False)),
                    language=evt.get("language"),
                )
            )
        elif t == SERVER_TTS_STARTED:
            logger.info("SLV tts_started sentence=%r", evt.get("sentence", "")[:80])
            await self._queue.put(TTSStarted(sentence=evt.get("sentence", "")))
        elif t == SERVER_TTS_SENTENCE_DONE:
            await self._queue.put(TTSSentenceDone(sentence=evt.get("sentence", "")))
        elif t == SERVER_TTS_DONE:
            session_complete = bool(evt.get("session_complete", True))
            logger.info("SLV tts_done session_complete=%s", session_complete)
            await self._queue.put(TTSDone(session_complete=session_complete))
        elif t == SERVER_TOOL_CALL:
            # Server-loop mode: SLV asks us to run a local tool. Additive —
            # only emitted when the server-side loop is enabled, so a legacy
            # always-client-loop session never reaches this branch.
            args = evt.get("arguments")
            await self._queue.put(
                ServerToolCall(
                    id=str(evt.get("id", "")),
                    name=str(evt.get("name", "")),
                    arguments=args if isinstance(args, dict) else {},
                    timeout_s=float(evt.get("timeout_s", 15.0)),
                )
            )
        elif t == SERVER_ERROR:
            await self._queue.put(SLVError(evt.get("error", "unknown")))
        else:
            logger.debug("Unknown SLV message type: %r", t)
