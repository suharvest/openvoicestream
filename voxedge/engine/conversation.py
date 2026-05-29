"""ConversationEngine + Session — the V2V orchestration core.

Faithful port of the orchestration buried in app/main.py's /v2v/stream
handler (dispatcher / asr_out_task / tts_out_task, app/main.py:2814-3432),
lifted out of FastAPI into an importable engine that drives any
:class:`~voxedge.transport.base.Transport`.

PORTING MAP (source ranges are app/main.py snapshots, 2026-05-29):
  * dispatcher()    ← app/main.py:2814-2990   (audio/control frame state machine)
  * asr_out_task()  ← app/main.py:2992-3205   (partial poll, gen gate, finalize)
  * tts_out_task()  ← app/main.py:3207-3432   (sentence queue, barge-in, done)
  * orchestrate     ← app/main.py:3434-3457   (spawn work tasks, cancel dispatcher)

DELIBERATE SIMPLIFICATIONS (kept faithful in shape, infra dropped):
  * No BackendCoordinator ``coord.acquire(...)`` / SessionLimiter / slot-pool
    (app/core, out of Phase-1a scope). The lock-around-backend calls are
    replaced by direct calls; concurrency ceilings are a later phase.
    WS-level admission (SessionLimiter) stays in the transport/app layer.
  * ASRSessionManager IS used now (M2, 2026-05-30): copied verbatim into
    ``voxedge/engine/asr_session_manager.py`` (dedup after Phase 1b). This
    gives the engine atomic per-generation partial snapshots
    (``get_partial_for_generation``), bounded cancel + worker-restart ladder
    (``cancel`` / ERROR_REBUILD), and ``finalize_with_status`` accepted/stale
    suppression — replacing the earlier hand-rolled ``_AsrTurn`` helper.
  * Wall-clock turn (M1) + TTS chunk/sentence (M3) watchdogs are now wired,
    but their thresholds are CONSTRUCTOR-INJECTED (``timeouts`` dict), never
    read from env (spec §2). Defaults mirror prod's env defaults.
  * Blocking backend calls run through the ASRSessionManager's executor hop
    (ASR) or ``asyncio.to_thread`` / ``run_in_executor`` (TTS) so the event
    loop is never blocked — same intent as prod (app/main.py:3050, 3310).

LLM closed loop (spec §4): when an ``llm`` backend is provided the asr_final
text is fed to the LLM and its text deltas drive the TTS sentence buffer
(equivalent to prod's CLIENT_TEXT path, app/main.py:2953-2960). With no LLM
the engine is a pure ASR↔TTS pass-through plus a direct CLIENT_TEXT→TTS path.
``tool_registry`` is accepted and a hook is reserved for the tool-calling
continuation, but the full ToolRunner loop is a later phase.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import threading
from typing import Any, Optional

import numpy as np

from voxedge.backends.base import ASRBackend, LLMBackend, TTSBackend, VADBackend
from voxedge.engine.asr_session_manager import (
    ASRSessionManager,
    ASRSessionUnavailable,
)
from voxedge.engine.coordinator import BackendCoordinator
from voxedge.transport.base import Transport


@contextlib.asynccontextmanager
async def _passthrough():
    """No-op async context manager — used when no coordinator is wired so the
    engine keeps its current direct-call behavior (backward compatible)."""
    yield

logger = logging.getLogger(__name__)


def _is_pool_saturated(exc: BaseException) -> tuple[bool, Optional[int]]:
    """Duck-type a backend ``PoolSaturatedError`` (M4).

    Mirrors app/main.py:592-610: recognize by ``status == 4429`` (+ class
    name as belt-and-braces) and surface a clean 4429 reject. These are
    deliberately NOT worker-protocol errors — a saturation is "backend
    busy", never a worker fault, so it must NOT trigger a worker restart.

    Returns ``(is_saturated, max_slots_or_None)``.
    """
    if getattr(exc, "status", None) == 4429 or type(exc).__name__ == "PoolSaturatedError":
        ms = getattr(exc, "max_slots", None)
        return True, ms if isinstance(ms, int) else None
    return False, None


# ── protocol constants (mirror app/core/v2v.py:33-53) ──────────────────
CLIENT_TEXT = "text"
CLIENT_ASR_EOS = "asr_eos"
CLIENT_TTS_FLUSH = "tts_flush"
CLIENT_ABORT = "abort"

SERVER_ASR_PARTIAL = "asr_partial"
SERVER_ASR_ENDPOINT = "asr_endpoint"
SERVER_ASR_FINAL = "asr_final"
SERVER_TTS_STARTED = "tts_started"
SERVER_TTS_SENTENCE_DONE = "tts_sentence_done"
SERVER_TTS_DONE = "tts_done"
SERVER_VAD_EVENT = "vad_event"
SERVER_ERROR = "error"

VAD_EVENT_SPEECH_START = "speech_start"
VAD_EVENT_SPEECH_END = "speech_end"


# ── lightweight sentence buffer (regex fallback, mirrors v2v.py path) ──
import re as _re

_SENTENCE_END_RE = _re.compile(r"[。！？；\n]+|[!?.](?=\s|$)")


class _SentenceBuffer:
    """Minimal pure-Python sentence buffer.

    Equivalent in role to app/core/v2v.py SentenceBuffer's regex-fallback
    path (v2v.py:202-224). voxedge core stays dep-free; the pysbd upgrade is
    the optional ``voxedge[text]`` extra.
    """

    def __init__(self, min_chars: int = 2, max_buffer: int = 200):
        self._buf = ""
        self._min = min_chars
        self._max = max_buffer

    def add(self, chunk: str):
        if not chunk:
            return
        self._buf += chunk
        while True:
            s = self._extract()
            if s is None:
                return
            yield s

    def flush(self):
        leftover = self._buf.strip()
        self._buf = ""
        if leftover:
            yield leftover

    def _extract(self) -> Optional[str]:
        pos = 0
        while True:
            m = _SENTENCE_END_RE.search(self._buf, pos)
            if m is None:
                if len(self._buf) >= self._max:
                    out = self._buf.strip()
                    self._buf = ""
                    return out or None
                return None
            end = m.end()
            prefix = self._buf[:end]
            if len(prefix.strip()) >= self._min:
                self._buf = self._buf[end:]
                return prefix.strip()
            pos = end


class Session:
    """One conversation over one Transport. Owns the per-conn state machine."""

    def __init__(self, engine: "ConversationEngine", transport: Transport):
        self.engine = engine
        self.transport = transport

        # Optional slot coordinator (concurrency abstraction migrated from
        # app/core, spec §3.1). When None the engine runs direct passthrough
        # (current behavior, backward compatible). When present, ASR/TTS
        # backend calls are wrapped in ``coord.acquire(...)`` so serialized /
        # exclusive modes truly mutually-exclude and concurrent mode overlaps.
        self._coord: Optional[BackendCoordinator] = engine.coordinator

        be = engine.backends
        self._asr_be: Optional[ASRBackend] = be.get("asr")
        self._tts_be: Optional[TTSBackend] = be.get("tts")
        self._vad_be: Optional[VADBackend] = be.get("vad")
        self._llm_be: Optional[LLMBackend] = be.get("llm")

        self.asr_enabled = self._asr_be is not None
        self._vad = (
            self._vad_be.create_session(silence_ms=engine.silence_ms)
            if self._vad_be is not None
            else None
        )
        # M2: ASRSessionManager replaces the hand-rolled _AsrTurn — gives
        # atomic per-generation partial snapshots + bounded cancel/restart
        # ladder + finalize accepted/stale suppression (mirrors prod
        # app/main.py asr_manager).
        self._asr_mgr: Optional[ASRSessionManager] = (
            ASRSessionManager(
                self._asr_be,
                language=engine.asr_language,
                sample_rate=self._asr_be.sample_rate,
            )
            if self._asr_be
            else None
        )
        self._tts_buffer = _SentenceBuffer() if self._tts_be else None

        self._tts_q: asyncio.Queue = asyncio.Queue()
        self._loop = asyncio.get_event_loop()

        # M1/M3: wall-clock watchdog thresholds — constructor-injected (spec
        # §2: no env reads in the engine). Defaults mirror prod env defaults
        # (OVS_ASR_TURN_TIMEOUT_S=45, OVS_TTS_CHUNK_TIMEOUT_S=10,
        # OVS_TTS_SENTENCE_TIMEOUT_S=15).
        self._asr_turn_timeout_s = engine.asr_turn_timeout_s
        self._tts_chunk_timeout_s = engine.tts_chunk_timeout_s
        self._tts_sentence_timeout_s = engine.tts_sentence_timeout_s

        # per-conn state (mirrors app/main.py:2751-2791 state dict)
        self.state = {
            "client_closed": False,
            "asr_active": False,
            "asr_active_gen": 0,
            "asr_session_closed": False,
            "endpoint_pending": None,
            "endpoint_pending_gen": None,
            "current_tts_task": None,
            "current_tts_stop": None,
            "tts_flush": False,
            "tts_started": False,
            # M1: per-ASR-turn wall-clock deadline anchor (app/main.py:2790).
            "asr_turn_started_at": None,
        }
        self._work_tasks: list[asyncio.Task] = []

    def _acquire(self, slot: str):
        """Acquire the slot via the coordinator, or a no-op when none wired.

        In ``serialized`` / ``exclusive`` mode the coordinator's shared lock
        makes ASR and TTS backend calls mutually exclusive; in ``concurrent``
        mode (or with no coordinator) this is a passthrough and they overlap.
        """
        if self._coord is None:
            return _passthrough()
        return self._coord.acquire(slot)  # type: ignore[arg-type]

    # ── transport send helpers (app/main.py:2795-2810) ─────────────────

    async def _send_event(self, payload: dict) -> None:
        try:
            await self.transport.send_event(payload)
        except Exception:
            self.state["client_closed"] = True

    async def _send_audio(self, data: bytes) -> None:
        try:
            await self.transport.send_audio(data)
        except Exception:
            self.state["client_closed"] = True

    async def _send_error(self, msg: str) -> None:
        await self._send_event({"type": SERVER_ERROR, "error": msg})

    # ══════════════════════════════════════════════════════════════════
    # dispatcher  ← app/main.py:2814-2990
    # ══════════════════════════════════════════════════════════════════

    async def _audio_loop(self) -> None:
        """Inbound audio frames → VAD segmentation + ASR feed.

        Port of the binary-frame branch of dispatcher() (app/main.py:2831-2943).
        """
        state = self.state
        multi = self.engine.multi_utterance
        try:
            async for data in self.transport.recv_audio():
                if state["client_closed"]:
                    break
                if not self.asr_enabled or state["asr_session_closed"]:
                    continue
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                speech_ended_now = False

                if self._vad is not None:
                    event = self._vad.process(samples)
                    if event == self._vad.SPEECH_START:
                        # Notify client first, then barge-in (app/main.py:2845-2893).
                        await self._send_event({
                            "type": SERVER_VAD_EVENT,
                            "event": VAD_EVENT_SPEECH_START,
                        })
                        await self._bargein_tts()
                        if not await self._open_asr_turn():
                            continue
                    elif event == self._vad.SPEECH_END:
                        # Defer endpoint flag until AFTER accepting this chunk
                        # (BUG 3, app/main.py:2894-2900).
                        speech_ended_now = True

                # No-VAD: open lazily on first audio (app/main.py:2901-2921).
                if self._vad is None and not state["asr_active"]:
                    if not await self._open_asr_turn():
                        continue

                if state["asr_active"]:
                    await self._asr_mgr.accept_audio(samples)

                # Now safe to latch endpoint (app/main.py:2929-2942).
                if speech_ended_now:
                    state["endpoint_pending"] = "vad"
                    state["endpoint_pending_gen"] = state["asr_active_gen"]
                    if not multi:
                        state["asr_session_closed"] = True
                    await self._send_event({
                        "type": SERVER_VAD_EVENT,
                        "event": VAD_EVENT_SPEECH_END,
                    })
        except Exception:
            logger.exception("voxedge audio_loop error")
            state["client_closed"] = True

    async def _event_loop(self) -> None:
        """Inbound control events. Port of the text-frame branch of
        dispatcher() (app/main.py:2944-2983)."""
        state = self.state
        multi = self.engine.multi_utterance
        try:
            async for payload in self.transport.recv_event():
                if state["client_closed"]:
                    break
                typ = payload.get("type")
                if typ == CLIENT_TEXT and self._tts_buffer is not None:
                    for sentence in self._tts_buffer.add(payload.get("text", "")):
                        await self._tts_q.put(sentence)
                elif typ == CLIENT_TTS_FLUSH:
                    if self._tts_buffer is not None:
                        for sentence in self._tts_buffer.flush():
                            await self._tts_q.put(sentence)
                    state["tts_flush"] = True
                elif typ == CLIENT_ASR_EOS:
                    state["endpoint_pending"] = "client_eos"
                    state["endpoint_pending_gen"] = state["asr_active_gen"]
                    if not multi:
                        state["asr_session_closed"] = True
                elif typ == CLIENT_ABORT:
                    await self._bargein_tts()
                    while not self._tts_q.empty():
                        try:
                            self._tts_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    if self._asr_mgr is not None and state["asr_active"]:
                        await self._asr_mgr.cancel("abort")
                        state["asr_active"] = False
                        state["asr_turn_started_at"] = None
        except Exception:
            logger.exception("voxedge event_loop error")
            state["client_closed"] = True

    async def _bargein_tts(self) -> None:
        """Cancel in-flight TTS + signal the synth thread to stop
        (app/main.py:2856-2861, 2966-2972)."""
        t = self.state["current_tts_task"]
        if t is not None and not t.done():
            t.cancel()
        stop = self.state["current_tts_stop"]
        if stop is not None:
            stop.set()

    async def _open_asr_turn(self) -> bool:
        """Start a fresh ASR utterance via the session manager.

        Port of app/main.py:2862-2893 (VAD speech_start) / 2901-2921
        (no-VAD lazy open). On ``ASRSessionUnavailable`` (rebuild ladder
        exhausted — race #1) surface ``asr_unavailable`` and DON'T flag the
        session active, so audio isn't silently dropped with the client
        stuck. Returns True iff a turn was opened.
        """
        state = self.state
        try:
            new_gen = await self._asr_mgr.on_speech_start()
        except ASRSessionUnavailable as e:
            state["asr_active"] = False
            state["endpoint_pending"] = None
            state["endpoint_pending_gen"] = None
            # M4: the manager wraps create_stream failures in
            # ASRSessionUnavailable (raised ``from`` the original). If the
            # root cause was a slot-pool saturation, surface the typed
            # pool_saturated reject instead of a generic asr_unavailable so
            # the client knows to retry (saturation is "busy", not a fault).
            sat, max_slots = _is_pool_saturated(e.__cause__ or e)
            if sat:
                await self._emit_pool_saturated(max_slots)
                return False
            logger.warning("voxedge: on_speech_start failed: %s", e)
            await self._send_event({"type": SERVER_ERROR, "error": "asr_unavailable"})
            return False
        except Exception as e:  # noqa: BLE001
            sat, max_slots = _is_pool_saturated(e)
            if sat:
                # M4: backend slot-pool saturated → typed 4429, not a fault.
                await self._emit_pool_saturated(max_slots)
                state["asr_active"] = False
                return False
            raise
        state["endpoint_pending"] = None
        state["endpoint_pending_gen"] = None
        state["asr_active"] = True
        state["asr_active_gen"] = new_gen
        # M1: anchor the per-turn wall-clock deadline (app/main.py:2892).
        state["asr_turn_started_at"] = self._loop.time()
        return True

    async def _emit_pool_saturated(self, max_slots: Optional[int]) -> None:
        """M4: typed pool_saturated event (app/main.py:3348-3353)."""
        payload = {"type": SERVER_ERROR, "error": "pool_saturated", "status": 4429}
        if max_slots is not None:
            payload["max_slots"] = max_slots
        await self._send_event(payload)

    # ══════════════════════════════════════════════════════════════════
    # asr_out_task  ← app/main.py:2992-3205
    # ══════════════════════════════════════════════════════════════════

    async def _asr_out_task(self) -> None:
        state = self.state
        multi = self.engine.multi_utterance
        last_streamed_final = None
        last_partial: tuple[int, str] = (-1, "")
        asr_turn_timeout_s = self._asr_turn_timeout_s
        while not state["client_closed"]:
            # ── M1: wall-clock per-turn deadline (app/main.py:3012-3077) ─
            # Active turn that hasn't finalized within the deadline → force
            # cancel + worker restart (via the manager's bounded cancel
            # ladder) so a wedged backend can't pin the session forever.
            turn_started = state.get("asr_turn_started_at")
            if (
                state.get("asr_active")
                and turn_started is not None
                and (self._loop.time() - turn_started) > asr_turn_timeout_s
            ):
                elapsed = self._loop.time() - turn_started
                logger.warning(
                    "voxedge ASR turn exceeded %.1fs wall-clock (elapsed=%.1fs); "
                    "aborting turn + force-cancel ASR session",
                    asr_turn_timeout_s, elapsed,
                )
                # Step 1: cooperative cancel with a tight budget; the manager
                # escalates to restart_worker internally on its own timeout
                # (app/main.py:3030-3054).
                if self._asr_mgr is not None:
                    try:
                        await asyncio.wait_for(
                            self._asr_mgr.cancel("turn_timeout"), timeout=2.0
                        )
                    except Exception as _exc:  # noqa: BLE001
                        logger.error(
                            "voxedge ASR cancel timed out / failed (%s)", _exc
                        )
                # Step 2: clear state + emit error so the client unwinds
                # (app/main.py:3055-3068).
                state["asr_active"] = False
                state["asr_turn_started_at"] = None
                state["endpoint_pending"] = None
                state["endpoint_pending_gen"] = None
                try:
                    await self._send_error(
                        f"asr: per-turn deadline {asr_turn_timeout_s:.0f}s exceeded"
                    )
                except Exception:
                    logger.exception("voxedge send_error after asr turn timeout failed")
                if multi and not state["asr_session_closed"]:
                    await asyncio.sleep(0.05)
                    continue
                return

            # ── partial poll (app/main.py:3084-3096) ──────────────────
            if state["asr_active"]:
                try:
                    # M2: atomic (gen, partial, is_endpoint) snapshot under
                    # the manager's lock — no torn read against a stream that
                    # a barge-in is replacing.
                    partial_gen, partial, is_endpoint = (
                        await self._asr_mgr.get_partial_for_generation()
                    )
                except Exception:
                    partial, is_endpoint, partial_gen = "", False, 0
                # Gen gate (BUG 4): drop partials from a replaced utterance.
                # Also dedupe identical consecutive partials so the fast poll
                # loop doesn't flood the client (prod tracks last_streamed_final
                # similarly, app/main.py:3001/3187).
                if (
                    partial
                    and partial_gen == state["asr_active_gen"]
                    and (partial_gen, partial) != last_partial
                ):
                    last_partial = (partial_gen, partial)
                    await self._send_event({
                        "type": SERVER_ASR_PARTIAL,
                        "text": partial,
                        "is_stable": bool(is_endpoint),
                    })
            else:
                is_endpoint = False

            # ── endpoint resolution (app/main.py:3098-3119) ───────────
            endpoint_reason = state["endpoint_pending"]
            # Gen-race gate (app/main.py:3107-3114): endpoint stamped against a
            # generation that has since been preempted → drop on the floor.
            if (
                endpoint_reason
                and state["endpoint_pending_gen"] is not None
                and state["endpoint_pending_gen"] != state["asr_active_gen"]
            ):
                state["endpoint_pending"] = None
                state["endpoint_pending_gen"] = None
                endpoint_reason = None

            endpoint_fired = bool(endpoint_reason) or (is_endpoint and state["asr_active"])

            if endpoint_fired:
                state["endpoint_pending"] = None
                state["endpoint_pending_gen"] = None
                if endpoint_reason != "client_eos":
                    await self._send_event({"type": SERVER_ASR_ENDPOINT})

                if state["asr_active"]:
                    finalize_gen = state["asr_active_gen"]
                    # M2: finalize_with_status suppresses stale/cancelled
                    # results (accepted=False) so a finalize that raced a
                    # barge-in doesn't emit a spurious final
                    # (app/core/asr_session_manager.py:265-320).
                    # Slot acquire: ASR finalize is the GPU-heavy decode; in
                    # serialized/exclusive mode it must not overlap a TTS synth.
                    async with self._acquire("asr"):
                        fin_gen, fin_text, accepted, detected_language = (
                            await self._asr_mgr.finalize_with_status(
                                endpoint_reason or "vad_end"
                            )
                        )
                    if accepted:
                        final_text = fin_text
                    else:
                        final_text, detected_language = "", None
                    # Only clear active if generation still current (BUG 2).
                    if state["asr_active_gen"] == finalize_gen:
                        state["asr_active"] = False
                        state["asr_turn_started_at"] = None
                else:
                    final_text, detected_language = "", None

                # ── emit asr_final (app/main.py:3164-3197) ────────────
                if multi:
                    is_closing = state["asr_session_closed"]
                    # Close-out endpoint with no fresh utterance (e.g. a
                    # client_eos / input-end arriving after the turn already
                    # finalized via VAD): there is nothing new to transcribe,
                    # so the session-complete final reaffirms the last streamed
                    # final rather than emitting a spurious empty text. Mirrors
                    # prod's duplicate_of_streamed close-out (app/main.py:3166-
                    # 3176) — the client gets a coherent terminal result.
                    if is_closing and not (final_text or "") and last_streamed_final:
                        emit_text = last_streamed_final
                    else:
                        emit_text = final_text or ""
                    payload = {
                        "type": SERVER_ASR_FINAL,
                        "text": emit_text,
                        "session_complete": is_closing,
                    }
                    if is_closing:
                        payload["duplicate_of_streamed"] = (
                            emit_text == (last_streamed_final or "")
                        )
                    if detected_language:
                        payload["language"] = detected_language
                    await self._send_event(payload)
                    # Closed-loop hook: feed final text to LLM→TTS (spec §4).
                    # Only drive the LLM on genuinely new ASR text — a close-out
                    # duplicate must not re-trigger another LLM→TTS turn.
                    if final_text and final_text.strip():
                        await self._on_asr_final(final_text)
                    if is_closing:
                        return
                    last_streamed_final = final_text or ""
                else:
                    payload = {"type": SERVER_ASR_FINAL, "text": final_text or ""}
                    if detected_language:
                        payload["language"] = detected_language
                    await self._send_event(payload)
                    await self._on_asr_final(final_text or "")
                    return

            # Exit when closed + nothing left (app/main.py:3202-3203).
            if state["asr_session_closed"] and not state["asr_active"]:
                return
            await asyncio.sleep(0.02)

    async def _on_asr_final(self, text: str) -> None:
        """asr_final → LLM (optional) → TTS sentence buffer (spec §4).

        With no LLM this is a no-op (pure ASR/TTS pass-through; the client
        drives TTS directly via CLIENT_TEXT). With an LLM backend, stream its
        text deltas into the TTS sentence buffer — equivalent to prod's
        CLIENT_TEXT path (app/main.py:2953-2960).

        TODO(phase-2): when ``tool_registry`` is set, run the ToolRunner
        continuation loop (spec §4 step 3) instead of the plain text stream.
        """
        if self._llm_be is None or not text.strip():
            return
        if self._tts_buffer is None:
            return
        messages = [{"role": "user", "content": text}]
        try:
            async for ev in self._llm_be.stream_events(messages):
                if ev.kind == "text" and ev.text:
                    for sentence in self._tts_buffer.add(ev.text):
                        await self._tts_q.put(sentence)
                # TODO(phase-2): ev.kind == "tool_call_delta" → ToolRunner.
            for sentence in self._tts_buffer.flush():
                await self._tts_q.put(sentence)
            self.state["tts_flush"] = True
        except Exception:
            logger.exception("voxedge LLM stream failed")

    # ══════════════════════════════════════════════════════════════════
    # tts_out_task  ← app/main.py:3207-3432
    # ══════════════════════════════════════════════════════════════════

    async def _tts_out_task(self) -> None:
        state = self.state
        multi = self.engine.multi_utterance
        sr_header_sent = False
        while not state["client_closed"]:
            # Exit / per-turn done when flush + drained (app/main.py:3217-3233).
            if state["tts_flush"] and self._tts_q.empty():
                if multi and not state["asr_session_closed"]:
                    state["tts_flush"] = False
                    if not state["client_closed"]:
                        await self._send_event({
                            "type": SERVER_TTS_DONE,
                            "session_complete": False,
                        })
                    continue
                break
            try:
                sentence = await asyncio.wait_for(self._tts_q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            audio_queue: asyncio.Queue = asyncio.Queue()
            stop_event = threading.Event()
            state["current_tts_stop"] = stop_event

            def _run_synth(s: str, ev: threading.Event, aq: asyncio.Queue):
                # Mirrors app/main.py:3246-3281 _run_synth (thread body).
                try:
                    for chunk in self._tts_be.generate_streaming(
                        s, language=self.engine.tts_language, cancel_token=ev
                    ):
                        if ev.is_set():
                            break
                        self._loop.call_soon_threadsafe(aq.put_nowait, chunk)
                except Exception as e:  # noqa: BLE001
                    # M4: a slot-pool saturation is "backend busy", NOT a
                    # synth fault — surface a typed reject marker, not a
                    # generic tts error (app/main.py:3342-3356).
                    sat, max_slots = _is_pool_saturated(e)
                    if sat:
                        self._loop.call_soon_threadsafe(
                            aq.put_nowait, ("__saturated__", max_slots)
                        )
                    else:
                        logger.exception("voxedge tts synth failed for %r", s[:80])
                        self._loop.call_soon_threadsafe(
                            aq.put_nowait, ("__error__", str(e))
                        )
                finally:
                    self._loop.call_soon_threadsafe(aq.put_nowait, None)

            chunk_timeout_s = self._tts_chunk_timeout_s

            async def drain(s: str, ev: threading.Event, aq: asyncio.Queue):
                nonlocal sr_header_sent
                # Mirrors app/main.py:3283-3361 drain().
                if not sr_header_sent:
                    sr = self._tts_be.sample_rate
                    await self._send_audio(struct.pack("<I", sr))
                    sr_header_sent = True
                await self._send_event({"type": SERVER_TTS_STARTED, "sentence": s})
                # Slot acquire: a TTS synth is the GPU-heavy op; in
                # serialized/exclusive mode it must not overlap an ASR finalize.
                # Held across the whole synth so the synth thread runs alone.
                async with self._acquire("tts"):
                    self._loop.run_in_executor(None, _run_synth, s, ev, aq)
                    state["tts_started"] = True
                    # M3: per-chunk watchdog — a wedged backend that produces no
                    # chunk within the budget aborts the sentence + emits an
                    # error so the client never waits forever (app/main.py:3322-3339).
                    while True:
                        try:
                            item = await asyncio.wait_for(aq.get(), timeout=chunk_timeout_s)
                        except asyncio.TimeoutError:
                            logger.warning(
                                "voxedge tts watchdog: no chunk within %.1fs for "
                                "sentence=%r — aborting synth", chunk_timeout_s, s[:80],
                            )
                            ev.set()
                            await self._send_error(
                                f"tts: synth produced no chunks within "
                                f"{chunk_timeout_s:.0f}s"
                            )
                            break
                        if item is None:
                            break
                        if isinstance(item, tuple) and item[0] == "__saturated__":
                            # M4: typed pool_saturated; keep the session alive.
                            await self._emit_pool_saturated(item[1])
                            break
                        if isinstance(item, tuple) and item[0] == "__error__":
                            await self._send_error(f"tts: {item[1]}")
                            break
                        await self._send_audio(item)
                await self._send_event({"type": SERVER_TTS_SENTENCE_DONE, "sentence": s})

            task = asyncio.create_task(drain(sentence, stop_event, audio_queue))
            state["current_tts_task"] = task
            # M3: outer per-sentence wall-clock deadline. Covers wedges BEFORE
            # the first chunk watchdog can fire (e.g. a backend that hangs in
            # generate_streaming setup before yielding) (app/main.py:3382-3409).
            sentence_timeout_s = self._tts_sentence_timeout_s
            try:
                await asyncio.wait_for(task, timeout=sentence_timeout_s)
            except asyncio.TimeoutError:
                logger.warning(
                    "voxedge tts: per-sentence deadline %.1fs exceeded for "
                    "sentence=%r — cancelling drain", sentence_timeout_s, sentence[:80],
                )
                stop_event.set()
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                if not state["client_closed"]:
                    try:
                        await self._send_error(
                            f"tts: per-sentence deadline "
                            f"{sentence_timeout_s:.0f}s exceeded"
                        )
                    except Exception:
                        logger.exception("voxedge send_error after tts deadline failed")
            except asyncio.CancelledError:
                # Barge-in: stop synth + drain residual chunks (app/main.py:3410-3420).
                stop_event.set()
                try:
                    while True:
                        item = audio_queue.get_nowait()
                        if item is None:
                            break
                except asyncio.QueueEmpty:
                    pass
            finally:
                state["current_tts_task"] = None
                state["current_tts_stop"] = None

        # Session-final tts_done (app/main.py:3424-3432).
        if not state["client_closed"]:
            payload = {"type": SERVER_TTS_DONE}
            if multi:
                payload["session_complete"] = True
            await self._send_event(payload)

    # ══════════════════════════════════════════════════════════════════
    # orchestrate  ← app/main.py:3434-3457
    # ══════════════════════════════════════════════════════════════════

    async def run(self) -> None:
        """Drive one full conversation, then close the transport.

        Like app/main.py:3439-3457: spawn the receive loops + work tasks,
        wait for the work tasks (which terminate on session close), then
        cancel the still-looping receive tasks.

        Graceful end-of-input: the production dispatcher treats a
        ``websocket.disconnect`` (no more client frames) as the session-end
        signal (app/main.py:2819-2830). The in-process equivalent is the
        inbound audio + event streams both being exhausted (the client called
        ``end_input()``). When that happens we mark the ASR session closed so
        the work tasks emit their close-out finals — asr_final /
        tts_done with ``session_complete=True`` in multi-utterance — and
        terminate cleanly, instead of polling forever until force-cancelled.
        """
        recv_tasks = [
            asyncio.create_task(self._audio_loop()),
            asyncio.create_task(self._event_loop()),
        ]
        work_tasks: list[asyncio.Task] = []
        if self.asr_enabled:
            work_tasks.append(asyncio.create_task(self._asr_out_task()))
        if self._tts_be is not None:
            work_tasks.append(asyncio.create_task(self._tts_out_task()))
        self._work_tasks = work_tasks

        async def _watch_input_end() -> None:
            # Once the client stops feeding (both recv loops drained), the
            # session is over. Flag the close-out so any in-flight ASR turn
            # finalizes and the work tasks exit (not just spin).
            await asyncio.gather(*recv_tasks, return_exceptions=True)
            self.state["asr_session_closed"] = True
            if not self.asr_enabled:
                # TTS-only: no asr task to drain — also flush so tts_out_task
                # emits its final tts_done and exits.
                self.state["tts_flush"] = True

        end_watcher = asyncio.create_task(_watch_input_end())

        try:
            if work_tasks:
                await asyncio.gather(*work_tasks, return_exceptions=False)
            else:
                # No work tasks (degenerate config) — still wait for input end.
                await end_watcher
        except asyncio.CancelledError:
            pass
        finally:
            # M5: ordered teardown (app/main.py:3503-3537). Tear down the
            # backends BEFORE closing the transport so the worker doesn't
            # leak the session and the synth thread is released for the next
            # connection.
            self.state["client_closed"] = True
            if not end_watcher.done():
                end_watcher.cancel()
            # (a) cancel the recv loops + work tasks.
            for t in recv_tasks:
                if not t.done():
                    t.cancel()
            for t in work_tasks:
                if not t.done():
                    t.cancel()
            # (b) signal the synth thread to bail so the TTS executor frees up
            # (app/main.py:3513-3517).
            stop = self.state.get("current_tts_stop")
            if stop is not None:
                try:
                    stop.set()
                except Exception:
                    pass
            # (c) await the cancelled work tasks before releasing the backend
            # slot (app/main.py:3518-3528, race #6).
            await asyncio.gather(
                end_watcher, *recv_tasks, *work_tasks, return_exceptions=True
            )
            # (d) cancel any in-flight ASR utterance so the worker doesn't
            # leak the session (app/main.py:3529-3535).
            if self._asr_mgr is not None:
                try:
                    await self._asr_mgr.cancel("ws_close")
                except Exception:
                    logger.exception("voxedge: asr cancel on close failed")
            # (e) finally close the transport (close-code 1003/1011 stays in
            # the transport/app layer — engine doesn't decide it).
            try:
                await self.transport.close()
            except Exception:
                pass


class ConversationEngine:
    """Importable V2V engine. Construct with resolved backends + config; no
    env reads (spec §2: ``__init__`` takes parsed config, not env).

    Args:
        backends: dict with keys ``asr`` / ``tts`` / ``vad`` / ``llm``; any
            subset. ASR-only, TTS-only, or full loop all valid.
        tool_registry: reserved for the tool-calling continuation (spec §4);
            wired in a later phase.
        multi_utterance: keep the session alive across turns (True) vs
            single-shot (False).
        timeouts: optional dict of wall-clock watchdog knobs (seconds),
            constructor-injected (spec §2: no env reads). Recognized keys:
            ``asr_turn`` (M1, default 45), ``tts_chunk`` (M3, default 10),
            ``tts_sentence`` (M3, default 15). Defaults mirror prod's env
            defaults (OVS_ASR_TURN_TIMEOUT_S / OVS_TTS_CHUNK_TIMEOUT_S /
            OVS_TTS_SENTENCE_TIMEOUT_S).
        silence_ms: VAD silence threshold passed to ``create_session``.
        asr_language: language hint forwarded to ASR stream creation.
        tts_language: default language hint forwarded to TTS streaming.
        coordinator: optional :class:`BackendCoordinator` (concurrency
            abstraction, spec §3.1). When provided, ASR/TTS backend calls are
            wrapped in ``coord.acquire(...)`` so serialized/exclusive modes
            truly mutually-exclude. When None (default) the engine runs direct
            passthrough — current behavior, backward compatible. Pass an
            instance built via ``BackendCoordinator.from_backends(...)`` to
            resolve the mode from backend capability.
    """

    def __init__(
        self,
        backends: dict[str, Any],
        *,
        tool_registry: Optional[Any] = None,
        multi_utterance: bool = False,
        timeouts: Optional[dict] = None,
        silence_ms: int = 400,
        asr_language: str = "auto",
        tts_language: Optional[str] = None,
        coordinator: Optional[BackendCoordinator] = None,
    ):
        self.backends = backends
        self.tool_registry = tool_registry
        self.multi_utterance = multi_utterance
        self.timeouts = timeouts or {}
        self.silence_ms = silence_ms
        self.asr_language = asr_language
        self.tts_language = tts_language
        self.coordinator = coordinator

        # M1/M3: resolve watchdog thresholds from the injected dict (env-free).
        self.asr_turn_timeout_s = float(self.timeouts.get("asr_turn", 45.0))
        self.tts_chunk_timeout_s = float(self.timeouts.get("tts_chunk", 10.0))
        self.tts_sentence_timeout_s = float(self.timeouts.get("tts_sentence", 15.0))

        # Preload ready backends once (app/main.py does this at startup).
        for be in backends.values():
            preload = getattr(be, "preload", None)
            if callable(preload):
                try:
                    if not (getattr(be, "is_ready", lambda: False)()):
                        preload()
                except Exception:
                    logger.exception("voxedge backend preload failed: %r", be)

    async def run(self, transport: Transport) -> None:
        """Drive one conversation over ``transport`` to completion."""
        await Session(self, transport).run()


__all__ = ["ConversationEngine", "Session"]
