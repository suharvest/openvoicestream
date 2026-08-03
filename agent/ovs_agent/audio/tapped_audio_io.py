"""TappedAudioIO — AudioIO subclass with multi-consumer capture taps.

One responsibility beyond the upstream AudioIO: capture tap fanout (see
start_capture_tap docstring), plus the mic gate that mutes capture while
our own TTS is playing.

Multi-channel mic capture (`mic_channels=6` for reSpeaker XVF3800) used to
live here too. It moved down into :class:`AudioIO` — every app hits the
reSpeaker PaErrorCode -9998 ("Invalid number of channels"), not just the
tap-capable ones — so this class simply inherits it.


The wake-word detector needs raw mic chunks in parallel with the SLV
streaming consumer that BaseApp's _mic_pump already drains. We cannot
open the mic twice (ALSA exclusive on reSpeaker / single PortAudio
RawInputStream callback), so the single sounddevice callback fans the
PCM out to every registered tap queue.

Backpressure rule: each tap has its own bounded queue. If a tap consumer
falls behind we drop the oldest buffered chunk for THAT tap only — never
block the primary _in_queue that feeds SLV. Wake-word detection is OK
with occasional gaps; user-utterance ASR is not.

NOTE: we override the private ``_safe_put`` method of AudioIO. The
upstream framework may change its signature; if that happens, switch
to a composition-based wrapper that owns the sounddevice callback
itself. Pinned to the API observed in ovs_agent 0.1.0.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from ovs_agent.audio_io import AudioIO

logger = logging.getLogger(__name__)


class TappedAudioIO(AudioIO):
    # Mic-gate-during-TTS hold-off (ms). After playback drains, we keep
    # the mic muted for this long so the tail of the speaker output (and
    # any room reverb) doesn't get fed back as a fake user utterance.
    # 300 ms covers typical Bluetooth-class latency + reSpeaker JST
    # speaker decay on the seeed-orin-nx hardware.
    _PLAYBACK_HOLDOFF_MS = int(os.getenv("MIC_GATE_HOLDOFF_MS", "300"))

    def __init__(self, *args, **kwargs) -> None:
        """Multi-channel mic handling (``mic_channels`` /
        ``mic_channel_select``) lives in :class:`AudioIO` now — every app
        needs it, not just the tap-capable ones — so it just passes through.
        """
        super().__init__(*args, **kwargs)
        self._taps: list[asyncio.Queue[bytes]] = []
        # Echo-suppression state. We can't use ``self.is_playing`` alone
        # because the framework's playback queue drains for hundreds of
        # milliseconds after ``TTSDone`` arrives — that tail is the most
        # likely time to be picked up by the open mic. Track the last
        # playback-end timestamp so we can extend the gate by holdoff.
        self._last_playback_end_ts_ns: int = 0

    def _mic_gate_open(self) -> bool:
        """True when mic chunks should reach downstream consumers.

        Closed while:
          * The agent is actively playing TTS audio (``self.is_playing``).
          * The configured holdoff window after playback end has not yet
            elapsed (room reverb / speaker tail).

        Why this matters for voice-arm: the reSpeaker XVF3800 mic and the
        JST speaker live on the SAME USB device (PortAudio idx 24). Any
        TTS reply is fed back into the open mic and re-enters the WS as
        a "user utterance" — the server VAD then triggers ASR on the
        agent's own speech, producing fake commands, phantom barge-ins,
        and (worst) echo loops where the assistant ends up replying to
        its own previous reply.

        codex 2026-05-26 architecture review: this gate is correctly
        placed in TappedAudioIO (the wrapper that already owns the
        physical mic fanout) rather than in BaseApp, because only
        deployments with shared mic/speaker need it.
        """
        # is_playing: true while ``audio.play()`` has bytes queued or the
        # output stream is non-empty. False once the framework has marked
        # playback complete via ``mark_playback_done``.
        if getattr(self, "is_playing", False):
            return False
        if self._last_playback_end_ts_ns == 0:
            return True  # never played yet
        elapsed_ms = (time.monotonic_ns() - self._last_playback_end_ts_ns) / 1_000_000
        if elapsed_ms < self._PLAYBACK_HOLDOFF_MS:
            return False
        return True

    def mark_playback_done(self) -> None:
        """BaseApp calls this when TTSDone fires (audio_io contract).

        We chain to the parent to keep the ``is_playing`` flag honest,
        then stamp the playback-end timestamp so ``_mic_gate_open``
        keeps the gate closed for ``_PLAYBACK_HOLDOFF_MS`` more.
        """
        parent_mark = getattr(super(), "mark_playback_done", None)
        if callable(parent_mark):
            parent_mark()
        self._last_playback_end_ts_ns = time.monotonic_ns()

    def _safe_put(self, data: bytes) -> None:
        # NOTE 2026-05-26: previous version had an echo-suppression gate
        # here. The reSpeaker XVF3800 has hardware AEC built in (ch 0 =
        # "Processed Conference") so we do NOT need software muting —
        # the FSM issues we hit were not caused by acoustic feedback.
        # Pass through unconditionally.
        # Primary path: hand the chunk to BaseApp's mic queue.
        super()._safe_put(data)
        # Fan out to every tap. list(...) copies the snapshot so a tap
        # registration during iteration can't trip RuntimeError.
        for q in list(self._taps):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                # Slow consumer: drop the oldest chunk in this tap only.
                try:
                    q.get_nowait()
                    q.put_nowait(data)
                except Exception:  # pragma: no cover - defensive
                    pass

    async def start_capture_tap(self, maxsize: int = 32) -> "asyncio.Queue[bytes]":
        """Return a fresh queue that will receive a copy of every mic chunk.

        Caller owns the queue; we keep a reference to fan out into it.
        Multiple taps may coexist. The queue uses int16 little-endian PCM
        at the input sample rate (16k) and channels=1 (AudioIO hard-codes
        single-channel capture — PortAudio down-mixes the multi-channel
        reSpeaker for us).
        """
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=maxsize)
        self._taps.append(q)
        logger.info("capture tap registered (total=%d)", len(self._taps))
        return q

    def stop_capture_tap(self, q: "asyncio.Queue[bytes]") -> None:
        """Unregister a tap so the input callback stops fanning chunks into it.

        Idempotent: a queue already removed (or never registered) is a no-op.
        Used by consumers that re-acquire a fresh tap on restart (e.g.
        OpenWakeWordSource) — without this their old queues accumulate in
        ``_taps`` and the sounddevice callback keeps doing useless work feeding
        orphaned queues (a slow leak)."""
        try:
            self._taps.remove(q)
        except ValueError:
            return
        logger.info("capture tap unregistered (total=%d)", len(self._taps))


__all__ = ["TappedAudioIO"]
