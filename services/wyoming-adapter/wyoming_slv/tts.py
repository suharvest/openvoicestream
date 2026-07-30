"""Wyoming TTS program backed by the voice service's ``POST /tts/stream``.

Two inbound shapes must both work:

1. **Streaming** (HA when ``supports_synthesize_streaming: true``):
   ``synthesize-start`` → ``synthesize-chunk``* → **a full ``synthesize`` with
   the COMPLETE text** ("for compatibility", per HA's own comment in
   ``homeassistant/components/wyoming/tts.py``) → ``synthesize-stop``.
   The trailing full ``Synthesize`` MUST be ignored once a stream has begun,
   otherwise every reply is synthesized and played TWICE.
2. **One-shot**: a bare ``synthesize`` with the whole text and no start/stop.

We synthesize per CLAUSE as chunks arrive (see ``clause.ClauseBuffer``) so
audio starts flowing before ``synthesize-stop``.
"""

from __future__ import annotations

import logging
from typing import Optional

import aiohttp
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .clause import ClauseBuffer
from .upstream import PCM_CHANNELS, PCM_WIDTH, SlvClient

_LOGGER = logging.getLogger(__name__)


class SlvTtsHandler(AsyncEventHandler):
    def __init__(self, *args, client: SlvClient, info: Info, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._client = client
        self._info = info
        self._session: Optional[aiohttp.ClientSession] = None
        self._reset()

    def _reset(self) -> None:
        self._streaming = False       # a synthesize-start was seen
        self._synthesized_any = False  # at least one clause was synthesized
        self._audio_started = False
        self._sample_rate = 0
        self._buffer: Optional[ClauseBuffer] = None
        self._speaker_id: Optional[int] = None
        self._total_bytes = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True

        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            lang = None
            if start.voice is not None:
                lang = start.voice.language
                self._speaker_id = _speaker_from_voice(start.voice.speaker)
            self._reset()
            self._streaming = True
            self._buffer = ClauseBuffer(language=lang)
            _LOGGER.info("synthesize-start (streaming) language=%s", lang)
            return True

        if SynthesizeChunk.is_type(event.type):
            chunk = SynthesizeChunk.from_event(event)
            if self._buffer is None:
                self._streaming = True
                self._buffer = ClauseBuffer()
            for clause in self._buffer.add(chunk.text):
                await self._synthesize_clause(clause)
            return True

        if SynthesizeStop.is_type(event.type):
            if self._buffer is not None:
                for clause in self._buffer.flush():
                    await self._synthesize_clause(clause)
            await self._finish()
            await self.write_event(SynthesizeStopped().event())
            _LOGGER.info("synthesize-stop: %d PCM bytes total", self._total_bytes)
            self._reset()
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            if self._streaming:
                # THE DOUBLE-SYNTHESIS TRAP: HA repeats the complete text here
                # for back-compat. Ignoring it is mandatory — synthesizing it
                # would play every reply twice. Only fall through when the
                # stream produced nothing at all (defensive fallback).
                if self._synthesized_any:
                    _LOGGER.info(
                        "ignoring trailing full synthesize (%d chars) — "
                        "already streamed",
                        len(synthesize.text or ""),
                    )
                    return True
                _LOGGER.warning(
                    "stream produced no clauses; falling back to full synthesize"
                )
            else:
                self._reset()
            if synthesize.voice is not None:
                self._speaker_id = _speaker_from_voice(synthesize.voice.speaker)
            text = (synthesize.text or "").strip()
            _LOGGER.info("synthesize (one-shot) %d chars", len(text))
            if text:
                await self._synthesize_clause(text)
            await self._finish()
            self._reset()
            return True

        return True

    # ------------------------------------------------------------------ impl
    async def _synthesize_clause(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        session = await self._get_session()
        _LOGGER.info("clause -> upstream /tts/stream: %r", text)
        try:
            async for sample_rate, pcm in self._client.tts_stream(
                session, text, speaker_id=self._speaker_id
            ):
                if not self._audio_started:
                    self._sample_rate = sample_rate
                    await self.write_event(
                        AudioStart(
                            rate=sample_rate,
                            width=PCM_WIDTH,
                            channels=PCM_CHANNELS,
                        ).event()
                    )
                    self._audio_started = True
                await self.write_event(
                    AudioChunk(
                        rate=self._sample_rate,
                        width=PCM_WIDTH,
                        channels=PCM_CHANNELS,
                        audio=pcm,
                    ).event()
                )
                self._total_bytes += len(pcm)
            self._synthesized_any = True
        except Exception:
            _LOGGER.exception("upstream TTS failed for clause %r", text)

    async def _finish(self) -> None:
        if self._audio_started:
            await self.write_event(AudioStop().event())
            self._audio_started = False

    async def disconnect(self) -> None:  # pragma: no cover - lifecycle
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None


def _speaker_from_voice(speaker: Optional[str]) -> Optional[int]:
    if not speaker:
        return None
    try:
        return int(speaker)
    except (TypeError, ValueError):
        return None
