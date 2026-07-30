"""Wyoming STT program backed by the voice service's ``WS /asr/stream``.

Design note — HA does NOT consume streaming transcripts. As of HA core
``homeassistant/components/wyoming/stt.py`` the read loop breaks on the first
``Transcript`` and never looks at ``supports_transcript_streaming`` /
TranscriptStart / TranscriptChunk / TranscriptStop. So this program advertises
``supports_transcript_streaming: false`` and emits exactly ONE final
``Transcript`` per utterance. Upstream partials are consumed (and logged at
debug) but never forwarded.

We still use the *streaming* WS upstream so a long utterance is pushed into the
decoder as it arrives instead of being buffered end-to-end here.
"""

from __future__ import annotations

import logging
from typing import Optional

import aiohttp
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .upstream import SlvClient, parse_asr_message

_LOGGER = logging.getLogger(__name__)


class SlvSttHandler(AsyncEventHandler):
    def __init__(self, *args, client: SlvClient, info: Info, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._client = client
        self._info = info
        self._language = "auto"
        self._sample_rate = 16000
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._sent_bytes = 0
        # HA's stt view forwards audio in arbitrary byte-length chunks (its
        # ``audio-chunk`` payloads are NOT guaranteed to be a whole number of
        # int16 samples). Sending an odd-length frame upstream makes
        # /asr/stream raise "buffer size must be a multiple of element size"
        # and the whole utterance comes back empty. Keep the straggler byte.
        self._align_tail = b""

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
            _LOGGER.debug("transcribe: language=%s", self._language)
            return True

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self._sample_rate = start.rate
            _LOGGER.info(
                "audio-start rate=%s width=%s channels=%s",
                start.rate,
                start.width,
                start.channels,
            )
            await self._open_upstream()
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self._ws is None:
                # Some clients skip audio-start; open lazily using chunk format.
                self._sample_rate = chunk.rate
                await self._open_upstream()
            assert self._ws is not None
            buf = self._align_tail + chunk.audio
            usable = len(buf) - (len(buf) % 2)
            self._align_tail = buf[usable:]
            if usable:
                await self._ws.send_bytes(buf[:usable])
                self._sent_bytes += usable
            return True

        if AudioStop.is_type(event.type):
            text = await self._finish_upstream()
            _LOGGER.info(
                "audio-stop after %d bytes -> transcript=%r", self._sent_bytes, text
            )
            await self.write_event(Transcript(text=text).event())
            self._sent_bytes = 0
            self._align_tail = b""
            return True

        return True

    # ------------------------------------------------------------------ impl
    async def _open_upstream(self) -> None:
        await self._close_upstream()
        self._session = aiohttp.ClientSession()
        url = self._client.asr_stream_url(
            language=self._language, sample_rate=self._sample_rate
        )
        _LOGGER.debug("opening upstream ASR ws: %s", url)
        self._ws = await self._session.ws_connect(url, max_msg_size=0, heartbeat=None)

    async def _finish_upstream(self) -> str:
        if self._ws is None:
            return ""
        text = ""
        try:
            # Empty binary frame = end-of-stream (forced endpoint).
            await self._ws.send_bytes(b"")
            async for msg in self._ws:
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue
                payload = parse_asr_message(msg.data)
                if payload.get("error"):
                    _LOGGER.error("upstream ASR error: %s", payload["error"])
                    break
                if payload.get("is_final"):
                    text = (payload.get("text") or "").strip()
                    break
                if payload.get("text"):
                    _LOGGER.debug("upstream partial: %s", payload["text"])
        except Exception:
            _LOGGER.exception("upstream ASR stream failed")
        finally:
            await self._close_upstream()
        return text

    async def _close_upstream(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    async def disconnect(self) -> None:  # pragma: no cover - lifecycle
        await self._close_upstream()
