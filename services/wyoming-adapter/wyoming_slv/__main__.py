"""Entrypoint: run the Wyoming STT and TTS servers side by side.

Both servers are thin translators in front of an already-running
seeed-local-voice instance (``SLV_BASE_URL``). No model code lives here, and
the voice image is not modified — this service is deployed alongside it so the
"we-are-the-brain" and "HA-is-the-brain" integration shapes can coexist.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os

from wyoming.info import (
    Artifact,
    AsrModel,
    AsrProgram,
    Attribution,
    Info,
    TtsProgram,
    TtsVoice,
)
from wyoming.server import AsyncServer

from .stt import SlvSttHandler
from .tts import SlvTtsHandler
from .upstream import SlvClient

_LOGGER = logging.getLogger("wyoming_slv")

ATTRIBUTION = Attribution(
    name="Seeed Studio / seeed-local-voice",
    url="https://github.com/Seeed-Projects/seeed-local-voice",
)
VERSION = "0.1.0"


def build_stt_info(languages: list[str], model_name: str) -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="seeed-local-voice",
                description="Streaming ASR served by seeed-local-voice",
                attribution=ATTRIBUTION,
                installed=True,
                version=VERSION,
                # HA's wyoming/stt.py breaks on the first Transcript and never
                # reads transcript-start/chunk/stop -> declare false and emit a
                # single final Transcript.
                supports_transcript_streaming=False,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=ATTRIBUTION,
                        installed=True,
                        version=None,
                        languages=languages,
                    )
                ],
            )
        ]
    )


def build_tts_info(languages: list[str], voice_name: str) -> Info:
    return Info(
        tts=[
            TtsProgram(
                name="seeed-local-voice",
                description="Streaming TTS served by seeed-local-voice",
                attribution=ATTRIBUTION,
                installed=True,
                version=VERSION,
                # HA's wyoming/tts.py gates streaming input on this flag.
                supports_synthesize_streaming=True,
                voices=[
                    TtsVoice(
                        name=voice_name,
                        description=voice_name,
                        attribution=ATTRIBUTION,
                        installed=True,
                        version=None,
                        languages=languages,
                    )
                ],
            )
        ]
    )


async def _serve(uri: str, factory) -> None:
    server = AsyncServer.from_uri(uri)
    _LOGGER.info("listening on %s", uri)
    await server.run(factory)


async def main_async(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    client = SlvClient(args.base_url)
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]

    stt_info = build_stt_info(languages, args.stt_model)
    tts_info = build_tts_info(languages, args.tts_voice)

    _LOGGER.info("upstream voice service: %s", client.base_url)

    await asyncio.gather(
        _serve(
            f"tcp://{args.host}:{args.stt_port}",
            functools.partial(SlvSttHandler, client=client, info=stt_info),
        ),
        _serve(
            f"tcp://{args.host}:{args.tts_port}",
            functools.partial(SlvTtsHandler, client=client, info=tts_info),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="wyoming-slv")
    parser.add_argument(
        "--base-url", default=os.environ.get("SLV_BASE_URL", "http://127.0.0.1:8621")
    )
    parser.add_argument("--host", default=os.environ.get("WYOMING_HOST", "0.0.0.0"))
    parser.add_argument(
        "--stt-port", type=int, default=int(os.environ.get("WYOMING_STT_PORT", "10300"))
    )
    parser.add_argument(
        "--tts-port", type=int, default=int(os.environ.get("WYOMING_TTS_PORT", "10200"))
    )
    parser.add_argument(
        "--languages", default=os.environ.get("WYOMING_LANGUAGES", "zh,en")
    )
    parser.add_argument(
        "--stt-model", default=os.environ.get("WYOMING_STT_MODEL", "slv-asr")
    )
    parser.add_argument(
        "--tts-voice", default=os.environ.get("WYOMING_TTS_VOICE", "slv-default")
    )
    parser.add_argument("--debug", action="store_true", default=bool(os.environ.get("WYOMING_DEBUG")))
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
