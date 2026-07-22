"""Pure wire adapters between Seeed Realtime V2 and cloud providers.

These classes intentionally perform no networking.  The Gateway relay owns
authentication, reconnects, rate conversion, and WebSocket I/O; adapters own
only deterministic JSON/base64 translation so they can be parity-tested from
frozen canonical traces.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderOutput:
    events: list[dict[str, Any]] = field(default_factory=list)
    audio: list[bytes] = field(default_factory=list)


def _audio_parts(session: dict[str, Any]) -> tuple[dict, dict, dict]:
    audio = session.get("audio") if isinstance(session.get("audio"), dict) else {}
    audio_in = audio.get("input") if isinstance(audio.get("input"), dict) else {}
    audio_out = audio.get("output") if isinstance(audio.get("output"), dict) else {}
    return audio, audio_in, audio_out


def _provider_tools(tools: Any) -> list[dict[str, Any]]:
    """Drop Seeed execution policy while retaining standard function schema."""
    return [
        {key: value for key, value in entry.items() if key != "x_v2v"}
        for entry in (tools or [])
        if isinstance(entry, dict)
    ]


def _provider_turn_detection(turn: Any) -> dict[str, Any] | None:
    if not isinstance(turn, dict) or turn.get("type") in (None, "none"):
        return None
    allowed = {
        "type", "threshold", "prefix_padding_ms", "silence_duration_ms",
        "create_response", "interrupt_response", "eagerness",
    }
    return {key: value for key, value in turn.items() if key in allowed}


class OpenAIRealtimeAdapter:
    """OpenAI Realtime GA JSON adapter (WebSocket audio remains base64)."""

    name = "openai"
    input_rate = 24000
    output_rate = 24000

    @staticmethod
    def capabilities() -> dict[str, bool]:
        return {
            "function_calling": True,
            "conversation_truncate": True,
            "conversation_reset": False,
            # Exact speech needs a deterministic TTS side channel; prompting a
            # generative realtime model is not an exact-speech guarantee.
            "direct_speak": False,
            "input_transcription": True,
        }

    def session_update(self, canonical: dict[str, Any]) -> dict[str, Any]:
        _, audio_in, audio_out = _audio_parts(canonical)
        turn = _provider_turn_detection(audio_in.get("turn_detection"))
        session: dict[str, Any] = {}
        for key in ("type", "model", "output_modalities", "instructions"):
            if key in canonical:
                session[key] = canonical[key]
        if "type" not in session:
            session["type"] = "realtime"
        if "audio" in canonical:
            session["audio"] = {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self.input_rate},
                    "turn_detection": turn,
                },
                "output": {
                    "format": {"type": "audio/pcm"},
                    "voice": audio_out.get("voice"),
                },
            }
        transcription = audio_in.get("transcription")
        if isinstance(transcription, dict) and transcription:
            session["audio"]["input"]["transcription"] = transcription
        if "tools" in canonical:
            session["tools"] = _provider_tools(canonical.get("tools"))
        return {
            "type": "session.update",
            "session": _without_none(session),
        }

    @staticmethod
    def audio_append(pcm: bytes) -> dict[str, Any]:
        return {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }

    def client_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        typ = event.get("type")
        if typ in {
            "input_audio_buffer.commit", "input_audio_buffer.clear",
            "response.create", "response.cancel", "conversation.item.create",
            "conversation.item.truncate",
        }:
            return dict(event)
        return None

    def server_event(self, event: dict[str, Any]) -> ProviderOutput:
        typ = event.get("type")
        if typ == "response.output_audio.delta":
            return ProviderOutput(audio=[base64.b64decode(event.get("delta") or "")])
        return ProviderOutput(events=[dict(event)])


class QwenRealtimeAdapter:
    """Alibaba Cloud Qwen-Audio/Omni Realtime event-shape adapter."""

    name = "qwen"
    input_rate = 16000
    output_rate = 24000

    @staticmethod
    def capabilities() -> dict[str, bool]:
        return {
            "function_calling": True,
            "conversation_truncate": False,
            "conversation_reset": False,
            "direct_speak": False,
            "input_transcription": True,
        }

    def session_update(self, canonical: dict[str, Any]) -> dict[str, Any]:
        _, audio_in, audio_out = _audio_parts(canonical)
        turn = _provider_turn_detection(audio_in.get("turn_detection"))
        session: dict[str, Any] = {}
        if "output_modalities" in canonical or "modalities" in canonical:
            session["modalities"] = ["text", "audio"]
        if "instructions" in canonical:
            session["instructions"] = canonical.get("instructions")
        if "audio" in canonical:
            session.update({
                "voice": audio_out.get("voice"),
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "turn_detection": turn,
                "input_audio_transcription": audio_in.get("transcription"),
            })
        if "tools" in canonical:
            session["tools"] = _provider_tools(canonical.get("tools"))
        return {"type": "session.update", "session": _without_none(session)}

    @staticmethod
    def audio_append(pcm: bytes) -> dict[str, Any]:
        return {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }

    def client_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if event.get("type") in {
            "input_audio_buffer.commit", "input_audio_buffer.clear",
            "response.create", "response.cancel", "conversation.item.create",
        }:
            return dict(event)
        return None

    def server_event(self, event: dict[str, Any]) -> ProviderOutput:
        mapped = dict(event)
        typ = mapped.get("type")
        if typ == "response.audio.delta":
            return ProviderOutput(audio=[base64.b64decode(mapped.get("delta") or "")])
        names = {
            "response.audio.done": "response.output_audio.done",
            "response.audio_transcript.delta": "response.output_audio_transcript.delta",
            "response.audio_transcript.done": "response.output_audio_transcript.done",
            "conversation.item.created": "conversation.item.added",
        }
        if typ in names:
            mapped["type"] = names[typ]
        return ProviderOutput(events=[mapped])


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def create_provider_adapter(name: str):
    normalized = name.strip().lower()
    if normalized == "openai":
        return OpenAIRealtimeAdapter()
    if normalized == "qwen":
        return QwenRealtimeAdapter()
    raise ValueError(f"unsupported realtime provider: {name!r}")
