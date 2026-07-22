from __future__ import annotations

import asyncio
import json

from server.core import v2v


class _Ids:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}_{self.n}"


def test_session_update_normalizes_to_internal_config() -> None:
    cfg = v2v.session_update_to_legacy_config({
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "sample_rate": 16000, "channels": 1},
                    "transcription": {"language": "auto"},
                    "turn_detection": {
                        "type": "server_vad",
                        "backend": "silero",
                        "silence_duration_ms": 550,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {"language": "zh", "voice": "default", "speed": 1.1},
            },
        },
    })
    assert cfg["type"] == "config"
    assert cfg["sample_rate"] == 16000
    assert cfg["asr_language"] == "auto"
    assert cfg["tts_language"] == "zh"
    assert cfg["vad"] == "silero"
    assert cfg["vad_silence_ms"] == 550
    assert cfg["multi_utterance"] is True
    assert cfg["_create_response"] is True


def test_response_lifecycle_has_one_terminal_event() -> None:
    adapter = v2v.RealtimeV2EventAdapter(id_factory=_Ids())
    started = adapter.translate({"type": "tts_started", "sentence": "hello"})
    assert [e["type"] for e in started] == [
        "response.created",
        "response.output_item.added",
        "x_v2v.tts_sentence.started",
    ]
    response_id = started[0]["response"]["id"]

    sentence_done = adapter.translate({
        "type": "tts_sentence_done", "sentence": "hello",
    })
    assert sentence_done[0]["type"] == "response.output_audio_transcript.delta"
    assert sentence_done[0]["delta"] == "hello"

    done = adapter.translate({"type": "tts_done"})
    assert [e["type"] for e in done] == [
        "response.output_audio_transcript.done",
        "response.output_audio.done",
        "response.done",
    ]
    assert done[0]["response_id"] == response_id
    assert done[0]["transcript"] == "hello"
    assert done[2]["response"]["id"] == response_id
    assert done[2]["response"]["status"] == "completed"


def test_session_updated_reports_effective_audio_format() -> None:
    adapter = v2v.RealtimeV2EventAdapter(
        input_sample_rate=16000,
        output_sample_rate=24000,
        id_factory=_Ids(),
    )
    event = adapter.session_updated({
        "audio": {
            "input": {
                "format": {"sample_rate": 8000},
                "turn_detection": {"type": "server_vad", "create_response": True},
            },
            "output": {"format": {"sample_rate": 16000}},
        }
    }, create_response=False, interrupt_response=True)
    audio = event["session"]["audio"]
    assert audio["input"]["format"]["rate"] == 16000
    assert audio["output"]["format"]["rate"] == 24000
    assert "sample_rate" not in audio["input"]["format"]
    assert audio["input"]["turn_detection"]["create_response"] is False
    assert audio["input"]["turn_detection"]["interrupt_response"] is True


def test_disabled_turn_detection_does_not_implicitly_create_response() -> None:
    cfg = v2v.session_update_to_legacy_config({
        "type": "session.update",
        "session": {
            "audio": {
                "input": {"turn_detection": {"type": "none"}},
                "output": {"language": "zh"},
            },
        },
    })
    assert cfg["_create_response"] is False


def test_cancelled_response_finishes_as_cancelled() -> None:
    adapter = v2v.RealtimeV2EventAdapter(id_factory=_Ids())
    adapter.translate({"type": "tts_started", "sentence": "hello"})
    adapter.mark_cancelled("turn_detected")
    done = adapter.translate({"type": "tts_done"})
    terminal = done[-1]["response"]
    assert terminal["status"] == "cancelled"
    assert terminal["status_details"] == {
        "type": "cancelled",
        "reason": "turn_detected",
    }


def test_vad_and_transcription_share_input_item_id() -> None:
    adapter = v2v.RealtimeV2EventAdapter(id_factory=_Ids())
    speech = adapter.translate({"type": "vad_event", "event": "speech_start"})[0]
    partial = adapter.translate({"type": "asr_partial", "text": "hel"})[0]
    committed = adapter.translate({"type": "asr_endpoint"})[0]
    final = adapter.translate({"type": "asr_final", "text": "hello"})[0]
    assert speech["type"] == "input_audio_buffer.speech_started"
    assert partial["type"] == "conversation.item.input_audio_transcription.delta"
    assert committed["type"] == "input_audio_buffer.committed"
    assert final["type"] == "conversation.item.input_audio_transcription.completed"
    assert {speech["item_id"], partial["item_id"], committed["item_id"], final["item_id"]} == {
        speech["item_id"]
    }


def test_session_final_legacy_tts_done_does_not_create_phantom_response() -> None:
    adapter = v2v.RealtimeV2EventAdapter(id_factory=_Ids())
    assert adapter.translate({"type": "tts_done", "session_complete": True}) == []


def test_voxedge_proxy_translates_events_and_strips_rate_header() -> None:
    from server.main import _RealtimeV2WebSocketProxy

    class FakeWS:
        def __init__(self) -> None:
            self.events = []
            self.audio = []

        async def send_json(self, payload):
            self.events.append(payload)

        async def send_bytes(self, data):
            self.audio.append(data)

    ws = FakeWS()
    adapter = v2v.RealtimeV2EventAdapter(output_sample_rate=24000, id_factory=_Ids())
    proxy = _RealtimeV2WebSocketProxy(ws, adapter)

    async def exercise() -> None:
        await proxy.send_json({"type": "tts_started", "sentence": "hello"})
        await proxy.send_bytes((24000).to_bytes(4, "little"))
        await proxy.send_bytes(b"\x01\x00" * 8)
        await proxy.send_json({"type": "tts_done", "session_complete": False})

    asyncio.run(exercise())

    assert [event["type"] for event in ws.events] == [
        "response.created",
        "response.output_item.added",
        "x_v2v.tts_sentence.started",
        "response.output_audio_transcript.done",
        "response.output_audio.done",
        "response.done",
    ]
    assert ws.audio == [b"\x01\x00" * 8]


def test_voxedge_proxy_preserves_manual_prompt_update_then_response_create() -> None:
    from server.main import _RealtimeV2WebSocketProxy

    class FakeWS:
        def __init__(self) -> None:
            self.incoming = [
                {
                    "type": "websocket.receive",
                    "text": json.dumps({
                        "type": "session.update",
                        "session": {"instructions": "[Faces: Alice]"},
                    }),
                },
                {
                    "type": "websocket.receive",
                    "text": json.dumps({"type": "response.create"}),
                },
            ]
            self.events = []

        async def receive(self):
            return self.incoming.pop(0)

        async def send_json(self, payload):
            self.events.append(payload)

    async def exercise():
        ws = FakeWS()
        adapter = v2v.RealtimeV2EventAdapter(id_factory=_Ids())
        proxy = _RealtimeV2WebSocketProxy(ws, adapter)
        update = json.loads((await proxy.receive())["text"])
        create = json.loads((await proxy.receive())["text"])
        return update, create

    update, create = asyncio.run(exercise())
    assert update == {
        "type": "tool_advertise",
        "tools": [],
        "system_prompt": "[Faces: Alice]",
        "llm_params": {},
        "warm_prefix": False,
    }
    assert create == {"type": "response.create"}


def test_tool_call_is_exposed_as_canonical_function_call() -> None:
    adapter = v2v.RealtimeV2EventAdapter(id_factory=_Ids())
    events = adapter.translate({
        "type": "tool_call",
        "call_id": "call_1",
        "name": "wave",
        "arguments": {"side": "left"},
        "timeout_s": 12.0,
    })
    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.function_call_arguments.done",
    ]
    done = events[-1]
    assert done["call_id"] == "call_1"
    assert done["name"] == "wave"
    assert done["arguments"] == '{"side":"left"}'
    assert done["x_v2v"]["timeout_s"] == 12.0


def test_direct_speak_response_is_tagged_history_free() -> None:
    adapter = v2v.RealtimeV2EventAdapter(id_factory=_Ids())
    adapter.mark_direct_speak()
    created = adapter.translate({"type": "tts_started", "sentence": "safe"})[0]
    assert created["response"]["metadata"]["x_v2v.direct_speak"] is True


def test_voxedge_proxy_normalizes_v2_tools_speech_and_truncate() -> None:
    import json
    from server.main import _RealtimeV2WebSocketProxy

    class FakeWS:
        def __init__(self, payloads) -> None:
            self.payloads = list(payloads)
            self.events = []

        async def receive(self):
            return {
                "type": "websocket.receive",
                "text": json.dumps(self.payloads.pop(0)),
            }

        async def send_json(self, payload):
            self.events.append(payload)

    async def exercise() -> None:
        ws = FakeWS([
            {"type": "session.update", "session": {
                "instructions": "SP",
                "tools": [{"type": "function", "function": {"name": "wave"}}],
            }},
            {"type": "conversation.item.create", "item": {
                "type": "function_call_output", "call_id": "call_1",
                "output": json.dumps({"ok": True, "name": "wave", "result": {"x": 1}}),
            }},
            {"type": "x_v2v.response.speak", "speech": {
                "text": "注意", "conversation": "none",
            }},
            {"type": "conversation.item.truncate", "item_id": "item_a",
             "content_index": 0, "audio_end_ms": 321},
            {"type": "input_audio_buffer.commit"},
        ])
        proxy = _RealtimeV2WebSocketProxy(
            ws, v2v.RealtimeV2EventAdapter(id_factory=_Ids())
        )
        advertise = json.loads((await proxy.receive())["text"])
        result = json.loads((await proxy.receive())["text"])
        speak = json.loads((await proxy.receive())["text"])
        flush = json.loads((await proxy.receive())["text"])
        commit = json.loads((await proxy.receive())["text"])
        assert advertise["type"] == "tool_advertise"
        assert advertise["system_prompt"] == "SP"
        assert result == {
            "type": "tool_result", "call_id": "call_1", "id": "call_1",
            "name": "wave", "ok": True, "result": {"x": 1},
        }
        assert speak == {"type": "text", "text": "注意"}
        assert flush == {"type": "tts_flush"}
        assert commit == {"type": "asr_eos"}
        assert any(
            event["type"] == "conversation.item.truncated"
            and event["audio_end_ms"] == 321
            for event in ws.events
        )

    asyncio.run(exercise())
