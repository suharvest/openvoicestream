from __future__ import annotations

import base64

from server.core.realtime_provider import OpenAIRealtimeAdapter, QwenRealtimeAdapter
from server.core.realtime_relay import _resample_pcm16, provider_settings


CANONICAL_SESSION = {
    "type": "realtime",
    "model": "provider-model",
    "output_modalities": ["audio"],
    "instructions": "Be concise",
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 16000},
            "transcription": {"language": "zh"},
            "turn_detection": {"type": "server_vad", "create_response": True},
        },
        "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "Tina"},
    },
    "tools": [{
        "type": "function", "name": "wave", "description": "wave",
        "parameters": {"type": "object"}, "x_v2v": {"timeout_s": 12},
    }],
}


def test_openai_adapter_uses_ga_session_shape_and_binary_bridge() -> None:
    adapter = OpenAIRealtimeAdapter()
    update = adapter.session_update(CANONICAL_SESSION)
    session = update["session"]
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["input"]["format"] == {
        "type": "audio/pcm", "rate": 24000,
    }
    assert session["tools"][0]["name"] == "wave"
    assert "x_v2v" not in session["tools"][0]
    tools_only = adapter.session_update({"tools": CANONICAL_SESSION["tools"]})
    assert set(tools_only["session"]) == {"type", "tools"}
    pcm = b"\x01\x00" * 8
    assert adapter.audio_append(pcm)["audio"] == base64.b64encode(pcm).decode()
    output = adapter.server_event({
        "type": "response.output_audio.delta",
        "delta": base64.b64encode(pcm).decode(),
    })
    assert output.audio == [pcm]


def test_qwen_adapter_flattens_session_and_normalizes_audio_events() -> None:
    adapter = QwenRealtimeAdapter()
    session = adapter.session_update(CANONICAL_SESSION)["session"]
    assert session["input_audio_format"] == "pcm"
    assert session["turn_detection"]["type"] == "server_vad"
    qwen_tools_only = adapter.session_update({"tools": CANONICAL_SESSION["tools"]})
    assert set(qwen_tools_only["session"]) == {"tools"}
    pcm = b"\x02\x00" * 4
    output = adapter.server_event({
        "type": "response.audio.delta",
        "delta": base64.b64encode(pcm).decode(),
    })
    assert output.audio == [pcm]
    done = adapter.server_event({"type": "response.audio.done", "response_id": "r"})
    assert done.events[0]["type"] == "response.output_audio.done"


def test_relay_resamples_openai_input_and_builds_provider_settings(monkeypatch) -> None:
    pcm_16k = (b"\x01\x00" * 160)
    pcm_24k = _resample_pcm16(pcm_16k, 16000, 24000)
    assert len(pcm_24k) == 480

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    url, headers, model = provider_settings("openai")
    assert model == "gpt-realtime-2.1"
    assert "model=gpt-realtime-2.1" in url
    assert headers == [("Authorization", "Bearer secret")]

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash")
    monkeypatch.setenv(
        "OVS_REALTIME_QWEN_URL",
        "wss://workspace.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
    )
    qwen_url, _, qwen_model = provider_settings("qwen")
    assert qwen_model == "qwen-audio-3.0-realtime-flash"
    assert "model=qwen-audio-3.0-realtime-flash" in qwen_url
