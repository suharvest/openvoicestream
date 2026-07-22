# OpenVoiceStream Examples

Small clients for validating and integrating the public streaming APIs.

## Stream TTS to WAV

No third-party Python packages required. The script calls `/tts/stream`, reads
the 4-byte sample-rate prefix, and writes a playable WAV.

```bash
python3 examples/stream_tts_to_wav.py \
  --url http://device:8621 \
  --text "你好，欢迎使用 OpenVoiceStream。" \
  --out /tmp/ovs-tts.wav
```

Use `http://device:8621` for the default compose deployment on every device.

## V2V WebSocket TTS-Only

Requires `websockets`. This demonstrates the unified `/v2v/stream` protocol in
TTS-only mode by sending text chunks and collecting returned PCM.

```bash
uv run --with websockets python examples/v2v_tts_only.py \
  --url ws://device:8621/v2v/stream \
  --text "Hello from a streaming client." \
  --out /tmp/ovs-v2v-tts.wav
```

Use this as the smallest copy-paste starting point for feeding LLM tokens into
OpenVoiceStream TTS.

## Agent Framework Examples (`agent/`)

面向想基于 `ovs_agent` 框架写自己语音应用的开发者示例（复制即改即跑）。
Developer examples for building voice apps on the `ovs_agent` framework:

- `agent/minimal_echo_app.py` — 最小语音应用（subclass `BaseApp` + 一个 hook）
- `agent/voice_tools_app.py` — 语音工具调用闭环（`@tool` 注册 + allowlist）
- `agent/custom_mode_app.py` — 自定义 `AppMode`（复读机模式 + mode 切换）

See [`agent/README.md`](agent/README.md) for prerequisites and how to run.
