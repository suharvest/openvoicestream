from __future__ import annotations

import asyncio
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from ovs_agent import Config
from ovs_agent.audio.tapped_audio_io import TappedAudioIO
from ovs_agent.kws.compiler import CompiledKeywords, PhraseCompiler
from ovs_agent.kws.sherpa_backend import SherpaKwsBackend
from ovs_agent.wake_sources.runtime_kws import RuntimeKwsSource


def test_phrase_compiler_uses_private_files_and_no_shell(monkeypatch):
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        raw, output = argv[-2:]
        observed["raw_mode"] = os.stat(raw).st_mode & 0o777
        observed["out_mode"] = os.stat(output).st_mode & 0o777
        text = open(raw, encoding="utf-8").read()
        assert "你好 小智 @你好_小智" in text
        with open(output, "w", encoding="utf-8") as f:
            f.write("ni 3 hao 3 @你好 小智\n")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    compiler = PhraseCompiler(tokens="tokens.txt", lexicon="lexicon.txt", timeout_s=3)
    result = compiler.compile(["  你好   小智  ", "你好 小智"])

    assert result.phrases == ("你好 小智",)
    assert result.keywords == "ni 3 hao 3 @你好 小智\n"
    assert observed["argv"][:2] == ["sherpa-onnx-cli", "text2token"]
    assert "shell" not in observed["kwargs"]
    assert observed["kwargs"]["timeout"] == 3
    assert observed["raw_mode"] == observed["out_mode"] == 0o600


def test_phrase_compiler_timeout_is_readable(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(TimeoutError, match="timed out"):
        PhraseCompiler(tokens="t", lexicon="l", timeout_s=1).compile(["Hey Seeed"])


class FakeSpotter:
    loads = 0

    def __init__(self, **kwargs):
        type(self).loads += 1
        self.kwargs = kwargs

    def create_stream(self, keywords=None):
        return SimpleNamespace(keywords=keywords, accepted=[], keyword="")

    def is_ready(self, stream):
        return False

    def decode_stream(self, stream):
        raise AssertionError("not ready")

    def get_result(self, stream):
        return SimpleNamespace(keyword=stream.keyword)

    def reset_stream(self, stream):
        stream.keyword = ""


def test_sherpa_backend_loads_model_once_for_keyword_updates():
    FakeSpotter.loads = 0
    module = SimpleNamespace(KeywordSpotter=FakeSpotter)
    cfg = {"tokens": "t", "encoder": "e", "decoder": "d", "joiner": "j"}
    backend = SherpaKwsBackend(cfg, module=module)
    first = backend.create_stream(CompiledKeywords(("a",), "A\n"))
    second = backend.create_stream(CompiledKeywords(("b",), "B\n"))
    assert FakeSpotter.loads == 1
    assert first.keywords == "A\n"
    assert second.keywords == "B\n"
    assert backend._spotter.kwargs["keywords_file"] == ""


class FakeCompiler:
    def compile(self, phrases):
        phrases = tuple(phrases)
        if phrases == ("bad",):
            raise ValueError("unsupported")
        return CompiledKeywords(phrases, "|".join(phrases))


class FakeBackend:
    def __init__(self):
        self.streams = []
        self.trigger = False

    def create_stream(self, compiled):
        stream = {"keywords": compiled.keywords}
        self.streams.append(stream)
        return stream

    def detect(self, stream, samples, sample_rate):
        assert samples.dtype == np.float32
        assert sample_rate == 16000
        if self.trigger:
            self.trigger = False
            return stream["keywords"]
        return None


class FakeAudio:
    def __init__(self):
        self.q = asyncio.Queue()
        self.tap_count = 0
        self.stop_count = 0

    async def start_capture_tap(self):
        self.tap_count += 1
        return self.q

    def stop_capture_tap(self, tap):
        self.stop_count += 1


class FakeApp:
    def __init__(self):
        self.audio = FakeAudio()
        self.config = SimpleNamespace(
            audio_input_sample_rate=16000, wake_phrases=["old", "legacy"]
        )
        self.wakes = []

    async def wake(self, source=""):
        self.wakes.append(source)


@pytest.mark.asyncio
async def test_runtime_update_is_atomic_on_failure():
    app = FakeApp()
    backend = FakeBackend()
    source = RuntimeKwsSource(
        app, phrases=["old"], compiler=FakeCompiler(), backend=backend, cooldown_s=0
    )
    assert source.setup()
    old_stream = source._stream

    with pytest.raises(ValueError, match="unsupported"):
        await source.update_phrases(["bad"])
    assert source.phrases == ("old",)
    assert source._stream is old_stream

    assert await source.update_phrases(["你好小智", "Hey Seeed"]) == (
        "你好小智", "Hey Seeed"
    )
    assert source._stream is not old_stream
    assert app.config.wake_phrases == ["你好小智", "Hey Seeed", "legacy"]


@pytest.mark.asyncio
async def test_runtime_source_wakes_and_releases_tap():
    app = FakeApp()
    backend = FakeBackend()
    source = RuntimeKwsSource(
        app, phrases=["你好小智"], compiler=FakeCompiler(), backend=backend, cooldown_s=0
    )
    assert source.setup()
    await source.start()
    try:
        await asyncio.sleep(0.55)
        backend.trigger = True
        await app.audio.q.put(np.ones(320, dtype=np.int16).tobytes())
        await asyncio.sleep(0.05)
        assert app.wakes == ["runtime_kws"]
        assert source.last_chunk_ts() is not None
    finally:
        await source.stop()
    assert app.audio.stop_count == 1


def test_conversation_registers_runtime_kws_only_for_opt_in(monkeypatch):
    from ovs_agent.apps.conversation import app as conversation

    made = []

    class FakeSource:
        name = "runtime_kws"
        local_audio = True

        def __init__(self, app, **kwargs):
            made.append(kwargs)

        def setup(self):
            return True

    monkeypatch.setattr(conversation, "RuntimeKwsSource", FakeSource)
    disabled = conversation.ConversationApp(Config(pipeline_mode="wake_word"))
    assert not any(getattr(p, "name", "") == "runtime_kws" for p in disabled.plugins)

    cfg = Config(
        pipeline_mode="wake_word",
        wake_sources=[],
        metadata={
            "wakeword": {
                "backend": "sherpa_onnx",
                "phrases": ["你好小智", "Hey Seeed"],
                "model": {"tokens": "/models/tokens.txt"},
                "compiler": {"lexicon": "/models/lexicon.txt"},
            }
        },
    )
    enabled = conversation.ConversationApp(cfg)
    assert isinstance(enabled.audio, TappedAudioIO)
    assert any(getattr(p, "name", "") == "runtime_kws" for p in enabled.plugins)
    assert made[0]["phrases"] == ["你好小智", "Hey Seeed"]
    assert enabled.config.wake_phrases[:2] == ["你好小智", "Hey Seeed"]


@pytest.mark.asyncio
async def test_base_app_local_audio_capability_controls_mic_skip():
    from ovs_agent.app_base import BaseApp
    from ovs_agent.state import ConvState

    app = BaseApp.__new__(BaseApp)
    app.config = SimpleNamespace(wake_mic_skip_ms=250)
    app.plugins = [SimpleNamespace(name="custom_mic_kws", local_audio=True)]
    app._state = ConvState.IDLE  # early return after source handling
    app._wake_mic_skip_until = 0.0
    await app.wake(source="custom_mic_kws")
    assert app._wake_mic_skip_until > 0

    app._wake_mic_skip_until = 0.0
    await app.wake(source="http")
    assert app._wake_mic_skip_until == 0.0
