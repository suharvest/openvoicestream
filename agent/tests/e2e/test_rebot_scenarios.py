"""reBot production-param versions of the classic timing scenarios.

The classic ``tests/e2e/test_{stop_intent,empty_final,idle_stability}.py`` run
on the DEFAULT config (``client_vad_backend='energy'``). Production reBot runs
a DIFFERENT code path — ``client_vad_backend='off'`` + the energy gate +
``wake_word`` mode (see ``_rebot_voice_config``). A scenario can pass on the
energy-VAD path and still regress on the gate path, so these re-run the same
intents under the production reBot voice params against the live SLV.

Run: pytest tests/e2e/test_rebot_scenarios.py -v -s  (needs orin-nx live)
"""
from __future__ import annotations

import asyncio

import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO
from .test_rebot_voice_capture import _rebot_voice_config, _wake, _last_user_text


@pytest.mark.asyncio
async def test_rebot_empty_final_no_llm(test_config):
    """Silence after wake must never open the energy gate → no utterance, no
    LLM call. (Classic empty-final, but on the gate path where a too-low close
    threshold could leak a spurious open.)"""
    cfg = _rebot_voice_config(test_config)
    audio = ScriptedAudioIO([])
    async with run_agent(cfg, audio) as (app, probe):
        await _wake(app, cfg)
        await probe.wait_event("on_wake", timeout=5)
        audio.inject(WAV_DIR / "silence_5s.wav")
        await asyncio.sleep(8)
        assert not _last_user_text(app).strip(), (
            f"silence yielded a non-empty final: {_last_user_text(app)!r}"
        )
        assert probe.assistant_tokens() == [], "LLM was called on silence"


@pytest.mark.asyncio
async def test_rebot_stop_intent_no_llm(test_config):
    """A stop word on the reBot gate path must fire stop-intent and bypass the
    LLM. Uses '别说了' — Chinese stop matching is EXACT full-string
    (app_base.py:497), so a phrase like '停一下' deliberately does NOT match the
    bare stop word '停'."""
    cfg = _rebot_voice_config(test_config)
    audio = ScriptedAudioIO([])
    async with run_agent(cfg, audio) as (app, probe):
        await _wake(app, cfg)
        await probe.wait_event("on_wake", timeout=5)
        await asyncio.sleep(0.5)  # clear the wake-tone suppression
        audio.inject(WAV_DIR / "stop_zh.wav")  # "别说了" — exact stop word
        await probe.wait_event("on_user_utterance", timeout=25)
        try:
            await probe.wait_event("on_user_stop_intent", timeout=8)
        except (TimeoutError, AssertionError):
            uttr = [e for e in probe.events if e.get("event") == "on_user_utterance"]
            pytest.fail(
                f"stop intent did not match on the gate path. utterances: "
                f"{[e.get('data') for e in uttr]}"
            )
        await asyncio.sleep(1.5)
        assert probe.assistant_tokens() == [], (
            f"stop must bypass the LLM; got tokens: {probe.assistant_tokens()!r}"
        )


@pytest.mark.asyncio
async def test_rebot_idle_stability(test_config):
    """After wake with no speech, the silent mic tail must not produce spurious
    utterances or errors while the gate idles (a leaky gate / RMS jitter would
    surface here)."""
    cfg = _rebot_voice_config(test_config)
    audio = ScriptedAudioIO([])
    async with run_agent(cfg, audio) as (app, probe):
        await _wake(app, cfg)
        await probe.wait_event("on_wake", timeout=5)
        await asyncio.sleep(12)
        utt = [e for e in probe.events if e.get("event") == "on_user_utterance"]
        assert utt == [], (
            f"idle produced spurious utterances: {[u.get('data') for u in utt]}"
        )
        assert probe.errors == [], f"idle produced errors: {probe.errors}"
