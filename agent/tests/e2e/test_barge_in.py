"""Barge-in: while TTS is playing, a new utterance must cancel it.

Sync model: the test scripts ONLY the turn-1 trigger WAV. Once we see
TTS bytes start landing in the fake sink (= agent is actively
playing), we dynamically inject the barge-in WAV via
`ScriptedAudioIO.inject()`. This avoids races where a too-fast LLM
response finishes TTS before a fixed pre-delay elapses.

ROBUSTNESS (2026-06-14): the default test_config system prompt forces CONCISE
replies (2-3 sentences), so any turn-1 trigger finishes its TTS in ~1-2s —
before the ~2.7s barge utterance is even recognized. The barge then lands as a
fresh turn 2 and BARGED_IN is never reached (a false failure unrelated to barge
handling). Fix: override the system prompt to make the model COUNT out loud
("请从一数到五十" → a long, deterministic multi-second reply), guaranteeing a
wide window of active TTS to barge into. We also inject the moment SPEAKING is
reached (not after a fixed delay) to maximize overlap.
"""
import asyncio
import time
from dataclasses import replace

import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_barge_in_cancels_tts(test_config):
    # Override the concise-by-default prompt so the count actually produces a
    # long reply → reliably long TTS window for the barge to land in.
    cfg = replace(
        test_config,
        system_prompt=(
            "你是语音助手。当用户要求数数时，从头到尾一个数字一个数字地"
            "完整数出来，不要省略、不要只回答一句话。"
        ),
    )
    # Count prompt → long, deterministic reply → wide TTS window.
    audio = ScriptedAudioIO([
        (500, WAV_DIR / "count_to_fifty.wav"),
    ])
    async with run_agent(cfg, audio) as (app, probe):
        # Step 1: wait until the agent is actively SPEAKING (TTS bytes flowing
        # AND state == speaking), so the barge lands firmly mid-playback.
        t_dl = time.monotonic() + 30
        while time.monotonic() < t_dl:
            speaking = any(s == "speaking" for _, s in probe.state_history)
            if speaking and len(audio.captured_tts) > 0:
                break
            await asyncio.sleep(0.05)
        assert len(audio.captured_tts) > 0, "TTS never started"
        assert any(s == "speaking" for _, s in probe.state_history), (
            f"expected SPEAKING before barge-in; saw {probe.state_history}"
        )

        # Step 2: inject the barge-in WAV immediately (turn 1 is still mid-count,
        # so a long window of TTS remains to be cancelled).
        bytes_at_inject = len(audio.captured_tts)
        audio.inject(WAV_DIR / "barge_in.wav")

        # Step 3: wait for barged_in state.
        t_dl = time.monotonic() + 15
        while time.monotonic() < t_dl:
            if any(s == "barged_in" for _, s in probe.state_history):
                break
            await asyncio.sleep(0.05)
        else:
            partials = [
                e.get("data") for e in probe.events
                if e.get("event") == "on_user_partial"
            ]
            pytest.fail(
                f"barged_in state never reached. "
                f"state_history={probe.state_history!r} "
                f"partials={partials!r}"
            )

        # Step 4: TTS bytes should stop growing within ~500 ms after
        # barge-in. Allow a small grace (one frame ≈ 32-64 KB at 24kHz).
        await asyncio.sleep(0.6)
        bytes_after = len(audio.captured_tts)
        grew = bytes_after - bytes_at_inject
        # Allow ~150 KB grace for in-flight chunks (≈ 3s of 24kHz mono
        # int16). If TTS keeps streaming a full LLM response after
        # barge-in, growth would be much larger.
        assert grew < 150_000, (
            f"TTS kept streaming after barge-in: "
            f"+{grew} bytes (inject@{bytes_at_inject}, after={bytes_after})"
        )
