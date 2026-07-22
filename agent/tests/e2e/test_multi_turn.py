"""Multi-turn: 3 utterances → session history preserved + WS stays alive.

NOTE (reconnect semantics changed — commit 07c6466 "revert proactive
reconnect on tts_done"): proactive reconnect on TTSDone (and on wake) was
INTENTIONALLY removed. SLV server v1.15+ added an ASR-turn wall-clock
timeout that force-releases the SessionLimiter slot, so the old client-side
proactive reconnect is no longer needed and was actively causing 4429
too_many_sessions. The WS now PERSISTS across turns (correct per the SLV
multi_utterance protocol where session_complete=False means the dialog
continues); reconnect fires only on a session_complete final or genuine WS
death. The old "≥2 reconnects per multi-turn session" assertion asserted the
removed behavior, so this test now asserts HEALTHY multi-turn instead:
3 turns each yielding user+assistant, full history preserved, WS still
healthy at the end, and NO per-turn reconnect storm.
"""
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_multi_turn(test_config):
    audio = ScriptedAudioIO([
        (800,  WAV_DIR / "hello.wav"),
        (8000, WAV_DIR / "weather.wav"),
        (8000, WAV_DIR / "hello.wav"),
    ])
    async with run_agent(test_config, audio) as (app, probe):
        # Wait for 3 user utterances + final idle.
        for i in range(3):
            await probe.wait_event("on_user_utterance", timeout=30)
            await probe.wait_state("speaking", timeout=30)
            # Wait for that turn's TTS to finish before checking the next.
            # Track via assistant_done events.
            await _wait_assistant_done_count(probe, i + 1, timeout=30)

        # History: 3 user + 3 assistant = 6 messages (each turn ran asr_final
        # → LLM → tts and was appended to the session history).
        assert len(app.session.history) == 6, (
            f"expected 6 messages, got {len(app.session.history)}: "
            f"{[(m['role'], m['content'][:20]) for m in app.session.history]}"
        )
        # Each turn produced an assistant_done (tts finished) — 3 of them.
        done = sum(1 for e in probe.events if e.get("event") == "on_assistant_done")
        assert done >= 3, f"expected ≥3 assistant_done, saw {done}"

        # Healthy multi-turn: the WS PERSISTS across turns (no proactive
        # reconnect on tts_done since 07c6466). It must still be healthy at
        # the end, and there must be NO per-turn reconnect storm — a single
        # persistent session should see well under one reconnect per turn.
        assert app.slv.is_healthy(), "SLV WS should still be healthy after 3 turns"
        reconnects = [e for e in probe.events if e.get("event") == "on_slv_reconnect"]
        assert len(reconnects) < 3, (
            f"expected the WS to persist across turns (no per-turn reconnect "
            f"storm), but saw {len(reconnects)} reconnects"
        )


@pytest.mark.asyncio
async def test_multi_turn_deep(test_config):
    """MT-006: N>4 dialogue on one persistent session.

    The 3-turn ``test_multi_turn`` proves the multi_utterance WS survives a
    couple of turns; this extends it into the N>4 regime that the no-VAD
    multi-turn ASR fix (commit 9b084a5) actually targets — every utterance
    after the first must still open/accept ASR. Five turns each yield a
    user+assistant pair (history → 10 messages, ≥5 assistant_done), the WS
    stays healthy, and there is no per-turn reconnect storm at depth. A
    regression of the ``asr_started_once`` gate would silently drop the 2nd+
    turn here (fewer than 10 messages), so this is a direct depth-guard for
    the fix now live in prod.
    """
    # Alternate two short-reply utterances so each turn's TTS finishes within
    # the inter-utterance gap (no long story replies → stable timing).
    script = [(800, WAV_DIR / "hello.wav")]
    for i in range(4):
        wav = "weather.wav" if i % 2 == 0 else "hello.wav"
        script.append((8000, WAV_DIR / wav))
    audio = ScriptedAudioIO(script)

    async with run_agent(test_config, audio) as (app, probe):
        for i in range(5):
            await probe.wait_event("on_user_utterance", timeout=30)
            await probe.wait_state("speaking", timeout=30)
            await _wait_assistant_done_count(probe, i + 1, timeout=30)

        # 5 user + 5 assistant = 10 messages: every turn (incl. 2nd–5th) ran
        # asr_final → LLM → tts. A dropped 2nd+ utterance would short this.
        assert len(app.session.history) == 10, (
            f"expected 10 messages across 5 turns, got {len(app.session.history)}: "
            f"{[(m['role'], m['content'][:16]) for m in app.session.history]}"
        )
        done = sum(1 for e in probe.events if e.get("event") == "on_assistant_done")
        assert done >= 5, f"expected ≥5 assistant_done across 5 turns, saw {done}"

        # Persistent WS at depth: healthy and no per-turn reconnect storm.
        assert app.slv.is_healthy(), "SLV WS should still be healthy after 5 turns"
        reconnects = [e for e in probe.events if e.get("event") == "on_slv_reconnect"]
        assert len(reconnects) < 5, (
            f"expected the WS to persist across 5 turns (no per-turn reconnect "
            f"storm), but saw {len(reconnects)} reconnects"
        )


async def _wait_assistant_done_count(probe, n: int, timeout: float = 30) -> None:
    import asyncio, time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cnt = sum(1 for e in probe.events if e.get("event") == "on_assistant_done")
        if cnt >= n:
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(f"expected {n} on_assistant_done, only saw "
                       f"{sum(1 for e in probe.events if e.get('event') == 'on_assistant_done')}")
