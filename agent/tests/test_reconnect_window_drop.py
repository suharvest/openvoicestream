"""T2 — audio spoken during an SLV reconnect window must be dropped (2026-06-13).

Real-machine symptom: "第一次没听到，要再说". With ``reconnect_on_wake=true``
every wake rebuilt the SLV WS; mic audio captured during that ~6s reconnect
window was either lost or — worse — carried across the gap so the next turn's
final came back as a mashed pre-gap + post-gap segment.

The fix (a5ef83a) drops chunks AND clears the pre-roll while
``slv.is_reconnecting()`` is true (app_base.py:1436-1440). The live-SLV e2e
suite can't faithfully open this window; these deterministic tests drive
``_mic_pump`` with a per-chunk ``is_reconnecting`` schedule and assert on the
exact bytes forwarded to SLV — no network, no flake.

See ``tests/sim_pump.py`` for the harness and memory
``voice_first_utterance_capture_2026_06_13`` for the root cause.
"""
from __future__ import annotations

import pytest

from .sim_pump import build_pump, pcm, real_chunks


@pytest.mark.asyncio
async def test_speech_during_reconnect_is_dropped(monkeypatch):
    pre, during, after = pcm(2000), pcm(2500), pcm(3000)
    script = [pre] + [during] * 3 + [after] * 3
    sched = [False] + [True] * 3 + [False] * 3  # reconnecting only over `during`
    app, sent, _ = build_pump(monkeypatch, script, reconnecting_schedule=sched)
    await app._mic_pump()

    reals = real_chunks(sent)
    assert during not in reals, (
        "T2 regression: speech during the SLV reconnect window reached SLV — it "
        "must be dropped so it isn't mashed into the next turn's final."
    )
    assert pre in reals, "pre-reconnect speech should forward normally"
    assert after in reals, "forwarding must resume once reconnect completes"


@pytest.mark.asyncio
async def test_no_carryover_across_the_window(monkeypatch):
    """Pre-roll is cleared on reconnect, so pre-window audio must NOT resurface
    after the window (else ASR decodes pre+post as one garbled utterance)."""
    pre, during, after = pcm(2000), pcm(2500), pcm(3000)
    script = [pre] + [during] * 2 + [after] * 3
    sched = [False] + [True] * 2 + [False] * 3
    app, sent, _ = build_pump(monkeypatch, script, reconnecting_schedule=sched)
    await app._mic_pump()

    reals = real_chunks(sent)
    first_after = reals.index(after)
    assert pre not in reals[first_after:], (
        "T2 regression: pre-reconnect audio leaked past the reconnect window — "
        "pre-roll was not cleared."
    )


@pytest.mark.asyncio
async def test_baseline_without_reconnect_forwards_everything(monkeypatch):
    """Control: the SAME script with no reconnect window forwards the middle
    audio — proving the drop above is caused by ``is_reconnecting``, not the
    gate swallowing it for another reason."""
    pre, mid, after = pcm(2000), pcm(2500), pcm(3000)
    script = [pre] + [mid] * 3 + [after] * 3
    app, sent, _ = build_pump(monkeypatch, script)  # schedule = all False
    await app._mic_pump()

    reals = real_chunks(sent)
    assert mid in reals, "without a reconnect window the middle audio must forward"
