"""T3/T4 — a command onset spoken inside a notification-tone suppression window
is dropped (2026-06-13).

After a wake tone (T3) or a completion tone (T4) the mic is suppressed for
``mic_suppress_tail_ms`` so the agent doesn't transcribe its own beep
(app_base.py:1450-1453, ``_play_wake_tone`` / ``_play_sleep_tone`` and the
grasp_plugin done-tone). The window not only drops audio, it ``preroll.clear()``
s — so the energy-gate pre-roll (which rescues a low-energy onset, T1) CANNOT
recover anything spoken inside it. If the user starts talking before the tail
elapses, the command onset is gone and ASR hears only the tail of the word.

This is why the tails were shortened (wake 600→250ms, done 200→120ms). These
deterministic tests pin the mechanism — identical for both tones since both
write the same ``_local_output_mic_suppress_until`` — and the sweep documents
the tail-vs-reaction trade-off without a microphone or the live SLV.
"""
from __future__ import annotations

import pytest

from .sim_pump import build_pump, pcm, real_chunks, LOUD, SILENCE


@pytest.mark.asyncio
async def test_onset_under_suppression_is_dropped(monkeypatch):
    onset, post = pcm(2000), pcm(3000)
    n_suppressed = 3
    script = [onset] * n_suppressed + [post] * 3
    app, sent, clock = build_pump(monkeypatch, script)
    # Tone suppression covers exactly the first n chunks (the user spoke too
    # soon after the beep).
    app._local_output_mic_suppress_until = clock.window_first(n_suppressed)
    await app._mic_pump()

    reals = real_chunks(sent)
    assert onset not in reals, (
        "T3/T4 regression: command onset spoken under tone-suppression reached "
        "SLV — it should be dropped (and the pre-roll cannot recover it)."
    )
    assert post in reals, "speech after the suppression window must forward"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tail_chunks,react_chunks,onset_survives",
    [
        (3, 0, False),  # speak immediately → onset inside the tail → lost
        (3, 1, False),  # 100ms after the beep → still inside → lost
        (3, 3, True),   # wait out the whole tail → onset survives (boundary)
        (1, 2, True),   # short tail, react later → survives
        (2, 4, True),   # short tail, slow reaction → survives
    ],
)
async def test_tail_vs_reaction_sweep(monkeypatch, tail_chunks, react_chunks, onset_survives):
    """Tail length × reaction time. The onset (a single soft leading chunk, the
    unvoiced '抓') survives iff the user starts speaking at/after the tail ends
    (``react_chunks >= tail_chunks``). Mirrors the live-SLV
    ``test_done_tone_tail_vs_onset`` sweep, deterministically."""
    onset_mark = pcm(800)  # RMS ≈ 0.024, just above gate-open — the soft onset
    script = [SILENCE] * react_chunks + [onset_mark] + [LOUD] * 4
    app, sent, clock = build_pump(monkeypatch, script)
    app._local_output_mic_suppress_until = clock.window_first(tail_chunks)
    await app._mic_pump()

    reals = real_chunks(sent)
    got = onset_mark in reals
    assert got == onset_survives, (
        f"tail={tail_chunks} react={react_chunks}: expected onset_survives="
        f"{onset_survives}, got {got}. reals={[r[:2] for r in reals]}"
    )


@pytest.mark.asyncio
async def test_no_suppression_keeps_onset(monkeypatch):
    """Control: with no suppression window the soft onset forwards — proving the
    drop above is the suppression window, not the gate rejecting a quiet onset
    (the pre-roll rescues it, T1)."""
    onset_mark = pcm(800)
    script = [onset_mark] + [LOUD] * 4
    app, sent, _ = build_pump(monkeypatch, script)  # suppress_until = 0
    await app._mic_pump()

    assert onset_mark in real_chunks(sent), (
        "without suppression the soft onset must reach SLV (pre-roll recovery)"
    )
