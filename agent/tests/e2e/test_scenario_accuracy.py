"""Voice timing-scenario ASR-accuracy sweep against the live SLV (Qwen3-ASR).

Beyond the pass/fail regressions, this is an EXPLORATORY measurement harness:
it positions a command utterance at the realistic timing offsets that occur
around a reBot tool call and reports how the live ASR transcribes it, so we can
*discover* where recognition degrades (not just confirm known bugs).

Scenarios (production reBot voice params: client_vad='off' + energy gate +
wake_word + mic_drop_while_speaking):

  baseline            speak well after wake — the reference accuracy.
  after_wake_fast     speak 150ms after wake — onset inside the wake-tone
                      suppression (250ms). ["唤醒后马上说"]
  donetone_react50    a completion tone fired (tail 120ms); user speaks 50ms
                      later — onset inside the done-tone suppression.
                      ["工具调用结束后马上说话"]
  donetone_react300   same tone, user waits 300ms (> tail) — control.
  during_reply        a 2nd command spoken 0.4s into the 1st turn's spoken
                      reply (agent SPEAKING/THINKING). ["工具调用中有声音 —
                      语音反馈期"] — expected DROPPED (echo gate).
  after_action_idle   a 2nd command spoken 2.5s later, after the reply ended
                      and the agent is back to IDLE (the silent action body of
                      a tool call). ["工具调用中有声音 — 静默动作期"] — should
                      be captured.
  soft_onset          a faded-in command whose onset sits below the gate — the
                      energy-gate pre-roll must recover it.

Run: pytest tests/e2e/test_scenario_accuracy.py -v -s   (needs orin-nx live)
The VALUE is the printed matrix; the test only sanity-asserts the baseline so
it doesn't flap on the exploratory cells.

FINDING (2026-06-14) — RESOLVED by server commit 9b084a5:
  This harness's first run surfaced after_action_idle = 0/3 — a 2ND consecutive
  command, spoken after the 1st turn's reply ended (agent back to IDLE, TTS
  done), was NOT transcribed. The gap diag (test_consecutive_gap_diag) showed
  the 2nd command's audio reached the mic pump (loud on_mic_rms) yet the SLV
  returned 0 partials/final; a forced reconnect recovered it
  (test_reconnect_recovers_2nd_turn) → per-session ASR state, not reset per
  utterance. Root cause: in no-VAD mode the server's lazy ASR-stream open was
  gated by the one-shot `asr_started_once` latch (server/main.py), so on a
  persistent multi_utterance session only the 1st utterance opened a stream.
  Fix: re-open per utterance in multi_utterance. Post-fix after_action_idle is
  3/3 and test_multi_turn passes.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO
from .test_rebot_voice_capture import _rebot_voice_config, _wake, _last_user_text

# Commands measured per scenario (Qwen3-TTS corpus; 挥手 excluded — known bad
# fixture). (wav stem, expected substring).
_CMDS = [
    ("tts_q_grab_box", "盒子"),
    ("tts_q_put_back", "放回"),
    ("tts_q_grab_cup", "杯子"),
]
_WARMUP = "tts_q_grab_box"  # first turn for the two-utterance scenarios


async def _wait_utterance_count(probe, n: int, timeout: float = 25.0) -> bool:
    """True once at least n on_user_utterance events have been observed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        c = sum(1 for e in probe.events if e.get("event") == "on_user_utterance")
        if c >= n:
            return True
        await asyncio.sleep(0.1)
    return False


async def _inject_read(app, probe, audio, wav, nth=1, timeout=25.0):
    audio.inject(WAV_DIR / f"{wav}.wav")
    ok = await _wait_utterance_count(probe, nth, timeout)
    return _last_user_text(app) if ok else None


# ── scenario bodies: each returns the recognised text of the MEASURED command
#    (or None if no final fired / it was dropped) ──────────────────────────

async def _scn_baseline(app, probe, audio, wav):
    await asyncio.sleep(0.6)
    return await _inject_read(app, probe, audio, wav)


async def _scn_after_wake_fast(app, probe, audio, wav):
    await asyncio.sleep(0.15)  # inside the 250ms wake-tone suppression
    return await _inject_read(app, probe, audio, wav)


async def _scn_donetone(react_s):
    async def _body(app, probe, audio, wav):
        await asyncio.sleep(0.6)  # clear the wake-tone suppression first
        app._local_output_mic_suppress_until = time.monotonic() + 0.120  # done-tone tail
        await asyncio.sleep(react_s)
        return await _inject_read(app, probe, audio, wav)
    return _body


async def _scn_during_reply(app, probe, audio, wav):
    """Speak the next command right as the 1st reply starts speaking. reBot
    replies are short, so the command lands as the reply ends and IS captured
    (responsive) — expect_capture=True. The pure echo gate (a command fully
    inside a sustained SPEAKING window is dropped) is covered deterministically
    by test_rebot_voice_capture::test_command_during_speaking_is_dropped."""
    await _inject_read(app, probe, audio, _WARMUP, nth=1)  # turn 1
    try:
        await probe.wait_state("speaking", timeout=12)
    except (TimeoutError, AssertionError):
        pass  # reply too fast to catch SPEAKING — inject anyway
    audio.inject(WAV_DIR / f"{wav}.wav")
    ok = await _wait_utterance_count(probe, 2, timeout=12)
    return _last_user_text(app) if ok else None


async def _scn_after_action_idle(app, probe, audio, wav):
    await _inject_read(app, probe, audio, _WARMUP, nth=1)  # turn 1
    await asyncio.sleep(2.5)  # reply over, agent back to IDLE
    audio.inject(WAV_DIR / f"{wav}.wav")
    ok = await _wait_utterance_count(probe, 2, timeout=25)
    return _last_user_text(app) if ok else None


async def _scn_soft_onset(app, probe, audio, wav):
    await asyncio.sleep(0.6)
    # Fixed fade-in fixture (say corpus) regardless of `wav` — its onset sits
    # below the gate, exercising the pre-roll recovery.
    audio.inject(WAV_DIR / "cmd_grab_box_fadein.wav")
    ok = await _wait_utterance_count(probe, 1, 25)
    return _last_user_text(app) if ok else None


def _scn_nth_consecutive(n):
    """Run n consecutive commands on ONE session and measure the n-th — catches
    ASR quality degradation over many utterances on a single streaming worker
    (a documented risk) and confirms the multi_utterance fix holds past turn 2."""
    async def _body(app, probe, audio, wav):
        for i in range(n - 1):
            await _inject_read(app, probe, audio, _WARMUP, nth=i + 1)
            await asyncio.sleep(2.5)  # each reply finishes → IDLE
        audio.inject(WAV_DIR / f"{wav}.wav")
        ok = await _wait_utterance_count(probe, n, timeout=25)
        return _last_user_text(app) if ok else None
    return _body


async def _scn_after_rewake(app, probe, audio, wav):
    """command → force SLEEPING → re-wake → command. Exercises the wake-time
    health/idle reconnect path for the post-rewake command."""
    await _inject_read(app, probe, audio, _WARMUP, nth=1)  # turn 1
    await asyncio.sleep(2.5)
    await app.sleep()                       # force SLEEPING
    await asyncio.sleep(0.5)
    await _wake(app, app.config)            # re-wake
    await probe.wait_event("on_wake", timeout=8)
    await asyncio.sleep(0.5)
    audio.inject(WAV_DIR / f"{wav}.wav")
    ok = await _wait_utterance_count(probe, 2, timeout=25)
    return _last_user_text(app) if ok else None


# (name, body, expect_capture) — expect_capture=False marks a scenario where a
# drop is the CORRECT behaviour (so "accuracy" there means "correctly dropped").
def _scenarios():
    return [
        ("baseline", _scn_baseline, True),
        ("after_wake_fast", _scn_after_wake_fast, True),
        ("donetone_react50", None, True),   # filled below (async factory)
        ("donetone_react300", None, True),
        ("during_reply", _scn_during_reply, True),
        ("after_action_idle", _scn_after_action_idle, True),
        ("turn3_consecutive", None, True),   # filled below
        ("turn4_consecutive", None, True),
        ("after_rewake", _scn_after_rewake, True),
        ("soft_onset", _scn_soft_onset, True),
    ]


@pytest.mark.asyncio
async def test_scenario_accuracy_sweep(test_config):
    cfg = _rebot_voice_config(test_config)
    dt50 = await _scn_donetone(0.05)
    dt300 = await _scn_donetone(0.30)
    bodies = {
        "donetone_react50": dt50,
        "donetone_react300": dt300,
        "turn3_consecutive": _scn_nth_consecutive(3),
        "turn4_consecutive": _scn_nth_consecutive(4),
    }

    rows = []  # (scenario, expect_capture, [(cmd, expect, got, ok)])
    for name, body, expect_capture in _scenarios():
        fn = bodies.get(name) or body
        # Bound runtime: the deep-turn scenarios pay for n turns per trial, so
        # measure fewer commands. soft_onset measures one fixed fixture.
        if name == "soft_onset":
            cmds = [("cmd_grab_box_fadein", "盒子")]
        elif name in ("turn3_consecutive", "turn4_consecutive"):
            cmds = _CMDS[:1]
        elif name == "after_rewake":
            cmds = _CMDS[:2]
        else:
            cmds = _CMDS
        cells = []
        for wav, expect in cmds:
            audio = ScriptedAudioIO([])
            try:
                async with run_agent(cfg, audio) as (app, probe):
                    await _wake(app, cfg)
                    await probe.wait_event("on_wake", timeout=5)
                    got = await fn(app, probe, audio, wav if name != "soft_onset" else wav)
            except Exception as e:  # keep the sweep going; record the failure
                got = f"<error: {type(e).__name__}>"
            if expect_capture:
                ok = bool(got) and expect in str(got)
            else:
                ok = not got  # correct == dropped
            cells.append((wav, expect, got, ok))
            print(f"[{name:18s}] {wav:20s} expect={expect!r:6} got={got!r}  {'OK' if ok else 'XX'}")
        rows.append((name, expect_capture, cells))

    # ── matrix summary ──
    print("\n================ SCENARIO ACCURACY MATRIX (live Qwen3-ASR) ================")
    print(f"{'scenario':20s} {'mode':8s} {'accuracy':10s} detail")
    for name, expect_capture, cells in rows:
        n_ok = sum(1 for *_, ok in cells if ok)
        mode = "capture" if expect_capture else "drop"
        detail = ", ".join(f"{w.replace('tts_q_','').replace('cmd_','')}={'OK' if ok else 'XX'}"
                           for w, _, _, ok in cells)
        print(f"{name:20s} {mode:8s} {n_ok}/{len(cells):<8} {detail}")
    print("===========================================================================\n")

    # Sanity floor only — baseline must mostly work, else the harness/SLV is
    # broken and the matrix is meaningless. Exploratory cells are not asserted.
    base = next(c for n, _, c in rows if n == "baseline")
    base_ok = sum(1 for *_, ok in base if ok)
    assert base_ok >= 2, f"baseline accuracy too low ({base_ok}/{len(base)}) — SLV/harness issue"


@pytest.mark.asyncio
@pytest.mark.parametrize("gap_s", [2.5, 5.0, 8.0])
async def test_consecutive_gap_diag(test_config, gap_s):
    """Diagnostic: disambiguate the after_action_idle=0/3 finding. Inject a 2nd
    command `gap_s` after the 1st turn and log the agent state + playback at the
    injection instant. If the 2nd command only lands once state==IDLE & not
    playing, the sweep's 0/3 was a too-short-gap artifact (reply still SPEAKING
    → mic_drop). If it drops even when IDLE, it's a real consecutive-turn bug."""
    cfg = _rebot_voice_config(test_config)
    audio = ScriptedAudioIO([])
    async with run_agent(cfg, audio) as (app, probe):
        await _wake(app, cfg)
        await probe.wait_event("on_wake", timeout=5)
        await _inject_read(app, probe, audio, _WARMUP, nth=1)  # turn 1
        await asyncio.sleep(gap_s)
        st = getattr(getattr(app, "_state", None), "value", getattr(app, "_state", None))
        playing = getattr(audio, "is_playing", None)
        before = len(probe.events)
        audio.inject(WAV_DIR / "tts_q_put_back.wav")
        ok = await _wait_utterance_count(probe, 2, timeout=25)
        got = _last_user_text(app) if ok else None
        # Event histogram after cmd2 injection — localises the drop.
        from collections import Counter
        new_events = probe.events[before:]
        hist = Counter(e.get("event") for e in new_events)
        mic_rms = [round(float((e.get("data") or {}).get("rms", 0)), 4)
                   for e in new_events if e.get("event") == "on_mic_rms"]
        loud = [r for r in mic_rms if r and r > 0.02]
        partials = [e for e in new_events if "partial" in str(e.get("event"))]
        print(f"\n[gap={gap_s}s] at-inject state={st} playing={playing} "
              f"-> captured={ok} got={got!r}")
        print(f"[gap={gap_s}s] events-after-cmd2={dict(hist)}")
        print(f"[gap={gap_s}s] mic_rms count={len(mic_rms)} loud(>0.02)={len(loud)} "
              f"max={max(mic_rms) if mic_rms else None} partials={len(partials)}")


@pytest.mark.asyncio
async def test_reconnect_recovers_2nd_turn(test_config):
    """Diagnostic for the after_action_idle=0/3 root cause: does forcing a fresh
    SLV WS between turns recover the 2nd utterance? If YES, the trt_edgellm ASR
    decode state is per-WS-session and not reset per-utterance (a new WS clears
    it) — corroborates the multi_utterance root cause and explains the field
    observation that the 2nd command 'sometimes' works (a wake/idle reconnect
    refreshes the ASR session). Run alongside test_consecutive_gap_diag (which
    shows it FAILS without a reconnect)."""
    cfg = _rebot_voice_config(test_config)
    audio = ScriptedAudioIO([])
    async with run_agent(cfg, audio) as (app, probe):
        await _wake(app, cfg)
        await probe.wait_event("on_wake", timeout=5)
        t1 = await _inject_read(app, probe, audio, _WARMUP, nth=1)  # turn 1
        await asyncio.sleep(3.0)  # turn-1 reply finishes → IDLE
        # Force a fresh WS / ASR session before turn 2.
        await asyncio.wait_for(app.slv.reconnect(), timeout=6.0)
        await asyncio.sleep(0.5)
        audio.inject(WAV_DIR / "tts_q_put_back.wav")
        ok = await _wait_utterance_count(probe, 2, timeout=25)
        t2 = _last_user_text(app) if ok else None
        print(f"\n[reconnect-between-turns] t1={t1!r} t2={t2!r} captured2={ok} "
              f"(if captured2=True a fresh WS heals the 2nd utterance → per-session "
              f"ASR state not reset per-utterance)")
