"""Energy-gate pre-roll: the command ONSET must survive the gate (2026-06-13).

Real-machine bug: with ``client_vad_backend: off`` the mic pump runs the
energy-gate path. The gate sent zero-fill until RMS crossed ``open_rms``, so
the low-energy onset of a command (the unvoiced '抓' in '抓盒子') reached the
server ASR as silence — the final came back as just the loud tail ('盒子').

The fix buffers real chunks while the gate is closed and replays that pre-roll
the instant the gate opens, so ASR sees the whole word. These tests drive
``_mic_pump`` directly with crafted RMS sequences and assert on the exact byte
stream forwarded to SLV.
"""
from __future__ import annotations

import struct

import pytest

from ovs_agent import Config
from ovs_agent.app_base import BaseApp
from ovs_agent.state import ConvState


def _pcm(int16_value: int, n: int = 1600) -> bytes:
    """A 100ms@16k mono chunk of a constant sample value (RMS = |value|/32768)."""
    return struct.pack("<h", int16_value) * n


# RMS of a constant-|v| chunk is |v|/32768. gate_open 0.020 → ~655; gate_close
# 0.010 → ~328. Pick onset BELOW open (so the gate is shut during the onset)
# but clearly non-zero, and loud well ABOVE open.
_SILENCE = _pcm(0)
_ONSET = _pcm(393)   # RMS ≈ 0.012  (< open 0.020): the gate stays CLOSED here
_LOUD = _pcm(2000)   # RMS ≈ 0.061  (> open): the gate OPENS


def _gate_app(script, *, makeup_gain: float = 1.0):
    """A BaseApp wired for the server-VAD energy-gate path, fed `script`
    (a list of PCM chunks), capturing every forwarded chunk."""
    app = BaseApp.__new__(BaseApp)
    app.config = Config(
        system_prompt="SYS",
        energy_gate_enabled=True,
        energy_gate_open_rms=0.020,
        energy_gate_close_rms=0.010,
        energy_gate_hangover_ms=250,
        mic_makeup_gain=makeup_gain,
        mic_drop_while_speaking=False,
        gate_drive_eos=False,
    )

    class _Audio:
        chunk_ms = 100

        async def start_capture(self):
            for c in script:
                yield c

    class _SLV:
        def is_reconnecting(self) -> bool:
            return False

    app.audio = _Audio()
    app.slv = _SLV()
    app._client_vad = None
    app._state = ConvState.LISTENING
    app._advertise_ready = None
    app._vad_state = "idle"
    app._last_mic_chunk_ts = 0.0
    app._wake_mic_skip_until = 0.0
    app._local_output_mic_suppress_until = 0.0
    app._schedule_mic_rms_broadcast = lambda d: False  # type: ignore[assignment]

    sent: list[bytes] = []

    async def _send(pcm):
        sent.append(pcm)

    app._send_audio_nonblocking = _send  # type: ignore[assignment]
    return app, sent


@pytest.mark.asyncio
async def test_preroll_replays_command_onset_when_gate_opens():
    # silence, then a quiet onset (gate shut), then loud (gate opens).
    app, sent = _gate_app([_SILENCE, _ONSET, _ONSET, _LOUD, _LOUD])
    await app._mic_pump()

    # The onset bytes (the quiet '抓') MUST appear in the forwarded stream —
    # replayed from pre-roll at the gate-open edge. Under the old behaviour the
    # onset was zero-filled and never reached ASR.
    assert _ONSET in sent, "command onset was lost — pre-roll did not replay it"
    assert sent.count(_ONSET) == 2, "both buffered onset chunks must replay"

    # Ordering: the replayed onset is forwarded BEFORE the loud chunk that
    # opened the gate (ASR must see 抓 then 盒子, not the reverse).
    first_onset = sent.index(_ONSET)
    first_loud = sent.index(_LOUD)
    assert first_onset < first_loud, "onset must precede the loud chunk in the stream"


@pytest.mark.asyncio
async def test_pure_silence_never_replays_anything_real():
    # A long idle stretch with no speech must forward only zeros (the gate never
    # opens, so the pre-roll is never drained — no stale audio leaks out).
    app, sent = _gate_app([_SILENCE] * 6)
    await app._mic_pump()
    assert all(set(c) == {0} for c in sent), "idle must forward only zero-fill"


@pytest.mark.asyncio
async def test_preroll_onset_carries_makeup_gain():
    # The replayed onset must be amplified by the same makeup gain as the gated
    # stream, otherwise the onset is quieter than the rest of the word.
    app, sent = _gate_app([_ONSET, _LOUD], makeup_gain=3.0)
    await app._mic_pump()
    # 393 * 3 = 1179, clipped/cast back to int16 little-endian.
    amplified = struct.unpack("<h", sent[sent.index(_amplified_onset())][:2])[0]
    assert amplified == 1179, "replayed onset must carry the makeup gain"


def _amplified_onset() -> bytes:
    return struct.pack("<h", 1179) * 1600


@pytest.mark.asyncio
async def test_gate_open_only_drains_once_not_every_loud_chunk():
    # Sustained speech: the pre-roll drains exactly once (at the open edge), not
    # on every subsequent loud chunk — so the onset isn't duplicated mid-word.
    app, sent = _gate_app([_ONSET, _LOUD, _LOUD, _LOUD])
    await app._mic_pump()
    assert sent.count(_ONSET) == 1, "onset must replay once, not per loud chunk"
