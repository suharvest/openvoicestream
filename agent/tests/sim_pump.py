"""Deterministic mic-pump simulator for voice timing-mechanics tests.

The live-SLV e2e suite (``tests/e2e/test_rebot_voice_capture.py``) proves the
END-TO-END ASR result, but it is slow, flaky, and needs orin-nx online — wrong
tool for asserting the *agent-side* timing logic. This module drives
``BaseApp._mic_pump`` directly with:

  * a scripted chunk timeline (list of PCM chunks),
  * a VIRTUAL clock that only advances when a chunk is yielded — so suppress /
    wake-skip / gate-hangover windows line up deterministically with chunk
    indices instead of wall-clock,
  * a fake SLV whose ``is_reconnecting()`` can be toggled per-chunk,

and captures the exact byte stream forwarded to SLV. This reproduces the
reconnect-window (T2) and notification-tone suppression (T3/T4) timing bugs
with ZERO network and full determinism — the same substrate as
``test_mic_pump_preroll.py`` (T1), generalised.

The pump checks, in order, BEFORE the energy gate (app_base.py:1421-1463):
  SLEEPING/SPEAKING drop → is_reconnecting → wake-skip → output-suppress →
  advertise-ready → energy gate.
Each gate ``continue``s and clears pre-roll, so audio spoken inside any window
never reaches ASR. These helpers let a test put a window over chosen chunks and
assert which audio survived.
"""
from __future__ import annotations

import struct
from typing import Iterable

from ovs_agent import Config
from ovs_agent.app_base import BaseApp
from ovs_agent.state import ConvState

CHUNK_SAMPLES = 1600  # 100 ms @ 16 kHz mono int16


def pcm(value: int, n: int = CHUNK_SAMPLES) -> bytes:
    """A 100 ms chunk of a constant sample value. RMS = |value| / 32768."""
    return struct.pack("<h", value) * n


SILENCE = pcm(0)
# RMS thresholds for the default gate below: open 0.020 → ~655, close 0.010 →
# ~328. A value >= ~700 opens the gate; 0 keeps it shut.
LOUD = pcm(2000)   # RMS ≈ 0.061 — opens the gate


def is_zero(chunk: bytes) -> bool:
    return set(chunk) == {0}


def is_real(chunk: bytes) -> bool:
    """A forwarded chunk that carries actual (non-zero) microphone audio."""
    return not is_zero(chunk)


class VirtualClock:
    """``monotonic()`` that advances by one chunk only when the audio generator
    yields. Lets a test express "suppress the first N chunks" exactly."""

    def __init__(self, chunk_ms: int = 100) -> None:
        self.base = 1000.0  # arbitrary non-zero epoch
        self.step = chunk_ms / 1000.0
        self.t = self.base

    def monotonic(self) -> float:
        return self.t

    def advance(self) -> None:
        self.t += self.step

    def window_first(self, n_chunks: int) -> float:
        """A ``*_until`` deadline that covers chunk indices 0..n-1 and releases
        at chunk n. The pump reads ``monotonic()`` AFTER the per-chunk advance,
        so chunk i sees ``base + step*(i+1)``; half-step past chunk n-1 covers
        exactly the first n chunks."""
        return self.base + self.step * (n_chunks + 0.5)


class _FakeTime:
    """Shim swapped into ``app_base.time``: virtual ``monotonic``, everything
    else delegates to the real ``time`` module."""

    def __init__(self, clock: VirtualClock) -> None:
        import time as _t

        self._clock = clock
        self._real = _t

    @property
    def monotonic(self):
        return self._clock.monotonic

    def __getattr__(self, name):
        return getattr(self._real, name)


class _ScriptAudio:
    """Minimal AudioIO surface that yields a fixed chunk list, advancing the
    virtual clock by one chunk_ms before each yield."""

    def __init__(self, script: Iterable[bytes], clock: VirtualClock, chunk_ms: int) -> None:
        self._script = list(script)
        self._clock = clock
        self.chunk_ms = chunk_ms

    async def start_capture(self):
        for chunk in self._script:
            self._clock.advance()
            yield chunk


class FakeSLV:
    """``is_reconnecting()`` returns values from a per-call schedule (one entry
    per mic chunk). Past the schedule it returns False. The pump calls it once
    per chunk (app_base.py:1436), so the schedule aligns 1:1 with the script."""

    def __init__(self, reconnecting_schedule: Iterable[bool] | None = None) -> None:
        self._sched = list(reconnecting_schedule or [])
        self._calls = 0
        self.reconnect_calls = 0

    def is_reconnecting(self) -> bool:
        i = self._calls
        self._calls += 1
        return self._sched[i] if i < len(self._sched) else False


def build_pump(
    monkeypatch,
    script,
    *,
    reconnecting_schedule=None,
    makeup_gain: float = 1.0,
    gate_open: float = 0.020,
    gate_close: float = 0.010,
    gate_hangover_ms: float = 250.0,
    state: ConvState = ConvState.LISTENING,
    chunk_ms: int = 100,
):
    """Wire a bare ``BaseApp`` for the server-VAD energy-gate path (the
    production reBot voice path: ``client_vad_backend='off'``), feed it
    ``script``, and capture every forwarded chunk.

    Returns ``(app, sent, clock)``. ``makeup_gain`` defaults to 1.0 so forwarded
    real chunks are byte-equal to the input — making value-based assertions
    (e.g. "the suppressed onset never reached SLV") exact.
    """
    import ovs_agent.app_base as ab

    clock = VirtualClock(chunk_ms)
    monkeypatch.setattr(ab, "time", _FakeTime(clock))

    app = BaseApp.__new__(BaseApp)
    app.config = Config(
        system_prompt="SYS",
        energy_gate_enabled=True,
        energy_gate_open_rms=gate_open,
        energy_gate_close_rms=gate_close,
        energy_gate_hangover_ms=gate_hangover_ms,
        mic_makeup_gain=makeup_gain,
        mic_drop_while_speaking=False,
        gate_drive_eos=False,
    )
    app.audio = _ScriptAudio(script, clock, chunk_ms)
    app.slv = FakeSLV(reconnecting_schedule)
    app._client_vad = None
    app._state = state
    app._advertise_ready = None
    app._vad_state = "idle"
    app._last_mic_chunk_ts = 0.0
    app._wake_mic_skip_until = 0.0
    app._local_output_mic_suppress_until = 0.0
    app._schedule_mic_rms_broadcast = lambda d: False  # type: ignore[assignment]

    sent: list[bytes] = []

    async def _send(pcm_bytes):
        sent.append(pcm_bytes)

    app._send_audio_nonblocking = _send  # type: ignore[assignment]
    return app, sent, clock


def real_chunks(sent: list[bytes]) -> list[bytes]:
    return [c for c in sent if is_real(c)]


__all__ = [
    "CHUNK_SAMPLES",
    "SILENCE",
    "LOUD",
    "pcm",
    "is_zero",
    "is_real",
    "real_chunks",
    "VirtualClock",
    "FakeSLV",
    "build_pump",
]
