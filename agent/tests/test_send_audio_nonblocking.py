"""Regression: mic_pump must NOT block indefinitely on slv.send_audio
when SLV is mid-reconnect.

Before the fix, an SLV WS close (keepalive ping timeout) followed by a
slow ws_connect (network blip / opening-handshake timeout) would have
every mic chunk park on SLVClient._send_lock for the full reconnect
window. mic_pump stops draining its input queue, sounddevice's callback
thread floods call_soon_threadsafe with PCM, and the log fills with
``mic queue full -- dropping chunk`` for the entire outage — the user
sees "agent feels dead, mic doesn't work" until reconnect finally lands.

The fix: cap send_audio with a 0.5 s timeout per chunk and drop on
TimeoutError so mic_pump keeps draining.
"""
from __future__ import annotations

import asyncio
import pytest

from ovs_agent.audio_io import AudioIO
from ovs_agent.app_base import BaseApp


class _SlowSLV:
    """send_audio() that blocks until we let it through — simulates an
    in-progress reconnect holding _send_lock."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.sent: list[bytes] = []

    async def send_audio(self, pcm: bytes) -> None:
        await asyncio.sleep(self.delay)
        self.sent.append(pcm)


@pytest.mark.asyncio
async def test_send_audio_nonblocking_returns_within_timeout():
    """If SLV send_audio is slow (reconnect in progress), the helper
    must return within the wait_for ceiling (2.0s) and not block the
    mic pump indefinitely. (Ceiling was raised from 0.5s → 2.0s as part
    of race #3 fix; the reconnecting-gate short-circuit handles the
    common reconnect case without consuming the full 2.0s budget.)"""
    app = BaseApp.__new__(BaseApp)
    app.slv = _SlowSLV(delay=5.0)  # would block 5 s

    start = asyncio.get_running_loop().time()
    await app._send_audio_nonblocking(b"\x00" * 100)
    elapsed = asyncio.get_running_loop().time() - start

    # Should bail out around 2.0 s (the helper's wait_for ceiling),
    # nowhere near the 5 s slow-send delay.
    assert elapsed < 2.5, f"send_audio_nonblocking blocked {elapsed:.2f}s"
    # The slow send is still pending in the background; that's fine, we
    # just must not have waited for it.
    assert app.slv.sent == []


@pytest.mark.asyncio
async def test_send_audio_nonblocking_passes_through_fast_send():
    """Normal-case: a fast send_audio completes without timing out."""
    app = BaseApp.__new__(BaseApp)
    app.slv = _SlowSLV(delay=0.01)
    await app._send_audio_nonblocking(b"\x01" * 50)
    assert app.slv.sent == [b"\x01" * 50]


@pytest.mark.asyncio
async def test_send_audio_nonblocking_swallows_exceptions():
    """A flaky send_audio that raises must NOT propagate into mic_pump
    (which would crash the whole capture loop)."""
    class _Bad:
        async def send_audio(self, pcm: bytes) -> None:
            raise RuntimeError("boom")

    app = BaseApp.__new__(BaseApp)
    app.slv = _Bad()
    # Must not raise.
    await app._send_audio_nonblocking(b"\x00")


@pytest.mark.asyncio
async def test_mic_rms_broadcast_is_fire_and_forget():
    """A slow dashboard/browser hook must not backpressure mic_pump.

    Before this regression fix, mic_pump awaited ``_broadcast("on_mic_rms")``.
    If the dashboard WS send stalled, VAD stopped seeing new mic chunks while
    TTS was playing, so the second barge-in could never fire.
    """

    class _SlowPlugin:
        name = "slow"

        def __init__(self) -> None:
            self.calls = 0

        async def on_mic_rms(self, data: dict) -> None:
            self.calls += 1
            await asyncio.sleep(5)

    app = BaseApp.__new__(BaseApp)
    plugin = _SlowPlugin()
    app.plugins = [plugin]
    app._mic_rms_broadcast_task = None

    start = asyncio.get_running_loop().time()
    assert app._schedule_mic_rms_broadcast({"rms": 0.1}) is True
    assert app._schedule_mic_rms_broadcast({"rms": 0.2}) is False
    elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 0.05
    assert app._mic_rms_broadcast_task is not None
    app._mic_rms_broadcast_task.cancel()
    try:
        await app._mic_rms_broadcast_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_playback_uses_callback_buffer_and_stop_clears_it():
    """Playback must not depend on blocking RawOutputStream.write.

    The PortAudio callback pulls from an in-memory buffer; barge-in clears
    that buffer so the current utterance stops within one callback period.
    """

    audio = AudioIO.__new__(AudioIO)
    audio._playback_response_id = None
    audio._discard_playback = False
    audio._is_playing = False
    audio._output_stream = object()
    audio.output_sr = 24000
    audio._source_sr = 24000

    await audio.play(b"\x01\x02\x03\x04")
    assert audio.is_playing is True
    assert bytes(audio._playback_buffer) == b"\x01\x02\x03\x04"

    out = bytearray(6)
    audio._output_callback(out, frames=3, time_info=None, status=None)
    assert bytes(out) == b"\x01\x02\x03\x04\x00\x00"
    assert bytes(audio._playback_buffer) == b""

    await audio.play(b"\x05\x06")
    await audio.stop_playback()
    assert audio.is_playing is False
    assert audio._discard_playback is True
    assert bytes(audio._playback_buffer) == b""


def test_is_playing_tracks_local_buffer_after_remote_tts_done():
    """TTSDone means SLV stopped sending, not that the speaker is silent.

    Barge-in must still fire while locally buffered PCM is audible.
    """

    audio = AudioIO.__new__(AudioIO)
    audio._playback_response_id = None
    audio._discard_playback = False
    audio._is_playing = True
    audio._playback_buffer = bytearray(b"\x01\x02\x03\x04")
    audio._playback_lock = __import__("threading").Lock()

    audio.mark_playback_done()
    assert audio.is_playing is True

    out = bytearray(4)
    audio._output_callback(out, frames=2, time_info=None, status=None)
    assert audio.is_playing is False
