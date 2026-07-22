"""Unit tests for the env-gated reBot remote audio-inject (debug) path.

Covers the pure logic (WAV decode + paced PCM feed into the capture queue);
the full server-loop arm drive is validated on-device, not here.
"""
from __future__ import annotations

import asyncio
import io
import wave

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.dashboard_plugin import (
    _wav_bytes_to_pcm16_mono,
)


def _make_wav(sr: int, seconds: float, freq: float = 220.0, nch: int = 1) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    arr = mono if nch == 1 else np.repeat(mono[:, None], nch, axis=1).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(arr.tobytes())
    return buf.getvalue()


def test_wav_decode_same_rate_mono():
    pcm = _wav_bytes_to_pcm16_mono(_make_wav(16000, 0.5), target_sr=16000)
    assert len(pcm) == 8000 * 2  # 0.5s @16k int16 = 8000 samples × 2 bytes


def test_wav_decode_resamples_and_downmixes():
    # 48k stereo → 16k mono. Length should be ~16k * 0.5 samples.
    pcm = _wav_bytes_to_pcm16_mono(_make_wav(48000, 0.5, nch=2), target_sr=16000)
    n_samples = len(pcm) // 2
    assert abs(n_samples - 8000) <= 2, n_samples


@pytest.mark.asyncio
async def test_inject_pcm_feeds_capture_queue_paced():
    """inject_pcm puts frame-aligned chunks into _in_queue (the same queue the
    mic callback feeds) so the existing pump forwards them unchanged."""
    from ovs_agent.audio_io import AudioIO

    a = AudioIO.__new__(AudioIO)
    a.input_sr = 16000
    a._in_queue = asyncio.Queue(maxsize=512)

    pcm = b"\x01\x02" * 16000  # 1s of 16k int16
    drained: list[bytes] = []

    async def _drain():
        while True:
            drained.append(await a._in_queue.get())

    dt = asyncio.create_task(_drain())
    n = await a.inject_pcm(pcm, chunk_ms=64.0)
    await asyncio.sleep(0.05)
    dt.cancel()

    assert n == len(pcm)
    assert b"".join(drained) == pcm  # byte-exact, nothing lost/reordered
    # frame-aligned (even byte counts), paced into multiple chunks.
    assert len(drained) > 1
    assert all(len(c) % 2 == 0 for c in drained)


@pytest.mark.asyncio
async def test_inject_pcm_raises_without_capture():
    from ovs_agent.audio_io import AudioIO

    a = AudioIO.__new__(AudioIO)
    a.input_sr = 16000
    a._in_queue = None
    with pytest.raises(RuntimeError):
        await a.inject_pcm(b"\x00\x00" * 100)
