"""Mic-less audio injection: feed a WAV clip in as if it had been spoken.

This is the only faithful way to exercise a **server-loop** deployment without a
microphone. Injecting a synthetic ``ASRFinal`` instead (see the debug dashboard's
``/api/control/inject_user_text``) runs the *client-side* dialogue runner — but
in server-loop the LLM and the tool dispatch live on the server, so that path
proves nothing about what actually ships. Audio is the real entrypoint.

Lifted out of the arm app's dashboard plugin so every app gets the same
implementation: the three bypasses below were each found on hardware, and a
second copy would drift away from them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ~64 ms frames, matching what a real mic pump delivers.
FRAME_S = 0.064
# Trailing silence so server-side VAD sees an endpoint even when the clip ends
# abruptly on the final syllable.
TRAILING_SILENCE_S = 0.3


def wav_bytes_to_pcm16_mono(data: bytes, target_sr: int = 16000) -> bytes:
    """Decode WAV bytes → mono int16 PCM at ``target_sr`` (linear resample)."""
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8
    elif sw == 4:
        arr = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
    else:
        arr = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1).astype(np.int16)
    if sr != target_sr and arr.size:
        n_out = int(len(arr) * target_sr / sr)
        x0 = np.linspace(0, 1, len(arr), endpoint=False)
        x1 = np.linspace(0, 1, n_out, endpoint=False)
        arr = np.interp(x1, x0, arr.astype(np.float32)).astype(np.int16)
    return arr.tobytes()


async def inject_wav(app: Any, data: bytes, *, wake: bool = True) -> dict:
    """Feed ``data`` (WAV bytes) to the SLV as one spoken utterance.

    Returns a JSON-able result dict. Raises ``ValueError`` for a bad clip and
    ``RuntimeError`` when the agent has no usable audio path.

    Three things had to be bypassed to make injected clips arrive intact (all
    observed on hardware, 2026-06-14):

    1. the energy-gated mic pump discarded low-energy syllables and the onset
       → send straight via ``slv.send_audio``, not through the mic queue;
    2. ``wake()`` can trigger an SLV reconnect (idle > 30 s) and PCM fed before
       the new ``/v2v`` stream accepts is lost → wait for the WS to be ready;
    3. the real mic pump runs concurrently on the SAME WS and its frames
       interleave with the injection (the SLV transcribed ambient room audio
       instead of the clip) → set ``app._injecting`` so mic forwarding is
       suppressed for the window.

    Real speech hits none of this.
    """
    audio = getattr(app, "audio", None)
    slv = getattr(app, "slv", None)
    if slv is None or not hasattr(slv, "send_audio"):
        raise RuntimeError("agent has no SLV audio path")
    if not data:
        raise ValueError("empty body; POST raw WAV bytes")

    sr = int(getattr(audio, "input_sr", 16000) or 16000)
    pcm = wav_bytes_to_pcm16_mono(data, target_sr=sr)
    logger.warning(
        "inject_wav: feeding %d PCM bytes (%.2fs @ %dHz) straight to SLV "
        "(bypassing energy gate + mic pump)",
        len(pcm),
        len(pcm) / 2 / sr,
        sr,
    )

    if wake:
        try:
            await app.wake(source="inject_wav")
        except Exception:
            logger.debug("inject_wav: wake failed", exc_info=True)
        await asyncio.sleep(0.5)  # let any wake tone finish (drop_while_speaking)

    for _ in range(60):  # up to ~6s for a possibly reconnecting WS
        try:
            if not slv.is_reconnecting() and slv.is_healthy():
                break
        except Exception:
            break
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.3)  # settle margin once the stream is ready

    step = max(2, int(sr * FRAME_S) * 2)  # even-aligned so int16 frames stay whole
    app._injecting = True
    try:
        for i in range(0, len(pcm), step):
            await slv.send_audio(pcm[i : i + step])
            await asyncio.sleep(FRAME_S)
        await slv.send_audio(b"\x00\x00" * int(sr * TRAILING_SILENCE_S))
        await asyncio.sleep(0.2)
        # Force the finalize so the clip lands regardless of VAD / endpoint
        # config (client_vad_drive_eos, vad: none, ...).
        send_eos = getattr(app, "send_asr_eos_once", None)
        if callable(send_eos):
            try:
                await send_eos()
            except Exception:
                logger.debug("inject_wav: asr_eos failed", exc_info=True)
    finally:
        app._injecting = False

    return {"ok": True, "pcm_bytes": len(pcm), "sr": sr, "via": "slv_direct"}


__all__ = ["inject_wav", "wav_bytes_to_pcm16_mono", "FRAME_S"]
