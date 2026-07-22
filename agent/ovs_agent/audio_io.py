"""sounddevice-backed mic capture + speaker playback.

Single persistent InputStream + OutputStream; sounddevice callbacks run
on background threads and push into asyncio.Queue via run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncIterator

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError) as _sd_exc:  # pragma: no cover - hardware-dependent
    sd = None  # type: ignore[assignment]
    _SD_IMPORT_ERR: Exception | None = _sd_exc
else:
    _SD_IMPORT_ERR = None

logger = logging.getLogger(__name__)


class AudioIO:
    """Mic in / speaker out, backed by sounddevice."""

    def __init__(
        self,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
        input_sr: int = 16000,
        output_sr: int = 24000,
        chunk_ms: int = 100,
    ) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self.input_sr = input_sr
        self.output_sr = output_sr  # fixed device output rate
        # Source rate of incoming TTS PCM. Defaults to output_sr (no resample).
        # When the TTS model emits a different rate (matcha=16k vs qwen3=24k),
        # ``set_source_sample_rate()`` updates this and ``play()`` resamples
        # rather than tearing down the underlying audio stream — macOS BT
        # devices often silently fail when their sample rate is switched.
        self._source_sr = output_sr
        self.chunk_ms = chunk_ms
        self._chunk_frames = int(input_sr * chunk_ms / 1000)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._in_queue: asyncio.Queue[bytes] | None = None
        self._out_queue: asyncio.Queue[bytes] | None = None
        self._input_stream: "sd.RawInputStream | None" = None
        self._input_callback = None  # set on first start_capture for reopen
        # PortAudio bad-state recovery: consecutive open failures trigger
        # a library terminate+reinit, which clears CoreAudio (macOS)
        # corruption caused by BT disconnect/reconnect or third-party
        # apps stealing the device. Reset after this many failures.
        self._input_open_failures: int = 0
        self._PA_RESET_THRESHOLD: int = 3
        self._output_stream: "sd.RawOutputStream | None" = None
        self._playback_task: asyncio.Task | None = None
        self._playback_buffer = bytearray()
        self._playback_lock = threading.Lock()
        self._is_playing = False
        # When True, play() drops incoming TTS pcm instead of queueing it.
        # Set by stop_playback (barge-in / sleep / stop-intent) so that
        # already-buffered TTS frames SLV keeps streaming over the WS
        # don't resume audible playback after we've silenced the speaker.
        # Cleared by arm_for_next_turn() at the start of the next utterance.
        self._discard_playback = False
        # Actual TTS bytes handed to the output callback for the active
        # response. Used to report conversation.item.truncate at barge-in.
        self._playback_response_id: str | None = None
        self._playback_device_bytes: int = 0
        # Device hot-plug watcher state.
        self._device_watcher_task: asyncio.Task | None = None
        self._device_signature: tuple | None = None
        self._device_watch_interval_s: float = float(
            __import__("os").environ.get("OVS_AUDIO_WATCH_S", "3.0")
        )
        # Auto-detect + exponential backoff for a missing/hot-plugged input
        # device (e.g. a USB reSpeaker that is not enumerated at boot or gets
        # re-plugged). When the configured mic is absent, the agent boots
        # WITHOUT crashing and the watcher keeps retrying — PortAudio is
        # terminate+reinitialized each attempt so a newly-appeared USB device
        # becomes visible (PortAudio caches the device list at init) — with the
        # delay doubling from min to max until the device shows up.
        self._input_reconnect_min_s: float = float(
            __import__("os").environ.get("OVS_AUDIO_RECONNECT_MIN_S", "1.0")
        )
        self._input_reconnect_max_s: float = float(
            __import__("os").environ.get("OVS_AUDIO_RECONNECT_MAX_S", "30.0")
        )
        self._input_reconnect_backoff_s: float = self._input_reconnect_min_s
        # True once start_capture has set up the callback + wants a live mic;
        # gates the watcher's reconnect loop.
        self._input_capture_active: bool = False

    @property
    def is_playing(self) -> bool:
        self._ensure_playback_buffer()
        with self._playback_lock:
            has_buffered_audio = bool(self._playback_buffer)
        return self._is_playing or has_buffered_audio

    # ── capture ─────────────────────────────────────────────────────

    async def start_capture(self) -> AsyncIterator[bytes]:
        if sd is None:  # pragma: no cover - hardware-dependent
            raise RuntimeError(
                f"sounddevice unavailable: {_SD_IMPORT_ERR!r}. Install libportaudio2."
            )
        self._loop = asyncio.get_running_loop()
        self._in_queue = asyncio.Queue(maxsize=64)

        def _cb(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                logger.debug("input status: %s", status)
            buf = bytes(indata)
            try:
                assert self._loop is not None and self._in_queue is not None
                # IMPORTANT: schedule _safe_put on the loop thread, not
                # put_nowait directly -- QueueFull would otherwise be
                # raised on the loop thread where this try/except cannot
                # catch it.
                self._loop.call_soon_threadsafe(self._safe_put, buf)
            except Exception as e:  # pragma: no cover
                logger.warning("mic cb error: %s", e)

        # Capture the callback for device-hot-plug reopen.
        self._input_callback = _cb
        self._input_capture_active = True

        # Resilient initial open: if the configured mic is not present yet
        # (USB reSpeaker not enumerated, mid-replug, etc.) DO NOT crash — boot
        # without it and let the watcher auto-detect + reconnect with
        # exponential backoff the moment it appears.
        try:
            self._open_input_stream()
        except Exception as e:
            logger.warning(
                "mic %r not available at start (%s); booting without it — "
                "will auto-detect with exponential backoff",
                self.input_device, e,
            )
            self._input_stream = None
            self._input_reconnect_backoff_s = self._input_reconnect_min_s
        self._device_signature = self._compute_device_signature()
        if self._device_watcher_task is None or self._device_watcher_task.done():
            self._device_watcher_task = self._loop.create_task(
                self._watch_devices(), name="audio-device-watcher"
            )

        try:
            while True:
                chunk = await self._in_queue.get()
                yield chunk
        finally:
            self._input_capture_active = False
            self._stop_input_stream()
            if self._device_watcher_task is not None:
                self._device_watcher_task.cancel()
                try:
                    await self._device_watcher_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._device_watcher_task = None

    async def inject_pcm(self, pcm: bytes, *, chunk_ms: float = 64.0) -> int:
        """DEBUG: feed raw PCM (``input_sr`` mono int16) into the live capture
        queue as if it came from the mic, paced at ``chunk_ms`` so the energy
        gate / VAD / ASR see a coherent utterance rather than one burst.

        Used by the env-gated reBot inject endpoint for remote, mic-less
        end-to-end testing (server-loop: the chunks are forwarded to SLV by the
        normal mic pump). Returns the number of bytes fed. Raises if capture
        hasn't started (no queue to push into)."""
        q = self._in_queue
        if q is None:
            raise RuntimeError("capture not started; cannot inject_pcm")
        # input_sr samples/s * chunk_ms/1000 s * 2 bytes/sample, frame-aligned.
        step = max(2, int(self.input_sr * (chunk_ms / 1000.0)) * 2)
        for i in range(0, len(pcm), step):
            await q.put(pcm[i:i + step])
            await asyncio.sleep(chunk_ms / 1000.0)
        return len(pcm)

    def _build_and_start_input_stream(self):
        """Construct + start a fresh RawInputStream. Closes the stream
        handle on start() failure to avoid leaks (Codex review #2)."""
        # device=None lets PortAudio resolve to the *current* system default,
        # so a hot-plug change picks up automatically on reopen.
        stream = sd.RawInputStream(
            samplerate=self.input_sr,
            blocksize=self._chunk_frames,
            device=self.input_device,
            channels=1,
            dtype="int16",
            callback=self._input_callback,
        )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:  # pragma: no cover
                pass
            raise
        return stream

    def _reset_portaudio_library(self) -> None:
        """Terminate + reinitialize PortAudio to clear a bad library state.

        macOS CoreAudio occasionally enters a state where every new
        ``Pa_OpenStream`` returns ``-9986`` even though the device
        itself works (verifiable from a fresh Python process). This
        sequence flushes that state in-process. All existing streams
        MUST be closed first — calling ``Pa_Terminate`` while streams
        are open is undefined behavior.
        """
        # Close every stream we own, ignoring failures.
        for attr in ("_input_stream", "_output_stream"):
            stream = getattr(self, attr, None)
            if stream is None:
                continue
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            setattr(self, attr, None)
        # PortAudio C library cycle. The sounddevice module exposes the
        # private helpers; this is the documented way to reinit.
        try:
            sd._terminate()
        except Exception as e:  # pragma: no cover
            logger.warning("PortAudio terminate failed: %s", e)
        try:
            sd._initialize()
        except Exception as e:  # pragma: no cover
            logger.warning("PortAudio reinitialize failed: %s", e)
        # Refresh the cached device signature so the watcher doesn't
        # treat the post-reset state as another topology change.
        self._device_signature = self._compute_device_signature()

    def _open_input_stream(self) -> None:
        """Open (or reopen) the RawInputStream using the current default device.

        Falls back to a PortAudio library reset (terminate+reinit) after
        ``_PA_RESET_THRESHOLD`` consecutive failures — recovers from
        macOS CoreAudio corruption (BT bounce, app conflict) without
        restarting the agent process.
        """
        assert self._input_callback is not None
        try:
            stream = self._build_and_start_input_stream()
        except Exception:
            self._input_open_failures += 1
            if self._input_open_failures >= self._PA_RESET_THRESHOLD:
                logger.warning(
                    "PortAudio refused input %d times in a row; "
                    "resetting library and retrying",
                    self._input_open_failures,
                )
                self._reset_portaudio_library()
                # one retry attempt after the reset
                stream = self._build_and_start_input_stream()
                self._input_open_failures = 0
            else:
                raise
        self._input_stream = stream
        self._input_open_failures = 0
        # Log the resolved device name for field debuggability — useful
        # to verify which mic the agent ended up on after a hot-plug.
        try:
            dev_idx = self._input_stream.device  # PortAudio resolves None
            name = sd.query_devices(dev_idx)["name"] if dev_idx is not None else "(default)"
            logger.info(
                "input stream open: device=%s sr=%d chunk=%d",
                name, self.input_sr, self._chunk_frames,
            )
        except Exception:  # pragma: no cover - defensive
            pass

    @staticmethod
    def _compute_device_signature() -> tuple:
        """Identity tuple over current PortAudio device topology + default.

        Reopen the streams whenever this changes (BT connect/disconnect,
        USB hot-plug, default device swap in System Preferences). Polling
        is preferable to a CoreAudio property listener because it's the
        same code on Linux/Pi/Jetson hosts.
        """
        try:
            default = tuple(sd.default.device) if isinstance(
                sd.default.device, (list, tuple)
            ) else (sd.default.device, sd.default.device)
            devs = sd.query_devices()
        except Exception:
            return ()
        # Name + channel-count signature; ignore latencies (jitter).
        return (
            default,
            tuple(
                (d["name"], d["max_input_channels"], d["max_output_channels"])
                for d in devs
            ),
        )

    async def _watch_devices(self) -> None:
        """Poll PortAudio device list; on change, reopen both streams.

        Cheap (a few hundred μs per poll) and runs at ``OVS_AUDIO_WATCH_S``
        cadence (default 3s). The reopen swaps the device transparently —
        the asyncio queue stays the same, so consumers see no break.
        """
        while True:
            # ── Auto-detect + exponential backoff: a mic is wanted but no
            # input stream is open (absent at boot, or lost to a hot-unplug /
            # failed reopen). Retry with a doubling delay; PortAudio is
            # reinitialized each attempt so a newly-enumerated USB device
            # becomes visible (its device list is cached at init). ──────────
            if self._input_capture_active and self._input_stream is None:
                try:
                    await asyncio.sleep(self._input_reconnect_backoff_s)
                except asyncio.CancelledError:
                    return
                # Safe to cycle PortAudio here — no input stream is open. This
                # also drops the output stream; it re-creates lazily on play().
                try:
                    self._reset_portaudio_library()
                except Exception:
                    pass
                try:
                    self._open_input_stream()
                except Exception:
                    self._input_reconnect_backoff_s = min(
                        self._input_reconnect_backoff_s * 2.0,
                        self._input_reconnect_max_s,
                    )
                    logger.info(
                        "mic %r still absent; next auto-detect retry in %.0fs",
                        self.input_device, self._input_reconnect_backoff_s,
                    )
                    continue
                # Connected — reset backoff + resync the topology signature.
                self._input_reconnect_backoff_s = self._input_reconnect_min_s
                self._device_signature = self._compute_device_signature()
                logger.info("mic %r connected via auto-detect", self.input_device)
                continue

            try:
                await asyncio.sleep(self._device_watch_interval_s)
            except asyncio.CancelledError:
                return
            sig = self._compute_device_signature()
            if sig and sig != self._device_signature:
                logger.info(
                    "audio device topology changed; reopening streams"
                )
                self._device_signature = sig
                try:
                    await self._reopen_streams()
                except Exception as e:
                    logger.warning("audio device reopen failed: %s", e)

    async def _reopen_streams(self) -> None:
        """Stop + reopen input/output streams against the new default device.

        Runs on the event loop thread. ``sd.RawInputStream.stop()`` blocks
        until the PortAudio callback returns, so this is race-safe.
        """
        # Input
        if self._input_stream is not None and self._input_callback is not None:
            self._stop_input_stream()
            try:
                self._open_input_stream()
            except Exception as e:
                logger.warning("input stream reopen failed: %s", e)
        # Output — re-create lazily on next play() to use the new default
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None

    def _safe_put(self, data: bytes) -> None:
        """Runs on the asyncio loop thread; drops the chunk if the queue is full."""
        if self._in_queue is None:
            return
        try:
            self._in_queue.put_nowait(data)
        except asyncio.QueueFull:
            logger.warning("mic queue full -- dropping chunk")

    def _stop_input_stream(self) -> None:
        if self._input_stream is not None:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:  # pragma: no cover
                pass
            self._input_stream = None

    # ── playback ────────────────────────────────────────────────────

    def _ensure_playback_buffer(self) -> None:
        if not hasattr(self, "_playback_buffer"):
            self._playback_buffer = bytearray()
        if not hasattr(self, "_playback_lock"):
            self._playback_lock = threading.Lock()

    def _output_callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("output status: %s", status)
        self._ensure_playback_buffer()
        needed = len(outdata)
        with self._playback_lock:
            n = min(needed, len(self._playback_buffer))
            if n:
                outdata[:n] = self._playback_buffer[:n]
                del self._playback_buffer[:n]
                if self._playback_response_id is not None:
                    self._playback_device_bytes += n
            if n < needed:
                outdata[n:needed] = b"\x00" * (needed - n)

    def _ensure_output(self) -> None:
        if sd is None:  # pragma: no cover - hardware-dependent
            raise RuntimeError(
                f"sounddevice unavailable: {_SD_IMPORT_ERR!r}. Install libportaudio2."
            )
        if self._output_stream is not None:
            return
        self._ensure_playback_buffer()
        # Query the device's preferred sample rate. macOS Bluetooth in HFP
        # profile is mono 16k; pushing 24k into it produces silent playback.
        # Use whatever the device wants, and let play() resample to it.
        device_sr = self._resolve_output_sample_rate()
        if device_sr != self.output_sr:
            logger.info(
                "output stream sample rate %d -> %d (device-preferred)",
                self.output_sr, device_sr,
            )
            self.output_sr = device_sr
        self._output_stream = sd.RawOutputStream(
            samplerate=self.output_sr,
            blocksize=max(1, int(self.output_sr * 0.02)),
            device=self.output_device,
            channels=1,
            dtype="int16",
            callback=self._output_callback,
        )
        self._output_stream.start()
        try:
            dev_idx = self._output_stream.device
            name = sd.query_devices(dev_idx)["name"] if dev_idx is not None else "(default)"
            logger.info(
                "output stream open: device=%s sr=%d", name, self.output_sr,
            )
        except Exception:  # pragma: no cover
            pass

    def _resolve_output_sample_rate(self) -> int:
        """Pick a sample rate the current default output device accepts.

        macOS Bluetooth in HFP/SCO profile only supports its native rate
        (typically 16k mono). Opening at 24k or 48k there results in
        silent playback. Query the device and prefer its declared
        ``default_samplerate``; fall back to the configured rate when
        the query fails.
        """
        try:
            if self.output_device is not None:
                info = sd.query_devices(self.output_device, kind="output")
            else:
                info = sd.query_devices(kind="output")
            sr = int(info.get("default_samplerate") or 0)
            if sr > 0:
                return sr
        except Exception:
            pass
        return self.output_sr

    async def _playback_loop(self) -> None:
        assert self._out_queue is not None
        try:
            while True:
                pcm = await self._out_queue.get()
                if pcm is None:
                    continue
                # NB: do NOT toggle _is_playing here. SLV streams TTS chunks
                # with variable inter-frame timing, so the queue can be
                # transiently empty between two chunks of the same utterance.
                # If we flipped to False during those gaps, barge-in checks
                # (`if audio.is_playing`) would race and miss real interrupts.
                # is_playing is owned by BaseApp dispatch:
                #   first TTSAudio frame → play() sets True
                #   TTSDone               → mark_playback_done() sets False
                #   barge-in / shutdown   → stop_playback() sets False
                try:
                    if self._output_stream is not None:
                        await asyncio.to_thread(self._output_stream.write, pcm)
                except Exception as e:  # pragma: no cover
                    logger.warning("playback write error: %s", e)
        except asyncio.CancelledError:
            raise

    def mark_playback_done(self) -> None:
        """Called by BaseApp when SLV emits TTSDone.

        This marks the remote TTS stream done, but local PortAudio may still
        have buffered PCM to play. `is_playing` therefore stays true until the
        callback drains `_playback_buffer`; otherwise barge-in during audible
        tail audio is missed.
        """
        self._is_playing = False

    def begin_response_playback(self, response_id: str) -> None:
        """Start actual-played-duration accounting for one response."""
        with self._playback_lock:
            self._playback_response_id = response_id
            self._playback_device_bytes = 0

    def playback_position_ms(self, response_id: str | None = None) -> int:
        """Return PCM duration already handed to the physical output stream."""
        with self._playback_lock:
            if response_id and response_id != self._playback_response_id:
                return 0
            byte_count = self._playback_device_bytes
        return max(0, int(byte_count * 1000 / (2 * max(1, self.output_sr))))

    def play_notification(self, pcm: bytes) -> None:
        """Play a short audio clip without affecting _is_playing state.

        Used for non-TTS feedback (wake beep, error tone) that must not
        trigger barge-in checks or block mic via drop_while_speaking.
        """
        self._ensure_output()
        self._ensure_playback_buffer()
        with self._playback_lock:
            self._playback_buffer.extend(pcm)

    async def play(self, pcm: bytes) -> None:
        # After barge-in (or sleep / stop-intent), SLV may keep streaming
        # the rest of the in-flight TTS for several hundred ms. Drop those
        # so the speaker actually stays silent until the next user turn.
        if self._discard_playback:
            return
        self._ensure_output()
        self._ensure_playback_buffer()
        self._is_playing = True
        # Resample to the device's fixed output rate when source differs.
        # Linear interpolation on int16 — fine for voice at modest ratios
        # (e.g. 16k↔24k); avoids scipy dependency on edge images.
        if self._source_sr != self.output_sr and pcm:
            pcm = self._resample_int16(pcm, self._source_sr, self.output_sr)
        with self._playback_lock:
            self._playback_buffer.extend(pcm)

    @staticmethod
    def _resample_int16(pcm: bytes, sr_in: int, sr_out: int) -> bytes:
        if sr_in == sr_out or not pcm:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        n_out = int(round(len(x) * sr_out / sr_in))
        if n_out <= 0:
            return b""
        y = np.interp(
            np.linspace(0, len(x) - 1, n_out, dtype=np.float64),
            np.arange(len(x), dtype=np.float64),
            x,
        )
        return np.clip(y, -32768, 32767).astype(np.int16).tobytes()

    def arm_for_next_turn(self) -> None:
        """Re-enable playback for the next turn after a barge-in / sleep /
        stop-intent.  Called from BaseApp when a new ASR final arrives."""
        self._discard_playback = False

    def set_output_sample_rate(self, sr: int) -> None:
        """Record the *source* sample rate of incoming TTS PCM.

        The device-side output stream stays at the rate that was chosen
        when the agent started (``self.output_sr``); any mismatch is
        handled by in-process resampling in ``play()``. This avoids
        tearing down the audio stream when models switch — which on
        macOS Bluetooth devices manifests as silent playback because
        Core Audio negotiates a new rate but the codec can't keep up.
        """
        self._source_sr = int(sr)

    async def stop_playback(self) -> None:
        """Drain queued audio (barge-in / sleep / stop-intent).

        Also arms `_discard_playback` so any TTS chunks SLV keeps streaming
        over the WS for the rest of the in-flight utterance are dropped on
        arrival instead of being re-queued by play().  arm_for_next_turn()
        clears the latch at the start of the next user turn.
        """
        self._ensure_playback_buffer()
        with self._playback_lock:
            self._playback_buffer.clear()
        self._is_playing = False
        self._discard_playback = True

    async def close(self) -> None:
        self._stop_input_stream()
        if self._playback_task is not None:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except (asyncio.CancelledError, Exception):
                pass
            self._playback_task = None
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:  # pragma: no cover
                pass
            self._output_stream = None


__all__ = ["AudioIO"]


# Helper to keep numpy import alive (some platforms need it loaded for
# sounddevice's CFFI bindings to find PortAudio's int16 path).
_ = np.zeros(1, dtype=np.int16)
