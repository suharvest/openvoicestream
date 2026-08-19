"""OpenWakeWordSource — local keyword spotter as an ovs-agent WakeSource.

Reads PCM chunks from a TappedAudioIO capture tap, accumulates 80ms
windows (1280 samples @ 16k), feeds them to openwakeword, and fires
``app.wake(source="openwakeword")`` whenever any model score crosses
the threshold (subject to cooldown).

Inputs:
  * AudioIO writes mono int16 PCM at 16 kHz (AudioIO hardcodes channels=1
    and the sounddevice layer asks PortAudio to down-mix the reSpeaker
    XVF3800 multi-channel device on our behalf).
  * The model expects 80ms frames at 16 kHz (1280 samples).
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

try:
    from openwakeword import Model as _OWWModel
except Exception as _exc:  # pragma: no cover - import-time check
    _OWWModel = None  # type: ignore[assignment]
    _OWW_IMPORT_ERR: Exception | None = _exc
else:
    _OWW_IMPORT_ERR = None

from ovs_agent.wake_source import WakeSource

logger = logging.getLogger(__name__)


class OpenWakeWordSource(WakeSource):
    name = "openwakeword"
    local_audio = True

    # 80ms @ 16k = 1280 samples; openwakeword's documented chunk size.
    CHUNK_SAMPLES = 1280

    def __init__(
        self,
        app,
        model_name: str = "hey jarvis",
        threshold: float = 0.5,
        cooldown_s: float = 2.0,
        vad_threshold: float = 0.0,
    ) -> None:
        super().__init__(app)
        self._model_name = model_name
        self._threshold = float(threshold)
        self._cooldown_s = float(cooldown_s)
        self._vad_threshold = float(vad_threshold)
        # Self-supervising loop: a long-lived supervisor restarts the inner
        # listen loop (_run_once) on crash, so a transient predict/model error
        # can't permanently kill wake detection. The heartbeat (_last_chunk_ts)
        # lets an external watchdog catch a SILENT stall (tap.get() blocking)
        # the supervisor can't see (no exception) — see app_base wake watchdog
        # + request_restart.
        self._supervisor_task: asyncio.Task | None = None
        self._run_task: asyncio.Task | None = None
        self._stopped: bool = False
        self._last_chunk_ts: float | None = None  # heartbeat: last tap chunk seen
        self._last_wake_ts: float = 0.0
        self._buffer = np.zeros(0, dtype=np.int16)
        self._model = None  # lazily constructed in setup()

    def last_chunk_ts(self) -> "float | None":
        """Heartbeat for the wake watchdog: monotonic time of the last tap
        chunk the listen loop pulled. ``None`` until the first chunk (grace)."""
        return self._last_chunk_ts

    def request_restart(self) -> None:
        """Ask the supervisor to restart the inner listen loop (cancels the
        current _run_once; the supervisor re-spawns it). Used by the wake
        watchdog when the heartbeat goes stale while the mic is fresh — i.e.
        tap.get() is wedged. Safe to call repeatedly."""
        task = self._run_task
        if task is not None and not task.done():
            task.cancel()

    # ── plugin lifecycle ───────────────────────────────────────────
    def setup(self) -> bool:
        if _OWWModel is None:
            logger.error(
                "openwakeword not importable: %r; wake source disabled",
                _OWW_IMPORT_ERR,
            )
            return False
        model_id = self._model_name.replace(" ", "_") + "_v0.1"
        try:
            self._model = _OWWModel(
                wakeword_models=[model_id],
                inference_framework="onnx",
                vad_threshold=self._vad_threshold,
            )
        except Exception:
            # Fall back to bare model name (openwakeword's loader also
            # accepts the path-less alias for bundled models).
            try:
                self._model = _OWWModel(
                    wakeword_models=[self._model_name],
                    inference_framework="onnx",
                    vad_threshold=self._vad_threshold,
                )
            except Exception:
                logger.exception(
                    "openwakeword model load failed (model=%r); disabling",
                    self._model_name,
                )
                return False
        logger.info(
            "OpenWakeWordSource: model=%s threshold=%.2f cooldown=%.1fs",
            self._model_name, self._threshold, self._cooldown_s,
        )
        return True

    async def start(self) -> None:
        await super().start()
        self._stopped = False
        self._supervisor_task = asyncio.create_task(
            self._supervisor(), name="openwakeword-supervisor"
        )

    async def stop(self) -> None:
        await super().stop()
        # Set the stop flag FIRST so the supervisor doesn't re-spawn _run_once
        # after we cancel it (race: cancel → supervisor's except → respawn).
        self._stopped = True
        for task in (self._run_task, self._supervisor_task):
            if task is None:
                continue
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        self._run_task = None
        self._supervisor_task = None

    # ── supervisor ─────────────────────────────────────────────────
    async def _supervisor(self) -> None:
        """Run the listen loop, restarting it on crash (exponential backoff,
        reset after a clean run). Handles the CRASH case; a SILENT stall
        (tap.get blocked) is recovered by the external wake watchdog calling
        request_restart() → cancels _run_once → supervisor re-spawns it."""
        backoff = 0.5
        while not self._stopped:
            started = time.monotonic()
            self._run_task = asyncio.create_task(self._run_once(), name="openwakeword-run")
            try:
                await self._run_task
            except asyncio.CancelledError:
                # Distinguish a request_restart "kick" (cancels self._run_task,
                # the awaited task → restart it) from the supervisor itself
                # being torn down (external cancel / stop interrupts THIS await
                # WITHOUT cancelling the run task). On teardown re-raise so we
                # actually stop — and cancel the run task first so it can't leak
                # and keep the loop alive ignoring cancellation.
                if self._stopped or not self._run_task.cancelled():
                    if not self._run_task.done():
                        self._run_task.cancel()
                    raise
                logger.info("openwakeword: listen loop restarting (kicked)")
                continue
            except Exception:
                logger.exception("openwakeword: _run_once crashed; restarting in %.1fs", backoff)
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2.0, 10.0)
                continue
            # _run_once returned normally (shouldn't — it loops). If it ran a
            # while, reset backoff so the next genuine crash starts cheap.
            if time.monotonic() - started > 30.0:
                backoff = 0.5

    # ── inference loop ─────────────────────────────────────────────
    async def _run_once(self) -> None:
        # AudioIO opens its streams lazily; wait a beat so start_capture
        # has had a chance to allocate the queue + start the input stream.
        await asyncio.sleep(0.5)
        start_tap = getattr(self.app.audio, "start_capture_tap", None)
        if not callable(start_tap):
            logger.error(
                "audio object %r is not a TappedAudioIO; wake source idle",
                type(self.app.audio).__name__,
            )
            return
        tap = await start_tap()
        # Fresh window state per run so a restart can't carry a half-filled
        # buffer from the previous (crashed/stalled) session into predict().
        self._buffer = np.zeros(0, dtype=np.int16)

        logger.info("OpenWakeWordSource: loop entered, waiting for audio")
        try:
            while not self._stopped:
                chunk = await tap.get()
                # Heartbeat: stamp BEFORE processing so the wake watchdog sees
                # the loop is alive even if a chunk takes a while to process.
                self._last_chunk_ts = time.monotonic()
                arr = np.frombuffer(chunk, dtype=np.int16)
                if arr.size == 0:
                    continue
                self._buffer = np.concatenate([self._buffer, arr])
                while len(self._buffer) >= self.CHUNK_SAMPLES:
                    window = self._buffer[: self.CHUNK_SAMPLES]
                    self._buffer = self._buffer[self.CHUNK_SAMPLES :]
                    try:
                        scores = self._model.predict(window)
                    except Exception:
                        logger.exception("openwakeword.predict raised")
                        continue
                    if not scores:
                        continue
                    triggered: str | None = None
                    top_score = 0.0
                    for wname, score in scores.items():
                        if score > top_score:
                            top_score = float(score)
                        if score > self._threshold and self._cooldown_ok():
                            triggered = wname
                            break
                    if triggered is not None:
                        logger.info(
                            "WAKE detected: model=%s score=%.3f",
                            triggered, top_score,
                        )
                        self._last_wake_ts = time.monotonic()
                        # Discard stale samples so the next wake doesn't
                        # immediately re-trigger on the trailing audio.
                        self._buffer = np.zeros(0, dtype=np.int16)
                        try:
                            await self.app.wake(source=self.name)
                        except Exception:
                            logger.exception("app.wake() failed")
                        break
        finally:
            # Always unregister the tap so a restart doesn't leak a dead queue
            # in TappedAudioIO._taps (the sounddevice callback would keep
            # fanning chunks into an orphaned queue forever).
            stop_tap = getattr(self.app.audio, "stop_capture_tap", None)
            if callable(stop_tap):
                try:
                    stop_tap(tap)
                except Exception:  # pragma: no cover - defensive
                    logger.debug("stop_capture_tap failed", exc_info=True)
        # NOTE: no broad ``except Exception`` here — a crash propagates to
        # _supervisor, which logs it and restarts the loop with backoff.

    def _cooldown_ok(self) -> bool:
        return (time.monotonic() - self._last_wake_ts) > self._cooldown_s


__all__ = ["OpenWakeWordSource"]
