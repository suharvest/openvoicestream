"""Open-vocabulary local wake source backed by sherpa-onnx."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Sequence

import numpy as np

from ovs_agent.kws import PhraseCompiler, SherpaKwsBackend
from ovs_agent.wake_source import WakeSource

logger = logging.getLogger(__name__)


class RuntimeKwsSource(WakeSource):
    """Continuously spot runtime-compiled Chinese or English phrases.

    ``backend`` and ``compiler`` are injectable so lifecycle, audio and atomic
    update behavior can be tested without importing sherpa or loading weights.
    """

    name = "runtime_kws"
    local_audio = True

    def __init__(
        self,
        app,
        *,
        phrases: Sequence[str],
        model_config: dict[str, Any] | None = None,
        compiler_config: dict[str, Any] | None = None,
        cooldown_s: float = 2.0,
        backend=None,
        compiler=None,
    ) -> None:
        super().__init__(app)
        model_config = dict(model_config or {})
        compiler_config = dict(compiler_config or {})
        self._phrases = tuple(phrases)
        self._cooldown_s = float(cooldown_s)
        self._backend = backend or SherpaKwsBackend(model_config)
        self._compiler = compiler or PhraseCompiler(
            tokens=str(compiler_config.get("tokens") or model_config.get("tokens") or ""),
            lexicon=str(compiler_config.get("lexicon") or ""),
            cli=str(compiler_config.get("cli", "sherpa-onnx-cli")),
            tokens_type=str(compiler_config.get("tokens_type", "phone+ppinyin")),
            timeout_s=float(compiler_config.get("timeout_s", 10.0)),
        )
        self._stream = None
        self._supervisor_task: asyncio.Task | None = None
        self._run_task: asyncio.Task | None = None
        self._stopped = False
        self._last_chunk_ts: float | None = None
        self._last_wake_ts = 0.0
        self._update_lock: asyncio.Lock | None = None

    @property
    def phrases(self) -> tuple[str, ...]:
        return self._phrases

    def last_chunk_ts(self) -> float | None:
        return self._last_chunk_ts

    def status(self) -> dict[str, Any]:
        return {
            "available": self._stream is not None,
            "running": bool(self._run_task and not self._run_task.done()),
            "phrases": list(self._phrases),
            "last_audio_chunk_ts": self._last_chunk_ts,
            "last_wake_ts": self._last_wake_ts or None,
        }

    async def validate_phrases(self, phrases: Sequence[str]) -> tuple[str, ...]:
        compiled = await asyncio.to_thread(self._compiler.compile, phrases)
        return compiled.phrases

    def request_restart(self) -> None:
        task = self._run_task
        if task is not None and not task.done():
            task.cancel()

    def setup(self) -> bool:
        try:
            compiled = self._compiler.compile(self._phrases)
            # load() is intentionally idempotent; create_stream loads once and
            # future phrase updates only allocate a decoder stream.
            stream = self._backend.create_stream(compiled)
        except Exception:
            logger.exception("runtime KWS setup failed; wake source disabled")
            return False
        self._phrases = compiled.phrases
        self._stream = stream
        logger.info("RuntimeKwsSource ready: %d phrase(s)", len(self._phrases))
        return True

    async def update_phrases(self, phrases: Sequence[str]) -> tuple[str, ...]:
        """Compile and construct first, then atomically publish the new stream.

        A bad phrase/compiler failure leaves both the old phrases and stream
        untouched, so live wake detection never enters a half-updated state.
        """
        if self._update_lock is None:
            self._update_lock = asyncio.Lock()
        async with self._update_lock:
            compiled = await asyncio.to_thread(self._compiler.compile, phrases)
            # Keep all sherpa spotter calls on the event-loop thread. Stream
            # construction is cheap (weights are already loaded) and this
            # avoids racing detect() in wheels that don't promise thread
            # safety for the shared native KeywordSpotter instance.
            stream = self._backend.create_stream(compiled)
            old_phrases = set(self._phrases)
            self._stream = stream
            self._phrases = compiled.phrases
            if hasattr(self.app, "config"):
                existing = [
                    phrase
                    for phrase in (getattr(self.app.config, "wake_phrases", []) or [])
                    if phrase not in old_phrases
                ]
                self.app.config.wake_phrases = list(
                    dict.fromkeys([*self._phrases, *existing])
                )
        return self._phrases

    async def start(self) -> None:
        await super().start()
        self._stopped = False
        self._supervisor_task = asyncio.create_task(
            self._supervisor(), name="runtime-kws-supervisor"
        )

    async def stop(self) -> None:
        await super().stop()
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

    async def _supervisor(self) -> None:
        backoff = 0.5
        while not self._stopped:
            started = time.monotonic()
            self._run_task = asyncio.create_task(self._run_once(), name="runtime-kws-run")
            try:
                await self._run_task
            except asyncio.CancelledError:
                if self._stopped or not self._run_task.cancelled():
                    if not self._run_task.done():
                        self._run_task.cancel()
                    raise
                logger.info("runtime KWS listen loop restarting (kicked)")
                continue
            except Exception:
                logger.exception("runtime KWS loop crashed; restarting in %.1fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
                continue
            if time.monotonic() - started > 30.0:
                backoff = 0.5

    async def _run_once(self) -> None:
        await asyncio.sleep(0.5)
        start_tap = getattr(self.app.audio, "start_capture_tap", None)
        if not callable(start_tap):
            logger.error("audio object has no capture tap; runtime KWS idle")
            return
        tap = await start_tap()
        logger.info("RuntimeKwsSource: loop entered, waiting for audio")
        try:
            while not self._stopped:
                chunk = await tap.get()
                self._last_chunk_ts = time.monotonic()
                pcm = np.frombuffer(chunk, dtype=np.int16)
                if pcm.size == 0:
                    continue
                samples = pcm.astype(np.float32) / 32768.0
                stream = self._stream
                if stream is None:
                    continue
                keyword = self._backend.detect(
                    stream,
                    samples,
                    int(getattr(self.app.config, "audio_input_sample_rate", 16000)),
                )
                if keyword and self._cooldown_ok():
                    self._last_wake_ts = time.monotonic()
                    logger.info("WAKE detected: keyword=%s", keyword)
                    try:
                        await self.app.wake(source=self.name)
                    except Exception:
                        logger.exception("app.wake() failed")
        finally:
            stop_tap = getattr(self.app.audio, "stop_capture_tap", None)
            if callable(stop_tap):
                try:
                    stop_tap(tap)
                except Exception:
                    logger.debug("stop_capture_tap failed", exc_info=True)

    def _cooldown_ok(self) -> bool:
        return (time.monotonic() - self._last_wake_ts) > self._cooldown_s


__all__ = ["RuntimeKwsSource"]
