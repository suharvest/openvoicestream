"""MOSS-TTS-Nano backend using the Jetson native worker process.

adapted from app/backends/jetson/moss_tts_nano.py (2026-05-30), dedup after
registry switch.

The worker protocol is JSONL over stdio; the backend spawns a subprocess and
demuxes per-request events over a queue. Supports: BASIC_TTS, STREAMING,
VOICE_CLONE (prompt-prefix reference audio), MULTI_LANGUAGE.

Decoupling from the production copy:
  * ABCs from ``voxedge.backends.base`` (TTSBackend / TTSCapability).
  * Production took a ``profile`` dict and read worker/engine paths from env
    (``MOSS_WORKER_BIN`` / ``MOSS_ENGINE_DIR`` / ``MOSS_TOKENIZER`` /
    ``MOSS_CODEC_ONNX_DIR``) plus ``MOSS_PY_REPO`` / ``MOSS_ORT_EP`` /
    ``MOSS_ORT_THREADS`` inside ``preload()``. All are now explicit fields on
    :class:`MossTtsNanoConfig`. voxedge has ZERO env reads (not even via the
    profile dict). The subprocess inherits the parent env unmodified.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import signal
import struct
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from voxedge.backends.base import TTSBackend, TTSCapability

logger = logging.getLogger(__name__)


# ── env/profile → config mapping (defaults byte-equal to production) ─────────
#   MOSS_WORKER_BIN          → worker_bin ("/opt/jv-workers/moss_tts_nano_worker")
#   MOSS_ENGINE_DIR          → engine_dir ("/opt/models/moss-tts-nano/engines")
#   MOSS_TOKENIZER           → tokenizer_model (<engines>/tokenizer.model)
#   MOSS_CODEC_ONNX_DIR      → codec_onnx_dir ("/opt/models/moss-tts-nano/codec_onnx")
#   profile moss_max_slots   → max_slots (1)
#   profile moss_max_seq_len → max_seq_len (2048)
#   profile moss_sample_rate/tts_sample_rate → sample_rate (48000)
#   profile moss_channels/tts_channels → channels (2)
#   MOSS_PY_REPO             → py_repo ("/opt/moss-tts-nano-py")   [.py worker only]
#   MOSS_ORT_EP              → ort_ep ("cpu")                      [.py worker only]
#   MOSS_ORT_THREADS         → ort_threads (4)                     [.py worker only]


@dataclass
class MossTtsNanoConfig:
    """Explicit construction-time config for :class:`MossTtsNanoBackend`."""

    worker_bin: str = "/opt/jv-workers/moss_tts_nano_worker"
    engine_dir: str = "/opt/models/moss-tts-nano/engines"
    tokenizer_model: Optional[str] = None
    codec_onnx_dir: str = "/opt/models/moss-tts-nano/codec_onnx"
    max_slots: int = 1
    max_seq_len: int = 2048
    sample_rate: int = 48000
    channels: int = 2
    # Only used when ``worker_bin`` ends in ``.py`` (ORT-mode persistent worker).
    py_repo: str = "/opt/moss-tts-nano-py"
    ort_ep: str = "cpu"
    ort_threads: int = 4

    def __post_init__(self) -> None:
        import os.path as _p
        if self.tokenizer_model is None:
            self.tokenizer_model = _p.join(self.engine_dir, "tokenizer.model")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")


class _WorkerDeadError(RuntimeError):
    """Raised when the worker process dies or cannot be reached."""


class _WorkerRequestError(RuntimeError):
    """Raised for structured per-request worker failures."""


class MossTtsNanoBackend(TTSBackend):
    """MOSS-TTS-Nano subprocess backend for Jetson devices."""

    supports_hot_reload = False

    _CONTROL_TIMEOUT_S = 30.0
    _REQUEST_TIMEOUT_S = 30.0
    _SHUTDOWN_TIMEOUT_S = 5.0

    def __init__(self, config: Optional[MossTtsNanoConfig] = None):
        self._config = config or MossTtsNanoConfig()
        self._worker_bin = self._config.worker_bin
        self._engine_dir = self._config.engine_dir
        self._tokenizer_model = self._config.tokenizer_model
        self._codec_onnx_dir = self._config.codec_onnx_dir

        self._max_slots = int(self._config.max_slots)
        self._max_seq_len = int(self._config.max_seq_len)
        self._sample_rate = int(self._config.sample_rate)
        self._channels = int(self._config.channels)

        self._proc: subprocess.Popen[bytes] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._proc_lock = threading.Lock()
        self._queues_lock = threading.Lock()
        self._request_queues: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._control_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._thread_local = threading.local()

    @property
    def name(self) -> str:
        return "jetson.moss_tts_nano"

    @property
    def capabilities(self) -> set[TTSCapability]:
        return frozenset({
            TTSCapability.BASIC_TTS,
            TTSCapability.STREAMING,
            TTSCapability.VOICE_CLONE,
            TTSCapability.MULTI_LANGUAGE,
        })

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def is_ready(self) -> bool:
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    def preload(self) -> None:
        """Start the worker and wait for its startup-ready event."""
        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._terminate_locked()
            self._control_queue = queue.Queue()
            with self._queues_lock:
                self._request_queues.clear()

            if self._worker_bin.endswith(".py"):
                cmd = [
                    "python3", "-u", self._worker_bin,
                    f"--model-dir={self._engine_dir}",
                    f"--repo={self._config.py_repo}",
                    f"--execution-provider={self._config.ort_ep}",
                    f"--cpu-threads={self._config.ort_threads}",
                ]
            else:
                cmd = [
                    self._worker_bin,
                    f"--engine-dir={self._engine_dir}",
                    f"--tokenizer-model={self._tokenizer_model}",
                    f"--codec-onnx-dir={self._codec_onnx_dir}",
                    f"--max-slots={self._max_slots}",
                    f"--max-seq-len={self._max_seq_len}",
                ]
            logger.info(
                "Starting MOSS-TTS-Nano worker: bin=%s engine_dir=%s tokenizer=%s codec_onnx_dir=%s",
                self._worker_bin, self._engine_dir, self._tokenizer_model, self._codec_onnx_dir,
            )
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=False,
            )
            self._stdout_thread = threading.Thread(
                target=self._stdout_reader, args=(self._proc,),
                name="moss-tts-nano-stdout", daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._stderr_drain, args=(self._proc,),
                name="moss-tts-nano-stderr", daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

        deadline = time.monotonic() + self._CONTROL_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.shutdown()
                raise TimeoutError("Timed out waiting for MOSS-TTS-Nano worker_ready event")
            try:
                event = self._control_queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                with self._proc_lock:
                    proc = self._proc
                    returncode = proc.poll() if proc is not None else None
                if returncode is not None:
                    raise RuntimeError(f"MOSS-TTS-Nano worker exited during preload with code {returncode}")
                continue

            kind = event.get("event")
            if kind == "worker_ready":
                logger.info("MOSS-TTS-Nano worker ready: %s", event)
                return
            if kind == "worker_exit":
                raise RuntimeError(f"MOSS-TTS-Nano worker exited during preload: {event}")
            logger.debug("Ignoring startup event before worker_ready: %s", event)

    def shutdown(self) -> None:
        """Terminate the worker process and clear request queues."""
        with self._proc_lock:
            proc = self._proc
            self._proc = None
            with self._queues_lock:
                for request_queue in self._request_queues.values():
                    request_queue.put({"event": "error", "message": "MOSS-TTS-Nano worker shutting down"})
                self._request_queues.clear()
            if proc is None:
                return
            self._stop_process(proc)

    def unload(self) -> None:
        self.shutdown()

    def generate_streaming(self, text: str, **kwargs: Any) -> Iterator[bytes]:
        """Yield raw stereo PCM s16le chunks from the worker."""
        request_id = uuid.uuid4().hex
        request = self._build_request(request_id, text, stream=True, kwargs=kwargs)
        attempt = 0
        while attempt < 2:
            attempt += 1
            request_queue: queue.Queue[dict[str, Any]] = queue.Queue()
            first_chunk_ms: float | None = None
            start_time = time.monotonic()
            self._thread_local.last_stream_metadata = {}
            self._register_request_queue(request_id, request_queue)
            try:
                self._send_request(request)
                while True:
                    try:
                        event = request_queue.get(timeout=self._REQUEST_TIMEOUT_S)
                    except queue.Empty as exc:
                        self._forget_request_queue(request_id, request_queue)
                        raise TimeoutError(
                            f"Timed out waiting for MOSS-TTS-Nano chunk for request {request_id}"
                        ) from exc

                    kind = event.get("event")
                    if kind == "ready":
                        logger.debug("MOSS-TTS-Nano request ready: id=%s", request_id)
                        continue
                    if kind == "chunk":
                        data = event.get("audio_b64") or event.get("data")
                        if not isinstance(data, str):
                            raise _WorkerRequestError(
                                f"MOSS-TTS-Nano chunk missing base64 data for request {request_id}"
                            )
                        try:
                            pcm = base64.b64decode(data, validate=True)
                        except Exception as exc:
                            raise _WorkerRequestError(
                                f"MOSS-TTS-Nano returned invalid base64 chunk for {request_id}"
                            ) from exc
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.monotonic() - start_time) * 1000.0
                        if pcm:
                            yield pcm
                        continue
                    if kind == "done":
                        done_meta = self._metadata_from_done(event, first_chunk_ms)
                        self._thread_local.last_stream_metadata = done_meta
                        return
                    if kind == "error":
                        message = event.get("message", "unknown worker error")
                        raise _WorkerRequestError(f"MOSS-TTS-Nano worker error for {request_id}: {message}")
                    if kind == "worker_exit":
                        raise _WorkerDeadError(f"MOSS-TTS-Nano worker exited during request {request_id}: {event}")
                    logger.debug("Ignoring unknown MOSS-TTS-Nano event for %s: %s", request_id, event)
            except _WorkerRequestError:
                self._forget_request_queue(request_id, request_queue)
                raise
            except (BrokenPipeError, OSError, _WorkerDeadError, TimeoutError):
                self._forget_request_queue(request_id, request_queue)
                if attempt >= 2:
                    raise
                logger.warning("MOSS-TTS-Nano request %s failed; respawning worker and retrying once", request_id)
                self._respawn_worker()
                continue
            finally:
                self._forget_request_queue(request_id, request_queue)

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = "auto",
        **kwargs: Any,
    ) -> tuple[bytes, dict]:
        """Synthesize text and return WAV bytes plus metadata."""
        if speaker_id is not None:
            kwargs.setdefault("speaker_id", speaker_id)
        if speed is not None:
            kwargs.setdefault("speed", speed)
        if pitch_shift is not None:
            kwargs.setdefault("pitch_shift", pitch_shift)

        start_time = time.monotonic()
        pcm_chunks = list(self.generate_streaming(text, language=language or "auto", **kwargs))
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        pcm = b"".join(pcm_chunks)
        wav_bytes = self._pcm_to_wav(pcm, sample_rate=self._sample_rate, channels=self._channels)
        stream_meta = getattr(self._thread_local, "last_stream_metadata", {}) or {}
        total_samples = len(pcm) // (2 * self._channels)
        wall_ms = stream_meta.get("wall_ms")
        if wall_ms is None:
            wall_ms = int(round(elapsed_ms))
        metadata = {
            "ttfa_ms": stream_meta.get("ttfa_ms"),
            "wall_ms": wall_ms,
            "total_samples": total_samples,
            "total_frames": stream_meta.get("total_frames"),
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "language": language or "auto",
        }
        return wav_bytes, metadata

    def clone_voice(
        self,
        text: str,
        speaker_embedding: Optional[bytes] = None,
        language: Optional[str] = None,
        *,
        reference_audio: Optional[bytes] = None,
        reference_sample_rate: int = 48000,
        **kwargs: Any,
    ) -> tuple[bytes, dict]:
        """Synthesize using prompt-prefix voice cloning audio."""
        audio = reference_audio if reference_audio is not None else speaker_embedding
        if audio is None:
            raise ValueError("clone_voice requires reference_audio bytes")
        ref_audio_b64 = base64.b64encode(audio).decode("ascii")
        return self.synthesize(
            text,
            language=language or "auto",
            ref_audio_b64=ref_audio_b64,
            ref_audio_sample_rate=int(reference_sample_rate),
            **kwargs,
        )

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        raise NotImplementedError(
            "MOSS-TTS-Nano uses prompt-prefix voice cloning from reference audio; "
            "it does not expose explicit reusable speaker embeddings."
        )

    def _build_request(self, request_id: str, text: str, *, stream: bool, kwargs: dict[str, Any]) -> dict[str, Any]:
        chunk_frames = int(kwargs.get("chunk_frames", 8))
        if chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
        request: dict[str, Any] = {
            "id": request_id,
            "request_id": request_id,
            "text": text,
            "stream": bool(stream),
            "chunk_transport": "base64",
            "chunk_format": "pcm_s16le",
            "chunk_frames": chunk_frames,
        }
        ref_audio_b64 = kwargs.get("ref_audio_b64")
        if ref_audio_b64:
            if isinstance(ref_audio_b64, bytes):
                ref_audio_b64 = ref_audio_b64.decode("ascii")
            request["ref_audio_b64"] = ref_audio_b64
            if kwargs.get("ref_audio_sample_rate") is not None:
                request["ref_audio_sample_rate"] = int(kwargs["ref_audio_sample_rate"])
        return request

    def _send_request(self, request: dict[str, Any]) -> None:
        line = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        with self._proc_lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise _WorkerDeadError("MOSS-TTS-Nano worker is not running")
            if proc.stdin is None:
                raise _WorkerDeadError("MOSS-TTS-Nano worker stdin is unavailable")
            proc.stdin.write(line)
            proc.stdin.flush()

    def _respawn_worker(self) -> None:
        self.shutdown()
        self.preload()

    def _register_request_queue(self, request_id: str, request_queue: queue.Queue[dict[str, Any]]) -> None:
        with self._queues_lock:
            self._request_queues[request_id] = request_queue

    def _forget_request_queue(self, request_id: str, request_queue: queue.Queue[dict[str, Any]]) -> None:
        with self._queues_lock:
            if self._request_queues.get(request_id) is request_queue:
                self._request_queues.pop(request_id, None)

    def _metadata_from_done(self, event: dict[str, Any], first_chunk_ms: float | None) -> dict[str, Any]:
        ttfa_ms = event.get("ttfa_ms")
        if ttfa_ms is None and first_chunk_ms is not None:
            ttfa_ms = int(round(first_chunk_ms))
        return {
            "total_frames": event.get("total_frames"),
            "wall_ms": event.get("wall_ms"),
            "ttfa_ms": ttfa_ms,
        }

    def _stdout_reader(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stdout is None:
            self._publish_worker_exit(proc, "stdout unavailable")
            return
        try:
            while True:
                raw = proc.stdout.readline()
                if raw == b"":
                    self._publish_worker_exit(proc, "stdout eof")
                    return
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON MOSS-TTS-Nano stdout: %s", line)
                    continue
                if not isinstance(event, dict):
                    logger.debug("Skipping non-object MOSS-TTS-Nano stdout event: %s", event)
                    continue
                self._route_stdout_event(event)
        except Exception as exc:
            logger.error("MOSS-TTS-Nano stdout reader failed: %s", exc)
            self._publish_worker_exit(proc, f"stdout reader failed: {exc}")

    def _stderr_drain(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stderr is None:
            return
        try:
            while True:
                raw = proc.stderr.readline()
                if raw == b"":
                    return
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    logger.debug("MOSS-TTS-Nano stderr: %s", line)
        except Exception as exc:
            logger.debug("MOSS-TTS-Nano stderr drain ended: %s", exc)

    def _route_stdout_event(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "worker_ready":
            self._control_queue.put(event)
            return
        request_id = event.get("request_id")
        if not (isinstance(request_id, str) and request_id):
            request_id = event.get("id")
        if isinstance(request_id, str) and request_id:
            with self._queues_lock:
                request_queue = self._request_queues.get(request_id)
            if request_queue is not None:
                request_queue.put(event)
            else:
                logger.debug("Dropping MOSS-TTS-Nano event for unknown request %s: %s", request_id, event)
            return
        logger.debug("Dropping MOSS-TTS-Nano event without request id: %s", event)

    def _publish_worker_exit(self, proc: subprocess.Popen[bytes], reason: str) -> None:
        returncode = proc.poll()
        event = {"event": "worker_exit", "returncode": returncode, "message": reason}
        self._control_queue.put(event)
        with self._queues_lock:
            request_queues = list(self._request_queues.values())
        for request_queue in request_queues:
            request_queue.put(event)

    def _terminate_locked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            self._stop_process(proc)

    def _stop_process(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=self._SHUTDOWN_TIMEOUT_S)
            logger.info("MOSS-TTS-Nano worker terminated")
            return
        except subprocess.TimeoutExpired:
            logger.warning("MOSS-TTS-Nano worker did not exit after SIGTERM; sending SIGKILL")
        except Exception as exc:
            logger.warning("Failed to SIGTERM MOSS-TTS-Nano worker cleanly: %s", exc)
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
                logger.info("MOSS-TTS-Nano worker killed")
            except Exception as exc:
                logger.error("Failed to SIGKILL MOSS-TTS-Nano worker: %s", exc)

    def _pcm_to_wav(self, pcm: bytes, *, sample_rate: int, channels: int) -> bytes:
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        bits_per_sample = 16
        bytes_per_sample = bits_per_sample // 8
        block_align = channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        data_size = len(pcm)
        chunk_size = 36 + data_size
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, channels, sample_rate,
            byte_rate, block_align, bits_per_sample, b"data", data_size,
        )
        return header + pcm
