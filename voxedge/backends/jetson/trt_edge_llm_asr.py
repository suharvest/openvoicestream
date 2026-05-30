"""ASR backend via TRT-Edge-LLM C++ worker (qwen3_asr_worker / llm_inference).

adapted from app/backends/jetson/trt_edge_llm_asr.py + app/core/worker_io.py
(2026-05-30), dedup after registry switch.

Audio is converted to a Whisper-compatible log-mel spectrogram in Python
(numpy-only), saved as a safetensors file, and passed to the LLM binary via
``--multimodalEngineDir`` for the audio encoder. The Python side spawns a C++
worker subprocess and talks JSON-line IPC, so it imports cleanly on a machine
with no CUDA / tensorrt.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ABCs imported from ``voxedge.backends.base`` (ASRBackend / ASRCapability /
    ASRStream / TranscriptionResult) and ``ConcurrencyCapability`` from
    ``voxedge.engine.concurrency_capability``.
  * ALL ~30 ``os.environ.get(...)`` reads (EDGE_LLM_* paths, ASR_* sampling,
    OVS_VAD_* offline-split, manifest path) are replaced by an explicit
    ``TRTEdgeLLMASRConfig`` dataclass injected at construction time. voxedge
    has ZERO module-scope or hardcoded env reads.
  * ``WorkerIO`` imported from the sibling ``voxedge.backends.jetson.worker_io``
    (not ``app.core.worker_io``).
  * The production offline-split path imported a DELETED module
    (``app.backends.jetson.qwen3_asr``) and ``app.core.vad`` /
    ``app.core.qwen3_artifact_downloader``. The silence splitters are
    reproduced env-free in ``._util``; the optional VAD-backend splitter and
    artifact auto-download are dropped (voxedge ships neither), so the long
    audio path uses the webrtcvad→energy splitter cascade only.
  * ``concurrency_capability`` is an instance method (voxedge base contract)
    reading ``config.max_slots`` instead of env/profile; the N>1
    ``--max_slots`` conditional (main fix b1cb1a5) is preserved.

Supports: OFFLINE, MULTI_LANGUAGE, STREAMING
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from voxedge.backends.base import (
    ASRBackend,
    ASRCapability,
    ASRStream,
    TranscriptionResult,
)
from voxedge.engine.concurrency_capability import ConcurrencyCapability

from ._trt_edge_llm_util import (
    VAD_MAX_SEG_SEC,
    _split_at_silence_energy,
    _split_at_silence_vad,
)
from .trt_edge_llm_ipc import audio_bytes_to_mel, run_binary, write_safetensors
from .worker_io import WorkerExitError as _WIOExitError
from .worker_io import WorkerIO

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var                       → TRTEdgeLLMASRConfig field
#   EDGE_LLM_ASR_BIN                      → asr_binary
#   EDGE_LLM_ASR_WORKER_BIN              → worker_binary
#   EDGE_LLM_ASR_PLUGIN_PATH/EDGELLM_ASR_PLUGIN_PATH → plugin_path
#   EDGE_LLM_ASR_ENGINE_DIR             → engine_dir
#   EDGE_LLM_ASR_AUDIO_ENC_DIR          → audio_encoder_dir
#   EDGE_LLM_ASR_WORKER                  → use_worker (default True)
#   EDGE_LLM_ASR_MEL_TENSOR_NAME        → mel_tensor_name ("mel")
#   EDGE_LLM_ASR_MAX_MEL_FRAMES         → max_mel_frames (6000)
#   EDGE_LLM_ASR_MAX_CONCURRENT          → max_slots (1)   ← N>1 gates --max_slots
#   EDGE_LLM_ASR_STREAM_MODE            → stream_mode ("accumulate")
#   EDGE_LLM_ASR_STREAM_CHUNK_SEC       → stream_chunk_sec (0.5)
#   EDGE_LLM_ASR_STREAM_UNFIXED_CHUNKS  → stream_unfixed_chunks (2)
#   EDGE_LLM_ASR_STREAM_UNFIXED_TOKENS  → stream_unfixed_tokens (5)
#   EDGE_LLM_ASR_MEL_SETTINGS           → mel_settings_path ("")
#   EDGE_LLM_ASR_MEL_FILTERS            → mel_filters_path ("")
#   ASR_TEMPERATURE                      → temperature (1.0)
#   ASR_TOP_P                            → top_p (1.0)
#   ASR_TOP_K                            → top_k (1)
#   ASR_MAX_GENERATE_LENGTH             → max_generate_length (200)
#   EDGE_LLM_ASR_MIN_AUDIO_FRAMES       → min_audio_frames (100)
#   EDGE_LLM_ASR_OFFLINE_SEGMENT        → offline_segment_enabled (True)
#   EDGE_LLM_ASR_OFFLINE_SEGMENT_SEC    → offline_segment_threshold_s (6.0)
#   EDGE_LLM_ASR_OFFLINE_MIN_SEGMENT_SEC → offline_segment_min_s (0.4)
#   EDGE_LLM_ASR_WORKER_WARMUP/SKIP_ASR_WARMUP → worker_warmup (True)
#   EDGE_LLM_ASR_PREWARM_MAX            → prewarm_max (6)
#   EDGE_LLM_ASR_CUDA_GRAPH             → worker_cuda_graph ("0")


@dataclass
class TRTEdgeLLMASRConfig:
    """Explicit construction-time config for :class:`TRTEdgeLLMASRBackend`.

    Every field default is identical to the production env default. Nothing
    here reads ``os.environ``: the path/engine fields have NO usable default
    (production resolved them from ``~/...`` artifact trees via env at module
    import) and MUST be supplied by the caller for a working backend — they
    default to empty strings so the module imports without CUDA/artifacts.
    """

    # Binaries / engines / plugin (no usable default — supply at construction).
    asr_binary: str = ""
    worker_binary: str = ""
    plugin_path: str = ""
    engine_dir: str = ""
    audio_encoder_dir: str = ""

    use_worker: bool = True
    mel_tensor_name: str = "mel"
    max_mel_frames: int = 6000
    # Slot-pool admission ceiling. Default 1 == legacy single-session. N>1
    # gates ``--max_slots`` (main fix b1cb1a5 — preserved).
    max_slots: int = 1

    stream_mode: str = "accumulate"
    stream_chunk_sec: float = 0.5
    stream_unfixed_chunks: int = 2
    stream_unfixed_tokens: int = 5
    mel_settings_path: str = ""
    mel_filters_path: str = ""

    # Sampling.
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 1
    max_generate_length: int = 200

    min_audio_frames: int = 100

    # Offline long-audio segmentation.
    offline_segment_enabled: bool = True
    offline_segment_threshold_s: float = 6.0
    offline_segment_min_s: float = 0.4

    # Worker warmup.
    worker_warmup: bool = True
    prewarm_max: int = 6
    worker_cuda_graph: str = "0"

    # Extra env to pass through to the worker subprocess (e.g. profile vars).
    # voxedge does not read process env; callers inject what the worker needs.
    extra_worker_env: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_slots = max(1, int(self.max_slots))
        self.stream_mode = (self.stream_mode or "accumulate").strip().lower()


class WorkerProtocolError(RuntimeError):
    """Base class for ASR worker protocol-level errors."""


class NoActiveSessionError(WorkerProtocolError):
    """Worker reported there is no active session (stale id / double-end)."""


class SessionAlreadyActiveError(WorkerProtocolError):
    """Worker reported a session is already active for the given id."""


class WorkerExitError(WorkerProtocolError):
    """Worker subprocess exited or didn't respond before the ack deadline."""


class PoolSaturatedError(RuntimeError):
    """Worker rejected a begin because every slot in its pool is busy.

    The C++ ``qwen3_asr_worker`` with ``--max_slots N`` returns
    ``{"error":"pool_saturated","status":4429,"max_slots":N}`` when an N+1st
    distinct session id tries to begin. Intentionally NOT a
    ``WorkerProtocolError`` subclass: a saturation is a fast-fail busy reject,
    not a worker fault that should trigger a destructive worker restart.
    """

    status: int = 4429

    def __init__(self, message: str, max_slots: Optional[int] = None) -> None:
        super().__init__(message)
        self.max_slots = max_slots


def _classify_worker_response(
    output_data: dict, *, request_event: str | None = None
) -> WorkerProtocolError | None:
    """Map a worker error JSON payload to a typed exception (or None)."""
    if not isinstance(output_data, dict):
        return None
    if output_data.get("event") != "error" and output_data.get("ok") is not False:
        return None
    msg = ""
    for key in ("error", "message", "reason", "detail"):
        v = output_data.get(key)
        if isinstance(v, str) and v:
            msg = v
            break
    if not msg:
        msg = str(output_data)
    low = msg.lower()
    if (
        output_data.get("status") == 4429
        or "pool_saturated" in low
        or "too_many_asr_sessions" in low
        or "too many asr sessions" in low
    ):
        return PoolSaturatedError(msg, max_slots=output_data.get("max_slots"))
    if "no active session" in low or "no_active_session" in low or "unknown session" in low:
        return NoActiveSessionError(msg)
    if "already active" in low or "session_already_active" in low or "already exists" in low:
        return SessionAlreadyActiveError(msg)
    if "exit" in low or "terminated" in low or "worker dead" in low:
        return WorkerExitError(msg)
    return None


class TRTEdgeLLMASRBackend(ASRBackend):
    """ASR via TRT-Edge-LLM qwen3_asr_worker subprocess."""

    @property
    def supports_hot_reload(self) -> bool:  # type: ignore[override]
        return self._use_worker()

    def concurrency_capability(self) -> ConcurrencyCapability:
        """Declare the ASR slot-pool ceiling.

        The backend multiplexes N distinct ASR sessions over one C++ worker
        subprocess (``--max_slots N`` + WorkerIO Semaphore(N)). Reads
        ``config.max_slots`` (was env ``EDGE_LLM_ASR_MAX_CONCURRENT`` / profile
        ``asr_max_slots``). N>1 enables ``supports_parallel``.
        """
        n = max(1, int(self._config.max_slots))
        return ConcurrencyCapability(
            supports_parallel=n > 1,
            max_concurrent=n,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="single_runtime_multiplex",
        )

    def __init__(self, config: Optional[TRTEdgeLLMASRConfig] = None):
        self._config = config or TRTEdgeLLMASRConfig()
        self._ready = False
        self._worker: Optional[subprocess.Popen] = None
        self._worker_lock = threading.Lock()
        self._restart_lock = threading.Lock()
        self._worker_ready_meta: dict = {}
        self._worker_stderr_tail: deque[str] = deque(maxlen=80)
        self._max_slots: int = max(1, int(self._config.max_slots))
        self._wio: Optional[WorkerIO] = None

    # -- ASRBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {
            ASRCapability.OFFLINE,
            ASRCapability.MULTI_LANGUAGE,
            ASRCapability.STREAMING,
        }

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        """Verify all required files exist."""
        cfg = self._config
        worker_binary = cfg.worker_binary
        asr_binary = cfg.asr_binary
        plugin_path = cfg.plugin_path
        engine_dir = cfg.engine_dir
        audio_encoder_dir = cfg.audio_encoder_dir
        required = [
            (worker_binary if self._use_worker() else asr_binary, "ASR binary"),
            (plugin_path, "TRT-Edge-LLM plugin"),
            (os.path.join(engine_dir, "config.json"), "LLM config"),
            (os.path.join(engine_dir, "llm.engine"), "LLM engine"),
            (os.path.join(audio_encoder_dir, "audio", "config.json"), "audio encoder config"),
            (os.path.join(audio_encoder_dir, "audio", "audio_encoder.engine"), "audio encoder engine"),
        ]
        missing = [(path, label) for path, label in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "ASR preload failed — missing:\n  "
                + "\n  ".join(f"{l}: {p}" for p, l in missing)
            )
        self._require_streaming_worker_assets()

        logger.info("ASR backend preload OK (config=%s)", self._config)
        if self._use_worker():
            self._ensure_worker()
        self._ready = True
        if self._use_worker():
            self._warm_worker()

    def unload(self) -> None:
        """Kill the resident ASR worker subprocess to fully release GPU memory."""
        if not self._ready and self._worker is None:
            return
        try:
            self.restart_worker()
        except Exception:
            logger.exception("TRTEdgeLLMASRBackend.unload failed; continuing")
        finally:
            self._ready = False

    def _warm_worker(self) -> None:
        """Pre-warm TRT audio_encoder optimization profile for batch shapes 1..N."""
        if not self._config.worker_warmup:
            logger.info("TRT-EdgeLLM ASR worker warmup skipped.")
            return
        prewarm_max = max(1, min(int(self._config.prewarm_max), 60))
        import time as _time

        t0 = _time.monotonic()
        warmed = 0
        for seconds in range(1, prewarm_max + 1):
            try:
                silence = np.zeros(16000 * seconds, dtype=np.float32)
                self.transcribe(_float_audio_to_wav_bytes(silence, 16000))
                warmed += 1
            except Exception as exc:
                msg = str(exc)
                if "cannot handle" in msg or "TensorRT Edge LLM" in msg:
                    logger.info(
                        "TRT-EdgeLLM ASR pre-warm: engine boundary at batch=%d "
                        "(expected, stopping)", seconds,
                    )
                else:
                    logger.warning(
                        "TRT-EdgeLLM ASR pre-warm batch=%d failed: %s", seconds, exc
                    )
                break
        elapsed = _time.monotonic() - t0
        logger.info(
            "TRT-EdgeLLM ASR worker pre-warmed shapes 1..%d in %.1fs", warmed, elapsed
        )

    def _use_worker(self) -> bool:
        return bool(self._config.use_worker)

    def _use_streaming_worker(self) -> bool:
        return self._config.stream_mode in (
            "worker", "stream", "streaming", "chunk_confirm", "prefix"
        )

    def _require_streaming_worker_assets(self) -> None:
        if not self._use_streaming_worker():
            return
        missing = []
        if not self._use_worker():
            missing.append("use_worker=True is required for streaming worker mode")
        for value, label in (
            (self._config.mel_settings_path, "mel_settings_path"),
            (self._config.mel_filters_path, "mel_filters_path"),
        ):
            if not value or not os.path.exists(value):
                missing.append(f"{label}: {value or '(unset)'}")
        if missing:
            raise FileNotFoundError(
                "stream_mode=worker requires PCM mel assets:\n  "
                + "\n  ".join(missing)
            )

    def _worker_env(self) -> dict:
        env = os.environ.copy()
        env.update(self._config.extra_worker_env)
        env["EDGELLM_PLUGIN_PATH"] = self._config.plugin_path
        env.setdefault("EDGE_LLM_ASR_CUDA_GRAPH", self._config.worker_cuda_graph)
        return env

    def _drain_worker_stderr(self, worker: subprocess.Popen) -> None:
        if worker.stderr is None:
            return
        for line in worker.stderr:
            text = line.rstrip()
            self._worker_stderr_tail.append(text)
            if "[JV_MEM]" in text:
                logger.info("ASR worker: %s", text)
            else:
                logger.debug("ASR worker stderr: %s", text)

    def _stderr_tail_text(self) -> str:
        return "\n".join(self._worker_stderr_tail)

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        cmd = [
            self._config.worker_binary,
            "--engineDir",
            self._config.engine_dir,
            "--multimodalEngineDir",
            self._config.audio_encoder_dir,
        ]
        # Only emit --max_slots when N>1 (main fix b1cb1a5): at N=1 we omit it
        # for byte-equivalent legacy behavior and back-compat with worker
        # binaries built before --max_slots existed.
        if self._max_slots and self._max_slots > 1:
            cmd += ["--max_slots", str(self._max_slots)]
        mel_settings = self._config.mel_settings_path or ""
        mel_filters = self._config.mel_filters_path or ""
        if mel_settings and mel_filters:
            cmd += ["--melSettings", mel_settings, "--melFilters", mel_filters]
        self._worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._worker_env(),
        )
        self._worker_stderr_tail.clear()
        threading.Thread(
            target=self._drain_worker_stderr,
            args=(self._worker,),
            name="trt-edgellm-asr-stderr",
            daemon=True,
        ).start()
        assert self._worker.stdout is not None
        ready_line = self._worker.stdout.readline()
        if not ready_line:
            stderr = self._stderr_tail_text()
            raise RuntimeError(f"ASR worker failed to start: {stderr}")
        ready = json.loads(ready_line)
        if ready.get("event") != "ready":
            raise RuntimeError(f"ASR worker did not become ready: {ready}")
        self._worker_ready_meta = ready
        self._max_slots = max(1, int(self._config.max_slots))
        # NB: ``_ensure_worker`` reads the worker's initial ``ready`` line
        # itself (above) BEFORE handing stdout to the WorkerIO reader thread.
        self._wio = WorkerIO(self._worker, concurrency=self._max_slots)

    def _worker_request(self, input_data: dict) -> dict:
        """Send one streaming protocol line to the worker, return its single reply."""
        req_event = input_data.get("event") if isinstance(input_data, dict) else None
        with self._worker_lock:
            self._ensure_worker()
            wio = self._wio
        assert wio is not None
        try:
            output_data: Optional[dict] = None
            gen = wio.request(input_data)
            try:
                for ev in gen:
                    output_data = ev
                    break
            finally:
                gen.close()
        except _WIOExitError as exc:
            stderr = self._stderr_tail_text()
            self._worker = None
            raise WorkerExitError(
                f"ASR worker exited before response: {exc}: {stderr}"
            ) from exc
        except (BrokenPipeError, OSError) as exc:
            stderr = self._stderr_tail_text()
            self._worker = None
            raise WorkerExitError(
                f"ASR worker stdin broken (likely killed): {exc}: {stderr}"
            ) from exc
        if output_data is None:
            stderr = self._stderr_tail_text()
            self._worker = None
            raise WorkerExitError(f"ASR worker exited before response: {stderr}")
        typed = _classify_worker_response(output_data, request_event=req_event)
        if typed is not None:
            raise typed
        if output_data.get("event") == "error" or output_data.get("ok") is False:
            raise WorkerProtocolError(f"ASR worker error: {output_data}")
        return output_data

    def restart_worker(self) -> None:
        """Forcibly kill the worker subprocess so the next request rebuilds it."""
        with self._restart_lock:
            worker = self._worker
            if worker is None:
                return
            self._worker = None
            self._worker_ready_meta = {}
            wio = self._wio
            self._wio = None
            if wio is not None:
                try:
                    wio.close()
                except Exception:
                    logger.debug("WorkerIO.close() during restart raised", exc_info=True)
            try:
                if worker.poll() is None:
                    try:
                        worker.kill()
                    except Exception:
                        pass
                    try:
                        worker.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                for fh in (worker.stdin, worker.stdout, worker.stderr):
                    try:
                        if fh is not None and not fh.closed:
                            fh.close()
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("restart_worker: kill failed: %s", exc)
        logger.info("ASR worker restarted (will respawn on next request)")

    @staticmethod
    def _strip_language_prefix(text: str) -> tuple[str, Optional[str]]:
        language_detected = None
        if text and len(text) >= 9 and text[:9] == "language ":
            known_languages = (
                "Chinese", "English", "Cantonese", "Japanese", "Korean",
                "French", "German", "Italian", "Portuguese", "Russian",
                "Spanish",
            )
            for name in known_languages:
                prefix = f"language {name}"
                if text.startswith(prefix):
                    language_detected = name
                    text = text[len(prefix):].lstrip()
                    break
            else:
                space = text.find(" ", 9)
                if space > 0:
                    language_detected = text[9:space]
                    text = text[space + 1:].lstrip()
                else:
                    language_detected = text[9:]
                    text = ""
        return text, language_detected

    def _transcribe_worker(self, mel_path: str, elapsed_mel_s: float) -> TranscriptionResult:
        req_id = uuid.uuid4().hex
        input_data = {
            "id": req_id,
            "requests": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "audio", "audio": mel_path}],
                        }
                    ],
                }
            ],
            "batch_size": 1,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "top_k": self._config.top_k,
            "max_generate_length": self._config.max_generate_length,
            "apply_chat_template": True,
            "add_generation_prompt": True,
        }
        with self._worker_lock:
            self._ensure_worker()
            wio = self._wio
        assert wio is not None
        t0 = time.time()
        try:
            output_data: dict = {}
            for ev in wio.request(input_data):
                output_data = ev
        except _WIOExitError as exc:
            stderr = self._stderr_tail_text()
            self._worker = None
            raise RuntimeError(
                f"ASR worker exited before response: {exc}: {stderr}"
            ) from exc
        elapsed_worker = time.time() - t0

        if not output_data.get("ok"):
            raise RuntimeError(f"ASR worker failed: {output_data}")

        responses = output_data.get("responses", [])
        if not responses:
            raise RuntimeError(f"ASR produced no responses: {output_data}")
        text = responses[0].get("output_text", "")
        if text == "TensorRT Edge LLM cannot handle this request. Fails.":
            raise RuntimeError(f"ASR inference failed (model returned error): {responses[0]}")
        text, language_detected = self._strip_language_prefix(text)
        total_s = elapsed_mel_s + elapsed_worker
        return TranscriptionResult(
            text=text,
            language=language_detected,
            meta={
                "inference_time_s": round(total_s, 3),
                "mel_time_s": round(elapsed_mel_s, 3),
                "worker_time_s": round(elapsed_worker, 3),
                "worker_init_ms": round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
            },
        )

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe audio via the resident C++ worker (or one-shot binary)."""
        if not self._ready:
            raise RuntimeError("ASR backend not preloaded")

        if self._config.offline_segment_enabled:
            try:
                audio, sample_rate = _wav_bytes_to_float_audio(audio_bytes)
                duration_s = len(audio) / max(sample_rate, 1)
            except Exception:
                audio = None
                sample_rate = 16000
                duration_s = 0.0
            if audio is not None and duration_s > self._config.offline_segment_threshold_s:
                return self._transcribe_segmented_offline(audio, sample_rate, language)

        with tempfile.TemporaryDirectory(prefix="trt_edgellm_asr_") as tmpdir:
            mel_t0 = time.time()
            mel = audio_bytes_to_mel(
                audio_bytes, min_audio_frames=self._config.min_audio_frames
            )  # [1, 128, T] float32
            max_mel_frames = int(self._config.max_mel_frames)
            if mel.shape[2] > max_mel_frames:
                raise ValueError(
                    f"Audio too long: {mel.shape[2]} frames (~{mel.shape[2]*0.01:.0f}s). "
                    f"Max {max_mel_frames} frames (~{max_mel_frames*0.01:.0f}s). Split into smaller chunks."
                )

            mel_fp16 = mel.astype(np.float16)
            mel_path = os.path.join(tmpdir, "mel.safetensors")
            write_safetensors(mel_fp16, self._config.mel_tensor_name, mel_path)
            elapsed_mel_s = time.time() - mel_t0
            logger.info(
                "Mel computed: shape=%s size=%s -> %s",
                list(mel_fp16.shape),
                mel_fp16.nbytes,
                mel_path,
            )

            if self._use_worker():
                return self._transcribe_worker(mel_path, elapsed_mel_s)

            input_data = {
                "requests": [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "audio", "audio": mel_path}],
                            }
                        ],
                    }
                ],
                "batch_size": 1,
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k,
                "max_generate_length": self._config.max_generate_length,
                "apply_chat_template": True,
                "add_generation_prompt": True,
            }

            input_path = os.path.join(tmpdir, "input.json")
            with open(input_path, "w") as f:
                json.dump(input_data, f)

            output_path = os.path.join(tmpdir, "output.json")
            cli_args = [
                "--engineDir", self._config.engine_dir,
                "--multimodalEngineDir", self._config.audio_encoder_dir,
                "--inputFile", input_path,
                "--outputFile", output_path,
            ]

            t0 = time.time()
            result = run_binary(
                self._config.asr_binary, cli_args, timeout=60,
                plugin_path=self._config.plugin_path,
            )
            elapsed = time.time() - t0

            if result.returncode != 0 or not os.path.exists(output_path):
                raise RuntimeError(
                    f"ASR subprocess failed (exit={result.returncode}): "
                    f"stdout={result.stdout[-300:]}, stderr={result.stderr[-300:]}"
                )

            with open(output_path) as f:
                output_data = json.load(f)

            responses = output_data.get("responses", [])
            if not responses:
                raise RuntimeError(f"ASR produced no responses: {output_data}")

            r = responses[0]
            text = r.get("output_text", "")
            if text == "TensorRT Edge LLM cannot handle this request. Fails.":
                raise RuntimeError(f"ASR inference failed (model returned error): {r}")

            text, language_detected = self._strip_language_prefix(text)
            return TranscriptionResult(
                text=text,
                language=language_detected,
                meta={"inference_time_s": round(elapsed, 3)},
            )

    def _transcribe_segmented_offline(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> TranscriptionResult:
        """Split long offline WAV uploads before sending them to the worker."""
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = max(1, int(round(len(audio) * ratio)))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
            sample_rate = 16000

        original_duration_s = len(audio) / sample_rate
        segments = _split_offline_audio(
            audio,
            sample_rate,
            max_segment_s=self._config.offline_segment_threshold_s,
        )
        texts: list[str] = []
        last_language = language
        total_inference_s = 0.0
        total_mel_s = 0.0
        total_worker_s = 0.0
        failed_segments = 0
        min_seg_s = self._config.offline_segment_min_s

        for seg in segments:
            seg_duration_s = len(seg) / sample_rate
            if seg_duration_s < min_seg_s:
                continue
            wav_bytes = _float_audio_to_wav_bytes(seg, sample_rate)
            try:
                result = self.transcribe(wav_bytes, language=language)
            except Exception as exc:
                failed_segments += 1
                logger.warning(
                    "TRT-EdgeLLM ASR offline segment failed (%.1fs): %s",
                    seg_duration_s,
                    exc,
                )
                continue
            if result.text:
                texts.append(result.text)
            last_language = result.language or last_language
            meta = result.meta or {}
            total_inference_s += float(meta.get("inference_time_s", 0.0) or 0.0)
            total_mel_s += float(meta.get("mel_time_s", 0.0) or 0.0)
            total_worker_s += float(meta.get("worker_time_s", 0.0) or 0.0)

        return TranscriptionResult(
            text=_join_segment_texts(texts, last_language or language),
            language=last_language,
            meta={
                "segmented": True,
                "segment_count": len(segments),
                "failed_segments": failed_segments,
                "original_duration_s": round(original_duration_s, 3),
                "inference_time_s": round(total_inference_s, 3),
                "mel_time_s": round(total_mel_s, 3),
                "worker_time_s": round(total_worker_s, 3),
                "worker_init_ms": round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
            },
        )

    def create_stream(self, language: str = "auto") -> ASRStream:
        """Accumulate stream audio and run the resident worker on finalize."""
        if not self._ready:
            raise RuntimeError("ASR backend not preloaded")
        if self._use_streaming_worker():
            return _TRTEdgeLLMStreamingASRStream(self, language=language)
        return _TRTEdgeLLMAccumulatingASRStream(self, language=language)


def _float_audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return out.getvalue()


def _wav_bytes_to_float_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), sample_rate


def _split_offline_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_segment_s: float,
) -> list[np.ndarray]:
    """Split long offline audio at silence (webrtcvad → energy fallback).

    The production version first tried a profile-configured VAD backend
    (``app.core.vad``); voxedge ships no VAD backend so it goes straight to
    the env-free webrtcvad→energy splitter cascade.
    """
    try:
        segments = _split_at_silence_vad(audio, sample_rate)
    except ImportError:
        segments = _split_at_silence_energy(audio, sample_rate)
    except Exception as exc:
        logger.warning("TRT-EdgeLLM ASR offline splitter failed: %s", exc)
        segments = [audio]

    max_samples = max(1, int(max_segment_s * sample_rate))
    bounded: list[np.ndarray] = []
    for seg in segments:
        if len(seg) <= max_samples:
            bounded.append(seg)
            continue
        for start in range(0, len(seg), max_samples):
            bounded.append(seg[start:start + max_samples])
    return [seg for seg in bounded if len(seg) > 0]


def _join_segment_texts(texts: list[str], language: str | None) -> str:
    texts = [text.strip() for text in texts if text and text.strip()]
    if not texts:
        return ""
    if len(texts) > 1:
        trail_punct = "。，、！？；,.!?;"
        texts = [text.rstrip(trail_punct).rstrip() for text in texts[:-1]] + [texts[-1]]
    cjk_langs = {"Chinese", "Japanese", "Korean", "Cantonese", "zh", "ja", "ko"}
    lang = language or ""
    is_cjk = lang in cjk_langs or any(lang.startswith(prefix) for prefix in ("zh", "ja", "ko"))
    return ("" if is_cjk else " ").join(texts).strip()


class _TRTEdgeLLMAccumulatingASRStream(ASRStream):
    def __init__(self, backend: TRTEdgeLLMASRBackend, language: str = "auto"):
        self._backend = backend
        self._language = language
        self._chunks: list[np.ndarray] = []
        self._cancelled = False
        self._final_text_cache = ""

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if self._cancelled:
            return
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(len(samples) * ratio)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)
        self._chunks.append(samples.copy())

    def cancel_and_finalize(self) -> None:
        if self._cancelled:
            return
        self._final_text_cache = ""
        self._cancelled = True
        self._chunks = []

    def finalize(self) -> tuple[str, Optional[str]]:
        if self._cancelled:
            return self._final_text_cache, None
        if not self._chunks:
            return "", None
        audio = np.concatenate(self._chunks)
        # The TRT audio_encoder engine is built with a fixed optimization
        # profile; forwarding >10s in one shot makes the worker reject the
        # request. Split at natural silence (webrtcvad → energy fallback),
        # then concatenate per-segment transcripts.
        try:
            segments = _split_at_silence_vad(audio)
        except ImportError:
            segments = _split_at_silence_energy(audio)
        except Exception:
            segments = [audio]

        MIN_SEG_S = 0.4
        texts: list[str] = []
        detected_language: Optional[str] = None
        for seg in segments:
            if len(seg) / 16000 < MIN_SEG_S:
                continue
            wav_bytes = _float_audio_to_wav_bytes(seg, 16000)
            try:
                result = self._backend.transcribe(wav_bytes, language=self._language)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "TRT-EdgeLLM ASR segment failed (%.1fs): %s",
                    len(seg) / 16000, e,
                )
                continue
            if result.text:
                texts.append(result.text)
                if detected_language is None and getattr(result, "language", None):
                    detected_language = result.language

        cjk_langs = {"Chinese", "Japanese", "Korean", "Cantonese", "zh", "ja", "ko"}
        is_cjk = self._language in cjk_langs
        if len(texts) > 1:
            trail_punct = "。，、！？；,.!?;"
            cleaned: list[str] = []
            for i, t in enumerate(texts):
                if i < len(texts) - 1:
                    cleaned.append(t.rstrip(trail_punct).rstrip())
                else:
                    cleaned.append(t)
            texts = cleaned
        separator = "" if is_cjk else " "
        return separator.join(texts).strip(), detected_language

    def get_partial(self) -> tuple[str, bool]:
        return "", False


class _TRTEdgeLLMStreamingASRStream(ASRStream):
    """TRT-EdgeLLM qwen3_asr_worker streaming protocol adapter.

    Enabled only with stream_mode=worker. The worker receives cumulative
    float32 PCM via ``pcm_b64`` and emits partial/final JSON events.
    """

    def __init__(self, backend: TRTEdgeLLMASRBackend, language: str = "auto"):
        self._backend = backend
        self._language = language
        self._session_id = uuid.uuid4().hex
        self._sample_rate = 16000
        self._hop_samples = max(
            1, int(float(backend._config.stream_chunk_sec) * self._sample_rate)
        )
        self._audio_accum = np.zeros(0, dtype=np.float32)
        self._samples_since_hop = 0
        self._partial_text = ""
        self._final_text = ""
        self._detected_language: Optional[str] = None
        self._cancelled = False
        self._closed = False
        self._begin()

    def _begin(self) -> None:
        ev = {
            "event": "begin",
            "id": self._session_id,
            "sample_rate": self._sample_rate,
            "chunk_size_sec": float(self._backend._config.stream_chunk_sec),
            "unfixed_chunk_num": int(self._backend._config.stream_unfixed_chunks),
            "unfixed_token_num": int(self._backend._config.stream_unfixed_tokens),
            "context": "",
        }
        if self._language and self._language != "auto":
            ev["force_language"] = self._language
        resp = self._backend._worker_request(ev)
        if resp.get("event") != "begin_ack":
            raise RuntimeError(f"ASR streaming worker begin failed: {resp}")

    def _send_chunk(self, *, last: bool) -> dict:
        pcm = np.asarray(self._audio_accum, dtype="<f4")
        pcm_b64 = base64.b64encode(pcm.tobytes()).decode("ascii")
        resp = self._backend._worker_request({
            "event": "chunk",
            "id": self._session_id,
            "pcm_b64": pcm_b64,
            "audio_sec": len(self._audio_accum) / self._sample_rate,
            "last": last,
        })
        event = resp.get("event")
        if event == "segment_rotation":
            carry_samples = int(float(resp.get("carryover_sec", 1.0)) * self._sample_rate)
            if carry_samples > 0 and len(self._audio_accum) > carry_samples:
                self._audio_accum = self._audio_accum[-carry_samples:].copy()
            return resp
        if event == "partial":
            stripped, lang = self._backend._strip_language_prefix(resp.get("text", "") or "")
            self._partial_text = stripped.strip()
            if lang:
                self._detected_language = lang
            return resp
        if event == "final":
            stripped, lang = self._backend._strip_language_prefix(resp.get("text", "") or "")
            self._final_text = stripped.strip()
            if lang:
                self._detected_language = lang
            self._closed = True
            return resp
        raise RuntimeError(f"unexpected ASR streaming worker event: {resp}")

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if self._cancelled or self._closed:
            return
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if sample_rate != self._sample_rate:
            ratio = self._sample_rate / sample_rate
            new_len = int(len(samples) * ratio)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)
        self._audio_accum = np.concatenate([self._audio_accum, samples])
        self._samples_since_hop += len(samples)
        while self._samples_since_hop >= self._hop_samples:
            self._send_chunk(last=False)
            self._samples_since_hop -= self._hop_samples

    def prepare_finalize(self) -> None:
        pass

    def finalize(self) -> tuple[str, Optional[str]]:
        if self._cancelled or self._closed:
            return self._final_text, self._detected_language
        if len(self._audio_accum) == 0:
            self._backend._worker_request({"event": "end", "id": self._session_id})
            self._closed = True
            return "", self._detected_language
        self._send_chunk(last=True)
        return self._final_text, self._detected_language

    def cancel_and_finalize(self) -> None:
        self._final_text = self._partial_text
        self._cancelled = True
        # Send the `end` event but bound the wait to 500ms; if the worker is
        # unresponsive raise WorkerExitError so the caller can restart_worker().
        import concurrent.futures as _cf

        pool = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-cancel")
        try:
            fut = pool.submit(
                self._backend._worker_request,
                {"event": "end", "id": self._session_id},
            )
            try:
                fut.result(timeout=0.5)
            except _cf.TimeoutError:
                self._closed = True
                pool.shutdown(wait=False)
                raise WorkerExitError(
                    f"ASR worker did not ack 'end' for session {self._session_id} within 500ms"
                )
            except WorkerProtocolError:
                self._closed = True
                raise
            except Exception:
                pass
            self._closed = True
        finally:
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass

    def get_partial(self) -> tuple[str, bool]:
        if self._closed:
            return self._final_text, True
        return self._partial_text, False
