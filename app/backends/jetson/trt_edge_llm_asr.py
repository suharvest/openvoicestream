"""ASR backend via TRT-Edge-LLM C++ binary (llm_inference).

Audio is converted to a Whisper-compatible log-mel spectrogram in Python
(scipy + numpy, no librosa), saved as a safetensors file, and passed to the
LLM binary via ``--multimodalEngineDir`` for the audio encoder.

Supports: OFFLINE, MULTI_LANGUAGE
Streaming: planned (Phase 2, requires llm_stream binary).
"""

from __future__ import annotations

import json
import logging
import os
import base64
import subprocess
import tempfile
import threading
import time
import uuid
import wave
import io
from collections import deque
from typing import Optional

import numpy as np

from app.core.asr_backend import ASRBackend, ASRCapability, ASRStream, TranscriptionResult
from app.core.worker_io import WorkerIO, WorkerExitError as _WIOExitError

from app.backends.jetson.trt_edge_llm_ipc import (
    ASR_BINARY,
    ASR_WORKER_BINARY,
    ASR_ENGINE_DIR,
    ASR_AUDIO_ENC_DIR,
    ASR_PLUGIN_PATH,
    audio_bytes_to_mel,
    run_binary,
    write_safetensors,
)

logger = logging.getLogger(__name__)

# Sampling defaults — read fresh per backend instance via _load_config().
# The module-level fallbacks below are kept for the rare external importer
# (none currently) and historical introspection; ALL request-time code must
# pull from self._config so hot reload of ASR_TEMPERATURE etc. is honored.
_DEFAULT_MAX_GENERATE_LENGTH = 200
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_TOP_P = 1.0
_DEFAULT_TOP_K = 1


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no")


def _offline_segment_enabled() -> bool:
    return _env_bool("EDGE_LLM_ASR_OFFLINE_SEGMENT", True)


def _offline_segment_threshold_s() -> float:
    return float(os.environ.get("EDGE_LLM_ASR_OFFLINE_SEGMENT_SEC", "6.0"))


def _offline_segment_min_s() -> float:
    return float(os.environ.get("EDGE_LLM_ASR_OFFLINE_MIN_SEGMENT_SEC", "0.4"))


def _default_offline_vad_backend() -> str:
    return (
        os.environ.get("OVS_VAD_BACKEND")
        or "silero"
    ).strip() or "silero"


def _default_offline_vad_silence_ms() -> int:
    raw = os.environ.get("OVS_VAD_SILENCE_MS")
    try:
        return int(raw) if raw else 400
    except ValueError:
        return 400


class WorkerProtocolError(RuntimeError):
    """Base class for ASR worker protocol-level errors.

    Distinct from generic RuntimeError so the ASRSessionManager can route
    these into its ERROR_REBUILD recovery path while leaving unrelated
    failures unhandled.
    """


class NoActiveSessionError(WorkerProtocolError):
    """Worker reported there is no active session (stale id / double-end)."""


class SessionAlreadyActiveError(WorkerProtocolError):
    """Worker reported a session is already active for the given id."""


class WorkerExitError(WorkerProtocolError):
    """Worker subprocess exited or didn't respond before the ack deadline."""


class PoolSaturatedError(RuntimeError):
    """Worker rejected a begin because every slot in its pool is busy.

    The C++ ``qwen3_asr_worker`` with ``--max_slots N`` returns
    ``{"error":"pool_saturated","status":4429,"max_slots":N}`` (also accepts
    the legacy ``too_many_asr_sessions`` wording) when an N+1st distinct
    session id tries to begin. This is the backend-internal fail-fast that
    backstops the Python ``WorkerIO`` semaphore admission gate; OVS callers
    map it to a 429/4429 busy response (reject-not-queue, spec §4).

    NB: intentionally NOT a ``WorkerProtocolError`` subclass. The
    ``ASRSessionManager`` duck-types worker-protocol errors by class-name MRO
    (``asr_session_manager._WORKER_ERROR_NAMES``) and routes them into the
    ERROR_REBUILD path (kill + respawn the worker). A saturation is not a
    worker fault — rebuilding would tear down the other N-1 healthy sessions.
    Keeping this off the ``WorkerProtocolError`` lineage makes it fail fast
    (busy) rather than trigger a destructive restart. The ``status`` field is
    carried so the WS/HTTP layer can surface the 4429/429 reject-not-queue
    response.
    """

    status: int = 4429

    def __init__(self, message: str, max_slots: Optional[int] = None) -> None:
        super().__init__(message)
        self.max_slots = max_slots


def _classify_worker_response(output_data: dict, *, request_event: str | None = None) -> WorkerProtocolError | None:
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
    # Slot-pool saturation: the C++ worker (--max_slots N) returns a typed
    # busy payload when an N+1st distinct session id tries to begin. Recognize
    # it by the structured ``status: 4429`` field OR the error wording so a
    # fast-fail busy rejection propagates as PoolSaturatedError (not a generic
    # WorkerProtocolError that the session manager would treat as a rebuild).
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
    """ASR via TRT-Edge-LLM llm_inference subprocess."""

    # supports_hot_reload: True when running in worker subprocess mode
    # (kill+respawn releases all GPU memory cleanly). When use_worker=False
    # the engine is held in-process and we have no clean release path.
    @property
    def supports_hot_reload(self) -> bool:  # type: ignore[override]
        return self._use_worker()

    @classmethod
    def concurrency_capability(cls, profile=None):
        """Declare the ASR slot-pool ceiling (Phase D-1 Step 4).

        The backend multiplexes N distinct ASR sessions over one C++ worker
        subprocess (``--max_slots N`` + WorkerIO Semaphore(N)). Mirrors the
        TTS backend's ``concurrency_capability`` shape so the unified
        ``capability_resolver`` derives the session-limiter ceiling from
        ``min(asr_n, tts_n)``.

        Resolution precedence (same source as ``_load_config``'s ``max_slots``):
          1. env ``EDGE_LLM_ASR_MAX_CONCURRENT`` (matches the worker side)
          2. profile ``asr_max_slots`` (top-level or nested ``asr.asr_max_slots``)
          3. default 1 (legacy single-session → N=1 safe, no behavior change)
        """
        from app.core.concurrency_capability import ConcurrencyCapability

        env_val = os.environ.get("EDGE_LLM_ASR_MAX_CONCURRENT")
        profile_val = None
        if isinstance(profile, dict):
            profile_val = profile.get("asr_max_slots")
            if profile_val is None:
                cfg = profile.get("asr")
                if isinstance(cfg, dict):
                    profile_val = cfg.get("asr_max_slots", cfg.get("max_concurrent"))
        try:
            n = int(env_val) if env_val is not None else (
                int(profile_val) if profile_val is not None else 1
            )
        except (TypeError, ValueError):
            n = 1
        n = max(1, n)
        return ConcurrencyCapability(
            supports_parallel=n > 1,
            max_concurrent=n,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="single_runtime_multiplex",
        )

    def __init__(self):
        self._config = self._load_config()
        self._ready = False
        self._worker: Optional[subprocess.Popen] = None
        # ``_worker_lock`` guards the lifecycle gate (``_ensure_worker``)
        # only — it no longer wraps the request stdin/stdout cycle.
        # Per the WorkerIO migration (capability-followups #5), request
        # demux is delegated to ``self._wio`` (single ``_stdin_lock`` +
        # daemon reader thread). This mirrors the TTS backend shape and
        # keeps the C++ ``qwen3_asr_worker`` single-session contract
        # (the worker still rejects a second concurrent session).
        self._worker_lock = threading.Lock()
        # Separate lock so restart_worker() can preempt a slow spawn
        # without blocking on _worker_lock. Post-WorkerIO migration,
        # request threads no longer hold _worker_lock during IO — they
        # block on WorkerIO's per-request queue.get instead. The lock
        # remaining here is purely the spawn gate.
        self._restart_lock = threading.Lock()
        self._worker_ready_meta: dict = {}
        self._worker_stderr_tail: deque[str] = deque(maxlen=80)
        # Slot-pool admission ceiling (Phase D-1 Step 4). Captured per-instance
        # from the current env/manifest so a fresh backend built after profile
        # reload honors the new value. Default 1 == legacy single-session.
        self._max_slots: int = int(self._config.get("max_slots", 1))
        # WorkerIO multiplexer — bound to ``self._worker`` on each spawn.
        # Concurrency = ``self._max_slots`` to match the C++ worker's
        # ``--max_slots N`` decoder slot pool. The WorkerIO semaphore is what
        # admits up to N concurrent callers in the Python layer; the worker's
        # ``pool_saturated`` (status 4429) is the backstop for the N+1st
        # begin. With max_slots=1 this is Semaphore(1) — byte-equivalent to
        # the previous single-session serialization.
        self._wio: Optional[WorkerIO] = None

    def _load_config(self) -> dict:
        manifest: dict = {}
        manifest_path = os.environ.get("EDGE_LLM_ASR_MANIFEST")
        if manifest_path:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        use_worker_default = bool(manifest.get("use_worker", True))
        return {
            "asr_binary": os.environ.get(
                "EDGE_LLM_ASR_BIN", manifest.get("asr_binary", ASR_BINARY)
            ),
            "worker_binary": os.environ.get(
                "EDGE_LLM_ASR_WORKER_BIN",
                manifest.get("worker_binary", ASR_WORKER_BINARY),
            ),
            "plugin_path": os.environ.get(
                "EDGE_LLM_ASR_PLUGIN_PATH",
                os.environ.get(
                    "EDGELLM_ASR_PLUGIN_PATH",
                    manifest.get(
                        "asr_plugin_path",
                        manifest.get("plugin_path", ASR_PLUGIN_PATH),
                    ),
                ),
            ),
            "engine_dir": os.environ.get(
                "EDGE_LLM_ASR_ENGINE_DIR", manifest.get("engine_dir", ASR_ENGINE_DIR)
            ),
            "audio_encoder_dir": os.environ.get(
                "EDGE_LLM_ASR_AUDIO_ENC_DIR",
                manifest.get("audio_encoder_dir", ASR_AUDIO_ENC_DIR),
            ),
            "use_worker": _env_bool("EDGE_LLM_ASR_WORKER", use_worker_default),
            "mel_tensor_name": os.environ.get(
                "EDGE_LLM_ASR_MEL_TENSOR_NAME",
                manifest.get("mel_tensor_name", "mel"),
            ),
            "max_mel_frames": int(
                os.environ.get(
                    "EDGE_LLM_ASR_MAX_MEL_FRAMES",
                    str(manifest.get("max_mel_frames", 6000)),
                )
            ),
            # ASR slot-pool admission ceiling (Phase D-1 Step 4). Default 1 ==
            # legacy single-session behavior. env name matches the C++ worker
            # side (EDGE_LLM_ASR_MAX_CONCURRENT); profile field is
            # ``asr_max_slots``. The worker is launched with ``--max_slots N``
            # and the WorkerIO semaphore is bound to the same N, so a fast-fail
            # rejection fires before a saturated slot pool is ever hit.
            "max_slots": max(
                1,
                int(
                    os.environ.get(
                        "EDGE_LLM_ASR_MAX_CONCURRENT",
                        str(manifest.get("asr_max_slots", manifest.get("max_concurrent", 1))),
                    )
                ),
            ),
            "stream_mode": os.environ.get(
                "EDGE_LLM_ASR_STREAM_MODE",
                manifest.get("stream_mode", "accumulate"),
            ).strip().lower(),
            "stream_chunk_sec": float(
                os.environ.get(
                    "EDGE_LLM_ASR_STREAM_CHUNK_SEC",
                    str(manifest.get("stream_chunk_sec", 0.5)),
                )
            ),
            "stream_unfixed_chunks": int(
                os.environ.get(
                    "EDGE_LLM_ASR_STREAM_UNFIXED_CHUNKS",
                    str(manifest.get("stream_unfixed_chunks", 2)),
                )
            ),
            "stream_unfixed_tokens": int(
                os.environ.get(
                    "EDGE_LLM_ASR_STREAM_UNFIXED_TOKENS",
                    str(manifest.get("stream_unfixed_tokens", 5)),
                )
            ),
            "mel_settings_path": os.environ.get(
                "EDGE_LLM_ASR_MEL_SETTINGS",
                manifest.get("mel_settings_path", ""),
            ),
            "mel_filters_path": os.environ.get(
                "EDGE_LLM_ASR_MEL_FILTERS",
                manifest.get("mel_filters_path", ""),
            ),
            # Sampling defaults — captured per-instance so a fresh backend
            # built after profile reload (BackendManager rebuilds on every
            # apply_profile) honors the new ASR_TEMPERATURE / ASR_TOP_P /
            # ASR_TOP_K / ASR_MAX_GENERATE_LENGTH values.
            "temperature": float(os.environ.get("ASR_TEMPERATURE", "1.0")),
            "top_p": float(os.environ.get("ASR_TOP_P", "1.0")),
            "top_k": int(os.environ.get("ASR_TOP_K", "1")),
            "max_generate_length": int(os.environ.get("ASR_MAX_GENERATE_LENGTH", "200")),
            "manifest_path": manifest_path,
        }

    # -- ASRBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.MULTI_LANGUAGE, ASRCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        """Verify all required files exist (auto-download missing artifacts if enabled)."""
        worker_binary = self._config["worker_binary"]
        asr_binary = self._config["asr_binary"]
        plugin_path = self._config["plugin_path"]
        engine_dir = self._config["engine_dir"]
        audio_encoder_dir = self._config["audio_encoder_dir"]
        required = [
            (worker_binary if self._use_worker() else asr_binary, "ASR binary"),
            (plugin_path, "TRT-Edge-LLM plugin"),
            (os.path.join(engine_dir, "config.json"), "LLM config"),
            (os.path.join(engine_dir, "llm.engine"), "LLM engine"),
            (os.path.join(
                audio_encoder_dir, "audio", "config.json"
            ), "audio encoder config"),
            (os.path.join(
                audio_encoder_dir, "audio", "audio_encoder.engine"
            ), "audio encoder engine"),
        ]
        missing = [(path, label) for path, label in required if not os.path.exists(path)]
        if missing:
            # Switching to a jetson-qwen3asr-* profile on a fresh image leaves
            # the engine dirs empty — auto-download them so the user doesn't
            # have to know to run deploy_qwen3_artifacts.py manually.
            try:
                from app.core.qwen3_artifact_downloader import ensure_artifacts
                ensure_artifacts([p for p, _ in missing])
            except Exception:
                logger.exception(
                    "Qwen3 artifact auto-download raised; will report original "
                    "missing files below"
                )
            missing = [(p, l) for p, l in missing if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "ASR preload failed — missing:\n  "
                + "\n  ".join(f"{l}: {p}" for p, l in missing)
            )
        self._require_streaming_worker_assets()

        logger.info(
            "ASR backend preload OK (config=%s)",
            self._config,
        )
        if self._use_worker():
            self._ensure_worker()
        self._ready = True
        if self._use_worker():
            self._warm_worker()

    def unload(self) -> None:
        """Kill the resident ASR worker subprocess to fully release GPU memory.

        Mirrors TRTEdgeLLMTTSBackend.unload(). Idempotent; safe to call from
        BackendManager.reload() rollback. The actual SIGKILL+pipe-close happens
        in restart_worker(); we just mark not-ready afterward.
        """
        if not self._ready and self._worker is None:
            return
        try:
            self.restart_worker()
        except Exception:
            logger.exception("TRTEdgeLLMASRBackend.unload failed; continuing")
        finally:
            self._ready = False

    def _warm_worker(self) -> None:
        """Pre-warm TRT audio_encoder optimization profile for batch shapes 1..N.

        The audio_encoder engine has a single optimization profile with
        dynamic batch dim [1..60]. TRT silently re-selects tactics + reallocs
        workspace the FIRST time each unique batch shape is seen, costing
        ~3-4× extra ms on that call. Without pre-warming, a `long_audio →
        short_audio` switch in production triggers tactic recompilation
        ~once per unique short-audio length (3s, 4s, 3.5s, ...) and the user
        feels several extra seconds of latency.

        We pre-warm 1..12 seconds (batch=1..12) which covers >99% of expected
        utterances. Boot cost ~1-3s, RAM ~80MB. Bypass with env vars below.
        Set EDGE_LLM_ASR_PREWARM_MAX to extend coverage (max 60).
        """
        if os.environ.get("SKIP_ASR_WARMUP", "").lower() in ("1", "true", "yes"):
            logger.info("TRT-EdgeLLM ASR worker warmup skipped (SKIP_ASR_WARMUP set).")
            return
        if os.environ.get("EDGE_LLM_ASR_WORKER_WARMUP", "1").lower() in ("0", "false", "no"):
            logger.info("TRT-EdgeLLM ASR worker warmup skipped.")
            return
        # Default max = 6 seconds, covering production calls. The streaming
        # finalize path splits long audio at 4.5s, so worker never sees
        # batches > 5 sec in real traffic. The audio encoder engine starts
        # rejecting (attention_mask profile max ~780×780 ≈ 9s) somewhere
        # between batch=9 and batch=10, so warming above 9 is wasted anyway.
        try:
            prewarm_max = int(os.environ.get("EDGE_LLM_ASR_PREWARM_MAX", "6"))
        except ValueError:
            prewarm_max = 6
        prewarm_max = max(1, min(prewarm_max, 60))
        import time as _time
        t0 = _time.monotonic()
        warmed = 0
        for seconds in range(1, prewarm_max + 1):
            try:
                silence = np.zeros(16000 * seconds, dtype=np.float32)
                self.transcribe(_float_audio_to_wav_bytes(silence, 16000))
                warmed += 1
            except Exception as exc:
                # The encoder profile has a hard max shape; hitting it is the
                # natural stop signal, not a real warning. Log as INFO.
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
                break  # larger shapes will also fail
        elapsed = _time.monotonic() - t0
        logger.info(
            "TRT-EdgeLLM ASR worker pre-warmed shapes 1..%d in %.1fs", warmed, elapsed
        )

    def _use_worker(self) -> bool:
        return bool(self._config["use_worker"])

    def _use_streaming_worker(self) -> bool:
        return self._config.get("stream_mode") in (
            "worker", "stream", "streaming", "chunk_confirm", "prefix"
        )

    def _require_streaming_worker_assets(self) -> None:
        if not self._use_streaming_worker():
            return
        missing = []
        if not self._use_worker():
            missing.append("EDGE_LLM_ASR_WORKER=1 is required for streaming worker mode")
        for key, label in (
            ("mel_settings_path", "EDGE_LLM_ASR_MEL_SETTINGS"),
            ("mel_filters_path", "EDGE_LLM_ASR_MEL_FILTERS"),
        ):
            path = self._config.get(key) or ""
            if not path or not os.path.exists(path):
                missing.append(f"{label}: {path or '(unset)'}")
        if missing:
            raise FileNotFoundError(
                "EDGE_LLM_ASR_STREAM_MODE=worker requires PCM mel assets:\n  "
                + "\n  ".join(missing)
            )

    def _worker_env(self) -> dict:
        env = os.environ.copy()
        env["EDGELLM_PLUGIN_PATH"] = self._config["plugin_path"]
        env.setdefault("EDGE_LLM_ASR_CUDA_GRAPH", "0")
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
            self._config["worker_binary"],
            "--engineDir",
            self._config["engine_dir"],
            "--multimodalEngineDir",
            self._config["audio_encoder_dir"],
        ]
        # Slot-pool size (Phase D-1 Step 4). The worker allocates N decoder
        # slots sharing one encoder context; the Python WorkerIO semaphore is
        # bound to the same N below. Only emit --max_slots when N>1: at N=1 we
        # omit it so behavior is byte-equivalent to the legacy single session
        # AND so worker binaries built before --max_slots existed (older images)
        # don't reject the unknown flag.
        if self._max_slots and self._max_slots > 1:
            cmd += ["--max_slots", str(self._max_slots)]
        mel_settings = self._config.get("mel_settings_path") or ""
        mel_filters = self._config.get("mel_filters_path") or ""
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
        # Re-read the slot ceiling in case the env/manifest was applied between
        # __init__ and this (first) spawn (BackendManager applies profile env
        # before building the backend; later hot reloads come via __init__ on a
        # fresh instance — so this is mostly defensive).
        self._max_slots = max(1, int(self._config.get("max_slots", self._max_slots)))
        # Bind a fresh WorkerIO multiplexer to the new subprocess. concurrency
        # = ``self._max_slots`` matches the C++ worker's ``--max_slots N``
        # decoder slot pool (see __init__ note). With max_slots=1 this is
        # Semaphore(1) — byte-equivalent to the legacy single-session path.
        # NB: ``_ensure_worker`` reads the worker's initial ``ready`` line
        # itself (above) BEFORE handing stdout to the WorkerIO reader thread,
        # so the reader thread only sees subsequent per-request events.
        self._wio = WorkerIO(self._worker, concurrency=self._max_slots)

    def _worker_request(self, input_data: dict) -> dict:
        """Send one streaming protocol line to the worker, return its single reply.

        Streaming events emitted by ``qwen3_asr_worker`` (``begin_ack``,
        ``partial``, ``final``, ``segment_rotation``, ``chunk_ack``,
        ``end_ack``, ``error``) are one-line responses keyed by the
        request's ``id`` — exactly one event per input line. We route the
        write + read through ``WorkerIO.request()`` (so the daemon reader
        thread demuxes by ``id``) but break out of the iterator after the
        first event since none of the streaming events are the ``done``/
        ``cancelled`` terminals that ``request()`` would otherwise loop
        for. ``request()``'s finally arm still unregisters the inflight
        queue and releases the semaphore on early break.

        The same ``id`` is reused across begin → N×chunk → end, but the
        WorkerIO inflight queue is registered fresh on each ``request()``
        call (and popped in finally). For a single streaming session the
        begin/chunk/end calls are naturally serialized by the caller
        (``_TRTEdgeLLMStreamingASRStream`` issues them one at a time), so the
        per-``id`` inflight registration aligns regardless of pool size.

        Slot-pool note (Phase D-1 Step 4): WorkerIO's semaphore is now sized
        to ``self._max_slots`` (>=1), so up to N *distinct* session ids may be
        in flight concurrently. The reader thread demuxes events by ``id``, so
        interleaved begin/chunk/end lines for different sessions route to the
        correct caller. With ``max_slots=1`` the Semaphore(1) restores the
        original strict serialization (byte-equivalent legacy behavior).
        """
        req_event = input_data.get("event") if isinstance(input_data, dict) else None
        # Lifecycle gate only.
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
            # WorkerIO surfaces a stdin write failure (worker subprocess
            # already dead, pipe broken) by letting the OSError propagate
            # out of ``request()``. Translate into the legacy
            # WorkerExitError contract that ASRSessionManager recovery
            # logic depends on (commit `bf24284` multi-utterance fix).
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
        """Forcibly kill the worker subprocess so the next request rebuilds it.

        Crucially, this method MUST NOT acquire ``_worker_lock``. After the
        WorkerIO migration (capability-followups #5), ``_worker_lock`` only
        gates ``_ensure_worker`` (spawn) — request threads now block on
        ``WorkerIO``'s queue.get rather than on bare ``stdout.readline()``.
        We still avoid taking ``_worker_lock`` here because a slow spawn
        (cold start of qwen3_asr_worker is multiple seconds) holds it and
        we want restart_worker() to be non-blocking from the caller's POV.

        ``wio.close()`` (below) wakes any in-flight callers waiting on
        ``WorkerIO``'s per-request queues with the ``_worker_exit``
        sentinel; they surface ``WorkerExitError`` and unwind cleanly.

        Concurrent callers collapse to a single restart via
        ``_restart_lock`` (cheap, doesn't block on IO).
        """
        # Snapshot WITHOUT _worker_lock — see docstring. _restart_lock
        # only serializes concurrent restarts.
        with self._restart_lock:
            worker = self._worker
            if worker is None:
                return
            # Clear our reference first so other threads that wake from
            # WorkerIO's ``_worker_exit`` sentinel see ``_worker = None``
            # and don't try to talk to a dead pipe.
            self._worker = None
            self._worker_ready_meta = {}
            # Drop the WorkerIO multiplexer first so its daemon reader
            # thread sees EOF cleanly when the subprocess is killed, and
            # any in-flight callers wake with the ``_worker_exit`` sentinel
            # raising ``WorkerExitError`` rather than hanging on q.get.
            wio = self._wio
            self._wio = None
            if wio is not None:
                try:
                    wio.close()
                except Exception:
                    logger.debug("WorkerIO.close() during restart raised", exc_info=True)
            try:
                if worker.poll() is None:
                    # Use kill() directly (SIGKILL on POSIX) so a wedged
                    # worker can't ignore SIGTERM. This force-closes its
                    # stdout pipe; the WorkerIO daemon reader thread sees
                    # EOF and (defensively) wakes any remaining inflight
                    # callers — though ``wio.close()`` above already did
                    # so explicitly via the ``_worker_exit`` sentinel.
                    try:
                        worker.kill()
                    except Exception:
                        pass
                    try:
                        worker.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                # Close pipes after kill for hygiene. WorkerIO's reader
                # thread will exit on stdout EOF; this is purely defensive.
                # close() on a stdout already EOF'd is a no-op.
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
                    text = text[len(prefix) :].lstrip()
                    break
            else:
                # No known language matched. The model may emit "language None"
                # (no trailing text, hallucinated as a bailout for silent/noise
                # segments) or "language Xxx <text>". Find the trailing space:
                #   - found → strip "language <Xxx> " prefix
                #   - missing → entire string is just "language <Xxx>", drop it
                space = text.find(" ", 9)
                if space > 0:
                    language_detected = text[9:space]
                    text = text[space + 1 :].lstrip()
                else:
                    language_detected = text[9:]   # e.g. "None"
                    text = ""                      # empty out the bailout
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
                            "content": [
                                {
                                    "type": "audio",
                                    "audio": mel_path,
                                }
                            ],
                        }
                    ],
                }
            ],
            "batch_size": 1,
            "temperature": self._config["temperature"],
            "top_p": self._config["top_p"],
            "top_k": self._config["top_k"],
            "max_generate_length": self._config["max_generate_length"],
            "apply_chat_template": True,
            "add_generation_prompt": True,
        }
        # Lifecycle gate only — request demux is delegated to WorkerIO.
        with self._worker_lock:
            self._ensure_worker()
            wio = self._wio
        assert wio is not None
        t0 = time.time()
        try:
            # The legacy one-shot transcribe path emits exactly one terminal
            # event (``done`` on success, ``error`` on failure) with the
            # matching ``id``. ``wio.request()`` loops until ``done`` /
            # ``cancelled``, which aligns naturally here.
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
            inference_time_s=round(total_s, 3),
            mel_time_s=round(elapsed_mel_s, 3),
            worker_time_s=round(elapsed_worker, 3),
            worker_init_ms=round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
        )

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe audio via subprocess.

        Workflow:
          1. Write incoming audio to a temp WAV file.
          2. Compute log-mel spectrogram (numpy+scipy).
          3. Save mel as FP16 safetensors.
          4. Build input JSON referencing the mel file.
          5. Run ``llm_inference --multimodalEngineDir ...``.
          6. Parse output JSON for transcribed text.
        """
        if not self._ready:
            raise RuntimeError("ASR backend not preloaded")

        if _offline_segment_enabled():
            try:
                audio, sample_rate = _wav_bytes_to_float_audio(audio_bytes)
                duration_s = len(audio) / max(sample_rate, 1)
            except Exception:
                audio = None
                sample_rate = 16000
                duration_s = 0.0
            if audio is not None and duration_s > _offline_segment_threshold_s():
                return self._transcribe_segmented_offline(audio, sample_rate, language)

        with tempfile.TemporaryDirectory(
            prefix="trt_edgellm_asr_"
        ) as tmpdir:
            # -- 1. Compute mel spectrogram (with duration guard) --
            mel_t0 = time.time()
            mel = audio_bytes_to_mel(audio_bytes)  # [1, 128, T] float32
            max_mel_frames = int(self._config["max_mel_frames"])
            if mel.shape[2] > max_mel_frames:  # 10ms hop
                raise ValueError(
                    f"Audio too long: {mel.shape[2]} frames (~{mel.shape[2]*0.01:.0f}s). "
                    f"Max {max_mel_frames} frames (~{max_mel_frames*0.01:.0f}s). Split into smaller chunks."
                )

            # Convert to FP16 for TRT
            mel_fp16 = mel.astype(np.float16)

            mel_path = os.path.join(tmpdir, "mel.safetensors")
            write_safetensors(mel_fp16, self._config["mel_tensor_name"], mel_path)
            elapsed_mel_s = time.time() - mel_t0
            logger.info(
                "Mel computed: shape=%s size=%s -> %s",
                list(mel_fp16.shape),
                mel_fp16.nbytes,
                mel_path,
            )

            if self._use_worker():
                return self._transcribe_worker(mel_path, elapsed_mel_s)

            # -- 2. Build input JSON --
            input_data = {
                "requests": [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "audio",
                                        "audio": mel_path,
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "batch_size": 1,
                "temperature": self._config["temperature"],
                "top_p": self._config["top_p"],
                "top_k": self._config["top_k"],
                "max_generate_length": self._config["max_generate_length"],
                "apply_chat_template": True,
                "add_generation_prompt": True,
            }

            input_path = os.path.join(tmpdir, "input.json")
            with open(input_path, "w") as f:
                json.dump(input_data, f)

            output_path = os.path.join(tmpdir, "output.json")

            # -- 3. Run binary --
            cli_args = [
                "--engineDir",
                self._config["engine_dir"],
                "--multimodalEngineDir",
                self._config["audio_encoder_dir"],
                "--inputFile",
                input_path,
                "--outputFile",
                output_path,
            ]

            t0 = time.time()
            result = run_binary(self._config["asr_binary"], cli_args, timeout=60)
            elapsed = time.time() - t0

            # -- 4. Parse output — fail loudly on errors
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
                raise RuntimeError(
                    f"ASR inference failed (model returned error): {r}"
                )

            text, language_detected = self._strip_language_prefix(text)

            meta = {
                "inference_time_s": round(elapsed, 3),
            }
            return TranscriptionResult(
                text=text, language=language_detected, **meta
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
            max_segment_s=_offline_segment_threshold_s(),
        )
        texts: list[str] = []
        last_language = language
        total_inference_s = 0.0
        total_mel_s = 0.0
        total_worker_s = 0.0
        failed_segments = 0
        min_seg_s = _offline_segment_min_s()

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
            total_inference_s += float(result.meta.get("inference_time_s", 0.0) or 0.0)
            total_mel_s += float(result.meta.get("mel_time_s", 0.0) or 0.0)
            total_worker_s += float(result.meta.get("worker_time_s", 0.0) or 0.0)

        return TranscriptionResult(
            text=_join_segment_texts(texts, last_language or language),
            language=last_language,
            segmented=True,
            segment_count=len(segments),
            failed_segments=failed_segments,
            original_duration_s=round(original_duration_s, 3),
            inference_time_s=round(total_inference_s, 3),
            mel_time_s=round(total_mel_s, 3),
            worker_time_s=round(total_worker_s, 3),
            worker_init_ms=round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
        )

    def create_stream(self, language: str = "auto") -> ASRStream:
        """Accumulate stream audio and run the resident worker on finalize.

        This matches the current V2V product behavior: the user has already
        stopped speaking before ASR is invoked, so partials are unnecessary.
        The resident C++ worker keeps the stop-to-final path warm.
        """
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
    configured_vad_segments = _split_offline_audio_with_configured_vad(
        audio,
        sample_rate,
        max_segment_s=max_segment_s,
    )
    if configured_vad_segments:
        return configured_vad_segments

    try:
        from app.backends.jetson.qwen3_asr import _split_at_silence_vad
        segments = _split_at_silence_vad(audio, sample_rate)
    except ImportError:
        from app.backends.jetson.qwen3_asr import _split_at_silence_energy
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


def _split_offline_audio_with_configured_vad(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_segment_s: float,
) -> list[np.ndarray]:
    backend = _default_offline_vad_backend()
    if backend.lower() in ("", "none", "off", "disabled"):
        return []
    try:
        from app.core.vad import VADSession, create_vad
        vad = create_vad(
            backend=backend,
            sample_rate=sample_rate,
            silence_ms=_default_offline_vad_silence_ms(),
        )
    except Exception as exc:
        logger.warning("Configured offline ASR VAD %r unavailable: %s", backend, exc)
        return []
    if vad is None:
        return []

    max_samples = max(1, int(max_segment_s * sample_rate))
    min_samples = max(1, int(_offline_segment_min_s() * sample_rate))
    chunk_samples = max(1, int(0.02 * sample_rate))
    cuts = [0]
    last_cut = 0

    for end in range(chunk_samples, len(audio) + chunk_samples, chunk_samples):
        chunk_end = min(end, len(audio))
        chunk = audio[end - chunk_samples:chunk_end] if end <= len(audio) else audio[end - chunk_samples:]
        if len(chunk) == 0:
            continue
        event = vad.process(chunk)
        elapsed = chunk_end - last_cut
        if event == VADSession.SPEECH_END and elapsed >= min_samples:
            cuts.append(chunk_end)
            last_cut = chunk_end
            vad.reset()
            continue
        while chunk_end - last_cut >= max_samples:
            last_cut += max_samples
            cuts.append(last_cut)
            vad.reset()

    if cuts[-1] != len(audio):
        cuts.append(len(audio))

    segments = [audio[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]
    return [seg for seg in segments if len(seg) >= min_samples]


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
        # No partials produced by this accumulate path.
        self._final_text_cache = ""
        self._cancelled = True
        self._chunks = []

    def finalize(self) -> tuple[str, Optional[str]]:
        if self._cancelled:
            return self._final_text_cache, None
        if not self._chunks:
            return "", None
        audio = np.concatenate(self._chunks)
        # The TRT audio_encoder engine is built with a fixed optimization profile
        # (~800-1500 mel frames ≈ 10-15s). Forwarding >10s of audio in one shot
        # makes the worker reject the request ("cannot handle this request").
        # Split at natural silence points using the same VAD splitter the older
        # qwen3_asr backend uses, then concatenate the per-segment transcripts.
        from app.backends.jetson.qwen3_asr import (
            _split_at_silence_vad, _split_at_silence_energy, VAD_MAX_SEG_SEC,
        )
        try:
            segments = _split_at_silence_vad(audio)
        except ImportError:
            # webrtcvad not installed on this image — fall back to energy-RMS
            segments = _split_at_silence_energy(audio)
        except Exception:
            # Any other VAD failure: be safe, single segment (will only break
            # for very long audio; short audio still works).
            segments = [audio]

        # Skip very short / near-silent segments: they tend to push the
        # Qwen3-ASR model into bailout outputs ("language None"). Threshold
        # of 0.4s drops VAD-trailing fragments without removing real content
        # — natural speech segments split by the VAD are >=1.0s by design.
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
                import logging
                logging.getLogger(__name__).warning(
                    "TRT-EdgeLLM ASR segment failed (%.1fs): %s",
                    len(seg) / 16000, e,
                )
                continue
            if result.text:
                # Drop trailing CJK/Latin sentence punctuation when more
                # segments follow — the split was mid-utterance, the model
                # over-eagerly punctuated the end. Keeps "变得 更善于"
                # instead of "变得。 更善于".
                texts.append(result.text)
                # Take the first non-empty segment's detected language as
                # the utterance language. (Mixed-language utterances are
                # rare; first-segment language is a safe default.)
                if detected_language is None and getattr(result, "language", None):
                    detected_language = result.language

        # Join: use the request language to pick separator. CJK languages
        # never use spaces; everything else does. Don't infer from output
        # content (output may contain mixed-script proper nouns).
        cjk_langs = {"Chinese", "Japanese", "Korean", "Cantonese", "zh", "ja", "ko"}
        is_cjk = self._language in cjk_langs
        # Trim segment-trailing punctuation when more text follows
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

    Enabled only with EDGE_LLM_ASR_STREAM_MODE=worker. The worker receives
    cumulative float32 PCM via `pcm_b64` and emits partial/final JSON events.
    """

    def __init__(self, backend: TRTEdgeLLMASRBackend, language: str = "auto"):
        self._backend = backend
        self._language = language
        self._session_id = uuid.uuid4().hex
        self._sample_rate = 16000
        self._hop_samples = max(
            1, int(float(backend._config["stream_chunk_sec"]) * self._sample_rate)
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
            "chunk_size_sec": float(self._backend._config["stream_chunk_sec"]),
            "unfixed_chunk_num": int(self._backend._config["stream_unfixed_chunks"]),
            "unfixed_token_num": int(self._backend._config["stream_unfixed_tokens"]),
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
            stripped, lang = self._backend._strip_language_prefix(
                resp.get("text", "") or ""
            )
            self._partial_text = stripped.strip()
            if lang:
                self._detected_language = lang
            return resp
        if event == "final":
            stripped, lang = self._backend._strip_language_prefix(
                resp.get("text", "") or ""
            )
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
        # The worker closes the session on the final chunk, so finalize() owns
        # the last=true event and returns the final text.
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
        # Send the `end` event but bound the wait to 500ms; if the worker
        # is unresponsive raise WorkerExitError so the session manager
        # can trigger restart_worker(). We must NOT use a
        # ``with ThreadPoolExecutor`` context manager here — its __exit__
        # waits for outstanding futures, so a wedged worker_request would
        # hold the cancel path forever, defeating the timeout.
        #
        # NB: post-WorkerIO migration (capability-followups #5), the
        # leaked worker thread blocks inside ``wio.request()``'s q.get
        # (60s timeout). When ``restart_worker()`` fires, ``wio.close()``
        # wakes it with the ``_worker_exit`` sentinel and it exits with
        # ``WorkerExitError`` — same end state, just unblocked sooner.
        # We do NOT use ``wio.cancel(session_id)`` here: the C++
        # ``qwen3_asr_worker`` (third_party/...qwen3_asr_worker.cpp ~L1395)
        # has no cancel-event handler; a stray ``{"type":"cancel",...}``
        # line lacking an ``event`` field would route to the legacy
        # one-shot handler and corrupt session state. Cancel semantics
        # remain unchanged from pre-migration ("end" event + Python-side
        # timeout) per the spec zero-behavior-change requirement.
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
                # Fire-and-forget shutdown; the leaked worker thread will
                # die once restart_worker() kills the subprocess (its
                # stdout will EOF and _worker_request returns/raises).
                pool.shutdown(wait=False)
                raise WorkerExitError(
                    f"ASR worker did not ack 'end' for session {self._session_id} within 500ms"
                )
            except WorkerProtocolError:
                # NoActiveSession etc. is already informative — propagate.
                self._closed = True
                raise
            except Exception:
                # Other errors are swallowed (legacy behavior).
                pass
            self._closed = True
        finally:
            # Best-effort: don't block — if the future is still running,
            # restart_worker() will unblock it shortly.
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass

    def get_partial(self) -> tuple[str, bool]:
        if self._closed:
            return self._final_text, True
        return self._partial_text, False
