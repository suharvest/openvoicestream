"""Speaker-embedding extraction — product-layer shim over voxedge.

The inference engine (sherpa-onnx CAM++ wrapper) + stateless helpers live in
``voxedge.capabilities.speaker_embedding`` and are env-free. This product-layer
module keeps the deployment concerns: the ``OVS_SPEAKER_EMB`` feature flag
(default off), model-path resolution, and lazy on-demand download (honoring
HF_ENDPOINT mirrors). Opt-in, default-OFF, lazy-loaded.

OVS is stateless — it emits the raw embedding + metadata only; matching/identity
lives on the consumer side. Public API is unchanged so callers need no edits;
the stateless helpers are re-exported from voxedge.
"""

from __future__ import annotations

import logging
import os
import threading

from server.core.env_helpers import truthy

# Stateless helpers + model id come straight from voxedge (single source).
try:
    from voxedge.capabilities.speaker_embedding import (  # noqa: F401
        SPEAKER_MODEL_NAME,
        decode_audio_to_16k_mono,
        embedding_payload,
        encode_embedding,
        pcm16_to_float32,
        resample_linear,
    )
except Exception:  # voxedge optional at import time
    SPEAKER_MODEL_NAME = "campplus_sv_zh_en_3dspeaker"

logger = logging.getLogger(__name__)

_HF_URL_DEFAULT = (
    "{endpoint}/csukuangfj/speaker-embedding-models/resolve/main/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)

_embedder = None        # cached voxedge SpeakerEmbedder
_lock = threading.Lock()
_load_failed = False


def _trt_engine_file() -> str:
    """Path to a prebuilt CAM++ TRT engine (a *file*, not a dir).

    When set and the file exists, the Jetson TRT backend is preferred over the
    sherpa CPU path. Unset/empty (the default, and every non-Jetson image) keeps
    the existing sherpa behavior byte-for-byte.
    """
    return os.environ.get("DIAR_CAMPPLUS_ENGINE_FILE", "").strip()


class _TRTEmbedderAdapter:
    """Expose ``JetsonCampplusTRT`` under the ``SpeakerEmbedder.compute()`` API
    so ``compute_embedding`` stays backend-agnostic (no caller edits)."""

    def __init__(self, ext):
        self._ext = ext

    def ready(self) -> bool:
        return self._ext.ready()

    @property
    def dim(self) -> int:
        return self._ext.dim

    def compute(self, samples, sample_rate):
        # JetsonCampplusTRT.extract: mono float32 [-1,1] -> 192-d L2-norm | None.
        return self._ext.extract(samples, sample_rate)


def speaker_embedding_enabled() -> bool:
    """Global default, from ``OVS_SPEAKER_EMB`` (default off). Overridable per
    connection via ``?speaker_embedding=`` / v2v config field.
    """
    return truthy(os.environ.get("OVS_SPEAKER_EMB", ""))


def _model_path() -> str:
    explicit = os.environ.get("OVS_SPEAKER_EMB_MODEL")
    if explicit:
        return explicit
    base = os.environ.get("MODEL_DIR", "/opt/models")
    return os.path.join(base, "speaker", "campplus.onnx")


def _hf_url() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return _HF_URL_DEFAULT.format(endpoint=endpoint)


def _ensure_model(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    import shutil
    import subprocess

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = _hf_url()
    logger.info("Speaker model missing; downloading %s -> %s", url, path)
    tmp = path + ".part"
    if shutil.which("curl"):
        # -L follows the HF-mirror 302 → LFS/CDN store; timeouts so a stuck or
        # unreachable mirror fails fast (feature degrades to off) instead of
        # hanging forever and wedging startup readiness.
        subprocess.run(
            ["curl", "-fSL", "--connect-timeout", "20", "--max-time", "1800",
             "--retry", "3", "-o", tmp, url],
            check=True, timeout=1900,
        )
    else:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    os.replace(tmp, path)
    logger.info("Speaker model ready (%d bytes).", os.path.getsize(path))


def _get_embedder():
    global _embedder, _load_failed
    if _embedder is not None:
        return _embedder
    if _load_failed:
        return None
    with _lock:
        if _embedder is not None:
            return _embedder
        if _load_failed:
            return None
        # Jetson TRT backend (opt-in): only when an engine *file* is configured
        # and present. On any problem (missing voxedge module, engine not ready)
        # fall through to the sherpa CPU path below — never raise, never wedge.
        engine_file = _trt_engine_file()
        if engine_file and os.path.exists(engine_file):
            try:
                from voxedge.capabilities.embedding_extractor import JetsonCampplusTRT

                ext = JetsonCampplusTRT(engine_file)
                if ext.ready():
                    _embedder = _TRTEmbedderAdapter(ext)
                    logger.info("Speaker embedding via Jetson TRT engine (%s).", engine_file)
                    return _embedder
                logger.warning(
                    "CAM++ TRT engine not ready (%s); falling back to sherpa CPU.",
                    engine_file,
                )
            except Exception:
                logger.exception(
                    "Jetson TRT speaker backend init failed; falling back to sherpa CPU."
                )
        try:
            from voxedge.capabilities.speaker_embedding import SpeakerEmbedder

            path = _model_path()
            _ensure_model(path)
            num_threads = int(os.environ.get("OVS_SPEAKER_THREADS", "2"))
            emb = SpeakerEmbedder(path, num_threads=num_threads)
            if not emb.ready():
                _load_failed = True
                return None
            _embedder = emb
        except Exception:
            _load_failed = True
            logger.exception("Failed to init speaker embedding; feature disabled.")
            return None
    return _embedder


def preload() -> bool:
    """Eagerly load (call at startup only when enabled). Returns readiness."""
    return _get_embedder() is not None


def embedding_dim() -> int:
    emb = _get_embedder()
    return emb.dim if emb is not None else 0


def compute_embedding(samples, sample_rate: int):
    """L2-normalized float32 vector for one utterance, or None. Never raises."""
    emb = _get_embedder()
    if emb is None:
        return None
    return emb.compute(samples, sample_rate)
