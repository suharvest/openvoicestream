"""Punctuation restoration — product-layer shim over voxedge.

The inference engine (sherpa-onnx CT-Transformer wrapper) lives in
``voxedge.capabilities.punctuation`` and is env-free. This product-layer module
keeps the deployment concerns voxedge deliberately excludes: the ``OVS_PUNCT``
feature flag (default off), model-path resolution, and lazy on-demand download
(honoring HF_ENDPOINT mirrors). Opt-in, default-OFF, lazy-loaded — when
disabled nothing is loaded and the feature costs nothing.

Public API is unchanged so callers (server/main.py endpoints + streaming
finalize) need no edits.
"""

from __future__ import annotations

import logging
import os
import threading

from server.core.env_helpers import truthy

logger = logging.getLogger(__name__)

# Re-exported from voxedge so the identifier has a single source of truth.
try:
    from voxedge.capabilities.punctuation import PUNCT_MODEL_NAME
except Exception:  # voxedge optional at import time (e.g. docs tooling)
    PUNCT_MODEL_NAME = "ct_transformer_zh_en_vocab272727_2024-04-12"

_HF_REPO = "csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"

_punct = None           # cached voxedge Punctuator
_lock = threading.Lock()
_load_failed = False


def punctuation_enabled() -> bool:
    """Global default for the feature, from ``OVS_PUNCT`` (default off).

    A per-connection ``?punctuate=`` query / v2v config field overrides this.
    """
    return truthy(os.environ.get("OVS_PUNCT", ""))


def _model_path() -> str:
    explicit = os.environ.get("OVS_PUNCT_MODEL")
    if explicit:
        return explicit
    base = os.environ.get("MODEL_DIR", "/opt/models")
    return os.path.join(base, "punctuation", "model.onnx")


def _hf_url() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{_HF_REPO}/resolve/main/model.onnx"


def _ensure_model(path: str) -> None:
    """Download the model to ``path`` on first use if missing (idempotent)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    import shutil
    import subprocess

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = _hf_url()
    logger.info("Punctuation model missing; downloading %s -> %s", url, path)
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
    logger.info("Punctuation model ready (%d bytes).", os.path.getsize(path))


def _get_punct():
    global _punct, _load_failed
    if _punct is not None:
        return _punct
    if _load_failed:
        return None
    with _lock:
        if _punct is not None:
            return _punct
        if _load_failed:
            return None
        try:
            from voxedge.capabilities.punctuation import Punctuator

            path = _model_path()
            _ensure_model(path)
            num_threads = int(os.environ.get("OVS_PUNCT_THREADS", "2"))
            punct = Punctuator(path, num_threads=num_threads)
            if not punct.ready():
                _load_failed = True
                return None
            _punct = punct
        except Exception:
            _load_failed = True
            logger.exception("Failed to init punctuation; feature disabled.")
            return None
    return _punct


def preload() -> bool:
    """Eagerly load (call at startup only when enabled). Returns readiness."""
    return _get_punct() is not None


def add_punctuation(text: str) -> str:
    """Return ``text`` with restored punctuation, unchanged if unavailable."""
    if not text or not text.strip():
        return text
    punct = _get_punct()
    if punct is None:
        return text
    return punct.add_punctuation(text)
