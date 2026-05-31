"""FastAPI server for NLLB-200 translation via CTranslate2.

Thin shell over :class:`voxedge.backends.NLLBTranslatorBackend` (translate
extraction Phase 2). All translation logic — CT2 + SentencePiece loading and
the three real-device tokenization bugs (EncodeAsPieces / ``</s>``+src appended
/ device_index forced int) — now lives in voxedge. This module is *only* the
service wiring: env→config, FastAPI app/lifespan/health, the ``/translate``
route + request/response schema, a CPU-bound thread pool, and the uvicorn entry.

The HTTP contract (route path, request/response fields, default language codes)
is unchanged from the pre-extraction service — ``agent/apps/translator`` and any
other client keep working byte-for-byte.
"""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from voxedge.backends.base import TranslatorConfig
from voxedge.backends.nllb_translator import NLLBTranslatorBackend

logger = logging.getLogger(__name__)


class TranslateRequest(BaseModel):
    """Request payload for /translate endpoint."""
    text: str
    src_lang: str = "zho_Hans"
    tgt_lang: str = "eng_Latn"


class TranslateResponse(BaseModel):
    """Response payload for /translate endpoint."""
    translation: str
    src_lang: str
    tgt_lang: str
    model: str = "nllb-200-distilled-600M"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    device: str


# Global translator backend instance (voxedge NLLBTranslatorBackend).
_backend: NLLBTranslatorBackend | None = None

# Worker pool for offloading the blocking CT2 / SentencePiece calls off the
# asyncio event loop. ctranslate2.Translator is internally thread-safe — it
# serialises on its own backing device / engine — so multiple threads can
# submit translate calls concurrently without external locking. Sizing:
# matches the typical small fan-in for InterpreterMode (1 voice user) + a
# couple of headroom slots for subtitle dashboards or admin probes. Tune via
# TRANSLATOR_WORKERS env var.
_executor: ThreadPoolExecutor | None = None


def _build_config() -> TranslatorConfig:
    """Construct a :class:`TranslatorConfig` from service env vars.

    This service is a standalone microservice (not inside the product app), so
    it owns its own env→config resolution here. Env var names are unchanged:
    ``TRANSLATOR_MODEL_PATH`` / ``TRANSLATOR_DEVICE`` / ``TRANSLATOR_DEVICE_INDEX``.
    """
    return TranslatorConfig(
        model_path=os.getenv(
            "TRANSLATOR_MODEL_PATH",
            "/models/nllb-200-distilled-600m-ct2-int8",
        ),
        device=os.getenv("TRANSLATOR_DEVICE", "cuda"),
        device_index=int(os.getenv("TRANSLATOR_DEVICE_INDEX", "0")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _backend, _executor
    try:
        config = _build_config()
        logger.info(
            "Loading translator from %s (device=%s:%d)",
            config.model_path, config.device, config.device_index,
        )
        backend = NLLBTranslatorBackend(config)
        backend.preload()
        _backend = backend
        logger.info("Translator model loaded successfully")

        workers = int(os.getenv("TRANSLATOR_WORKERS", "4"))
        _executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ct2-worker"
        )
        logger.info("Translator service started (workers=%d)", workers)
    except Exception as e:
        logger.error("Failed to start translator service: %s", e)
        raise
    yield
    logger.info("Translator service shutting down")
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
    if _backend is not None:
        _backend.unload()


app = FastAPI(
    title="NLLB-200 Translator Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if _backend is None or not _backend.is_ready():
        raise HTTPException(status_code=503, detail="Translator not initialized")

    device = os.getenv("TRANSLATOR_DEVICE", "cuda")
    return HealthResponse(
        status="ok",
        model="nllb-200-distilled-600M",
        device=device,
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """Translate text from src_lang to tgt_lang.

    The CT2 call is offloaded to ``_executor`` so the asyncio event loop
    stays free for other in-flight requests. CT2's ``translate_batch`` is
    internally thread-safe; concurrent submissions serialise on the
    backing device but the Python side returns control to the loop while
    they wait.
    """
    if _backend is None or not _backend.is_ready() or _executor is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor,
            _backend.translate,
            text, request.src_lang, request.tgt_lang,
        )
        translated = result.text

        logger.debug(
            "Translated (%s→%s): %r → %r",
            request.src_lang, request.tgt_lang, text, translated
        )

        return TranslateResponse(
            translation=translated,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
        )
    except Exception as e:
        logger.error("Translation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}") from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9001,
        log_level="info",
    )
