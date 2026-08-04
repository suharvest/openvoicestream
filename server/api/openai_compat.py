"""Small OpenAI-compatible audio adapter.

The adapter deliberately stays at the transport boundary.  Backend selection,
speaker resolution, admission and execution are delegated to the shared v1
helpers in :mod:`server.main` and :mod:`server.core.api_execution`; this module
only validates the OpenAI wire shape and serializes the result.
"""

from __future__ import annotations

import logging
import math
import struct
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import (
    http_exception_handler as _default_http_exception_handler,
    request_validation_exception_handler as _default_validation_handler,
)
from fastapi.responses import JSONResponse, Response
from fastapi import HTTPException
from starlette.datastructures import UploadFile

logger = logging.getLogger(__name__)

router = APIRouter()


def _main():
    # Import at call time: ``server.main`` installs this router after defining
    # its private resolver/execution helpers, so importing it at module import
    # time would create an avoidable circular dependency.
    from server import main

    return main


def _require_api_key(request: Request) -> None:
    """Reuse the exact authentication dependency used by native routes."""

    return _main()._require_api_key(request)


def _is_audio_path(request: Request) -> bool:
    return request.url.path.startswith("/v1/audio/")


def _error_type(status: int) -> str:
    if status == 401:
        return "authentication_error"
    if status == 429:
        return "rate_limit_error"
    if status >= 500:
        return "server_error"
    return "invalid_request_error"


def _openai_error(
    message: str,
    *,
    status_code: int,
    code: str,
    param: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {
        "message": str(message),
        "type": _error_type(int(status_code)),
        "code": str(code),
    }
    if param:
        error["param"] = param
    return JSONResponse(
        {"error": error},
        status_code=int(status_code),
        headers=dict(headers or {}),
    )


def _serialize_exception(exc: BaseException) -> JSONResponse:
    """Map shared/domain/FastAPI errors to the OpenAI error envelope."""

    main = _main()
    from server.core.api_execution import APIExecutionError

    if isinstance(exc, APIExecutionError):
        return _openai_error(
            exc.message,
            status_code=exc.status_code,
            code=exc.code,
            param=exc.param,
            headers=exc.headers,
        )
    saturated, _max_slots = main._is_pool_saturated(exc)
    if saturated:
        return _openai_error(
            "backend is busy",
            status_code=429,
            code="backend_busy",
            headers={"Retry-After": "1"},
        )
    if isinstance(exc, HTTPException):
        status = int(exc.status_code)
        detail = exc.detail
        if isinstance(detail, Mapping):
            code = str(detail.get("error") or detail.get("code") or "http_error")
            # Never expose manager/backend internals through a 5xx adapter.
            message = (
                str(detail.get("message") or detail.get("detail") or code)
                if status < 500
                else "service unavailable"
            )
            param = detail.get("param")
        else:
            code = "http_error"
            message = str(detail) if status < 500 else "service unavailable"
            param = None
        return _openai_error(
            message,
            status_code=status,
            code=code,
            param=str(param) if param else None,
            headers=dict(exc.headers or {}),
        )
    logger.exception("OpenAI-compatible audio execution failed", exc_info=exc)
    return _openai_error(
        "service unavailable",
        status_code=503,
        code="backend_error",
        headers={"Retry-After": "1"},
    )


def _validation_param(exc: RequestValidationError) -> str | None:
    for item in exc.errors():
        loc = item.get("loc") or ()
        for value in reversed(tuple(loc)):
            if value not in {"body", "query", "path", "form"}:
                return str(value)
    return None


def _validation_response(exc: RequestValidationError) -> JSONResponse:
    return _openai_error(
        "invalid request",
        status_code=422,
        code="invalid_request",
        param=_validation_param(exc),
    )


def _normalize_voice(value: Any) -> int | str | None:
    # OpenAI's voice is a string, but the native resolver accepts numeric ids
    # as either JSON numbers or numeric strings.  Preserve both forms.
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, str)):
        return value
    return str(value)


def _wav_pcm_payload(wav: bytes) -> tuple[bytes, int, int]:
    """Extract PCM16 little-endian data from a RIFF/WAVE byte stream.

    Chunk walking (rather than slicing at byte 44) handles optional LIST,
    JUNK and non-canonical fmt/data ordering.  Only the service contract's
    PCM s16le mono/stereo format is accepted.
    """

    if not isinstance(wav, (bytes, bytearray, memoryview)):
        raise ValueError("backend returned non-binary audio")
    raw = bytes(wav)
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("backend did not return a valid WAV stream")
    riff_size = struct.unpack_from("<I", raw, 4)[0]
    riff_end = riff_size + 8
    if riff_size < 4 or riff_end > len(raw):
        raise ValueError("backend returned a truncated WAV stream")
    fmt: tuple[int, int, int, int, int, int] | None = None
    data: bytes | None = None
    offset = 12
    while offset < riff_end:
        if offset + 8 > riff_end:
            raise ValueError("backend returned a truncated WAV chunk header")
        chunk_id = raw[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
        start = offset + 8
        end = start + chunk_size
        padded_end = end + (chunk_size & 1)
        if end > riff_end or padded_end > riff_end:
            raise ValueError("backend returned a truncated WAV chunk")
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError("WAV fmt chunk is too short")
            audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from(
                "<HHIIHH", raw, start
            )
            fmt = (audio_format, channels, sample_rate, byte_rate, block_align, bits)
        elif chunk_id == b"data":
            data = raw[start:end]
        # RIFF chunks are padded to an even boundary.
        offset = padded_end
    if fmt is None or data is None:
        raise ValueError("WAV stream is missing fmt or data chunk")
    audio_format, channels, sample_rate, byte_rate, block_align, bits = fmt
    if audio_format != 1 or channels not in {1, 2} or bits != 16:
        raise ValueError("only mono/stereo PCM s16le WAV is supported")
    expected_block_align = channels * 2
    if sample_rate <= 0 or block_align != expected_block_align:
        raise ValueError("WAV PCM format metadata is invalid")
    if byte_rate != sample_rate * expected_block_align:
        raise ValueError("WAV PCM byte rate is invalid")
    if len(data) % (2 * channels):
        raise ValueError("WAV data is not aligned to PCM frames")
    return data, sample_rate, channels


def _native_tts_request(
    *,
    model: str,
    text: str,
    voice: Any,
    speed: Any,
) -> Any:
    main = _main()
    try:
        return main.NativeTTSRequest(
            model=model,
            text=text,
            voice=_normalize_voice(voice),
            speed=speed,
        )
    except Exception as exc:
        # Keep malformed body errors in the OpenAI envelope even when a
        # client sends a non-numeric speed or an invalid union value.
        from server.core.api_execution import APIExecutionError

        raise APIExecutionError(
            "invalid speech request",
            status_code=400,
            code="invalid_request",
        ) from exc


async def _run_speech(
    *,
    model: str,
    text: str,
    voice: Any,
    speed: float | None,
) -> Any:
    main = _main()
    from server.core.api_execution import APIExecutionError
    from server.core.session_limiter import acquire_http

    if not isinstance(model, str) or not model.strip():
        raise APIExecutionError(
            "model is required",
            status_code=400,
            code="missing_required_parameter",
            param="model",
        )
    if not isinstance(text, str) or not text:
        raise APIExecutionError(
            "input is required",
            status_code=400,
            code="missing_required_parameter",
            param="input",
        )
    main._v1_validate_text(text)
    req = _native_tts_request(model=model, text=text, voice=voice, speed=speed)
    async with acquire_http("/v1/audio/speech"):
        manager = await main._ensure_tts_manager_started()

        def _prepare(backend):
            active_model = main._v1_backend_model(backend)
            main._v1_check_model(req.model, active_model)
            main._v1_validate_controls(req, backend)
            return main._v1_resolve_voice_kwargs(req, backend)

        # ``_execute_tts_core`` is the transport-neutral path used by native
        # v1 and legacy routes; no HTTP response is parsed here.
        return await main._execute_tts_core(req, manager=manager, prepare=_prepare)


@router.post("/v1/audio/speech")
async def audio_speech(request: Request, _: None = Depends(_require_api_key)):
    """OpenAI-compatible speech synthesis returning WAV or headerless PCM."""

    try:
        payload = await request.json()
    except Exception as exc:
        from server.core.api_execution import APIExecutionError

        return _serialize_exception(
            APIExecutionError(
                "request body must be valid JSON",
                status_code=400,
                code="invalid_request",
            )
        )
    if not isinstance(payload, dict):
        from server.core.api_execution import APIExecutionError

        return _serialize_exception(
            APIExecutionError(
                "request body must be a JSON object",
                status_code=400,
                code="invalid_request",
            )
        )
    allowed = {"model", "input", "voice", "speed", "response_format"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        from server.core.api_execution import APIExecutionError

        return _serialize_exception(
            APIExecutionError(
                f"unsupported parameter {unknown[0]!r}",
                status_code=400,
                code="unsupported_parameter",
                param=unknown[0],
            )
        )
    model = payload.get("model")
    text = payload.get("input")
    response_format = payload.get("response_format", "wav")
    if response_format is None:
        response_format = "wav"
    if not isinstance(response_format, str) or response_format.strip().lower() not in {"wav", "pcm"}:
        from server.core.api_execution import APIExecutionError

        return _serialize_exception(
            APIExecutionError(
                "only wav and pcm response formats are supported",
                status_code=400,
                code="unsupported_format",
                param="response_format",
            )
        )
    response_format = response_format.strip().lower()
    speed = payload.get("speed")
    if speed is not None:
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            from server.core.api_execution import APIExecutionError

            return _serialize_exception(
                APIExecutionError(
                    "speed must be numeric",
                    status_code=400,
                    code="unsupported_control",
                    param="speed",
                )
            )
    try:
        result = await _run_speech(
            model=model,
            text=text,
            voice=payload.get("voice"),
            speed=speed,
        )
        content = result.audio
        headers = {
            "X-Audio-Duration": str(result.metadata.get("duration", result.metadata.get("duration_s", 0))),
            "X-Inference-Time": str(result.metadata.get("inference_time", result.metadata.get("inference_time_s", 0))),
            "X-RTF": str(result.metadata.get("rtf", 0)),
        }
        if response_format == "pcm":
            try:
                content, sample_rate, channels = _wav_pcm_payload(content)
            except ValueError as exc:
                from server.core.api_execution import APIExecutionError

                raise APIExecutionError(
                    str(exc),
                    status_code=503,
                    code="invalid_backend_audio",
                ) from exc
            headers["X-Sample-Rate"] = str(sample_rate)
            headers["X-Audio-Channels"] = str(channels)
            return Response(content=content, media_type="audio/pcm", headers=headers)
        return Response(content=content, media_type="audio/wav", headers=headers)
    except Exception as exc:
        return _serialize_exception(exc)


def _form_values(form: Any, key: str) -> list[Any]:
    try:
        values = form.getlist(key)
    except Exception:
        values = []
    if values:
        return list(values)
    value = form.get(key)
    return [] if value is None else [value]


def _form_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


async def _run_transcription(*, upload: UploadFile, model: str, language: str) -> Any:
    main = _main()
    from server.core.api_execution import APIExecutionError, read_bounded_upload
    from server.core.api_capabilities import canonical_asr_model_id
    from server.core.session_limiter import acquire_http

    # Keep OpenAI transcription under the same HTTP admission limiter as
    # native /asr and /v1/asr.  The lease covers upload consumption and decode
    # so a large multipart body cannot bypass the configured session cap.
    async with acquire_http("/v1/audio/transcriptions"):
        audio = await read_bounded_upload(
            upload,
            max_bytes=main._v1_limit("OVS_API_MAX_AUDIO_BYTES", 32 * 1024 * 1024),
        )
        manager = main._get_asr_manager()

        def _prepare(backend):
            active_model = main._v1_active_asr_model(backend)
            if canonical_asr_model_id(model) != active_model:
                raise APIExecutionError(
                    f"model {model!r} is not the active ASR model",
                    status_code=404,
                    code="unknown_model",
                    param="model",
                )

        return await main._execute_asr_core(
            audio,
            language,
            prepare=_prepare,
            manager_override=manager,
        )


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request, _: None = Depends(_require_api_key)):
    """OpenAI-compatible multipart transcription (JSON or plain text)."""

    from server.core.api_execution import APIExecutionError

    try:
        content_type = request.headers.get("content-type", "").casefold()
        if "multipart/form-data" not in content_type:
            raise APIExecutionError(
                "multipart/form-data is required",
                status_code=400,
                code="invalid_multipart",
            )
        try:
            form = await request.form()
        except Exception as exc:
            raise APIExecutionError(
                "invalid multipart body",
                status_code=400,
                code="invalid_multipart",
            ) from exc
        upload = form.get("file")
        model = form.get("model")
        if not isinstance(upload, UploadFile):
            raise APIExecutionError(
                "file is required",
                status_code=400,
                code="missing_required_parameter",
                param="file",
            )
        if not isinstance(model, str) or not model.strip():
            raise APIExecutionError(
                "model is required",
                status_code=400,
                code="missing_required_parameter",
                param="model",
            )
        language = form.get("language")
        language = str(language).strip() if language is not None else ""
        language = language or "auto"
        response_format = form.get("response_format", "json")
        response_format = str(response_format).strip().casefold() if response_format is not None else "json"
        if response_format not in {"json", "text"}:
            raise APIExecutionError(
                "only json and text response formats are supported",
                status_code=400,
                code="unsupported_format",
                param="response_format",
            )
        prompt = form.get("prompt")
        if prompt not in (None, ""):
            raise APIExecutionError(
                "prompt is not supported",
                status_code=400,
                code="unsupported_control",
                param="prompt",
            )
        temperature = form.get("temperature")
        if temperature not in (None, ""):
            try:
                temperature_value = float(temperature)
            except (TypeError, ValueError):
                raise APIExecutionError(
                    "temperature must be numeric",
                    status_code=400,
                    code="unsupported_control",
                    param="temperature",
                )
            if not math.isfinite(temperature_value) or temperature_value != 0.0:
                raise APIExecutionError(
                    "only temperature=0 is supported",
                    status_code=400,
                    code="unsupported_control",
                    param="temperature",
                )
        timestamp_values = _form_values(form, "timestamp_granularities[]") + _form_values(
            form, "timestamp_granularities"
        )
        if any(str(value).strip() for value in timestamp_values):
            raise APIExecutionError(
                "timestamp_granularities is not supported",
                status_code=400,
                code="unsupported_control",
                param="timestamp_granularities[]",
            )
        verbose = form.get("verbose_json")
        if verbose is not None and _form_bool(verbose):
            raise APIExecutionError(
                "verbose_json is not supported",
                status_code=400,
                code="unsupported_format",
                param="response_format",
            )
        result = await _run_transcription(
            upload=upload,
            model=model,
            language=language,
        )
        if response_format == "text":
            return Response(content=result.text, media_type="text/plain")
        return {"text": result.text}
    except Exception as exc:
        return _serialize_exception(exc)


def _active_model_state() -> list[dict[str, Any]]:
    """Return configured ASR/TTS models without triggering lazy preload."""

    main = _main()
    from server.core.api_capabilities import canonical_asr_model_id
    from server.core.tts_speakers import canonical_model_id
    from server.core.profile_loader import current_profile
    from server.core import tts_service

    try:
        profile = current_profile() or {}
    except Exception:
        profile = {}

    tts_manager = main._get_tts_manager()
    tts_backend = None
    tts_ready = False
    if tts_manager is not None and tts_manager.is_ready():
        try:
            tts_backend = tts_manager.get_backend_unsafe()
            tts_ready = bool(tts_backend and tts_backend.is_ready())
        except Exception:
            tts_backend = None
    elif tts_manager is None:
        try:
            if tts_service.is_ready():
                tts_backend = tts_service.get_backend()
                tts_ready = bool(tts_backend and tts_backend.is_ready())
        except Exception:
            tts_backend = None

    asr_manager = main._get_asr_manager()
    asr_backend = None
    asr_ready = False
    if asr_manager is not None and asr_manager.is_ready():
        try:
            asr_backend = asr_manager.get_backend_unsafe()
            asr_ready = bool(asr_backend and asr_backend.is_ready())
        except Exception:
            asr_backend = None
    elif asr_manager is None:
        try:
            asr_backend = main._get_asr_backend()
            asr_ready = bool(asr_backend and asr_backend.is_ready())
        except Exception:
            asr_backend = None

    rows: dict[str, dict[str, Any]] = {}

    def add(
        kind: str,
        model_id: str | None,
        backend: Any,
        ready: bool,
        *,
        aliases: list[str] | None = None,
    ) -> None:
        if not model_id:
            return
        mid = canonical_model_id(model_id) if kind == "tts" else canonical_asr_model_id(model_id)
        if not mid:
            return
        row = rows.setdefault(
            mid,
            {
                "id": mid,
                "object": "model",
                "created": 0,
                "owned_by": "seeed-studio",
                "modalities": [],
                "aliases": [],
                "backend": None,
                "ready": False,
                "readiness": "not_ready",
            },
        )
        if kind not in row["modalities"]:
            row["modalities"].append(kind)
        for alias in aliases or []:
            alias = str(alias).strip()
            if alias and alias != mid and alias not in row["aliases"]:
                row["aliases"].append(alias)
        backend_name = getattr(backend, "name", None) if backend is not None else None
        configured_backend = profile.get(f"{kind}_backend")
        name = str(backend_name or configured_backend) if (backend_name or configured_backend) else None
        previous = row.get("backend")
        if name and previous is None:
            row["backend"] = name
        elif name and previous != name:
            names = previous if isinstance(previous, list) else ([previous] if previous else [])
            if name not in names:
                names.append(name)
            row["backend"] = names
        row["ready"] = bool(row["ready"] or ready)
        row["readiness"] = "ready" if row["ready"] else "not_ready"

    tts_model = getattr(tts_backend, "model_id", None) or profile.get("tts_model_id") or os_environ("OVS_TTS_MODEL_ID")
    asr_model = getattr(asr_backend, "model_id", None) or profile.get("asr_model_id") or os_environ("OVS_ASR_MODEL_ID")
    tts_aliases = []
    asr_aliases = []
    if tts_model:
        tts_aliases.append(str(tts_model))
    if asr_model:
        asr_aliases.append(str(asr_model))
    # Keep declared public aliases as metadata on the canonical row.  They are
    # accepted by the strict native model resolver but never become duplicate
    # OpenAI model records.
    if canonical_model_id("qwen3-tts-0.6b-base") == canonical_model_id(str(tts_model or "")):
        tts_aliases.extend(["qwen3-tts-base", "qwen3-tts-0.6b"])
    if canonical_asr_model_id(str(asr_model or "")) == "qwen3-asr":
        asr_aliases.extend(["qwen3-asr-0.6b", "Qwen/Qwen3-ASR-0.6B"])
    if profile.get("tts_backend") or tts_manager is not None or tts_backend is not None:
        add("tts", tts_model, tts_backend, tts_ready, aliases=tts_aliases)
    if profile.get("asr_backend") or asr_manager is not None or asr_backend is not None:
        add("asr", asr_model, asr_backend, asr_ready, aliases=asr_aliases)
    for row in rows.values():
        row["modalities"] = sorted(row["modalities"])
    return list(rows.values())


def os_environ(key: str) -> str | None:
    # Tiny indirection keeps _active_model_state easy to monkeypatch in tests
    # without importing os into route code paths.
    import os

    return os.environ.get(key)


@router.get("/v1/models")
async def models(_: None = Depends(_require_api_key)):
    """List only the configured canonical ASR/TTS model identities."""

    try:
        return {"object": "list", "data": _active_model_state()}
    except Exception as exc:
        return _serialize_exception(exc)


def register(app: Any) -> None:
    """Install routes and audio-only error normalizers on the application."""

    if not any(getattr(route, "path", None) == "/v1/audio/speech" for route in app.routes):
        app.include_router(router)

    previous_validation = app.exception_handlers.get(RequestValidationError)
    previous_http = app.exception_handlers.get(HTTPException)
    previous_exception = app.exception_handlers.get(Exception)

    async def validation_handler(request: Request, exc: RequestValidationError):
        if _is_audio_path(request):
            return _validation_response(exc)
        if previous_validation is not None:
            return await previous_validation(request, exc)
        return await _default_validation_handler(request, exc)

    async def http_handler(request: Request, exc: HTTPException):
        if _is_audio_path(request):
            return _serialize_exception(exc)
        if previous_http is not None:
            return await previous_http(request, exc)
        # FastAPI installs its stock exception handlers in the application
        # dispatch stack rather than always exposing them in
        # ``app.exception_handlers``.  Falling back to the framework handler
        # keeps legacy/native auth and HTTPException wire formats byte-for-
        # byte unchanged outside /v1/audio/*.
        return await _default_http_exception_handler(request, exc)

    async def exception_handler(request: Request, exc: Exception):
        if _is_audio_path(request):
            return _serialize_exception(exc)
        if previous_exception is not None:
            return await previous_exception(request, exc)
        raise exc

    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(HTTPException, http_handler)
    app.add_exception_handler(Exception, exception_handler)
