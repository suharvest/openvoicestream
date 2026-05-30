"""Qwen3-TTS backend via C++ TRT native engine (pybind11).

adapted from app/backends/jetson/qwen3_trt.py (2026-05-30), dedup after
registry switch.

Supports: BASIC_TTS, VOICE_CLONE, MULTI_LANGUAGE, STREAMING. Models load once
at preload(); the C++ ``qwen3_speech_engine.Pipeline`` stays resident.

Decoupling from the production copy:
  * ABCs from ``voxedge.backends.base`` (TTSBackend / TTSCapability).
  * ALL module-scope + method-internal config env reads (QWEN3_MODEL_BASE +
    all QWEN3_* paths, QWEN3_TTS_VARIANT, OVS_TTS_MODEL_ID, TTS_INT8_EOS_LOGIT_OFFSET,
    TTS_TALKER_CUDA_GRAPH, TTS_TRT_VOCODER_MAX_FRAMES, TTS_VOCODER_TRT,
    QWEN3_TTS_OFFLINE_STREAMING_FOR_LONG, QWEN3_TTS_NUMPY_SAMPLING, OVS_TTS_SEED,
    QWEN3_TTS_PRODUCT_*) → explicit :class:`Qwen3TRTConfig`. voxedge has ZERO
    module-scope env reads. The two values the resident C++ engine reads from
    process env (``TTS_INT8_EOS_LOGIT_OFFSET`` + ``TTS_TALKER_CUDA_GRAPH``) are
    still pushed into ``os.environ`` at preload from config so the native side
    sees them — that is a deliberate subprocess/engine-env bridge.
  * ``resolve_speaker_kwargs`` from ``._util`` (was ``app.core.tts_speakers``;
    voxedge has no registry so preset speaker-name lookups return only ids).
  * The Qwen3 artifact auto-downloader (``app.core.qwen3_artifact_downloader``)
    is dropped — voxedge ships no downloader, so missing files raise.
  * ``model_id`` is a config field; the CustomVoice ``model_id`` pin is folded
    into config resolution.
  * The heavy pybind ``qwen3_speech_engine`` + ``tokenizers`` imports stay
    method-local so the module imports on a box without the engine.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from voxedge.backends.base import TTSBackend, TTSCapability

from ._util import resolve_speaker_kwargs

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
#   QWEN3_MODEL_BASE              → model_base ("/opt/models/qwen3-tts")
#   QWEN3_SHERPA_DIR              → sherpa_dir (<base>/onnx)
#   QWEN3_MODEL_DIR               → model_dir (<base>/onnx)
#   QWEN3_TALKER_ENGINE           → talker_engine (<base>/engines/talker_decode_bf16.engine)
#   QWEN3_CP_ENGINE               → cp_engine (<base>/engines/cp_bf16.engine)
#   QWEN3_SPEAKER_ENCODER         → speaker_encoder (<base>/onnx/speaker_encoder.onnx)
#   QWEN3_TOKENIZER_DIR           → tokenizer_dir (<base>/tokenizer)
#   QWEN3_EXTRACT_SCRIPT          → extract_script (<base>/extract_speaker_emb.py)
#   QWEN3_TTS_VARIANT/OVS_TTS_MODEL_ID → supports_voice_cloning (auto, customvoice→False)
#   OVS_TTS_MODEL_ID              → model_id ("qwen3-tts")
#   TTS_INT8_EOS_LOGIT_OFFSET     → int8_eos_logit_offset (-10.0)   [pushed to engine env]
#   TTS_TALKER_CUDA_GRAPH         → talker_cuda_graph (True)        [pushed to engine env]
#   TTS_TRT_VOCODER_MAX_FRAMES    → vocoder_max_frames (100)
#   TTS_VOCODER_TRT               → use_trt_vocoder (True)
#   QWEN3_TTS_OFFLINE_STREAMING_FOR_LONG → offline_streaming_for_long (True)
#   QWEN3_TTS_NUMPY_SAMPLING      → numpy_sampling (True)
#   OVS_TTS_SEED                  → default_seed (0)
#   QWEN3_TTS_PRODUCT_SEGMENT_TEXT → product_segment_text (False)
#   QWEN3_TTS_PRODUCT_SEGMENT_MAX_CHARS → product_segment_max_chars (20)
#   QWEN3_TTS_PRODUCT_COMMA_PAUSE_MS → product_comma_pause_ms (120)
#   QWEN3_TTS_PRODUCT_HARD_PAUSE_MS → product_hard_pause_ms (180)


@dataclass
class Qwen3TRTConfig:
    """Explicit construction-time config for :class:`Qwen3TRTBackend`."""

    model_base: str = "/opt/models/qwen3-tts"
    sherpa_dir: Optional[str] = None
    model_dir: Optional[str] = None
    talker_engine: Optional[str] = None
    cp_engine: Optional[str] = None
    speaker_encoder: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    extract_script: Optional[str] = None

    # None → auto-detect from ``is_customvoice`` (CustomVoice has no cloning).
    supports_voice_cloning: Optional[bool] = None
    is_customvoice: bool = False
    model_id: str = "qwen3-tts"

    int8_eos_logit_offset: float = -10.0
    talker_cuda_graph: bool = True
    vocoder_max_frames: int = 100
    use_trt_vocoder: bool = True
    offline_streaming_for_long: bool = True
    numpy_sampling: bool = True
    default_seed: int = 0

    product_segment_text: bool = False
    product_segment_max_chars: int = 20
    product_comma_pause_ms: int = 120
    product_hard_pause_ms: int = 180

    def __post_init__(self) -> None:
        import os.path as _p
        base = self.model_base
        if self.sherpa_dir is None:
            self.sherpa_dir = _p.join(base, "onnx")
        if self.model_dir is None:
            self.model_dir = _p.join(base, "onnx")
        if self.talker_engine is None:
            self.talker_engine = _p.join(base, "engines", "talker_decode_bf16.engine")
        if self.cp_engine is None:
            self.cp_engine = _p.join(base, "engines", "cp_bf16.engine")
        if self.speaker_encoder is None:
            self.speaker_encoder = _p.join(base, "onnx", "speaker_encoder.onnx")
        if self.tokenizer_dir is None:
            self.tokenizer_dir = _p.join(base, "tokenizer")
        if self.extract_script is None:
            self.extract_script = _p.join(base, "extract_speaker_emb.py")
        if self.supports_voice_cloning is None:
            self.supports_voice_cloning = not self.is_customvoice


def _sampling_uniforms(seed: int, max_frames: int, numpy_sampling: bool) -> list[float]:
    if seed == 0 or not numpy_sampling:
        return []
    n = max(64, (max_frames + 4) * 16)
    return np.random.RandomState(seed).random_sample(n).astype(float).tolist()


def _detect_language(text: str) -> str:
    """Simple language detection — returns config-compatible language strings."""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            return "chinese"
        if 0x3040 <= cp <= 0x30FF:
            return "japanese"
        if 0xAC00 <= cp <= 0xD7AF:
            return "korean"
    return "english"


def _pcm16_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return out.getvalue()


def _contains_cjk(text: str) -> bool:
    return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in "_+-./")


def _is_ascii_word_boundary(text: str, idx: int) -> bool:
    before = text[idx - 1] if idx > 0 else ""
    after = text[idx] if idx < len(text) else ""
    return not (_is_ascii_word_char(before) and _is_ascii_word_char(after))


def _safe_product_tts_cut(text: str, max_chars: int) -> int:
    if len(text) <= max_chars:
        return len(text)

    floor = max(1, int(max_chars * 0.55))
    for idx in range(max_chars, floor - 1, -1):
        if not _is_ascii_word_boundary(text, idx):
            continue
        prev_ch = text[idx - 1]
        next_ch = text[idx] if idx < len(text) else ""
        if prev_ch.isspace() or next_ch.isspace():
            return idx
        if (prev_ch.isascii() and not next_ch.isascii()) or (not prev_ch.isascii() and next_ch.isascii()):
            return idx

    idx = max_chars
    while idx < len(text) and not _is_ascii_word_boundary(text, idx):
        idx += 1
    return idx if idx < len(text) else len(text)


def _split_product_tts_text(text: str, max_chars: int = 20) -> list[str]:
    """Split only where the Qwen3-TTS product path loses conditioning."""
    text = text.strip()
    if not text or not _contains_cjk(text):
        return [text] if text else []
    breaks = set("，,、。！？!?；;：:\n")
    raw_parts: list[str] = []
    current: list[str] = []
    for ch in text:
        current.append(ch)
        part = "".join(current).strip()
        if ch in breaks:
            if part:
                raw_parts.append(part)
            current.clear()
    tail = "".join(current).strip()
    if tail:
        raw_parts.append(tail)

    parts: list[str] = []
    punctuation_only = set("，,、。！？!?；;：:")
    for raw in raw_parts:
        rest = raw
        while len(rest) > max_chars:
            cut = _safe_product_tts_cut(rest, max_chars)
            part = rest[:cut]
            rest = rest[cut:]
            if part.strip():
                parts.append(part)
        if rest.strip():
            if parts and all(ch in punctuation_only for ch in rest):
                parts[-1] += rest
            else:
                parts.append(rest)
    return parts or [text]


def _concat_wav_bytes(parts: list[bytes], pauses_ms: Optional[list[int]] = None) -> bytes:
    non_empty = [part for part in parts if part]
    if not non_empty:
        return b""
    if len(non_empty) == 1:
        return non_empty[0]

    params = None
    frames: list[bytes] = []
    for idx, part in enumerate(non_empty):
        with wave.open(io.BytesIO(part), "rb") as reader:
            current = (
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
                reader.getcomptype(),
                reader.getcompname(),
            )
            if params is None:
                params = current
            elif current != params:
                raise RuntimeError(f"Cannot concatenate WAV segments with different formats: {current} != {params}")
            frames.append(reader.readframes(reader.getnframes()))
            if pauses_ms and idx < len(non_empty) - 1:
                pause_samples = int(current[2] * max(0, pauses_ms[idx]) / 1000)
                if pause_samples > 0:
                    frames.append(b"\x00" * pause_samples * current[0] * current[1])

    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(params[0])
        writer.setsampwidth(params[1])
        writer.setframerate(params[2])
        writer.setcomptype(params[3], params[4])
        for frame_bytes in frames:
            writer.writeframes(frame_bytes)
    return out.getvalue()


class Qwen3TRTBackend(TTSBackend):
    """Qwen3-TTS via C++ TRT native inference (pybind11 module, models resident)."""

    def __init__(self, config: Optional[Qwen3TRTConfig] = None):
        self._config = config or Qwen3TRTConfig()
        self._engine = None
        self._tokenizer = None
        self._ready = False
        self.supports_voice_cloning = bool(self._config.supports_voice_cloning)

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def name(self) -> str:
        return "qwen3_trt"

    @property
    def capabilities(self) -> set[TTSCapability]:
        import os
        caps = {TTSCapability.BASIC_TTS, TTSCapability.MULTI_LANGUAGE,
                TTSCapability.STREAMING}
        if self.supports_voice_cloning and os.path.exists(self._config.speaker_encoder):
            caps.add(TTSCapability.VOICE_CLONE)
        return caps

    @property
    def sample_rate(self) -> int:
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def _segment_pause_ms(self, segment: str) -> int:
        if segment.rstrip().endswith(("。", "！", "？", "!", "?", "；", ";")):
            return max(0, int(self._config.product_hard_pause_ms))
        return max(0, int(self._config.product_comma_pause_ms))

    def unload(self) -> None:
        """Best-effort release of the pybind11 Pipeline + tokenizer."""
        if not self._ready and self._engine is None:
            return
        try:
            self._engine = None
            self._tokenizer = None
            import gc
            gc.collect()
        except Exception:
            logger.exception("Qwen3TRTBackend.unload failed; continuing")
        finally:
            self._ready = False

    def preload(self) -> None:
        """Load C++ TRT engine + tokenizer. Models stay resident."""
        import os

        c = self._config
        required = [
            (c.talker_engine, "talker engine"),
            (c.cp_engine, "CP engine"),
            (os.path.join(c.model_base, "engines", "vocoder_fp16.engine"), "vocoder engine"),
            (os.path.join(c.sherpa_dir, "config.json"), "config.json (authoritative)"),
        ]
        missing = [(path, desc) for path, desc in required if not os.path.exists(path)]
        if missing:
            path, desc = missing[0]
            raise FileNotFoundError(f"Missing {desc}: {path}")

        self._load_tokenizer()

        logger.info("Loading Qwen3 TRT engine (this takes ~25s)...")
        t0 = time.time()

        # Bridge: the resident C++ engine reads these two from process env. We
        # push the config-driven values into os.environ (setdefault so an
        # operator override still wins) so the native side sees them.
        os.environ.setdefault("TTS_INT8_EOS_LOGIT_OFFSET", str(c.int8_eos_logit_offset))

        import qwen3_speech_engine
        self._engine = qwen3_speech_engine.Pipeline(
            c.model_dir, c.sherpa_dir, c.talker_engine, c.cp_engine,
        )
        logger.info("Qwen3 TRT engine loaded in %.1fs", time.time() - t0)

        if c.talker_cuda_graph:
            try:
                self._engine.enable_cuda_graph(True)
                logger.info("CUDA Graph enabled for talker decode (cached mode)")
            except Exception as e:
                logger.warning("CUDA Graph enable failed (non-fatal): %s", e)
        else:
            logger.info("CUDA Graph disabled for talker decode")

        self._ready = True

    def _load_tokenizer(self):
        import os
        vocab_path = os.path.join(self._config.tokenizer_dir, "vocab.json")
        merges_path = os.path.join(self._config.tokenizer_dir, "merges.txt")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Tokenizer not found: {vocab_path}")

        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel

        self._tokenizer = Tokenizer(BPE(vocab_path, merges_path))
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        logger.info("Tokenizer loaded from %s", self._config.tokenizer_dir)

    def _tokenize(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._tokenizer.encode(text).ids

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        c = self._config
        voice = resolve_speaker_kwargs(
            self.model_id,
            allow_embedding=self.supports_voice_cloning,
            speaker_id=speaker_id,
            **kwargs,
        )
        if voice.get("speaker_embedding"):
            if not self.supports_voice_cloning:
                logger.warning("Qwen3-CustomVoice backend received speaker_embedding; not supported")
                raise NotImplementedError(
                    "Qwen3-CustomVoice does not support voice cloning. "
                    "Use a preset speaker_id (one of the 9 built-in voices) instead."
                )
            kwargs.setdefault("speaker_embedding", voice["speaker_embedding"])
        resolved_speaker_name = voice.get("speaker") if voice else None
        resolved_speaker_id = voice.get("speaker_id") if voice else None
        if language is None:
            language = _detect_language(text)

        token_ids = self._tokenize(text)
        requested_max_frames = int(kwargs.get("max_audio_length", kwargs.get("max_frames", 200)))
        vocoder_cap = int(c.vocoder_max_frames)
        use_trt_vocoder = bool(c.use_trt_vocoder)
        expected_frames = max(50, len(token_ids) * 3)
        collect_streaming = (
            use_trt_vocoder
            and requested_max_frames > vocoder_cap
            and expected_frames > vocoder_cap
            and c.offline_streaming_for_long
        )
        seed = int(kwargs.get("seed", c.default_seed))
        segment_text = kwargs.get("product_segment_text", True)
        if isinstance(segment_text, str):
            segment_text = segment_text.lower() not in ("0", "false", "no")
        if segment_text and c.product_segment_text:
            max_chars = int(c.product_segment_max_chars)
            segments = _split_product_tts_text(text, max_chars=max_chars)
            if len(segments) > 1:
                start = time.time()
                wav_parts: list[bytes] = []
                segment_meta: list[dict] = []
                segment_kwargs = dict(kwargs)
                segment_kwargs["product_segment_text"] = False
                segment_kwargs.pop("seed", None)
                segment_kwargs.setdefault("max_audio_length", min(requested_max_frames, vocoder_cap if use_trt_vocoder else requested_max_frames))
                for segment in segments:
                    wav, meta = self.synthesize(
                        segment,
                        speaker_id=resolved_speaker_id if resolved_speaker_id is not None else speaker_id,
                        speed=speed,
                        pitch_shift=pitch_shift,
                        language=language,
                        seed=seed,
                        **segment_kwargs,
                    )
                    wav_parts.append(wav)
                    segment_meta.append({"text": segment, **meta})
                pauses_ms = [self._segment_pause_ms(segment) for segment in segments[:-1]]
                wav_bytes = _concat_wav_bytes(wav_parts, pauses_ms=pauses_ms)
                duration = 0.0
                samples = 0
                with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
                    samples = reader.getnframes()
                    duration = samples / reader.getframerate() if reader.getframerate() else 0.0
                elapsed = time.time() - start
                return wav_bytes, {
                    "duration": round(duration, 3),
                    "inference_time": round(elapsed, 3),
                    "rtf": round(elapsed / duration, 3) if duration else 0,
                    "sample_rate": self.sample_rate,
                    "samples": samples,
                    "seed": seed,
                    "product_segmented": True,
                    "segment_count": len(segments),
                    "segment_pauses_ms": pauses_ms,
                    "segments": segment_meta,
                }

        if collect_streaming:
            start = time.time()
            stream_kwargs: dict[str, Any] = {
                "language": language,
                "max_frames": requested_max_frames,
                "seed": seed,
                "first_chunk_frames": int(kwargs.get("first_chunk_frames", 25)),
                "chunk_frames": int(kwargs.get("chunk_frames", 25)),
            }
            if kwargs.get("speaker_embedding"):
                stream_kwargs["speaker_embedding"] = kwargs["speaker_embedding"]
            if resolved_speaker_name:
                stream_kwargs["speaker"] = resolved_speaker_name
            if resolved_speaker_id is not None:
                stream_kwargs["speaker_id"] = resolved_speaker_id
            pcm = b"".join(self.generate_streaming(text, **stream_kwargs))
            elapsed = time.time() - start
            duration = len(pcm) / 2 / self.sample_rate if pcm else 0.0
            return _pcm16_to_wav(pcm, self.sample_rate), {
                "duration": round(duration, 3),
                "inference_time": round(elapsed, 3),
                "rtf": round(elapsed / duration, 3) if duration else 0,
                "sample_rate": self.sample_rate,
                "samples": len(pcm) // 2,
                "seed": seed,
                "offline_collected_streaming": True,
            }

        max_frames = requested_max_frames
        if use_trt_vocoder:
            max_frames = min(max_frames, vocoder_cap)
        random_values = _sampling_uniforms(seed, max_frames, c.numpy_sampling)

        start = time.time()
        engine_kwargs = dict(
            text=text,
            lang=language,
            token_ids=token_ids,
            max_frames=max_frames,
            seed=seed,
            random_values=random_values,
        )
        if resolved_speaker_name:
            engine_kwargs["speaker"] = resolved_speaker_name
        try:
            result = self._engine.synthesize(**engine_kwargs)
        except TypeError as exc:
            if "speaker" in engine_kwargs and "speaker" in str(exc):
                engine_kwargs.pop("speaker", None)
                result = self._engine.synthesize(**engine_kwargs)
            else:
                raise
        elapsed = time.time() - start

        wav_bytes = result["wav_bytes"]
        duration = result.get("duration", 0)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(result.get("rtf", 0), 3),
            "sample_rate": self.sample_rate,
            "n_frames": result.get("n_frames", 0),
            "per_step_ms": round(result.get("per_step_ms", 0), 1),
            "seed": seed,
        }
        return wav_bytes, meta

    def clone_voice(
        self,
        text: str,
        speaker_embedding: bytes,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if not self.supports_voice_cloning:
            logger.warning("clone_voice called on Qwen3-CustomVoice backend; rejecting")
            raise NotImplementedError(
                "Qwen3-CustomVoice does not support voice cloning. "
                "Switch to a clone-capable backend (e.g. MOSS-TTS-Nano) or use built-in speakers."
            )
        if language is None:
            language = _detect_language(text)

        token_ids = self._tokenize(text)

        start = time.time()
        result = self._engine.synthesize_clone(
            text=text,
            lang=language,
            token_ids=token_ids,
            speaker_emb_bytes=speaker_embedding,
        )
        elapsed = time.time() - start

        wav_bytes = result["wav_bytes"]
        duration = result.get("duration", 0)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(result.get("rtf", 0), 3),
            "sample_rate": self.sample_rate,
        }
        return wav_bytes, meta

    def generate_streaming(self, text: str, **kwargs):
        """Yield PCM int16 chunks via C++ callback-based streaming."""
        import queue as queue_mod
        import threading

        c = self._config
        voice = resolve_speaker_kwargs(
            self.model_id,
            allow_embedding=self.supports_voice_cloning,
            **kwargs,
        )
        language = kwargs.get("language") or _detect_language(text)
        speaker_embedding = voice.get("speaker_embedding") or kwargs.get("speaker_embedding")
        resolved_speaker_name = voice.get("speaker") if voice else None
        if not resolved_speaker_name and isinstance(kwargs.get("speaker"), str):
            resolved_speaker_name = kwargs["speaker"]
        if speaker_embedding and not self.supports_voice_cloning:
            logger.warning("generate_streaming received speaker_embedding on CustomVoice backend; rejecting")
            raise NotImplementedError(
                "Qwen3-CustomVoice does not support voice cloning (streaming clone path)."
            )
        first_chunk_frames = kwargs.get("first_chunk_frames", 5)
        chunk_frames = kwargs.get("chunk_frames", 25)
        max_frames = kwargs.get("max_frames", 200)
        seed = int(kwargs.get("seed", c.default_seed))
        random_values = _sampling_uniforms(seed, int(max_frames), c.numpy_sampling)

        token_ids = self._tokenize(text)

        chunk_queue: queue_mod.Queue = queue_mod.Queue()
        SENTINEL = object()

        def _on_chunk(chunk_dict):
            wav_bytes = chunk_dict["wav_bytes"]
            if len(wav_bytes) > 44:
                chunk_queue.put(wav_bytes[44:])

        def _run_engine():
            try:
                if speaker_embedding:
                    self._engine.synthesize_streaming_clone_callback(
                        text=text,
                        lang=language,
                        token_ids=token_ids,
                        speaker_emb_bytes=speaker_embedding,
                        callback=_on_chunk,
                        first_chunk_frames=first_chunk_frames,
                        chunk_frames=chunk_frames,
                        max_frames=max_frames,
                        seed=seed,
                    )
                else:
                    streaming_kwargs = dict(
                        text=text,
                        lang=language,
                        token_ids=token_ids,
                        callback=_on_chunk,
                        first_chunk_frames=first_chunk_frames,
                        chunk_frames=chunk_frames,
                        max_frames=max_frames,
                        seed=seed,
                        random_values=random_values,
                    )
                    if resolved_speaker_name:
                        streaming_kwargs["speaker"] = resolved_speaker_name
                    try:
                        self._engine.synthesize_streaming_callback(**streaming_kwargs)
                    except TypeError as exc:
                        if "speaker" in streaming_kwargs and "speaker" in str(exc):
                            streaming_kwargs.pop("speaker", None)
                            self._engine.synthesize_streaming_callback(**streaming_kwargs)
                        else:
                            raise
            finally:
                chunk_queue.put(SENTINEL)

        threading.Thread(target=_run_engine, daemon=True).start()

        while True:
            item = chunk_queue.get()
            if item is SENTINEL:
                break
            yield item

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        """Extract speaker embedding using the external extract script + ORT."""
        if not self.supports_voice_cloning:
            raise NotImplementedError(
                "Qwen3-CustomVoice does not support speaker embedding extraction."
            )
        import os
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wf.write(audio_wav_bytes)
            wav_path = wf.name
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as ef:
            emb_path = ef.name

        try:
            result = subprocess.run(
                ["python3", self._config.extract_script,
                 "--audio", wav_path,
                 "--model", self._config.speaker_encoder,
                 "--output", emb_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Embedding extraction failed: {result.stderr}")
            return open(emb_path, "rb").read()
        finally:
            for p in [wav_path, emb_path]:
                if os.path.exists(p):
                    os.unlink(p)
