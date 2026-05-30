"""Paraformer streaming ASR — encoder + decoder via TensorRT.

adapted from app/backends/jetson/paraformer_trt.py (2026-05-30), dedup after
registry switch.

Supports: OFFLINE, STREAMING. numpy-only fbank + TRT encoder/decoder (CUDA).
CIF (Continuous Integrate-and-Fire) handles token timing + endpoint detection.

Decoupling from the production copy:
  * ABCs from ``voxedge.backends.base`` (ASRBackend / ASRCapability / ASRStream
    / TranscriptionResult) and ``ConcurrencyCapability`` from
    ``voxedge.engine.concurrency_capability``.
  * The module-scope ``PARAFORMER_*`` path reads + ``PARAFORMER_PREROLL_MS`` are
    replaced by an explicit :class:`ParaformerTRTConfig` injected at
    construction. voxedge has ZERO module-scope env reads.
  * Production imported ``tensorrt`` / ``cuda`` at module scope (guarded by a
    ``_HAS_TRT`` try/except). voxedge defers ALL of tensorrt + cuda into
    methods so the module imports on a CUDA-less box; ``preload()`` raises a
    clear error when the runtime is missing.
  * ``PREROLL_MS`` is a per-instance value derived from config (was a module
    constant), threaded through the preroll helpers as an arg.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from voxedge.backends.base import (
    ASRBackend,
    ASRCapability,
    ASRStream,
    TranscriptionResult,
)
from voxedge.engine.concurrency_capability import ConcurrencyCapability

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
#   PARAFORMER_MODEL_DIR     → model_dir ("/opt/models/paraformer-streaming")
#   PARAFORMER_ENC_ENGINE    → enc_engine (<dir>/engines/paraformer_encoder_sp1_80.plan)
#   PARAFORMER_ENC_ONNX      → enc_onnx (<dir>/encoder.onnx)
#   PARAFORMER_DEC_ONNX      → dec_onnx (<dir>/decoder.onnx)
#   PARAFORMER_DEC_ENGINE    → dec_engine (<dir>/engines/paraformer_decoder_fp16.plan)
#   PARAFORMER_TOKENS        → tokens_path (<dir>/tokens.txt)
#   PARAFORMER_PREROLL_MS    → preroll_ms (100)


@dataclass
class ParaformerTRTConfig:
    """Explicit construction-time config for :class:`ParaformerTRTBackend`."""

    model_dir: str = "/opt/models/paraformer-streaming"
    enc_engine: Optional[str] = None
    enc_onnx: Optional[str] = None
    dec_onnx: Optional[str] = None
    dec_engine: Optional[str] = None
    tokens_path: Optional[str] = None
    preroll_ms: int = 100

    def __post_init__(self) -> None:
        import os.path as _p
        d = self.model_dir
        if self.enc_engine is None:
            self.enc_engine = _p.join(d, "engines", "paraformer_encoder_sp1_80.plan")
        if self.enc_onnx is None:
            self.enc_onnx = _p.join(d, "encoder.onnx")
        if self.dec_onnx is None:
            self.dec_onnx = _p.join(d, "decoder.onnx")
        if self.dec_engine is None:
            self.dec_engine = _p.join(d, "engines", "paraformer_decoder_fp16.plan")
        if self.tokens_path is None:
            self.tokens_path = _p.join(d, "tokens.txt")
        self.preroll_ms = max(0, int(self.preroll_ms))


# Streaming parameters — match sherpa-onnx training distribution.
CHUNK_SIZE_SEC = 0.67
LEFT_CONTEXT_SEC = 2.68

# FBank parameters (kaldi-compatible)
SAMPLE_RATE = 16000
FFT_SIZE = 512
WINDOW_SIZE = 400
HOP_SIZE = 160
NUM_MEL_BINS = 80
NUM_STACKED = 7
NUM_STRIDE = 6
PRE_EMPH = 0.97
LOW_FREQ = 20
HIGH_FREQ = 8000

# CIF parameters
CIF_THRESHOLD = 1.0
RIGHT_LOOKAHEAD_LFR = 15
CIF_TAIL_THRESHOLD = 0.5

# Tokens
BLANK_ID = 0
SOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 8404

ENC_INPUT_NAMES = ["speech", "speech_lengths"]
ENC_OUTPUT_NAMES = ["enc", "enc_len", "alphas"]
DEC_INPUT_NAMES = ["enc", "enc_len", "acoustic_embeds", "acoustic_embeds_len"] + \
                  [f"in_cache_{i}" for i in range(16)]
DEC_OUTPUT_NAMES = ["logits", "sample_ids"] + \
                   [f"out_cache_{i}" for i in range(16)]

# ---------------------------------------------------------------------------
# FBank extraction (numpy-only, kaldi-style)
# ---------------------------------------------------------------------------

_MEL_FILTERBANK: Optional[np.ndarray] = None


def _get_mel_filterbank() -> np.ndarray:
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is not None:
        return _MEL_FILTERBANK

    num_bins = NUM_MEL_BINS
    fft_size = FFT_SIZE
    sr = SAMPLE_RATE
    low_mel = 2595.0 * np.log10(1.0 + LOW_FREQ / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + HIGH_FREQ / 700.0)
    mel_points = np.linspace(low_mel, high_mel, num_bins + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_indices = np.floor(hz_points * (fft_size // 2 + 1) / (sr / 2.0)).astype(np.int32)

    fbank = np.zeros((num_bins, fft_size // 2 + 1), dtype=np.float32)
    for i in range(num_bins):
        left, center, right = bin_indices[i], bin_indices[i + 1], bin_indices[i + 2]
        for j in range(left, center):
            fbank[i, j] = (j - left) / (center - left) if center != left else 1.0
        for j in range(center, right):
            fbank[i, j] = (right - j) / (right - center) if right != center else 1.0

    _MEL_FILTERBANK = fbank
    return fbank


def compute_fbank(audio: np.ndarray) -> np.ndarray:
    """Compute 80-dim log-fbank features from 16kHz audio. Returns [num_frames, 80]."""
    if len(audio) < WINDOW_SIZE:
        audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))

    audio = np.concatenate([[audio[0]], audio[1:] - PRE_EMPH * audio[:-1]])

    num_frames = (len(audio) - WINDOW_SIZE) // HOP_SIZE + 1
    frames = np.zeros((num_frames, WINDOW_SIZE), dtype=np.float32)
    for i in range(num_frames):
        start = i * HOP_SIZE
        frames[i] = audio[start:start + WINDOW_SIZE]

    hamming = np.hamming(WINDOW_SIZE).astype(np.float32)
    frames = frames * hamming

    spectrum = np.fft.rfft(frames, n=FFT_SIZE)
    power = (spectrum.real ** 2 + spectrum.imag ** 2) / FFT_SIZE

    fbank = _get_mel_filterbank()
    mel_feats = power @ fbank.T

    mel_feats = np.maximum(mel_feats, 1e-10)
    mel_feats = np.log(mel_feats)

    mean = mel_feats.mean(axis=0, keepdims=True)
    std = mel_feats.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-10)
    mel_feats = (mel_feats - mean) / std

    return mel_feats.astype(np.float32)


def stack_frames(feats: np.ndarray) -> np.ndarray:
    """Apply Paraformer LFR: stack 7 frames with stride 6. Returns [ceil(N/6), 560]."""
    n, d = feats.shape
    out_n = (n + NUM_STRIDE - 1) // NUM_STRIDE
    needed = (out_n - 1) * NUM_STRIDE + NUM_STACKED
    if needed > n:
        pad = np.repeat(feats[-1:], needed - n, axis=0)
        feats = np.concatenate([feats, pad], axis=0)
    stacked = np.zeros((out_n, d * NUM_STACKED), dtype=np.float32)
    for i in range(out_n):
        start = i * NUM_STRIDE
        stacked[i] = feats[start:start + NUM_STACKED].ravel()
    return stacked


# ---------------------------------------------------------------------------
# CIF (Continuous Integrate-and-Fire)
# ---------------------------------------------------------------------------

def cif(
    enc: np.ndarray,
    alphas: np.ndarray,
    threshold: float = CIF_THRESHOLD,
    tail_threshold: float = CIF_TAIL_THRESHOLD,
    carry_weight: float = 0.0,
    carry_embed: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Continuous Integrate-and-Fire for token boundary detection."""
    if carry_embed is None:
        carry_embed = np.zeros(512, dtype=np.float32)

    acoustic_embeds = []
    accum_weight = carry_weight
    accum_embed = carry_embed.copy()

    for t in range(len(enc)):
        alpha = float(alphas[t])
        if alpha <= 0:
            continue

        accum_weight += alpha
        accum_embed += alpha * enc[t]

        while accum_weight >= threshold:
            excess = accum_weight - threshold
            token_embed = (accum_embed - excess * enc[t]) / threshold
            acoustic_embeds.append(token_embed)
            accum_weight = excess
            accum_embed = excess * enc[t]

    return np.stack(acoustic_embeds) if acoustic_embeds else np.empty((0, 512), dtype=np.float32), \
           accum_weight, accum_embed


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def load_tokens(path: str) -> list[str]:
    """Load token-to-string mapping from tokens.txt."""
    with open(path, "r", encoding="utf-8") as f:
        tokens = []
        for line in f:
            token = line.rstrip("\n")
            parts = token.rsplit(None, 1)
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                token = parts[0]
            else:
                token = token.strip()
            tokens.append(token)
    return tokens


def decode_ids(token_ids: list[int], tokens: list[str]) -> str:
    """Decode token IDs to text, filtering special tokens."""
    pieces = []
    for tid in token_ids:
        if tid in (BLANK_ID, SOS_ID, EOS_ID):
            continue
        if 0 <= tid < len(tokens):
            token = tokens[tid]
            if token.startswith("<") and token.endswith(">"):
                continue
            if token.endswith("@@"):
                token = token[:-2]
            pieces.append(token)
    return "".join(pieces)


def add_preroll_silence(audio: np.ndarray, preroll_ms: int) -> np.ndarray:
    """Add a short leading context pad for zero-start utterances."""
    if preroll_ms <= 0 or len(audio) == 0:
        return audio
    pad = np.zeros(int(SAMPLE_RATE * preroll_ms / 1000), dtype=np.float32)
    return np.concatenate([pad, audio.astype(np.float32, copy=False)])


def initial_preroll_audio(preroll_ms: int) -> np.ndarray:
    if preroll_ms <= 0:
        return np.array([], dtype=np.float32)
    return np.zeros(int(SAMPLE_RATE * preroll_ms / 1000), dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-stream context bundle (N>=2 concurrency safety)
# ---------------------------------------------------------------------------


class _ParaformerCtxBundle:
    """Per-stream TRT execution contexts + device buffer cache.

    ``cuda``/``cudart`` import stays method-local so this module imports
    without CUDA.
    """

    def __init__(self, enc_engine, dec_engine):
        self.enc_ctx = enc_engine.create_execution_context() if enc_engine is not None else None
        self.dec_ctx = dec_engine.create_execution_context() if dec_engine is not None else None
        self.enc_bindings: dict[str, dict] = {}
        self.dec_bindings: dict[str, dict] = {}
        self.enc_active_profile: Optional[int] = None
        self._allocations: list[int] = []
        self._destroyed = False

    def alloc(self, nbytes: int) -> int:
        from cuda import cudart
        err, ptr = cudart.cudaMalloc(nbytes)
        if int(err) != 0:
            raise RuntimeError(f"cudaMalloc({nbytes}) failed: {err}")
        self._allocations.append(int(ptr))
        return int(ptr)

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        try:
            from cuda import cudart
            cudart.cudaDeviceSynchronize()
            for ptr in self._allocations:
                try:
                    cudart.cudaFree(ptr)
                except Exception:
                    pass
        except Exception:
            pass
        self._allocations.clear()
        self.enc_bindings.clear()
        self.dec_bindings.clear()
        self.enc_ctx = None
        self.dec_ctx = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ParaformerTRTStream
# ---------------------------------------------------------------------------

class ParaformerTRTStream(ASRStream):
    """Streaming ASR session backed by TRT encoder + decoder."""

    def __init__(self, backend: "ParaformerTRTBackend"):
        self._backend = backend
        self._tokens = backend._tokens
        self._preroll_ms = backend._config.preroll_ms

        self._audio_buf = initial_preroll_audio(self._preroll_ms)
        self._processed_chunks = 0

        left_ctx_samples = int(LEFT_CONTEXT_SEC * SAMPLE_RATE)
        self._history_audio = np.array([], dtype=np.float32)
        self._left_ctx_samples = left_ctx_samples
        self._all_audio = np.array([], dtype=np.float32)
        self._prev_total_frames = 0
        self._cif_processed_lfr = 0

        self._all_token_ids: list[int] = []
        self._partial_text: str = ""
        self._is_endpoint: bool = False

        self._carry_weight: float = 0.0
        self._carry_embed: np.ndarray = np.zeros(512, dtype=np.float32)

        self._cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]

        self._chunk_count = 0
        self._total_enc_ms = 0.0
        self._total_dec_ms = 0.0

        self._cancelled = False
        self._final_text_cache = ""

        self._ctx_bundle: Optional[_ParaformerCtxBundle] = backend.create_context_bundle()

    def close(self) -> None:
        bundle = self._ctx_bundle
        self._ctx_bundle = None
        if bundle is not None:
            try:
                bundle.destroy()
            except Exception:
                logger.exception("ParaformerTRTStream.close: bundle.destroy raised")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _reset_utterance_state(self) -> None:
        self._audio_buf = initial_preroll_audio(self._preroll_ms)
        self._processed_chunks = 0
        self._all_token_ids = []
        self._partial_text = ""
        self._is_endpoint = False
        self._carry_weight = 0.0
        self._carry_embed = np.zeros(512, dtype=np.float32)
        self._cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]
        self._chunk_count = 0
        self._total_enc_ms = 0.0
        self._total_dec_ms = 0.0
        self._history_audio = np.array([], dtype=np.float32)
        self._all_audio = np.array([], dtype=np.float32)
        self._prev_total_frames = 0
        self._cif_processed_lfr = 0

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if self._cancelled:
            return
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if sample_rate != SAMPLE_RATE:
            ratio = SAMPLE_RATE / sample_rate
            new_len = int(len(samples) * ratio)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)), samples,
            ).astype(np.float32)

        self._audio_buf = np.concatenate([self._audio_buf, samples])
        self._process_chunks()

    def _process_chunks(self) -> None:
        chunk_samples = int(CHUNK_SIZE_SEC * SAMPLE_RATE)
        while len(self._audio_buf) >= chunk_samples:
            chunk_audio = self._audio_buf[:chunk_samples]
            self._audio_buf = self._audio_buf[chunk_samples:]
            self._process_one_chunk(chunk_audio)

    def _process_one_chunk(self, audio: np.ndarray) -> None:
        t0 = time.perf_counter()

        self._all_audio = np.concatenate([self._all_audio, audio])

        all_feats = compute_fbank(self._all_audio)
        all_lfr = stack_frames(all_feats)
        cur_total_lfr = all_lfr.shape[0]
        new_lfr = cur_total_lfr - self._prev_total_frames
        if new_lfr <= 0:
            return
        self._prev_total_frames = cur_total_lfr

        feats = all_lfr if cur_total_lfr <= 400 else all_lfr[-400:]
        hist_stacked = feats.shape[0] - new_lfr
        new_stacked = new_lfr

        t1 = time.perf_counter()
        enc, alphas = self._backend._run_encoder(feats, self._ctx_bundle)
        enc_time = (time.perf_counter() - t1) * 1000
        self._total_enc_ms += enc_time

        logger.debug(
            "[paraformer-stream] enc_call frames_total=%d new_frames=%d history_frames=%d",
            feats.shape[0], new_stacked, hist_stacked,
        )

        if enc is None or alphas is None:
            logger.warning("Encoder returned None for chunk %d", self._chunk_count)
            self._chunk_count += 1
            return

        enc_t = enc[0]
        alphas_t = alphas[0]

        cif_end = enc_t.shape[0] - RIGHT_LOOKAHEAD_LFR
        cif_start = self._cif_processed_lfr
        if cif_end <= cif_start:
            self._chunk_count += 1
            return

        cif_enc = enc_t[cif_start:cif_end]
        cif_alphas = alphas_t[cif_start:cif_end]
        self._cif_processed_lfr = cif_end

        acoustic_embeds, self._carry_weight, self._carry_embed = cif(
            cif_enc, cif_alphas,
            carry_weight=self._carry_weight,
            carry_embed=self._carry_embed,
        )

        if len(acoustic_embeds) == 0:
            self._chunk_count += 1
            return

        t2 = time.perf_counter()
        sample_ids = self._backend._run_decoder(
            enc, enc.shape[1],
            acoustic_embeds, len(acoustic_embeds),
            self._cache,
            self._ctx_bundle,
        )
        dec_time = (time.perf_counter() - t2) * 1000
        self._total_dec_ms += dec_time
        self._chunk_count += 1

        if sample_ids is None:
            return

        new_ids = sample_ids.tolist()
        self._all_token_ids.extend(new_ids)
        old_len = len(self._partial_text)
        self._partial_text = decode_ids(self._all_token_ids, self._tokens)

        if old_len != len(self._partial_text):
            logger.debug(
                "Chunk %d: %d new tokens, text += '%s'",
                self._chunk_count, len(new_ids),
                self._partial_text[old_len:],
            )

    def get_partial(self) -> tuple[str, bool]:
        text = self._partial_text
        is_endpoint = self._is_endpoint
        if is_endpoint:
            self._reset_utterance_state()
        return text, is_endpoint

    def cancel_and_finalize(self) -> None:
        if self._cancelled:
            return
        self._final_text_cache = self._partial_text
        self._cancelled = True
        self._audio_buf = np.array([], dtype=np.float32)

    def finalize(self):
        if self._cancelled:
            return self._final_text_cache, None
        residual_audio = self._audio_buf
        if len(residual_audio) > 0:
            self._all_audio = np.concatenate([self._all_audio, residual_audio])

        if len(self._all_audio) >= WINDOW_SIZE:
            all_feats = compute_fbank(self._all_audio)
            all_lfr = stack_frames(all_feats)
            cur_total_lfr = all_lfr.shape[0]

            feats = all_lfr if cur_total_lfr <= 400 else all_lfr[-400:]
            enc, alphas = self._backend._run_encoder(feats, self._ctx_bundle)
            if enc is not None and alphas is not None:
                cif_start = max(self._cif_processed_lfr,
                                feats.shape[0] - cur_total_lfr + cur_total_lfr - feats.shape[0])
                cif_start_local = max(0, cif_start - (cur_total_lfr - feats.shape[0]))
                cif_enc = enc[0][cif_start_local:]
                cif_alphas = alphas[0][cif_start_local:]
                self._cif_processed_lfr = cur_total_lfr

                acoustic_embeds, self._carry_weight, self._carry_embed = cif(
                    cif_enc, cif_alphas,
                    carry_weight=self._carry_weight,
                    carry_embed=self._carry_embed,
                )

                if len(acoustic_embeds) > 0:
                    sample_ids = self._backend._run_decoder(
                        enc, enc.shape[1],
                        acoustic_embeds, len(acoustic_embeds),
                        self._cache,
                        self._ctx_bundle,
                    )
                    if sample_ids is not None:
                        self._all_token_ids.extend(sample_ids.tolist())
                        self._partial_text = decode_ids(self._all_token_ids, self._tokens)

        self._audio_buf = np.array([], dtype=np.float32)

        self._flush_cif_tail()

        text = self._partial_text
        chunk_count = self._chunk_count
        total_enc_ms = self._total_enc_ms
        total_dec_ms = self._total_dec_ms
        self._is_endpoint = True
        self._reset_utterance_state()

        logger.info(
            "Paraformer finalize: %d chunks, enc=%.0fms dec=%.0fms, text='%s'",
            chunk_count, total_enc_ms, total_dec_ms, text,
        )
        return text, None

    def _flush_cif_tail(self) -> None:
        if self._carry_weight >= CIF_TAIL_THRESHOLD:
            acoustic_embed = self._carry_embed / self._carry_weight
            acoustic_embeds = acoustic_embed[np.newaxis, :]
            dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
            sample_ids = self._backend._run_decoder(
                dummy_enc, 1,
                acoustic_embeds, 1,
                self._cache,
                self._ctx_bundle,
            )
            if sample_ids is not None:
                new_ids = sample_ids.tolist()
                self._all_token_ids.extend(new_ids)
                self._partial_text = decode_ids(self._all_token_ids, self._tokens)

    def force_endpoint(self) -> str:
        self._flush_cif_tail()
        text = self._partial_text
        self._is_endpoint = True
        self._reset_utterance_state()
        return text


# ---------------------------------------------------------------------------
# ParaformerTRTBackend
# ---------------------------------------------------------------------------

class ParaformerTRTBackend(ASRBackend):

    def concurrency_capability(self) -> ConcurrencyCapability:
        # Per-stream _ParaformerCtxBundle: each ASRStream owns its own enc/dec
        # TRT execution contexts + buffer cache. Backend holds shared engines
        # only; concurrency scales with open streams, bounded by VRAM.
        return ConcurrencyCapability(
            supports_parallel=True,
            max_concurrent=None,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="multi_runtime_per_slot",
        )

    def __init__(self, config: Optional[ParaformerTRTConfig] = None):
        self._config = config or ParaformerTRTConfig()
        self._engines: dict = {}
        self._enc_ort_session = None
        self._enc_provider = "trt"
        self._tokens: list[str] = []
        self._ready = False
        self._enc_profile_ranges: list[tuple[int, int, int]] = []

    def create_context_bundle(self) -> "_ParaformerCtxBundle":
        enc_eng = self._engines.get("enc")
        dec_eng = self._engines.get("dec")
        if dec_eng is None:
            raise RuntimeError(
                "Paraformer TRT decoder engine not loaded; call preload() first"
            )
        return _ParaformerCtxBundle(enc_eng, dec_eng)

    @property
    def name(self) -> str:
        return "paraformer_trt"

    @property
    def providers(self) -> dict[str, str]:
        return {"encoder": self._enc_provider, "decoder": "trt"}

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        import os
        try:
            import tensorrt  # noqa: F401
            from cuda import cudart  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT + CUDA Python not available; paraformer_trt requires "
                "a CUDA/TensorRT runtime"
            ) from exc

        for label, path in [("encoder engine", self._config.enc_engine),
                            ("decoder engine", self._config.dec_engine)]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Paraformer {label} not found: {path}")
        if not os.path.isfile(self._config.tokens_path):
            raise FileNotFoundError(f"Paraformer tokens not found: {self._config.tokens_path}")

        self._tokens = load_tokens(self._config.tokens_path)
        logger.info("Loaded %d tokens from %s", len(self._tokens), self._config.tokens_path)

        self._engines["enc"] = self._load_trt_engine(self._config.enc_engine)
        eng = self._engines["enc"]
        tensor_names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        logger.info("Encoder engine (%d I/O): %s", len(tensor_names), tensor_names)
        self._enc_profile_ranges = self._load_encoder_profile_ranges(eng)

        self._engines["dec"] = self._load_trt_engine(self._config.dec_engine)
        dec_eng = self._engines["dec"]
        dec_tensor_names = [dec_eng.get_tensor_name(i) for i in range(dec_eng.num_io_tensors)]
        logger.info("Decoder engine (%d I/O): %s", len(dec_tensor_names), dec_tensor_names)

        warmup_audio = (np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE) * 0.3).astype(np.float32)
        warmup_feats = compute_fbank(warmup_audio)
        warmup_feats = stack_frames(warmup_feats)
        n_warmup = min(warmup_feats.shape[0], 40)
        warmup_feats = warmup_feats[:n_warmup]

        warmup_bundle = self.create_context_bundle()
        try:
            enc, alphas = self._run_encoder_trt(warmup_feats, warmup_bundle)
            if enc is not None and alphas is not None and not np.isnan(alphas).any():
                logger.info("Encoder TRT engine validated (no NaN)")
                self._enc_provider = "trt"
            else:
                logger.warning(
                    "Encoder TRT engine produces NaN, falling back to ORT CUDA EP."
                )
                self._enc_provider = "ort_cuda"
                import onnxruntime
                enc_ort_opts = onnxruntime.SessionOptions()
                enc_ort_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                enc_ort_opts.log_severity_level = 3
                self._enc_ort_session = onnxruntime.InferenceSession(
                    self._config.enc_onnx,
                    sess_options=enc_ort_opts,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                logger.info("Encoder ORT session loaded (providers: %s)", self._enc_ort_session.get_providers())
                self._run_encoder_ort(warmup_feats)

            dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
            dummy_ae = np.zeros((1, 512), dtype=np.float32)
            dummy_cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]
            self._run_decoder(dummy_enc, 1, dummy_ae, 1, dummy_cache, warmup_bundle)
        finally:
            try:
                warmup_bundle.destroy()
            except Exception:
                logger.exception("paraformer preload: warmup_bundle destroy raised")

        logger.info("Paraformer TRT backend ready (encoder=%s, decoder=trt)", self._enc_provider)
        self._ready = True

    @staticmethod
    def _load_trt_engine(path: str):
        import tensorrt as trt
        logger_obj = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger_obj)
        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if not self._ready:
            raise RuntimeError("Paraformer TRT backend not loaded; call preload() first")

        try:
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            import wave
            with wave.open(io.BytesIO(audio_bytes)) as w:
                sr = w.getframerate()
                raw = w.readframes(w.getnframes())
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        if data.ndim > 1:
            data = data.mean(axis=1)

        if sr != SAMPLE_RATE:
            ratio = SAMPLE_RATE / sr
            new_len = int(len(data) * ratio)
            data = np.interp(
                np.linspace(0, len(data) - 1, new_len),
                np.arange(len(data)), data,
            ).astype(np.float32)
        data = add_preroll_silence(data, self._config.preroll_ms)

        feats = compute_fbank(data)
        feats = stack_frames(feats)

        ENGINE_MAX_FRAMES = 400
        chunk_frames = min(ENGINE_MAX_FRAMES, max(40, feats.shape[0]))
        all_token_ids: list[int] = []
        carry_w = 0.0
        carry_e = np.zeros(512, dtype=np.float32)
        cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]

        bundle = self.create_context_bundle()
        try:
            for start in range(0, feats.shape[0], chunk_frames):
                chunk = feats[start:start + chunk_frames]
                if chunk.shape[0] < chunk_frames:
                    pad = np.zeros((chunk_frames - chunk.shape[0], 560), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=0)

                enc, alphas = self._run_encoder(chunk, bundle)
                if enc is None:
                    continue

                enc_t = enc[0]
                alphas_t = alphas[0]

                acoustic_embeds, carry_w, carry_e = cif(
                    enc_t, alphas_t, carry_weight=carry_w, carry_embed=carry_e,
                )

                if len(acoustic_embeds) == 0:
                    continue

                sample_ids = self._run_decoder(
                    enc, alphas.shape[1],
                    acoustic_embeds, len(acoustic_embeds),
                    cache,
                    bundle,
                )
                if sample_ids is not None:
                    all_token_ids.extend(sample_ids.tolist())

            if carry_w >= CIF_TAIL_THRESHOLD:
                acoustic_embeds = (carry_e / carry_w)[np.newaxis, :]
                dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
                sample_ids = self._run_decoder(
                    dummy_enc, 1, acoustic_embeds, 1, cache, bundle,
                )
                if sample_ids is not None:
                    all_token_ids.extend(sample_ids.tolist())
        finally:
            try:
                bundle.destroy()
            except Exception:
                logger.exception("paraformer transcribe: bundle destroy raised")

        full_text = decode_ids(all_token_ids, self._tokens)
        return TranscriptionResult(text=full_text, language=language)

    def transcribe_audio(self, audio: np.ndarray, language: str = "auto") -> TranscriptionResult:
        """Transcribe float32 audio array (16kHz, [-1,1])."""
        if not self._ready:
            raise RuntimeError("Paraformer TRT backend not loaded; call preload() first")

        audio = add_preroll_silence(audio, self._config.preroll_ms)
        feats = compute_fbank(audio)
        feats = stack_frames(feats)

        ENGINE_MAX_FRAMES = 400
        chunk_frames = min(ENGINE_MAX_FRAMES, max(40, feats.shape[0]))
        all_text_parts = []
        carry_w = 0.0
        carry_e = np.zeros(512, dtype=np.float32)
        cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]

        bundle = self.create_context_bundle()
        try:
            for start in range(0, feats.shape[0], chunk_frames):
                chunk = feats[start:start + chunk_frames]
                if chunk.shape[0] < chunk_frames:
                    pad = np.zeros((chunk_frames - chunk.shape[0], 560), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=0)

                enc, alphas = self._run_encoder(chunk, bundle)
                if enc is None:
                    continue

                enc_t = enc[0]
                alphas_t = alphas[0]

                acoustic_embeds, carry_w, carry_e = cif(
                    enc_t, alphas_t, carry_weight=carry_w, carry_embed=carry_e,
                )

                if len(acoustic_embeds) == 0:
                    continue

                sample_ids = self._run_decoder(
                    enc, alphas.shape[1],
                    acoustic_embeds, len(acoustic_embeds),
                    cache,
                    bundle,
                )
                if sample_ids is not None:
                    new_ids = sample_ids.tolist()
                    text = decode_ids(new_ids, self._tokens)
                    if text:
                        all_text_parts.append(text)

            if carry_w >= CIF_TAIL_THRESHOLD:
                acoustic_embeds = (carry_e / carry_w)[np.newaxis, :]
                dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
                sample_ids = self._run_decoder(
                    dummy_enc, 1, acoustic_embeds, 1, cache, bundle,
                )
                if sample_ids is not None:
                    text = decode_ids(sample_ids.tolist(), self._tokens)
                    if text:
                        all_text_parts.append(text)
        finally:
            try:
                bundle.destroy()
            except Exception:
                logger.exception("paraformer transcribe_audio: bundle destroy raised")

        full_text = "".join(all_text_parts)
        return TranscriptionResult(text=full_text, language=language)

    def create_stream(self, language: str = "auto") -> ASRStream:
        if not self._ready:
            raise RuntimeError("Paraformer TRT backend not loaded")
        return ParaformerTRTStream(self)

    @staticmethod
    def _cuda_err(result):
        if isinstance(result, tuple):
            return result[0]
        return result

    def _run_encoder(
        self,
        feats: np.ndarray,
        bundle: "_ParaformerCtxBundle",
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._enc_provider == "ort_cuda":
            return self._run_encoder_ort(feats)
        return self._run_encoder_trt(feats, bundle)

    def _run_encoder_ort(self, feats: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        n_frames = feats.shape[0]
        enc_min_frames = 40
        orig_n_frames = n_frames

        if n_frames < enc_min_frames:
            pad_len = enc_min_frames - n_frames
            feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="edge")
            n_frames = enc_min_frames

        speech = np.ascontiguousarray(feats[np.newaxis, :].astype(np.float32))
        speech_len = np.array([n_frames], dtype=np.int32)

        outputs = self._enc_ort_session.run(
            output_names=["enc", "enc_len", "alphas"],
            input_feed={"speech": speech, "speech_lengths": speech_len},
        )

        enc_out, enc_len_out, alphas_out = outputs

        if orig_n_frames < n_frames:
            enc_out = enc_out[:, :orig_n_frames, :]
            alphas_out = alphas_out[:, :orig_n_frames]

        return enc_out, alphas_out

    def _run_encoder_trt(
        self,
        feats: np.ndarray,
        bundle: "_ParaformerCtxBundle",
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        from cuda import cudart
        ctx = bundle.enc_ctx
        n_frames = feats.shape[0]
        profile_idx, enc_min_frames, enc_max_frames = self._select_encoder_profile(n_frames)
        if n_frames > enc_max_frames:
            logger.error(
                "Encoder TRT input has %d frames, exceeding selected profile %d max=%d",
                n_frames, profile_idx, enc_max_frames,
            )
            return None, None

        orig_n_frames = n_frames
        if n_frames < enc_min_frames:
            pad_len = enc_min_frames - n_frames
            feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="edge")
            n_frames = enc_min_frames

        key = f"enc_p{profile_idx}_{n_frames}"
        if key not in bundle.enc_bindings:
            bundle.enc_bindings[key] = self._alloc_enc_buffers(n_frames, bundle)

        bufs = bundle.enc_bindings[key]

        err, stream = cudart.cudaStreamCreate()
        if self._cuda_err(err) != 0:
            logger.error("cudaStreamCreate failed: %s", err)
            return None, None

        if bundle.enc_active_profile != profile_idx:
            cudart.cudaDeviceSynchronize()
            if hasattr(ctx, "set_optimization_profile_async"):
                success = ctx.set_optimization_profile_async(profile_idx, stream)
                if not success:
                    logger.error(
                        "Encoder TRT set_optimization_profile_async failed (profile=%d, n_frames=%d)",
                        profile_idx, n_frames,
                    )
                    cudart.cudaStreamDestroy(stream)
                    return None, None
                cudart.cudaStreamSynchronize(stream)
            elif hasattr(ctx, "active_optimization_profile"):
                ctx.active_optimization_profile = profile_idx
            bundle.enc_active_profile = profile_idx

        ctx.set_tensor_address("speech", bufs["speech"])
        ctx.set_tensor_address("speech_lengths", bufs["speech_lengths"])
        ctx.set_tensor_address("enc", bufs["enc"])
        ctx.set_tensor_address("enc_len", bufs["enc_len"])
        ctx.set_tensor_address("alphas", bufs["alphas"])

        ctx.set_input_shape("speech", (1, n_frames, 560))
        ctx.set_input_shape("speech_lengths", (1,))

        speech = np.ascontiguousarray(feats[np.newaxis, :])
        err = cudart.cudaMemcpy(
            bufs["speech"], speech.ctypes.data, speech.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if self._cuda_err(err) != 0:
            cudart.cudaStreamDestroy(stream)
            return None, None

        speech_len = np.array([n_frames], dtype=np.int32)
        err = cudart.cudaMemcpy(
            bufs["speech_lengths"], speech_len.ctypes.data, speech_len.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if self._cuda_err(err) != 0:
            cudart.cudaStreamDestroy(stream)
            return None, None

        success = ctx.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)
        cudart.cudaStreamDestroy(stream)

        if not success:
            logger.error("Encoder TRT execute_async_v3 failed (n_frames=%d)", n_frames)
            return None, None

        enc_out = np.empty((1, n_frames, 512), dtype=np.float32)
        err = cudart.cudaMemcpy(
            enc_out.ctypes.data, bufs["enc"], enc_out.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if self._cuda_err(err) != 0:
            return None, None

        alphas_out = np.empty((1, n_frames), dtype=np.float32)
        err = cudart.cudaMemcpy(
            alphas_out.ctypes.data, bufs["alphas"], alphas_out.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if self._cuda_err(err) != 0:
            return None, None

        if orig_n_frames < n_frames:
            enc_out = enc_out[:, :orig_n_frames, :]
            alphas_out = alphas_out[:, :orig_n_frames]

        return enc_out, alphas_out

    def _load_encoder_profile_ranges(self, eng) -> list[tuple[int, int, int]]:
        ranges: list[tuple[int, int, int]] = []
        n_profiles = getattr(eng, "num_optimization_profiles", 1)
        for profile_idx in range(n_profiles):
            try:
                shapes = eng.get_tensor_profile_shape("speech", profile_idx)
                min_frames = int(shapes[0][1])
                max_frames = int(shapes[2][1])
            except Exception as exc:
                logger.warning(
                    "Unable to inspect encoder profile %d shape range: %s", profile_idx, exc,
                )
                continue
            ranges.append((profile_idx, min_frames, max_frames))
        if not ranges:
            ranges = [(0, 40, 400)]
        ranges.sort(key=lambda item: (item[1], item[2]))
        logger.info("Encoder profile ranges: %s", ranges)
        return ranges

    def _select_encoder_profile(self, n_frames: int) -> tuple[int, int, int]:
        ranges = self._enc_profile_ranges or [(0, 40, 400)]

        compatible = [
            item for item in ranges
            if item[1] <= n_frames <= item[2]
        ]
        if compatible:
            return min(compatible, key=lambda item: item[2] - item[1])

        pad_candidates = [item for item in ranges if n_frames < item[1]]
        if pad_candidates:
            return min(pad_candidates, key=lambda item: item[1])

        return max(ranges, key=lambda item: item[2])

    def _alloc_enc_buffers(self, n_frames: int, bundle: "_ParaformerCtxBundle") -> dict:
        bufs = {}
        bufs["speech"] = bundle.alloc(1 * n_frames * 560 * 4)
        bufs["speech_lengths"] = bundle.alloc(4)
        bufs["enc"] = bundle.alloc(1 * n_frames * 512 * 4)
        bufs["enc_len"] = bundle.alloc(4)
        bufs["alphas"] = bundle.alloc(1 * n_frames * 4)
        return bufs

    def _run_decoder(
        self,
        enc: np.ndarray,
        enc_len: int,
        acoustic_embeds: np.ndarray,
        acoustic_embeds_len: int,
        cache: list[np.ndarray],
        bundle: "_ParaformerCtxBundle",
    ) -> Optional[np.ndarray]:
        from cuda import cudart
        ctx = bundle.dec_ctx if bundle is not None else None
        if ctx is None:
            logger.error("Decoder TRT context not available (bundle missing or destroyed)")
            return None

        n_tokens = acoustic_embeds.shape[0]
        if n_tokens == 0:
            return np.array([], dtype=np.int64)

        enc_nframes = enc.shape[1]
        key = f"dec_{enc_nframes}_{n_tokens}"

        if key not in bundle.dec_bindings:
            bundle.dec_bindings[key] = self._alloc_dec_buffers(enc_nframes, n_tokens, bundle)
        bufs = bundle.dec_bindings[key]

        ctx.set_tensor_address("enc", bufs["enc"])
        ctx.set_tensor_address("enc_len", bufs["enc_len"])
        ctx.set_tensor_address("acoustic_embeds", bufs["acoustic_embeds"])
        ctx.set_tensor_address("acoustic_embeds_len", bufs["acoustic_embeds_len"])
        ctx.set_tensor_address("pad_mask", bufs["pad_mask"])
        ctx.set_tensor_address("enc_pad_mask", bufs["enc_pad_mask"])
        for i in range(16):
            ctx.set_tensor_address(f"in_cache_{i}", bufs[f"in_cache_{i}"])
            ctx.set_tensor_address(f"out_cache_{i}", bufs[f"out_cache_{i}"])
        ctx.set_tensor_address("logits", bufs["logits"])
        ctx.set_tensor_address("sample_ids", bufs["sample_ids"])

        ctx.set_input_shape("enc", (1, enc_nframes, 512))
        ctx.set_input_shape("acoustic_embeds", (1, n_tokens, 512))
        ctx.set_input_shape("pad_mask", (1, n_tokens))
        ctx.set_input_shape("enc_pad_mask", (1, enc_nframes))

        err, stream = cudart.cudaStreamCreate()
        if self._cuda_err(err) != 0:
            logger.error("cudaStreamCreate failed for decoder")
            return None

        try:
            cudart.cudaMemcpy(bufs["enc"], enc.ctypes.data, enc.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            cudart.cudaMemcpy(bufs["enc_len"], np.array([enc_nframes], dtype=np.int32).ctypes.data, 4,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            ae_batch = np.ascontiguousarray(acoustic_embeds[np.newaxis, :])
            cudart.cudaMemcpy(bufs["acoustic_embeds"], ae_batch.ctypes.data, ae_batch.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            cudart.cudaMemcpy(bufs["acoustic_embeds_len"], np.array([n_tokens], dtype=np.int32).ctypes.data, 4,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            pad_mask = np.ones((1, n_tokens), dtype=np.float32)
            enc_pad_mask = np.ones((1, enc_nframes), dtype=np.float32)
            cudart.cudaMemcpy(bufs["pad_mask"], pad_mask.ctypes.data, pad_mask.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            cudart.cudaMemcpy(bufs["enc_pad_mask"], enc_pad_mask.ctypes.data, enc_pad_mask.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            for i in range(16):
                cudart.cudaMemcpy(bufs[f"in_cache_{i}"], cache[i].ctypes.data, cache[i].nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            success = ctx.execute_async_v3(stream)
            cudart.cudaStreamSynchronize(stream)

            if not success:
                logger.error("Decoder TRT execute_async_v3 failed (enc=%d, tokens=%d)", enc_nframes, n_tokens)
                return None

            sample_ids = np.empty((1, n_tokens), dtype=np.int64)
            cudart.cudaMemcpy(sample_ids.ctypes.data, bufs["sample_ids"], sample_ids.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            for i in range(16):
                cudart.cudaMemcpy(cache[i].ctypes.data, bufs[f"out_cache_{i}"], cache[i].nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        finally:
            cudart.cudaStreamDestroy(stream)

        return sample_ids[0]

    def _alloc_dec_buffers(self, enc_nframes: int, n_tokens: int, bundle: "_ParaformerCtxBundle") -> dict:
        bufs = {}
        bufs["enc"] = bundle.alloc(1 * enc_nframes * 512 * 4)
        bufs["enc_len"] = bundle.alloc(4)
        bufs["acoustic_embeds"] = bundle.alloc(1 * n_tokens * 512 * 4)
        bufs["acoustic_embeds_len"] = bundle.alloc(4)
        bufs["pad_mask"] = bundle.alloc(n_tokens * 4)
        bufs["enc_pad_mask"] = bundle.alloc(enc_nframes * 4)
        for i in range(16):
            bufs[f"in_cache_{i}"] = bundle.alloc(1 * 512 * 10 * 4)
            bufs[f"out_cache_{i}"] = bundle.alloc(1 * 512 * 10 * 4)
        bufs["logits"] = bundle.alloc(1 * n_tokens * 8404 * 4)
        bufs["sample_ids"] = bundle.alloc(1 * n_tokens * 8)
        return bufs
