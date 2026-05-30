"""Product-layer config builders: env/profile → voxedge backend config.

voxedge backends are env-free: they take an explicit config dataclass at
construction (``voxedge.backends.jetson.trt_edge_llm_asr.TRTEdgeLLMASRConfig`` /
``voxedge.backends.jetson.matcha_trt.MatchaTRTConfig``) with path/engine fields
that default to empty / a layout root. The product, by contrast, resolves all
those values from ``os.environ`` (and an optional ASR manifest JSON) exactly
the way the legacy ``app/backends/jetson`` backends did.

These builders are the single translation layer: they read the SAME env vars
(and manifest) the legacy ``_load_config`` / ``_resolve_matcha_paths`` read,
with byte-identical defaults, and emit the voxedge config dataclass. Keeping
the mapping here (not in voxedge) preserves voxedge's zero-env property.

Field-by-field mapping is documented inline against the legacy source:
  ASR  ← app/backends/jetson/trt_edge_llm_asr.py ``_load_config`` (+ module env)
  TTS  ← app/backends/jetson/matcha_trt.py ``_resolve_matcha_paths`` (+ env)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    """Match legacy ``trt_edge_llm_asr._env_bool``."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no")


def _profile_get(profile: Optional[dict], *keys, default=None):
    """Read the first present top-level key from a profile dict."""
    if not isinstance(profile, dict):
        return default
    for key in keys:
        if key in profile and profile[key] is not None:
            return profile[key]
    return default


def build_trt_edge_llm_asr_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``TRTEdgeLLMASRConfig`` from env + optional ASR manifest.

    Mirrors ``TRTEdgeLLMASRBackend._load_config`` field-for-field. ``env`` is
    accepted for symmetry/testing but the legacy backend read ``os.environ``
    directly, so we default to it. The manifest JSON (``EDGE_LLM_ASR_MANIFEST``)
    supplies fallbacks below each env var, identical to legacy precedence
    ``env → manifest → hardcoded default``.

    env / profile → TRTEdgeLLMASRConfig field map (legacy _load_config):
      EDGE_LLM_ASR_BIN              → asr_binary            (manifest asr_binary / ASR_BINARY)
      EDGE_LLM_ASR_WORKER_BIN       → worker_binary         (manifest worker_binary / ASR_WORKER_BINARY)
      EDGE_LLM_ASR_PLUGIN_PATH      → plugin_path           (or EDGELLM_ASR_PLUGIN_PATH / manifest / ASR_PLUGIN_PATH)
      EDGE_LLM_ASR_ENGINE_DIR       → engine_dir            (manifest engine_dir / ASR_ENGINE_DIR)
      EDGE_LLM_ASR_AUDIO_ENC_DIR    → audio_encoder_dir     (manifest audio_encoder_dir / ASR_AUDIO_ENC_DIR)
      EDGE_LLM_ASR_WORKER           → use_worker            (manifest use_worker / True)
      EDGE_LLM_ASR_MEL_TENSOR_NAME  → mel_tensor_name       (manifest mel_tensor_name / "mel")
      EDGE_LLM_ASR_MAX_MEL_FRAMES   → max_mel_frames        (manifest max_mel_frames / 6000)
      EDGE_LLM_ASR_MAX_CONCURRENT   → max_slots             (manifest asr_max_slots/max_concurrent / profile asr_max_slots / 1)
      EDGE_LLM_ASR_STREAM_MODE      → stream_mode           (manifest stream_mode / "accumulate")
      EDGE_LLM_ASR_STREAM_CHUNK_SEC → stream_chunk_sec      (manifest stream_chunk_sec / 0.5)
      EDGE_LLM_ASR_STREAM_UNFIXED_CHUNKS → stream_unfixed_chunks (manifest / 2)
      EDGE_LLM_ASR_STREAM_UNFIXED_TOKENS → stream_unfixed_tokens (manifest / 5)
      EDGE_LLM_ASR_MEL_SETTINGS     → mel_settings_path     (manifest mel_settings_path / "")
      EDGE_LLM_ASR_MEL_FILTERS      → mel_filters_path      (manifest mel_filters_path / "")
      ASR_TEMPERATURE               → temperature           (1.0)
      ASR_TOP_P                     → top_p                 (1.0)
      ASR_TOP_K                     → top_k                 (1)
      ASR_MAX_GENERATE_LENGTH       → max_generate_length   (200)
      EDGE_LLM_ASR_OFFLINE_SEGMENT  → offline_segment_enabled (True)
      EDGE_LLM_ASR_OFFLINE_SEGMENT_SEC → offline_segment_threshold_s (6.0)
      EDGE_LLM_ASR_OFFLINE_MIN_SEGMENT_SEC → offline_segment_min_s (0.4)
      SKIP_ASR_WARMUP / EDGE_LLM_ASR_WORKER_WARMUP → worker_warmup (True)
      EDGE_LLM_ASR_PREWARM_MAX      → prewarm_max           (6)
      EDGE_LLM_ASR_CUDA_GRAPH       → worker_cuda_graph     ("0", via extra_worker_env passthrough)

    NB legacy module-level path defaults (ASR_BINARY etc.) live in
    ``app.backends.jetson.trt_edge_llm_ipc``; we import them so the empty-string
    voxedge defaults are replaced by the real production artifact-tree paths.
    """
    from voxedge.backends.jetson.trt_edge_llm_asr import TRTEdgeLLMASRConfig
    from app.backends.jetson.trt_edge_llm_ipc import (
        ASR_BINARY,
        ASR_WORKER_BINARY,
        ASR_ENGINE_DIR,
        ASR_AUDIO_ENC_DIR,
        ASR_PLUGIN_PATH,
    )

    if env is None:
        env = os.environ

    # -- manifest (EDGE_LLM_ASR_MANIFEST) --
    manifest: dict = {}
    manifest_path = env.get("EDGE_LLM_ASR_MANIFEST")
    if manifest_path:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    use_worker_default = bool(manifest.get("use_worker", True))

    # -- slot ceiling: env → manifest → profile → 1 (matches _load_config +
    #    concurrency_capability precedence) --
    profile_slots = _profile_get(profile, "asr_max_slots")
    if profile_slots is None:
        asr_cfg = _profile_get(profile, "asr")
        if isinstance(asr_cfg, dict):
            profile_slots = asr_cfg.get("asr_max_slots", asr_cfg.get("max_concurrent"))
    max_slots_raw = env.get(
        "EDGE_LLM_ASR_MAX_CONCURRENT",
        str(
            manifest.get(
                "asr_max_slots",
                manifest.get("max_concurrent", profile_slots if profile_slots is not None else 1),
            )
        ),
    )

    # -- warmup: legacy gates on SKIP_ASR_WARMUP OR EDGE_LLM_ASR_WORKER_WARMUP --
    skip_warmup = env.get("SKIP_ASR_WARMUP", "").lower() in ("1", "true", "yes")
    warmup_disabled = env.get("EDGE_LLM_ASR_WORKER_WARMUP", "1").lower() in ("0", "false", "no")
    worker_warmup = not (skip_warmup or warmup_disabled)

    try:
        prewarm_max = int(env.get("EDGE_LLM_ASR_PREWARM_MAX", "6"))
    except ValueError:
        prewarm_max = 6

    # min_audio_frames: legacy read EDGE_LLM_ASR_MIN_AUDIO_FRAMES at module
    # scope in trt_edge_llm_ipc (default 100); voxedge lifts it to a config
    # field consumed by audio_bytes_to_mel.
    try:
        min_audio_frames = int(env.get("EDGE_LLM_ASR_MIN_AUDIO_FRAMES", "100"))
    except ValueError:
        min_audio_frames = 100

    cfg = TRTEdgeLLMASRConfig(
        asr_binary=env.get("EDGE_LLM_ASR_BIN", manifest.get("asr_binary", ASR_BINARY)),
        worker_binary=env.get(
            "EDGE_LLM_ASR_WORKER_BIN", manifest.get("worker_binary", ASR_WORKER_BINARY)
        ),
        plugin_path=env.get(
            "EDGE_LLM_ASR_PLUGIN_PATH",
            env.get(
                "EDGELLM_ASR_PLUGIN_PATH",
                manifest.get("asr_plugin_path", manifest.get("plugin_path", ASR_PLUGIN_PATH)),
            ),
        ),
        engine_dir=env.get("EDGE_LLM_ASR_ENGINE_DIR", manifest.get("engine_dir", ASR_ENGINE_DIR)),
        audio_encoder_dir=env.get(
            "EDGE_LLM_ASR_AUDIO_ENC_DIR", manifest.get("audio_encoder_dir", ASR_AUDIO_ENC_DIR)
        ),
        use_worker=_env_bool("EDGE_LLM_ASR_WORKER", use_worker_default),
        mel_tensor_name=env.get(
            "EDGE_LLM_ASR_MEL_TENSOR_NAME", manifest.get("mel_tensor_name", "mel")
        ),
        max_mel_frames=int(
            env.get("EDGE_LLM_ASR_MAX_MEL_FRAMES", str(manifest.get("max_mel_frames", 6000)))
        ),
        max_slots=max(1, int(max_slots_raw)),
        stream_mode=env.get(
            "EDGE_LLM_ASR_STREAM_MODE", manifest.get("stream_mode", "accumulate")
        ),
        stream_chunk_sec=float(
            env.get("EDGE_LLM_ASR_STREAM_CHUNK_SEC", str(manifest.get("stream_chunk_sec", 0.5)))
        ),
        stream_unfixed_chunks=int(
            env.get(
                "EDGE_LLM_ASR_STREAM_UNFIXED_CHUNKS",
                str(manifest.get("stream_unfixed_chunks", 2)),
            )
        ),
        stream_unfixed_tokens=int(
            env.get(
                "EDGE_LLM_ASR_STREAM_UNFIXED_TOKENS",
                str(manifest.get("stream_unfixed_tokens", 5)),
            )
        ),
        mel_settings_path=env.get(
            "EDGE_LLM_ASR_MEL_SETTINGS", manifest.get("mel_settings_path", "")
        ),
        mel_filters_path=env.get(
            "EDGE_LLM_ASR_MEL_FILTERS", manifest.get("mel_filters_path", "")
        ),
        temperature=float(env.get("ASR_TEMPERATURE", "1.0")),
        top_p=float(env.get("ASR_TOP_P", "1.0")),
        top_k=int(env.get("ASR_TOP_K", "1")),
        max_generate_length=int(env.get("ASR_MAX_GENERATE_LENGTH", "200")),
        min_audio_frames=min_audio_frames,
        offline_segment_enabled=_env_bool("EDGE_LLM_ASR_OFFLINE_SEGMENT", True),
        offline_segment_threshold_s=float(
            env.get("EDGE_LLM_ASR_OFFLINE_SEGMENT_SEC", "6.0")
        ),
        offline_segment_min_s=float(
            env.get("EDGE_LLM_ASR_OFFLINE_MIN_SEGMENT_SEC", "0.4")
        ),
        worker_warmup=worker_warmup,
        prewarm_max=prewarm_max,
        worker_cuda_graph=env.get("EDGE_LLM_ASR_CUDA_GRAPH", "0"),
    )
    return cfg


def build_paraformer_trt_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``ParaformerTRTConfig`` from env.

    Mirrors the module-scope ``PARAFORMER_*`` path reads + ``PARAFORMER_PREROLL_MS``
    in legacy ``app/backends/jetson/paraformer_trt.py``. Path fields default to
    ``<model_dir>/...`` exactly like the legacy ``os.path.join`` defaults.

    env → ParaformerTRTConfig field map (legacy module scope):
      PARAFORMER_MODEL_DIR   → model_dir   ("/opt/models/paraformer-streaming")
      PARAFORMER_ENC_ENGINE  → enc_engine  (<dir>/engines/paraformer_encoder_sp1_80.plan)
      PARAFORMER_ENC_ONNX    → enc_onnx    (<dir>/encoder.onnx)
      PARAFORMER_DEC_ONNX    → dec_onnx    (<dir>/decoder.onnx)
      PARAFORMER_DEC_ENGINE  → dec_engine  (<dir>/engines/paraformer_decoder_fp16.plan)
      PARAFORMER_TOKENS      → tokens_path (<dir>/tokens.txt)
      PARAFORMER_PREROLL_MS  → preroll_ms  (100, clamped >=0)
    """
    from voxedge.backends.jetson.paraformer_trt import ParaformerTRTConfig

    if env is None:
        env = os.environ

    model_dir = env.get("PARAFORMER_MODEL_DIR", "/opt/models/paraformer-streaming")
    base = model_dir

    try:
        preroll_ms = int(env.get("PARAFORMER_PREROLL_MS", "100"))
    except ValueError:
        preroll_ms = 100

    return ParaformerTRTConfig(
        model_dir=model_dir,
        enc_engine=env.get("PARAFORMER_ENC_ENGINE")
        or os.path.join(base, "engines", "paraformer_encoder_sp1_80.plan"),
        enc_onnx=env.get("PARAFORMER_ENC_ONNX") or os.path.join(base, "encoder.onnx"),
        dec_onnx=env.get("PARAFORMER_DEC_ONNX") or os.path.join(base, "decoder.onnx"),
        dec_engine=env.get("PARAFORMER_DEC_ENGINE")
        or os.path.join(base, "engines", "paraformer_decoder_fp16.plan"),
        tokens_path=env.get("PARAFORMER_TOKENS") or os.path.join(base, "tokens.txt"),
        preroll_ms=preroll_ms,
    )


def build_sherpa_asr_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``SherpaASRConfig`` from env.

    Mirrors the module-scope reads in legacy ``app/backends/cpu/sherpa_asr.py``.
    ``streaming_model_dir`` / ``offline_provider`` are left ``None`` so the
    dataclass ``__post_init__`` reproduces the legacy language-conditional /
    provider-fallback defaults exactly.

    env → SherpaASRConfig field map (legacy module scope):
      LANGUAGE_MODE              → language_mode ("zh_en")
      STREAMING_MODEL_DIR        → streaming_model_dir (None → __post_init__ picks per language_mode)
      STREAMING_ASR_PROVIDER     → streaming_provider ("cuda")
      OFFLINE_ASR_PROVIDER / ASR_PROVIDER → offline_provider (None → __post_init__ = streaming_provider)
      STREAMING_ASR_NUM_THREADS  → num_threads (4)
      MODEL_DIR                  → model_root ("/opt/models")
    """
    from voxedge.backends.sherpa.asr import SherpaASRConfig

    if env is None:
        env = os.environ

    streaming_provider = env.get("STREAMING_ASR_PROVIDER", "cuda")
    # legacy: OFFLINE_ASR_PROVIDER → ASR_PROVIDER → streaming_provider
    offline_provider = env.get(
        "OFFLINE_ASR_PROVIDER", env.get("ASR_PROVIDER", streaming_provider)
    )

    try:
        num_threads = int(env.get("STREAMING_ASR_NUM_THREADS", "4"))
    except ValueError:
        num_threads = 4

    return SherpaASRConfig(
        language_mode=env.get("LANGUAGE_MODE", "zh_en"),
        streaming_model_dir=env.get("STREAMING_MODEL_DIR") or None,
        streaming_provider=streaming_provider,
        offline_provider=offline_provider,
        num_threads=num_threads,
        model_root=env.get("MODEL_DIR", "/opt/models"),
    )


def build_rk_asr_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``RKASRConfig`` from env.

    Mirrors the env reads in legacy ``app/backends/rk/asr.py``: ``RK_PLATFORM``
    (in ``__init__``) plus the per-call energy-split reads inside
    ``_split_at_silence_energy``. ``long_audio_threshold_s`` was a module
    constant (``_LONG_AUDIO_THRESHOLD_S = 15.0``), so it has no env override.

    env → RKASRConfig field map (legacy):
      RK_PLATFORM                 → platform (legacy default "rk3576")
      ASR_ENERGY_SPLIT_RMS        → energy_split_rms (0.003)
      ASR_ENERGY_MIN_SILENCE_MS   → energy_min_silence_ms (120)
      (constant _LONG_AUDIO_THRESHOLD_S) → long_audio_threshold_s (15.0)
    """
    from voxedge.backends.rk.asr import RKASRConfig

    if env is None:
        env = os.environ

    try:
        energy_split_rms = float(env.get("ASR_ENERGY_SPLIT_RMS", "0.003"))
    except ValueError:
        energy_split_rms = 0.003
    try:
        energy_min_silence_ms = int(env.get("ASR_ENERGY_MIN_SILENCE_MS", "120"))
    except ValueError:
        energy_min_silence_ms = 120

    return RKASRConfig(
        platform=env.get("RK_PLATFORM", "rk3576"),
        energy_split_rms=energy_split_rms,
        energy_min_silence_ms=energy_min_silence_ms,
        long_audio_threshold_s=15.0,
    )


def build_matcha_tts_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``MatchaTRTConfig`` from env.

    Mirrors ``MatchaTRTBackend._resolve_matcha_paths`` + the per-method env
    reads (``MATCHA_ACOUSTIC_EP``, ``MATCHA_STREAM_CHUNK_MS``,
    ``MATCHA_MIN_MEL_FRAMES``, ``OVS_TTS_STREAM_MAX_WORKERS``,
    ``OVS_MATCHA_ARENA_SIZE_MB`` / ``OVS_CUDA_ARENA_SIZE_MB``,
    ``OVS_TTS_MODEL_ID``).

    env / profile → MatchaTRTConfig field map:
      MATCHA_MODEL_BASE             → model_base ("/opt/models/matcha-icefall-zh-en")
      LANGUAGE_MODE                 → language_mode ("zh_en")
      VOCOS_ENGINE                  → vocos_engine (<base>/engines/vocos_fp16.engine)
      ACOUSTIC_ONNX                 → acoustic_onnx (<base>/model-steps-3.onnx)
      MATCHA_SPLIT_ENCODER_ONNX     → split_encoder_onnx (<base>/onnx/matcha_encoder_trt.onnx)
      MATCHA_SPLIT_ESTIMATOR_ENGINE → split_estimator_engine (<base>/engines/matcha_estimator_step0_bf16.engine)
      LEXICON_PATH                  → lexicon_path (<base>/lexicon.txt)
      TOKENS_PATH                   → tokens_path (<base>/tokens.txt)
      MATCHA_MIN_MEL_FRAMES         → min_mel_frames (72)
      MATCHA_ACOUSTIC_EP            → acoustic_ep ("")
      OVS_TTS_STREAM_MAX_WORKERS    → stream_max_workers (profile tts_stream_max_workers / 2)
      OVS_MATCHA_ARENA_SIZE_MB / OVS_CUDA_ARENA_SIZE_MB → arena_size_mb (16)
      MATCHA_STREAM_CHUNK_MS        → stream_chunk_ms (40)
      OVS_TTS_MODEL_ID              → model_id ("matcha_trt")
    """
    from voxedge.backends.jetson.matcha_trt import MatchaTRTConfig

    if env is None:
        env = os.environ

    model_base = env.get("MATCHA_MODEL_BASE", "/opt/models/matcha-icefall-zh-en")

    # -- stream_max_workers: env → profile → 2 (matches matcha
    #    concurrency_capability precedence) --
    sw_env = env.get("OVS_TTS_STREAM_MAX_WORKERS")
    if sw_env is not None:
        try:
            stream_max_workers = int(sw_env)
        except ValueError:
            stream_max_workers = 2
    else:
        profile_sw = _profile_get(profile, "tts_stream_max_workers")
        if profile_sw is None:
            tcfg = _profile_get(profile, "tts_backend_config")
            if isinstance(tcfg, dict):
                profile_sw = tcfg.get("stream_max_workers")
        try:
            stream_max_workers = int(profile_sw) if profile_sw is not None else 2
        except (TypeError, ValueError):
            stream_max_workers = 2

    # -- arena: OVS_MATCHA_ARENA_SIZE_MB → OVS_CUDA_ARENA_SIZE_MB → 16
    #    (matches matcha _read_arena_size_bytes("OVS_MATCHA_ARENA_SIZE_MB")) --
    arena_fallback = env.get("OVS_CUDA_ARENA_SIZE_MB", "16")
    arena_raw = env.get("OVS_MATCHA_ARENA_SIZE_MB", arena_fallback)
    try:
        arena_size_mb = int(arena_raw)
    except ValueError:
        logger.warning("Invalid OVS_MATCHA_ARENA_SIZE_MB=%r; falling back to 16", arena_raw)
        arena_size_mb = 16

    try:
        min_mel_frames = int(env.get("MATCHA_MIN_MEL_FRAMES", "72"))
    except ValueError:
        min_mel_frames = 72

    try:
        stream_chunk_ms = int(env.get("MATCHA_STREAM_CHUNK_MS", "40"))
    except ValueError:
        stream_chunk_ms = 40

    # model_id: legacy TTSBackend.model_id reads OVS_TTS_MODEL_ID, falling back
    # to the backend ``name`` ("matcha_trt").
    model_id = env.get("OVS_TTS_MODEL_ID") or "matcha_trt"

    return MatchaTRTConfig(
        model_base=model_base,
        language_mode=env.get("LANGUAGE_MODE", "zh_en"),
        vocos_engine=env.get("VOCOS_ENGINE") or None,
        acoustic_onnx=env.get("ACOUSTIC_ONNX") or None,
        split_encoder_onnx=env.get("MATCHA_SPLIT_ENCODER_ONNX") or None,
        split_estimator_engine=env.get("MATCHA_SPLIT_ESTIMATOR_ENGINE") or None,
        lexicon_path=env.get("LEXICON_PATH") or None,
        tokens_path=env.get("TOKENS_PATH") or None,
        min_mel_frames=min_mel_frames,
        acoustic_ep=env.get("MATCHA_ACOUSTIC_EP", ""),
        stream_max_workers=stream_max_workers,
        arena_size_mb=arena_size_mb,
        stream_chunk_ms=stream_chunk_ms,
        model_id=model_id,
    )


def build_kokoro_trt_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``KokoroTRTConfig`` from env.

    Mirrors the module-scope + per-method env reads in legacy
    ``app/backends/jetson/kokoro_trt.py``. Path fields are left ``None`` so the
    dataclass ``__post_init__`` rebuilds them from ``model_base`` exactly like
    the legacy ``os.path.join`` defaults; only env *overrides* are passed.

    env → KokoroTRTConfig field map (see voxedge kokoro_trt header):
      KOKORO_MODEL_BASE                → model_base ("/opt/models/kokoro-multi-lang-v1_0")
      KOKORO_ONNX                      → model_onnx (<base>/model.onnx)
      KOKORO_TRT_ENGINE                → engine_path (<base>/engines/kokoro_fp16.engine)
      KOKORO_HYBRID_DIR                → hybrid_dir (<base>/hybrid)
      KOKORO_HYBRID_PREFIX_ENGINE      → hybrid_prefix_engine_env (None)
      KOKORO_HYBRID_SUFFIX_ONNX        → hybrid_suffix_onnx (<hybrid>/...)
      KOKORO_SPLIT_ENCODER_ENGINE …    → split_*_engine / split_*_onnx
      KOKORO_VOICES                    → voices_bin (<base>/voices.bin)
      KOKORO_TOKENS                    → tokens_path (<base>/tokens.txt)
      KOKORO_MAX_TOKENS                → max_tokens (510)
      KOKORO_DEFAULT_SID/TTS_DEFAULT_SID → default_speaker_id (52)
      TTS_DEFAULT_SPEED                → default_speed (1.0)
      KOKORO_STREAM_MAX_SEGMENT_TOKENS → stream_segment_tokens (64)
      KOKORO_STREAM_SEGMENT_TEXT       → stream_segment_text (True)
      KOKORO_SYNTH_SEGMENT_TEXT        → synth_segment_text (True)
      KOKORO_SYNTH_MAX_SEGMENT_TOKENS  → synth_max_segment_tokens (= stream_segment_tokens)
      KOKORO_TRT_RUNTIME               → runtime_mode ("auto")
      OVS_TTS_STREAM_MAX_WORKERS       → stream_max_workers (profile tts_stream_max_workers / 2)
      OVS_KOKORO_ARENA_SIZE_MB/OVS_CUDA_ARENA_SIZE_MB → arena_size_mb (16)
      KOKORO_STREAM_CHUNK_MS           → stream_chunk_ms (40)
      KOKORO_SPLIT_CPU_FALLBACK        → split_cpu_fallback (True)
      KOKORO_SPLIT_MAX_SEQ_LEN/KOKORO_HYBRID_MAX_SEQ_LEN → max_seq_len_fallback (128)
      KOKORO_HYBRID_TOKEN_LEN          → hybrid_token_len (0)
      OVS_TTS_MODEL_ID                 → model_id ("kokoro_trt")
    """
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTConfig

    if env is None:
        env = os.environ

    def _int(name: str, default: int) -> int:
        try:
            return int(env.get(name, str(default)))
        except ValueError:
            return default

    def _bool(name: str, default: bool) -> bool:
        v = env.get(name)
        if v is None:
            return default
        return v.lower() not in ("0", "false", "no")

    # stream_max_workers: env → profile → 2 (same precedence as matcha)
    sw_env = env.get("OVS_TTS_STREAM_MAX_WORKERS")
    if sw_env is not None:
        try:
            stream_max_workers = int(sw_env)
        except ValueError:
            stream_max_workers = 2
    else:
        profile_sw = _profile_get(profile, "tts_stream_max_workers")
        if profile_sw is None:
            tcfg = _profile_get(profile, "tts_backend_config")
            if isinstance(tcfg, dict):
                profile_sw = tcfg.get("stream_max_workers")
        try:
            stream_max_workers = int(profile_sw) if profile_sw is not None else 2
        except (TypeError, ValueError):
            stream_max_workers = 2

    # arena: OVS_KOKORO_ARENA_SIZE_MB → OVS_CUDA_ARENA_SIZE_MB → 16
    arena_fallback = env.get("OVS_CUDA_ARENA_SIZE_MB", "16")
    arena_raw = env.get("OVS_KOKORO_ARENA_SIZE_MB", arena_fallback)
    try:
        arena_size_mb = int(arena_raw)
    except ValueError:
        arena_size_mb = 16

    # default_speaker_id: KOKORO_DEFAULT_SID → TTS_DEFAULT_SID → 52
    default_sid = _int("KOKORO_DEFAULT_SID", _int("TTS_DEFAULT_SID", 52))

    stream_segment_tokens = _int("KOKORO_STREAM_MAX_SEGMENT_TOKENS", 64)
    synth_max_segment_tokens = _int("KOKORO_SYNTH_MAX_SEGMENT_TOKENS", stream_segment_tokens)

    # max_seq_len_fallback: legacy split path reads KOKORO_SPLIT_MAX_SEQ_LEN,
    # hybrid path reads KOKORO_HYBRID_MAX_SEQ_LEN; both default 128. Honour the
    # split var first (the production runtime mode), falling back to the hybrid.
    max_seq_len_fallback = _int(
        "KOKORO_SPLIT_MAX_SEQ_LEN", _int("KOKORO_HYBRID_MAX_SEQ_LEN", 128)
    )

    return KokoroTRTConfig(
        model_base=env.get("KOKORO_MODEL_BASE", "/opt/models/kokoro-multi-lang-v1_0"),
        model_onnx=env.get("KOKORO_ONNX") or None,
        engine_path=env.get("KOKORO_TRT_ENGINE") or None,
        hybrid_dir=env.get("KOKORO_HYBRID_DIR") or None,
        hybrid_prefix_engine_env=env.get("KOKORO_HYBRID_PREFIX_ENGINE") or None,
        hybrid_suffix_onnx=env.get("KOKORO_HYBRID_SUFFIX_ONNX") or None,
        split_encoder_engine=env.get("KOKORO_SPLIT_ENCODER_ENGINE") or None,
        split_length_onnx=env.get("KOKORO_SPLIT_LENGTH_ONNX") or None,
        split_decoder_engine=env.get("KOKORO_SPLIT_DECODER_ENGINE") or None,
        split_decoder_engine_long=env.get("KOKORO_SPLIT_DECODER_ENGINE_LONG") or None,
        split_source_engine=env.get("KOKORO_SPLIT_SOURCE_ENGINE") or None,
        split_source_engine_long=env.get("KOKORO_SPLIT_SOURCE_ENGINE_LONG") or None,
        split_source_onnx=env.get("KOKORO_SPLIT_SOURCE_ONNX") or None,
        split_generator_engine=env.get("KOKORO_SPLIT_GENERATOR_ENGINE") or None,
        split_generator_engine_long=env.get("KOKORO_SPLIT_GENERATOR_ENGINE_LONG") or None,
        split_istft_onnx=env.get("KOKORO_SPLIT_ISTFT_ONNX") or None,
        voices_bin=env.get("KOKORO_VOICES") or None,
        tokens_path=env.get("KOKORO_TOKENS") or None,
        max_tokens=_int("KOKORO_MAX_TOKENS", 510),
        default_speaker_id=default_sid,
        default_speed=float(env.get("TTS_DEFAULT_SPEED", "1.0")),
        stream_segment_tokens=stream_segment_tokens,
        stream_segment_text=_bool("KOKORO_STREAM_SEGMENT_TEXT", True),
        synth_segment_text=_bool("KOKORO_SYNTH_SEGMENT_TEXT", True),
        synth_max_segment_tokens=synth_max_segment_tokens,
        runtime_mode=env.get("KOKORO_TRT_RUNTIME", "auto"),
        stream_max_workers=stream_max_workers,
        arena_size_mb=arena_size_mb,
        stream_chunk_ms=_int("KOKORO_STREAM_CHUNK_MS", 40),
        split_cpu_fallback=_bool("KOKORO_SPLIT_CPU_FALLBACK", True),
        max_seq_len_fallback=max_seq_len_fallback,
        hybrid_token_len=_int("KOKORO_HYBRID_TOKEN_LEN", 0),
        model_id=env.get("OVS_TTS_MODEL_ID") or "kokoro_trt",
    )


def build_qwen3_trt_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``Qwen3TRTConfig`` from env.

    Mirrors the module-scope + per-method env reads in legacy
    ``app/backends/jetson/qwen3_trt.py`` (``_resolve_paths`` + the customvoice
    detection + the synth-time engine env reads). Path fields are left ``None``
    so ``__post_init__`` rebuilds them from ``model_base``.

    env → Qwen3TRTConfig field map (see voxedge qwen3_trt header):
      QWEN3_MODEL_BASE                     → model_base ("/opt/models/qwen3-tts")
      QWEN3_SHERPA_DIR                     → sherpa_dir (<base>/onnx)
      QWEN3_MODEL_DIR                      → model_dir (<base>/onnx)
      QWEN3_TALKER_ENGINE                  → talker_engine (<base>/engines/talker_decode_bf16.engine)
      QWEN3_CP_ENGINE                      → cp_engine (<base>/engines/cp_bf16.engine)
      QWEN3_SPEAKER_ENCODER                → speaker_encoder (<base>/onnx/speaker_encoder.onnx)
      QWEN3_TOKENIZER_DIR                  → tokenizer_dir (<base>/tokenizer)
      QWEN3_EXTRACT_SCRIPT                 → extract_script (<base>/extract_speaker_emb.py)
      QWEN3_TTS_VARIANT/OVS_TTS_MODEL_ID   → is_customvoice (→ supports_voice_cloning auto)
      OVS_TTS_MODEL_ID                     → model_id ("qwen3-tts"; customvoice→"qwen3-tts-customvoice")
      TTS_INT8_EOS_LOGIT_OFFSET            → int8_eos_logit_offset (-10.0)
      TTS_TALKER_CUDA_GRAPH                → talker_cuda_graph (True)
      TTS_TRT_VOCODER_MAX_FRAMES           → vocoder_max_frames (100)
      TTS_VOCODER_TRT                      → use_trt_vocoder (True)
      QWEN3_TTS_OFFLINE_STREAMING_FOR_LONG → offline_streaming_for_long (True)
      QWEN3_TTS_NUMPY_SAMPLING             → numpy_sampling (True)
      OVS_TTS_SEED                         → default_seed (0)
      QWEN3_TTS_PRODUCT_SEGMENT_TEXT       → product_segment_text (False)
      QWEN3_TTS_PRODUCT_SEGMENT_MAX_CHARS  → product_segment_max_chars (20)
      QWEN3_TTS_PRODUCT_COMMA_PAUSE_MS     → product_comma_pause_ms (120)
      QWEN3_TTS_PRODUCT_HARD_PAUSE_MS      → product_hard_pause_ms (180)
    """
    from voxedge.backends.jetson.qwen3_trt import Qwen3TRTConfig

    if env is None:
        env = os.environ

    def _int(name: str, default: int) -> int:
        try:
            return int(env.get(name, str(default)))
        except ValueError:
            return default

    def _bool(name: str, default: bool) -> bool:
        v = env.get(name)
        if v is None:
            return default
        return v.lower() not in ("0", "false", "no")

    # CustomVoice detection mirrors legacy ``_is_customvoice_variant``:
    #   QWEN3_TTS_VARIANT=customvoice, OR "customvoice" in OVS_TTS_MODEL_ID.
    variant = env.get("QWEN3_TTS_VARIANT", "").strip().lower()
    base_model_id = (env.get("OVS_TTS_MODEL_ID") or "").strip().lower()
    is_customvoice = (variant == "customvoice") or ("customvoice" in base_model_id)

    # model_id: legacy pins to "qwen3-tts-customvoice" when the customvoice
    # variant is detected via env but OVS_TTS_MODEL_ID wasn't the customvoice key.
    model_id = env.get("OVS_TTS_MODEL_ID") or "qwen3-tts"
    if is_customvoice and "customvoice" not in base_model_id:
        model_id = "qwen3-tts-customvoice"

    return Qwen3TRTConfig(
        model_base=env.get("QWEN3_MODEL_BASE", "/opt/models/qwen3-tts"),
        sherpa_dir=env.get("QWEN3_SHERPA_DIR") or None,
        model_dir=env.get("QWEN3_MODEL_DIR") or None,
        talker_engine=env.get("QWEN3_TALKER_ENGINE") or None,
        cp_engine=env.get("QWEN3_CP_ENGINE") or None,
        speaker_encoder=env.get("QWEN3_SPEAKER_ENCODER") or None,
        tokenizer_dir=env.get("QWEN3_TOKENIZER_DIR") or None,
        extract_script=env.get("QWEN3_EXTRACT_SCRIPT") or None,
        supports_voice_cloning=None,  # auto from is_customvoice in __post_init__
        is_customvoice=is_customvoice,
        model_id=model_id,
        int8_eos_logit_offset=float(env.get("TTS_INT8_EOS_LOGIT_OFFSET", "-10.0")),
        talker_cuda_graph=(env.get("TTS_TALKER_CUDA_GRAPH", "1") == "1"),
        vocoder_max_frames=_int("TTS_TRT_VOCODER_MAX_FRAMES", 100),
        use_trt_vocoder=_bool("TTS_VOCODER_TRT", True),
        offline_streaming_for_long=_bool("QWEN3_TTS_OFFLINE_STREAMING_FOR_LONG", True),
        numpy_sampling=_bool("QWEN3_TTS_NUMPY_SAMPLING", True),
        default_seed=_int("OVS_TTS_SEED", 0),
        product_segment_text=_bool("QWEN3_TTS_PRODUCT_SEGMENT_TEXT", False),
        product_segment_max_chars=_int("QWEN3_TTS_PRODUCT_SEGMENT_MAX_CHARS", 20),
        product_comma_pause_ms=_int("QWEN3_TTS_PRODUCT_COMMA_PAUSE_MS", 120),
        product_hard_pause_ms=_int("QWEN3_TTS_PRODUCT_HARD_PAUSE_MS", 180),
    )


def build_trt_edge_llm_tts_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``TRTEdgeLLMTTSConfig`` from env + profile.

    Mirrors the legacy ``app/backends/jetson/trt_edge_llm_tts.py`` module-scope
    + ``__init__`` env reads field-for-field. Artifact path fields (talker /
    code_predictor / tokenizer / code2wav / worker_binary / plugin) are
    resolved via the legacy ``trt_edge_llm_ipc`` fresh-read resolvers so the
    empty-string voxedge defaults are replaced by the real production
    artifact-tree paths (identical precedence: explicit env → vocab-pruned /
    highperf probe → ``~/qwen3-tts-*`` default tree). ``tts_binary`` /
    ``speaker_encoder`` mirror the module constants / ``_resolve_speaker_encoder``.

    ``worker_concurrency`` (env ``OVS_TTS_WORKER_CONCURRENCY`` → profile
    ``tts_worker_concurrency`` / ``tts_backend_config.worker_concurrency`` → 1)
    gates the worker's ``--max_slots`` arg when N>1 (preserves b1cb1a5
    semantics inside the voxedge backend).

    env / profile → TRTEdgeLLMTTSConfig field map (legacy module + __init__):
      EDGE_LLM_TTS_BIN                                  → tts_binary (TTS_BINARY)
      (resolve_tts_worker_binary)                       → worker_binary
      EDGELLM_PLUGIN_PATH                              → plugin_path (PLUGIN_PATH)
      (resolve_tts_talker_dir)                          → talker_dir
      (resolve_tts_code_predictor_dir)                  → code_predictor_dir
      (resolve_tts_tokenizer_dir)                       → tokenizer_dir
      (resolve_tts_code2wav_dir)                        → code2wav_dir
      QWEN3_SPEAKER_ENCODER/QWEN3_ARTIFACT_ROOT/...     → speaker_encoder
      OVS_TTS_MODEL_ID                                  → model_id ("trt_edgellm")
      OVS_TTS_BACKEND/EDGE_LLM_TTS_BACKEND             → backend_mode ("edgellm_worker")
      EDGE_LLM_TTS_WORKER                              → use_worker (True)
      OVS_TTS_WORKER_CONCURRENCY / profile tts_worker_concurrency → worker_concurrency (1)
      EDGE_LLM_QWEN3_PROFILE/OVS_QWEN3_PROFILE        → qwen3_runtime_profile ("highperf")
      EDGE_LLM_TTS_PERF_PROFILE                        → perf_profile ("quality")
      EDGE_LLM_TTS_STATEFUL_CODE2WAV                   → stateful_code2wav (None→profile-derived)
      OVS_TTS_SEED                                     → seed (42)
      OVS_TTS_TALKER_TEMPERATURE/TTS_TALKER_TEMPERATURE → talker_temperature (0.9)
      OVS_TTS_TALKER_TOP_K/TTS_TALKER_TOP_K            → talker_top_k (50)
      OVS_TTS_TOP_P/TTS_TOP_P                          → talker_top_p (1.0)
      OVS_TTS_PREDICTOR_TEMPERATURE/TTS_PREDICTOR_TEMPERATURE → predictor_temperature (0.9)
      OVS_TTS_PREDICTOR_TOP_K/TTS_PREDICTOR_TOP_K      → predictor_top_k (50)
      OVS_TTS_PREDICTOR_TOP_P/TTS_PREDICTOR_TOP_P      → predictor_top_p (1.0)
      TTS_MAX_AUDIO_LENGTH                              → max_audio_length (1024)
      TTS_MIN_AUDIO_LENGTH                              → min_audio_length (30)
      TTS_REPETITION_PENALTY                            → repetition_penalty (1.05)
      TTS_CODEC_EOS_LOGIT_OFFSET                        → codec_eos_logit_offset (0.0)
      EDGE_LLM_TTS_SEGMENT_TEXT                         → segment_text (True)
      EDGE_LLM_TTS_SEGMENT_MAX_CHARS                    → segment_max_chars_latin (120)
      EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS               → segment_max_chars_cjk (48)
      EDGE_LLM_TTS_SEGMENT_PAUSE_MS                     → segment_pause_ms (80)
      EDGE_LLM_TTS_HARD_SEGMENT_PAUSE_MS               → segment_hard_pause_ms (120)
      EDGE_LLM_TTS_STREAMING_PROFILE                   → streaming_profile ("continuous_playback")
      OVS_TTS_MODEL_BASE                               → product_model_base (legacy /home/harvest/voice_test/...)
      OVS_TTS_NATIVE_MODULE_DIR                        → product_overlay_dir (legacy /home/harvest/voice_test/app_overlay)

    NB: the production model_id resolution is ``OVS_TTS_MODEL_ID`` → backend
    name "trt_edgellm" (legacy ``TTSBackend.model_id`` fallback to ``self.name``).
    """
    from voxedge.backends.jetson.trt_edge_llm_tts import TRTEdgeLLMTTSConfig
    from app.backends.jetson.trt_edge_llm_ipc import (
        TTS_BINARY,
        PLUGIN_PATH,
        resolve_tts_talker_dir,
        resolve_tts_code_predictor_dir,
        resolve_tts_tokenizer_dir,
        resolve_tts_code2wav_dir,
        resolve_tts_worker_binary,
        qwen3_runtime_profile,
    )

    if env is None:
        env = os.environ

    def _first(*names: str, default: str = "") -> str:
        """First non-empty env value among ``names`` (legacy ``_env``)."""
        for name in names:
            v = env.get(name)
            if v not in (None, ""):
                return v
        return default

    def _fl(default: float, *names: str) -> float:
        try:
            return float(_first(*names, default=str(default)))
        except (TypeError, ValueError):
            return default

    def _in(default: int, *names: str) -> int:
        try:
            return int(_first(*names, default=str(default)))
        except (TypeError, ValueError):
            return default

    def _flag(name: str, default: bool) -> bool:
        v = env.get(name)
        if v is None:
            return default
        return v.lower() not in ("0", "false", "no", "off")

    # -- speaker encoder: QWEN3_SPEAKER_ENCODER → QWEN3_ARTIFACT_ROOT probe →
    #    <model_base>/onnx/speaker_encoder.onnx  (legacy _resolve_speaker_encoder)
    qwen3_tts_model_base = _first(
        "OVS_TTS_MODEL_BASE", "QWEN3_MODEL_BASE", default="/opt/models/qwen3-tts"
    )
    speaker_encoder = env.get("QWEN3_SPEAKER_ENCODER", "") or ""
    if not speaker_encoder:
        qwen3_root = env.get("QWEN3_ARTIFACT_ROOT", "")
        candidate = ""
        if qwen3_root:
            candidate = os.path.join(
                qwen3_root, "tts", "speaker_encoder", "speaker_encoder.onnx"
            )
            if not os.path.exists(candidate):
                candidate = ""
        speaker_encoder = candidate or os.path.join(
            qwen3_tts_model_base, "onnx", "speaker_encoder.onnx"
        )

    # -- worker_concurrency: env → profile (top-level or nested) → 1.
    #    N>1 gates --max_slots (b1cb1a5). Mirrors legacy
    #    concurrency_capability precedence.
    env_conc = env.get("OVS_TTS_WORKER_CONCURRENCY")
    if env_conc is not None:
        try:
            worker_concurrency = int(env_conc)
        except ValueError:
            worker_concurrency = 1
    else:
        profile_conc = _profile_get(profile, "tts_worker_concurrency")
        if profile_conc is None:
            tcfg = _profile_get(profile, "tts_backend_config")
            if isinstance(tcfg, dict):
                profile_conc = tcfg.get("worker_concurrency")
        try:
            worker_concurrency = int(profile_conc) if profile_conc is not None else 1
        except (TypeError, ValueError):
            worker_concurrency = 1
    worker_concurrency = max(1, worker_concurrency)

    # -- stateful_code2wav: legacy default is the env flag
    #    EDGE_LLM_TTS_STATEFUL_CODE2WAV defaulting to qwen3_highperf_enabled().
    #    Leave None when unset so the dataclass derives it from the runtime
    #    profile (byte-equivalent expression); pass the explicit bool otherwise.
    stateful_raw = env.get("EDGE_LLM_TTS_STATEFUL_CODE2WAV")
    if stateful_raw is None:
        stateful_code2wav = None
    else:
        stateful_code2wav = stateful_raw.lower() not in ("0", "false", "no", "off")

    # model_id: OVS_TTS_MODEL_ID → backend name "trt_edgellm" (legacy fallback).
    model_id = env.get("OVS_TTS_MODEL_ID") or "trt_edgellm"

    return TRTEdgeLLMTTSConfig(
        tts_binary=env.get("EDGE_LLM_TTS_BIN") or TTS_BINARY,
        worker_binary=resolve_tts_worker_binary(),
        plugin_path=env.get("EDGELLM_PLUGIN_PATH") or PLUGIN_PATH,
        talker_dir=resolve_tts_talker_dir(),
        code_predictor_dir=resolve_tts_code_predictor_dir(),
        tokenizer_dir=resolve_tts_tokenizer_dir(),
        code2wav_dir=resolve_tts_code2wav_dir(),
        speaker_encoder=speaker_encoder,
        model_id=model_id,
        backend_mode=_first(
            "OVS_TTS_BACKEND", "EDGE_LLM_TTS_BACKEND", default="edgellm_worker"
        ),
        use_worker=_flag("EDGE_LLM_TTS_WORKER", True),
        worker_concurrency=worker_concurrency,
        qwen3_runtime_profile=qwen3_runtime_profile(),
        perf_profile=env.get("EDGE_LLM_TTS_PERF_PROFILE", "quality"),
        stateful_code2wav=stateful_code2wav,
        seed=_in(42, "OVS_TTS_SEED"),
        talker_temperature=_fl(0.9, "OVS_TTS_TALKER_TEMPERATURE", "TTS_TALKER_TEMPERATURE"),
        talker_top_k=_in(50, "OVS_TTS_TALKER_TOP_K", "TTS_TALKER_TOP_K"),
        talker_top_p=_fl(1.0, "OVS_TTS_TOP_P", "TTS_TOP_P"),
        predictor_temperature=_fl(
            0.9, "OVS_TTS_PREDICTOR_TEMPERATURE", "TTS_PREDICTOR_TEMPERATURE"
        ),
        predictor_top_k=_in(50, "OVS_TTS_PREDICTOR_TOP_K", "TTS_PREDICTOR_TOP_K"),
        predictor_top_p=_fl(1.0, "OVS_TTS_PREDICTOR_TOP_P", "TTS_PREDICTOR_TOP_P"),
        max_audio_length=_in(1024, "TTS_MAX_AUDIO_LENGTH"),
        min_audio_length=_in(30, "TTS_MIN_AUDIO_LENGTH"),
        repetition_penalty=_fl(1.05, "TTS_REPETITION_PENALTY"),
        codec_eos_logit_offset=_fl(0.0, "TTS_CODEC_EOS_LOGIT_OFFSET"),
        segment_text=_flag("EDGE_LLM_TTS_SEGMENT_TEXT", True),
        segment_max_chars_latin=_in(120, "EDGE_LLM_TTS_SEGMENT_MAX_CHARS"),
        segment_max_chars_cjk=_in(48, "EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS"),
        segment_pause_ms=_in(80, "EDGE_LLM_TTS_SEGMENT_PAUSE_MS"),
        segment_hard_pause_ms=_in(120, "EDGE_LLM_TTS_HARD_SEGMENT_PAUSE_MS"),
        streaming_profile=env.get(
            "EDGE_LLM_TTS_STREAMING_PROFILE", "continuous_playback"
        ),
        product_model_base=_first(
            "OVS_TTS_MODEL_BASE",
            default="/home/harvest/voice_test/models/qwen3-tts",
        ),
        product_overlay_dir=_first(
            "OVS_TTS_NATIVE_MODULE_DIR",
            default="/home/harvest/voice_test/app_overlay",
        ),
    )


def build_moss_tts_nano_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``MossTtsNanoConfig`` from env + profile.

    Mirrors the env reads (``MOSS_*``) + profile reads (slot/seq/audio shape)
    in legacy ``app/backends/jetson/moss_tts_nano.py``. ``tokenizer_model`` is
    left ``None`` so ``__post_init__`` derives ``<engine_dir>/tokenizer.model``.

    env/profile → MossTtsNanoConfig field map (see voxedge moss header):
      MOSS_WORKER_BIN     → worker_bin ("/opt/jv-workers/moss_tts_nano_worker")
      MOSS_ENGINE_DIR     → engine_dir ("/opt/models/moss-tts-nano/engines")
      MOSS_TOKENIZER      → tokenizer_model (<engine_dir>/tokenizer.model)
      MOSS_CODEC_ONNX_DIR → codec_onnx_dir ("/opt/models/moss-tts-nano/codec_onnx")
      profile moss_max_slots                    → max_slots (1)
      profile moss_max_seq_len                  → max_seq_len (2048)
      profile moss_sample_rate/tts_sample_rate  → sample_rate (48000)
      profile moss_channels/tts_channels        → channels (2)
      MOSS_PY_REPO        → py_repo ("/opt/moss-tts-nano-py")   [.py worker only]
      MOSS_ORT_EP         → ort_ep ("cpu")                      [.py worker only]
      MOSS_ORT_THREADS    → ort_threads (4)                     [.py worker only]
    """
    from voxedge.backends.jetson.moss_tts_nano import MossTtsNanoConfig

    if env is None:
        env = os.environ
    p = profile if isinstance(profile, dict) else {}

    def _pint(*keys, default):
        for k in keys:
            v = p.get(k)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return default

    return MossTtsNanoConfig(
        worker_bin=env.get("MOSS_WORKER_BIN", "/opt/jv-workers/moss_tts_nano_worker"),
        engine_dir=env.get("MOSS_ENGINE_DIR", "/opt/models/moss-tts-nano/engines"),
        tokenizer_model=env.get("MOSS_TOKENIZER") or None,
        codec_onnx_dir=env.get("MOSS_CODEC_ONNX_DIR", "/opt/models/moss-tts-nano/codec_onnx"),
        max_slots=_pint("moss_max_slots", default=1),
        max_seq_len=_pint("moss_max_seq_len", default=2048),
        sample_rate=_pint("moss_sample_rate", "tts_sample_rate", default=48000),
        channels=_pint("moss_channels", "tts_channels", default=2),
        py_repo=env.get("MOSS_PY_REPO", "/opt/moss-tts-nano-py"),
        ort_ep=env.get("MOSS_ORT_EP", "cpu"),
        ort_threads=int(env.get("MOSS_ORT_THREADS", "4"))
        if env.get("MOSS_ORT_THREADS", "4").isdigit()
        else 4,
    )


def build_sherpa_tts_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``SherpaTTSConfig`` from env.

    Mirrors the module-scope reads in legacy ``app/backends/cpu/sherpa.py``.
    ``model_dir`` / ``default_speaker_id`` are left ``None`` so the dataclass
    ``__post_init__`` reproduces the language-conditional defaults exactly.

    env → SherpaTTSConfig field map (see voxedge sherpa/tts header):
      LANGUAGE_MODE                      → language_mode ("zh_en")
      SHERPA_TTS_MODEL_DIR/TTS_MODEL_DIR → model_dir (None → per language_mode)
      TTS_PROVIDER                       → provider ("cuda")
      TTS_NUM_THREADS                    → num_threads (4)
      TTS_DEFAULT_SID                    → default_speaker_id (None → per language_mode)
      TTS_DEFAULT_SPEED                  → default_speed (1.0)
      TTS_PITCH_SHIFT                    → pitch_shift (0.0)
      OVS_TTS_MODEL_ID                   → model_id ("sherpa")
    """
    from voxedge.backends.sherpa.tts import SherpaTTSConfig

    if env is None:
        env = os.environ

    # model_dir: SHERPA_TTS_MODEL_DIR → TTS_MODEL_DIR → None(→ per language_mode)
    model_dir = env.get("SHERPA_TTS_MODEL_DIR") or env.get("TTS_MODEL_DIR") or None

    sid_env = env.get("TTS_DEFAULT_SID")
    default_speaker_id = None
    if sid_env is not None:
        try:
            default_speaker_id = int(sid_env)
        except ValueError:
            default_speaker_id = None

    try:
        num_threads = int(env.get("TTS_NUM_THREADS", "4"))
    except ValueError:
        num_threads = 4

    return SherpaTTSConfig(
        language_mode=env.get("LANGUAGE_MODE", "zh_en"),
        model_dir=model_dir,
        provider=env.get("TTS_PROVIDER", "cuda"),
        num_threads=num_threads,
        default_speaker_id=default_speaker_id,
        default_speed=float(env.get("TTS_DEFAULT_SPEED", "1.0")),
        pitch_shift=float(env.get("TTS_PITCH_SHIFT", "0")),
        model_id=env.get("OVS_TTS_MODEL_ID") or "sherpa",
    )


def build_rk_tts_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``RKTTSConfig`` from env.

    The RK adapter delegates backend selection to rkvoice-stream via the
    ``TTS_BACKEND`` env (read inside rkvoice-stream, not here). The only
    product-layer config is ``model_id`` (legacy ``OVS_TTS_MODEL_ID`` →
    backend-name fallback "rk").

    env → RKTTSConfig field map (see voxedge rk/tts header):
      OVS_TTS_MODEL_ID → model_id ("rk")
    """
    from voxedge.backends.rk.tts import RKTTSConfig

    if env is None:
        env = os.environ

    return RKTTSConfig(model_id=env.get("OVS_TTS_MODEL_ID") or "rk")
