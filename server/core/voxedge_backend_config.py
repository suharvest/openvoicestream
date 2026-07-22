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

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool, env: Optional[dict] = None) -> bool:
    """Match legacy ``trt_edge_llm_asr._env_bool``.

    Reads from ``env`` when supplied (defaults to ``os.environ``) so callers
    passing an explicit env mapping get consistent behaviour with the other
    ``env.get(...)`` reads in each builder.
    """
    source = os.environ if env is None else env
    value = source.get(name)
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

    Delegates to the canonical voxedge factory function
    ``voxedge.backends.jetson.trt_edge_llm_asr.build_config_from_env``.
    The ``profile`` argument (asr_max_slots precedence) is handled here for
    backward compatibility: if env does not set EDGE_LLM_ASR_MAX_CONCURRENT,
    the profile slot value is injected so the factory picks it up.
    """
    from voxedge.backends.jetson.trt_edge_llm_asr import build_config_from_env

    if env is None:
        env = os.environ

    # profile asr_max_slots injection: env → manifest → profile → 1.
    # The voxedge factory reads EDGE_LLM_ASR_MAX_CONCURRENT; inject profile
    # value as a synthetic env override when the env var is absent.
    if "EDGE_LLM_ASR_MAX_CONCURRENT" not in env:
        profile_slots = _profile_get(profile, "asr_max_slots")
        if profile_slots is None:
            asr_cfg = _profile_get(profile, "asr")
            if isinstance(asr_cfg, dict):
                profile_slots = asr_cfg.get("asr_max_slots", asr_cfg.get("max_concurrent"))
        if profile_slots is not None:
            env = dict(env)
            env["EDGE_LLM_ASR_MAX_CONCURRENT"] = str(profile_slots)

    return build_config_from_env(env=env)


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
      PARAFORMER_MAX_CONCURRENT → max_concurrent (env → profile asr_max_slots → 2,
                                  clamped >=1; bounds per-stream TRT context fan-out)
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

    # -- max_concurrent ceiling: env → profile asr_max_slots → 2. Bounded
    #    (default 2) so a burst of streams can't OOM the device; tune per VRAM.
    mc_env = env.get("PARAFORMER_MAX_CONCURRENT")
    if mc_env is not None:
        try:
            max_concurrent = int(mc_env)
        except ValueError:
            max_concurrent = 2
    else:
        profile_slots = _profile_get(profile, "asr_max_slots")
        if profile_slots is None:
            asr_cfg = _profile_get(profile, "asr")
            if isinstance(asr_cfg, dict):
                profile_slots = asr_cfg.get("asr_max_slots", asr_cfg.get("max_concurrent"))
        try:
            max_concurrent = int(profile_slots) if profile_slots is not None else 2
        except (TypeError, ValueError):
            max_concurrent = 2
    max_concurrent = max(1, max_concurrent)

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
        max_concurrent=max_concurrent,
    )


def build_sensevoice_trt_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``SenseVoiceTRTConfig`` from env.

    env → SenseVoiceTRTConfig field map:
      SENSEVOICE_TRT_MODEL_DIR → model_dir ("/opt/models/sensevoice-trt")
      SENSEVOICE_TRT_ENGINE    → engine    (<model_dir>/sensevoice.plan)
      SENSEVOICE_TRT_BPE       → bpe_model (<model_dir>/chn_jpn_yue_eng_ko_spectok.bpe.model)
    """
    from voxedge.backends.jetson.sensevoice_trt import SenseVoiceTRTConfig

    if env is None:
        env = os.environ
    model_dir = env.get("SENSEVOICE_TRT_MODEL_DIR", "/opt/models/sensevoice-trt")
    return SenseVoiceTRTConfig(
        engine=env.get("SENSEVOICE_TRT_ENGINE") or os.path.join(model_dir, "sensevoice.plan"),
        model_dir=model_dir,
        bpe_model=env.get("SENSEVOICE_TRT_BPE") or None,
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


def build_trt_edge_llm_tts_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``TRTEdgeLLMTTSConfig`` from env + profile.

    Delegates to the canonical voxedge factory function
    ``voxedge.backends.jetson.trt_edge_llm_tts.build_config_from_env``.
    The ``profile`` argument (tts_worker_concurrency precedence) is handled
    here for backward compatibility: if env does not set
    OVS_TTS_WORKER_CONCURRENCY, the profile value is injected so the factory
    picks it up.
    """
    from voxedge.backends.jetson.trt_edge_llm_tts import build_config_from_env

    if env is None:
        env = os.environ

    # profile worker_concurrency injection: env → profile (top-level or nested) → 1.
    if "OVS_TTS_WORKER_CONCURRENCY" not in env:
        profile_conc = _profile_get(profile, "tts_worker_concurrency")
        if profile_conc is None:
            tcfg = _profile_get(profile, "tts_backend_config")
            if isinstance(tcfg, dict):
                profile_conc = tcfg.get("worker_concurrency")
        if profile_conc is not None:
            env = dict(env)
            env["OVS_TTS_WORKER_CONCURRENCY"] = str(profile_conc)

    return build_config_from_env(env=env)


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
        model_id=env.get("OVS_TTS_MODEL_ID") or "moss-tts-nano",
        py_repo=env.get("MOSS_PY_REPO", "/opt/moss-tts-nano-py"),
        ort_ep=env.get("MOSS_ORT_EP", "cpu"),
        ort_threads=int(env.get("MOSS_ORT_THREADS", "4"))
        if env.get("MOSS_ORT_THREADS", "4").isdigit()
        else 4,
    )


def build_sparktts_trt_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``SparkTTSConfig`` from env + profile (controllable SparkTTS worker).

    env → SparkTTSConfig field map (see voxedge sparktts_trt header):
      SPARKTTS_WORKER_BINARY          → worker_binary
      SPARKTTS_PLUGIN_PATH            → plugin_path (edge-llm plugin .so, LD_PRELOAD'd)
      SPARKTTS_LLM_ENGINE_DIR         → llm_engine_dir (mixed-precision LLM engine dir)
      SPARKTTS_TOKENIZER_DIR          → tokenizer_dir (None → llm_engine_dir)
      SPARKTTS_BICODEC_ENGINE         → bicodec_engine (.engine file)
      SPARKTTS_SPEAKER_DECODER_ENGINE → speaker_decoder_engine (.engine file)
      SPARKTTS_LD_LIBRARY_PATH        → ld_library_path (edge-llm build dir)
      SPARKTTS_SAMPLE_RATE            → sample_rate (16000)
      SPARKTTS_FIRST_CHUNK_TOKENS     → first_chunk_tokens (6)
      SPARKTTS_CHUNK_TOKENS           → chunk_tokens (16)
      SPARKTTS_LEFT_OVERLAP_TOKENS    → left_overlap_tokens (12)
      SPARKTTS_MAX_TOKENS             → max_tokens (800, runaway cap)
      SPARKTTS_MAX_SEMANTIC           → max_semantic (600, BiCodec T ceiling)
      SPARKTTS_DEFAULT_GENDER/PITCH/SPEED → default style labels
      SPARKTTS_VOICES_DIR             → voices_dir (clone VoiceProfile registry dir; None→off)
      SPARKTTS_CLONE_USE_REF_SEMANTIC → clone_use_ref_semantic (strategy B; default off)
      OVS_TTS_WORKER_CONCURRENCY / profile sparktts_worker_concurrency /
        tts_worker_concurrency → worker_concurrency (1; gates worker --max_slots)
      OVS_TTS_MODEL_ID                → model_id ("sparktts-0p5b")

    NOTE: env reads happen HERE (product layer), never at voxedge import/module
    scope — preserves voxedge's zero-env property (trt_edge_llm_tts_env_staleness).
    """
    from voxedge.backends.jetson.sparktts_trt import SparkTTSConfig

    if env is None:
        env = os.environ
    p = profile if isinstance(profile, dict) else {}

    def _pint(envkey, *pkeys, default):
        v = env.get(envkey)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
        for k in pkeys:
            pv = p.get(k)
            if pv is not None:
                try:
                    return int(pv)
                except (TypeError, ValueError):
                    pass
        return default

    def _pfloat(envkey, default):
        v = env.get(envkey)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        return default

    # worker_concurrency precedence: env OVS_TTS_WORKER_CONCURRENCY → profile → 1.
    conc = _pint("OVS_TTS_WORKER_CONCURRENCY",
                 "sparktts_worker_concurrency", "tts_worker_concurrency", default=1)

    return SparkTTSConfig(
        worker_binary=env.get("SPARKTTS_WORKER_BINARY", "/opt/jv-workers/spark_tts_worker"),
        plugin_path=env.get("SPARKTTS_PLUGIN_PATH", "/opt/edgellm/libNvInfer_edgellm_plugin.so"),
        llm_engine_dir=env.get("SPARKTTS_LLM_ENGINE_DIR", "/opt/models/sparktts-0p5b/llm_engine"),
        tokenizer_dir=env.get("SPARKTTS_TOKENIZER_DIR") or None,
        bicodec_engine=env.get("SPARKTTS_BICODEC_ENGINE",
                               "/opt/models/sparktts-0p5b/bicodec_decoder_dynT.fp16.engine"),
        speaker_decoder_engine=env.get("SPARKTTS_SPEAKER_DECODER_ENGINE",
                                       "/opt/models/sparktts-0p5b/sparktts_speaker_decoder.fp32.engine"),
        ld_library_path=env.get("SPARKTTS_LD_LIBRARY_PATH") or None,
        sample_rate=_pint("SPARKTTS_SAMPLE_RATE", "tts_sample_rate", default=16000),
        first_chunk_tokens=_pint("SPARKTTS_FIRST_CHUNK_TOKENS", default=6),
        chunk_tokens=_pint("SPARKTTS_CHUNK_TOKENS", default=16),
        left_overlap_tokens=_pint("SPARKTTS_LEFT_OVERLAP_TOKENS", default=12),
        max_tokens=_pint("SPARKTTS_MAX_TOKENS", default=800),
        max_semantic=_pint("SPARKTTS_MAX_SEMANTIC", default=600),
        temperature=_pfloat("SPARKTTS_TEMPERATURE", default=1.0),
        top_k=_pint("SPARKTTS_TOP_K", default=1),
        top_p=_pfloat("SPARKTTS_TOP_P", default=1.0),
        default_gender=env.get("SPARKTTS_DEFAULT_GENDER", "female"),
        default_pitch=env.get("SPARKTTS_DEFAULT_PITCH", "moderate"),
        default_speed=env.get("SPARKTTS_DEFAULT_SPEED", "moderate"),
        voices_dir=env.get("SPARKTTS_VOICES_DIR") or None,
        clone_use_ref_semantic=str(
            env.get("SPARKTTS_CLONE_USE_REF_SEMANTIC", "0")
        ).strip().lower() in ("1", "true", "yes", "on"),
        worker_concurrency=conc,
        model_id=env.get("OVS_TTS_MODEL_ID") or "sparktts-0p5b",
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


# ── Capability resolution for voxedge backends ─────────────────────────────
# voxedge backends declare ``concurrency_capability`` as an INSTANCE method
# reading the injected config (env-free). The legacy capability_resolver called
# it as a classmethod ``cls.concurrency_capability(profile)`` — which raises on
# the voxedge backends and silently fell back to a serialized default (this
# broke N>1 concurrency-mode resolution). Resolve it correctly here: build the
# config from profile (same as create_*_backend) and instantiate the backend
# (cheap __init__ — stores config, no model load) to read the capability.
_ASR_CONFIG_BUILDERS = {
    "jetson.trt_edge_llm": build_trt_edge_llm_asr_config,
    "jetson.paraformer_trt": build_paraformer_trt_config,
    "cpu.sherpa_asr": build_sherpa_asr_config,
    "rk.asr": build_rk_asr_config,
}
_TTS_CONFIG_BUILDERS = {
    "jetson.trt_edge_llm": build_trt_edge_llm_tts_config,
    "jetson.matcha_trt": build_matcha_tts_config,
    "jetson.kokoro_trt": build_kokoro_trt_config,
    "jetson.moss_tts_nano": build_moss_tts_nano_config,
    "jetson.sparktts": build_sparktts_trt_config,
    "cpu.sherpa": build_sherpa_tts_config,
    "rk.tts": build_rk_tts_config,
}


def build_config_for_spec(spec, kind, profile=None):
    """Build the voxedge config dataclass for ``spec`` (kind='asr'|'tts')."""
    builders = _ASR_CONFIG_BUILDERS if kind == "asr" else _TTS_CONFIG_BUILDERS
    builder = builders.get(spec)
    if builder is None:
        return None
    return builder(profile=profile)


def concurrency_capability_for_spec(spec, cls, kind, profile=None):
    """ConcurrencyCapability for a voxedge backend without loading models.

    Returns ``None`` when ``spec`` is not a known voxedge spec (caller falls
    back to the legacy classmethod path).

    NOTE: we deliberately do NOT call ``cls(config=config)`` — some backends do
    heavy work in ``__init__`` (e.g. RK calls ``create_asr()`` which imports
    ``rkvoice_stream`` / inits the NPU), which would crash or init hardware
    during a pure capability probe. Every backend's ``concurrency_capability``
    only reads ``self._config`` (or returns a constant / is a classmethod), so
    we build a config-bearing stub via ``__new__`` and call the method on it,
    skipping ``__init__`` entirely.
    """
    config = build_config_for_spec(spec, kind, profile)
    if config is None:
        return None
    stub = cls.__new__(cls)
    stub._config = config
    try:
        return stub.concurrency_capability()
    except TypeError:
        # classmethod-style ``concurrency_capability(cls, profile=None)``
        # (sherpa / rk) bound through the class.
        return cls.concurrency_capability(profile)
