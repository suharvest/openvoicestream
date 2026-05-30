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
