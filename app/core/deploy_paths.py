"""Product/profile deployment-path constants and resolvers for TRT-Edge-LLM.

Extracted verbatim from ``app/backends/jetson/trt_edge_llm_ipc.py`` (2026-05-31)
during the voxedge migration. The voxedge library backends are env-free; the
product translation layer (``app/core/voxedge_backend_config.py``) consumes
these env-derived path constants/resolvers to fill the voxedge config
dataclasses. Keeping them here (not in voxedge) preserves voxedge's zero-env
property while keeping byte-identical env/default behaviour with the legacy
backend module.

Provides:
  - Path constants (override via env vars): binaries, plugin, engine/artifact
    dirs.
  - Fresh-read resolver functions that re-read os.environ each call (hot-reload
    safe).

The generic, env-free helpers (``run_binary`` / ``write_safetensors`` /
``audio_bytes_to_mel`` / mel constants) live in
``voxedge.backends.jetson.trt_edge_llm_ipc`` and are NOT duplicated here.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths — all overridable via environment variables
# ---------------------------------------------------------------------------

_EDGE_LLM_BASE = os.environ.get(
    "EDGE_LLM_BASE", os.path.expanduser("~/project/tensorrt-edge-llm")
)
_EDGE_LLM_BUILD = os.path.join(
    _EDGE_LLM_BASE,
    os.environ.get("EDGE_LLM_BUILD_DIR", "build_sm87"),
)
_OPENVOICESTREAM_BASE = (
    os.environ.get("OVS_BASE")
    or os.path.expanduser("~/project/openvoicestream")
)
_VOICE_WORKER_BUILD = os.environ.get(
    "OVS_WORKER_BUILD",
    os.path.join(_OPENVOICESTREAM_BASE, "build", "edgellm_voice_worker", "workers"),
)


QWEN3_RUNTIME_PROFILE = os.environ.get(
    "EDGE_LLM_QWEN3_PROFILE",
    os.environ.get("OVS_QWEN3_PROFILE", "highperf"),
).strip().lower().replace("-", "_")


def qwen3_runtime_profile() -> str:
    """Resolve the qwen3 runtime profile (highperf / official / etc.)
    from the *current* os.environ.

    Mirrors the module-level QWEN3_RUNTIME_PROFILE assignment above. Reading
    env fresh on each call so hot reload picks up new profile defaults without
    forcing the whole process to reimport the module.
    """
    raw = os.environ.get(
        "EDGE_LLM_QWEN3_PROFILE",
        os.environ.get("OVS_QWEN3_PROFILE", "highperf"),
    )
    return raw.strip().lower().replace("-", "_")


def qwen3_highperf_enabled() -> bool:
    return qwen3_runtime_profile() in ("highperf", "perf", "performance", "v2v")


def _prefer_existing(primary: str, fallback: str) -> str:
    return primary if os.path.exists(primary) else fallback

# Binaries
TTS_BINARY = os.environ.get(
    "EDGE_LLM_TTS_BIN",
    os.path.join(_EDGE_LLM_BUILD, "examples/omni/qwen3_tts_inference"),
)
TTS_WORKER_BINARY = os.environ.get(
    "EDGE_LLM_TTS_WORKER_BIN",
    _prefer_existing(
        os.path.join(_VOICE_WORKER_BUILD, "qwen3_tts_worker"),
        os.path.join(_EDGE_LLM_BUILD, "examples/omni/qwen3_tts_worker"),
    ),
)
ASR_BINARY = os.environ.get(
    "EDGE_LLM_ASR_BIN",
    os.path.join(_EDGE_LLM_BUILD, "examples/llm/llm_inference"),
)
ASR_WORKER_BINARY = os.environ.get(
    "EDGE_LLM_ASR_WORKER_BIN",
    _prefer_existing(
        os.path.join(_VOICE_WORKER_BUILD, "qwen3_asr_worker"),
        os.path.join(_EDGE_LLM_BUILD, "examples/llm/qwen3_asr_worker"),
    ),
)
PLUGIN_PATH = os.environ.get(
    "EDGELLM_PLUGIN_PATH",
    os.path.join(_EDGE_LLM_BUILD, "libNvInfer_edgellm_plugin.so"),
)
DEFAULT_PLUGIN_PATH = os.path.join(_EDGE_LLM_BUILD, "libNvInfer_edgellm_plugin.so")
ASR_PLUGIN_PATH = os.environ.get(
    "EDGE_LLM_ASR_PLUGIN_PATH",
    os.environ.get("EDGELLM_ASR_PLUGIN_PATH", DEFAULT_PLUGIN_PATH),
)

# TTS engine directories
_TTS_FIXED_RUNTIME = os.path.expanduser("~/qwen3-tts-edgellm-runtime")
_TTS_DEFAULT_ROOT = (
    _TTS_FIXED_RUNTIME
    if os.path.exists(os.path.join(_TTS_FIXED_RUNTIME, "engines", "talker", "llm.engine"))
    else os.path.expanduser("~/qwen3-tts-trt-edge-llm-export")
)


def _first_existing_dir(*paths: str) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[-1]


TTS_TALKER_DIR = os.environ.get(
    "EDGE_LLM_TTS_TALKER_DIR",
    os.path.join(_TTS_DEFAULT_ROOT, "engines", "talker"),
)
TTS_FULL_TALKER_DIR = os.environ.get("EDGE_LLM_TTS_FULL_TALKER_DIR", TTS_TALKER_DIR)
TTS_PRUNED_TALKER_DIR = os.environ.get("EDGE_LLM_TTS_PRUNED_TALKER_DIR", TTS_TALKER_DIR)
_TTS_VOCAB_PRUNED = os.environ.get("EDGE_LLM_TTS_VOCAB_PRUNED", os.environ.get("QWEN3_TTS_VOCAB_PRUNED", "0")).lower()
if "EDGE_LLM_TTS_TALKER_DIR" not in os.environ:
    if _TTS_VOCAB_PRUNED in ("1", "true", "yes"):
        TTS_TALKER_DIR = TTS_PRUNED_TALKER_DIR
    elif _TTS_VOCAB_PRUNED in ("0", "false", "no"):
        TTS_TALKER_DIR = TTS_FULL_TALKER_DIR
_TTS_DEFAULT_CODE_PREDICTOR_DIR = os.path.join(os.path.dirname(TTS_TALKER_DIR), "code_predictor")
_TTS_BF16_IO_CODE_PREDICTOR_DIR = os.environ.get(
    "EDGE_LLM_TTS_CP_BF16_IO_DIR",
    "/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/cp_dir",
)
TTS_CODE_PREDICTOR_DIR = os.environ.get(
    "EDGE_LLM_TTS_CP_DIR",
    _first_existing_dir(_TTS_BF16_IO_CODE_PREDICTOR_DIR, _TTS_DEFAULT_CODE_PREDICTOR_DIR)
    if qwen3_highperf_enabled()
    else _TTS_DEFAULT_CODE_PREDICTOR_DIR,
)
TTS_CODE2WAV_DIR = os.environ.get(
    "EDGE_LLM_TTS_CODE2WAV_DIR",
    _first_existing_dir(
        os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder_vocoder100_compat/code2wav"),
        os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder_vocoder50_compat/code2wav"),
        os.path.join(_TTS_DEFAULT_ROOT, "engines", "code2wav"),
        os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder/code2wav"),
    ),
)
TTS_TOKENIZER_DIR = os.environ.get(
    "EDGE_LLM_TTS_TOKENIZER_DIR",
    _TTS_DEFAULT_ROOT
    if os.path.exists(os.path.join(_TTS_DEFAULT_ROOT, "processed_chat_template.json"))
    else os.path.expanduser("~/qwen3-tts-trt-edge-llm-export"),
)


# ---------------------------------------------------------------------------
# Fresh-read resolvers for TTS artifact paths
#
# The module-level constants above capture os.environ at *import time*. Hot
# reload via BackendManager.apply_profile() updates os.environ post-import,
# so backends that consume the module-level constants would see stale paths
# on the second profile swap. The resolver functions below mirror the exact
# cold-boot logic but re-read os.environ on each call; consumers should call
# them inside ``__init__`` (or per-use) instead of importing the constants.
# ---------------------------------------------------------------------------


def _tts_vocab_pruned_now() -> str:
    """Return the current EDGE_LLM_TTS_VOCAB_PRUNED value (lowercased)."""
    return os.environ.get(
        "EDGE_LLM_TTS_VOCAB_PRUNED",
        os.environ.get("QWEN3_TTS_VOCAB_PRUNED", "0"),
    ).lower()


def resolve_tts_talker_dir() -> str:
    """Resolve the talker engine dir from the *current* os.environ.

    Mirrors the module-level TTS_TALKER_DIR resolution at lines 111-122 but
    re-reads env each call so hot reload picks up profile-applied values.
    """
    explicit = os.environ.get("EDGE_LLM_TTS_TALKER_DIR")
    default_talker = os.path.join(_TTS_DEFAULT_ROOT, "engines", "talker")
    if explicit:
        return explicit
    full_dir = os.environ.get("EDGE_LLM_TTS_FULL_TALKER_DIR", default_talker)
    pruned_dir = os.environ.get("EDGE_LLM_TTS_PRUNED_TALKER_DIR", default_talker)
    vocab = _tts_vocab_pruned_now()
    if vocab in ("1", "true", "yes"):
        return pruned_dir
    if vocab in ("0", "false", "no"):
        return full_dir
    return default_talker


def resolve_tts_code_predictor_dir() -> str:
    """Resolve the code-predictor dir from the *current* os.environ.

    Mirrors the module-level TTS_CODE_PREDICTOR_DIR resolution at
    lines 123-133 (incl. the qwen3-highperf bf16-io override probe).
    """
    explicit = os.environ.get("EDGE_LLM_TTS_CP_DIR")
    if explicit:
        return explicit
    talker_dir = resolve_tts_talker_dir()
    default_cp = os.path.join(os.path.dirname(talker_dir), "code_predictor")
    bf16_io_cp = os.environ.get(
        "EDGE_LLM_TTS_CP_BF16_IO_DIR",
        "/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/cp_dir",
    )
    if qwen3_highperf_enabled():
        return _first_existing_dir(bf16_io_cp, default_cp)
    return default_cp


def resolve_tts_tokenizer_dir() -> str:
    """Resolve the tokenizer dir from the *current* os.environ.

    Mirrors the module-level TTS_TOKENIZER_DIR resolution at lines 143-148.
    """
    explicit = os.environ.get("EDGE_LLM_TTS_TOKENIZER_DIR")
    if explicit:
        return explicit
    if os.path.exists(os.path.join(_TTS_DEFAULT_ROOT, "processed_chat_template.json")):
        return _TTS_DEFAULT_ROOT
    return os.path.expanduser("~/qwen3-tts-trt-edge-llm-export")


def resolve_tts_code2wav_dir() -> str:
    """Resolve the code2wav engine dir from the *current* os.environ.

    Mirrors the module-level TTS_CODE2WAV_DIR resolution at lines 134-142.
    """
    explicit = os.environ.get("EDGE_LLM_TTS_CODE2WAV_DIR")
    if explicit:
        return explicit
    return _first_existing_dir(
        os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder_vocoder100_compat/code2wav"),
        os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder_vocoder50_compat/code2wav"),
        os.path.join(_TTS_DEFAULT_ROOT, "engines", "code2wav"),
        os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder/code2wav"),
    )


def resolve_tts_worker_binary() -> str:
    """Resolve the TTS worker binary path from the *current* os.environ.

    Mirrors the module-level TTS_WORKER_BINARY resolution at lines 67-73.
    Hot reload may rewrite EDGE_LLM_TTS_WORKER_BIN; instance state captured
    at __init__ then becomes stale until the BackendManager rebuilds the
    backend, but transient resolves still need to honor the new env.
    """
    explicit = os.environ.get("EDGE_LLM_TTS_WORKER_BIN")
    if explicit:
        return explicit
    return _prefer_existing(
        os.path.join(_VOICE_WORKER_BUILD, "qwen3_tts_worker"),
        os.path.join(_EDGE_LLM_BUILD, "examples/omni/qwen3_tts_worker"),
    )


def resolve_asr_worker_binary() -> str:
    """Resolve the ASR worker binary path from the *current* os.environ.

    Mirrors the module-level ASR_WORKER_BINARY resolution at lines 78-84.
    """
    explicit = os.environ.get("EDGE_LLM_ASR_WORKER_BIN")
    if explicit:
        return explicit
    return _prefer_existing(
        os.path.join(_VOICE_WORKER_BUILD, "qwen3_asr_worker"),
        os.path.join(_EDGE_LLM_BUILD, "examples/llm/qwen3_asr_worker"),
    )


def resolve_plugin_path() -> str:
    """Resolve the TRT-Edge-LLM plugin .so path from the *current* os.environ."""
    explicit = os.environ.get("EDGELLM_PLUGIN_PATH")
    if explicit:
        return explicit
    return os.path.join(_EDGE_LLM_BUILD, "libNvInfer_edgellm_plugin.so")



# ASR engine directories
_ASR_PRUNED_ENGINE_DIR = os.path.expanduser(
    "~/qwen3-asr-edgellm-runtime/engines/thinker_prunedembed35k_kv512"
)
_ASR_OFFICIAL_PRUNED_ENGINE_DIR = os.path.expanduser(
    "~/qwen3-asr-edgellm-runtime/engines/thinker_pruned35k_kv512"
)
_ASR_DIALOG_ENGINE_DIR = os.path.expanduser(
    "~/qwen3-asr-edgellm-runtime/engines/thinker_kv512"
)
_ASR_FP8_EMBED_FULL_ENGINE_DIR = os.path.expanduser(
    "~/qwen3-asr-edgellm-runtime/engines/thinker_full_in128_kv256_fp8embed_0510"
)
_ASR_SMALL_FULL_ENGINE_DIR = os.path.expanduser(
    "~/qwen3-asr-edgellm-runtime/engines/thinker_full_in128_kv256_0510"
)
_ASR_EXPORT_ENGINE_DIR = os.path.expanduser("~/qwen3-asr-trt-edge-llm-export/engines/thinker")
ASR_FULL_ENGINE_DIR = os.environ.get(
    "EDGE_LLM_ASR_FULL_ENGINE_DIR",
    _ASR_FP8_EMBED_FULL_ENGINE_DIR
    if os.path.exists(os.path.join(_ASR_FP8_EMBED_FULL_ENGINE_DIR, "llm.engine"))
    else _ASR_SMALL_FULL_ENGINE_DIR
    if os.path.exists(os.path.join(_ASR_SMALL_FULL_ENGINE_DIR, "llm.engine"))
    else _ASR_DIALOG_ENGINE_DIR
    if os.path.exists(os.path.join(_ASR_DIALOG_ENGINE_DIR, "llm.engine"))
    else _ASR_EXPORT_ENGINE_DIR,
)
ASR_PRUNED_ENGINE_DIR = os.environ.get(
    "EDGE_LLM_ASR_PRUNED_ENGINE_DIR",
    _ASR_PRUNED_ENGINE_DIR
    if os.path.exists(os.path.join(_ASR_PRUNED_ENGINE_DIR, "llm.engine"))
    else _ASR_OFFICIAL_PRUNED_ENGINE_DIR,
)
_ASR_VOCAB_PRUNED = os.environ.get("EDGE_LLM_ASR_VOCAB_PRUNED", "0").lower()
ASR_ENGINE_DIR = os.environ.get(
    "EDGE_LLM_ASR_ENGINE_DIR",
    ASR_PRUNED_ENGINE_DIR
    if _ASR_VOCAB_PRUNED in ("1", "true", "yes")
    else ASR_FULL_ENGINE_DIR
    if _ASR_VOCAB_PRUNED in ("0", "false", "no")
    else _ASR_PRUNED_ENGINE_DIR
    if os.path.exists(os.path.join(_ASR_PRUNED_ENGINE_DIR, "llm.engine"))
    else _ASR_OFFICIAL_PRUNED_ENGINE_DIR
    if os.path.exists(os.path.join(_ASR_OFFICIAL_PRUNED_ENGINE_DIR, "llm.engine"))
    else ASR_FULL_ENGINE_DIR,
)
ASR_AUDIO_ENC_DIR = os.environ.get(
    "EDGE_LLM_ASR_AUDIO_ENC_DIR",
    os.path.expanduser(
        "~/qwen3-asr-trt-edge-llm-export/engines/audio_encoder"
    ),
)
