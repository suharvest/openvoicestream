"""Fidelity table for app/core/voxedge_backend_config.py builders.

These builders are the single env/profile → voxedge config translation layer
(see the module docstring). A silently-renamed or dropped env key would route a
deploy-time override into the void — the backend would run on a default while
the operator believes the override took effect. This file locks the mapping:

  * (a) DEFAULT  — with the relevant env cleared, the builder emits the
    documented legacy default for each asserted field.
  * (b) OVERRIDE — setting the env key (path / numeric param / B4 chunk keys)
    is reflected in the corresponding config field.

Each builder gets at least one default + one override case. Builders whose path
fields derive from module-level constants in
``voxedge.backends.jetson.trt_edge_llm_ipc`` (trt_edge_llm asr/tts) assert the
scalar/env-driven fields directly and the path fields via override-flow only
(the module constants are import-cached and machine-derived).

NB these tests pass an explicit ``env`` dict to each builder, so they never
touch / depend on the ambient process environment.
"""

from __future__ import annotations

import pytest

from app.core import voxedge_backend_config as vbc


# ── jetson.trt_edge_llm ASR ──────────────────────────────────────────────────


def test_trt_edge_llm_asr_defaults():
    cfg = vbc.build_trt_edge_llm_asr_config(env={})
    assert cfg.use_worker is True
    assert cfg.mel_tensor_name == "mel"
    assert cfg.max_mel_frames == 6000
    assert cfg.max_slots == 1
    assert cfg.stream_mode == "accumulate"
    assert cfg.stream_chunk_sec == 0.5
    assert cfg.stream_unfixed_chunks == 2
    assert cfg.stream_unfixed_tokens == 5
    assert cfg.segment_cap_sec == 5.5
    assert cfg.temperature == 1.0
    assert cfg.top_p == 1.0
    assert cfg.top_k == 1
    assert cfg.max_generate_length == 200
    assert cfg.prewarm_max == 6
    assert cfg.offline_segment_enabled is True


def test_trt_edge_llm_asr_overrides():
    env = {
        "EDGE_LLM_ASR_ENGINE_DIR": "/custom/asr/engines",
        "EDGE_LLM_ASR_MEL_TENSOR_NAME": "mel2",
        "EDGE_LLM_ASR_MAX_MEL_FRAMES": "8000",
        "EDGE_LLM_ASR_MAX_CONCURRENT": "3",
        "EDGE_LLM_ASR_STREAM_MODE": "final",
        "EDGE_LLM_ASR_STREAM_CHUNK_SEC": "1.5",
        "EDGE_LLM_ASR_STREAM_UNFIXED_CHUNKS": "4",
        "EDGE_LLM_ASR_STREAM_UNFIXED_TOKENS": "9",
        "EDGE_LLM_ASR_SEGMENT_CAP_SEC": "7.0",
        "ASR_TEMPERATURE": "0.7",
        "ASR_TOP_P": "0.8",
        "ASR_TOP_K": "5",
        "ASR_MAX_GENERATE_LENGTH": "256",
        "EDGE_LLM_ASR_PREWARM_MAX": "12",
        "EDGE_LLM_ASR_WORKER": "0",
        "EDGE_LLM_ASR_OFFLINE_SEGMENT": "0",
    }
    cfg = vbc.build_trt_edge_llm_asr_config(env=env)
    assert cfg.engine_dir == "/custom/asr/engines"
    assert cfg.mel_tensor_name == "mel2"
    assert cfg.max_mel_frames == 8000
    assert cfg.max_slots == 3
    assert cfg.stream_mode == "final"
    assert cfg.stream_chunk_sec == 1.5
    assert cfg.stream_unfixed_chunks == 4
    assert cfg.stream_unfixed_tokens == 9
    assert cfg.segment_cap_sec == 7.0
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.8
    assert cfg.top_k == 5
    assert cfg.max_generate_length == 256
    assert cfg.prewarm_max == 12
    assert cfg.use_worker is False
    assert cfg.offline_segment_enabled is False


def test_trt_edge_llm_asr_max_slots_from_profile():
    cfg = vbc.build_trt_edge_llm_asr_config(profile={"asr_max_slots": 4}, env={})
    assert cfg.max_slots == 4


# ── jetson.paraformer_trt (G2) ───────────────────────────────────────────────


def test_paraformer_defaults():
    cfg = vbc.build_paraformer_trt_config(env={})
    assert cfg.model_dir == "/opt/models/paraformer-streaming"
    assert cfg.enc_engine.endswith("engines/paraformer_encoder_sp1_80.plan")
    assert cfg.dec_engine.endswith("engines/paraformer_decoder_fp16.plan")
    assert cfg.tokens_path.endswith("tokens.txt")
    assert cfg.preroll_ms == 100
    # G2: bounded concurrency default
    assert cfg.max_concurrent == 2


def test_paraformer_overrides():
    env = {
        "PARAFORMER_MODEL_DIR": "/custom/paraformer",
        "PARAFORMER_ENC_ENGINE": "/custom/enc.plan",
        "PARAFORMER_TOKENS": "/custom/tokens.txt",
        "PARAFORMER_PREROLL_MS": "250",
        "PARAFORMER_MAX_CONCURRENT": "4",
    }
    cfg = vbc.build_paraformer_trt_config(env=env)
    assert cfg.model_dir == "/custom/paraformer"
    assert cfg.enc_engine == "/custom/enc.plan"
    assert cfg.tokens_path == "/custom/tokens.txt"
    assert cfg.preroll_ms == 250
    assert cfg.max_concurrent == 4


def test_paraformer_max_concurrent_from_profile():
    cfg = vbc.build_paraformer_trt_config(profile={"asr_max_slots": 3}, env={})
    assert cfg.max_concurrent == 3
    # env beats profile
    cfg2 = vbc.build_paraformer_trt_config(
        profile={"asr_max_slots": 3}, env={"PARAFORMER_MAX_CONCURRENT": "1"}
    )
    assert cfg2.max_concurrent == 1


# ── cpu.sherpa ASR ───────────────────────────────────────────────────────────


def test_sherpa_asr_defaults():
    cfg = vbc.build_sherpa_asr_config(env={})
    assert cfg.language_mode == "zh_en"
    # builder passes None; SherpaASRConfig.__post_init__ resolves the per-language
    # streaming model dir default. The builder's job is to NOT override it.
    assert cfg.streaming_model_dir is not None
    assert cfg.streaming_provider == "cuda"
    assert cfg.offline_provider == "cuda"  # falls back to streaming_provider
    assert cfg.num_threads == 4
    assert cfg.model_root == "/opt/models"


def test_sherpa_asr_overrides():
    env = {
        "LANGUAGE_MODE": "en",
        "STREAMING_MODEL_DIR": "/custom/stream",
        "STREAMING_ASR_PROVIDER": "cpu",
        "OFFLINE_ASR_PROVIDER": "cuda",
        "STREAMING_ASR_NUM_THREADS": "8",
        "MODEL_DIR": "/custom/models",
    }
    cfg = vbc.build_sherpa_asr_config(env=env)
    assert cfg.language_mode == "en"
    assert cfg.streaming_model_dir == "/custom/stream"
    assert cfg.streaming_provider == "cpu"
    assert cfg.offline_provider == "cuda"
    assert cfg.num_threads == 8
    assert cfg.model_root == "/custom/models"


# ── rk.asr ───────────────────────────────────────────────────────────────────


def test_rk_asr_defaults():
    cfg = vbc.build_rk_asr_config(env={})
    assert cfg.platform == "rk3576"
    assert cfg.energy_split_rms == 0.003
    assert cfg.energy_min_silence_ms == 120
    assert cfg.long_audio_threshold_s == 15.0


def test_rk_asr_overrides():
    env = {
        "RK_PLATFORM": "rk3588",
        "ASR_ENERGY_SPLIT_RMS": "0.01",
        "ASR_ENERGY_MIN_SILENCE_MS": "200",
    }
    cfg = vbc.build_rk_asr_config(env=env)
    assert cfg.platform == "rk3588"
    assert cfg.energy_split_rms == 0.01
    assert cfg.energy_min_silence_ms == 200


# ── jetson.matcha_trt ────────────────────────────────────────────────────────


def test_matcha_defaults():
    cfg = vbc.build_matcha_tts_config(env={})
    assert cfg.model_base == "/opt/models/matcha-icefall-zh-en"
    assert cfg.language_mode == "zh_en"
    assert cfg.min_mel_frames == 72
    assert cfg.acoustic_ep == ""
    assert cfg.stream_max_workers == 2
    assert cfg.arena_size_mb == 16
    assert cfg.stream_chunk_ms == 40
    assert cfg.model_id == "matcha_trt"


def test_matcha_overrides():
    env = {
        "MATCHA_MODEL_BASE": "/custom/matcha",
        "LANGUAGE_MODE": "en",
        "VOCOS_ENGINE": "/custom/vocos.engine",
        "MATCHA_MIN_MEL_FRAMES": "96",
        "MATCHA_ACOUSTIC_EP": "cuda",
        "OVS_TTS_STREAM_MAX_WORKERS": "3",
        "OVS_MATCHA_ARENA_SIZE_MB": "32",
        "MATCHA_STREAM_CHUNK_MS": "80",
        "OVS_TTS_MODEL_ID": "matcha_custom",
    }
    cfg = vbc.build_matcha_tts_config(env=env)
    assert cfg.model_base == "/custom/matcha"
    assert cfg.language_mode == "en"
    assert cfg.vocos_engine == "/custom/vocos.engine"
    assert cfg.min_mel_frames == 96
    assert cfg.acoustic_ep == "cuda"
    assert cfg.stream_max_workers == 3
    assert cfg.arena_size_mb == 32
    assert cfg.stream_chunk_ms == 80
    assert cfg.model_id == "matcha_custom"


def test_matcha_stream_workers_from_profile():
    cfg = vbc.build_matcha_tts_config(
        profile={"tts_stream_max_workers": 4}, env={}
    )
    assert cfg.stream_max_workers == 4


# ── jetson.kokoro_trt ────────────────────────────────────────────────────────


def test_kokoro_defaults():
    cfg = vbc.build_kokoro_trt_config(env={})
    assert cfg.model_base == "/opt/models/kokoro-multi-lang-v1_0"
    assert cfg.max_tokens == 510
    assert cfg.default_speaker_id == 52
    assert cfg.default_speed == 1.0
    assert cfg.stream_segment_tokens == 64
    assert cfg.stream_segment_text is True
    assert cfg.synth_segment_text is True
    assert cfg.synth_max_segment_tokens == 64
    assert cfg.runtime_mode == "auto"
    assert cfg.stream_max_workers == 2
    assert cfg.arena_size_mb == 16
    assert cfg.stream_chunk_ms == 40
    assert cfg.split_cpu_fallback is True
    assert cfg.max_seq_len_fallback == 128
    assert cfg.hybrid_token_len == 0
    assert cfg.model_id == "kokoro_trt"


def test_kokoro_overrides():
    env = {
        "KOKORO_MODEL_BASE": "/custom/kokoro",
        "KOKORO_TRT_ENGINE": "/custom/kokoro.engine",
        "KOKORO_MAX_TOKENS": "256",
        "KOKORO_DEFAULT_SID": "7",
        "TTS_DEFAULT_SPEED": "1.2",
        "KOKORO_STREAM_MAX_SEGMENT_TOKENS": "32",
        "KOKORO_STREAM_SEGMENT_TEXT": "0",
        "KOKORO_SYNTH_SEGMENT_TEXT": "0",
        "KOKORO_TRT_RUNTIME": "split",
        "OVS_TTS_STREAM_MAX_WORKERS": "3",
        "OVS_KOKORO_ARENA_SIZE_MB": "64",
        "KOKORO_STREAM_CHUNK_MS": "80",
        "KOKORO_SPLIT_CPU_FALLBACK": "0",
        "KOKORO_SPLIT_MAX_SEQ_LEN": "256",
        "KOKORO_HYBRID_TOKEN_LEN": "5",
        "OVS_TTS_MODEL_ID": "kokoro_custom",
    }
    cfg = vbc.build_kokoro_trt_config(env=env)
    assert cfg.model_base == "/custom/kokoro"
    assert cfg.engine_path == "/custom/kokoro.engine"
    assert cfg.max_tokens == 256
    assert cfg.default_speaker_id == 7
    assert cfg.default_speed == 1.2
    assert cfg.stream_segment_tokens == 32
    assert cfg.stream_segment_text is False
    assert cfg.synth_segment_text is False
    # synth_max defaults to stream_segment_tokens when unset
    assert cfg.synth_max_segment_tokens == 32
    assert cfg.runtime_mode == "split"
    assert cfg.stream_max_workers == 3
    assert cfg.arena_size_mb == 64
    assert cfg.stream_chunk_ms == 80
    assert cfg.split_cpu_fallback is False
    assert cfg.max_seq_len_fallback == 256
    assert cfg.hybrid_token_len == 5
    assert cfg.model_id == "kokoro_custom"


# ── jetson.qwen3_trt ─────────────────────────────────────────────────────────


def test_qwen3_defaults():
    cfg = vbc.build_qwen3_trt_config(env={})
    assert cfg.model_base == "/opt/models/qwen3-tts"
    assert cfg.is_customvoice is False
    assert cfg.model_id == "qwen3-tts"
    assert cfg.int8_eos_logit_offset == -10.0
    assert cfg.talker_cuda_graph is True
    assert cfg.vocoder_max_frames == 100
    assert cfg.use_trt_vocoder is True
    assert cfg.offline_streaming_for_long is True
    assert cfg.numpy_sampling is True
    assert cfg.default_seed == 0
    assert cfg.product_segment_text is False
    assert cfg.product_segment_max_chars == 20
    assert cfg.product_comma_pause_ms == 120
    assert cfg.product_hard_pause_ms == 180


def test_qwen3_overrides():
    env = {
        "QWEN3_MODEL_BASE": "/custom/qwen3",
        "QWEN3_TALKER_ENGINE": "/custom/talker.engine",
        "TTS_INT8_EOS_LOGIT_OFFSET": "-5.0",
        "TTS_TALKER_CUDA_GRAPH": "0",
        "TTS_TRT_VOCODER_MAX_FRAMES": "200",
        "TTS_VOCODER_TRT": "0",
        "QWEN3_TTS_NUMPY_SAMPLING": "0",
        "OVS_TTS_SEED": "7",
        "QWEN3_TTS_PRODUCT_SEGMENT_TEXT": "1",
        "QWEN3_TTS_PRODUCT_SEGMENT_MAX_CHARS": "40",
        "QWEN3_TTS_PRODUCT_COMMA_PAUSE_MS": "200",
        "QWEN3_TTS_PRODUCT_HARD_PAUSE_MS": "300",
    }
    cfg = vbc.build_qwen3_trt_config(env=env)
    assert cfg.model_base == "/custom/qwen3"
    assert cfg.talker_engine == "/custom/talker.engine"
    assert cfg.int8_eos_logit_offset == -5.0
    assert cfg.talker_cuda_graph is False
    assert cfg.vocoder_max_frames == 200
    assert cfg.use_trt_vocoder is False
    assert cfg.numpy_sampling is False
    assert cfg.default_seed == 7
    assert cfg.product_segment_text is True
    assert cfg.product_segment_max_chars == 40
    assert cfg.product_comma_pause_ms == 200
    assert cfg.product_hard_pause_ms == 300


def test_qwen3_customvoice_detection():
    cfg = vbc.build_qwen3_trt_config(env={"QWEN3_TTS_VARIANT": "customvoice"})
    assert cfg.is_customvoice is True
    assert cfg.model_id == "qwen3-tts-customvoice"
    # via OVS_TTS_MODEL_ID containing "customvoice"
    cfg2 = vbc.build_qwen3_trt_config(env={"OVS_TTS_MODEL_ID": "qwen3-tts-customvoice"})
    assert cfg2.is_customvoice is True


# ── jetson.trt_edge_llm TTS (incl. B4 chunk keys) ────────────────────────────


def test_trt_edge_llm_tts_defaults():
    cfg = vbc.build_trt_edge_llm_tts_config(env={})
    assert cfg.model_id == "trt_edgellm"
    assert cfg.backend_mode == "edgellm_worker"
    assert cfg.use_worker is True
    assert cfg.worker_concurrency == 1
    assert cfg.perf_profile == "quality"
    assert cfg.seed == 42
    assert cfg.talker_temperature == 0.9
    assert cfg.talker_top_k == 50
    assert cfg.talker_top_p == 1.0
    assert cfg.max_audio_length == 1024
    assert cfg.min_audio_length == 30
    assert cfg.repetition_penalty == 1.05
    assert cfg.segment_max_chars_latin == 120
    assert cfg.segment_max_chars_cjk == 48
    assert cfg.streaming_profile == "continuous_playback"
    # B4 chunk keys: unset => None (streaming_profile-derived)
    assert cfg.first_chunk_frames is None
    assert cfg.chunk_frames is None
    assert cfg.adaptive_chunks is None
    assert cfg.max_chunk_frames is None
    assert cfg.chunk_growth_frames is None


def test_trt_edge_llm_tts_overrides():
    env = {
        "OVS_TTS_MODEL_ID": "trt_custom",
        "OVS_TTS_BACKEND": "edgellm_native",
        "EDGE_LLM_TTS_WORKER": "0",
        "OVS_TTS_WORKER_CONCURRENCY": "3",
        "EDGE_LLM_TTS_PERF_PROFILE": "speed",
        "OVS_TTS_SEED": "99",
        "OVS_TTS_TALKER_TEMPERATURE": "0.5",
        "OVS_TTS_TALKER_TOP_K": "20",
        "TTS_MAX_AUDIO_LENGTH": "2048",
        "TTS_MIN_AUDIO_LENGTH": "60",
        "TTS_REPETITION_PENALTY": "1.2",
        "EDGE_LLM_TTS_SEGMENT_MAX_CHARS": "200",
        "EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS": "60",
        "EDGE_LLM_TTS_STREAMING_PROFILE": "low_latency",
    }
    cfg = vbc.build_trt_edge_llm_tts_config(env=env)
    assert cfg.model_id == "trt_custom"
    assert cfg.backend_mode == "edgellm_native"
    assert cfg.use_worker is False
    assert cfg.worker_concurrency == 3
    assert cfg.perf_profile == "speed"
    assert cfg.seed == 99
    assert cfg.talker_temperature == 0.5
    assert cfg.talker_top_k == 20
    assert cfg.max_audio_length == 2048
    assert cfg.min_audio_length == 60
    assert cfg.repetition_penalty == 1.2
    assert cfg.segment_max_chars_latin == 200
    assert cfg.segment_max_chars_cjk == 60
    assert cfg.streaming_profile == "low_latency"


def test_trt_edge_llm_tts_chunk_keys_b4():
    """B4: the 5 chunk-frame env keys must each map through."""
    env = {
        "EDGE_LLM_TTS_FIRST_CHUNK_FRAMES": "8",
        "EDGE_LLM_TTS_CHUNK_FRAMES": "16",
        "EDGE_LLM_TTS_ADAPTIVE_CHUNKS": "1",
        "EDGE_LLM_TTS_MAX_CHUNK_FRAMES": "64",
        "EDGE_LLM_TTS_CHUNK_GROWTH_FRAMES": "4",
    }
    cfg = vbc.build_trt_edge_llm_tts_config(env=env)
    assert cfg.first_chunk_frames == 8
    assert cfg.chunk_frames == 16
    assert cfg.adaptive_chunks is True
    assert cfg.max_chunk_frames == 64
    assert cfg.chunk_growth_frames == 4


def test_trt_edge_llm_tts_worker_concurrency_from_profile():
    cfg = vbc.build_trt_edge_llm_tts_config(
        profile={"tts_worker_concurrency": 4}, env={}
    )
    assert cfg.worker_concurrency == 4


# ── jetson.moss_tts_nano ─────────────────────────────────────────────────────


def test_moss_defaults():
    cfg = vbc.build_moss_tts_nano_config(env={})
    assert cfg.worker_bin == "/opt/jv-workers/moss_tts_nano_worker"
    assert cfg.engine_dir == "/opt/models/moss-tts-nano/engines"
    assert cfg.codec_onnx_dir == "/opt/models/moss-tts-nano/codec_onnx"
    assert cfg.max_slots == 1
    assert cfg.max_seq_len == 2048
    assert cfg.sample_rate == 48000
    assert cfg.channels == 2
    assert cfg.py_repo == "/opt/moss-tts-nano-py"
    assert cfg.ort_ep == "cpu"
    assert cfg.ort_threads == 4


def test_moss_overrides():
    env = {
        "MOSS_WORKER_BIN": "/custom/moss_worker",
        "MOSS_ENGINE_DIR": "/custom/moss/engines",
        "MOSS_CODEC_ONNX_DIR": "/custom/moss/codec",
        "MOSS_PY_REPO": "/custom/moss-py",
        "MOSS_ORT_EP": "cuda",
        "MOSS_ORT_THREADS": "8",
    }
    cfg = vbc.build_moss_tts_nano_config(env=env)
    assert cfg.worker_bin == "/custom/moss_worker"
    assert cfg.engine_dir == "/custom/moss/engines"
    assert cfg.codec_onnx_dir == "/custom/moss/codec"
    assert cfg.py_repo == "/custom/moss-py"
    assert cfg.ort_ep == "cuda"
    assert cfg.ort_threads == 8


def test_moss_shape_from_profile():
    cfg = vbc.build_moss_tts_nano_config(
        profile={
            "moss_max_slots": 2,
            "moss_max_seq_len": 4096,
            "moss_sample_rate": 24000,
            "moss_channels": 1,
        },
        env={},
    )
    assert cfg.max_slots == 2
    assert cfg.max_seq_len == 4096
    assert cfg.sample_rate == 24000
    assert cfg.channels == 1


# ── cpu.sherpa TTS ───────────────────────────────────────────────────────────


def test_sherpa_tts_defaults():
    cfg = vbc.build_sherpa_tts_config(env={})
    assert cfg.language_mode == "zh_en"
    # builder passes None; SherpaTTSConfig.__post_init__ resolves model_dir +
    # default_speaker_id per language_mode. The builder must not override them.
    assert cfg.model_dir is not None
    assert cfg.provider == "cuda"
    assert cfg.num_threads == 4
    assert cfg.default_speaker_id is not None
    assert cfg.default_speed == 1.0
    assert cfg.pitch_shift == 0.0
    assert cfg.model_id == "sherpa"


def test_sherpa_tts_overrides():
    env = {
        "LANGUAGE_MODE": "en",
        "SHERPA_TTS_MODEL_DIR": "/custom/tts",
        "TTS_PROVIDER": "cpu",
        "TTS_NUM_THREADS": "6",
        "TTS_DEFAULT_SID": "3",
        "TTS_DEFAULT_SPEED": "1.3",
        "TTS_PITCH_SHIFT": "2.0",
        "OVS_TTS_MODEL_ID": "sherpa_custom",
    }
    cfg = vbc.build_sherpa_tts_config(env=env)
    assert cfg.language_mode == "en"
    assert cfg.model_dir == "/custom/tts"
    assert cfg.provider == "cpu"
    assert cfg.num_threads == 6
    assert cfg.default_speaker_id == 3
    assert cfg.default_speed == 1.3
    assert cfg.pitch_shift == 2.0
    assert cfg.model_id == "sherpa_custom"


# ── G3: kokoro env/profile → config → capability precedence ──────────────────


def _kokoro_cap(cfg):
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend

    stub = KokoroTRTBackend.__new__(KokoroTRTBackend)
    stub._config = cfg
    return stub.concurrency_capability()


def test_kokoro_capability_default():
    cfg = vbc.build_kokoro_trt_config(env={})
    cap = _kokoro_cap(cfg)
    assert cap.max_concurrent == 2
    assert cap.supports_parallel is True
    assert cap.scaling_mode == "single_runtime_multiplex"


def test_kokoro_capability_env_override():
    cfg = vbc.build_kokoro_trt_config(env={"OVS_TTS_STREAM_MAX_WORKERS": "3"})
    assert cfg.stream_max_workers == 3
    assert _kokoro_cap(cfg).max_concurrent == 3


def test_kokoro_capability_profile_then_env_precedence():
    # profile alone
    cfg = vbc.build_kokoro_trt_config(profile={"tts_stream_max_workers": 4}, env={})
    assert cfg.stream_max_workers == 4
    assert _kokoro_cap(cfg).max_concurrent == 4
    # env wins over profile
    cfg2 = vbc.build_kokoro_trt_config(
        profile={"tts_stream_max_workers": 4},
        env={"OVS_TTS_STREAM_MAX_WORKERS": "1"},
    )
    assert cfg2.stream_max_workers == 1
    assert _kokoro_cap(cfg2).supports_parallel is False


# ── rk.tts ───────────────────────────────────────────────────────────────────


def test_rk_tts_defaults():
    cfg = vbc.build_rk_tts_config(env={})
    assert cfg.model_id == "rk"


def test_rk_tts_overrides():
    cfg = vbc.build_rk_tts_config(env={"OVS_TTS_MODEL_ID": "rk_custom"})
    assert cfg.model_id == "rk_custom"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
