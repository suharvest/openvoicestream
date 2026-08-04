import json
from pathlib import Path

from server.core.capability_resolver import resolve
from server.core.coordinator import BackendCoordinator


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "deploy/docker/Dockerfile.jetson.edgellm-v091-runtime"
COMPOSE = ROOT / "deploy/docker-compose.edgellm-v091-voice.yml"
RELEASE_LOCK = ROOT / "deploy/artifacts/v091-release-lock.json"
BASE_PROFILE = ROOT / "configs/profiles/jetson-edgellm-v091-qwen3ttsbase.json"
BASE_N2_PROFILE = ROOT / "configs/profiles/jetson-edgellm-v091-qwen3ttsbase-isolated-n2.json"
MATCHA_PROFILE = ROOT / "configs/profiles/jetson-edgellm-v091-matcha.json"


def test_runtime_image_packages_every_v091_profile():
    dockerfile = DOCKERFILE.read_text()
    production_profiles = {
        "jetson-edgellm-v091-customvoice.json",
        "jetson-edgellm-v091-asr-validation.json",
        "jetson-edgellm-v091-moss.json",
        "jetson-edgellm-v091-matcha.json",
        "jetson-edgellm-v091-n2.json",
        "jetson-edgellm-v091-qwen3ttsbase-isolated-n2.json",
        "jetson-edgellm-v091-qwen3ttsbase-triple.json",
        "jetson-edgellm-v091-qwen3ttsbase.json",
        "jetson-edgellm-v091-sparktts.json",
    }

    for profile_name in production_profiles:
        expected = (
            f"COPY configs/profiles/{profile_name} "
            "/opt/speech/configs/profiles/"
        )
        assert expected in dockerfile, profile_name


def test_v091_profiles_with_asr_publish_canonical_model_identity():
    for path in (ROOT / "configs/profiles").glob("jetson-edgellm-v091*.json"):
        profile = json.loads(path.read_text())
        if profile.get("asr_backend"):
            assert profile.get("asr_model_id") == "qwen3-asr", path


def test_runtime_image_uses_model_level_artifacts_and_download_mirrors():
    dockerfile = DOCKERFILE.read_text()
    compose = COMPOSE.read_text()

    assert "com.seeed.model-artifact-schema=\"2\"" in dockerfile
    assert "HF_MODEL_CACHE_ROOT=/opt/models" in dockerfile
    assert "HF_MODEL_CACHE_ROOT: /opt/models" in compose
    assert "QWEN3_ARTIFACT_SET" not in dockerfile
    assert "QWEN3_ARTIFACT_SET" not in compose
    assert "QWEN3_HF_REPO_ID" not in dockerfile
    assert "QWEN3_HF_REPO_ID" not in compose
    assert (
        "seeed-local-voice:jetson-jp62-trt103-edgellm-v091"
        in compose
    )
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in dockerfile
    assert "HF_ENDPOINT=https://hf-mirror.com" in dockerfile
    assert "HF_ENDPOINT: ${HF_ENDPOINT:-https://hf-mirror.com}" in compose
    assert 'PIP_INDEX_URL="${PIP_INDEX_URL}" pip install' in dockerfile


def test_runtime_image_owns_workers_plugin_but_compose_volume_owns_engines():
    dockerfile = DOCKERFILE.read_text()
    compose = COMPOSE.read_text()

    assert "COPY deploy/artifacts/v091-release-gate/bin/" in dockerfile
    assert "COPY deploy/artifacts/v091-release-gate/libNvInfer_edgellm_plugin.so" in dockerfile
    assert "COPY deploy/artifacts/v091-release-gate/python/" in dockerfile
    assert "speech-models:/opt/models" in compose
    assert "/home/harvest/edgellm-artifacts/" not in compose
    assert "HF_MODEL_CACHE_ROOT: /opt/models" in compose

    for path in (ROOT / "configs/profiles").glob("jetson-edgellm-v091*.json"):
        profile = json.loads(path.read_text())
        serialized = json.dumps(profile, sort_keys=True)
        assert "/opt/edgellm-v091/engines" not in serialized, path
        for engine in profile.get("required_engines", []):
            engine_path = str(engine.get("engine_path", ""))
            if engine_path.startswith("/opt/models/"):
                model_roots = {
                    entry["root"].rstrip("/") + "/"
                    for entry in profile["model_artifacts"]
                }
                assert any(engine_path.startswith(root) for root in model_roots), path


def test_v091_base_profiles_cap_generation_to_the_512_frame_vocoder():
    for path in (BASE_PROFILE, BASE_N2_PROFILE):
        profile = json.loads(path.read_text())
        assert profile["env"]["TTS_MAX_AUDIO_LENGTH"] == "512", path
        assert profile["env"]["EDGE_LLM_TTS_CODE2WAV_DIR"].endswith(
            "/tts_base_code2wav_512"
        ), path


def test_formal_v091_base_profiles_remove_unconsumed_speaker_encoder_contract():
    for path in (
        BASE_PROFILE,
        ROOT / "configs/profiles/jetson-edgellm-v091-qwen3ttsbase-triple.json",
        BASE_N2_PROFILE,
    ):
        profile = json.loads(path.read_text())
        serialized = json.dumps(profile, sort_keys=True)
        assert "EDGE_LLM_TTS_SPK_ENCODER" not in serialized, path
        assert profile["env"]["EDGE_LLM_TTS_BASE_SPK_EMBED_PATH"].endswith(
            "/models/qwen3-tts-base/ref_embedding.b64.txt"
        ), path


def test_v091_compose_persists_voice_data_separately_from_models():
    compose = COMPOSE.read_text()
    assert "seeed-local-voice-data:/opt/seeed-local-voice/data" in compose
    assert "OVS_TTS_SPEAKERS_FILE: /opt/seeed-local-voice/data/speakers.json" in compose
    assert "SPARKTTS_VOICES_DIR: /opt/seeed-local-voice/data/sparktts_voices" in compose
    assert "name: seeed-local-voice-data" in compose
    assert "external: true" not in compose
    assert "speech-models:/opt/models" in compose


def test_v091_runtime_defaults_to_device_qualified_matcha_profile():
    dockerfile = DOCKERFILE.read_text()
    compose = COMPOSE.read_text()
    matcha_profile = json.loads(MATCHA_PROFILE.read_text())

    assert "OVS_PROFILE=jetson-edgellm-v091-matcha" in dockerfile
    assert (
        "OVS_PROFILE: ${OVS_PROFILE:-jetson-edgellm-v091-matcha}"
        in compose
    )
    assert matcha_profile["asr_model_id"] == "qwen3-asr"
    assert matcha_profile["tts_model_id"] == "matcha-icefall-zh-en"
    assert matcha_profile["execution_policy"]["cross_modal_overlap"] is True
    assert matcha_profile["max_concurrent_sessions"] == 2
    assert matcha_profile["env"]["OVS_TTS_SAMPLE_RATE"] == "16000"
    assert matcha_profile["env"]["OVS_TTS_STREAM_MAX_WORKERS"] == "2"

    resolved = resolve(
        profile=matcha_profile,
        policy=matcha_profile["execution_policy"],
        env={},
    )
    assert resolved.session_ceiling == 2
    assert resolved.coordinator_mode == "concurrent"
    assert BackendCoordinator(
        matcha_profile["execution_policy"], profile=matcha_profile
    ).mode == "concurrent"
