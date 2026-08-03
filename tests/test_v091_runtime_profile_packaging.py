import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "deploy/docker/Dockerfile.jetson.edgellm-v091-runtime"
COMPOSE = ROOT / "deploy/docker-compose.edgellm-v091-voice.yml"
RELEASE_LOCK = ROOT / "deploy/artifacts/v091-release-lock.json"
BASE_PROFILE = ROOT / "configs/profiles/jetson-edgellm-v091-qwen3ttsbase.json"
BASE_N2_PROFILE = ROOT / "configs/profiles/jetson-edgellm-v091-qwen3ttsbase-isolated-n2.json"


def test_runtime_image_packages_every_v091_profile():
    dockerfile = DOCKERFILE.read_text()
    production_profiles = {
        "jetson-edgellm-v091-customvoice.json",
        "jetson-edgellm-v091-moss.json",
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


def test_runtime_image_and_compose_pin_final_r5_artifact_and_pypi_mirror():
    artifact_set = "orin-nx-edgellm-v091-jp62-trt103-sm87-20260803-r5"
    dockerfile = DOCKERFILE.read_text()
    compose = COMPOSE.read_text()
    release_lock = json.loads(RELEASE_LOCK.read_text())

    assert artifact_set in dockerfile
    assert artifact_set in compose
    assert "seeed-local-voice:v0.9.1-edgellm-runtime-r12-20260803" in compose
    assert release_lock["artifact_set"] == artifact_set
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in dockerfile
    assert 'PIP_INDEX_URL="${PIP_INDEX_URL}" pip install' in dockerfile


def test_v091_base_profiles_cap_generation_to_the_512_frame_vocoder():
    for path in (BASE_PROFILE, BASE_N2_PROFILE):
        profile = json.loads(path.read_text())
        assert profile["env"]["TTS_MAX_AUDIO_LENGTH"] == "512", path
        assert profile["env"]["EDGE_LLM_TTS_CODE2WAV_DIR"].endswith(
            "/tts_base_code2wav_512"
        ), path
