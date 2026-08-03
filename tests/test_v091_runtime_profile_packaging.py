from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "deploy/docker/Dockerfile.jetson.edgellm-v091-runtime"


def test_runtime_image_packages_every_v091_profile():
    dockerfile = DOCKERFILE.read_text()
    production_profiles = {
        "jetson-edgellm-v091-customvoice.json",
        "jetson-edgellm-v091-moss.json",
        "jetson-edgellm-v091-n2.json",
        "jetson-edgellm-v091-qwen3ttsbase-isolated-n2.json",
        "jetson-edgellm-v091-qwen3ttsbase.json",
        "jetson-edgellm-v091-sparktts.json",
    }

    for profile_name in production_profiles:
        expected = (
            f"COPY configs/profiles/{profile_name} "
            "/opt/speech/configs/profiles/"
        )
        assert expected in dockerfile, profile_name
