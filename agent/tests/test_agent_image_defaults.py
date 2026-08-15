"""Regression guards for image-level defaults that are not app presets."""

from pathlib import Path


def test_generic_agent_image_does_not_enable_server_loop():
    dockerfile = Path(__file__).parents[1] / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    assert "OVS_AGENT_SERVER_LOOP=1" not in text
    assert "OVS_AGENT_SERVER_LOOP=0" in text
