"""Regression guards for image-level defaults that are not app presets."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "dockerfile",
    [
        Path(__file__).parents[1] / "Dockerfile",
        Path(__file__).parents[2] / "deploy/docker/Dockerfile.rk.agent",
    ],
)
def test_agent_images_do_not_enable_server_loop(dockerfile: Path):
    text = dockerfile.read_text(encoding="utf-8")
    assert "OVS_AGENT_SERVER_LOOP=1" not in text
    assert "OVS_AGENT_SERVER_LOOP=0" in text
