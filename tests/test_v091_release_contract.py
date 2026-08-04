from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "deploy/artifacts/v091-release-lock.json").read_text())
VOICE = yaml.safe_load(
    (ROOT / "deploy/docker-compose.edgellm-v091-voice.yml").read_text()
)
LLM = yaml.safe_load(
    (ROOT / "deploy/docker-compose.edgellm-v091-cutover.yml").read_text()
)
LLM_4K = yaml.safe_load(
    (ROOT / "deploy/docker-compose.edgellm-v091-cutover-4k.yml").read_text()
)


def test_qualified_8k_compose_matches_release_lock() -> None:
    env = LLM["services"]["edge-llm"]["environment"]
    locked = LOCK["model_artifacts"]["qwen3.5-4b-gdn-mtp-8k"]
    assert "runtime-20260804-v12" in LLM["services"]["edge-llm"]["image"]
    assert env["EDGELLM_MODEL_PROFILE"] == "qwen35-4b-gdn-mtp"
    assert env["EDGELLM_ENGINE_PROFILE"] == "8k"
    assert env["EDGELLM_EXPECTED_MAX_INPUT_LEN"] == "8192"
    assert env["EDGELLM_EXPECTED_MAX_KV_CACHE_CAPACITY"] == "8192"
    assert env["EDGELLM_ENGINE_REPO"] == locked["repo"]
    assert env["EDGELLM_ENGINE_REVISION"] == (
        "${EDGELLM_8K_ENGINE_REVISION:-__REPLACE_WITH_PUBLISHED_8K_REVISION__}"
    )
    assert locked["revision"] == "__REPLACE_WITH_PUBLISHED_8K_REVISION__"
    assert env["EDGELLM_EXPECTED_PAYLOAD_SHA256"] == locked["payload_sha256"]
    assert locked["payload_sha256"] == (
        "9208e46d61a4f1440ac68a312e35dde3d04b88edf0e4ee12b32210e7190d3325"
    )
    assert env["EDGELLM_SPECULATIVE_TOKEN_SLACK"] == "128"
    assert env["EDGELLM_SKIP_ENGINE_PROVENANCE_CHECK"] == "0"


def test_qualified_4k_compose_matches_release_lock_and_fails_closed_marker():
    env = LLM_4K["services"]["edge-llm"]["environment"]
    locked = LOCK["model_artifacts"]["qwen3.5-4b-gdn-mtp-4k"]
    assert env["EDGELLM_MODEL_PROFILE"] == "qwen35-4b-gdn-mtp"
    assert env["EDGELLM_ENGINE_PROFILE"] == "4k"
    assert env["EDGELLM_EXPECTED_MAX_INPUT_LEN"] == "4096"
    assert env["EDGELLM_EXPECTED_MAX_KV_CACHE_CAPACITY"] == "4096"
    assert env["EDGELLM_ENGINE_REPO"] == locked["repo"]
    assert env["EDGELLM_ENGINE_REVISION"] == (
        "${EDGELLM_4K_ENGINE_REVISION:-__REPLACE_WITH_PUBLISHED_4K_REVISION__}"
    )
    assert locked["revision"] == "__REPLACE_WITH_PUBLISHED_4K_REVISION__"
    assert env["EDGELLM_EXPECTED_PAYLOAD_SHA256"] == locked["payload_sha256"]
    assert locked["payload_sha256"] == (
        "06273e358a579590bb8344b451aa35c89983cd99401339fb1858d61af4dbd107"
    )
    assert env["EDGELLM_SPECULATIVE_TOKEN_SLACK"] == "128"
    assert env["EDGELLM_SKIP_ENGINE_PROVENANCE_CHECK"] == "0"
    assert "runtime-20260804-v12" in LLM_4K["services"]["edge-llm"]["image"]


def test_orin_services_use_model_level_caches_and_mirror() -> None:
    speech = VOICE["services"]["speech"]
    llm = LLM["services"]["edge-llm"]
    assert speech["environment"]["OVS_AUTO_DOWNLOAD_ARTIFACTS"] == "1"
    assert "hf-mirror.com" in speech["environment"]["HF_ENDPOINT"]
    assert "hf-mirror.com" in llm["environment"]["HF_ENDPOINT"]
    assert any("speech-models" in volume for volume in speech["volumes"])
    assert any("edge-llm-models-v091" in volume for volume in llm["volumes"])


def test_install_script_makes_orin_nx_v091_explicit_and_rollback_available() -> None:
    script = (ROOT / "deploy/install.sh").read_text()
    assert 'echo "orin-nx"' in script
    assert 'deploy/docker-compose.edgellm-v091-voice.yml' in script
    assert 'deploy/docker-compose.edgellm-v091-cutover.yml' in script
    assert 'deploy/install.sh --target jetson' in script
    assert 'deploy/verify-llm.sh' in script
