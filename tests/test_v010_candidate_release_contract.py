from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "deploy/artifacts/v010-candidate-release-lock.json"
LOCK = json.loads(LOCK_PATH.read_text())
V091_LOCK = json.loads((ROOT / "deploy/artifacts/v091-release-lock.json").read_text())
PROFILES = tuple(sorted((ROOT / "configs/profiles").glob("jetson-edgellm-v010-candidate-*.json")))
VOICE_COMPOSE_PATH = ROOT / "deploy/docker-compose.edgellm-v010-candidate-voice.yml"
LLM_COMPOSE_PATH = ROOT / "deploy/docker-compose.edgellm-v010-candidate-llm.yml"
DOCKERFILE = ROOT / "deploy/docker/Dockerfile.jetson.edgellm-v010-candidate-runtime"
CHECKER = ROOT / "scripts/check_edgellm_v010_release_lock.py"
NX_BASELINE = ROOT / "bench/perf/baselines/edgellm-v091-orin-nx-qualification.json"
NANO_BASELINE = ROOT / "bench/perf/baselines/edgellm-v091-orin-nano-qualification.json"


def _run_checker(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--lock",
            str(LOCK_PATH),
            "--repo-root",
            str(ROOT),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_v010_candidate_is_additive_and_v091_remains_rollback_identity() -> None:
    assert LOCK["release_state"] == "candidate_unpublished"
    assert LOCK["deployable"] is False
    assert LOCK["rollback_release_lock"] == "deploy/artifacts/v091-release-lock.json"
    assert V091_LOCK["release_state"] == "published_and_qualified"
    assert V091_LOCK["artifact_set"].startswith("orin-nx-edgellm-v091-")
    assert "runtime-20260804-v13" in V091_LOCK["runtime_image"]["ref"]


def test_candidate_source_is_bound_to_gitlink_and_overlay_manifests() -> None:
    assert LOCK["source"]["upstream_sha"] == "71dd1bae032e70771265917ec74d3ff4cad07a10"
    gitlink = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--stage", "third_party/jetson-voice-engine"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert gitlink[:2] == ["160000", LOCK["source"]["submodule_sha"]]
    result = _run_checker()
    assert result.returncode == 0, result.stderr
    assert '"release_state": "candidate_unpublished"' in result.stdout


def test_candidate_cannot_pass_the_published_release_gate() -> None:
    result = _run_checker("--require-published")
    assert result.returncode != 0
    assert "still candidate_unpublished" in result.stderr


def test_unpublished_model_and_runtime_identities_are_explicit_nulls() -> None:
    unpublished_models = [
        artifact
        for artifact in LOCK["model_artifacts"].values()
        if artifact["status"].startswith("candidate_")
    ]
    assert unpublished_models
    for artifact in unpublished_models:
        assert artifact["revision"] is None
        assert artifact["payload_sha256"] is None
        assert artifact["payload_size"] is None
    for artifact in LOCK["runtime_artifacts"].values():
        assert artifact == {
            "sha256": None,
            "size": None,
            "status": "candidate_not_staged",
        }
    assert set(LOCK["runtime_images"]) == {"speech", "llm"}
    for image in LOCK["runtime_images"].values():
        assert all(image[key] is None for key in (
            "ref", "image_id", "registry_digest", "size_bytes", "service_revision"
        ))


def test_candidate_profiles_are_isolated_and_fail_closed_on_asr_revision() -> None:
    assert {path.stem for path in PROFILES} == {
        "jetson-edgellm-v010-candidate-asr",
        "jetson-edgellm-v010-candidate-matcha",
    }
    for path in PROFILES:
        profile = json.loads(path.read_text())
        serialized = json.dumps(profile, sort_keys=True)
        assert profile["name"] == path.stem
        assert profile["artifact_set"] == "edgellm-v010-candidate"
        assert profile["deployment_scope"] == "gray-only-unpublished"
        assert "/opt/edgellm-v091" not in serialized
        assert "/opt/models/" not in serialized
        assert "/opt/edgellm-v010/" in serialized
        asr = next(item for item in profile["model_artifacts"] if item["canonical_model_id"] == "qwen3-asr-0.6b")
        assert asr["revision"] == "UNPUBLISHED_V010_REVISION"
        assert asr["repo"] == LOCK["model_artifacts"]["qwen3-asr-0.6b"]["repo"]


def test_candidate_compose_uses_separate_namespaces_and_double_opt_in() -> None:
    compose = yaml.safe_load(VOICE_COMPOSE_PATH.read_text())
    service = compose["services"]["speech-v010-candidate"]
    assert service["restart"] == "no"
    assert service["build"]["args"]["V010_ALLOW_UNPUBLISHED_CANDIDATE"] == "${V010_ALLOW_UNPUBLISHED_CANDIDATE:-0}"
    assert service["environment"]["EDGELLM_V010_ALLOW_UNPUBLISHED_CANDIDATE"] == "${V010_ALLOW_UNPUBLISHED_CANDIDATE:-0}"
    assert service["environment"]["OVS_AUTO_DOWNLOAD_ARTIFACTS"] == "0"
    assert any("speech-models-v010-candidate" in volume for volume in service["volumes"])
    assert "speech-models-v091" not in VOICE_COMPOSE_PATH.read_text()


def test_candidate_llm_compose_requires_all_unpublished_identities() -> None:
    text = LLM_COMPOSE_PATH.read_text()
    compose = yaml.safe_load(text)
    service = compose["services"]["edge-llm-v010-candidate"]
    assert service["restart"] == "no"
    assert "EDGE_LLM_V010_CANDIDATE_IMAGE:?" in text
    assert "EDGELLM_V010_ENGINE_REVISION:?" in text
    assert "EDGELLM_V010_EXPECTED_PAYLOAD_SHA256:?" in text
    assert service["environment"]["EDGELLM_SKIP_ENGINE_PROVENANCE_CHECK"] == "0"
    assert "edge-llm-models-v091" not in text


def test_candidate_dockerfile_cannot_silently_inherit_v091_runtime_binaries() -> None:
    text = DOCKERFILE.read_text()
    assert "v010-candidate-release-gate/" in text
    assert "v091-release-gate/" not in text
    assert "CANDIDATE-SHA256SUMS" in text
    assert "--require-published" in text
    assert "V010_ALLOW_UNPUBLISHED_CANDIDATE=0" in text
    assert LOCK["source"]["submodule_sha"] in text


def test_v091_no_regression_baselines_are_release_scoped_and_fail_closed() -> None:
    nx = json.loads(NX_BASELINE.read_text())
    assert nx["release"] == "TensorRT-Edge-LLM v0.9.1"
    assert nx["comparison_policy"]["missing_metric"] == "fail"
    assert nx["lanes"]["qwen3_asr_int4_b2_1024_1536"][
        "streaming_eos_to_final_p95_ms"
    ] == 117.21
    assert nx["lanes"]["qwen35_4b_gdn_mtp_8k"][
        "throughput_tokens_per_s"
    ] == 48.2558

    nano = json.loads(NANO_BASELINE.read_text())
    assert nano["release_baseline_status"] == "pending_capture"
    assert nano["missing_metric"] == "fail"
