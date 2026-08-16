from __future__ import annotations

import json
import hashlib
import re
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
PRODUCTION_DOCKERFILE = ROOT / "deploy/docker/Dockerfile.jetson.edgellm-v010-production-runtime"
PRODUCTION_COMPOSE = ROOT / "deploy/docker-compose.edgellm-v010-production-voice.yml"
PRODUCTION_LLM_DOCKERFILE = ROOT / "deploy/docker/Dockerfile.jetson.edgellm-v010-production-llm-runtime"
PRODUCTION_LLM_COMPOSE = ROOT / "deploy/docker-compose.edgellm-v010-production-llm.yml"
CHECKER = ROOT / "scripts/check_edgellm_v010_release_lock.py"
NX_BASELINE = ROOT / "bench/perf/baselines/edgellm-v091-orin-nx-qualification.json"
NANO_BASELINE = ROOT / "bench/perf/baselines/edgellm-v091-orin-nano-qualification.json"


def _run_checker(
    *extra: str, lock_path: Path = LOCK_PATH
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--lock",
            str(lock_path),
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
    assert LOCK["gate_policy"] == {
        "candidate_gate": "require-candidate",
        "embedded_image_gate": "require-image-build-ready",
        "external_publication_gate": "require-published",
        "image_build_state": "artifacts_qualified_image_pending",
        "image_self_identity_fields": [
            "ref",
            "image_id",
            "registry_digest",
            "size_bytes",
        ],
    }


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
    assert "not published_and_qualified" in result.stderr


def _build_ready_lock(runtime_root: Path) -> tuple[dict, str]:
    lock = json.loads(json.dumps(LOCK))
    revision = "a" * 40
    lock["release_state"] = "artifacts_qualified_image_pending"
    lock["deployable"] = False
    lock["source"]["status"] = "published"
    for artifact in lock["model_artifacts"].values():
        identities = artifact.get("target_variants") or {"shared": artifact}
        for identity in identities.values():
            if identity["status"] != "published_retained_v091":
                identity.update(
                    revision="b" * 40,
                    payload_sha256="c" * 64,
                    payload_size=1,
                    status="published",
                )
    for index, (artifact_key, artifact) in enumerate(
        lock["runtime_artifacts"].items()
    ):
        relative = artifact["path"]
        payload = f"runtime-{index}".encode()
        artifact.update(
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
            status="qualified_for_image",
        )
        if artifact_key.startswith("speech/"):
            path = runtime_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            if relative.startswith("bin/"):
                path.chmod(0o755)
    for target in lock["targets"].values():
        target["qualification_status"] = "passed"
    for image in lock["runtime_images"].values():
        image.update(
            ref=None,
            image_id=None,
            registry_digest=None,
            size_bytes=None,
            service_revision=revision,
            status="build_pending",
        )
    return lock, revision


def test_two_phase_gate_breaks_image_digest_cycle_without_weakening_inputs(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    lock, revision = _build_ready_lock(runtime_root)
    path = tmp_path / "build-ready.json"
    path.write_text(json.dumps(lock))

    build = _run_checker(
        "--require-image-build-ready",
        "--runtime-root",
        str(runtime_root),
        "--image-key",
        "speech",
        "--expected-service-revision",
        revision,
        lock_path=path,
    )
    assert build.returncode == 0, build.stderr

    external_before_push = _run_checker("--require-published", lock_path=path)
    assert external_before_push.returncode != 0
    assert "not published_and_qualified" in external_before_push.stderr

    lock["release_state"] = "published_and_qualified"
    lock["deployable"] = True
    for artifact in lock["runtime_artifacts"].values():
        artifact["status"] = "published_in_image"
    for index, image in enumerate(lock["runtime_images"].values()):
        digest = f"{index + 1:x}" * 64
        image.update(
            ref=f"registry.example/edgellm/image-{index}@sha256:{digest}",
            image_id="sha256:" + "e" * 64,
            registry_digest="sha256:" + digest,
            size_bytes=1024,
            status="published",
        )
    path.write_text(json.dumps(lock))
    external = _run_checker("--require-published", lock_path=path)
    assert external.returncode == 0, external.stderr


def test_build_ready_gate_rejects_prepopulated_self_identity(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    lock, _ = _build_ready_lock(runtime_root)
    lock["runtime_images"]["speech"]["registry_digest"] = "sha256:" + "d" * 64
    path = tmp_path / "self-referential.json"
    path.write_text(json.dumps(lock))

    result = _run_checker("--require-image-build-ready", lock_path=path)
    assert result.returncode != 0
    assert "must remain null at image build" in result.stderr


def test_published_gate_rejects_tag_only_image_ref(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    lock, _ = _build_ready_lock(runtime_root)
    lock["release_state"] = "published_and_qualified"
    lock["deployable"] = True
    for artifact in lock["runtime_artifacts"].values():
        artifact["status"] = "published_in_image"
    for image in lock["runtime_images"].values():
        image.update(
            ref="registry.example/edgellm:v0.10",
            image_id="sha256:" + "e" * 64,
            registry_digest="sha256:" + "d" * 64,
            size_bytes=1024,
            status="published",
        )
    path = tmp_path / "tag-only-image.json"
    path.write_text(json.dumps(lock))

    result = _run_checker("--require-published", lock_path=path)
    assert result.returncode != 0
    assert "ref must be pinned to registry_digest" in result.stderr


def test_build_ready_gate_rejects_unpublished_runtime_artifact(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    lock, _ = _build_ready_lock(runtime_root)
    lock["runtime_artifacts"]["speech/qwen3-asr-worker"]["status"] = (
        "candidate_not_staged"
    )
    path = tmp_path / "unpublished-runtime.json"
    path.write_text(json.dumps(lock))

    result = _run_checker("--require-image-build-ready", lock_path=path)
    assert result.returncode != 0
    assert "status is not qualified_for_image" in result.stderr


def test_build_ready_gate_rejects_unpublished_source(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    lock, _ = _build_ready_lock(runtime_root)
    lock["source"]["status"] = "candidate_pinned"
    path = tmp_path / "unpublished-source.json"
    path.write_text(json.dumps(lock))

    result = _run_checker("--require-image-build-ready", lock_path=path)
    assert result.returncode != 0
    assert "source.status is not published" in result.stderr


def test_build_ready_gate_rejects_runtime_bytes_not_matching_lock(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    lock, _ = _build_ready_lock(runtime_root)
    path = tmp_path / "runtime-mismatch.json"
    path.write_text(json.dumps(lock))
    (runtime_root / "bin/qwen3_asr_worker").write_bytes(b"tampered")

    result = _run_checker(
        "--require-image-build-ready",
        "--runtime-root",
        str(runtime_root),
        "--image-key",
        "speech",
        lock_path=path,
    )
    assert result.returncode != 0
    assert "runtime artifact size mismatch" in result.stderr


def test_unpublished_model_and_runtime_identities_are_explicit_nulls() -> None:
    unpublished_models = []
    for artifact in LOCK["model_artifacts"].values():
        identities = artifact.get("target_variants") or {"shared": artifact}
        for identity in identities.values():
            if identity["status"].startswith("candidate_"):
                unpublished_models.append(identity)
                assert identity["revision"] is None
                if identity["status"] not in {"candidate_packaged_not_published"}:
                    assert identity["payload_sha256"] is None
                    assert identity["payload_size"] is None
    assert unpublished_models
    for artifact in LOCK["runtime_artifacts"].values():
        assert artifact["status"] == "candidate_staged_qualified"
        assert len(artifact["sha256"]) == 64
        assert artifact["size"] > 0
        assert artifact["path"]
    assert set(LOCK["runtime_images"]) == {"speech", "llm"}
    for image in LOCK["runtime_images"].values():
        assert all(image[key] is None for key in (
            "ref", "image_id", "registry_digest", "size_bytes", "service_revision"
        ))


def test_target_scope_keeps_unqualified_plan_lanes_off_orin_nano() -> None:
    speech_lanes = {
        "speech/qwen3-asr-0.6b",
        "speech/qwen3-tts-0.6b-base",
        "speech/qwen3-tts-0.6b-customvoice",
        "speech/moss-tts-nano",
        "speech/matcha-icefall-zh-en",
        "speech/sparktts-0.5b",
    }
    llm_lanes = {
        "llm/qwen3.5-4b-gdn-mtp-4k",
        "llm/qwen3.5-4b-gdn-mtp-8k",
    }
    nano_speech_lanes = {
        "speech/qwen3-asr-0.6b",
        "speech/qwen3-tts-0.6b-base",
        "speech/qwen3-tts-0.6b-customvoice",
        "speech/matcha-icefall-zh-en",
    }

    nx = LOCK["targets"]["orin-nx-16gb"]
    assert nx["deployment_scope"] == "full_speech_and_llm"
    assert set(nx["supported_lanes"]) == speech_lanes | llm_lanes
    assert nx["excluded_lanes"] == []

    nano = LOCK["targets"]["orin-nano"]
    assert nano["deployment_scope"] == "speech_only"
    assert set(nano["supported_lanes"]) == nano_speech_lanes
    assert {item["lane"] for item in nano["excluded_lanes"]} == (
        (speech_lanes - nano_speech_lanes) | llm_lanes
    )
    assert all(item["reason"].strip() for item in nano["excluded_lanes"])
    assert not any(lane.startswith("llm/") for lane in nano["supported_lanes"])

    assert LOCK["runtime_images"]["speech"]["target_ids"] == [
        "orin-nx-16gb",
        "orin-nano",
    ]
    assert LOCK["runtime_images"]["llm"]["target_ids"] == ["orin-nx-16gb"]
    for model_id in (
        "qwen3-asr-0.6b",
        "qwen3-tts-0.6b-base",
        "qwen3-tts-0.6b-customvoice",
    ):
        assert set(LOCK["model_artifacts"][model_id]["target_variants"]) == {
            "orin-nx-16gb",
            "orin-nano",
        }
    assert LOCK["model_artifacts"]["matcha-icefall-zh-en"]["portability"] == (
        "onnx-runtime-cross-target"
    )


def test_passed_targets_bind_the_frozen_hardware_evidence_report() -> None:
    expected_report = (
        "third_party/jetson-voice-engine/engine-overlay-v010/"
        "VALIDATION-20260814.md"
    )
    expected_sha = hashlib.sha256((ROOT / expected_report).read_bytes()).hexdigest()
    for target in LOCK["targets"].values():
        evidence = target["qualification_evidence"]
        assert evidence["report_path"] == expected_report
        assert evidence["report_sha256"] == expected_sha
        assert evidence["gates"]
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in evidence["gates"].values()
        )

    nx_gates = LOCK["targets"]["orin-nx-16gb"]["qualification_evidence"]["gates"]
    assert {
        "asr_b2_isolation",
        "tts_base_n2_cancel_recovery",
        "tts_base_no_regression",
        "tts_customvoice_n2_cancel_recovery",
        "qwen35_4k_no_regression",
        "qwen35_8k_no_regression",
        "qwen35_abort_recovery",
        "qwen35_tts_base_co_residency",
        "moss_n2_cancel_recovery",
        "moss_no_regression",
        "spark_n2_cancel_recovery_soak",
    } == set(nx_gates)


def test_checker_rejects_qualification_report_hash_drift(tmp_path: Path) -> None:
    mutated = json.loads(json.dumps(LOCK))
    mutated["targets"]["orin-nx-16gb"]["qualification_evidence"][
        "report_sha256"
    ] = "0" * 64
    path = tmp_path / "bad-qualification-report.json"
    path.write_text(json.dumps(mutated))

    result = _run_checker(lock_path=path)
    assert result.returncode != 0
    assert "qualification_evidence report hash differs" in result.stderr


def test_checker_rejects_qwen35_supported_on_nano(tmp_path: Path) -> None:
    mutated = json.loads(json.dumps(LOCK))
    mutated["targets"]["orin-nano"]["supported_lanes"].append(
        "llm/qwen3.5-4b-gdn-mtp-4k"
    )
    path = tmp_path / "bad-target-lock.json"
    path.write_text(json.dumps(mutated))

    result = _run_checker(lock_path=path)
    assert result.returncode != 0
    assert "cannot both support and exclude" in result.stderr


def test_checker_rejects_nx_built_moss_as_nano_supported(tmp_path: Path) -> None:
    mutated = json.loads(json.dumps(LOCK))
    nano = mutated["targets"]["orin-nano"]
    nano["supported_lanes"].append("speech/moss-tts-nano")
    nano["excluded_lanes"] = [
        item
        for item in nano["excluded_lanes"]
        if item["lane"] != "speech/moss-tts-nano"
    ]
    path = tmp_path / "bad-nano-moss-lock.json"
    path.write_text(json.dumps(mutated))

    result = _run_checker(lock_path=path)
    assert result.returncode != 0
    assert "only Nano-qualified speech lanes" in result.stderr


def test_checker_rejects_missing_target_specific_tts_variant(tmp_path: Path) -> None:
    mutated = json.loads(json.dumps(LOCK))
    del mutated["model_artifacts"]["qwen3-tts-0.6b-base"]["target_variants"][
        "orin-nano"
    ]
    path = tmp_path / "missing-target-variant.json"
    path.write_text(json.dumps(mutated))

    result = _run_checker(lock_path=path)
    assert result.returncode != 0
    assert "target_variants must match supported target scope" in result.stderr


def test_checker_rejects_llm_image_targeting_nano(tmp_path: Path) -> None:
    mutated = json.loads(json.dumps(LOCK))
    mutated["runtime_images"]["llm"]["target_ids"].append("orin-nano")
    path = tmp_path / "bad-image-scope-lock.json"
    path.write_text(json.dumps(mutated))

    result = _run_checker(lock_path=path)
    assert result.returncode != 0
    assert "llm image must target Orin NX only" in result.stderr


def test_candidate_profiles_are_isolated_and_fail_closed_on_asr_revision() -> None:
    assert {path.stem for path in PROFILES} == {
        "jetson-edgellm-v010-candidate-asr",
        "jetson-edgellm-v010-candidate-customvoice",
        "jetson-edgellm-v010-candidate-matcha",
        "jetson-edgellm-v010-candidate-moss",
        "jetson-edgellm-v010-candidate-qwen3ttsbase",
        "jetson-edgellm-v010-candidate-sparktts",
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
        candidate_artifacts = [
            item for item in profile["model_artifacts"]
            if item["revision"] == "UNPUBLISHED_V010_REVISION"
        ]
        assert candidate_artifacts
        for artifact in candidate_artifacts:
            locked = LOCK["model_artifacts"][artifact["canonical_model_id"]]
            assert artifact["repo"] == locked["repo"]
        if profile["asr_backend"] is not None:
            asr = next(
                item for item in profile["model_artifacts"]
                if item["canonical_model_id"] == "qwen3-asr-0.6b"
            )
            assert asr["repo"] == LOCK["model_artifacts"]["qwen3-asr-0.6b"]["repo"]

    spark = json.loads(
        (ROOT / "configs/profiles/jetson-edgellm-v010-candidate-sparktts.json").read_text()
    )
    spark_llm = next(
        item for item in spark["required_engines"]
        if item["env_var"] == "SPARKTTS_LLM_ENGINE_DIR"
    )
    assert spark_llm["engine_path"].endswith("/llm.engine")
    assert spark_llm["env_path"] == spark["env"]["SPARKTTS_LLM_ENGINE_DIR"]


def test_candidate_compose_uses_separate_namespaces_and_double_opt_in() -> None:
    compose = yaml.safe_load(VOICE_COMPOSE_PATH.read_text())
    service = compose["services"]["speech-v010-candidate"]
    assert service["restart"] == "no"
    assert service["build"]["args"]["V010_ALLOW_UNPUBLISHED_CANDIDATE"] == "${V010_ALLOW_UNPUBLISHED_CANDIDATE:-0}"
    assert service["environment"]["EDGELLM_V010_ALLOW_UNPUBLISHED_CANDIDATE"] == "${V010_ALLOW_UNPUBLISHED_CANDIDATE:-0}"
    assert service["environment"]["OVS_AUTO_DOWNLOAD_ARTIFACTS"] == "0"
    assert service["environment"]["LANGUAGE_MODE"] == "multilanguage"
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
    assert service["environment"]["EDGELLM_MTP_ENABLED"] == "1"
    assert service["environment"]["EDGELLM_DRAFT_TOP_K"] == "1"
    assert service["environment"]["EDGELLM_DRAFT_STEP"] == "3"
    assert service["environment"]["EDGELLM_VERIFY_TREE_SIZE"] == "4"
    assert service["environment"]["EDGELLM_SPEC_DECODE_ENGINE_DIR"] == (
        service["environment"]["EDGELLM_ENGINE_DIR"]
    )
    assert service["environment"]["EDGELLM_EXPECTED_TENSORRT"] == "10.3.0.30"
    assert "edge-llm-models-v091" not in text


def test_candidate_llm_image_moves_native_runtime_as_one_unit() -> None:
    dockerfile = (
        ROOT / "deploy/docker/Dockerfile.jetson.edgellm-v010-candidate-llm-runtime"
    ).read_text(encoding="utf-8")
    manifest = json.loads(
        (ROOT / "deploy/artifacts/v010-candidate-llm-runtime-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "COPY experimental/server/" in dockerfile
    assert "COPY runtime/_edgellm_runtime" in dockerfile
    assert "COPY runtime/libNvInfer_edgellm_plugin" in dockerfile
    assert "python-multipart==0.0.20" in dockerfile
    assert "UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple" in dockerfile
    assert "pip install --system" in dockerfile
    assert "python3 -m pip" not in dockerfile
    assert manifest["python_multipart_version"] == "0.0.20"
    assert manifest["trt_edge_llm_commit"] == LOCK["source"]["upstream_sha"]


def test_candidate_dockerfile_cannot_silently_inherit_v091_runtime_binaries() -> None:
    text = DOCKERFILE.read_text()
    assert "v010-candidate-release-gate/" in text
    assert "v091-release-gate/" not in text
    assert "CANDIDATE-SHA256SUMS" in text
    assert "--require-candidate" in text
    assert "--require-published" not in text
    assert "--require-image-build-ready" not in text
    assert "V010_ALLOW_UNPUBLISHED_CANDIDATE=0" in text
    assert "LANGUAGE_MODE=multilanguage" in text
    assert LOCK["source"]["submodule_sha"] in text
    for worker in (
        "qwen3_asr_worker",
        "qwen3_tts_streaming_worker",
        "moss_tts_nano_worker",
        "spark_tts_worker",
    ):
        assert f"test -x bin/{worker}" in text
    assert "libonnxruntime.so.1.20.0" in text
    assert "moss_tts_nano_worker_v010" in text
    wrapper = (ROOT / "scripts/run_moss_tts_v010.sh").read_text()
    assert "MOSS_ORT_ROOT=/opt/edgellm-v010/moss-runtime" in wrapper
    assert "exec /opt/edgellm-v010/bin/moss_tts_nano_worker" in wrapper
    candidate_entrypoint = (
        ROOT / "scripts/start_edgellm_v010_candidate_runtime.py"
    ).read_text()
    assert "--require-candidate" in candidate_entrypoint
    assert "--require-published" not in candidate_entrypoint
    assert "--require-image-build-ready" not in candidate_entrypoint


def test_production_image_uses_build_ready_gate_without_candidate_override() -> None:
    dockerfile = PRODUCTION_DOCKERFILE.read_text()
    entrypoint = (
        ROOT / "scripts/start_edgellm_v010_production_runtime.py"
    ).read_text()
    compose = PRODUCTION_COMPOSE.read_text()

    assert "--require-image-build-ready" in dockerfile
    assert "--runtime-root /opt/edgellm-v010" in dockerfile
    assert "COPY deploy/artifacts/v010-build-ready-release-lock.json" in dockerfile
    assert "COPY deploy/artifacts/v010-candidate-release-lock.json" not in dockerfile
    assert "v010-embedded-build-lock.json" in dockerfile
    assert "VALIDATION-20260814.md" in dockerfile
    assert "render_edgellm_v010_production_profiles.py" in dockerfile
    assert "--output-dir /tmp/v010-production-profiles" in dockerfile
    assert "rm -f /opt/speech/configs/profiles/jetson-edgellm-v010-candidate-*.json" in dockerfile
    assert "V010_ALLOW_UNPUBLISHED_CANDIDATE" not in dockerfile
    assert "--require-published" not in dockerfile
    assert "--require-image-build-ready" in entrypoint
    assert "--expected-service-revision" in entrypoint
    assert "candidate" in entrypoint
    assert (
        "SPEECH_V010_IMAGE:-sensecraft-missionpack.seeed.cn/solution/"
        "seeed-local-voice@sha256:df1b000f3142f3fd9876bf3f4380fef0a1a48861d886a4e96e52c3b419bedaec"
        in compose
    )
    assert "OVS_V010_PROFILE:?" in compose
    assert "models-v010-candidate" not in compose
    assert "v010-candidate" not in compose
    assert "build:" not in compose


def test_production_llm_image_uses_same_two_phase_gate() -> None:
    dockerfile = PRODUCTION_LLM_DOCKERFILE.read_text()
    entrypoint = (
        ROOT / "scripts/start_edgellm_v010_production_llm_runtime.py"
    ).read_text()
    compose = PRODUCTION_LLM_COMPOSE.read_text()

    assert "--require-image-build-ready" in dockerfile
    assert "--runtime-root /opt/edgellm-v010" in dockerfile
    assert "--image-key llm" in dockerfile
    assert "VALIDATION-20260814.md" in dockerfile
    assert (
        'io.seeed.tensorrt-edge-llm.revision="71dd1bae032e70771265917ec74d3ff4cad07a10"'
        in dockerfile
    )
    assert "V010_ALLOW_UNPUBLISHED_CANDIDATE" not in dockerfile
    assert "--require-published" not in dockerfile
    assert "--require-image-build-ready" in entrypoint
    assert '"llm"' in entrypoint
    assert (
        "EDGE_LLM_V010_IMAGE:-sensecraft-missionpack.seeed.cn/solution/"
        "edge-llm-chat-service@sha256:0101c04fffdc3801575da9856525cd1b92b4066449c2cd96cb980c07dae87f72"
        in compose
    )
    assert "EDGELLM_V010_ENGINE_REVISION:-21e50311cb9809445f3714d9925c5e69618a41d7" in compose
    assert (
        "EDGELLM_V010_EXPECTED_PAYLOAD_SHA256:-"
        "83964439c8d309e330a0ebcdb586694c92b55f36f77de646547c28def7da2138"
        in compose
    )
    assert "models-v010-candidate" not in compose
    assert "build:" not in compose


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
    assert nano["release_baseline_status"] == "qualified_with_first_native_platform_lanes"
    assert nano["missing_metric"] == "fail"
    assert nano["lanes"]["qwen3_asr_int4_b2_1024_1536"][
        "v010_no_regression_passed"
    ] is True
    assert nano["remaining_lanes"] == []
    assert nano["lanes"]["qwen3_tts_base_int4_isolated"][
        "first_native_platform_gate_passed"
    ] is True
    assert nano["lanes"]["qwen3_tts_customvoice_int4_isolated"][
        "first_native_platform_gate_passed"
    ] is True
