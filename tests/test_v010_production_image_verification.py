from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = json.loads(
    (ROOT / "deploy/artifacts/v010-production-image-verification.json").read_text()
)
LOCK = json.loads(
    (ROOT / "deploy/artifacts/v010-build-ready-release-lock.json").read_text()
)
SHA256 = re.compile(r"(?:sha256:)?[0-9a-f]{64}")


def test_pre_push_image_evidence_matches_build_ready_identity() -> None:
    assert EVIDENCE["status"] == "qualified_pre_push"
    assert EVIDENCE["artifact_set"] == LOCK["artifact_set"]
    assert EVIDENCE["embedded_gate"] == "require-image-build-ready"
    assert EVIDENCE["service_revision"] == LOCK["runtime_images"]["speech"][
        "service_revision"
    ]
    for key, image in EVIDENCE["images"].items():
        assert image["targets"] == LOCK["runtime_images"][key]["target_ids"]
        assert SHA256.fullmatch(image["image_id"])
        assert image["size_bytes"] > 0
        assert image["labels"]["org.opencontainers.image.revision"] == EVIDENCE[
            "service_revision"
        ]
        assert image["labels"]["com.seeed.release-gate"] == EVIDENCE[
            "embedded_gate"
        ]


def test_speech_image_is_byte_identical_across_nx_and_nano() -> None:
    speech = EVIDENCE["images"]["speech"]
    transfer = EVIDENCE["device_transfer"]
    assert speech["target_image_ids_match"] is True
    assert transfer["loaded_image_id_matches_source"] is True
    assert transfer["transport"] == "tailscale_direct_lan_stream"
    assert transfer["temporary_server_removed"] is True
    assert SHA256.fullmatch(transfer["archive_sha256"])


def test_every_final_runtime_smoke_passed_and_uses_locked_revision() -> None:
    for target, lanes in EVIDENCE["runtime_smokes"].items():
        for lane, result in lanes.items():
            assert result["passed"] is True, (target, lane)
            assert result["http_status"] == 200, (target, lane)
            assert re.fullmatch(r"[0-9a-f]{40}", result["revision"])
    assert EVIDENCE["runtime_smokes"]["orin-nano"]["tts_customvoice"][
        "deterministic_replay"
    ] is True


def test_qwen_continuous_batching_is_not_overclaimed() -> None:
    qwen = EVIDENCE["runtime_smokes"]["orin-nx-16gb"]["qwen3.5-8k"]
    assert qwen["speculative_decoding"] is True
    assert qwen["continuous_batching"] is False
    assert qwen["engine_max_batch_size"] == 1


def test_all_frozen_no_regression_gates_pass() -> None:
    gates = EVIDENCE["no_regression_gates"]
    assert gates
    for gate in gates.values():
        assert gate["passed"] is True
        assert SHA256.fullmatch(gate["evidence_sha256"])
