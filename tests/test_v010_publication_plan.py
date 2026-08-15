from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "deploy/artifacts/v010-publication-plan.json").read_text())
LOCK = json.loads(
    (ROOT / "deploy/artifacts/v010-candidate-release-lock.json").read_text()
)
SHA256 = re.compile(r"[0-9a-f]{64}")


def _locked_identity(model_id: str, target_id: str) -> dict:
    artifact = LOCK["model_artifacts"][model_id]
    variants = artifact.get("target_variants")
    return variants[target_id] if variants is not None else artifact


def test_publication_plan_covers_every_new_target_payload_once() -> None:
    expected = set()
    for target_id, target in LOCK["targets"].items():
        for lane in target["supported_lanes"]:
            model_id = lane.split("/", 1)[1]
            if model_id != "matcha-icefall-zh-en":
                expected.add(f"{model_id}/{target_id}")
    assert set(PLAN["packages"]) == expected
    assert PLAN["external_upload_authorized"] is False


def test_staged_packages_match_release_lock_and_have_complete_identity() -> None:
    staged = 0
    for key, package in PLAN["packages"].items():
        model_id, target_id = key.rsplit("/", 1)
        locked = _locked_identity(model_id, target_id)
        assert package["repo"] == LOCK["model_artifacts"][model_id]["repo"]
        assert package["proposed_branch"].startswith("v010-")
        assert package["published_revision"] is None
        if package["status"] == "staged_verified":
            staged += 1
            assert package["staging_uri"].startswith("fleet://spark/")
            assert package["payload_sha256"] == locked["payload_sha256"]
            assert package["payload_size"] == locked["payload_size"]
            assert SHA256.fullmatch(package["manifest_sha256"])
            assert SHA256.fullmatch(package["sums_sha256"])
        else:
            assert package["status"] == "blocked_wsl_offline"
            for field in (
                "staging_uri",
                "payload_sha256",
                "payload_size",
                "manifest_sha256",
                "sums_sha256",
            ):
                assert package[field] is None
    assert staged == 10


def test_proposed_branches_are_unique_within_each_repo() -> None:
    refs = [
        (package["repo"], package["proposed_branch"])
        for package in PLAN["packages"].values()
    ]
    assert len(refs) == len(set(refs))
