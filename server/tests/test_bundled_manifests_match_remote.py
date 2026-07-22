"""Catch bundled artifact manifests drifting from what the artifact repo serves.

``_load_manifest`` prefers the bundled copy over the remote one, so a stale
bundle does not merely miss an update -- it actively breaks a working install.
The download path replaces the destination before verifying, so a bundled md5
that no longer matches HF makes provisioning fetch the current file, reject it
against the old hash, and unlink it. That happened for real on an Orin NX:
``moss_tts_decode_step.plan`` was deleted off a healthy device.

These are network tests, skipped unless ``OVS_TEST_REMOTE_MANIFESTS=1``, so a
default run stays offline. Point CI at them after publishing artifacts.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "deploy" / "artifacts"

BUNDLED = [
    ("moss_manifest.json", "models/moss-tts-nano"),
    ("edgellm_v090_manifest.json", "models/edgellm-v090-asr"),
]

pytestmark = pytest.mark.skipif(
    os.environ.get("OVS_TEST_REMOTE_MANIFESTS") != "1",
    reason="set OVS_TEST_REMOTE_MANIFESTS=1 to check bundled manifests against HF",
)


def _remote(repo: str, prefix: str) -> dict:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{repo}/resolve/main/{prefix}/manifest.json"
    req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream-tests/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


@pytest.mark.parametrize("filename,prefix", BUNDLED, ids=lambda v: Path(str(v)).stem)
def test_bundled_hashes_match_the_artifact_repo(filename: str, prefix: str) -> None:
    bundled = json.loads((ARTIFACTS / filename).read_text())
    remote = _remote(bundled["hf_repo"], bundled.get("hf_prefix", prefix))

    remote_by_path = {f["path"]: f for f in remote["files"]}
    drifted = []
    for entry in bundled["files"]:
        other = remote_by_path.get(entry["path"])
        if other is None:
            continue  # bundle may carry entries the published set does not
        if (entry.get("size"), entry.get("md5")) != (other.get("size"), other.get("md5")):
            drifted.append(
                f"{entry['path']}: bundled {entry.get('md5')}/{entry.get('size')} "
                f"vs remote {other.get('md5')}/{other.get('size')}"
            )

    assert not drifted, (
        f"{filename} is stale against {bundled['hf_repo']}. Provisioning would "
        f"download the current file, fail it against the bundled hash, and delete "
        f"it:\n  " + "\n  ".join(drifted)
    )
