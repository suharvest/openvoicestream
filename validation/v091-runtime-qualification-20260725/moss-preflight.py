#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
from pathlib import Path

from server.core import profile_loader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


profile = profile_loader.apply_profile_from_env()
missing = profile_loader.find_missing_artifacts(profile)
env_keys = sorted(
    key
    for key in (profile.get("env") or {})
    if key.startswith(("MOSS_", "EDGE_LLM_ASR_", "EDGELLM_PLUGIN"))
)
resolved_env = {key: os.environ.get(key) for key in env_keys}

inventory = []
root = Path("/opt/edgellm-v091/engines/moss")
for path in sorted(root.rglob("*")):
    stat = path.lstat()
    record = {
        "path": str(path),
        "mode": oct(stat.st_mode & 0o777),
        "size": stat.st_size,
        "symlink": path.is_symlink(),
    }
    if path.is_symlink():
        record["target"] = os.readlink(path)
    elif path.is_file():
        record["sha256"] = sha256(path)
    inventory.append(record)

worker = Path("/opt/edgellm-v091/bin/moss_tts_nano_worker")
ldd = subprocess.run(
    ["ldd", "-r", str(worker)],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
)
report = {
    "profile": profile.get("name"),
    "missing": missing,
    "resolved_env": resolved_env,
    "required_engines": profile.get("required_engines"),
    "worker": {
        "path": str(worker),
        "exists": worker.exists(),
        "executable": os.access(worker, os.X_OK),
        "sha256": sha256(worker) if worker.is_file() else None,
        "ldd_returncode": ldd.returncode,
        "ldd_output": ldd.stdout,
    },
    "inventory": inventory,
    "onnxruntime_candidates": [
        str(path)
        for base in (Path("/usr/local/lib"), Path("/usr/lib"), Path("/opt"))
        for path in base.glob("**/libonnxruntime.so*")
    ],
}
output = Path(
    "/validation/results/moss-profile-resolve-artifact-ldd-preflight.json"
)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {
            "profile": report["profile"],
            "missing": missing,
            "resolved_env": resolved_env,
            "worker": report["worker"],
            "inventory_count": len(inventory),
            "onnxruntime_candidates": report["onnxruntime_candidates"],
        },
        ensure_ascii=False,
        indent=2,
    )
)
