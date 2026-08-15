#!/usr/bin/env python3
"""Measure deterministic ASR HTTP solo, N=2, and recovery behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _multipart(wav: Path) -> tuple[bytes, str]:
    boundary = f"edgellm-{uuid.uuid4().hex}"
    fields = (("model", "qwen3-asr"), ("response_format", "json"))
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            )
        )
    chunks.extend(
        (
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{wav.name}"\r\n'
            ).encode(),
            b"Content-Type: audio/wav\r\n\r\n",
            wav.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return b"".join(chunks), boundary


def _request(url: str, wav: Path, timeout: float) -> dict[str, Any]:
    body, boundary = _multipart(wav)
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        status = response.status
    elapsed = time.perf_counter() - started
    parsed = json.loads(payload)
    text = str(parsed.get("text", "")).strip()
    if status != 200 or not text:
        raise RuntimeError(f"ASR request failed: status={status} payload={parsed!r}")
    return {
        "elapsed_seconds": elapsed,
        "status": status,
        "text": text,
        "response_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def run(
    base_url: str,
    wav_a: Path,
    wav_b: Path,
    rounds: int,
    timeout: float,
    label: str,
) -> dict[str, Any]:
    if rounds < 3:
        raise ValueError("rounds must be at least 3")
    url = base_url.rstrip("/") + "/v1/audio/transcriptions"
    warm_a = _request(url, wav_a, timeout)
    warm_b = _request(url, wav_b, timeout)
    solo: list[dict[str, Any]] = []
    n2: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for _ in range(rounds):
            solo.append({"a": _request(url, wav_a, timeout), "b": _request(url, wav_b, timeout)})
            started = time.perf_counter()
            future_a = executor.submit(_request, url, wav_a, timeout)
            future_b = executor.submit(_request, url, wav_b, timeout)
            result_a = future_a.result()
            result_b = future_b.result()
            n2.append(
                {
                    "a": result_a,
                    "b": result_b,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    recovery = _request(url, wav_a, timeout)
    if recovery["text"] != warm_a["text"]:
        raise RuntimeError("deterministic recovery transcription changed")
    return {
        "schema_version": 1,
        "label": label,
        "base_url": base_url,
        "rounds": rounds,
        "inputs": {
            "a": {"path": str(wav_a), "sha256": _sha256(wav_a)},
            "b": {"path": str(wav_b), "sha256": _sha256(wav_b)},
        },
        "warmup": {"a": warm_a, "b": warm_b},
        "solo": solo,
        "n2": n2,
        "recovery": recovery,
        "metrics": {
            "solo_a_seconds": _stats([item["a"]["elapsed_seconds"] for item in solo]),
            "solo_b_seconds": _stats([item["b"]["elapsed_seconds"] for item in solo]),
            "n2_wall_seconds": _stats([item["wall_seconds"] for item in n2]),
            "n2_a_seconds": _stats([item["a"]["elapsed_seconds"] for item in n2]),
            "n2_b_seconds": _stats([item["b"]["elapsed_seconds"] for item in n2]),
        },
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--wav-a", type=Path, required=True)
    parser.add_argument("--wav-b", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.base_url,
        args.wav_a.resolve(),
        args.wav_b.resolve(),
        args.rounds,
        args.timeout,
        args.label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
