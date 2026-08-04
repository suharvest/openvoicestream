#!/usr/bin/env python3
"""ASR N=2 plus TTS N=2 cross-component service gate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import requests


ERROR_RE = re.compile(
    r"(CUDA(?: runtime)? (?:error|failure)|illegal memory access|"
    r"Myelin.*(?:error|fail|already loaded binary graph)|"
    r"(?:TensorRT|\[TRT\]).*(?:\[E\]|error|fail)|"
    r"segmentation fault|core dumped|out of memory)",
    re.IGNORECASE,
)


def timed(call: Callable[[], Any], barrier: threading.Barrier | None = None) -> dict[str, Any]:
    if barrier is not None:
        barrier.wait()
    started = time.perf_counter()
    value = call()
    ended = time.perf_counter()
    return {
        "started_monotonic": started,
        "ended_monotonic": ended,
        "elapsed_ms": (ended - started) * 1000,
        "value": value,
    }


def post_tts(url: str, text: str, timeout: float) -> tuple[int, bytes]:
    response = requests.post(
        url.rstrip("/") + "/tts",
        json={"text": text, "language": "chinese"},
        timeout=timeout,
    )
    return response.status_code, response.content


def post_asr(
    url: str,
    wav_path: Path,
    language: str,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    with wav_path.open("rb") as wav:
        response = requests.post(
            url.rstrip("/") + "/asr",
            files={"file": (wav_path.name, wav, "audio/wav")},
            data={"language": language},
            timeout=timeout,
        )
    try:
        payload = response.json()
    except Exception:
        payload = {"raw": response.text[:1000]}
    return response.status_code, payload


def clean(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.endswith("_monotonic") and key != "value"
    }


def overlap_ms(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(
        0.0,
        (
            min(a["ended_monotonic"], b["ended_monotonic"])
            - max(a["started_monotonic"], b["started_monotonic"])
        )
        * 1000,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8622")
    parser.add_argument("--wav-a", type=Path, required=True)
    parser.add_argument("--wav-b", type=Path, required=True)
    parser.add_argument("--expect-a", required=True)
    parser.add_argument("--expect-b", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started_iso = dt.datetime.now(dt.timezone.utc).isoformat()

    tts_texts = (
        "这是双路语音合成请求甲，用于验证跨模型并发稳定性。",
        "这是双路语音合成请求乙，用于确认音频结果互不串扰。",
    )
    baselines = []
    for text in tts_texts:
        status, body = post_tts(args.base_url, text, args.timeout)
        if status != 200 or not body:
            raise RuntimeError(f"TTS baseline failed: status={status}")
        baselines.append(hashlib.sha256(body).hexdigest())
    if baselines[0] == baselines[1]:
        raise RuntimeError("distinct TTS baselines are identical")

    rounds = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index in range(1, args.rounds + 1):
            barrier = threading.Barrier(4)
            futures = {
                "tts_a": pool.submit(
                    timed,
                    lambda: post_tts(args.base_url, tts_texts[0], args.timeout),
                    barrier,
                ),
                "tts_b": pool.submit(
                    timed,
                    lambda: post_tts(args.base_url, tts_texts[1], args.timeout),
                    barrier,
                ),
                "asr_a": pool.submit(
                    timed,
                    lambda: post_asr(
                        args.base_url, args.wav_a, "Chinese", args.timeout
                    ),
                    barrier,
                ),
                "asr_b": pool.submit(
                    timed,
                    lambda: post_asr(
                        args.base_url, args.wav_b, "English", args.timeout
                    ),
                    barrier,
                ),
            }
            results = {name: future.result() for name, future in futures.items()}
            tts_status_a, tts_body_a = results["tts_a"]["value"]
            tts_status_b, tts_body_b = results["tts_b"]["value"]
            asr_status_a, asr_body_a = results["asr_a"]["value"]
            asr_status_b, asr_body_b = results["asr_b"]["value"]
            tts_matches = (
                hashlib.sha256(tts_body_a).hexdigest() == baselines[0]
                and hashlib.sha256(tts_body_b).hexdigest() == baselines[1]
            )
            asr_matches = (
                re.search(
                    args.expect_a,
                    str(asr_body_a.get("text", "")),
                    re.IGNORECASE,
                )
                is not None
                and re.search(
                    args.expect_b,
                    str(asr_body_b.get("text", "")),
                    re.IGNORECASE,
                )
                is not None
            )
            cross_overlap = max(
                overlap_ms(results[tts], results[asr])
                for tts in ("tts_a", "tts_b")
                for asr in ("asr_a", "asr_b")
            )
            ok = (
                tts_status_a == 200
                and tts_status_b == 200
                and asr_status_a == 200
                and asr_status_b == 200
                and tts_matches
                and asr_matches
                and cross_overlap > 0
            )
            rounds.append(
                {
                    "round": index,
                    "tts_a": clean(results["tts_a"]),
                    "tts_b": clean(results["tts_b"]),
                    "asr_a": clean(results["asr_a"]),
                    "asr_b": clean(results["asr_b"]),
                    "tts_matches_baseline": tts_matches,
                    "asr_matches_expected": asr_matches,
                    "cross_component_overlap_ms": cross_overlap,
                    "ok": ok,
                }
            )
            print(
                f"[{index:03d}/{args.rounds}] ok={ok} "
                f"tts={results['tts_a']['elapsed_ms']:.1f}/"
                f"{results['tts_b']['elapsed_ms']:.1f}ms "
                f"asr={results['asr_a']['elapsed_ms']:.1f}/"
                f"{results['asr_b']['elapsed_ms']:.1f}ms "
                f"cross_overlap={cross_overlap:.1f}ms",
                flush=True,
            )

    logs = subprocess.run(
        ["docker", "logs", "--since", started_iso, args.container],
        text=True,
        capture_output=True,
        check=False,
    )
    error_hits = [
        line
        for line in (logs.stdout + logs.stderr).splitlines()
        if ERROR_RE.search(line)
    ]
    inspect = subprocess.run(
        [
            "docker",
            "inspect",
            "--format={{.RestartCount}}|{{.State.OOMKilled}}",
            args.container,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    report = {
        "test": "voice_n2_cross_component",
        "rounds_requested": args.rounds,
        "rounds_passed": sum(bool(row["ok"]) for row in rounds),
        "tts_baseline_sha256": baselines,
        "rounds": rounds,
        "container_state": inspect.stdout.strip(),
        "runtime_error_hits": error_hits,
    }
    report["passed"] = (
        report["rounds_passed"] == args.rounds
        and not error_hits
        and inspect.stdout.strip() == "0|false"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rounds_requested": report["rounds_requested"],
        "rounds_passed": report["rounds_passed"],
        "container_state": report["container_state"],
        "runtime_error_hits": error_hits,
        "passed": report["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
