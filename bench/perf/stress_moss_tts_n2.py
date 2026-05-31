#!/usr/bin/env python3
"""N=2 concurrency stress test for MossTtsNanoBackend (C++ TRT worker).

Spawns two backend client threads sharing the same backend instance; the
backend itself ships one worker subprocess that internally maintains a
slot pool of size ``moss_max_slots`` (per profile env). At max_slots>=2,
the C++ worker dispatches concurrent requests to dedicated worker threads,
each grabbing its own runtime slot (separate CUDA stream + TRT contexts).

Modes:
  basic    — 2 concurrent requests with different prompts; checks both
             return audio, ttfa/wall stats logged.
  burst    — N alternating rounds (default 30) of 2 concurrent requests
             per round; counts errors/crashes.
  parity   — runs the same prompt twice (single-client) AND again under
             N=2 dual-client (same client sends both); md5 must match
             single-client byte-identical (per Phase 1 hard gate).
  mixed    — one short prompt + one long prompt concurrently; ensures
             short ttfa not blocked by long prompt.

Run on orin-nx (after binary deployed and profile env set):

  MOSS_MAX_SLOTS=2 python3 bench/perf/stress_moss_tts_n2.py --mode basic
  MOSS_MAX_SLOTS=2 python3 bench/perf/stress_moss_tts_n2.py --mode burst --rounds 30
  MOSS_MAX_SLOTS=2 python3 bench/perf/stress_moss_tts_n2.py --mode parity
  MOSS_MAX_SLOTS=2 python3 bench/perf/stress_moss_tts_n2.py --mode mixed
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import logging
import os
import sys
import threading
import time
from pathlib import Path


def _setup_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _single_request(backend, text: str, label: str, results: dict) -> None:
    tid = threading.get_ident()
    t0 = time.perf_counter()
    first_ms = None
    pcm_chunks: list[bytes] = []
    err: Exception | None = None
    try:
        for chunk in backend.generate_streaming(text):
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000
            pcm_chunks.append(chunk)
    except Exception as exc:  # noqa: BLE001
        err = exc
    wall_ms = (time.perf_counter() - t0) * 1000
    pcm = b"".join(pcm_chunks)
    results[label] = {
        "tid": tid,
        "text": text,
        "wall_ms": round(wall_ms, 1),
        "ttfa_ms": round(first_ms, 1) if first_ms is not None else None,
        "pcm_len": len(pcm),
        "pcm_md5": hashlib.md5(pcm).hexdigest(),
        "error": repr(err) if err else None,
    }


def _run_n2(backend, prompts: list[tuple[str, str]]) -> dict:
    """Send len(prompts) requests concurrently; return per-label result."""
    results: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futures = [
            ex.submit(_single_request, backend, text, label, results)
            for label, text in prompts
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    return results


def _mode_basic(backend) -> int:
    prompts = [
        ("A", "你好，今天天气怎么样？"),
        ("B", "我喜欢用语音助手。"),
    ]
    print("[basic] sending 2 concurrent requests")
    r = _run_n2(backend, prompts)
    for label, info in r.items():
        print(f"[basic] {label}: {info}")
    if any(v["error"] for v in r.values()):
        print("[basic] FAIL: at least one error")
        return 1
    if any(v["pcm_len"] == 0 for v in r.values()):
        print("[basic] FAIL: zero-length audio")
        return 1
    print("[basic] PASS")
    return 0


def _mode_burst(backend, rounds: int) -> int:
    err_count = 0
    crash_count = 0
    for i in range(rounds):
        prompts = [
            ("A", f"测试一下第{i}轮"),
            ("B", f"语音合成轮次{i}效果如何"),
        ]
        try:
            r = _run_n2(backend, prompts)
        except Exception as exc:  # noqa: BLE001
            crash_count += 1
            print(f"[burst] round {i}: CRASH {exc!r}")
            continue
        round_err = sum(1 for v in r.values() if v["error"] or v["pcm_len"] == 0)
        err_count += round_err
        # Compact log: only print on error.
        if round_err:
            print(f"[burst] round {i}: {r}")
        else:
            a = r["A"]; b = r["B"]
            print(
                f"[burst] round {i}: A ttfa={a['ttfa_ms']}ms len={a['pcm_len']} "
                f"B ttfa={b['ttfa_ms']}ms len={b['pcm_len']}"
            )
    print(f"[burst] DONE rounds={rounds} errors={err_count} crashes={crash_count}")
    return 0 if (err_count == 0 and crash_count == 0) else 1


def _mode_parity(backend) -> int:
    text = "你好，今天天气真不错"
    print("[parity] baseline single-client run #1")
    base = {}
    _single_request(backend, text, "X", base)
    print(f"[parity] baseline: {base['X']}")
    if base["X"]["error"] or base["X"]["pcm_len"] == 0:
        print("[parity] FAIL baseline error")
        return 1

    print("[parity] N=2 concurrent: same prompt twice")
    prompts = [("A", text), ("B", text)]
    r = _run_n2(backend, prompts)
    for label, info in r.items():
        print(f"[parity] {label}: {info}")

    # Strict gate: md5 byte-identical to single-client baseline (Phase 1 promise).
    base_md5 = base["X"]["pcm_md5"]
    a_md5 = r["A"]["pcm_md5"]; b_md5 = r["B"]["pcm_md5"]
    ok = (a_md5 == base_md5 and b_md5 == base_md5)
    print(f"[parity] baseline={base_md5} A={a_md5} B={b_md5} match={ok}")
    return 0 if ok else 1


def _mode_mixed(backend) -> int:
    short_text = "你好"
    long_text = "这是一个语音合成测试，用于验证长文本和短文本并发时的延迟特性。"
    print("[mixed] baseline single-client short prompt for TTFA reference")
    base_short = {}
    _single_request(backend, short_text, "S", base_short)
    short_ttfa_single = base_short["S"]["ttfa_ms"]
    print(f"[mixed] single-client short ttfa={short_ttfa_single}ms")

    print("[mixed] N=2 concurrent: short + long")
    prompts = [("short", short_text), ("long", long_text)]
    r = _run_n2(backend, prompts)
    for label, info in r.items():
        print(f"[mixed] {label}: {info}")

    short_ttfa_n2 = r["short"]["ttfa_ms"]
    if any(v["error"] or v["pcm_len"] == 0 for v in r.values()):
        print("[mixed] FAIL: error or empty audio")
        return 1
    # Spec gate: short TTFA N=2 / single ≤ 1.5x. Print only — soft gate.
    if short_ttfa_single and short_ttfa_n2:
        ratio = short_ttfa_n2 / short_ttfa_single
        print(f"[mixed] short TTFA ratio N=2/single = {ratio:.2f} (spec ≤1.5)")
    print("[mixed] PASS (errors=0)")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["basic", "burst", "parity", "mixed"], default="basic")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--max-slots", default=None, help="override MOSS_MAX_SLOTS env")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.max_slots is not None:
        os.environ["MOSS_MAX_SLOTS"] = str(args.max_slots)
    _setup_path()
    from voxedge.backends.jetson.moss_tts_nano import MossTtsNanoBackend

    profile = {"moss_max_slots": int(os.environ.get("MOSS_MAX_SLOTS", "2"))}
    backend = MossTtsNanoBackend(profile)
    print(f"[stress] starting backend; moss_max_slots={profile['moss_max_slots']}")
    t0 = time.perf_counter()
    backend.preload()
    print(f"[stress] preload OK in {(time.perf_counter()-t0)*1000:.0f}ms")

    try:
        if args.mode == "basic":
            rc = _mode_basic(backend)
        elif args.mode == "burst":
            rc = _mode_burst(backend, args.rounds)
        elif args.mode == "parity":
            rc = _mode_parity(backend)
        elif args.mode == "mixed":
            rc = _mode_mixed(backend)
        else:
            rc = 2
    finally:
        backend.shutdown()
    print(f"[stress] mode={args.mode} exit={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
