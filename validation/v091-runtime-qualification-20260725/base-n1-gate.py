#!/usr/bin/env python3
"""Strict Base N=1 ASR true-streaming and HTTP TTS cancellation gate."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
from pathlib import Path


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "p95": ordered[p95_index],
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18621")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-rounds", type=int, default=3)
    parser.add_argument("--tts-rounds", type=int, default=3)
    parser.add_argument("--cancel-rounds", type=int, default=20)
    parser.add_argument("--recovery-timeout", type=float, default=15)
    args = parser.parse_args()

    helper_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(helper_root))
    from qwen_asr_stream_n2_service import clean, stream_one
    from tts_http_stream_gate import cancel_recovery, stream_once

    websocket_url = args.base_url.replace("http://", "ws://").replace(
        "https://", "wss://"
    )
    asr_rows = []
    for index in range(1, args.asr_rounds + 1):
        row = clean(
            stream_one(
                websocket_url,
                args.wav,
                "Chinese",
                "非常震惊",
                250,
                True,
                threading.Barrier(1),
                30,
            )
        )
        row["round"] = index
        row["passed"] = bool(
            row["matches"]
            and row["text"]
            and row["partial_before_eos"]
            and row["partial_count"] > 0
        )
        asr_rows.append(row)
        print(
            f"ASR {index}/{args.asr_rounds}: pass={row['passed']} "
            f"partial={row['first_partial_ms']:.1f}ms "
            f"eos-final={row['eos_to_final_ms']:.1f}ms",
            flush=True,
        )

    tts_rows = []
    for index in range(1, args.tts_rounds + 1):
        row = stream_once(
            args.base_url,
            "这是新版运行时的低延迟流式语音验证。",
            90,
            24000,
        )
        row["round"] = index
        audio_seconds = row["pcm_bytes"] / (24000 * 2)
        row["audio_seconds"] = audio_seconds
        row["rtf"] = (
            row["wall_ms"] / 1000 / audio_seconds if audio_seconds > 0 else None
        )
        row["passed"] = bool(row["passed"] and row["pcm_bytes"] > 0)
        tts_rows.append(row)
        print(
            f"TTS {index}/{args.tts_rounds}: pass={row['passed']} "
            f"ttfa={row['ttfa_ms']:.1f}ms rtf={row['rtf']:.3f}",
            flush=True,
        )

    cancel_rows = []
    for index in range(1, args.cancel_rounds + 1):
        row = cancel_recovery(args.base_url, 90, 24000, args.recovery_timeout)
        row["round"] = index
        cancel_rows.append(row)
        print(
            f"CANCEL {index}/{args.cancel_rounds}: pass={row['passed']} "
            f"recovery={row['recovery_ms']:.1f}ms "
            f"pcm={row['recovery'].get('pcm_bytes')}",
            flush=True,
        )
        if not row["passed"]:
            break

    report = {
        "test": "v091_r2_base_n1",
        "asr": {
            "rounds": asr_rows,
            "partial_before_eos_count": sum(
                row["partial_before_eos"] for row in asr_rows
            ),
            "first_partial_ms": summary(
                [row["first_partial_ms"] for row in asr_rows]
            ),
            "eos_to_final_ms": summary(
                [row["eos_to_final_ms"] for row in asr_rows]
            ),
        },
        "tts": {
            "rounds": tts_rows,
            "ttfa_ms": summary([row["ttfa_ms"] for row in tts_rows]),
            "rtf": summary([row["rtf"] for row in tts_rows]),
        },
        "cancel_recovery": {
            "rounds_requested": args.cancel_rounds,
            "rounds_completed": len(cancel_rows),
            "rounds": cancel_rows,
            "recovery_ms": summary(
                [row["recovery_ms"] for row in cancel_rows]
            ),
        },
    }
    report["passed"] = bool(
        len(asr_rows) == args.asr_rounds
        and all(row["passed"] for row in asr_rows)
        and len(tts_rows) == args.tts_rounds
        and all(row["passed"] for row in tts_rows)
        and len(cancel_rows) == args.cancel_rounds
        and all(row["passed"] for row in cancel_rows)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"]}, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
