#!/usr/bin/env python3
"""Offline whole-file /asr benchmark: bypasses VAD/streaming endpoint entirely.

Separates "model capability" (this script) from "streaming endpoint
behavior" (asr_stream_ws_bench.py): POST /asr sends the whole WAV in one
request and reads back the full-file transcript, so early-VAD-endpoint
truncation on long/paused audio cannot happen here — see
docs/perf/rk3576-matrix-20260906.md for why that distinction matters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from asr_stream_ws_bench import _char_err_rate, _err_rate, load_corpus_items, mean


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--corpus", default="bench/perf/corpus")
    p.add_argument("--category", default="short")
    p.add_argument("--lang", choices=["zh", "en"], required=True)
    p.add_argument("--limit", type=int, default=5)
    args = p.parse_args()

    corpus = Path(args.corpus)
    items = load_corpus_items(corpus, args.category, args.lang, args.limit)

    rows = []
    for item in items:
        ref = item.get("eval_transcript") or item["transcript"]
        wav_path = corpus / item["filename"]
        with open(wav_path, "rb") as f:
            resp = requests.post(
                f"http://{args.host}/asr",
                params={"language": args.lang},
                files={"file": (wav_path.name, f, "audio/wav")},
                timeout=60,
            )
        resp.raise_for_status()
        data = resp.json()
        hyp = data.get("text", "")
        row = {
            "id": item["id"],
            "ref": ref,
            "text": hyp,
            "error_rate": _err_rate(ref, hyp, args.lang),
            "char_error_rate": _char_err_rate(ref, hyp, args.lang),
            "rtf": data.get("rtf"),
            "backend": data.get("backend"),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print(json.dumps({
        "summary": {
            "lang": args.lang, "category": args.category, "n": len(rows),
            "mean_error_rate": mean(rows, "error_rate"),
            "mean_char_error_rate": mean(rows, "char_error_rate"),
        }
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
