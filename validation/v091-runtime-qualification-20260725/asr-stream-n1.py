#!/usr/bin/env python3
import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_asr_stream_n2_service import clean, stream_one


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--expect", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="ws://127.0.0.1:8621")
    args = parser.parse_args()
    result = clean(
        stream_one(
            args.base_url,
            args.wav,
            args.language,
            args.expect,
            250,
            True,
            threading.Barrier(1),
            30,
        )
    )
    result["passed"] = bool(result["matches"] and result["text"])
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
