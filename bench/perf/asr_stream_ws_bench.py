#!/usr/bin/env python3
"""Benchmark /asr/stream WebSocket stop-to-final latency."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websocket


def _norm_zh(text: str) -> str:
    return re.sub(r"[\s，。、“”\"'（）()：:；;,.!?！？-]", "", text).lower()


def _norm_en(text: str) -> list[str]:
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text).lower()
    return [w for w in text.split() if w]


def _norm_en_chars(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def _edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _err_rate(ref: str, hyp: str, lang: str) -> float:
    if lang == "zh":
        r, h = _norm_zh(ref), _norm_zh(hyp)
        return _edit_distance(r, h) / max(1, len(r))
    r, h = _norm_en(ref), _norm_en(hyp)
    return _edit_distance(r, h) / max(1, len(r))


def _char_err_rate(ref: str, hyp: str, lang: str) -> float:
    if lang == "zh":
        r, h = _norm_zh(ref), _norm_zh(hyp)
    else:
        r, h = _norm_en_chars(ref), _norm_en_chars(hyp)
    return _edit_distance(r, h) / max(1, len(r))


def load_corpus_items(corpus: Path, category: str, lang: str, limit: int) -> list[dict]:
    """Manifest rows for one (category, lang) slice, capped at `limit`."""
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    return [
        x for x in manifest["files"]
        if x["category"] == category and x["lang"] == lang
    ][:limit]


def mean(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(1, len(rows))


def _load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        x_old = np.linspace(0, len(audio) - 1, len(audio))
        x_new = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def _drain(ws, deadline_s: float) -> list[dict]:
    messages: list[dict] = []
    while time.perf_counter() < deadline_s:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            break
        except websocket.WebSocketConnectionClosedException:
            break
        try:
            messages.append(json.loads(raw))
        except Exception:
            messages.append({"raw": raw})
    return messages


def run_one(
    url: str,
    wav_path: Path,
    ref: str,
    lang: str,
    chunk_ms: int,
    realtime: bool,
    prepare_lead_ms: int,
) -> dict:
    audio = _load_audio(wav_path)
    audio_i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    chunk_n = max(1, int(16000 * chunk_ms / 1000))
    ws = websocket.create_connection(url, timeout=30)
    ws.settimeout(0.001)

    messages: list[dict] = []
    feed_start = time.perf_counter()
    for start in range(0, len(audio_i16), chunk_n):
        t0 = time.perf_counter()
        ws.send_binary(audio_i16[start:start + chunk_n].tobytes())
        messages.extend(_drain(ws, time.perf_counter() + 0.001))
        if realtime:
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, chunk_ms / 1000 - elapsed))
    feed_ms = (time.perf_counter() - feed_start) * 1000

    prepare_at = None
    if prepare_lead_ms >= 0:
        prepare_at = time.perf_counter()
        ws.send(json.dumps({"type": "prepare"}))
        time.sleep(max(0.0, prepare_lead_ms / 1000))

    ws.settimeout(10)
    eos_at = time.perf_counter()
    ws.send_binary(b"")
    final = None
    while True:
        try:
            msg = json.loads(ws.recv())
        except (websocket.WebSocketTimeoutException, websocket.WebSocketConnectionClosedException):
            break
        messages.append(msg)
        if msg.get("is_final"):
            final = msg
            break
    final_wait_ms = (time.perf_counter() - eos_at) * 1000
    prepare_to_final_ms = (
        (time.perf_counter() - prepare_at) * 1000
        if prepare_at is not None
        else None
    )
    try:
        ws.close()
    except Exception:
        pass

    text = (final or {}).get("text", "")
    return {
        "text": text,
        "ref": ref,
        "error_rate": _err_rate(ref, text, lang),
        "char_error_rate": _char_err_rate(ref, text, lang),
        "feed_wall_ms": feed_ms,
        "prepare_lead_ms": prepare_lead_ms,
        "eos_to_final_ms": final_wait_ms,
        "prepare_to_final_ms": prepare_to_final_ms,
        "messages": len(messages),
        "final": final or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="host:port, for example 100.89.94.11:8621")
    parser.add_argument("--corpus", default="bench/perf/corpus")
    parser.add_argument("--category", default="short")
    parser.add_argument("--lang", choices=["zh", "en"], required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--chunk-ms", type=int, default=250)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--prepare-lead-ms",
        type=int,
        default=-1,
        help="Send {type: prepare}, wait this long, then send EOS. -1 disables.",
    )
    args = parser.parse_args()

    corpus = Path(args.corpus)
    items = load_corpus_items(corpus, args.category, args.lang, args.limit)
    url = f"ws://{args.host}/asr/stream?language=auto&sample_rate=16000"

    rows = []
    for idx, item in enumerate(items):
        if idx:
            time.sleep(3.0)  # let SessionLimiter (effective_limit=1) release prior slot
        ref = item.get("eval_transcript") or item["transcript"]
        row = run_one(
            url,
            corpus / item["filename"],
            ref,
            args.lang,
            args.chunk_ms,
            args.realtime,
            args.prepare_lead_ms,
        )
        row.update({"id": item["id"], "duration_s": item["duration_s"]})
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print(json.dumps({
        "summary": {
            "lang": args.lang,
            "n": len(rows),
            "chunk_ms": args.chunk_ms,
            "realtime": args.realtime,
            "prepare_lead_ms": args.prepare_lead_ms,
            "mean_error_rate": mean(rows, "error_rate"),
            "mean_char_error_rate": mean(rows, "char_error_rate"),
            "mean_feed_wall_ms": mean(rows, "feed_wall_ms"),
            "mean_eos_to_final_ms": mean(rows, "eos_to_final_ms"),
            "mean_prepare_to_final_ms": (
                mean(rows, "prepare_to_final_ms") if args.prepare_lead_ms >= 0 else None
            ),
        }
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
