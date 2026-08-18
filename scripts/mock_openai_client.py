#!/usr/bin/env python3
"""Mock device-side client for POST /v1/audio/transcriptions.

Simulates an openai_whisper-style adapter continuously pushing audio to the
OpenAI-compatible transcription endpoint. Pure standard library — no extra
dependencies.

Examples:
    python scripts/mock_openai_client.py --count 20
    python scripts/mock_openai_client.py --api-key sk-xxx --interval 0.5
    python scripts/mock_openai_client.py --wav a.wav --wav b.wav --concurrency 4
    python scripts/mock_openai_client.py --use-openai-sdk --count 5

Behavior:
    - Multipart POST to <base-url>/v1/audio/transcriptions.
    - On 429: honors Retry-After, exponential backoff capped at 30 s.
    - Prints status / latency / text summary per request.
    - On Ctrl-C or when --count completes: prints success/429/error counts
      and latency p50/p95.
"""
from __future__ import annotations

import argparse
import io
import math
import mimetypes
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

MAX_BACKOFF_S = 30.0


# ---------------------------------------------------------------------------
# Audio payload
# ---------------------------------------------------------------------------

def synth_wav(seconds: float = 3.0, sample_rate: int = 16000) -> bytes:
    """3 s 16 kHz mono PCM16: 1 s silence + sine sweep (stdlib only)."""
    n = int(sample_rate * seconds)
    frames = bytearray()
    for i in range(n):
        t = i / sample_rate
        if t < 1.0:
            sample = 0
        else:
            sample = int(12000 * math.sin(2 * math.pi * 440.0 * t))
        frames += struct.pack("<h", sample)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def load_payloads(wav_paths: list[str]) -> list[tuple[str, bytes]]:
    if not wav_paths:
        return [("synthetic.wav", synth_wav())]
    payloads = []
    for p in wav_paths:
        with open(p, "rb") as f:
            payloads.append((os.path.basename(p), f.read()))
    return payloads


# ---------------------------------------------------------------------------
# Multipart request (stdlib)
# ---------------------------------------------------------------------------

def encode_multipart(fields: dict[str, str], filename: str, file_bytes: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    lines: list[bytes] = []
    for name, value in fields.items():
        lines += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            str(value).encode(),
        ]
    ctype = mimetypes.guess_type(filename)[0] or "audio/wav"
    lines += [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
        f"Content-Type: {ctype}".encode(),
        b"",
        file_bytes,
        f"--{boundary}--".encode(),
        b"",
    ]
    return b"\r\n".join(lines), f"multipart/form-data; boundary={boundary}"


def post_transcription(args, filename: str, file_bytes: bytes) -> tuple[int, float, str, dict]:
    """One POST. Returns (status, latency_s, text_summary, headers)."""
    fields: dict[str, str] = {"model": "whisper-1", "response_format": args.response_format}
    if args.language:
        fields["language"] = args.language
    body, content_type = encode_multipart(fields, filename, file_bytes)
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/audio/transcriptions",
        data=body,
        method="POST",
        headers={"Content-Type": content_type},
    )
    if args.api_key:
        req.add_header("Authorization", f"Bearer {args.api_key}")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            return resp.status, time.perf_counter() - t0, payload[:120], dict(resp.headers)
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", errors="replace")
        return e.code, time.perf_counter() - t0, payload[:120], dict(e.headers or {})


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ok = 0
        self.throttled = 0
        self.errors = 0
        self.latencies: list[float] = []
        self.inflight = 0
        self.peak_inflight = 0
        self.peak_ok_inflight = 0

    def enter(self) -> None:
        """Mark a request as in flight; tracks the peak for capacity claims."""
        with self.lock:
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)

    def leave(self, status: int) -> None:
        with self.lock:
            if status == 200:
                # Peak overlap of *accepted* requests is the real channel count:
                # a 429 proves the limiter rejected, not that it served.
                self.peak_ok_inflight = max(self.peak_ok_inflight, self.inflight)
            self.inflight -= 1

    def record(self, status: int, latency: float) -> None:
        with self.lock:
            if status == 200:
                self.ok += 1
                self.latencies.append(latency)
            elif status == 429:
                self.throttled += 1
            else:
                self.errors += 1

    @staticmethod
    def _pct(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
        return sorted_vals[idx]

    def report(self) -> str:
        with self.lock:
            lat = sorted(self.latencies)
            return (
                f"total={self.ok + self.throttled + self.errors} "
                f"success={self.ok} 429={self.throttled} errors={self.errors} "
                f"latency_p50={self._pct(lat, 0.50) * 1000:.1f}ms "
                f"latency_p95={self._pct(lat, 0.95) * 1000:.1f}ms "
                f"peak_inflight={self.peak_inflight} "
                f"peak_accepted_inflight={self.peak_ok_inflight}"
            )


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def run_worker(worker_id: int, args, payloads, stats: Stats, stop: threading.Event,
               barrier: "threading.Barrier | None" = None) -> None:
    try:
        _run_worker_loop(worker_id, args, payloads, stats, stop, barrier)
    finally:
        if barrier is not None:
            # Release anyone still waiting: a worker throttled into retries
            # consumes no count slot, so it lags and would otherwise block on a
            # barrier the finished workers will never reach again.
            barrier.abort()


def _run_worker_loop(worker_id: int, args, payloads, stats: Stats, stop: threading.Event,
                     barrier: "threading.Barrier | None") -> None:
    i = 0
    backoff = 1.0
    while not stop.is_set() and (args.count == 0 or i < args.count):
        if barrier is not None:
            # Burst mode: every worker fires the same round simultaneously, so
            # overlap is guaranteed rather than left to timing luck. Without
            # this, a fast decode plus a non-zero --interval means requests
            # never coincide and the limiter is never actually exercised.
            try:
                barrier.wait(timeout=args.timeout + 30)
            except threading.BrokenBarrierError:
                return
        filename, file_bytes = payloads[i % len(payloads)]
        stats.enter()
        try:
            status, latency, summary, headers = post_transcription(args, filename, file_bytes)
        except Exception as exc:  # connection refused, timeout, ...
            stats.leave(-1)
            stats.record(-1, 0.0)
            print(f"[w{worker_id} #{i}] ERROR {type(exc).__name__}: {exc}", flush=True)
            i += 1
            stop.wait(args.interval)
            continue
        stats.leave(status)
        stats.record(status, latency)
        print(
            f"[w{worker_id} #{i}] status={status} latency={latency * 1000:.1f}ms "
            f"resp={summary!r}",
            flush=True,
        )
        if status == 429:
            retry_after = headers.get("Retry-After")
            try:
                wait_s = float(retry_after) if retry_after else backoff
            except ValueError:
                wait_s = backoff
            wait_s = min(max(wait_s, backoff), MAX_BACKOFF_S)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
            print(f"[w{worker_id}] throttled, backing off {wait_s:.1f}s", flush=True)
            stop.wait(wait_s)
            continue  # retry the same request, do not consume a count slot
        backoff = 1.0
        i += 1
        if args.interval > 0:
            stop.wait(args.interval)


# ---------------------------------------------------------------------------
# Optional: openai SDK path
# ---------------------------------------------------------------------------

def run_openai_sdk(args, payloads, stats: Stats) -> int:
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed — skipping SDK mode "
              "(run without --use-openai-sdk, or install openai separately)")
        return 0
    client = OpenAI(base_url=args.base_url.rstrip("/") + "/v1",
                    api_key=args.api_key or "unused")
    count = args.count or 5
    for i in range(count):
        filename, file_bytes = payloads[i % len(payloads)]
        t0 = time.perf_counter()
        try:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=(filename, io.BytesIO(file_bytes)),
                language=args.language or None,
                response_format=args.response_format,
            )
            latency = time.perf_counter() - t0
            text = tr if isinstance(tr, str) else getattr(tr, "text", "")
            stats.record(200, latency)
            print(f"[sdk #{i}] ok latency={latency * 1000:.1f}ms text={str(text)[:60]!r}")
        except Exception as exc:
            stats.record(-1, time.perf_counter() - t0)
            print(f"[sdk #{i}] ERROR {type(exc).__name__}: {exc}")
        time.sleep(args.interval)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8621")
    ap.add_argument("--api-key", default=os.environ.get("OVS_API_KEY", ""))
    ap.add_argument("--wav", action="append", default=[],
                    help="WAV file to send (repeatable); default: generated 3s 16k mono")
    ap.add_argument("--count", type=int, default=0, help="requests per worker; 0 = infinite")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--burst", action="store_true",
                    help="synchronize workers on a barrier so each round fires "
                         "simultaneously (use this to measure real channel capacity)")
    ap.add_argument("--response-format", default="json",
                    choices=["json", "text", "verbose_json", "srt", "vtt"])
    ap.add_argument("--language", default="")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--use-openai-sdk", action="store_true",
                    help="use the openai package if installed (skips if missing)")
    args = ap.parse_args()

    payloads = load_payloads(args.wav)
    stats = Stats()

    if args.use_openai_sdk:
        rc = run_openai_sdk(args, payloads, stats)
        print("STATS:", stats.report())
        return rc

    stop = threading.Event()
    n_workers = max(1, args.concurrency)
    barrier = threading.Barrier(n_workers) if args.burst and n_workers > 1 else None
    threads = [
        threading.Thread(target=run_worker,
                         args=(w, args, payloads, stats, stop, barrier), daemon=True)
        for w in range(n_workers)
    ]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\ninterrupted, shutting down...", flush=True)
        stop.set()
        for t in threads:
            t.join(timeout=5)
    print("STATS:", stats.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
