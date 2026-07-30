#!/usr/bin/env python3
"""Speech-driven barge-in probe — the NPU co-residency stress test.

WHY THIS EXISTS (and why tests/test_v2v_bargein_latency.py does not cover it)
-----------------------------------------------------------------------------
``tests/test_v2v_bargein_latency.py`` drives barge-in via the explicit
``CLIENT_ABORT`` frame and injects *text* to trigger TTS. That path never runs
the recognizer, so ASR and TTS never touch the accelerator at the same time.

The interesting failure mode on a shared-NPU device (RK3588: ASR RKNN encoder +
RKLLM decoder and matcha's Vocos RKNN all live on the same 3-core NPU) is the
other barge-in path: ``audio_dispatcher.py`` sees a VAD ``speech_start`` while
TTS is mid-sentence and calls ``Session._bargein_tts()``. To reach it the client
has to *speak over the reply*, which makes the ASR encoder (one inference per
400 ms chunk) and the RKLLM partial decode run concurrently with a Vocos
inference. That overlap is what the shared NPU lock exists to serialize.

So this probe: speak, let the server answer, then speak again *while the answer
is still playing*, and check three things:

  1. barge-in actually cuts the audio (and how fast),
  2. the interrupting utterance still transcribes correctly,
  3. no RKNN fault is produced by the overlap (check the service log separately;
     this probe only reports what the client can observe).

Usage:
    uv run --with websocket-client --with numpy python bench/perf/bargein_npu_probe.py \
        --host 127.0.0.1:8621 --wav bench/perf/corpus/short/zh_short_01.wav --trials 3

Exit status is 0 if every trial achieved real overlap AND still transcribed the
interrupting utterance; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
import wave

import websocket

SAMPLE_RATE = 16000
CHUNK_MS = 100
# How long to let the reply play before speaking over it. Must be > 0 so we are
# genuinely interrupting mid-sentence rather than racing the first frame.
INTERRUPT_AFTER_MS = 300
# Audio is considered stopped once no PCM frame has arrived for this long.
QUIET_MS = 800


def load_wav_16k_mono(path: str) -> tuple[bytes, float]:
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise SystemExit(f"{path}: need mono 16-bit PCM")
        if wf.getframerate() != SAMPLE_RATE:
            raise SystemExit(f"{path}: need {SAMPLE_RATE} Hz, got {wf.getframerate()}")
        pcm = wf.readframes(wf.getnframes())
    return pcm, len(pcm) / 2 / SAMPLE_RATE


class Reader(threading.Thread):
    """Timestamps every inbound frame so the main thread can send freely."""

    daemon = True

    def __init__(self, ws):
        super().__init__()
        self.ws = ws
        self.lock = threading.Lock()
        self.finals: list[tuple[float, str, bool | None]] = []
        self.events: list[tuple[float, str]] = []
        self.t_first_pcm: float | None = None
        self.t_last_pcm: float | None = None
        self.pcm_bytes = 0
        self.error: str | None = None
        self.closed = False

    def run(self) -> None:
        self.ws.settimeout(0.2)
        while not self.closed:
            try:
                msg = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:                      # socket torn down
                self.error = self.error or f"{type(e).__name__}: {e}"
                return
            if msg is None or msg == "":
                self.closed = True
                return
            now = time.monotonic()
            with self.lock:
                if isinstance(msg, bytes):
                    # First binary frame is a 4-byte sample-rate header.
                    if len(msg) > 4:
                        if self.t_first_pcm is None:
                            self.t_first_pcm = now
                        self.t_last_pcm = now
                        self.pcm_bytes += len(msg)
                    continue
                try:
                    data = json.loads(msg)
                except (ValueError, TypeError):
                    continue
                t = data.get("type")
                self.events.append((now, t or "?"))
                if t == "asr_final":
                    self.finals.append(
                        (now, data.get("text", "") or "", data.get("session_complete"))
                    )
                elif t == "error":
                    self.error = data.get("error", "(no detail)")

    def snapshot(self):
        with self.lock:
            return (self.t_first_pcm, self.t_last_pcm, self.pcm_bytes,
                    list(self.finals), self.error)


def send_pcm(ws, pcm: bytes, realtime: bool = True) -> None:
    chunk_bytes = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2
    for off in range(0, len(pcm), chunk_bytes):
        chunk = pcm[off:off + chunk_bytes]
        if not chunk:
            break
        ws.send_binary(chunk)
        if realtime:
            time.sleep(CHUNK_MS / 1000.0)


def wait_for(pred, budget_s: float, poll_s: float = 0.02) -> bool:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(poll_s)
    return False


def trial(host: str, wav_pcm: bytes, language: str, budget_s: float,
          tts_language: str = "auto", interrupt_on: str = "final",
          interrupt_after_ms: int = 0) -> dict:
    out: dict = {"overlapped": False, "inject_to_last_pcm_ms": None,
                 "reply_pcm_bytes": 0, "final_1": None, "final_2": None,
                 "error": None, "pcm_after_inject_bytes": 0,
                 "reply_completed": None}
    ws = websocket.WebSocket()
    ws.connect(f"ws://{host}/v2v/stream", timeout=10)
    reader = Reader(ws)
    try:
        # multi_utterance is required: the interrupting speech is a *second*
        # utterance in the same session, so the session must not close on the
        # first final.
        # tts_language is what ARMS the TTS leg — omit it and the server opens
        # the session with "tts=off", never calls the LLM, and the probe has
        # nothing to barge into (it silently measures nothing).
        ws.send(json.dumps({
            "type": "config",
            "asr_language": language,
            "tts_language": tts_language,
            "sample_rate": SAMPLE_RATE,
            "vad": "silero",
            "multi_utterance": True,
        }))
        reader.start()

        # ── turn 1: speak, then let the server answer ──────────────────────
        send_pcm(ws, wav_pcm)
        silence = b"\x00" * (int(SAMPLE_RATE * 0.7) * 2)
        send_pcm(ws, silence)

        if not wait_for(lambda: len(reader.snapshot()[3]) >= 1, budget_s):
            out["error"] = "no asr_final for utterance 1"
            return out
        out["final_1"] = reader.snapshot()[3][0][1]

        if interrupt_on == "audio":
            # Cut-latency mode: wait for the reply to actually be playing.
            # NOTE this window is tiny in practice — the server pushes PCM as
            # fast as it synthesizes (matcha RTF ~0.09), so a short reply is
            # fully delivered in a few hundred ms and there is nothing left to
            # interrupt. Use it only with a deliberately long reply.
            if not wait_for(lambda: reader.snapshot()[0] is not None, budget_s):
                out["error"] = "reply never produced audio (nothing to barge into)"
                return out
        # Default ("final"): start speaking the moment the transcript lands, so
        # the interrupting utterance's ASR work (RKNN encoder per 400 ms chunk +
        # RKLLM partial decode) is guaranteed to overlap the LLM + Vocos
        # synthesis of the reply. That overlap is the thing under test; cut
        # latency is secondary.
        if interrupt_after_ms:
            time.sleep(interrupt_after_ms / 1000.0)

        # ── barge in: speak over the reply ─────────────────────────────────
        _, t_last_before, bytes_before, _, _ = reader.snapshot()
        t_inject = time.monotonic()
        send_pcm(ws, wav_pcm)          # real speech → VAD speech_start → cut

        # Audio has stopped once QUIET_MS passes with no new PCM frame.
        def quiet() -> bool:
            _, t_last, _, _, _ = reader.snapshot()
            return t_last is not None and (time.monotonic() - t_last) > QUIET_MS / 1000.0

        wait_for(quiet, budget_s)
        _, t_last_after, bytes_after, finals, err = reader.snapshot()
        out["pcm_after_inject_bytes"] = bytes_after - bytes_before
        if t_last_after is not None and t_last_after > t_inject:
            out["inject_to_last_pcm_ms"] = (t_last_after - t_inject) * 1000
        elif interrupt_on == "audio":
            # Audio had already stopped before we injected — the reply was too
            # short to interrupt. Report it rather than calling it a pass.
            out["error"] = "reply finished before injection; no overlap occurred"
            return out
        out["overlapped"] = out["pcm_after_inject_bytes"] > 0
        out["reply_pcm_bytes"] = bytes_after
        # Distinguish "barge-in cut the reply short" from "the reply simply
        # finished": a tts_sentence_done / tts_done after the injection means it
        # ran to completion, so inject_to_last_pcm_ms is NOT a cut latency.
        with reader.lock:
            out["reply_completed"] = any(
                t > t_inject and k in ("tts_sentence_done", "tts_done")
                for t, k in reader.events)

        # ── did the interrupting utterance still transcribe? ───────────────
        send_pcm(ws, silence)
        wait_for(lambda: len(reader.snapshot()[3]) >= 2, budget_s)
        finals = reader.snapshot()[3]
        if len(finals) >= 2:
            out["final_2"] = finals[1][1]
        out["error"] = reader.snapshot()[4]
        return out
    finally:
        reader.closed = True
        try:
            ws.close()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="host:port (no scheme)")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--tts-language", default="auto")
    ap.add_argument("--interrupt-on", choices=["final", "audio"], default="final",
                    help="start the interrupting speech as soon as the transcript "
                         "lands (final, default: guarantees ASR/TTS NPU overlap) or "
                         "once reply audio is flowing (audio: measures cut latency, "
                         "needs a long reply)")
    ap.add_argument("--interrupt-after-ms", type=int, default=0,
                    help="extra delay before speaking over the reply")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--budget-s", type=float, default=30.0,
                    help="per-wait budget for finals / audio")
    args = ap.parse_args()

    wav_pcm, dur = load_wav_16k_mono(args.wav)
    print(f"wav={args.wav} dur={dur:.2f}s  interrupt_after={INTERRUPT_AFTER_MS}ms  "
          f"trials={args.trials}", file=sys.stderr)

    results = []
    for i in range(args.trials):
        r = trial(args.host, wav_pcm, args.language, args.budget_s,
                  tts_language=args.tts_language,
                  interrupt_on=args.interrupt_on,
                  interrupt_after_ms=args.interrupt_after_ms)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))
        _l = r["inject_to_last_pcm_ms"]
        print(f"[trial {i+1}/{args.trials}] overlapped={r['overlapped']} "
              f"inject_to_last_pcm={_l if _l is None else round(_l)}ms "
              f"reply_completed={r['reply_completed']} "
              f"final_2={r['final_2']!r} err={r['error']}", file=sys.stderr)
        # Let the session limiter release the slot before the next trial.
        time.sleep(2.0)

    ok = [r for r in results if r["overlapped"] and r["final_2"]]
    lat = [r["inject_to_last_pcm_ms"] for r in results
           if r["inject_to_last_pcm_ms"] is not None]
    cut = [r for r in results if r["overlapped"] and r["reply_completed"] is False]
    print("\n=== speech-over-reply NPU overlap summary ===", file=sys.stderr)
    print(f"trials={len(results)}  overlapped+transcribed={len(ok)}  "
          f"replies actually cut short={len(cut)}", file=sys.stderr)
    if lat:
        lat.sort()
        print(f"inject -> last PCM: p50={lat[len(lat)//2]:.0f}ms "
              f"min={lat[0]:.0f}ms max={lat[-1]:.0f}ms  "
              f"(a CUT latency only for trials where reply_completed is false)",
              file=sys.stderr)
    for r in results:
        if r["error"]:
            print(f"  error: {r['error']}", file=sys.stderr)
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
