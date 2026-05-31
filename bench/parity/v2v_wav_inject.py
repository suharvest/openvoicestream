#!/usr/bin/env python3
"""WAV-injection harness for SLV /v2v/stream — #37 parity before/after tool.

WHAT THIS DRIVES
----------------
SLV's /v2v/stream (port 8621 on seeed-orin-nx) is an ASR -> (text-in) -> TTS
*service*. It does NOT run an LLM and does NOT register any robot-arm tools.
The VoiceArm agent (voice-arm container, :8765) is a *separate client* that
owns the LLM + tool loop: it connects to SLV /v2v/stream, receives asr_final,
runs edge-llm (:8000) with its 10 arm tools, streams the reply text back into
SLV via CLIENT_TEXT, then CLIENT_TTS_FLUSH to synthesize.

This harness IMPERSONATES that client at the SLV protocol level, but injects
a *fixed deterministic text* for the TTS leg instead of calling the LLM. That
means:
  * It exercises the real ASR -> TTS server path (the SLV half of parity).
  * It NEVER invokes the agent's LLM or tool loop -> structurally CANNOT
    trigger a robot-arm action. (No /actions/{name}/test is ever called.)

It is deterministic and repeatable: same WAV in -> same asr_final out, same
injected text -> same TTS. Perfect for #37 before/after diffing.

PROTOCOL (mirrors agent/openvoicestream_agent/slv_client.py)
------------------------------------------------------------
  1. open WS, send {"type":"config", ...} (multi_utterance, vad as configured)
  2. stream WAV PCM chunks (realtime-paced) + trailing silence
  3. (optional) send {"type":"asr_eos"} to force finalize
  4. receive asr_final  -> record text + t_final  (TTFA clock starts here)
  5. send {"type":"text","text": <fixed reply>} then {"type":"tts_flush"}
  6. receive tts_started / binary PCM / tts_sentence_done / tts_done
     -> first binary PCM byte = TTFA endpoint

MEASUREMENTS (logged + emitted as JSON)
  * asr_final text
  * injected reply text (the "LLM reply" stand-in)
  * tts: started?, total PCM bytes, #sentences, tts_done seen?, sample_rate
  * TTFA_ms = (first TTS PCM byte) - (asr_final)
  * raw event log (parity grep keys: asr_final / tts_started / tts_sentence_done
    / tts_done)

Usage:
  uv run python bench/parity/v2v_wav_inject.py \
      --host 100.111.134.124 --port 8621 \
      --wav bench/perf/corpus/short/zh_short_04.wav \
      --reply "今天是星期五。" \
      --out /tmp/voicearm-baseline/no-arm/run1.json

Guardrail: the --reply text is what gets synthesized. It is NEVER routed to
the agent LLM, so it cannot select an arm tool. Keep it a plain neutral
sentence regardless.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import struct
import sys
import time
import wave

import websockets
from websockets.asyncio.client import connect as ws_connect


# ── Arm-trigger guardrail ─────────────────────────────────────────────
# Mirror of voice-arm /app/default_config/actions.yaml trigger phrases.
# Used ONLY as a defensive lint on the *asr_final* text and the injected
# reply, so a run that accidentally transcribes/echoes an action phrase
# is flagged loudly. (The harness can't actually fire an arm action — it
# never reaches the agent LLM — but we surface it anyway per task spec.)
ARM_TRIGGER_PHRASES = [
    # home / reset
    "回到原位", "复位", "go home", "reset",
    # pick
    "准备抓取", "get ready to pick", "pick position",
    # gripper
    "张开", "松开", "open gripper", "open the hand",
    "闭合", "抓紧", "close gripper", "grab", "hold",
    # look
    "抬头", "向上看", "look up", "低头", "向下看", "look down",
    # gestures
    "打招呼", "挥手", "wave", "say hi",
    "点头", "同意", "nod", "say yes",
    "摇头", "不行", "shake head", "say no",
]


def scan_arm_triggers(text: str) -> list[str]:
    t = (text or "").lower()
    return [p for p in ARM_TRIGGER_PHRASES if p.lower() in t]


def load_wav(path: str) -> tuple[bytes, int, float]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        pcm = wf.readframes(n)
    if ch != 1 or sw != 2:
        raise SystemExit(f"WAV must be mono 16-bit, got ch={ch} sw={sw}")
    return pcm, sr, n / sr


def pcm_chunks(pcm: bytes, sr: int, chunk_ms: int) -> list[bytes]:
    frame_bytes = 2  # mono int16
    chunk_frames = int(sr * chunk_ms / 1000)
    chunk_bytes = chunk_frames * frame_bytes
    return [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]


async def run(args) -> dict:
    pcm, sr, dur = load_wav(args.wav)
    chunks = pcm_chunks(pcm, sr, args.chunk_ms)
    chunk_dt = args.chunk_ms / 1000.0
    ws_url = f"ws://{args.host}:{args.port}/v2v/stream"

    events: list[dict] = []
    t0 = time.monotonic()

    def log(kind: str, **kw):
        events.append({"t_ms": round((time.monotonic() - t0) * 1000, 1), "kind": kind, **kw})

    result: dict = {
        "wav": os.path.basename(args.wav),
        "wav_dur_s": round(dur, 3),
        "ws_url": ws_url,
        "reply_injected": args.reply,
        "asr_final": None,
        "asr_language": None,
        "tts_started": False,
        "tts_sentence_done_count": 0,
        "tts_done": False,
        "tts_pcm_bytes": 0,
        "tts_sample_rate": None,
        "ttfa_ms": None,
        "asr_eos_to_final_ms": None,
        "arm_trigger_hits": [],
        "error": None,
    }

    # proxy=None: bypass any ALL_PROXY/SOCKS env on the dev machine — this is
    # a direct Tailscale/LAN connection to the device, not internet egress.
    try:
        _cm = ws_connect(ws_url, max_size=None, proxy=None)
    except TypeError:
        # Older websockets without the proxy kwarg.
        _cm = ws_connect(ws_url, max_size=None)
    async with _cm as ws:
        config = {
            "type": "config",
            "asr_language": args.asr_language,
            "tts_language": args.tts_language,
            "vad": args.vad,
            "sample_rate": sr,
            "multi_utterance": False,
        }
        if args.vad != "none":
            config["vad_silence_ms"] = args.vad_silence_ms
        await ws.send(json.dumps(config))
        log("config_sent", config=config)

        t_final: float | None = None
        t_eos: float | None = None
        t_first_pcm: float | None = None
        tts_sr_seen = False

        async def reader():
            nonlocal t_final, t_first_pcm, tts_sr_seen
            try:
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray)):
                        data = bytes(msg)
                        # First binary frame is <little-endian uint32 sr><pcm>.
                        if not tts_sr_seen:
                            if len(data) >= 4:
                                (srx,) = struct.unpack("<I", data[:4])
                                result["tts_sample_rate"] = srx
                                pcmpart = data[4:]
                            else:
                                pcmpart = b""
                            tts_sr_seen = True
                            if pcmpart and t_first_pcm is None:
                                t_first_pcm = time.monotonic()
                            result["tts_pcm_bytes"] += len(pcmpart)
                            log("tts_pcm_first", bytes=len(pcmpart))
                        else:
                            if t_first_pcm is None:
                                t_first_pcm = time.monotonic()
                            result["tts_pcm_bytes"] += len(data)
                            log("tts_pcm", bytes=len(data))
                        continue
                    evt = json.loads(msg)
                    et = evt.get("type")
                    if et == "asr_partial":
                        log("asr_partial", text=evt.get("text", ""))
                    elif et == "asr_endpoint":
                        log("asr_endpoint")
                    elif et == "asr_final":
                        t_final = time.monotonic()
                        txt = evt.get("text", "")
                        result["asr_final"] = txt
                        result["asr_language"] = evt.get("language")
                        log("asr_final", text=txt, language=evt.get("language"))
                        return  # hand back control to inject text
                    elif et == "tts_started":
                        result["tts_started"] = True
                        log("tts_started", sentence=evt.get("sentence", "")[:120])
                    elif et == "tts_sentence_done":
                        result["tts_sentence_done_count"] += 1
                        log("tts_sentence_done", sentence=evt.get("sentence", "")[:120])
                    elif et == "tts_done":
                        result["tts_done"] = True
                        log("tts_done", session_complete=evt.get("session_complete"))
                        return
                    elif et == "error":
                        result["error"] = evt.get("error", "unknown")
                        log("error", error=result["error"])
                        return
                    else:
                        log("other", type=et)
            except websockets.ConnectionClosed as e:
                log("ws_closed", reason=str(e))

        # Stage 1: stream WAV (realtime paced).
        for c in chunks:
            await ws.send(c)
            if args.realtime:
                await asyncio.sleep(chunk_dt)
        # Stage 2: trailing silence so VAD can endpoint (when vad != none).
        sil_ms = max(args.vad_silence_ms + args.chunk_ms, 600)
        sil = b"\x00\x00" * int(sr * args.chunk_ms / 1000)
        for _ in range(int(sil_ms / args.chunk_ms)):
            await ws.send(sil)
            if args.realtime:
                await asyncio.sleep(chunk_dt)
        # Stage 3: force finalize (deterministic regardless of VAD config).
        t_eos = time.monotonic()
        await ws.send(json.dumps({"type": "asr_eos"}))
        log("asr_eos_sent")

        # Stage 4: wait for asr_final.
        try:
            await asyncio.wait_for(reader(), timeout=args.timeout)
        except asyncio.TimeoutError:
            result["error"] = "timeout waiting for asr_final"
            return result
        if result["asr_final"] is None:
            if result["error"] is None:
                result["error"] = "no asr_final received"
            return result
        if t_final and t_eos:
            result["asr_eos_to_final_ms"] = round((t_final - t_eos) * 1000, 1)

        # Guardrail lint: would this text resemble an arm trigger?
        hits = scan_arm_triggers(result["asr_final"]) + scan_arm_triggers(args.reply)
        result["arm_trigger_hits"] = sorted(set(hits))

        # Stage 5: inject fixed reply text + flush -> TTS.
        await ws.send(json.dumps({"type": "text", "text": args.reply}))
        log("text_injected", text=args.reply)
        await ws.send(json.dumps({"type": "tts_flush"}))
        log("tts_flush_sent")

        # Stage 6: collect TTS until tts_done.
        try:
            await asyncio.wait_for(reader(), timeout=args.timeout)
        except asyncio.TimeoutError:
            result["error"] = "timeout waiting for tts_done"

        if t_first_pcm and t_final:
            result["ttfa_ms"] = round((t_first_pcm - t_final) * 1000, 1)

    result["events"] = events
    return result


def main():
    ap = argparse.ArgumentParser(description="SLV /v2v/stream WAV-injection harness (#37 parity)")
    ap.add_argument("--host", default="100.111.134.124")
    ap.add_argument("--port", type=int, default=8621)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--reply", required=True, help="Fixed text injected for TTS (LLM stand-in)")
    ap.add_argument("--asr-language", default="auto")
    ap.add_argument("--tts-language", default="zh")
    ap.add_argument("--vad", default="none", help="silero|none (match agent slv_config.vad)")
    ap.add_argument("--vad-silence-ms", type=int, default=400)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--realtime", action="store_true", default=True)
    ap.add_argument("--no-realtime", dest="realtime", action="store_false")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", default=None, help="Write result JSON here")
    args = ap.parse_args()

    res = asyncio.run(run(args))

    # Human summary to stderr; machine JSON to stdout / --out.
    s = sys.stderr
    print("── v2v WAV-inject result ──", file=s)
    print(f"  wav           : {res['wav']} ({res['wav_dur_s']}s)", file=s)
    print(f"  asr_final     : {res['asr_final']!r} (lang={res['asr_language']})", file=s)
    print(f"  reply_injected: {res['reply_injected']!r}", file=s)
    print(f"  tts_started   : {res['tts_started']}", file=s)
    print(f"  tts_sent_done : {res['tts_sentence_done_count']}", file=s)
    print(f"  tts_done      : {res['tts_done']}", file=s)
    print(f"  tts_pcm_bytes : {res['tts_pcm_bytes']} (sr={res['tts_sample_rate']})", file=s)
    print(f"  TTFA_ms       : {res['ttfa_ms']}", file=s)
    print(f"  eos->final_ms : {res['asr_eos_to_final_ms']}", file=s)
    print(f"  arm_triggers  : {res['arm_trigger_hits']}", file=s)
    print(f"  error         : {res['error']}", file=s)

    out = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(out)
        print(f"  -> wrote {args.out}", file=s)
    else:
        print(out)

    if res["error"]:
        sys.exit(2)
    if res["arm_trigger_hits"]:
        print("WARNING: arm-trigger phrase detected in asr_final/reply!", file=s)


if __name__ == "__main__":
    main()
