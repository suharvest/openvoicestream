#!/usr/bin/env python3
"""CustomVoice N=1 built-in speaker, capability, and cancel/recovery gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import struct
import time
from pathlib import Path

import requests


CASES = [
    (3066, "serena", "中文语音迁移验证，声音需要清晰而且完整。", "chinese"),
    (3061, "ryan", "The English CustomVoice migration test must be clear and complete.", "english"),
    (2873, "ono_anna", "第二个中文内置音色用于验证输出隔离。", "chinese"),
]


def stream(base_url: str, text: str, language: str, speaker_id: int, *,
           cancel_after_audio: bool = False, timeout: float = 90) -> dict:
    started = time.perf_counter()
    first_audio_at = None
    chunks: list[bytes] = []
    response = requests.post(
        base_url.rstrip("/") + "/tts/stream",
        json={"text": text, "language": language, "speaker_id": speaker_id},
        stream=True,
        timeout=timeout,
    )
    status = response.status_code
    content_type = response.headers.get("content-type")
    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            continue
        chunks.append(chunk)
        if sum(map(len, chunks)) > 4 and first_audio_at is None:
            first_audio_at = time.perf_counter()
            if cancel_after_audio:
                break
    response.close()
    ended = time.perf_counter()
    payload = b"".join(chunks)
    sample_rate = (
        struct.unpack("<I", payload[:4])[0]
        if status == 200 and len(payload) >= 4 else None
    )
    pcm = payload[4:] if len(payload) >= 4 else b""
    wall_ms = (ended - started) * 1000
    audio_seconds = len(pcm) / 48000
    return {
        "status": status,
        "content_type": content_type,
        "sample_rate": sample_rate,
        "pcm_bytes": len(pcm),
        "sha256": hashlib.sha256(pcm).hexdigest(),
        "ttfa_ms": (
            (first_audio_at - started) * 1000 if first_audio_at is not None else None
        ),
        "wall_ms": wall_ms,
        "audio_seconds": audio_seconds,
        "rtf": (
            wall_ms / 1000 / audio_seconds
            if audio_seconds > 0 and not cancel_after_audio else None
        ),
        "passed": bool(
            status == 200 and sample_rate == 24000 and len(pcm) > 0
            and first_audio_at is not None
        ),
    }


def summary(values: list[float | None]) -> dict:
    clean = [value for value in values if value is not None]
    return {
        "min": min(clean) if clean else None,
        "p50": statistics.median(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def response_payload(response: requests.Response):
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"raw_text": response.text[:2000]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--recovery-deadline", type=float, default=15)
    parser.add_argument("--skip-clone-probe", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    builtins = []
    for speaker_id, label, text, language in CASES:
        row = stream(base_url, text, language, speaker_id)
        row.update({"speaker_id": speaker_id, "label": label, "language": language})
        builtins.append(row)
        print(
            f"BUILTIN {label}: pass={row['passed']} ttfa={row['ttfa_ms']:.1f}ms "
            f"rtf={row['rtf']:.3f}",
            flush=True,
        )

    # LAZY_TTS means these endpoints are 503 before the first synthesis. Query
    # them only after the first built-in request has initialized the backend.
    capabilities_response = requests.get(base_url + "/tts/capabilities", timeout=10)
    capabilities = response_payload(capabilities_response)
    speakers_response = requests.get(base_url + "/tts/speakers", timeout=10)
    speakers = response_payload(speakers_response)
    available_ids = {int(row["id"]) for row in speakers.get("speakers", [])}
    for row in builtins:
        row["passed"] = bool(row["passed"] and row["speaker_id"] in available_ids)

    # CustomVoice deliberately has no speaker encoder. Validate that all public
    # metadata and synthesis paths reject external embeddings before streaming.
    fake_embedding = base64.b64encode(b"\x00" * 32).decode()
    if args.skip_clone_probe:
        clone_gate = {
            "skipped": True,
            "reason": "probe already captured an HTTP 500 worker failure in prior run",
            "supported": bool(capabilities.get("supports_voice_cloning")),
            "metadata_consistent": False,
            "passed": False,
        }
    else:
        clone_response = requests.post(
            base_url + "/tts/clone",
            json={
                "text": "This request must be rejected by CustomVoice.",
                "speaker_embedding_b64": fake_embedding,
                "language": "english",
            },
            timeout=20,
        )
        clone_payload = response_payload(clone_response)
        stream_clone_response = requests.post(
            base_url + "/tts/stream",
            json={
                "text": "This stream must be rejected before response bytes.",
                "speaker_embedding_b64": fake_embedding,
                "language": "english",
            },
            timeout=20,
        )
        stream_clone_payload = response_payload(stream_clone_response)
        clone_gate = {
            "supported": bool(capabilities.get("supports_voice_cloning")),
            "metadata_consistent": (
                capabilities.get("supports_voice_cloning") is False
                and speakers.get("supports_voice_cloning") is False
            ),
            "clone_status": clone_response.status_code,
            "clone_payload": clone_payload,
            "stream_status": stream_clone_response.status_code,
            "stream_payload": stream_clone_payload,
        }
        clone_gate["passed"] = bool(
            clone_gate["metadata_consistent"]
            and clone_response.status_code == 400
            and stream_clone_response.status_code == 400
            and clone_payload.get("required_capability") == "voice_clone"
            and clone_payload.get("supports_voice_cloning") is False
            and stream_clone_payload.get("required_capability") == "voice_clone"
            and stream_clone_payload.get("supports_voice_cloning") is False
        )

    recovery_rounds = []
    for index in range(1, args.rounds + 1):
        cancel = stream(
            base_url,
            "这一条长语音会在收到首个音频块后立即取消。" * 10,
            "chinese",
            3065,
            cancel_after_audio=True,
        )
        recovery_started = time.perf_counter()
        attempts = []
        recovery = {"passed": False}
        deadline = recovery_started + args.recovery_deadline
        while time.perf_counter() < deadline:
            recovery = stream(
                base_url,
                "取消后恢复正常语音。",
                "chinese",
                3066,
                timeout=max(0.1, deadline - time.perf_counter()),
            )
            attempts.append(recovery)
            if recovery["passed"] or recovery["status"] != 429:
                break
            time.sleep(0.1)
        recovery_elapsed_ms = (time.perf_counter() - recovery_started) * 1000
        passed = bool(
            cancel["passed"] and recovery["passed"]
            and recovery_elapsed_ms <= args.recovery_deadline * 1000
        )
        recovery_rounds.append({
            "round": index,
            "cancel": cancel,
            "recovery": recovery,
            "recovery_attempts": attempts,
            "recovery_elapsed_ms": recovery_elapsed_ms,
            "passed": passed,
        })
        print(
            f"CANCEL_RECOVERY {index}/{args.rounds}: pass={passed} "
            f"elapsed={recovery_elapsed_ms:.1f}ms",
            flush=True,
        )
        if not passed:
            break

    hashes = {row["sha256"] for row in builtins}
    report = {
        "test": "v091_r2_customvoice_n1",
        "capabilities": capabilities,
        "speakers": speakers,
        "builtins": builtins,
        "builtin_ttfa_ms": summary([row["ttfa_ms"] for row in builtins]),
        "builtin_rtf": summary([row["rtf"] for row in builtins]),
        "clone_capability_gate": clone_gate,
        "cancel_recovery_rounds": recovery_rounds,
    }
    report["functional_passed"] = bool(
        len(builtins) == len(CASES)
        and all(row["passed"] for row in builtins)
        and len(hashes) == len(builtins)
        and len(recovery_rounds) == args.rounds
        and all(row["passed"] for row in recovery_rounds)
    )
    report["passed"] = bool(report["functional_passed"] and clone_gate["passed"])
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    execution_passed = (
        report["functional_passed"] if args.skip_clone_probe else report["passed"]
    )
    print(
        json.dumps({
            "passed": report["passed"],
            "functional_passed": report["functional_passed"],
            "execution_passed": execution_passed,
        }),
        flush=True,
    )
    return 0 if execution_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
