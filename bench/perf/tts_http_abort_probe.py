#!/usr/bin/env python3
"""Force an HTTP RST after the first TTS PCM byte.

Used to verify that an abrupt streaming client disconnect reaches the
cooperative WorkerIO cancel path instead of leaving synthesis in flight.
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import struct
import time
from urllib.parse import urlparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8622")
    parser.add_argument("--text", default="今天天气真不错，适合出门散步。")
    args = parser.parse_args()

    parsed = urlparse(args.base_url)
    body = json.dumps({"text": args.text}, ensure_ascii=False).encode("utf-8")
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
    started = time.perf_counter()
    conn.request(
        "POST",
        "/tts/stream",
        body=body,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Connection": "close",
        },
    )
    response = conn.getresponse()
    prefix_and_pcm = response.read(5)
    ttfa_ms = (time.perf_counter() - started) * 1000

    sock = conn.sock
    if sock is not None:
        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_LINGER,
            struct.pack("ii", 1, 0),
        )
        sock.close()
    conn.close()
    print(
        json.dumps(
            {
                "status": response.status,
                "bytes_read": len(prefix_and_pcm),
                "ttfa_ms": ttfa_ms,
                "reset": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
