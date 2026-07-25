from __future__ import annotations

import json
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bench.perf import gdn_sse_abort_recovery
from bench.perf import qwen_asr_n2_service
from bench.perf import tts_n2_cancel_isolation


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, _request: object, _client_address: object) -> None:
        # An explicit client disconnect is the behavior under test.
        pass


class _Server:
    def __init__(self, handler: type[BaseHTTPRequestHandler]):
        self.server = _QuietThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, *unused: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class _QuietHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: object) -> None:
        pass

    def _json_body(self) -> dict:
        size = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(size) or b"{}")

    def _chunk(self, value: bytes) -> None:
        self.wfile.write(f"{len(value):x}\r\n".encode() + value + b"\r\n")
        self.wfile.flush()

    def _end_chunks(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class _GdnHandler(_QuietHandler):
    def do_GET(self) -> None:
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        payload = self._json_body()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        count = 8 if payload["max_tokens"] > 100 else 2
        try:
            for index in range(count):
                event = {
                    "id": f"req-{index}",
                    "choices": [{"delta": {"content": f"tok{index}"}}],
                }
                self._chunk(f"data: {json.dumps(event)}\n\n".encode())
                time.sleep(0.003)
            self._chunk(b"data: [DONE]\n\n")
            self._end_chunks()
        except (BrokenPipeError, ConnectionResetError):
            pass


class _TtsHandler(_QuietHandler):
    PCM = b"\x12\x34" * 4096

    def do_POST(self) -> None:
        self._json_body()
        self.send_response(200)
        self.send_header("content-type", "application/octet-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        body = (24000).to_bytes(4, "little") + self.PCM
        try:
            for offset in range(0, len(body), 1024):
                self._chunk(body[offset : offset + 1024])
                time.sleep(0.008)
            self._end_chunks()
        except (BrokenPipeError, ConnectionResetError):
            pass


class _AsrHandler(_QuietHandler):
    active = 0
    lock = threading.Lock()

    def do_POST(self) -> None:
        with self.lock:
            self.__class__.active += 1
            active = self.__class__.active
        try:
            size = int(self.headers.get("content-length", "0"))
            body_in = self.rfile.read(size)
            if active > 2:
                body = b'{"detail":{"error":"too_many_sessions","status":4429}}'
                status = 429
            else:
                time.sleep(0.04)
                text = "hello world" if b"English" in body_in else "ni hao shi jie"
                body = json.dumps({"text": text}).encode()
                status = 200
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with self.lock:
                self.__class__.active -= 1


def _empty_log(tmp_path: Path) -> Path:
    path = tmp_path / "server.log"
    path.write_text("validation interval started\n", encoding="utf-8")
    return path


def test_gdn_abort_immediate_recovery_mock(tmp_path: Path) -> None:
    with _Server(_GdnHandler) as base_url:
        summary, passed = gdn_sse_abort_recovery.run(
            Namespace(
                base_url=base_url,
                model="mock",
                rounds=2,
                abort_after_events=1,
                abort_max_tokens=256,
                recovery_max_tokens=8,
                abort_prompt="count",
                recovery_prompt="recover",
                seed=7,
                health_path="/health",
                timeout=5,
                max_next_delay_ms=100,
                server_log=[_empty_log(tmp_path)],
            )
        )
    assert passed
    assert summary["rounds_ok"] == 2
    assert all(r["abort_to_next_start_ms"] <= 100 for r in summary["rounds"])
    assert all(r["recovery"]["done_seen"] for r in summary["rounds"])


def test_tts_cancel_a_continue_b_mock(tmp_path: Path) -> None:
    with _Server(_TtsHandler) as base_url:
        summary, passed = tts_n2_cancel_isolation.run(
            Namespace(
                base_url=base_url,
                endpoint="/tts/stream",
                rounds=2,
                text_a="cancel A",
                text_b="complete B",
                payload_a_json=None,
                payload_b_json=None,
                cancel_after_pcm_chunks=1,
                min_pcm_bytes=1024,
                require_byte_equal=True,
                timeout=5,
                server_log=[_empty_log(tmp_path)],
                capture_dir=tmp_path / "capture",
            )
        )
    assert passed
    assert summary["rounds_ok"] == 2
    assert all(r["timing"]["cancel_a_during_b"] for r in summary["rounds"])
    assert (tmp_path / "capture" / "round-001-b.bin").stat().st_size > 1024


def test_qwen_asr_n2_distinct_wav_mock(tmp_path: Path) -> None:
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_a.write_bytes(b"RIFFmock-a")
    wav_b.write_bytes(b"RIFFmock-b")
    _AsrHandler.active = 0
    with _Server(_AsrHandler) as base_url:
        summary, passed = qwen_asr_n2_service.run(
            Namespace(
                base_url=base_url,
                endpoint="/asr",
                wav_a=wav_a,
                wav_b=wav_b,
                language_a="Chinese",
                language_b="English",
                expect_a=r"ni hao",
                expect_b=r"hello world",
                rounds=2,
                check_oversubscribe=True,
                timeout=5,
                server_log=[_empty_log(tmp_path)],
            )
        )
    assert passed
    assert summary["rounds_ok"] == 2
    assert summary["saturation"]["rejected_count"] >= 1


def test_log_scans_fail_on_gpu_errors(tmp_path: Path) -> None:
    log = tmp_path / "bad.log"
    log.write_text("TensorRT Error Code 1: CUDA illegal memory access\n")
    assert gdn_sse_abort_recovery.scan_logs([log])["hits"]
    assert tts_n2_cancel_isolation.scan_logs([log])["hits"]
    assert qwen_asr_n2_service.scan_logs([log])["hits"]
