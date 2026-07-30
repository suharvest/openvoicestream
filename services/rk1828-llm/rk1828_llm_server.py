"""OpenAI-compatible streaming HTTP shim for Qwen3-4B on the RK1828 PCIe EP.

Owns one persistent ``rknn_qwen3_demo`` server-mode subprocess (init once) and
exposes ``POST /v1/chat/completions`` (SSE streaming + non-streaming) and
``GET /v1/models``.

Worker IPC protocol (see examples/Qwen3/cpp/main.cc):
  stderr : "READY 1" handshake once model init completes; all diagnostics.
  stdin  : one request line per turn: ``<max_new_tokens>\\t<escaped prompt>``
  stdout : per token ``[uint32 LE len][utf8 bytes]``; ``0xFFFFFFFE`` = EOS.

The EP is a single-context device: every request is serialised on a lock.
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import os
import queue
import re
import struct
import subprocess
import threading
import time
import uuid
from typing import Iterator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

LOG = logging.getLogger("rk1828-llm")

END_OF_STREAM = 0xFFFFFFFE
_LEN = struct.Struct("<I")
MAX_FRAME_BYTES = 8 * 1024 * 1024  # a bigger length prefix means stdout desync
MODEL_ID = "Qwen3-4B"

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class WorkerError(RuntimeError):
    pass


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


class Qwen3Worker:
    def __init__(
        self,
        binary: str,
        model_dir: str,
        core_mask: str = "ff",
        max_context: int = 2048,
        start_attempts: int = 3,
        ready_timeout: float = 180.0,
    ) -> None:
        self.binary = binary
        self.model_dir = model_dir
        self.core_mask = core_mask
        self.max_context = max_context
        self.start_attempts = start_attempts
        self.ready_timeout = ready_timeout
        self.proc: Optional[subprocess.Popen] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self.last_stderr: List[str] = []

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        last: Optional[BaseException] = None
        for attempt in range(1, self.start_attempts + 1):
            try:
                self._spawn()
                LOG.info("worker ready (attempt %d)", attempt)
                return
            except BaseException as exc:  # noqa: BLE001
                last = exc
                LOG.error("worker start attempt %d failed: %s", attempt, exc)
                self.stop()
                if attempt < self.start_attempts:
                    time.sleep(2.0 * attempt)
        raise WorkerError(
            f"RK1828 worker failed to start after {self.start_attempts} attempts: {last} "
            "-- EP may be degraded; a clean host reboot is likely required"
        )

    def _spawn(self) -> None:
        self._ready.clear()
        env = dict(os.environ)
        libdir = os.path.join(os.path.dirname(self.binary), "lib")
        env["LD_LIBRARY_PATH"] = f"{libdir}:/lib:" + env.get("LD_LIBRARY_PATH", "")
        args = [
            self.binary,
            self.model_dir,
            "--core-mask",
            self.core_mask,
            "--max-context",
            str(self.max_context),
            "-",
        ]
        LOG.info("spawning worker: %s", " ".join(args))
        self.proc = subprocess.Popen(
            args,
            cwd=os.path.dirname(self.binary),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="worker-stderr", daemon=True
        )
        self._stderr_thread.start()

        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if self._ready.wait(0.5):
                return
            if self.proc.poll() is not None:
                raise WorkerError(
                    f"worker exited during init rc={self.proc.returncode} "
                    f"tail={self.last_stderr[-6:]}"
                )
        raise WorkerError(f"worker READY handshake timed out tail={self.last_stderr[-6:]}")

    def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            self.last_stderr.append(line)
            if len(self.last_stderr) > 200:
                del self.last_stderr[:100]
            # Case-INsensitive: the TTS binary emits "ready", this one "READY 1".
            if "ready" in line.lower():
                self._ready.set()
            LOG.info("[worker] %s", line)

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        self._ready.clear()
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()

    def is_ready(self) -> bool:
        return (
            self.proc is not None
            and self.proc.poll() is None
            and self._ready.is_set()
        )

    # ── framed IO ────────────────────────────────────────────────────────
    def _read_exact(self, n: int) -> bytes:
        assert self.proc and self.proc.stdout
        buf = b""
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise WorkerError(
                    f"worker stdout EOF (rc={self.proc.returncode}) "
                    f"tail={self.last_stderr[-6:]}"
                )
            buf += chunk
        return buf

    def _run_request(self, prompt: str, max_new_tokens: int, q: "queue.Queue") -> None:
        """Drive one request to its EOS frame, pushing pieces into ``q``.

        Runs on its own thread and owns the worker lock for the whole request.
        Decoupling it from the HTTP response iterator is load-bearing: if the
        client disconnects mid-stream (``curl | head``), this thread still reads
        every remaining frame through the EOS sentinel before releasing the lock.
        Abandoning a half-read request would leave the next one reading the
        previous request's tokens (desync) or block forever on the lock.
        """
        try:
            with self._lock:
                if not self.is_ready():
                    raise WorkerError("worker is not ready")
                assert self.proc and self.proc.stdin
                line = f"{max_new_tokens}\t{_escape(prompt)}\n".encode("utf-8")
                self.proc.stdin.write(line)
                self.proc.stdin.flush()

                dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    (length,) = _LEN.unpack(self._read_exact(4))
                    if length == END_OF_STREAM:
                        tail = dec.decode(b"", final=True)
                        if tail:
                            q.put(("text", tail))
                        break
                    if length > MAX_FRAME_BYTES:
                        raise WorkerError(
                            f"frame length {length} > MAX_FRAME_BYTES={MAX_FRAME_BYTES}: "
                            "stdout desync (stray text on the frame channel)"
                        )
                    piece = dec.decode(self._read_exact(length))
                    if piece:
                        q.put(("text", piece))
        except BaseException as exc:  # noqa: BLE001
            LOG.exception("request failed")
            q.put(("error", str(exc)))
        finally:
            q.put(("done", None))

    def generate(
        self, prompt: str, max_new_tokens: int, timeout: float = 600.0
    ) -> Iterator[str]:
        """Serialised streaming generation. Yields decoded text pieces."""
        q: "queue.Queue" = queue.Queue()
        threading.Thread(
            target=self._run_request, args=(prompt, max_new_tokens, q), daemon=True
        ).start()
        deadline = time.time() + timeout
        while True:
            try:
                kind, payload = q.get(timeout=max(1.0, deadline - time.time()))
            except queue.Empty:
                raise WorkerError(f"request timed out after {timeout}s")
            if kind == "text":
                yield payload
            elif kind == "error":
                raise WorkerError(payload)
            else:
                return


# ── prompt assembly ──────────────────────────────────────────────────────
# The RKNN3 runtime applies the Qwen3 ChatML template itself (verified: an
# 11-token user string prefills as 24 tokens) and `enable_thinking=false` is set
# on the C++ side, so no <|im_start|> tags are injected here.  Multi-turn
# history / system prompts are flattened into the single prompt string.
def build_prompt(messages: List[dict]) -> str:
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    system = [m for m in messages if m.get("role") == "system"]
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not turns:
        raise HTTPException(status_code=400, detail="no user/assistant messages")

    parts: List[str] = []
    for m in system:
        parts.append(str(m.get("content") or "").strip())
    if len(turns) == 1:
        parts.append(str(turns[0].get("content") or ""))
    else:
        for m in turns[:-1]:
            tag = "User" if m.get("role") == "user" else "Assistant"
            parts.append(f"{tag}: {str(m.get('content') or '').strip()}")
        parts.append(str(turns[-1].get("content") or ""))
    return "\n\n".join(p for p in parts if p)


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[dict]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None


app = FastAPI(title="RK1828 Qwen3-4B OpenAI shim")
WORKER: Optional[Qwen3Worker] = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if (WORKER and WORKER.is_ready()) else "unavailable",
        "model": MODEL_ID,
        "device": "rk1828",
    }


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "rk1828",
            }
        ],
    }


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if WORKER is None or not WORKER.is_ready():
        raise HTTPException(status_code=503, detail="RK1828 worker not ready")

    prompt = build_prompt(req.messages)
    max_new = int(req.max_tokens or 512)
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if not req.stream:
        text = "".join(WORKER.generate(prompt, max_new))
        text = _THINK_RE.sub("", text)
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    def event_stream() -> Iterator[str]:
        yield _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )
        n = 0
        try:
            for piece in WORKER.generate(prompt, max_new):
                # Defensive: the runtime has enable_thinking=false, but drop any
                # literal think tags rather than surfacing them to the client.
                if piece in ("<think>", "</think>"):
                    continue
                n += 1
                yield _sse(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("generation failed")
            yield _sse({"error": {"message": str(exc), "type": "worker_error"}})
        yield _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": n,
                    "total_tokens": n,
                },
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--binary",
        default="/home/radxa/rk1828/rknn3-model-zoo/install/rk3588_linux_aarch64/"
        "rknn_Qwen3_demo/rknn_qwen3_demo",
    )
    ap.add_argument(
        "--model-dir",
        default="/home/radxa/rk1828/rknn3-model-zoo/install/rk3588_linux_aarch64/"
        "rknn_Qwen3_demo/model",
    )
    ap.add_argument("--core-mask", default="ff")
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1828)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    global WORKER
    WORKER = Qwen3Worker(
        binary=args.binary,
        model_dir=args.model_dir,
        core_mask=args.core_mask,
        max_context=args.max_context,
    )
    WORKER.start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
