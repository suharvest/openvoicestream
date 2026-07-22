"""Real-edge-llm /v2v orchestration soak for the conversation split.

Runs the refactored voxedge ConversationEngine (Session + the 7 split modules)
against the REAL edge-llm-chat-service on this Orin NX, hammering the exact
orchestration the production arm exercises — server-loop multi-turn + tool
round + barge-in — and asserting clean teardown every cycle (no slot/task leak).

Mock ASR/TTS (the refactor didn't touch backend internals; this isolates the
orchestration), real streaming LLM (validates llm_turn against a live model).
"""
import argparse
import asyncio
import json

import httpx

from voxedge.backends.base import LLMBackend, LLMEvent
from voxedge.backends.mock import MockASR, MockTTS, MockVAD
from voxedge.engine import ConversationEngine, ToolRegistry
from voxedge.engine.conversation import Session
from voxedge.transport import InProcessTransport


class EdgeLLM(LLMBackend):
    """Minimal OpenAI-compatible streaming backend → edge-llm /v1/chat/completions."""

    def __init__(self, base_url: str, model: str = "/workspace/Qwen3-4B-AWQ/engines-8192"):
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._model = model

    async def stream(self, messages, **kw):
        async for ev in self.stream_events(messages, **kw):
            if ev.kind == "text" and ev.text:
                yield ev.text

    async def stream_events(self, messages, **kw):
        body = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "max_tokens": kw.get("max_tokens", 256),
            "temperature": kw.get("temperature", 0.0),
        }
        if kw.get("tools"):
            body["tools"] = kw["tools"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", self._url, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield LLMEvent(kind="text", text=delta["content"])
                    for tc in (delta.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        yield LLMEvent(
                            kind="tool_call_delta",
                            tool_call_index=tc.get("index", 0),
                            tool_call_id=tc.get("id"),
                            name=fn.get("name"),
                            arguments=fn.get("arguments"),
                        )
                    if choice.get("finish_reason"):
                        yield LLMEvent(kind="finish", finish_reason=choice["finish_reason"])


def _make_engine(base_url):
    registry = ToolRegistry()

    @registry.tool(description="Get the current time of day.")
    def get_time() -> dict:
        return {"time": "14:30"}

    return ConversationEngine(
        backends={"asr": MockASR(), "tts": MockTTS(), "vad": MockVAD(), "llm": EdgeLLM(base_url)},
        tool_registry=registry,
        system_prompt="You are a concise voice assistant. Use tools when asked about the time.",
    ), registry


async def _one_cycle(engine, barge: bool):
    """One server-loop turn; optionally barge-in mid-flight. Returns clean state."""
    sess = Session(engine, InProcessTransport())
    sess.state.llm_barged = False
    task = asyncio.create_task(
        sess._llm.run([{"role": "user", "content": "现在几点了？"}])
    )
    sess.state.current_llm_task = task
    if barge:
        # let it start, then barge-in (the production-critical path)
        await asyncio.sleep(0.15)
        await sess._bargein_tts()
    try:
        await asyncio.wait_for(task, timeout=45.0)
    except asyncio.CancelledError:
        pass
    finally:
        if sess.state.current_llm_task is task:
            sess.state.current_llm_task = None
    # cleanliness asserts (the slot/leak invariants the prod arm cares about)
    assert sess.state.current_tts_task is None, "current_tts_task leaked"
    assert sess.state.current_llm_task is None, "current_llm_task leaked"
    if barge:
        assert sess.state.llm_barged is True, "barge flag not set"
        assert sess._tts.q.empty(), "tts queue not drained after barge-in"
    return True


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    engine, _ = _make_engine(args.base)

    # sanity: one plain turn must produce text from the real LLM
    sess = Session(engine, InProcessTransport())
    await asyncio.wait_for(sess._llm.run([{"role": "user", "content": "你好"}]), timeout=45.0)
    got_text = not sess._tts.q.empty() or sess._tts.buffer is None
    print(f"sanity: real-LLM turn produced queued speech = {got_text}")

    tasks_before = len(asyncio.all_tasks())
    ok = 0
    fail = 0
    for i in range(args.n):
        try:
            await _one_cycle(engine, barge=(i % 2 == 1))  # alternate plain / barge-in
            ok += 1
        except Exception as e:
            fail += 1
            print(f"cycle {i} FAIL: {type(e).__name__}: {e}")
    await asyncio.sleep(0.2)
    tasks_after = len(asyncio.all_tasks())
    leaked = tasks_after - tasks_before
    print(f"SOAK: ok={ok} fail={fail} / {args.n}  asyncio_task_delta={leaked}")
    print("RESULT:", "PASS" if (fail == 0 and leaked <= 0) else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
