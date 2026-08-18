"""POST /asr must not block the event loop while the backend decodes.

``_asr_impl`` used to call ``backend.transcribe()`` synchronously inside the
coroutine, so a multi-second (or multi-minute) offline decode froze the whole
event loop: /readyz and /metrics stopped answering and the docker healthcheck
(interval 30s / timeout 5s / retries 3) eventually flagged the container
unhealthy mid-transcription.

The decode is now submitted to the shared ASR executor. These tests pin that
behaviour with a fake backend whose ``transcribe`` does ``time.sleep(0.5)``:
while the /asr request is in flight, /readyz must still answer promptly.

Reuses the fake-backend manager harness from ``test_main_hot_swap`` (no GPU,
no real models).
"""
from __future__ import annotations

import asyncio
import io
import time
import wave

import httpx
import pytest

from server.tests.test_main_hot_swap import _FakeASRBackend, _install_managers

DECODE_SECONDS = 0.5
# Generous vs. DECODE_SECONDS but far below it: a blocked loop would make the
# probe wait out the whole decode, so any value well under DECODE_SECONDS
# separates "executor" from "inline call" without being flaky on slow CI.
PROBE_BUDGET_SECONDS = 0.25


class _FakeResult:
    def __init__(self, text: str, language: str):
        self.text = text
        self.language = language
        self.meta: dict = {}


class _SlowASRBackend(_FakeASRBackend):
    """Fake ASR backend whose decode blocks the calling thread."""

    name = "slow-fake"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> _FakeResult:
        self.calls.append({"language": language, "n_bytes": len(audio_bytes)})
        time.sleep(DECODE_SECONDS)
        return _FakeResult(text="慢速转写", language="zh")


def _make_wav(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """PCM16 mono silence WAV of the given duration (stdlib only)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return buf.getvalue()


@pytest.fixture
def slow_backend(monkeypatch):
    monkeypatch.delenv("OVS_API_KEYS", raising=False)
    asr_be = _SlowASRBackend()
    _install_managers(asr=asr_be)
    return asr_be


def _client():
    from server.main import app
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


@pytest.mark.anyio
async def test_readyz_responds_while_asr_decodes(slow_backend):
    """/readyz answers promptly while a 0.5s /asr decode is in flight."""
    async with _client() as client:
        asr_task = asyncio.create_task(
            client.post(
                "/asr",
                files={"file": ("audio.wav", _make_wav(), "audio/wav")},
            )
        )
        # Let /asr reach the blocking decode before probing.
        await asyncio.sleep(0.05)
        assert not asr_task.done(), "decode finished too fast to prove anything"

        t0 = time.perf_counter()
        probe = await client.get("/readyz")
        elapsed = time.perf_counter() - t0

        # /readyz may legitimately be 503 under the fake wiring; what matters
        # is that it was *served* rather than starved behind the decode.
        assert probe.status_code in (200, 503), probe.text
        assert elapsed < PROBE_BUDGET_SECONDS, (
            f"/readyz took {elapsed:.3f}s during ASR decode — event loop blocked"
        )
        assert not asr_task.done(), "decode already finished; probe proves nothing"

        asr_resp = await asr_task

    assert asr_resp.status_code == 200, asr_resp.text
    assert asr_resp.json()["text"] == "慢速转写"
    assert len(slow_backend.calls) == 1


@pytest.mark.anyio
async def test_asr_response_shape_unchanged(slow_backend):
    """Executor submission preserves the payload and forwards ``language``."""
    async with _client() as client:
        r = await client.post(
            "/asr",
            params={"language": "zh"},
            files={"file": ("audio.wav", _make_wav(), "audio/wav")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "慢速转写"
    assert body["language"] == "zh"
    assert body["backend"] == "slow-fake"
    assert slow_backend.calls[0]["language"] == "zh"


@pytest.mark.anyio
async def test_concurrent_readyz_probes_all_fast(slow_backend):
    """Repeated probes during one decode all stay inside the budget."""
    async with _client() as client:
        asr_task = asyncio.create_task(
            client.post(
                "/asr",
                files={"file": ("audio.wav", _make_wav(), "audio/wav")},
            )
        )
        await asyncio.sleep(0.05)

        latencies: list[float] = []
        for _ in range(3):
            t0 = time.perf_counter()
            resp = await client.get("/readyz")
            latencies.append(time.perf_counter() - t0)
            assert resp.status_code in (200, 503)

        assert max(latencies) < PROBE_BUDGET_SECONDS, latencies
        # All three probes were served before the decode finished — impossible
        # if the decode ran inline on the event loop.
        assert not asr_task.done(), "decode finished before the probes landed"
        await asr_task
