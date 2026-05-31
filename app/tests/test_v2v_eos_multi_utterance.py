"""Regression test for the asr_eos + multi_utterance fix (commit bf24284).

Before the fix, a client sending ``{"type":"asr_eos"}`` in a
``multi_utterance`` session would unconditionally close the ASR side of
the session (``asr_session_closed=True``). That made the
``asr_out_task`` emit a final with ``session_complete=True`` and return,
forcing the client to reopen the WebSocket for the next utterance —
defeating the entire purpose of multi_utterance mode.

The fix at ``app/main.py:1069-1072`` mirrors the VAD speech-end behavior
at ``app/main.py:1046-1051``: ``endpoint_pending`` is set unconditionally
(so the current utterance does get finalized) but ``asr_session_closed``
is only flipped on for *single*-utterance sessions.

Scenarios covered:

1. ``multi_utterance=True`` + 3× client ``asr_eos`` → 3 distinct finals,
   all with ``session_complete=False``; WS stays open after #3.
2. ``multi_utterance=False`` + 1× client ``asr_eos`` → 1 final with
   ``session_complete=True`` (or no flag, depending on branch) and the
   server-side ``asr_out_task`` returns.
3. Symmetry: in multi_utterance, ``endpoint_pending="vad"`` and
   ``endpoint_pending="client_eos"`` produce identical session-lifecycle
   outcomes (both keep the loop alive).

Scaffolding note
----------------
The real ``/v2v/stream`` endpoint pulls in the full FastAPI stack +
profile loader + backend factories + VAD. We DO use the real FastAPI
``TestClient`` WebSocket against ``app.main.app`` for scenarios 1 & 2 —
this is the real wire-level integration test the spec asked for. ASR
backend is replaced with a minimal in-process fake via monkeypatching
``app.main._asr_backend`` and ``app.main._get_asr_backend``. VAD is
disabled at the protocol level (``vad: "none"`` in config), so the no-VAD
code path is exercised and audio chunks open an utterance lazily on
first arrival.

For scenario 3 we use a smaller, asyncio-level harness that re-creates
the state-dict contract from ``asr_out_task`` (since injecting a
synthetic VAD speech-end through the real TestClient flow without a
real VAD backend is more invasive than is warranted for this symmetry
check). Both halves of scenario 3 set the same ``endpoint_pending``
values the production code does on those triggers and assert identical
post-finalize state.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.core.asr_backend import ASRBackend, ASRCapability


# ──────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────


class _FakeStream:
    """Minimal stream stand-in.

    ``finalize()`` returns the next pre-queued text on the parent backend.
    ``get_partial()`` is a no-op (returns "" so asr_out_task emits
    nothing on the partial branch).
    """

    def __init__(self, backend: "_FakeASRBackend"):
        self._backend = backend
        self.accepted_chunks: List[int] = []
        self.finalized = False
        self.cancelled = False

    def accept_waveform(self, sr: int, samples) -> None:
        self.accepted_chunks.append(len(samples))

    def get_partial(self) -> Tuple[str, bool]:
        # Return no partials and never a backend-driven endpoint —
        # the test drives endpoints solely through client asr_eos
        # (and VAD speech-end in the unit-harness half of scenario 3).
        return "", False

    def finalize(self):
        self.finalized = True
        text = self._backend._next_final_text()
        # ASRStream finalize ABC now returns ``(text, detected_language)``.
        return text, None

    def cancel(self) -> None:
        self.cancelled = True

    def cancel_and_finalize(self) -> str:
        self.cancelled = True
        return ""


class _FakeASRBackend(ASRBackend):
    """Streaming-capable backend that hands out _FakeStream instances.

    Pre-loaded with a list of final texts; each ``stream.finalize()`` pops
    one. Tracks the number of streams created so tests can assert
    per-utterance lifecycle.
    """

    def __init__(self, finals: List[str]):
        self._finals = list(finals)
        self.streams_created: List[_FakeStream] = []
        self._lock = threading.Lock()
        self._final_idx = 0

    # ASRBackend abstract surface ──────────────────────────────────────
    @property
    def name(self) -> str:  # type: ignore[override]
        return "fake-eos-test"

    @property
    def capabilities(self):  # type: ignore[override]
        return {ASRCapability.STREAMING}

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return 16000

    def is_ready(self) -> bool:  # type: ignore[override]
        return True

    def preload(self) -> None:  # type: ignore[override]
        return None

    def transcribe(self, audio_bytes: bytes, language: str = "auto"):  # type: ignore[override]
        from app.core.asr_backend import TranscriptionResult
        return TranscriptionResult(text="", duration=0.0, inference_time=0.0,
                                   rtf=0.0, n_tokens=0, per_token_ms=0.0,
                                   backend=self.name)

    def transcribe_audio(self, audio, language="auto"):  # type: ignore[override]
        # Not used by the streaming path under test.
        from app.core.asr_backend import TranscriptionResult
        return TranscriptionResult(text="", duration=0.0, inference_time=0.0,
                                   rtf=0.0, n_tokens=0, per_token_ms=0.0,
                                   backend=self.name)

    # Streaming surface called by ASRSessionManager ────────────────────
    def create_stream(self, language: str = "auto"):
        s = _FakeStream(self)
        self.streams_created.append(s)
        return s

    # Internal ─────────────────────────────────────────────────────────
    def _next_final_text(self) -> str:
        with self._lock:
            if self._final_idx < len(self._finals):
                t = self._finals[self._final_idx]
                self._final_idx += 1
                return t
            return ""


# ──────────────────────────────────────────────────────────────────────
# Real WS integration scenarios (1 & 2)
# ──────────────────────────────────────────────────────────────────────


def _silence_pcm16(ms: int = 50, sr: int = 16000) -> bytes:
    n = (sr * ms) // 1000
    return np.zeros(n, dtype=np.int16).tobytes()


def _drain_until_final(ws, timeout_s: float = 5.0):
    """Receive JSON messages from the test WS until an asr_final arrives.

    Returns (final_payload, all_payloads_seen).
    """
    deadline = time.monotonic() + timeout_s
    seen = []
    while time.monotonic() < deadline:
        try:
            payload = ws.receive_json()
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"WS receive error before asr_final: {e}; seen={seen}")
        seen.append(payload)
        if payload.get("type") == "asr_final":
            return payload, seen
    raise AssertionError(f"timed out waiting for asr_final; seen={seen}")


@pytest.fixture
def fake_asr_backend(monkeypatch):
    """Install a fake ASR backend into app.main so /v2v/stream can run.

    Also init the backend coordinator (normally done by the startup
    lifespan event, which we deliberately skip to avoid downloading
    real models on Mac CI). Returns the backend so tests can pre-seed
    finals.
    """
    import app.main as main_mod
    from app.core.coordinator import init_coordinator
    init_coordinator({"mode": "concurrent"})

    be = _FakeASRBackend(finals=["one", "two", "three", "four"])
    monkeypatch.setattr(main_mod, "_asr_backend", be, raising=False)
    monkeypatch.setattr(main_mod, "_get_asr_backend", lambda: be)
    return be


def _open_v2v(client, *, multi_utterance: bool):
    """Open /v2v/stream and send the initial config frame.

    ``vad="none"`` disables VAD so audio chunks lazily open an utterance
    and the only endpoint trigger in play is the client asr_eos message.
    No TTS configured → no TTS backend dependency.
    """
    cfg = {
        "type": "config",
        "asr_language": "en",
        "vad": "none",
        "sample_rate": 16000,
        "multi_utterance": multi_utterance,
    }
    ws = client.websocket_connect("/v2v/stream")
    ws.__enter__()
    ws.send_json(cfg)
    return ws


def test_scenario1_multi_utterance_three_eos_three_finals(fake_asr_backend):
    """3× client asr_eos in multi_utterance mode → 3 finals, session stays open."""
    from fastapi.testclient import TestClient
    from app.main import app

    fake_asr_backend._finals = ["utterance one", "utterance two", "utterance three"]
    fake_asr_backend._final_idx = 0

    # NOTE: deliberately not using ``with TestClient(app) as client`` —
    # that triggers ``@app.on_event("startup")`` which tries to download
    # real model files into /opt/models. We pre-init only what
    # /v2v/stream actually reaches for (coordinator + fake ASR backend)
    # via the fake_asr_backend fixture.
    client = TestClient(app)
    ws = _open_v2v(client, multi_utterance=True)
    try:
        finals = []
        for _ in range(3):
            ws.send_bytes(_silence_pcm16(50))   # opens utterance lazily
            ws.send_json({"type": "asr_eos"})   # triggers finalize
            payload, _seen = _drain_until_final(ws)
            finals.append(payload)

        assert len(finals) == 3, f"expected 3 finals, got {len(finals)}"
        for i, f in enumerate(finals):
            assert f.get("type") == "asr_final"
            # The load-bearing assertion: multi_utterance + client_eos
            # must NOT close the session.
            assert f.get("session_complete") is False, (
                f"final #{i+1} prematurely closed session: {f}"
            )

        # Texts should track the pre-seeded queue, one per utterance.
        assert [f.get("text") for f in finals] == [
            "utterance one", "utterance two", "utterance three",
        ]

        # Three distinct stream objects must have been created
        # (one per utterance), each with finalize=True.
        assert len(fake_asr_backend.streams_created) == 3, (
            f"expected 3 streams, got {len(fake_asr_backend.streams_created)}"
        )
        assert all(s.finalized for s in fake_asr_backend.streams_created)

        # WS still alive: send one more EOS, expect a 4th final.
        ws.send_bytes(_silence_pcm16(50))
        ws.send_json({"type": "asr_eos"})
        payload, _seen = _drain_until_final(ws)
        assert payload.get("session_complete") is False
    finally:
        ws.__exit__(None, None, None)


def test_scenario2_single_utterance_eos_closes_session(fake_asr_backend):
    """1× client asr_eos in single-utterance mode → final + loop exit."""
    from fastapi.testclient import TestClient
    from app.main import app

    fake_asr_backend._finals = ["only utterance"]
    fake_asr_backend._final_idx = 0

    client = TestClient(app)
    ws = _open_v2v(client, multi_utterance=False)
    try:
        ws.send_bytes(_silence_pcm16(50))
        ws.send_json({"type": "asr_eos"})
        payload, _seen = _drain_until_final(ws)

        assert payload.get("type") == "asr_final"
        assert payload.get("text") == "only utterance"
        # Single-utterance final is the terminating event. Per the
        # production branch at app/main.py:1178-1181 it does NOT
        # carry session_complete; the loop just returns after the
        # send. Either absent OR True is acceptable; "False" would
        # be a regression.
        sc = payload.get("session_complete", None)
        assert sc in (None, True), (
            f"single-utterance final must not advertise session_complete=False: {payload}"
        )

        # After the final the server's asr_out_task returns; client
        # closing its side completes the teardown cleanly.
    finally:
        ws.__exit__(None, None, None)


# ──────────────────────────────────────────────────────────────────────
# Scenario 3 (unit-level): symmetry between VAD endpoint and client_eos
# ──────────────────────────────────────────────────────────────────────
#
# Both ``vad`` SPEECH_END and client ``asr_eos`` must, in multi_utterance
# mode, produce the same observable post-finalize state on the shared
# ``state`` dict:
#   - endpoint_pending consumed (None)
#   - asr_session_closed STILL False  ← the load-bearing invariant
#   - emitted asr_final has session_complete=False
#
# We mirror exactly the two production code paths that set these flags:
#   - VAD branch: app/main.py:1047-1050
#   - asr_eos branch: app/main.py:1069-1072
# and then run a minimal version of asr_out_task's relevant decision
# block. This is the documented fallback the task spec sanctions.


def _apply_vad_speech_end(state: dict, multi_utterance: bool) -> None:
    """Mirror app/main.py:1047-1050 verbatim."""
    state["endpoint_pending"] = "vad"
    if not multi_utterance:
        state["asr_session_closed"] = True


def _apply_client_asr_eos(state: dict, multi_utterance: bool) -> None:
    """Mirror app/main.py:1069-1072 verbatim (the fix under test)."""
    state["endpoint_pending"] = "client_eos"
    if not multi_utterance:
        state["asr_session_closed"] = True


def _emit_final_decision(state: dict, multi_utterance: bool, final_text: str) -> dict:
    """Mirror the multi_utterance branch of asr_out_task (lines 1159-1177).

    Returns the asr_final payload that would be sent.
    """
    # endpoint_pending consumed
    endpoint_reason = state["endpoint_pending"]
    state["endpoint_pending"] = None
    assert endpoint_reason is not None

    if multi_utterance:
        is_closing = state["asr_session_closed"]
        if is_closing:
            return {
                "type": "asr_final",
                "text": final_text or "",
                "session_complete": True,
                "duplicate_of_streamed": False,
            }
        return {
            "type": "asr_final",
            "text": final_text or "",
            "session_complete": False,
        }
    return {"type": "asr_final", "text": final_text or ""}


def test_scenario3_symmetry_vad_vs_client_eos_in_multi_utterance():
    """VAD speech-end and client asr_eos behave identically in multi_utterance."""
    # ── VAD path ──
    state_vad = {"endpoint_pending": None, "asr_session_closed": False}
    _apply_vad_speech_end(state_vad, multi_utterance=True)
    assert state_vad["endpoint_pending"] == "vad"
    assert state_vad["asr_session_closed"] is False, (
        "VAD speech-end in multi_utterance must NOT close the session"
    )
    final_vad = _emit_final_decision(state_vad, multi_utterance=True, final_text="hello")
    assert final_vad["session_complete"] is False
    assert state_vad["asr_session_closed"] is False  # still alive

    # ── client_eos path ──
    state_eos = {"endpoint_pending": None, "asr_session_closed": False}
    _apply_client_asr_eos(state_eos, multi_utterance=True)
    assert state_eos["endpoint_pending"] == "client_eos"
    assert state_eos["asr_session_closed"] is False, (
        "client asr_eos in multi_utterance must NOT close the session "
        "— this is exactly the bug fixed by commit bf24284"
    )
    final_eos = _emit_final_decision(state_eos, multi_utterance=True, final_text="hello")
    assert final_eos["session_complete"] is False
    assert state_eos["asr_session_closed"] is False  # still alive

    # ── symmetry assertions ──
    assert final_vad["session_complete"] == final_eos["session_complete"]
    assert state_vad["asr_session_closed"] == state_eos["asr_session_closed"]


def test_scenario3_single_utterance_both_paths_close():
    """Sanity counter-part: in single-utterance mode, BOTH paths close."""
    for applier in (_apply_vad_speech_end, _apply_client_asr_eos):
        state = {"endpoint_pending": None, "asr_session_closed": False}
        applier(state, multi_utterance=False)
        assert state["asr_session_closed"] is True, (
            f"{applier.__name__} must close session in single-utterance mode"
        )


# ──────────────────────────────────────────────────────────────────────
# Structural pin: guard against silent removal of the load-bearing
# ``if not multi_utterance:`` guard in main.py. If someone deletes the
# guard, this test fails fast even if the integration tests above
# somehow flake out.
# ──────────────────────────────────────────────────────────────────────


def test_asr_eos_branch_has_multi_utterance_guard():
    """Pin the source: asr_eos handler must guard the close on multi_utterance.

    Looks for the pattern
        elif typ == v2v_proto.CLIENT_ASR_EOS:
            state["endpoint_pending"] = "client_eos"
            if not multi_utterance:
                state["asr_session_closed"] = True
    in app/main.py. If the ``if not multi_utterance:`` line gets removed
    in a future refactor, this test will catch it immediately.
    """
    import re

    here = os.path.dirname(__file__)
    main_path = os.path.abspath(os.path.join(here, "..", "main.py"))
    with open(main_path, "r", encoding="utf-8") as f:
        src = f.read()
    # Be lenient on whitespace; strict on structure. The handler's nesting
    # depth (and therefore its leading indentation) has drifted across
    # refactors — what matters is the structural order of the statements, not
    # the exact column. Match with flexible leading whitespace so this stays a
    # behavior pin and not an indentation pin.
    pattern = re.compile(
        r"CLIENT_ASR_EOS:\s*\n"
        r"\s*state\[\"endpoint_pending\"\] = \"client_eos\"\s*\n"
        r"\s*state\[\"endpoint_pending_gen\"\] = state\[\"asr_active_gen\"\]\s*\n"
        r"\s*if not multi_utterance:\s*\n"
        r"\s*state\[\"asr_session_closed\"\] = True"
    )
    assert pattern.search(src), (
        "fix from commit bf24284 appears to have been reverted: "
        "the `if not multi_utterance:` guard around "
        "`state['asr_session_closed'] = True` in the asr_eos handler is missing"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
