"""wake() arms the mic-skip window so the wake-word tail isn't forwarded.

Root cause it guards against: with a continuous "Hey Jarvis <command>", the
agent forwarded the wake-word audio tail into the same ASR turn, so the server
decoded wake-word+command as one garbled segment. On a local wake-word fire we
now drop a short window of mic audio (``wake_mic_skip_ms``).
"""
from __future__ import annotations

import asyncio
import time

from ovs_agent.app_base import BaseApp, ConvState


def _bare_app(*, skip_ms: float, state: ConvState) -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app._wake_mic_skip_until = 0.0
    app._state = state

    class _Cfg:
        wake_mic_skip_ms = skip_ms

    app.config = _Cfg()
    app.plugins = [type("_LocalWake", (), {"name": "openwakeword", "local_audio": True})()]
    # wake() reaches further only when SLEEPING; for these tests we only assert
    # the skip-arming that happens BEFORE the SLEEPING gate, so stub the rest.
    return app


def _run(coro):
    return asyncio.run(coro)


def test_openwakeword_arms_skip_even_when_already_idle():
    # Re-wake spoken mid-conversation (already IDLE) must still arm the skip.
    app = _bare_app(skip_ms=500.0, state=ConvState.IDLE)
    before = time.monotonic()
    _run(app.wake(source="openwakeword"))
    assert app._wake_mic_skip_until >= before + 0.4


def test_external_wake_does_not_arm_skip():
    # Non-audio wake (button / API) has no wake word to skip.
    app = _bare_app(skip_ms=500.0, state=ConvState.IDLE)
    _run(app.wake(source="external"))
    assert app._wake_mic_skip_until == 0.0


def test_zero_skip_ms_disables():
    app = _bare_app(skip_ms=0.0, state=ConvState.IDLE)
    _run(app.wake(source="openwakeword"))
    assert app._wake_mic_skip_until == 0.0
