"""CLIENT_PING keepalive (P1, server half).

The idle watchdog reaps a client that sends no frame for
OVS_V2V_IDLE_TIMEOUT_S. Wake-word clients legitimately send nothing while
nobody speaks, so SLVClient now sends a no-op ping. Two things must hold:
the ping must reach the dispatcher's frame loop (that is what resets the
watchdog) and it must not touch any session state.

Structural assertions against live source, matching the house style in
test_v2v_admission_release_ordering.py — the full handler needs ASR/TTS/VAD
wiring to drive end to end.
"""
from __future__ import annotations

import inspect
import re

from server.core import v2v as v2v_proto


def test_client_ping_constant_exists():
    assert v2v_proto.CLIENT_PING == "ping"


def test_agent_vendored_constant_matches_server():
    """protocol.py is vendored, not imported — it can drift silently."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "agent" / "ovs_agent" / "protocol.py").read_text()
    m = re.search(r'^CLIENT_PING\s*=\s*"([^"]+)"', src, re.M)
    assert m is not None, "agent/ovs_agent/protocol.py is missing CLIENT_PING"
    assert m.group(1) == v2v_proto.CLIENT_PING


def test_ping_is_handled_before_any_state_mutating_branch():
    """A ping must short-circuit ahead of the text/flush/eos/abort chain.

    If it fell through to the tail instead, a future refactor rejecting
    unknown types would start killing keepalives.
    """
    from server import main as appmod

    src = inspect.getsource(appmod.v2v_stream)
    ping_at = src.find("v2v_proto.CLIENT_PING")
    assert ping_at != -1, "dispatcher has no CLIENT_PING branch"
    for later in (
        "v2v_proto.CLIENT_TEXT and tts_buffer is not None",
        "elif typ == v2v_proto.CLIENT_TTS_FLUSH",
        "elif typ == v2v_proto.CLIENT_ASR_EOS",
        "elif typ == v2v_proto.CLIENT_ABORT",
    ):
        at = src.find(later)
        assert at != -1, f"anchor vanished: {later}"
        assert ping_at < at, f"CLIENT_PING must be dispatched before {later}"


def test_ping_branch_is_a_pure_no_op():
    """The branch body may only `continue` — no session state, no send."""
    from server import main as appmod

    src = inspect.getsource(appmod.v2v_stream)
    m = re.search(
        r"if typ == v2v_proto\.CLIENT_PING:\n(?P<body>(?:[ \t]*(?:#[^\n]*)?\n)*"
        r"[ \t]*continue\n)",
        src,
    )
    assert m is not None, (
        "expected `if typ == v2v_proto.CLIENT_PING:` whose body is comments "
        "then a bare `continue`"
    )
    body = m.group("body")
    for forbidden in ("state[", "await ", "tts_q", "asr_manager", "send_json"):
        assert forbidden not in body, (
            f"ping branch must be a no-op, found {forbidden!r}"
        )


def test_keepalive_cadence_leaves_margin_under_watchdog():
    """The cadence is sized against the tightest DEPLOYED idle timeout.

    The server default is 90s, but deploy/docker-compose.jetson-rebot.yml
    pins OVS_V2V_IDLE_TIMEOUT_S=45 on the arm stack — sizing against 90s
    would leave only 1.5x margin there.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "agent" / "ovs_agent" / "slv_client.py").read_text()
    cadence = re.search(r"_KEEPALIVE_DEFAULT_S\s*=\s*([0-9.]+)", src)
    tightest = re.search(
        r"_TIGHTEST_DEPLOYED_IDLE_TIMEOUT_S\s*=\s*([0-9.]+)", src
    )
    assert cadence is not None and tightest is not None
    assert float(tightest.group(1)) <= 90.0, (
        "the pinned tightest value must not exceed the server default"
    )
    assert float(cadence.group(1)) * 3 <= float(tightest.group(1))
