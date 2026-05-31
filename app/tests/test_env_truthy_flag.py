"""Regression: _env_truthy must tolerate --env-file literal quoting.

The /v2v server-loop gate (app/main.py: ``OVS_V2V_SERVER_LOOP``) and the
agent-side flag both read env flags. Production injects them via ``--env-file``
whose values can carry literal quotes, so ``FLAG="1"`` arrives in os.environ as
the 3-char string ``"1"``. A plain ``.strip().lower()`` never matched ``"1"``,
so the SLV silently stayed in client pass-through (tool_registry=None → no
remote tool registration) — the 2026-05-31 3b-ii server-loop activation bug,
SLV half (the agent half was fixed in openvoicestream_agent/config.py).
"""

import pytest

from app.main import _env_truthy


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ('"1"', True),       # <-- the --env-file quoting bug
        ("'1'", True),
        (" 1 ", True),
        ('" 1 "', True),
        ("true", True),
        ('"true"', True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ('"0"', False),
        ("false", False),
        ('"false"', False),
        ("", False),
        ("  ", False),
        (None, False),
    ],
)
def test_env_truthy_tolerates_quotes(value, expected):
    assert _env_truthy(value) is expected
