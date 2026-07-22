"""Contract for the shared env-var coercion helpers."""

from server.core.env_helpers import env_float, env_int, truthy


def test_truthy_accepts_the_on_set():
    for v in ("1", "true", "TRUE", " yes ", "on", "On"):
        assert truthy(v) is True
    for v in ("", "0", "false", "no", "off", "nope", None):
        assert truthy(v) is False


def test_env_int_parses_or_falls_back():
    env = {"A": "7", "BAD": "x"}
    assert env_int("A", 1, env) == 7
    assert env_int("BAD", 1, env) == 1      # unparseable → default
    assert env_int("MISSING", 3, env) == 3  # absent → default


def test_env_float_parses_or_falls_back():
    env = {"A": "0.55", "BAD": ""}
    assert env_float("A", 1.0, env) == 0.55
    assert env_float("BAD", 0.5, env) == 0.5
    assert env_float("MISSING", 0.7, env) == 0.7
