"""Unit tests for app.core.session_limiter."""

import pytest

from app.core import session_limiter, metrics


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("OVS_MAX_CONCURRENT_SESSIONS", raising=False)
    session_limiter._reset_for_tests()
    metrics._reset_for_tests()


# ── resolve_limit ──────────────────────────────────────────────────

def test_env_override_clamped_to_ceiling(monkeypatch):
    # Spec §7 (commit fe33bf1): env/profile overrides may only DOWNGRADE.
    # No backend declared → unknown target → ceiling 1. env=7 is above the
    # ceiling, so it is warn-logged and silently clamped to the ceiling.
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "7")
    assert session_limiter.resolve_limit({"max_concurrent_sessions": 2}) == 1


def test_profile_override_clamped_to_target_default():
    # orin-nano default (ceiling) is 1; profile asks for 3 (an upgrade),
    # which is clamped back down to the ceiling per spec §7.
    profile = {"name": "jetson-orin-nano-zh", "max_concurrent_sessions": 3}
    assert session_limiter.resolve_limit(profile) == 1


def test_orin_nx_default():
    assert session_limiter.resolve_limit({"name": "jetson-orin-nx-highperf"}) == 2


def test_orin_nano_default():
    assert session_limiter.resolve_limit({"name": "jetson-orin-nano-default"}) == 1


def test_rk_default():
    assert session_limiter.resolve_limit({"name": "rk3576-default"}) == 1


def test_desktop_default():
    assert session_limiter.resolve_limit({"name": "desktop-ci"}) == 4


def test_unknown_default():
    assert session_limiter.resolve_limit({"name": "weird-profile"}) == 1
    assert session_limiter.resolve_limit({}) == 1


def test_zero_env_raises(monkeypatch):
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "0")
    with pytest.raises(ValueError):
        session_limiter.resolve_limit({})


def test_negative_env_raises(monkeypatch):
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "-1")
    with pytest.raises(ValueError):
        session_limiter.resolve_limit({})


def test_non_int_env_raises(monkeypatch):
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "five")
    with pytest.raises(ValueError):
        session_limiter.resolve_limit({})


def test_zero_profile_raises():
    with pytest.raises(ValueError):
        session_limiter.resolve_limit({"max_concurrent_sessions": 0})


# ── SessionLimiter ─────────────────────────────────────────────────

def test_acquire_succeeds_below_limit():
    sl = session_limiter.SessionLimiter(2)
    t1 = sl.try_acquire()
    assert t1 is not None
    assert sl.active == 1
    t2 = sl.try_acquire()
    assert t2 is not None
    assert sl.active == 2


def test_acquire_fails_at_limit():
    sl = session_limiter.SessionLimiter(1)
    t1 = sl.try_acquire()
    assert t1 is not None
    t2 = sl.try_acquire()
    assert t2 is None


def test_release_decrements_active():
    sl = session_limiter.SessionLimiter(2)
    t = sl.try_acquire()
    assert sl.active == 1
    t.release()
    assert sl.active == 0


def test_double_release_idempotent():
    sl = session_limiter.SessionLimiter(2)
    t = sl.try_acquire()
    t.release()
    t.release()
    t.release()
    assert sl.active == 0


def test_rejection_increments_metrics_counter():
    # The limiter itself doesn't touch metrics on reject — acquire_http
    # / try_acquire_ws do. So just check that acquire/release shape
    # ovs_sessions_active correctly.
    sl = session_limiter.SessionLimiter(1)
    assert metrics.get_sessions_active() == 0
    t = sl.try_acquire()
    assert metrics.get_sessions_active() == 1
    t.release()
    assert metrics.get_sessions_active() == 0


def test_zero_limit_constructor_rejects():
    with pytest.raises(ValueError):
        session_limiter.SessionLimiter(0)


def test_init_and_get_limiter():
    sl = session_limiter.init_limiter({"name": "desktop"})
    assert session_limiter.get_limiter() is sl
    assert sl.limit == 4


# ── HTTP integration ───────────────────────────────────────────────

def test_http_429_when_full(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "1")
    sl = session_limiter.init_limiter({})
    app = FastAPI()

    @app.get("/work")
    async def work():
        async with session_limiter.acquire_http("/work"):
            return {"ok": True}

    with TestClient(app) as c:
        # Hold a slot manually to simulate concurrent in-flight work.
        held = sl.try_acquire()
        r = c.get("/work")
        assert r.status_code == 429
        assert r.headers.get("Retry-After") == "5"
        body = r.json()
        assert body["detail"]["error"] == "too_many_sessions"
        assert body["detail"]["limit"] == 1
        held.release()
        # Now the slot is free again.
        r = c.get("/work")
        assert r.status_code == 200
