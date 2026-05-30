"""Unit tests for app.core.backend_manager.BackendManager.

Avoids pytest-asyncio dep by routing async tests through ``asynctest``.
``profile_loader.apply_profile`` and ``current_profile`` are monkey-patched
on each test that touches profile-reload behavior, so we never touch real
``configs/profiles`` JSON.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import HTTPException

from app.core import backend_manager as bm_mod
from app.core.backend_manager import (
    BackendManager,
    BackendState,
    asr_manager,
    init_backend_managers,
    tts_manager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def asynctest(fn):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(fn(*args, **kwargs))
        finally:
            loop.close()

    wrapper.__name__ = fn.__name__
    return wrapper


class FakeBackend:
    counter = 0
    # PR5: existing reload tests assume reload succeeds. Real backends that
    # opt out of hot reload now get 400; FakeBackend explicitly opts in so
    # the manager-level tests stay focused on lifecycle, not capability.
    supports_hot_reload = True

    def __init__(
        self,
        *,
        name: str | None = None,
        fail_preload: bool = False,
        sleep_preload: float = 0.0,
    ) -> None:
        FakeBackend.counter += 1
        self.name = name or f"fake-{FakeBackend.counter}"
        self.unloaded = False
        self._fail_preload = fail_preload
        self._sleep_preload = sleep_preload

    def preload(self) -> None:
        if self._sleep_preload:
            time.sleep(self._sleep_preload)
        if self._fail_preload:
            raise RuntimeError("preload fail")

    def unload(self) -> None:
        self.unloaded = True

    async def work(self, ms: int = 50) -> str:
        await asyncio.sleep(ms / 1000)
        return "ok"


def _make_mgr(
    *,
    name: str = "tts",
    factory=None,
    preload_calls: list | None = None,
    unload_calls: list | None = None,
    drain_timeout_s: float = 30.0,
) -> BackendManager:
    """Build a manager. By default each factory() returns a fresh FakeBackend."""
    if factory is None:
        def factory():
            return FakeBackend()

    preload_calls = preload_calls if preload_calls is not None else []
    unload_calls = unload_calls if unload_calls is not None else []

    def preloader(b):
        preload_calls.append(b)
        b.preload()

    def unloader(b):
        unload_calls.append(b)
        b.unload()

    return BackendManager(
        name=name,
        factory=factory,
        preloader=preloader,
        unloader=unloader,
        drain_timeout_s=drain_timeout_s,
    )


@pytest.fixture(autouse=True)
def _stub_profile_loader(monkeypatch):
    """Default: current_profile returns matching kinds; apply_profile is a no-op."""
    state = {"profile": {"name": "p-old", "tts_backend": "fake", "asr_backend": "fake"}}

    def current_profile():
        return dict(state["profile"])

    apply_mock = MagicMock(side_effect=lambda ref, **kw: state["profile"])
    monkeypatch.setattr(bm_mod.profile_loader, "current_profile", current_profile)
    monkeypatch.setattr(bm_mod.profile_loader, "apply_profile", apply_mock)
    # Stub the JSON profile loader used for kind validation:
    monkeypatch.setattr(
        BackendManager,
        "_load_profile_kind",
        lambda self, ref: {
            "name": ref,
            "tts_backend": "fake",
            "asr_backend": "fake",
        },
    )
    yield state
    bm_mod._reset_for_tests()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@asynctest
async def test_start_transitions_init_to_ready():
    mgr = _make_mgr()
    assert mgr.state == BackendState.INIT
    await mgr.start()
    assert mgr.state == BackendState.READY
    assert mgr.is_ready()


@asynctest
async def test_start_factory_failure_state_failed():
    def bad_factory():
        raise RuntimeError("boom")

    mgr = _make_mgr(factory=bad_factory)
    with pytest.raises(RuntimeError):
        await mgr.start()
    assert mgr.state == BackendState.FAILED
    with pytest.raises(HTTPException) as ei:
        async with mgr.acquire():
            pass
    assert ei.value.status_code == 503


@asynctest
async def test_acquire_increments_inflight():
    mgr = _make_mgr()
    await mgr.start()
    assert mgr.status()["inflight_http"] == 0
    async with mgr.acquire() as b:
        assert b is not None
        assert mgr.status()["inflight_http"] == 1
    assert mgr.status()["inflight_http"] == 0


@asynctest
async def test_acquire_when_not_ready_raises_503():
    mgr = _make_mgr()
    # Never called start; state is INIT.
    with pytest.raises(HTTPException) as ei:
        async with mgr.acquire():
            pass
    assert ei.value.status_code == 503


@asynctest
async def test_reload_changes_backend_instance():
    mgr = _make_mgr()
    await mgr.start()
    old = mgr.get_backend_unsafe()
    out = await mgr.reload("p-new")
    assert out["status"] == "reloaded"
    new = mgr.get_backend_unsafe()
    assert new is not old


@asynctest
async def test_reload_calls_unload_on_old():
    unload_calls: list = []
    mgr = _make_mgr(unload_calls=unload_calls)
    await mgr.start()
    old = mgr.get_backend_unsafe()
    await mgr.reload("p-new")
    assert old.unloaded is True
    assert unload_calls and unload_calls[0] is old


@asynctest
async def test_reload_waits_for_inflight_drain():
    mgr = _make_mgr(drain_timeout_s=5.0)
    await mgr.start()

    drain_observed: list[int] = []

    async def long_request():
        async with mgr.acquire() as b:
            # Record inflight just after entering.
            drain_observed.append(mgr.status()["inflight_http"])
            await asyncio.sleep(0.2)

    reload_completed_at = {}

    async def do_reload():
        # Let the request enter acquire first.
        await asyncio.sleep(0.05)
        await mgr.reload("p-new")
        reload_completed_at["t"] = time.monotonic()

    t0 = time.monotonic()
    await asyncio.gather(long_request(), do_reload())
    elapsed = reload_completed_at["t"] - t0
    # Reload must not finish before the 0.2s request drains.
    assert elapsed >= 0.18, f"reload finished too early: {elapsed}"
    assert mgr.is_ready()
    assert mgr.status()["inflight_http"] == 0


@asynctest
async def test_reload_drain_timeout_hard_proceeds():
    mgr = _make_mgr(drain_timeout_s=0.05)
    await mgr.start()

    async def hanging_request():
        async with mgr.acquire():
            await asyncio.sleep(0.5)

    req_task = asyncio.create_task(hanging_request())
    await asyncio.sleep(0.02)  # ensure inflight=1

    out = await mgr.reload("p-new")
    assert out["status"] == "reloaded"
    assert out["drained_cleanly"] is False
    # Let the hanging request finish.
    await req_task
    assert mgr.status()["inflight_http"] == 0


@asynctest
async def test_reload_closes_ws_sessions_1012():
    mgr = _make_mgr()
    await mgr.start()

    close_calls: list = []
    cancel_calls: list = []

    class FakeWS:
        def __init__(self):
            self.closed = False

        async def close(self, code: int = 1000, reason: str = ""):
            close_calls.append((code, reason))
            self.closed = True

    class FakeTask:
        def __init__(self):
            self._done = False

        def done(self):
            return self._done

        def cancel(self):
            cancel_calls.append(True)
            self._done = True

    class FakeHandle:
        def __init__(self):
            self.websocket = FakeWS()
            self.task = FakeTask()

    h = FakeHandle()
    mgr.register_ws(h)
    assert mgr.status()["inflight_ws"] == 1

    await mgr.reload("p-new")
    assert close_calls == [(1012, "backend reload")]
    assert cancel_calls == [True]


@asynctest
async def test_concurrent_reload_returns_409():
    mgr = _make_mgr()
    await mgr.start()

    # Register a WS handle whose close() awaits long enough that a second
    # concurrent reload can observe the held _reload_lock.
    pause = asyncio.Event()
    resume = asyncio.Event()

    class SlowWS:
        async def close(self, code=1000, reason=""):
            pause.set()
            await resume.wait()

    class SlowHandle:
        def __init__(self):
            self.websocket = SlowWS()
            self.task = None

    _slow_handle = SlowHandle()  # keep strong ref; WeakSet drops temps
    mgr.register_ws(_slow_handle)

    async def first():
        return await mgr.reload("p-a")

    async def second():
        await pause.wait()  # first reload is now inside _force_close_ws_sessions
        try:
            return await mgr.reload("p-b")
        finally:
            resume.set()  # let first proceed

    results = await asyncio.gather(first(), second(), return_exceptions=True)
    codes = [r.status_code if isinstance(r, HTTPException) else None for r in results]
    assert 409 in codes, f"expected a 409, got {results!r}"
    successes = [r for r in results if isinstance(r, dict)]
    assert len(successes) == 1


@asynctest
async def test_reload_refuses_when_supports_hot_reload_false():
    """PR5 FIX_A: reload must 400 when the live backend opts out."""

    class _NoHotReload(FakeBackend):
        supports_hot_reload = False

    def factory():
        return _NoHotReload()

    mgr = _make_mgr(factory=factory)
    await mgr.start()
    with pytest.raises(HTTPException) as ei:
        await mgr.reload("p-new")
    assert ei.value.status_code == 400
    detail = ei.value.detail
    assert detail["error"] == "hot_reload_not_supported"
    # State must stay READY — the manager should not have entered DRAINING.
    assert mgr.state == BackendState.READY


@asynctest
async def test_reload_succeeds_when_supports_hot_reload_true():
    """PR5 FIX_A: a True flag must not block reload."""
    mgr = _make_mgr()  # FakeBackend has supports_hot_reload = True
    await mgr.start()
    out = await mgr.reload("p-new")
    assert out["status"] == "reloaded"
    assert mgr.state == BackendState.READY


@asynctest
async def test_reload_cross_kind_succeeds():
    """Cross-kind reload should succeed: the factory (registry-dispatched in
    real code) builds whatever the new profile declares. The manager no longer
    gates on old_kind != new_kind."""
    mgr = _make_mgr(name="tts")
    await mgr.start()

    # Patch the kind loader to declare a different tts_backend on the new profile.
    original = BackendManager._load_profile_kind
    BackendManager._load_profile_kind = lambda self, ref: {  # type: ignore[assignment]
        "name": ref, "tts_backend": "other", "asr_backend": "fake"
    }
    try:
        out = await mgr.reload("p-new")
        assert out["status"] == "reloaded"
        assert mgr.state == BackendState.READY
    finally:
        BackendManager._load_profile_kind = original  # type: ignore[assignment]


@asynctest
async def test_reload_factory_failure_rolls_back():
    call = {"n": 0}

    def factory():
        call["n"] += 1
        if call["n"] == 2:  # the reload attempt
            raise RuntimeError("new factory broke")
        return FakeBackend()

    mgr = _make_mgr(factory=factory)
    await mgr.start()
    old = mgr.get_backend_unsafe()

    out = await mgr.reload("p-new")
    assert out["status"] == "rolled_back"
    assert "new factory broke" in out["error"]
    assert mgr.state == BackendState.READY
    # Rollback built a fresh backend (call 3), not the same instance as old.
    assert mgr.get_backend_unsafe() is not old


@asynctest
async def test_reload_rollback_failure_state_failed():
    call = {"n": 0}

    def factory():
        call["n"] += 1
        if call["n"] >= 2:  # both reload and rollback fail
            raise RuntimeError(f"factory broke #{call['n']}")
        return FakeBackend()

    mgr = _make_mgr(factory=factory)
    await mgr.start()
    with pytest.raises(HTTPException) as ei:
        await mgr.reload("p-new")
    assert ei.value.status_code == 500
    assert mgr.state == BackendState.FAILED


@asynctest
async def test_acquire_during_reload_raises_503():
    mgr = _make_mgr()
    await mgr.start()

    # Park the reload inside _force_close_ws_sessions via a WS whose close()
    # awaits an Event we control. While parked, manager.state == DRAINING.
    pause = asyncio.Event()
    resume = asyncio.Event()

    class SlowWS:
        async def close(self, code=1000, reason=""):
            pause.set()
            await resume.wait()

    class SlowHandle:
        def __init__(self):
            self.websocket = SlowWS()
            self.task = None

    _slow_handle = SlowHandle()  # keep strong ref; WeakSet drops temps
    mgr.register_ws(_slow_handle)

    reload_task = asyncio.create_task(mgr.reload("p-new"))
    await pause.wait()
    assert mgr.state == BackendState.DRAINING

    with pytest.raises(HTTPException) as ei:
        async with mgr.acquire():
            pass
    assert ei.value.status_code == 503

    resume.set()
    await reload_task
    assert mgr.is_ready()


@asynctest
async def test_status_shape():
    mgr = _make_mgr()
    await mgr.start()
    s = mgr.status()
    assert set(s.keys()) == {
        "state",
        "profile_name",
        "backend_name",
        "inflight_http",
        "inflight_ws",
    }
    assert s["state"] == "ready"
    assert s["inflight_http"] == 0
    assert s["inflight_ws"] == 0
    assert s["backend_name"]


def test_init_backend_managers_singleton():
    bm_mod._reset_for_tests()

    def f():
        return FakeBackend()

    def p(b):
        b.preload()

    def u(b):
        b.unload()

    # Before init, accessors must raise.
    with pytest.raises(RuntimeError):
        tts_manager()
    with pytest.raises(RuntimeError):
        asr_manager()

    init_backend_managers(
        tts_factory=f, tts_preloader=p, tts_unloader=u,
        asr_factory=f, asr_preloader=p, asr_unloader=u,
    )
    assert tts_manager().name == "tts"
    assert asr_manager().name == "asr"

    # Second init refuses.
    with pytest.raises(RuntimeError):
        init_backend_managers(
            tts_factory=f, tts_preloader=p, tts_unloader=u,
            asr_factory=f, asr_preloader=p, asr_unloader=u,
        )
    bm_mod._reset_for_tests()


# ---------------------------------------------------------------------------
# FIX_4 (PR4b): rollback must re-apply the original profile *reference*
# (which could be a custom path), not just the logical profile name.
# ---------------------------------------------------------------------------

def test_reload_rollback_uses_original_profile_ref(monkeypatch):
    """A failed reload rolls back via the same ref used to load the prior profile.

    Not wrapped with @asynctest because we need monkeypatch (a pytest fixture)
    to coexist with the autouse _stub_profile_loader fixture's state. We drive
    the event loop manually.
    """
    # Force the second factory call (the reload attempt) to fail, so rollback runs.
    call = {"n": 0}

    def factory():
        call["n"] += 1
        if call["n"] == 3:  # successful initial start, successful first reload,
                             # then the SECOND reload attempt fails.
            raise RuntimeError("reload broke")
        return FakeBackend()

    mgr = _make_mgr(factory=factory)

    # Override apply_profile to capture call history (the autouse fixture's
    # MagicMock would also do this, but we want clean per-test state).
    apply_calls: list = []

    def apply_mock(ref, *, overrides=None, resolve_engines=False):
        apply_calls.append({"ref": ref, "resolve_engines": resolve_engines})

    monkeypatch.setattr(bm_mod.profile_loader, "apply_profile", apply_mock)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(mgr.start())

        # Simulate prior load from a custom path: a successful reload records
        # _last_profile_ref = custom_path.
        custom_path = "/tmp/custom-profile-A.json"
        out = loop.run_until_complete(mgr.reload(custom_path))
        assert out["status"] == "reloaded"
        assert mgr._last_profile_ref == custom_path

        # Now a reload that fails — rollback should re-apply custom_path,
        # NOT the logical profile name "p-old".
        apply_calls.clear()
        out2 = loop.run_until_complete(mgr.reload("/tmp/profile-B.json"))
        assert out2["status"] == "rolled_back"
        # Both reload and rollback now apply with resolve_engines=True (B1: the
        # rollback must re-resolve the old profile's engines too); identify the
        # rollback by re-applying the original ref (custom_path), not by the
        # resolve_engines flag.
        rollback_calls = [c for c in apply_calls if c["ref"] == custom_path]
        assert rollback_calls, f"expected a rollback apply_profile call, got {apply_calls}"
        assert rollback_calls[-1]["ref"] == custom_path, (
            f"rollback must use original profile ref, got {rollback_calls[-1]!r}"
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# FIX_4_completion (PR4c): start() seeds _last_profile_ref from
# initial_profile_ref so the *very first* reload's rollback re-applies via
# the bootstrap source (e.g. a custom OVS_PROFILE_JSON path) even though no
# prior successful reload has run.
# ---------------------------------------------------------------------------

def test_start_seeds_last_profile_ref_from_initial(monkeypatch):
    """BackendManager seeded with initial_profile_ref records it on start()."""
    custom_path = "/tmp/custom-startup-X.json"
    mgr = BackendManager(
        name="tts",
        factory=lambda: FakeBackend(),
        preloader=lambda b: b.preload(),
        unloader=lambda b: b.unload(),
        initial_profile_ref=custom_path,
    )
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(mgr.start())
    finally:
        loop.close()
    assert mgr._last_profile_ref == custom_path


def test_first_reload_after_custom_path_startup_rollback_uses_path(monkeypatch):
    """FIX_4_completion: a failed FIRST reload (no prior reload() success) must
    roll back via the bootstrap profile ref, not the logical profile ``name``.
    """
    # Factory: succeed on initial start, fail on the reload attempt → rollback runs.
    call = {"n": 0}

    def factory():
        call["n"] += 1
        if call["n"] == 2:  # the reload attempt
            raise RuntimeError("reload broke on first try")
        return FakeBackend()

    custom_path = "/tmp/custom-A.json"
    mgr = BackendManager(
        name="tts",
        factory=factory,
        preloader=lambda b: b.preload(),
        unloader=lambda b: b.unload(),
        initial_profile_ref=custom_path,
    )

    apply_calls: list = []

    def apply_mock(ref, *, overrides=None, resolve_engines=False):
        apply_calls.append({"ref": ref, "resolve_engines": resolve_engines})

    monkeypatch.setattr(bm_mod.profile_loader, "apply_profile", apply_mock)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(mgr.start())
        # No intermediate successful reload — go straight to a failing one.
        out = loop.run_until_complete(mgr.reload("/tmp/profile-B.json"))
        assert out["status"] == "rolled_back"
        # B1: both reload and rollback apply with resolve_engines=True; the
        # rollback is the one re-applying the original ref (custom_path).
        rollback_calls = [c for c in apply_calls if c["ref"] == custom_path]
        assert rollback_calls, f"expected rollback apply_profile call, got {apply_calls}"
        assert rollback_calls[-1]["ref"] == custom_path, (
            f"rollback must re-apply the bootstrap ref {custom_path!r}, "
            f"got {rollback_calls[-1]!r}"
        )
    finally:
        loop.close()


def test_init_backend_managers_propagates_initial_profile_ref(monkeypatch):
    """init_backend_managers passes initial_profile_ref through to both managers."""
    bm_mod._reset_for_tests()

    def f():
        return FakeBackend()

    def p(b):
        b.preload()

    def u(b):
        b.unload()

    init_backend_managers(
        tts_factory=f, tts_preloader=p, tts_unloader=u,
        asr_factory=f, asr_preloader=p, asr_unloader=u,
        initial_profile_ref="/tmp/seeded-ref.json",
    )
    assert tts_manager()._initial_profile_ref == "/tmp/seeded-ref.json"
    assert asr_manager()._initial_profile_ref == "/tmp/seeded-ref.json"
    bm_mod._reset_for_tests()


# ---------------------------------------------------------------------------
# FIX_DRYRUN: pre-flight artifact validation rejects unloadable profiles
# with HTTP 400 *before* draining the running backend.
# ---------------------------------------------------------------------------

@asynctest
async def test_reload_rejects_profile_with_missing_engine_paths():
    """Profile with a path-like env value that doesn't exist on disk → 400."""
    mgr = _make_mgr()
    await mgr.start()
    old = mgr.get_backend_unsafe()

    bad_path = "/nonexistent/seeed/test/dryrun/foo.engine"

    def loader(self, ref):
        return {
            "name": ref,
            "tts_backend": "fake",
            "asr_backend": "fake",
            "env": {"EDGE_LLM_ASR_ENGINE_DIR": bad_path},
        }

    original = BackendManager._load_profile_kind
    BackendManager._load_profile_kind = loader  # type: ignore[assignment]
    try:
        with pytest.raises(HTTPException) as ei:
            await mgr.reload("p-bad")
    finally:
        BackendManager._load_profile_kind = original  # type: ignore[assignment]

    assert ei.value.status_code == 400
    detail = ei.value.detail
    assert detail["error"] == "profile_artifacts_missing"
    assert detail["kind"] == "tts"
    assert detail["profile"] == "p-bad"
    missing_paths = [m["path"] for m in detail["missing"]]
    assert bad_path in missing_paths
    # Manager must NOT have entered DRAINING — old backend still serving.
    assert mgr.state == BackendState.READY
    assert mgr.get_backend_unsafe() is old


@asynctest
async def test_reload_passes_artifact_check_when_paths_exist():
    """Profile env points at /tmp (always exists) → check passes, reload proceeds."""
    mgr = _make_mgr()
    await mgr.start()

    def loader(self, ref):
        return {
            "name": ref,
            "tts_backend": "fake",
            "asr_backend": "fake",
            # /tmp exists on every supported platform; passes the heuristic.
            "env": {"SOME_DIR": "/tmp"},
        }

    original = BackendManager._load_profile_kind
    BackendManager._load_profile_kind = loader  # type: ignore[assignment]
    try:
        out = await mgr.reload("p-ok")
    finally:
        BackendManager._load_profile_kind = original  # type: ignore[assignment]

    assert out["status"] == "reloaded"
    assert mgr.state == BackendState.READY


# ---------------------------------------------------------------------------
# Regression: rollback must unload the failed NEW backend before re-invoking
# the factory, so a factory that caches its instance in a module-level global
# (mirrors main.py's _asr_backend / _tts_service_mod._backend pattern) gets
# the cache cleared and rebuilds fresh against the restored env.
# Without this, the rollback path returns the broken cached NEW instance and
# preload fails with the same error, sending the manager to FAILED.
# ---------------------------------------------------------------------------

@asynctest
async def test_reload_rollback_clears_cached_failed_backend():
    """When new profile's preload fails, rollback must unload the failed new
    instance before re-instantiating, so the factory rebuilds against
    restored env rather than returning the cached broken one."""

    # Simulate main.py's "cache on None" factory pattern.
    cache: dict[str, FakeBackend | None] = {"backend": None}
    factory_calls = {"n": 0}
    # Mark a backend as "broken" if it was built when broken_env is True; its
    # preload then fails. After unload, cache resets to None so the next
    # factory() call sees broken_env=False (env restored) and builds fresh.
    broken_env = {"flag": False}

    class CachingFakeBackend(FakeBackend):
        def __init__(self, *, broken: bool):
            super().__init__()
            self.broken = broken

        def preload(self) -> None:
            if self.broken:
                raise FileNotFoundError("engine artifact missing (new profile)")

        def unload(self) -> None:
            self.unloaded = True

    def factory():
        factory_calls["n"] += 1
        if cache["backend"] is None:
            cache["backend"] = CachingFakeBackend(broken=broken_env["flag"])
        return cache["backend"]

    def preloader(b):
        b.preload()

    def unloader(b):
        b.unload()
        # Mirror main.py: unloader clears the module-level cache.
        cache["backend"] = None

    mgr = BackendManager(
        name="asr",
        factory=factory,
        preloader=preloader,
        unloader=unloader,
    )

    # Start: env is healthy → build OK instance.
    await mgr.start()
    old = mgr.get_backend_unsafe()
    factory_calls_after_start = factory_calls["n"]

    # Simulate "apply new profile flips env to broken paths" by toggling the
    # flag right when apply_profile is called. The autouse fixture installed
    # a MagicMock; reset its side_effect so we don't recursively call
    # ourselves through the mock.
    apply_calls: list[dict] = []

    def apply_side_effect(ref, **kw):
        apply_calls.append({"ref": ref, **kw})
        # B1: both reload and rollback now apply with resolve_engines=True, so
        # we distinguish by ref — the reload applies the new "p-new" ref (env
        # becomes broken), the rollback re-applies the old ref (env healthy).
        if ref == "p-new":
            broken_env["flag"] = True  # new env is broken
        else:
            broken_env["flag"] = False  # rollback restores healthy env
        return {"name": ref}

    bm_mod.profile_loader.apply_profile.side_effect = apply_side_effect

    out = await mgr.reload("p-new")

    # 1. Manager rolled back successfully.
    assert out["status"] == "rolled_back", out
    assert mgr.state == BackendState.READY
    assert "engine artifact missing" in out["error"]

    # 2. The broken NEW instance must have had unload() called on it
    #    (this is what clears the cache).
    # The broken instance was built between start and reload; capture it
    # indirectly by checking that the factory was called at least twice
    # *after* start (once for the failed new, once for the rollback rebuild).
    assert factory_calls["n"] - factory_calls_after_start >= 2, (
        f"factory must be called twice during reload+rollback, "
        f"got {factory_calls['n'] - factory_calls_after_start}"
    )

    # 3. The rolled-back live backend is a *different* Python object from
    #    the original old one (old was unloaded too) — proves rollback
    #    rebuilt rather than reusing a stale reference.
    rolled = mgr.get_backend_unsafe()
    assert rolled is not old
    # And it isn't broken — preload succeeded → it's a working instance.
    assert isinstance(rolled, CachingFakeBackend)
    assert rolled.broken is False
