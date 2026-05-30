"""BackendManager: lifecycle + hot-reload for ASR / TTS backend instances.

PR3 scaffold. main.py wiring lands in PR4; real per-backend ``unload()``
implementations land in PR5. This module is intentionally backend-agnostic:
it works with anything obeying the callable contracts injected at construction
time (``factory``, ``preloader``, ``unloader``).

State machine
-------------

    INIT --start()--> READY
                       │
                       │ reload()
                       ▼
                    DRAINING (close WS, wait for inflight HTTP)
                       │
                       ▼
                    RELOADING (unload old, apply profile, build new)
                       │
                       ├─── success ──> READY
                       └─── failure ──> rollback → READY
                                        rollback fails → FAILED

A ``_reload_lock`` keeps reloads strictly serial; concurrent attempts get
``HTTPException(409)``. Request gating happens via :meth:`acquire`, which
holds a context manager bumping ``inflight_http``. WS sessions register
themselves and get hard-closed (code 1012) on reload.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import weakref
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Generic, Protocol, TypeVar

from fastapi import HTTPException, WebSocket

from app.core import profile_loader

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _emit_state(manager_name: str, state: "BackendState") -> None:
    """Defensive helper: update ``ovs_backend_state`` for this manager.

    Imported lazily and wrapped in try/except so an early import failure
    can never break the BackendManager state machine.
    """
    try:
        from app.core import metrics as _m
        _m.set_backend_state(manager_name, state.value)
    except Exception:  # pragma: no cover - never block reload on metrics
        pass


def _emit_reload(result: str) -> None:
    try:
        from app.core import metrics as _m
        _m.record_backend_reload(result)
    except Exception:  # pragma: no cover
        pass


class BackendState(str, Enum):
    INIT = "init"
    READY = "ready"
    DRAINING = "draining"
    RELOADING = "reloading"
    FAILED = "failed"


class WebSocketHandle(Protocol):
    websocket: WebSocket
    task: "asyncio.Task[Any] | None"


def _resolve_profile_path(profile_ref: str) -> Path:
    """Mirror of profile_loader._profile_path (kept private there)."""
    candidate = Path(profile_ref)
    if candidate.is_file():
        return candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs" / "profiles" / candidate.name


def _backend_name_of(backend: Any) -> str:
    """Best-effort label for diagnostics."""
    name = getattr(backend, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(backend).__name__


class BackendManager(Generic[T]):
    """Lifecycle owner for one backend kind (``"tts"`` or ``"asr"``)."""

    def __init__(
        self,
        *,
        name: str,
        factory: Callable[[], T],
        preloader: Callable[[T], None],
        unloader: Callable[[T], None],
        drain_timeout_s: float = 30.0,
        reload_timeout_s: float = 120.0,
        initial_profile_ref: str | None = None,
    ) -> None:
        if name not in ("tts", "asr"):
            raise ValueError(f"BackendManager.name must be 'tts' or 'asr', got {name!r}")
        self.name = name
        self._factory = factory
        self._preloader = preloader
        self._unloader = unloader
        self._drain_timeout_s = drain_timeout_s
        self._reload_timeout_s = reload_timeout_s
        # FIX_4_completion: remember the profile ref that bootstrapped this
        # manager so the very first reload's rollback can re-apply via the
        # original source (custom path or name). Without this seed,
        # ``_last_profile_ref`` would still be None when the first reload
        # fails, forcing the rollback fallback to use the logical profile
        # ``name`` — which is wrong for custom OVS_PROFILE_JSON paths whose
        # ``name`` field doesn't match any file basename.
        self._initial_profile_ref: str | None = initial_profile_ref

        self._state: BackendState = BackendState.INIT
        _emit_state(self.name, self._state)
        self._current: T | None = None
        self._inflight_http: int = 0
        # FIX_4: remember the *original* profile reference (could be a name or a
        # custom path supplied via OVS_PROFILE_JSON / admin reload). Required so
        # rollback re-applies the same source even when the profile's logical
        # ``name`` doesn't match its file basename.
        self._last_profile_ref: str | None = None
        # WeakSet of WS handles so a dropped session doesn't leak.
        self._ws_handles: "weakref.WeakSet[WebSocketHandle]" = weakref.WeakSet()

        self._state_lock = asyncio.Lock()
        self._reload_lock = asyncio.Lock()
        self._drained_cond = asyncio.Condition(self._state_lock)

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """First-time init: build + preload backend; INIT → READY."""
        if self._state != BackendState.INIT:
            raise RuntimeError(
                f"BackendManager[{self.name}].start called in state {self._state.value}"
            )
        try:
            backend = self._factory()
            self._preloader(backend)
        except Exception:
            self._state = BackendState.FAILED
            _emit_state(self.name, self._state)
            logger.exception("BackendManager[%s] start failed", self.name)
            raise
        self._current = backend
        # FIX_4_completion: seed _last_profile_ref with the ref supplied at
        # construction time (typically OVS_PROFILE_JSON / OVS_PROFILE from
        # startup). This ensures the first failed reload's rollback can
        # re-apply via the original source — including custom paths whose
        # profile ``name`` doesn't map back to a default filename.
        if self._initial_profile_ref is not None and self._last_profile_ref is None:
            self._last_profile_ref = self._initial_profile_ref
        async with self._state_lock:
            self._state = BackendState.READY
        _emit_state(self.name, self._state)
        logger.info("BackendManager[%s] ready (%s)", self.name, _backend_name_of(backend))

    async def shutdown(self) -> None:
        """Tear down the current backend. Intended for tests/fixtures."""
        async with self._state_lock:
            self._state = BackendState.DRAINING
        _emit_state(self.name, self._state)
        if self._current is not None:
            try:
                self._unloader(self._current)
            except Exception:
                logger.exception("BackendManager[%s] unload during shutdown failed", self.name)
        self._current = None
        async with self._state_lock:
            self._state = BackendState.FAILED  # not-serving sentinel; caller usually discards
        _emit_state(self.name, self._state)

    # ------------------------------------------------------------------- query

    @property
    def state(self) -> BackendState:
        return self._state

    @property
    def backend_name(self) -> str:
        return _backend_name_of(self._current) if self._current is not None else ""

    @property
    def profile_name(self) -> str | None:
        prof = profile_loader.current_profile()
        n = prof.get("name") if isinstance(prof, dict) else None
        return str(n) if n else None

    def is_ready(self) -> bool:
        return self._state == BackendState.READY and self._current is not None

    def get_backend_unsafe(self) -> T:
        """Return the live backend without bumping inflight counters.

        Only safe for readiness probes / metadata queries; HTTP handlers should
        use :meth:`acquire` so drain logic can see them.
        """
        if self._state != BackendState.READY or self._current is None:
            raise HTTPException(
                status_code=503,
                detail={"error": "backend_unavailable", "state": self._state.value},
            )
        return self._current

    def status(self) -> dict:
        return {
            "state": self._state.value,
            "profile_name": self.profile_name,
            "backend_name": self.backend_name,
            "inflight_http": self._inflight_http,
            "inflight_ws": len(self._ws_handles),
        }

    # --------------------------------------------------------- request gating

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[T]:
        async with self._state_lock:
            if self._state != BackendState.READY or self._current is None:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "backend_unavailable", "state": self._state.value},
                )
            self._inflight_http += 1
            backend = self._current
        try:
            yield backend
        finally:
            async with self._state_lock:
                self._inflight_http -= 1
                if self._inflight_http <= 0:
                    self._inflight_http = max(0, self._inflight_http)
                    self._drained_cond.notify_all()

    def register_ws(self, handle: WebSocketHandle) -> None:
        self._ws_handles.add(handle)

    def unregister_ws(self, handle: WebSocketHandle) -> None:
        self._ws_handles.discard(handle)

    # -------------------------------------------------------------- internals

    async def _wait_for_http_drain(self, timeout: float) -> bool:
        """Wait until ``_inflight_http == 0`` or ``timeout`` elapses.

        Returns True on clean drain, False on timeout (caller proceeds anyway).
        """
        async with self._state_lock:
            if self._inflight_http == 0:
                return True

        # Python 3.10 lacks `asyncio.timeout()` (3.11+). Use `wait_for` over a
        # nested coroutine — same semantics, works on Jetson's 3.10 image.
        async def _wait_zero() -> None:
            async with self._state_lock:
                while self._inflight_http > 0:
                    await self._drained_cond.wait()

        try:
            await asyncio.wait_for(_wait_zero(), timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "BackendManager[%s] drain timeout, inflight_http=%d",
                self.name,
                self._inflight_http,
            )
            return False

    async def _force_close_ws_sessions(self) -> None:
        handles = list(self._ws_handles)
        for h in handles:
            ws = getattr(h, "websocket", None)
            if ws is not None:
                try:
                    res = ws.close(code=1012, reason="backend reload")
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    logger.exception("BackendManager[%s] ws close failed", self.name)
            task = getattr(h, "task", None)
            if task is not None and not task.done():
                task.cancel()

    def _load_profile_kind(self, profile_ref: str) -> dict:
        """Load profile JSON without applying env; used for kind validation."""
        path = _resolve_profile_path(profile_ref)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ----------------------------------------------------------------- reload

    async def reload(
        self,
        profile_ref: str | None = None,
        *,
        reason: str = "admin",
    ) -> dict:
        """Hot-swap the backend, optionally re-applying a profile.

        Always closes registered WS sessions (code 1012). Waits up to
        ``drain_timeout_s`` for in-flight HTTP requests; proceeds regardless.

        Returns a dict describing the outcome (``status``: ``reloaded`` |
        ``rolled_back``). On unrecoverable failure raises ``HTTPException(500)``.
        """
        if self._reload_lock.locked():
            raise HTTPException(
                status_code=409,
                detail={"error": "reload_in_progress"},
            )

        async with self._reload_lock:
            # 1. Pre-conditions ------------------------------------------------
            if self._state != BackendState.READY or self._current is None:
                raise HTTPException(
                    status_code=409,
                    detail={"error": "backend_not_ready", "state": self._state.value},
                )

            # PR5 / FIX_A: backend self-reports whether it can truly release
            # resources in-process. Refuse early (before drain) when the live
            # backend doesn't support hot reload — restarting the whole
            # process is the supported path for those (Jetson in-process TRT).
            if not getattr(self._current, "supports_hot_reload", False):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "hot_reload_not_supported",
                        "kind": self.name,
                        "backend": _backend_name_of(self._current),
                    },
                )

            # 2. Snapshot old state -------------------------------------------
            old_backend: T = self._current  # type: ignore[assignment]
            old_profile = dict(profile_loader.current_profile() or {})
            old_profile_name = old_profile.get("name")
            # FIX_4: rollback must re-apply via the *same* reference we used to
            # load this profile (could be a custom path). Fall back to the
            # profile's logical name if no prior ref is recorded (e.g. the
            # very first reload after startup applied via env, not via reload()).
            old_profile_ref = self._last_profile_ref or (
                str(old_profile_name) if old_profile_name is not None else None
            )

            # 3. Validate new profile kind (if provided) ----------------------
            new_profile_preview: dict = {}
            if profile_ref is not None:
                try:
                    new_profile_preview = self._load_profile_kind(profile_ref)
                except Exception as exc:
                    raise HTTPException(
                        status_code=400,
                        detail={"error": "invalid_profile", "message": str(exc)},
                    ) from exc

                # Note: backend_kind_mismatch gate removed — create_tts_backend /
                # create_asr_backend dispatch via a registry keyed on the
                # profile's *_backend field, so self._factory() after
                # apply_profile() builds the correct new-kind backend.

                # FIX_DRYRUN: pre-flight artifact check. Catches "profile for the
                # wrong device" before we tear down the running backend. Without
                # this, a missing engine path forces a failed preload + rollback
                # dance (see commit fd19877 and memory:
                # backend_manager_rollback_env_pollution).
                missing = profile_loader.find_missing_artifacts(new_profile_preview)
                if missing:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "profile_artifacts_missing",
                            "kind": self.name,
                            "profile": new_profile_preview.get("name"),
                            "missing": missing,
                        },
                    )

            # 4. Drain ---------------------------------------------------------
            async with self._state_lock:
                self._state = BackendState.DRAINING
            _emit_state(self.name, self._state)

            await self._force_close_ws_sessions()
            drained = await self._wait_for_http_drain(self._drain_timeout_s)
            if not drained:
                logger.warning(
                    "BackendManager[%s] drain timed out, hard-proceeding to reload",
                    self.name,
                )

            # 5. Reload --------------------------------------------------------
            async with self._state_lock:
                self._state = BackendState.RELOADING
            _emit_state(self.name, self._state)
            self._current = None

            new_backend: T | None = None
            try:
                try:
                    self._unloader(old_backend)
                except Exception:
                    logger.exception(
                        "BackendManager[%s] old unload raised; continuing", self.name
                    )

                if profile_ref is not None:
                    profile_loader.apply_profile(profile_ref, resolve_engines=True)

                new_backend = self._factory()
                self._preloader(new_backend)

                self._current = new_backend
                # FIX_4: record the successful ref so a *future* failed reload
                # rolls back via this exact reference.
                if profile_ref is not None:
                    self._last_profile_ref = profile_ref
                async with self._state_lock:
                    self._state = BackendState.READY
                _emit_state(self.name, self._state)
                _emit_reload("success")

                logger.info(
                    "BackendManager[%s] reloaded (reason=%s) %s → %s",
                    self.name,
                    reason,
                    old_profile_name,
                    profile_loader.current_profile().get("name") if profile_ref else old_profile_name,
                )
                return {
                    "status": "reloaded",
                    "kind": self.name,
                    "old_profile": old_profile_name,
                    "new_profile": (
                        profile_loader.current_profile().get("name")
                        if profile_ref else old_profile_name
                    ),
                    "backend_name": self.backend_name,
                    "drained_cleanly": drained,
                }

            except Exception as exc:
                _emit_reload("fail")
                logger.exception("BackendManager[%s] reload failed; rolling back", self.name)
                # Unload the partially-constructed NEW backend so the factory's
                # module-level cache (main.py _asr_backend / _tts_service_mod._backend)
                # is cleared. Without this, the rollback's self._factory() returns the
                # broken cached instance whose _config snapshot still points at the
                # NEW profile's (missing) artifact paths. See memory:
                # backend_manager_rollback_env_pollution for the orin-nano repro.
                if new_backend is not None:
                    try:
                        self._unloader(new_backend)
                    except Exception:
                        logger.exception(
                            "BackendManager[%s] failed-new-backend unload raised; continuing",
                            self.name,
                        )
                # --- rollback to old profile + fresh factory ----------------
                try:
                    if profile_ref is not None and old_profile_ref is not None:
                        # FIX_4: re-apply via the original ref (path or name), not
                        # the logical profile name — paths like
                        # ``OVS_PROFILE_JSON=/custom/foo.json`` can carry an
                        # arbitrary ``name`` that doesn't resolve via the default
                        # ``configs/profiles/<name>.json`` lookup.
                        # resolve_engines=True so the rolled-back backend gets
                        # the old profile's engine env re-injected AND the failed
                        # reload's injected engine keys cleared via the unified
                        # _APPLIED_KEYS reconciliation (avoids env pollution).
                        profile_loader.apply_profile(
                            old_profile_ref, resolve_engines=True
                        )
                    restored = self._factory()
                    self._preloader(restored)
                    self._current = restored
                    async with self._state_lock:
                        self._state = BackendState.READY
                    _emit_state(self.name, self._state)
                    _emit_reload("rollback")
                    return {
                        "status": "rolled_back",
                        "kind": self.name,
                        "error": str(exc),
                        "old_profile": old_profile_name,
                    }
                except Exception as rb_exc:
                    logger.exception(
                        "BackendManager[%s] rollback also failed; entering FAILED",
                        self.name,
                    )
                    self._current = None
                    async with self._state_lock:
                        self._state = BackendState.FAILED
                    _emit_state(self.name, self._state)
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": "reload_and_rollback_failed",
                            "reload_error": str(exc),
                            "rollback_error": str(rb_exc),
                        },
                    ) from rb_exc


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_tts_manager: BackendManager | None = None
_asr_manager: BackendManager | None = None


def init_backend_managers(
    *,
    tts_factory: Callable[[], Any],
    tts_preloader: Callable[[Any], None],
    tts_unloader: Callable[[Any], None],
    asr_factory: Callable[[], Any],
    asr_preloader: Callable[[Any], None],
    asr_unloader: Callable[[Any], None],
    drain_timeout_s: float = 30.0,
    reload_timeout_s: float = 120.0,
    initial_profile_ref: str | None = None,
) -> None:
    """Install global TTS/ASR managers. Call exactly once during startup.

    ``initial_profile_ref`` (keyword-only, optional) is the profile reference
    used to bootstrap the process (typically ``OVS_PROFILE_JSON`` or
    ``OVS_PROFILE``). It seeds ``BackendManager._last_profile_ref`` so the
    first failed reload's rollback re-applies via the same source — required
    for custom paths whose profile ``name`` doesn't map back to a default
    filename (see FIX_4_completion).
    """
    global _tts_manager, _asr_manager
    if _tts_manager is not None or _asr_manager is not None:
        raise RuntimeError("init_backend_managers called twice")
    _tts_manager = BackendManager(
        name="tts",
        factory=tts_factory,
        preloader=tts_preloader,
        unloader=tts_unloader,
        drain_timeout_s=drain_timeout_s,
        reload_timeout_s=reload_timeout_s,
        initial_profile_ref=initial_profile_ref,
    )
    _asr_manager = BackendManager(
        name="asr",
        factory=asr_factory,
        preloader=asr_preloader,
        unloader=asr_unloader,
        drain_timeout_s=drain_timeout_s,
        reload_timeout_s=reload_timeout_s,
        initial_profile_ref=initial_profile_ref,
    )


def tts_manager() -> BackendManager:
    if _tts_manager is None:
        raise RuntimeError("tts_manager() called before init_backend_managers()")
    return _tts_manager


def asr_manager() -> BackendManager:
    if _asr_manager is None:
        raise RuntimeError("asr_manager() called before init_backend_managers()")
    return _asr_manager


def _reset_for_tests() -> None:
    """Test-only hook: drop module-level managers."""
    global _tts_manager, _asr_manager
    _tts_manager = None
    _asr_manager = None
