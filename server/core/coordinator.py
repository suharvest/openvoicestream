"""Backend execution coordinator.

╔══════════════════════════════════════════════════════════════════════════╗
║ MIGRATION TWIN — read before editing.                                      ║
║ This is the ACTIVE, profile-driven coordinator wired into the HTTP path    ║
║ (server/main.py acquires asr/tts slots here). Its env-free counterpart is  ║
║ ``voxedge.engine.coordinator`` — the convergence target the product is     ║
║ migrating onto (used today only on the /v2v ``OVS_V2V_ENGINE=voxedge``     ║
║ path). They intentionally share the spec                                   ║
║ docs/specs/concurrency-capability-framework.md.                            ║
║ DO NOT add new behaviour here — add it to the voxedge copy and adapt. See  ║
║ ARCHITECTURE.md → "Known structural debt".                                 ║
╚══════════════════════════════════════════════════════════════════════════╝


execution_policy in profile JSON drives this:
- concurrent  : no lock, ASR and TTS run in parallel
- serialized  : single asyncio.Lock shared by both slots; mutually exclusive
- exclusive   : same lock + slot tracking; switching slot calls dormant
                backend.unload() before yielding. Best-effort: backends not
                overriding unload() stay resident.

Spec docs/specs/concurrency-capability-framework.md §4: backend capability
is the ceiling, profile.execution_policy is the floor. ``concurrent`` is
permitted only when BOTH active backends declare
``supports_parallel=True`` with ``max_concurrent > 1``. If either is
``supports_parallel=False``, the mode is downgraded to ``serialized``.
Profile ``exclusive`` is always honored as-is.
"""
from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Dict, Literal, Optional

logger = logging.getLogger(__name__)

Slot = Literal["asr", "tts"]


def _resolve_mode(policy: dict, profile: Optional[dict]) -> str:
    """Resolve the effective coordinator mode per spec §4.

    Thin wrapper over ``capability_resolver.resolve`` — the actual
    aggregation + downgrade rules live there. Without ``profile`` we
    return the raw ``policy.mode`` to preserve legacy/test behavior
    (matches the pre-resolver contract).
    """
    requested = (policy or {}).get("mode", "concurrent")
    if requested == "exclusive":
        return "exclusive"
    if not isinstance(profile, dict):
        return requested

    try:
        from server.core.capability_resolver import resolve as _resolve_cap
    except Exception:
        return requested

    resolved = _resolve_cap(profile=profile, policy=policy)
    if (
        requested == "concurrent"
        and resolved.coordinator_mode == "serialized"
    ):
        asr_cap = resolved.asr_cap
        tts_cap = resolved.tts_cap
        logger.info(
            "coordinator: downgrading concurrent -> serialized "
            "(asr.supports_parallel=%s/max=%s, tts.supports_parallel=%s/max=%s)",
            asr_cap.supports_parallel, asr_cap.max_concurrent,
            tts_cap.supports_parallel, tts_cap.max_concurrent,
        )
    return resolved.coordinator_mode


class BackendCoordinator:
    def __init__(self, policy: dict, profile: Optional[dict] = None):
        self._mode = _resolve_mode(policy, profile)
        self._lock: Optional[asyncio.Lock] = None
        if self._mode in ("serialized", "exclusive"):
            self._lock = asyncio.Lock()
        self._active_slot: Optional[Slot] = None
        # store callables to fetch backends lazily (set after services start)
        self._backend_getters: Dict[Slot, Callable] = {}

    @property
    def mode(self) -> str:
        return self._mode

    def register_backend(self, slot: Slot, getter: Callable):
        """Register a callable returning the currently-loaded backend for the slot."""
        self._backend_getters[slot] = getter

    @asynccontextmanager
    async def acquire(self, slot: Slot) -> AsyncIterator[None]:
        if self._mode == "concurrent" or self._lock is None:
            yield
            return
        async with self._lock:
            if self._mode == "exclusive" and self._active_slot not in (None, slot):
                # unload the previously active slot's backend if available
                other = self._active_slot
                getter = self._backend_getters.get(other)
                if getter is not None:
                    backend = getter()
                    if backend is not None and hasattr(backend, "unload"):
                        backend.unload()
            self._active_slot = slot
            yield


_coordinator: Optional[BackendCoordinator] = None


def init_coordinator(
    policy: dict, profile: Optional[dict] = None
) -> BackendCoordinator:
    global _coordinator
    _coordinator = BackendCoordinator(policy, profile=profile)
    return _coordinator


def get_coordinator() -> BackendCoordinator:
    if _coordinator is None:
        raise RuntimeError("coordinator not initialized; call init_coordinator() at startup")
    return _coordinator
