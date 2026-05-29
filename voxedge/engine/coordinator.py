"""Backend execution coordinator (voxedge engine slot layer).

Copied/adapted from app/core/coordinator.py, dedup after Phase 1b.
Migrated per docs/specs/edge-voice-library-architecture.md §3.1 — the
"N=2 small-GPU doesn't crash" moat. stdlib-only (asyncio); no app imports.

Execution mode drives slot acquire:
  * concurrent  : no lock, ASR and TTS run in parallel.
  * serialized  : single asyncio.Lock shared by both slots; mutually
                  exclusive.
  * exclusive   : same lock + slot tracking; switching slot calls the
                  dormant backend's ``unload()`` before yielding. Best-effort:
                  backends not overriding ``unload()`` stay resident.

Spec docs/specs/concurrency-capability-framework.md §4: backend capability is
the ceiling, the requested mode is the floor. ``concurrent`` is permitted only
when BOTH active backends declare a parallel-capable
``ConcurrencyCapability``. If either cannot run parallel, the mode is
downgraded to ``serialized``. ``exclusive`` is always honored as-is.

DECOUPLING vs app/core/coordinator.py:
  * The original lazily imported ``app.core.capability_resolver.resolve``
    (coordinator.py:43) and consulted a ``profile`` dict. voxedge instead
    constructs from live backend INSTANCES and uses
    ``voxedge.engine.capability_resolver.resolve`` (no profile, no env,
    no registry). No ``app.*`` import remains.
  * The module-level singleton (``init_coordinator`` / ``get_coordinator``,
    coordinator.py:99-113) is dropped — the engine owns its coordinator
    instance and passes it explicitly (no global state in the library).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, Literal, Optional

from voxedge.engine.capability_resolver import resolve as _resolve_capability

logger = logging.getLogger(__name__)

Slot = Literal["asr", "tts"]


class BackendCoordinator:
    """Coordinates ASR/TTS slot acquisition under a resolved execution mode.

    Construct one of two ways:

    * ``BackendCoordinator.from_backends(asr=..., tts=..., requested_mode=...)``
      — resolves the effective mode from the live backends' capabilities
      (preferred; the engine path). The mode may downgrade concurrent →
      serialized per spec §4.
    * ``BackendCoordinator(mode="serialized")`` — pin an explicit mode
      directly (tests / advanced callers that have already resolved it).
    """

    def __init__(self, mode: str = "concurrent"):
        self._mode = mode
        self._lock: Optional[asyncio.Lock] = None
        if self._mode in ("serialized", "exclusive"):
            self._lock = asyncio.Lock()
        self._active_slot: Optional[Slot] = None
        # callables returning the currently-loaded backend per slot (for the
        # exclusive-mode unload(); set via register_backend or from_backends).
        self._backend_getters: Dict[Slot, Callable[[], Any]] = {}

    # -- construction from live backends (decoupled resolver) ---------------

    @classmethod
    def from_backends(
        cls,
        *,
        asr: Any = None,
        tts: Any = None,
        requested_mode: str = "concurrent",
    ) -> "BackendCoordinator":
        """Build a coordinator whose mode is resolved from backend capability.

        Replaces app/core/coordinator.py:64-68 (``__init__(policy, profile)``
        → ``_resolve_mode``) with instance-driven resolution.
        """
        resolved = _resolve_capability(
            asr_backend=asr, tts_backend=tts, requested_mode=requested_mode
        )
        if (
            requested_mode == "concurrent"
            and resolved.coordinator_mode == "serialized"
        ):
            logger.info(
                "coordinator: downgrading concurrent -> serialized "
                "(asr.supports_parallel=%s/max=%s, tts.supports_parallel=%s/max=%s)",
                resolved.asr_cap.supports_parallel,
                resolved.asr_cap.max_concurrent,
                resolved.tts_cap.supports_parallel,
                resolved.tts_cap.max_concurrent,
            )
        coord = cls(mode=resolved.coordinator_mode)
        # Pre-register backends so exclusive-mode unload() can reach them.
        if asr is not None:
            coord.register_backend("asr", lambda b=asr: b)
        if tts is not None:
            coord.register_backend("tts", lambda b=tts: b)
        return coord

    @property
    def mode(self) -> str:
        return self._mode

    def register_backend(self, slot: Slot, getter: Callable[[], Any]) -> None:
        """Register a callable returning the currently-loaded backend for the
        slot (used by exclusive-mode unload). Mirrors coordinator.py:77-79."""
        self._backend_getters[slot] = getter

    @asynccontextmanager
    async def acquire(self, slot: Slot) -> AsyncIterator[None]:
        """Acquire the slot for the duration of a backend call.

        Verbatim semantics from app/core/coordinator.py:81-96.
        """
        if self._mode == "concurrent" or self._lock is None:
            yield
            return
        async with self._lock:
            if self._mode == "exclusive" and self._active_slot not in (None, slot):
                # unload the previously active slot's backend if available.
                other = self._active_slot
                getter = self._backend_getters.get(other)
                if getter is not None:
                    backend = getter()
                    if backend is not None and hasattr(backend, "unload"):
                        backend.unload()
            self._active_slot = slot
            yield


__all__ = ["BackendCoordinator", "Slot"]
