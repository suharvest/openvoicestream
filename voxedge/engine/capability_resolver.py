"""Concurrency-capability resolver (voxedge engine slot layer).

Copied/adapted from app/core/capability_resolver.py, dedup after Phase 1b.
Migrated per docs/specs/edge-voice-library-architecture.md §3.1 — only the
parts the engine's slot orchestration actually needs are ported; the rest is
left as TODO (see below).

WHAT IS MIGRATED (1:1 with app/core/capability_resolver.py):
  * ``_aggregate_ceiling``  ← capability_resolver.py:136-153
        min(asr.max_concurrent, tts.max_concurrent), None == +inf.
  * ``_parallel_ok``        ← capability_resolver.py:156-162
        spec §4 parallel-capability predicate.
  * coordinator-mode resolution (exclusive honored; concurrent → serialized
    downgrade when either backend can't run parallel) ← capability_resolver
    .py:308-330.

KEY DECOUPLINGS vs the app/core original:
  * NO env reads. The original ``_infer_target`` consulted ``os.environ``
    RK_PLATFORM / LANGUAGE_MODE (capability_resolver.py:75-86) — dropped.
    Profile-target inference is not needed for slot orchestration here.
  * NO app backend registry. The original looked the backend CLASS up in
    ``app.core.asr_backend._ASR_REGISTRY`` / ``_TTS_REGISTRY`` via lazy
    import (capability_resolver.py:254-258, 483) and called the classmethod
    ``cls.concurrency_capability(profile)``. voxedge instead takes the live
    backend INSTANCES and reads ``backend.concurrency_capability()``
    (the ABC instance method on voxedge.backends.base, no profile/env).

WHAT IS NOT MIGRATED (intentionally — TODO after Phase 1b):
  * Profile-target → default ceiling table ``_TARGET_DEFAULTS`` +
    ``_infer_target`` (capability_resolver.py:49-88). This is a profile/env
    concern, not slot orchestration. The engine resolves the ceiling purely
    from backend reality; a profile-default fallback can be layered on later.
  * Session-ceiling clamp + OVS_MAX_CONCURRENT_SESSIONS env handling
    (capability_resolver.py:272-306). That is the transport-layer admission
    gate (session_limiter, spec M4), not the engine.
  * Executor max_workers resolution + OVS_TTS_STREAM_MAX_WORKERS env
    (capability_resolver.py:332-415, 423-485). ThreadPoolExecutor sizing is
    an app.main concern, unrelated to the engine slot acquire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

from voxedge.engine.concurrency_capability import ConcurrencyCapability

logger = logging.getLogger(__name__)


CoordinatorMode = Literal["concurrent", "serialized", "exclusive"]


# ---------------------------------------------------------------------------
# Backend-instance capability extraction (decoupled: no registry, no env).
#
# voxedge.backends.base declares ``concurrency_capability()`` as an INSTANCE
# method returning a plain dict ``{"max_concurrency": int, "mode": str}``
# (base.py:144-152 / 261-263). The app/core resolver instead called a
# CLASSMETHOD that returned a typed ``ConcurrencyCapability`` from a profile.
# We normalise either shape into a ``ConcurrencyCapability`` here so the
# aggregation logic below is identical to app/core.
# ---------------------------------------------------------------------------


def _normalize_capability(raw: Any) -> ConcurrencyCapability:
    """Coerce a backend's ``concurrency_capability()`` return into the typed
    descriptor.

    Accepts:
      * a ``ConcurrencyCapability`` already (passed through),
      * the voxedge dict form ``{"max_concurrency": int, "mode": str}``
        (base.py default) — ``mode == "concurrent"`` (or max>1) implies
        ``supports_parallel``,
      * ``None`` / anything else → conservative default.
    """
    if isinstance(raw, ConcurrencyCapability):
        return raw
    if isinstance(raw, Mapping):
        max_n = raw.get("max_concurrency", raw.get("max_concurrent", 1))
        try:
            max_n = int(max_n) if max_n is not None else None
        except (TypeError, ValueError):
            max_n = 1
        mode = str(raw.get("mode", "serialized")).lower()
        # supports_parallel is True when the backend declares a concurrent
        # mode OR an explicit ceiling > 1.
        supports_parallel = mode == "concurrent" or (
            max_n is not None and max_n > 1
        )
        return ConcurrencyCapability(
            supports_parallel=supports_parallel,
            max_concurrent=max_n,
        )
    return ConcurrencyCapability.default()


def capability_of(backend: Any) -> ConcurrencyCapability:
    """Read a live backend instance's concurrency capability.

    Decoupled replacement for capability_resolver.py:122-133 ``_capability_for``
    + the registry lazy-import at :254-260. Graceful fallback to the
    conservative default on any failure.
    """
    if backend is None:
        return ConcurrencyCapability.default()
    getter = getattr(backend, "concurrency_capability", None)
    if not callable(getter):
        return ConcurrencyCapability.default()
    try:
        return _normalize_capability(getter())
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "capability_resolver: concurrency_capability() raised for %r: %s",
            backend, exc,
        )
        return ConcurrencyCapability.default()


# ---------------------------------------------------------------------------
# Core aggregation — copied 1:1 from app/core/capability_resolver.py:136-162.
# ---------------------------------------------------------------------------


def _aggregate_ceiling(
    asr_cap: ConcurrencyCapability, tts_cap: ConcurrencyCapability
) -> tuple[Optional[int], str]:
    """Compute ``min(asr.max_concurrent, tts.max_concurrent)``.

    ``None`` means "no fixed cap" (treated as +inf per spec §1).
    Returns ``(ceiling, label)`` where ``label`` is a human-readable
    diagnostic. Verbatim from app/core/capability_resolver.py:136-153.
    """
    asr_n = asr_cap.max_concurrent
    tts_n = tts_cap.max_concurrent
    if asr_n is None and tts_n is None:
        return None, "asr=inf,tts=inf"
    if asr_n is None:
        return tts_n, f"asr=inf,tts={tts_n}"
    if tts_n is None:
        return asr_n, f"asr={asr_n},tts=inf"
    return min(asr_n, tts_n), f"asr={asr_n},tts={tts_n}"


def _parallel_ok(cap: ConcurrencyCapability) -> bool:
    """spec §4 parallel-capability predicate.

    Verbatim from app/core/capability_resolver.py:156-162.
    """
    if not cap.supports_parallel:
        return False
    if cap.max_concurrent is None:
        return True
    return cap.max_concurrent > 1


# ---------------------------------------------------------------------------
# Resolved snapshot (slimmed: only the slot-orchestration fields).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedCapability:
    """Slot-layer resolution result.

    Trimmed vs app/core's ResolvedCapability (capability_resolver.py:170-198):
    the ``session_ceiling`` / ``executor_max_workers`` fields belonged to the
    transport + app.main concerns that are NOT migrated (see module docstring).

    Fields:
      * ``ceiling``: ``min(asr, tts)`` aggregate, ``None`` == no fixed cap.
      * ``coordinator_mode``: resolved execution policy (spec §4).
      * ``ceiling_source``: short diagnostic label.
      * ``asr_cap`` / ``tts_cap``: raw descriptors (for diagnostics / logs).
    """

    ceiling: Optional[int]
    coordinator_mode: CoordinatorMode
    ceiling_source: str
    asr_cap: ConcurrencyCapability
    tts_cap: ConcurrencyCapability


def resolve(
    *,
    asr_backend: Any = None,
    tts_backend: Any = None,
    requested_mode: str = "concurrent",
) -> ResolvedCapability:
    """Resolve aggregate capability + coordinator mode from backend instances.

    Decoupled replacement for app/core/capability_resolver.py ``resolve``
    (:224-360): no profile, no env, no registry — capability comes straight
    off the live backend instances via :func:`capability_of`.

    Args:
        asr_backend / tts_backend: live backend instances (or None).
        requested_mode: the desired execution policy. ``exclusive`` is always
            honored; ``concurrent`` downgrades to ``serialized`` when either
            backend cannot run in parallel (spec §4); ``serialized`` and any
            other explicit value pass through unchanged.
    """
    asr_cap = capability_of(asr_backend)
    tts_cap = capability_of(tts_backend)
    ceiling, ceiling_source = _aggregate_ceiling(asr_cap, tts_cap)

    # ---- Coordinator mode (spec §4) — mirrors capability_resolver.py:308-330.
    requested = str(requested_mode or "concurrent")
    if requested == "exclusive":
        coordinator_mode: CoordinatorMode = "exclusive"
    elif requested == "concurrent" and not (
        _parallel_ok(asr_cap) and _parallel_ok(tts_cap)
    ):
        coordinator_mode = "serialized"
    elif requested == "concurrent":
        coordinator_mode = "concurrent"
    elif requested == "serialized":
        coordinator_mode = "serialized"
    else:
        coordinator_mode = requested  # type: ignore[assignment]

    return ResolvedCapability(
        ceiling=ceiling,
        coordinator_mode=coordinator_mode,
        ceiling_source=ceiling_source,
        asr_cap=asr_cap,
        tts_cap=tts_cap,
    )


__all__ = [
    "ConcurrencyCapability",
    "ResolvedCapability",
    "CoordinatorMode",
    "capability_of",
    "resolve",
]
