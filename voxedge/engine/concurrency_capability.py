"""Concurrency capability descriptor for ASR/TTS backends.

Copied/adapted from app/core/concurrency_capability.py, dedup after Phase 1b.
This is the pure-Python "slot layer" descriptor migrated into voxedge per
docs/specs/edge-voice-library-architecture.md §3.1 ("并发抽象(slot 层)归属").
The dataclass is byte-equivalent to the app/core original (zero coupling —
no env, no registry, no app imports).

Describes runtime concurrency properties (separate from feature capabilities
such as ``STREAMING``) so the coordinator can derive ceilings from backend
reality instead of profile-only defaults.

See ``docs/specs/concurrency-capability-framework.md`` (Section 1) for the
full field semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


ScalingMode = Literal[
    "single_runtime_multiplex",
    "multi_runtime_per_slot",
    "per_call_isolated",
    "external_managed",
]


@dataclass(frozen=True)
class ConcurrencyCapability:
    """Runtime concurrency descriptor.

    Fields match Section 1 of the spec:

    - ``supports_parallel``: backend can serve >1 concurrent in-flight request.
    - ``max_concurrent``: hard ceiling, ``>=1`` or ``None`` (no fixed cap;
      treated as ``+inf`` when participating in a ``min()`` aggregate).
    - ``is_stateful``: backend holds per-session/per-call state.
    - ``requires_exclusive_device``: cross-backend mutex on the device;
      *not* a within-backend single-slot constraint (that is expressed by
      ``supports_parallel`` + ``max_concurrent``).
    - ``scaling_mode``: how additional concurrency is realized.
    - ``vram_mb_per_slot``: optional metadata for P2 VRAM budgeting.
    """

    supports_parallel: bool = False
    max_concurrent: Optional[int] = 1
    is_stateful: bool = True
    requires_exclusive_device: bool = True
    scaling_mode: ScalingMode = "single_runtime_multiplex"
    vram_mb_per_slot: Optional[int] = None

    @classmethod
    def default(cls) -> "ConcurrencyCapability":
        """Conservative default: serialized, single-slot, stateful, exclusive.

        Backends that do not override the descriptor resolve to this, which
        preserves N=1 safety for legacy/untouched backends.
        """
        return cls()
