"""Per-utterance ASR session manager — re-export shim (G5 dedup).

The canonical implementation now lives in
``voxedge.engine.asr_session_manager``. This module is a thin re-export so
existing ``app.core.asr_session_manager`` imports keep working against the
single source of truth (which probes ``backend.sample_rate`` instead of
hardcoding 16000 Hz — env-free, works with any ASR sample rate).

Previously this file held a duplicate copy kept during the voxedge migration;
keeping two copies risked silent divergence when an ASR fix landed in only one.
This dedups it. The git history of the old in-repo implementation is preserved.

Requires the ``voxedge`` package to be importable (editable install for dev;
the docker images must ship/``pip install`` voxedge — see DEVELOP.md).
"""
from voxedge.engine.asr_session_manager import (  # noqa: F401
    ASRSessionManager,
    ASRSessionUnavailable,
    SessionState,
)

__all__ = ["ASRSessionManager", "ASRSessionUnavailable", "SessionState"]
