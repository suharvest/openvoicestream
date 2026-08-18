"""Production metrics module (Week 2).

Backs the public Week 1 helpers with Prometheus counters/gauges while
keeping shadow in-process integers so the legacy ``inc_* -> int`` and
``snapshot()`` contracts continue to work for unit tests and the
``/readyz`` probe.

New helpers (TTFA/RTF/ASR decode/backend state/reload/worker
cancel/queue depth/active WS/GPU watchdog) expose Prometheus metrics
only; they have no Week 1 shadow.

Naming follows ``ovs_<noun>_<verb>[_total]`` (see
``docs/specs/prod-hardening-week2.md``).
"""

from __future__ import annotations

import math
import threading
from typing import Dict, Iterable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Use a module-owned registry so tests can call ``_reset_for_tests()`` to
# rebuild fresh collectors without touching the global default registry,
# which keeps process/platform collectors stable across reloads.

_registry: CollectorRegistry = CollectorRegistry()

_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Week 1 shadow counters (preserved so existing getters return integers)
# ---------------------------------------------------------------------------
_sessions_active: int = 0
_sessions_rejected_total: Dict[str, int] = {}
_auth_rejected_total: Dict[str, int] = {}
_active_ws_sessions: int = 0

# ---------------------------------------------------------------------------
# Backend state catalogue (kept in sync with BackendState enum)
# ---------------------------------------------------------------------------
_BACKEND_STATES: tuple[str, ...] = ("init", "ready", "draining", "reloading", "failed")
_BACKEND_MANAGERS: tuple[str, ...] = ("asr", "tts")


# ---------------------------------------------------------------------------
# Prometheus collectors
# ---------------------------------------------------------------------------

def _build_collectors() -> dict:
    """(Re)create all collectors on ``_registry``.

    Called once on module import and again from ``_reset_for_tests()``.
    """
    c: dict = {}

    c["sessions_active"] = Gauge(
        "ovs_sessions_active",
        "Number of admitted voice sessions currently held by the limiter.",
        registry=_registry,
    )
    c["sessions_rejected"] = Counter(
        "ovs_sessions_rejected_total",
        "Sessions rejected by the global concurrency limiter.",
        ["reason"],
        registry=_registry,
    )
    c["auth_rejected"] = Counter(
        "ovs_auth_rejected_total",
        "API-key auth rejections per endpoint.",
        ["endpoint"],
        registry=_registry,
    )
    c["tts_ttfa"] = Histogram(
        "ovs_tts_ttfa_seconds",
        "Time-to-first-audio for streaming TTS (first PCM after header).",
        ["backend"],
        buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
        registry=_registry,
    )
    c["tts_rtf"] = Histogram(
        "ovs_tts_rtf",
        "TTS real-time factor (wall / synthesised duration).",
        ["backend"],
        buckets=(0.1, 0.3, 0.5, 1.0, 2.0),
        registry=_registry,
    )
    c["asr_decode"] = Histogram(
        "ovs_asr_decode_duration_seconds",
        "ASR decode/finalize wall duration.",
        ["backend"],
        # Upper bound reaches minutes: offline /asr decodes of long clips
        # otherwise all land in +Inf and the histogram carries no signal.
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
        registry=_registry,
    )
    c["asr_cer"] = Gauge(
        "ovs_asr_cer",
        "Rolling ASR CER hook (fed by external eval jobs).",
        ["backend"],
        registry=_registry,
    )
    c["backend_state"] = Gauge(
        "ovs_backend_state",
        "BackendManager state indicator (1 = active, 0 = inactive).",
        ["manager", "state"],
        registry=_registry,
    )
    # Initialise all known (manager,state) combinations to zero so the
    # exposition is stable even before any state transition fires.
    for mgr in _BACKEND_MANAGERS:
        for st in _BACKEND_STATES:
            c["backend_state"].labels(manager=mgr, state=st).set(0)

    c["backend_reload"] = Counter(
        "ovs_backend_reload_total",
        "BackendManager reload outcomes.",
        ["result"],
        registry=_registry,
    )
    c["worker_cancels"] = Counter(
        "ovs_worker_cancels_total",
        "Worker cancellations per backend/reason.",
        ["backend", "reason"],
        registry=_registry,
    )
    c["active_ws"] = Gauge(
        "ovs_active_ws_sessions",
        "Number of accepted streaming WebSocket sessions currently open.",
        registry=_registry,
    )
    c["queue_depth"] = Gauge(
        "ovs_queue_depth",
        "Depth of in-process queues (tts_stream / asr / tts_worker).",
        ["queue"],
        registry=_registry,
    )
    # Watchdog metrics (Deliverable 2 publishes via this module)
    c["watchdog_ok"] = Gauge(
        "ovs_gpu_watchdog_ok",
        "GPU/NPU watchdog cached health (1 = ok, 0 = failed).",
        registry=_registry,
    )
    c["watchdog_check_dur"] = Histogram(
        "ovs_gpu_watchdog_check_duration_seconds",
        "Duration of a single GPU/NPU watchdog probe.",
        buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
        registry=_registry,
    )
    c["watchdog_failures"] = Counter(
        "ovs_gpu_watchdog_failures_total",
        "GPU/NPU watchdog raw probe failures.",
        ["platform", "reason"],
        registry=_registry,
    )
    return c


_C = _build_collectors()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_finite_positive(value: float) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if math.isnan(v) or math.isinf(v):
        return False
    return v >= 0.0


# ---------------------------------------------------------------------------
# Week 1 public API — preserved signatures
# ---------------------------------------------------------------------------

def inc_sessions_active() -> int:
    global _sessions_active
    with _lock:
        _sessions_active += 1
        value = _sessions_active
    _C["sessions_active"].set(value)
    return value


def dec_sessions_active() -> int:
    global _sessions_active
    with _lock:
        if _sessions_active > 0:
            _sessions_active -= 1
        value = _sessions_active
    _C["sessions_active"].set(value)
    return value


def get_sessions_active() -> int:
    with _lock:
        return _sessions_active


def inc_sessions_rejected(reason: str) -> int:
    with _lock:
        _sessions_rejected_total[reason] = _sessions_rejected_total.get(reason, 0) + 1
        value = _sessions_rejected_total[reason]
    _C["sessions_rejected"].labels(reason=reason).inc()
    return value


def get_sessions_rejected(reason: str | None = None) -> int | Dict[str, int]:
    with _lock:
        if reason is None:
            return dict(_sessions_rejected_total)
        return _sessions_rejected_total.get(reason, 0)


def inc_auth_rejected(endpoint: str) -> int:
    with _lock:
        _auth_rejected_total[endpoint] = _auth_rejected_total.get(endpoint, 0) + 1
        value = _auth_rejected_total[endpoint]
    _C["auth_rejected"].labels(endpoint=endpoint).inc()
    return value


def get_auth_rejected(endpoint: str | None = None) -> int | Dict[str, int]:
    with _lock:
        if endpoint is None:
            return dict(_auth_rejected_total)
        return _auth_rejected_total.get(endpoint, 0)


def snapshot() -> dict:
    """Read-only snapshot for tests/readiness."""
    with _lock:
        return {
            "ovs_sessions_active": _sessions_active,
            "ovs_sessions_rejected_total": dict(_sessions_rejected_total),
            "ovs_auth_rejected_total": dict(_auth_rejected_total),
        }


# ---------------------------------------------------------------------------
# Week 1 friendly aliases (spec §Deliverable 1 §2 record_* names)
# ---------------------------------------------------------------------------

def record_session_acquired() -> int:
    return inc_sessions_active()


def record_session_released() -> int:
    return dec_sessions_active()


def record_session_rejected(reason: str) -> int:
    return inc_sessions_rejected(reason)


def record_auth_rejected(endpoint: str) -> int:
    return inc_auth_rejected(endpoint)


# ---------------------------------------------------------------------------
# Week 2 helpers — Prometheus only
# ---------------------------------------------------------------------------

def record_tts_ttfa(backend: str, seconds: float) -> None:
    if not _is_finite_positive(seconds):
        return
    _C["tts_ttfa"].labels(backend=backend or "unknown").observe(float(seconds))


def record_tts_rtf(backend: str, rtf: float) -> None:
    if not _is_finite_positive(rtf):
        return
    _C["tts_rtf"].labels(backend=backend or "unknown").observe(float(rtf))


def record_asr_decode_duration(backend: str, seconds: float) -> None:
    if not _is_finite_positive(seconds):
        return
    _C["asr_decode"].labels(backend=backend or "unknown").observe(float(seconds))


def set_asr_cer(backend: str, cer: float) -> None:
    if not _is_finite_positive(cer):
        return
    _C["asr_cer"].labels(backend=backend or "unknown").set(float(cer))


def set_backend_state(manager: str, state: str, value: int | float = 1) -> None:
    if manager not in _BACKEND_MANAGERS:
        # Don't blow up production on a stray label; ignore silently.
        return
    if state not in _BACKEND_STATES:
        return
    # Set selected to ``value`` and all others to 0 so the manager appears
    # in exactly one state at a time.
    for st in _BACKEND_STATES:
        _C["backend_state"].labels(manager=manager, state=st).set(
            float(value) if st == state else 0.0
        )


def record_backend_reload(result: str) -> None:
    if result not in ("success", "fail", "rollback", "drain_timeout"):
        result = "fail"
    _C["backend_reload"].labels(result=result).inc()


def record_worker_cancel(backend: str, reason: str) -> None:
    _C["worker_cancels"].labels(
        backend=backend or "unknown", reason=reason or "unknown"
    ).inc()


def inc_active_ws_sessions() -> int:
    global _active_ws_sessions
    with _lock:
        _active_ws_sessions += 1
        value = _active_ws_sessions
    _C["active_ws"].set(value)
    return value


def dec_active_ws_sessions() -> int:
    global _active_ws_sessions
    with _lock:
        if _active_ws_sessions > 0:
            _active_ws_sessions -= 1
        value = _active_ws_sessions
    _C["active_ws"].set(value)
    return value


def set_queue_depth(queue: str, depth: int) -> None:
    try:
        d = int(depth)
    except (TypeError, ValueError):
        return
    if d < 0:
        d = 0
    _C["queue_depth"].labels(queue=queue or "unknown").set(d)


# ---------------------------------------------------------------------------
# GPU watchdog metric helpers (called from gpu_watchdog.py)
# ---------------------------------------------------------------------------

def set_gpu_watchdog_ok(ok: bool) -> None:
    _C["watchdog_ok"].set(1 if ok else 0)


def observe_gpu_watchdog_check_duration(seconds: float) -> None:
    if not _is_finite_positive(seconds):
        return
    _C["watchdog_check_dur"].observe(float(seconds))


def record_gpu_watchdog_failure(platform: str, reason: str) -> None:
    _C["watchdog_failures"].labels(
        platform=platform or "unknown", reason=reason or "unknown"
    ).inc()


# ---------------------------------------------------------------------------
# Exposition
# ---------------------------------------------------------------------------

def render_prometheus() -> bytes:
    return generate_latest(_registry)


def prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------------
# Test hook
# ---------------------------------------------------------------------------

def _reset_for_tests() -> None:
    """Test-only hook: reset shadow counters AND rebuild collectors.

    Rebuilding the registry avoids leftover label samples bleeding into
    subsequent tests.
    """
    global _sessions_active, _active_ws_sessions, _registry, _C
    with _lock:
        _sessions_active = 0
        _active_ws_sessions = 0
        _sessions_rejected_total.clear()
        _auth_rejected_total.clear()
    _registry = CollectorRegistry()
    _C = _build_collectors()
