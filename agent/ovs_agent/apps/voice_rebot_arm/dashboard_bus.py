"""In-process bus between the grasp pipeline and the arm dashboard.

The pipeline (worker threads) publishes decision frames + result events; the
dashboard plugin (asyncio) reads snapshots. Everything is bounded (ring
buffers) and lock-guarded; publishing never blocks on a slow consumer and a
missing dashboard costs nothing but the annotation encode.

A module-level singleton (``BUS``) keeps the wiring trivial: grasp_plugin and
dashboard_plugin live in the same process and there is exactly one arm.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Optional


class DashboardBus:
    def __init__(self, frame_history: int = 8, event_history: int = 50) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._frame: Optional[dict] = None  # latest only; history keeps meta
        self._frame_meta_hist: deque = deque(maxlen=frame_history)
        self._events: deque = deque(maxlen=event_history)

    # ── producer side (pipeline worker threads) ──────────────────────
    def publish_frame(
        self,
        jpg: bytes,
        depth_jpg: Optional[bytes],
        meta: dict,
    ) -> None:
        with self._lock:
            self._seq += 1
            meta = {**meta, "seq": self._seq, "ts": time.time()}
            self._frame = {"jpg": jpg, "depth_jpg": depth_jpg, "meta": meta}
            self._frame_meta_hist.append(meta)

    def publish_event(self, kind: str, payload: dict) -> None:
        with self._lock:
            self._events.append({"kind": kind, "ts": time.time(), **payload})

    # ── consumer side (dashboard plugin) ─────────────────────────────
    def latest_jpg(self) -> Optional[bytes]:
        with self._lock:
            return self._frame["jpg"] if self._frame else None

    def latest_depth_jpg(self) -> Optional[bytes]:
        with self._lock:
            return (self._frame or {}).get("depth_jpg")

    def snapshot(self) -> dict:
        """Meta-only state for the polling /api/state endpoint (no images)."""
        with self._lock:
            return {
                "frame_seq": self._seq,
                "frame_meta": dict(self._frame["meta"]) if self._frame else None,
                "frame_history": [dict(m) for m in self._frame_meta_hist],
                "events": [dict(e) for e in self._events],
            }


BUS = DashboardBus()
