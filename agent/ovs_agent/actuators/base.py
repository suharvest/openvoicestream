"""Actuator ABC — the pluggable motion-device contract.

A concrete actuator (SO-ARM, pan/tilt head, gripper, …) owns:

  * a serial / bus connection to a physical motor group,
  * an observation cache (latest joint positions) served read-only to the
    observation HTTP server without re-touching the bus,
  * a torque kill-switch that gates whether motion commands are accepted.

Discipline (mirrors ``voxedge/voxedge/backends/base.py``):

  * **Synchronous.** Serial I/O is blocking; the actuator methods block.
    Async callers (ArmPlugin) wrap them in ``asyncio.to_thread`` — the
    actuator never owns an event loop.
  * **Env-free.** The constructor takes explicit config (port, ids,
    delays). Nothing reads ``os.environ`` at import or construction time;
    the factory translates a config dict into ctor kwargs.

Torque model:
  ``set_torque(enable)`` drives the physical bus AND records the new state
  on the actuator. ``torque_enabled`` is the single public read of that
  state — callers (the voice pipeline's ``execute_action`` safety check
  and the observation server's ``/torque`` endpoints) read it instead of
  poking any private field.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ActuatorClosed(RuntimeError):
    """disconnect() landed while connect() was still in flight.

    connect() may run on a worker thread that task cancellation cannot kill, so
    a shutdown racing it can only be signalled by the actuator itself. Callers
    that retry connect() must treat this as "stop", not as a transient failure:
    retrying re-energises hardware the operator just asked to release.
    """


class Actuator(ABC):
    """Owns one physical motion device + its observation cache."""

    # ── lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> None:
        """Open the bus, seed the observation schema + cache.

        Blocking. Raises on hardware/driver failure.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the bus. Best-effort; never raises."""

    # ── observation cache ────────────────────────────────────────────

    @abstractmethod
    def update_cache(self) -> Dict[str, Any]:
        """Read a fresh observation from the device and update the cache.

        Returns the freshly read flat ``{field: value}`` dict (empty if
        the device is not connected).
        """

    @abstractmethod
    def get_cached_observation(self) -> Dict[str, Any]:
        """Return the last cached observation without touching the bus."""

    @abstractmethod
    def observation_features(self) -> Dict[str, Any]:
        """Return the schema (``{field: {"type": ...}}``) for the device.

        Drives ``GET /observation/schema`` and the configurable
        required-field set in ``ActionsManager``.
        """

    # ── action dispatch ──────────────────────────────────────────────

    @abstractmethod
    def execute_sequence(
        self,
        frames: List[Dict[str, Any]],
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        """Execute a normalized sequence of ``{joints, delay}`` frames.

        If ``cancel_event`` fires mid-sequence the loop breaks early.
        Refreshes the observation cache when done. Returns True if the
        sequence was dispatched, False otherwise (not connected / empty).
        """

    # ── torque control ───────────────────────────────────────────────

    @abstractmethod
    def set_torque(self, enable: bool) -> None:
        """Enable or disable joint torque (whole device).

        Drives the physical bus AND updates ``torque_enabled``. Raises
        RuntimeError if the device is not connected.
        """

    @property
    @abstractmethod
    def torque_enabled(self) -> bool:
        """Whether torque is currently enabled.

        Single source of truth for "can we move?". The voice pipeline's
        action-dispatch safety check and the observation server's
        ``/torque`` endpoints both read this public property — no caller
        pokes a private torque field.
        """


__all__ = ["Actuator"]
