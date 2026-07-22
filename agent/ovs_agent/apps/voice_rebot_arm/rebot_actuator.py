"""rebot_actuator — RebotArmActuator: the B601-DM concrete :class:`Actuator`.

Phase A: basic cartesian-waypoint motion. Wraps the vendored
:class:`~ovs_agent.apps.voice_rebot_arm.rebot_arm.RebotArm` (B601-DM SDK +
CAN gripper force-control state machine) behind the framework's
:class:`~ovs_agent.actuators.base.Actuator` ABC so it reuses ArmPlugin's
tool registration, observation HTTP server, torque gate and cancellation —
no bespoke wiring.

Frame model (differs from SO-ARM joint angles):
  Each ``execute_sequence`` frame's ``joints`` dict is a **cartesian
  waypoint**, not joint angles. Recognised fields:

    x, y, z         — TCP position in metres (IK target)
    roll, pitch, yaw — TCP orientation in radians
    gripper         — signed-magnitude gripper command (per-frame amplitude):
                        > 0  → open to that WIDTH in metres (clamp max_open)
                        < 0  → grasp with that FORCE in N·m (= |gripper|)
                        == 0 → hold (leave the gripper untouched)
    delay (frame-level) — settle pause after the waypoint

  ``move_to`` runs IK internally so we never hand-calibrate joints. Missing
  position fields default to "hold current" semantics via the cache.

Connection gotchas (see config.yaml comments — fixed in config, surfaced
here for the reader):
  ① channel defaults to ttyACM0 (SO-ARM) in the SDK — we MUST pass the
     B601-DM port (ttyACM1) explicitly via config.
  ② the SDK uses ``channel.startswith("/dev/tty")`` to pick serial vs
     SocketCAN, so the channel MUST be a realpath ``/dev/ttyACM1``, never a
     ``/dev/serial/by-id/...`` symlink (which would be misread as CAN).
  ③ /dev/ttyACM1 needs ``--device`` + root in the container (dialout group
     is bypassed by running as root).

Discipline: synchronous (ArmPlugin wraps in ``asyncio.to_thread``) and
env-free at construction (the builder translates config → ctor kwargs).
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from ovs_agent.actuators.base import Actuator
from ovs_agent.actuators.factory import register_actuator
from ovs_agent.actuators.serial_resolve import resolve_serial_port

logger = logging.getLogger(__name__)

# Deferred import: the RebotArm constructor touches the C-extension SDK, so we
# only construct it inside connect(). Importing the *class* is SDK-free (its
# heavy imports live in __init__), so a Mac without the SDK can import this
# module and even instantiate the actuator (which holds no RebotArm until
# connect()).
from .rebot_arm import RebotArm


def sleep_cancellable(
    delay: float, cancel_event: Optional[threading.Event]
) -> bool:
    """Sleep ``delay`` seconds in small steps, polling ``cancel_event``.

    Returns True if the full delay elapsed (or there was nothing to wait for),
    False if ``cancel_event`` fired before the deadline. Shared by the actuator
    settle pause and the grasp pipeline's motion-settle wait so both honour the
    same cancellation cadence.
    """
    if delay <= 0:
        return True
    if cancel_event is None:
        time.sleep(delay)
        return True
    end = time.monotonic() + delay
    while time.monotonic() < end:
        if cancel_event.is_set():
            return False
        time.sleep(min(0.05, max(0.0, end - time.monotonic())))
    return not cancel_event.is_set()


# Observation schema fields exposed by GET /observation/schema and used by
# ActionsManager to validate that every saved frame supplies these fields.
_CARTESIAN_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")

# The frame "gripper" field is a signed magnitude (see _apply_gripper):
# positive = open width (metres), negative = grasp force (N·m), 0 = hold.


class RebotArmActuator(Actuator):
    """Owns the B601-DM CAN connection (via RebotArm) + observation cache."""

    def __init__(
        self,
        channel: str,
        repo_root: Optional[str] = None,
        config_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        gripper_cfg_path: Optional[str] = None,
        move_duration: float = 2.0,
        grasp_force: Optional[float] = None,
        open_distance_m: float = 0.09,
    ) -> None:
        self._channel = channel
        self._repo_root = repo_root
        self._config_path = config_path
        self._urdf_path = urdf_path
        self._gripper_cfg_path = gripper_cfg_path
        # SAFETY: validate the gripper/motion amplitude clamps up front. A
        # non-finite or negative clamp would silently defeat the per-frame
        # clamps in _apply_gripper (e.g. a negative open_distance_m → min()
        # never bounds the request; a NaN compares False to everything).
        self._move_duration = self._validate_nonneg_finite(
            move_duration, "move_duration"
        )
        if grasp_force is not None:
            grasp_force = self._validate_nonneg_finite(grasp_force, "grasp_force")
        self._grasp_force = grasp_force
        self._open_distance_m = self._validate_nonneg_finite(
            open_distance_m, "open_distance_m"
        )

        self._robot: Optional[RebotArm] = None
        self._latest_obs: Dict[str, Any] = {}
        self._schema: Dict[str, Any] = {
            f: {"type": "float"} for f in _CARTESIAN_FIELDS
        }
        # Single actuator-level lock serializes all bus-touching ops
        # (move_to, gripper commands, observation reads). The gripper's own
        # 500Hz force-control thread already self-synchronizes on the CAN
        # bus via the SDK's RLock; this lock additionally prevents a move
        # and a gripper command from interleaving from the asyncio side.
        self._lock = threading.RLock()
        # Torque is enabled at connect(enable=True); track it as the single
        # source of truth for "can we move?" (mirrors SO-ARM semantics).
        self._torque_state: str = "off"

    @staticmethod
    def _validate_nonneg_finite(value: Any, name: str) -> float:
        f = float(value)
        if not math.isfinite(f) or f < 0.0:
            raise ValueError(
                f"rebot_arm actuator {name!r} must be a non-negative finite "
                f"number; got {value!r}"
            )
        return f

    @property
    def robot(self):
        """The underlying :class:`RebotArm`, or ``None`` before connect().

        Phase B's grasp pipeline (``grasp_service.run_grasp_once``) drives the
        arm directly (move_to / get_tcp_pose / grasp / gripper) rather than
        through the actuation-sequence abstraction, so it needs the raw arm.
        """
        return self._robot

    @contextlib.contextmanager
    def acquire_motion_lock(self) -> Iterator[None]:
        """Hold the actuator's bus lock for a single arm motion.

        Phase B's grasp pipeline drives the raw arm directly (bypassing
        execute_sequence). To stay mutually exclusive with execute_sequence's
        per-op locking and with observation reads, it MUST take this same lock
        around each discrete bus operation (one move / one gripper command).
        Like execute_sequence, callers acquire it per-op and release it across
        any blocking wait so torque-off / cache reads are never starved.
        """
        with self._lock:
            yield

    # ── lifecycle ────────────────────────────────────────────────────

    def connect(self) -> None:
        """Construct RebotArm, connect+enable, init gripper, seed schema.

        Blocking. Raises on SDK/hardware failure. The RebotArm constructor
        is what first touches the SDK C-extensions, so on a Mac without the
        SDK this raises ImportError/FileNotFoundError here (NOT at import).
        """
        # CRITICAL: pass channel through. The SDK reads the bus only from its
        # arm.yaml `channel` field (no kwarg) and defaults to ttyACM0 — the
        # SO-ARM's port. RebotArm copies the source arm.yaml, overrides the
        # channel to our configured realpath (ttyACM1), and feeds that temp
        # cfg to the SDK so the B601-DM connects to the correct bus. The
        # gripper rides on the arm's controller, so it inherits this channel.
        self._robot = RebotArm(
            config_path=self._config_path,
            urdf_path=self._urdf_path,
            repo_root=self._repo_root,
            channel=self._channel,
        )
        self._robot.connect(enable=True)
        self._torque_state = "on"
        try:
            self._robot.init_gripper(self._gripper_cfg_path)
        except Exception as exc:  # pragma: no cover — best-effort gripper
            print(f"[RebotArmActuator] init_gripper failed (continuing): {exc}")
        # Prime the observation cache so verify panels don't flash NaN.
        self.update_cache()

    def disconnect(self) -> None:
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception as exc:  # pragma: no cover — best-effort
                print(f"[RebotArmActuator] disconnect error: {exc}")
            finally:
                self._robot = None
                self._torque_state = "off"

    # ── observation cache ────────────────────────────────────────────

    def update_cache(self) -> Dict[str, Any]:
        """Read a fresh TCP pose + gripper state and update the cache."""
        if self._robot is None:
            return {}
        with self._lock:
            obs = self._read_observation_locked()
            self._latest_obs = dict(obs)
        return obs

    def _read_observation_locked(self) -> Dict[str, Any]:
        """Read x/y/z + gripper from the arm. Caller holds the lock."""
        obs: Dict[str, Any] = {}
        robot = self._robot
        if robot is None:
            return obs
        try:
            T = robot.get_tcp_pose()
            obs["x"] = float(T[0, 3])
            obs["y"] = float(T[1, 3])
            obs["z"] = float(T[2, 3])
            # Orientation is left to FK consumers; Phase A reports position +
            # gripper. roll/pitch/yaw kept in schema for forward-compat but
            # not decoded from the rotation matrix here (TODO Phase B).
        except Exception as exc:  # pragma: no cover — best-effort read
            print(f"[RebotArmActuator] get_tcp_pose failed: {exc}")
        try:
            gp, gv, gt = robot.get_gripper_state()
            obs["gripper"] = float(gp)
            obs["gripper_vel"] = float(gv)
            obs["gripper_torq"] = float(gt)
            obs["gripper_holding"] = bool(robot.gripper_is_holding)
        except Exception:  # pragma: no cover — gripper may be absent
            pass
        return obs

    def get_cached_observation(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_obs)

    def observation_features(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._schema)

    # ── action dispatch ──────────────────────────────────────────────

    def execute_sequence(
        self,
        frames: List[Dict[str, Any]],
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        """Execute a normalized sequence of cartesian-waypoint frames.

        Each frame's ``joints`` dict holds {x,y,z,roll,pitch,yaw,gripper}.

        LOCKING (SAFETY): the actuator lock is held only around each discrete
        bus operation (one move_to / one gripper command, the final cache
        read) — NOT across the whole sequence and NOT across the settle
        ``sleep_cancellable``. Holding it across the multi-second sleep would
        block an emergency ``set_torque(False)`` and ``update_cache`` for the
        entire motion; releasing it between ops keeps each bus touch atomic
        while letting torque-off pre-empt the sequence promptly.

        Between every frame we re-check ``cancel_event`` AND ``torque_enabled``
        — a torque-off (emergency disable) mid-sequence aborts immediately.

        Returns True if dispatched (ran ≥0 frames cleanly), False if not
        connected / empty / a bus op failed / torque was off at start.
        """
        if self._robot is None or not frames:
            return False
        if not self.torque_enabled:
            print(
                "[RebotArmActuator] REFUSING sequence: torque is "
                f"{self._torque_state!r}. Enable torque first."
            )
            return False

        ok = True
        for frame in frames:
            if cancel_event is not None and cancel_event.is_set():
                break
            # SAFETY: torque dropped mid-sequence (emergency disable) → abort.
            if not self.torque_enabled:
                print(
                    "[RebotArmActuator] ABORT sequence: torque went "
                    f"{self._torque_state!r} mid-sequence."
                )
                ok = False
                break
            wp = frame.get("joints") if isinstance(frame, dict) else None
            if not isinstance(wp, dict):
                continue
            # Each bus op is atomic under the lock; the sleep below is not.
            if not self._apply_waypoint(wp):
                ok = False
                break
            delay = (
                float(frame.get("delay", self._move_duration))
                if isinstance(frame, dict)
                else self._move_duration
            )
            # Interruptible settle pause — LOCK RELEASED so set_torque(False)
            # / update_cache can run during the multi-second wait.
            sleep_cancellable(delay, cancel_event)

        # Refresh cache (single locked bus read).
        with self._lock:
            try:
                obs = self._read_observation_locked()
                self._latest_obs = dict(obs)
            except Exception as exc:  # pragma: no cover — best-effort
                print(f"[RebotArmActuator] cache refresh after sequence failed: {exc}")
        return ok

    def _apply_waypoint(self, wp: Dict[str, Any]) -> bool:
        """Apply one cartesian waypoint: move_to (if any pose field) +
        gripper command (if a gripper field).

        Each bus op is wrapped in the actuator lock so it stays atomic against
        the 500Hz gripper thread / observation reads / the grasp pipeline.
        Returns False if a bus op raised (caller aborts the sequence), True
        otherwise. A non-OK frame stops the sequence rather than silently
        continuing on a stale/failed pose.
        """
        robot = self._robot
        if robot is None:
            return False

        has_pose = any(k in wp for k in ("x", "y", "z", "roll", "pitch", "yaw"))
        if has_pose:
            try:
                with self._lock:
                    cur = self._latest_obs
                    x = float(wp.get("x", cur.get("x", 0.0)))
                    y = float(wp.get("y", cur.get("y", 0.0)))
                    z = float(wp.get("z", cur.get("z", 0.0)))
                    roll = float(wp.get("roll", 0.0))
                    pitch = float(wp.get("pitch", 0.0))
                    yaw = float(wp.get("yaw", 0.0))
                    duration = float(wp.get("duration", self._move_duration))
                    robot.move_to(x, y, z, roll, pitch, yaw, duration=duration)
            except Exception as exc:
                print(f"[RebotArmActuator] move_to failed: {exc}")
                return False

        if "gripper" in wp:
            try:
                g = float(wp["gripper"])
            except (TypeError, ValueError):
                g = 0.0
            if not self._apply_gripper(g):
                return False
        return True

    def _apply_gripper(self, g: float) -> bool:
        """Map the frame gripper field (signed magnitude) to an SDK call.

        The gripper field is a per-frame SIGNED MAGNITUDE so each action can
        choose its own opening width / grasp force (not a global config knob):

          * g > 0  → OPEN to ``g`` metres, clamped to ``max_open_m``
                     (``open_distance_m`` config). e.g. 0.06 = open 6 cm.
          * g < 0  → GRASP with force ``|g|`` N·m, clamped to ``max_grasp_force``
                     (``grasp_force`` config, if set). e.g. -0.2 = grasp 0.2 N·m.
          * g == 0 → hold (leave the gripper untouched; arm-motion frames).

        Units are deliberately encoded in the sign (metres when +, N·m when -)
        so a single ``gripper`` field survives the framework's frame validator,
        which strips any non-required keys on save/preview.
        """
        robot = self._robot
        if robot is None or g == 0.0:
            return True
        try:
            with self._lock:
                if g > 0.0:
                    # Clamp to [0, mechanical max]: lower bound guards against
                    # a pathological negative config slipping past validation.
                    dist = max(0.0, min(g, self._open_distance_m))
                    robot.open_gripper(dist)
                else:
                    force = abs(g)
                    if self._grasp_force is not None:
                        # Clamp to [0, safe max].
                        force = max(0.0, min(force, self._grasp_force))
                    robot.grasp(force=force)
        except Exception as exc:
            print(f"[RebotArmActuator] gripper command failed: {exc}")
            return False
        return True

    # ── torque control ───────────────────────────────────────────────

    def set_torque(self, enable: bool) -> None:
        """Enable or disable joint torque.

        Prefer ``RebotArm.set_torque`` — it re-energises the motors AND restarts
        the ArmEndPos controller, which a bare ``_arm.enable()`` does not (after
        an off→on cycle the motors power up but the stale/torn-down controller
        holds no target, so the arm won't move: "torque on but motors dead").
        Falls back to the low-level enable/disable for older wrappers/stubs that
        lack ``set_torque``; that fallback only flips the state flag on enable.
        """
        if self._robot is None:
            raise RuntimeError("arm not connected")
        with self._lock:
            robot_set_torque = getattr(self._robot, "set_torque", None)
            if callable(robot_set_torque):
                try:
                    robot_set_torque(enable)
                    self._torque_state = "on" if enable else "off"
                    return
                except Exception as exc:  # pragma: no cover — hardware path
                    print(
                        f"[RebotArmActuator] RebotArm.set_torque({enable}) failed: "
                        f"{exc}; falling back to low-level enable/disable"
                    )
            arm = getattr(self._robot, "_arm", None)
            if enable:
                enable_fn = getattr(arm, "enable", None)
                if callable(enable_fn):
                    try:
                        enable_fn()
                    except Exception as exc:  # pragma: no cover
                        print(f"[RebotArmActuator] arm.enable failed: {exc}")
                self._torque_state = "on"
            else:
                disable_fn = getattr(arm, "disable", None)
                if callable(disable_fn):
                    try:
                        disable_fn()
                    except Exception as exc:  # pragma: no cover
                        print(f"[RebotArmActuator] arm.disable failed: {exc}")
                # else: SDK has no runtime disable — record intent so the
                # torque gate refuses motion regardless. TODO confirm low-
                # level disable name on powered hardware.
                self._torque_state = "off"

    @property
    def torque_enabled(self) -> bool:
        return self._torque_state == "on"


# Default USB-identity match for the B601-DM (Damiao DM-serial bus, which uses
# an HDSC HC32 USB-CDC bridge). VID:PID 2e88:4603 was read off the real device
# (ID_VENDOR=HDSC). This is the stable primary key; ``vendor`` is a belt-and-
# braces fallback. ``exclude`` is intentionally empty until the SO-ARM's USB
# signature is captured with both arms plugged in (see project memory) — the
# VID:PID match alone already excludes the SO-ARM unless it is the same chip.
_B601_DM_MATCH = {"usb_id": ["2e88:4603"], "vendor": ["hdsc"]}


def _resolve_channel(
    channel: str,
    *,
    match: dict | None = None,
    exclude: dict | None = None,
    ambiguous: str = "error",
) -> str:
    """Resolve the B601-DM serial channel.

    An explicit ``/dev/tty*`` path or ``/dev/serial/by-id/...`` symlink is
    returned as a realpath (operator escape hatch). ``'auto'`` matches by stable
    USB identity via :func:`serial_resolve.resolve_serial_port` — never by
    ``ttyACMx`` number, and never a blind "first port" fallback (that risks
    grabbing the neighbouring SO-ARM). See that module for the safety contract.
    """
    return resolve_serial_port(
        channel,
        match=match or _B601_DM_MATCH,
        exclude=exclude,
        ambiguous=ambiguous,
        log=logger.info,
    )


def _make_rebot_arm(config: dict) -> Actuator:
    """Build a :class:`RebotArmActuator` from the actuator config dict.

    *channel* can be ``'auto'`` (scans ``/dev/serial/by-id/`` for the B601-DM),
    a ``/dev/serial/by-id/...`` symlink, or a direct ``/dev/ttyACM*`` realpath.
    Optional: repo_root, config_path, urdf_path, gripper_cfg_path,
    move_duration, and the gripper-amplitude SAFETY CLAMPS:
      * ``open_distance_m`` — max open width (m); a frame's +gripper is clamped
        to this (default 0.09 = mechanical max).
      * ``grasp_force`` — max grasp force (N·m); a frame's -gripper magnitude is
        clamped to this if set (default None = rely on the SDK's own clamp).
    """
    # Config values often arrive via "${VAR:-default}" env substitution, so an
    # unset var yields an EMPTY STRING (not None). Treat ""/whitespace as
    # "unset" for every optional field — otherwise float('') / a literal ''
    # path would blow up actuator construction (and disable the whole arm).
    def _opt_str(key: str) -> Optional[str]:
        v = config.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    def _opt_float(key: str) -> Optional[float]:
        s = _opt_str(key)
        return None if s is None else float(s)

    raw_channel = _opt_str("channel")
    if not raw_channel:
        raise ValueError(
            "rebot_arm actuator requires a 'channel' in config "
            "(e.g. 'auto', '/dev/ttyACM1', or a /dev/serial/by-id/... path)"
        )
    # Optional USB-identity match overrides from config (metadata.actuator).
    # When channel == 'auto', these drive serial_resolve; explicit paths ignore
    # them. A bare dict in YAML arrives as a dict; tolerate None.
    match = config.get("channel_match") or None
    exclude = config.get("channel_exclude") or None
    ambiguous = (_opt_str("channel_ambiguous") or "error").lower()
    channel = _resolve_channel(
        raw_channel, match=match, exclude=exclude, ambiguous=ambiguous
    )
    return RebotArmActuator(
        channel=channel,
        repo_root=_opt_str("repo_root"),
        config_path=_opt_str("config_path"),
        urdf_path=_opt_str("urdf_path"),
        gripper_cfg_path=_opt_str("gripper_cfg_path"),
        move_duration=_opt_float("move_duration") or 2.0,
        grasp_force=_opt_float("grasp_force"),
        open_distance_m=_opt_float("open_distance_m") or 0.09,
    )


# Self-register so the factory can build us by name once the owning app
# (apps/voice_rebot_arm) is imported.
register_actuator("rebot_arm", _make_rebot_arm)


__all__ = ["RebotArmActuator", "_make_rebot_arm"]
