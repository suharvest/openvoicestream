"""rebot_arm — vendored RebotArm driver for the B601-DM arm.

Vendored from reBot-DevArm-Grasp ``drivers/robot/rebot_arm.py``. The only
substantive change vs the upstream copy is :func:`find_rebot_repo_root`,
which no longer carries hard-coded developer paths (``/home/chlorine/seeed``
etc.). Instead it resolves the ``reBotArm_control_py`` SDK from, in order:

  1. an explicit ``repo_root`` hint (wired from agent config
     ``metadata.actuator.config.repo_root``),
  2. the ``REBOT_REPO_ROOT`` environment variable,
  3. the container's canonical install location
     ``/opt/rebot/reBotArm_control_py``.

Like ``apps/voice_arm/so_arm.py``, the heavy SDK import (``motorbridge`` /
``reBotArm_control_py`` C-extensions) is **deferred to connect time** so this
module imports cleanly on a developer Mac that has no SDK installed. The
:class:`RebotArm` constructor performs the first SDK touch, so a Mac unit
test can instantiate the *actuator* (which holds an unconnected RebotArm)
without tripping ImportError — the SDK is only needed once ``connect()``
runs against real hardware.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


_REBOT_REPO_NAME = "reBotArm_control_py"

# Canonical container install location (see Dockerfile.rebot-arm: the SDK is
# cloned here and pip-installed editable).
_CONTAINER_SDK_ROOT = Path("/opt/rebot")


def _is_rebot_repo_root(path: Path) -> bool:
    return path.is_dir() and (path / _REBOT_REPO_NAME).is_dir()


def find_rebot_repo_root(hint: Optional[str] = None) -> Path:
    """Locate the directory that *contains* ``reBotArm_control_py``.

    Resolution order (first match wins):
      1. ``hint`` (agent config ``metadata.actuator.config.repo_root``),
      2. ``$REBOT_REPO_ROOT`` env var,
      3. the container default ``/opt/rebot``.

    Raises ``FileNotFoundError`` with an actionable message when none match.
    NOTE: the upstream developer-machine fallbacks (``~/seeed``,
    ``/home/chlorine/seeed``, ``<cameraws>/sdk``) are intentionally removed —
    this driver only ever runs inside the rebot-arm container or against a
    config-supplied path.
    """
    candidates: list[Path] = []
    if hint:
        candidates.append(Path(hint).expanduser().resolve())
    env_hint = os.environ.get("REBOT_REPO_ROOT")
    if env_hint:
        candidates.append(Path(env_hint).expanduser().resolve())
    candidates.append(_CONTAINER_SDK_ROOT)

    for p in candidates:
        if _is_rebot_repo_root(p):
            return p

    tried = ", ".join(str(p) for p in candidates) or "(none)"
    raise FileNotFoundError(
        f"Cannot locate {_REBOT_REPO_NAME!r} SDK. Tried: {tried}. "
        "Set metadata.actuator.config.repo_root in the agent YAML, the "
        "REBOT_REPO_ROOT env var, or install the SDK to "
        f"{_CONTAINER_SDK_ROOT / _REBOT_REPO_NAME}."
    )


def ensure_rebot_sdk_in_syspath(hint: Optional[str] = None) -> Path:
    repo = find_rebot_repo_root(hint)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def normalize_channel(channel: str) -> str:
    """Resolve ``channel`` to a serial realpath and validate it.

    The SDK's ``_make_controller`` decides serial-vs-SocketCAN with
    ``channel.startswith("/dev/tty")``. A ``/dev/serial/by-id/...`` symlink
    would be misread as a SocketCAN interface, so we ``realpath`` it first
    (resolving by-id symlinks to their ``/dev/ttyACM*`` target) and then
    require the result to start with ``/dev/tty``. Phase A only supports the
    DM-serial transport — non-tty paths (e.g. SocketCAN ``can0``) are rejected
    with an actionable error.
    """
    resolved = os.path.realpath(str(channel))
    if not resolved.startswith("/dev/tty"):
        raise ValueError(
            f"rebot_arm channel must resolve to a serial device "
            f"(/dev/ttyACM*); got {channel!r} → realpath {resolved!r}. "
            "Pass the realpath /dev/ttyACM* (NOT a /dev/serial/by-id/... "
            "symlink that resolves elsewhere, and NOT a SocketCAN path — "
            "Phase A only supports DM-serial)."
        )
    return resolved


def _write_channel_override_yaml(src_cfg_path: str, channel: str) -> str:
    """Load ``src_cfg_path``, override the top-level ``channel`` field, and
    write the result to a process-level temp file. Returns the temp path.

    Only the ``channel`` field is touched — all other fields (motor configs,
    joints, gains, gripper config) are passed through unchanged. The SDK has
    no ``channel`` kwarg; it only reads it from the yaml, so this override is
    the only way to retarget the bus port.
    """
    with open(src_cfg_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"rebot_arm cfg yaml {src_cfg_path!r} did not parse to a mapping"
        )
    data["channel"] = channel
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="rebot_chan_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path


# ── 夹爪状态机常量 ────────────────────────────────────────────────────────────
_G_MAX_DIST_M      = 0.100  # 100mm operational max (ladder-anchored 2026-07-10: ~18mm/rad to 104mm; physical ceiling 105-110mm)
_G_ANGLE_OPEN      = -5.763  # open target; driver settles ~0.1 shy -> ~100mm true opening
_G_OPEN_SOFT_LIMIT = -5.85  # worst case ~104mm, ruler-verified safe (claw borders intact)
_G_ARRIVE_TOL      = 0.12
_G_HARD_STOP_ANGLE = -0.05
_G_TAU_MAX         = 1.5
_G_KP_MOVE         = 5.0
_G_KD_MOVE         = 1.0
_G_OPEN_RATE       = 4.0     # rad/s in free travel (to _G_SLOW_ZONE)
_G_OPEN_RATE_SLOW  = 4.0     # rad/s past _G_SLOW_ZONE; stiction fixed in hardware 2026-07-14, creep no longer needed (was 0.12)
_G_SLOW_ZONE       = -5.0    # boundary of the former stick-slip zone (kept so the creep leg can be re-enabled by one number)
_G_OPEN_TIMEOUT_S  = 5.0     # blocking-open backstop; full open now ~1.5s at constant 4.0 rad/s
_G_CLOSE_TORQUE    = 1.0
_G_KD_CLOSE        = 0.5
_G_STALL_VEL       = 0.05
_G_STARTUP_DIST    = 0.30
_G_KP_HOLD         = 5.0
_G_KD_HOLD         = 1.0
_G_DEFAULT_FORCE   = 0.30
_G_CTRL_RATE       = 500.0
# Physical-evidence thresholds. The control-mode enum is NOT a reliable
# record of whether an object is held (any later gripper command rewrites
# it), so holding/open-complete decisions are grounded in encoder + torque:
_G_HOLD_MIN_GAP    = -0.30   # rad; jaw at least this ajar (~5mm) ⇒ something can be between the fingers
_G_HOLD_TORQ_MIN   = 0.12    # N·m sustained grip torque ⇒ physically clamping an object
# After an INTENTIONAL open, "holding" additionally requires the jaw to have
# stopped at least this far (rad) short of the commanded open target — i.e. an
# object physically blocked it. The soft-limit shortfall at full open is
# ~0.1 rad (0.0853m measured vs 0.09m commanded) WITH residual limit torque,
# which used to false-positive the release verification ("release failed —
# jaw still gripping after full open" on a 0.057m object, real machine
# 2026-06-12). NOTE the physical blind spot: an object ≥~0.082m wide is
# indistinguishable from the soft limit by gap — release verification is only
# trustworthy for objects ≤~0.078m (use a ≤75mm demo box).
_G_OPEN_BLOCKED_RAD = 0.35
_G_OPEN_STALL_S    = 0.20    # s stalled after the open ramp finished ⇒ jaw at its physical open limit
_G_CLOSE_GRACE_S   = 0.40    # s in CLOSING after which stall ⇒ contact even with no start-up travel
                             # (jaw already resting on the object moves < _G_STARTUP_DIST)
# Feedback plausibility rails. The first frame(s) after boot can be pure
# garbage pinned at the register limits (pos≈±PMAX 12.5, vel≈±VMAX 30, seen
# 2026-07-13); caching one poisons the tau_ff compensation in _g_safe_mit
# into a hard slam past the open limit, so such frames must never be cached.
_G_FB_POS_SANE     = 7.0     # rad; |pos| beyond this = corrupt frame
_G_FB_VEL_SANE     = 25.0    # rad/s; |vel| beyond this = corrupt frame
# Boot zero re-referencing. The DM-J4310 encoder is SINGLE-TURN and gripper
# travel exceeds one turn (hardware team, 2026-07-13), so the power-on zero
# is only correct if the claw was fully closed at power-up — otherwise it
# comes up shifted (+4.63 rad on 2026-07-10, +1.67 rad on 2026-07-13). All
# width limits are denominated in encoder angle, so a shifted zero silently
# opens the claw past its ~105mm breaking width. Stall-reference every boot.
_G_ZERO_CHECK_TOL  = 0.15    # rad; stall must land within this of 0
_G_ZERO_CHECK_CAP  = 6.5     # rad; max closing travel hunting the stop (covers full travel)
_G_ZERO_CHECK_S    = 30.0    # s; overall time budget for the check


class _GS:
    IDLE    = 0
    OPENING = 1
    CLOSING = 2
    CONTACT = 3
    HOLDING = 4
    HOMING  = 5


class RebotArm:
    """B601-DM arm ↔ high-level interface, with built-in gripper force-control
    state machine.

    Args:
        config_path: source arm.yaml path; None = SDK default
                     (``<repo_root>/config/arm.yaml``). When ``channel`` is
                     given, this yaml is the *source* whose ``channel`` field
                     is overridden into a temp file passed to the SDK.
        urdf_path:   URDF path; None = SDK default
        repo_root:   reBotArm_control_py parent dir; None = auto-search
        channel:     serial port realpath (e.g. ``/dev/ttyACM1``). The SDK has
                     no ``channel`` kwarg — it only reads ``channel`` from the
                     yaml (defaulting to ``/dev/ttyACM0``, the SO-ARM's port).
                     When provided, we override the source arm.yaml's
                     ``channel`` into a temp cfg so the B601-DM connects to the
                     correct bus. ``None`` → use the source yaml's channel
                     verbatim (SDK default ttyACM0).
    """

    # Probe the two layouts the SDK config files can live under, relative to
    # the located repo root: the packaged ``reBotArm_control_py/config/`` first,
    # then the SDK-relative ``config/`` fallback the upstream driver used.
    def _sdk_cfg_path(self, filename: str) -> Optional[str]:
        cand = self._repo_root / _REBOT_REPO_NAME / "config" / filename
        if cand.exists():
            return str(cand)
        cand = self._repo_root / "config" / filename
        if cand.exists():
            return str(cand)
        return None

    # Where the SDK ships its default arm.yaml relative to the located repo
    # root. ``init_gripper`` probes the same two layouts for gripper.yaml.
    def _default_arm_cfg_path(self) -> Optional[str]:
        return self._sdk_cfg_path("arm.yaml")

    def __init__(
        self,
        config_path: Optional[str] = None,
        urdf_path:   Optional[str] = None,
        repo_root:   Optional[str] = None,
        channel:     Optional[str] = None,
    ) -> None:
        # The SDK import lives here (constructor), not at module scope, so the
        # module imports on a Mac without the SDK. Constructing a RebotArm
        # still requires the SDK; the actuator defers construction to
        # connect() to keep the import-only smoke test SDK-free.
        repo = ensure_rebot_sdk_in_syspath(repo_root)
        self._repo_root = repo

        # Channel (if any) is normalized to a serial realpath up front so an
        # invalid path fails fast before any SDK / bus touch.
        self._channel = normalize_channel(channel) if channel else None
        # Track temp cfg files created for channel override so disconnect()
        # can clean them up.
        self._tmp_cfg_paths: list[str] = []

        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.controllers import ArmEndPos
        from reBotArm_control_py.kinematics import (
            IKSolverParams,
            compute_fk,
            get_end_effector_frame_id,
            load_robot_model,
            pos_rot_to_se3,
        )
        from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik

        # Resolve the source arm.yaml: caller-supplied config_path wins,
        # otherwise the SDK-shipped default. When a channel override is
        # requested we MUST have a concrete source yaml to copy+patch (the SDK
        # accepts no channel kwarg), so fail clearly if it cannot be located.
        cfg = str(config_path) if config_path else None
        if self._channel is not None:
            src_cfg = cfg or self._default_arm_cfg_path()
            if src_cfg is None:
                raise FileNotFoundError(
                    "rebot_arm channel override requires a source arm.yaml: "
                    "pass config_path or install the SDK default at "
                    f"{self._repo_root / _REBOT_REPO_NAME / 'config' / 'arm.yaml'}"
                )
            cfg = _write_channel_override_yaml(src_cfg, self._channel)
            self._tmp_cfg_paths.append(cfg)
        # From here on a failure must NOT leak the channel-override temp file:
        # disconnect() won't run (no object is returned), so clean up inline.
        try:
            self._arm = RobotArm(cfg_path=cfg)

            if urdf_path:
                self._model = load_robot_model(urdf_path=str(urdf_path))
            else:
                self._model = load_robot_model()

            self._data = self._model.createData()
            self._ee_frame_id = get_end_effector_frame_id(self._model)
            self._compute_fk = compute_fk
            self._pos_rot_to_se3 = pos_rot_to_se3
            self._solve_ik = solve_ik
            self._ik_check_params = IKSolverParams(
                max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6
            )
        except Exception:
            self._cleanup_tmp_cfgs()
            raise

        self._endpos_ctrl = None
        self._ArmEndPos = ArmEndPos

        self._connected = False

        # Gripper motor (registered onto the arm's existing CAN bus).
        self._gripper_mot  = None
        self._gripper_kp   = _G_KP_HOLD
        self._gripper_kd   = _G_KD_HOLD
        self._gripper_ctrl = None

        # Gripper state machine.
        self._g_state            = _GS.IDLE
        self._g_lock             = threading.Lock()
        self._g_pos              = 0.0
        self._g_vel              = 0.0
        self._g_torq             = 0.0
        self._g_pos_start        = 0.0
        self._g_q_contact        = 0.0
        self._g_contact_elapsed  = 0.0
        self._g_open_q_des       = _G_OPEN_SOFT_LIMIT
        self._g_open_target      = _G_OPEN_SOFT_LIMIT
        # Set after an intentional open completes/gives up; cleared by any
        # close/grasp. Lets gripper_is_holding distinguish "parked at the
        # commanded open (soft-limit shortfall + residual torque)" from
        # "an object blocked the open" — see _G_OPEN_BLOCKED_RAD.
        self._g_open_last_target: "float | None" = None
        self._g_open_stall_s     = 0.0
        self._g_close_elapsed    = 0.0
        self._g_target_force     = _G_DEFAULT_FORCE
        self._g_bad_frames       = 0
        self._g_loop_thread: Optional[threading.Thread] = None
        self._g_loop_running     = False
        self._g_loop_stop        = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self, enable: bool = True) -> None:
        self._arm.connect()
        if enable:
            self._arm.enable()
            # SAFETY: once enable() succeeds the motors are energised. If any
            # subsequent step (settle / ArmEndPos start) raises, the outer
            # caller will NOT call disconnect() (it never got a constructed
            # object back from connect), so the arm would be left torqued with
            # no controller. Best-effort disable() then re-raise so the arm
            # ends in a safe, de-energised state.
            try:
                time.sleep(0.5)
                self._endpos_ctrl = self._ArmEndPos(self._arm)
                self._endpos_ctrl.start()
            except Exception:
                self._endpos_ctrl = None
                disable_fn = getattr(self._arm, "disable", None)
                if callable(disable_fn):
                    try:
                        disable_fn()
                    except Exception:
                        pass
                raise
            print("[RebotArm] connected, motors enabled")
        else:
            self._arm._request_and_poll()
            print("[RebotArm] connected, motors stay disabled (read-only)")
        self._connected = True

    def disconnect(self) -> None:
        self._g_stop_loop()
        if self._endpos_ctrl is not None:
            try:
                self._endpos_ctrl.end()
            except Exception:
                pass
            self._endpos_ctrl = None
        # SAFETY: explicitly de-energise the joints before tearing down the
        # bus. The SDK's disconnect() may disable internally, but we add an
        # explicit best-effort disable() so the motors are guaranteed to drop
        # torque even if the SDK path changes.
        disable_fn = getattr(self._arm, "disable", None)
        if callable(disable_fn):
            try:
                disable_fn()
            except Exception:
                pass
        try:
            self._arm.disconnect()
        except Exception:
            pass
        self._cleanup_tmp_cfgs()
        self._connected = False
        print("[RebotArm] disconnected")

    def set_torque(self, enable: bool) -> None:
        """Runtime joint-torque toggle that ALSO manages the ArmEndPos controller.

        A bare ``self._arm.enable()`` re-powers the motors but leaves the
        ArmEndPos controller (started in :meth:`connect`) torn down/stale after
        a prior ``disable()``, so the arm holds no target and won't track
        commands — the "torque shows on but the motors don't move" symptom after
        an off→on cycle for manual posing. Mirror connect()/disconnect():

          enable=True  → enable motors → restart a FRESH ArmEndPos controller
                         (it latches the CURRENT pose, so no jump).
          enable=False → end() the controller first (so it neither fights the
                         manual move nor leaves a stale target that snaps on
                         re-enable) → disable motors.

        The gripper force-control loop is independent and left running.
        """
        if not self._connected:
            raise RuntimeError("arm not connected")
        if enable:
            self._arm.enable()
            # Drop any stale controller from before the disable, then start fresh.
            if self._endpos_ctrl is not None:
                try:
                    self._endpos_ctrl.end()
                except Exception:
                    pass
                self._endpos_ctrl = None
            try:
                time.sleep(0.5)
                self._endpos_ctrl = self._ArmEndPos(self._arm)
                self._endpos_ctrl.start()
            except Exception:
                # Leave the arm de-energised rather than torqued with no
                # controller (same safety contract as connect()).
                self._endpos_ctrl = None
                disable_fn = getattr(self._arm, "disable", None)
                if callable(disable_fn):
                    try:
                        disable_fn()
                    except Exception:
                        pass
                raise
            print("[RebotArm] torque re-enabled (motors energised + controller restarted)")
        else:
            # Stop the controller BEFORE de-energising so it does not drive
            # against the manual move or hold a stale target.
            if self._endpos_ctrl is not None:
                try:
                    self._endpos_ctrl.end()
                except Exception:
                    pass
                self._endpos_ctrl = None
            disable_fn = getattr(self._arm, "disable", None)
            if callable(disable_fn):
                disable_fn()
            print("[RebotArm] torque disabled (motors de-energised for manual move)")

    def _cleanup_tmp_cfgs(self) -> None:
        """Unlink any channel-override temp cfg files this instance created."""
        for p in self._tmp_cfg_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        self._tmp_cfg_paths = []

    # ── gripper init ───────────────────────────────────────────────────────────

    def init_gripper(self, cfg_path: Optional[str] = None) -> None:
        """Register the gripper motor onto the arm's CAN bus and start the
        force-control state machine."""
        from motorbridge import CallError, Mode
        from reBotArm_control_py.actuator.gripper import load_cfg as load_gripper_cfg

        if cfg_path is None:
            cfg_path = self._sdk_cfg_path("gripper.yaml")
            if cfg_path is None:
                # Neither layout has the file; fall back to the SDK-relative
                # path string so load_cfg surfaces a clear missing-file error.
                cfg_path = str(self._repo_root / "config" / "gripper.yaml")

        gcfg = load_gripper_cfg(cfg_path)
        gc = gcfg["gripper"]

        vendor = gc.vendor
        if vendor not in self._arm._ctrl_map:
            raise RuntimeError(
                f"gripper vendor={vendor!r} differs from arm vendor; cannot "
                "share Controller"
            )
        ctrl = self._arm._ctrl_map[vendor]

        if vendor == "damiao":
            self._gripper_mot = ctrl.add_damiao_motor(gc.motor_id, gc.feedback_id, gc.model)
        elif vendor == "myactuator":
            self._gripper_mot = ctrl.add_myactuator_motor(gc.motor_id, gc.feedback_id, gc.model)
        elif vendor == "robstride":
            self._gripper_mot = ctrl.add_robstride_motor(gc.motor_id, gc.feedback_id, gc.model)
        else:
            raise ValueError(f"unsupported gripper vendor: {vendor!r}")

        self._gripper_kp   = gc.kp
        self._gripper_kd   = gc.kd
        self._gripper_ctrl = ctrl

        # One RLock serializes arm-loop and gripper-loop bus ops.
        if not hasattr(ctrl, "_bus_lock"):
            ctrl._bus_lock = threading.RLock()
        lock = ctrl._bus_lock

        def _wrap(fn, _lock=lock):
            def _locked(*a, **kw):
                with _lock:
                    return fn(*a, **kw)
            return _locked

        if not hasattr(ctrl, "_bus_lock_patched"):
            ctrl.poll_feedback_once = _wrap(ctrl.poll_feedback_once)
            ctrl._bus_lock_patched = True

        if not hasattr(self._arm, "_bus_lock_patched"):
            for jc in self._arm._joints:
                mot = self._arm._motor_map[jc.name]
                for _mattr in ("send_pos_vel", "send_mit", "request_feedback"):
                    if hasattr(mot, _mattr):
                        setattr(mot, _mattr, _wrap(getattr(mot, _mattr)))
            self._arm._bus_lock_patched = True

        try:
            ctrl.enable_all()
            time.sleep(0.3)
        except CallError as e:
            print(f"[RebotArm] gripper enable warning: {e}")
        try:
            self._gripper_mot.ensure_mode(Mode.MIT, 1000)
        except CallError as e:
            raise RuntimeError(f"gripper MIT mode switch failed: {e}") from e

        self._g_boot_zero_check()

        self._g_start_loop()
        print("[RebotArm] gripper registered on CAN bus, force-control loop started")

    def _g_boot_zero_check(self) -> None:
        """Close to the mechanical stop and verify the encoder reads ~0 there.

        The encoder is single-turn while gripper travel exceeds one turn, so
        the power-on zero is only valid if the claw was fully closed at
        power-up; otherwise every angle-denominated width limit silently
        over-opens the claw (2× observed). Stall-referencing here makes a
        shifted zero impossible to miss.

        DETECTION ONLY (2026-07-13, after the auto-heal re-zeroed onto a
        box left in the jaw): this never calls set_zero. Landing ≉ 0 →
        raise, which makes init_gripper leave the gripper DOWN — no open
        is possible until a human runs the manual re-zero script. The
        normal rest state is closed (put_down closes the claw), so the
        healthy boot lands ≈ 0 in a second and nothing is refused.
        Set REBOT_G_BOOT_ZEROCHECK=0 to skip (e.g. bench debugging).
        """
        if os.environ.get("REBOT_G_BOOT_ZEROCHECK", "1") == "0":
            print("[RebotArm] gripper boot zero-check SKIPPED (env)", flush=True)
            return
        lock = self._gripper_ctrl._bus_lock

        def _fresh(tries: int = 60):
            for _ in range(tries):
                with lock:
                    self._gripper_mot.request_feedback()
                    self._gripper_ctrl.poll_feedback_once()
                time.sleep(0.005)
                s = self._gripper_mot.get_state()
                if s is not None and abs(s.pos) < _G_FB_POS_SANE and abs(s.vel) < _G_FB_VEL_SANE:
                    return s
            return None

        st = _fresh()
        if st is None:
            raise RuntimeError("gripper boot zero-check: no sane feedback frame")

        pos = float(st.pos)
        start_pos = pos
        q = pos
        cap = pos + _G_ZERO_CHECK_CAP
        plateau_ref, plateau_t = pos, time.monotonic()
        deadline = time.monotonic() + _G_ZERO_CHECK_S
        stalled = False
        while time.monotonic() < deadline and q < cap:
            q = min(cap, q + 0.3 / 100.0)          # 0.3 rad/s closing creep
            with lock:
                self._gripper_mot.send_mit(q, 0.0, _G_KP_MOVE, _G_KD_MOVE, 0.0)
                self._gripper_ctrl.poll_feedback_once()
            s = self._gripper_mot.get_state()
            if s is not None and abs(s.pos) < _G_FB_POS_SANE and abs(s.vel) < _G_FB_VEL_SANE:
                pos = float(s.pos)
                if abs(s.torq) > 0.6:
                    stalled = True                  # contact torque at the stop
                    break
                if abs(pos - plateau_ref) > 0.02:
                    plateau_ref, plateau_t = pos, time.monotonic()
                elif (time.monotonic() - plateau_t > 0.8 and q - pos > 0.15
                      and abs(float(s.torq)) >= 0.25):
                    # A plateau only counts as the stop if the motor is
                    # actually pushing against something — frozen feedback
                    # otherwise fakes an instant stall (seen 2026-07-13).
                    stalled = True
                    break
            time.sleep(1.0 / 100.0)

        def _park_quiet() -> None:
            with lock:
                self._gripper_mot.send_mit(pos, 0.0, 0.0, _G_KD_MOVE, 0.0)

        if not stalled:
            _park_quiet()
            raise RuntimeError(
                "gripper boot zero-check: no stall found — encoder frame "
                "unverified (frozen feedback?); run the manual re-zero "
                "(gripper_rezero_v2.py) before using the gripper")

        # settle briefly at light press, then read the landing
        for _ in range(50):
            with lock:
                self._gripper_mot.send_mit(min(cap, pos + 0.05), 0.0, _G_KP_MOVE, _G_KD_MOVE, 0.0)
                self._gripper_ctrl.poll_feedback_once()
            s = self._gripper_mot.get_state()
            if s is not None and abs(s.pos) < _G_FB_POS_SANE:
                pos = float(s.pos)
            time.sleep(1.0 / 100.0)

        if abs(pos) <= _G_ZERO_CHECK_TOL:
            print(f"[RebotArm] gripper boot zero-check OK: closed reads {pos:+.4f}", flush=True)
            # relax to damping-only so the loop starts from a quiet closed claw
            with lock:
                self._gripper_mot.send_mit(_G_HARD_STOP_ANGLE, 0.0, 0.0, _G_KD_MOVE, 0.0)
            return

        _park_quiet()
        if abs(pos - start_pos) < 0.10:
            raise RuntimeError(
                f"gripper boot zero-check: stalled without travel at {pos:+.4f} "
                f"— jaw blocked or feedback frozen; zero NOT changed, inspect "
                f"and restart")
        if pos < 0.0:
            # Landed while still OPEN in the current frame: the jaw hit an
            # OBJECT, not the closed stop (a power-on wrap reads POSITIVE:
            # +4.63 and +1.67 observed). Never treat an object as the stop.
            raise RuntimeError(
                f"gripper boot zero-check: jaw stopped {-pos:.2f} rad short of "
                f"closed — object between the claws? Remove it and restart; "
                f"zero NOT changed")
        raise RuntimeError(
            f"gripper boot zero-check: ZERO SHIFTED — closed reads {pos:+.4f} "
            f"(single-turn wrap); run the manual re-zero "
            f"(gripper_rezero_v2.py), then restart")

    @property
    def has_gripper(self) -> bool:
        return self._gripper_mot is not None

    @property
    def gripper_is_holding(self) -> bool:
        """True when the jaw is physically gripping an object.

        Grounded in PHYSICAL evidence (encoder gap + sustained grip torque),
        not just the control-mode enum: any later gripper command rewrites the
        enum (e.g. a misheard "close_gripper" while carrying re-enters CLOSING
        and used to make this report False with the box still clamped), but it
        cannot rewrite the measured jaw angle / torque.
        """
        with self._g_lock:
            s = self._g_state
        if s in (_GS.CONTACT, _GS.HOLDING):
            return True
        if s in (_GS.OPENING, _GS.HOMING):
            return False  # actively releasing — never report "holding"
        # IDLE / CLOSING: gripping iff the jaw is noticeably ajar AND the
        # motor is exerting sustained clamp torque (a parked-empty jaw shows
        # ~0 torque regardless of its angle).
        ajar = self._g_pos < _G_HOLD_MIN_GAP
        torq = abs(self._g_torq) >= _G_HOLD_TORQ_MIN
        last_open = getattr(self, "_g_open_last_target", None)
        if last_open is not None:
            # After an INTENTIONAL open: only report holding when the jaw
            # stopped well SHORT of the commanded target (an object blocked
            # it). Parking at/near the target — including the ~0.1 rad
            # soft-limit shortfall at full open, with its residual limit
            # torque — is a successful release, not a grip (this exact case
            # false-positived the put_down release verification).
            blocked = (self._g_pos - last_open) >= _G_OPEN_BLOCKED_RAD
            return ajar and torq and blocked
        return ajar and torq

    def gripper_opening_m(self) -> float:
        """Current jaw opening in metres, estimated from the encoder."""
        return float(np.clip(self._g_pos / _G_ANGLE_OPEN, 0.0, 1.0) * _G_MAX_DIST_M)

    # ── gripper state-machine internals ──────────────────────────────────────

    def _g_safe_mit(self, pos: float, vel: float, kp: float, kd: float, tau_ff: float = 0.0) -> None:
        pos_cmd  = float(np.clip(pos, _G_OPEN_SOFT_LIMIT, 0.0))
        pos_term = kp * (pos_cmd - self._g_pos) + kd * (-self._g_vel)
        tau_safe = float(np.clip(pos_term + tau_ff, -_G_TAU_MAX, _G_TAU_MAX)) - pos_term
        lock = getattr(self._gripper_ctrl, "_bus_lock", None)
        try:
            with (lock or contextlib.nullcontext()):
                self._gripper_mot.send_mit(pos_cmd, vel, kp, kd, tau_safe)
                self._gripper_mot.request_feedback()
                self._gripper_ctrl.poll_feedback_once()
        except Exception:
            pass

    def _g_tick(self, dt: float) -> None:
        try:
            st = self._gripper_mot.get_state()
            if st is not None:
                if abs(st.pos) < _G_FB_POS_SANE and abs(st.vel) < _G_FB_VEL_SANE:
                    self._g_pos  = float(st.pos)
                    self._g_vel  = float(st.vel)
                    self._g_torq = float(st.torq)
                else:
                    self._g_bad_frames += 1
                    if self._g_bad_frames <= 5 or self._g_bad_frames % 500 == 0:
                        print(f"[RebotArm] gripper: rejected corrupt feedback frame "
                              f"pos={st.pos:+.3f} vel={st.vel:+.3f} (n={self._g_bad_frames})",
                              flush=True)
        except Exception:
            pass

        pos = self._g_pos
        vel = self._g_vel

        with self._g_lock:
            s  = self._g_state
            tf = self._g_target_force

        if s == _GS.OPENING:
            with self._g_lock:
                target = self._g_open_target
                rate = _G_OPEN_RATE if self._g_open_q_des > _G_SLOW_ZONE else _G_OPEN_RATE_SLOW
                self._g_open_q_des = max(self._g_open_q_des - rate * dt, target)
                q = self._g_open_q_des
                ramp_done = q <= target + 1e-9
            self._g_safe_mit(q, 0.0, _G_KP_MOVE, _G_KD_MOVE)
            arrived = abs(pos - target) < _G_ARRIVE_TOL
            stalled = False
            if ramp_done and not arrived:
                # The commanded ramp finished but the encoder never got within
                # tolerance: the jaw is blocked by an obstacle, or the creep
                # leg lost the race against stiction (past _G_SLOW_ZONE the
                # linkage only glides under a gentle sustained push). A short sustained stall is
                # the completion signal — without it the loop pushes at the
                # torque clamp forever (motor-overheat hazard) and every full
                # open "fails" despite the jaw being open.
                with self._g_lock:
                    if abs(vel) < _G_STALL_VEL:
                        self._g_open_stall_s += dt
                    else:
                        self._g_open_stall_s = 0.0
                    stalled = self._g_open_stall_s >= _G_OPEN_STALL_S
            if arrived or stalled:
                if stalled:
                    # Park at the achieved position (damping only) so the
                    # motor stops regulating into the hard stop.
                    self._g_safe_mit(pos, 0.0, 0.0, _G_KD_MOVE, 0.0)
                with self._g_lock:
                    self._g_state = _GS.IDLE
                    self._g_open_stall_s = 0.0

        elif s == _GS.CLOSING:
            self._g_safe_mit(0.0, 0.0, 0.0, _G_KD_CLOSE, _G_CLOSE_TORQUE)
            with self._g_lock:
                ps = self._g_pos_start
                self._g_close_elapsed += dt
                grace_over = self._g_close_elapsed >= _G_CLOSE_GRACE_S
            # Stall detection normally waits for _G_STARTUP_DIST of travel so
            # rest-state vel≈0 isn't mistaken for contact. But a jaw that is
            # ALREADY resting on the object (misheard close while holding)
            # never travels that far — after a short grace period a stall
            # counts as contact regardless, otherwise the state machine sits
            # in CLOSING pushing _G_CLOSE_TORQUE into the object forever.
            if abs(pos - ps) >= _G_STARTUP_DIST or grace_over:
                if pos > _G_HARD_STOP_ANGLE:
                    with self._g_lock:
                        self._g_state = _GS.IDLE
                elif abs(vel) < _G_STALL_VEL:
                    with self._g_lock:
                        self._g_q_contact       = pos
                        self._g_contact_elapsed = 0.0
                        self._g_state           = _GS.CONTACT

        elif s == _GS.CONTACT:
            with self._g_lock:
                qc = self._g_q_contact
            self._g_safe_mit(qc, 0.0, _G_KP_HOLD, _G_KD_HOLD)
            with self._g_lock:
                self._g_contact_elapsed += dt
                if self._g_contact_elapsed >= 0.02:
                    self._g_state = _GS.HOLDING

        elif s == _GS.HOLDING:
            with self._g_lock:
                qc = self._g_q_contact
            self._g_safe_mit(qc, 0.0, _G_KP_HOLD, _G_KD_HOLD, tf)

        elif s == _GS.HOMING:
            self._g_safe_mit(0.0, 0.0, _G_KP_MOVE, _G_KD_MOVE)
            if abs(pos) < _G_ARRIVE_TOL:
                with self._g_lock:
                    self._g_state = _GS.IDLE

    def _g_ctrl_loop(self) -> None:
        dt = 1.0 / _G_CTRL_RATE
        last = time.perf_counter()
        while not self._g_loop_stop.is_set():
            now = time.perf_counter()
            elapsed = now - last
            if elapsed >= dt:
                last += dt
                self._g_tick(elapsed)
            else:
                time.sleep(1e-4)

    def _g_start_loop(self) -> None:
        if self._g_loop_running:
            return
        self._g_loop_stop.clear()
        self._g_loop_thread = threading.Thread(target=self._g_ctrl_loop, daemon=True)
        self._g_loop_thread.start()
        self._g_loop_running = True

    def _g_stop_loop(self) -> None:
        if not self._g_loop_running:
            return
        self._g_loop_stop.set()
        thread_alive = False
        if self._g_loop_thread is not None:
            self._g_loop_thread.join(timeout=1.0)
            if self._g_loop_thread.is_alive():
                # The 500Hz control thread did NOT exit within the timeout —
                # it may still be touching the gripper motor / CAN bus. It is
                # UNSAFE to send a follow-up soft-stop frame (we'd race the
                # still-running tick on the same bus), so mark the loop as
                # unavailable, log, and leave the shared resources alone.
                thread_alive = True
                print(
                    "[RebotArm] ERROR: gripper control thread did not stop "
                    "within 1.0s; skipping soft-stop frame to avoid racing "
                    "the live thread. Gripper marked unavailable."
                )
                self._g_loop_running = False
                self._gripper_mot = None
                return
            self._g_loop_thread = None
        self._g_loop_running = False
        if not thread_alive and self._gripper_mot is not None:
            try:
                self._gripper_mot.send_mit(self._g_pos, 0.0, 0.0, _G_KD_MOVE, 0.0)
                self._gripper_mot.request_feedback()
                self._gripper_ctrl.poll_feedback_once()
            except Exception:
                pass

    def _g_wait_idle(self, timeout: float = 3.0) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._g_lock:
                if self._g_state == _GS.IDLE:
                    return True
            time.sleep(0.01)
        return False

    # ── gripper public API ──────────────────────────────────────────────────

    def open_gripper(self, distance_m: float = _G_MAX_DIST_M) -> bool:
        """Open the gripper (blocking, up to _G_OPEN_TIMEOUT_S).

        Returns True when the open COMPLETED — either the encoder reached the
        target or the jaw stalled at its physical open limit (the control loop
        treats a sustained post-ramp stall as completion and parks). Returns
        False only on the timeout backstop (state machine wedged), which
        force-parks to avoid motor overheat.
        """
        if self._gripper_mot is None:
            return False
        d = float(np.clip(distance_m, 0.0, _G_MAX_DIST_M))
        target = max((d / _G_MAX_DIST_M) * _G_ANGLE_OPEN, _G_OPEN_SOFT_LIMIT)
        with self._g_lock:
            self._g_open_target  = target
            self._g_open_q_des   = self._g_pos
            self._g_open_stall_s = 0.0
            self._g_state = _GS.OPENING
        with self._g_lock:
            self._g_open_last_target = target
        if not self._g_wait_idle(_G_OPEN_TIMEOUT_S):
            # Backstop only (the in-loop stall completion should fire first):
            # force IDLE, then park at the CURRENT position with damping only
            # so the 500Hz loop stops regulating into whatever blocked it.
            print(
                "[RebotArm] open_gripper: state machine did not settle within "
                f"3s; force-parking at current position (opening≈{self.gripper_opening_m():.3f}m)"
            )
            with self._g_lock:
                self._g_state = _GS.IDLE
                self._g_open_stall_s = 0.0
            self._g_safe_mit(self._g_pos, 0.0, 0.0, _G_KD_MOVE, 0.0)
            return False
        return True

    def close_gripper(self) -> None:
        """Pure-torque close (non-blocking)."""
        if self._gripper_mot is None:
            return
        with self._g_lock:
            self._g_open_last_target = None
            self._g_pos_start = self._g_pos
            self._g_close_elapsed = 0.0
            self._g_state = _GS.CLOSING

    def grasp(
        self,
        force: Optional[float] = None,
        timeout: float = 5.0,
        *,
        adaptive: bool = False,
        adaptive_start: float = 0.2,
        adaptive_step: float = 0.1,
        adaptive_window_s: float = 0.3,
        adaptive_creep_rad: float = 0.04,
    ) -> bool:
        """Compliant grasp: close → contact detect → force-hold (blocking).

        adaptive=False (default): hold at ``force`` — byte-equivalent to the
        historical behaviour. Used for objects whose grip force is configured
        per class (grasp_force_by_class).

        adaptive=True: ``force`` becomes the CEILING. Start holding at
        ``adaptive_start`` and watch the encoder for ``adaptive_window_s``:
        if the gap is still creeping shut by more than ``adaptive_creep_rad``
        (a soft object compressing / not yet stable), bump the hold force by
        ``adaptive_step`` and re-anchor the hold angle at the current
        position, until the gap is stable or the ceiling is reached. A rigid
        box stabilises in the first window (one ~0.3s check); soft fruit
        settles at the lowest force that actually holds it instead of being
        crushed at the ceiling. PHYSICAL LIMIT: the encoder only sees the
        GAP — an object sliding ALONG the jaws is invisible here; the grasp
        pipeline's post-carry holding re-check + retry covers that.
        """
        if self._gripper_mot is None:
            return False
        cap = float(np.clip(
            force if force is not None else self._g_target_force,
            0.05, _G_TAU_MAX,
        ))
        initial = float(np.clip(adaptive_start, 0.05, cap)) if adaptive else cap
        if force is not None or adaptive:
            with self._g_lock:
                self._g_target_force = initial
        with self._g_lock:
            self._g_open_last_target = None
            self._g_pos_start = self._g_pos
            self._g_close_elapsed = 0.0
            self._g_state = _GS.CLOSING
        t_end = time.monotonic() + timeout
        holding = False
        t0 = time.monotonic()
        t_contact = None
        while time.monotonic() < t_end:
            with self._g_lock:
                s = self._g_state
            if s == _GS.CONTACT and t_contact is None:
                t_contact = time.monotonic()
            if s == _GS.HOLDING:
                holding = True
                break
            if s == _GS.IDLE:
                return False
            time.sleep(0.01)
        # Phase timing (close-window tuning data): how long the physical
        # close travel took vs the contact-confirm window. Tune only with
        # this evidence — the CLOSING grace/stall logic has real-machine
        # history behind every constant.
        if holding:
            now = time.monotonic()
            print(f"[RebotArm] close timing: contact={((t_contact or now) - t0):.2f}s "
                  f"holding={(now - t0):.2f}s")
        if not holding:
            with self._g_lock:
                self._g_state = _GS.IDLE
            return False
        if not adaptive:
            return True

        # ── adaptive ramp (HOLDING reached at `initial`) ──────────────────
        tf = initial
        while time.monotonic() < t_end and tf < cap:
            p0 = self._g_pos
            t_w = time.monotonic() + max(0.05, float(adaptive_window_s))
            while time.monotonic() < t_w:
                time.sleep(0.02)
            with self._g_lock:
                still_holding = self._g_state == _GS.HOLDING
            if not still_holding:
                break  # released / re-commanded mid-ramp — stop adjusting
            # Closing direction is pos → 0 (less negative): creep means the
            # gap is still shrinking under the current force.
            if (self._g_pos - p0) < float(adaptive_creep_rad):
                break  # stable — this force is enough
            tf = float(min(cap, tf + float(adaptive_step)))
            with self._g_lock:
                self._g_target_force = tf
                # Re-anchor the position hold at the compressed angle so the
                # spring term stops fighting the deeper grip.
                self._g_q_contact = self._g_pos
        logger_force = tf
        try:
            print(f"[RebotArm] adaptive grasp settled at {logger_force:.2f} N·m "
                  f"(cap {cap:.2f})")
        except Exception:
            pass
        return True

    def release_gripper(self, timeout: float = 4.0) -> None:
        """Open the gripper and home it (blocking)."""
        if self._gripper_mot is None:
            return
        with self._g_lock:
            self._g_open_q_des   = self._g_pos
            self._g_open_stall_s = 0.0
            self._g_state = _GS.OPENING
        self._g_wait_idle(2.0)
        with self._g_lock:
            self._g_state = _GS.HOMING
        self._g_wait_idle(timeout)

    def get_gripper_state(self) -> tuple:
        """Return (pos_rad, vel_rad_s, torq_nm)."""
        return (self._g_pos, self._g_vel, self._g_torq)

    def set_gripper_zero(self) -> bool:
        """Set the current position as the zero point (pauses the ctrl loop)."""
        if self._gripper_mot is None:
            return False
        self._g_stop_loop()
        # If _g_stop_loop could not stop the thread it nulls _gripper_mot and
        # marks the gripper unavailable; bail out (cannot zero a live/absent
        # motor).
        if self._gripper_mot is None:
            print("[RebotArm] gripper zero aborted: gripper unavailable")
            return False
        from motorbridge import CallError
        ok = False
        try:
            self._gripper_mot.set_zero_position()
            print("[RebotArm] gripper zero set")
            ok = True
        except CallError as e:
            print(f"[RebotArm] gripper zero set failed: {e}")
            ok = False
        finally:
            # ALWAYS restart the control loop — a raise from set_zero_position
            # (e.g. an unexpected non-CallError) must not leave the gripper
            # permanently un-controlled.
            self._g_start_loop()
            with self._g_lock:
                self._g_state = _GS.IDLE
        return ok

    # ── state read ────────────────────────────────────────────────────────────

    def get_tcp_pose(self) -> np.ndarray:
        """Read current TCP pose via FK; returns a (4, 4) homogeneous transform."""
        self._arm._request_and_poll()
        q, _, _ = self._arm.get_state()
        position, rotation, _ = self._compute_fk(self._model, q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3,  3] = position
        return T

    # ── motion control ─────────────────────────────────────────────────────────

    def check_ik(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
    ) -> tuple[bool, float]:
        """Solve IK only; send no motion command."""
        self._arm._request_and_poll()
        q_curr, _, _ = self._arm.get_state()
        target = self._pos_rot_to_se3(
            np.array([x, y, z], dtype=np.float64),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
        result = self._solve_ik(
            self._model,
            self._data,
            self._ee_frame_id,
            target,
            q_curr,
            self._ik_check_params,
        )
        return bool(result.success), float(result.error)

    def move_to(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        if self._endpos_ctrl is None:
            raise RuntimeError("arm not connected; call connect() first")
        return bool(self._endpos_ctrl.move_to_traj(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=duration,
        ))

    def wait_motion(self, duration: float, extra: float = 0.6) -> None:
        """Wait for the current TCP-trajectory send thread to finish."""
        if self._endpos_ctrl is None:
            return
        thread = getattr(self._endpos_ctrl, "_send_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=duration + extra + 2.0)
        else:
            time.sleep(duration + extra)

    def safe_home(self, duration: float = 3.0) -> None:
        """Home the arm (all joints to zero)."""
        if self._endpos_ctrl is None:
            raise RuntimeError("arm not connected; call connect() first")
        self._endpos_ctrl.safe_home()

    # ── context manager ─────────────────────────────────────────────────────────

    def __enter__(self) -> "RebotArm":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()


__all__ = [
    "RebotArm",
    "find_rebot_repo_root",
    "ensure_rebot_sdk_in_syspath",
]
