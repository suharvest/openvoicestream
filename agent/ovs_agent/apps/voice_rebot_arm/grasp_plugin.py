"""GraspPlugin — registers the ``grasp_object`` LLM tool (Phase B).

Wires the torch-free vision-grasp pipeline (:func:`grasp_service.run_grasp_once`)
into the voice agent as a single ``grasp_object(object_name)`` tool with
``response_mode="parallel"``: the tool body dispatches the grasp onto a worker
thread and returns ``{"started": True, "target": ...}`` within ~200ms so the
LLM's spoken acknowledgement overlaps the multi-second physical grasp (same
fast-dispatch pattern as ``ArmPlugin.dispatch_action``).

Cancellation: a ``threading.Event`` is set on stop-intent ("停") or sleep
(``on_user_stop_intent`` / ``on_sleep``); the grasp service polls it before
every arm motion, safe-parks the gripper (open), and aborts. Ordinary user
speech does NOT cancel an in-flight motion — an early design cancelled on
``on_user_speech_start``, which meant saying ANYTHING mid-grasp silently
dropped the held object and the new command then bounced off the
already_running guard ("第一遍没反应"). Only an explicit stop word or sleep
interrupts the arm now; a new motion command while one runs is refused with
the busy motion's name so the LLM can tell the user to wait. A short
completion tone (success/failure pitch) tells the user when the arm is done
and ready for the next command.

Heavy deps (onnxruntime via the segmenter, camera SDK) are imported lazily on
first grasp so the agent boots on hosts without them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ovs_agent.plugin import Plugin

logger = logging.getLogger(__name__)


class GraspPlugin(Plugin):
    name = "grasp"

    def __init__(self, app: Any, config: Optional[dict] = None) -> None:
        super().__init__(app)
        self.cfg = dict(config or {})
        self._arm_plugin: Any = None  # resolved at start()
        self._segmenter: Any = None
        self._camera: Any = None
        self._hand_eye: Optional[np.ndarray] = None
        self._ggcnn: Any = None
        self._cancel_event = threading.Event()
        self._grasp_task: Optional[asyncio.Task] = None
        self._prime_task: Optional[asyncio.Task] = None
        self._registered = False
        # Single-reader camera discipline: the pipeline (motion) and the idle
        # dashboard observer never read the Orbbec concurrently. Motion holds
        # the lock for its whole run; the idle observer try-acquires per frame
        # (worst case it delays a grasp start by one ~100ms read).
        self._cam_lock = threading.Lock()
        self._idle_stop = threading.Event()
        self._idle_thread: Optional[threading.Thread] = None
        # Last successful grasp result (grasp_pose / pregrasp_pose /
        # open_distance_m). put_down replays it so the object is placed back
        # at the camera-visible spot it was picked from. Cleared on a
        # successful put_down.
        self._last_grasp: Optional[dict] = None
        # Deferred completion tone (failure path): the "your turn" tone is
        # played AFTER the spoken failure reason finishes (on_assistant_done),
        # so the tone is always the LAST output — a reliable signal that the
        # user may speak the next command. Holds the tone kind until played.
        self._pending_ready_tone: Optional[str] = None

    # ── config helpers ─────────────────────────────────────────────
    def _list_override(self, env_var: str, cfg_key: str) -> Any:
        """Return a list config value, preferring a JSON env override.

        Table-calibrated list values (place_bounds, scan_poses) are baked into
        the image's config.yaml. YAML lists can't use the ``${VAR:-default}``
        scalar substitution the other keys rely on, so re-calibrating after a
        table change would otherwise need an image rebuild. This lets a JSON
        env var override the baked default at container-recreate time:

            REBOT_PLACE_BOUNDS='[0.20,0.60,-0.26,0.40]'
            REBOT_SCAN_POSES='[[0.27,0,0.30,0,0.30,0],[0.25,0.10,0.30,0,0.30,0.35]]'

        If the env var is unset/empty, or its value isn't valid JSON / isn't a
        list, fall back to ``self.cfg.get(cfg_key)`` (a warning is logged for
        malformed values). The downstream length / float validation at each
        call site is unchanged — this only chooses the source.
        """
        raw = os.environ.get(env_var)
        if not raw or not raw.strip():
            return self.cfg.get(cfg_key)
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning(
                "GraspPlugin: %s is not valid JSON (%r); using config %s",
                env_var, raw, cfg_key,
            )
            return self.cfg.get(cfg_key)
        if not isinstance(parsed, list):
            logger.warning(
                "GraspPlugin: %s must be a JSON list (got %s); using config %s",
                env_var, type(parsed).__name__, cfg_key,
            )
            return self.cfg.get(cfg_key)
        return parsed

    # ── lifecycle ──────────────────────────────────────────────────
    def setup(self) -> bool:
        # Register the tool eagerly so it is advertised on the first wake,
        # exactly like ArmPlugin registers its action tools in setup().
        if self.cfg.get("enabled", True) is False:
            logger.info("GraspPlugin: disabled by config; not registering grasp_object")
            return True
        self._register_tool()
        return True

    async def start(self) -> None:
        await super().start()
        # Find the ArmPlugin so we can reach the underlying RebotArm. The
        # actuator owns the CAN bus; we drive it directly for grasp moves.
        for plugin in getattr(self.app, "plugins", []) or []:
            if plugin.__class__.__name__ == "ArmPlugin":
                self._arm_plugin = plugin
                break
        if self._arm_plugin is None:
            logger.warning("GraspPlugin: no ArmPlugin found; grasp_object will error")
        else:
            # Cross-plugin motion mutex: let ArmPlugin's static actions see
            # our in-flight grasp/search/put_down (and refuse to interleave).
            reg = getattr(self._arm_plugin, "register_motion_source", None)
            if callable(reg):
                reg(self._busy_motion_name)
        # Prime perception in the background: the YOLO onnx session load and
        # the camera stream start are otherwise paid by the FIRST grasp — and
        # a cold Orbbec returns None frames for a while (wait_for_frames(500)
        # timeouts), which used to make the demo's first grasp fail. Priming
        # moves that cost to boot. Failure is non-fatal (lazy init remains).
        if self.cfg.get("enabled", True) is not False and self._prime_enabled():
            self._prime_task = asyncio.create_task(
                asyncio.to_thread(self._prime_perception), name="grasp-prime"
            )
        # Idle dashboard observer: low-rate "detection viewfinder" frames when
        # the arm is NOT moving (demo prep: place the object, glance at the
        # dashboard, confirm it is detected before speaking). Off → decision
        # frames still flow during motions.
        if self.cfg.get("enabled", True) is not False and self._idle_frames_enabled():
            self._idle_stop.clear()
            self._idle_thread = threading.Thread(
                target=self._idle_observer, name="grasp-idle-frames", daemon=True
            )
            self._idle_thread.start()

    def _unknown_object_mode(self) -> str:
        """How to handle a spoken object that resolves to NO catalog label:
          'first'  — fall back to catalog[0] (DEFAULT; today's behaviour).
          'reject' — refuse, do not grasp anything (vocab-decoupled mode pairs
                     with this so naming an object outside yolo_classes is
                     declined rather than mis-grasped)."""
        v = str(self.cfg.get("unknown_object", "first") or "first").strip().lower()
        return "reject" if v == "reject" else "first"

    @staticmethod
    def _resolve_catalog_label(target: str, catalog: list) -> Optional[str]:
        """Resolve a spoken object to an EXACT catalog label (the detector
        filters by exact class name). Exact (case-insensitive) → substring
        either way → None. No catalog[0] fallback here — the caller decides
        what to do with an unmatched object based on the unknown_object mode."""
        if not target or not catalog:
            return None
        tl = target.strip().lower()
        return (
            next((c for c in catalog if c.lower() == tl), None)
            or next((c for c in catalog if c.lower() in tl or tl in c.lower()), None)
        )

    def _ggcnn_enabled(self) -> bool:
        v = self.cfg.get("ggcnn_refiner", False)
        return v if isinstance(v, bool) else str(v).strip().lower() in {"1", "true", "yes"}

    def _prime_enabled(self) -> bool:
        v = self.cfg.get("prime_on_start", True)
        return v if isinstance(v, bool) else str(v).strip().lower() not in {"0", "false", "no"}

    def _prime_perception(self) -> None:
        import time as _time

        t0 = _time.monotonic()
        try:
            self._ensure_perception()
            cam = self._camera
            if cam is not None:
                try:
                    cam.warm_up(5)
                except Exception:
                    logger.debug("GraspPlugin: prime warm_up failed", exc_info=True)
                # Pull frames until one is real — a cold camera returns None
                # for the first second or two.
                c = d = None
                for _ in range(10):
                    try:
                        c, d = cam.get_frame()
                    except Exception:
                        break
                    if c is not None and d is not None:
                        break
                # Force the ORT session now — TRT engine load (~6s) and the
                # memory-gate decision belong at boot, not inside the first
                # grasp (2026-07-14).
                if c is not None and self._segmenter is not None:
                    try:
                        self._segmenter.predict(c, conf=0.9)
                        logger.info("GraspPlugin: prime inference session ready")
                    except Exception:
                        logger.debug("GraspPlugin: prime predict failed",
                                     exc_info=True)
            logger.info(
                "GraspPlugin: perception primed in %.1fs (first grasp will "
                "skip model load + camera cold-start)",
                _time.monotonic() - t0,
            )
        except Exception:
            logger.warning(
                "GraspPlugin: perception priming failed after %.1fs — first "
                "grasp falls back to lazy init",
                _time.monotonic() - t0,
                exc_info=True,
            )

    def _idle_frames_enabled(self) -> bool:
        v = self.cfg.get("idle_frames", True)
        return v if isinstance(v, bool) else str(v).strip().lower() not in {"0", "false", "no"}

    def _idle_interval_s(self) -> float:
        try:
            return max(0.5, float(str(self.cfg.get("idle_frame_interval_s", 2.0)).strip()))
        except (TypeError, ValueError):
            return 2.0

    def _idle_observer(self) -> None:
        """Background thread: one frame + detection every interval while the
        arm is idle. Strictly a second-class camera citizen: skips the tick
        whenever a motion is in flight or holds the camera lock."""
        while not self._idle_stop.wait(self._idle_interval_s()):
            if self._busy_motion_name() is not None:
                continue
            cam, seg = self._camera, self._segmenter
            if cam is None or seg is None:
                continue
            if not self._cam_lock.acquire(blocking=False):
                continue
            try:
                if self._busy_motion_name() is not None:
                    continue
                color, depth = cam.get_frame()
                if color is None or depth is None:
                    continue
                try:
                    conf = float(str(self.cfg.get("conf", 0.25)).strip() or 0.25)
                except (TypeError, ValueError):
                    conf = 0.25
                results = seg.predict(color, conf=conf)
                self._publish_frame(color, depth, results, None, "idle")
            except Exception:
                logger.debug("idle observer tick failed", exc_info=True)
            finally:
                self._cam_lock.release()

    def _run_locked(self, fn, *args, **kwargs):
        """Run a pipeline function holding the camera lock (worker thread)."""
        with self._cam_lock:
            return fn(*args, **kwargs)

    def _publish_frame(self, color_bgr, depth_mm, results, best, stage) -> None:
        """frame_sink for the pipeline + idle observer → dashboard bus.
        Annotation/encode is ~10ms; failures must never reach the pipeline."""
        try:
            from .dashboard_bus import BUS
            from .perception.annotate import annotate_frame, depth_colormap

            dets = []
            for r in results or []:
                names = getattr(r, "names", {}) or {}
                for b in getattr(r, "boxes", []) or []:
                    import numpy as _np

                    dets.append({
                        "class": str(names.get(int(_np.asarray(b.cls).reshape(-1)[0]),
                                               "?")),
                        "conf": round(float(_np.asarray(b.conf).reshape(-1)[0]), 3),
                    })
            meta: dict = {"stage": str(stage), "detections": dets}
            if best is not None:
                meta["best"] = {
                    "class": str(getattr(best, "class_name", "?")),
                    "conf": round(float(getattr(best, "conf", 0.0)), 3),
                    "jaw_width_m": round(float(getattr(best, "jaw_width_m", 0.0)), 4),
                    "method": str(getattr(best, "method", "?")),
                    "center_px": [int(v) for v in getattr(best, "center_px", (0, 0))],
                }
            jpg = annotate_frame(color_bgr, results, best, label=str(stage))
            depth_jpg = depth_colormap(depth_mm) if depth_mm is not None else None
            if jpg:
                BUS.publish_frame(jpg, depth_jpg, meta)
        except Exception:
            logger.debug("dashboard frame publish failed", exc_info=True)

    def _busy_motion_name(self) -> Optional[str]:
        """Name of our in-flight motion (e.g. 'grasp-box'), or None."""
        task = self._grasp_task
        if task is not None and not task.done():
            try:
                return task.get_name()
            except Exception:  # pragma: no cover — defensive
                return "grasp"
        return None

    async def stop(self) -> None:
        await super().stop()
        # Idle dashboard observer first — it must stop touching the camera
        # before perception teardown.
        self._idle_stop.set()
        if self._idle_thread is not None:
            self._idle_thread.join(timeout=3.0)
            self._idle_thread = None
        # Wind down the boot-time perception priming first (it only loads the
        # model / pulls frames — bounded work, brief wait then abandon).
        if self._prime_task is not None and not self._prime_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._prime_task), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        self._prime_task = None
        # Signal the in-flight grasp to abort. The grasp body runs in a worker
        # thread (asyncio.to_thread) which CANNOT be force-cancelled — the
        # pipeline polls _cancel_event before each motion and safe-parks. So
        # set the event FIRST, then wait (bounded) for the worker to wind down
        # rather than just abandoning the thread mid-bus-op.
        self._cancel_event.set()
        if self._grasp_task is not None and not self._grasp_task.done():
            try:
                # Bounded wait for the worker to observe the cancel and return.
                await asyncio.wait_for(asyncio.shield(self._grasp_task), timeout=6.0)
            except asyncio.TimeoutError:
                logger.warning("GraspPlugin: grasp worker did not stop within 6s")
                self._grasp_task.cancel()
                try:
                    await self._grasp_task
                except (asyncio.CancelledError, Exception):
                    pass
            except (asyncio.CancelledError, Exception):
                pass
        self._grasp_task = None
        # Release the camera (plugin owns it; the grasp pipeline only borrows
        # the already-opened handle and must never close the shared camera).
        cam = self._camera
        if cam is not None:
            close_fn = getattr(cam, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.exception("GraspPlugin: camera close failed")
            self._camera = None

    # ── cancellation hooks (barge-in / stop / sleep) ────────────────
    async def on_user_stop_intent(self, data: str) -> None:
        self._cancel_event.set()
        logger.info("GraspPlugin: stop-intent → cancel grasp")

    async def on_user_speech_start(self) -> None:
        # Speech-start barge-in cancel is OFF by default: in a noisy demo hall
        # ambient noise / TTS echo opens the mic and would abort an in-flight
        # grasp mid-motion ("arm twitches then stops"). A grasp is a short,
        # committed motion; let it finish. The user can still abort via an
        # explicit stop-intent (on_user_stop_intent) or by going to sleep. Set
        # grasp.cancel_on_speech: true to restore the old barge-in behaviour.
        if not self.cfg.get("cancel_on_speech", False):
            return
        if self._grasp_task is not None and not self._grasp_task.done():
            self._cancel_event.set()
            logger.info("GraspPlugin: barge-in → cancel grasp")

    async def on_sleep(self, data) -> None:  # noqa: ANN001
        self._cancel_event.set()

    # ── tool registration ───────────────────────────────────────────
    def _register_tool(self) -> None:
        if self._registered:
            return
        registry = getattr(self.app, "tool_registry", None)
        if registry is None:
            logger.warning("GraspPlugin: app has no tool_registry; cannot register")
            return

        plugin = self

        # The vision model only recognises a fixed catalog of class labels (the
        # configured ``yolo_classes``). The detector filters its results by the
        # ``object_name`` we pass, so the LLM MUST fill object_name with one of
        # these EXACT labels — not the user's spoken word. The LLM maps the
        # user's intent ("抓盒子"/"把那个箱子拿起来") to the closest catalog label.
        catalog = list(self.cfg.get("yolo_classes", []))
        catalog_str = ", ".join(repr(c) for c in catalog) or "'box'"
        reject_mode = self._unknown_object_mode() == "reject"
        # The description is mode-aware: in 'reject' mode we tell the LLM to pass
        # the user's named object as-is and that ONLY the catalog labels are
        # graspable (no closest-mapping — an unknown object is declined, not
        # forced onto a known label). In 'first' mode keep the closest-mapping
        # instruction so the unknown still resolves to catalog[0].
        if reject_mode:
            grasp_desc = (
                "Pick up / grasp an object using the camera-guided arm when the "
                "user asks to grab/pick something up ('grab','pick up','grasp'). "
                "Pass the user's named object in object_name. ONLY these catalog "
                f"labels are graspable: [{catalog_str}]; any other object is "
                "declined (not mis-grasped). The detector only knows the catalog "
                "labels above."
            )
        else:
            grasp_desc = (
                "Pick up / grasp an object using the camera-guided arm when the "
                "user asks to grab/pick something up ('grab','pick up','grasp'). "
                f"object_name MUST be exactly one of these catalog labels: [{catalog_str}]. "
                "Map the user's spoken object to the closest catalog label and pass "
                "that label verbatim (e.g. user says 'grab the box' -> "
                "object_name='box'). The detector only knows the catalog labels above."
            )

        @registry.tool(
            name="grasp_object",
            description=grasp_desc,
            timeout_s=2.0,
            preamble_text="Okay, grasping.",
            response_mode="parallel",
        )
        async def grasp_object(object_name: str) -> dict:  # noqa: ANN001
            return await plugin._dispatch_grasp(object_name)

        # search_object — sweep the arm-mounted camera across observation poses
        # to FIND an object that may be outside the current view, then point at
        # it WITHOUT grasping. Separate from grasp_object so a demo can show
        # "find" and "grasp" as distinct steps.
        if reject_mode:
            search_desc = (
                "Search for / locate an object by sweeping the camera around when "
                "the user asks to FIND or LOOK FOR something but not (yet) pick it "
                "up ('find','look for','search for'). The arm scans several "
                "viewpoints, stops when it sees the object and points at it "
                "WITHOUT grasping. Pass the user's named object in object_name. "
                f"ONLY these catalog labels are searchable: [{catalog_str}]; any "
                "other object is declined."
            )
        else:
            search_desc = (
                "Search for / locate an object by sweeping the camera around when "
                "the user asks to FIND or LOOK FOR something but not (yet) pick it "
                "up ('find','look for','search for'). The arm scans several "
                "viewpoints, stops when it sees the object and points at it "
                "WITHOUT grasping. "
                f"object_name MUST be exactly one of these catalog labels: [{catalog_str}]. "
                "Map the user's spoken object to the closest catalog label "
                "(e.g. 'find the box' -> object_name='box')."
            )

        @registry.tool(
            name="search_object",
            description=search_desc,
            timeout_s=2.0,
            preamble_text="Okay, looking for it.",
            response_mode="parallel",
        )
        async def search_object(object_name: str) -> dict:  # noqa: ANN001
            return await plugin._dispatch_search(object_name)

        # put_down — place the held object back where grasp_object picked it
        # up (a spot the camera detected it at, so the NEXT grasp can find it
        # again), then return home. Lives here rather than actions.yaml so it
        # can replay the recorded grasp/pregrasp poses and release width.
        put_down_desc = (
            "Put down / place / release the object currently held by the "
            "gripper. The arm sets it back down at the spot it was picked up "
            "from (so the camera can find it again), opens the gripper, and "
            "returns home. Use this whenever the user wants the held object "
            "put down or returned, EVEN IF they name the object. 'put it back' "
            "means release the HELD object (put_down); do NOT confuse with "
            "go_home, which moves the empty arm home. Triggers: 'put it down', "
            "'put down', 'put it back', 'put the box back', 'place it', "
            "'set it down', 'drop it', 'release it'."
        )

        @registry.tool(
            name="put_down",
            description=put_down_desc,
            timeout_s=2.0,
            preamble_text="Okay, putting it back.",
            response_mode="parallel",
        )
        async def put_down() -> dict:
            return await plugin._dispatch_put_down()

        self._registered = True
        # Tool list changed → invalidate the warmed prefix cache (mirrors
        # ArmPlugin._reregister_tools).
        session = getattr(self.app, "session", None)
        if session is not None and getattr(session, "cache_warmed", False):
            session.cache_warmed = False
        logger.info("GraspPlugin: registered grasp_object tool")

    # ── dispatch (fast-return, worker-thread grasp) ─────────────────
    async def _dispatch_grasp(self, object_name: str) -> dict:
        target = (object_name or "").strip()
        if not target:
            return {"started": False, "error": "empty object_name"}
        # Safety net: the detector filters by class label, which only knows the
        # configured catalog (English). If the LLM passed the user's word
        # instead (e.g. '盒子'), remap to a catalog label so detections aren't
        # silently filtered out. Exact or substring match against the catalog is
        # honoured; an UNMATCHED object is handled per the unknown_object mode:
        #   'first'  → fall back to the first catalog class (today's behaviour).
        #   'reject' → refuse without starting any motion.
        catalog = list(self.cfg.get("yolo_classes", []))
        if catalog:
            resolved = self._resolve_catalog_label(target, catalog)
            if resolved is None:
                if self._unknown_object_mode() == "reject":
                    logger.info(
                        "GraspPlugin: object_name %r not in catalog %s → reject "
                        "(unknown_object=reject)", target, catalog
                    )
                    return {
                        "started": False,
                        "error": f"object {target!r} is not in the graspable catalog",
                        "unknown_object": target,
                        "catalog": list(catalog),
                    }
                # mode 'first': fall back to the first catalog class.
                resolved = catalog[0]
            if resolved != target:
                logger.info(
                    "GraspPlugin: object_name %r → catalog label %r", target, resolved
                )
                target = resolved
        if self._arm_plugin is None or getattr(self._arm_plugin, "arm", None) is None:
            return {"started": False, "target": target, "error": "arm not available"}

        actuator = self._arm_plugin.arm
        arm = getattr(actuator, "robot", None)
        if arm is None:
            return {"started": False, "target": target, "error": "arm not connected"}

        # SAFETY: the grasp pipeline drives the arm directly. Refuse to start
        # if torque is off — the torque gate is the single source of truth for
        # "may we move?", and a parallel grasp must honour it just like
        # execute_sequence does.
        if not getattr(actuator, "torque_enabled", False):
            return {"started": False, "target": target, "error": "torque disabled"}

        # SAFETY: refuse re-entry / interleave. A second grasp while one is in
        # flight would overwrite _cancel_event / _grasp_task and let two grasp
        # workers race the same arm/bus; a static action in flight would
        # interleave waypoints with ours.
        busy_err = self._refuse_if_motion_busy()
        if busy_err is not None:
            return {"started": False, "target": target, **busy_err}

        # Fresh cancel token for this grasp.
        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event

        try:
            self._ensure_perception()
        except Exception as exc:
            logger.exception("GraspPlugin: perception init failed")
            return {"started": False, "target": target, "error": str(exc)}

        async def _runner() -> dict:
            from .grasp_service import run_grasp_once

            params = self._grasp_params()
            # Force policy. Two modes, switched by adaptive_force_default:
            #
            #   adaptive_force_default=TRUE (the default): EVERY grasp uses the
            #   adaptive ramp. grasp_force becomes the per-class CEILING — a
            #   listed class caps at its configured value (soft fruit low, box
            #   high), an unlisted class caps at the global grasp_force. Rigid
            #   objects climb to the ceiling (firm grip); soft objects settle
            #   below it (can't be crushed). Zero hand-tuning per object.
            #
            #   adaptive_force_default=FALSE: the legacy policy — a class listed
            #   in grasp_force_by_class uses its FIXED configured force (no ramp,
            #   deterministic, zero added latency); any unlisted class ramps to
            #   the global grasp_force ceiling. Byte-identical to the validated
            #   box demo.
            ceiling, is_listed = self._force_ceiling_for_class(target)
            if self._adaptive_force_default():
                # Per-class FIXED-force OVERRIDE: a rigid class in
                # grasp_force_fixed_classes uses its configured force as a fixed
                # grip (no ramp) even though adaptive is the default — the ramp
                # under-grips rigid boxes (no compression → stops at the ~0.2
                # start → the flush-aligned jaw slips). Soft classes fall through
                # to the ramp below (configured value = crush-avoiding ceiling).
                if (
                    is_listed
                    and ceiling is not None
                    and self._force_is_fixed_for_class(target)
                ):
                    params["grasp_force"] = ceiling
                    params["adaptive_force"] = False
                else:
                    params["adaptive_force"] = True
                    if ceiling is not None:
                        params["grasp_force"] = ceiling
            elif is_listed and ceiling is not None:
                params["grasp_force"] = ceiling
                params["adaptive_force"] = False
            else:
                params["adaptive_force"] = True
            return await asyncio.to_thread(
                self._run_locked,
                run_grasp_once,
                target,
                arm=arm,
                actuator=actuator,
                segmenter=self._segmenter,
                camera=self._camera,
                T_hand_eye=self._hand_eye,
                cancel_event=cancel_event,
                ggcnn=self._ggcnn,
                frame_sink=self._publish_frame,
                **params,
            )

        self._grasp_task = asyncio.create_task(_runner(), name=f"grasp-{target}")
        # Fire-and-forget: surface completion in the log; the LLM round 2 /
        # spoken ack overlaps the physical motion (parallel mode).
        self._grasp_task.add_done_callback(self._on_grasp_done)
        return {"started": True, "target": target}

    async def _dispatch_search(self, object_name: str) -> dict:
        """Fast-return dispatch for search_object (sweep + locate, no grasp)."""
        target = (object_name or "").strip()
        catalog = list(self.cfg.get("yolo_classes", []))
        if catalog and target:
            resolved = self._resolve_catalog_label(target, catalog)
            if resolved is None:
                if self._unknown_object_mode() == "reject":
                    logger.info(
                        "GraspPlugin: search object_name %r not in catalog %s → "
                        "reject (unknown_object=reject)", target, catalog
                    )
                    return {
                        "started": False,
                        "error": f"object {target!r} is not in the searchable catalog",
                        "unknown_object": target,
                        "catalog": list(catalog),
                    }
                resolved = catalog[0]
            target = resolved
        elif not target and catalog:
            target = catalog[0]
        if not target:
            return {"started": False, "error": "empty object_name"}
        if self._arm_plugin is None or getattr(self._arm_plugin, "arm", None) is None:
            return {"started": False, "target": target, "error": "arm not available"}
        actuator = self._arm_plugin.arm
        arm = getattr(actuator, "robot", None)
        if arm is None:
            return {"started": False, "target": target, "error": "arm not connected"}
        if not getattr(actuator, "torque_enabled", False):
            return {"started": False, "target": target, "error": "torque disabled"}
        # Single arm-motion slot shared with grasp + ArmPlugin actions.
        busy_err = self._refuse_if_motion_busy()
        if busy_err is not None:
            return {"started": False, "target": target, **busy_err}
        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event
        try:
            self._ensure_perception()
        except Exception as exc:
            logger.exception("GraspPlugin: perception init failed (search)")
            return {"started": False, "target": target, "error": str(exc)}

        async def _runner() -> dict:
            from .grasp_service import run_search_once

            return await asyncio.to_thread(
                self._run_locked,
                run_search_once,
                target,
                arm=arm,
                actuator=actuator,
                segmenter=self._segmenter,
                camera=self._camera,
                T_hand_eye=self._hand_eye,
                cancel_event=cancel_event,
                frame_sink=self._publish_frame,
                **self._search_params(),
            )

        self._grasp_task = asyncio.create_task(_runner(), name=f"search-{target}")
        self._grasp_task.add_done_callback(self._on_grasp_done)
        return {"started": True, "target": target}

    def _search_params(self) -> dict:
        params: dict[str, Any] = {}
        sp = self._list_override("REBOT_SCAN_POSES", "scan_poses")
        if sp:
            # each pose is a list/tuple [x,y,z,roll,pitch,yaw]
            params["scan_poses"] = [tuple(float(v) for v in p) for p in sp]
        for k in ("conf", "move_duration", "warm_up_frames", "frames", "indicate"):
            if k in self.cfg:
                v = self.cfg[k]
                if k == "indicate":
                    params[k] = v if isinstance(v, bool) else str(v).strip().lower() in {"1", "true", "yes"}
                elif k in ("warm_up_frames", "frames"):
                    try:
                        params[k] = int(float(str(v).strip()))
                    except (TypeError, ValueError):
                        pass
                else:
                    try:
                        params[k] = float(str(v).strip())
                    except (TypeError, ValueError):
                        pass
        return params

    async def _dispatch_put_down(self) -> dict:
        """Fast-return dispatch for put_down (place back where picked up)."""
        if self._arm_plugin is None or getattr(self._arm_plugin, "arm", None) is None:
            return {"started": False, "error": "arm not available"}
        actuator = self._arm_plugin.arm
        arm = getattr(actuator, "robot", None)
        if arm is None:
            return {"started": False, "error": "arm not connected"}
        if not getattr(actuator, "torque_enabled", False):
            return {"started": False, "error": "torque disabled"}
        # Single arm-motion slot shared with grasp/search + ArmPlugin actions.
        busy_err = self._refuse_if_motion_busy()
        if busy_err is not None:
            return {"started": False, **busy_err}
        last = self._last_grasp or {}
        # Admission: a recorded grasp is sufficient — place-back is harmless
        # even if the jaw somehow let go in the meantime, and the physical
        # holding check can be momentarily False mid-transition. Refuse only
        # when BOTH there is no recorded grasp AND the gripper is physically
        # not clamping anything (encoder gap + grip torque on the real arm).
        if not last.get("grasp_pose"):
            try:
                holding_attr = getattr(arm, "gripper_is_holding", None)
                held = holding_attr() if callable(holding_attr) else holding_attr
            except Exception:
                held = None
            if held is False:
                return {"started": False, "error": "nothing held"}
        kwargs: dict[str, Any] = {
            "grasp_pose": last.get("grasp_pose"),
            "pregrasp_pose": last.get("pregrasp_pose"),
        }
        try:
            kwargs["open_distance_m"] = float(last.get("open_distance_m", 0.09))
        except (TypeError, ValueError):
            kwargs["open_distance_m"] = 0.09
        pp = self.cfg.get("place_pose")
        if pp:
            try:
                kwargs["place_pose"] = tuple(float(v) for v in pp)
            except (TypeError, ValueError):
                logger.warning("GraspPlugin: ignoring malformed place_pose")
        pb = self._list_override("REBOT_PLACE_BOUNDS", "place_bounds")
        if pb:
            try:
                bounds = [float(v) for v in pb]
                if len(bounds) == 4:
                    kwargs["place_bounds"] = bounds
                else:
                    logger.warning("GraspPlugin: place_bounds needs 4 values")
            except (TypeError, ValueError):
                logger.warning("GraspPlugin: ignoring malformed place_bounds")
        pm = self.cfg.get("place_margin_m")
        if pm is not None:
            try:
                kwargs["place_margin_m"] = float(str(pm).strip())
            except (TypeError, ValueError):
                pass
        md = self.cfg.get("move_duration")
        if md is not None:
            try:
                kwargs["move_duration"] = float(str(md).strip())
            except (TypeError, ValueError):
                pass

        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event

        async def _runner() -> dict:
            from .grasp_service import run_put_down_once

            res = await asyncio.to_thread(
                run_put_down_once,
                arm=arm,
                actuator=actuator,
                cancel_event=cancel_event,
                **kwargs,
            )
            if res.get("success"):
                # Placed back — the recorded pose is consumed; the next grasp
                # re-detects from camera.
                self._last_grasp = None
            return res

        self._grasp_task = asyncio.create_task(_runner(), name="put_down")
        self._grasp_task.add_done_callback(self._on_grasp_done)
        return {"started": True, "used_recorded_pose": bool(last.get("grasp_pose"))}

    def _refuse_if_motion_busy(self) -> Optional[dict]:
        """Shared single-motion-slot guard for grasp/search/put_down, plus the
        cross-plugin check against ArmPlugin's static actions. Returns an
        error payload naming the in-flight motion (so the LLM can tell the
        user what the arm is still doing), or None when the arm is free."""
        mine = self._busy_motion_name()
        if mine:
            # The error string is LLM-facing (server-loop feeds it back as the
            # tool result): say explicitly NOT to retry, or the model fires
            # the same tool every round until the iteration cap (real-machine
            # loop, 2026-06-12).
            return {
                "error": (
                    f"already_running: the arm is still executing '{mine}'. "
                    "Do NOT call any motion tool again this turn — tell the "
                    "user the arm is busy and to wait for the completion tone."
                ),
                "current": mine,
            }
        chk = getattr(self._arm_plugin, "busy_action", None)
        if callable(chk):
            try:
                busy = chk()
            except Exception:  # pragma: no cover — defensive
                busy = None
            # busy_action also consults our registered source, but our slot
            # was checked above and is idle — anything it reports now is a
            # static action sequence.
            if busy:
                return {
                    "error": (
                        f"arm_busy: the arm is still executing '{busy}'. "
                        "Do NOT call any motion tool again this turn — tell "
                        "the user the arm is busy and to wait."
                    ),
                    "current": str(busy),
                }
        return None

    def _on_grasp_done(self, task: asyncio.Task) -> None:
        try:
            res = task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("GraspPlugin: grasp task crashed")
            self._play_done_tone("fail")
            return
        # Remember a successful grasp so put_down can place the object back at
        # the (camera-visible, IK-validated) pickup spot. Search / put_down
        # results have no grasp_pose, so they never overwrite this.
        if res and res.get("success") and res.get("grasp_pose"):
            self._last_grasp = dict(res)
        logger.info("GraspPlugin: grasp result: %s", res)
        if res:
            try:
                from .dashboard_bus import BUS

                BUS.publish_event(
                    str(task.get_name() or "motion"),
                    {k: v for k, v in res.items()
                     if isinstance(v, (str, int, float, bool, list, dict))},
                )
            except Exception:
                logger.debug("dashboard event publish failed", exc_info=True)
        # Audible completion feedback: the parallel tool result returned long
        # ago ("好的"), so without this the user cannot tell when the arm is
        # done and safe to command again. Skip on cancel — the user stopped it
        # and already knows.
        if res and not res.get("cancelled"):
            ok = bool(res.get("success") or res.get("found"))
            kind = "ok"
            if not ok:
                err = str(res.get("error") or "")
                stage = str(res.get("stage") or "")
                if res.get("too_wide") or "too wide" in err:
                    kind = "out_of_range"       # descending sweep
                elif "no valid grasp" in err or res.get("found") is False:
                    kind = "not_found"          # double low beep
                elif stage == "plausibility" or "implausible" in err or "too low" in err:
                    kind = "out_of_range"       # descending sweep
                else:
                    kind = "fail"               # single low
            # Ordering of the audible "your turn" signal:
            #  * success / no-spoken-reason: play the tone NOW — it is already
            #    the last output.
            #  * failure WITH a spoken reason: announce WHY first, and DEFER
            #    the tone until that reply finishes (on_assistant_done), so the
            #    tone is always the final sound the user hears before speaking.
            #    Realtime V2 direct speech is history-free (no LLM round), so
            #    the reason plays the moment the motion ends.
            phrase = (
                self._failure_phrase(kind, res)
                if (kind != "ok" and self._announce_enabled())
                else ""
            )
            if phrase:
                self._pending_ready_tone = kind
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._announce(phrase), name="grasp-announce")
                    loop.create_task(
                        self._ready_tone_fallback(kind), name="grasp-ready-fallback"
                    )
                except RuntimeError:
                    self._pending_ready_tone = None
                    self._play_done_tone(kind)
            else:
                self._play_done_tone(kind)

    async def on_assistant_done(self) -> None:
        """SLV finished playing a TTS reply. If a failure announcement was just
        spoken, the completion tone was deferred to land AFTER it — play it now
        so the tone is the final "your turn" signal regardless of outcome."""
        kind = self._pending_ready_tone
        if kind is not None:
            self._pending_ready_tone = None
            self._play_done_tone(kind)

    async def _ready_tone_fallback(self, kind: str) -> None:
        """Safety net: if no tts_done arrives for the failure announcement
        (e.g. the direct-TTS path doesn't emit one), play the deferred tone
        after a bounded wait so the user is never left without the signal."""
        await asyncio.sleep(6.0)
        if self._pending_ready_tone == kind:
            self._pending_ready_tone = None
            self._play_done_tone(kind)
            logger.info("GraspPlugin: ready-tone fallback fired (no tts_done)")

    def _announce_enabled(self) -> bool:
        v = self.cfg.get("announce_failures", True)
        return v if isinstance(v, bool) else str(v).strip().lower() not in {"0", "false", "no"}

    @staticmethod
    def _failure_phrase(kind: str, res: dict) -> str:
        err = str(res.get("error") or "")
        target = str(res.get("target") or "object")
        if res.get("too_wide") or "too wide" in err:
            return "The box is too big for me to grip."
        if kind == "not_found":
            return f"I couldn't find the {target}."
        if kind == "out_of_range":
            if "jaw width" in err or "太宽" in err:
                return "It's too wide to grip."
            if "too low" in err:
                return "The target is too low."
            return "That's out of reach. Please move it closer."
        if "release failed" in err:
            return "I couldn't put it down. Please check the gripper."
        if "nothing held" in err or "lost during carry" in err:
            return "I didn't grab it. Please try again."
        if "IK failed" in err:
            return "I can't reach that position."
        return "The action didn't succeed."

    async def _announce(self, text: str) -> None:
        """Deterministic, history-free speech through Realtime V2."""
        try:
            speak = getattr(self.app, "speak", None)
            if callable(speak):
                await speak(text, conversation="none")
                logger.info("GraspPlugin: announced %r", text)
                return

            # Compatibility for lightweight app fakes and protocol-v1 agents.
            # SLVClient.speak() selects Realtime V2 or legacy text + flush.
            slv = getattr(self.app, "slv", None)
            if slv is None:
                return
            speak = getattr(slv, "speak", None)
            if callable(speak):
                await speak(text, conversation="none")
            else:
                await slv.send_text(text)
                await slv.flush_tts()
            logger.info("GraspPlugin: announced %r", text)
        except Exception:
            logger.debug("GraspPlugin: announce failed", exc_info=True)

    def _play_done_tone(self, kind) -> None:
        """Audible motion-outcome feedback, reason-differentiated so the
        presenter knows WHAT failed without reading logs:
          ok           — single high beep (1175Hz)
          fail         — single low beep (330Hz; grasp/release failure)
          not_found    — DOUBLE low beep (nothing detected — reposition/retry)
          out_of_range — descending sweep (target unreachable/ungraspable —
                         move the object)
        Config ``grasp.done_tone: {hz_ok, hz_fail, ms}``; ms: 0 disables.
        bool kinds accepted for backward compat (True→ok, False→fail)."""
        if isinstance(kind, bool):
            kind = "ok" if kind else "fail"
        try:
            audio = getattr(self.app, "audio", None)
            notify = getattr(audio, "play_notification", None)
            if not callable(notify):
                return
            cfg = dict(self.cfg.get("done_tone") or {})
            hz_ok = float(cfg.get("hz_ok", 1175))
            hz_fail = float(cfg.get("hz_fail", 330))
            ms = float(cfg.get("ms", 180))
            if ms <= 0:
                return
            import math
            import struct
            import time as _time

            sr = int(getattr(audio, "output_sr", None) or 16000)
            amp = 16000

            def _seg(f0: float, f1: float, seg_ms: float) -> bytes:
                n = int(sr * seg_ms / 1000)
                fade = min(n // 4, int(sr * 0.01))
                pcm = bytearray(n * 2)
                for i in range(n):
                    t = i / max(n - 1, 1)
                    f = f0 + (f1 - f0) * t
                    v = amp * math.sin(2 * math.pi * f * i / sr)
                    if i < fade:
                        v *= i / fade
                    elif i >= n - fade:
                        v *= (n - 1 - i) / fade
                    struct.pack_into("<h", pcm, i * 2, max(-32768, min(32767, int(v))))
                return bytes(pcm)

            gap = bytes(int(sr * 0.06) * 2)
            if kind == "ok":
                pcm = _seg(hz_ok, hz_ok, ms)
            elif kind == "not_found":
                pcm = _seg(hz_fail, hz_fail, ms * 0.6) + gap + _seg(hz_fail, hz_fail, ms * 0.6)
            elif kind == "out_of_range":
                pcm = _seg(hz_ok * 0.56, hz_fail, ms * 1.6)
            else:  # fail
                pcm = _seg(hz_fail, hz_fail, ms)
            notify(bytes(pcm))
            # Suppress the mic while the tone plays + a short tail for its
            # decay, so the tone isn't captured as a command. Tail 200→120ms
            # (2026-06-13, e2e tail-sweep on live ASR): at 200ms a command
            # spoken fast after the beep lost its onset ('抓盒子'→'花盒子');
            # ≤120ms keeps the onset even at a 120ms (super-fast) reaction,
            # while typical human reaction (200-300ms) leaves margin. The tone
            # spans the longer kinds too (not_found double beep, out_of_range
            # 1.6× sweep), so suppress for the ACTUAL pcm duration, not `ms`.
            tail_ms = float((self.cfg.get("done_tone") or {}).get(
                "mic_suppress_tail_ms", 120.0
            ))
            tone_ms = len(pcm) / 2 / max(sr, 1) * 1000.0
            sup = getattr(self.app, "_local_output_mic_suppress_until", None)
            if sup is not None:
                self.app._local_output_mic_suppress_until = max(
                    float(sup), _time.monotonic() + (tone_ms + tail_ms) / 1000.0
                )
            logger.info("GraspPlugin: done tone kind=%s", kind)
        except Exception:  # pragma: no cover — defensive
            logger.debug("GraspPlugin: done tone failed", exc_info=True)

    # ── perception / calibration init (lazy, device-only) ───────────
    def _ensure_perception(self) -> None:
        if self._segmenter is None:
            from .perception.yolo_onnx import YoloOnnxSegmenter

            model_path = self.cfg.get("yolo_model_path")
            if not model_path:
                raise RuntimeError("grasp config missing yolo_model_path")
            # Detection-recall vocab (Item D): env override REBOT_GRASP_CLASSES
            # (JSON list) lets the live vocabulary be retuned at
            # container-recreate time without an image rebuild; malformed /
            # unset falls back to the baked yolo_classes (see _list_override).
            names_src = self._list_override("REBOT_GRASP_CLASSES", "yolo_classes")
            if not isinstance(names_src, list):
                names_src = self.cfg.get("yolo_classes", [])
            names = [str(n) for n in (names_src or [])]
            providers = self.cfg.get("onnx_providers")
            kwargs: dict[str, Any] = {}
            if providers:
                # yaml expresses ORT (name, options) provider tuples as
                # [name, {options}] lists — convert; strings pass through.
                kwargs["providers"] = tuple(
                    tuple(p) if isinstance(p, (list, tuple)) else p
                    for p in providers
                )
            # Optional detector input size (Item D): default unset → segmenter's
            # 640 default (unchanged). A larger size needs a matching re-exported
            # engine; pass an int (square) or [w, h].
            iz = self.cfg.get("yolo_input_size")
            if iz is not None:
                try:
                    if isinstance(iz, (list, tuple)):
                        size = (int(iz[0]), int(iz[1]))
                    else:
                        size = (int(iz), int(iz))
                    kwargs["input_size"] = size
                except (TypeError, ValueError, IndexError):
                    logger.warning(
                        "GraspPlugin: ignoring malformed yolo_input_size %r", iz
                    )
            # Vocab-decoupled ("embin") mode: opt-in via config. When a
            # text_encoder_path is set (or vocab_mode == "embeddings"), the
            # class vocabulary is NOT baked into the detector — instead the
            # class names from yolo_classes are encoded to text PE rows and fed
            # as class_embeddings on every predict. Nothing here changes for the
            # default baked box engine (text_encoder_path unset → embin False).
            text_encoder_path = self.cfg.get("text_encoder_path")
            vocab_mode = str(self.cfg.get("vocab_mode", "") or "").lower()
            if text_encoder_path or vocab_mode == "embeddings":
                if not text_encoder_path:
                    raise RuntimeError(
                        "grasp vocab_mode=embeddings requires text_encoder_path"
                    )
                from .perception.text_pe import TextPromptEncoder

                pad_slots = int(self.cfg.get("embin_pad_slots", 16))
                encoder = TextPromptEncoder(text_encoder_path, pad_slots=pad_slots)
                class_embeddings = encoder.encode(names)
                kwargs["class_embeddings"] = class_embeddings
                kwargs["active_n"] = encoder.active_n
                logger.info(
                    "GraspPlugin: embin mode — %d class embeddings built "
                    "from %s (pad_slots=%d)",
                    encoder.active_n,
                    text_encoder_path,
                    pad_slots,
                )
            self._segmenter = YoloOnnxSegmenter(model_path, names, **kwargs)
        if self._camera is None:
            from .perception.camera import make_camera

            cam_cfg = {"camera": dict(self.cfg.get("camera", {}))}
            calib_dir = self.cfg.get("calib_dir")
            self._camera = make_camera(cam_cfg, calib_dir=calib_dir)
            self._camera.open()
        if self._hand_eye is None:
            self._hand_eye = self._load_hand_eye()
        # Optional GG-CNN refiner (Phase 3): second opinion for plane grasps
        # + primary estimator for curved objects. Ships dark (config
        # ggcnn_refiner default false) until real-fruit validation.
        if self._ggcnn is None and self._ggcnn_enabled():
            try:
                from .perception.ggcnn_refiner import GgcnnRefiner
                from pathlib import Path as _P

                default_path = str(
                    _P(__file__).parent / "tools" / "artifacts" / "ggcnn2-300.onnx"
                )
                self._ggcnn = GgcnnRefiner(
                    str(self.cfg.get("ggcnn_model_path") or default_path)
                )
                logger.info("GraspPlugin: ggcnn refiner enabled (%s)",
                            self._ggcnn.model_path)
            except Exception:
                logger.warning("GraspPlugin: ggcnn refiner init failed; "
                               "continuing without it", exc_info=True)
                self._ggcnn = None

    def _load_hand_eye(self) -> Optional[np.ndarray]:
        path = self.cfg.get("hand_eye_path")
        if not path:
            logger.warning("GraspPlugin: no hand_eye_path; grasp transform will fail")
            return None
        p = Path(path)
        if not p.exists():
            logger.warning("GraspPlugin: hand_eye_path %s not found", path)
            return None
        try:
            if p.suffix == ".npz":
                with np.load(str(p)) as data:
                    for key in ("T_hand_eye", "T_result"):
                        if key in data:
                            return np.asarray(data[key], dtype=np.float64)
                    if not data.files:
                        logger.warning("GraspPlugin: hand-eye npz %s is empty", path)
                        return None
                    return np.asarray(data[data.files[0]], dtype=np.float64)
            return np.asarray(np.load(str(p)), dtype=np.float64)
        except Exception:
            logger.exception("GraspPlugin: failed to load hand-eye from %s", path)
            return None

    def _adaptive_force_default(self) -> bool:
        """When true (default), every grasp uses the adaptive ramp with the
        per-class value treated as the CEILING. False → legacy fixed-force
        policy for listed classes. Env-overridable via REBOT_ADAPTIVE_FORCE
        (rendered into adaptive_force_default at config load)."""
        v = self.cfg.get("adaptive_force_default", True)
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() not in {"0", "false", "no", "off"}

    def _force_ceiling_for_class(self, target: str) -> tuple[Optional[float], bool]:
        """Look up the per-class force value from grasp_force_by_class.

        Returns ``(value, is_listed)``:
          • value      — the configured number (a CEILING under adaptive mode,
                          a FIXED force under legacy mode), or None when the
                          class is not listed / not numeric.
          • is_listed  — True iff the class name appears in grasp_force_by_class
                          (regardless of whether the value parsed as a number).
        """
        fb = self.cfg.get("grasp_force_by_class")
        if not isinstance(fb, dict) or not target:
            return None, False
        tl = target.strip().lower()
        for k, v in fb.items():
            if str(k).strip().lower() == tl:
                try:
                    s = str(v).strip()
                    return (float(s) if s else None), True
                except (TypeError, ValueError):
                    logger.warning(
                        "GraspPlugin: ignoring non-numeric grasp_force_by_class[%r]=%r",
                        k, v,
                    )
                    return None, True
        return None, False

    def _force_for_class(self, target: str) -> Optional[float]:
        """Fixed per-class grasp force from grasp_force_by_class, or None
        when the class is not configured (→ adaptive ramp). Retained for
        backward compatibility; see _force_ceiling_for_class for the
        (value, is_listed) form used by the dispatch policy."""
        value, _ = self._force_ceiling_for_class(target)
        return value

    def _force_is_fixed_for_class(self, target: str) -> bool:
        """True iff this class uses its configured grasp_force_by_class value as
        a FIXED grip (no adaptive ramp) EVEN when adaptive_force_default=true.

        Driven by grasp_force_fixed_classes (env REBOT_FORCE_FIXED_CLASSES, JSON
        list). Rigid objects (box family) belong here: the adaptive ramp settles
        at the lowest non-creeping force, which under-grips a rigid box (it never
        compresses → reads "stable" at the ramp start → the flush-aligned jaw
        slips). Soft classes stay off the list so they keep the crush-avoiding
        ramp."""
        names = self._list_override(
            "REBOT_FORCE_FIXED_CLASSES", "grasp_force_fixed_classes"
        )
        if not isinstance(names, list) or not target:
            return False
        tl = target.strip().lower()
        return any(str(n).strip().lower() == tl for n in names)

    def _grasp_params(self) -> dict:
        # Numeric grasp params often arrive as "${VAR:-default}" → an env-
        # substituted STRING (e.g. conf "0.15"), which would break the numpy
        # `c < conf` gate in predict and float maths downstream. Coerce floats
        # here so run_grasp_once always receives real numbers. Empty string →
        # treat as unset (drop the key, fall back to the function default).
        float_keys = {
            "conf", "iou", "depth_quantile", "pregrasp_offset_m",
            "insertion_depth_m", "lift_height_m", "grasp_force",
            "open_distance_m", "move_duration",
        }
        int_keys = {"warm_up_frames", "detect_frames"}
        bool_keys = {"release_after", "reobserve", "servo_correct"}
        out: dict[str, Any] = {}
        # grasp_retries (config name) → retries (run_grasp_once param): extra
        # full detect→grasp attempts after a retriable failure.
        gr = self.cfg.get("grasp_retries")
        if gr is not None:
            try:
                s = str(gr).strip()
                if s:
                    out["retries"] = int(float(s))
            except (TypeError, ValueError):
                logger.warning("GraspPlugin: ignoring non-numeric grasp_retries=%r", gr)
        # Auto-search: pass the configured scan_poses so grasp_object sweeps to
        # find the object when it is not in the immediate view (same poses
        # search_object uses).
        sp = self._list_override("REBOT_SCAN_POSES", "scan_poses")
        if sp:
            try:
                out["scan_poses"] = [tuple(float(v) for v in p) for p in sp]
            except (TypeError, ValueError):
                logger.warning("GraspPlugin: ignoring malformed scan_poses")
        # Plausibility box [x0,x1,y0,y1,z0,z1] (base frame) — rejects depth-
        # noise grasp points before any motion; retriable so a fresh frame
        # gets another chance.
        pb = self.cfg.get("plausible_box")
        if pb:
            try:
                box = [float(v) for v in pb]
                if len(box) == 6:
                    out["plausible_box"] = box
                else:
                    logger.warning("GraspPlugin: plausible_box needs 6 values")
            except (TypeError, ValueError):
                logger.warning("GraspPlugin: ignoring malformed plausible_box")
        for k in float_keys | int_keys | bool_keys:
            if k not in self.cfg:
                continue
            v = self.cfg[k]
            try:
                if k in bool_keys:
                    out[k] = v if isinstance(v, bool) else str(v).strip().lower() in {"1", "true", "yes"}
                elif k in int_keys:
                    s = str(v).strip()
                    if s:
                        out[k] = int(float(s))
                else:  # float
                    s = str(v).strip()
                    if s:
                        out[k] = float(s)
            except (TypeError, ValueError):
                logger.warning("GraspPlugin: ignoring non-numeric %s=%r", k, v)
        return out


__all__ = ["GraspPlugin"]
