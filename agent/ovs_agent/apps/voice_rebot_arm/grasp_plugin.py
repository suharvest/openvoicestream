"""GraspPlugin — registers the ``grasp_object`` LLM tool (Phase B).

Wires the torch-free vision-grasp pipeline (:func:`grasp_service.run_grasp_once`)
into the voice agent as a single ``grasp_object(object_name)`` tool with
``response_mode="parallel"``: the tool body dispatches the grasp onto a worker
thread and returns ``{"started": True, "target": ...}`` within ~200ms so the
LLM's spoken acknowledgement overlaps the multi-second physical grasp (same
fast-dispatch pattern as ``ArmPlugin.dispatch_action``).

Cancellation: a ``threading.Event`` is set on barge-in / stop-intent / sleep
(``on_user_stop_intent`` / ``on_user_speech_start`` / ``on_sleep``); the grasp
service polls it before every arm motion, safe-parks the gripper (open), and
aborts. This keeps Phase A's ArmPlugin (framework core) untouched — the grasp
tool lives entirely in this app-local plugin.

Heavy deps (onnxruntime via the segmenter, camera SDK) are imported lazily on
first grasp so the agent boots on hosts without them.
"""

from __future__ import annotations

import asyncio
import logging
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
        self._cancel_event = threading.Event()
        self._grasp_task: Optional[asyncio.Task] = None
        self._registered = False

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

    async def stop(self) -> None:
        await super().stop()
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
        # User started talking mid-grasp → treat as barge-in cancel.
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

        @registry.tool(
            name="grasp_object",
            description=(
                "Pick up / grasp an object by name using the camera-guided "
                "arm. Trigger words: '抓', '抓起', '拿起', '夹起', '抓取', "
                "'grasp', 'pick up', 'grab'. The object_name is the thing to "
                "pick up (e.g. 'banana', 'bottle', 'cup', '香蕉', '瓶子')."
            ),
            timeout_s=2.0,
            preamble_text="好的。",
            response_mode="parallel",
        )
        async def grasp_object(object_name: str) -> dict:  # noqa: ANN001
            return await plugin._dispatch_grasp(object_name)

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

        # SAFETY: refuse re-entry. A second grasp while one is in flight would
        # overwrite _cancel_event / _grasp_task and let two grasp workers race
        # the same arm/bus.
        if self._grasp_task is not None and not self._grasp_task.done():
            return {"started": False, "target": target, "error": "already_running"}

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
            return await asyncio.to_thread(
                run_grasp_once,
                target,
                arm=arm,
                actuator=actuator,
                segmenter=self._segmenter,
                camera=self._camera,
                T_hand_eye=self._hand_eye,
                cancel_event=cancel_event,
                **params,
            )

        self._grasp_task = asyncio.create_task(_runner(), name=f"grasp-{target}")
        # Fire-and-forget: surface completion in the log; the LLM round 2 /
        # spoken ack overlaps the physical motion (parallel mode).
        self._grasp_task.add_done_callback(self._on_grasp_done)
        return {"started": True, "target": target}

    def _on_grasp_done(self, task: asyncio.Task) -> None:
        try:
            res = task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("GraspPlugin: grasp task crashed")
            return
        logger.info("GraspPlugin: grasp result: %s", res)

    # ── perception / calibration init (lazy, device-only) ───────────
    def _ensure_perception(self) -> None:
        if self._segmenter is None:
            from .perception.yolo_onnx import YoloOnnxSegmenter

            model_path = self.cfg.get("yolo_model_path")
            if not model_path:
                raise RuntimeError("grasp config missing yolo_model_path")
            names = list(self.cfg.get("yolo_classes", []))
            providers = self.cfg.get("onnx_providers")
            kwargs: dict[str, Any] = {}
            if providers:
                kwargs["providers"] = tuple(providers)
            self._segmenter = YoloOnnxSegmenter(model_path, names, **kwargs)
        if self._camera is None:
            from .perception.camera import make_camera

            cam_cfg = {"camera": dict(self.cfg.get("camera", {}))}
            calib_dir = self.cfg.get("calib_dir")
            self._camera = make_camera(cam_cfg, calib_dir=calib_dir)
            self._camera.open()
        if self._hand_eye is None:
            self._hand_eye = self._load_hand_eye()

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
                data = np.load(str(p))
                for key in ("T_hand_eye", "T_result"):
                    if key in data:
                        return np.asarray(data[key], dtype=np.float64)
                return np.asarray(data[data.files[0]], dtype=np.float64)
            return np.asarray(np.load(str(p)), dtype=np.float64)
        except Exception:
            logger.exception("GraspPlugin: failed to load hand-eye from %s", path)
            return None

    def _grasp_params(self) -> dict:
        keys = (
            "conf", "iou", "depth_quantile", "pregrasp_offset_m",
            "insertion_depth_m", "lift_height_m", "grasp_force",
            "open_distance_m",
            "move_duration", "warm_up_frames", "release_after",
        )
        return {k: self.cfg[k] for k in keys if k in self.cfg}


__all__ = ["GraspPlugin"]
