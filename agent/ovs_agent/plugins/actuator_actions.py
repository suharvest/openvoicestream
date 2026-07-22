"""ArmPlugin — owns the actuator serial bus + observation HTTP server.

Lifecycle:
  setup()  (SYNC):
    - load actions.yaml via ActionsManager
    - register one ovs-agent tool per action
  start()  (ASYNC):
    - connect to the arm (serial I/O wrapped in asyncio.to_thread)
    - launch observation-cache refresh task
    - start the FastAPI observation server in a daemon thread
  stop()   (ASYNC):
    - cancel obs-cache task, disconnect arm

The plugin holds a single ``Actuator`` instance and a single
``ActionsManager``; tools dispatch into ``execute_action`` which runs
the blocking serial sequence on a worker thread.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from ovs_agent.actuators.actions import ActionsManager
from ovs_agent.actuators.base import Actuator
from ovs_agent.actuators.factory import create_actuator
from ovs_agent.plugin import Plugin
from ovs_agent.tools.action_tools import register_arm_tools

logger = logging.getLogger(__name__)


class ArmPlugin(Plugin):
    name = "arm"

    def __init__(self, app, config: dict) -> None:
        super().__init__(app)
        self.cfg = dict(config or {})
        self.arm: Actuator | None = None
        self.actions: ActionsManager | None = None
        self._obs_task: asyncio.Task | None = None
        self._obs_server_thread: threading.Thread | None = None
        # Names we have currently registered on the global tool registry.
        # Used to clean up before re-registering when actions.yaml is
        # edited via the /actions POST endpoint.
        self._registered_tool_names: list[str] = []
        # In-flight dispatch tasks keyed by action name. The fast-return
        # path (dispatch_action) populates this; wait_action_completion
        # awaits and pops. We keep them alive on the running loop so
        # the GC doesn't drop a still-running serial task.
        self._inflight_tasks: dict[str, asyncio.Task] = {}
        # External motion sources (e.g. an app-local grasp pipeline that
        # drives the same arm/bus outside execute_sequence). Each callable
        # returns the name of its in-flight motion, or None when idle.
        # dispatch_action refuses to start an action while any source
        # reports busy — the single serial bus tolerates one motion at a
        # time, and interleaving two waypoint sequences produces garbage
        # motion. Registered via register_motion_source().
        self._external_motion_sources: list = []

    # ── cross-plugin motion mutex ──────────────────────────────────
    def register_motion_source(self, check) -> None:
        """Register a callable ``() -> Optional[str]`` that reports an
        external in-flight motion (its name) using the same arm, so
        dispatch_action can refuse to interleave with it."""
        self._external_motion_sources.append(check)

    def busy_action(self) -> str | None:
        """Name of any in-flight motion (own actions + external sources),
        or None when the arm is idle. Used both by dispatch_action and by
        external pipelines (e.g. GraspPlugin) for the reverse check."""
        for name, task in self._inflight_tasks.items():
            if task is not None and not task.done():
                return name
        for check in self._external_motion_sources:
            try:
                busy = check()
            except Exception:  # pragma: no cover — defensive
                continue
            if busy:
                return str(busy)
        return None

    # ── lifecycle ──────────────────────────────────────────────────
    def setup(self) -> bool:  # SYNC (per Plugin.setup contract)
        actions_path = Path(self.cfg["actions_yaml_path"])
        if not actions_path.exists():
            logger.error("actions.yaml not found at %s; ArmPlugin disabled", actions_path)
            return False
        # Build the actuator (factory by backend name) BEFORE the
        # ActionsManager so we can derive the required-field set from it.
        # The actual serial connect happens in start() so we don't block
        # the synchronous setup pipeline.
        backend = self.cfg.get("backend", "so_arm")
        actuator_cfg = dict(self.cfg.get("actuator_config", {}) or {})
        try:
            self.arm = create_actuator(backend, actuator_cfg)
        except Exception:
            logger.exception(
                "actuator %r init failed; ArmPlugin disabled", backend
            )
            return False
        # The frame-validation joint set comes from the actuator, NOT a
        # hard-coded SO-ARM list. Prefer an explicit ``required_fields``
        # config override; otherwise fall back to the actuator's
        # observation_features() keys (empty pre-connect → ActionsManager
        # uses its built-in default). This is the migration fix for the
        # old hard-coded REQUIRED_JOINTS.
        required_fields = self.cfg.get("required_fields")
        if not required_fields:
            try:
                feats = self.arm.observation_features()
                required_fields = list(feats.keys()) if feats else None
            except Exception:
                required_fields = None
        try:
            self.actions = ActionsManager(
                actions_path, required_fields=required_fields
            )
        except Exception:
            logger.exception("ActionsManager init failed; ArmPlugin disabled")
            return False
        # Register one tool per action.
        self._reregister_tools()
        return True

    def _reregister_tools(self) -> None:
        """Sync the tool registry with the current actions.yaml.

        Called once at setup() and again from the observation server's
        on_actions_changed hook whenever a POST /actions succeeds, so a
        newly recorded action becomes voice-callable without restart.

        We touch ``ToolRegistry._tools`` directly because the upstream
        framework does not expose an unregister method yet — if upstream
        adds one, swap to it. The mutation is idempotent.
        """
        if self.actions is None:
            return
        try:
            entries = self.actions.list_with_descriptions()
        except Exception:
            logger.exception("ActionsManager.list_with_descriptions failed")
            entries = []
        registry = self.app.tool_registry
        # Drop previously-registered arm tools so a renamed/removed action
        # doesn't linger in the LLM's tool list. Prefer the public
        # ``unregister`` (ovs-agent >= the registry-unregister commit);
        # fall back to the private ``_tools`` dict for older installs.
        unregister = getattr(registry, "unregister", None)
        if callable(unregister):
            for old in self._registered_tool_names:
                unregister(old)
        else:
            store = getattr(registry, "_tools", None)
            if isinstance(store, dict):
                for old in self._registered_tool_names:
                    store.pop(old, None)
        prev_count = len(self._registered_tool_names)
        disabled_actions = set(self.cfg.get("disabled_actions", []) or [])
        count = register_arm_tools(
            registry=registry,
            arm_plugin=self,
            actions=entries,
            disabled_actions=disabled_actions,
        )
        self._registered_tool_names = [
            e.get("name")
            for e in entries
            if isinstance(e.get("name"), str) and e.get("name") not in disabled_actions
        ]
        logger.info(
            "ArmPlugin tools synced: %d registered (was %d before)",
            count, prev_count,
        )
        # The tools spec is part of the prefix the server caches against;
        # changing the tool list invalidates whatever prefix the previous
        # turn warmed. Force-clear ``cache_warmed`` so the next LLM call
        # takes the cold-but-save path (sends ``save_system_prompt_kv_cache``
        # rather than ``prefix_cache=True`` against a now-stale prefix).
        session = getattr(self.app, "session", None)
        if session is not None and getattr(session, "cache_warmed", False):
            session.cache_warmed = False
            logger.debug("cleared session.cache_warmed (tool list changed)")
        # Optional opt-in: wipe conversation history too. Useful if the
        # operator worries old turns referencing now-deleted action names
        # would bias the LLM. Off by default — multi-turn dialogue is the
        # common case and the model handles "tool no longer available"
        # gracefully by picking from the new list.
        if session is not None and self.cfg.get("clear_history_on_tool_change", False):
            reset = getattr(session, "reset", None)
            if callable(reset):
                reset()
                logger.info("cleared session history (clear_history_on_tool_change=true)")
        # NOTE: LLM warmup (prefix KV cache + CUDA-graph capture) is no
        # longer the application's responsibility. The agent framework
        # invokes ``LLMBackend.warmup()`` once at run() startup — for
        # edge-llm that does both /v1/cache/system_prompt and a real-
        # shape 1-token completion. When tools change mid-session we
        # only invalidate ``session.cache_warmed`` (already done above);
        # the next turn will repopulate the prefix cache naturally as a
        # cold-but-save call (``save_system_prompt_kv_cache=True``).

    async def start(self) -> None:
        await super().start()
        # Serial connect on a worker thread — LeRobot is fully blocking.
        try:
            await asyncio.to_thread(self.arm.connect)
            _actuator_cfg = self.cfg.get("actuator_config", {}) or {}
            logger.info(
                "actuator connected (backend=%s port=%s)",
                self.cfg.get("backend", "so_arm"),
                _actuator_cfg.get("port", _actuator_cfg.get("arm_port")),
            )
        except Exception:
            logger.exception("arm.connect failed; HTTP server will run with cache-only mode")
        # Periodic observation cache refresh.
        self._obs_task = asyncio.create_task(self._obs_loop(), name="arm-obs-cache")
        # Observation HTTP server (daemon thread; OK to leak on shutdown).
        try:
            from ovs_agent.plugins.actuator_observation_server import (
                start_observation_server,
            )
            port = int(self.cfg.get("observation_port", 8765))
            # ``on_actions_changed`` fires after every successful POST
            # /actions — re-register tools so the LLM picks up the new
            # action without an agent restart.
            self._obs_server_thread = start_observation_server(
                self.arm, self.actions, port,
                on_actions_changed=self._reregister_tools,
            )
            logger.info("observation_server started on :%d", port)
        except Exception:
            logger.exception("observation_server start failed")

    async def on_assistant_done(self) -> None:
        """Single-turn mode: clear conversation history after every turn.

        voice-arm is a command interface, not a chatbot — multi-turn
        context provides no value here and actively harms Qwen3-4B-AWQ's
        tool-calling reliability. See KNOWN_ISSUES.md ISSUE-001: once the
        history accumulates 2+ tool-call turns, the small model starts
        mimicking the "user → assistant '已X'" pattern from history and
        skips emitting tool_calls entirely, so the arm never moves even
        though the user hears a spoken confirmation.

        We clear ``history`` directly (not via ``session.reset()``)
        because ``reset()`` also wipes the prefix/graph warmup flags,
        which would force a cold-but-save LLM call every single turn
        (~200ms extra). The warmup state is independent of the message
        list — it only depends on system prompt + tool spec, which do
        not change between turns.

        Opt-out via ``metadata.actuator.clear_history_on_turn_end: false`` in
        agent.yaml for operators who want to experiment with multi-turn.
        """
        if not self.cfg.get("clear_history_on_turn_end", True):
            return
        session = getattr(self.app, "session", None)
        if session is None:
            return
        history = getattr(session, "history", None)
        if history is None:
            return
        n = len(history)
        if n == 0:
            return
        try:
            history.clear()
        except Exception:
            logger.exception("ArmPlugin: history.clear() failed")
            return
        logger.info(
            "ArmPlugin: cleared session history (single-turn mode, was %d turns)",
            n,
        )

    async def stop(self) -> None:
        await super().stop()
        if self._obs_task is not None and not self._obs_task.done():
            self._obs_task.cancel()
            try:
                await self._obs_task
            except (asyncio.CancelledError, Exception):
                pass
            self._obs_task = None
        if self.arm is not None:
            try:
                await asyncio.to_thread(self.arm.disconnect)
            except Exception:
                logger.exception("arm.disconnect failed")

    # ── helpers ────────────────────────────────────────────────────
    async def _obs_loop(self) -> None:
        while True:
            try:
                if self.arm is not None:
                    await asyncio.to_thread(self.arm.update_cache)
            except Exception:
                logger.debug("arm observation cache refresh failed", exc_info=True)
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return

    async def execute_action(self, name: str) -> dict:
        """Run one action by name and block until completion.

        Used by ``response_mode="await"`` tool callers (default). For
        ``response_mode="parallel"`` use ``dispatch_action`` instead so
        the LLM round 2 + TTS reply can overlap the arm motion.

        Returns ``{"success": bool, "action": str, "error"?: str}``.
        """
        if self.arm is None or self.actions is None:
            return {"success": False, "action": name, "error": "arm not initialised"}
        frames = self.actions.get_sequence(name)
        if frames is None:
            return {"success": False, "action": name, "error": f"unknown action: {name}"}
        try:
            ok = await asyncio.to_thread(self.arm.execute_sequence, frames)
        except Exception as e:
            logger.exception("execute_sequence(%s) raised", name)
            return {"success": False, "action": name, "error": str(e)}
        return {"success": bool(ok), "action": name}

    async def dispatch_action(self, name: str) -> dict:
        """Start an action and return as soon as it's confirmed running.

        Actuator.execute_sequence holds the serial-bus lock for the
        whole multi-frame motion (typically 2-5s) and has no first-frame
        ack signal. Minimum-viable parallelisation: launch the blocking
        sequence on a worker task, sleep briefly, then check the task
        is still alive — if it is, the first ``_send_frame`` has been
        issued (or is queued) and we can return ``started=True`` while
        the remaining frames keep running on the worker thread. The
        wait_action_completion coroutine awaits the task to learn the
        final outcome.

        Returns ``{"started": bool, "action": str, "error"?: str}``. On
        the rare path where the task finishes within the dispatch
        window (e.g. single-frame gripper open with 0.4s delay), we
        return the final result instead of leaving an empty in-flight
        slot.
        """
        if self.arm is None or self.actions is None:
            return {"started": False, "action": name, "error": "arm not initialised"}
        frames = self.actions.get_sequence(name)
        if frames is None:
            return {"started": False, "action": name, "error": f"unknown action: {name}"}

        # If a previous dispatch for the same action is still running,
        # don't stack — bus is single-threaded and stacked calls would
        # serialise via the internal _lock anyway. The serial bus only
        # tolerates one motion at a time; the LLM has no reason to
        # double-fire.
        prev = self._inflight_tasks.get(name)
        if prev is not None and not prev.done():
            return {
                "started": True,
                "action": name,
                "already_running": True,
            }
        # A DIFFERENT motion in flight (another action, or an external
        # pipeline like the vision grasp): refuse rather than interleave.
        # The bus lock is per-op on some actuators, so two concurrent
        # sequences would interleave waypoint-by-waypoint into a garbage
        # motion. Returning started=False with the busy name lets the LLM
        # tell the user what the arm is still doing.
        busy = self.busy_action()
        if busy is not None:
            return {
                "started": False,
                "action": name,
                "error": (
                    f"arm busy: '{busy}' still running. Do NOT call any "
                    "motion tool again this turn — tell the user to wait."
                ),
            }

        async def _runner() -> bool:
            return await asyncio.to_thread(self.arm.execute_sequence, frames)

        task = asyncio.create_task(_runner(), name=f"arm-action-{name}")
        self._inflight_tasks[name] = task

        # Give the worker a slice to acquire the bus lock and emit
        # frame #1. 200ms is well above the LeRobot serial write RTT
        # (~10-30ms per joint) but well under any noticeable user-
        # perceived latency. If the task is still alive after this
        # window, the first frame is in flight or the lock has been
        # acquired — either way "started" is true.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.2)
            # Task finished within the window — return the real result.
            ok = task.result()
            self._inflight_tasks.pop(name, None)
            return {
                "started": True,
                "action": name,
                "success": bool(ok),
                "completed_in_dispatch": True,
            }
        except asyncio.TimeoutError:
            # Still running → expected fast-return path.
            return {"started": True, "action": name}
        except Exception as e:
            logger.exception("dispatch_action(%s) failed during launch", name)
            self._inflight_tasks.pop(name, None)
            return {"started": False, "action": name, "error": str(e)}

    async def wait_action_completion(self, name: str) -> dict:
        """Block until the in-flight dispatch of ``name`` finishes.

        Symmetric to ``dispatch_action``: returns the same shape as
        ``execute_action`` once the serial sequence is fully done.
        Returns immediately with ``no_inflight=True`` if nothing is
        currently dispatching for that name.
        """
        task = self._inflight_tasks.get(name)
        if task is None:
            return {"success": True, "action": name, "no_inflight": True}
        try:
            ok = await task
            return {"success": bool(ok), "action": name}
        except Exception as e:
            logger.exception("wait_action_completion(%s) raised", name)
            return {"success": False, "action": name, "error": str(e)}
        finally:
            # Clear the slot only if the task we awaited is still the
            # one stored (a fast re-dispatch could have replaced it).
            if self._inflight_tasks.get(name) is task:
                self._inflight_tasks.pop(name, None)


__all__ = ["ArmPlugin"]
