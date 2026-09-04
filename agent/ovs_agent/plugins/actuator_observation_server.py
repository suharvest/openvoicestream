"""observation_server.py — Tiny FastAPI server running in a daemon thread.

Exposes:
  GET  /observation                — flat {field: number} JSON; with cache fallback
  GET  /observation/schema         — field-type map from RobotArm.observation_features()
  POST /torque/off                 — release torque (manual posing)
  POST /torque/on                  — re-enable torque
  GET  /actions                    — full action list + etag
  GET  /actions/etag               — etag only (polled by verify panel)
  POST /actions                    — save a sequence (atomic; supports If-Match)
  POST /actions/{name}/test        — run a single saved action without LLM
  POST /actions/preview            — play ad-hoc frames without saving
  POST /actions/cancel             — cancel a running test/preview (safety stop)

State ownership:
  * RobotArm owns the serial bus + observation cache
  * ActionsManager owns the YAML file + ACTION_MAP
  * This module owns: torque flag, test-busy flag, cancel event, obs cache TS

The voice pipeline calls `start_observation_server()` once at startup and
passes its ActionsManager instance so action-execution endpoints can
fetch sequences out of the same in-memory map the pipeline uses.
"""
from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from ovs_agent.actuators.actions import (
    NAME_RE,
    ActionsError,
    ActionsManager,
)

if TYPE_CHECKING:  # pragma: no cover
    from ovs_agent.actuators.base import Actuator


# Cache fallback config: how stale can a cached observation be before we
# refuse to serve it? Tunable per-deployment via env (default 2s).
OBS_CACHE_MAX_AGE_MS_DEFAULT = 2000

# Debounce after torque-on before we accept /actions/*/test. Prevents the
# arm from snapping while the operator's hand is still on it.
TORQUE_DEBOUNCE_NS = 500_000_000  # 500 ms

# Per-frame motion overhead estimate (servo settle + command latency).
# Used to surface a duration_ms hint back to the verify panel so its
# safety timeout matches reality. Tuned for SO-100/SO-ARM at default
# move_delay; if you change ARM_MOVE_DELAY in config.py, update this too.
ESTIMATED_MOVE_MS_PER_FRAME = 500


def start_observation_server(
    robot_arm: "Actuator",
    actions_manager: ActionsManager,
    port: int = 8765,
    on_actions_changed: Optional[Callable[[], None]] = None,
) -> threading.Thread:
    """Start the FastAPI server in a daemon thread (returns the thread).

    ``on_actions_changed`` (optional) fires after a successful POST
    /actions, so callers like ArmPlugin can re-register LLM tools and
    new actions become callable without an agent restart.
    """
    app = _build_app(robot_arm, actions_manager, on_actions_changed)

    def _serve() -> None:
        import uvicorn  # type: ignore[import-not-found]

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(port),
            log_level="error",
        )

    thread = threading.Thread(target=_serve, daemon=True, name="observation-server")
    thread.start()
    print(f"[obs-server] listening on 0.0.0.0:{port}")
    return thread


def _build_app(
    robot_arm: "Actuator",
    actions_manager: ActionsManager,
    on_actions_changed: Optional[Callable[[], None]] = None,
) -> FastAPI:
    """Constructed as a separate function so tests can build the app without uvicorn."""
    app = FastAPI(title="voice-arm observation", docs_url=None, redoc_url=None)

    # ── server-owned state ──────────────────────────────────────────
    state_lock = threading.Lock()
    state: Dict[str, Any] = {
        "torque_state": "on",                  # we assume torque-on at startup
        "torque_on_since_ns": time.monotonic_ns(),
        "test_running": False,
        "latest_obs": {},
        "latest_obs_ts_ns": 0,
    }
    cancel_event = threading.Event()

    cache_max_age_ms = int(os.getenv("OBS_CACHE_MAX_AGE_MS", OBS_CACHE_MAX_AGE_MS_DEFAULT))

    # Expose the state dict + cancel_event for the pipeline's main loop
    # to refresh `latest_obs` directly (avoids serial contention on bursty reads).
    app.state.obs_state = state
    app.state.obs_state_lock = state_lock
    app.state.cancel_event = cancel_event

    # ── observation endpoints ───────────────────────────────────────

    @app.get("/observation")
    def get_observation() -> Response:
        # Try a fresh read first; fall back to cache if the bus is busy
        # (e.g. torque off, sync_read raises) but cache is still warm.
        try:
            obs = robot_arm.update_cache()
            now_ns = time.monotonic_ns()
            with state_lock:
                state["latest_obs"] = obs
                state["latest_obs_ts_ns"] = now_ns
            return JSONResponse(obs)
        except Exception as exc:
            print(f"[obs-server] live read failed, trying cache: {exc}")

        with state_lock:
            obs = dict(state["latest_obs"])
            ts_ns = state["latest_obs_ts_ns"]

        if not obs or ts_ns == 0:
            raise HTTPException(status_code=503, detail={"error": "stale_observation"})

        age_ms = (time.monotonic_ns() - ts_ns) / 1_000_000
        if age_ms > cache_max_age_ms:
            raise HTTPException(status_code=503, detail={"error": "stale_observation"})

        return JSONResponse(obs, headers={"X-Observation-Source": "cache"})

    @app.get("/observation/schema")
    def get_schema() -> dict:
        return robot_arm.observation_features()

    @app.get("/gripper")
    def gripper_health() -> dict:
        """Jaw availability, separate from /observation.

        Kept off the observation payload on purpose: that dict is the joint
        schema ActionsManager validates frames against, and this is health, not
        a joint. Backends without a gripper report ready=True with no error, so
        a caller can treat "not ready" as a real fault rather than absence.
        """
        err = getattr(robot_arm, "gripper_error", None)
        ready = getattr(robot_arm, "gripper_ready", None)
        if ready is None:                    # backend has no gripper concept
            return {"ready": True, "error": None, "supported": False}
        return {"ready": bool(ready), "error": err, "supported": True}

    # ── torque endpoints ────────────────────────────────────────────

    @app.get("/torque")
    def torque_state() -> dict:
        """Current torque state so the UI's on/off toggle can show the
        right initial label without guessing. Mirrors what /torque/{on,off}
        set on the same state dict."""
        with state_lock:
            return {"torque": state.get("torque_state", "on")}

    @app.post("/torque/off")
    def torque_off() -> dict:
        try:
            robot_arm.set_torque(False)
        except Exception as exc:
            raise HTTPException(status_code=500, detail={"error": "torque_set_failed", "msg": str(exc)})
        with state_lock:
            state["torque_state"] = "off"
        # The voice pipeline's execute_action() path reads the actuator's
        # public ``torque_enabled`` property for the same safety check
        # (B3). ``set_torque(False)`` already flipped that state in lockstep
        # with the physical bus, so there's nothing to mirror here.
        return {"ok": True, "torque": "off"}

    @app.post("/torque/on")
    def torque_on() -> dict:
        try:
            robot_arm.set_torque(True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail={"error": "torque_set_failed", "msg": str(exc)})
        with state_lock:
            state["torque_state"] = "on"
            state["torque_on_since_ns"] = time.monotonic_ns()
        # ``set_torque(True)`` already updated the actuator's public
        # ``torque_enabled`` state in lockstep with the bus (B3); no
        # private mirror needed.
        return {"ok": True, "torque": "on"}

    # ── actions endpoints ───────────────────────────────────────────

    @app.get("/actions")
    def list_actions() -> dict:
        return actions_manager.get_all()

    @app.get("/actions/etag")
    def get_actions_etag() -> dict:
        return {"etag": actions_manager.get_etag()}

    @app.post("/actions")
    async def save_action(request: Request, if_match: Optional[str] = Header(default=None)) -> dict:
        # Guard against saves during a test run — back-pressure prevents
        # overwriting an action while the arm is mid-motion executing it.
        with state_lock:
            if state["test_running"]:
                raise HTTPException(status_code=409, detail={"error": "busy", "msg": "test in progress"})

        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"error": "bad_json", "msg": str(exc)})
        name = body.get("name") if isinstance(body, dict) else None
        frames = body.get("frames") if isinstance(body, dict) else None
        # description is the LLM tool docstring — what tells the function-calling
        # model when to fire this action. Optional on update (existing value is
        # preserved); strongly encouraged on insert or the new action will not
        # be triggerable by voice.
        description = body.get("description") if isinstance(body, dict) else None
        try:
            result = actions_manager.save(
                name, frames, description=description, if_match=if_match
            )
        except ActionsError as exc:
            raise HTTPException(status_code=exc.http_status, detail={"error": exc.code, "msg": str(exc)})
        # New / renamed actions must reach the LLM's tool list, otherwise
        # "嘿 Jarvis 触发新动作" does nothing (LLM doesn't know the tool
        # exists). The callback is best-effort — never let a re-register
        # failure poison the save response the UI is waiting on.
        if on_actions_changed is not None:
            try:
                on_actions_changed()
            except Exception:
                # Logging here is harmless; saved action is still callable
                # via /actions/{name}/test even if the LLM bridge is stale.
                import traceback
                print(f"[obs-server] on_actions_changed hook raised:\n{traceback.format_exc()}")
        return result

    @app.post("/actions/{name}/test")
    def test_action(name: str) -> dict:
        # 1. Defensive name re-check (path traversal guard — host might
        #    decode %2F differently; do it here regardless of routing).
        if not NAME_RE.match(name):
            raise HTTPException(status_code=400, detail={"error": "bad_name"})

        # 2. Pre-fetch the sequence OUTSIDE the lock (it has its own lock
        #    inside ActionsManager — nested-lock would be fine here but
        #    avoid it for clarity). A missing sequence short-circuits
        #    before we touch any busy state.
        frames = actions_manager.get_sequence(name)
        if frames is None:
            raise HTTPException(status_code=404, detail={"error": "not_found"})

        # 3. ATOMIC precondition check + busy-claim. Previously these were
        #    two separate locked sections with a window in between — two
        #    concurrent /test requests could both pass the busy check
        #    before either set the flag. Now we do the check-and-set
        #    in a single critical section.
        with state_lock:
            if state["torque_state"] != "on":
                raise HTTPException(status_code=409, detail={"error": "torque_off"})
            since_ns = state["torque_on_since_ns"]
            now_ns = time.monotonic_ns()
            if now_ns - since_ns < TORQUE_DEBOUNCE_NS:
                raise HTTPException(status_code=409, detail={"error": "debounce"})
            if state["test_running"]:
                raise HTTPException(status_code=409, detail={"error": "busy"})
            # Claim busy inside the same critical section.
            state["test_running"] = True

        cancel_event.clear()

        # Estimate duration so the client UI can size its safety timer.
        total_delay_s = 0.0
        for fr in frames:
            try:
                total_delay_s += float(fr.get("delay", 0.4))
            except (TypeError, ValueError):
                total_delay_s += 0.4
        duration_ms = int(round(total_delay_s * 1000)) + len(frames) * ESTIMATED_MOVE_MS_PER_FRAME

        def _runner() -> None:
            start = time.monotonic_ns()
            try:
                robot_arm.execute_sequence(frames, cancel_event=cancel_event)
            except Exception as exc:  # pragma: no cover — best-effort
                print(f"[obs-server] test_action runner error: {exc}")
            finally:
                actual_ms = (time.monotonic_ns() - start) / 1_000_000
                print(f"[obs-server] test {name!r} done in {actual_ms:.0f}ms")
                with state_lock:
                    state["test_running"] = False

        threading.Thread(target=_runner, daemon=True, name=f"test-{name}").start()
        return {
            "ok": True,
            "name": name,
            "frames": len(frames),
            "duration_ms": duration_ms,
        }

    @app.post("/actions/preview")
    async def preview_frames(request: Request) -> dict:
        """Play ad-hoc frames straight from the request body without
        touching actions.yaml. Lets the verify-panel recorder test a
        draft sequence before deciding whether to save it — saving
        recreates the container (~30 s lockout) so we want a cheap
        try-before-commit path.

        Shares the same torque / debounce / busy gates with
        /actions/{name}/test so the safety semantics stay identical.
        """
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail={"error": "bad_json", "msg": str(exc)}
            )
        frames_raw = body.get("frames") if isinstance(body, dict) else None
        try:
            frames = ActionsManager._validate_frames(frames_raw)
        except ActionsError as exc:
            raise HTTPException(
                status_code=exc.http_status,
                detail={"error": exc.code, "msg": str(exc)},
            )

        with state_lock:
            if state["torque_state"] != "on":
                raise HTTPException(status_code=409, detail={"error": "torque_off"})
            since_ns = state["torque_on_since_ns"]
            now_ns = time.monotonic_ns()
            if now_ns - since_ns < TORQUE_DEBOUNCE_NS:
                raise HTTPException(status_code=409, detail={"error": "debounce"})
            if state["test_running"]:
                raise HTTPException(status_code=409, detail={"error": "busy"})
            state["test_running"] = True

        cancel_event.clear()

        total_delay_s = 0.0
        for fr in frames:
            try:
                total_delay_s += float(fr.get("delay", 0.4))
            except (TypeError, ValueError):
                total_delay_s += 0.4
        duration_ms = (
            int(round(total_delay_s * 1000))
            + len(frames) * ESTIMATED_MOVE_MS_PER_FRAME
        )

        def _runner() -> None:
            start = time.monotonic_ns()
            try:
                robot_arm.execute_sequence(frames, cancel_event=cancel_event)
            except Exception as exc:  # pragma: no cover — best-effort
                print(f"[obs-server] preview runner error: {exc}")
            finally:
                actual_ms = (time.monotonic_ns() - start) / 1_000_000
                print(
                    f"[obs-server] preview ({len(frames)} frames) done in "
                    f"{actual_ms:.0f}ms"
                )
                with state_lock:
                    state["test_running"] = False

        threading.Thread(
            target=_runner, daemon=True, name="preview"
        ).start()
        return {
            "ok": True,
            "preview": True,
            "frames": len(frames),
            "duration_ms": duration_ms,
        }

    @app.post("/actions/cancel")
    def cancel_action() -> dict:
        cancel_event.set()
        return {"ok": True}

    return app
