"""AppMode: Strategy-pattern extension point for voice apps.

A `BaseApp` owns the pipeline + transport + state machine. An `AppMode`
plugged into it decides what to do with each user utterance: standard
chat, interpreter, monologue, transcribe, recipe assistant, ...

Modes interact with the agent exclusively through `ModeContext` — they
never hold a back-ref to `BaseApp`. This forbids feature-creep and keeps
the surface area testable.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class LLMTimeoutError(RuntimeError):
    """LLM 调用超时（首 token 或流式 idle）。"""

    def __init__(self, kind: str, timeout_s: float, partial_text: str = ""):
        super().__init__(f"LLM {kind} timeout after {timeout_s:.1f}s")
        self.kind = kind  # "first_token" | "stream_idle"
        self.timeout_s = timeout_s
        self.partial_text = partial_text  # 已收到的 token 拼接


@dataclass
class ModeContext:
    """Dependency-injection bundle handed to every Mode method.

    Constructed fresh per call by `MultiModeApp._make_mode_ctx()`, so
    Modes always observe the current state of pipeline objects.
    """

    config: Any
    slv: Any
    llm: Any
    session: Any
    audio: Any
    events: Any
    broadcast: Callable[..., Awaitable[None]]
    translator: Any = None  # TranslatorBackend; optional (defaults to None for backward compat)
    mode_manager: "ModeManager | None" = None  # Optional ModeManager handle
    # ASR-reported language for the utterance currently being handled
    # (human-readable name like "Chinese" / "English"). None when the
    # backend doesn't perform language ID, or when no utterance is being
    # processed (e.g. enter/exit hooks). InterpreterMode uses this to
    # auto-pick the translator src language.
    detected_language: "str | None" = None

    async def speak(self, text: str) -> None:
        """Stream a pre-composed ``text`` to TTS without invoking the LLM.

        Used by modes that produce assistant text by some non-LLM path
        (e.g. InterpreterMode → NLLB translation). The text is pushed
        verbatim to SLV's text channel and flushed; no history append,
        no system prompt, no token streaming.
        """
        if not text or not text.strip():
            logger.info("ModeContext.speak: empty text — skipping")
            return
        try:
            await self.slv.send_text(text)
            await self.slv.flush_tts()
        except Exception:  # pragma: no cover - best effort
            logger.exception("ModeContext.speak: send/flush failed")
            raise

    async def run_default_dialogue_turn(
        self,
        text: str,
        system_prompt_override: str | None = None,
    ) -> None:
        """Standard chat turn: append user → stream LLM tokens to SLV →
        flush → append assistant to history.

        Reused by ChatMode, InterpreterMode, MonologueMode self-prompts,
        RecipeHelper, etc.

        System-prompt resolution order:
          1. `system_prompt_override` arg
          2. `config.mode_overrides[current_mode.name].system_prompt`
          3. `current_mode.system_prompt`
          4. `config.system_prompt`
        """
        if not text or not text.strip():
            logger.warning("run_default_dialogue_turn: empty text — skipping")
            return

        # ── Server-loop mode (#37 Phase 2-product, flag-gated) ──
        # When the agent runs in server-loop mode the LLM + tool loop lives
        # on SLV: we already advertised our tool schemas + system prompt at
        # session open, the server runs the LLM, dispatches any tool calls
        # back to us via SERVER_TOOL_CALL, and emits TTS directly. The
        # client must NOT run its own LLM here, must NOT append to history
        # (the server owns conversation state in this mode), and must NOT
        # send CLIENT_TEXT. The ASR final has already reached SLV over the
        # same WS, so the server turn is already in flight — we simply
        # return and let the TTS events drive the FSM forward.
        #
        # Flag off (default) → this branch never runs; everything below is
        # byte-for-byte the legacy client-loop path.
        cfg0 = self.config
        try:
            _server_loop = bool(cfg0.server_loop_enabled())
        except AttributeError:
            _server_loop = bool(getattr(cfg0, "server_loop", False))
        if _server_loop:
            logger.info(
                "run_default_dialogue_turn: server-loop mode — skipping local "
                "LLM/tool loop (server owns it); text=%r", text,
            )
            return

        # Fail-fast on a known-DOWN LLM. The breaker lives on the app
        # (set by LLMAvailabilityPlugin.start). The ctx doesn't carry an
        # app handle, but the event bus does via its parent — instead we
        # stash the plugin on the ModeContext indirectly through
        # ctx.events._app if present. Simpler: read it off the events bus
        # via a dedicated attribute. The plugin attaches itself to
        # ``app.llm_availability``; ``ctx.events`` has no app pointer.
        # We resolve via the broadcast closure: the broadcast callable is
        # bound to ``app.broadcast`` → its ``__self__`` is the app.
        avail = None
        bc = getattr(self.broadcast, "__self__", None)
        if bc is not None:
            avail = getattr(bc, "llm_availability", None)
        if avail is not None:
            # Local import to avoid a cycle at module load.
            from .plugins.llm_availability import AvailabilityState, LLMUnavailable
            # Fail-fast on both DOWN (confirmed failures) and UNKNOWN
            # (probe can't reach the LLM — connection refused, DNS gone,
            # etc.). Either way the next user-facing call would burn the
            # full first_token_timeout for nothing. UNKNOWN may recover
            # any second; the user will retry, but they should not wait
            # the full LLM timeout to find out.
            if avail.state in (AvailabilityState.DOWN, AvailabilityState.UNKNOWN):
                raise LLMUnavailable(
                    f"LLM is {avail.state.value.upper()} "
                    f"(consecutive failures: {avail.consecutive_failures})"
                )

        system_prompt = self._resolve_system_prompt(system_prompt_override)
        temperature = self._resolve_mode_value("temperature")

        logger.info(
            "run_default_dialogue_turn: text=%r (history=%d msgs, sp_len=%d)",
            text, len(self.session.history) + 1, len(system_prompt),
        )

        self.session.add_user(text)
        chunks: list[str] = []
        # Race #5: track ALL text we forwarded to SLV (not just LLM
        # token chunks). Tool preamble + completion_text also go to
        # SLV via send_text and must be aborted on cancel/error,
        # otherwise SLV speaks the preamble of a tool the user just
        # barged in on.
        sent_any_text = False
        cancelled = False
        completed = False
        try:
            llm_kwargs: dict[str, Any] = {}
            if temperature is not None:
                llm_kwargs["temperature"] = temperature
            cfg = self.config
            first_timeout = float(getattr(cfg, "llm_first_token_timeout_s", 15.0))
            idle_timeout = float(getattr(cfg, "llm_stream_idle_timeout_s", 30.0))

            # Resolve tools_enabled + allowlist per-turn (mode override
            # > global default). Empty allowlist + tools_enabled=True is
            # equivalent to tools disabled (no tools schema sent to LLM).
            #
            # NOTE (codex review HIGH #1): distinguish "override not set"
            # from "override explicitly set to False". Truthy fallback
            # broke per-mode opt-out: a mode with tools_enabled=False
            # would still inherit the global True. Use None sentinel.
            override = self._resolve_mode_value("tools_enabled")
            if override is None:
                tools_enabled = bool(getattr(cfg, "tools_enabled", False))
            else:
                tools_enabled = bool(override)
            allowlist_raw = self._resolve_mode_value("tools_allowlist")
            if allowlist_raw is None:
                allowlist_raw = getattr(cfg, "tools_default_allowlist", []) or []
            # Semantics matrix:
            #   tools_enabled=False                      → no tools (set())
            #   tools_enabled=True  + allowlist non-empty → that subset (set(...))
            #   tools_enabled=True  + allowlist empty     → ALL registered (None)
            #
            # `None` flows through to ``ToolRegistry.list_openai_tools(None)``
            # which already exposes the full registry. The empty-list-means-all
            # branch was documented in the user guide ("expose every
            # registered tool") but the original implementation collapsed
            # it to ``set()`` (no tools). That made
            # ``tools_enabled=True + allowlist=[]`` operationally identical
            # to ``tools_enabled=False`` — a footgun for solutions like
            # voice-arm where every action a plugin registers should be
            # callable without re-listing each name in YAML.
            allowed_tools: set[str] | None
            if not tools_enabled:
                allowed_tools = set()
            elif allowlist_raw:
                allowed_tools = set(allowlist_raw)
            else:
                allowed_tools = None
            max_iters = int(getattr(cfg, "tools_max_iterations", 5))

            # Resolve registry: prefer the BaseApp-owned registry (so
            # tests can inject), else the global default.
            registry = None
            bc_app = getattr(self.broadcast, "__self__", None)
            if bc_app is not None:
                registry = getattr(bc_app, "tool_registry", None)
            if registry is None:
                # Lazy import to avoid a cycle when tools module imports
                # only at first call (cheap once cached).
                from .tools import default_registry as _r  # noqa
                registry = _r

            from .tools import ToolCallCtx, stream_with_tools

            tool_ctx = ToolCallCtx(
                session=self.session,
                mode_manager=self.mode_manager,
                event_bus=self.events,
                config=cfg,
            )

            async def _on_token(token: str) -> None:
                nonlocal sent_any_text
                chunks.append(token)
                try:
                    self.events.emit("assistant_token", token)
                except Exception:  # pragma: no cover - defensive
                    pass
                await self.broadcast("on_assistant_token", token)
                await self.slv.send_text(token)
                sent_any_text = True

            async def _on_tool_started(tc: dict) -> None:
                payload = {
                    "id": tc.get("id"),
                    "name": tc.get("function", {}).get("name"),
                    "arguments_json": tc.get("function", {}).get("arguments"),
                }
                try:
                    self.events.emit("tool_call_started", payload)
                except Exception:  # pragma: no cover - defensive
                    pass
                await self.broadcast("on_tool_call_started", payload)

            async def _on_tool_preamble(text: str) -> None:
                # Stream the per-tool preamble verbatim to SLV. The
                # server's sentence-boundary accumulator will synthesise
                # this immediately (punctuation terminates the
                # sentence) so the user hears an acknowledgement while
                # the (potentially slow) tool dispatch is still in
                # flight. Fail-open — a dropped/reconnecting SLV must
                # not abort the tool round.
                nonlocal sent_any_text
                if not text:
                    return
                try:
                    await self.slv.send_text(text)
                    sent_any_text = True  # race #5: track for cancel abort
                    logger.info(
                        "dispatched tool preamble to SLV: %r", text
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.warning(
                        "tool preamble send_text failed", exc_info=True
                    )

            async def _on_tool_completion_text(text: str) -> None:
                # Counterpart to _on_tool_preamble — synthesised at the
                # end of a "template" response_mode tool round in lieu of
                # an LLM round 2. Fail-open: TTS drop must not abort
                # the dialogue.
                nonlocal sent_any_text
                if not text:
                    return
                try:
                    await self.slv.send_text(text)
                    sent_any_text = True  # race #5: track for cancel abort
                    logger.info(
                        "dispatched tool completion_text to SLV: %r", text
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.warning(
                        "tool completion_text send_text failed", exc_info=True
                    )

            async def _on_tool_completed(
                tc: dict, result: dict, dt_ms: float
            ) -> None:
                ok = not (
                    isinstance(result, dict) and result.get("success") is False
                )
                name = tc.get("function", {}).get("name")
                cid = tc.get("id")
                completed_payload = {
                    "id": cid,
                    "name": name,
                    "ok": ok,
                    "duration_ms": dt_ms,
                }
                try:
                    self.events.emit("tool_call_completed", completed_payload)
                except Exception:  # pragma: no cover - defensive
                    pass
                await self.broadcast("on_tool_call_completed", completed_payload)
                if not ok:
                    err_payload = {
                        "id": cid,
                        "name": name,
                        "error": str(result.get("error")) if isinstance(result, dict) else "",
                    }
                    try:
                        self.events.emit("tool_call_error", err_payload)
                    except Exception:  # pragma: no cover - defensive
                        pass
                    await self.broadcast("on_tool_call_error", err_payload)

            def _on_timeout(kind: str, t_used: float, partial: str) -> BaseException:
                return LLMTimeoutError(kind, t_used, partial)

            messages_for_llm = self.session.messages(system_prompt)
            # Inject `session` kwarg under llm_kwargs so it reaches the
            # backend (legacy contract for prefix_cache control).
            llm_kwargs["session"] = self.session

            await stream_with_tools(
                self.llm,
                messages_for_llm,
                session=self.session,
                registry=registry,
                allowed_tools=allowed_tools,
                ctx=tool_ctx,
                max_iterations=max_iters,
                on_assistant_token=_on_token,
                # ``allowed_tools`` may be ``None`` (= all registered) so
                # the truthy check below would skip the callbacks. Gate on
                # ``tools_enabled`` directly — that's the real semantic.
                on_tool_started=_on_tool_started if tools_enabled else None,
                on_tool_preamble=_on_tool_preamble if tools_enabled else None,
                on_tool_completion_text=_on_tool_completion_text if tools_enabled else None,
                on_tool_completed=_on_tool_completed if tools_enabled else None,
                llm_kwargs=llm_kwargs,
                first_token_timeout_s=first_timeout,
                idle_timeout_s=idle_timeout,
                on_timeout=_on_timeout,
            )
            completed = True
        except asyncio.CancelledError:
            cancelled = True
            # CRITICAL: tell SLV to drop the per-turn text buffer.
            # Tokens already streamed via ``slv.send_text()`` sit on the
            # server until a ``tts_flush`` or ``abort``. If this cancel
            # was a barge-in (or a thinking-watchdog recovery, or a new
            # asr_final pre-empting an in-flight turn), the next turn's
            # tokens get APPENDED to ours → server TTSes the merge of
            # both turns and our state machine sees confused replies.
            # Best-effort; if SLV is unreachable we'll fail open and the
            # next turn at least starts on a fresh connect.
            # Race #5: also abort when only tool preamble / completion_text
            # was sent (no LLM tokens yet) — those texts are equally
            # buffered on SLV's TTS pipeline and must not bleed into the
            # next turn.
            if chunks or sent_any_text:
                try:
                    await self.slv.abort()
                except Exception:  # pragma: no cover - best effort
                    logger.debug("SLV abort during cancel failed", exc_info=True)
                stop = getattr(self.audio, "stop_playback", None)
                if callable(stop):
                    try:
                        await stop()
                    except Exception:  # pragma: no cover - best effort
                        logger.debug(
                            "stop_playback during cancel failed", exc_info=True
                        )
            raise
        except Exception:
            # Partial tokens already flushed to SLV's TTS buffer — abort
            # so we don't speak a half-sentence or persist it as an
            # assistant reply. The runner already rolled session.history
            # back on its own exception path for tool rounds.
            # Race #5: include tool preamble / completion_text sends.
            if chunks or sent_any_text:
                logger.info(
                    "LLM stream failed after %d partial token(s) (sent_any_text=%s); aborting partial TTS",
                    len(chunks), sent_any_text,
                )
                try:
                    await self.slv.abort()
                except Exception:  # pragma: no cover - best effort
                    logger.debug("SLV abort failed after partial LLM error", exc_info=True)
                stop = getattr(self.audio, "stop_playback", None)
                if callable(stop):
                    try:
                        await stop()
                    except Exception:  # pragma: no cover - best effort
                        logger.debug(
                            "stop_playback failed after partial LLM error",
                            exc_info=True,
                        )
            raise
        finally:
            if completed and not cancelled:
                try:
                    await self.slv.flush_tts()
                except Exception:  # pragma: no cover - best effort
                    pass
            cm = getattr(self.llm, "last_cache_metrics", None)
            if cm:
                try:
                    await self.broadcast("on_llm_cache_metrics", cm)
                except Exception:  # pragma: no cover - defensive
                    pass

    def _resolve_system_prompt(self, override: str | None) -> str:
        if override is not None:
            return override
        mode_name = None
        if self.mode_manager is not None:
            mode_name = self.mode_manager.current_name
        # Per-mode overrides from config.mode_overrides[<name>].
        # Use `in` + `is not None` (NOT truthiness) so an explicit
        # empty-string override is honoured — users may legitimately
        # want "no system message" for a mode.
        overrides = getattr(self.config, "mode_overrides", None) or {}
        if mode_name and isinstance(overrides, dict):
            mo = overrides.get(mode_name) or {}
            if isinstance(mo, dict) and "system_prompt" in mo and mo["system_prompt"] is not None:
                return str(mo["system_prompt"])
        # Mode-class-declared system prompt (None means inherit;
        # empty string would also be honoured here).
        if self.mode_manager is not None and self.mode_manager._current is not None:
            sp = self.mode_manager._current.system_prompt
            if sp is not None:
                return sp
        # Fall back to global app system_prompt.
        return getattr(self.config, "system_prompt", "") or ""

    def _resolve_mode_value(self, key: str) -> Any:
        """Resolve a non-prompt AppMode value from config override → class default."""
        mode = self.mode_manager._current if self.mode_manager is not None else None
        mode_name = mode.name if mode is not None else None
        overrides = getattr(self.config, "mode_overrides", None) or {}
        if mode_name and isinstance(overrides, dict):
            mo = overrides.get(mode_name) or {}
            if isinstance(mo, dict) and key in mo:
                return mo[key]
        if mode is not None:
            return getattr(mode, key, None)
        return None


class AppMode(ABC):
    """Base class for a voice-app mode (Strategy pattern).

    Subclasses MUST implement `on_user_utterance`. All other hooks have
    no-op defaults and are optional.

    Class-level attributes (`system_prompt`, `temperature`, ...) are
    declarative defaults. They can be overridden per-deployment via
    `config.mode_overrides[<mode_name>] = {...}`.
    """

    name: str = "unnamed"
    display_name: str = "Unnamed"
    icon: str = "•"
    description: str = ""

    # Config overrides (None = inherit from app config).
    system_prompt: str | None = None
    temperature: float | None = None
    max_history: int | None = None
    barge_in_enabled: bool | None = None

    # If False, this mode does NOT produce TTS in response to user
    # utterances. The dispatcher uses this to restore IDLE after a turn
    # so the SPEAKING→IDLE path (which never fires for silent modes)
    # isn't required to advance the FSM. Default True — most modes talk.
    produces_tts: bool = True

    async def enter(self, ctx: ModeContext) -> None:
        """Called when this mode becomes active. Override for setup."""

    async def exit(self, ctx: ModeContext) -> None:
        """Called when this mode is being switched away from."""

    @abstractmethod
    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        """REQUIRED: how to handle the user's recognized text."""

    async def on_assistant_done(self, ctx: ModeContext) -> None:
        """Optional: called after the assistant finished speaking."""

    async def on_session_idle(self, ctx: ModeContext, idle_seconds: float) -> None:
        """Optional: periodic tick during long IDLE. monologue uses this."""

    def preprocess_user_text(self, text: str) -> str | None:
        """Optional: transform/filter ASR text before on_user_utterance.

        Return None to drop the turn entirely (e.g., wake-word filtering).
        """
        return text


class ModeManager:
    """Owns the registry of AppModes and the currently-active mode.

    Constructed with a `ctx_factory` so a fresh ModeContext is created
    for each hook invocation (avoids stale references to session/slv).
    """

    def __init__(self, ctx_factory: Callable[[], ModeContext]) -> None:
        self._ctx_factory = ctx_factory
        self._modes: dict[str, AppMode] = {}
        self._current: AppMode | None = None
        # Set True after start(); used to suppress on_mode_registered
        # broadcasts during initial bulk registration (would spam plugins).
        self._started: bool = False

    def register(self, mode: AppMode) -> None:
        if not isinstance(mode, AppMode):
            raise TypeError(f"register() expected AppMode, got {type(mode).__name__}")
        if not mode.name or mode.name == "unnamed":
            raise ValueError(f"AppMode subclass {type(mode).__name__} must set .name")
        if mode.name in self._modes:
            raise ValueError(f"mode {mode.name!r} already registered")
        self._modes[mode.name] = mode
        logger.info("registered mode %r (%s)", mode.name, mode.display_name)
        # Late registration (after start) → fire-and-forget broadcast so the
        # dashboard can refresh its mode dropdown without polling.
        if self._started:
            payload = {
                "name": mode.name,
                "display_name": mode.display_name,
                "icon": mode.icon,
                "description": mode.description,
            }
            try:
                ctx = self._make_ctx()
            except Exception:
                logger.exception("on_mode_registered: ctx build failed")
                return
            import asyncio as _asyncio
            try:
                loop = _asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                async def _bcast():
                    try:
                        await ctx.broadcast("on_mode_registered", payload)
                    except Exception:
                        logger.exception("on_mode_registered broadcast failed")
                loop.create_task(_bcast())

    @property
    def current(self) -> AppMode:
        if self._current is None:
            raise RuntimeError("No mode active — call start() first")
        return self._current

    @property
    def current_name(self) -> str | None:
        return self._current.name if self._current is not None else None

    def get(self, name: str) -> AppMode | None:
        return self._modes.get(name)

    def list_all(self) -> list[dict]:
        out: list[dict] = []
        cur = self.current_name
        for m in self._modes.values():
            out.append({
                "name": m.name,
                "display_name": m.display_name,
                "icon": m.icon,
                "description": m.description,
                "current": (m.name == cur),
            })
        return out

    def _make_ctx(self) -> ModeContext:
        ctx = self._ctx_factory()
        ctx.mode_manager = self
        return ctx

    async def switch(self, name: str) -> None:
        """Exit current (if any) → set new → enter new. No-op if same."""
        if name not in self._modes:
            raise KeyError(f"unknown mode: {name!r}")
        if self._current is not None and self._current.name == name:
            return
        prev = self._current
        next_mode = self._modes[name]
        ctx = self._make_ctx()
        if prev is not None:
            try:
                await prev.exit(ctx)
            except Exception:
                logger.exception("mode %s exit() failed", prev.name)
        self._current = next_mode
        try:
            await next_mode.enter(ctx)
        except Exception:
            logger.exception("mode %s enter() failed", next_mode.name)
        # Broadcast mode_change hook.
        payload = {
            "name": next_mode.name,
            "display_name": next_mode.display_name,
            "icon": next_mode.icon,
            "prev": prev.name if prev is not None else None,
        }
        try:
            await ctx.broadcast("on_mode_change", payload)
        except Exception:
            logger.exception("on_mode_change broadcast failed")

    async def start(self, default_name: str = "chat") -> None:
        """Initial switch to the default mode."""
        if default_name not in self._modes:
            # Fall back to first registered mode if default missing.
            if not self._modes:
                raise RuntimeError("ModeManager.start: no modes registered")
            default_name = next(iter(self._modes))
            logger.warning(
                "default mode not registered, falling back to %r", default_name
            )
        await self.switch(default_name)
        self._started = True


__all__ = ["AppMode", "ModeContext", "ModeManager", "LLMTimeoutError"]
