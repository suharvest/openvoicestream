"""Config dataclass + YAML loader with ${VAR} env substitution."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields as _dataclass_fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("openvoicestream-agent requires PyYAML (uv add pyyaml)") from exc


_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _normalize_env_flag(raw: str) -> str:
    """Normalize an env-var flag value for truthy/falsy comparison.

    Strips surrounding whitespace, lowercases, and — critically — peels a
    *single* pair of matching literal quote characters (``"`` or ``'``).

    Root cause this guards against (2026-05-31): production injects flags via
    ``--env-file`` whose values can carry literal quotes, so
    ``OVS_AGENT_SERVER_LOOP="1"`` arrives in ``os.environ`` as the 3-char
    string ``"1"`` (with the quotes). A plain ``.strip().lower()`` leaves the
    quotes in place, so it never matched ``"1"`` and silently fell through to
    False, disabling server-loop in prod while isolated ``-e FLAG=1`` runs
    (no quotes) passed. Peeling the quote pair makes ``"1"`` / ``'1'`` /
    ``" 1 "`` / ``1`` all normalize to ``1``.

    Only a *matched* outer pair is stripped (one level), and only after
    whitespace is trimmed, so the existing truth semantics of ``1`` / ``true``
    are unchanged.
    """
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("\"", "'"):
        v = v[1:-1].strip()
    return v.lower()


def _default_slv_config() -> dict[str, Any]:
    return {
        "asr_language": "zh",
        "tts_language": "zh",
        "tts_speaker_id": None,   # None = use model default speaker
        "tts_voice": "default",   # deprecated, prefer tts_speaker_id
        "tts_speed": 1.0,
        "sample_rate": 16000,
        "vad": "silero",
        "vad_silence_ms": 400,
        "multi_utterance": True,
    }


@dataclass
class Config:
    """Top-level agent config."""

    slv_url: str = "ws://localhost:8621/v2v/stream"
    slv_config: dict[str, Any] = field(default_factory=_default_slv_config)
    llm_backend: str = "edge_llm"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "qwen2.5-3b-instruct"
    system_prompt: str = "You are a helpful, concise voice assistant."
    audio_input_device: str | int | None = None
    audio_output_device: str | int | None = None
    audio_input_sample_rate: int = 16000
    audio_output_sample_rate: int = 24000
    log_level: str = "INFO"
    metadata: dict[str, Any] = field(default_factory=dict)
    # Client-side VAD (replaces server VAD when slv_config.vad == "none").
    # backend: "silero" | "energy" | "auto" | "off"
    client_vad_backend: str = "auto"
    client_vad_threshold: float | None = None
    client_vad_speech_min_ms: int = 200
    client_vad_silence_ms: int = 600
    # SLV closes the WS when asr_eos arrives (even in multi_utterance), so
    # firing it from the client requires reconnect-per-turn. Off by default:
    # we keep the persistent WS and let Paraformer's CIF endpoint detect
    # utterance boundaries instead. Enable only if Paraformer endpoints
    # arrive too late or not at all for your audio.
    client_vad_drive_eos: bool = False

    # ── continuous-dialogue mic-pump (server-VAD path, client_vad off) ──
    # All OFF by default so existing deployments are unchanged. A solution
    # tunes these in its agent.yaml for its specific mic / acoustics.
    # energy_gate: substitute true-zero PCM for sub-threshold chunks so the
    # server VAD sees clean silence between utterances and endpoints (else
    # continuous room/echo audio never reaches speech_end → "silent mute").
    energy_gate_enabled: bool = False
    energy_gate_open_rms: float = 0.08        # >= open (raw RMS) → gate opens
    energy_gate_close_rms: float = 0.05       # < close for hangover_ms → gate shuts
    energy_gate_hangover_ms: float = 250.0    # bridge word-internal dips
    # makeup_gain: linear gain on forwarded mic audio so a quiet mic reaches
    # the server VAD/ASR's trained level range. 1.0 = no-op.
    mic_makeup_gain: float = 1.0
    # drive an explicit asr_eos on the gate's open→close edge so the server
    # finalizes each utterance immediately instead of relying on its own VAD
    # endpoint (which can wedge). Needs multi_utterance so the session stays
    # open. Only fires after >= eos_min_speech_ms of real speech.
    gate_drive_eos: bool = False
    gate_eos_min_speech_ms: float = 250.0
    # drop mic audio while the agent is SPEAKING/THINKING (its own TTS echo)
    # so it can't open a server-VAD segment that never cleanly ends.
    mic_drop_while_speaking: bool = False

    # Stop-intent recognition: when the ASR final exactly matches one of
    # these (after normalisation), abort current TTS, drop the turn, and
    # transition state→IDLE without consulting the LLM. Chinese strings
    # match the whole utterance; English strings match case-insensitive
    # whole-utterance or word-boundary prefix.
    stop_words: list[str] = field(default_factory=lambda: [
        "停", "停下", "停下来", "别说了", "闭嘴", "安静",
        "stop", "shut up", "be quiet", "silence",
    ])
    # AppMode framework: which mode the app boots into, plus per-mode
    # overrides keyed by mode name, e.g.
    #   mode_overrides: {chat: {system_prompt: "..."}, interpreter: {...}}
    default_mode: str = "chat"
    mode_overrides: dict[str, Any] = field(default_factory=dict)
    # pipeline_mode: controls HOW user audio enters the agent. Orthogonal
    # to AppMode (which controls what the agent DOES with the text).
    #   always_on     — current behaviour. Mic always streams; client VAD
    #                   drives turn boundaries.
    #   wake_word     — agent boots SLEEPING. A WakeSource plugin (HTTP,
    #                   MQTT, serial, local keyword spotter) fires
    #                   app.wake() → state→IDLE for one turn, then
    #                   auto-sleep after sleep_timeout_s of IDLE.
    #   push_to_talk  — agent boots SLEEPING. POST /api/control/ptt/start
    #                   wakes + jumps to LISTENING; POST /api/control/ptt/end
    #                   sends asr_eos + sleeps.
    pipeline_mode: str = "always_on"
    sleep_timeout_s: float = 30.0
    wake_sources: list[str] = field(default_factory=lambda: ["http"])
    # In push_to_talk mode, optionally disable the client-VAD silence
    # detector — relying entirely on the explicit ptt/end signal for EOS.
    # Default True since PTT users typically don't want VAD second-guessing.
    push_to_talk_no_vad_silence: bool = True
    # LLM 防卡死超时（秒）
    # llm_first_token_timeout_s: 发请求 → 首 token 的最长等待
    # llm_stream_idle_timeout_s: 流式过程中两 token 间最长间隔
    llm_first_token_timeout_s: float = 15.0
    llm_stream_idle_timeout_s: float = 30.0
    # ASR 防卡死超时（秒）— SLV 在 always_on pipeline 下不发空 final，
    # 没这个 watchdog 第一次 mic 噪声触发 EOS 后 FSM 永远卡 THINKING。
    asr_final_timeout_s: float = 3.0
    # Transparent retry for transient upstream LLM failures (network
    # resets, 5xx, connect timeouts) that happen *before any token has
    # been yielded*. Once the model has started speaking we never retry
    # — that would duplicate audio. Set to 0 to disable.
    llm_retry_on_transient: int = 1
    llm_retry_backoff_s: float = 0.5
    # LLM availability probe + circuit breaker (combined state machine —
    # see plugins/llm_availability.py). The probe hits a real
    # /v1/chat/completions with max_tokens=1, not /v1/models, so a server
    # that returns metadata but fails on inference still flips to DOWN.
    llm_availability_enabled: bool = True
    llm_availability_probe_interval_s: float = 30.0
    llm_availability_probe_timeout_s: float = 5.0
    llm_availability_failures_to_down: int = 3
    # MED-3: consecutive "unknown" probe results (timeout / connect error)
    # before we transition to UNKNOWN state. UNKNOWN surfaces a grey dot
    # on the dashboard (vs HEALTHY's green) so operators notice a network
    # partition or hung server instead of mistakenly trusting a stale
    # "everything is fine" indicator. Set to a large number to disable.
    llm_availability_unknowns_to_unknown_state: int = 3
    # Session history trim (A2). When set, the oldest turns are dropped
    # before the prompt is shipped to the LLM so total input tokens stay
    # below this ceiling. Trim fires at ``session_max_input_tokens * 0.75``
    # (see Session._trim_to_budget). The fixed prefix (system_prompt +
    # tools schema) is charged against the same budget, so this value
    # must be large enough that the fixed prefix is a small fraction of
    # ``max * 0.75`` — otherwise every turn trims, clears cache_warmed,
    # and the upstream KV-cache hot path is permanently defeated.
    #
    # Default 7000: tuned for an 8K (8192-token) engine context window
    # with ~1K output headroom (7000 + ~1000 generated ≈ 8K). Trim
    # budget (history-only) = 7000 * 0.75 = 5250 tokens; with a typical
    # 3-4K system+tools prefix that still leaves ~1500-2000 tokens for
    # history (~5-6 turns). Set to None to disable trimming (matches
    # the original append-only invariant).
    #
    # Override per-deployment if the engine uses a different context
    # window (engines-3072 → ~2000; 16K engines → ~14000). EdgeLLMBackend
    # warmup() will log an INFO/WARNING comparing this value to the
    # observed engine context when it can be inferred (currently best-
    # effort — the upstream server does not yet expose max_seq_len via
    # /v1/info, so we rely on operator configuration).
    session_max_input_tokens: int | None = 7000
    # Tokenizer used to estimate prompt size. Default matches the most
    # common edge-llm engine; override per-deployment if your engine
    # ships a different vocabulary.
    session_tokenizer_model: str = "Qwen/Qwen3-4B-AWQ"
    # Translator backend: "noop" (pass-through) or "ctranslate2" (HTTP client).
    # Used by TranslatorApp for sentence-level translation (wait for ASRFinal,
    # translate, stream to TTS). Default "noop" means translation is disabled.
    translator_backend: str = "noop"
    # Base URL of the translator service (when translator_backend="ctranslate2").
    translator_url: str = "http://localhost:9001"
    # NLLB-200 language codes for source and target languages.
    # Examples: "zho_Hans" (Chinese), "eng_Latn" (English), "fra_Latn" (French).
    translator_src_lang: str = "zho_Hans"
    translator_tgt_lang: str = "eng_Latn"
    # Request timeout for translator service (seconds).
    translator_timeout_s: float = 5.0
    # ── Tool calling (see docs/agent/tool-usage.md) ────────────────
    # Master switch. When False, app_mode bypasses the tool runner
    # entirely and behaves identically to the pre-tool implementation
    # (single LLM stream → TTS). When True, the runner is invoked with
    # the effective allowlist resolved per turn.
    tools_enabled: bool = False
    # Global default allowlist. Per-mode override via
    #   mode_overrides[<mode>].tools_allowlist
    # takes precedence. Tools list MUST stay stable per session+mode for
    # the edge-llm prefix_cache to hit (changing the list mid-session is
    # safe but degrades to a cache miss).
    tools_default_allowlist: list[str] = field(default_factory=list)
    # Maximum number of LLM ↔ tool round trips per user turn. After this
    # the runner rolls the partial round back and returns empty text.
    tools_max_iterations: int = 5
    # ── Server-loop client mode (#37 Phase 2-product, spec §5/§6) ──
    # When False (default), the agent runs the LLM + tool loop locally
    # (current behaviour, byte-for-byte unchanged). When True, the agent
    # switches to "server-loop" mode: on session open it advertises its
    # tool schemas (+ system prompt + llm params) to SLV via
    # CLIENT_TOOL_ADVERTISE, the SERVER runs the LLM + tool loop, and the
    # agent only executes remote SERVER_TOOL_CALL frames against its local
    # handlers (arm tools live where the arm is) and replies with
    # CLIENT_TOOL_RESULT. The agent does NOT call its own LLM in this mode.
    #
    # Env override: OVS_AGENT_SERVER_LOOP=1/true/yes/on enables it without
    # touching YAML. Resolved by ``server_loop_enabled()`` (env wins so a
    # deployment can flip the flag without editing config).
    server_loop: bool = False
    # Path the config was loaded from (set by `load_config`); used by
    # the dashboard's per-mode override editor to persist changes back
    # to disk. None when the Config was constructed in code.
    _source_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        allowed = {"always_on", "wake_word", "push_to_talk"}
        if self.pipeline_mode not in allowed:
            raise ValueError(
                f"pipeline_mode must be one of {sorted(allowed)}; got {self.pipeline_mode!r}"
            )
        if not (isinstance(self.llm_first_token_timeout_s, (int, float))
                and self.llm_first_token_timeout_s > 0):
            raise ValueError(
                f"llm_first_token_timeout_s must be a positive number; "
                f"got {self.llm_first_token_timeout_s!r}"
            )
        if not (isinstance(self.llm_stream_idle_timeout_s, (int, float))
                and self.llm_stream_idle_timeout_s > 0):
            raise ValueError(
                f"llm_stream_idle_timeout_s must be a positive number; "
                f"got {self.llm_stream_idle_timeout_s!r}"
            )
        if not (isinstance(self.llm_retry_on_transient, int)
                and self.llm_retry_on_transient >= 0):
            raise ValueError(
                f"llm_retry_on_transient must be a non-negative int; "
                f"got {self.llm_retry_on_transient!r}"
            )
        if not (isinstance(self.llm_retry_backoff_s, (int, float))
                and self.llm_retry_backoff_s >= 0):
            raise ValueError(
                f"llm_retry_backoff_s must be a non-negative number; "
                f"got {self.llm_retry_backoff_s!r}"
            )
        # Validate translator backend
        translator_allowed = {"noop", "ctranslate2"}
        if self.translator_backend not in translator_allowed:
            raise ValueError(
                f"translator_backend must be one of {sorted(translator_allowed)}; "
                f"got {self.translator_backend!r}"
            )
        if not (isinstance(self.translator_timeout_s, (int, float))
                and self.translator_timeout_s > 0):
            raise ValueError(
                f"translator_timeout_s must be a positive number; "
                f"got {self.translator_timeout_s!r}"
            )
        # Validate NLLB language codes (format: xxx_Xxxx per FLORES-200)
        if not re.match(r"^[a-z]{3}_[A-Z][a-z]{3}$", self.translator_src_lang):
            raise ValueError(
                f"translator_src_lang must match NLLB format (e.g. 'zho_Hans'); "
                f"got {self.translator_src_lang!r}"
            )
        if not re.match(r"^[a-z]{3}_[A-Z][a-z]{3}$", self.translator_tgt_lang):
            raise ValueError(
                f"translator_tgt_lang must match NLLB format (e.g. 'eng_Latn'); "
                f"got {self.translator_tgt_lang!r}"
            )

    def server_loop_enabled(self) -> bool:
        """Resolve the server-loop flag: ``OVS_AGENT_SERVER_LOOP`` env wins,
        else fall back to the ``server_loop`` config field.

        Env values ``1/true/yes/on`` (case-insensitive) enable it;
        ``0/false/no/off`` force-disable even if YAML set it True. Any
        other / unset value defers to the config field. This keeps the
        hard contract "flag off → zero behaviour change" cheap to assert
        in tests by toggling a single env var.
        """
        raw = os.environ.get("OVS_AGENT_SERVER_LOOP")
        if raw is not None:
            v = _normalize_env_flag(raw)
            if v in ("1", "true", "yes", "on"):
                return True
            if v in ("0", "false", "no", "off", ""):
                return False
        return bool(self.server_loop)

    @property
    def slv_http_base(self) -> str:
        """HTTP base derived from slv_url (ws://host:port/path → http://host:port).

        Used by the dashboard plugin to proxy TTS speaker/clone calls to the
        SLV service. wss:// → https://, ws:// → http://. If slv_url cannot be
        parsed, falls back to http://localhost:8621.
        """
        from urllib.parse import urlparse
        try:
            u = urlparse(self.slv_url)
            scheme = "https" if u.scheme in ("wss", "https") else "http"
            netloc = u.netloc or "localhost:8621"
            return f"{scheme}://{netloc}"
        except Exception:
            return "http://localhost:8621"


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} in strings."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    """Load YAML config, apply env substitution, return a Config."""
    p = Path(path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping; got {type(raw).__name__}")

    # SLV config sub-block: merge with defaults so users don't have to
    # restate every key.
    slv_cfg = _default_slv_config()
    slv_cfg.update(raw.get("slv_config", {}) or {})
    # Force the framework invariant: persistent WS across utterances.
    slv_cfg["multi_utterance"] = True

    fields = {k: v for k, v in raw.items() if k != "slv_config"}
    # Tolerate template ↔ Config drift: a base-image ``agent.yaml.tmpl`` may
    # carry keys this Config version dropped (e.g. ``energy_gate_*`` from the
    # v6.1 base). Passing them straight to ``Config(**fields)`` raises
    # ``TypeError: unexpected keyword argument`` and crashes the whole agent at
    # boot (surfaced during 3b-ii prod-faithful verify, 2026-05-31). Drop
    # unknown keys (logged) instead of failing.
    known = {f.name for f in _dataclass_fields(Config)}
    unknown = sorted(k for k in fields if k not in known)
    if unknown:
        logger.warning(
            "config %s: ignoring %d unknown key(s) not on Config: %s",
            p, len(unknown), ", ".join(unknown),
        )
        fields = {k: v for k, v in fields.items() if k in known}
    cfg = Config(slv_config=slv_cfg, **fields)
    cfg._source_path = p
    return cfg
