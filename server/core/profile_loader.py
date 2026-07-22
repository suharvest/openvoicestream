"""Runtime profile loader for deploy-time backend selection.

Profiles set environment defaults before backend modules are imported.
Operator-supplied env (present at process start) is preserved across reloads;
profile-applied keys are tracked so a subsequent ``apply_profile()`` cleans
stale keys before writing the new profile's defaults.
"""

from __future__ import annotations

import json
import logging
import os
import string
import threading
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_OPERATOR_KEY_PREFIXES: tuple[str, ...] = (
    "OVS_", "LANGUAGE_MODE", "MODEL_DIR", "ASR_", "TTS_", "EDGE_LLM_",
    "QWEN3_", "KOKORO_", "MATCHA_", "RK_", "RKLLM_", "SHERPA_",
    "STREAMING_", "VOCOS_", "HF_", "CUDA_", "TRT_", "NVIDIA_",
)


def _snapshot_operator_keys() -> frozenset[str]:
    """Snapshot every env key matching an operator prefix at import time.

    These keys are considered owned by the operator (or the surrounding
    container/CI environment) and will NEVER be overwritten or cleared by
    ``apply_profile``.

    Empty values are excluded: docker-compose passes declared-but-unset
    variables (e.g. ``QWEN3_ARTIFACT_MANIFEST:`` with no value) into the
    container as empty strings, not unset, and treating those as
    operator-owned would suppress the profile defaults they were meant to
    inherit from.
    """
    return frozenset(
        k for k, v in os.environ.items()
        if k.startswith(_OPERATOR_KEY_PREFIXES) and v != ""
    )


# Snapshot taken exactly once at module import.
_OPERATOR_KEYS: frozenset[str] = _snapshot_operator_keys()

# Keys for which an operator/profile mismatch is a hard failure (not a soft
# "operator wins" silent override). These keys steer downstream backend
# selection AND model-download routing; a silent mismatch produces
# misleading runtime failures (e.g. profile picks Qwen3 ASR but
# LANGUAGE_MODE=zh_en in env shadows it, so the Qwen3 artifacts never
# download). Fail loud at startup instead.
CRITICAL_KEYS: frozenset[str] = frozenset({
    "LANGUAGE_MODE", "ASR_BACKEND", "TTS_BACKEND",
})

# Keys written by the most recent ``apply_profile`` call. Used to clear stale
# values when reloading a different profile.
_APPLIED_KEYS: set[str] = set()

# Most recently loaded profile dict (empty when none has been applied).
_CURRENT_PROFILE: dict = {}

# Reentrant lock guarding the three pieces of state above.
_PROFILE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _project_root() -> Path:
    # __file__ = <repo>/server/core/profile_loader.py → parents[2] = <repo>
    return Path(__file__).resolve().parents[2]


def _profile_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    return _project_root() / "configs" / "profiles" / candidate.name


def _select_profile_ref() -> str | None:
    """Resolve a profile reference from env (legacy ``apply_profile_from_env`` path)."""
    profile_ref = _env("OVS_PROFILE_JSON") or _env("OVS_PROFILE") or _env("OVS_PROFILE_DEFAULT")
    if not profile_ref:
        language_mode = os.environ.get("LANGUAGE_MODE", "").strip()
        if language_mode == "zh_en":
            profile_ref = "jetson-zh-en"
        elif language_mode == "multilanguage":
            profile_ref = "jetson-multilang-highperf"
    if not profile_ref:
        preset = _env("OVS_PRESET")
        if preset:
            from server.core.profile_selector import resolve_profile_name, UnsupportedPreset
            try:
                profile_ref = resolve_profile_name(preset)
            except UnsupportedPreset as exc:
                logger.error("preset %r not supported on this device: %s", preset, exc)
                raise
            logger.info("preset %r → profile %r", preset, profile_ref)
    return profile_ref


def _derive_tts_model_id(profile: dict) -> str | None:
    """Compute the OVS_TTS_MODEL_ID value implied by ``profile``.

    Returns ``None`` when the profile has no usable hint; callers decide
    whether to write it into env (and whether operator ownership overrides).
    """
    model_id = profile.get("tts_model_id")
    if model_id:
        return str(model_id)

    engines = profile.get("required_engines") or []
    tts_keys = ("kokoro", "matcha", "qwen3", "piper")
    for engine in engines:
        mid = engine.get("model_id", "")
        if any(k in str(mid).lower() for k in tts_keys):
            return str(mid)

    logger.warning(
        "Profile %r has no tts_model_id; speaker tables may mis-scope. "
        "Add tts_model_id to your profile JSON.",
        profile.get("name"),
    )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def current_profile() -> dict:
    """Return the most recently loaded profile dict (or ``{}``)."""
    with _PROFILE_LOCK:
        return _CURRENT_PROFILE


def get_max_concurrent_sessions() -> int | None:
    """Return the profile's top-level ``max_concurrent_sessions`` (if set).

    Week 1 production hardening: profiles MAY declare a session limit as
    a top-level integer (not under ``env``); env override
    ``OVS_MAX_CONCURRENT_SESSIONS`` always wins. See
    ``docs/specs/prod-hardening-week1.md`` Deliverable 2.

    Returns ``None`` when the profile does not set the field. Raises
    ``ValueError`` if the value is present but not a positive integer.
    """
    with _PROFILE_LOCK:
        raw = _CURRENT_PROFILE.get("max_concurrent_sessions")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"profile.max_concurrent_sessions must be an integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"profile.max_concurrent_sessions must be > 0, got {value}"
        )
    return value


def get_applied_keys() -> frozenset[str]:
    """Return a snapshot of env keys written by the most recent ``apply_profile``."""
    with _PROFILE_LOCK:
        return frozenset(_APPLIED_KEYS)


def apply_profile(
    profile_ref: str | None = None,
    *,
    overrides: Mapping[str, str] | None = None,
    resolve_engines: bool = False,
    kind: str | None = None,
) -> dict:
    """Load a profile and reconcile process env against it.

    Args:
        profile_ref: Profile name or absolute path. ``None`` falls back to env
            resolution (compatible with ``apply_profile_from_env``).
        overrides: Extra env entries layered on top of the profile (respecting
            operator ownership).
        resolve_engines: Reserved for PR4 — when True, future versions will
            call ``engine_resolver`` to materialize artifact paths. Currently
            a no-op.

    Returns:
        The parsed profile dict (or ``{}`` when no profile could be resolved).
    """
    global _CURRENT_PROFILE

    with _PROFILE_LOCK:
        ref = profile_ref if profile_ref else _select_profile_ref()
        if not ref:
            return {}

        path = _profile_path(ref)
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        # Compute the desired env from profile JSON (with $VAR expansion).
        # Two-pass expansion lets a profile reference its own keys, e.g.:
        #   QWEN3_ARTIFACT_ROOT=/opt/models/qwen3-edgellm
        #   EDGE_LLM_ASR_ENGINE_DIR=${QWEN3_ARTIFACT_ROOT}/engines/...
        # without depending on the current process env having an identically
        # named (and correct) value already set.
        env_block = profile.get("env") or {}
        new_env: dict[str, str] = {}
        for key, value in env_block.items():
            new_env[key] = _expand_with_profile_env(str(value), env_block)

        derived: dict[str, str] = {
            "OVS_PROFILE_NAME": str(profile.get("name", path.stem)),
        }
        tts_model_id = _derive_tts_model_id(profile)
        if tts_model_id is not None:
            derived["OVS_TTS_MODEL_ID"] = tts_model_id

        merged: dict[str, str] = {**new_env, **derived}
        if overrides:
            for k, v in overrides.items():
                merged[k] = str(v)

        # Kind-scoped reload (swap only ASR or only TTS): touch ONLY this
        # modality's own env keys. Shared keys (roots/labels used by both
        # backends) and the OTHER kind's keys are left exactly as the last
        # full / other-kind reload set them. Without this, switching ASR to a
        # bundle would stomp OVS_TTS_MODEL_ID / *_TTS_* / MOSS_* etc., and the
        # live TTS backend (whose model_id is read from env) would drift; a
        # failed kind reload's rollback could likewise clobber a concurrently
        # reloaded other kind.
        def _in_scope(k: str) -> bool:
            return kind is None or _key_kind(k) == kind

        if kind is not None:
            merged = {k: v for k, v in merged.items() if _in_scope(k)}

        new_keys = set(merged.keys())

        # Profile-owned operator keys: a profile may declare operator-prefixed
        # keys it FULLY owns (e.g. a CustomVoice profile owning its EDGE_LLM_TTS_*
        # engine wiring). For those, the profile's value overrides the import-time
        # operator snapshot (image-baked defaults / inherited env) instead of being
        # silently shadowed by it. Default (field absent) = no owned keys =
        # byte-identical behavior for every existing profile. Use this only when a
        # profile carries the COMPLETE set of values for those keys; partial
        # profiles that rely on baked defaults must NOT list them.
        owned = {str(k) for k in (profile.get("profile_owned_env") or [])}
        eff_operator = _OPERATOR_KEYS - owned

        # 1. Clear stale keys (in previous profile but not in this one),
        #    skipping operator-owned keys AND (for a kind-scoped reload) any key
        #    outside this modality's scope — those belong to the other backend.
        stale = _APPLIED_KEYS - new_keys
        for k in stale:
            if k in eff_operator or not _in_scope(k):
                continue
            os.environ.pop(k, None)

        # 2. Write new values, unconditionally overwriting unless operator-owned.
        for k, v in merged.items():
            if k in eff_operator:
                existing = os.environ.get(k)
                if existing != v:
                    if k in CRITICAL_KEYS:
                        raise RuntimeError(
                            f"Profile {derived['OVS_PROFILE_NAME']} requires "
                            f"{k}={v}, but environment has {k}={existing}. "
                            f"Remove {k} from your environment "
                            f"(compose/.env/shell) or change OVS_PROFILE to "
                            f"one that matches."
                        )
                    logger.warning(
                        "Profile %s wants %s=%s but env overrides to %s",
                        derived["OVS_PROFILE_NAME"], k, v, existing,
                    )
                continue
            os.environ[k] = v

        # 3. Update bookkeeping. For a kind-scoped reload only replace this
        #    modality's tracked keys; the other kind's stay recorded so a later
        #    full reload can still reconcile them.
        if kind is None:
            _APPLIED_KEYS.clear()
        else:
            for k in [k for k in _APPLIED_KEYS if _in_scope(k)]:
                _APPLIED_KEYS.discard(k)
        _APPLIED_KEYS.update(k for k in new_keys if k not in eff_operator)
        _CURRENT_PROFILE = profile

        if resolve_engines:
            # Inject required_engines[].env_var -> engine_path into os.environ
            # so a backend built right after (e.g. on hot-reload) reads the new
            # profile's engine paths instead of stale ones. resolve_all() is
            # cheap when engines are already built (cache hit). The injected
            # keys are NOT part of the static profile env dict, so we fold them
            # into _APPLIED_KEYS — otherwise the next apply_profile's stale-clear
            # would not remove them and they'd pollute across reloads/rollbacks
            # (cf. backend_manager rollback env-pollution class of bug).
            try:
                from server.core.engine_resolver import resolve_all
                injected = resolve_all(profile, kind=kind)
                _APPLIED_KEYS.update(
                    k for k in injected if k not in _OPERATOR_KEYS
                )
            except Exception:
                logger.exception(
                    "apply_profile: engine resolution failed; a freshly built "
                    "backend may see stale/empty engine paths"
                )

        logger.info(
            "Applied profile %s from %s (%d env keys; %d stale cleared)",
            derived["OVS_PROFILE_NAME"],
            path,
            len(_APPLIED_KEYS),
            len(stale),
        )
        return profile


def apply_profile_from_env() -> dict:
    """Compatibility shim: resolve profile from env and apply it.

    Resolution order:
      1. ``OVS_PROFILE_JSON`` — explicit path.
      2. ``OVS_PROFILE`` / ``OVS_PROFILE_DEFAULT`` — profile name.
      3. ``LANGUAGE_MODE`` heuristic (``zh_en`` / ``multilanguage``).
      4. ``OVS_PRESET`` — resolved via ``profile_selector``.
    """
    return apply_profile(None)


# ---------------------------------------------------------------------------
# Artifact pre-flight helpers
# ---------------------------------------------------------------------------

# Env-key suffixes that signal "this value is a filesystem path / artifact".
# Used by ``expected_artifact_paths`` / ``find_missing_artifacts`` to decide
# which env entries are worth existence-checking before a profile reload.
#
# Coverage notes (audit 2026-05-21 against configs/profiles/*.json):
#   - _DIR / _ENGINE / _PATH / _BIN / _MANIFEST / _SETTINGS / _FILTERS: original
#     set covers most jetson/rk profile keys.
#   - _ONNX: KOKORO_SPLIT_*_ONNX, MATCHA_SPLIT_ENCODER_ONNX, PARAFORMER_*_ONNX.
#   - _ROOT: QWEN3_ARTIFACT_ROOT, QWEN3_EDGELLM_JETSON_ROOT.
#   - _BASE: KOKORO_MODEL_BASE, MATCHA_MODEL_BASE.
#   - _VOICES: future Kokoro voices.bin path (not currently in any shipped
#     profile, but cheap to include — startswith("/") filter catches misuse).
#   - _TOKENS: future kokoro/matcha tokens.txt path; current profiles use
#     non-path *_STREAM_MAX_SEGMENT_TOKENS=64 (scalar) — caught by the
#     startswith("/") safety filter below.
#   - _LONG: KOKORO_SPLIT_*_ENGINE_LONG (engine paths used for long context).
#
# Safety net: ``_JSON`` matches both ``EDGE_LLM_ASR_MANIFEST_JSON`` (a path)
# and ``OVS_TTS_SPEAKERS_JSON`` (a JSON blob like '{"0":""}', NOT a path).
# We rely on the ``expanded.startswith("/")`` filter to discard non-path
# values regardless of suffix — never remove that filter without rethinking
# how non-path string env entries flow through this helper.
_PATH_LIKE_SUFFIXES: tuple[str, ...] = (
    "_DIR", "_ENGINE", "_PATH", "_BIN", "_MANIFEST",
    "_SETTINGS", "_FILTERS", "_JSON",
    "_ONNX", "_ROOT", "_BASE", "_VOICES", "_TOKENS", "_LONG",
)


def _expand_with_profile_env(value: str, profile_env: Mapping[str, object]) -> str:
    """Expand ``$VAR`` / ``${VAR}`` using ``profile_env`` first, then ``os.environ``.

    Two-pass expansion handles cases where a profile both defines a key AND
    references it from another key, e.g.::

        QWEN3_ARTIFACT_ROOT=/opt/models/qwen3-edgellm        # profile-defined
        EDGE_LLM_ASR_ENGINE_DIR=${QWEN3_ARTIFACT_ROOT}/...   # profile-referenced

    With plain ``os.path.expandvars`` the second key would resolve against
    whatever ``QWEN3_ARTIFACT_ROOT`` happened to be in the *current* process
    env (often empty or wrong), defeating dry-run artifact pre-flight.

    Unknown variables are preserved as-is (``safe_substitute`` behaviour),
    so a malformed ``$`` in a value never raises.
    """
    merged: dict[str, str] = {}
    # os.environ wins as the fallback layer for things the profile doesn't define.
    merged.update({k: v for k, v in os.environ.items()})
    # Profile values override env so a profile-declared key resolves to its
    # profile value, not whatever stale value is in the current process env.
    for k, v in profile_env.items():
        if isinstance(v, str):
            merged[k] = v
    try:
        return string.Template(value).safe_substitute(merged)
    except (ValueError, KeyError):
        return value


# Kind attribution for path-like env keys, so a *kind-scoped* reload (swap
# only TTS, or only ASR) validates just that modality's engines and ignores
# the other half of a bundled profile. Keys matching neither marker set are
# treated as shared and always validated. This is the SINGLE source of truth
# for kind attribution — engine_resolver imports _key_kind from here so the two
# kind-scoping call sites can never drift (a missed engine family, e.g. MOSS,
# would otherwise leak the other modality's engines into a kind-scoped reload).
# Every TTS/ASR engine family MUST have a marker here.
_ASR_KEY_MARKERS = ("ASR", "PARAFORMER", "SENSEVOICE")
_TTS_KEY_MARKERS = (
    "TTS", "MATCHA", "KOKORO", "VOCOS", "SPARK", "MOSS", "SPEAKER_ENCODER",
)


def _key_kind(key: str) -> str | None:
    """Classify a path-like env key as 'asr' / 'tts' / None (shared)."""
    ku = key.upper()
    is_asr = any(m in ku for m in _ASR_KEY_MARKERS)
    is_tts = any(m in ku for m in _TTS_KEY_MARKERS)
    if is_asr and not is_tts:
        return "asr"
    if is_tts and not is_asr:
        return "tts"
    return None


def expected_artifact_paths(profile: dict, kind: str | None = None) -> dict[str, str]:
    """Return ``{env_key: expanded_absolute_path}`` for env entries that
    look like filesystem artifacts.

    When ``kind`` is 'tts' or 'asr', keys that belong exclusively to the
    *other* modality are skipped — a kind-scoped reload only needs its own
    engines present, not the paired backend's (see :func:`_key_kind`).

    Variable expansion uses the profile's own env block layered on top of
    ``os.environ`` (see :func:`_expand_with_profile_env`), so profiles that
    self-reference (e.g. ``${QWEN3_ARTIFACT_ROOT}/engines/...``) resolve
    against the profile's own value, not whatever happens to be in the
    current process env.

    Heuristic: include env keys whose suffix matches
    :data:`_PATH_LIKE_SUFFIXES` AND whose expanded value starts with ``/``
    (absolute path only). Non-absolute paths and non-string values are
    skipped — they're either repo-relative (e.g. ``deploy/artifacts/...``)
    or non-path config (e.g. ``OVS_TTS_SPEAKERS_JSON='{"0":""}'``).
    """
    result: dict[str, str] = {}
    env_block = profile.get("env") or {}
    for k, v in env_block.items():
        if not isinstance(v, str):
            continue
        if not any(k.endswith(s) for s in _PATH_LIKE_SUFFIXES):
            continue
        if kind is not None:
            kk = _key_kind(k)
            if kk is not None and kk != kind:
                continue
        expanded = _expand_with_profile_env(v, env_block)
        if expanded.startswith("/"):
            result[k] = expanded
    return result


def find_missing_artifacts(profile: dict, kind: str | None = None) -> list[dict]:
    """Return missing-path records for the given profile.

    Empty list means all expected absolute paths exist on disk. Record
    shape: ``{"env_var": <key>, "path": <expanded>}``. Pass ``kind`` to scope
    the check to one modality (a TTS-only reload skips ASR engine paths).
    """
    paths = expected_artifact_paths(profile, kind=kind)
    missing: list[dict] = []
    for k, p in paths.items():
        if not os.path.exists(p):
            missing.append({"env_var": k, "path": p})
    return missing
