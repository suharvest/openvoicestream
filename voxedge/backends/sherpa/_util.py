"""Decoupled helpers for the voxedge sherpa adapter.

adapted from app/core/language.py + app/core/tts_speakers.py (2026-05-30),
dedup after registry switch.

These are minimal, env-free reproductions of the small helper functions the
production sherpa backends imported from ``app.core``. voxedge must not import
``app.*`` (open-core split), so the necessary logic is reproduced here with
**zero** module-scope env reads and zero file I/O.

Why a trimmed copy instead of the full ``app.core.tts_speakers`` registry:
the sherpa backends are single-speaker (the production ``_PRESETS["sherpa"]``
table is ``_SINGLE_SPEAKER``), and the only call site passes
``allow_embedding=False``. So the full registry — file persistence,
``OVS_TTS_SPEAKERS_JSON`` env parsing, embedding registration — is unused on
this path. We reproduce only ``resolve_speaker_kwargs`` for preset ids.
"""

from __future__ import annotations

from typing import Optional


# ── from app/core/language.py:8-45 ──────────────────────────────────────────

_AUTO_VALUES = {"", "auto", "detect", "default"}


def normalize_auto_language(language: Optional[str]) -> Optional[str]:
    """Return None when the caller asked the backend to auto-detect."""
    if language is None:
        return None
    lang = str(language).strip()
    if lang.lower() in _AUTO_VALUES:
        return None
    return lang


def detect_zh_en(text: str, language: Optional[str] = None) -> str:
    """Detect the TTS language used by bilingual Matcha-style backends.

    The zh-en Matcha model can handle embedded English in Chinese text. For
    mixed input we therefore anchor to ``zh`` when any CJK character exists,
    and only return ``en`` for pure Latin/non-CJK input.
    """
    explicit = normalize_auto_language(language)
    if explicit:
        lowered = explicit.lower()
        if lowered in {"chinese", "mandarin", "cn", "zh-cn", "zh_hans"}:
            return "zh"
        if lowered in {"english", "en-us", "en_us", "us"}:
            return "en"
        return explicit

    for ch in text:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
        ):
            return "zh"
    return "en"


# ── trimmed from app/core/tts_speakers.py:555-590 ────────────────────────────
# Single-speaker preset path only. No registry / file / env. The production
# helper resolves through a model-scoped registry; for the sherpa adapter the
# registry is the single-speaker table so the resolved kwargs are simply the
# passed-in speaker_id (or none).


def resolve_speaker_kwargs(
    *,
    allow_embedding: bool = False,
    **kwargs: object,
) -> dict[str, object]:
    """Env-free, registry-free speaker kwargs resolver for single-speaker TTS.

    Input priority (first wins), mirroring app/core/tts_speakers.py:
    1. ``speaker_embedding`` — raw bytes (rejected here: sherpa has no clone).
    2. ``speaker_id`` — numeric id passed straight through.
    3. ``sid`` — deprecated alias for speaker_id.

    Returns ``{"speaker_id": int}`` when an id is provided, else ``{}``. The
    sherpa backend layers its own ``default_speaker_id`` on top when absent.
    """
    emb = kwargs.get("speaker_embedding")
    if emb is not None:
        if not allow_embedding:
            raise ValueError("sherpa backend does not support voice clone embeddings")
        return {"speaker_embedding": emb}

    sid = kwargs.get("speaker_id", kwargs.get("sid"))
    if sid is not None:
        return {"speaker_id": int(sid)}

    return {}
