"""Decoupled helpers for the voxedge RK adapter.

adapted from app/core/language.py + app/core/tts_speakers.py (2026-05-30),
dedup after registry switch.

These are minimal, env-free reproductions of the small helper functions the
production RK backends imported from ``app.core``. voxedge must not import
``app.*`` (open-core split), so the necessary logic is reproduced here with
**zero** module-scope env reads and zero file I/O.

The RK TTS adapter is single-speaker (no voice clone; the only call sites pass
``allow_embedding=False``), so — as with the sherpa adapter — the full
registry (file persistence, ``OVS_TTS_SPEAKERS_JSON`` parsing, embedding
registration) is unused on this path. We reproduce only the single-speaker
``resolve_speaker_kwargs`` (keeping the ``model_id`` positional arg for
call-site compatibility) and ``detect_zh_en``.
"""

from __future__ import annotations

from typing import Optional


# ── from app/core/language.py ───────────────────────────────────────────────

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
    """Detect the TTS language used by bilingual zh-en backends.

    Anchors to ``zh`` when any CJK character exists, and only returns ``en``
    for pure Latin/non-CJK input.
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
# Single-speaker preset path only. No registry / file / env. ``model_id`` is
# kept as a positional arg to match the production call sites
# (``resolve_speaker_kwargs(self.model_id, allow_embedding=False, ...)``) but
# is otherwise unused on the single-speaker RK path.


def resolve_speaker_kwargs(
    model_id: str,
    *,
    allow_embedding: bool = False,
    **kwargs: object,
) -> dict[str, object]:
    """Env-free, registry-free speaker kwargs resolver for single-speaker TTS.

    Input priority (first wins), mirroring app/core/tts_speakers.py:
    1. ``speaker_embedding`` — raw bytes (rejected here: RK TTS has no clone).
    2. ``speaker_id`` — numeric id passed straight through.
    3. ``sid`` — deprecated alias for speaker_id.

    Returns ``{"speaker_id": int}`` when an id is provided, else ``{}``. The RK
    backend layers its own default speaker id (``0``) on top when absent.
    """
    emb = kwargs.get("speaker_embedding")
    if emb is not None:
        if not allow_embedding:
            raise ValueError(
                f"Model {model_id!r} does not support voice clone embeddings"
            )
        return {"speaker_embedding": emb}

    sid = kwargs.get("speaker_id", kwargs.get("sid"))
    if sid is not None:
        return {"speaker_id": int(sid)}

    return {}
