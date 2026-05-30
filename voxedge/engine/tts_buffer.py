"""Low-latency TTS chunk buffer — env-free port of app/core/v2v.py.

Faithful port of ``app/core/v2v.py:227-351`` ``LowLatencyTTSBuffer``, the
clause-level chunker the legacy /v2v handler selects for voice-agent TTFA
(emits short CJK clauses / bounded no-punctuation spans early instead of
waiting for a full sentence). The only change vs the legacy source is spec
§2 compliance: the per-language chunk-sizing knobs (``min_chars`` /
``target_chars`` / ``max_chars``) are CONSTRUCTOR-INJECTED rather than read
from ``OVS_TTS_LOW_LATENCY_{CJK,LATIN}_{MIN,TARGET,MAX}_CHARS`` env vars.

The defaults below mirror the legacy env defaults exactly
(app/core/v2v.py:246-248):

    CJK   : min=15  target=24  max=40
    Latin : min=24  target=48  max=80

so a ``LowLatencyTTSBuffer(language="zh")`` here splits byte-identically to
the production buffer with no env set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

# Mirrors app/core/v2v.py:94 — minimum chars before a hard break is honored.
DEFAULT_MIN_SENTENCE_CHARS = 2

# Per-language default chunk-sizing knobs. These mirror the legacy env
# defaults (app/core/v2v.py:246-248) so behavior is identical with no env.
_CJK_DEFAULTS = {"min_chars": 15, "target_chars": 24, "max_chars": 40}
_LATIN_DEFAULTS = {"min_chars": 24, "target_chars": 48, "max_chars": 80}

_CJK_LANGS = ("zh", "chinese", "ja", "japanese", "ko", "korean")


def _contains_cjk(text: str) -> bool:
    # Port of app/core/v2v.py:98-103.
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
            return True
    return False


@dataclass
class LowLatencyTTSBuffer:
    """Emit short TTS-ready chunks without waiting for full sentences.

    Intentionally separate from the conservative ``_SentenceBuffer``: this
    buffer optimizes voice-agent TTFA by emitting CJK clauses and bounded
    no-punctuation spans early.

    Chunk-sizing knobs (``min_chars`` / ``target_chars`` / ``max_chars``) are
    constructor-injected (spec §2: no env reads). When left ``None`` they
    resolve to the per-language defaults that mirror the legacy env defaults
    (CJK 15/24/40, Latin 24/48/80).
    """

    language: Optional[str] = None
    min_chars: Optional[int] = None
    target_chars: Optional[int] = None
    max_chars: Optional[int] = None
    _buf: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        lang = (self.language or "").strip().lower()
        cjk = lang in _CJK_LANGS
        defaults = _CJK_DEFAULTS if cjk else _LATIN_DEFAULTS
        if self.min_chars is None:
            self.min_chars = defaults["min_chars"]
        if self.target_chars is None:
            self.target_chars = defaults["target_chars"]
        if self.max_chars is None:
            self.max_chars = defaults["max_chars"]
        # Clamp ordering exactly as the legacy buffer (app/core/v2v.py:255-257).
        self.min_chars = max(2, int(self.min_chars))
        self.target_chars = max(self.min_chars, int(self.target_chars))
        self.max_chars = max(self.target_chars, int(self.max_chars))

    def add(self, chunk: str) -> Iterator[str]:
        if not chunk:
            return
        self._buf += chunk
        yield from self._emit_ready(final=False)

    def flush(self) -> Iterator[str]:
        yield from self._emit_ready(final=True)

    def is_empty(self) -> bool:
        return not self._buf.strip()

    def _emit_ready(self, *, final: bool) -> Iterator[str]:
        while True:
            part = self._next_chunk(final=final)
            if part is None:
                return
            yield part

    def _next_chunk(self, *, final: bool) -> Optional[str]:
        text = self._buf.lstrip()
        if text != self._buf:
            self._buf = text
        if not self._buf:
            return None

        if final:
            out = self._buf.strip()
            self._buf = ""
            return out or None

        is_cjk = _contains_cjk(self._buf) or (self.language or "").lower() in _CJK_LANGS
        hard_breaks = "。！？!?；;\n"
        soft_breaks = "，,、：:" if is_cjk else ",;:"

        hard_idx = self._first_break_index(self._buf, hard_breaks)
        if hard_idx >= 0:
            end = hard_idx + 1
            if len(self._buf[:end].strip()) >= DEFAULT_MIN_SENTENCE_CHARS:
                return self._take(end)

        soft_idx = self._last_break_index(self._buf, soft_breaks, limit=len(self._buf))
        if soft_idx >= 0 and len(self._buf[: soft_idx + 1].strip()) >= self.min_chars:
            return self._take(soft_idx + 1)
        if soft_idx >= 0 and len(self._buf.strip()) >= self.target_chars:
            return self._take(len(self._buf))

        length_cut_threshold = self.max_chars if is_cjk else self.target_chars
        if len(self._buf.strip()) < length_cut_threshold:
            return None

        end = self._choose_length_cut(is_cjk=is_cjk)
        if end <= 0:
            return None
        return self._take(end)

    def _take(self, end: int) -> Optional[str]:
        out = self._buf[:end].strip()
        self._buf = self._buf[end:].lstrip()
        return out or None

    @staticmethod
    def _first_break_index(text: str, chars: str) -> int:
        found = [text.find(ch) for ch in chars if text.find(ch) >= 0]
        return min(found) if found else -1

    @staticmethod
    def _last_break_index(text: str, chars: str, *, limit: int) -> int:
        window = text[:limit]
        found = [window.rfind(ch) for ch in chars if window.rfind(ch) >= 0]
        return max(found) if found else -1

    def _choose_length_cut(self, *, is_cjk: bool) -> int:
        limit = min(len(self._buf), self.max_chars)
        if is_cjk:
            soft_idx = self._last_break_index(self._buf, "，,、：:", limit=limit)
            if soft_idx >= self.min_chars - 1:
                return soft_idx + 1
            return limit

        window = self._buf[:limit]
        for idx in range(len(window) - 1, self.min_chars - 2, -1):
            if window[idx].isspace():
                return idx + 1
        return min(len(self._buf), self.target_chars)


__all__ = ["LowLatencyTTSBuffer"]
