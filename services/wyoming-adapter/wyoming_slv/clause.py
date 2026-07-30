"""Low-latency clause buffer for streaming synthesis.

Behaviour + sizing mirror ``server/core/v2v.py::LowLatencyTTSBuffer`` (CJK
min 15 / target 24 / max 40; Latin 24 / 48 / 80) so the Wyoming path chunks
text the same way our own v2v path does. It is re-implemented here rather than
imported because this adapter is a standalone service that must not depend on
the voice image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

MIN_SENTENCE_CHARS = 2

_CJK_LANGS = ("zh", "chinese", "ja", "japanese", "ko", "korean")


def contains_cjk(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        if (
            0x4E00 <= o <= 0x9FFF      # CJK unified ideographs
            or 0x3400 <= o <= 0x4DBF   # ext A
            or 0x3040 <= o <= 0x30FF   # kana
            or 0xAC00 <= o <= 0xD7AF   # hangul
        ):
            return True
    return False


@dataclass
class ClauseBuffer:
    language: Optional[str] = None
    min_chars: Optional[int] = None
    target_chars: Optional[int] = None
    max_chars: Optional[int] = None
    _buf: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        lang = (self.language or "").strip().lower()
        cjk = lang in _CJK_LANGS
        prefix = "OVS_TTS_LOW_LATENCY_CJK" if cjk else "OVS_TTS_LOW_LATENCY_LATIN"
        d_min = int(os.environ.get(f"{prefix}_MIN_CHARS", "15" if cjk else "24"))
        d_tgt = int(os.environ.get(f"{prefix}_TARGET_CHARS", "24" if cjk else "48"))
        d_max = int(os.environ.get(f"{prefix}_MAX_CHARS", "40" if cjk else "80"))
        self.min_chars = max(2, int(self.min_chars if self.min_chars is not None else d_min))
        self.target_chars = max(
            self.min_chars, int(self.target_chars if self.target_chars is not None else d_tgt)
        )
        self.max_chars = max(
            self.target_chars, int(self.max_chars if self.max_chars is not None else d_max)
        )

    def add(self, chunk: str) -> Iterator[str]:
        if not chunk:
            return
        self._buf += chunk
        yield from self._emit(final=False)

    def flush(self) -> Iterator[str]:
        yield from self._emit(final=True)

    def is_empty(self) -> bool:
        return not self._buf.strip()

    # ------------------------------------------------------------------ impl
    def _emit(self, *, final: bool) -> Iterator[str]:
        while True:
            part = self._next(final=final)
            if part is None:
                return
            yield part

    def _next(self, *, final: bool) -> Optional[str]:
        stripped = self._buf.lstrip()
        if stripped != self._buf:
            self._buf = stripped
        if not self._buf:
            return None

        if final:
            out = self._buf.strip()
            self._buf = ""
            return out or None

        is_cjk = contains_cjk(self._buf) or (self.language or "").lower() in _CJK_LANGS
        hard_breaks = "。！？!?；;\n"
        soft_breaks = "，,、：:" if is_cjk else ",;:"

        hard_idx = self._first_break(self._buf, hard_breaks)
        if hard_idx >= 0:
            end = hard_idx + 1
            if len(self._buf[:end].strip()) >= MIN_SENTENCE_CHARS:
                return self._take(end)

        soft_idx = self._last_break(self._buf, soft_breaks, limit=len(self._buf))
        if soft_idx >= 0 and len(self._buf[: soft_idx + 1].strip()) >= self.min_chars:
            return self._take(soft_idx + 1)
        if soft_idx >= 0 and len(self._buf.strip()) >= self.target_chars:
            return self._take(len(self._buf))

        threshold = self.max_chars if is_cjk else self.target_chars
        if len(self._buf.strip()) < threshold:
            return None

        end = self._length_cut(is_cjk=is_cjk)
        if end <= 0:
            return None
        return self._take(end)

    def _take(self, end: int) -> Optional[str]:
        out = self._buf[:end].strip()
        self._buf = self._buf[end:].lstrip()
        return out or None

    @staticmethod
    def _first_break(text: str, chars: str) -> int:
        found = [text.find(c) for c in chars if text.find(c) >= 0]
        return min(found) if found else -1

    @staticmethod
    def _last_break(text: str, chars: str, *, limit: int) -> int:
        window = text[:limit]
        found = [window.rfind(c) for c in chars if window.rfind(c) >= 0]
        return max(found) if found else -1

    def _length_cut(self, *, is_cjk: bool) -> int:
        limit = min(len(self._buf), self.max_chars)
        if is_cjk:
            soft_idx = self._last_break(self._buf, "，,、：:", limit=limit)
            if soft_idx >= self.min_chars - 1:
                return soft_idx + 1
            return limit
        window = self._buf[:limit]
        for idx in range(len(window) - 1, self.min_chars - 2, -1):
            if window[idx].isspace():
                return idx + 1
        return min(len(self._buf), self.target_chars)
