"""V2V WebSocket protocol + sentence buffering.

Used by:
- WS /v2v/stream — unified ASR + TTS + VAD + barge-in endpoint
- (optionally exposed as TTS-only / ASR-only by which config keys the
  client supplies)

Protocol spec: docs/api/v2v-stream.md
"""
from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

# pysbd — Python Sentence Boundary Disambiguation. Rule-based, no model
# files, 22 languages, handles abbreviations ("Dr. Smith", "U.S.A."),
# numbers ("3.14"), URLs ("example.com"). ~100 KB pure Python. If it's
# missing (older image, dev env), we fall back to a simple regex that
# over-splits abbreviations but still works.
try:
    import pysbd
    _PYSBD_AVAILABLE = True
except ImportError:
    pysbd = None  # type: ignore
    _PYSBD_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────
# Client → Server JSON message types
# ────────────────────────────────────────────────────────────────────────
CLIENT_CONFIG     = "config"        # initial setup, must be first message
CLIENT_TEXT       = "text"          # streaming text input for TTS
CLIENT_ASR_EOS    = "asr_eos"       # manually finalize ASR (overrides VAD)
CLIENT_TTS_FLUSH  = "tts_flush"     # flush remaining TTS buffer
CLIENT_ABORT      = "abort"         # barge-in: cancel current TTS
# Remote server-side tool loop (spec §4 Mode B). Additive; legacy clients that
# never enable the server tool loop never see/send these.
CLIENT_TOOL_RESULT = "tool_result"  # device client returns a remote-tool result

# ────────────────────────────────────────────────────────────────────────
# Server → Client JSON message types
# ────────────────────────────────────────────────────────────────────────
SERVER_ASR_PARTIAL        = "asr_partial"
SERVER_ASR_ENDPOINT       = "asr_endpoint"       # VAD detected end of speech
SERVER_ASR_FINAL          = "asr_final"
SERVER_TTS_STARTED        = "tts_started"        # first audio frame about to ship
SERVER_TTS_SENTENCE_DONE  = "tts_sentence_done"  # one sentence finished
SERVER_TTS_DONE           = "tts_done"           # flush complete, no more audio
SERVER_VAD_EVENT          = "vad_event"          # server-side VAD speech_start/speech_end
SERVER_ERROR              = "error"
# Remote server-side tool loop (spec §4 Mode B). Server asks a device client to
# run a tool and report back via CLIENT_TOOL_RESULT. Additive; only emitted when
# the server-side tool loop is enabled (OVS_V2V_SERVER_LOOP) with remote tools.
SERVER_TOOL_CALL          = "tool_call"

# vad_event "event" field values
VAD_EVENT_SPEECH_START    = "speech_start"
VAD_EVENT_SPEECH_END      = "speech_end"


# ────────────────────────────────────────────────────────────────────────
# Sentence buffering for streaming TTS input
# ────────────────────────────────────────────────────────────────────────

# Languages pysbd 0.3.4 supports out-of-the-box (ISO-639-1).
_PYSBD_LANGS = {
    "am", "ar", "bg", "da", "de", "el", "en", "es", "fa", "fr",
    "hi", "hy", "it", "ja", "kk", "mr", "my", "nl", "pl", "ru",
    "ur", "zh",
}

# Verbose names → ISO codes. Customer configs sometimes pass these.
_LANG_ALIASES = {
    "english": "en",    "chinese": "zh",    "japanese": "ja",
    "korean": "ko",     "spanish": "es",    "french": "fr",
    "german": "de",     "italian": "it",    "portuguese": "pt",
    "russian": "ru",    "arabic": "ar",     "hindi": "hi",
    "dutch": "nl",      "polish": "pl",     "greek": "el",
    "burmese": "my",    "marathi": "mr",
}


def _normalize_lang(lang: Optional[str]) -> Optional[str]:
    """Return ISO 639-1 code if pysbd supports it, else None (caller
    falls back to the regex splitter)."""
    if not lang:
        return None
    lc = str(lang).strip().lower()
    code = _LANG_ALIASES.get(lc, lc)
    return code if code in _PYSBD_LANGS else None


# Regex-fallback sentence boundary: CJK terminators always count; ASCII
# `.!?` only count when followed by whitespace or buffer-end (avoids
# "3.14" but still over-splits "Dr. Smith" — that's why we prefer pysbd
# when available).
_SENTENCE_END_RE = re.compile(r"[。！？；\n]+|[!?.](?=\s|$)")

DEFAULT_MIN_SENTENCE_CHARS = 2
DEFAULT_MAX_BUFFER_CHARS   = 200


def _contains_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
            return True
    return False


@dataclass
class SentenceBuffer:
    """Accumulates streaming text and emits complete sentences.

    Used to bridge a token-streaming source (LLM) to a sentence-batched
    sink (TTS engine). Two implementations:

    1. pysbd-backed (default when language is recognized & pysbd is
       installed) — correctly handles abbreviations, numbers, URLs.
    2. regex-backed fallback — splits on punctuation; over-splits
       abbreviations like "Dr. Smith" but works everywhere.

    Usage::

        buf = SentenceBuffer(language="en")     # or "zh"/"ja"/...
        for token in llm_tokens:
            for sentence in buf.add(token):
                tts.synthesize(sentence)
        for sentence in buf.flush():            # at end-of-stream
            tts.synthesize(sentence)

    Note on streaming latency: when the pysbd path is active, a sentence
    is only emitted once the buffer contains the NEXT sentence's first
    characters (pysbd needs lookahead to confidently split). For typical
    LLM streams with sub-50 ms inter-token gaps this is invisible. If
    you have a one-shot final sentence, call `flush()` to force it out.
    """

    language:   Optional[str] = None
    min_chars:  int = DEFAULT_MIN_SENTENCE_CHARS
    max_buffer: int = DEFAULT_MAX_BUFFER_CHARS
    _buf:       str = field(default="", init=False, repr=False)
    _seg:       object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        code = _normalize_lang(self.language)
        if _PYSBD_AVAILABLE and code is not None:
            try:
                self._seg = pysbd.Segmenter(language=code, clean=False)
            except Exception:
                self._seg = None

    # ─── public API ────────────────────────────────────────────────

    def add(self, chunk: str) -> Iterator[str]:
        """Append text, yield any sentences now complete."""
        if not chunk:
            return
        self._buf += chunk
        if self._seg is not None:
            yield from self._emit_pysbd()
        else:
            yield from self._emit_regex()

    def flush(self) -> Iterator[str]:
        """Yield remaining text as a final sentence (no min-length check)."""
        leftover = self._buf.strip()
        self._buf = ""
        if leftover:
            yield leftover

    def is_empty(self) -> bool:
        return not self._buf.strip()

    @property
    def using_pysbd(self) -> bool:
        """For tests / observability — confirms which splitter is active."""
        return self._seg is not None

    # ─── pysbd path ────────────────────────────────────────────────

    def _emit_pysbd(self) -> Iterator[str]:
        # pysbd.segment returns *all* sentences in the input. The LAST
        # element might be incomplete (still buffering); the prefix
        # elements are confirmed sentence boundaries.
        sentences = self._seg.segment(self._buf)   # type: ignore[union-attr]
        if len(sentences) > 1:
            for s in sentences[:-1]:
                stripped = s.strip()
                if len(stripped) >= self.min_chars:
                    yield stripped
                # else: too short, swallow it (rare edge case — pysbd
                # rarely emits sub-min sentences; merging back would
                # confuse pysbd state in the next call)
            self._buf = sentences[-1]
            return
        # Single sentence so far — wait for more text. But guard against
        # runaway buffer (e.g. an LLM with no punctuation).
        if len(self._buf) >= self.max_buffer:
            out = self._buf.strip()
            self._buf = ""
            if out:
                yield out

    # ─── regex fallback path ───────────────────────────────────────

    def _emit_regex(self) -> Iterator[str]:
        while True:
            sentence = self._extract_next_sentence_regex()
            if sentence is None:
                return
            yield sentence

    def _extract_next_sentence_regex(self) -> Optional[str]:
        pos = 0
        while True:
            m = _SENTENCE_END_RE.search(self._buf, pos)
            if m is None:
                if len(self._buf) >= self.max_buffer:
                    out = self._buf.strip()
                    self._buf = ""
                    return out or None
                return None
            end = m.end()
            prefix = self._buf[:end]
            if len(prefix.strip()) >= self.min_chars:
                self._buf = self._buf[end:]
                return prefix.strip()
            pos = end


@dataclass
class LowLatencyTTSBuffer:
    """Emit short TTS-ready chunks without waiting for full sentences.

    This is intentionally separate from ``SentenceBuffer``. SentenceBuffer is
    conservative and linguistically cleaner; this buffer optimizes voice-agent
    TTFA by emitting CJK clauses and bounded no-punctuation spans early.
    """

    language: Optional[str] = None
    min_chars: Optional[int] = None
    target_chars: Optional[int] = None
    max_chars: Optional[int] = None
    _buf: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        lang = (self.language or "").strip().lower()
        cjk = lang in ("zh", "chinese", "ja", "japanese", "ko", "korean")
        prefix = "OVS_TTS_LOW_LATENCY_CJK" if cjk else "OVS_TTS_LOW_LATENCY_LATIN"
        default_min = int(os.environ.get(f"{prefix}_MIN_CHARS", "15" if cjk else "24"))
        default_target = int(os.environ.get(f"{prefix}_TARGET_CHARS", "24" if cjk else "48"))
        default_max = int(os.environ.get(f"{prefix}_MAX_CHARS", "40" if cjk else "80"))
        if self.min_chars is None:
            self.min_chars = default_min
        if self.target_chars is None:
            self.target_chars = default_target
        if self.max_chars is None:
            self.max_chars = default_max
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

        is_cjk = _contains_cjk(self._buf) or (self.language or "").lower() in (
            "zh",
            "chinese",
            "ja",
            "japanese",
            "ko",
            "korean",
        )
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
