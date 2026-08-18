"""Automatic segmentation for the offline ASR path — no more silent truncation.

Background: fixed-shape offline engines (SenseVoice TRT/RKNN accepts exactly
344 LFR frames ≈ 20.4 s of audio) drop the tail of a longer clip with no error
and no log line, so the caller receives a normal 200 carrying a truncated
transcript. ``server/core/asr_segmenter.py`` cuts long clips into segments —
at VAD-detected silence when a VAD is available, overlapped fixed slices
otherwise — decodes each one and re-joins the text.

Coverage:

  1. Clip under the threshold → exactly one ``transcribe()`` call, no
     segmentation metadata (legacy path untouched).
  2. 30 s clip → several ``transcribe()`` calls, every segment's text present.
  3. Joined text is in segment order.
  4. ``OVS_ASR_AUTO_SEGMENT=0`` → one call again (kill switch).
  5. Join spacing: no space between CJK, one space between Latin.
  6. ``/v1/audio/transcriptions`` segments long clips end-to-end and fills the
     ``verbose_json`` ``segments`` array.
  7. Non-WAV / truncated header → no segmentation, but a WARNING is logged
     (never a silent truncation).
  8. VAD boundary planning, exercised with a mocked VAD session, plus the
     fixed-slice fallback when the VAD blows up.

The tests never require a real Silero model: VAD is either mocked or forced
off, and the fixed-length fallback carries the HTTP-level cases.
"""
from __future__ import annotations

import io
import logging
import os
import wave

import pytest
from fastapi.testclient import TestClient

from server.core import asr_segmenter as seg
from server.tests.test_main_hot_swap import _FakeASRBackend, _install_managers

SR = 16000


# ── Helpers ──────────────────────────────────────────────────────────

def _make_wav(seconds: float = 3.0, sample_rate: int = SR, channels: int = 1) -> bytes:
    """PCM16 silence WAV of the given duration (stdlib only)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * channels * int(sample_rate * seconds))
    return buf.getvalue()


class _FakeResult:
    def __init__(self, text: str, language: str | None = "zh"):
        self.text = text
        self.language = language
        self.meta: dict = {}


class _CountingASRBackend(_FakeASRBackend):
    """Returns a distinguishable transcript per call so joins are checkable."""

    name = "counting-fake"

    def __init__(self, texts: list[str] | None = None, language: str | None = "zh"):
        super().__init__()
        self.calls: list[dict] = []
        self._texts = texts
        self._language = language

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> _FakeResult:
        idx = len(self.calls)
        self.calls.append({"language": language, "n_bytes": len(audio_bytes)})
        if self._texts is not None:
            text = self._texts[idx % len(self._texts)]
        else:
            text = f"seg{idx}"
        return _FakeResult(text, self._language)


class _TruncatingASRBackend(_FakeASRBackend):
    """Emulates a fixed-shape engine: only the first ``limit`` seconds decode.

    Words are one per second of audio, so a truncated decode is visibly short.
    """

    name = "truncating-fake"
    LIMIT_SECONDS = 20.4

    def __init__(self, words: list[str]):
        super().__init__()
        self.words = words
        self.calls: list[float] = []

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> _FakeResult:
        clip = seg.probe_wav(audio_bytes)
        assert clip is not None
        # Which words of the original clip does this slice cover? The fake
        # server hands us slices, so recover the offset from the byte length of
        # everything decoded so far is not possible — instead the caller passes
        # slices in order, and we track a cursor.
        self.calls.append(clip.duration)
        start = sum(self.calls[:-1])
        # Overlapped fixed slices re-read a little audio; clamp to the clip.
        visible = min(clip.duration, self.LIMIT_SECONDS)
        i0 = int(start)
        i1 = min(len(self.words), int(start + visible))
        return _FakeResult(" ".join(self.words[i0:i1]), "en")


@pytest.fixture
def harness(monkeypatch):
    """(client_factory, backend_installer) with managers/limiter installed."""
    monkeypatch.delenv("OVS_API_KEYS", raising=False)
    # Force the deterministic fixed-slice planner: no Silero model needed.
    monkeypatch.setenv(seg.ENV_SEGMENT_VAD, "none")
    monkeypatch.delenv(seg.ENV_AUTO_SEGMENT, raising=False)
    monkeypatch.delenv(seg.ENV_MAX_SEGMENT_SECONDS, raising=False)

    def install(backend):
        _install_managers(asr=backend)
        from server.main import app
        return TestClient(app)

    return install


def _post_asr(client: TestClient, audio: bytes, **params):
    return client.post(
        "/asr", params=params, files={"file": ("a.wav", audio, "audio/wav")}
    )


# ── 1. Short audio keeps the legacy single-shot path ─────────────────

def test_short_audio_decodes_once(harness):
    be = _CountingASRBackend()
    client = harness(be)
    r = _post_asr(client, _make_wav(3.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(be.calls) == 1, "short clip must not be segmented"
    assert body["text"] == "seg0"
    assert body["language"] == "zh"
    assert body["backend"] == "counting-fake"
    assert "segmentation" not in body
    assert "segments_decoded" not in body


def test_short_audio_receives_whole_clip(harness):
    """The single call still gets the original bytes, not a re-encoded slice."""
    be = _CountingASRBackend()
    client = harness(be)
    audio = _make_wav(5.0)
    _post_asr(client, audio)
    assert be.calls[0]["n_bytes"] == len(audio)


# ── 2/3. Long audio is split, decoded per segment, joined in order ───

def test_long_audio_is_segmented(harness):
    be = _CountingASRBackend()
    client = harness(be)
    r = _post_asr(client, _make_wav(30.0))
    assert r.status_code == 200, r.text
    body = r.json()

    assert len(be.calls) > 1, "30 s clip must be split"
    assert body["segments_decoded"] == len(be.calls)
    # Every segment's transcript survives into the joined text.
    for i in range(len(be.calls)):
        assert f"seg{i}" in body["text"]
    # ...and in decode order.
    expected = " ".join(f"seg{i}" for i in range(len(be.calls)))
    assert body["text"] == expected

    meta = body["segmentation"]
    assert meta["applied"] is True
    assert meta["strategy"] == "fixed"
    assert meta["audio_duration"] == pytest.approx(30.0, abs=0.01)
    assert len(meta["segments"]) == len(be.calls)
    # Boundaries are monotonic, non-degenerate, and cover the whole clip.
    assert meta["segments"][0]["start"] == 0.0
    assert meta["segments"][-1]["end"] == pytest.approx(30.0, abs=0.01)
    prev_start = -1.0
    for s in meta["segments"]:
        assert s["end"] > s["start"]
        assert s["start"] > prev_start
        prev_start = s["start"]


def test_long_audio_segment_lengths_respect_threshold(harness, monkeypatch):
    monkeypatch.setenv(seg.ENV_MAX_SEGMENT_SECONDS, "10.5")
    be = _CountingASRBackend()
    client = harness(be)
    r = _post_asr(client, _make_wav(30.0))
    for s in r.json()["segmentation"]["segments"]:
        assert (s["end"] - s["start"]) <= 10.5 + 1e-6


def test_language_forwarded_to_every_segment(harness):
    be = _CountingASRBackend()
    client = harness(be)
    _post_asr(client, _make_wav(30.0), language="en")
    assert len(be.calls) > 1
    assert all(c["language"] == "en" for c in be.calls)


def test_join_order_is_segment_order():
    """Joining is positional — later segments never jump ahead."""
    assert seg.join_segment_texts(["one", "two", "three"]) == "one two three"
    assert seg.join_segment_texts(["", "two", "", "three"]) == "two three"


# ── 4. Kill switch ───────────────────────────────────────────────────

def test_auto_segment_disabled_restores_legacy_behaviour(harness, monkeypatch):
    monkeypatch.setenv(seg.ENV_AUTO_SEGMENT, "0")
    be = _CountingASRBackend()
    client = harness(be)
    r = _post_asr(client, _make_wav(30.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(be.calls) == 1, "OVS_AUTO_SEGMENT=0 must not segment"
    assert body["text"] == "seg0"
    assert "segmentation" not in body


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_auto_segment_flag_falsy_values(value, monkeypatch):
    monkeypatch.setenv(seg.ENV_AUTO_SEGMENT, value)
    assert seg.auto_segment_enabled() is False


@pytest.mark.parametrize("value", [None, "", "1", "true", "yes"])
def test_auto_segment_flag_default_on(value, monkeypatch):
    if value is None:
        monkeypatch.delenv(seg.ENV_AUTO_SEGMENT, raising=False)
    else:
        monkeypatch.setenv(seg.ENV_AUTO_SEGMENT, value)
    assert seg.auto_segment_enabled() is True


# ── 5. Join spacing rules ────────────────────────────────────────────

def test_join_cjk_without_space():
    assert seg.join_segment_texts(["今天天气", "非常好"]) == "今天天气非常好"


def test_join_latin_with_space():
    assert seg.join_segment_texts(["hello there", "world again"]) == (
        "hello there world again"
    )


def test_join_mixed_boundary_gets_space():
    assert seg.join_segment_texts(["中文", "english"]) == "中文 english"
    assert seg.join_segment_texts(["english", "中文"]) == "english 中文"


def test_join_respects_existing_whitespace():
    assert seg.join_segment_texts(["abc", "  def"]) == "abc def"
    assert seg.join_segment_texts(["中文。", "下一句"]) == "中文。下一句"


def test_is_cjk_classification():
    for ch in "中文漢字あアハン한":
        assert seg.is_cjk(ch), ch
    for ch in "aZ1 ,.":
        assert not seg.is_cjk(ch), ch


def test_segmented_response_spacing_is_cjk_aware(harness):
    be = _CountingASRBackend(texts=["第一段", "第二段", "第三段", "第四段"])
    client = harness(be)
    r = _post_asr(client, _make_wav(30.0))
    text = r.json()["text"]
    assert " " not in text
    assert text.startswith("第一段第二段")


# ── 6. OpenAI-compatible route ───────────────────────────────────────

def test_openai_transcriptions_segments_long_audio(harness):
    be = _CountingASRBackend()
    client = harness(be)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _make_wav(30.0), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    assert len(be.calls) > 1
    body = r.json()
    assert set(body) == {"text"}, "json format shape must stay {text}"
    for i in range(len(be.calls)):
        assert f"seg{i}" in body["text"]


def test_openai_verbose_json_fills_segments(harness):
    be = _CountingASRBackend()
    client = harness(be)
    r = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "verbose_json"},
        files={"file": ("a.wav", _make_wav(30.0), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "transcribe"
    assert body["duration"] == pytest.approx(30.0, abs=0.01)
    assert len(body["segments"]) == len(be.calls) > 1
    first = body["segments"][0]
    assert first["id"] == 0
    assert first["start"] == 0.0
    assert first["text"] == "seg0"
    assert body["segments"][-1]["end"] == pytest.approx(30.0, abs=0.01)


def test_openai_short_audio_segments_stay_empty(harness):
    """Existing contract: single-pass decode still reports segments == []."""
    be = _CountingASRBackend()
    client = harness(be)
    r = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "verbose_json"},
        files={"file": ("a.wav", _make_wav(3.0), "audio/wav")},
    )
    assert r.json()["segments"] == []
    assert len(be.calls) == 1


# ── 7. Undecodable containers degrade loudly, never silently ─────────

def test_non_wav_payload_logs_warning_and_skips_segmentation(caplog):
    be = _CountingASRBackend()
    with caplog.at_level(logging.WARNING, logger="server.core.asr_segmenter"):
        out = seg.maybe_transcribe_segmented(be, b"OggS\x00" + b"\x11" * 4096)
    assert out is None, "unknown duration must not be guessed at"
    assert be.calls == [], "the segmenter must not decode on the skip path"
    assert any(
        "cannot read audio duration" in rec.getMessage() for rec in caplog.records
    ), caplog.text
    assert "truncate" in caplog.text.lower(), "warning must name the risk"


def test_broken_wav_header_logs_warning(caplog):
    be = _CountingASRBackend()
    broken = _make_wav(30.0)[:20]  # header cut mid-way
    with caplog.at_level(logging.WARNING, logger="server.core.asr_segmenter"):
        out = seg.maybe_transcribe_segmented(be, broken)
    assert out is None
    assert caplog.records, "degradation must be observable"


def test_non_wav_upload_still_returns_200(harness, caplog):
    """HTTP behaviour on an unreadable container is unchanged (no new 5xx)."""
    be = _CountingASRBackend()
    client = harness(be)
    with caplog.at_level(logging.WARNING):
        r = client.post("/asr", files={"file": ("a.ogg", b"OggS" + b"\x00" * 8192)})
    assert r.status_code == 200, r.text
    assert len(be.calls) == 1
    assert "segmentation" not in r.json()


def test_probe_wav_and_duration_helpers():
    assert seg.probe_wav(b"not audio") is None
    assert seg.wav_duration_seconds(b"not audio") == 0.0
    clip = seg.probe_wav(_make_wav(2.5))
    assert clip is not None
    assert clip.sample_rate == SR
    assert clip.duration == pytest.approx(2.5, abs=1e-6)
    assert seg.wav_duration_seconds(_make_wav(7.25)) == pytest.approx(7.25, abs=1e-6)


# ── 8. Planning: fixed slices, VAD, and VAD failure ──────────────────

def test_fixed_bounds_cover_clip_with_overlap():
    bounds = seg.fixed_segment_bounds(30.0, 10.5, 0.4)
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == pytest.approx(30.0)
    for (s0, e0), (s1, _e1) in zip(bounds, bounds[1:]):
        assert s1 < e0, "adjacent slices must overlap"
        assert e0 - s1 == pytest.approx(0.4, abs=1e-6)
        assert e0 - s0 <= 10.5 + 1e-9


def test_fixed_bounds_single_when_short():
    assert seg.fixed_segment_bounds(5.0, 10.5, 0.4) == [(0.0, 5.0)]
    assert seg.fixed_segment_bounds(0.0, 10.5, 0.4) == []


def test_fixed_bounds_absorb_tiny_tail():
    bounds = seg.fixed_segment_bounds(10.6, 10.5, 0.0)
    assert len(bounds) == 1
    assert bounds[0] == (0.0, pytest.approx(10.6))


def test_group_regions_merges_and_splits():
    # Three short speech regions: the first two fit together, the third starts
    # too late to join them.
    regions = [(0.0, 4.0), (4.5, 9.0), (12.0, 15.0)]
    grouped = seg.group_regions(regions, 10.0, 0.4)
    assert grouped == [(0.0, 9.0), (12.0, 15.0)]

    # A single region longer than the budget has no silence to cut at, so it
    # gets sliced.
    grouped = seg.group_regions([(0.0, 25.0)], 10.0, 0.5)
    assert len(grouped) > 1
    assert grouped[0][0] == 0.0
    assert grouped[-1][1] == pytest.approx(25.0)


class _MockVAD:
    """Emits speech_start / speech_end at scripted window indices."""

    WINDOW_16K = 256

    def __init__(self, script: dict[int, str]):
        self.script = script
        self.i = -1

    def process(self, _samples):
        self.i += 1
        return self.script.get(self.i)

    def reset(self):
        self.i = -1


def test_plan_segments_uses_vad_boundaries():
    clip = seg.probe_wav(_make_wav(30.0))
    assert clip is not None
    win = _MockVAD.WINDOW_16K
    per_sec = SR / win  # 62.5 windows per second
    # Speech 0-9 s, then 11-20 s, then 22-30 s.
    script = {
        0: "speech_start",
        int(9 * per_sec): "speech_end",
        int(11 * per_sec): "speech_start",
        int(20 * per_sec): "speech_end",
        int(22 * per_sec): "speech_start",
    }
    bounds, strategy = seg.plan_segments(
        clip,
        max_segment_seconds=10.5,
        overlap_seconds=0.4,
        vad_backend="silero",
        vad_factory=lambda: _MockVAD(script),
    )
    assert strategy == "vad"
    assert len(bounds) == 3
    # Cuts land in the scripted silences (± the 0.1 s pad), not at 10.5 s.
    assert bounds[0][1] == pytest.approx(9.1, abs=0.05)
    assert bounds[1][0] == pytest.approx(10.9, abs=0.05)
    assert bounds[1][1] == pytest.approx(20.1, abs=0.05)
    assert bounds[2][1] == pytest.approx(30.0, abs=0.05)


def test_plan_segments_falls_back_when_vad_raises():
    clip = seg.probe_wav(_make_wav(30.0))
    assert clip is not None

    def _boom():
        raise RuntimeError("silero model missing")

    bounds, strategy = seg.plan_segments(
        clip, 10.5, 0.4, vad_backend="silero", vad_factory=_boom
    )
    assert strategy == "fixed_vad_unavailable"
    assert len(bounds) > 1
    assert bounds[-1][1] == pytest.approx(30.0)


def test_plan_segments_falls_back_when_vad_finds_nothing():
    clip = seg.probe_wav(_make_wav(30.0))
    assert clip is not None
    bounds, strategy = seg.plan_segments(
        clip, 10.5, 0.4, vad_backend="silero", vad_factory=lambda: _MockVAD({})
    )
    assert strategy == "fixed_vad_empty"
    assert len(bounds) > 1


def test_plan_segments_skips_vad_for_unsupported_rate():
    clip = seg.probe_wav(_make_wav(30.0, sample_rate=8000))
    assert clip is not None
    bounds, strategy = seg.plan_segments(clip, 10.5, 0.4, vad_backend="silero")
    assert strategy == "fixed_vad_unavailable"
    assert len(bounds) > 1


def test_vad_path_end_to_end_with_mock(monkeypatch):
    """Full segmented decode driven by a mocked VAD (no Silero model needed)."""
    win = _MockVAD.WINDOW_16K
    per_sec = SR / win
    script = {
        0: "speech_start",
        int(8 * per_sec): "speech_end",
        int(10 * per_sec): "speech_start",
        int(18 * per_sec): "speech_end",
        int(20 * per_sec): "speech_start",
    }
    monkeypatch.setenv(seg.ENV_SEGMENT_VAD, "silero")
    be = _CountingASRBackend()
    out = seg.maybe_transcribe_segmented(
        be, _make_wav(30.0), language="zh", vad_factory=lambda: _MockVAD(script)
    )
    assert out is not None
    assert out.strategy == "vad"
    assert out.segments_decoded == len(be.calls) == 3
    assert out.text == "seg0 seg1 seg2"  # ASCII boundaries → single space
    assert out.language == "zh"


# ── Threshold resolution ─────────────────────────────────────────────

def test_max_segment_seconds_default(monkeypatch):
    monkeypatch.delenv(seg.ENV_MAX_SEGMENT_SECONDS, raising=False)
    assert seg.resolve_max_segment_seconds(None) == pytest.approx(10.5)


def test_max_segment_seconds_env_override(monkeypatch):
    monkeypatch.setenv(seg.ENV_MAX_SEGMENT_SECONDS, "15.5")
    assert seg.resolve_max_segment_seconds(None) == pytest.approx(15.5)


def test_max_segment_seconds_from_backend(monkeypatch):
    monkeypatch.delenv(seg.ENV_MAX_SEGMENT_SECONDS, raising=False)

    class _Declaring(_CountingASRBackend):
        max_offline_audio_seconds = 18.0

    assert seg.resolve_max_segment_seconds(_Declaring()) == pytest.approx(18.0)


def test_env_beats_backend_declaration(monkeypatch):
    monkeypatch.setenv(seg.ENV_MAX_SEGMENT_SECONDS, "6")

    class _Declaring(_CountingASRBackend):
        max_offline_audio_seconds = 18.0

    assert seg.resolve_max_segment_seconds(_Declaring()) == pytest.approx(6.0)


def test_backend_threshold_drives_segmentation(monkeypatch):
    monkeypatch.delenv(seg.ENV_MAX_SEGMENT_SECONDS, raising=False)
    monkeypatch.setenv(seg.ENV_SEGMENT_VAD, "none")

    class _Declaring(_CountingASRBackend):
        max_segment_seconds = 6.0

    be = _Declaring()
    out = seg.maybe_transcribe_segmented(be, _make_wav(20.0))
    assert out is not None
    assert len(be.calls) >= 4
    for s in out.segments:
        assert s.duration <= 6.0 + 1e-6


# ── The regression this whole module exists for ──────────────────────

def test_segmentation_recovers_text_a_fixed_shape_engine_would_drop(monkeypatch):
    """30 s of speech: unsegmented decode loses everything past ~20.4 s.

    ``_TruncatingASRBackend`` mimics SenseVoice TRT's fixed 344-frame input.
    One word per second makes the loss countable.
    """
    monkeypatch.setenv(seg.ENV_SEGMENT_VAD, "none")
    monkeypatch.setenv(seg.ENV_MAX_SEGMENT_SECONDS, "10.5")
    words = [f"w{i:02d}" for i in range(30)]
    audio = _make_wav(30.0)

    # (a) legacy behaviour — one shot into the fixed-shape engine
    legacy_be = _TruncatingASRBackend(words)
    legacy_text = legacy_be.transcribe(audio).text
    legacy_words = legacy_text.split()
    assert len(legacy_words) == 20, legacy_text
    assert "w29" not in legacy_words, "engine was supposed to truncate"

    # (b) segmented
    seg_be = _TruncatingASRBackend(words)
    out = seg.maybe_transcribe_segmented(seg_be, audio, language="en")
    assert out is not None
    assert out.segments_decoded > 1
    seg_words = out.text.split()

    missing = [w for w in words if w not in seg_words]
    assert missing == [], f"segmented decode still dropped {missing}"
    recovered = [w for w in words if w not in legacy_words and w in seg_words]
    assert len(recovered) == 10, recovered
    print(
        f"\nunsegmented: {len(legacy_words)}/30 words, last={legacy_words[-1]}"
        f"\nsegmented  : {len(set(seg_words))}/30 words in "
        f"{out.segments_decoded} segments, last={seg_words[-1]}"
        f"\nrecovered  : {recovered}"
    )


# ── Backends that chunk internally must not be double-segmented ──────
# Paraformer's own transcribe_audio() walks the whole clip in 400-frame chunks
# while carrying CIF state and the decoder cache across boundaries. Splitting
# here would reset that carried state at every cut, so segmentation must stay
# out of its way.

class _ParaformerLikeBackend(_CountingASRBackend):
    name = "paraformer_trt"


def test_internal_chunking_backend_is_not_segmented(monkeypatch):
    monkeypatch.delenv(seg.ENV_MAX_SEGMENT_SECONDS, raising=False)
    backend = _ParaformerLikeBackend()

    assert seg.backend_chunks_internally(backend) is True
    assert seg.resolve_max_segment_seconds(backend, os.environ) == 0.0

    out = seg.maybe_transcribe_segmented(backend, _make_wav(30.0), language="zh")

    assert out is None, "paraformer-like backend must keep the single-pass path"
    assert backend.calls == [], "segmenter must not call transcribe() itself"


def test_sensevoice_like_backend_still_segments(monkeypatch):
    monkeypatch.delenv(seg.ENV_MAX_SEGMENT_SECONDS, raising=False)
    backend = _CountingASRBackend()  # name="counting-fake" -> not in the deny-list

    assert seg.backend_chunks_internally(backend) is False
    assert seg.resolve_max_segment_seconds(backend, os.environ) == (
        seg.DEFAULT_MAX_SEGMENT_SECONDS
    )

    out = seg.maybe_transcribe_segmented(backend, _make_wav(30.0), language="zh")

    assert out is not None and out.segments_decoded > 1


def test_env_override_forces_segmentation_on_internal_chunking_backend(monkeypatch):
    """The operator escape hatch still wins over the deny-list."""
    monkeypatch.setenv(seg.ENV_MAX_SEGMENT_SECONDS, "10.5")
    backend = _ParaformerLikeBackend()

    assert seg.resolve_max_segment_seconds(backend, os.environ) == 10.5

    out = seg.maybe_transcribe_segmented(backend, _make_wav(30.0), language="zh")

    assert out is not None and out.segments_decoded > 1


def test_declared_limit_wins_over_deny_list(monkeypatch):
    monkeypatch.delenv(seg.ENV_MAX_SEGMENT_SECONDS, raising=False)
    backend = _ParaformerLikeBackend()
    backend.max_segment_seconds = 8.0

    assert seg.resolve_max_segment_seconds(backend, os.environ) == 8.0
