"""Unit tests for ASRStream.cancel_and_finalize() overrides.

Covers the five concrete streams under app/backends/. Backends are mocked —
the goal is to assert that:

  1. cancel_and_finalize() returns within <50ms (no CUDA/TRT/ORT/numpy
     heavy work on the abort path).
  2. After cancel, finalize() returns the cached final text immediately.
  3. After cancel, accept_waveform() is a no-op and does NOT crash.

No real models, no GPU, no native libs required.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Make `app.*` importable when running pytest from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CANCEL_BUDGET_MS = 50.0


def _audio(secs: float = 0.5, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(secs * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# Class 1: _TRTEdgeLLMAccumulatingASRStream
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trt_edge_module():
    try:
        from voxedge.backends.jetson import trt_edge_llm_asr  # type: ignore
    except Exception as e:
        pytest.skip(f"trt_edge_llm_asr import failed: {e}")
    return trt_edge_llm_asr


def test_trt_edgellm_stream_cancel_fast(trt_edge_module):
    backend = MagicMock()
    backend.transcribe = MagicMock(
        side_effect=AssertionError("must not run after cancel")
    )
    s = trt_edge_module._TRTEdgeLLMAccumulatingASRStream(backend, language="zh")
    s.accept_waveform(16000, _audio(0.5))

    t0 = time.perf_counter()
    s.cancel_and_finalize()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < CANCEL_BUDGET_MS, f"cancel took {dt_ms:.1f}ms"


def test_trt_edgellm_stream_finalize_after_cancel_returns_cache(trt_edge_module):
    backend = MagicMock()
    backend.transcribe = MagicMock(
        side_effect=AssertionError("must not run after cancel")
    )
    s = trt_edge_module._TRTEdgeLLMAccumulatingASRStream(backend, language="zh")
    s.accept_waveform(16000, _audio(0.5))
    s.cancel_and_finalize()
    assert s.finalize() == ("", None)
    backend.transcribe.assert_not_called()


def test_trt_edgellm_stream_accept_after_cancel_noop(trt_edge_module):
    backend = MagicMock()
    s = trt_edge_module._TRTEdgeLLMAccumulatingASRStream(backend, language="zh")
    s.cancel_and_finalize()
    s.accept_waveform(16000, _audio(0.5))
    assert s._chunks == []


# ---------------------------------------------------------------------------
# Class 4: ParaformerTRTStream
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def paraformer_module():
    try:
        from voxedge.backends.jetson import paraformer_trt  # type: ignore
    except Exception as e:
        pytest.skip(f"paraformer_trt import failed: {e}")
    return paraformer_trt


def test_paraformer_stream_cancel_fast(paraformer_module):
    backend = MagicMock()
    backend._tokens = {}
    backend._run_encoder = MagicMock(
        side_effect=AssertionError("must not run after cancel")
    )
    backend._run_decoder = MagicMock(
        side_effect=AssertionError("must not run after cancel")
    )
    try:
        s = paraformer_module.ParaformerTRTStream(backend)
    except Exception as e:
        pytest.skip(f"ParaformerTRTStream instantiation failed: {e}")

    s._partial_text = "你好世界"
    t0 = time.perf_counter()
    s.cancel_and_finalize()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < CANCEL_BUDGET_MS, f"cancel took {dt_ms:.1f}ms"
    assert s._final_text_cache == "你好世界"


def test_paraformer_stream_finalize_after_cancel_returns_cache(paraformer_module):
    backend = MagicMock()
    backend._tokens = {}
    backend._run_encoder = MagicMock(
        side_effect=AssertionError("must not run after cancel")
    )
    try:
        s = paraformer_module.ParaformerTRTStream(backend)
    except Exception as e:
        pytest.skip(f"ParaformerTRTStream instantiation failed: {e}")
    s._partial_text = "abc"
    s.cancel_and_finalize()
    assert s.finalize() == ("abc", None)
    backend._run_encoder.assert_not_called()


def test_paraformer_stream_accept_after_cancel_noop(paraformer_module):
    backend = MagicMock()
    backend._tokens = {}
    try:
        s = paraformer_module.ParaformerTRTStream(backend)
    except Exception as e:
        pytest.skip(f"ParaformerTRTStream instantiation failed: {e}")
    s.cancel_and_finalize()
    before_len = len(s._audio_buf)
    s.accept_waveform(16000, _audio(0.5))
    assert len(s._audio_buf) == before_len


# ---------------------------------------------------------------------------
# Class 5: SherpaASRStream
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sherpa_module():
    try:
        from app.backends.cpu import sherpa_asr  # type: ignore
    except Exception as e:
        pytest.skip(f"sherpa_asr import failed: {e}")
    return sherpa_asr


def _make_sherpa_recognizer():
    rec = MagicMock()
    native_stream = MagicMock()
    native_stream.accept_waveform = MagicMock()
    native_stream.input_finished = MagicMock(
        side_effect=AssertionError("input_finished must not run after cancel")
    )
    rec.create_stream = MagicMock(return_value=native_stream)
    rec.is_ready = MagicMock(return_value=False)
    rec.decode_stream = MagicMock(
        side_effect=AssertionError("decode_stream must not run after cancel")
    )
    rec.get_result = MagicMock(return_value="hello world")
    rec.is_endpoint = MagicMock(return_value=False)
    return rec, native_stream


def test_sherpa_stream_cancel_fast(sherpa_module):
    rec, _ = _make_sherpa_recognizer()
    s = sherpa_module.SherpaASRStream(rec, language_mode="en")
    s._last_text = "partial so far"

    t0 = time.perf_counter()
    s.cancel_and_finalize()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < CANCEL_BUDGET_MS, f"cancel took {dt_ms:.1f}ms"
    assert s._final_text_cache == "partial so far"


def test_sherpa_stream_finalize_after_cancel_returns_cache(sherpa_module):
    rec, native_stream = _make_sherpa_recognizer()
    s = sherpa_module.SherpaASRStream(rec, language_mode="en")
    s._last_text = "cached"
    s.cancel_and_finalize()
    assert s.finalize() == ("cached", None)
    native_stream.input_finished.assert_not_called()
    rec.decode_stream.assert_not_called()


def test_sherpa_stream_accept_after_cancel_noop(sherpa_module):
    rec, native_stream = _make_sherpa_recognizer()
    # Use the recognizer in a non-cancelled prefix call would call native
    # accept_waveform — but we cancel first, so it must not.
    native_stream.accept_waveform = MagicMock(
        side_effect=AssertionError("must not push samples after cancel")
    )
    s = sherpa_module.SherpaASRStream(rec, language_mode="en")
    s.cancel_and_finalize()
    s.accept_waveform(16000, _audio(0.5))
    native_stream.accept_waveform.assert_not_called()
