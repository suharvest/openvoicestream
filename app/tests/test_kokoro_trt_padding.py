"""Unit tests for KokoroTRTBackend.synthesize() token padding policy.

Defensive padding ensures short inputs (e.g. "Hi", "OK") never trigger the
TRT-profile shape-mismatch fallback to CPU ORT (which leaks an untracked ORT
session per call). See `_synthesize_one` in app/backends/jetson/kokoro_trt.py.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend


def _build_backend(mode: str, *, fixed_len=None, min_len=None, max_len=128):
    b = KokoroTRTBackend()
    b._runtime_mode = mode
    b._hybrid_fixed_seq_len = fixed_len
    b._hybrid_min_seq_len = min_len
    b._hybrid_max_seq_len = max_len
    return b


def _patched_synth(backend, token_ids, capture):
    """Run _synthesize_one with token + downstream stubs and capture input_ids."""

    def fake_run_split_generator(input_ids, style, speed_arr, **_kwargs):
        # _kwargs absorbs pool / split_ctxs / split_long_ctxs added by the
        # per-call concurrency rework (2026-05-25).
        capture["input_ids"] = input_ids
        # Return a small fake audio buffer.
        return np.zeros(1024, dtype=np.float32)

    def fake_load_style(sid, n):
        # Return a small fake style tensor; shape doesn't matter for the mocked run.
        return np.zeros((1, 256), dtype=np.float32)

    with patch.object(backend, "_text_to_token_ids", return_value=token_ids), \
         patch.object(backend, "_run_split_generator", side_effect=fake_run_split_generator), \
         patch.object(backend, "_load_style", side_effect=fake_load_style):
        wav, meta = backend._synthesize_one("x")
    return wav, meta


def test_synthesize_pads_to_min_seq_when_below_floor():
    """Short input (1 token) should be padded to engine's min_seq (4)."""
    b = _build_backend("split_generator", fixed_len=None, min_len=4, max_len=256)
    capture: dict = {}
    _patched_synth(b, [5], capture)
    input_ids = capture["input_ids"]
    assert input_ids.shape[0] == 1
    assert input_ids.shape[1] >= 4, f"expected pad to >=4, got shape {input_ids.shape}"
    # First/last should still be BOS/EOS=0 around the real token.
    assert int(input_ids[0, 0]) == 0
    assert int(input_ids[0, 1]) == 5
    assert int(input_ids[0, 2]) == 0


def test_synthesize_no_padding_when_input_above_floor():
    """10-token input should NOT be padded (10+BOS+EOS = 12, above min=4)."""
    b = _build_backend("split_generator", fixed_len=None, min_len=4, max_len=256)
    capture: dict = {}
    _patched_synth(b, list(range(1, 11)), capture)
    input_ids = capture["input_ids"]
    assert input_ids.shape == (1, 12), f"expected (1,12), got {input_ids.shape}"


def test_synthesize_uses_fixed_seq_when_set():
    """Fixed-shape engine: short input padded to fixed length exactly."""
    b = _build_backend("split_generator", fixed_len=64, min_len=None, max_len=64)
    capture: dict = {}
    _patched_synth(b, [5], capture)
    input_ids = capture["input_ids"]
    assert input_ids.shape == (1, 64), f"expected (1,64), got {input_ids.shape}"
