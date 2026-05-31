"""Unit tests for the kokoro ORT-IO-name + TRT-meta caches (perf micro-opts).

We can't drive a real ORT/TRT session on the dev Mac, so the tests use small
fakes that just track ``get_inputs`` / ``get_outputs`` / ``run`` call counts.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from voxedge.backends.jetson.kokoro_trt import _OrtIoNames, _run_cpu_onnx


class _FakeNamed:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    """Minimal stand-in for an onnxruntime InferenceSession."""

    def __init__(self, inputs, outputs):
        self._inputs = [_FakeNamed(n) for n in inputs]
        self._outputs = [_FakeNamed(n) for n in outputs]
        self.get_inputs_calls = 0
        self.get_outputs_calls = 0
        self.run_calls = []  # list[(output_names_tuple, feeds_keys_tuple)]

    def get_inputs(self):
        self.get_inputs_calls += 1
        return list(self._inputs)

    def get_outputs(self):
        self.get_outputs_calls += 1
        return list(self._outputs)

    def run(self, output_names, feeds):
        self.run_calls.append((tuple(output_names), tuple(sorted(feeds.keys()))))
        return [np.array([0], dtype=np.float32) for _ in output_names]


def test_run_cpu_onnx_no_cache_falls_back_to_session_metadata():
    sess = _FakeSession(["x", "y"], ["out0", "out1"])
    out = _run_cpu_onnx(sess, {"x": np.zeros(1), "y": np.zeros(1), "extra": 9})
    # Without cache → must call get_inputs + get_outputs (one of each).
    assert sess.get_inputs_calls == 1
    assert sess.get_outputs_calls == 1
    # Extra feed key was filtered out.
    assert sess.run_calls == [(("out0", "out1"), ("x", "y"))]
    assert set(out.keys()) == {"out0", "out1"}


def test_run_cpu_onnx_with_cache_skips_session_metadata():
    sess = _FakeSession(["x", "y"], ["out0", "out1"])
    cache = _OrtIoNames(frozenset(["x", "y"]), ("out0", "out1"))
    out = _run_cpu_onnx(sess, {"x": np.zeros(1), "y": np.zeros(1), "extra": 9}, io_names=cache)
    # With cache → must NOT touch session metadata.
    assert sess.get_inputs_calls == 0
    assert sess.get_outputs_calls == 0
    # Run still receives the filtered feeds in the cached output order.
    assert sess.run_calls == [(("out0", "out1"), ("x", "y"))]
    assert set(out.keys()) == {"out0", "out1"}


def test_run_cpu_onnx_cache_filters_extra_inputs():
    """Extra feed keys (e.g. pass-through stage dict) are dropped per cache."""
    sess = _FakeSession(["x"], ["z"])
    cache = _OrtIoNames(frozenset(["x"]), ("z",))
    _run_cpu_onnx(sess, {"x": 1, "noise": 2, "speed": 3}, io_names=cache)
    assert sess.run_calls == [(("z",), ("x",))]
