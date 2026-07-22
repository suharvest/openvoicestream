"""Unit test for _TrtSession multi-input binding/feed logic (no real TRT).

The native-TRT path is the blocker for the embin detector (two inputs:
``images`` + ``class_embeddings``). We can't run TensorRT on Mac, so we
unit-test the pure binding/feed bookkeeping by mocking the engine + the cudart
shim: every declared engine input must be (a) tracked and (b) memcpy'd to its
device binding on run(), and a missing feed must raise.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.perception import yolo_onnx


class _FakeTensorMode:
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class _FakeLogger:
    WARNING = 0

    def __init__(self, *_a, **_k):
        pass


class _FakeContext:
    def __init__(self):
        self.addresses = {}
        self.executed = False

    def set_tensor_address(self, name, addr):
        self.addresses[name] = addr

    def execute_async_v3(self, stream_handle):  # noqa: ARG002
        self.executed = True
        return True


class _FakeEngine:
    """Two inputs (images, class_embeddings) + two outputs (output0, output1)."""

    _IO = [
        ("images", (1, 3, 4, 4), _FakeTensorMode.INPUT),
        ("class_embeddings", (1, 16, 8), _FakeTensorMode.INPUT),
        ("output0", (1, 5, 6), _FakeTensorMode.OUTPUT),
        ("output1", (1, 2, 3, 3), _FakeTensorMode.OUTPUT),
    ]

    @property
    def num_io_tensors(self):
        return len(self._IO)

    def get_tensor_name(self, i):
        return self._IO[i][0]

    def get_tensor_shape(self, name):
        return next(s for n, s, _ in self._IO if n == name)

    def get_tensor_dtype(self, name):  # noqa: ARG002
        return "float32"

    def get_tensor_mode(self, name):
        return next(m for n, _, m in self._IO if n == name)

    def create_execution_context(self):
        return _FakeContext()


class _FakeRuntime:
    def __init__(self, _logger):
        pass

    def deserialize_cuda_engine(self, _blob):
        return _FakeEngine()


def _install_fake_trt(monkeypatch):
    fake_trt = types.ModuleType("tensorrt")
    fake_trt.Logger = _FakeLogger
    fake_trt.Runtime = _FakeRuntime
    fake_trt.TensorIOMode = _FakeTensorMode
    fake_trt.init_libnvinfer_plugins = lambda *_a, **_k: None
    fake_trt.nptype = lambda _dt: np.float32
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)


class _FakeCudart:
    """Records every H2D copy keyed by device pointer so we can assert all
    inputs were fed. malloc hands out monotonically increasing fake pointers."""

    H2D = 1
    D2H = 2

    def __init__(self):
        self._next = 1000
        self.h2d = {}  # ptr -> nbytes copied
        self.ptr_for = {}

    def malloc(self, nbytes):
        p = types.SimpleNamespace(value=self._next)
        self._next += 1
        self.ptr_for[p.value] = nbytes
        return p

    def free(self, *_a):
        pass

    def memcpy(self, dst, src, nbytes, kind):  # noqa: ARG002
        if kind == self.H2D:
            self.h2d[getattr(dst, "value", dst)] = nbytes

    def stream_create(self):
        return types.SimpleNamespace(value=42)

    def stream_sync(self, *_a):
        pass


def _build_session(monkeypatch, tmp_path):
    _install_fake_trt(monkeypatch)
    monkeypatch.setattr(yolo_onnx, "_Cudart", _FakeCudart)
    engine_file = tmp_path / "fake.engine"
    engine_file.write_bytes(b"not-a-real-engine")
    return yolo_onnx._TrtSession(str(engine_file))


def test_all_inputs_tracked(monkeypatch, tmp_path):
    sess = _build_session(monkeypatch, tmp_path)
    assert set(sess.input_names) == {"images", "class_embeddings"}
    # first input is preserved for single-input callers
    assert sess.input_name == "images"
    # two outputs, ordered det (ndim 3) before proto (ndim 4)
    assert len(sess._outputs) == 2
    assert sess._outputs[0][0] == "output0"
    assert sess._outputs[1][0] == "output1"


def test_run_feeds_every_input(monkeypatch, tmp_path):
    sess = _build_session(monkeypatch, tmp_path)
    cu: _FakeCudart = sess._cu  # type: ignore[assignment]
    images = np.ones((1, 3, 4, 4), dtype=np.float32)
    embeds = np.ones((1, 16, 8), dtype=np.float32)
    outs = sess.run(None, {"images": images, "class_embeddings": embeds})
    # both input device pointers received an H2D copy of the right size
    img_ptr = sess._inputs["images"][3].value
    emb_ptr = sess._inputs["class_embeddings"][3].value
    assert cu.h2d.get(img_ptr) == images.nbytes
    assert cu.h2d.get(emb_ptr) == embeds.nbytes
    # outputs come back as float32 in contract order
    assert len(outs) == 2
    assert outs[0].shape == (1, 5, 6)
    assert outs[1].shape == (1, 2, 3, 3)


def test_run_missing_feed_raises(monkeypatch, tmp_path):
    sess = _build_session(monkeypatch, tmp_path)
    images = np.ones((1, 3, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="missing feeds"):
        sess.run(None, {"images": images})  # class_embeddings omitted


def test_run_size_mismatch_raises(monkeypatch, tmp_path):
    sess = _build_session(monkeypatch, tmp_path)
    bad = np.ones((1, 3, 4, 5), dtype=np.float32)  # wrong nbytes
    embeds = np.ones((1, 16, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="size mismatch"):
        sess.run(None, {"images": bad, "class_embeddings": embeds})
