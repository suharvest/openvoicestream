"""Unit tests for the device-resident-chain sync optimization in kokoro_trt.

Covers:
1. ``_run_trt_context(..., sync=False, return_device=True)`` does NOT call
   ``pool.synchronize()`` (kernels remain in-flight on the CUDA stream).
2. ``_run_trt_context(..., sync=True)`` always calls ``pool.synchronize()``.
3. ``_run_trt_context(..., sync=False, return_device=False)`` MUST still sync
   before copy_dtoh — safety override.
4. ``_KokoroCtxSlot.reset_per_request()`` calls ``pool.synchronize()`` BEFORE
   ``pool.free_all()`` (defensive sync against in-flight kernels writing
   arena memory that's about to be returned).
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend, _KokoroCtxSlot


def _make_engine_with_one_output(name: str, shape: tuple, dtype):
    """A mocked TRT engine reporting a single output tensor.

    NB: we never exercise the legacy ``meta is None`` fallback (which would
    require importing tensorrt). Tests always pass a real ``_TrtEngineMeta``.
    """
    engine = MagicMock(name="engine")
    return engine


def _make_meta(name: str, dtype):
    from voxedge.backends.jetson.kokoro_trt import _TrtEngineMeta, _TrtOutputMeta

    return _TrtEngineMeta(outputs=(_TrtOutputMeta(name=name, dtype=dtype),))


def _make_pool():
    pool = MagicMock(name="pool")
    pool.allocate.side_effect = lambda nbytes: 0x1000  # bogus device pointer
    pool.stream_handle.return_value = 0
    return pool


def _make_ctx_with_shape(shape: tuple):
    ctx = MagicMock(name="ctx")
    ctx.get_tensor_shape.return_value = shape
    ctx.set_input_shape.return_value = True
    ctx.execute_async_v3.return_value = True
    return ctx


def test_run_trt_context_sync_false_device_return_skips_synchronize():
    """sync=False AND return_device=True → no pool.synchronize() call."""
    b = KokoroTRTBackend()
    pool = _make_pool()
    out_name = "device_out"
    out_shape = (1, 4)
    engine = _make_engine_with_one_output(out_name, out_shape, np.float32)
    ctx = _make_ctx_with_shape(out_shape)

    result = b._run_trt_context(
        engine, ctx,
        inputs={},
        pool=pool,
        return_device=True,
        sync=False,
        meta=_make_meta(out_name, np.float32),
    )

    assert pool.synchronize.call_count == 0, (
        "device-resident hop must NOT block CPU on pool.synchronize"
    )
    assert pool.copy_dtoh.call_count == 0
    assert out_name in result


def test_run_trt_context_sync_true_synchronizes():
    """sync=True → pool.synchronize() must be called exactly once."""
    b = KokoroTRTBackend()
    pool = _make_pool()
    out_name = "host_out"
    out_shape = (1, 4)
    engine = _make_engine_with_one_output(out_name, out_shape, np.float32)
    ctx = _make_ctx_with_shape(out_shape)

    b._run_trt_context(
        engine, ctx,
        inputs={},
        pool=pool,
        return_device=False,
        sync=True,
        meta=_make_meta(out_name, np.float32),
    )

    assert pool.synchronize.call_count == 1
    assert pool.copy_dtoh.call_count == 1


def test_run_trt_context_sync_false_host_return_still_syncs():
    """Safety override: sync=False + host output MUST sync before copy_dtoh."""
    b = KokoroTRTBackend()
    pool = _make_pool()
    out_name = "host_out"
    out_shape = (1, 4)
    engine = _make_engine_with_one_output(out_name, out_shape, np.float32)
    ctx = _make_ctx_with_shape(out_shape)

    b._run_trt_context(
        engine, ctx,
        inputs={},
        pool=pool,
        return_device=False,
        sync=False,  # ignored — host return forces sync
        meta=_make_meta(out_name, np.float32),
    )

    assert pool.synchronize.call_count == 1
    assert pool.copy_dtoh.call_count == 1


def test_reset_per_request_synchronizes_before_free_all(monkeypatch):
    """defensive sync: pool.synchronize must precede pool.free_all."""
    # Build a slot without invoking __init__ (which would try to construct a
    # real CudaMemoryPool). We just need the reset_per_request method.
    slot = _KokoroCtxSlot.__new__(_KokoroCtxSlot)
    parent = MagicMock(name="parent")
    pool = MagicMock(name="pool")
    parent.attach_mock(pool, "pool")
    slot.pool = pool

    slot.reset_per_request()

    method_names = [c[0] for c in parent.mock_calls if c[0].startswith("pool.")]
    assert "pool.synchronize" in method_names
    assert "pool.free_all" in method_names
    assert method_names.index("pool.synchronize") < method_names.index("pool.free_all"), (
        f"synchronize must precede free_all but got order: {method_names}"
    )


def test_reset_per_request_continues_when_synchronize_raises():
    """sync failure must NOT prevent free_all (we still need to reset arena)."""
    slot = _KokoroCtxSlot.__new__(_KokoroCtxSlot)
    pool = MagicMock(name="pool")
    pool.synchronize.side_effect = RuntimeError("simulated cuda sync failure")
    slot.pool = pool

    slot.reset_per_request()  # must not raise

    assert pool.synchronize.call_count == 1
    assert pool.free_all.call_count == 1
