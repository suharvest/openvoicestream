"""Unit tests for KokoroTRTBackend.unload() release ordering + hot-reload flag.

Mirrors test_matcha_trt_unload.py — kokoro applies the same release pattern
(sync → ctx → engine → ORT → pool.destroy → gc) but has more engine/context
pairs due to the split-generator + hybrid architecture.

Hardware (VRAM) verification on orin-nano is a separate follow-up.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend


def _make_loaded_backend(parent: MagicMock) -> KokoroTRTBackend:
    """Build an instance that looks 'loaded' with mock TRT/ORT/pool objects."""
    b = KokoroTRTBackend()
    b._ready = True
    b._ort_sess = MagicMock(name="ort_sess")
    b._suffix_sess = MagicMock(name="suffix_sess")
    b._split_length_sess = MagicMock(name="split_length_sess")
    b._split_source_sess = MagicMock(name="split_source_sess")
    b._split_istft_sess = MagicMock(name="split_istft_sess")

    pool = MagicMock(name="pool")
    parent.attach_mock(pool, "pool")
    b._pool = pool

    # Main engine + ctx (hybrid mode).
    main_engine = MagicMock(name="main_engine")
    main_ctx = MagicMock(name="main_ctx")
    parent.attach_mock(main_engine, "main_engine")
    parent.attach_mock(main_ctx, "main_ctx")
    b._engine = main_engine
    b._ctx = main_ctx

    # Split engines / ctxs (encoder/decoder/source/generator).
    names = ("encoder", "decoder", "source", "generator")
    split_engines = {}
    split_ctxs = {}
    for name in names:
        eng = MagicMock(name=f"split_eng_{name}")
        ctx = MagicMock(name=f"split_ctx_{name}")
        parent.attach_mock(eng, f"split_eng_{name}")
        parent.attach_mock(ctx, f"split_ctx_{name}")
        split_engines[name] = eng
        split_ctxs[name] = ctx
    b._split_engines = split_engines
    b._split_ctxs = split_ctxs

    # Long-bucket engines + ctxs.
    long_engines = {}
    long_ctxs = {}
    for name in ("decoder", "source", "generator"):
        eng = MagicMock(name=f"long_eng_{name}")
        ctx = MagicMock(name=f"long_ctx_{name}")
        parent.attach_mock(eng, f"long_eng_{name}")
        parent.attach_mock(ctx, f"long_ctx_{name}")
        long_engines[name] = eng
        long_ctxs[name] = ctx
    b._split_long_engines = long_engines
    b._split_long_ctxs = long_ctxs

    b._token_to_id = {"a": 1, "b": 2}
    return b


def test_supports_hot_reload_is_true():
    b = KokoroTRTBackend()
    assert b.supports_hot_reload is True


def test_unload_release_order():
    """pool.synchronize must precede pool.destroy; both called exactly once."""
    parent = MagicMock(name="parent")
    b = _make_loaded_backend(parent)

    b.unload()

    method_names = [c[0] for c in parent.mock_calls if c[0].startswith("pool.")]
    assert "pool.synchronize" in method_names
    assert "pool.destroy" in method_names
    assert method_names.index("pool.synchronize") < method_names.index("pool.destroy")

    # Final state: refs cleared and not-ready.
    assert b._ready is False
    assert b._engine is None
    assert b._ctx is None
    assert b._pool is None
    assert b._ort_sess is None
    assert b._suffix_sess is None
    assert b._split_length_sess is None
    assert b._split_source_sess is None
    assert b._split_istft_sess is None
    assert b._split_engines == {}
    assert b._split_ctxs == {}
    assert b._split_long_engines == {}
    assert b._split_long_ctxs == {}
    assert b._token_to_id == {}


def test_unload_idempotent_fresh_instance():
    """Two unload() calls on a fresh (never-preloaded) instance: both no-throw."""
    b = KokoroTRTBackend()
    b.unload()
    b.unload()
    assert b._ready is False


def test_unload_idempotent_after_teardown():
    """After a loaded-instance teardown, calling unload again should no-op."""
    parent = MagicMock(name="parent")
    b = _make_loaded_backend(parent)
    b.unload()
    b.unload()  # early-return path.
    assert b._ready is False


def test_unload_continues_on_stage_failure():
    """pool.synchronize raises → engines still cleared, pool.destroy still called."""
    parent = MagicMock(name="parent")
    b = _make_loaded_backend(parent)
    b._pool.synchronize.side_effect = RuntimeError("simulated sync failure")

    b.unload()

    method_names = [c[0] for c in parent.mock_calls]
    assert "pool.synchronize" in method_names
    assert "pool.destroy" in method_names

    assert b._ready is False
    assert b._split_engines == {}
    assert b._split_long_engines == {}
    assert b._engine is None
    assert b._pool is None
