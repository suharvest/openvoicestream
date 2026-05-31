"""PR5 tests: per-backend unload() idempotence + supports_hot_reload flags.

These tests do NOT preload real models. They construct each backend, then
invoke ``unload()`` against the freshly-constructed object (no preload) and
again on the unloaded object. Both calls must be no-throw.

Many backends import native deps (TensorRT, cuda-python, sherpa-onnx,
rkvoice-stream). On mac / CPU dev machines those imports fail at module-load
time, so each backend's import is wrapped in try/except and the test
``pytest.skip``s when the backend can't be loaded.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ---------------------------------------------------------------------------
# Per-backend lazy importers — return class or None on ImportError/other
# ---------------------------------------------------------------------------

def _import_or_none(modpath: str, clsname: str):
    try:
        mod = __import__(modpath, fromlist=[clsname])
    except Exception as exc:
        pytest.skip(f"{modpath} unimportable here: {exc}")
        return None  # pragma: no cover
    try:
        return getattr(mod, clsname)
    except AttributeError as exc:
        pytest.skip(f"{modpath}.{clsname} missing: {exc}")
        return None  # pragma: no cover


# ---------------------------------------------------------------------------
# supports_hot_reload flag table
# ---------------------------------------------------------------------------

_FLAG_TABLE = [
    ("voxedge.backends.jetson.trt_edge_llm_tts", "TRTEdgeLLMTTSBackend", True),
    ("voxedge.backends.sherpa.tts",          "SherpaTTSBackend",     True),
    ("voxedge.backends.sherpa.asr",          "SherpaASRBackend",     True),
    ("voxedge.backends.jetson.kokoro_trt",       "KokoroTRTBackend",     True),
    ("voxedge.backends.jetson.matcha_trt",       "MatchaTRTBackend",     True),
    ("voxedge.backends.jetson.qwen3_trt",        "Qwen3TRTBackend",      False),
    ("voxedge.backends.rk.tts",              "RKTTSBackend",         False),
    ("voxedge.backends.rk.asr",              "RKASRBackend",         False),
]


@pytest.mark.parametrize("modpath,clsname,expected", _FLAG_TABLE)
def test_supports_hot_reload_flag(modpath, clsname, expected):
    """Verify each backend's supports_hot_reload flag.

    PR5b: TRTEdgeLLMTTSBackend.supports_hot_reload is an instance property
    (mode-dependent), so we check via instance attribute access. For other
    backends it remains a plain class attribute, which is also readable via
    instance attribute access.
    """
    cls = _import_or_none(modpath, clsname)
    assert cls is not None
    inst = _safe_construct(cls)
    got = inst.supports_hot_reload
    assert got is expected, (
        f"{clsname}.supports_hot_reload expected {expected}, got {got}"
    )


# ---------------------------------------------------------------------------
# Idempotent unload() — construct, unload, unload again
# ---------------------------------------------------------------------------

def _safe_construct(cls):
    """Try to instantiate the backend without preload. Skip if the bare
    ``__init__`` already wants a native module (e.g. RKTTSBackend calls
    rkvoice_stream.create_tts() in __init__)."""
    try:
        return cls()
    except Exception as exc:
        pytest.skip(f"{cls.__name__}() unconstructable here: {exc}")
        return None  # pragma: no cover


@pytest.mark.parametrize("modpath,clsname", [(m, c) for m, c, _ in _FLAG_TABLE])
def test_unload_idempotent_without_preload(modpath, clsname):
    """unload() on a freshly-constructed (never preloaded) backend must not raise.
    A second unload() must also be a no-op.
    """
    cls = _import_or_none(modpath, clsname)
    inst = _safe_construct(cls)
    # First unload — no preload, should early-return.
    inst.unload()
    # Second unload — idempotent.
    inst.unload()
    # is_ready() must not raise after unload. PR5b: previously this harness
    # swallowed AttributeError, masking FIX_2/FIX_3 (RK backends crashed
    # because is_ready() dereferenced None _inner). Assert directly.
    ready = inst.is_ready()
    assert ready is False or ready == 0 or ready is None


def test_sherpa_tts_unload_after_fake_ready():
    """Force ``_ready=True`` to exercise the non-early-return path."""
    cls = _import_or_none("voxedge.backends.sherpa.tts", "SherpaTTSBackend")
    inst = _safe_construct(cls)
    # Pretend preload succeeded (without actually loading sherpa-onnx).
    inst._ready = True
    inst.unload()
    assert inst.is_ready() is False
    # Idempotent.
    inst.unload()
    assert inst.is_ready() is False


def test_trt_edgellm_tts_unload_with_no_worker():
    """TRTEdgeLLMTTSBackend.unload() with _ready=True but no worker shouldn't crash."""
    cls = _import_or_none(
        "voxedge.backends.jetson.trt_edge_llm_tts", "TRTEdgeLLMTTSBackend"
    )
    inst = _safe_construct(cls)
    inst._ready = True
    inst._worker = None  # explicitly no subprocess
    inst.unload()
    assert inst.is_ready() is False
    inst.unload()  # idempotent


def test_trtedgellm_supports_hot_reload_depends_on_mode():
    """PR5b FIX_1: supports_hot_reload must reflect the resolved mode.

    edgellm_worker / official → True (subprocess can be killed).
    product_explicit_kv / explicit_kv → False (embeds in-process Qwen3 TRT).
    """
    cls = _import_or_none(
        "voxedge.backends.jetson.trt_edge_llm_tts", "TRTEdgeLLMTTSBackend"
    )
    inst = _safe_construct(cls)

    inst._resolved_mode = "edgellm_worker"
    assert inst.supports_hot_reload is True

    inst._resolved_mode = "official"
    assert inst.supports_hot_reload is True

    inst._resolved_mode = "product_explicit_kv"
    assert inst.supports_hot_reload is False

    inst._resolved_mode = "explicit_kv"
    assert inst.supports_hot_reload is False


def test_trtedgellm_unload_calls_product_backend_unload():
    """PR5b FIX_1: when product_explicit_kv mode is active, unload() must
    invoke the embedded Qwen3 backend's unload() before discarding it."""
    cls = _import_or_none(
        "voxedge.backends.jetson.trt_edge_llm_tts", "TRTEdgeLLMTTSBackend"
    )
    inst = _safe_construct(cls)

    class _FakeProductBackend:
        def __init__(self) -> None:
            self.unload_calls = 0

        def unload(self) -> None:
            self.unload_calls += 1

    fake = _FakeProductBackend()
    inst._ready = True
    inst._worker = None
    inst._product_backend = fake
    inst._resolved_mode = "product_explicit_kv"

    inst.unload()
    assert fake.unload_calls == 1
    assert inst._product_backend is None
    assert inst.is_ready() is False
    # Idempotent: second unload is a no-op (early-return), unload_calls unchanged.
    inst.unload()
    assert fake.unload_calls == 1


def test_trtedgellm_unload_with_only_product_backend():
    """PR5c FIX_1: when a preload fails halfway and leaves the resident
    backend with ``_ready=False``, ``_worker=None`` but a live
    ``_product_backend``, ``unload()`` must still invoke the embedded
    backend's unload() instead of early-returning and leaking its GPU
    memory across profile swaps.
    """
    cls = _import_or_none(
        "voxedge.backends.jetson.trt_edge_llm_tts", "TRTEdgeLLMTTSBackend"
    )
    inst = _safe_construct(cls)

    class _FakeProductBackend:
        def __init__(self) -> None:
            self.unload_calls = 0

        def unload(self) -> None:
            self.unload_calls += 1

    fake = _FakeProductBackend()
    inst._ready = False
    inst._worker = None
    inst._product_backend = fake
    inst._resolved_mode = "product_explicit_kv"

    inst.unload()
    assert fake.unload_calls == 1, "product_backend.unload was not called"
    assert inst._product_backend is None
    assert inst._resolved_mode is None
    assert inst.is_ready() is False
    # Idempotent: second unload now hits the all-empty early-return.
    inst.unload()
    assert fake.unload_calls == 1


def test_trtedgellm_product_backend_unload_raises_still_clears():
    """PR5d FIX_2: even if ``_product_backend.unload()`` raises, the
    ``finally`` block must still null ``_product_backend`` and
    ``_resolved_mode`` and flip ``_ready`` to False. Otherwise the next
    preload attempt would see a stale handle to a backend whose teardown
    failed and silently leak GPU memory.
    """
    import threading
    from unittest.mock import MagicMock

    cls = _import_or_none(
        "voxedge.backends.jetson.trt_edge_llm_tts", "TRTEdgeLLMTTSBackend"
    )
    backend = cls.__new__(cls)
    backend._ready = False
    backend._worker = None
    backend._worker_lock = threading.Lock()
    backend._worker_stderr_tail = __import__("collections").deque()
    backend._worker_ready_meta = {}
    backend._resolved_mode = "product_explicit_kv"

    fake_product = MagicMock()
    fake_product.unload.side_effect = RuntimeError(
        "simulated product unload failure"
    )
    backend._product_backend = fake_product

    # Must NOT raise — the unload() try/except inside the finally swallows it.
    backend.unload()

    fake_product.unload.assert_called_once()
    assert backend._product_backend is None
    assert backend._resolved_mode is None
    assert backend._ready is False


def test_rk_asr_stream_adapter_after_unload():
    """PR5d FIX_3: stream adapter on an RK ASR backend whose ``_inner`` was
    nulled must raise RuntimeError (not AttributeError) so callers can
    distinguish the unload race from a real bug.
    """
    from unittest.mock import MagicMock

    try:
        from voxedge.backends.rk.asr import RKASRBackend, _RKASRStreamAdapter
    except Exception as exc:
        pytest.skip(f"rkvoice_stream not available: {exc}")
        return  # pragma: no cover

    # Construct without __init__ to skip native dep load.
    backend = RKASRBackend.__new__(RKASRBackend)
    fake_inner_stream = MagicMock()
    backend._inner = MagicMock()

    adapter = _RKASRStreamAdapter(fake_inner_stream, backend, language="auto")

    # Simulate manager-driven unload: drop both the adapter's local handle
    # and the backend's inner handle.
    backend._inner = None
    adapter._inner = None

    import numpy as _np

    with pytest.raises(RuntimeError):
        adapter.accept_waveform(16000, _np.zeros(320, dtype=_np.float32))
    with pytest.raises(RuntimeError):
        adapter.finalize()
    with pytest.raises(RuntimeError):
        adapter.prepare_finalize()
    with pytest.raises(RuntimeError):
        adapter.cancel_and_finalize()
    with pytest.raises(RuntimeError):
        adapter.get_partial()


def test_kokoro_trt_unload_clears_engine_fields():
    cls = _import_or_none("voxedge.backends.jetson.kokoro_trt", "KokoroTRTBackend")
    inst = _safe_construct(cls)
    inst._ready = True
    inst._engine = object()
    inst._ctx = object()
    inst._split_engines = {"x": object()}
    inst.unload()
    assert inst._engine is None
    assert inst._ctx is None
    assert inst._split_engines == {}
    assert inst.is_ready() is False
    inst.unload()  # idempotent


def test_matcha_trt_unload_clears_fields():
    cls = _import_or_none("voxedge.backends.jetson.matcha_trt", "MatchaTRTBackend")
    inst = _safe_construct(cls)
    inst._ready = True
    inst._acoustic_ort = object()
    inst._vocos_engine = object()
    inst.unload()
    assert inst._acoustic_ort is None
    assert inst._vocos_engine is None
    assert inst.is_ready() is False
    inst.unload()


def test_qwen3_trt_unload_drops_engine_and_tokenizer():
    cls = _import_or_none("voxedge.backends.jetson.qwen3_trt", "Qwen3TRTBackend")
    inst = _safe_construct(cls)
    inst._ready = True
    inst._engine = object()
    inst._tokenizer = object()
    inst.unload()
    assert inst._engine is None
    assert inst._tokenizer is None
    assert inst.is_ready() is False
    inst.unload()
