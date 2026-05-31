"""Regression test for codex Week 3 BLOCKER 2.

The TTS N=2 stability gate (`bench/perf/stability_tts_n2_common.py`) used
to only emit a `WARNING` when the detected backend didn't match the
gate's expected backend substring, then proceeded to run and could
report ``pass=True``. That meant a kokoro-labelled gate run against a
qwen3 server would falsely PASS without ever exercising the kokoro code
path.

The fix: backend mismatch → return a failure report with reason
``backend_mismatch: ...`` so ``main_entry`` exits with code 2.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# The stability gate module imports `requests` at module load. Skip the
# entire file when it isn't available (e.g. minimal test environments
# without bench-time deps installed).
pytest.importorskip("requests")

from bench.perf import stability_tts_n2_common as gate_mod  # noqa: E402


def _make_cfg(tmp_path: Path, expected: tuple[str, ...]) -> gate_mod.GateConfig:
    return gate_mod.GateConfig(
        backend_label="kokoro_trt",
        expected_backend_substr=expected,
        fallback_env_var="OVS_TTS_STREAM_MAX_WORKERS_KOKORO",
        prompt_lang="zh",
        base_url="http://test.invalid",
        bursts=1,
        warmup=1,
        timeout=5.0,
        fail_on_ratio=1.5,
        output_dir=tmp_path,
        prompt_id=None,
        category=None,
        scan_log=None,
        container=None,
    )


def test_backend_mismatch_fails_gate(tmp_path, monkeypatch):
    """Expected kokoro_trt but the server reports qwen3_trt → pass=False."""
    cfg = _make_cfg(tmp_path, expected=("kokoro_trt",))

    # Stub the health probe to advertise a qwen3 backend.
    monkeypatch.setattr(
        gate_mod, "fetch_backend_info",
        lambda _url: {"health": {"tts_backend": "qwen3_trt"}},
    )

    # Make sure no real HTTP / prompt loads happen if the early return
    # isn't honored — they would error since the URL is invalid.
    monkeypatch.setattr(
        gate_mod, "load_prompts",
        lambda lang=None, category=None: [{"id": "p1", "text": "你好"}],
    )

    report = gate_mod.run_gate(cfg)
    assert report["pass"] is False, (
        f"expected pass=False on backend mismatch, got {report['pass']}; "
        f"failure_reasons={report.get('failure_reasons')}"
    )
    reasons = report.get("failure_reasons") or []
    assert any("backend_mismatch" in r for r in reasons), (
        f"expected a backend_mismatch reason, got {reasons}"
    )


def test_backend_match_does_not_short_circuit(tmp_path, monkeypatch):
    """When backend substring matches, the gate proceeds (does NOT trigger
    the new backend_mismatch fast-fail path)."""
    cfg = _make_cfg(tmp_path, expected=("kokoro_trt",))

    monkeypatch.setattr(
        gate_mod, "fetch_backend_info",
        lambda _url: {"health": {"tts_backend": "jetson.kokoro_trt.fp16"}},
    )

    # Replace prompt + post helpers with stubs so we don't hit the network.
    monkeypatch.setattr(
        gate_mod, "load_prompts",
        lambda lang=None, category=None: [{"id": "p1", "text": "hello"}],
    )

    # Stale-test fix: TtsCallResult gained fields (status, total_ms,
    # bytes_total, started_at, ...) since this stub was written. The gate's
    # warmup loop now reads r.status, so the stub must mirror the current
    # dataclass shape to avoid AttributeError before the backend-check
    # assertion is reached.
    class _FakeResp:
        text = "hello"
        prompt_id = "p1"
        started_at = 0.0
        error = "stub: bypassing further execution"
        ttfa_ms = None
        total_ms = 0.0
        status = 0
        bytes_total = 0
        pcm_present = False
        body = b""

    monkeypatch.setattr(
        gate_mod, "post_tts_stream",
        lambda *a, **kw: _FakeResp(),
    )

    # The gate should NOT report backend_mismatch — it will likely fail
    # for other reasons (no real TTFA samples) but that proves the
    # backend check did not short-circuit.
    report = gate_mod.run_gate(cfg)
    reasons = report.get("failure_reasons") or []
    assert not any("backend_mismatch" in r for r in reasons), (
        f"unexpected backend_mismatch on matching backend: {reasons}"
    )


def test_main_entry_returns_exit_code_2_on_mismatch(tmp_path, monkeypatch):
    """`main_entry` returns 2 when run_gate produces a failing report."""
    import sys

    argv = [
        "prog",
        "--base-url", "http://test.invalid",
        "--bursts", "1",
        "--warmup", "1",
        "--timeout", "5",
        "--output-dir", str(tmp_path),
        "--lang", "zh",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    monkeypatch.setattr(
        gate_mod, "fetch_backend_info",
        lambda _url: {"health": {"tts_backend": "qwen3_trt"}},
    )
    monkeypatch.setattr(
        gate_mod, "load_prompts",
        lambda lang=None, category=None: [{"id": "p1", "text": "你好"}],
    )

    rc = gate_mod.main_entry(
        backend_label="kokoro_trt",
        expected_backend_substr=("kokoro_trt",),
        fallback_env_var="OVS_TTS_STREAM_MAX_WORKERS_KOKORO",
        default_lang="zh",
    )
    assert rc == 2, f"expected exit code 2 on backend mismatch, got {rc}"
