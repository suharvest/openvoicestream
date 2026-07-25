"""Week 2 Prometheus metrics tests.

Asserts:
- new helpers emit Prometheus samples,
- backend state Gauge clears non-selected states,
- exposition is bytes with the Prometheus content type,
- Week 1 helpers still return integers and clamp at zero.
"""

from __future__ import annotations

from server.core import metrics


def setup_function(_func) -> None:
    metrics._reset_for_tests()


def _exposition() -> str:
    return metrics.render_prometheus().decode("utf-8")


def test_render_prometheus_returns_bytes_with_content_type():
    body = metrics.render_prometheus()
    assert isinstance(body, bytes)
    assert b"ovs_sessions_active" in body
    assert metrics.prometheus_content_type().startswith("text/plain")


def test_week1_helpers_still_return_int():
    assert metrics.inc_sessions_active() == 1
    assert metrics.inc_sessions_active() == 2
    assert metrics.dec_sessions_active() == 1
    assert metrics.dec_sessions_active() == 0
    # clamp at zero
    assert metrics.dec_sessions_active() == 0


def test_active_ws_sessions_gauge():
    assert metrics.inc_active_ws_sessions() == 1
    assert metrics.inc_active_ws_sessions() == 2
    assert metrics.dec_active_ws_sessions() == 1
    assert metrics.dec_active_ws_sessions() == 0
    assert metrics.dec_active_ws_sessions() == 0  # clamp
    body = _exposition()
    assert "ovs_active_ws_sessions 0.0" in body


def test_tts_ttfa_histogram_emits_buckets():
    metrics.record_tts_ttfa("mock", 0.12)
    body = _exposition()
    assert 'ovs_tts_ttfa_seconds_bucket{backend="mock"' in body
    assert 'ovs_tts_ttfa_seconds_count{backend="mock"} 1.0' in body


def test_tts_rtf_histogram_emits_buckets():
    metrics.record_tts_rtf("mock", 0.5)
    assert 'ovs_tts_rtf_bucket{backend="mock"' in _exposition()


def test_asr_decode_histogram_emits_buckets():
    metrics.record_asr_decode_duration("mock", 0.05)
    assert 'ovs_asr_decode_duration_seconds_bucket{backend="mock"' in _exposition()


def test_asr_cer_gauge():
    metrics.set_asr_cer("mock", 0.07)
    assert 'ovs_asr_cer{backend="mock"} 0.07' in _exposition()


def test_set_backend_state_clears_other_states():
    metrics.set_backend_state("tts", "ready")
    body = _exposition()
    assert 'ovs_backend_state{manager="tts",state="ready"} 1.0' in body
    assert 'ovs_backend_state{manager="tts",state="init"} 0.0' in body
    assert 'ovs_backend_state{manager="tts",state="draining"} 0.0' in body
    # Transition to draining clears ready.
    metrics.set_backend_state("tts", "draining")
    body = _exposition()
    assert 'ovs_backend_state{manager="tts",state="ready"} 0.0' in body
    assert 'ovs_backend_state{manager="tts",state="draining"} 1.0' in body


def test_set_backend_state_ignores_unknown_manager():
    metrics.set_backend_state("nope", "ready")
    metrics.set_backend_state("tts", "weirdstate")
    # No exception, no sample for unknown labels.
    body = _exposition()
    assert 'manager="nope"' not in body
    assert 'state="weirdstate"' not in body


def test_record_backend_reload_counter():
    metrics.record_backend_reload("success")
    metrics.record_backend_reload("fail")
    metrics.record_backend_reload("rollback")
    metrics.record_backend_reload("drain_timeout")
    body = _exposition()
    assert 'ovs_backend_reload_total{result="success"} 1.0' in body
    assert 'ovs_backend_reload_total{result="fail"} 1.0' in body
    assert 'ovs_backend_reload_total{result="rollback"} 1.0' in body
    assert 'ovs_backend_reload_total{result="drain_timeout"} 1.0' in body


def test_record_backend_reload_normalises_unknown():
    metrics.record_backend_reload("bogus")
    assert 'ovs_backend_reload_total{result="fail"} 1.0' in _exposition()


def test_record_worker_cancel():
    metrics.record_worker_cancel("tts", "client_abort")
    metrics.record_worker_cancel("asr", "bargein")
    body = _exposition()
    assert 'ovs_worker_cancels_total{backend="tts",reason="client_abort"} 1.0' in body
    assert 'ovs_worker_cancels_total{backend="asr",reason="bargein"} 1.0' in body


def test_set_queue_depth():
    metrics.set_queue_depth("tts_stream", 3)
    metrics.set_queue_depth("asr", 0)
    body = _exposition()
    assert 'ovs_queue_depth{queue="tts_stream"} 3.0' in body
    assert 'ovs_queue_depth{queue="asr"} 0.0' in body


def test_set_queue_depth_clamps_negative():
    metrics.set_queue_depth("tts_stream", -5)
    assert 'ovs_queue_depth{queue="tts_stream"} 0.0' in _exposition()


def test_invalid_histogram_values_are_dropped():
    metrics.record_tts_ttfa("mock", float("nan"))
    metrics.record_tts_ttfa("mock", float("inf"))
    metrics.record_tts_ttfa("mock", -1.0)
    body = _exposition()
    # No count incremented because all observations dropped.
    assert 'ovs_tts_ttfa_seconds_count{backend="mock"}' not in body


def test_session_rejected_counter_in_exposition():
    metrics.inc_sessions_rejected("http")
    metrics.inc_sessions_rejected("ws")
    metrics.inc_sessions_rejected("http")
    body = _exposition()
    assert 'ovs_sessions_rejected_total{reason="http"} 2.0' in body
    assert 'ovs_sessions_rejected_total{reason="ws"} 1.0' in body


def test_auth_rejected_counter_in_exposition():
    metrics.inc_auth_rejected("/tts")
    body = _exposition()
    assert 'ovs_auth_rejected_total{endpoint="/tts"} 1.0' in body


def test_watchdog_metrics_helpers():
    metrics.set_gpu_watchdog_ok(False)
    metrics.observe_gpu_watchdog_check_duration(0.012)
    metrics.record_gpu_watchdog_failure("jetson", "probe_timeout")
    body = _exposition()
    assert "ovs_gpu_watchdog_ok 0.0" in body
    assert "ovs_gpu_watchdog_check_duration_seconds_count 1.0" in body
    assert (
        'ovs_gpu_watchdog_failures_total{platform="jetson",reason="probe_timeout"} 1.0'
        in body
    )


def test_reset_for_tests_clears_counters_and_collectors():
    metrics.inc_sessions_active()
    metrics.record_tts_ttfa("mock", 0.2)
    metrics._reset_for_tests()
    body = _exposition()
    assert "ovs_sessions_active 0.0" in body
    assert 'ovs_tts_ttfa_seconds_count{backend="mock"}' not in body
    assert metrics.snapshot()["ovs_sessions_active"] == 0


def test_snapshot_preserves_week1_keys():
    metrics.inc_sessions_active()
    metrics.inc_sessions_rejected("http")
    metrics.inc_auth_rejected("/tts")
    snap = metrics.snapshot()
    assert snap["ovs_sessions_active"] == 1
    assert snap["ovs_sessions_rejected_total"] == {"http": 1}
    assert snap["ovs_auth_rejected_total"] == {"/tts": 1}
