"""P0 actionable-failure UX: the resolver classifies each provisioning failure
into a stable F-code with a copy-pasteable remediation, collects ALL failures in
one pass, and surfaces them via resolve_all's raised report + get_last_report().

Spec: docs/specs/engine-host-coverage-and-builder-sidecar.md (internal).
"""
import pytest

from server.core import engine_resolver as er
from server.core import hf_artifacts


HOST = er.HostSignature("87", "10.3", "6.2", "12.6")  # sm87-trt10.3-jp6.2-cuda12.6


def _profile(tmp_path, n=1):
    return {
        "required_engines": [
            {
                "model_id": "demo",
                "engine_file": f"enc{i}.plan",
                "engine_path": str(tmp_path / f"enc{i}.plan"),  # does not exist → cache miss
                "env_var": f"ENC{i}",
                "required": True,
            }
            for i in range(n)
        ]
    }


def test_F1_uncovered_host_lists_supported_signatures(tmp_path, monkeypatch):
    # Manifest exists but only ships a bundle for a DIFFERENT host signature.
    monkeypatch.setattr(er, "detect_host_signature", lambda: HOST)
    monkeypatch.setattr(
        hf_artifacts, "fetch_manifest",
        lambda mid: {"files": {"engines/sm99-trt99.9-jp9.9-cuda99.9.tar.gz": {"sha256": "x"}}},
    )

    report = er.build_report(_profile(tmp_path))

    assert not report.ok
    assert len(report.failures) == 1
    f = report.failures[0]
    assert f.code == "F1"
    assert f.state == "FAILED:F1"
    # remediation names the detected host AND the supported one
    assert "sm87-trt10.3-jp6.2-cuda12.6" in f.remediation
    assert "sm99-trt99.9-jp9.9-cuda99.9" in f.remediation
    # human report renders the fix line
    text = er.format_report_text(report)
    assert "✗ enc0.plan [F1]" in text
    assert "→ fix:" in text


def test_F3_hf_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "detect_host_signature", lambda: HOST)

    def boom(mid):
        raise hf_artifacts.ArtifactError("HTTPSConnectionPool: connection timeout")

    monkeypatch.setattr(hf_artifacts, "fetch_manifest", boom)

    report = er.build_report(_profile(tmp_path))
    f = report.failures[0]
    assert f.code == "F3"
    assert "unreachable" in f.cause.lower()
    assert "hf download" in f.remediation  # offline pre-stage hint


def test_collect_all_reports_every_failure_not_just_first(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "detect_host_signature", lambda: HOST)
    monkeypatch.setattr(hf_artifacts, "fetch_manifest", lambda mid: {"files": {}})

    report = er.build_report(_profile(tmp_path, n=3))
    # all three classified, not just the first
    assert len(report.failures) == 3
    assert all(e.code == "F1" for e in report.failures)


def test_resolve_all_raises_EngineResolutionError_and_stashes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "_acquire_lock", lambda: -1)  # skip the shared-volume flock
    monkeypatch.setattr(er, "_release_lock", lambda fd: None)
    monkeypatch.setattr(er, "detect_host_signature", lambda: HOST)
    monkeypatch.setattr(hf_artifacts, "fetch_manifest", lambda mid: {"files": {}})

    with pytest.raises(er.EngineResolutionError) as ei:
        er.resolve_all(_profile(tmp_path))

    # it IS a RuntimeError (existing callers/tests rely on this)
    assert isinstance(ei.value, RuntimeError)
    # the message carries the actionable block
    assert "ENGINE PROVISIONING" in str(ei.value)
    assert "→ fix:" in str(ei.value)
    # report stashed for /readyz
    last = er.get_last_report()
    assert last is not None
    assert last.to_dict()["host_signature"] == "sm87-trt10.3-jp6.2-cuda12.6"
    assert last.to_dict()["ready"] is False


def test_resolve_all_returns_paths_when_all_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "_acquire_lock", lambda: -1)
    monkeypatch.setattr(er, "_release_lock", lambda fd: None)
    monkeypatch.setattr(er, "detect_host_signature", lambda: HOST)
    # pretend every engine resolves + is a cache hit
    monkeypatch.setattr(er, "_resolve_one", lambda spec, host, force_rebuild: None)
    monkeypatch.setattr(er, "_meta_matches", lambda path, host: True)

    resolved = er.resolve_all(_profile(tmp_path, n=2))
    assert set(resolved.keys()) == {"ENC0", "ENC1"}
    assert er.get_last_report().ok is True
