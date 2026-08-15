from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.perf.qwen35_mtp_profile_gate import (
    ValidationError,
    evaluate,
    main,
    parse_profile,
)


def _profile(
    scale: float = 1.0,
    *,
    include_official: bool = True,
    include_extra: bool = True,
) -> dict:
    payload = {
        "schema_version": 1,
        "stages": {
            "draft_prefill": {"tokens": 100, "duration_ms": 100.0 * scale},
            "draft_proposal": {"tokens": 30, "duration_ms": 50.0 * scale},
            "base_verification": {"tokens": 40, "duration_ms": 50.0 * scale},
        },
    }
    if include_extra:
        payload["stages"]["spec_decode_draft_accept"] = {
            "tokens": 30,
            "duration_ms": 25.0 * scale,
        }
    if include_official:
        # Deliberately differs from the compatible-stage result.  The gate
        # must report this field without using it for the ratio.
        payload["throughput_tokens_per_s"] = 123.0 / scale
    return payload


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_compatible_stage_gate_passes_and_excludes_v010_accept_stage(tmp_path: Path):
    baseline = _write(tmp_path, "baseline.json", _profile(scale=1.0))
    candidate = _write(tmp_path, "candidate.json", _profile(scale=1.0))
    wall = _write(
        tmp_path,
        "wall.json",
        {"comparison": {"wall_v010_over_v091": 1.04}},
    )

    report = evaluate([baseline], [candidate], wall_summary=wall)

    assert report["passed"] is True
    assert report["comparison"]["candidate_over_baseline"] == pytest.approx(1.0)
    assert report["comparison"]["compatible_stage_gate"]["passed"] is True
    assert report["comparison"]["wall_gate"]["passed"] is True
    assert report["baseline"]["official_fields"]["throughput_tokens_per_s"][
        "median"
    ] == pytest.approx(123.0)
    assert report["metric_contract"]["excluded_official_stage"] == (
        "spec_decode_draft_accept"
    )


def test_ratio_gate_fails_using_compatible_stage_medians(tmp_path: Path):
    baseline = _write(tmp_path, "baseline.json", _profile(scale=1.0))
    candidate = _write(tmp_path, "candidate.json", _profile(scale=1.1))

    report = evaluate([baseline], [candidate])

    assert report["passed"] is False
    assert report["comparison"]["candidate_over_baseline"] == pytest.approx(1 / 1.1)
    assert report["comparison"]["candidate_over_baseline"] < 0.95
    assert report["comparison"]["compatible_stage_gate"]["passed"] is False


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p["stages"].pop("draft_proposal"), "missing stage"),
        (lambda p: p.update(schema_version=2), "schema_version"),
    ],
)
def test_missing_stage_or_schema_fails_strictly(mutate, message):
    payload = _profile()
    mutate(payload)
    with pytest.raises(ValidationError, match=message):
        parse_profile(payload, source="fixture.json")


def test_wall_gate_fails_independently_of_throughput(tmp_path: Path):
    baseline = _write(tmp_path, "baseline.json", _profile())
    candidate = _write(tmp_path, "candidate.json", _profile())
    wall = _write(
        tmp_path,
        "wall.json",
        {"comparison": {"wall_v010_over_v091": 1.06}},
    )

    report = evaluate([baseline], [candidate], wall_summary=wall)

    assert report["comparison"]["compatible_stage_gate"]["passed"] is True
    assert report["comparison"]["wall_gate"]["passed"] is False
    assert report["passed"] is False


def test_cli_emits_structured_json_and_nonzero_for_gate_failure(tmp_path: Path, capsys):
    baseline = _write(tmp_path, "baseline.json", _profile())
    candidate = _write(tmp_path, "candidate.json", _profile(scale=1.1))

    rc = main(["--baseline", str(baseline), "--candidate", str(candidate)])
    captured = capsys.readouterr()

    assert rc == 1
    report = json.loads(captured.out)
    assert report["passed"] is False
    assert report["comparison"]["compatible_stage_gate"]["passed"] is False


def test_accepts_native_edge_llm_profiles_and_ignores_accept_stage(tmp_path: Path):
    def native(*, accept_ms: float | None) -> dict:
        stages = [
            {"stage_id": "spec_decode_base_verification", "total_gpu_time_ms": 150.0},
            {"stage_id": "spec_decode_draft_prefill", "total_gpu_time_ms": 6.0},
            {"stage_id": "spec_decode_draft_proposal", "total_gpu_time_ms": 34.0},
            {"stage_id": "llm_prefill", "total_gpu_time_ms": 65.0},
        ]
        if accept_ms is not None:
            stages.append(
                {"stage_id": "spec_decode_draft_accept", "total_gpu_time_ms": accept_ms}
            )
        return {
            "mtp_generation": {
                "total_generated_tokens": 9,
                "overall_tokens_per_second_excluding_base_prefill": 44.0,
            },
            "stages": stages,
        }

    baseline = _write(tmp_path, "v091.json", native(accept_ms=None))
    candidate = _write(tmp_path, "v010.json", native(accept_ms=12.0))
    report = evaluate([baseline], [candidate])

    assert report["passed"] is True
    assert report["comparison"]["candidate_over_baseline"] == pytest.approx(1.0)
    assert report["candidate"]["profiles"][0][
        "compatible_throughput_tokens_per_s"
    ] == pytest.approx(9 / 0.190)
