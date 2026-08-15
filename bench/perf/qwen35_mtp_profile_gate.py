#!/usr/bin/env python3
"""Compare Qwen3.5 GDN+MTP profiles across the v0.9.1/v0.10 metric contract.

The Edge-LLM formatter changed its reported throughput in v0.10 by adding
``spec_decode_draft_accept`` to the timed stages.  That field is useful for
diagnostics, but is not comparable with the v0.9.1 reported field.  This gate
therefore recomputes a versioned throughput using only
``draft_prefill``, ``draft_proposal`` and ``base_verification``.

Input profile schema (version 1)::

    {
      "schema_version": 1,
      "stages": {
        "draft_prefill": {"tokens": 128, "duration_ms": 2.0},
        "draft_proposal": {"tokens": 3, "duration_ms": 1.0},
        "base_verification": {"tokens": 4, "duration_ms": 5.0},
        "spec_decode_draft_accept": {"tokens": 3, "duration_ms": 0.5}
      },
      "throughput_tokens_per_s": 48.2
    }

``throughput_tokens_per_s`` is optional and is reported as the official
formatter value when present; it is never used by the cross-version gate.
One or more ``--baseline`` and ``--candidate`` profiles are accepted.  Each
profile produces one compatible throughput sample, and the comparison uses
the median of each side.

An optional wall-summary JSON must contain
``comparison.wall_v010_over_v091``.  It is independently gated against 1.05
by default.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
COMPATIBLE_STAGES = (
    "draft_prefill",
    "draft_proposal",
    "base_verification",
)
OFFICIAL_EXTRA_STAGE = "spec_decode_draft_accept"
KNOWN_STAGES = frozenset((*COMPATIBLE_STAGES, OFFICIAL_EXTRA_STAGE))
EDGE_STAGE_NAMES = {
    "spec_decode_draft_prefill": "draft_prefill",
    "spec_decode_draft_proposal": "draft_proposal",
    "spec_decode_base_verification": "base_verification",
    "spec_decode_draft_accept": OFFICIAL_EXTRA_STAGE,
}
DEFAULT_MIN_RATIO = 0.95
DEFAULT_MAX_WALL_RATIO = 1.05


class ValidationError(ValueError):
    """A profile or wall-summary does not satisfy the gate schema."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_number(value: Any, *, label: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ValidationError(f"{label} must be a finite number")
    result = float(value)
    if result <= 0:
        raise ValidationError(f"{label} must be > 0")
    return result


def _positive_integer(value: Any, *, label: str) -> int:
    # Token counts are deliberately strict: accepting 3.5 or true here would
    # make an otherwise plausible-looking profile silently incomparable.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _validate_schema_version(payload: Mapping[str, Any], *, source: str) -> None:
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValidationError(
            f"{source}: schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )


def _stage_entries(raw_stages: Any, *, source: str) -> dict[str, Mapping[str, Any]]:
    """Normalize the two unambiguous v1 stage representations.

    A mapping is the canonical representation.  A list is accepted for
    profiler emitters that preserve repeated records, but duplicate names are
    rejected so a profile cannot accidentally double-count a stage.
    """

    entries: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_stages, Mapping):
        iterable = raw_stages.items()
        for name, stage in iterable:
            if not isinstance(name, str):
                raise ValidationError(f"{source}: stage names must be strings")
            if not isinstance(stage, Mapping):
                raise ValidationError(f"{source}: stage {name!r} must be an object")
            if name in entries:
                raise ValidationError(f"{source}: duplicate stage {name!r}")
            entries[name] = stage
    elif isinstance(raw_stages, list):
        for index, stage in enumerate(raw_stages):
            if not isinstance(stage, Mapping):
                raise ValidationError(f"{source}: stages[{index}] must be an object")
            name = stage.get("name")
            if not isinstance(name, str) or not name:
                raise ValidationError(
                    f"{source}: stages[{index}].name must be a non-empty string"
                )
            if name in entries:
                raise ValidationError(f"{source}: duplicate stage {name!r}")
            entries[name] = stage
    else:
        raise ValidationError(f"{source}: stages must be an object or array")

    unknown = sorted(set(entries) - KNOWN_STAGES)
    if unknown:
        raise ValidationError(f"{source}: unknown stage(s): {', '.join(unknown)}")
    missing = [name for name in COMPATIBLE_STAGES if name not in entries]
    if missing:
        raise ValidationError(f"{source}: missing stage(s): {', '.join(missing)}")
    return entries


def _validate_stage(stage: Mapping[str, Any], *, source: str, name: str) -> dict[str, Any]:
    tokens = _positive_integer(stage.get("tokens"), label=f"{source}: {name}.tokens")
    duration_ms = _positive_number(
        stage.get("duration_ms"), label=f"{source}: {name}.duration_ms"
    )
    return {"tokens": tokens, "duration_ms": duration_ms}


def _official_fields(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Extract formatter fields without making them part of the gate.

    The canonical field is at the profile root.  The two explicit aliases are
    accepted because older capture wrappers used an ``official`` object or a
    more descriptive key while keeping the same numeric meaning.
    """

    result: dict[str, Any] = {}
    for key in (
        "throughput_tokens_per_s",
        "official_throughput_tokens_per_s",
        "reported_throughput_tokens_per_s",
    ):
        if key in payload:
            result[key] = _positive_number(payload[key], label=f"{source}: {key}")

    official = payload.get("official")
    if official is not None:
        if not isinstance(official, Mapping):
            raise ValidationError(f"{source}: official must be an object")
        if "throughput_tokens_per_s" in official:
            result["official.throughput_tokens_per_s"] = _positive_number(
                official["throughput_tokens_per_s"],
                label=f"{source}: official.throughput_tokens_per_s",
            )
    return result


def parse_profile(payload: Any, *, source: str = "profile") -> dict[str, Any]:
    """Validate and normalize one profile, returning its computed metrics."""

    if not isinstance(payload, Mapping):
        raise ValidationError(f"{source}: top-level JSON value must be an object")
    if "mtp_generation" in payload:
        return _parse_edge_llm_profile(payload, source=source)
    _validate_schema_version(payload, source=source)
    entries = _stage_entries(payload.get("stages"), source=source)
    stages = {
        name: _validate_stage(stage, source=source, name=name)
        for name, stage in entries.items()
    }
    compatible_tokens = sum(stages[name]["tokens"] for name in COMPATIBLE_STAGES)
    compatible_duration_ms = sum(
        stages[name]["duration_ms"] for name in COMPATIBLE_STAGES
    )
    compatible_throughput = compatible_tokens / (compatible_duration_ms / 1000.0)
    return {
        "source": source,
        "stages": stages,
        "compatible_tokens": compatible_tokens,
        "compatible_duration_ms": compatible_duration_ms,
        "compatible_throughput_tokens_per_s": compatible_throughput,
        "official_fields": _official_fields(payload, source=source),
    }


def _parse_edge_llm_profile(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Normalize the JSON emitted by the Edge-LLM MTP profiler.

    Edge-LLM reports generated-token count once at ``mtp_generation`` rather
    than repeating it per stage.  The cross-version metric therefore divides
    that count by the sum of GPU time for the three stages present in both
    releases.  ``llm_prefill`` and the v0.10-only accept stage remain visible
    diagnostics but do not affect the gate.
    """

    generation = payload.get("mtp_generation")
    if not isinstance(generation, Mapping):
        raise ValidationError(f"{source}: mtp_generation must be an object")
    generated_tokens = _positive_integer(
        generation.get("total_generated_tokens"),
        label=f"{source}: mtp_generation.total_generated_tokens",
    )
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list):
        raise ValidationError(f"{source}: stages must be an array")

    stages: dict[str, dict[str, Any]] = {}
    for index, raw_stage in enumerate(raw_stages):
        if not isinstance(raw_stage, Mapping):
            raise ValidationError(f"{source}: stages[{index}] must be an object")
        edge_name = raw_stage.get("stage_id")
        if not isinstance(edge_name, str) or not edge_name:
            raise ValidationError(
                f"{source}: stages[{index}].stage_id must be a non-empty string"
            )
        if edge_name == "llm_prefill":
            continue
        name = EDGE_STAGE_NAMES.get(edge_name)
        if name is None:
            raise ValidationError(f"{source}: unknown stage {edge_name!r}")
        if name in stages:
            raise ValidationError(f"{source}: duplicate stage {edge_name!r}")
        stages[name] = {
            "duration_ms": _positive_number(
                raw_stage.get("total_gpu_time_ms"),
                label=f"{source}: {edge_name}.total_gpu_time_ms",
            )
        }

    missing = [name for name in COMPATIBLE_STAGES if name not in stages]
    if missing:
        raise ValidationError(f"{source}: missing stage(s): {', '.join(missing)}")
    compatible_duration_ms = sum(
        stages[name]["duration_ms"] for name in COMPATIBLE_STAGES
    )
    official_fields: dict[str, Any] = {}
    official_key = "overall_tokens_per_second_excluding_base_prefill"
    if official_key in generation:
        official_fields[official_key] = _positive_number(
            generation[official_key], label=f"{source}: mtp_generation.{official_key}"
        )
    return {
        "source": source,
        "stages": stages,
        "compatible_tokens": generated_tokens,
        "compatible_duration_ms": compatible_duration_ms,
        "compatible_throughput_tokens_per_s": generated_tokens
        / (compatible_duration_ms / 1000.0),
        "official_fields": official_fields,
    }


def load_profile(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ValidationError(f"{path}: cannot read JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc.msg}") from exc
    return parse_profile(payload, source=str(path))


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValidationError("at least one profile is required")
    return float(statistics.median(values))


def _summarize_profiles(paths: Sequence[str | Path], *, label: str) -> dict[str, Any]:
    if not paths:
        raise ValidationError(f"{label}: at least one profile is required")
    profiles = [load_profile(path) for path in paths]
    compatible_values = [
        profile["compatible_throughput_tokens_per_s"] for profile in profiles
    ]

    # Preserve each official field independently.  Missing formatter fields
    # are visible in the report and do not change the compatible-stage gate.
    official_names = sorted(
        {
            key
            for profile in profiles
            for key in profile["official_fields"]
        }
    )
    official: dict[str, Any] = {}
    for name in official_names:
        values = [
            profile["official_fields"][name]
            for profile in profiles
            if name in profile["official_fields"]
        ]
        official[name] = {
            "values": values,
            "median": _median(values),
            "sample_count": len(values),
            "missing_count": len(profiles) - len(values),
        }

    return {
        "sample_count": len(profiles),
        "profiles": [
            {
                "source": profile["source"],
                "compatible_tokens": profile["compatible_tokens"],
                "compatible_duration_ms": profile["compatible_duration_ms"],
                "compatible_throughput_tokens_per_s": profile[
                    "compatible_throughput_tokens_per_s"
                ],
                "official_fields": profile["official_fields"],
            }
            for profile in profiles
        ],
        "compatible_throughput_tokens_per_s": {
            "values": compatible_values,
            "median": _median(compatible_values),
            "sample_count": len(compatible_values),
        },
        "official_fields": official,
    }


def _load_wall_summary(path: str | Path) -> float:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ValidationError(f"{path}: cannot read JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ValidationError(f"{path}: top-level JSON value must be an object")
    comparison = payload.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ValidationError(f"{path}: comparison must be an object")
    if "wall_v010_over_v091" not in comparison:
        raise ValidationError(f"{path}: missing comparison.wall_v010_over_v091")
    return _positive_number(
        comparison["wall_v010_over_v091"],
        label=f"{path}: comparison.wall_v010_over_v091",
    )


def evaluate(
    baseline_paths: Sequence[str | Path],
    candidate_paths: Sequence[str | Path],
    *,
    min_ratio: float = DEFAULT_MIN_RATIO,
    wall_summary: str | Path | None = None,
    max_wall_ratio: float = DEFAULT_MAX_WALL_RATIO,
) -> dict[str, Any]:
    """Evaluate the compatible-stage and optional wall-time gates."""

    min_ratio = _positive_number(min_ratio, label="min_ratio")
    max_wall_ratio = _positive_number(max_wall_ratio, label="max_wall_ratio")
    baseline = _summarize_profiles(baseline_paths, label="baseline")
    candidate = _summarize_profiles(candidate_paths, label="candidate")
    baseline_median = baseline["compatible_throughput_tokens_per_s"]["median"]
    candidate_median = candidate["compatible_throughput_tokens_per_s"]["median"]
    ratio = candidate_median / baseline_median
    compatible_passed = ratio >= min_ratio

    wall_value: float | None = None
    wall_passed = True
    if wall_summary is not None:
        wall_value = _load_wall_summary(wall_summary)
        wall_passed = wall_value <= max_wall_ratio

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_contract": {
            "name": "qwen35_mtp_v091_compatible_throughput",
            "stages": list(COMPATIBLE_STAGES),
            "excluded_official_stage": OFFICIAL_EXTRA_STAGE,
            "formula": "compatible_tokens / (sum(compatible_stage_duration_ms) / 1000)",
            "minimum_ratio": min_ratio,
            "official_fields_are_report_only": True,
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "candidate_over_baseline": ratio,
            "compatible_stage_gate": {
                "minimum_ratio": min_ratio,
                "passed": compatible_passed,
            },
            "wall_v010_over_v091": wall_value,
            "wall_gate": {
                "maximum_ratio": max_wall_ratio,
                "passed": wall_passed,
            },
            "passed": compatible_passed and wall_passed,
        },
        "passed": compatible_passed and wall_passed,
    }
    return result


def _error_report(message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "errors": [message],
    }


def _emit(report: Mapping[str, Any], output: str | None) -> None:
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is not None:
        Path(output).write_text(encoded)
    sys.stdout.write(encoded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        "--baseline-profile",
        dest="baseline",
        action="append",
        required=True,
        metavar="JSON",
        help="v0.9.1 profile JSON; repeat for multiple samples",
    )
    parser.add_argument(
        "--candidate",
        "--candidate-profile",
        dest="candidate",
        action="append",
        required=True,
        metavar="JSON",
        help="v0.10 profile JSON; repeat for multiple samples",
    )
    parser.add_argument(
        "--wall-summary",
        metavar="JSON",
        help="optional summary containing comparison.wall_v010_over_v091",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=DEFAULT_MIN_RATIO,
        help=f"minimum compatible candidate/baseline ratio (default: {DEFAULT_MIN_RATIO})",
    )
    parser.add_argument(
        "--max-wall-ratio",
        type=float,
        default=DEFAULT_MAX_WALL_RATIO,
        help=f"maximum wall_v010_over_v091 ratio (default: {DEFAULT_MAX_WALL_RATIO})",
    )
    parser.add_argument("--output", metavar="JSON", help="also write the report to a file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate(
            args.baseline,
            args.candidate,
            min_ratio=args.min_ratio,
            wall_summary=args.wall_summary,
            max_wall_ratio=args.max_wall_ratio,
        )
    except (ValidationError, OSError) as exc:
        report = _error_report(str(exc))
        _emit(report, args.output)
        return 2
    _emit(report, args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
