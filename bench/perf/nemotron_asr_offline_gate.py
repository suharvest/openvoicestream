#!/usr/bin/env python3
"""Reproducible offline quality/performance gate for Edge-LLM Nemotron ASR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from runners import _normalize_for_match, metric_transcript


_TRANSCRIPT_RE = re.compile(r"^Transcript:\s*(.*)$", re.MULTILINE)
_RTF_RE = re.compile(r"^RTF:\s*([0-9.]+)", re.MULTILINE)
_LANGUAGE_TAG_RE = re.compile(r"\s*<[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?>\s*")
_QWEN_LANGUAGE_PREFIX_RE = re.compile(
    r"^\s*language\s+(?:Chinese|English|Cantonese|Japanese|Korean|French|German|Italian|Portuguese|Russian|Spanish)",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_transcript(output: str) -> str:
    match = _TRANSCRIPT_RE.search(output)
    if not match:
        raise ValueError("Nemotron CLI output has no Transcript line")
    return strip_runtime_language_prefix(match.group(1))


def strip_runtime_language_prefix(text: str) -> str:
    text = _LANGUAGE_TAG_RE.sub("", text)
    text = _QWEN_LANGUAGE_PREFIX_RE.sub("", text)
    return text.strip()


def parse_rtf(output: str) -> float:
    match = _RTF_RE.search(output)
    if not match:
        raise ValueError("Nemotron benchmark output has no RTF line")
    return float(match.group(1))


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def exact_error_rate(reference: str, hypothesis: str, lang: str) -> float:
    """Exact Levenshtein CER for zh and WER for en; no optional dependency."""
    reference = _normalize_for_match(reference, lang)
    hypothesis = _normalize_for_match(hypothesis, lang)
    ref_units = list(reference) if lang == "zh" else reference.split()
    hyp_units = list(hypothesis) if lang == "zh" else hypothesis.split()
    if not ref_units:
        return float(bool(hyp_units))
    previous = list(range(len(hyp_units) + 1))
    for ref_index, ref_unit in enumerate(ref_units, 1):
        current = [ref_index]
        for hyp_index, hyp_unit in enumerate(hyp_units, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_unit != hyp_unit),
                )
            )
        previous = current
    return previous[-1] / len(ref_units)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for lang in ("zh", "en"):
        selected = [row for row in rows if row["lang"] == lang]
        errors = [float(row["error_rate"]) for row in selected]
        walls = [float(row["wall_ms"]) for row in selected]
        result[lang] = {
            "count": len(selected),
            "metric": "CER" if lang == "zh" else "WER",
            "mean_error_rate": statistics.fmean(errors) if errors else None,
            "median_error_rate": statistics.median(errors) if errors else None,
            "p95_error_rate": percentile(errors, 0.95) if errors else None,
            "median_cold_process_wall_ms": statistics.median(walls) if walls else None,
        }
    return result


def run_cli(command: list[str], plugin: Path) -> tuple[str, str, float]:
    environment = os.environ.copy()
    environment["EDGELLM_PLUGIN_PATH"] = str(plugin)
    environment["LD_PRELOAD"] = str(plugin)
    started = time.perf_counter()
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    return result.stdout, result.stderr, (time.perf_counter() - started) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-qwen-report", type=Path)
    parser.add_argument("--benchmark-id", action="append", default=[])
    parser.add_argument(
        "--auto-language",
        action="store_true",
        help="Use checkpoint default prompt 101 instead of zh-CN=4/en-US=0.",
    )
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--max-mean-zh-cer", type=float, default=1.0)
    parser.add_argument("--max-mean-en-wer", type=float, default=1.0)
    parser.add_argument("--max-p95-rtf", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["files"][: args.limit]
    rows: list[dict[str, Any]] = []
    stderr_error_lines: list[str] = []
    entry_by_id = {entry["id"]: entry for entry in entries}

    for entry in entries:
        audio = args.corpus_root / entry["filename"]
        actual_sha = sha256(audio)
        if actual_sha != entry["sha256"]:
            raise RuntimeError(f"{entry['id']}: WAV SHA mismatch: {actual_sha}")
        command = [
                str(args.binary),
                f"--engineDir={args.engine_dir}",
                f"--tokenizerDir={args.engine_dir}",
                f"--audioFile={audio}",
        ]
        if not args.auto_language:
            command.append(f"--promptId={4 if entry['lang'] == 'zh' else 0}")
        stdout, stderr, wall_ms = run_cli(command, args.plugin)
        hypothesis = parse_transcript(stdout)
        reference = metric_transcript(entry)
        error_rate = exact_error_rate(reference, hypothesis, entry["lang"])
        rows.append(
            {
                "id": entry["id"],
                "lang": entry["lang"],
                "category": entry["category"],
                "duration_s": entry["duration_s"],
                "reference": reference,
                "hypothesis": hypothesis,
                "error_rate": error_rate,
                "wall_ms": wall_ms,
                "rtf_including_process_init": wall_ms / 1000.0 / entry["duration_s"],
            }
        )
        stderr_error_lines.extend(
            line for line in stderr.splitlines() if re.search(r"\b(error|failed|exception)\b", line, re.I)
        )

    benchmarks: list[dict[str, Any]] = []
    for entry_id in args.benchmark_id:
        entry = entry_by_id.get(entry_id)
        if entry is None:
            raise RuntimeError(f"benchmark id not present in selected corpus: {entry_id}")
        audio = args.corpus_root / entry["filename"]
        command = [
                str(args.binary),
                f"--engineDir={args.engine_dir}",
                f"--tokenizerDir={args.engine_dir}",
                f"--audioFile={audio}",
                "--benchmark",
                f"--iters={args.benchmark_iters}",
                f"--warmup={args.benchmark_warmup}",
        ]
        if not args.auto_language:
            command.append(f"--promptId={4 if entry['lang'] == 'zh' else 0}")
        stdout, stderr, wall_ms = run_cli(command, args.plugin)
        benchmarks.append({"id": entry_id, "rtf": parse_rtf(stdout), "process_wall_ms": wall_ms})
        stderr_error_lines.extend(
            line for line in stderr.splitlines() if re.search(r"\b(error|failed|exception)\b", line, re.I)
        )

    summary = summarize(rows)
    baseline: dict[str, Any] | None = None
    if args.baseline_qwen_report:
        baseline_payload = json.loads(args.baseline_qwen_report.read_text())
        qwen_by_id = {
            Path(row["wav"]).stem: row for row in baseline_payload.get("results", [])
        }
        baseline_rows: list[dict[str, Any]] = []
        for entry in entries:
            qwen_row = qwen_by_id.get(entry["id"])
            if qwen_row is None:
                raise RuntimeError(f"Qwen baseline is missing {entry['id']}")
            hypothesis = strip_runtime_language_prefix(qwen_row["text"])
            baseline_rows.append(
                {
                    "id": entry["id"],
                    "lang": entry["lang"],
                    "error_rate": exact_error_rate(
                        metric_transcript(entry), hypothesis, entry["lang"]
                    ),
                    "wall_ms": float(qwen_row["elapsed_ms"]),
                    "rtf": float(qwen_row["elapsed_ms"]) / 1000.0 / entry["duration_s"],
                }
            )
        baseline_summary = summarize(baseline_rows)
        baseline = {
            "source": str(args.baseline_qwen_report),
            "summary": baseline_summary,
            "rows": baseline_rows,
            "candidate_minus_qwen_mean_error_rate": {
                lang: summary[lang]["mean_error_rate"]
                - baseline_summary[lang]["mean_error_rate"]
                for lang in ("zh", "en")
            },
        }
    benchmark_rtfs = [float(row["rtf"]) for row in benchmarks]
    passed = (
        not stderr_error_lines
        and summary["zh"]["mean_error_rate"] <= args.max_mean_zh_cer
        and summary["en"]["mean_error_rate"] <= args.max_mean_en_wer
        and (not benchmark_rtfs or percentile(benchmark_rtfs, 0.95) <= args.max_p95_rtf)
    )
    report = {
        "schema_version": 1,
        "runtime_semantics": "offline_batch_1_no_cache_aware_streaming",
        "prompt_mode": "auto_101" if args.auto_language else "zh-CN_4_en-US_0",
        "artifacts": {
            "binary": {"path": str(args.binary), "sha256": sha256(args.binary)},
            "plugin": {"path": str(args.plugin), "sha256": sha256(args.plugin)},
            "audio_encoder_engine": {
                "path": str(args.engine_dir / "audio_encoder.engine"),
                "sha256": sha256(args.engine_dir / "audio_encoder.engine"),
            },
            "rnnt_step_engine": {
                "path": str(args.engine_dir / "rnnt_step.engine"),
                "sha256": sha256(args.engine_dir / "rnnt_step.engine"),
            },
        },
        "thresholds": {
            "max_mean_zh_cer": args.max_mean_zh_cer,
            "max_mean_en_wer": args.max_mean_en_wer,
            "max_p95_rtf": args.max_p95_rtf,
        },
        "summary": summary,
        "qwen_baseline": baseline,
        "benchmarks": benchmarks,
        "rows": rows,
        "stderr_error_lines": stderr_error_lines,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
