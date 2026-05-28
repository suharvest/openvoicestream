#!/usr/bin/env python3
"""Perf regression gate.

Compares a perf result JSON (produced by perf.py / stats.save_results) against a
per-device baseline and emits PASS/FAIL. Exit code 1 if any metric regresses, so
it can be wired into CI / a post-run check.

Two subcommands:

  gate.py check  <result.json|dir...> [--baseline baselines/baseline.json]
                 [--device NAME] [--strict] [--json out.json]
      Compare results against the baseline. With --strict, a missing baseline
      entry is a FAIL instead of a SKIP.

  gate.py update <result.json|dir...> [--baseline baselines/baseline.json]
                 [--device NAME] [--note "..."]
      Write/merge the result summary into the baseline (seed or refresh).
      Use this after a known-good full run to establish the reference numbers.

Device identity = meta.client_host (== fleet node name in --mode-label local runs),
or inferred from a `_from_<node>/` parent dir, or --device override.

Metric direction:
  - error_rate : lower is better, absolute tolerance (limit = base + error_rate_abs)
  - similarity : higher is better, pct tolerance  (limit = base * (1 - similarity_pct))
  - all others : lower is better, pct tolerance   (limit = base * (1 + pct))
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE / "baselines" / "baseline.json"

DEFAULT_TOLERANCES = {
    "pct": 0.25,            # latency / rtf: fail if actual > base * 1.25
    "error_rate_abs": 0.10,  # CER/WER: fail if actual > base + 0.10
    "similarity_pct": 0.10,  # speaker sim: fail if actual < base * 0.90
}

# Which summary stat to gate per metric, and which to store on --update.
STAT_FOR_METRIC = {
    "error_rate": "mean",
    "similarity": "mean",
}
DEFAULT_STAT = "p95"  # latency + rtf metrics: gate the tail

# Metrics worth gating per scenario family. Others in the summary are ignored.
GATED_METRICS = {
    "asr": ("finalize_rtf", "error_rate", "eos_to_final_ms"),
    "tts": ("rtf", "tfd_ms", "total_ms"),
    "v2v": ("eos_to_first_audio_ms", "asr_finalize_ms", "tts_tfd_ms"),
    "v2v_stream": ("endpoint_latency_ms", "asr_finalize_ms", "total_latency_ms"),
    "concurrent": ("rtf", "tfd_ms", "eos_to_final_ms"),
    "clone": ("rtf", "total_ms", "similarity"),
}

_FAMILIES = ("v2v_stream", "concurrent", "asr", "tts", "v2v", "clone", "noise",
             "stability", "boot")


def scenario_family(scenario: str) -> str:
    s = scenario.lower()
    for fam in _FAMILIES:
        if s.startswith(fam):
            return fam
    return s


def stat_for(metric: str) -> str:
    return STAT_FOR_METRIC.get(metric, DEFAULT_STAT)


def device_of(result: dict, path: Path, override: str | None) -> str | None:
    if override:
        return override
    host = result.get("meta", {}).get("client_host")
    if host:
        return host
    for part in path.parts:
        if part.startswith("_from_"):
            return part[len("_from_"):]
    return None


def iter_result_files(inputs: list[str]):
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            yield from sorted(p.glob("*.json"))
        elif any(ch in item for ch in "*?["):
            yield from (Path(x) for x in sorted(glob.glob(item)))
        else:
            yield p


def load_result(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    # Must look like a perf result, not a baseline / collected blob.
    if not isinstance(d, dict) or "summary" not in d or "scenario" not in d:
        return None
    return d


def load_baseline(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"_meta": {}, "_tolerances": dict(DEFAULT_TOLERANCES), "devices": {}}


def evaluate(metric: str, actual: float, base: float, tol: dict):
    """Return (ok, limit, direction)."""
    if metric == "error_rate":
        limit = base + tol["error_rate_abs"]
        return actual <= limit, limit, "<="
    if metric == "similarity":
        limit = base * (1 - tol["similarity_pct"])
        return actual >= limit, limit, ">="
    limit = base * (1 + tol["pct"])
    return actual <= limit, limit, "<="


def cmd_check(args):
    baseline = load_baseline(Path(args.baseline))
    tol = {**DEFAULT_TOLERANCES, **baseline.get("_tolerances", {})}
    devices = baseline.get("devices", {})

    rows = []  # (device, fam, group, metric, stat, base, actual, limit, verdict)
    n_pass = n_fail = n_skip = 0

    for path in iter_result_files(args.inputs):
        result = load_result(path)
        if result is None:
            continue
        device = device_of(result, path, args.device)
        fam = scenario_family(result["scenario"])
        gated = GATED_METRICS.get(fam)
        if gated is None:
            continue
        dev_base = devices.get(device, {}).get(fam, {})
        for group, gstats in result["summary"].items():
            if not isinstance(gstats, dict):
                continue
            for metric in gated:
                mstat = gstats.get(metric)
                if not isinstance(mstat, dict):
                    continue
                stat = stat_for(metric)
                actual = mstat.get(stat)
                if actual is None:
                    continue
                base_entry = dev_base.get(group, {}).get(metric)
                if base_entry is None:
                    n_skip += 1
                    if args.strict:
                        n_fail += 1
                        rows.append((device, fam, group, metric, stat,
                                     None, actual, None, "NO-BASE"))
                    continue
                base = base_entry["value"]
                ok, limit, _ = evaluate(metric, actual, base, tol)
                verdict = "PASS" if ok else "FAIL"
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                rows.append((device, fam, group, metric, stat,
                             base, actual, limit, verdict))

    _print_table(rows)
    print(f"\n{n_pass} pass, {n_fail} fail, {n_skip} skipped (no baseline)")

    if args.json:
        Path(args.json).write_text(json.dumps([
            {"device": d, "scenario": f, "group": g, "metric": m, "stat": s,
             "baseline": b, "actual": a, "limit": lim, "verdict": v}
            for (d, f, g, m, s, b, a, lim, v) in rows
        ], indent=2))

    failed = any(r[-1] in ("FAIL", "NO-BASE") for r in rows)
    sys.exit(1 if failed else 0)


def _fmt(v):
    if v is None:
        return "-"
    if abs(v) < 10:
        return f"{v:.3f}"
    return f"{v:.0f}"


def _print_table(rows):
    if not rows:
        print("(no comparable metrics found)")
        return
    headers = ("device", "scenario", "group", "metric", "stat",
               "baseline", "actual", "limit", "verdict")
    table = [headers] + [
        (d, f, g, m, s, _fmt(b), _fmt(a), _fmt(lim), v)
        for (d, f, g, m, s, b, a, lim, v) in rows
    ]
    widths = [max(len(str(r[i])) for r in table) for i in range(len(headers))]
    for ri, r in enumerate(table):
        line = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
        print(line)
        if ri == 0:
            print("  ".join("-" * widths[i] for i in range(len(headers))))


def cmd_update(args):
    bpath = Path(args.baseline)
    baseline = load_baseline(bpath)
    baseline.setdefault("_tolerances", dict(DEFAULT_TOLERANCES))
    devices = baseline.setdefault("devices", {})
    n_written = 0

    for path in iter_result_files(args.inputs):
        result = load_result(path)
        if result is None:
            continue
        device = device_of(result, path, args.device)
        if not device:
            print(f"skip (no device): {path}")
            continue
        fam = scenario_family(result["scenario"])
        gated = GATED_METRICS.get(fam)
        if gated is None:
            continue
        dev = devices.setdefault(device, {})
        fam_entry = dev.setdefault(fam, {})
        for group, gstats in result["summary"].items():
            if not isinstance(gstats, dict):
                continue
            for metric in gated:
                mstat = gstats.get(metric)
                if not isinstance(mstat, dict):
                    continue
                stat = stat_for(metric)
                if mstat.get(stat) is None:
                    continue
                fam_entry.setdefault(group, {})[metric] = {
                    "stat": stat,
                    "value": round(mstat[stat], 4),
                    "src": path.name,
                }
                n_written += 1

    baseline["_meta"] = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": args.note or baseline.get("_meta", {}).get("note", ""),
    }
    bpath.parent.mkdir(parents=True, exist_ok=True)
    bpath.write_text(json.dumps(baseline, indent=2, ensure_ascii=False))
    print(f"Wrote {n_written} metric baselines for "
          f"{len(devices)} device(s) -> {bpath}")


def main():
    p = argparse.ArgumentParser(prog="gate")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_c = sub.add_parser("check", help="compare results against baseline")
    sp_c.add_argument("inputs", nargs="+", help="result JSON files, globs, or dirs")
    sp_c.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    sp_c.add_argument("--device", default=None, help="override device name")
    sp_c.add_argument("--strict", action="store_true",
                      help="missing baseline entry counts as FAIL")
    sp_c.add_argument("--json", default=None, help="write verdict rows to JSON")
    sp_c.set_defaults(func=cmd_check)

    sp_u = sub.add_parser("update", help="seed/refresh baseline from results")
    sp_u.add_argument("inputs", nargs="+", help="result JSON files, globs, or dirs")
    sp_u.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    sp_u.add_argument("--device", default=None, help="override device name")
    sp_u.add_argument("--note", default=None, help="note stored in baseline _meta")
    sp_u.set_defaults(func=cmd_update)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
