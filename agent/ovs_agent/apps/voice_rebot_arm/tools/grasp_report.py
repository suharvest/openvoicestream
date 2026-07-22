"""Aggregate grasp/put_down/search results from agent logs into a session report.

Feed it `docker logs voice-rebot-arm` output (file or stdin) — it parses the
``GraspPlugin: grasp result: {...}`` lines (and the cycle checker's
``GRASP n {...}`` / ``PUTDOWN n {...}`` lines) and prints success rates,
failure-stage distribution, per-stage timing percentiles, retry recovery and
strategy-trigger rates. The measurement substrate for every future tuning
decision: run a demo, pull the log, get numbers.

Usage:
    docker logs voice-rebot-arm 2>&1 | python grasp_report.py
    python grasp_report.py session.log --json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict

# `GraspPlugin: grasp result: {...}` uses python-repr dicts (single quotes,
# True/False/None); the cycle tool prints real JSON. Accept both.
_LINE_RES = [
    re.compile(r"GraspPlugin: grasp result: (\{.*\})\s*$"),
    re.compile(r"^(?:GRASP|PUTDOWN|SEARCH)\s+\d+\s+(\{.*\})\s*$"),
]


def _parse_payload(text: str):
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue
    return None


def classify(rec: dict) -> str:
    """grasp / put_down / search, from the result's distinguishing keys."""
    if "found" in rec:
        return "search"
    if "released" in rec or "placed_at" in rec or "release_opening_m" in rec:
        return "put_down"
    return "grasp"


def parse_lines(lines) -> list[tuple[str, dict]]:
    out = []
    for line in lines:
        for rx in _LINE_RES:
            m = rx.search(line)
            if not m:
                continue
            rec = _parse_payload(m.group(1))
            if rec is not None:
                out.append((classify(rec), rec))
            break
    return out


def _pct(values, q):
    if not values:
        return None
    vs = sorted(values)
    idx = min(len(vs) - 1, max(0, int(round(q * (len(vs) - 1)))))
    return vs[idx]


def build_report(records: list[tuple[str, dict]]) -> dict:
    grasps = [r for k, r in records if k == "grasp"]
    putdowns = [r for k, r in records if k == "put_down"]
    searches = [r for k, r in records if k == "search"]

    g_ok = [g for g in grasps if g.get("success")]
    stage_ms: dict[str, list[int]] = defaultdict(list)
    for g in grasps:
        for stage, ms in (g.get("stage_ms") or {}).items():
            stage_ms[stage].append(int(ms))
    fail_stages = Counter(
        str(g.get("stage")) for g in grasps if not g.get("success") and not g.get("cancelled")
    )
    retried = [g for g in grasps if int(g.get("attempt") or 1) > 1]

    def _total_ms(g):
        return sum((g.get("stage_ms") or {}).values()) or None

    totals = [t for t in (_total_ms(g) for g in g_ok) if t]
    report = {
        "grasp": {
            "total": len(grasps),
            "ok": len(g_ok),
            "cancelled": sum(1 for g in grasps if g.get("cancelled")),
            "success_rate": round(len(g_ok) / len(grasps), 3) if grasps else None,
            "fail_stage_dist": dict(fail_stages),
            "retry_rate": round(len(retried) / len(grasps), 3) if grasps else None,
            "retry_recovered": sum(1 for g in retried if g.get("success")),
            "reobserve_rate": round(
                sum(1 for g in grasps if g.get("reobserved")) / len(grasps), 3
            ) if grasps else None,
            "servo_corrections": [g.get("servo_drift_mm") for g in grasps
                                  if g.get("servo_drift_mm") is not None],
            "widths_m": [round(float(g["jaw_width_m"]), 4) for g in grasps
                         if g.get("jaw_width_m")],
            "total_ms_p50": _pct(totals, 0.5),
            "total_ms_p90": _pct(totals, 0.9),
        },
        "put_down": {
            "total": len(putdowns),
            "ok": sum(1 for p in putdowns if p.get("success")),
            "fail_stage_dist": dict(Counter(
                str(p.get("stage")) for p in putdowns if not p.get("success")
            )),
        },
        "search": {
            "total": len(searches),
            "found": sum(1 for s in searches if s.get("found")),
        },
        "stage_ms": {
            stage: {"p50": _pct(v, 0.5), "p90": _pct(v, 0.9), "n": len(v)}
            for stage, v in sorted(stage_ms.items())
        },
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logfile", nargs="?", help="log file (default: stdin)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    fh = open(args.logfile, encoding="utf-8", errors="replace") if args.logfile else sys.stdin
    try:
        records = parse_lines(fh)
    finally:
        if args.logfile:
            fh.close()

    report = build_report(records)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        g = report["grasp"]; p = report["put_down"]; s = report["search"]
        print(f"grasps   : {g['ok']}/{g['total']} ok "
              f"(cancelled {g['cancelled']}, retry_rate {g['retry_rate']}, "
              f"retry_recovered {g['retry_recovered']})")
        print(f"put_down : {p['ok']}/{p['total']} ok  fail_stages={p['fail_stage_dist']}")
        print(f"search   : {s['found']}/{s['total']} found")
        print(f"fail stages : {g['fail_stage_dist']}")
        print(f"reobserve rate {g['reobserve_rate']}  servo {g['servo_corrections']}")
        print(f"widths m: {g['widths_m']}")
        print(f"grasp total ms p50={g['total_ms_p50']} p90={g['total_ms_p90']}")
        for stage, st in report["stage_ms"].items():
            print(f"  {stage:<12} p50={st['p50']:>6} p90={st['p90']:>6} n={st['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
