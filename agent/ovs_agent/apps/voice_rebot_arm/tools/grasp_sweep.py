"""Synthetic grasp-PLANNING sweep over a documented box/pose/yaw grid (Mac, no
device). Drives the production estimator (``perception.ordinary_grasp`` +
``perception.transforms``) through the torch-free ``synthetic_grasp_harness``
renderer and records, per case, the planned grasp + a set of ANOMALY FLAGS that
encode the regression classes we care about (over-wide top_face, side-z below
table, valid-but-unreachable, width mismatch, method instability, expected
reject). Writes a CSV and prints a SUMMARY MATRIX + ranked worst-region list.

Run:
    uv run python -m ovs_agent.apps.voice_rebot_arm.tools.grasp_sweep --out /tmp/sweep.csv

The grid is a SMART SUBSET (not full cartesian) — a few hundred cases — and a
representative slice is ALSO re-run under the D405-class :class:`NoiseModel`
(fixed seed) to exercise the noise-driven top+side fusion path.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .synthetic_grasp_harness import (
    IMG_HW,
    NoiseModel,
    default_K,
    default_T_cam2base,
    plan_grasp,
    reachable,
)

# ── grid definition (documented discrete sets) ───────────────────────────────
TABLE_Z = 0.05  # base-frame table surface the box bottom rests on

# footprint = (short, long) in metres. Square + one non-square.
FOOTPRINTS: tuple[tuple[float, float], ...] = (
    (0.04, 0.04),
    (0.06, 0.06),
    (0.075, 0.075),
    (0.085, 0.085),
    (0.10, 0.10),
    (0.13, 0.13),
    (0.05, 0.12),  # non-square
)
HEIGHTS: tuple[float, ...] = (0.03, 0.08, 0.15, 0.19, 0.25)
POS_X: tuple[float, ...] = (0.30, 0.40, 0.50, 0.55)
POS_Y: tuple[float, ...] = (-0.15, 0.0, 0.15)
YAWS_DEG: tuple[float, ...] = (0.0, 30.0, 45.0, 90.0)

# epsilons / thresholds for the anomaly flags
Z_TABLE_EPS = 0.005          # >5mm below table surface ⇒ Z_BELOW_TABLE
OVERWIDE_TOP_W = 0.085       # post-8fb88ac top_face must never exceed this
WIDTH_MISMATCH_FRAC = 0.30   # top_face width off true short dim by >30%
JAW_LIMIT = 0.085            # footprint w/ no sub-jaw face ⇒ EXPECTED_REJECT_OK


@dataclass
class CaseResult:
    # inputs
    short_m: float
    long_m: float
    height_m: float
    x: float
    y: float
    yaw_deg: float
    noisy: bool
    # planned grasp
    method: Optional[str]
    jaw_width_m: Optional[float]
    object_length_m: Optional[float]
    is_valid: bool
    rejected_reason: Optional[str]
    # base grasp pose (after production transform); None if no/invalid grasp
    base_x: Optional[float]
    base_y: Optional[float]
    base_z: Optional[float]
    base_pitch: Optional[float]
    base_yaw: Optional[float]
    reachable: Optional[bool]
    reach_why: str
    # anomaly flags
    flags: list[str] = field(default_factory=list)

    def row(self) -> dict:
        d = {
            "short_m": self.short_m,
            "long_m": self.long_m,
            "height_m": self.height_m,
            "x": self.x,
            "y": self.y,
            "yaw_deg": self.yaw_deg,
            "noisy": int(self.noisy),
            "method": self.method or "",
            "jaw_width_m": "" if self.jaw_width_m is None else round(self.jaw_width_m, 4),
            "object_length_m": (
                "" if self.object_length_m is None else round(self.object_length_m, 4)
            ),
            "is_valid": int(self.is_valid),
            "rejected_reason": self.rejected_reason or "",
            "base_x": "" if self.base_x is None else round(self.base_x, 4),
            "base_y": "" if self.base_y is None else round(self.base_y, 4),
            "base_z": "" if self.base_z is None else round(self.base_z, 4),
            "base_pitch": "" if self.base_pitch is None else round(self.base_pitch, 4),
            "base_yaw": "" if self.base_yaw is None else round(self.base_yaw, 4),
            "reachable": "" if self.reachable is None else int(self.reachable),
            "reach_why": self.reach_why,
            "flags": "|".join(self.flags),
        }
        return d


CSV_FIELDS = [
    "short_m", "long_m", "height_m", "x", "y", "yaw_deg", "noisy",
    "method", "jaw_width_m", "object_length_m", "is_valid", "rejected_reason",
    "base_x", "base_y", "base_z", "base_pitch", "base_yaw",
    "reachable", "reach_why", "flags",
]


def _eval_case(
    short_m: float,
    long_m: float,
    height_m: float,
    x: float,
    y: float,
    yaw_deg: float,
    T: np.ndarray,
    K: np.ndarray,
    noise: Optional[NoiseModel],
    probe_unstable: bool = True,
) -> CaseResult:
    from ..perception.transforms import transform_grasp_pose_to_base

    dims = (long_m, short_m, height_m)  # (Lx, Ly, Lz): Ly=short horizontal
    yaw = math.radians(yaw_deg)
    pose = (x, y, TABLE_Z, yaw)

    g = plan_grasp(dims, pose, T, K, IMG_HW, noise=noise)

    method = None if g is None else g.method
    jaw = None if g is None else float(g.jaw_width_m)
    olen = None if g is None else float(g.object_length_m)
    is_valid = bool(g is not None and g.is_valid)
    rej = None if g is None else g.rejected_reason

    base = (None, None, None, None, None)
    reach_ok: Optional[bool] = None
    reach_why = "no grasp" if g is None else ""
    if is_valid:
        try:
            grasp6, _pre = transform_grasp_pose_to_base(
                np.asarray(g.position, dtype=np.float64),
                np.asarray(g.tcp_rotation, dtype=np.float64),
                np.asarray(T, dtype=np.float64),
                pregrasp_offset_m=0.08,
                insertion_depth_m=0.025,
            )
            base = (
                float(grasp6[0]), float(grasp6[1]), float(grasp6[2]),
                float(grasp6[4]), float(grasp6[5]),
            )
            reach_ok, reach_why = reachable(g, T)
        except Exception as e:  # pragma: no cover - defensive
            reach_why = f"transform/reach error: {e}"

    res = CaseResult(
        short_m=short_m, long_m=long_m, height_m=height_m, x=x, y=y,
        yaw_deg=yaw_deg, noisy=noise is not None,
        method=method, jaw_width_m=jaw, object_length_m=olen,
        is_valid=is_valid, rejected_reason=rej,
        base_x=base[0], base_y=base[1], base_z=base[2],
        base_pitch=base[3], base_yaw=base[4],
        reachable=reach_ok, reach_why=reach_why,
    )
    _flag(res, dims, T, K, probe_unstable=probe_unstable)
    return res


def _flag(res: CaseResult, dims, T, K, *, probe_unstable: bool = True) -> None:
    flags = res.flags
    short_m = res.short_m
    footprint_max = max(res.short_m, res.long_m)

    # OVERWIDE_TOP — must be impossible post-8fb88ac
    if res.method == "top_face" and res.jaw_width_m is not None and res.jaw_width_m > OVERWIDE_TOP_W + 1e-6:
        flags.append("OVERWIDE_TOP")

    # Z_BELOW_TABLE — a valid grasp whose base-frame z is below the table by >eps
    if res.is_valid and res.base_z is not None and res.base_z < TABLE_Z - Z_TABLE_EPS:
        flags.append("Z_BELOW_TABLE")

    # VALID_BUT_UNREACHABLE
    if res.is_valid and res.reachable is False:
        flags.append("VALID_BUT_UNREACHABLE")

    # WIDTH_MISMATCH — top_face width off the true short horizontal dim by >30%
    if res.method == "top_face" and res.jaw_width_m is not None and short_m > 0:
        if abs(res.jaw_width_m - short_m) / short_m > WIDTH_MISMATCH_FRAC:
            flags.append("WIDTH_MISMATCH")

    # EXPECTED_REJECT_OK — footprint > jaw limit on BOTH horizontal dims and
    # no valid grasp produced → correctly rejected (GOOD, not a failure).
    if min(res.short_m, res.long_m) > JAW_LIMIT and not res.is_valid:
        flags.append("EXPECTED_REJECT_OK")

    # METHOD_UNSTABLE — method label flips under ±1cm x or ±5° yaw perturbation.
    # Only probe clean (un-noised) valid cases to keep it deterministic & cheap.
    if probe_unstable and not res.noisy and res.method is not None:
        if _method_unstable(res, dims, T, K):
            flags.append("METHOD_UNSTABLE")


def _method_unstable(res: CaseResult, dims, T, K) -> bool:
    base_method = res.method
    perturbs = (
        (res.x + 0.01, res.y, res.yaw_deg),
        (res.x - 0.01, res.y, res.yaw_deg),
        (res.x, res.y, res.yaw_deg + 5.0),
        (res.x, res.y, res.yaw_deg - 5.0),
    )
    for px, py, pyaw in perturbs:
        yaw = math.radians(pyaw)
        g = plan_grasp(dims, (px, py, TABLE_Z, yaw), T, K, IMG_HW, noise=None)
        m = None if g is None else g.method
        if m != base_method:
            return True
    return False


# ── representative noisy subset selector ─────────────────────────────────────
def _is_noisy_rep(short_m, long_m, height_m, x, y, yaw_deg) -> bool:
    """A small representative slice for the noise pass: tall + far boxes (the
    fusion-prone regime) at the central y, axis-aligned + 45° yaw."""
    if height_m < 0.15:
        return False
    if x < 0.50:
        return False
    if abs(y) > 1e-9:
        return False
    if yaw_deg not in (0.0, 45.0):
        return False
    return True


def run_sweep(
    noise_seed: int = 12345,
    include_noisy: bool = True,
    probe_unstable: bool = True,
    reduced: bool = False,
) -> list[CaseResult]:
    """Run the documented grid. Returns the list of CaseResult.

    Smart subset (not full cartesian): for each footprint × height we sweep all
    x but restrict y to {center, +0.15} for square footprints (y is near-
    symmetric) and add the full y set only for the non-square footprint; yaw is
    full for square footprints but {0,45,90} for the non-square (30° adds little
    over the square cases). This keeps the count in the few-hundred range while
    still covering every (footprint, height, x, yaw) combination at least once.

    ``probe_unstable=False`` skips the ±1cm/±5° METHOD_UNSTABLE perturbation
    probe (4 extra plans per clean case) — used by the fast invariants test,
    which only asserts the HARD flags. ``reduced=True`` further trims y to
    {center} and yaw to {0,45,90} for a sub-60s invariant sweep that still
    spans every footprint × height × x.
    """
    T = default_T_cam2base()
    K = default_K()
    noise = NoiseModel(seed=noise_seed)

    results: list[CaseResult] = []
    for (short_m, long_m) in FOOTPRINTS:
        square = abs(short_m - long_m) < 1e-9
        if reduced:
            ys = (0.0,)
            yaws = (0.0, 45.0, 90.0)
        elif square:
            # trim y for square footprints to {center,+0.15} to keep count down.
            ys = (0.0, 0.15)
            yaws = YAWS_DEG
        else:
            ys = POS_Y  # full y for the non-square footprint
            yaws = (0.0, 45.0, 90.0)
        for height_m in HEIGHTS:
            for x in POS_X:
                for y in ys:
                    for yaw_deg in yaws:
                        results.append(
                            _eval_case(short_m, long_m, height_m, x, y,
                                       yaw_deg, T, K, noise=None,
                                       probe_unstable=probe_unstable)
                        )
                        if include_noisy and _is_noisy_rep(
                            short_m, long_m, height_m, x, y, yaw_deg
                        ):
                            results.append(
                                _eval_case(short_m, long_m, height_m, x, y,
                                           yaw_deg, T, K, noise=noise,
                                           probe_unstable=probe_unstable)
                            )
    return results


# ── reporting ────────────────────────────────────────────────────────────────
ANOMALY_FLAGS = [
    "OVERWIDE_TOP",
    "Z_BELOW_TABLE",
    "VALID_BUT_UNREACHABLE",
    "WIDTH_MISMATCH",
    "METHOD_UNSTABLE",
    "EXPECTED_REJECT_OK",
]
METHOD_LABELS = ["top_face", "side_face", "legacy", "(none)"]


def summarize(results: list[CaseResult]) -> str:
    lines: list[str] = []
    n = len(results)
    lines.append(f"\n{'=' * 72}")
    lines.append(f"GRASP SWEEP SUMMARY — {n} cases")
    lines.append("=" * 72)

    # method × anomaly matrix
    def mkey(r: CaseResult) -> str:
        return r.method if r.method else "(none)"

    counts: dict[tuple[str, str], int] = {}
    method_totals: dict[str, int] = {m: 0 for m in METHOD_LABELS}
    flag_totals: dict[str, int] = {f: 0 for f in ANOMALY_FLAGS}
    for r in results:
        m = mkey(r)
        method_totals[m] = method_totals.get(m, 0) + 1
        for f in r.flags:
            counts[(m, f)] = counts.get((m, f), 0) + 1
            flag_totals[f] = flag_totals.get(f, 0) + 1

    # matrix table
    hdr = f"{'method':<10} " + " ".join(f"{f[:11]:>12}" for f in ANOMALY_FLAGS) + f" {'TOTAL':>7}"
    lines.append("\nMETHOD × ANOMALY MATRIX")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for m in METHOD_LABELS:
        cells = " ".join(f"{counts.get((m, f), 0):>12}" for f in ANOMALY_FLAGS)
        lines.append(f"{m:<10} {cells} {method_totals.get(m, 0):>7}")
    tot_cells = " ".join(f"{flag_totals.get(f, 0):>12}" for f in ANOMALY_FLAGS)
    lines.append(f"{'TOTAL':<10} {tot_cells} {n:>7}")

    # ranked worst-region list (hard anomalies first)
    lines.append("\nRANKED WORST-REGION LIST (by anomaly, region = footprint/x band)")
    region_hits: dict[tuple[str, str], int] = {}
    for r in results:
        fp = max(r.short_m, r.long_m)
        fp_band = ">=0.10" if fp >= 0.10 else ("0.075-0.085" if fp >= 0.075 else "<0.075")
        x_band = ">=0.50" if r.x >= 0.50 else "<0.50"
        for f in r.flags:
            if f == "EXPECTED_REJECT_OK":
                continue  # GOOD outcome — not a "worst region"
            key = (f, f"footprint{fp_band} & x{x_band}")
            region_hits[key] = region_hits.get(key, 0) + 1
    if region_hits:
        # hard flags ranked above soft ones
        hard = {"OVERWIDE_TOP", "Z_BELOW_TABLE"}
        ranked = sorted(
            region_hits.items(),
            key=lambda kv: (kv[0][0] not in hard, -kv[1]),
        )
        for (flag, region), c in ranked:
            tag = "HARD" if flag in hard else "soft"
            lines.append(f"  [{tag}] {flag:<22} {c:>4}×  @ {region}")
    else:
        lines.append("  (no anomalies of any kind — clean sweep)")

    # explicit GOOD / hard summary
    n_reject_ok = flag_totals.get("EXPECTED_REJECT_OK", 0)
    n_overwide = flag_totals.get("OVERWIDE_TOP", 0)
    n_ztab = flag_totals.get("Z_BELOW_TABLE", 0)
    lines.append(
        f"\nHARD invariants: OVERWIDE_TOP={n_overwide} (must be 0), "
        f"Z_BELOW_TABLE={n_ztab} (must be 0)"
    )
    lines.append(f"EXPECTED_REJECT_OK (correctly rejected over-jaw footprints): {n_reject_ok}")
    return "\n".join(lines)


def write_csv(results: list[CaseResult], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(r.row())


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="grasp_sweep_results.csv",
                    help="CSV output path (default: ./grasp_sweep_results.csv)")
    ap.add_argument("--no-noisy", action="store_true",
                    help="skip the noisy representative subset")
    ap.add_argument("--noise-seed", type=int, default=12345)
    args = ap.parse_args(argv)

    t0 = time.time()
    results = run_sweep(noise_seed=args.noise_seed, include_noisy=not args.no_noisy)
    dt = time.time() - t0

    write_csv(results, args.out)
    print(summarize(results))
    print(f"\nCSV written: {Path(args.out).resolve()}")
    print(f"Total cases: {len(results)}   wall time: {dt:.2f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
