"""HARD-invariant gate over a reduced synthetic grasp sweep (Mac, no device).

Runs a reduced ``grasp_sweep.run_sweep`` and asserts the two invariants that
must NEVER break, regardless of footprint / pose / yaw / noise:

  * ZERO ``OVERWIDE_TOP``  — no ``top_face`` grasp may carry a jaw width > 0.085
    (the 8fb88ac regression class).
  * ZERO ``Z_BELOW_TABLE`` — no is_valid grasp may sit below the table surface
    by more than the epsilon (the P2 side-z class).

The SOFT flags (WIDTH_MISMATCH / METHOD_UNSTABLE / VALID_BUT_UNREACHABLE) are
diagnostic signal — they are COUNTED and PRINTED (run with ``-s`` to see) but do
NOT fail the test. Kept < ~60s by using the reduced grid and disabling the
METHOD_UNSTABLE perturbation probe.
"""

from __future__ import annotations

import time

from ovs_agent.apps.voice_rebot_arm.tools.grasp_sweep import (
    run_sweep,
    summarize,
)


def test_sweep_hard_invariants():
    t0 = time.time()
    results = run_sweep(include_noisy=True, probe_unstable=False, reduced=True)
    dt = time.time() - t0

    # diagnostic report (visible with -s)
    print(summarize(results))
    print(f"\n[invariants] reduced sweep: {len(results)} cases in {dt:.2f}s")

    flag_counts: dict[str, int] = {}
    overwide: list = []
    zbelow: list = []
    for r in results:
        for f in r.flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1
        if "OVERWIDE_TOP" in r.flags:
            overwide.append(r)
        if "Z_BELOW_TABLE" in r.flags:
            zbelow.append(r)

    print(f"[invariants] flag counts: {flag_counts}")

    # ── HARD invariants ──
    assert not overwide, (
        f"OVERWIDE_TOP invariant broken in {len(overwide)} case(s): "
        + ", ".join(
            f"(fp={r.short_m}x{r.long_m} h={r.height_m} x={r.x} yaw={r.yaw_deg} "
            f"noisy={r.noisy} w={r.jaw_width_m})"
            for r in overwide[:5]
        )
    )
    assert not zbelow, (
        f"Z_BELOW_TABLE invariant broken in {len(zbelow)} case(s): "
        + ", ".join(
            f"(fp={r.short_m}x{r.long_m} h={r.height_m} x={r.x} yaw={r.yaw_deg} "
            f"noisy={r.noisy} base_z={r.base_z})"
            for r in zbelow[:5]
        )
    )

    # keep the gate fast
    assert dt < 60.0, f"reduced sweep took {dt:.1f}s (> 60s budget)"
