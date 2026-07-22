=== SIDE / FORWARD GRASP GEOMETRIC EVALUATION — FINDING (2026-06-14/15) ===

QUESTION: With the corrected real-CAD gripper (-X approach, tool_offset_x=-0.128),
can the reBot arm geometrically grasp a table box with a SIDE/FORWARD (not top-down)
approach? Build on prior agent's proven "top-down is kinematically impossible".

CHOSEN GRASP ORIENTATION (the reachable side/forward approach):
  roll=0, pitch~0.0 (horizontal), yaw=0.
  gripper -X (approach) world dir = [-1.00, 0.00, ~0]  -> points straight FORWARD
    (toward the base / into the box front face), essentially horizontal.
  finger separation axis = world Y (sepY=[0,+-1,0]) -> OPEN fingers straddle the
    box's +Y / -Y side faces. CONFIRMED straddle=True for all 48 sweep configs
    (padL.y ~ +0.042, padR.y ~ -0.042, box half-Y 0.015-0.025).

KEY KINEMATIC MAP (probe_tradeoff / probe_zfloor, pure pinocchio FK):
  - The JAW/pad reaches box-MID height (z=table+lz/2, i.e. BESIDE the body) ONLY at a
    near-HORIZONTAL approach (pitch 0.0-0.3, approach_down 0 to -0.25 = 0-14 deg below
    horizontal). At those low z's the approach is essentially forward, not down.
  - DOWN-pointing orientations (pitch~1.6, approach_down ~ -0.95) are reachable ONLY at
    high z (pad floor ~0.13), i.e. ABOVE any table box -> top-edge pinch only (this is
    the prior agent's top-down wall, re-confirmed). Straight-down (-X.-Z=1) unreachable
    everywhere (best ~0.39); positive-pitch wrap reaches -0.97 down but only at z>=0.13.

INSERTION-SIGN REVERSAL (the thing the task asked to demonstrate):
  TOP-DOWN  geom (geom_sweep.csv):  insertion_mm = -47 .. -83 mm  (pad ABOVE box top),
                                    pad_in_body_z = never.
  SIDE/FWD  geom (geom_side_sweep.csv): fwd_insertion_mm = +1 .. +25 mm at x>=0.34,
                                    pad_in_body_z = TRUE for all lz>=0.05 (36/48),
                                    straddle = TRUE for all 48.
  => The side approach genuinely puts the pad BESIDE the body with POSITIVE insertion
     and the open fingers straddling the box — qualitatively what top-down can NEVER do.

THE WALL (why geom_ok = 0/48 anyway): the approach SWEEPS the box.
  geom_side_sweep.csv: geom_ok=False for ALL 48. knock_mm = 52 .. 171 (median 110).
  Fail split: 21 UNREACH (box-mid pose at x<=0.34 not cleanly IK-reachable) + 27 KNOCK.
  Advance trace (probe_advance): the box starts moving while the pad is still ~80-120mm
    IN FRONT of the box front face. Cause = the OPEN fingers are two ~80mm blades that
    extend forward of the pad-contact point (extents_out.txt: finger far-tips reach
    80mm along the finger axis from end_link; pad-contact behind them); under a position-
    controlled horizontal advance the blade tips / gripper body at box height shove the
    box before the contact pad is ever positioned beside it.
  Angled descent does NOT escape it (probe_angled): there is an inescapable tradeoff —
    shallow enough to put the pad beside the body (pitch 0.3, +13mm insertion) sweeps the
    box 124-173mm; steep enough to reduce knock (pitch 0.7, knock 45-88mm) lifts the pad
    back ABOVE the body (insertion goes -4..-7mm). Knock never drops below ~45mm AND
    stays positive-insertion simultaneously.

VERDICT:
  The side/forward approach is the arm's REACHABLE natural grasp orientation and it DOES
  achieve the correct grasp GEOMETRY (fingers straddle the box's side faces, pad reaches
  beside the body with positive insertion) — unlike top-down, which can only pad-above
  the top. BUT a clean grasp is NOT geometrically attainable for a FREE table box with a
  position-controlled arm: every reachable approach that places the pad beside the body
  first sweeps the box away (knock 45-171mm) because the open finger blades extend ~80mm
  forward of the contact and collide with the box on entry. There is no footprint x
  height x x region with knock<8mm AND positive insertion AND straddle.
  (The held2 friction sweep's "HELD" results are a high-mu/high-kp top-EDGE pinch on a
   pinned/heavy box, not a clean side envelope — they do not contradict this.)

  To make a side grasp clean would need either: a fixed-joint attach-on-contact to model
  a successful grip (out of scope: requires pipeline change), a non-position controller
  that stops on contact, or the object braced/against a wall so it cannot be swept.

ARTIFACTS:
  geom_side_sweep.csv      — the 48-config side/forward geom map (this finding's data)
  geom_sweep.csv           — top-down geom (negative insertion, for the before/after)
  probe_tradeoff.py/.out   — reachable approach-angle vs box-mid height map
  probe_zfloor.py          — jaw z-floor per x (z=0.020 reachable only at ~horizontal)
  probe_straddle.py        — USD straddle + forward-insertion confirm
  probe_advance.py         — advance trace: box knocks ~80-120mm before pad arrives
  probe_extent2.py, extents_out.txt — finger blades reach 80mm forward of pad-contact
  probe_angled.py          — angled-descent tradeoff (no clean escape)
  run_held2.py --geomside  — the side/forward geom sweep mode (only file edited)
numpy 1.26.4. Pipeline / URDF / USD / isaac_arm kinematics UNCHANGED.
