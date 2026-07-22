=== HELD GRASP ATTEMPT — FINAL FINDING (Path A exhausted, hard gripper wall) ===

GOAL: produce >=1 HELD (box lifted >30mm, held between both fingers).
RESULT: NO HELD. Reproducible hard wall in the gripper USD geometry.

CLOSEST RESULT (run_held.py default, raw line from held_result.txt):
  dims=(0.05,0.05,0.08) pose=(0.30,0.0,0.02,yaw0) grasp executed at yaw=90deg
  move_above clean: box_after_above=[0.3,0.0,0.06] (NOT knocked)
  jaw_to_box_xy_mm=1.6  (jaw mm-accurate over the box)
  jaw settled z=0.1103 (= box_top 0.10 +~0.01) -> fingertips REST ON BOX TOP
  close_half_target=0.0  finger_L=0.0226 finger_R=0.0293 (pinch box TOP edge)
  status=SLIPPED  lifted_mm=0.1  box_z0=0.06 box_z1=0.0601

ROOT-CAUSE DIAGNOSIS (each step verified with an isolated probe):
1. Gripper jaw frame ~= fingertip plane (tool_offset 0.045 along tool+X = -Z in the
   vertical grasp). Reachable vertical JAW band (pinocchio+settle): x=0.40 z[0.08,0.16];
   x=0.36 z[0.08,0.18]; single elbow-down branch, continuous descent never branch-jumps.
2. At yaw=0 the gripper STRUCTURE overlaps the box X-footprint -> open jaw lowered over a
   box lands on its top, cannot descend (dbg_width: narrowing box-Y did nothing).
3. At yaw=90 the finger-separation axis rotates to X; reachable column is clean only at
   x<=0.32 (dbg_yawmap: z[0.06,0.14], jaw_err ~2mm). At x=0.30 the approach is clean
   (box NOT knocked) for boxes up to top~0.10; taller boxes (top>=0.11) get hit by the
   wrist during the multiseed move_above and are flung aside.
4. THE WALL (dbg_iso / dbg_narrowx, all reproducible single-run):
   - With a box present, the descent ALWAYS stalls with the jaw at box_top+~0.01,
     i.e. the FINGERTIPS rest on the box TOP FACE. They never get alongside the body.
   - This is independent of: pad friction (default vs 4.0), finger drive kp (625 vs 8000),
     box X-width (50/30/20mm — fingers straddle by up to 32mm/side and STILL stall),
     box Y-width, box height.
   - The fingertips reach the free-descent floor (jaw 0.0485) ONLY when the box is
     absent/knocked aside -> a solid INTER-FINGER BRIDGE/PALM rests on any object top,
     not the finger width, is the blocker.
   Conclusion: this gripper USD cannot perform a top-down enveloping grasp of a table
   object. Top-down only yields a top-edge pinch, which cannot resist a vertical lift.
5. The only geometry that places fingertips beside the body is a SIDE entry at grasp
   height — which the position-controlled arm executes by translating laterally INTO
   the box, sweeping it (= the original P4 24/24 KNOCKED/SLIPPED result).

PATHS TRIED:
  Path A (tall box): boxes 0.07-0.20m tall, widths 0.02-0.06m, x 0.30-0.42, yaw 0 & 90.
    All SLIP (rest on top) or KNOCKED (wrist hits a tall box / lateral sweep).
  Path B (branch-locked vertical/lateral): vertical descend (clean, but rests on top);
    lateral elbow-down entry (sweeps box). Strong squeeze (kp 8000, mu 4.0) + deep-
    descend-then-raise: still a top-edge pinch -> SLIP.

NEXT STEP TO BREAK THE WALL (needs a USD change, out of scope for sim_bridge edits):
  Fix the gripper USD so the finger collision meshes extend BELOW the inter-finger
  bridge (longer fingers / recessed palm), OR add a fixed-joint attach on contact to
  emulate a successful grasp. With the shipped collision geometry, top-down envelop is
  physically impossible.

numpy: 1.26.4 (<2). Pipeline (perception/*, transforms.py) UNCHANGED.
