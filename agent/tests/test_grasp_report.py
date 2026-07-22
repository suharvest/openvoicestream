"""grasp_report.py parser/aggregator — fixtures are REAL log lines captured
from seeed-orin-nx on 2026-06-12 (agent-format python-repr dicts + cycle-tool
JSON), plus noise lines and one truncated record that must not crash."""
from __future__ import annotations

from ovs_agent.apps.voice_rebot_arm.tools.grasp_report import build_report, parse_lines

FIXTURE = r"""
2026-06-12 14:12:26,472 INFO ovs_agent.slv_client: SLV advertise 10 tool(s) (sp_len=3057, llm_params=True)
2026-06-12 12:32:04,080 INFO ovs_agent.apps.voice_rebot_arm.grasp_plugin: GraspPlugin: grasp result: {'success': True, 'target': 'box', 'cancelled': False, 'grasp_class': 'box', 'grasp_conf': 0.6801813244819641, 'jaw_width_m': 0.08497507870197296, 'grasp_pose': [0.59, 0.21, 0.12, -0.12, 0.37, 0.47], 'grasp_closed': True, 'adaptive_force': False, 'returned_home': True, 'lifted': True, 'holding': True, 'stage': 'done', 'stage_ms': {'init': 0, 'capture': 640, 'detect': 0, 'transform': 3, 'open': 661, 'pregrasp': 2225, 'grasp_move': 1628, 'grasp': 2037, 'lift': 1280, 'carry_home': 2187}, 'attempt': 1}
2026-06-12 12:32:14,203 INFO ovs_agent.apps.voice_rebot_arm.grasp_plugin: GraspPlugin: grasp result: {'success': False, 'released': False, 'cancelled': False, 'used_recorded_pose': True, 'release_opening_m': 0.0853, 'stage': 'release', 'error': 'release failed — jaw still gripping after full open'}
2026-06-12 12:32:52,746 INFO ovs_agent.apps.voice_rebot_arm.grasp_plugin: GraspPlugin: grasp result: {'found': True, 'target': 'box', 'cancelled': False, 'scan_index': 0, 'conf': 0.167, 'position_base': [0.749, 0.058, 0.124], 'position_plausible': False, 'scanned_poses': 1}
GRASP 1 {"success": false, "target": "box", "cancelled": false, "jaw_width_m": 0.16596192121505737, "reobserved": true, "stage": "plausibility", "stage_ms": {"init": 0, "capture": 517, "detect": 0, "transform": 3, "reobserve": 2628}, "error": "implausible jaw width 0.166m", "attempt": 2}
GRASP 2 {"success": true, "target": "box", "cancelled": false, "jaw_width_m": 0.058447452019224175, "reobserved": true, "servo_drift_mm": 27.8, "grasp_closed": true, "holding": true, "stage": "done", "stage_ms": {"init": 24, "capture": 717, "detect": 0, "transform": 3, "reobserve": 3111, "open": 404, "pregrasp": 2170, "servo": 1187, "grasp_move": 1543, "grasp": 1685, "lift": 1270, "carry_home": 2160}, "attempt": 1}
PUTDOWN 2 {"success": true, "released": true, "cancelled": false, "used_recorded_pose": true, "release_opening_m": 0.0679, "placed_at": [0.514, 0.136, 0.156], "stage": "done"}
GraspPlugin: grasp result: {'success': True, 'truncated and broken
random unrelated line containing { braces } that must be ignored
"""


def test_parse_real_lines_and_classification():
    records = parse_lines(FIXTURE.splitlines())
    kinds = [k for k, _ in records]
    # 3 grasps, 2 put_downs (release-failure + success), 1 search; the
    # truncated line and noise lines are dropped.
    assert kinds.count("grasp") == 3
    assert kinds.count("put_down") == 2
    assert kinds.count("search") == 1


def test_report_numbers_match_manual_count():
    report = build_report(parse_lines(FIXTURE.splitlines()))
    g = report["grasp"]
    assert g["total"] == 3 and g["ok"] == 2
    assert g["fail_stage_dist"] == {"plausibility": 1}
    assert g["retry_rate"] == round(1 / 3, 3)      # one attempt=2 record
    assert g["retry_recovered"] == 0               # that retry still failed
    assert 27.8 in g["servo_corrections"]
    assert 0.085 == round(g["widths_m"][0], 3)
    p = report["put_down"]
    assert p["total"] == 2 and p["ok"] == 1
    assert p["fail_stage_dist"] == {"release": 1}
    assert report["search"]["found"] == 1
    # stage timings aggregated across grasps
    assert report["stage_ms"]["capture"]["n"] == 3


def test_empty_and_garbage_input_safe():
    assert build_report([])["grasp"]["total"] == 0
    assert parse_lines(["", "{not a record}", "GRASP x {bad json"]) == []
