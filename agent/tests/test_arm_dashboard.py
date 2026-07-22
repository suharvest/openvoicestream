"""Arm dashboard: bus, annotation, pipeline tee, idle observer, HTTP API."""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.dashboard_bus import DashboardBus
from ovs_agent.apps.voice_rebot_arm.perception.annotate import (
    annotate_frame,
    depth_colormap,
)
from ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx import (
    YoloResult,
    _Box,
    _Boxes,
    _Masks,
)


def _result(h=120, w=160):
    mask = np.zeros((h, w), dtype=np.float32)
    mask[40:80, 60:110] = 1.0
    return YoloResult(
        names={0: "box"},
        boxes=_Boxes([_Box([60, 40, 110, 80], 0, 0.77)]),
        masks=_Masks(mask[None]),
        orig_shape=(h, w),
    )


# ── bus ──────────────────────────────────────────────────────────────
def test_bus_frame_and_event_ring():
    bus = DashboardBus(frame_history=3, event_history=2)
    assert bus.latest_jpg() is None
    for i in range(5):
        bus.publish_frame(f"jpg{i}".encode(), b"d", {"stage": f"s{i}"})
        bus.publish_event("grasp-box", {"success": i % 2 == 0, "i": i})
    assert bus.latest_jpg() == b"jpg4"
    snap = bus.snapshot()
    assert snap["frame_seq"] == 5
    assert snap["frame_meta"]["stage"] == "s4"
    assert len(snap["frame_history"]) == 3          # ring bounded
    assert len(snap["events"]) == 2                 # ring bounded
    assert snap["events"][-1]["i"] == 4
    assert "ts" in snap["frame_meta"]


def test_bus_thread_safety_smoke():
    bus = DashboardBus()
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            bus.publish_frame(b"x", None, {"stage": "t"})
            bus.publish_event("e", {"success": True})

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.2:
        bus.snapshot()
        bus.latest_jpg()
    stop.set()
    t.join(timeout=2)


# ── annotation ───────────────────────────────────────────────────────
def test_annotate_frame_produces_jpeg():
    pytest.importorskip("cv2")
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    jpg = annotate_frame(img, [_result()], None, label="detect")
    assert jpg[:2] == b"\xff\xd8"  # JPEG SOI
    assert len(jpg) > 500


def test_annotate_frame_with_best_grasp_geometry():
    pytest.importorskip("cv2")
    from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import GraspPose

    best = GraspPose(
        class_name="box", conf=0.8, bbox_xyxy=(60, 40, 110, 80),
        center_px=(85, 60), position=None, rotation=None, tcp_rotation=None,
        jaw_width_m=0.056, object_length_m=0.1, angle_deg=10.0,
        rect_points=np.array([[60, 40], [110, 40], [110, 80], [60, 80]]),
        short_edge_points=np.array([[60, 60], [110, 60]]),
        valid_depth_pixels=100, method="side_face",
    )
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    jpg = annotate_frame(img, [_result()], best, label="servo")
    assert jpg[:2] == b"\xff\xd8"


def test_depth_colormap_marks_invalid_black():
    cv2 = pytest.importorskip("cv2")
    depth = np.full((60, 80), 400, dtype=np.uint16)
    depth[:, :20] = 0  # sensor hole
    jpg = depth_colormap(depth)
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img[:, :10].max() <= 20      # hole ≈ black (jpeg noise tolerance)
    assert img[:, 30:].max() > 100      # valid region colored


# ── pipeline tee ─────────────────────────────────────────────────────
def test_grasp_pipeline_calls_frame_sink():
    from tests.test_grasp_service import FakeArm, FakeCamera, FakeSegmenter, _make_result, _scene
    from ovs_agent.apps.voice_rebot_arm.grasp_service import run_grasp_once

    color, depth, K = _scene()
    seen = []
    res = run_grasp_once(
        "banana",
        arm=FakeArm(), segmenter=FakeSegmenter(_make_result()),
        camera=FakeCamera(color, depth, K), K=K, T_hand_eye=np.eye(4),
        warm_up_frames=0, move_duration=0.02,
        frame_sink=lambda c, d, r, b, s: seen.append((s, b is not None)),
    )
    assert res["success"] is True
    assert len(seen) >= 1
    assert seen[0][1] is True  # winning frame carried a best candidate


def test_frame_sink_errors_never_break_the_grasp():
    from tests.test_grasp_service import FakeArm, FakeCamera, FakeSegmenter, _make_result, _scene
    from ovs_agent.apps.voice_rebot_arm.grasp_service import run_grasp_once

    color, depth, K = _scene()

    def _boom(*a):
        raise RuntimeError("sink crash")

    res = run_grasp_once(
        "banana",
        arm=FakeArm(), segmenter=FakeSegmenter(_make_result()),
        camera=FakeCamera(color, depth, K), K=K, T_hand_eye=np.eye(4),
        warm_up_frames=0, move_duration=0.02, frame_sink=_boom,
    )
    assert res["success"] is True


# ── dashboard HTTP API ───────────────────────────────────────────────
class _FakeApp:
    plugins: list = []


def test_dashboard_state_and_frame_endpoints():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer
    from aiohttp import web
    from ovs_agent.apps.voice_rebot_arm.dashboard_plugin import ArmDashboardPlugin
    from ovs_agent.apps.voice_rebot_arm import dashboard_bus

    dashboard_bus.BUS.publish_frame(b"\xff\xd8fakejpg", b"\xff\xd8fakedepth",
                                    {"stage": "idle", "detections": []})
    plugin = ArmDashboardPlugin(_FakeApp(), {
        "place_bounds": [0.2, 0.6, -0.26, 0.4],
        "observation_port": 1,  # closed port → arm proxied as None
    })

    async def _drive():
        web_app = web.Application()
        web_app.router.add_get("/api/state", plugin._api_state)  # noqa: SLF001
        web_app.router.add_get("/api/frame.jpg", plugin._api_frame)  # noqa: SLF001
        web_app.router.add_get("/", plugin._handle_index)  # noqa: SLF001
        async with TestClient(TestServer(web_app)) as client:
            r = await client.get("/api/state")
            assert r.status == 200
            st = await r.json()
            assert st["place_bounds"] == [0.2, 0.6, -0.26, 0.4]
            assert st["frame_seq"] >= 1
            assert st["frame_meta"]["stage"] == "idle"
            assert st["arm"] is None
            r2 = await client.get("/api/frame.jpg")
            assert r2.status == 200
            assert (await r2.read())[:2] == b"\xff\xd8"
            r3 = await client.get("/")
            assert r3.status == 200
            assert "reBot" in (await r3.text())

    asyncio.run(_drive())
