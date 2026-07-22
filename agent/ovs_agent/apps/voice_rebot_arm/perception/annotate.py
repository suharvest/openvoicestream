"""Render dashboard decision frames: detections + grasp geometry on the color
image, and a colormapped depth view. cv2 is a device-image dependency already
(letterboxing/masks use it); import stays inside the functions so the module
imports on hosts without it.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

_GREEN = (80, 220, 80)
_YELLOW = (40, 200, 230)
_RED = (60, 60, 230)
_WHITE = (240, 240, 240)


def annotate_frame(
    color_bgr: np.ndarray,
    results: Optional[list] = None,
    best: Any = None,
    label: str = "",
    jpeg_quality: int = 80,
) -> bytes:
    """Detections (boxes + mask outlines) plus the winning grasp's OBB and
    jaw line drawn onto a copy of the frame; returns JPEG bytes."""
    import cv2

    img = np.ascontiguousarray(color_bgr).copy()
    for r in results or []:
        names = getattr(r, "names", {}) or {}
        boxes = getattr(getattr(r, "boxes", None), "__iter__", None)
        masks = getattr(getattr(r, "masks", None), "data", None)
        if boxes is None:
            continue
        for i, b in enumerate(r.boxes):
            x1, y1, x2, y2 = (int(v) for v in np.asarray(b.xyxy).reshape(-1)[:4])
            conf = float(np.asarray(b.conf).reshape(-1)[0])
            cls_id = int(np.asarray(b.cls).reshape(-1)[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), _GREEN, 2)
            cv2.putText(
                img, f"{names.get(cls_id, cls_id)} {conf:.2f}",
                (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN, 1,
            )
            if masks is not None and i < len(masks):
                m = (np.asarray(masks[i]) > 0.5).astype(np.uint8)
                if m.shape[:2] == img.shape[:2]:
                    contours, _ = cv2.findContours(
                        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(img, contours, -1, _YELLOW, 1)
    if best is not None:
        try:
            rect = np.asarray(best.rect_points, dtype=np.int32).reshape(-1, 2)
            if len(rect) >= 4:
                cv2.polylines(img, [rect], True, _RED, 2)
            line = np.asarray(best.short_edge_points, dtype=np.int32).reshape(-1, 2)
            if len(line) >= 2:
                cv2.line(img, tuple(line[0]), tuple(line[1]), _RED, 3)
            cx, cy = (int(v) for v in best.center_px)
            cv2.drawMarker(img, (cx, cy), _RED, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                img,
                f"{best.method} w={best.jaw_width_m:.3f}m",
                (cx + 12, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _RED, 2,
            )
        except Exception:  # geometry fields are best-effort across estimators
            pass
    if label:
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _WHITE, 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return buf.tobytes() if ok else b""


def depth_colormap(
    depth_mm: np.ndarray,
    near_mm: float = 150.0,
    far_mm: float = 1200.0,
    jpeg_quality: int = 80,
) -> bytes:
    """JET-colormapped depth (near=red → far=blue); invalid/zero depth renders
    black so sensor holes (e.g. the gripper-occlusion blind zone) are obvious."""
    import cv2

    d = np.asarray(depth_mm, dtype=np.float32)
    valid = (d > 1.0)
    norm = np.clip((d - near_mm) / max(1.0, far_mm - near_mm), 0.0, 1.0)
    img8 = (255 - norm * 255).astype(np.uint8)  # near bright, far dark
    cm = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    cm[~valid] = 0
    ok, buf = cv2.imencode(".jpg", cm, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return buf.tobytes() if ok else b""
