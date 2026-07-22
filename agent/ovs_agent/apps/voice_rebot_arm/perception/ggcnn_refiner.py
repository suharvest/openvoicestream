"""GG-CNN2 depth-grasp refiner (Phase 3) — optional second opinion / curved-
object primary for the grasp pipeline.

Where it sits in the estimator chain (see grasp_service / ordinary_grasp):

  1. top-face plane fit succeeds + width fits → TOP grasp (boxes; GG-CNN is
     only a 35ms CONSISTENCY CHECK: angle within 25° and width within 2cm →
     agree; disagree → caller triggers a re-observation),
  2. side-face candidate → SIDE grasp (tall objects),
  3. NO plane fits (curved/irregular objects — banana, fruit) → GG-CNN is
     PRIMARY: per-pixel grasp quality on the depth crop, argmax inside the
     instance mask → grasp point + angle + width.

Model: tools/artifacts/ggcnn2-300.onnx (BSD-3, Cornell-trained, 62K params,
~35ms CPU). Preprocessing contract from EXPORT_NOTES.md: inpaint zero-depth
(cv2 INPAINT_NS, 1px border trick), crop+resize to 300², then
``clip(depth - depth.mean(), -1, 1)`` float32. Outputs pos/cos/sin/width
maps; θ = 0.5·atan2(sin, cos); width_px = width·150 (input-pixel scale).

Disabled by default (config ``grasp.ggcnn_refiner``): the box demo's plane
fit is already 5/5 — this ships dark and is enabled after real-fruit
validation (#7).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

INPUT_SIZE = 300
WIDTH_SCALE = 150.0  # model width map → pixels at the 300² input scale


@dataclass
class GgcnnGrasp:
    """Camera-frame grasp hypothesis from the quality map."""

    center_px: tuple[int, int]      # full-image pixel coords
    quality: float                  # [0..1] from the pos map
    angle_rad: float                # in-image-plane grasp angle
    width_m: float                  # metric width via depth + intrinsics
    depth_m: float                  # depth at the grasp point


class GgcnnRefiner:
    """Lazy ONNX wrapper. Construction is cheap; the session loads on first
    use so the agent boots on hosts without the artifact/onnxruntime."""

    def __init__(self, model_path: str) -> None:
        self.model_path = str(model_path)
        self._session: Any = None
        self._input_name: Optional[str] = None

    def _ensure(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort  # noqa: PLC0415 — optional dep

        self._session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    # ── preprocessing (EXPORT_NOTES contract) ─────────────────────────
    @staticmethod
    def _preprocess(depth_mm: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
        crop = depth_mm[y0:y0 + size, x0:x0 + size].astype(np.float32) / 1000.0
        # inpaint invalid depth with the 1px-border trick from the repo.
        invalid = (crop <= 0).astype(np.uint8)
        if invalid.any():
            padded = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
            pmask = cv2.copyMakeBorder(invalid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
            scale = float(np.abs(padded).max()) or 1.0
            painted = cv2.inpaint(
                (padded / scale).astype(np.float32), pmask, 1, cv2.INPAINT_NS
            )
            crop = painted[1:-1, 1:-1] * scale
        if crop.shape != (INPUT_SIZE, INPUT_SIZE):
            crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        return np.clip(crop - crop.mean(), -1.0, 1.0).astype(np.float32)

    def predict(
        self,
        depth_mm: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        center_hint_px: Optional[tuple[int, int]] = None,
    ) -> Optional[GgcnnGrasp]:
        """Best grasp INSIDE ``mask`` from the GG-CNN quality map.

        The crop is a square window around the mask (or hint) so the model
        sees the object at roughly the viewing scale it was trained on.
        Returns None when the model/artifact is unavailable or no positive-
        quality pixel lands inside the mask.
        """
        try:
            self._ensure()
        except Exception:
            logger.warning("ggcnn: session unavailable (%s)", self.model_path,
                           exc_info=True)
            return None
        h, w = depth_mm.shape
        ys, xs = np.nonzero(mask > 0)
        if len(xs) == 0:
            return None
        cx = int(center_hint_px[0]) if center_hint_px else int(xs.mean())
        cy = int(center_hint_px[1]) if center_hint_px else int(ys.mean())
        span = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.6)
        size = int(np.clip(span, INPUT_SIZE, min(h, w)))
        x0 = int(np.clip(cx - size // 2, 0, w - size))
        y0 = int(np.clip(cy - size // 2, 0, h - size))

        blob = self._preprocess(depth_mm, x0, y0, size)[None, None]
        pos, cos2, sin2, width = [np.squeeze(o) for o in
                                  self._session.run(None, {self._input_name: blob})]

        # mask resampled into the crop grid: argmax of quality INSIDE it.
        crop_mask = (mask[y0:y0 + size, x0:x0 + size] > 0).astype(np.uint8)
        crop_mask = cv2.resize(crop_mask, (INPUT_SIZE, INPUT_SIZE),
                               interpolation=cv2.INTER_NEAREST)
        q = np.where(crop_mask > 0, pos, -np.inf)
        if not np.isfinite(q).any() or float(q.max()) <= 0.0:
            return None
        iy, ix = np.unravel_index(int(np.argmax(q)), q.shape)
        quality = float(pos[iy, ix])
        angle = 0.5 * float(np.arctan2(sin2[iy, ix], cos2[iy, ix]))
        width_px_300 = float(width[iy, ix]) * WIDTH_SCALE

        # back to full-image pixels + metric width via depth & fx.
        sx = size / INPUT_SIZE
        px = int(round(x0 + ix * sx))
        py = int(round(y0 + iy * sx))
        z = depth_mm[min(py, h - 1), min(px, w - 1)]
        if z <= 0:
            valid = depth_mm[ys, xs]
            valid = valid[valid > 0]
            if len(valid) == 0:
                return None
            z = float(np.median(valid))
        z_m = float(z) / 1000.0
        fx = max(float(K[0, 0]), 1e-6)
        width_m = width_px_300 * sx * z_m / fx
        return GgcnnGrasp(
            center_px=(px, py),
            quality=quality,
            angle_rad=angle,
            width_m=float(width_m),
            depth_m=z_m,
        )


def consistent(
    plane_angle_deg: float,
    plane_width_m: float,
    gg: GgcnnGrasp,
    angle_tol_deg: float = 25.0,
    width_tol_m: float = 0.02,
) -> bool:
    """Agreement vote between the plane-fit grasp and the GG-CNN hypothesis.

    Angles compare modulo 180° (a parallel jaw is symmetric); widths compare
    absolutely. Used by the caller as: agree → proceed with the plane fit
    (metric, explainable); disagree → trigger a re-observation.
    """
    da = abs(plane_angle_deg - np.degrees(gg.angle_rad)) % 180.0
    da = min(da, 180.0 - da)
    return da <= angle_tol_deg and abs(plane_width_m - gg.width_m) <= width_tol_m


__all__ = ["GgcnnRefiner", "GgcnnGrasp", "consistent", "INPUT_SIZE"]
