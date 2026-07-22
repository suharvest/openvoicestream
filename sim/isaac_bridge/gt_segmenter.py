"""GtSegmenter — ground-truth instance segmentation -> YoloResult.

Bypasses YOLO (spec decision 1). Builds the production YoloResult from Isaac's
instance-segmentation of the target box, exactly as
tools/synthetic_grasp_harness.make_detection does (cls=0, conf=1.0, mask HxW,
bbox from mask extents).
"""
import sys
sys.path.insert(0, "/root/agent/ovs_agent/apps")

import numpy as np
from voice_rebot_arm.perception.yolo_onnx import YoloResult, _Box, _Boxes, _Masks


def make_detection(box_mask, K=None, class_name="box", conf=1.0):
    """Build a real YoloResult from a HxW uint8 instance mask (1 over object)."""
    box_mask = np.asarray(box_mask)
    ys, xs = np.nonzero(box_mask > 0)
    if len(xs) == 0:
        raise ValueError("empty box mask — nothing rendered into view")
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    box = _Box(np.array([x1, y1, x2, y2], dtype=np.float32), cls_id=0, conf=conf)
    boxes = _Boxes([box])
    masks = _Masks(np.asarray(box_mask, dtype=np.float32)[None, ...])  # (1,H,W)
    H, W = box_mask.shape
    return YoloResult(names={0: class_name}, boxes=boxes, masks=masks, orig_shape=(H, W))


class GtSegmenter:
    """Holds a reference to the scene so predict() can pull the live GT mask.

    `mask_provider()` -> HxW uint8 mask of the target box for the current frame.
    """
    def __init__(self, mask_provider, class_name="box"):
        self._mask_provider = mask_provider
        self._class_name = class_name

    def predict(self, bgr, only_names=None):
        mask = self._mask_provider()
        if mask is None or int(np.sum(mask > 0)) < 20:
            return [YoloResult(names={0: self._class_name},
                               boxes=_Boxes([]), masks=None, orig_shape=bgr.shape[:2])]
        return [make_detection(mask, class_name=self._class_name)]
