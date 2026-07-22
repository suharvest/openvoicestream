"""End-to-end embin-mode segmenter tests on the REAL artifacts.

Drives YoloOnnxSegmenter in vocab-decoupled ("embin") mode on
``yoloe-26s-seg-embin.onnx`` + the real text encoder + bus.jpg. Skipped when
either artifact is absent so CI without them still passes.

Asserts:
  * embin mode is detected (class_embeddings fed on every predict);
  * NO detection has cls_id >= active_n (the pad-slot guard);
  * positive-image parity: vocab ["bus","person"] on bus.jpg yields the frozen
    person/bus counts at conf>=0.25 (regression freeze, codex ask);
  * a richer vocab ["box",...,"yellow banana"] runs end-to-end.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import pytest

from ovs_agent.apps.voice_rebot_arm.perception.text_pe import TextPromptEncoder
from ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx import YoloOnnxSegmenter

EMBIN_DIR = Path("/Users/harvest/project/_scratch/yolo-onnx-probe/embin_out")
MODEL = EMBIN_DIR / "yoloe-26s-seg-embin.onnx"
ENCODER = EMBIN_DIR / "text_encoder_pe.onnx"
IMG = Path(__file__).parent / "bus.jpg"

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and ENCODER.exists() and IMG.exists()),
    reason="embin model / text encoder / bus.jpg fixture absent",
)


def _make_segmenter(vocab, tmp_path):
    enc = TextPromptEncoder(str(ENCODER), pad_slots=16, cache_dir=str(tmp_path))
    emb = enc.encode(vocab)
    seg = YoloOnnxSegmenter(
        str(MODEL),
        vocab,
        providers=["CPUExecutionProvider"],
        class_embeddings=emb,
        active_n=enc.active_n,
    )
    return seg, enc


def _counts(res):
    cnt = Counter()
    max_cls = -1
    for i in range(len(res.boxes)):
        cid = int(res.boxes[i].cls[0])
        max_cls = max(max_cls, cid)
        cnt[res.names[cid]] += 1
    return cnt, max_cls


def test_embin_mode_detected_and_pad_guard(tmp_path):
    vocab = ["bus", "person"]
    seg, enc = _make_segmenter(vocab, tmp_path)
    seg._ensure_session()
    assert seg._embin is True
    assert seg._active_n == 2

    img = cv2.imread(str(IMG))
    res = seg.predict(img, conf=0.25)[0]
    _, max_cls = _counts(res)
    # pad-slot guard: no detection may land on a padded class id (>= active_n)
    assert max_cls < enc.active_n
    for i in range(len(res.boxes)):
        assert 0 <= int(res.boxes[i].cls[0]) < enc.active_n


def test_positive_image_parity_bus_person(tmp_path):
    """Frozen regression: bus.jpg with ["bus","person"] at conf>=0.25 ->
    4 person + 1 bus (captured 2026-06-15 on yoloe-26s-seg-embin.onnx)."""
    vocab = ["bus", "person"]
    seg, _ = _make_segmenter(vocab, tmp_path)
    img = cv2.imread(str(IMG))
    res = seg.predict(img, conf=0.25)[0]
    cnt, _ = _counts(res)
    assert cnt.get("person", 0) == 4
    assert cnt.get("bus", 0) == 1
    assert len(res.boxes) == 5
    # every detected label is from the configured vocab
    for i in range(len(res.boxes)):
        assert res.names[int(res.boxes[i].cls[0])] in set(vocab)


def test_grasp_vocab_runs_end_to_end(tmp_path):
    """A grasp-style vocab runs without error end-to-end and never emits a
    padded class id (bus.jpg has no boxes → 0 detections, which is fine)."""
    vocab = ["box", "cardboard box", "carton", "yellow banana"]
    seg, enc = _make_segmenter(vocab, tmp_path)
    img = cv2.imread(str(IMG))
    res = seg.predict(img, conf=0.25)[0]
    _, max_cls = _counts(res)
    assert max_cls < enc.active_n  # -1 (no dets) or a valid class id
    # names mapping reflects the configured vocab
    assert res.names == {i: n for i, n in enumerate(vocab)}
