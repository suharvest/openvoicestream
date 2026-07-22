"""Torch-free YOLOE-seg segmenter on onnxruntime + numpy/cv2.

Replaces the ultralytics ``YOLO.predict`` path used by ``reBot-DevArm-Grasp``
with a pure onnxruntime + numpy/cv2 implementation that produces a
``YoloResult`` whose ``names`` / ``boxes`` / ``masks`` / ``orig_shape`` fields
mirror the subset of the ultralytics ``Results`` API consumed downstream
(``ordinary_grasp`` + ``yolo_utils``): ``result.names`` (dict),
``result.boxes[i].xyxy/.cls/.conf``, ``result.masks.data[i]`` (numpy HxW),
``result.orig_shape`` (h, w). No torch, no ``.cpu()``, no ultralytics.

ONNX export contract (validated, see _scratch/yolo-onnx-probe):
  * The fixed-vocabulary YOLOE-seg ONNX bakes both the open-vocab class set
    AND NMS into the graph, so we do NOT run NMS ourselves.
  * ``output0``: ``[1, 300, 4 + 1 + 1 + 32]`` rows of
    ``[x1, y1, x2, y2, conf, cls_id, *32 mask_coeffs]`` in 640-letterbox xyxy.
  * ``output1``: ``[1, 32, 160, 160]`` mask prototypes.

Mask assembly replicates ultralytics ``process_mask(upsample=True)``:
  coeffs @ protos -> 160x160 logits, crop by box*ratio in proto space,
  bilinear upsample to 640, threshold > 0 (== sigmoid > 0.5), THEN strip the
  letterbox padding and resize the *content region* to the original image
  (NEAREST). The padding strip is load-bearing: ``ultralytics`` ``masks.data``
  is already letterbox-cropped to the scaled content (e.g. 640x480, not
  640x640); skipping the strip systematically under-estimates mask area by
  ~25%.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np

_LOG = logging.getLogger(__name__)

# Default ONNX Runtime provider preference: TensorRT (Jetson) → CUDA → CPU.
# onnxruntime silently drops unavailable providers, so this list is safe on a
# CPU-only Mac (falls back to CPUExecutionProvider).
DEFAULT_PROVIDERS: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


# ── ultralytics-compatible result containers (numpy-only) ────────────────────
class _Box:
    """One detection's box, mirroring ``ultralytics.engine.results.Boxes[i]``.

    ``xyxy`` is a ``(1, 4)`` array so downstream ``box.xyxy[0]`` indexing works;
    ``cls`` / ``conf`` are length-1 arrays so ``int(box.cls[0])`` /
    ``float(box.conf[0])`` work unchanged. No ``.cpu()`` needed (already numpy).
    """

    __slots__ = ("xyxy", "cls", "conf")

    def __init__(self, xyxy: np.ndarray, cls_id: int, conf: float) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(1, 4)
        self.cls = np.asarray([cls_id], dtype=np.float32)
        self.conf = np.asarray([conf], dtype=np.float32)


class _Boxes:
    """Indexable / len-able collection of :class:`_Box`."""

    __slots__ = ("_items",)

    def __init__(self, items: Sequence[_Box]) -> None:
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> _Box:
        return self._items[index]


class _Masks:
    """Holds ``data``: an ``(N, H, W)`` numpy mask stack (orig image space).

    Mirrors ``ultralytics`` ``result.masks.data`` which downstream code reads
    as ``result.masks.data[i]`` (per-instance HxW float/uint mask). Already
    numpy → ``tensor_to_numpy`` short-circuits, no ``.cpu()`` invoked.
    """

    __slots__ = ("data",)

    def __init__(self, data: np.ndarray) -> None:
        self.data = data


@dataclass
class YoloResult:
    """Minimal stand-in for ``ultralytics.engine.results.Results``.

    Exposes exactly the attributes the vendored grasp pipeline reads:
    ``names`` (id→label dict), ``boxes`` (:class:`_Boxes`), ``masks``
    (:class:`_Masks` or ``None``), ``orig_shape`` ``(h, w)``. ``obb`` is always
    ``None`` (YOLOE-seg has no oriented boxes) so the grasp code falls through
    to its mask/box path.
    """

    names: dict[int, str]
    boxes: _Boxes
    masks: Optional[_Masks]
    orig_shape: tuple[int, int]
    obb: None = None


def _letterbox(
    image_bgr: np.ndarray,
    new_shape: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, float, float]:
    """Resize+pad to a square ``new_shape`` keeping aspect ratio.

    Returns ``(padded, ratio, dw, dh)`` where ``ratio`` is the uniform scale
    and ``dw`` / ``dh`` are the *total* horizontal / vertical padding (split
    evenly across both sides). Matches the ultralytics LetterBox transform
    used at export time.
    """
    h, w = image_bgr.shape[:2]
    ratio = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * ratio)), int(round(h * ratio)))
    dw = (new_shape - new_unpad[0]) / 2.0
    dh = (new_shape - new_unpad[1]) / 2.0
    resized = image_bgr
    if (w, h) != new_unpad:
        resized = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, ratio, dw, dh


class _Cudart:
    """Minimal ctypes shim over the host-mounted ``libcudart`` — malloc /
    memcpy / stream only. Deliberately NOT cuda-python/pycuda: the slim
    container gets CUDA + TensorRT exclusively via host bind-mounts (the
    edge-llm/SLV contract), so there is nothing to pip-install and nothing
    to redo after a container recreate."""

    H2D = 1
    D2H = 2

    def __init__(self) -> None:
        import ctypes

        self._ct = ctypes
        lib = None
        for cand in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
            try:
                lib = ctypes.CDLL(cand)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError(
                "libcudart not loadable — host CUDA lib dir must be bind-"
                "mounted and on LD_LIBRARY_PATH (slim-container contract)"
            )
        self._lib = lib

    def _check(self, rc: int, what: str) -> None:
        if int(rc) != 0:
            raise RuntimeError(f"{what} failed (cudaError {int(rc)})")

    def malloc(self, nbytes: int):
        ptr = self._ct.c_void_p()
        self._check(self._lib.cudaMalloc(self._ct.byref(ptr), self._ct.c_size_t(nbytes)), "cudaMalloc")
        return ptr

    def free(self, ptr) -> None:
        try:
            self._lib.cudaFree(ptr)
        except Exception:
            pass

    def memcpy(self, dst, src, nbytes: int, kind: int) -> None:
        self._check(
            self._lib.cudaMemcpy(dst, src, self._ct.c_size_t(nbytes), self._ct.c_int(kind)),
            "cudaMemcpy",
        )

    def stream_create(self):
        s = self._ct.c_void_p()
        self._check(self._lib.cudaStreamCreate(self._ct.byref(s)), "cudaStreamCreate")
        return s

    def stream_sync(self, s) -> None:
        self._check(self._lib.cudaStreamSynchronize(s), "cudaStreamSynchronize")


class _TrtSession:
    """Native-TensorRT session exposing the same ``run(None, feeds)`` calling
    convention :class:`YoloOnnxSegmenter` uses with onnxruntime, so the whole
    pre/post-process pipeline is backend-agnostic.

    Expects a serialized engine built (ON-DEVICE, same GPU arch) from the
    YOLOE-seg ONNX, e.g.::

        /usr/src/tensorrt/bin/trtexec --onnx=yoloe-26s-seg-box.onnx \
            --saveEngine=yoloe-26s-seg-box.engine --fp16 --skipInference

    Outputs are reordered to the ONNX contract (det rows ``[1,300,38]`` first,
    mask prototypes ``[1,32,h,w]`` second) and cast to float32 so the numpy
    post-process is byte-compatible with the CPU path.
    """

    def __init__(self, engine_path: str) -> None:
        import tensorrt as trt  # host dist-packages, bind-mounted

        self._trt = trt
        trt_logger = trt.Logger(trt.Logger.WARNING)
        # init_libnvinfer_plugins: harmless if no plugins are used.
        try:
            trt.init_libnvinfer_plugins(trt_logger, "")
        except Exception:
            pass
        with open(engine_path, "rb") as fh:
            blob = fh.read()
        runtime = trt.Runtime(trt_logger)
        self._engine = runtime.deserialize_cuda_engine(blob)
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize TRT engine {engine_path!r}")
        self._context = self._engine.create_execution_context()
        self._cu = _Cudart()
        self._stream = self._cu.stream_create()

        # All input bindings, keyed by tensor name (multi-input support, e.g.
        # the embeddings-as-input embin engine: ``images`` + ``class_embeddings``).
        # ``input_name`` keeps the FIRST input for the single-input callers that
        # still read it (baked box engine path is unchanged).
        self.input_name: Optional[str] = None
        self._inputs: dict[str, tuple] = {}
        outputs = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self._engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(
                    f"dynamic dim in tensor {name!r} {shape} — build the "
                    "engine with static shapes (the seg export is static)"
                )
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            dev = self._cu.malloc(nbytes)
            self._context.set_tensor_address(name, dev.value)
            entry = (name, shape, dtype, dev, nbytes)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if self.input_name is None:
                    self.input_name = name
                self._inputs[name] = entry
            else:
                outputs.append(entry)
        if not self._inputs or not outputs:
            raise RuntimeError("TRT engine missing input/output tensors")
        # ONNX contract order: detection rows (ndim 3) before mask protos
        # (ndim 4) — engine binding order is not guaranteed to match.
        self._outputs = sorted(outputs, key=lambda e: len(e[1]))

    @property
    def input_names(self) -> list[str]:
        """All engine input tensor names (order they were bound)."""
        return list(self._inputs.keys())

    def run(self, _output_names, feeds: dict) -> list:
        import ctypes

        # Every engine input must be supplied. Copy each feed to its binding.
        missing = [n for n in self._inputs if n not in feeds]
        if missing:
            raise ValueError(
                f"missing feeds for engine inputs {missing}; "
                f"got {sorted(feeds)}"
            )
        for name, (_n, _shape, dtype, dev, nbytes) in self._inputs.items():
            arr = np.ascontiguousarray(
                np.asarray(feeds[name]).astype(dtype, copy=False)
            )
            if arr.nbytes != nbytes:
                raise ValueError(
                    f"input {name!r} size mismatch: got {arr.nbytes}B, "
                    f"engine expects {nbytes}B"
                )
            self._cu.memcpy(
                dev, arr.ctypes.data_as(ctypes.c_void_p), nbytes, _Cudart.H2D
            )
        if not self._context.execute_async_v3(stream_handle=self._stream.value):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        self._cu.stream_sync(self._stream)
        outs = []
        for oname, oshape, odtype, odev, onbytes in self._outputs:
            host = np.empty(oshape, dtype=odtype)
            self._cu.memcpy(host.ctypes.data_as(ctypes.c_void_p), odev, onbytes, _Cudart.D2H)
            # fp16 engines emit fp16 IO in some builds — normalise to float32
            # so the numpy post-process matches the CPU path bit-for-bit-ish.
            outs.append(host.astype(np.float32, copy=False))
        return outs


class YoloOnnxSegmenter:
    """ONNX-Runtime YOLOE-seg segmenter with numpy/cv2 seg post-process.

    Args:
        model_path: path to the exported ``*-seg.onnx`` (vocab + NMS baked in).
        names: ordered class labels; index ``i`` is class id ``i``. MUST match
            the vocabulary order used at export time.
        input_size: ``(h, w)`` network input. The export is square so only the
            first element is used as the letterbox target.
        providers: onnxruntime execution providers, most-preferred first.
            Defaults to TensorRT → CUDA → CPU; unavailable providers are
            silently dropped by onnxruntime.

    The :class:`onnxruntime.InferenceSession` is created lazily on the first
    :meth:`predict` so constructing a segmenter is cheap and SDK-free (useful
    for tests that mock inference).
    """

    EMBED_INPUT_NAME = "class_embeddings"

    def __init__(
        self,
        model_path: str,
        names: list[str],
        input_size: tuple[int, int] = (640, 640),
        providers: Sequence[str] = DEFAULT_PROVIDERS,
        class_embeddings: Optional[np.ndarray] = None,
        active_n: Optional[int] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.names: dict[int, str] = {i: str(n) for i, n in enumerate(names)}
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.providers = tuple(providers)
        self._session: Any = None
        self._input_name: Optional[str] = None
        # Embeddings-as-input ("embin") mode (vocab-decoupled). When
        # ``class_embeddings`` is given the loaded model is expected to take a
        # ``class_embeddings`` input fed on every predict; the class vocabulary
        # therefore comes from config + computed text PE, NOT from a baked head.
        self._class_embeddings: Optional[np.ndarray] = (
            None
            if class_embeddings is None
            else np.ascontiguousarray(class_embeddings, dtype=np.float32)
        )
        # Number of REAL classes (pad rows beyond this are inert padding). Used
        # by the postprocess pad-slot guard to drop any cls_id >= active_n.
        # Defaults to len(names) so the guard works even without embeddings.
        self._active_n: int = (
            int(active_n) if active_n is not None else len(self.names)
        )
        self._embin: bool = False  # resolved at session creation

    # ── session lifecycle ────────────────────────────────────────────────
    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        # Native-TRT path (2026-06-12): a ``*.engine`` / ``*.plan`` model path
        # selects the TensorRT backend. The container gets TRT by bind-
        # mounting the HOST's /usr/lib/python3.10/dist-packages/tensorrt +
        # CUDA libs (the same contract edge-llm/SLV run with) — no wheels, no
        # writable-layer installs, nothing to redo after a container
        # recreate. ONNX paths keep the onnxruntime CPU path unchanged.
        if self.model_path.endswith((".engine", ".plan")):
            sess = _TrtSession(self.model_path)
            self._session = sess
            self._input_name = sess.input_name
            self._embin = self.EMBED_INPUT_NAME in sess.input_names
            self._validate_embin()
            return
        # Deferred import: onnxruntime is an optional/device dep. Keep it out
        # of module import so the package loads on hosts without the wheel.
        import onnxruntime as ort  # noqa: PLC0415

        providers = list(self.providers)
        _LOG.info("yolo_onnx: session create requested providers=%r", providers)
        # Memory gate (2026-07-14): a GPU EP session pins ~0.8-1.6GB that this
        # box cannot always spare (voice stack pressure; see the
        # onnx_providers note in config.yaml). Below the floor, run this
        # session on CPU rather than risk an OOM that could hit the voice
        # stack. Floor tunable via REBOT_GPU_MIN_AVAIL_MB.
        if any((p[0] if isinstance(p, (list, tuple)) else p) != "CPUExecutionProvider"
               for p in providers):
            import os as _os
            try:
                import time as _time

                def _avail_mb() -> int:
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemAvailable"):
                                return int(line.split()[1]) // 1024
                    return 0

                floor_mb = int(_os.environ.get("REBOT_GPU_MIN_AVAIL_MB", "1200"))
                avail_mb = _avail_mb()
                # The boot prime races the voice stack's own warmup, which
                # transiently depresses MemAvailable; wait out the dip
                # (bounded) before deciding.
                for _ in range(int(_os.environ.get("REBOT_GPU_GATE_RETRIES", "4"))):
                    if not avail_mb or avail_mb >= floor_mb:
                        break
                    _time.sleep(4.0)
                    avail_mb = _avail_mb()
                if avail_mb and avail_mb < floor_mb:
                    _LOG.warning(
                        "yolo_onnx: MemAvailable %dMB < %dMB floor — CPU "
                        "providers this session", avail_mb, floor_mb)
                    providers = ["CPUExecutionProvider"]
            except Exception:
                _LOG.debug("yolo_onnx: memory gate failed", exc_info=True)
        try:
            self._session = ort.InferenceSession(
                self.model_path, providers=providers
            )
        except Exception:
            if providers == ["CPUExecutionProvider"]:
                raise
            _LOG.warning(
                "yolo_onnx: GPU session creation failed — CPU fallback",
                exc_info=True)
            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        in_names = [i.name for i in inputs]
        self._input_name = in_names[0]
        # Embin mode iff the model declares the class_embeddings input. The
        # baked single-input box model has only ``images`` → _embin False →
        # everything below is byte-identical to the legacy path.
        self._embin = self.EMBED_INPUT_NAME in in_names
        # Pick the IMAGE input by name when multi-input (do not assume index 0).
        if self._embin:
            for n in in_names:
                if n != self.EMBED_INPUT_NAME:
                    self._input_name = n
                    break
        self._validate_embin()

    def _validate_embin(self) -> None:
        """Sanity-check that embin mode has the embeddings it must feed."""
        if self._embin and self._class_embeddings is None:
            raise RuntimeError(
                f"model {self.model_path!r} declares a "
                f"{self.EMBED_INPUT_NAME!r} input but no class_embeddings were "
                "supplied to YoloOnnxSegmenter (build them via TextPromptEncoder)"
            )

    # ── inference ─────────────────────────────────────────────────────────
    def predict(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,  # noqa: ARG002 — NMS is baked into the graph
        only_names: Optional[set] = None,
    ) -> list[YoloResult]:
        """Run inference on one BGR frame; return ``[YoloResult]`` (list for
        ultralytics API parity — always length 1).

        ``iou`` is accepted for signature compatibility but unused: the export
        bakes NMS in. We only apply the ``conf`` confidence gate on rows.

        ``only_names`` (optional): a set of class label strings. When given,
        rows whose class name is not in the set are dropped *before* the
        expensive per-row mask assembly. This is purely a cost optimisation —
        the result is identical to running full inference and filtering the
        boxes/masks by name afterwards (``None`` keeps the legacy behaviour).
        """
        self._ensure_session()
        img0 = np.asarray(image_bgr)
        h0, w0 = img0.shape[:2]
        net = self.input_size[0]

        padded, ratio, dw, dh = _letterbox(img0, net)
        # BGR→RGB, HWC→CHW, scale, add batch dim.
        blob = np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        )
        feeds = {self._input_name: blob}
        if self._embin:
            # Vocab-decoupled mode: the class set is supplied at inference time
            # as text PE rows. Fed on EVERY predict (the engine has no baked
            # vocab). Shape/dtype already validated at construction.
            feeds[self.EMBED_INPUT_NAME] = self._class_embeddings
        try:
            outs = self._session.run(None, feeds)
        except Exception:
            # GPU runtime failure mid-session (OOM under pressure): rebuild
            # on CPU and keep the grasp alive; later predicts stay on CPU.
            _LOG.warning(
                "yolo_onnx: session.run failed — rebuilding session on CPU",
                exc_info=True)
            import onnxruntime as ort  # noqa: PLC0415
            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"])
            outs = self._session.run(None, feeds)
        return [
            self._postprocess(
                outs, conf, (h0, w0), ratio, dw, dh, net, only_names
            )
        ]

    # ── numpy/cv2 seg post-process (validated against ultralytics) ─────────
    def _postprocess(
        self,
        outs: list[np.ndarray],
        conf: float,
        orig_shape: tuple[int, int],
        ratio: float,
        dw: float,
        dh: float,
        net: int,
        only_names: Optional[set] = None,
    ) -> YoloResult:
        det = np.asarray(outs[0])[0]      # [300, 38]
        proto = np.asarray(outs[1])[0]    # [32, 160, 160]
        mh, mw = proto.shape[1:]
        proto_flat = proto.reshape(proto.shape[0], -1)
        h0, w0 = orig_shape

        # Letterbox padding edges in 640 space (stripped from each mask).
        top = int(round(dh - 0.1))
        left = int(round(dw - 0.1))
        bot = net - int(round(dh + 0.1))
        rgt = net - int(round(dw + 0.1))

        boxes: list[_Box] = []
        masks: list[np.ndarray] = []
        dropped_pad = 0
        for row in det:
            c = float(row[4])
            # Keep detections AT the threshold (conf is an inclusive floor);
            # only drop strictly-below-threshold rows.
            if c < conf:
                continue
            cls_id = int(round(float(row[5])))
            # Pad-slot safety guard (defense-in-depth for the embin path): the
            # class_embeddings tensor is padded to a fixed width (e.g. 16) with
            # zero rows beyond the real vocabulary. A detection whose cls_id
            # lands on a padded slot has no valid label → drop it by
            # construction so a phantom padded-class detection is impossible.
            # Empirically pad slots are inert (conf 0.0), but the guard removes
            # any reliance on that. active_n == len(names) in the baked path, so
            # this is a no-op for the legacy single-input box engine.
            if cls_id >= self._active_n or cls_id < 0:
                dropped_pad += 1
                continue
            # Class-name gate (optional, before the costly mask assembly). The
            # caller filters to the same name set downstream, so dropping here
            # is behaviour-equivalent but skips the per-row mask work.
            if only_names is not None and self.names.get(cls_id) not in only_names:
                continue
            box640 = row[:4].astype(np.float32)
            coeffs = row[6:]

            # box in original-image space (un-letterbox).
            box_orig = box640.copy()
            box_orig[[0, 2]] -= dw
            box_orig[[1, 3]] -= dh
            box_orig /= ratio
            box_orig[[0, 2]] = box_orig[[0, 2]].clip(0, w0)
            box_orig[[1, 3]] = box_orig[[1, 3]].clip(0, h0)

            # mask logits in proto (160) space, cropped to the box.
            m = (coeffs @ proto_flat).reshape(mh, mw)
            rx, ry = mw / net, mh / net
            cx1 = int(round(max(box640[0] * rx, 0)))
            cy1 = int(round(max(box640[1] * ry, 0)))
            cx2 = int(round(box640[2] * rx))
            cy2 = int(round(box640[3] * ry))
            cropped = np.zeros_like(m)
            cropped[cy1:cy2, cx1:cx2] = m[cy1:cy2, cx1:cx2]

            # upsample to 640, threshold (>0 == sigmoid>0.5), strip padding,
            # resize content to original (NEAREST) — matches ultralytics
            # masks.data consumer path.
            m640 = cv2.resize(cropped, (net, net), interpolation=cv2.INTER_LINEAR)
            bin640 = (m640 > 0.0).astype(np.float32)
            content = bin640[top:bot, left:rgt]
            m_orig = cv2.resize(content, (w0, h0), interpolation=cv2.INTER_NEAREST)

            boxes.append(_Box(box_orig, cls_id, c))
            masks.append((m_orig > 0.5).astype(np.float32))

        if dropped_pad:
            _LOG.debug(
                "yolo_onnx: dropped %d detection(s) on padded class slots "
                "(cls_id >= active_n=%d)",
                dropped_pad,
                self._active_n,
            )

        mask_obj: Optional[_Masks] = None
        if masks:
            mask_obj = _Masks(np.stack(masks, axis=0))
        return YoloResult(
            names=self.names,
            boxes=_Boxes(boxes),
            masks=mask_obj,
            orig_shape=(h0, w0),
        )


__all__ = ["YoloOnnxSegmenter", "YoloResult", "DEFAULT_PROVIDERS"]
