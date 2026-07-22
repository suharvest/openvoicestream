"""Text-prompt encoder for the vocab-decoupled ("embin") YOLOE-seg path.

The embeddings-as-input detector (``yoloe-26s-seg-embin.onnx``) takes a
``class_embeddings [1, P, 512]`` tensor instead of baking the class vocabulary
into the graph. Each row is the final CLIP/PE projection of one class name; the
P slots beyond the real vocabulary are zero padding (inert pad slots).

This module turns a config-supplied list of class names into that padded
tensor, fully torch-free:

    names -> clip_tokenizer.tokenize -> text_encoder_pe.onnx (ORT, CPU) -> PE row

``text_encoder_pe.onnx`` maps ``tokens [1,77] int32 -> class_pe [1,512] float``
(the folded final PE; cosine 1.0 vs the reference ``get_text_pe``). We run it
once per class (batch size is fixed at 1 in the export) and L2-stack the rows
into the ``[1, P, 512]`` tensor the detector expects.

The result is cached to a ``.npy`` keyed by a hash of the ORDERED vocabulary +
the encoder file md5, so flipping a class name or swapping the encoder
invalidates the cache automatically. The same npy is reloaded on the next run.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import Optional, Sequence

import numpy as np

from . import clip_tokenizer

_LOG = logging.getLogger(__name__)

#: Padded slot count for the embin detector's ``class_embeddings`` input. The
#: published ``yoloe-26s-seg-embin.onnx`` fixes this at 16.
DEFAULT_PAD_SLOTS = 16
#: Projection / embedding dimension emitted by ``text_encoder_pe.onnx``.
EMBED_DIM = 512


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TextPromptEncoder:
    """Build a padded ``class_embeddings [1, P, 512]`` tensor from class names.

    Args:
        text_encoder_pe_onnx_path: path to ``text_encoder_pe.onnx``
            (tokens[1,77] int32 → class_pe[1,512] float).
        pad_slots: total slot count P of the detector embeddings input.
        providers: ORT providers. Defaults to CPU only — the text encoder is
            tiny and runs once (then cached), so CPU keeps GPU memory for the
            detector + voice stack.
        cache_dir: where the ``.npy`` cache lives. Must be WRITABLE — defaults
            to ``REBOT_TEXT_PE_CACHE`` or a temp dir (the encoder typically sits
            on a read-only host mount, so a sibling ``.cache`` cannot be written).
    """

    def __init__(
        self,
        text_encoder_pe_onnx_path: str,
        pad_slots: int = DEFAULT_PAD_SLOTS,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        cache_dir: Optional[str] = None,
    ) -> None:
        self.encoder_path = str(text_encoder_pe_onnx_path)
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(
                f"text encoder onnx not found: {self.encoder_path!r}"
            )
        self.pad_slots = int(pad_slots)
        self.providers = tuple(providers)
        # Cache dir must be WRITABLE. The encoder usually lives on a READ-ONLY
        # host mount (e.g. /opt/rebot-models/staging), so a ``.cache`` sibling
        # there fails every write (recompute each boot + a traceback in the
        # logs). Default to a writable temp dir; override with the explicit
        # ``cache_dir`` arg or the ``REBOT_TEXT_PE_CACHE`` env var.
        self.cache_dir = (
            str(cache_dir)
            if cache_dir is not None
            else (
                os.environ.get("REBOT_TEXT_PE_CACHE")
                or os.path.join(tempfile.gettempdir(), "rebot_text_pe_cache")
            )
        )
        self._encoder_md5: Optional[str] = None
        self._session = None
        # Set by encode(): number of REAL (non-pad) class rows.
        self.active_n: int = 0

    # ── cache key ─────────────────────────────────────────────────────────
    def _cache_path(self, names: Sequence[str]) -> str:
        if self._encoder_md5 is None:
            self._encoder_md5 = _file_md5(self.encoder_path)
        key = hashlib.sha1()
        key.update(f"pad={self.pad_slots}\x00".encode("utf-8"))
        key.update(f"enc={self._encoder_md5}\x00".encode("utf-8"))
        for n in names:  # ORDER matters — slot i is class id i
            key.update(str(n).encode("utf-8"))
            key.update(b"\x00")
        return os.path.join(self.cache_dir, f"text_pe_{key.hexdigest()}.npy")

    # ── ORT session (lazy) ────────────────────────────────────────────────
    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort  # noqa: PLC0415 — optional device dep

            self._session = ort.InferenceSession(
                self.encoder_path, providers=list(self.providers)
            )
            ins = self._session.get_inputs()
            self._tok_input_name = ins[0].name
        return self._session

    # ── main entry ────────────────────────────────────────────────────────
    def encode(self, names: Sequence[str]) -> np.ndarray:
        """Return the padded ``class_embeddings [1, pad_slots, 512]`` float32.

        Active rows (one per name, in order) are the L2-normalised PE; the
        remaining rows are zero. Caches the result to ``.npy`` keyed by
        (ordered vocab + encoder md5); reloads it if present.
        """
        names = [str(n) for n in names]
        self.active_n = len(names)
        if self.active_n == 0:
            raise ValueError("TextPromptEncoder.encode: empty class names")
        if self.active_n > self.pad_slots:
            raise ValueError(
                f"{self.active_n} class names exceed pad_slots={self.pad_slots}"
            )

        cache_path = self._cache_path(names)
        if os.path.exists(cache_path):
            try:
                arr = np.load(cache_path)
                if arr.shape == (1, self.pad_slots, EMBED_DIM):
                    _LOG.debug("TextPromptEncoder: cache hit %s", cache_path)
                    return arr.astype(np.float32, copy=False)
                _LOG.warning(
                    "TextPromptEncoder: cache %s wrong shape %s, recomputing",
                    cache_path,
                    arr.shape,
                )
            except Exception:  # pragma: no cover — corrupt cache → recompute
                _LOG.warning(
                    "TextPromptEncoder: failed to load cache %s, recomputing",
                    cache_path,
                    exc_info=True,
                )

        sess = self._ensure_session()
        out = np.zeros((1, self.pad_slots, EMBED_DIM), dtype=np.float32)
        for i, name in enumerate(names):
            tokens = clip_tokenizer.tokenize(
                [name], context_length=77, truncate=True
            ).astype(np.int32)
            pe = sess.run(None, {self._tok_input_name: tokens})[0]
            pe = np.asarray(pe, dtype=np.float32).reshape(-1)
            if pe.shape[0] != EMBED_DIM:
                raise RuntimeError(
                    f"text encoder returned dim {pe.shape[0]}, expected {EMBED_DIM}"
                )
            norm = float(np.linalg.norm(pe))
            if norm > 0.0:
                pe = pe / norm
            out[0, i] = pe

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cache_path, out)
            _LOG.debug("TextPromptEncoder: cached %s", cache_path)
        except Exception as exc:  # pragma: no cover — best-effort cache
            # Read-only cache dir is an expected, non-fatal condition (the
            # embeddings were already computed in-memory) — one concise line,
            # no traceback noise.
            _LOG.warning(
                "TextPromptEncoder: cache write skipped (%s): %s",
                cache_path,
                exc,
            )
        return out


__all__ = ["TextPromptEncoder", "DEFAULT_PAD_SLOTS", "EMBED_DIM"]
