"""Lazily imported sherpa-onnx open-vocabulary keyword spotter."""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .compiler import CompiledKeywords


class SherpaKwsBackend:
    """Thin, injectable adapter around ``sherpa_onnx.KeywordSpotter``.

    ``load`` is idempotent so a model is constructed once per source. Keyword
    updates create only a new decoder stream; model weights stay resident.
    """

    def __init__(self, config: dict[str, Any], *, module: Any | None = None) -> None:
        self.config = dict(config)
        self._module = module
        self._spotter = None

    def load(self) -> None:
        if self._spotter is not None:
            return
        module = self._module
        if module is None:
            module = importlib.import_module("sherpa_onnx")
            self._module = module
        required = ("tokens", "encoder", "decoder", "joiner")
        missing = [key for key in required if not self.config.get(key)]
        if missing:
            raise ValueError(f"missing sherpa KWS model setting(s): {', '.join(missing)}")
        kwargs = {
            "tokens": self.config["tokens"],
            "encoder": self.config["encoder"],
            "decoder": self.config["decoder"],
            "joiner": self.config["joiner"],
            # The Python API requires this constructor argument even when all
            # phrases are supplied dynamically by create_stream(). Empty is
            # the official inline-keywords mode.
            "keywords_file": self.config.get("keywords_file", ""),
            "num_threads": int(self.config.get("num_threads", 1)),
            "sample_rate": float(self.config.get("sample_rate", 16000)),
            "feature_dim": int(self.config.get("feature_dim", 80)),
            "max_active_paths": int(self.config.get("max_active_paths", 4)),
            "provider": self.config.get("provider", "cpu"),
            "device": int(self.config.get("device", 0)),
            "keywords_score": float(self.config.get("keywords_score", 1.5)),
            "keywords_threshold": float(self.config.get("keywords_threshold", 0.25)),
            "num_trailing_blanks": int(self.config.get("num_trailing_blanks", 1)),
        }
        self._spotter = module.KeywordSpotter(**kwargs)

    def create_stream(self, compiled: CompiledKeywords):
        self.load()
        try:
            return self._spotter.create_stream(keywords=compiled.keywords)
        except TypeError:
            # Older wheels expose the runtime keyword string positionally.
            return self._spotter.create_stream(compiled.keywords)

    def detect(self, stream, samples: np.ndarray, sample_rate: int) -> str | None:
        data = np.asarray(samples, dtype=np.float32)
        stream.accept_waveform(int(sample_rate), data)
        while self._spotter.is_ready(stream):
            self._spotter.decode_stream(stream)
        result = self._spotter.get_result(stream)
        keyword = result if isinstance(result, str) else getattr(result, "keyword", "")
        if not keyword:
            return None
        self._spotter.reset_stream(stream)
        return str(keyword)


__all__ = ["SherpaKwsBackend"]
