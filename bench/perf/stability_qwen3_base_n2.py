#!/usr/bin/env python3
"""Qwen3-TTS Base N=2 HTTP streaming stability and TTFA gate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stability_tts_n2_common import main_entry  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        main_entry(
            backend_label="qwen3_base",
            expected_backend_substr=("trt_edge_llm", "qwen3", "edgellm"),
            fallback_env_var="OVS_TTS_WORKER_CONCURRENCY",
            default_lang="zh",
        )
    )
