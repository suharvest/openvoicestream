from __future__ import annotations

import ast
from pathlib import Path


def test_reference_gate_is_offline_and_uses_explicit_language_prompts():
    path = Path(__file__).resolve().parents[1] / "bench" / "perf" / "nemotron_asr_hf_reference.py"
    source = path.read_text()
    ast.parse(source)
    assert 'HF_HUB_OFFLINE", "1"' in source
    assert 'TRANSFORMERS_OFFLINE", "1"' in source
    assert '"zh-CN" if entry["lang"] == "zh" else "en-US"' in source
    assert "local_files_only=True" in source
    assert "processor.feature_extractor.sampling_rate" in source
    assert "load_audio(str(audio_path)" in source
