from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_gate():
    path = Path(__file__).resolve().parents[1] / "bench" / "perf" / "nemotron_asr_offline_gate.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("nemotron_asr_offline_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_transcript_strips_model_language_tag():
    gate = _load_gate()
    assert gate.parse_transcript("Transcript: <zh-CN> 今天天气很好。\n") == "今天天气很好。"
    assert gate.parse_transcript("Transcript: 今天天气很好。 <zh-CN>\n") == "今天天气很好。"
    assert gate.parse_transcript("Transcript: 今天天气 <zh-CN> 很好。\n") == "今天天气很好。"


def test_strip_runtime_language_prefix_handles_qwen_worker_output():
    gate = _load_gate()
    assert gate.strip_runtime_language_prefix("language EnglishHello world") == "Hello world"


def test_parse_rtf_reads_official_benchmark_line():
    gate = _load_gate()
    assert gate.parse_rtf("RTF: 0.042  (23.8x faster than real-time)\n") == 0.042


def test_summarize_keeps_cer_and_wer_separate():
    gate = _load_gate()
    summary = gate.summarize(
        [
            {"lang": "zh", "error_rate": 0.1, "wall_ms": 10.0},
            {"lang": "zh", "error_rate": 0.3, "wall_ms": 30.0},
            {"lang": "en", "error_rate": 0.2, "wall_ms": 20.0},
        ]
    )
    assert summary["zh"]["metric"] == "CER"
    assert summary["zh"]["mean_error_rate"] == 0.2
    assert summary["zh"]["median_cold_process_wall_ms"] == 20.0
    assert summary["en"]["metric"] == "WER"


def test_exact_error_rate_is_levenshtein_not_sequence_matcher():
    gate = _load_gate()
    assert gate.exact_error_rate("a b c", "a x c", "en") == 1 / 3
    assert gate.exact_error_rate("天气很好", "天气真好", "zh") == 1 / 4
