"""Regression: load_config must tolerate unknown YAML keys (template drift).

A base-image ``agent.yaml.tmpl`` (e.g. the v6.1 VoiceArm base) can carry keys
this Config version has dropped — ``energy_gate_enabled`` + 8 siblings. Passing
them straight to ``Config(**fields)`` raised ``TypeError: unexpected keyword
argument`` and crashed the whole agent at boot, which blocked the 3b-ii
prod-faithful verify (2026-05-31). Unknown keys must be dropped (logged), not
fatal.
"""

import logging

from openvoicestream_agent.config import load_config


def _write(tmp_path, text):
    p = tmp_path / "agent.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_unknown_keys_are_ignored_not_fatal(tmp_path, caplog):
    path = _write(
        tmp_path,
        "energy_gate_enabled: true\n"
        "energy_gate_threshold: 0.5\n"
        "mic_device: default\n"
        "reconnect_on_wake: true\n"
        "slv_config:\n"
        "  asr_language: zh\n",
    )
    with caplog.at_level(logging.WARNING):
        cfg = load_config(path)
    # Did not raise; known config still loaded.
    assert cfg.slv_config["asr_language"] == "zh"
    # And it warned about the dropped keys.
    assert "ignoring" in caplog.text
    assert "energy_gate_enabled" in caplog.text


def test_clean_config_still_loads_without_warning(tmp_path, caplog):
    path = _write(tmp_path, "slv_config:\n  asr_language: en\n")
    with caplog.at_level(logging.WARNING):
        cfg = load_config(path)
    assert cfg.slv_config["asr_language"] == "en"
    assert "ignoring" not in caplog.text
