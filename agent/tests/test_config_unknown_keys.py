"""Regression: load_config must tolerate unknown YAML keys (template drift).

A base-image ``agent.yaml.tmpl`` can carry keys this Config version does not
recognise. Passing them straight to ``Config(**fields)`` raised ``TypeError:
unexpected keyword argument`` and crashed the whole agent at boot, which blocked
the 3b-ii prod-faithful verify (2026-05-31). Unknown keys must be dropped
(logged), not fatal.

Note: ``energy_gate_enabled`` / ``reconnect_on_wake`` are now REAL Config
fields again (mic-pump + reconnect-on-wake optimisations cherry-picked back), so
they load as real values and are NOT in the ignored set. The drift-tolerance is
exercised here with keys that are genuinely not on Config
(``energy_gate_threshold``, ``mic_device``).
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
    # Restored real fields load as real values (NOT dropped).
    assert cfg.energy_gate_enabled is True
    assert cfg.reconnect_on_wake is True
    # And it warned only about the genuinely-unknown keys.
    assert "ignoring" in caplog.text
    assert "energy_gate_threshold" in caplog.text
    assert "mic_device" in caplog.text
    # The restored fields must NOT appear in the ignored set.
    assert "energy_gate_enabled" not in caplog.text
    assert "reconnect_on_wake" not in caplog.text


def test_clean_config_still_loads_without_warning(tmp_path, caplog):
    path = _write(tmp_path, "slv_config:\n  asr_language: en\n")
    with caplog.at_level(logging.WARNING):
        cfg = load_config(path)
    assert cfg.slv_config["asr_language"] == "en"
    assert "ignoring" not in caplog.text
