from __future__ import annotations

from pathlib import Path

from ovs_agent.config import load_config


def _cfg():
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "ovs_agent"
        / "apps"
        / "voice_rebot_arm"
        / "config.yaml"
    )
    return load_config(cfg_path)


def test_voice_rebot_arm_stays_resident_server_loop():
    # The arm runs a RESIDENT server-loop session after wake — we intentionally
    # do NOT enable the single-turn wake-command state machine.
    cfg = _cfg()
    assert cfg.pipeline_mode == "wake_word"
    assert cfg.wake_command_single_turn is False
    # Short finalize + gate-driven EOS for terse commands.
    assert cfg.gate_drive_eos is True
    assert float(cfg.asr_final_timeout_s) <= 1.0
    assert float(cfg.gate_eos_delay_ms) > 0
    # Tool trigger guard OFF (deliberate, 2026-06-12): on the real device ASR
    # mis-hears short Mandarin commands and the guard blocked the CORRECT tool
    # the LLM picked ("只回复不干活"); the 4B recovers homophones on its own.
    # The exempt list is retained for a future re-enable: semantic tools plus
    # put_down (blocking it would strand a held object).
    assert cfg.tool_trigger_guard is False
    exempt = set(cfg.tool_trigger_guard_exempt)
    assert {"grasp_object", "search_object", "put_down"} <= exempt


def test_voice_rebot_arm_first_utterance_capture_tuning():
    # 2026-06-13 "第一次没听到" fixes — lock the values so they can't drift back:
    #  * reconnect_on_wake OFF so a hot/healthy wake doesn't drop the first
    #    command in a 6s WS-rebuild window (health/idle reconnect still applies).
    #  * energy gate ON (the path that needs the pre-roll onset recovery).
    #  * wake-tone mic suppression trimmed so it can't eat a closely-following
    #    command's onset.
    cfg = _cfg()
    assert cfg.reconnect_on_wake is False
    assert cfg.energy_gate_enabled is True
    tone = (getattr(cfg, "metadata", {}) or {}).get("wake_tone", {}) or {}
    assert float(tone.get("mic_suppress_tail_ms", 600)) <= 300


def test_voice_rebot_arm_enables_gated_tts_hardening():
    # dup-TTS drop + playback drain are opt-in framework features; the arm app
    # turns them on via config (they stay off everywhere else).
    cfg = _cfg()
    assert float(cfg.tts_drop_duplicate_window_s) > 0
    assert cfg.playback_drain_enabled is True


def test_voice_rebot_arm_keeps_parallel_preamble_prompt():
    # v7 (2026-06-12): the tool PREAMBLE already speaks the acknowledgement
    # ("好的"), so the prompt now forbids the model from writing its own text
    # before/alongside the call — the old "FIRST emit a brief acknowledgement"
    # guidance made it speak twice. The model only confirms AFTER the tool
    # returns.
    cfg = _cfg()
    prompt = " ".join(cfg.system_prompt.split())
    assert "FIRST emit a brief" not in prompt
    assert "do NOT write any text before" in prompt
    # v8: post-tool confirmation text is gone too (round-2 LLM+TTS latency);
    # the completion tone closes the loop.
    assert "EMPTY message" in prompt
