"""Mic channel-layout resolution (config ↔ reSpeaker profile detection).

Guards the precedence rules in ``ovs_agent.audio.profiles.resolve_mic_setup``
and the BaseApp wiring that hands the result to AudioIO — a wrong channel
count makes PortAudio reject the reSpeaker with -9998 and crash-loops the
mic pump, so this is load-bearing for every app, not just the arm ones.
"""
from __future__ import annotations

import pytest

from ovs_agent.audio import profiles as P


@pytest.fixture
def fake_device(monkeypatch):
    """Patch device detection to report a given (channels, name)."""

    def _set(channels: int, name: str):
        monkeypatch.setattr(
            P, "_device_signature", lambda idx, _c=channels, _n=name: (_c, _n)
        )

    return _set


def test_detects_6ch_flex_firmware(fake_device):
    fake_device(6, "reSpeaker Flex XVF3800 C16K6Ch")
    s = P.resolve_mic_setup(None, "auto", "auto")
    assert (s.channels, s.channel_select) == (6, 0)
    assert s.makeup_gain == 12.0
    assert s.profile_name == "xvf3800-flex-6ch"


def test_detects_2ch_4mic_firmware(fake_device):
    """Same physical product, newer firmware, different channel count — the
    whole reason detection exists."""
    fake_device(2, "reSpeaker XVF3800 4-Mic Array")
    s = P.resolve_mic_setup(None, "auto", "auto")
    assert (s.channels, s.channel_select) == (2, 0)
    assert s.makeup_gain == 2.0  # 12x clips this firmware → garbled ASR
    assert s.profile_name == "xvf3800-4mic-2ch"


def test_unknown_device_falls_back_to_native_count(fake_device):
    fake_device(1, "Some USB Headset")
    s = P.resolve_mic_setup(None, "auto", "auto")
    assert s.channels == 1
    assert s.makeup_gain == 1.0  # no-op: unknown mic gets no makeup


def test_explicit_channel_count_pins_and_skips_detection(monkeypatch):
    """Back-compat: deployments that hardcode MIC_CHANNELS keep their value
    and must not pick up a profile's makeup gain."""
    def _boom(idx):  # pragma: no cover - must never be called
        raise AssertionError("detection ran despite an explicit mic_channels")

    monkeypatch.setattr(P, "_device_signature", _boom)
    s = P.resolve_mic_setup(None, 6, 0)
    assert (s.channels, s.channel_select) == (6, 0)
    assert s.makeup_gain is None
    assert s.profile_name is None


def test_explicit_select_overrides_profile(fake_device):
    fake_device(6, "reSpeaker Flex XVF3800 C16K6Ch")
    s = P.resolve_mic_setup(None, "auto", 3)
    assert (s.channels, s.channel_select) == (6, 3)


def test_mean_select_means_downmix(fake_device):
    fake_device(6, "reSpeaker Flex XVF3800 C16K6Ch")
    assert P.resolve_mic_setup(None, "auto", "mean").channel_select is None


def test_out_of_range_select_clamps_to_zero(fake_device):
    fake_device(2, "reSpeaker XVF3800 4-Mic Array")
    assert P.resolve_mic_setup(None, "auto", 5).channel_select == 0


def test_empty_env_value_is_auto(fake_device):
    """``${MIC_CHANNELS:-}`` expands to an empty string, not None."""
    fake_device(2, "reSpeaker XVF3800 4-Mic Array")
    s = P.resolve_mic_setup(None, "", "")
    assert (s.channels, s.channel_select) == (2, 0)


def test_detection_failure_degrades_to_mono(monkeypatch):
    def _raise(idx):
        raise RuntimeError("no sounddevice here")

    monkeypatch.setattr(P, "_device_signature", _raise)
    s = P.resolve_mic_setup(None, "auto", "auto")
    assert s.channels == 1


def test_profile_clamps_to_what_the_device_offers(fake_device):
    """A stale profile must never re-introduce the -9998 crash."""
    fake_device(2, "reSpeaker Flex XVF3800 C16K6Ch")  # name says 6ch, device says 2
    s = P.resolve_mic_profile(None)
    assert s.mic_channels <= 2


def test_base_app_passes_mic_setup_to_audio_io(monkeypatch, fake_device):
    """End of the chain: the resolved layout must reach the AudioIO ctor."""
    from ovs_agent import app_base

    fake_device(6, "reSpeaker Flex XVF3800 C16K6Ch")
    calls = []

    class FakeAudioIO:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(app_base.BaseApp, "AUDIO_IO_CLASS", FakeAudioIO)

    from ovs_agent.config import Config

    cfg = Config(slv_url="ws://localhost:1/ws")
    app = app_base.BaseApp(cfg)

    assert calls[0]["mic_channels"] == 6
    assert calls[0]["mic_channel_select"] == 0
    assert app.mic_setup.profile_name == "xvf3800-flex-6ch"
    # Config left makeup at the 1.0 no-op default → profile fills it in.
    assert cfg.mic_makeup_gain == 12.0


def test_base_app_keeps_explicit_makeup_gain(monkeypatch, fake_device):
    from ovs_agent import app_base
    from ovs_agent.config import Config

    class FakeAudioIO:
        def __init__(self, **kwargs):
            pass

    fake_device(6, "reSpeaker Flex XVF3800 C16K6Ch")
    monkeypatch.setattr(app_base.BaseApp, "AUDIO_IO_CLASS", FakeAudioIO)

    cfg = Config(slv_url="ws://localhost:1/ws", mic_makeup_gain=4.0)
    app_base.BaseApp(cfg)
    assert cfg.mic_makeup_gain == 4.0
