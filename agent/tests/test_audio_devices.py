"""Tests for the /proc/asound/cards-based audio device resolver.

The resolver reads ALSA-native card identity (immune to PortAudio dropping a
busy/late USB device) and hands sounddevice a stable NAME SUBSTRING (e.g.
``XVF3800``) rather than a pinned index. So ``resolve_*`` returns either an int
index (explicit/numeric input) or a name-token string.
"""
from __future__ import annotations

from ovs_agent.audio import devices

# A realistic Jetson Orin NX card list: reSpeaker (real) + HDA/APE (phantom).
_CARDS_WITH_RESPEAKER = [
    (0, "C16K6Ch", "USB-Audio - reSpeaker Flex XVF3800 C16K6Ch"),
    (1, "HDA", "tegra-hda - NVIDIA Jetson Orin NX HDA"),
    (2, "APE", "tegra-ape - NVIDIA Jetson Orin NX APE"),
]
_CARDS_NO_RESPEAKER = [
    (1, "HDA", "tegra-hda - NVIDIA Jetson Orin NX HDA"),
    (2, "APE", "tegra-ape - NVIDIA Jetson Orin NX APE"),
    (3, "Generic", "USB-Audio - Some Generic USB Mic"),
]


def test_explicit_int_index_passthrough():
    assert devices.resolve_input_index(24) == 24


def test_numeric_string_resolves_to_index():
    assert devices.resolve_input_index("24") == 24


def test_explicit_name_substring_passthrough():
    # A non-"auto", non-numeric value is handed straight to sounddevice.
    assert devices.resolve_input_index("reSpeaker") == "reSpeaker"


def test_linux_topology_signature_changes_without_proc_asound(monkeypatch):
    """Docker exposes sysfs/dev hot-plug state without a procfs bind mount."""
    monkeypatch.setattr(devices.os.path, "exists", lambda _path: False)
    state = {
        "/sys/class/sound": (("card0", "../../devices/fallback"),),
        "/dev/snd/by-id": (),
    }
    monkeypatch.setattr(
        devices,
        "_directory_link_signature",
        lambda path: state[path],
    )
    before = devices.linux_audio_topology_signature()

    state["/sys/class/sound"] += (("card3", "../../devices/xvf3800"),)
    state["/dev/snd/by-id"] = (
        ("usb-Seeed_Studio_reSpeaker_XVF3800-00", "../controlC3"),
    )
    after = devices.linux_audio_topology_signature()

    assert before != after
    assert before[0] == after[0] == ()


def test_auto_finds_respeaker_and_returns_stable_token(monkeypatch):
    monkeypatch.setattr(devices, "_read_sound_cards", lambda: _CARDS_WITH_RESPEAKER)
    monkeypatch.setattr(devices.time, "sleep", lambda _s: None)
    # Both directions resolve to the stable USB product token, not a card index.
    assert devices.resolve_input_index("auto") == "XVF3800"
    assert devices.resolve_output_index("auto") == "XVF3800"


def test_auto_never_falls_back_to_a_phantom_card(monkeypatch):
    # No reSpeaker present: must pick the real Generic card, never HDA/APE.
    monkeypatch.setenv("MIC_RESOLVE_TIMEOUT_S", "1")
    monkeypatch.setattr(devices, "_read_sound_cards", lambda: _CARDS_NO_RESPEAKER)
    monkeypatch.setattr(devices.time, "sleep", lambda _s: None)
    resolved = devices.resolve_input_index("auto")
    assert "HDA" not in str(resolved)
    assert "APE" not in str(resolved)
    assert "Generic" in str(resolved)
