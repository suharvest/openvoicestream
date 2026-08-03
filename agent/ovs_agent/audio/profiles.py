"""reSpeaker device-profile auto-detection.

Different reSpeaker XVF3800 firmware variants expose a different number of
USB-UAC input channels (the 6-channel "Flex C16K6Ch" original vs the
2-channel "4-Mic Array" newer firmware). A hardcoded ``mic_channels`` that
matches one variant makes PortAudio reject the other with
``Invalid number of channels [-9998]`` and the mic pump crash-loops.

This module maps a *connected* device — keyed on its native input-channel
count and ALSA name, both of which ``sounddevice.query_devices`` already
exposes — to the ``mic_channels`` / ``mic_channel_select`` the agent should
open it with. Built-in profiles cover the two known variants; an optional
``audio_profiles.yaml`` in the config dir overrides/extends them without a
code change.

Used by :class:`~ovs_agent.app_base.BaseApp` for EVERY app when
``mic_channels`` resolves to ``auto`` (the default) — see
:func:`resolve_mic_setup`, which is the single entry point apps should call.
An explicit numeric ``mic_channels`` in the YAML/env still wins, so existing
deployments that pin a value are unaffected.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Built-in fallback table — used when audio_profiles.yaml is absent or none of
# its entries match. Most specific first; the empty-match ``fallback`` last.
# Fingerprints captured 2026-06-09:
#   original : USB 2886:001e, 6 in, "reSpeaker Flex XVF3800 C16K6Ch"
#   newer fw : USB 2886:001a, 2 in, "reSpeaker XVF3800 4-Mic Array"
# mic_channel_select=0 is the field-verified working value for BOTH (the 6ch
# unit's env/yaml/runtime log all read 0, despite wiki notes suggesting ch1).
_BUILTIN_PROFILES: list[dict[str, Any]] = [
    {
        "name": "xvf3800-flex-6ch",
        "match": {"input_channels": 6, "name_regex": r"(?i)C16K6Ch|Flex XVF3800"},
        "mic_channels": 6,
        "mic_channel_select": 0,
        # ch0 of the 6ch firmware is quiet → needs heavy makeup so the server
        # silero VAD/ASR sees its trained level range.
        "mic_makeup_gain": 12.0,
    },
    {
        "name": "xvf3800-4mic-2ch",
        "match": {"input_channels": 2, "name_regex": r"(?i)4-Mic Array|XVF3800"},
        "mic_channels": 2,
        "mic_channel_select": 0,
        # the 2ch firmware runs much louder; 12x clips → garbled ASR. 2x is the
        # field-tuned value (10.8.0.170, 2026-06-10).
        "mic_makeup_gain": 2.0,
    },
    {
        "name": "fallback",
        "match": {},
        "mic_channels": "auto",  # open the device's native channel count
        "mic_channel_select": 0,
        # unknown mic → no makeup (1.0 no-op), let the deployment tune it.
        "mic_makeup_gain": 1.0,
    },
]


@dataclass
class MicProfile:
    name: str
    mic_channels: int
    mic_channel_select: int | None
    # Linear mic makeup gain for this firmware. None = profile has no opinion
    # (caller keeps whatever the YAML/config set). The 6ch/2ch firmwares need
    # very different gains, so it belongs with the profile, not a flat config.
    mic_makeup_gain: float | None = None


def _device_signature(device_index: int | None) -> tuple[int, str]:
    """Return ``(max_input_channels, name)`` for the selected input device.

    ``device_index=None`` means "the system default input". Note that
    ``query_devices(None)`` returns the whole DeviceList, not a device — the
    ``kind`` argument is what resolves the default, and it must be passed for
    every app that leaves ``audio_input_device: null``.
    """
    import sounddevice as sd

    info = sd.query_devices(device_index, kind="input")
    return int(info.get("max_input_channels", 0) or 0), str(info.get("name", "") or "")


def _load_yaml_profiles(config_dir: str | None) -> list[dict] | None:
    """Load ``audio_profiles.yaml`` from the config dir, or None if absent."""
    if not config_dir:
        config_dir = os.getenv("CONFIG_DIR", "/opt/seeed/voice_arm/config")
    path = os.path.join(config_dir, "audio_profiles.yaml")
    if not os.path.isfile(path):
        return None
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        profiles = data.get("profiles")
        if isinstance(profiles, list) and profiles:
            logger.info(
                "audio_profiles.yaml: loaded %d profile(s) from %s",
                len(profiles), path,
            )
            return profiles
        logger.warning("audio_profiles.yaml at %s has no 'profiles' list — using built-ins", path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("audio_profiles.yaml load failed (%s) — using built-ins", exc)
    return None


def _match_profile(profiles: list[dict], max_in_ch: int, name: str) -> dict | None:
    """First profile whose every declared match key holds. ``match: {}`` always
    matches (use as the trailing fallback)."""
    for p in profiles:
        m = p.get("match", {}) or {}
        if "input_channels" in m and int(m["input_channels"]) != max_in_ch:
            continue
        rx = m.get("name_regex")
        if rx and not re.search(str(rx), name or ""):
            continue
        return p
    return None


def resolve_mic_profile(device_index: int | None, config_dir: str | None = None) -> MicProfile:
    """Pick the mic channel config for the connected input device.

    Queries the device's native channel count + name, matches it against
    ``audio_profiles.yaml`` (or the built-in table), and resolves
    ``mic_channels`` (``auto`` → device native count) + ``mic_channel_select``.
    Never raises — on any failure it falls back to opening the native count.
    """
    try:
        max_in_ch, name = _device_signature(device_index)
    except Exception as exc:
        logger.warning(
            "audio profile: could not query device [%s] (%s) — defaulting to mono",
            device_index, exc,
        )
        return MicProfile("unknown", 1, None, None)

    profiles = _load_yaml_profiles(config_dir) or _BUILTIN_PROFILES
    matched = _match_profile(profiles, max_in_ch, name)
    if matched is None:
        logger.warning(
            "audio profile: no match for device [%s] %r (%d ch) — opening native count",
            device_index, name, max_in_ch,
        )
        return MicProfile("native", max(1, max_in_ch), 0, None)

    raw_ch = matched.get("mic_channels", "auto")
    ch = max_in_ch if str(raw_ch).strip().lower() == "auto" else int(raw_ch)
    ch = max(1, ch)
    # Clamp the requested channel count to what the device actually offers so a
    # stale profile can never re-introduce the -9998 crash.
    if max_in_ch > 0 and ch > max_in_ch:
        logger.warning(
            "audio profile '%s' requests %d ch but device offers %d — clamping",
            matched.get("name"), ch, max_in_ch,
        )
        ch = max_in_ch

    sel_raw = matched.get("mic_channel_select", 0)
    sel = None if sel_raw in (None, "", "mean") else int(sel_raw)
    if sel is not None and not (0 <= sel < ch):
        logger.warning(
            "audio profile '%s' select=%d out of range for %d ch — using 0",
            matched.get("name"), sel, ch,
        )
        sel = 0

    mg_raw = matched.get("mic_makeup_gain")
    makeup = None
    if mg_raw is not None:
        try:
            makeup = float(mg_raw)
        except (TypeError, ValueError):
            logger.warning(
                "audio profile '%s' mic_makeup_gain=%r not a number — ignoring",
                matched.get("name"), mg_raw,
            )

    logger.info(
        "audio profile '%s' matched device [%s] %r (%d ch) → mic_channels=%d select=%r makeup_gain=%s",
        matched.get("name"), device_index, name, max_in_ch, ch, sel,
        makeup if makeup is not None else "(config)",
    )
    return MicProfile(str(matched.get("name", "?")), ch, sel, makeup)


@dataclass
class MicSetup:
    """How to open the mic, after config ↔ profile precedence is applied."""

    channels: int
    channel_select: int | None
    # Gain the profile wants, or None when the profile has no opinion / the
    # value was pinned in config. Callers only apply it when their own config
    # is still at the 1.0 no-op default.
    makeup_gain: float | None = None
    # Name of the matched profile, or None when ``channels`` was pinned in
    # config and no detection ran.
    profile_name: str | None = None


def _is_auto(value: Any) -> bool:
    return value is None or str(value).strip().lower() in ("", "auto", "none")


def resolve_mic_setup(
    device_index: int | None,
    channels_cfg: Any = "auto",
    channel_select_cfg: Any = None,
    config_dir: str | None = None,
) -> MicSetup:
    """Resolve ``(channels, select, makeup)`` from config + device detection.

    Precedence, per field:

    * ``channels_cfg`` numeric  → pinned, no detection (back-compat: existing
      deployments that hardcode ``MIC_CHANNELS`` behave exactly as before).
    * ``channels_cfg`` ``auto``/empty → :func:`resolve_mic_profile`.
    * ``channel_select_cfg`` numeric → always wins over the profile.
    * ``channel_select_cfg`` ``mean`` → ``None`` (mean downmix across channels).
    * ``channel_select_cfg`` ``auto``/empty → the profile's select when
      detecting, else channel 0.

    Never raises — detection failures degrade to mono.
    """
    sel_is_mean = str(channel_select_cfg).strip().lower() == "mean"
    sel_explicit = None
    if not _is_auto(channel_select_cfg) and not sel_is_mean:
        try:
            sel_explicit = int(channel_select_cfg)
        except (TypeError, ValueError):
            logger.warning(
                "mic_channel_select=%r is not a number — falling back to auto",
                channel_select_cfg,
            )

    if _is_auto(channels_cfg):
        prof = resolve_mic_profile(device_index, config_dir=config_dir)
        channels = prof.mic_channels
        makeup = prof.mic_makeup_gain
        profile_name = prof.name
        select = None if sel_is_mean else (
            prof.mic_channel_select if sel_explicit is None else sel_explicit
        )
    else:
        try:
            channels = max(1, int(channels_cfg))
        except (TypeError, ValueError):
            logger.warning(
                "mic_channels=%r is not a number — treating as auto", channels_cfg
            )
            return resolve_mic_setup(device_index, "auto", channel_select_cfg, config_dir)
        makeup = None
        profile_name = None
        select = None if sel_is_mean else (0 if sel_explicit is None else sel_explicit)

    if select is not None and not (0 <= select < channels):
        logger.warning(
            "mic_channel_select=%d out of range for %d ch — using 0", select, channels
        )
        select = 0
    return MicSetup(channels, select, makeup, profile_name)
