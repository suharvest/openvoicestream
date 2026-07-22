"""VoiceRebotArmApp — reBot B601-DM voice agent built on MultiModeApp.

Phase A. Identical wiring to ``apps/voice_arm/app.py`` (TappedAudioIO,
ArmPlugin, OpenWakeWordSource); the only differences are the app name and
that the actuator backend resolves to ``rebot_arm`` via the config's
``metadata.actuator.backend`` (see config.yaml).

Importing this module registers the ``rebot_arm`` actuator builder with the
factory (via ``rebot_actuator`` self-registration), so ArmPlugin's
``create_actuator("rebot_arm", ...)`` resolves.
"""

from __future__ import annotations

import logging

from ovs_agent.apps.multi_mode.app import MultiModeApp
from ovs_agent.audio.devices import resolve_input_index, resolve_output_index
from ovs_agent.audio.tapped_audio_io import TappedAudioIO
from ovs_agent.plugins.actuator_actions import ArmPlugin
from ovs_agent.wake_sources.openwakeword import OpenWakeWordSource

# Importing the actuator module self-registers the "rebot_arm" backend with
# the factory. Re-export the class for convenience / direct import.
from .rebot_actuator import RebotArmActuator  # noqa: F401

# Phase B: camera-guided grasp tool. Importing is cheap (heavy deps —
# onnxruntime, camera SDK — load lazily on the first grasp).
from .dashboard_plugin import ArmDashboardPlugin
from .grasp_plugin import GraspPlugin

logger = logging.getLogger(__name__)


def _resolve_actuator_cfg(meta: dict) -> dict:
    """Build the ArmPlugin config from ``metadata.actuator``.

    Shape (``metadata.actuator``):
        backend: rebot_arm
        config: {channel, repo_root, ...}
        actions_yaml_path: /path/to/actions.yaml
        # optional: observation_port, required_fields,
        #   clear_history_on_turn_end, clear_history_on_tool_change

    Returns the dict ArmPlugin expects:
        {backend, actuator_config, actions_yaml_path, observation_port?, ...}
    """
    actuator = dict(meta.get("actuator", {}) or {})
    backend = actuator.get("backend", "rebot_arm")
    actuator_config = dict(actuator.get("config", {}) or {})
    plugin_cfg: dict = {
        "backend": backend,
        "actuator_config": actuator_config,
    }
    for key in (
        "actions_yaml_path",
        "observation_port",
        "required_fields",
        "clear_history_on_turn_end",
        "clear_history_on_tool_change",
        "disabled_actions",
    ):
        if key in actuator:
            plugin_cfg[key] = actuator[key]
        elif key in actuator_config:
            plugin_cfg[key] = actuator_config[key]
    return plugin_cfg


class VoiceRebotArmApp(MultiModeApp):
    def __init__(self, config) -> None:  # noqa: ANN001
        raw_input_device = getattr(config, "audio_input_device", None) or "reSpeaker"
        raw_output_device = getattr(config, "audio_output_device", None) or "reSpeaker"
        input_device = resolve_input_index(raw_input_device)
        output_device = resolve_output_index(raw_output_device)
        logger.info(
            "VoiceRebotArmApp: resolved audio devices input=%s output=%s",
            input_device, output_device,
        )
        config.audio_input_device = input_device
        config.audio_output_device = output_device
        super().__init__(config)

        # Swap BaseApp's plain AudioIO for the tap-capable variant before
        # run() opens any audio streams.
        logger.info("VoiceRebotArmApp: replacing AudioIO with TappedAudioIO")
        self.audio = TappedAudioIO(
            input_device=config.audio_input_device,
            output_device=config.audio_output_device,
            input_sr=config.audio_input_sample_rate,
            output_sr=config.audio_output_sample_rate,
        )

        meta = getattr(config, "metadata", {}) or {}
        wake_cfg = dict(meta.get("wakeword", {}) or {})
        arm_cfg = _resolve_actuator_cfg(meta)

        # Re-open AudioIO with the mic channel count once known (reSpeaker =
        # 6 channels, exclusive USB device that rejects channels=1).
        mic_channels = int(wake_cfg.get("mic_channels", 1))
        mic_channel_select_raw = wake_cfg.get("mic_channel_select")
        mic_channel_select = (
            None if mic_channel_select_raw in (None, "", "mean")
            else int(mic_channel_select_raw)
        )
        if mic_channels > 1:
            logger.info(
                "Re-opening AudioIO with mic_channels=%d select=%r",
                mic_channels, mic_channel_select,
            )
            self.audio = TappedAudioIO(
                input_device=config.audio_input_device,
                output_device=config.audio_output_device,
                input_sr=config.audio_input_sample_rate,
                output_sr=config.audio_output_sample_rate,
                mic_channels=mic_channels,
                mic_channel_select=mic_channel_select,
            )

        # ArmPlugin owns the CAN/serial port + obs HTTP server. Register it
        # before the wake source so tools are available the moment the first
        # wake fires.
        self.register(ArmPlugin(self, arm_cfg))

        # Phase B: camera-guided grasp. Registered AFTER ArmPlugin so its
        # start() can resolve the ArmPlugin (→ underlying RebotArm). Its
        # ``grasp_object`` tool is registered in setup(), alongside the
        # ArmPlugin action tools. Driven by ``metadata.grasp`` (optional —
        # absent → grasp_object still registers but errors at call time with
        # a clear "missing yolo_model_path" message).
        grasp_cfg = dict(meta.get("grasp", {}) or {})
        self.register(GraspPlugin(self, grasp_cfg))

        # Read-only arm dashboard (frames via GraspPlugin's tee + idle
        # observer; arm state proxied from the observation server). The
        # place_bounds copy lets the topdown panel draw the safe region.
        dash_cfg = dict(meta.get("dashboard", {}) or {})
        dash_cfg.setdefault("place_bounds", grasp_cfg.get("place_bounds"))
        dash_cfg.setdefault(
            "observation_port", (meta.get("actuator", {}) or {}).get("observation_port", 8775)
        )
        self.register(ArmDashboardPlugin(self, dash_cfg))

        self.register(OpenWakeWordSource(
            self,
            model_name=wake_cfg.get("model", "hey jarvis"),
            threshold=float(wake_cfg.get("threshold", 0.5)),
            cooldown_s=float(wake_cfg.get("cooldown_s", 2.0)),
            vad_threshold=float(wake_cfg.get("vad_threshold", 0.0)),
        ))


# CLI loader expects an ``App`` symbol at module top level.
App = VoiceRebotArmApp


__all__ = ["VoiceRebotArmApp", "App"]
