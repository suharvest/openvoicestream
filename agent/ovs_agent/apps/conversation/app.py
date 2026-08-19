"""Minimal conversational app: one chat mode, no domain-specific tools.

All transport, streaming, audio hot-plug, client VAD, playback drain,
barge-in and LLM warmup behavior remains owned by the shared Agent runtime.
Solutions only provide configuration and platform-specific service URLs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ovs_agent.app_mode import ModeManager
from ovs_agent.audio.tapped_audio_io import TappedAudioIO
from ovs_agent.apps.multi_mode.app import MultiModeApp
from ovs_agent.modes import ChatMode
from ovs_agent.wake_sources.runtime_kws import RuntimeKwsSource

logger = logging.getLogger(__name__)


def _load_persisted_phrases(wake: dict, fallback: list[str]) -> list[str]:
    """Load a small root-owned runtime override without mutating shipped YAML."""
    path = Path(str(wake.get("state_path") or "/var/lib/ovs-agent/wakeword.json"))
    try:
        if not path.is_file() or path.stat().st_size > 64 * 1024:
            return fallback
        raw = json.loads(path.read_text(encoding="utf-8"))
        phrases = raw.get("phrases") if isinstance(raw, dict) else None
        if isinstance(phrases, list):
            clean = [str(value).strip() for value in phrases if str(value).strip()]
            if clean:
                return clean
    except Exception:
        logger.warning("ignoring invalid persisted wake-word state: %s", path, exc_info=True)
    return fallback


class ConversationApp(MultiModeApp):
    """The smallest full-duplex LLM voice application shipped by OVS."""

    AUDIO_IO_CLASS = TappedAudioIO

    def __init__(self, config) -> None:  # noqa: ANN001
        wake = dict((getattr(config, "metadata", {}) or {}).get("wakeword", {}) or {})
        kws_enabled = (
            getattr(config, "pipeline_mode", "always_on") == "wake_word"
            and wake.get("backend") == "sherpa_onnx"
        )
        phrases = wake.get("phrases") or []
        if isinstance(phrases, str):
            phrases = [phrases]
        phrases = _load_persisted_phrases(wake, list(phrases)) if kws_enabled else list(phrases)
        if kws_enabled:
            # KWS fires after hearing the full phrase, so the same phrase can
            # already be present in an ASR final. Feed runtime phrases into the
            # shared leak filter as well as the acoustic spotter.
            existing = list(getattr(config, "wake_phrases", []) or [])
            config.wake_phrases = list(dict.fromkeys([*phrases, *existing]))
        super().__init__(config)

        # MultiModeApp supplies the production pipeline and shared plugins.
        # Replace its broad mode catalog with chat only so this app exposes a
        # stable, unsurprising surface for downstream solutions to reuse.
        self.modes = ModeManager(self._make_mode_ctx)
        self.modes.register(ChatMode())

        if kws_enabled:
            model = dict(wake.get("model", {}) or {})
            compiler = dict(wake.get("compiler", {}) or {})
            self.register(
                RuntimeKwsSource(
                    self,
                    phrases=phrases,
                    model_config=model,
                    compiler_config=compiler,
                    cooldown_s=float(wake.get("cooldown_s", 2.0)),
                )
            )


App = ConversationApp

__all__ = ["App", "ConversationApp"]
