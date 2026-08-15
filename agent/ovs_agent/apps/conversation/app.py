"""Minimal conversational app: one chat mode, no domain-specific tools.

All transport, streaming, audio hot-plug, client VAD, playback drain,
barge-in and LLM warmup behavior remains owned by the shared Agent runtime.
Solutions only provide configuration and platform-specific service URLs.
"""
from __future__ import annotations

from ovs_agent.app_mode import ModeManager
from ovs_agent.apps.multi_mode.app import MultiModeApp
from ovs_agent.modes import ChatMode


class ConversationApp(MultiModeApp):
    """The smallest full-duplex LLM voice application shipped by OVS."""

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)

        # MultiModeApp supplies the production pipeline and shared plugins.
        # Replace its broad mode catalog with chat only so this app exposes a
        # stable, unsurprising surface for downstream solutions to reuse.
        self.modes = ModeManager(self._make_mode_ctx)
        self.modes.register(ChatMode())


App = ConversationApp

__all__ = ["App", "ConversationApp"]
