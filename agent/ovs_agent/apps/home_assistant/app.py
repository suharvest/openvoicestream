"""HomeAssistantApp — voice control for a Home Assistant smart home.

``MultiModeApp`` plus a Home Assistant REST client wired into the HA tools. In a
server-loop deployment this process is the CLIENT: it owns the microphone and
speaker, advertises the HA tool schemas to the server-side LLM over
``/v2v/stream``, and executes the tool calls the LLM sends back. The LLM itself
never touches the user's home network or credentials.

Configuration comes from ``config.yaml`` (see next to this file); the two values
that must be set are the Home Assistant base URL and a long-lived access token.
"""
from __future__ import annotations

import logging
import os

from ovs_agent.apps.multi_mode.app import MultiModeApp

from . import ha_tools
from .ha_client import HAClient

logger = logging.getLogger(__name__)


class HomeAssistantApp(MultiModeApp):
    """``MultiModeApp`` + a configured Home Assistant client.

    The HA client is built once here and handed to the tool module, so every
    tool call reuses one connection pool and one device cache rather than
    re-discovering the house on each utterance.
    """

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        self.ha: HAClient | None = None
        self._init_ha()

    def _init_ha(self) -> None:
        """Build the HA client and install it into the tools.

        Settings live under ``metadata:`` in config.yaml, NOT at the top level:
        ``load_config`` silently DROPS unknown top-level keys (it only logs
        them), so a top-level ``ha_base_url`` would vanish without an error.
        ``metadata`` is the supported per-app escape hatch, and env substitution
        (``${HA_TOKEN}``) is applied to it like everything else.

        Env vars win over config so a container can be handed a token without
        rewriting a baked config file — and so the token never has to be
        committed.

        Deliberately non-fatal: a bad URL or an expired token must not stop the
        voice agent from booting. The tools each report a connection failure to
        the LLM (which can then say so out loud), and that is far friendlier
        than a process that refuses to start with a stack trace.
        """
        meta = getattr(self.config, "metadata", None) or {}
        base = (os.environ.get("HA_BASE_URL")
                or meta.get("ha_base_url") or "").strip()
        token = (os.environ.get("HA_TOKEN")
                 or meta.get("ha_token") or "").strip()
        if not base or not token:
            logger.warning(
                "[ha] ha_base_url / ha_token not configured — HA tools will "
                "report a configuration error when called"
            )
            return
        extra = tuple(meta.get("ha_extra_domains") or ())
        exclude = tuple(meta.get("ha_exclude_entity_ids") or ())
        self.ha = HAClient(base, token, extra_domains=extra,
                           exclude_entity_ids=exclude)
        ha_tools.configure(self.ha)
        # Ping once at startup so a wrong URL or a stale token shows up in the
        # log now, rather than as a puzzling "无法连接" mid-conversation.
        if self.ha.ping():
            try:
                n = len(self.ha.devices(refresh=True))
                logger.info("[ha] connected to %s — %d controllable devices",
                            base, n)
            except Exception as e:  # noqa: BLE001
                logger.warning("[ha] connected but device listing failed: %r", e)
        else:
            logger.warning(
                "[ha] cannot reach %s with the configured token — check the URL "
                "is reachable FROM THIS DEVICE and that the long-lived token is "
                "still valid", base,
            )


__all__ = ["HomeAssistantApp"]
App = HomeAssistantApp  # for cli.py dynamic loader
