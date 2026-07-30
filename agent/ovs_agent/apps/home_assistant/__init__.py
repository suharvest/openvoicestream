"""HomeAssistantApp entry point package."""
from .app import HomeAssistantApp

# Side-effect import: registers the HA tools (list_devices / turn_on / turn_off /
# set_brightness / set_cover_position / get_state / call_service) onto
# default_registry so they exist before ``_advertise_tools_if_server_loop()``
# runs at session open. Without this import the agent advertises an empty tool
# set and the LLM can never control anything.
from . import ha_tools as _ha_tools  # noqa: F401,E402

App = HomeAssistantApp

__all__ = ["HomeAssistantApp", "App"]
