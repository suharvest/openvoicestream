from ovs_agent import Config
from ovs_agent.apps.conversation.app import ConversationApp


def test_conversation_app_exposes_only_chat_mode():
    app = ConversationApp(Config())

    assert [mode["name"] for mode in app.modes.list_all()] == ["chat"]
