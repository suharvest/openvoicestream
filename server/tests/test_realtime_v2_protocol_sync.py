from __future__ import annotations

import ast
from pathlib import Path


V2_NAMES = {
    "REALTIME_V2_SUBPROTOCOL",
    "CLIENT_SESSION_UPDATE",
    "CLIENT_INPUT_AUDIO_BUFFER_COMMIT",
    "CLIENT_INPUT_AUDIO_BUFFER_CLEAR",
    "CLIENT_RESPONSE_CREATE",
    "CLIENT_RESPONSE_CANCEL",
    "CLIENT_CONVERSATION_ITEM_CREATE",
    "CLIENT_CONVERSATION_ITEM_TRUNCATE",
    "CLIENT_DIRECT_SPEAK",
    "CLIENT_CONVERSATION_RESET",
    "SERVER_SESSION_CREATED",
    "SERVER_SESSION_UPDATED",
    "SERVER_INPUT_AUDIO_BUFFER_SPEECH_STARTED",
    "SERVER_INPUT_AUDIO_BUFFER_SPEECH_STOPPED",
    "SERVER_INPUT_AUDIO_BUFFER_COMMITTED",
    "SERVER_INPUT_AUDIO_TRANSCRIPTION_DELTA",
    "SERVER_INPUT_AUDIO_TRANSCRIPTION_COMPLETED",
    "SERVER_RESPONSE_CREATED",
    "SERVER_RESPONSE_OUTPUT_ITEM_ADDED",
    "SERVER_RESPONSE_OUTPUT_AUDIO_DONE",
    "SERVER_RESPONSE_DONE",
    "SERVER_RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA",
    "SERVER_RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE",
    "SERVER_CONVERSATION_ITEM_TRUNCATED",
    "SERVER_CONVERSATION_RESET_DONE",
}


def _constants(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in V2_NAMES:
            values[target.id] = ast.literal_eval(node.value)
    return values


def test_server_and_agent_v2_wire_constants_stay_in_sync() -> None:
    root = Path(__file__).resolve().parents[2]
    server = _constants(root / "server/core/v2v.py")
    agent = _constants(root / "agent/ovs_agent/protocol.py")
    assert server.keys() == V2_NAMES
    assert agent.keys() == V2_NAMES
    assert server == agent
