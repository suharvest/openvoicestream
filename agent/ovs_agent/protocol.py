"""V2V WebSocket protocol message-type constants (agent-side copy).

These mirror the server's ``app/core/v2v.py`` (the SLV /v2v/stream protocol).
They are vendored here so ``ovs_agent`` is a self-contained package with NO
import dependency on the SLV server source tree — the agent image ships only
``agent/`` (see docs/BUILD_IMAGES.md), and the server-side ``app/`` is being
renamed to ``server/``, which would break a ``from app.core.v2v import`` here.

⚠️  MUST stay byte-for-byte in sync with ``app/core/v2v.py`` on the server.
    These are stable wire-protocol identifiers; changing one without changing
    the matching server constant silently breaks the session. If you add a new
    message type to the server protocol, add it here too.
"""
from __future__ import annotations

REALTIME_V2_SUBPROTOCOL = "seeed.realtime.v2"

# ── Client → Server JSON message types ──────────────────────────────────
CLIENT_CONFIG = "config"          # initial setup, must be first message
CLIENT_TEXT = "text"              # streaming text input for TTS
CLIENT_ASR_EOS = "asr_eos"        # manually finalize ASR (overrides VAD)
CLIENT_TTS_FLUSH = "tts_flush"    # flush remaining TTS buffer
CLIENT_ABORT = "abort"            # barge-in: cancel current TTS
CLIENT_TOOL_RESULT = "tool_result"        # device returns a remote-tool result
CLIENT_TOOL_ADVERTISE = "tool_advertise"  # device uploads local tool schemas

# Realtime V2 canonical client events.
CLIENT_SESSION_UPDATE = "session.update"
CLIENT_INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
CLIENT_INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
CLIENT_RESPONSE_CREATE = "response.create"
CLIENT_RESPONSE_CANCEL = "response.cancel"
CLIENT_CONVERSATION_ITEM_CREATE = "conversation.item.create"
CLIENT_CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
CLIENT_DIRECT_SPEAK = "x_v2v.response.speak"
CLIENT_CONVERSATION_RESET = "x_v2v.conversation.reset"

# ── Server → Client JSON message types ──────────────────────────────────
SERVER_ASR_PARTIAL = "asr_partial"
SERVER_ASR_ENDPOINT = "asr_endpoint"        # VAD detected end of speech
SERVER_ASR_FINAL = "asr_final"
SERVER_TTS_STARTED = "tts_started"          # first audio frame about to ship
SERVER_TTS_SENTENCE_DONE = "tts_sentence_done"  # one sentence finished
SERVER_TTS_DONE = "tts_done"                # flush complete, no more audio
SERVER_VAD_EVENT = "vad_event"              # server-side VAD speech_start/end
SERVER_ERROR = "error"
SERVER_TOOL_CALL = "tool_call"              # server asks device to run a tool

# Realtime V2 canonical server events.
SERVER_SESSION_CREATED = "session.created"
SERVER_SESSION_UPDATED = "session.updated"
SERVER_INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
SERVER_INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
SERVER_INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
SERVER_INPUT_AUDIO_TRANSCRIPTION_DELTA = (
    "conversation.item.input_audio_transcription.delta"
)
SERVER_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = (
    "conversation.item.input_audio_transcription.completed"
)
SERVER_RESPONSE_CREATED = "response.created"
SERVER_RESPONSE_OUTPUT_AUDIO_DONE = "response.output_audio.done"
SERVER_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA = (
    "response.output_audio_transcript.delta"
)
SERVER_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE = (
    "response.output_audio_transcript.done"
)
SERVER_RESPONSE_DONE = "response.done"
SERVER_RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
SERVER_RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = (
    "response.function_call_arguments.delta"
)
SERVER_RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = (
    "response.function_call_arguments.done"
)
SERVER_CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
SERVER_CONVERSATION_RESET_DONE = "x_v2v.conversation.reset.done"

# ── vad_event "event" field values ──────────────────────────────────────
VAD_EVENT_SPEECH_START = "speech_start"
VAD_EVENT_SPEECH_END = "speech_end"
