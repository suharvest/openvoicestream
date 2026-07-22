#!/usr/bin/env python3
"""Build-time patch: re-open the ASR stream per utterance in no-VAD
multi_utterance on the /v2v/stream path.

Mirrors repo commit 9b084a5. In no-VAD mode (vad:none) the server opens the
ASR stream lazily on first audio, gated by the one-shot ``asr_started_once``
latch which is never reset — so on a persistent multi_utterance session only
the FIRST utterance ever opens a stream; the 2nd+ finds asr_active=False, no
stream re-opens, accept_audio() is skipped, and the audio is silently dropped
(0 partial / 0 final). Fix: allow the lazy-open in multi_utterance regardless
of the latch (the existing ``not asr_active`` guard prevents a double-open).

Applied by Dockerfile.jetson.v2v-multiturn-overlay on top of a server/-layout
base image (has /opt/speech/server/main.py). Idempotent: a no-op (success) if
the base is already fixed; fails loudly if neither the fixed line nor the
unfixed anchor is found exactly once (base diverged → rebuild against the new
base instead of silently shipping an unpatched image).
"""
import sys

PATH = "/opt/speech/server/main.py"

# Unique anchor: the lazy-open gate's last two conditions. `endpoint_pending`
# pins this to the no-VAD lazy-open block (server/main.py).
OLD = (
    '                            and state["endpoint_pending"] is None\n'
    '                            and not state["asr_started_once"]\n'
    '                        ):'
)

NEW = (
    '                            and state["endpoint_pending"] is None\n'
    '                            # `asr_started_once` is a one-shot latch that is never\n'
    '                            # reset, so in no-VAD mode it would open the ASR stream\n'
    '                            # only for the FIRST utterance — the 2nd+ utterance on a\n'
    '                            # persistent multi_utterance session would find\n'
    '                            # asr_active=False and never re-open, so accept_audio()\n'
    '                            # below is skipped and the audio is silently dropped\n'
    '                            # (0 partial/final). In multi_utterance the session is\n'
    '                            # explicitly kept alive for more turns, so re-open every\n'
    '                            # utterance. (`not asr_active` above prevents double-open.)\n'
    '                            and (multi_utterance or not state["asr_started_once"])\n'
    '                        ):'
)

# Marker that the fix is already present (source-built base, or re-run).
ALREADY = 'and (multi_utterance or not state["asr_started_once"])'


def main() -> None:
    with open(PATH, encoding="utf-8") as f:
        s = f.read()
    if ALREADY in s:
        print(f"v2v multiturn patch: already present in {PATH} — no-op")
        return
    n = s.count(OLD)
    if n != 1:
        sys.exit(
            f"v2v multiturn patch: anchor matched {n} times (expected 1) in "
            f"{PATH} — base image diverged, rebuild overlay against it"
        )
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(s.replace(OLD, NEW))
    print(f"v2v multiturn patch applied to {PATH}")


if __name__ == "__main__":
    main()
