#!/usr/bin/env python3
"""Build-time patch: make /asr/stream's server-VAD path multi-utterance.

Applied by Dockerfile.jetson.asr-multiturn-overlay on top of a server/-layout
base image whose ``_asr_stream_backend`` still `break`s after a server-VAD
endpoint (closing the socket per utterance). Mirrors the repo fix in
server/main.py. Fails loudly if the anchor isn't found exactly once (base
diverged → rebuild the overlay against the new base instead of silently no-op).
"""
import sys

PATH = "/opt/speech/server/main.py"

OLD = '''                        await ws.send_json(payload)
                    except Exception:
                        # Client may have disconnected during a slow finalize
                        # (e.g. TRT-EdgeLLM on Jetson). Nothing to send to.
                        pass
                    break'''

NEW = '''                        await ws.send_json(payload)
                    except Exception:
                        # Client gone during a slow finalize (e.g. TRT-EdgeLLM
                        # on Jetson) — nothing to send to, close out.
                        break
                    # Multi-utterance: reset the ASR stream + VAD and KEEP the
                    # socket open for the next utterance (was: break = close per
                    # utterance, forcing per-sentence reconnects). Complete finals
                    # via prepare_finalize+finalize (not force_endpoint).
                    try:
                        _old_close = getattr(stream, "close", None)
                        if _old_close is not None:
                            _old_close()
                    except Exception:
                        logger.exception("ASR VAD endpoint: stream close raised")
                    stream = asr_be.create_stream(language=language)
                    try:
                        vad_session.reset()
                    except Exception:
                        logger.debug("VAD reset after endpoint raised", exc_info=True)
                    continue'''


def main() -> None:
    with open(PATH, encoding="utf-8") as f:
        s = f.read()
    n = s.count(OLD)
    if n != 1:
        sys.exit(f"ASR multiturn patch: anchor matched {n} times (expected 1) "
                 f"in {PATH} — base image diverged, rebuild overlay against it")
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(s.replace(OLD, NEW))
    print(f"ASR multiturn patch applied to {PATH}")


if __name__ == "__main__":
    main()
