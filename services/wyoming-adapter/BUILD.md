# Wyoming adapter — build & run notes

Exposes an existing **seeed-local-voice** instance to Home Assistant's Assist
pipeline over the [Wyoming protocol](https://github.com/rhasspy/wyoming)
(newline-delimited JSON header + optional binary payload over TCP).

Two Wyoming servers, one per program (HA's Wyoming integration is configured
per `host:port`, and one config entry = one program):

| Program | Default port | HA entity |
|---|---|---|
| STT (`asr`) | `10300` | `stt.seeed_local_voice` |
| TTS (`tts`) | `10200` | `tts.seeed_local_voice` |

**This is a standalone service — it is deliberately NOT baked into the voice
image.** It only speaks HTTP/WS to the voice service, so the "we-are-the-brain"
shape (our own v2v pipeline) and the "HA-is-the-brain" shape (Assist calls our
ASR/TTS) can run side by side without touching the voice image.

## Upstream endpoints used

| Wyoming side | Voice-service side |
|---|---|
| `transcribe` + `audio-*` → one `transcript` | `WS /asr/stream?language=..&sample_rate=..&vad=none` — raw int16 mono PCM in, empty binary frame = EOS, JSON `{"type":"final","text":..}` out |
| `synthesize*` → `audio-*` | `POST /tts/stream` — body `{"text": ...}`, response = `uint32 LE sample_rate` header then raw int16 mono PCM |

Verified live against radxa (`http://100.77.150.16:8621`, 2026-07-30):
`/tts/stream` returns `content-type: application/octet-stream`, sample-rate
header **16000**, width 2, channels 1. The Wyoming `audio-start` is populated
from that in-band header, never from a hard-coded constant — a wrong rate is
chipmunk/slow-motion audio that a log will not show you.

## Two protocol facts this adapter is built around

1. **HA does not consume streaming transcripts.** `homeassistant/components/wyoming/stt.py`
   never reads `supports_transcript_streaming` and breaks its read loop on the
   first `Transcript`. We therefore advertise
   `supports_transcript_streaming: false` and emit exactly one final
   `Transcript`. Upstream partials are logged at DEBUG only.
2. **HA does consume streaming synthesis, and repeats the text.** With
   `supports_synthesize_streaming: true`, `wyoming/tts.py` sends
   `synthesize-start` → `synthesize-chunk`* → **a full `synthesize` carrying the
   COMPLETE text** ("for compatibility", per its own comment) → `synthesize-stop`.
   Treating that trailing `Synthesize` as work would synthesize and play every
   reply **twice**. `wyoming_slv/tts.py` ignores it once any clause has been
   streamed, and only falls back to it if the stream produced nothing. A bare
   `Synthesize` with no start/stop (flag-false clients) still works.

Streaming synthesis is done **per clause** (`wyoming_slv/clause.py`, sizing
mirrored from `server/core/v2v.py::LowLatencyTTSBuffer`: CJK min 15 / target 24
/ max 40, Latin 24 / 48 / 80) so audio starts flowing before `synthesize-stop`.

`audio-chunk` payloads from HA are **not** guaranteed to be a whole number of
int16 samples; odd-length frames make `/asr/stream` raise
`buffer size must be a multiple of element size` and the utterance returns
empty. The STT handler keeps the straggler byte and forwards only 2-byte
aligned blocks.

## Run locally (uv)

```bash
cd services/wyoming-adapter
uv sync
SLV_BASE_URL=http://100.77.150.16:8621 uv run python -m wyoming_slv --debug
```

Env / flags:

| Env | Flag | Default |
|---|---|---|
| `SLV_BASE_URL` | `--base-url` | `http://127.0.0.1:8621` |
| `WYOMING_HOST` | `--host` | `0.0.0.0` |
| `WYOMING_STT_PORT` | `--stt-port` | `10300` |
| `WYOMING_TTS_PORT` | `--tts-port` | `10200` |
| `WYOMING_LANGUAGES` | `--languages` | `zh,en` |
| `WYOMING_STT_MODEL` | `--stt-model` | `slv-asr` |
| `WYOMING_TTS_VOICE` | `--tts-voice` | `slv-default` |
| `WYOMING_DEBUG` | `--debug` | off |

## Docker

```bash
docker build -t seeed-local-voice:wyoming-adapter-v0.1.0 services/wyoming-adapter
docker run -d --name wyoming-adapter -p 10300:10300 -p 10200:10200 \
  -e SLV_BASE_URL=http://100.77.150.16:8621 \
  seeed-local-voice:wyoming-adapter-v0.1.0
```

Or the bundled compose file (does not touch any existing compose project):

```bash
SLV_BASE_URL=http://100.77.150.16:8621 \
  docker compose -f services/wyoming-adapter/docker-compose.wyoming.yml up -d
```

Compose block if you prefer to fold it into an existing file:

```yaml
  wyoming-adapter:
    image: seeed-local-voice:wyoming-adapter-v0.1.0
    build: ./services/wyoming-adapter
    restart: unless-stopped
    environment:
      SLV_BASE_URL: http://voice:8621     # or host.docker.internal / a Tailscale IP
      WYOMING_STT_PORT: 10300
      WYOMING_TTS_PORT: 10200
    ports:
      - "10300:10300"
      - "10200:10200"
```

## Register in Home Assistant

UI: **Settings → Devices & Services → Add Integration → Wyoming Protocol**,
then add the two endpoints separately (host + port). Or over the REST API:

```bash
TOKEN=<long-lived token>
for PORT in 10300 10200; do
  FLOW=$(curl -s -X POST -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' -d '{"handler":"wyoming"}' \
    http://127.0.0.1:8123/api/config/config_entries/flow | jq -r .flow_id)
  curl -s -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
    -d "{\"host\":\"host.docker.internal\",\"port\":$PORT}" \
    http://127.0.0.1:8123/api/config/config_entries/flow/$FLOW
done
```

If HA runs in Docker and the adapter on the host, the host HA must use is
`host.docker.internal` (verify from inside the container:
`docker exec homeassistant getent hosts host.docker.internal`), not `127.0.0.1`.

Then pick the two entities in an Assist pipeline
(**Settings → Voice assistants → Add assistant**, speech-to-text =
`seeed-local-voice`, text-to-speech = `seeed-local-voice`). That selection is
also API-drivable via the websocket API (`assist_pipeline/pipeline/create`).

## Verify

```bash
cd services/wyoming-adapter
uv run python verify_protocol.py            # describe/info flags, STT with a real WAV, streaming TTS
```

`verify_protocol.py` asserts: both capability flags; a non-empty transcript for
`bench/perf/corpus/short/zh_short_01.wav`; exactly **one** `audio-start`;
RMS > 0 (byte-non-empty is not proof of audio); and a duration-per-character
bound that a double synthesis would blow through.
