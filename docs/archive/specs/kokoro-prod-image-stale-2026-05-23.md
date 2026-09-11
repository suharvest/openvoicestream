# Radxa kokoro image stale — speaker_id TypeError (hot-patched 2026-05-23)

## Symptom

`POST /tts/stream` on radxa (`openvoicestream-kokoro` container, image
`openvoicestream:rk-kokoro-2026-05-23`) returns `HTTP 200` but only **4
bytes** (the WAV `sample_rate` header, no PCM payload). Container log
shows every request hitting:

```
TypeError: rkvoice_stream.backends.tts.kokoro_rknn.KokoroRKNNBackend.synthesize_stream()
got multiple values for keyword argument 'speaker_id'
```

Effective production outage of TTS on radxa.

## Root cause

The `kwargs.pop("speaker_id", None)` guard at
`server/backends/rk/tts.py:137` (commit `6155ebe`, 2026-05-23 10:32 +0800)
**is not present in the deployed image.**

Build timeline:

| When | Event |
|---|---|
| 2026-05-23 10:27 +0800 (02:27 UTC) | Image `rk-kokoro-2026-05-23` built |
| 2026-05-23 10:32 +0800 | Commit `6155ebe` lands the speaker_id pop fix |

Image was built ~5 minutes before the fix commit. The image's
`/opt/speech/server/backends/rk/tts.py` therefore contains the pre-fix
version where `speaker_id` is both extracted via `resolve_speaker_kwargs`
and re-passed via `**kwargs` to `synthesize_stream(...)`.

Earlier ssh-side validation (matcha + kokoro 17% benchmarks) appeared to
work because the file had been `docker cp`'d in as an ad-hoc hot-patch.
A recent perf-diagnostic / RTF-reconciliation agent restored the
container to stock image state during its cleanup, re-exposing the bug.

| Location | MD5 | State |
|---|---|---|
| Repo `server/backends/rk/tts.py` (HEAD `6155ebe`) | `4961497d910cac5531ceafe35e4f1713` | Fixed |
| Image `/opt/speech/server/backends/rk/tts.py` (before patch) | `6ef45a137e1d062341744c72618cd707` | Broken (pre-fix) |
| Container `/opt/speech/server/backends/rk/tts.py` (after `docker cp`) | `4961497d910cac5531ceafe35e4f1713` | Fixed (matches repo) |

## Hot-patch applied 2026-05-23

```
fleet push radxa server/backends/rk/tts.py /tmp/tts.py
docker cp /tmp/tts.py openvoicestream-kokoro:/opt/speech/server/backends/rk/tts.py
docker restart openvoicestream-kokoro
```

Verification: 10 consecutive `POST /tts/stream {"text":"abc."}` calls
returned `HTTP 200` with `SIZE=251908` bytes each, byte-identical
audio MD5 `13cd893168ab9f917ada5107fbe87d47` across all 10. Startup
log shows `Speech service ready.`, no TypeError after restart.

## Image-level follow-up

`deploy/docker/Dockerfile.rk:74` already does
`COPY server/ /opt/speech/server/` — there is **no special overlay needed**.
Any rebuild of `openvoicestream:rk-kokoro-*` from current `HEAD`
(or any commit ≥ `6155ebe`) will automatically pick up the fix.

**Decision: no urgent rebuild required.** Hot-patched container is
production-stable. Schedule the rebuild whenever the next normal
image refresh ships (e.g. bundled with the next perf/feature pass);
30+ min rebuild cost is not justified for a single one-line patch
that already runs correctly on the live container.

## CAVEAT: hot-patch is volatile

The current radxa kokoro working state depends on the `docker cp`'d
file inside the container's writable layer. **Any of the following will
revert to the broken pre-fix code:**

- `docker compose up -d --force-recreate openvoicestream-kokoro`
- `docker rm openvoicestream-kokoro` followed by recreate
- `docker pull` of a same-tag image that was pushed without the fix
- Host reboot **if** compose project is set to recreate on start
  (currently NOT the case — `restart: unless-stopped` preserves the
  writable layer across host reboot)

Before any such operation, either:

1. Rebuild the image from a commit ≥ `6155ebe`, **or**
2. Re-apply the `docker cp` hot-patch after the recreate.

## Repro for verification

```bash
# Should return SIZE=251908, not SIZE=4
fleet exec radxa -- 'curl -sS -o /tmp/t.wav \
  -w "%{http_code} %{size_download}\n" \
  -X POST http://localhost:8621/tts/stream \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"abc.\"}"'
```

If `SIZE=4`, the hot-patch was lost — re-apply or rebuild image.
