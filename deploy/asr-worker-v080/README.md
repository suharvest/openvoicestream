# Production ASR worker (qwen3_asr_worker) — v0.8.0 deployed variant + first-word-drop fix

This directory pins the **exact C++ source of the streaming ASR worker that runs
in production** (seeed-orin-nx, SLV container `seeed-voice-v080-prefix`), together
with the fix for the streaming "first English word dropped" bug.

## Why this lives here

The worker source has historically been vendored as scattered, **non-git** copies
across staging dirs (≥5 divergent variants exist by md5). The production binary was
hot-built on dev orin-nx from `~/project/v080-worker-build/` (not a git repo), so
the deployed source had no durable home. This commit gives it one.

`qwen3_asr_worker.cpp` here (`md5 13b34dcb`) = the production build source
(`md5 1236d9f2`, orin-nx `v080-worker-build`, git `ee4c4ed`) **+ the strip fix**.
It is NOT the same generation as the older vendored copy at
`third_party/jetson-voice-engine/native/edgellm_voice_worker/qwen3_asr_worker.cpp`
(`md5 5f1620d5`, uses `runStreamingHop`/`parseAsrText`) — do not confuse them.

## The bug (real-machine + isolated experiment 2026-06-15)

reBot demo: every English `asr_final` dropped its first word ("grab the box" →
"the box.", "go home" → "home."). Single-word commands ("wave") lost their only
word → no response; "please wave" worked. **Chinese was unaffected.**

Root cause: `stripLangTag()` greedily consumed a run of ASCII letters after the
`"language "` tag. The model emits `"language <Lang><asr_text><transcript>"`; in the
streaming path the `<asr_text>` separator is a special token decoded to empty, so the
language name and the first transcript word are **glued** ("language EnglishGo home.").
The greedy letter-run ate "EnglishGo" as one language word → returned "home.". CJK
transcripts are non-ASCII, so the letter-run stopped at the language name → Chinese
always worked. The model decoded correctly (`output_text` = "language EnglishGo home.");
this is purely a string-cleaning bug. Offline `POST /asr` uses a different (correct)
parse, which is why offline returned "Go home." while streaming returned "home.".

Decisive isolation: offline `/asr` (bypasses SLV VAD, one-shot) = correct vs streaming
`/v2v` = wrong → narrowed to SLV-vs-worker; then ran the production-identical worker
binary standalone on orin-nx with the FULL audio → raw `output_text` correct but emitted
`text` = "home." → pinned to `stripLangTag`. (Earlier dead-ends: a VAD-onset/preroll
hypothesis on the SLV dispatcher — wrong layer, reverted; a prefix-rollback double-template
hypothesis — analysed the wrong source variant.)

## The fix

`stripLangTag-firstword-fix.patch` — bound the language strip to a **known language-name
set** (longest match) instead of a greedy ASCII-letter run; prefer the `<asr_text>` tag
when present; unify `parseAsrText` onto the same helper. CJK and tag-present cases are
unchanged; unknown/absent tags leave the string untouched.

Verified on orin-nx: English "go home" → "Go home." (was "home."); Chinese
("我们都非常震惊。这位母亲表示。") unchanged, no "language Chinese" residual.

## Build (dev orin-nx — does NOT touch production)

The EdgeLLM runtime static lib is already built; only the worker needs rebuilding.

```bash
# on orin-nx
cp qwen3_asr_worker.cpp ~/project/v080-worker-build/native/edgellm_voice_worker/qwen3_asr_worker.cpp
cd ~/project/v080-worker-build/build_v080 && make qwen3_asr_worker   # ~2-5 min
# -> build_v080/workers/qwen3_asr_worker  (fixed binary md5 8cf0b8df)
```

## Deploy (production SLV — image-baked binary, so overlay)

The worker is baked into the SLV image at `/opt/jv-workers/qwen3_asr_worker` (NOT a
mount). Overlay it onto the running base:

```bash
# context = dir containing the rebuilt qwen3_asr_worker, on seeed-orin-nx
docker build -f Dockerfile.jetson.slv-asrfix-overlay \
  --build-arg BASE=<...:v0.8.0-edgellm-20260611-prefix-voxedgeasrfix> \
  -t <BASE>-asrfix .
# update docker-compose.v080-prefix.yml image: line, then:
docker compose -f docker-compose.v080-prefix.yml up -d seeed-voice-v080-prefix
```

Deployed image: `...voxedgeasrfix-asrfix` (worker md5 `8cf0b8df`). Rollback: set the
compose `image:` back to `...voxedgeasrfix` + `up -d seeed-voice-v080-prefix`
(compose backup `.bak-asrfix` on the device).

Production env that selects the affected path: `OVS_ASR_STREAM_PREFIX=1` +
`OVS_ASR_STREAM_PREFIX_FINAL_ONESHOT=1` (final = one-shot, but its text still passes
through `stripLangTag`, so it was hit).

## TODO

Reconcile the divergent worker variants into a single tracked source-of-truth (this
file is the deployed one; the `third_party/` vendored copy is an older generation).
