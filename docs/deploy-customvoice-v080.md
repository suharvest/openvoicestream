# Deploy: Qwen3-TTS CustomVoice (v0.8.0, int4+fp8)

CustomVoice (CV) Qwen3-TTS-12Hz-0.6B with an **int4 talker + fp8 text_embedding**,
named speakers (e.g. `serena=3066`), served by the v0.8.0 streaming worker
(`qwen3_tts_streaming_worker`). One worker binary serves **both** Base (8-row
prefix) and CustomVoice (9-row) via a runtime-if on `langId` (overlay
`UPSTREAM_PIN=c48c0de`, fork branch `wip/cv-9row-v080-n1n2`).

Profile: [`configs/profiles/jetson-edgellm-v080-customvoice.json`](../configs/profiles/jetson-edgellm-v080-customvoice.json).

## Status — through-service gate PASS (orin-nano, non-production)

| case | HTTP | faster-whisper roundtrip |
|------|------|--------------------------|
| zh (`language=chinese`) | 200 | `你好，很高兴见到你。` (exact) |
| en (`language=english`) | 200 | `Hello! Nice to meet you.` (exact) |
| Base (no `language`) | 200 | `你好，很高兴见到你。` (exact, no regression) |

Worker reached ready, no `tts_manager_start_failed`, 0 CUDA errors. Standalone
worker gate (langId 2055/2050 → 9-row, langId -1 → 8-row, all EOS) also PASS.

## Shipped image

`sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.8.0-n1n2-cv`
digest `sha256:7fa8ba47e6b315516cc12a021ff87994d8a179a5110fce97218520b529a87dde`
(baked-path through-service gate PASS, zh/en/Base exact). NOTE: this pushed image
predates the `profile_owned_env` change, so it still needs the `-e EDGE_LLM_TTS_*`
overrides at runtime. A future image built from current `main` (which has both
`profile_owned_env` and the `*_ENGINE_FILE` fix below) needs `OVS_PROFILE` only.

### Two baked-`/opt` deploy bugs found + fixed (the `/opt` path had never been served before)

1. **Engine-resolver cache miss** — baked engines must ship `.meta.json` sidecars
   (host-correct, e.g. `sm87-trt10.3-jp6.2-cuda12.6`) next to each `*.engine`, or
   the resolver attempts an HF fetch and fails offline. Generate them with the
   resolver's own `_write_meta` when assembling the bundle.
2. **`required_engines[].env_var` clobber** — those `env_var`s must NOT reuse the
   `EDGE_LLM_TTS_TALKER_DIR`/`CP_DIR`/`CODE2WAV_DIR` names the TTS backend reads as
   **directories**: `resolve_all` overwrites them with the engine **file** path →
   `tts_manager_start_failed`. The profile now uses non-clobbering
   `EDGE_LLM_TTS_*_ENGINE_FILE` names (resolver still validates the files; the dir
   vars from the `env` block survive).

## Image

Overlay the CV worker + a **dedicated CV plugin** onto the `:v0.8.0-n1n2-rebake`
base (do not rebuild other workers — the base ASR/MOSS workers + their plugins
are untouched):

| in-image path | md5 | what |
|---|---|---|
| `/opt/jv-workers/qwen3_tts_streaming_worker` | `50d586de…` | CV (Base+CV) streaming worker, from `c48c0de` |
| `/opt/jv-workers/libNvInfer_edgellm_plugin_cv.so` | `e723ffc7…` | CV-matched plugin (dedicated path) |
| `/opt/edgellm-v080/plugins/libNvInfer_edgellm_plugin.so` | `0c058bee…` | ASR + Base TTS plugin — **untouched** |
| `/opt/jv-workers/libNvInfer_edgellm_plugin.so` | `09ad20a8…` | MOSS plugin — **untouched** |

The CV model bundle (int4 talker + fp8 text_embedding + code_predictor +
code2wav + tokenizer) bakes to `/opt/models/qwen3-tts-customvoice/`.

## Three load-bearing gotchas (why the first gate failed)

1. **The plugin must be the CV-matched one.** The streaming worker built from
   `c48c0de` core-dumps at TRT engine deserialize (*"Using an engine plan file
   across different models of devices"*) against any other plugin. It is installed
   at a **dedicated path** so the shared ASR/MOSS plugins stay untouched, and
   `EDGELLM_PLUGIN_PATH` points at it.

2. **`EDGELLM_PLUGIN_PATH` is NOT operator-prefixed.** `profile_loader.py`'s
   `_OPERATOR_KEY_PREFIXES` lists `EDGE_LLM_` (with underscore), not `EDGELLM_`.
   So a container `-e EDGELLM_PLUGIN_PATH=…` is **overwritten** by the profile's
   value. The CV plugin path therefore MUST live in the **profile env** (it does,
   above) — not just `-e`. *(See "Open items" — this asymmetry is a footgun.)*

3. **`EDGE_LLM_TTS_STATEFUL_CODE2WAV=1` is REQUIRED** — the streaming worker uses
   `_synthesize_worker_via_stream` (gated on `stateful_code2wav_enabled()`); `=0`
   takes the non-streaming branch that raises `KeyError: 'output_file'`. (This is
   set in the profile env.)

### Solved: the entrypoint-prestamp shadowing (`profile_owned_env`)

The baked entrypoint pre-stamps `EDGE_LLM_TTS_*` + `OVS_TTS_*`. Those are
operator-prefixed, so `profile_loader` snapshots them as operator-owned and the
profile's values were silently shadowed → talker pointed at the wrong (baked Base)
path → `tts_manager_start_failed`. This is now fixed: the profile declares
`profile_owned_env` (the list of operator-prefixed keys it FULLY owns), and
`apply_profile` lets the profile override the operator snapshot for exactly those
keys. **The CV profile owns its TTS engine wiring, so `OVS_PROFILE` alone drives
it — no `-e` overrides needed.** This is opt-in per profile: profiles that omit
`profile_owned_env` (e.g. the multilang profiles that rely on baked TTS defaults,
and the MOSS profile) are byte-identical to before.

## Run recipe

Bring up the CV image with just the profile (and the slim-image host lib mounts):

```bash
-e OVS_PROFILE=jetson-edgellm-v080-customvoice
# The profile owns EDGE_LLM_TTS_* + OVS_TTS_* via profile_owned_env, and carries
# EDGELLM_PLUGIN_PATH (not operator-prefixed). No other -e overrides needed.
# The :v0.8.0-n1n2-rebake base is a SLIM image: also bind-mount host CUDA/TRT libs
# (/host-cuda, /host-nvidia-libs, /host-libs) per deploy/jetson-release-highperf.sh.
```

> NOTE: this applies to images whose server code includes the `profile_owned_env`
> support in `profile_loader.py`. A CV image built BEFORE that change still needs
> the explicit `-e EDGE_LLM_TTS_*` / `OVS_TTS_*` overrides for the same paths.

## Open items (for review / follow-up)

- **`EDGELLM_` vs `EDGE_LLM_` operator-prefix asymmetry** in `profile_loader.py`
  is a latent footgun (it currently *helps* — the profile carries the CV plugin).
  Optional cleanup: rename to `EDGE_LLM_PLUGIN_PATH` for consistency. Not urgent.
- **Bake the CV bundle to `/opt/models/qwen3-tts-customvoice/`** and re-run the
  through-service gate against the baked `/opt` paths (the PASS above used a
  RO-mounted `/cv-bundle`).
