# Deploy: v0.8.0 N=2 (Qwen3-TTS Base + streaming ASR)

Runbook for deploying the **v0.8.0 N>1** voice stack (concurrent ASR streaming +
concurrent Qwen3-TTS Base) on a Jetson Orin Nano / NX **profiling** device.

> ⚠️ **Do NOT deploy this to `seeed-orin-nx`** (the production robot-arm stack)
> or to the shared production `speech-models` volume. These instructions target
> the `orin-nano` / `orin-nx` profiling devices only, using an isolated engine
> volume. The N=2 gates in [`../BENCHMARKS.md`](../BENCHMARKS.md) were run this way.

Verified 2026-06-21. Image `seeed-local-voice:v0.8.0-n1n2-rebake`. See
[`../BENCHMARKS.md`](../BENCHMARKS.md) for the measured numbers and gates.

---

## 1. What you get

- **ASR N=2 streaming** — two concurrent sessions (e.g. one zh + one en) with no
  cross-talk; a 3rd concurrent session is rejected with `4429 too_many_sessions`.
- **TTS N=2** — slot-pool concurrency (independent, staggered-friendly lanes).
  Two variants:
  - **int4 talker** (default for Orin Nano): 245.9 MB talker, ~4 GB system RAM at
    N=2, fits 8 GB and 16 GB.
  - **shared-engine**: 2nd slot adds only +1.6 GB (context/KV, not a 2nd weight
    copy) — saves ~436 MB vs two independent instances.

---

## 2. Prerequisites

- Jetson Orin Nano (8 GB or 16 GB) or Orin NX, JetPack with CUDA 12.6.
- Docker with the NVIDIA runtime (`--gpus all` / `--runtime nvidia`).
- Clocks locked to max (run once after boot):
  ```bash
  sudo ./scripts/setup-performance.sh   # MAXN power mode + locked clocks
  ```
- An **isolated** engine directory on the host, e.g. `/opt/edgellm-v080/engines`
  (never the production `speech-models` volume).

---

## 3. Pull the image

```bash
docker pull seeed-local-voice:v0.8.0-n1n2-rebake
```

This image bundles the v0.8.0 workers + the `libNvInfer_edgellm_plugin.so`
plugin. Reproduction anchors (see BENCHMARKS):

| Artifact | Hash |
|---|---|
| fork tip | `port/qwen3-tts-base-v080-n1n2` @ `7142a30` |
| ASR worker | `5ebd436b` |
| TTS shared-engine worker | `190178f6` |

---

## 4. Get the engines

Two options.

### Option A — pull prebuilt engines from Hugging Face (recommended)

```bash
# int4+fp8 Base talker bundle (recommended on Orin Nano)
hf download harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8 \
  --local-dir /opt/edgellm-v080/engines/qwen3-tts-base

# int4 ASR engine
hf download harvestsu/qwen3-asr-0.6b-int4-v080 \
  --local-dir /opt/edgellm-v080/engines/qwen3-asr
```

Engine hashes to expect: int4 talker (`harvestsu/...base...int4fp8`),
asr-b2 `4122dfcc`, talker-b2 `f7339e02`.

### Option B — mount engines built on-device

If you built engines on the device (`ENABLE_CUTE_DSL=OFF`, KV capped at
`maxKVCacheCapacity=1536` so N=2 fits 8 GB), mount their directory at
`/opt/models/qwen3-tts-base/engines` in the run command below.

---

## 5. Profile: `jetson-edgellm-v080-n2`

This profile is the N=1 `jetson-edgellm-v080-qwen3ttsbase` profile with the
**session-gate triple** flipped so N=2 works out of the box:

| Env key | N=1 | **N=2** | Meaning |
|---|---|---|---|
| `LAZY_TTS` | (unset) | **`1`** | lazy-load the TTS worker so both slots warm independently |
| `OVS_TTS_WORKER_CONCURRENCY` | `1` | **`2`** | second TTS slot-pool lane |
| `OVS_MAX_CONCURRENT_SESSIONS` | `1` | **`2`** | admit 2 sessions; 3rd → `4429 too_many_sessions` |

Everything else (engine paths, plugin path, ASR worker config) is inherited from
`jetson-edgellm-v080-qwen3ttsbase`. Keep `EDGE_LLM_ASR_MAX_CONCURRENT=2` so the
ASR streaming slot-pool also admits 2.

### int4 vs shared-engine

- **int4 talker** (default): point `EDGE_LLM_TTS_TALKER_DIR` at the int4+fp8
  talker engine (245.9 MB). Lowest RAM; recommended for 8 GB Orin Nano.
- **shared-engine**: use the shared-engine worker (`190178f6`) so the 2nd slot
  reuses resident weights (+1.6 GB instead of a full 2nd copy). Use this when you
  want fp16 quality at N=2 on a 16 GB device, or to minimise the marginal cost of
  the 2nd lane.

---

## 6. `docker run` example (N=2 out of the box)

```bash
docker run -d --name seeed-voice-v080-n2 \
  --runtime nvidia --gpus all \
  -p 8621:8000 \
  -e OVS_PROFILE=jetson-edgellm-v080-n2 \
  -e LAZY_TTS=1 \
  -e OVS_TTS_WORKER_CONCURRENCY=2 \
  -e OVS_MAX_CONCURRENT_SESSIONS=2 \
  -e EDGE_LLM_ASR_MAX_CONCURRENT=2 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v /opt/edgellm-v080/engines:/opt/models \
  seeed-local-voice:v0.8.0-n1n2-rebake
```

Notes:
- `-v .../engines:/opt/models` uses an **isolated** host directory — never the
  production `speech-models` volume.
- Drop `HF_ENDPOINT` if you have direct Hugging Face access; some images are not
  compatible with the mirror endpoint.
- This is a `docker run` (not `compose`) on purpose, matching the C7 / baseline
  run command — `compose up` can drift the image tag.

---

## 7. Verify

```bash
# Health
curl -s http://127.0.0.1:8621/health

# Two concurrent ASR streams should both transcribe; a 3rd should get 4429.
# Two concurrent TTS requests should each return audio (slot-pool, staggered).
docker logs seeed-voice-v080-n2 2>&1 | grep -iE 'error|crash|fail|too_many_sessions'
```

Expected: 2 sessions admitted, the 3rd `4429 too_many_sessions`, **0** CUDA
errors, and concurrent audio byte-identical to solo (the N=2 gate criterion).

---

## 8. Rollback

Stop the N=2 container and start the prior N=1 profile:

```bash
docker stop seeed-voice-v080-n2 && docker rm seeed-voice-v080-n2
# then re-run with OVS_PROFILE=jetson-edgellm-v080-qwen3ttsbase (N=1 defaults)
```

---

## Scope reminder

These steps run on `orin-nano` / `orin-nx` profiling devices with an isolated
engine volume. **The `seeed-orin-nx` production robot-arm stack is not part of
this runbook and must not be redeployed here.**
