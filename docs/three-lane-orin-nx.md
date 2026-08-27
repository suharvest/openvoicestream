# Three concurrent voice sessions on Orin NX (Qwen3-ASR b3 + Matcha n3)

Runbook for raising the Orin NX 16GB voice stack from one lane to three, for the
smart_warehouse Tier 2B deployment where a single reComputer Super J4012 runs the
warehouse system, face recognition and the voice stack.

## Why a rebuild is needed at all

The ASR lane count is compiled into the TensorRT engine. `qwen3_asr_worker.cpp`
clamps the requested lanes to the engine's physical batch:

```
int32_t const physBatch = runtime->maxSessionBatchSize();
int32_t asrMax = args.maxSlots;
if (asrMax > physBatch) asrMax = physBatch;
```

(`deploy/asr-worker-v080/qwen3_asr_worker.cpp:1644-1650`)

A `b1` engine therefore serves one lane no matter what the profile declares, and
the degradation is silent. As of this document `configs/profiles/jetson-edgellm-v091-matcha.json`
declares `max_concurrent_sessions: 2` while pointing `EDGE_LLM_ASR_ENGINE_DIR` at
`asr_thinker_full_int4_b1` — that profile serves **one** ASR lane.

TTS is different. `matcha_trt` declares `supports_parallel=True` with
`max_concurrent` = the size of a pre-allocated context pool, engine weights
shared, one context borrowed per call
(`docs/specs/concurrency-capability-framework.md` Section 3). A third TTS lane is
`OVS_TTS_STREAM_MAX_WORKERS=3`, not a new engine.

So: rebuild the ASR engine, configure the TTS pool.

## Prerequisites

- An Orin NX with JetPack 6.2 (fleet: `orin-nx`, `seeed-orin-nx`, `recomputer-desktop`)
- A patched TensorRT-Edge-LLM tree at the pin the builder enforces
  (`EXPECTED_UPSTREAM_PIN` in `build-engines-for-device.sh`)
- `HF_ENDPOINT=https://hf-mirror.com` exported in a **non-login** shell
  (`bash -c 'echo $HF_ENDPOINT'` must print it)
- ~40GB free disk for the export + engine build

## Step 1 — build the b3 engine

`build-engines-for-device.sh` now takes the batch set as an env var. The default
is `1 2`, so existing builds are unchanged.

```bash
EXPORT_ROOT=/work/build/export MODEL_ROOT=/work/build/models \
UPSTREAM=/work/build/upstream \
ASR_MAX_BATCH_SIZES="1 2 3" \
bash third_party/jetson-voice-engine/engine-overlay-v010/build-engines-for-device.sh qwen3-asr
```

Produces `${EXPORT_ROOT}/qwen3-asr/thinker-b3/llm.engine` alongside b1 and b2.
Record the `_meta` output (size + sha256) — it is the provenance you will pin.

## Step 2 — publish and pin

1. Upload `thinker-b3/` to `harvestsu/qwen3-asr-0.6b-jetson-artifacts` as
   `engines/asr_thinker_full_int4_b3`.
2. Add the new revision to `deploy/artifacts/v091-release-lock.json` under
   `model_artifacts.qwen3-asr-0.6b`.

Step 2 is not optional: `tests/test_v091_runtime_profile_packaging.py::test_every_v091_profile_model_is_pinned_by_release_lock`
fails any `jetson-edgellm-v091*` profile whose revision is not in the lock. That
gate is why the n3 profile is not in this commit.

3. Replace `revision: PENDING_B3_ENGINE_UPLOAD` in
   `configs/leaves/qwen3-asr-nx-v091.yaml` (`asr.qwen3_asr_v091.orin-nx.n3`) with
   the real commit.

## Step 3 — add the n3 profile

Copy `configs/profiles/jetson-edgellm-v091-matcha.json` to
`jetson-edgellm-v091-matcha-n3.json` and change exactly these:

| Key | From | To |
|-----|------|-----|
| `name` | `jetson-edgellm-v091-matcha` | `jetson-edgellm-v091-matcha-n3` |
| `model_artifacts[qwen3-asr].revision` | `9a82e1ae…` | the b3 revision |
| `model_artifacts[qwen3-asr].required_files` | `engines/asr_thinker_full_int4_b1` | `engines/asr_thinker_full_int4_b3` |
| `env.EDGE_LLM_ASR_ENGINE_DIR` | `…/asr_thinker_full_int4_b1` | `…/asr_thinker_full_int4_b3` |
| `env.OVS_TTS_STREAM_MAX_WORKERS` | `2` | `3` |
| `env.OVS_MAX_CONCURRENT_SESSIONS` | (unset) | `3` |
| `max_concurrent_sessions` | `2` | `3` |
| `required_engines[EDGE_LLM_ASR_ENGINE_FILE].engine_path` | `…b1/llm.engine` | `…b3/llm.engine` |

## Step 4 — measure before believing

Memory is the binding constraint, and there is a recorded failure to respect:
`configs/profiles/jetson-edgellm-v091-n2.json` carries the note *"Do not use for
ASR+TTS co-residency on a 16 GB Orin NX: loading both N=2 workers caused kernel
OOM"*. That was Qwen3-TTS N=2, which is much heavier than Matcha, but it is proof
that 16GB runs out on this device.

Tier 2B additionally co-locates the warehouse container and the face recognition
container (TensorRT, `face-rec-api:v1.1-jetson`) on the same J4012.

Measure on hardware, with the face stack running:

```bash
# three concurrent ASR sessions, per-session timelines, cross-talk check
python3 bench/perf/v2v_concurrency_probe.py --url ws://<j4012>:8621 --wav clip.wav --n 3
# a 4th session must be rejected with 4429 too_many_sessions
```

Record and update, replacing the current placeholders:

- `configs/leaves/qwen3-asr-nx-v091.yaml` — the n3 leaf has no `resources:` block yet
- `configs/leaves/matcha-tts.yaml` — `tts.matcha_trt.orin.n3.peak_unified_mb` is a
  guess (1800), and the n2 value it was derived from is itself `TBD-measure`
- `docker-compose-jetson-voice.yml` in sensecraft-solutions sets `mem_limit: 8g`
  on the speech container; three lanes will need this raised

Gate criteria, same as the N=2 qualification in `docs/deploy-v090.md`: three
concurrent sessions transcribe with no cross-talk, a 4th gets 4429, zero CUDA
errors, and peak unified RAM leaves headroom with face-rec resident.

## Step 5 — switch the solution over

In `sensecraft-solutions`, `solutions/smart_warehouse/assets/docker/docker-compose-jetson-voice.yml`:

- `OVS_PROFILE`: `jetson-edgellm-v091-matcha` → `jetson-edgellm-v091-matcha-n3`
- `mem_limit` / `memswap_limit`: raise to the measured figure from Step 4

Then write the measured lane count into `guide.md` / `guide_zh.md` for the Tier 2B
preset. Until Step 4 produces numbers, that documentation should not claim three
lanes — the preset currently ships one.
