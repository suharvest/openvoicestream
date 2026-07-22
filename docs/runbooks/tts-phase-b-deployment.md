# TTS Phase B Deployment & Reproduction Runbook

Status: **NOT yet baked into image as of 2026-05-23**. The Phase B + pipeline parallelism changes are currently HOT-DEPLOYED (via `docker cp`) on `orin-nx/deploy-speech-1`. A `docker compose down/up` or system reboot **will revert to the pre-Phase-B baseline**.

This document captures everything needed to:
1. Verify the current hot-deployed state matches what's in git
2. Bake Phase B into a new image and ship it to the registry
3. Reproduce the work from scratch on a fresh Orin NX

## §1 What landed

### Main repo (`seeed-local-voice`), branch `main`
- `fe2ba23` Part D disconnect watcher (Starlette cancellation hardening)
- `bc2c29f` + `54d5d00` + `c756b16` specs (audit + patch + ASR follow-up)
- `0fb752e` + `8d6d1b2` `OVS_TTS_STREAM_MAX_WORKERS` env knob (default 2)
- `e67bd37` **sentence pipeline parallelism** (this is the new optimization)

### Fork repo (`tensorrt-edge-llm`), branch `highperf/runtime-service`
9 commits in `64185fa..8a286ce`:
- `64185fa` C++ worker cooperative cancel protocol
- `a7637db` C5 Code2Wav worker mutex
- `e1abd90` C1 codecHiddensBuffer per-request
- `718b233` Phase B stability gate worker mutex (now superseded by the slot fixes, but kept for fallback)
- `fff8a38` C2/C3 talker + CP scratch per-request locals
- `99cf14a` CP output tensors per-request + C5 re-added
- `b2f1ed6` C6 TRT context routed through slot
- `f0c4698` C6 finding #2: slot lifetime across full request (begin/endRequest)
- `f1b91ac` C7 per-request CUDA stream in dispatch path
- `8a286ce` **Option B per-slot pre-allocated scratch tensors** (this is the final binary)

### Built artifact
- `deploy/jetson-workers/qwen3_tts_worker`
- Size: 60793984 bytes
- MD5: `4cf47793532b3951ffa5c77585b0eae4`
- SHA256: `a5ed07ceb5d7e6086aaf314f37e7b85026111f781de42ec5d167ffbed488fc94`
- Built from fork commit `8a286ce` inside `docker.m.daocloud.io/dustynv/l4t-pytorch:r36.4.0` on Orin NX

## §2 Current deployed state on orin-nx

Confirmed 2026-05-23:

| Location | Identity | Source of truth |
|---|---|---|
| Container image | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.14-hotswap` | Pre-Phase-B baseline image (untouched) |
| `/opt/jv-workers/qwen3_tts_worker` in container | md5 `4cf47793...` | HOT-COPIED from `deploy/jetson-workers/qwen3_tts_worker` (post-Option B). Persists only while container is not recreated. |
| `/opt/jv-workers/qwen3_tts_worker.precc1.bak` | md5 `428bcfa4...` | Pre-Phase-B rollback |
| `/opt/jv-workers/qwen3_tts_worker.precc5.bak` | md5 `60762272...` | Pre-C5-mutex rollback |
| `/opt/speech/app/main.py` in container | md5 `09c495cc...` | HOT-COPIED from main repo HEAD (`e67bd37`). |
| `/opt/speech/app/backends/jetson/trt_edge_llm_tts.py` in container | md5 `19e01be0...` | HOT-COPIED from main repo HEAD. |
| `/opt/speech/configs/profiles/jetson-multilang-highperf-nx.json` | md5 `0836ff0a...` | HOT-COPIED. Current default uses the FP16 kv1024 Talker with `OVS_TTS_WORKER_CONCURRENCY=1`; b2/concurrency=2 is opt-in. |

**This is fragile.** Any of:
- `docker compose down deploy-speech-1 && docker compose up -d`
- System reboot
- `docker rm -f deploy-speech-1` (cleanup)

… will revert ALL of the above to the baked image baseline, dropping Phase B + pipeline parallelism.

## §3 To真正固化 (bake into a new image) — RECOMMENDED: overlay path

**TL;DR — 30 seconds on orin-nx:**

```bash
cd /home/harvest/seeed-local-voice-hotswap
DOCKER_BUILDKIT=0 docker build \
    -f deploy/docker/Dockerfile.jetson.tts-phase-b-overlay \
    -t openvoicestream:jetson-v1.15-tts-phase-b .
docker tag openvoicestream:jetson-v1.15-tts-phase-b \
    sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.15-tts-phase-b
```

The overlay Dockerfile (`deploy/docker/Dockerfile.jetson.tts-phase-b-overlay`) takes the production `jetson-v1.14-hotswap` image as base and just COPYs the 4 files that changed:

- `deploy/jetson-workers/qwen3_tts_worker` (Phase B binary)
- `app/main.py` (Part D watcher + pipeline parallelism)
- `app/backends/jetson/trt_edge_llm_tts.py` (cancel counter)
- `configs/profiles/jetson-multilang-highperf-nx.json` (default `OVS_TTS_WORKER_CONCURRENCY=1`; b2/concurrency=2 is opt-in)

Builds in ~30 seconds (vs ~30 minutes for `jetson-release-highperf.sh` full rebuild). All system / pip / submodule layers are inherited from the base image untouched, so there's no risk of dependency drift.

Verified post-build:
```
binary  md5 4cf47793532b3951ffa5c77585b0eae4
main.py md5 09c495cc7527c65f5ca6fb598224ad75
asr.py  md5 19e01be0c8f0ea69d256c59be42bc1ea
profile md5 0836ff0a6da6d33c35f841984ab96294
```

These are byte-identical to the files currently hot-deployed in the live `deploy-speech-1` container, so swapping in this image is a no-op for behaviour and only fixes the persistence gap.

### Full rebuild path (ALTERNATE, only if base image needs to change)

Use this only when you also need to update system/pip dependencies or third_party submodules. Otherwise it's a 30-minute waste.

## §3-bis Full rebuild (legacy reference)

### Prerequisites
- Orin NX (or AGX) build host with at least 4GB free RAM and 20GB free disk
- `docker` with nvidia runtime
- Access to `sensecraft-missionpack.seeed.cn` registry (push credentials)
- All Phase B commits in main + fork pulled to the build host

### Steps

1. **Sync repos to the build host**

   ```bash
   # On orin-nx (or wherever you build):
   cd /home/harvest/seeed-local-voice
   git pull origin main         # main repo with pipeline + Part D
   ```

2. **Verify the binary in deploy/ matches the Phase B build**

   ```bash
   md5sum deploy/jetson-workers/qwen3_tts_worker
   # Expected: 4cf47793532b3951ffa5c77585b0eae4

   sha256sum deploy/jetson-workers/qwen3_tts_worker
   # Expected: a5ed07ceb5d7e6086aaf314f37e7b85026111f781de42ec5d167ffbed488fc94
   ```

   These match the SHA256 in `deploy/jetson-workers/MANIFEST.json:8`.

3. **Bump the image tag and bake**

   Edit `deploy/jetson-release-highperf.sh` to bump:
   - `IMAGE` default: `openvoicestream:jetson-v1.14-hotswap` → `openvoicestream:jetson-v1.15-tts-phase-b`
   - `REGISTRY_IMAGE` default: `…:jetson-v1.14-hotswap` → `…:jetson-v1.15-tts-phase-b`

   Then build:

   ```bash
   cd /home/harvest/seeed-local-voice
   ./deploy/jetson-release-highperf.sh --artifact-set orin-nx-highperf-2026-05-14
   ```

   The script will:
   - Build `openvoicestream:jetson-v1.15-tts-phase-b` from `deploy/docker/Dockerfile.jetson`
   - Bake in `deploy/jetson-workers/qwen3_tts_worker` (Phase B binary)
   - Bake in `app/main.py` (pipeline parallelism + Part D)
   - Bake in `app/backends/jetson/trt_edge_llm_tts.py` (`_WorkerIO` + cancel)
   - Bake in `configs/profiles/jetson-multilang-highperf-nx.json` (default FP16 kv1024 Talker, `OVS_TTS_WORKER_CONCURRENCY=1`)
   - Run roundtrip TTS→ASR loopback + ASR streaming gate verification
   - Re-tag as the registry image

4. **Push to registry**

   ```bash
   ./deploy/jetson-release-highperf.sh --push --artifact-set orin-nx-highperf-2026-05-14
   ```

5. **Update production docker-compose**

   In `/tmp/seeed-local-voice-release/deploy/docker-compose.yml` (or wherever the live one is), update the image tag to `jetson-v1.15-tts-phase-b` and:

   ```bash
   docker compose up -d speech
   ```

   This applies the new image (single-service hot-swap). No full `down` needed.

## §4 Verification gauntlet after image swap

```bash
# 1. Confirm new image is running
docker exec deploy-speech-1 md5sum /opt/jv-workers/qwen3_tts_worker
# Expected: 4cf47793532b3951ffa5c77585b0eae4

docker exec deploy-speech-1 grep OVS_TTS_STREAM_MAX_WORKERS /opt/speech/app/main.py | head -1
# Expected: max_workers=int(os.environ.get("OVS_TTS_STREAM_MAX_WORKERS", "2"))

# 2. Audio MD5 byte-equivalence gate (N=1)
uv run --with requests python bench/perf/stress_cancel_n1.py \
    --scenario early-break --n 5 --host <orin-ip>:8621
# Gate: audio MD5 must equal f515a4376962cca876f21089130d7253

# 3. Multi-sentence pipeline gate
uv run --with requests python bench/perf/multi_sentence_pipeline.py \
    --host <orin-ip>:8621 --runs 3 --text multi
# Gate: 3 runs same audio MD5, total ~9-10s

# 4. N=2 stability gauntlet
for i in $(seq 1 30); do
    uv run --with requests python bench/perf/load_2client_tts.py <orin-ip>:8621 2
done
# Gate: zero "broken-state TTFA" (i.e., TTFA < 100ms) across 30 runs.
# Slow-client TTFA at ~3s is the Orin NX hardware ceiling (expected, not a regression).
```

CUDA error count check:
```bash
docker logs deploy-speech-1 --tail 2000 2>&1 | grep -iE 'illegal|cuda runtime error|state\.read' | wc -l
# Gate: 0
```

## §5 Reproduce the work from scratch (clean Orin NX)

If a teammate gets a fresh Orin NX and wants to reproduce Phase B:

```bash
# 1. Clone repos
cd ~/
git clone -b main <seeed-local-voice-repo> seeed-local-voice
git clone -b highperf/runtime-service https://github.com/suharvest/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git remote add fork https://github.com/suharvest/TensorRT-Edge-LLM.git

# 2. Sanity-check fork commits
git log --oneline 64185fa..8a286ce
# Should see 9 Phase B commits (cancel protocol → C1/C2/C3/C5/C6/C7 → Option B)

# 3. Build the TTS worker binary
docker run --rm --runtime=nvidia \
  -v $HOME/TensorRT-Edge-LLM:/repo \
  -v /usr/lib/python3/dist-packages/pybind11/share/cmake/pybind11:/usr/lib/python3/dist-packages/pybind11/share/cmake/pybind11:ro \
  -v /usr/include/pybind11:/usr/include/pybind11:ro \
  -v /usr/lib/python3/dist-packages/pybind11/include:/usr/lib/python3/dist-packages/pybind11/include:ro \
  docker.m.daocloud.io/dustynv/l4t-pytorch:r36.4.0 \
  bash -lc 'cd /repo/build_container && \
            cmake -DCMAKE_BUILD_TYPE=Release \
                  -Dpybind11_DIR=/usr/lib/python3/dist-packages/pybind11/share/cmake/pybind11 . && \
            make -j6 qwen3_tts_worker'

# Verify
md5sum build_container/examples/omni/qwen3_tts_worker
# Expected: 4cf47793532b3951ffa5c77585b0eae4 (modulo minor compiler determinism — bytes
# may not match exactly across machines; verify functionally via audio MD5 gate below)

# 4. Copy into seeed-local-voice deploy/
cp build_container/examples/omni/qwen3_tts_worker \
   ~/seeed-local-voice/deploy/jetson-workers/qwen3_tts_worker

# 5. Bake into image (see §3 above for the full bake + push + deploy)
```

## §6 Engine artifacts (NOT changed by Phase B)

The TRT engine files are downloaded separately and unchanged by Phase B work:
- `/opt/models/qwen3-edgellm/engines/orin-nx/highperf/talker_w8a16_outputk/` (Talker engine)
- `/opt/models/qwen3-edgellm/engines/orin-nx/highperf/code_predictor/cp_dir/` (CodePredictor)
- `/opt/models/qwen3-edgellm/engines/orin-nx/highperf/code2wav_stateful/` (Code2Wav)

These are the same engines that were running pre-Phase-B. Phase B only changed the **runtime code that uses them**, not the engines themselves. This is why N=1 audio is byte-identical.

## §7 What's NOT shipped (deferred work)

1. **ASR concurrency** — Strategy B (multi-runtime) deferred to AGX upgrade. Today's behavior: second concurrent ASR client queues at Python `_get_asr_executor max_workers=1`. See `an internal engineering spec (not in the public mirror)`.

2. **TTS N=2 slow-client TTFA optimization** — Reaching the ≤1.5× spec gate for the slow client at N=2 requires either pipeline overlap between Code2Wav and next-talker, or AGX hardware. Phase B fast-client achieves 1.4× (within gate); slow-client at ~5× is hardware-bound.

3. **Larger prefetch window** — `OVS_TTS_STREAM_PREFETCH` defaults to 2. Bumping to 3+ requires either more slots per worker (re-eval VRAM) or accepting more inter-sentence contention.

## §8 Cross-reference

- TTS Phase B detailed memory: `~/.claude/projects/-Users-harvest-project-seeed-local-voice/memory/tts_n2_phase_b_stability_landed.md`
- TTS Phase B audit spec: `an internal engineering spec (not in the public mirror)`
- TTS Phase B patch spec: `an internal engineering spec (not in the public mirror)`
- ASR audit spec: `an internal engineering spec (not in the public mirror)`
- Shared tensor audit: `an internal engineering spec (not in the public mirror)`
