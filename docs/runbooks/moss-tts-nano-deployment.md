# MOSS-TTS-Nano Deployment Runbook

**Status:** Production ready on two paths as of 2026-05-24.

| Path | Profile | TTFA (Orin NX) | When to use |
|---|---|---|---|
| C++ TRT (default) | `jetson-moss-tts-nano-trt` | **~157 ms** | Production. Native binary, FP32/FP16 KV via TensorRT. |
| ORT Python (fallback) | `jetson-moss-tts-nano` | ~3000 ms | Fallback if the C++ binary fails to load or you need CPU-only deterministic path. |

The C++ TRT path was unblocked on 2026-05-24 by a KV-buffer dtype ABI fix — see `docs/specs/moss-tts-nano-kv-dtype-abi-fix.md`.

## Quick start (C++ TRT, default)

```bash
OVS_PROFILE=jetson-moss-tts-nano-trt docker-compose -f deploy/docker-compose.yml up
```

Quick start (ORT fallback):

```bash
OVS_PROFILE=jetson-moss-tts-nano docker-compose -f deploy/docker-compose.yml up
```

Then:

```bash
curl -s -X POST http://localhost:8000/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，今天天气真不错"}' \
  -o /tmp/out.wav
```

Endpoints (inherited from `app/main.py`):

| Endpoint | Purpose |
|---|---|
| `POST /tts` | Sync synth → WAV bytes |
| `POST /tts/stream` | Streaming PCM chunks |
| `POST /tts/clone` | Voice clone via reference WAV |
| `POST /tts/clone/stream` | Streaming voice clone |
| `GET /tts/capabilities` | Backend capability check |
| `GET /tts/speakers` | Speaker preset list |

## Components

| Layer | Artifact | Path |
|---|---|---|
| C++ worker | `moss_tts_nano_worker` (561 KB) | `deploy/jetson-workers/moss_tts_nano_worker` → image `/opt/jv-workers/` |
| Python backend | `MossTtsNanoBackend` | `app/backends/jetson/moss_tts_nano.py` |
| Registry | `jetson.moss_tts_nano` | `app/core/tts_backend.py:135` |
| Profile | 6 required engines + env vars | `configs/profiles/jetson-moss-tts-nano.json` |
| Engine bundle | 5 TTS plan + 1 codec plan + tokenizer + meta | host `/opt/models/moss-tts-nano/` (1.42 GB) → container `/opt/models/` (via `speech-models` volume) |
| Engine build script | trtexec recipe | `scripts/build_moss_tts_engines.sh` |
| ONNX patches | KV paged FP16 + codec If-rank fix | `scripts/patches/moss-tts-nano-paged-kv-fp16.patch`, `scripts/fix_moss_codec_onnx_if_rank.py` |

## Setting up engines on a new Jetson

On a Jetson with TRT 10.3 + CUDA 12.6:

```bash
# 1. Get the ONNX exports (one-time, on a dev GPU machine with PyTorch):
git clone https://github.com/OpenMOSS/MOSS-TTS-Nano.git
cd MOSS-TTS-Nano
git apply path/to/seeed-local-voice/scripts/patches/moss-tts-nano-paged-kv-fp16.patch
huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano-100M --local-dir ~/models/MOSS-TTS-Nano-100M
python onnx/export_hf_to_tts_onnx.py \
  --checkpoint-path ~/models/MOSS-TTS-Nano-100M \
  --output-dir ~/models/moss-tts-nano-onnx \
  --opset 17

# 2. Get codec ONNX, apply If-rank fix:
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX --local-dir ~/models/MOSS-Audio-Tokenizer-Nano-ONNX
python path/to/seeed-local-voice/scripts/fix_moss_codec_onnx_if_rank.py \
  --in-dir ~/models/MOSS-Audio-Tokenizer-Nano-ONNX \
  --out-dir ~/models/MOSS-Audio-Tokenizer-Nano-ONNX-trtfix

# 3. Copy ONNX bundle to target Jetson, then on Jetson:
ONNX_DIR=~/moss-tts-nano-onnx \
CODEC_ONNX_DIR=~/MOSS-Audio-Tokenizer-Nano-ONNX-trtfix \
OUT_DIR=/opt/models/moss-tts-nano \
bash scripts/build_moss_tts_engines.sh
# ~5-15 min wall on Orin NX. Produces 6 .plan files + sidecar meta/data.
```

## Tested on

- Orin NX 16GB / JetPack R36.4.3 / TRT 10.3 / CUDA 12.6 — TTFA **157 ms** (C++ TRT post-fix, 2026-05-24), ASR CER=0 across 3 Chinese prompts (short/medium/41-char-long), 5-round-repeat output byte-identical. N=2 concurrency verified (fork d92a306 / main 62448ef, binary md5 `fed07741bd93a5bf2f132989716e514c`).
- Orin NX 16GB — **ORT 1.23.2 relink (issue #48, 2026-05-31)**: worker rebuilt with `ORT_ROOT=/opt/onnxruntime-1.23.2` so it links the standard image's `onnxruntime 1.23.2` instead of the old 1.20 dev tree (`VERS_1.20.0` → `VERS_1.23.2`, `ldd` resolves `libonnxruntime.so.1` via rpath `$ORIGIN/../lib`). Verified against the standard Python ORT 1.23.2 (no bind-mount): `worker_ready` + synthesize "今天天气真不错。" → TTFA **137 ms**, wall 746 ms, 268800 samples (5.6 s), RMS **1526** (non-silent). New binary md5 `c037641d1279cbbcb90dcf197ba95430` / sha256 `b09f7f8d183f1e1dda8d261a5fbf22e4f429e25b84dfa125de5f8a820dc706f7` (562272 B). Mirrored at `harvestsu/seeed-local-voice-artifacts/models/moss-tts-nano/moss_tts_nano_worker`.
- Orin Nano 8GB / JetPack R36.4.3 / TRT 10.3 / CUDA 12.6 — TTFA **290 ms** N=1 (4.24 s audio for "你好，今天天气真不错"), N=2 concurrency verified 2026-05-25: parity 3-way MD5 byte-identical (`e15c5a601bb208feeb3add9044d5d1b5`), burst 30/30 rounds 0 errors 0 crashes, mixed-length TTFA ratio **1.01** (well within ≤1.5× spec gate), basic dual-client PASS. Peak RAM used 5530 MB / 7.4 GB total (min avail 1735 MB) — safely under OOM threshold. Same binary as Orin NX (md5 `fed07741bd93a5bf2f132989716e514c`).
- Orin NX ORT fallback path: TTFA ~3000 ms (CPU EP) — production-validated 2026-05-23 (`[[moss_tts_nano_ort_path_production_ready]]`).

Pending (per `[[moss_tts_nano_smoke_e2e_done]]` follow-ups): AGX Orin.

### Deploying to Orin Nano

Same `jetson-moss-tts-nano-trt` profile works as-is on Orin Nano 8GB (no separate N=1 fallback profile needed). Steps:

```bash
# 1. Create target dirs (sudo)
sudo mkdir -p /opt/jv-workers /opt/models/moss-tts-nano/engines /opt/models/moss-tts-nano/codec_onnx
sudo chown -R $USER:$USER /opt/jv-workers /opt/models/moss-tts-nano

# 2. Install runtime deps
sudo apt-get install -y libsentencepiece-dev libsentencepiece0
# ensure /home/<user>/ort-from-container/lib/libonnxruntime.so.1 exists (worker rpath)

# 3. Sync engines + codec + worker binary from Orin NX (via LAN rsync recommended,
#    ~2.5 GB engines + ~120 MB codec; ~25 s on gigabit LAN at ~110 MB/s):
#    on orin-nx:
rsync -av --exclude '*.bak' --exclude '*.before_*' --exclude '*.fp16bak' \
  --exclude '*.fp32test' --exclude '*.ropepin.plan' \
  /opt/models/moss-tts-nano/engines/ harvest@<orin-nano-lan-ip>:/opt/models/moss-tts-nano/engines/
rsync -av /opt/models/moss-tts-nano/codec_onnx/ harvest@<orin-nano-lan-ip>:/opt/models/moss-tts-nano/codec_onnx/
scp /opt/jv-workers/moss_tts_nano_worker harvest@<orin-nano-lan-ip>:/tmp/
#    on orin-nano:
sudo install -m 0755 /tmp/moss_tts_nano_worker /opt/jv-workers/moss_tts_nano_worker
md5sum /opt/jv-workers/moss_tts_nano_worker
# expect fed07741bd93a5bf2f132989716e514c

# 4. Smoke + N=2 stress (same scripts as Orin NX):
python3 bench/perf/smoke_moss_tts_backend.py --text "你好，今天天气真不错" --output /tmp/moss_smoke.wav
MOSS_MAX_SLOTS=2 python3 bench/perf/stress_moss_tts_n2.py --mode parity
```

Monitor RAM with `watch -n2 free -m` during N=2 burst; if peak `used` exceeds ~7000 MB, fall back to `MOSS_MAX_SLOTS=1`. In practice on Orin Nano 8GB peak observed at 5530 MB.

## Building the C++ TRT worker (Orin NX host)

The fork repo `TensorRT-Edge-LLM` branch `qwen3-tts-highperf-runtime-w8a16` contains both the runtime (`cpp/runtime/mossTtsNanoRuntime.{cpp,h}`) and the standalone worker (`cpp/workers/moss_tts_nano_worker.cpp` + `cpp/workers/build_moss_worker.sh`). The build script auto-rebuilds the runtime `.o` if its source has been touched since the last build.

```bash
# On Orin NX host (NOT inside container — needs CUDA toolkit + nvcc):
cd ~/TensorRT-Edge-LLM
EDGELLM_SRC=$PWD \
ORT_ROOT=/usr/local/onnxruntime \
SP_ROOT=/usr \
OUT=/tmp/moss_tts_nano_worker \
bash cpp/workers/build_moss_worker.sh

# Then back up production binary + install:
sudo cp /opt/jv-workers/moss_tts_nano_worker /opt/jv-workers/moss_tts_nano_worker.bak.$(date +%Y%m%d-%H%M)
sudo install -m 0755 /tmp/moss_tts_nano_worker /opt/jv-workers/moss_tts_nano_worker
md5sum /opt/jv-workers/moss_tts_nano_worker  # should match build output
```

On worker startup look for the KV dtype probe line in stderr:

```
[moss] KV element dtype=0 size=4 bytes (FP32=0 FP16=1)
```

`dtype=0 size=4` → FP32 KV engines (v16+ rebuild). `dtype=1 size=2` → FP16 KV engines. Both are valid — the runtime probes dynamically post-fix.

## Known issues / gotchas

### 1. Worker binary ORT version (RESOLVED 2026-05-31, issue #48)

**History:** the original `moss_tts_nano_worker` was built against an ORT 1.20
dev tree (`/home/harvest/ort-from-container/lib`), so the binary carried
`VERS_1.20.0` symbol versions and would fail to load against the standard image
ORT 1.23.2 (`/tts` 500s). The old workaround was to bind-mount the 1.20 host ORT
into the container.

**Fix:** rebuild with `ORT_ROOT=/opt/onnxruntime-1.23.2` (the official
`onnxruntime-linux-aarch64-1.23.2.tgz` dev tarball — C++ headers + lib, not the
pip wheel). The relinked binary carries `VERS_1.23.2` and, via the embedded
rpath `$ORIGIN/../lib`, resolves the in-image `libonnxruntime.so.1` (1.23.2). No
bind-mount needed. Other backends already use Python ORT (version-flexible);
only this C++ worker linked the dev tree directly.

```bash
# Get ORT 1.23.2 aarch64 dev tarball (headers + lib):
curl -fL -o /tmp/ort1232.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-aarch64-1.23.2.tgz
sudo tar xzf /tmp/ort1232.tgz -C /opt
sudo ln -sfn /opt/onnxruntime-linux-aarch64-1.23.2 /opt/onnxruntime-1.23.2

# Rebuild (see "Building the C++ TRT worker" above) with ORT_ROOT pointed at it:
ORT_ROOT=/opt/onnxruntime-1.23.2 ... bash cpp/workers/build_moss_worker.sh

# Verify the relink:
nm -D --with-symbol-versions <worker> | grep -oE 'VERS_[0-9.]+'   # expect only VERS_1.23.2
ldd <worker> | grep onnx                                          # libonnxruntime.so.1 -> 1.23.2
```

The build script at `cpp/workers/build_moss_worker.sh` (in TensorRT-Edge-LLM
fork) takes `ORT_ROOT` as the only knob that needs changing for this.

### 2. Engine profile caps cap audio duration

Current engines built with `--maxShapes=past_key_*:1x512x12x64` →
~40s max audio after a ~80-token prompt. For longer utterances rebuild with
higher max past_seq.

Worker enforces this via `kKvProfileMaxPast = 256` constant; bump it together
with engine rebuild.

### 3. ORT 1.20 vs 1.23 ABI

Worker built against ORT 1.20.0 (from container snapshot). On a fresh Jetson
without `/home/harvest/ort-from-container/lib`, install onnxruntime-gpu via
apt or NVIDIA Jetson Zoo and adjust `ORT_ROOT` before running `build_moss_worker.sh`.

### 4. SentencePiece dependency

Worker dynamically links `libsentencepiece.so.0`. Install on Jetson:
`apt install libsentencepiece-dev libsentencepiece0` (Ubuntu 22.04 has it).

### 5. (HISTORICAL, FIXED 2026-05-24) KV buffer dtype hardcoded as `sizeof(half)`

`mossTtsNanoRuntime.cpp` previously hardcoded `sizeof(half)` (2 bytes) in 8 KV-buffer sizing call-sites. When engines were rebuilt with FP32 KV IO (v16 rebuild), buffers were half-size → per-layer KV overlap → frame≥1 corruption → trailing-token hallucination on ASR. **Fixed** by probing `mDecodeEngine->getTensorDataType("past_key_0")` and computing element size dynamically. Full bug report + diagnostic trail: `docs/specs/moss-tts-nano-kv-dtype-abi-fix.md`.

If you rebuild the worker from a fork commit older than `3c6c263`, you will hit this regression again. Pull the fix or apply the equivalent dtype-probe patch.

### 6. Codec file lookup hardcoded under engine-dir

Worker (via `MossTtsNanoRuntime::loadCodecEngine`) reads `codec_decode_step.plan`
and `codec_browser_onnx_meta.json` from `engineDir`, NOT from `--codec-onnx-dir`
(that arg only feeds the voice-clone encode ORT session).

The build script handles this with symlinks `engines/codec_*.{plan,json}` →
`../codec_onnx/...`. If you build engines manually skipping the script, add
these symlinks or pass `--codec-onnx-dir=<same as engine-dir>`.

## Observability

- Worker `worker_ready` event includes `voice_clone_enabled` and
  `prompt_template_loaded` — log both at backend startup.
- Worker emits `ttfa_ms` and `wall_ms` in `done` events; backend forwards
  these into `TTSResult.metadata`.
- For per-request timing add `X-Inference-Time` and `X-RTF` headers in
  `/tts` route handler (qwen3_trt pattern already there).

## Smoke verification

```bash
# Standalone Python backend smoke (no HTTP):
python3 bench/perf/smoke_moss_tts_backend.py \
  --text "你好，今天天气真不错" \
  --output /tmp/moss_smoke.wav

# Expected output:
# [smoke] preload OK in ~8000 ms
# [smoke] first chunk N bytes at ~120 ms
# [smoke] streaming done: ... duration=13.76s, wall=2427ms, ttfa=120ms
# [smoke] backend.shutdown OK
```

WAV md5 baseline (deterministic except RNG):
`ac08ee5da347e241ba1ecb2887938757` (rng-dependent; use `sox … -n stat` for
quality check: RMS ~0.26, rough freq ~1.1 kHz on healthy Chinese speech).

## Roll back

Switch `OVS_PROFILE` back to `jetson-multilang-highperf` (or whatever was prior).
MOSS engines / worker stay on disk but are unused. No volume cleanup needed
unless reclaiming the 1.42 GB.

## References

- ONNX patch & codec fix landing: [[moss_tts_nano_smoke_e2e_done]]
- C++ runtime: [[moss_tts_nano_worker_p1_done]] + `docs/specs/moss-tts-nano-paged-kv-cpp.md`
- Edge port playbook: `docs/playbooks/tts-model-edge-port-playbook.md`
- Backend pattern: `app/backends/jetson/qwen3_trt.py` (mirror template)
