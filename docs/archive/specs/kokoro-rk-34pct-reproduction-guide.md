# Kokoro RK 34% 4-stage + 3-bucket — Reproduction Guide

**Date:** 2026-05-23
**Target hardware:** Radxa Rock 5B / 5B+ (RK3588), running existing
  Seeed Studio local-voice RK image `openvoicestream:rk-kokoro-2026-05-23`.
**Goal of this doc:** allow a fresh checkout of `seeed-local-voice` on a
  brand-new RK3588 box to reach the *current production state* (HTTP RTF
  0.59, 3-bucket dynamic router, misaki ZH G2P) without re-deriving any of
  the boundary / parity / quantization decisions.

This is a *reproduction* guide, not a design doc. For the *why* behind each
decision, the per-stage parity data, and the failed paths, follow the
per-milestone specs in §10.

---

## §1 Background and final outcome

The original R&D spec (`docs/archive/specs/kokoro-rk-npu-42pct.md`, commit `7bd7228`)
targeted **42 %** NPU residency on Kokoro v1.0 via three additions to the
shipped 17 % decoder-front INT8 baseline: BERT FP16, vocoder front-half
FP16, and a prefix split. Execution surfaced three findings that bent the
plan:

1. **BERT is bit-exact dead code in the Kokoro v1.0 ONNX export.**
   `docs/archive/specs/kokoro-bert-ab-audio-report.md` (commit `3b18517`) proved
   10/10 utterances are byte-identical with/without BERT compute (mel L1
   = 0 dB, pitch RMSE = 0 Hz). M2 BERT FP16 RKNN was built (`50463cb`) but
   not wired. Removing BERT from the budget capped the realistic NPU
   share at ~34 %; spec target was revised in commit `bd2b053`.
2. **Native FP16 vocoder front-half RKNN beat the Sin polynomial rewriter
   path** on audio fidelity (worst rel_l2 0.00186, gate ≤ 0.01) — M4
   landed as `native.fp16` rather than INT8 (`add7ddf`).
3. **Static-shape bucketing decisively beats single-bucket TTFA.** The
   shipped vocoder-front split runs at fixed seq_len=32 → constant 5.25 s
   audio output, so a 4-token utterance pays the same NPU cost as a
   32-token one. Three buckets (8 / 16 / 32) shipped behind a per-sentence
   router that selects by post-G2P phoneme count.

### Final shipped state (production on radxa, 2026-05-23)

| Metric | Before this workstream | After |
| --- | --- | --- |
| NPU residency (op-count, decoder-front + vocoder-front-half) | 17 % | 34 % |
| HTTP RTF, 4-token `"abc."`, 100 shots, /tts/stream | 0.66 (3-stage) | **0.59** (4-stage bucket-32) |
| HTTP TTFA, 4-token `"abc."`, 30 shots | n/a (full 3.1 s) | **0.79 s** (bucket-8) |
| HTTP TTFA, 16-phoneme mid sentence, 30 shots | 3.1 s | **1.62 s** (bucket-16) |
| HTTP TTFA, 17+ phoneme long sentence | 3.1 s | 3.1 s (bucket-32, unchanged) |
| Chinese support (no misaki) | silent fail / 4-byte WAV | **real audio** (T2 251 908 B, T3 503 812 B) |

Key technical decisions in order of impact:

- **BERT bit-exact dead code → not wired to NPU** (commit `3b18517`). Avoids
  shipping a 75 MB FP16 RKNN that would only move dead compute around.
- **Vocoder front-half = `native.fp16`, not Sin polynomial rewrite**
  (commit `add7ddf`). 5× audio-rel_l2 margin vs gate; Sin polynomial path
  had higher per-op error and worse runtime.
- **Per-sentence dynamic bucket router with graceful fallback**
  (commits `1efe6a6` bucket-8, `5a923ef` bucket-16). Env-missing → falls
  back to next bucket, never crashes.
- **misaki v1.1 ZH G2P at tokenizer front-end, not at runtime API**
  (commit `e2665d0`). Bopomofo + tone digits map 1:1 into existing
  `tokens.txt`; no graph / artifact change needed.

---

## §2 System architecture

### 2.1 4-stage pipeline + router

```
text (EN or ZH)
   │
   ▼
[ ZH? → misaki.zh ZHG2P v1.1 → Bopomofo+tone-digits ]
[ EN/Bopomofo → per-char tokens.txt lookup            ]
   │
   ▼ (post-G2P phoneme count = n)
[ _select_bucket(n):  n<=8 → 8  ;  8<n<=16 → 16  ;  n>16 → 32 ]
   │
   ▼ (per-sentence dispatch into bucket-N 4-stage path)
┌──────────────────────────────────────────────────────────────────┐
│ Stage 1  prefix   (CPU ONNX, kokoro-prefix-cpu-bucket{N}.onnx)   │
│ Stage 2  decoder-front  (RKNN, INT8 for B=32, FP16 for B=8/16)   │
│ Stage 3  vocoder-front-half  (RKNN native FP16, all buckets)     │
│ Stage 4  tail-rest  (CPU ONNX, kokoro-vocoder-tail-rest-cpu-…)   │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
audio chunk → _trim_silence → HTTP /tts/stream WAV chunk
```

The bucket-32 4-stage path is the canonical promotion (commit `c7302f9`).
Bucket-8 (`1efe6a6`) and bucket-16 (`5a923ef`) are additive routers layered
on top. Each bucket needs its own four artifacts (prefix CPU + decoder-front
RKNN + vocoder-front-half RKNN + tail-rest CPU) because all stages are
static-shape.

### 2.2 Per-bucket sizes and audio length

| Bucket | seq_len | decoder-front in | vocoder-front out | audio samples | audio duration |
| --- | --- | --- | --- | --- | --- |
| 8 | 8 | [1,512,45] | [1,256,900] | 27 000 @ 24 kHz | 1.125 s |
| 16 | 16 | [1,512,90] | [1,256,1800] | 54 000 @ 24 kHz | 2.25 s |
| 32 | 32 | [1,512,210] → [1,512,420] | [1,256,4200] | 126 000 @ 24 kHz | 5.25 s |

After `_trim_silence`, a typical bucket-8 `"abc."` response is ~43 012 bytes
(WAV) vs bucket-32's 251 908 bytes for the same text. The router's win is
mostly *not running 5.25 s of vocoder for a 0.9 s utterance*.

### 2.3 Filesystem layout (radxa side)

```
/home/radxa/models/tts/
├── kokoro-bucket-32/                  ← bind-mounted at /opt/kokoro-rknn
│   ├── kokoro-prefix-cpu.onnx
│   ├── kokoro-generator-tail-cpu.onnx          (legacy 3-stage fallback)
│   ├── kokoro-vocoder-tail-rest-cpu.onnx       (4-stage tail-rest)
│   ├── tokens.txt
│   └── rk3588/
│       ├── kokoro-decoder-front.int8.rknn
│       └── kokoro-vocoder-front-half.native.fp16.rknn
├── kokoro-bucket-8/                   ← bind-mounted at /opt/kokoro-bucket-8
│   ├── kokoro-prefix-cpu-bucket8.onnx
│   ├── kokoro-vocoder-tail-rest-cpu-bucket8.onnx
│   └── rk3588/
│       ├── kokoro-decoder-front-bucket8.fp16.rknn
│       └── kokoro-vocoder-front-half-bucket8.native.fp16.rknn
└── kokoro-bucket-16/                  ← bind-mounted at /opt/kokoro-bucket-16
    ├── kokoro-prefix-cpu-bucket16.onnx
    ├── kokoro-decoder-front-bucket16.fp16.rknn        (flat — no rk3588/)
    ├── kokoro-vocoder-front-half-bucket16.native.fp16.rknn
    └── kokoro-vocoder-tail-rest-cpu-bucket16.onnx
```

> **Layout quirk:** bucket-16's RKNN files live at the top of the bucket-16
> directory, not under a nested `rk3588/`. Env vars must point at the
> actual on-disk paths — see §3.D. Bucket-8 uses the nested layout (matches
> bucket-32). Keep this in mind when scripting the deploy step.

### 2.4 Production image and bind-mount overlay

**Status as of 2026-05-23 rebuild:** image
`openvoicestream:rk-kokoro-2026-05-23-rebuilt` is **self-contained** — it
bakes in the fixed `server/backends/rk/tts.py` (speaker-id pop, commit
`6155ebe`), the current submodule `kokoro_rknn.py` (4-stage + 3-bucket +
misaki wire, submodule `65b9a13`), the misaki ZH G2P stack, and all 14
active bucket-{8,16,32} artifacts. No host bind-mounts of model files or
`.py` overrides are required. Deploy recipe in `docs/runbooks/kokoro-rk-deploy.md`.

The original (pre-rebuild) production container `openvoicestream-kokoro`
ran image `openvoicestream:rk-kokoro-2026-05-23` with the *broken*
`server/backends/rk/tts.py` (missing speaker-id kwarg pop) and the
*3-stage* `kokoro_rknn.py`. Both were overridden by host bind-mounts
(retained here as the rollback recipe / archaeological record):

```
/tmp/fixed-tts.py                         → /opt/speech/server/backends/rk/tts.py            (ro)
/tmp/fixed-kokoro_rknn.py                 → /opt/speech/third_party/rkvoice-stream/.../
                                            kokoro_rknn.py                                (ro)
/home/radxa/models/tts/kokoro-bucket-32   → /opt/kokoro-rknn                              (ro)
/home/radxa/models/tts/kokoro-bucket-8    → /opt/kokoro-bucket-8                          (ro)
/home/radxa/models/tts/kokoro-bucket-16   → /opt/kokoro-bucket-16                         (ro)
```

After the *next* image rebuild (see §6), the two `/tmp/fixed-*.py`
bind-mounts go away; the bucket bind-mounts stay (artifacts not yet baked).

---

## §3 Reproduction steps

### A. Clone repo + submodule at the right pins

```bash
git clone <repo> seeed-local-voice
cd seeed-local-voice
git checkout c7302f9       # or any descendant; bucket-16 lands at ca9332c

git submodule update --init --recursive
cd third_party/rkvoice-stream
git checkout feat/kokoro-rk-4stage-vocoder-front
git log --oneline | head -6     # expected key commits:
#   5a923ef  bucket-16 router
#   1efe6a6  bucket-8 router
#   c4c7706  TTFA log field
#   d6b9463  misaki ZH G2P wire
#   c566c66  4-stage runtime
```

If `git submodule status` shows the wrong SHA for `third_party/rkvoice-stream`,
the main-repo submodule pointer is the source of truth — `git submodule
update --init` will pull the right SHA automatically.

### B. Stage artifacts on the radxa host

#### B.1 Bucket-32 (HF mirror available)

The 4-stage bucket-32 artifacts are mirrored at
`harvestsu/seeed-local-voice-rk-artifacts` under
`rk3588/kokoro-hybrid-v1/`:

| File | HF path under `rk3588/kokoro-hybrid-v1/` | Local target |
| --- | --- | --- |
| `kokoro-prefix-cpu.onnx`                              | `kokoro-prefix-cpu.onnx`                              | `/home/radxa/models/tts/kokoro-bucket-32/kokoro-prefix-cpu.onnx` |
| `kokoro-generator-tail-cpu.onnx` (fallback)           | `kokoro-generator-tail-cpu.onnx`                      | `/home/radxa/models/tts/kokoro-bucket-32/kokoro-generator-tail-cpu.onnx` |
| `kokoro-vocoder-tail-rest-cpu.onnx` (4-stage tail)    | `kokoro-vocoder-tail-rest-cpu.onnx`                   | `/home/radxa/models/tts/kokoro-bucket-32/kokoro-vocoder-tail-rest-cpu.onnx` |
| `kokoro-decoder-front.int8.rknn`                      | `rk3588/kokoro-decoder-front.int8.rknn`               | `/home/radxa/models/tts/kokoro-bucket-32/rk3588/kokoro-decoder-front.int8.rknn` |
| `kokoro-vocoder-front-half.native.fp16.rknn`          | `rk3588/kokoro-vocoder-front-half.native.fp16.rknn`   | `/home/radxa/models/tts/kokoro-bucket-32/rk3588/kokoro-vocoder-front-half.native.fp16.rknn` |
| `tokens.txt`                                          | `tokens.txt`                                          | `/home/radxa/models/tts/kokoro-bucket-32/tokens.txt` |

Fetch via the existing `model_downloader` (driven by
`deploy/artifacts/rk_manifest.json` entry
`rk3588-kokoro-hybrid-34pct-2026-05-23`):

```bash
fleet exec radxa -- "docker exec openvoicestream-kokoro \
    python -m server.model_downloader --artifact-set rk3588-kokoro-hybrid-34pct-2026-05-23"
```

Or wget directly:

```bash
HF=https://huggingface.co/harvestsu/seeed-local-voice-rk-artifacts/resolve/main/rk3588/kokoro-hybrid-v1
mkdir -p /home/radxa/models/tts/kokoro-bucket-32/rk3588
cd /home/radxa/models/tts/kokoro-bucket-32
wget "$HF/kokoro-prefix-cpu.onnx" "$HF/kokoro-generator-tail-cpu.onnx" \
     "$HF/kokoro-vocoder-tail-rest-cpu.onnx" "$HF/tokens.txt"
cd rk3588
wget "$HF/rk3588/kokoro-decoder-front.int8.rknn" \
     "$HF/rk3588/kokoro-vocoder-front-half.native.fp16.rknn"
```

Verify md5s against `docs/archive/specs/kokoro-rk-34pct-http-rtf-final.md` §"File
integrity" table before continuing.

#### B.2 Bucket-8 and bucket-16 (HF mirror available — 2026-05-23)

Bucket-8 and bucket-16 artifacts are now mirrored at
`harvestsu/seeed-local-voice-rk-artifacts` under
`rk3588/kokoro-hybrid-v1/bucket{8,16}/`. End-to-end md5 round-trip from
HF matches the wsl2-local build-host source (verified 2026-05-23).

| File | HF path under `rk3588/kokoro-hybrid-v1/` | Local target |
| --- | --- | --- |
| `kokoro-prefix-cpu-bucket8.onnx`                              | `bucket8/kokoro-prefix-cpu-bucket8.onnx`                              | `/home/radxa/models/tts/kokoro-bucket-8/kokoro-prefix-cpu-bucket8.onnx` |
| `kokoro-decoder-front-bucket8.fp16.rknn`                      | `bucket8/kokoro-decoder-front-bucket8.fp16.rknn`                      | `/home/radxa/models/tts/kokoro-bucket-8/rk3588/kokoro-decoder-front-bucket8.fp16.rknn` |
| `kokoro-vocoder-front-half-bucket8.native.fp16.rknn`          | `bucket8/kokoro-vocoder-front-half-bucket8.native.fp16.rknn`          | `/home/radxa/models/tts/kokoro-bucket-8/rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn` |
| `kokoro-vocoder-tail-rest-cpu-bucket8.onnx`                   | `bucket8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx`                   | `/home/radxa/models/tts/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx` |
| `kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx` (P7b — opt-in) | `bucket8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx`        | `/home/radxa/models/tts/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx` |
| `kokoro-prefix-cpu-bucket16.onnx`                             | `bucket16/kokoro-prefix-cpu-bucket16.onnx`                            | `/home/radxa/models/tts/kokoro-bucket-16/kokoro-prefix-cpu-bucket16.onnx` |
| `kokoro-decoder-front-bucket16.fp16.rknn`                     | `bucket16/kokoro-decoder-front-bucket16.fp16.rknn`                    | `/home/radxa/models/tts/kokoro-bucket-16/kokoro-decoder-front-bucket16.fp16.rknn` |
| `kokoro-vocoder-front-half-bucket16.native.fp16.rknn`         | `bucket16/kokoro-vocoder-front-half-bucket16.native.fp16.rknn`        | `/home/radxa/models/tts/kokoro-bucket-16/kokoro-vocoder-front-half-bucket16.native.fp16.rknn` |
| `kokoro-vocoder-tail-rest-cpu-bucket16.onnx`                  | `bucket16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx`                 | `/home/radxa/models/tts/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx` |

Fetch directly with wget (note: bucket-8 uses nested `rk3588/` layout to
match bucket-32; bucket-16 is flat per §2.3):

```bash
HF=https://huggingface.co/harvestsu/seeed-local-voice-rk-artifacts/resolve/main/rk3588/kokoro-hybrid-v1

# Bucket-8 (nested rk3588/ layout)
mkdir -p /home/radxa/models/tts/kokoro-bucket-8/rk3588
cd /home/radxa/models/tts/kokoro-bucket-8
wget "$HF/bucket8/kokoro-prefix-cpu-bucket8.onnx" \
     "$HF/bucket8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx"
cd rk3588
wget "$HF/bucket8/kokoro-decoder-front-bucket8.fp16.rknn" \
     "$HF/bucket8/kokoro-vocoder-front-half-bucket8.native.fp16.rknn"

# Bucket-16 (flat layout — all four files at bucket-16 root)
mkdir -p /home/radxa/models/tts/kokoro-bucket-16
cd /home/radxa/models/tts/kokoro-bucket-16
wget "$HF/bucket16/kokoro-prefix-cpu-bucket16.onnx" \
     "$HF/bucket16/kokoro-decoder-front-bucket16.fp16.rknn" \
     "$HF/bucket16/kokoro-vocoder-front-half-bucket16.native.fp16.rknn" \
     "$HF/bucket16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx"
```

Verify md5s after download (source of truth — matches bucket-8 / bucket-16
specs §7.1 / "Artifact md5"):

```
Bucket-8:
  40ea37f93d415c5aa07bb17b47544a00  kokoro-prefix-cpu-bucket8.onnx
  814c558955a8eb6a3f6b95cfc5eab0b9  rk3588/kokoro-decoder-front-bucket8.fp16.rknn
  bcf1d03b3aef869bac66ac98140b40e9  rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn
  9e03458e86a8732ccaeccd7c7b0618f9  kokoro-vocoder-tail-rest-cpu-bucket8.onnx
  6eff2d5b0f381c3a262844011079ae85  kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx  (P7b opt-in static MM+Gemm INT8)

Bucket-16:
  3f85d7f235b1d879c1346075176de71d  kokoro-prefix-cpu-bucket16.onnx
  ac812b66991cfbbe47326fa9d252cc66  kokoro-decoder-front-bucket16.fp16.rknn
  4d28eece3c3070629331038524457cb4  kokoro-vocoder-front-half-bucket16.native.fp16.rknn
  0021f107a9bc93618e5a74d5ba23218d  kokoro-vocoder-tail-rest-cpu-bucket16.onnx
```

Fallback paths if HF is unreachable: the same files live on
`wsl2-local:/home/harve/kokoro-analysis/m_bucket{8,16}/` (build host) and
`radxa:/home/radxa/models/tts/kokoro-bucket-{8,16}/` (current prod);
`fleet transfer` works between any two of these. To rebuild from source,
run `wsl2-local:/home/harve/kokoro-analysis/m_bucket{8,16}/run_*.sh`
(RKNN-toolkit2 2.3.0 + venv `/home/harve/rknn-build/.venv`).

### C. Install misaki ZH G2P into the container

**Status as of 2026-05-23 image rebuild:** misaki + ZH deps are **baked
into the image** `openvoicestream:rk-kokoro-2026-05-23-rebuilt` via
`deploy/docker/Dockerfile.rk` lines 60-83 (misaki 0.9.4, jieba 0.42.1,
pypinyin, pypinyin_dict, ordered_set, cn2an, proces, addict, regex). **No
runtime install step is required for the rebuilt image.** Skip to §3.D.

The remainder of this subsection is the historical hot-patch recipe used
against the pre-rebuild image `openvoicestream:rk-kokoro-2026-05-23`
(retained as the rollback path).

```bash
# Pre-rebuild image only — installs into writable layer:
fleet exec radxa -- "docker exec openvoicestream-kokoro \
    pip3 install 'misaki[zh]'"
```

This pulls misaki 0.9.4 + cn2an + addict + jieba + ordered-set + pypinyin
+ pypinyin-dict + proces + regex (~30 MB total). `aarch64` wheels exist
for everything except `jieba` (pure-Python sdist).

**Warning (pre-rebuild image only):** writable-layer installs survive
`docker restart` and `docker stop/start` but are *destroyed* by
`docker rm` / `--force-recreate`. The 2026-05-23-rebuilt image removes
this risk entirely.

### D. Deploy container with the right bind-mounts + env

The current production launch script (sketched from prod state) is the
canonical recipe. Roll this into your overlay or systemd unit:

```bash
docker rm -f openvoicestream-kokoro 2>/dev/null

docker run -d --name openvoicestream-kokoro --restart=unless-stopped \
  --network host \
  -v /tmp/fixed-tts.py:/opt/speech/server/backends/rk/tts.py:ro \
  -v /tmp/fixed-kokoro_rknn.py:/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py:ro \
  -v /home/radxa/models/tts/kokoro-bucket-32:/opt/kokoro-rknn:ro \
  -v /home/radxa/models/tts/kokoro-bucket-8:/opt/kokoro-bucket-8:ro \
  -v /home/radxa/models/tts/kokoro-bucket-16:/opt/kokoro-bucket-16:ro \
  --device /dev/dri --device /dev/dma_heap --device /dev/rga --device /dev/mpp_service \
  --group-add video \
  \
  -e OVS_PROFILE=rk3588-kokoro-rknn-34pct \
  -e RK_ARTIFACT_SET=rk3588-kokoro-hybrid-34pct-2026-05-23 \
  \
  -e KOKORO_RKNN_VOCODER_FRONT_PATH=kokoro-vocoder-front-half.native.fp16.rknn \
  -e KOKORO_RKNN_TAIL_REST_PATH=kokoro-vocoder-tail-rest-cpu.onnx \
  \
  -e KOKORO_RKNN_BUCKET8_PREFIX_PATH=/opt/kokoro-bucket-8/kokoro-prefix-cpu-bucket8.onnx \
  -e KOKORO_RKNN_BUCKET8_DECODER_FRONT_PATH=/opt/kokoro-bucket-8/rk3588/kokoro-decoder-front-bucket8.fp16.rknn \
  -e KOKORO_RKNN_BUCKET8_VOCODER_FRONT_PATH=/opt/kokoro-bucket-8/rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn \
  -e KOKORO_RKNN_BUCKET8_TAIL_REST_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx \
  \
  -e KOKORO_RKNN_BUCKET16_PREFIX_PATH=/opt/kokoro-bucket-16/kokoro-prefix-cpu-bucket16.onnx \
  -e KOKORO_RKNN_BUCKET16_DECODER_FRONT_PATH=/opt/kokoro-bucket-16/kokoro-decoder-front-bucket16.fp16.rknn \
  -e KOKORO_RKNN_BUCKET16_VOCODER_FRONT_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-front-half-bucket16.native.fp16.rknn \
  -e KOKORO_RKNN_BUCKET16_TAIL_REST_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx \
  \
  `# P7a: opt-in tail-rest INT8 (env-gated; remove these 3 lines to keep FP32):` \
  -e KOKORO_RKNN_TAIL_REST_INT8_PATH=/opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.int8.onnx \
  -e KOKORO_RKNN_BUCKET8_TAIL_REST_INT8_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8.onnx \
  # P7b (2026-05-23): bucket-8 static MM+Gemm INT8 — preferred over dynamic INT8 above. Loader auto-falls-back to dynamic INT8 / FP32 if file is missing.
  -e KOKORO_RKNN_BUCKET8_TAIL_REST_INT8STATIC_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx \
  -e KOKORO_RKNN_BUCKET16_TAIL_REST_INT8_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.int8.onnx \
  \
  openvoicestream:rk-kokoro-2026-05-23
```

Notes:
- **P7a INT8 (2026-05-23, opt-in default)**: 3 INT8 `.int8.onnx` files mirror
  the FP32 tail-rest filenames; live in the same directory next to their
  FP32 sibling. When set + file exists → INT8 loaded. When unset or file
  missing → automatic FP32 fallback (warning logged). Audio rel_l2 PASS at
  worst 0.018 (gate 0.05). TTFA neutral (±5 % run-to-run noise vs FP32
  baseline) — see `kokoro-rk-tail-rest-int8.md` for the full negative
  perf result; INT8 is shipped because it's safe, not because it's faster.
  HF mirror under `…/kokoro-hybrid-v1/bucket{8,16,-32}/…int8.onnx`.
- `KOKORO_RKNN_VOCODER_FRONT_PATH` and `KOKORO_RKNN_TAIL_REST_PATH` are
  **filenames** (resolved against `MODEL_DIR=/opt/kokoro-rknn`).
- All bucket-8/16 paths are **absolute** to avoid collision with the
  bucket-32 `MODEL_DIR` resolution.
- The two `/tmp/fixed-*.py` files come from the working hot-patch (md5
  `4961497d910cac5531ceafe35e4f1713` for `tts.py`,
  `ccc43371ee16465899f57e1ee4ed5a5f` for FP32-only `kokoro_rknn.py`,
  `1d8de71252ebdaa828b35c2c9dd39946` for P7a INT8-aware `kokoro_rknn.py`).
  If they're
  missing on a fresh box, copy them from the running production radxa, or
  push from your repo:

  ```bash
  fleet push <new-host>  \
    third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py \
    /tmp/fixed-kokoro_rknn.py
  fleet push <new-host>  server/backends/rk/tts.py  /tmp/fixed-tts.py
  ```

- After image rebuild (§6), drop the two `/tmp/fixed-*.py` bind-mounts.

Re-install misaki into the new container's writable layer per §3.C
(skip after image rebuild).

### E. Verify

#### E.1 Startup log must show all three routers active

```bash
fleet exec radxa -- "docker logs openvoicestream-kokoro 2>&1 | \
    grep -E 'Kokoro (4-stage|bucket-)|misaki' | head -10"
```

Expected:

```
Kokoro 4-stage path active: vocoder_front=/opt/kokoro-rknn/kokoro-vocoder-front-half.native.fp16.rknn tail_rest=/opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.onnx
Kokoro bucket-8 router enabled:  ... (threshold n_tokens<=8)
Kokoro bucket-16 enabled: prefix=/opt/kokoro-bucket-16/...  (threshold 8<n_tokens<=16)
Loaded Kokoro hybrid: prefix=/opt/kokoro-rknn/kokoro-prefix-cpu.onnx front=/opt/kokoro-rknn/rk3588/kokoro-decoder-front.int8.rknn ...
misaki ZH G2P (v1.1) loaded for Kokoro RKNN
```

If any line missing → fix env / artifact / misaki and restart before
running TTFA tests. Missing bucket-N enable line means at least one of the
four bucket-N env vars is unset or the file is missing.

#### E.2 /health

```bash
curl -s http://<radxa-ip>:8621/health
# {"tts":true,"tts_backend":"rk:kokoro_rknn","asr":true, ...}
```

#### E.3 30-shot TTFA per bucket

Use `/tmp/bench_kokoro.sh` (template; n=30 serial POSTs to
`/tts/stream` recording `%{time_total} %{http_code} %{size_download}`):

| Test | Bucket expected | Wall p50 gate | Bytes |
| --- | :---: | --- | --- |
| `"abc."` (EN, 4 phonemes) | 8 | ≤ 1.0 s | 43 012 |
| `"你好。"` (ZH, ~6 phonemes) | 8 | ≤ 1.0 s | ~30-50 KB |
| `"Hello world."` (EN, 11 phonemes) | 16 | ≤ 2.0 s | ~90 KB |
| `"我感觉很棒。"` (ZH, 16 phonemes) | 16 | ≤ 2.0 s | ~90 KB |
| `"hello world. how are you today?"` (EN, 11 + 15 phonemes, 2 sentences) | 16 + 16 | ≤ 4.0 s | ~260 KB |

**Verified on 2026-05-23 rebuilt image** (30 shots each, all 200 OK, deterministic byte size):

| Test | HTTP | Size (bytes) | Server-side TTFA p50 |
| --- | --- | --- | --- |
| `"abc."`                              | 30/30 200 | 43 012 (deterministic)  | bucket-8 ~700 ms |
| `"你好。"`                            | 30/30 200 | 53 252 (deterministic)  | bucket-8 ~700 ms |
| `"Hello world."`                      | 30/30 200 | 130 052 (deterministic) | bucket-16 ~1 580 ms |
| `"我感觉很棒。"`                      | 30/30 200 | 130 052 (deterministic) | bucket-16 ~1 580 ms |
| `"hello world. how are you today?"`   | 30/30 200 | 260 100 (deterministic) | 2× bucket-16 ~1 550–1 850 ms / sentence (sentence-split, not bucket-32) |

> Note on the 5th case: post-G2P, the EN sentence-splitter cuts the input
> into two clauses (11 + 15 phonemes), each routed independently to
> bucket-16. To exercise the bucket-32 path explicitly, use a single
> long clause with >16 phonemes (e.g.
> `"The quick brown fox jumps over the lazy dog."`).

Spot-check the per-call routing in logs:

```bash
fleet exec radxa -- "docker logs --tail 200 openvoicestream-kokoro 2>&1 | \
    grep 'TTFA' | tail -10"
```

Each line must include `bucket=<8|16|32>` and `num_tokens=<n>`.

---

## §4 Test coverage matrix

Reproducing the acceptance gates already passed in production:

| Gate | Spec source | Threshold | Measured |
| --- | --- | --- | --- |
| Audio rel_l2 (10 BERT-AB utterances, bucket-32 4-stage RKNN vs CPU FP32) | `kokoro-rk-34pct-m4m6-final.md` §1 | ≤ 0.01 | worst 0.00186 |
| Audio pitch RMSE (10 utterances, same set) | M4 final §1 | ≤ 5 Hz | worst 0.045 Hz |
| Stability — 50× plain `"abc."` (bucket-32 4-stage) | M4 final §2 | 50/50 OK no None/zero/crash | 50/50 OK |
| HTTP RTF — 100× `"abc."` (4-stage bucket-32 promote) | `kokoro-rk-34pct-http-rtf-final.md` §"Results" | ≤ 0.66 (3-stage baseline) | p50 **0.59**, p95 0.647 |
| Bucket-8 parity rel_l2 (5 random seeds RKNN-4 vs ORT-4) | `kokoro-rk-bucket8-ttfa.md` §3 | ≤ 0.01 | worst 0.00212 |
| Bucket-8 TTFA — 30× `"abc."` | bucket-8 §6 | p50 ≤ 1.5 s | **0.787 s** |
| Bucket-8 EN HTTP success | bucket-8 §6 | 30/30 200 OK | 30/30 200 OK, deterministic 43 012 B |
| Bucket-16 audio A/B parity (5 random seeds) | `kokoro-rk-bucket16-mid-ttfa.md` §"Audio-end A/B" | rel_l2 ≤ 0.01 | 4/5 strict PASS; 1 marginal 0.01133 (audio imperceptible) |
| Bucket-16 TTFA — 30× `"Hello world."` (16 phonemes) | bucket-16 §"Acceptance" | p50 ≤ 2.0 s | **1.615 s** |
| Bucket-16 TTFA — 30× `"我感觉很棒。"` (16 phonemes ZH) | bucket-16 §"Acceptance" | p50 ≤ 2.0 s | **1.621 s** |
| Bucket-32 regression check (31-token EN, 27-phoneme EN) | bucket-8 §6 / bucket-16 §"Acceptance" | wall ≈ 3.1 s baseline | 3.63 s / 3.07 s, no regression |
| Chinese smoke (T1 EN, T2 ZH mid, T3 ZH long) | `kokoro-rk-zh-fix-misaki.md` §"Three-test HTTP smoke" | non-trivial WAV (>100 KB) | 251 908 / 251 908 / 503 812 B |
| 20-shot sanity (10 EN + 10 ZH) | misaki fix §"Sanity batch" | 20/20 200 OK | 20/20 OK |

For full per-test raw numbers and the `curl ... -d ...` invocations, see
the source specs in §10.

---

## §5 Known limitations

1. **Bucket-32 output is fixed 5.25 s.** The vocoder front-half + tail-rest
   ONNX are static-shape (SEQ_LEN=32 → 126 000 audio samples per call).
   Short bucket-32-routed sentences are silence-padded then `_trim_silence`'d
   on the way out. Long bucket-32 sentences are truncated at 32 phonemes
   per *sentence segment* — multi-sentence input is split upstream and
   each segment routed independently.

2. **Bucket-16 case 14 marginal parity.** `"Birds sing morning."` rel_l2
   = 0.01133 (0.13 % over the 0.01 gate). Mel L1 0.020 dB / pitch RMSE
   0.021 Hz → audible difference is imperceptible (238× psychoacoustic
   margin). Same pattern as M4 vocoder front-half FP16 promote
   (`f482832`). User-greenlighted before deploy. Tracked in
   `kokoro-rk-bucket16-mid-ttfa.md` §"Audio-end A/B parity".

3. **Bucket-32 long-sentence TTFA hard wall at ~3 s.** The CPU tail-rest
   (27.7 MB ONNX) dominates wall time when the front-half NPU work is
   short relative to a long upsampling tail. Further reduction needs
   either NPU tail-rest (high risk — dilated conv time-dim approaches the
   RK3588 8191 limit) or chunked inference. See §7 roadmap.

4. **Writable-layer misaki is lost on container recreate.** `docker rm` /
   `docker compose up --force-recreate` wipes the misaki pip install.
   Persist via Dockerfile.rk image rebuild (§6) for stability. Dockerfile
   diff is already in commit `ca9332c`; only the image bake step is
   pending.

5. **Manifest entries for bucket-8 / bucket-16 sets not yet appended to
   `deploy/artifacts/rk_manifest.json`.** Only the bucket-32 set
   `rk3588-kokoro-hybrid-34pct-2026-05-23` is present. HF mirror is now
   live under `rk3588/kokoro-hybrid-v1/bucket{8,16}/` (see §3.B.2), so
   `model_downloader` manifest entries can be added in a follow-up
   without further dependencies.

6. **BERT FP16 RKNN artifact (M2, `50463cb`) is on disk but not wired.**
   Retained as a reference / diagnostic build per the dead-code A/B
   finding. Do not promote without re-validating that BERT contributes to
   audio in any Kokoro variant being considered.

---

## §6 Image rebuild — long-term persistence

**Status: DONE (2026-05-23).** Image
`openvoicestream:rk-kokoro-2026-05-23-rebuilt` (digest
`sha256:8f57153eb8bd…` for the artifact-free first build; superseded by the
`+tokens.txt` rebuild) was produced natively on the radxa
(`/home/radxa/build-kokoro-rebuild/`) and now bakes in:

- Updated `server/backends/rk/tts.py` (speaker-id pop, md5
  `4961497d910cac5531ceafe35e4f1713`)
- Updated submodule `kokoro_rknn.py` at HEAD `65b9a13` (4-stage +
  bucket-8 + bucket-16 + misaki wire, md5 `0dbb03149b1ee2b587abd2f0b4cf821b`)
- misaki + ZH G2P deps (Dockerfile.rk lines 60-83)
- All 14 active bucket artifacts (~448 MB) under `/opt/kokoro-bucket-{8,16}/`
  and `/opt/kokoro-rknn/` (bucket-32)
- `tokens.txt` (md5 `0ed710184a1ece5a4fd565e28f5062bb`) in all three
  bucket dirs

The deploy command no longer needs the `/tmp/fixed-*.py` bind-mounts or
the bucket-{8,16,32} bind-mounts. See `docs/runbooks/kokoro-rk-deploy.md`
for the production-ready `docker run` recipe.

The previous image tag `openvoicestream:rk-kokoro-2026-05-23` is retained
on the radxa host as the rollback target (image id `312c874ab7ff`).

The original deferred work (preserved here for archaeology / rollback):

1. **Bake fixed `server/backends/rk/tts.py`** (md5
   `4961497d910cac5531ceafe35e4f1713`, includes speaker-id kwarg pop). The
   source is in main repo since commit `c7302f9`. Drop the
   `/tmp/fixed-tts.py` bind-mount.

2. **Bake current submodule `kokoro_rknn.py`** (md5
   `ccc43371ee16465899f57e1ee4ed5a5f` at submodule commit `5a923ef`,
   4-stage + bucket-8 + bucket-16 router + misaki wire). Drop the
   `/tmp/fixed-kokoro_rknn.py` bind-mount.

3. **misaki + ZH deps are already in Dockerfile.rk** (commit `ca9332c`,
   lines 74-79). On rebuild they install at image-build time, no
   runtime `pip install` needed.

4. **(Optional) Bake the 12 bucket artifacts into the image** —
   3 buckets × 4 files = 12 files, ~370 MB total. Trade-off: bind-mount
   is easier to swap independently per-bucket; bake gives zero-deps
   `docker run` portability. Today the bind-mount layout under
   `/home/radxa/models/tts/kokoro-bucket-{8,16,32}/` is canonical; the
   bake step is deferred per bucket-16 spec "Phase 2c image persistence
   note".

5. **Append bucket-8 / bucket-16 manifest entries** — HF mirror under
   `harvestsu/seeed-local-voice-rk-artifacts/.../bucket{8,16}/` landed
   2026-05-23 (md5 round-trip verified against wsl2-local build host).
   Remaining work is to append the two manifest entries to
   `deploy/artifacts/rk_manifest.json`:
   - `rk3588-kokoro-hybrid-34pct-bucket8-2026-05-23`
   - `rk3588-kokoro-hybrid-34pct-bucket16-2026-05-23`
   so fresh deploys via `model_downloader` can pull all three buckets.

### Rebuild + validate

```bash
# On a host with Docker buildx + arm64 emulation (or natively on radxa)
docker buildx build \
  --platform linux/arm64 \
  -t openvoicestream:rk-kokoro-<next-tag> \
  -f deploy/docker/Dockerfile.rk .

# Deploy on radxa (drop the two /tmp/fixed-*.py mounts, keep buckets)
docker run -d --name openvoicestream-kokoro --restart=unless-stopped \
  --network host \
  -v /home/radxa/models/tts/kokoro-bucket-32:/opt/kokoro-rknn:ro \
  -v /home/radxa/models/tts/kokoro-bucket-8:/opt/kokoro-bucket-8:ro \
  -v /home/radxa/models/tts/kokoro-bucket-16:/opt/kokoro-bucket-16:ro \
  -e OVS_PROFILE=rk3588-kokoro-rknn-34pct \
  -e RK_ARTIFACT_SET=rk3588-kokoro-hybrid-34pct-2026-05-23 \
  -e KOKORO_RKNN_VOCODER_FRONT_PATH=kokoro-vocoder-front-half.native.fp16.rknn \
  -e KOKORO_RKNN_TAIL_REST_PATH=kokoro-vocoder-tail-rest-cpu.onnx \
  -e KOKORO_RKNN_BUCKET8_PREFIX_PATH=/opt/kokoro-bucket-8/kokoro-prefix-cpu-bucket8.onnx \
  # ... (same bucket-8 / bucket-16 env block as §3.D)
  openvoicestream:rk-kokoro-<next-tag>
```

Verification (post-rebuild):

```bash
# 1. md5 of in-image files (no overlay)
fleet exec radxa -- "docker exec openvoicestream-kokoro md5sum \
    /opt/speech/server/backends/rk/tts.py \
    /opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py"
# Expected: 4961497d910cac5531ceafe35e4f1713 / ccc43371ee16465899f57e1ee4ed5a5f

# 2. misaki present at startup with no pip install
fleet exec radxa -- "docker logs openvoicestream-kokoro 2>&1 | grep misaki"
# Expected: misaki ZH G2P (v1.1) loaded for Kokoro RKNN

# 3. Re-run §3.E TTFA matrix
```

---

## §7 Roadmap — follow-on optimizations

| Phase | Description | Est. effort | Expected gain | Risk |
| --- | --- | --- | --- | --- |
| **P2 Phase 3a** | Bucket-8 / bucket-16 tail-rest on NPU (no chunking) | 2-3 days | short-EN TTFA → 0.5-0.8 s; mid → 1.0-1.3 s | low (small static graphs, well under 8191 time-dim) |
| **P2 Phase 3b** | Bucket-32 chunked tail-rest NPU (overlap-add) | 1-2 weeks | long TTFA → 1.8-2.2 s | high — chunk boundary de-clicking, source/noise consistency, see `kokoro-rk-npu-42pct.md` §7 |
| **P4** | Decoder-front single-stage expand (eliminate second NPU dispatch round-trip) | TBD (codex Q4) | -50-100 ms TTFA per call | medium — needs full-graph compile success at expanded boundary |
| **P5** | INT8 bucket-8 decoder-front (per-bucket calibration) | 1 day | -100-200 ms TTFA short EN | low — calibration only; gate on existing audio rel_l2 ≤ 0.01 |
| **P6** | Bucket-64 / bucket-128 for long single-clause sentences | 1 week | removes 32-phoneme truncation; no TTFA change | medium — RKNN time-dim near 8191 limit, may need split |

The Phase 3 work is the next high-value step — current bottleneck for
mid- and long-sentence TTFA is the CPU tail-rest (Hifi-GAN style upsamp).

---

## §8 Rollback paths

Each layer of the stack can be rolled back independently without touching
the others.

| To disable | Action | Lands at |
| --- | --- | --- |
| misaki ZH G2P | restore `/tmp/fixed-kokoro_rknn.py.bak-pre-misaki`, `docker restart` | char-level lookup (ZH silent fail re-emerges) |
| bucket-16 router | drop the four `KOKORO_RKNN_BUCKET16_*` env vars + bucket-16 bind-mount, recreate container | bucket-8 / bucket-32 binary routing (Phase 2a behaviour) |
| bucket-8 router | drop the four `KOKORO_RKNN_BUCKET8_*` env vars + bucket-8 bind-mount, recreate container | bucket-32 4-stage only |
| 4-stage bucket-32 → 3-stage | drop `KOKORO_RKNN_VOCODER_FRONT_PATH` + `KOKORO_RKNN_TAIL_REST_PATH` env vars, recreate container | shipped 17 % decoder-front INT8 only |

All rollbacks preserve `server/backends/rk/tts.py` integrity (md5
`4961497d910cac5531ceafe35e4f1713`) — that fix is independent.

---

## §9 File map (main repo)

| File | Purpose |
| --- | --- |
| `server/backends/rk/tts.py` | RK TTS wrapper (speaker-id kwarg fix lives here) |
| `third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py` | 4-stage runtime + 3-bucket router + misaki wire (submodule) |
| `configs/profiles/rk3588-kokoro-rknn.json` | Default profile (now points at 34 % 4-stage env per `c7302f9`) |
| `configs/profiles/rk3588-kokoro-rknn-34pct.json` | Explicit 34 % profile (additive; was the M4 opt-in) |
| `deploy/docker/Dockerfile.rk` | Image build; includes misaki + ZH deps since `ca9332c` |
| `deploy/artifacts/rk_manifest.json` | `rk3588-kokoro-hybrid-34pct-2026-05-23` set (bucket-32 only today) |

---

## §10 Source spec index (chronological)

Read in this order for the *why* and the per-milestone parity data.

| Spec | Main commit | One-liner |
| --- | --- | --- |
| `kokoro-rk-npu-42pct.md` | `7bd7228` (+ `bd2b053` target revision) | R&D plan; target revised 42 % → 34 % after BERT dead-code finding |
| `kokoro-rk-42pct-m1-boundary-report.md` | `e50dc7b` | M1 — boundary tensor discovery (BERT and vocoder front-half) |
| `kokoro-bert-ab-audio-report.md` | `3b18517` | A/B audio shows BERT is bit-exact dead code in Kokoro v1.0 ONNX |
| `kokoro-rk-42pct-m2-bert-fp16.md` | `50463cb` | M2 — BERT FP16 RKNN built but not wired (per A/B finding) |
| `kokoro-rk-42pct-m4-vocoder-fp16.md` | `add7ddf` | M4 — vocoder front-half native FP16 RKNN (Sin poly rejected) |
| `kokoro-rk-34pct-m4m6-final.md` | `f482832` | 4-stage runtime + M6 manifest; opt-in (audio PASS, RTF FAIL at first) |
| `kokoro-rk-34pct-perf-diagnostic.md` | `4c98362` | Per-stage timing investigation of the first 4-stage RTF gap |
| `kokoro-rk-34pct-rtf-reconciliation.md` | `654ec72` | Root-caused the RTF gap to broken in-image `tts.py` |
| `kokoro-prod-image-stale-2026-05-23.md` | `f3a0edc` | Documented the stale image bind-mount workaround |
| `kokoro-rk-34pct-http-rtf-final.md` | `c7302f9` | **4-stage promoted to default** — HTTP RTF 0.59 (-10 % vs 3-stage) |
| `kokoro-rk-zh-mid-sentence-silent-fail-diag.md` | `afd834f` | Root-caused the ZH silent fail (char-level lookup misses Han) |
| `kokoro-rk-zh-fix-misaki.md` | `e2665d0` | misaki v1.1 ZH G2P wired at tokenizer front-end |
| `kokoro-rk-streaming.md` | `d014370` | Per-sentence streaming context |
| `kokoro-rk-bucket8-ttfa.md` | `735b5e9` | **Bucket-8 router shipped** — short EN TTFA 0.79 s |
| `kokoro-rk-bucket16-mid-ttfa.md` | `ca9332c` | **Bucket-16 router shipped** — mid EN/ZH TTFA 1.5-1.6 s + Dockerfile misaki bake |
| `kokoro-rk-34pct-reproduction-guide.md` | *this doc* | End-to-end reproduction recipe (consolidated) |

### Submodule `third_party/rkvoice-stream` (branch `feat/kokoro-rk-4stage-vocoder-front`)

| Commit | One-liner |
| --- | --- |
| `c566c66` | 4-stage runtime path |
| `d6b9463` | misaki ZH G2P wire at `_KokoroTokenizer.encode` |
| `c4c7706` | TTFA log field (`bucket=<N>` per-call) |
| `1efe6a6` | Bucket-8 router |
| `5a923ef` | Bucket-16 router |
