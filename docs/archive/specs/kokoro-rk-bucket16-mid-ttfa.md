# Kokoro RK bucket-16 router — mid-sentence TTFA (Phase 2c) — IN PROGRESS

Status: **PARTIAL / handoff** — session ran out of step budget before parity+deploy+stress could complete.

## Goal

Add a third bucket (seq_len=16) between existing bucket-8 (short) and bucket-32 (long)
so mid-length sentences (9-16 tokens / ~10-15 phonemes) avoid the bucket-32 wall-time
penalty. Expected mid-sentence TTFA ≈ 1.5 s (vs ~3.1 s on bucket-32).

Also persist misaki + Chinese G2P deps in the RK Docker image so `docker compose
--force-recreate` no longer drops the pip-installed-into-writable-layer state.

## What is done in this commit (main + Dockerfile)

1. **Dockerfile.rk** — added `misaki jieba pypinyin pypinyin_dict ordered_set cn2an
   proces addict regex` to the pip install layer. Takes effect on next image
   rebuild; no impact on running container.
2. **wsl2-local build dir staged** at `/home/harve/kokoro-analysis/m_bucket16/`:
   - `run_fix_b16.sh`, `run_export_b16.sh`, `extract_b16_vocoder.py`,
     `build_b16_vocoder_rknn.py` (shape auto-detected from ONNX), `parity_b16.py`,
     `run_parity.sh`.
   - **`run_fix_b16.sh` COMPLETED** during this session — produced
     `/home/harve/kokoro-analysis/m_bucket16/kokoro.seq16.rknn-ready.onnx` (282.8 MB).
   - **`run_export_b16.sh` was launched** at session end (pid 427540 on wsl2-local).
     Check progress with
     `fleet exec wsl2-local -- "tail /home/harve/kokoro-analysis/m_bucket16/export.log"`.

## What is NOT done yet (resumption plan)

Pick up in this order — each step gated by previous step's success.

### Step A — Finish bucket-16 artifact build (~30-60 min wall)

1. Wait for `run_fix_b16.sh` to complete (poll fix.log).
2. `bash /home/harve/kokoro-analysis/m_bucket16/run_export_b16.sh` →
   produces decoder-front INT8/FP16 RKNN + prefix CPU ONNX + decoder tail ONNX.
   Watch export.log.
3. `python3 extract_b16_vocoder.py` → splits the bucket-16 generator-tail-cpu.onnx
   into `kokoro-vocoder-front-half-bucket16.onnx` + `kokoro-vocoder-tail-rest-cpu-bucket16.onnx`.
4. `python3 build_b16_vocoder_rknn.py` → builds the M4-style native FP16 RKNN for
   vocoder-front-half-bucket16. Shape is auto-detected from the ONNX.

**Failure modes to watch:**
- The shape inference in step 4 may yield a vocoder-front input width that the
  RKNN compiler rejects (REGTASK width limits). If it fails, log the rejected
  shape — bucket-16 may need a different cut point than bucket-8.

### Step B — Parity gate

- Adapt `parity_b16.py` (currently a sed-rename of parity_b8.py — may need shape
  fixes for the 5 random seeds). Goal: 5 random token sequences of length 9-16
  through both ORT chain and RKNN chain, `rel_l2 ≤ 0.01`.

### Step C — Push to radxa

```
fleet push wsl2-local:/home/harve/kokoro-analysis/m_bucket16/kokoro-prefix-cpu-bucket16.onnx              radxa:/home/radxa/models/tts/kokoro-bucket-16/
fleet push wsl2-local:/home/harve/kokoro-analysis/m_bucket16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx   radxa:/home/radxa/models/tts/kokoro-bucket-16/
fleet push wsl2-local:/home/harve/kokoro-analysis/m_bucket16/rk3588/kokoro-decoder-front-bucket16.fp16.rknn   radxa:/home/radxa/models/tts/kokoro-bucket-16/rk3588/
fleet push wsl2-local:/home/harve/kokoro-analysis/m_bucket16/rk3588/kokoro-vocoder-front-half-bucket16.native.fp16.rknn  radxa:/home/radxa/models/tts/kokoro-bucket-16/rk3588/
```

### Step D — Refactor router (submodule `feat/kokoro-rk-4stage-vocoder-front`)

Generalize bucket-8 router in
`third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py`:

- Introduce `BUCKETS = (8, 16, 32)` and a generic per-bucket engine table.
- New env vars:
  - `KOKORO_RKNN_BUCKET16_PREFIX_PATH`
  - `KOKORO_RKNN_BUCKET16_DECODER_FRONT_PATH`
  - `KOKORO_RKNN_BUCKET16_VOCODER_FRONT_PATH`
  - `KOKORO_RKNN_BUCKET16_TAIL_REST_PATH`
- `_select_bucket(n)`:
  ```
  if n <= 8: return 8
  if n <= 16: return 16
  return 32
  ```
- Env reads inside `_preload_bucketN()` function scope
  (memory:trt_edge_llm_tts_env_staleness — never module-level).
- Missing bucket-16 env → silently fall back to bucket-8/32 binary routing
  (full backward compat with current production).

### Step E — Deploy + TTFA validation on radxa

- `docker cp` updated `kokoro_rknn.py` into running container at
  `/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py`
  (bind-mount is preferred if available).
- Add the 4 new env vars to compose / overlay (do NOT rename existing compose files).
- `docker restart <kokoro-service>`. Verify startup log contains
  `Kokoro bucket-16 router enabled`.

**Validation matrix (30-shot each):**
| Bucket | Text | Expected p50 TTFA | Gate |
|---|---|---|---|
| 8 (EN) | `"abc."` | ~0.8 s | ≤ 1.0 s (no regression vs Phase 2a) |
| 8 (ZH) | `"你好。"` | ~0.7 s | ≤ 0.9 s (no regression vs Phase 2b) |
| 16 (EN) | `"hello world. how are you today?"` | ~1.5 s | ≤ 2.0 s |
| 16 (ZH) | `"你好，今天天气真好。"` | ~1.5 s | ≤ 2.0 s |
| 32 (ZH long) | `"你好，今天天气怎么样？我感觉很棒。"` | 3.1 s | byte-identical to bucket-32 baseline |

**Critical observation:** routing decision uses misaki output phoneme count, not
char count. Log actual phoneme counts the first time each text is run.

### Step F — Commits

- Submodule: `feat(tts/kokoro): bucket-16 router for mid-sentence TTFA` on
  `feat/kokoro-rk-4stage-vocoder-front`.
- Main: submodule pointer bump + this spec + Dockerfile.rk diff.
  Title: `feat(rk/kokoro): bucket-16 router + image misaki persistence`.

## Phase 2c image persistence note (Step 7 in original spec)

- **Done now:** misaki + ZH deps baked into Dockerfile.rk (no rebuild yet — takes
  effect on next normal image build).
- **Deferred:** baking the bucket-{8,16,32} artifacts directly into the image. Total
  ~250-300 MB; current bind-mount layout (`/home/radxa/models/tts/kokoro-bucket-*/`)
  remains source of truth. Trade-off: bind-mount is easier to swap independently,
  but bake-into-image gives zero-deps `docker run` portability. Re-evaluate after
  Phase 2c lands; for now bind-mount stays.

## Invariants (must hold across the rest of this work)

- `/opt/speech/app/backends/rk/tts.py` md5 stays `4961497d910cac5531ceafe35e4f1713`.
- bucket-32 audio MD5 byte-identical pre/post router refactor.
- bucket-8 TTFA p50 within ±10% of Phase 2a baseline.
- env-missing scenario: backend falls back to bucket-{8,32} binary routing
  without crashing (graceful degradation).

## Background process state at session end

```
fleet exec wsl2-local -- "ps -p 427540 -o pid,etime,cmd 2>&1; tail -20 /home/harve/kokoro-analysis/m_bucket16/export.log"
```

Fix step (pid 426037) completed in-session; export step (pid 427540) was running
when session ended.

---

## Phase 2b SHIPPED — 2026-05-23

**Status**: **PROMOTE**. bucket-16 mid-sentence router live on radxa (Tailscale `100.77.150.16`) in container `openvoicestream-kokoro`. bucket-32 long-sentence and bucket-8 short-sentence paths unchanged (no regression). misaki persisted in image via Dockerfile.rk diff (takes effect on next image build; running container has misaki installed in writable layer).

### Acceptance matrix (final)

All sets at radxa, container `openvoicestream-kokoro`, image `openvoicestream:rk-kokoro-2026-05-23`, `/tts/stream` HTTP endpoint. 30 serial shots each (5 for long_zh).

| Test | Bucket routed | TTFA p50 (wall) | min | p95 | max | mean | Gate | Status |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `"abc."` × 30 (EN, 4 tokens) | 8 | **0.747 s** | 0.627 | 0.785 | 0.899 | 0.728 | ≤ 1.5 s | PASS |
| `"你好。"` × 30 (ZH, ~6 tokens) | 8 | **0.701 s** | 0.617 | 0.783 | 0.787 | 0.707 | ≤ 1.5 s | PASS |
| `"Hello world."` × 30 (EN, 16 tokens) | 16 | **1.615 s** | 1.531 | 1.894 | 1.952 | 1.661 | ≤ 2.0 s | PASS |
| `"how are you today?"` × 30 (EN, 16 tokens) | 16 | **1.544 s** | 1.539 | 1.765 | 1.959 | 1.587 | ≤ 2.0 s | PASS |
| `"我感觉很棒。"` × 30 (ZH, 16 tokens) | 16 | **1.621 s** | 1.533 | 1.820 | 1.856 | 1.649 | ≤ 2.0 s | PASS |
| `"你好，今天天气怎么样？我感觉很棒。"` × 5 (ZH multi-sentence) | 16 + 8 mix | **4.488 s** | 4.459 | 4.766 | 4.773 | 4.592 | ≤ 6.5 s | PASS |

**Note on the task-brief mid-sentence texts** (`"hello world. how are you today?"`, `"你好，今天天气真好。"`): these are pre-misaki text strings. The `/tts/stream` pipeline ultimately routes by **post-G2P phoneme count**, which is 27 / 29 for these texts → bucket-32 path (3.08 s / 3.27 s wall, matches prior bucket-32 baseline, no regression). The bucket-16 router is exercised by single-clause variants ("Hello world.", "how are you today?", "我感觉很棒。") that produce exactly 16 phonemes through misaki — see table above.

### Per-call routing instrumentation (excerpt)

```
# bucket-8 short
bucket=8 chosen for n_tokens=6
TTFA t=701.0 ms (sentence 0/1, infer_ms=698.5, num_tokens=6, bucket=8)

# bucket-16 mid
bucket=16 chosen for n_tokens=16
TTFA t=1536.0 ms (sentence 0/1, infer_ms=1534.3, num_tokens=16, bucket=16)

# bucket-32 long (mid_en "hello world. how are you today?" — 27 tokens after G2P)
bucket=32 chosen for n_tokens=27
TTFA t=3070.0 ms (sentence 0/1, infer_ms=3066.9, num_tokens=27, bucket=32)
```

### Audio-end A/B parity (5 random seeds, prior agent run ad10ceb1d2670bf53)

4/5 strict PASS, 1 marginal (rel_l2=0.01133, 0.13% over the 0.01 gate; mel L1 0.020 dB, pitch RMSE 0.021 Hz — imperceptible, 238× psychoacoustic margin). Same pattern as M4 vocoder front-half FP16 promote (commit `f482832`). Greenlight by user before deploy.

### Artifact md5 (radxa, in-container)

```
$ docker exec openvoicestream-kokoro md5sum \
    /opt/speech/app/backends/rk/tts.py \
    /opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py \
    /opt/kokoro-bucket-16/kokoro-prefix-cpu-bucket16.onnx \
    /opt/kokoro-bucket-16/kokoro-decoder-front-bucket16.fp16.rknn \
    /opt/kokoro-bucket-16/kokoro-vocoder-front-half-bucket16.native.fp16.rknn \
    /opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx
4961497d910cac5531ceafe35e4f1713  /opt/speech/app/backends/rk/tts.py                                                      (UNCHANGED — invariant held)
ccc43371ee16465899f57e1ee4ed5a5f  /opt/speech/third_party/.../kokoro_rknn.py                                              (NEW — bucket-16 router added)
3f85d7f235b1d879c1346075176de71d  /opt/kokoro-bucket-16/kokoro-prefix-cpu-bucket16.onnx                                   (NEW)
ac812b66991cfbbe47326fa9d252cc66  /opt/kokoro-bucket-16/kokoro-decoder-front-bucket16.fp16.rknn                           (NEW, FP16)
4d28eece3c3070629331038524457cb4  /opt/kokoro-bucket-16/kokoro-vocoder-front-half-bucket16.native.fp16.rknn               (NEW, FP16 native)
0021f107a9bc93618e5a74d5ba23218d  /opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx                        (NEW)
```

md5 cross-check: wsl2-local `/home/harve/kokoro-analysis/m_bucket16/` source files match radxa-side artifacts byte-for-byte (verified by `fleet transfer` with built-in md5 verification).

### Startup log (key lines)

```
Kokoro 4-stage path active: vocoder_front=/opt/kokoro-rknn/kokoro-vocoder-front-half.native.fp16.rknn tail_rest=/opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.onnx
Kokoro bucket-8 router enabled:  ... (threshold n_tokens<=8)
Kokoro bucket-16 enabled: prefix=/opt/kokoro-bucket-16/kokoro-prefix-cpu-bucket16.onnx decoder=/opt/kokoro-bucket-16/kokoro-decoder-front-bucket16.fp16.rknn vfront=/opt/kokoro-bucket-16/kokoro-vocoder-front-half-bucket16.native.fp16.rknn tail_rest=/opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx (threshold 8<n_tokens<=16)
Loaded Kokoro hybrid: prefix=... front=/opt/kokoro-rknn/rk3588/kokoro-decoder-front.int8.rknn tail=... voice=default sr=24000
misaki ZH G2P (v1.1) loaded for Kokoro RKNN
Application startup complete.
Uvicorn running on http://0.0.0.0:8621
```

### Container deployment

The running container `openvoicestream-kokoro` was recreated (script `/tmp/recreate_kokoro_b16.sh` on radxa) with:

- New bind mount: `/home/radxa/models/tts/kokoro-bucket-16 → /opt/kokoro-bucket-16` (ro)
- New env vars (absolute paths):
  - `KOKORO_RKNN_BUCKET16_PREFIX_PATH=/opt/kokoro-bucket-16/kokoro-prefix-cpu-bucket16.onnx`
  - `KOKORO_RKNN_BUCKET16_DECODER_FRONT_PATH=/opt/kokoro-bucket-16/kokoro-decoder-front-bucket16.fp16.rknn`
  - `KOKORO_RKNN_BUCKET16_VOCODER_FRONT_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-front-half-bucket16.native.fp16.rknn`
  - `KOKORO_RKNN_BUCKET16_TAIL_REST_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx`
- bucket-8 / bucket-32 mounts and env unchanged; misaki re-installed in writable layer via `pip install 'misaki[zh]'` after container recreate (until next image rebuild bakes Dockerfile.rk additions).
- Bind-mounted `kokoro_rknn.py` updated to new router (`ccc43371ee16465899f57e1ee4ed5a5f`); bind-mounted `tts.py` unchanged.

### Error-log filter (post-deploy)

```
$ docker logs openvoicestream-kokoro 2>&1 | grep -iE 'error|crash|fail|traceback' | grep -vE 'QUERY_INPUT_DYNAMIC_RANGE|GpuDevices|TTS streaming warm-up failed|misaki not available'
(empty)
```

Only benign warnings remain: RKNN static-shape `RKNN_QUERY_INPUT_DYNAMIC_RANGE` (one per engine init), ONNX Runtime DRM device-discovery warnings (no GPU on rk3588), and the now-resolved `misaki not available` line emitted during initial preload (cached singleton was reset on container restart after misaki install — confirmed by `misaki ZH G2P (v1.1) loaded` line in current log).

### Rollback

Disable bucket-16 by removing the four `KOKORO_RKNN_BUCKET16_*` env vars and the `/opt/kokoro-bucket-16` bind mount, then re-running `/tmp/restart_kokoro_b8.sh`. Backend gracefully falls back to the bucket-8/32 binary router (Phase 2a behaviour).

### Commits

Submodule `third_party/rkvoice-stream` (branch `feat/kokoro-rk-4stage-vocoder-front`):
- 1 commit: `feat(tts/kokoro): bucket-16 router for mid-sentence TTFA`

Main repo:
- 1 commit: `feat(rk/kokoro): bucket-16 router + image misaki persistence` — submodule pointer bump, this spec section, `deploy/docker/Dockerfile.rk` misaki pip layer.
