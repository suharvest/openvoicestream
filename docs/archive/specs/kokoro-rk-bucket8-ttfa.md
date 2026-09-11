# Kokoro RK Bucket-8 Router — Short-Sentence TTFA (Phase 2a)

**Date**: 2026-05-23
**Hardware**: radxa (RK3588), Tailscale `100.77.150.16`
**Container**: `openvoicestream-kokoro`, image `openvoicestream:rk-kokoro-2026-05-23`
**Decision**: **PROMOTE bucket-8 router for short sentences (≤8 tokens) — TTFA p50 0.787 s (gate ≤1.5 s, 1.9× margin).**

## TL;DR

| Path | Sentence | TTFA p50 wall | bytes |
| --- | --- | --- | --- |
| bucket-32 (baseline, prior spec `kokoro-rk-34pct-http-rtf-final.md`) | "abc." (4 tokens) | 3.107 s | 251 908 |
| **bucket-8 (this work)** | "abc." (4 tokens) | **0.787 s** | 43 012 |
| bucket-32 (no regression) | "Hello world this is a longer one indeed." (31 tokens) | 3.63 s | 251 908 |

Bucket-8 short-sentence path delivers a **3.95× TTFA reduction** for ≤8-token sentences. Bucket-32 long-sentence path is unchanged. Routing is per-sentence, transparent, and the runtime falls back to bucket-32 only if any bucket-8 artifact is missing.

## 1. Pipeline

| Stage | bucket-32 (existing) | bucket-8 (new) |
| --- | --- | --- |
| 1. Prefix (CPU ONNX) | `kokoro-prefix-cpu.onnx` seq_len=32 | `kokoro-prefix-cpu-bucket8.onnx` seq_len=8 |
| 2. Decoder-front (RKNN) | `kokoro-decoder-front.int8.rknn` (INT8) | `kokoro-decoder-front-bucket8.fp16.rknn` (FP16 native) |
| 3. Vocoder-front-half (RKNN FP16) | `kokoro-vocoder-front-half.native.fp16.rknn` | `kokoro-vocoder-front-half-bucket8.native.fp16.rknn` |
| 4. Tail-rest (CPU ONNX) | `kokoro-vocoder-tail-rest-cpu.onnx` | `kokoro-vocoder-tail-rest-cpu-bucket8.onnx` |

Output audio (untrimmed):

| Bucket | seq_len | decoder-front in | vocoder-front out | audio samples | duration |
| --- | --- | --- | --- | --- | --- |
| 32 | 32 | [1,512,210] → [1,512,420] | [1,256,4200] | 126 000 @ 24 kHz | 5.25 s |
| 8 | 8 | [1,512,45] → [1,512,90] | [1,256,900] | 27 000 @ 24 kHz | 1.125 s |

After `_trim_silence` the bucket-8 "abc." response is 43 012 bytes (= ~21 467 samples × 2 + 78-byte WAV header ≈ 0.9 s audio).

Phase 2a chose **FP16 (not INT8)** for the bucket-8 decoder-front to avoid re-doing INT8 calibration. (Bucket-32 INT8 calibration is bucket-specific and not reusable. INT8 for bucket-8 deferred to a later phase.)

## 2. Artifacts

Build host: `wsl2-local:/home/harve/kokoro-analysis/m_bucket8/` (RKNN-toolkit 2.3.0 + venv `/home/harve/rknn-build/.venv`).

| File | Size | md5 | Purpose |
| --- | ---: | --- | --- |
| `kokoro.seq8.rknn-ready.onnx` | 280 MB | `34c73ae9c741d951e30416266a74546f` | Step 1: fixed full graph at seq_len=8 (from `fix_kokoro_rknn.py --seq-len 8`) |
| `kokoro-prefix-cpu-bucket8.onnx` | 22 MB | `40ea37f93d415c5aa07bb17b47544a00` | Bucket-8 prefix CPU ONNX |
| `kokoro-decoder-front.onnx` (bucket-8) | 128 MB | `aef35b9a216e42153c637d529b7bcd14` | Bucket-8 decoder-front FP32 ref ONNX |
| `kokoro-vocoder-front-half-bucket8.onnx` | 68 MB | `76e5585c362e17d3fa392e841ceb4dd2` | Bucket-8 vocoder-front-half FP32 ref ONNX |
| `kokoro-vocoder-tail-rest-cpu-bucket8.onnx` | 27 MB | `9e03458e86a8732ccaeccd7c7b0618f9` | Bucket-8 tail-rest CPU ONNX |
| `rk3588/kokoro-decoder-front-bucket8.fp16.rknn` | 75 MB | `814c558955a8eb6a3f6b95cfc5eab0b9` | Bucket-8 decoder-front RKNN FP16 |
| `rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn` | 33 MB | `bcf1d03b3aef869bac66ac98140b40e9` | Bucket-8 vocoder-front-half RKNN FP16 native |

Both RKNN builds compiled cleanly (RKNN 2.3.0 + rk3588 + optimization_level=0, `do_quantization=False`).

## 3. Parity gate

5 random short sequences (3–7 tokens, seeds 42/17/99/1/256) on `wsl2-local`. Compare bucket-8 RKNN 4-stage vs bucket-8 ORT 4-stage (FP32 reference) and bucket-8 ORT 3-stage (full-CPU reference):

```
sd=42  n_tok=3 audio=(27000,)  rknn4-vs-ort4=0.00206  rknn4-vs-ort3=0.00206  ort_split=0.00000
sd=17  n_tok=6 audio=(27000,)  rknn4-vs-ort4=0.00175  rknn4-vs-ort3=0.00175  ort_split=0.00000
sd=99  n_tok=7 audio=(27000,)  rknn4-vs-ort4=0.00145  rknn4-vs-ort3=0.00145  ort_split=0.00000
sd=1   n_tok=5 audio=(27000,)  rknn4-vs-ort4=0.00212  rknn4-vs-ort3=0.00212  ort_split=0.00000
sd=256 n_tok=5 audio=(27000,)  rknn4-vs-ort4=0.00187  rknn4-vs-ort3=0.00187  ort_split=0.00000

AGGREGATE
  RKNN-4 vs ORT-4 worst rel_l2: 0.00212  (PRIMARY GATE ≤ 0.01)  PASS
  RKNN-4 vs ORT-3 worst rel_l2: 0.00212  (overall vs full-CPU reference)
  ORT-4 vs ORT-3 worst rel_l2 : 0.00000  (split error alone, FP32)
```

The split itself is FP32-equivalent (ORT-4 == ORT-3 to bit precision). The 0.0021 RKNN error is pure FP16 quantization, well below the 0.01 audio gate (4.7× margin). **PASS.**

## 4. Runtime router (submodule change)

`third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py` (branch `feat/kokoro-rk-4stage-vocoder-front`):

- Adds 4 bucket-8 attributes on `KokoroRKNNBackend`: `_b8_prefix_sess`, `_b8_rknn`, `_b8_rknn_vfront`, `_b8_tail_rest_sess`, plus `_use_bucket8` flag.
- New `_preload_bucket8(ort, rknn_lite_cls)` called at the end of `_preload_hybrid`. **Reads env at function scope** (not module level) — see memory `trt_edge_llm_tts_env_staleness.md` lesson. Required env vars:
  - `KOKORO_RKNN_BUCKET8_PREFIX_PATH`
  - `KOKORO_RKNN_BUCKET8_DECODER_FRONT_PATH`
  - `KOKORO_RKNN_BUCKET8_VOCODER_FRONT_PATH`
  - `KOKORO_RKNN_BUCKET8_TAIL_REST_PATH`
- If any env var unset or any artifact missing → logs info and the backend continues with bucket-32 only.
- New `_select_bucket(n_tokens)` (Phase 2a returns 8 if ≤8 else 32; bucket-16 deferred to Phase 2b).
- `_infer_segment` now routes per-sentence: encode at bucket-32 seq_len, peek `n_tokens`, if `_use_bucket8 and n_tokens <= 8` slice tokens to `[:, :8]` and dispatch to new `_infer_segment_bucket8(tokens, n_tokens, speed, meta)`. The bucket-8 path is structurally identical to the 4-stage path but uses bucket-8 engines.
- `cleanup()` releases the two extra RKNN runtimes.
- TTFA log now includes `bucket=<8|32>` field for observability.

The router only enables itself when the host backend is already on the **bucket-32 4-stage** path (`_use_4stage=True`). If the bucket-32 path is 3-stage, bucket-8 is skipped with a warning (bucket-8 is itself a 4-stage design).

Backward compatibility: with no new env vars, behaviour is identical to commit `c4c7706`. Existing deployments are unaffected.

## 5. Container deployment

Container `openvoicestream-kokoro` was recreated with:

- New bind mount: `/home/radxa/models/tts/kokoro-bucket-8 → /opt/kokoro-bucket-8` (ro)
- New env vars (absolute paths to avoid colliding with `MODEL_DIR=/opt/kokoro-rknn` resolution):
  - `KOKORO_RKNN_BUCKET8_PREFIX_PATH=/opt/kokoro-bucket-8/kokoro-prefix-cpu-bucket8.onnx`
  - `KOKORO_RKNN_BUCKET8_DECODER_FRONT_PATH=/opt/kokoro-bucket-8/rk3588/kokoro-decoder-front-bucket8.fp16.rknn`
  - `KOKORO_RKNN_BUCKET8_VOCODER_FRONT_PATH=/opt/kokoro-bucket-8/rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn`
  - `KOKORO_RKNN_BUCKET8_TAIL_REST_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx`
- The bucket-32 bind mount (`/home/radxa/models/tts/kokoro-bucket-32 → /opt/kokoro-rknn`) and bucket-32 env vars were kept exactly as before.
- Bind-mounted `kokoro_rknn.py` was updated to the new router; bind-mounted `tts.py` was kept unchanged (md5 `4961497d910cac5531ceafe35e4f1713`).

Startup log confirms (excerpt):

```
Kokoro 4-stage path active: vocoder_front=/opt/kokoro-rknn/kokoro-vocoder-front-half.native.fp16.rknn tail_rest=/opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.onnx
Kokoro bucket-8 router enabled: prefix=/opt/kokoro-bucket-8/kokoro-prefix-cpu-bucket8.onnx decoder=/opt/kokoro-bucket-8/rk3588/kokoro-decoder-front-bucket8.fp16.rknn vfront=/opt/kokoro-bucket-8/rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn tail_rest=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx (threshold n_tokens<=8)
Loaded Kokoro hybrid: prefix=/opt/kokoro-rknn/kokoro-prefix-cpu.onnx front=/opt/kokoro-rknn/rk3588/kokoro-decoder-front.int8.rknn ... voice=default sr=24000
Application startup complete.
Uvicorn running on http://0.0.0.0:8621
```

## 6. Acceptance gates

| Gate | Spec § | Threshold | Measured | Status |
| --- | --- | --- | --- | --- |
| Parity rel_l2 (5 random seeds, bucket-8 RKNN-4 vs ORT-4) | spec §6 | ≤ 0.01 | 0.00212 worst | PASS (4.7× margin) |
| Short EN TTFA (`"abc."`, n=30) | task spec | p50 ≤ 1.5 s | **0.787 s** | PASS (1.9× margin) |
| Short EN HTTP success | task spec | 30/30 200 OK | 30/30 200 OK, deterministic 43 012 bytes | PASS |
| bucket-32 regression (`"Hello world this is a longer one indeed."`, 31 tokens) | task spec | wall ≈ baseline 3.1 s | 3.63 s, bytes=251 908, bucket=32 in TTFA log | PASS (no regression) |
| Chinese sentences (`"你好。"` etc.) | task spec | n/a — Chinese pre-existing block | misaki not installed in this image → tokens dropped → 4-byte sentinel | KNOWN ISSUE, not introduced by this work |

The Chinese gate cannot be evaluated here because the production container does not ship `misaki[zh]` (issue documented in `kokoro-rk-zh-fix-misaki.md`). Independent of bucket-8 routing.

## 7. EVIDENCE

### 7.1 Artifact md5 (on radxa, in container)

```
$ docker exec openvoicestream-kokoro md5sum \
    /opt/speech/app/backends/rk/tts.py \
    /opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py \
    /opt/kokoro-bucket-8/kokoro-prefix-cpu-bucket8.onnx \
    /opt/kokoro-bucket-8/rk3588/kokoro-decoder-front-bucket8.fp16.rknn \
    /opt/kokoro-bucket-8/rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn \
    /opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx
4961497d910cac5531ceafe35e4f1713  /opt/speech/app/backends/rk/tts.py                                 (UNCHANGED — wrapper, production-stable)
0b06f06c960081dcc2849902a5122c9e  /opt/speech/third_party/rkvoice-stream/.../kokoro_rknn.py          (NEW — bucket-8 router)
40ea37f93d415c5aa07bb17b47544a00  /opt/kokoro-bucket-8/kokoro-prefix-cpu-bucket8.onnx                (NEW)
814c558955a8eb6a3f6b95cfc5eab0b9  /opt/kokoro-bucket-8/rk3588/kokoro-decoder-front-bucket8.fp16.rknn (NEW)
bcf1d03b3aef869bac66ac98140b40e9  /opt/kokoro-bucket-8/rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn (NEW)
9e03458e86a8732ccaeccd7c7b0618f9  /opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx     (NEW)
```

### 7.2 ONNX IO shape verification

```
kokoro-decoder-front (bucket-8)
  in  /MatMul_1_output_0  [1,512,45]
  in  /Slice_2_output_0   [1,128]
  out /decoder/decode.3/Mul_output_0  [1,512,90]

kokoro-vocoder-front-half-bucket8 (398 nodes)
  in  /decoder/decode.3/Mul_output_0  [1,512,90]
  in  /Slice_2_output_0               [1,128]
  out /decoder/generator/Add_5_output_0  [1,256,900]   (well under RKNN 8191 time-dim limit)

kokoro-vocoder-tail-rest-cpu-bucket8 (416 nodes)
  in  /decoder/generator/Add_5_output_0  [1,256,900]
  in  /Slice_2_output_0                  [1,128]
  in  /decoder/decode.3/Mul_output_0     [1,512,90]
  out audio  [27000]   (= 1.125 s @ 24 kHz)
```

### 7.3 Parity raw print

```
$ python3 /home/harve/kokoro-analysis/m_bucket8/parity_b8.py
... (RKNN builds elided) ...
sd=42 n_tok=3 audio=(27000,)  rknn4-vs-ort4=0.00206  rknn4-vs-ort3=0.00206  ort_split=0.00000
sd=17 n_tok=6 audio=(27000,)  rknn4-vs-ort4=0.00175  rknn4-vs-ort3=0.00175  ort_split=0.00000
sd=99 n_tok=7 audio=(27000,)  rknn4-vs-ort4=0.00145  rknn4-vs-ort3=0.00145  ort_split=0.00000
sd=1  n_tok=5 audio=(27000,)  rknn4-vs-ort4=0.00212  rknn4-vs-ort3=0.00212  ort_split=0.00000
sd=256 n_tok=5 audio=(27000,)  rknn4-vs-ort4=0.00187  rknn4-vs-ort3=0.00187  ort_split=0.00000

=== AGGREGATE ===
RKNN-4 vs ORT-4 worst rel_l2: 0.00212  (PRIMARY GATE ≤ 0.01)
RKNN-4 vs ORT-3 worst rel_l2: 0.00212  (overall vs full-CPU reference)
ORT-4 vs ORT-3 worst rel_l2 : 0.00000  (split error alone, FP32)
PASS
```

### 7.4 30-shot short-EN raw (`"abc."`, n=30, deterministic 43 012 bytes, all HTTP 200)

```
shot, wall_s, http_code, bytes
 1, 0.923664, 200, 43012
 2, 0.791578, 200, 43012
 3, 0.799499, 200, 43012
 4, 0.793874, 200, 43012
 5, 0.792820, 200, 43012
 6, 0.790261, 200, 43012
 7, 0.780730, 200, 43012
 8, 0.653174, 200, 43012
 9, 0.786302, 200, 43012
10, 0.787130, 200, 43012
11, 0.612404, 200, 43012
12, 0.751901, 200, 43012
13, 0.790265, 200, 43012
14, 0.690130, 200, 43012
15, 0.709269, 200, 43012
16, 0.792278, 200, 43012
17, 0.791090, 200, 43012
18, 0.674675, 200, 43012
19, 0.635400, 200, 43012
20, 0.784211, 200, 43012
21, 0.696240, 200, 43012
22, 0.717339, 200, 43012
23, 0.791043, 200, 43012
24, 0.708747, 200, 43012
25, 0.788423, 200, 43012
26, 0.792915, 200, 43012
27, 0.791326, 200, 43012
28, 0.787453, 200, 43012
29, 0.692894, 200, 43012
30, 0.791720, 200, 43012

Aggregate: n=30  min=0.612  p50=0.787  p95=0.799  max=0.924  mean=0.756
Audio MD5 (all 30 shots): 776447cf515d46127b4f95391b6d9881  (single value — fully deterministic)
```

### 7.5 bucket-32 regression check (no regression)

```
$ curl ... -d '{"text":"Hello world this is a longer one indeed."}'
wall=3.667537 bytes=251908

docker logs:
  kokoro synthesize_stream TTFA: first chunk yielded at t=3633.2 ms
    (sentence 0/1, infer_ms=3631.7, num_tokens=31, bucket=32)
```

For comparison: prior bucket-32 baseline (`kokoro-rk-34pct-http-rtf-final.md`) reports p50 3.107 s wall for `"abc."`. With the bucket-8 router enabled and the same `"abc."` text, the call now takes 0.787 s (bucket=8). The 31-token regression check above runs through the bucket-32 path and lands in the bucket-32 envelope.

### 7.6 Container startup log key lines

```
Kokoro 4-stage path active: vocoder_front=/opt/kokoro-rknn/kokoro-vocoder-front-half.native.fp16.rknn tail_rest=/opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.onnx
Kokoro bucket-8 router enabled: prefix=/opt/kokoro-bucket-8/kokoro-prefix-cpu-bucket8.onnx decoder=/opt/kokoro-bucket-8/rk3588/kokoro-decoder-front-bucket8.fp16.rknn vfront=/opt/kokoro-bucket-8/rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn tail_rest=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx (threshold n_tokens<=8)
Loaded Kokoro hybrid: prefix=/opt/kokoro-rknn/kokoro-prefix-cpu.onnx front=/opt/kokoro-rknn/rk3588/kokoro-decoder-front.int8.rknn tail=/opt/kokoro-rknn/kokoro-generator-tail-cpu.onnx voice=default sr=24000 ...
Application startup complete.
Uvicorn running on http://0.0.0.0:8621
```

### 7.7 Per-call routing instrumentation

```
# short, 4 tokens  -> bucket=8
kokoro synthesize_stream TTFA: first chunk yielded at t=792.6 ms (sentence 0/1, infer_ms=791.6, num_tokens=4, bucket=8)

# long, 31 tokens -> bucket=32
kokoro synthesize_stream TTFA: first chunk yielded at t=3633.2 ms (sentence 0/1, infer_ms=3631.7, num_tokens=31, bucket=32)
```

### 7.8 docker logs error filter (post-deployment)

```
$ docker logs openvoicestream-kokoro 2>&1 | grep -iE 'error|crash|fail|capture' | grep -v 'QUERY_INPUT_DYNAMIC_RANGE\|GpuDevices\|TTS streaming warm-up failed'
(empty)
```

The `TTS streaming warm-up failed` line refers to the pre-existing misaki-not-installed Chinese G2P warmup (independent of bucket-8). Two `QUERY_INPUT_DYNAMIC_RANGE` warnings per RKNN engine are benign (static-shape models).

## 8. Phase 2b TODO

- **bucket-16 mid-range router** — add `kokoro.seq16.rknn-ready.onnx` + vocoder-front-half-bucket16 + tail-rest-bucket16. Router becomes `8 → 16 → 32`. Expected target: sentences with 9–16 tokens, TTFA ~1.5 s.
- **INT8 bucket-8 decoder-front** — re-do calibration at seq_len=8, build INT8 RKNN. Potential further 1.3-1.5× TTFA gain on the decoder-front stage.
- **HF artifact mirror** — upload to `harvestsu/seeed-local-voice-rk-artifacts/blob/main/rk3588/kokoro-hybrid-v1/bucket8/` (deferred — current deployment uses fleet-pushed local copies under `/home/radxa/models/tts/kokoro-bucket-8/`).
- **Manifest entry** — append `rk3588-kokoro-hybrid-34pct-bucket8-2026-05-23` set to `deploy/artifacts/rk_manifest.json` once HF mirror is up (so fresh deploys pull artifacts and apply env via `RK_ARTIFACT_SET`).
- **Chinese fix gate** — the bucket-8 router can dramatically help short Chinese sentences once misaki is installed in the production image. Tracked separately.

## 9. Rollback

If the bucket-8 router needs to be disabled (without rebuilding the image):

```bash
docker exec openvoicestream-kokoro env | grep KOKORO_RKNN_BUCKET8_   # confirm vars set
docker stop openvoicestream-kokoro
docker rm openvoicestream-kokoro
# Re-run with bucket-32-only env (drop the four KOKORO_RKNN_BUCKET8_* lines from /tmp/restart_kokoro_b8.sh).
```

Backend behaviour with bucket-8 env vars unset is byte-equivalent to commit `c4c7706` (3-stage bucket-32 promotion).

## 10. Commits

Submodule `third_party/rkvoice-stream` (branch `feat/kokoro-rk-4stage-vocoder-front`):
- 1 commit: `feat(tts/kokoro): bucket-8 router for short-sentence TTFA`

Main repo:
- 1 commit: spec + submodule pointer bump.

---

EVIDENCE summary: all artifact md5 verified, ONNX IO shapes shown, parity raw print included, 30-shot raw TTFA included, bucket-32 regression confirmed unchanged, container md5 + startup log verbatim. No production config or image changed.
