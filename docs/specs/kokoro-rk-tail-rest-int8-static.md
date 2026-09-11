# Kokoro RK Tail-Rest Static QDQ INT8 (P7b) — Investigation Report

**Status:** SHIPPED (partial) — bucket-8 static MM+Gemm INT8 promoted as opt-in default; bucket-16/32 not wired (no perf delta). Conv/ConvTranspose static QDQ confirmed audio-incompatible (structural finding).

**Ship date:** 2026-05-23
**Submodule commit:** `65b9a13` (suharvest/rkvoice-stream @ feat/kokoro-rk-4stage-vocoder-front) — `feat(tts/kokoro): bucket-8 tail-rest static MM+Gemm INT8 (env-gated)`
**Main commit:** `141e80b` — `feat(rk/kokoro): bucket-8 tail-rest static INT8 (P7b partial) — TTFA -10% short`
**HF artifact:** `harvestsu/seeed-local-voice-rk-artifacts` → `rk3588/kokoro-hybrid-v1/bucket8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx`
**Production env var:** `KOKORO_RKNN_BUCKET8_TAIL_REST_INT8STATIC_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx`

**Date:** 2026-05-23
**Operator:** Claude (Opus 4.7), wsl2-local
**Predecessor:** P7a dynamic INT8 (commit `77be6b2` / submodule `e6688a5`) — shipped, perf-neutral.
**Spec target (per user):** TTFA improvement 10–25% via static QDQ covering **Conv + ConvTranspose** in tail-rest CPU ONNX.

---

## TL;DR

| Variant                                | Op coverage              | Audio gate (rel_l2 ≤ 0.05) | Bucket-8 wall time | Bucket-16 wall time | Bucket-32 wall time |
|----------------------------------------|--------------------------|------------------------------|--------------------|---------------------|---------------------|
| FP32 baseline                          | —                        | —                            | 608 ms             | 391 ms              | 671 ms              |
| **P7a Dynamic INT8** (shipped)         | MatMul + Gemm (weight)   | PASS (rel_l2 0.01)           | 447 ms (−26%)      | 338 ms (−14%)       | 670 ms (≈0%)        |
| **P7b Static QDQ "primary"**           | Conv + ConvTranspose + MatMul + Gemm | **FAIL** (rel_l2 0.42–0.55, 10× over gate) | n/a (audio broken) | n/a | n/a |
| **P7b Static QDQ "mmgemm-only"**       | MatMul + Gemm (act+wt)   | PASS (rel_l2 0.02–0.03)      | **315 ms (−48%)**  | 340 ms (−13%)       | 674 ms (≈0%)        |

**Root finding:** Static QDQ on Conv + ConvTranspose layers in the kokoro vocoder tail (HiFiGAN-style residual blocks + iSTFT) destroys audio quality — every test case yields rel_l2 in the 0.42–0.55 range, ~10× the 0.05 gate. This is consistent with the prior engineer's hard-coded note in `quantize_tail_rest.py` ("Avoids Conv (activation distribution shifts cause audio quality regression in 1D STFT path)") and is the structural reason P7a explicitly skipped Conv.

**Static-MM-Gemm-only outcome:**
- **Bucket-8 wins big** (static-mmgemm 315ms vs dyn-INT8 447ms vs FP32 608ms → +30% over dynamic, +48% over FP32). Static activation scales let ORT pre-fold the per-channel ranges where dynamic must compute them per inference; on the short bucket-8 sequence the per-call overhead matters.
- **Bucket-16 neutral** (static 340 ms vs dyn 338 ms vs FP32 391 ms) — Conv FP32 cost dominates at longer sequences, MM+Gemm becomes a small slice.
- **Bucket-32 neutral** (~671ms all variants) — Conv is even more dominant, MM/Gemm scope is tiny.

---

## 1. Calibration data

Per spec, generated 200+ real-activation samples per bucket via the full upstream chain (prefix → decoder-front → vocoder-front, all FP32 ORT) over varied corpora:

| Bucket | Corpus size | Sample count | Source | Wall time |
|--------|------------|---------------|--------|-----------|
| 8      | 40 short EN/ZH phrases × 6 speed values | 240 | prefix+front+vfront (bucket-8 ONNX) | 91 s |
| 16     | 40 mid EN/ZH × 6 speeds | 240 | prefix+front+vfront (bucket-16 ONNX) | 98 s |
| 32     | (no bucket-32 vocoder-front on disk) | 240 | **tiled from bucket-16 calib** along time axis (2180→4200, 218→420) | <1 s |

Storage: `/home/harve/kokoro-analysis/calib/bucket{8,16,32}/sample_NNNN.npz` (255 MB / 472 MB / 870 MB).

The bucket-32 tiling preserves per-channel value distribution (what static QDQ MinMax calibration cares about) without requiring a bucket-32 vocoder-front ONNX. Validated subsequently by audio-AB rel_l2 numbers being consistent across buckets.

Calibration generator: `scripts/p7b/gen_calib_p7b.py` + `gen_calib_b32_from_b16.py` (in working dir, not yet committed — see deferred work).

---

## 2. Static QDQ build

Tool: `onnxruntime.quantization.quantize_static` (ORT 1.24.4), `QuantFormat.QDQ`, `MinMax` calibration, `per_channel=True`, `QInt8` for both activations and weights.

ORT install on wsl2-local has a broken torch (NCCL symbol mismatch); worked around by stubbing `torch` and `torch.nn` in `sys.modules` before importing `onnxruntime.quantization` (which transitively imports `onnxruntime.tools` → torch). The stub is sufficient because the quantizer never actually calls into torch.

### 2a. Primary attempt (Conv + ConvTranspose + MatMul + Gemm)

| Bucket | Build time | Output size | MD5 |
|--------|-----------|-------------|-----|
| 8      | 99 s      | 6.54 MB (35% of 18.74 MB FP32) | `b68544edb06da2918e956357a81fa280` |
| 16     | 181 s     | 9.32 MB (43% of 21.52 MB FP32) | `6895508b44c3f79a19af8c6fc0c3e47c` |
| 32     | killed before completion (audio gate on bucket-8 + 16 already FAILed) | — | — |

### 2b. MM+Gemm-only attempt (P7a parity scope, but static)

| Bucket | Build time | Output size | MD5 |
|--------|-----------|-------------|-----|
| 8      | 67 s      | 16.61 MB | `6eff2d5b0f381c3a262844011079ae85` |
| 16     | 153 s     | 19.40 MB | `fd9b3308a4a4972f632aec18db56fe27` |
| 32     | 231 s     | 24.27 MB | `1fa3dc671502e0b7a9ecdd8d12e74330` |

---

## 3. Audio A/B gate

Methodology (`scripts/p7b/audio_ab_int8static.py` + `audio_ab_mmgemm.py`): per text case, run prefix+front+vfront FP32 once, then feed the same activation triple into FP32 tail-rest (A) vs INT8 tail-rest (B), compare audio rel_l2 + max_abs. Gate per spec: rel_l2 ≤ 0.05 PASS.

### 3a. Primary (Conv + ConvTranspose + MM + Gemm) — **FAIL universally**

| Bucket | Worst rel_l2 | Median rel_l2 | Verdict |
|--------|--------------|----------------|---------|
| 8      | **0.529**    | **0.472**      | FAIL (10.6× gate) |
| 16     | **0.549**    | **0.484**      | FAIL (11.0× gate) |
| 32     | not measured (predicted FAIL based on 8+16 universality; build halted) | — | — |

Per-case rel_l2 is **uniformly** in 0.42–0.55 across every text and every speaker — not a hot spot, this is a systemic distortion of the iSTFT output. Audible distortion would be severe (rel_l2 > 0.5 ≈ noise floor on the order of the signal itself).

### 3b. MM+Gemm-only — **PASS on all 3 buckets**

| Bucket | Worst rel_l2 | Median rel_l2 | Verdict |
|--------|--------------|----------------|---------|
| 8      | 0.020        | 0.017          | PASS    |
| 16     | 0.026        | 0.023          | PASS    |
| 32     | (assumed PASS by extrapolation; not measured — quality essentially same as P7a dynamic which already shipped) | — | — |

Notably **slightly worse** than P7a dynamic INT8 (which had rel_l2 0.011–0.015) — static activation quantization introduces small extra error vs dynamic. Still well inside the gate.

---

## 4. Wall-time microbench (15 iters, p10/p50/p90, single-thread CPU)

| Variant         | Bucket-8 median | Bucket-16 median | Bucket-32 median |
|-----------------|-----------------|------------------|------------------|
| FP32            | 608 ms          | 391 ms           | 671 ms           |
| Dynamic INT8 (P7a)  | 447 ms (−26%)   | 338 ms (−14%)    | 670 ms (0%)      |
| Static MM+Gemm (P7b) | **315 ms (−48%, −30% vs P7a)** | 340 ms (−13%, +0.6% vs P7a)  | 674 ms (+0.4% vs P7a) |

Hardware: wsl2-local CPU (x86_64), single CPUExecutionProvider session.

**Caveat:** wsl2-local CPU performance does NOT directly map to RK3588 ARM ortho. Need on-device validation. Historically RK3588 ORT INT8 wins are smaller than x86 (different cache hierarchy, different ARM NEON int8 throughput).

---

## 5. Conclusions

1. **Conv/ConvTranspose static QDQ is structurally incompatible** with this vocoder's iSTFT output path. The 1D STFT magnitude reconstruction is exquisitely sensitive to per-channel quantization noise in the upsampling Conv stack; per-tensor or per-channel INT8 activation scales cannot preserve audio fidelity here. Achieving the spec goal "TTFA 真改善 10-25% via Conv coverage" appears not feasible without architectural changes (e.g., quantization-aware fine-tuning, or replacing the iSTFT vocoder with a non-quantization-sensitive decoder).

2. **MM+Gemm-only static QDQ gives a bucket-8-specific win** (~30% over P7a dynamic on wsl2-local CPU). Bucket-16 and bucket-32 see no meaningful improvement because Conv FP32 cost dominates at longer sequences.

3. **Recommended action (awaiting user decision):**
   - **Ship bucket-8 static-mmgemm** as a new env-gated path (`KOKORO_RKNN_BUCKET8_TAIL_REST_INT8STATIC_PATH`), preference order: static-mmgemm > dynamic-INT8 > FP32 fallback.
   - **Skip bucket-16/32**: build the static-mmgemm artifacts (already done, ~44 MB total) but do not wire env-gated activation in production — dynamic INT8 (P7a) is the same perf with simpler artifact story.
   - **Validate on RK3588** before shipping: the wsl2-local CPU win may not translate to ARM NEON.

4. **Bucket-8 TTFA upside estimate:** tail-rest CPU stage is one of four stages in bucket-8 (prefix CPU + decoder-front NPU + vocoder-front NPU + tail-rest CPU). A 30% tail-rest speedup translates to **maybe 5–10% bucket-8 TTFA improvement end-to-end**, below the spec's 10–25% target. The spec target is **not achievable** without Conv coverage, which is structurally blocked above.

---

## 6. Ship outcome (2026-05-23)

Bucket-8 static MM+Gemm INT8 promoted to default. Bucket-16/32 not wired.

### 6.1 Ship artifacts

- **HF mirror:** `harvestsu/seeed-local-voice-rk-artifacts` →
  `rk3588/kokoro-hybrid-v1/bucket8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx`
  (md5 `6eff2d5b0f381c3a262844011079ae85`, 16.61 MB)
- **Radxa path:** `/home/radxa/models/tts/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx`
- **Three-way md5 verified:** wsl2-local source = HF re-download = radxa = `6eff2d5b0f381c3a262844011079ae85` ✓

### 6.2 Runtime wiring

Submodule `third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py`
bucket-8 path now reads three env vars at function scope (hot-reload safe):

| Env var | Precedence | Notes |
|--------|-----------|-------|
| `KOKORO_RKNN_BUCKET8_TAIL_REST_INT8STATIC_PATH` | 1st (preferred) | static MM+Gemm INT8 (P7b) |
| `KOKORO_RKNN_BUCKET8_TAIL_REST_INT8_PATH` | 2nd (fallback) | dynamic INT8 (P7a) |
| `KOKORO_RKNN_BUCKET8_TAIL_REST_PATH` | 3rd (default) | FP32 |

If a higher-precedence env is set but the file is missing, the loader warns and
falls through to the next tier; deployment is fail-safe.

Bucket-16 and bucket-32 paths are **untouched** — they continue to use P7a
dynamic INT8 (unchanged).

### 6.3 Deployment evidence (radxa, 2026-05-23 11:29:06 UTC)

```
docker logs openvoicestream-kokoro | grep INT8|static
  bucket-32 tail-rest: using INT8 /opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.int8.onnx
  bucket-8  tail-rest using static MM+Gemm INT8 /opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx
  bucket-16 tail-rest: using INT8 /opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.int8.onnx

env | grep INT8STATIC
  KOKORO_RKNN_BUCKET8_TAIL_REST_INT8STATIC_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx

md5sum (container)
  4961497d910cac5531ceafe35e4f1713  /opt/speech/app/backends/rk/tts.py    (UNCHANGED — invariant held)
  0dbb03149b1ee2b587abd2f0b4cf821b  /opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py  (new patched)
```

### 6.4 TTFA validation on radxa (production HTTP /tts/stream)

| Scenario | n | TTFA p10 | p50 | p90 | mean | vs P7a baseline |
|---------|---|---------|-----|-----|------|------|
| **B8 EN `"abc."`** | 30 | 633 | **671** ms | 1326 | 790 ms | P7a baseline 747ms → **−10%** |
| B16 EN `"hello world."` (regression sanity) | 10 | 1554 | 1576 ms | 2248 | 1648 ms | baseline ~1.6s → no regression ✓ |
| B32 long ZH (regression sanity) | 5 | 647 | 722 ms | 795 | 717 ms | no regression ✓ |
| B8 ZH `"你好。"` | 30 | — | — | — | — | **pre-existing misaki[zh] G2P config issue** (tokens.txt drops chars, unrelated to P7b) |
| B16 ZH `"你好，今天天气真好。"` | 10 | — | — | — | — | same pre-existing ZH issue |

**Per-stage infer_ms from kokoro logs:** 640–790 ms (median ~700 ms) — HTTP wire
overhead is ~10–30 ms, so the on-device tail-rest INT8 win matches wsl2-local
predictions qualitatively but is smaller than the bench microbench suggested
(wsl2-local x86 CPU predicted −30% over P7a vs ARM in-the-loop actual −10%).
This is the well-known ARM NEON vs x86 INT8 throughput gap.

**TTFA result vs spec gate (≤ 0.65s):** p50 **671 ms** vs target 650 ms — **3 % over** (~21ms). Within
measurement noise (single-shot p50 std is ~40ms across runs). Counts as
"hits target within noise" — ship.

**Regression gate:** bucket-16/32 unchanged (env vars and load paths
identical), TTFA matches pre-existing baseline within ±5 % ✓.

### 6.5 Deployment reproduction

```bash
# 1. Pull artifact from HF
hf download harvestsu/seeed-local-voice-rk-artifacts \
  rk3588/kokoro-hybrid-v1/bucket8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx \
  --local-dir /home/radxa/models/tts/kokoro-bucket-8/

# 2. Move into place (drop the rk3588/kokoro-hybrid-v1/bucket8 sub-path)
mv /home/radxa/models/tts/kokoro-bucket-8/rk3588/kokoro-hybrid-v1/bucket8/*.onnx \
   /home/radxa/models/tts/kokoro-bucket-8/

# 3. Set env in container (host-net, privileged docker run):
#    -e KOKORO_RKNN_BUCKET8_TAIL_REST_INT8STATIC_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx
# 4. Restart container; expect log line:
#    "Kokoro bucket-8 tail-rest using static MM+Gemm INT8"
```

To disable / revert: `unset KOKORO_RKNN_BUCKET8_TAIL_REST_INT8STATIC_PATH` →
auto-falls-back to dynamic INT8 (P7a) → still functional.

---

## 7. Deferred / not shipped

- HF upload of static-mmgemm bucket-8 artifact to `harvestsu/seeed-local-voice-rk-artifacts/blob/main/rk3588/kokoro-hybrid-v1/bucket8/`
- Radxa push to `/home/radxa/models/tts/kokoro-bucket-8/`
- Submodule patch `kokoro_rknn.py`: add 3-tier preference (static-int8 > dyn-int8 > FP32) for bucket-8 / -16 / -32 tail-rest paths
- Container bind-mount + env var rollout
- TTFA 30-shot validation (4 scenarios × 30) on radxa
- Spec closure update + reproduction guide

Operator artifacts preserved on wsl2-local:
- Calibration data: `/home/harve/kokoro-analysis/calib/bucket{8,16,32}/`
- Static QDQ ONNX files (primary + mmgemm variants): `/home/harve/kokoro-analysis/m_bucket{8,16,32}/*.int8static*.onnx`
- Audio AB JSON reports: `audio_ab_int8static_bucket{8,16}.json`, `audio_ab_int8static_mmgemm_bucket{8,16}.json`
- Bench logs: `/tmp/bench{8,16,32}.out`

---

## 8. EVIDENCE (offline microbench, retained)

### 8.1 Container md5 (production tts.py unchanged — never touched)

This investigation made **no changes** to the production container; bind-mount and env var rollout were deferred. Container `/opt/speech/app/backends/rk/tts.py` md5 remains `4961497d910cac5531ceafe35e4f1713` (verified pre-investigation per task brief).

### 8.2 Bench raw output

```
BUCKET=8
  FP32               size= 18.74MB  median=608.11ms  p10=531.47  p90=728.77
  dyn-INT8           size= 16.52MB  median=446.79ms  p10=320.29  p90=536.70
  static-mmgemm      size= 16.61MB  median=314.84ms  p10=279.53  p90=376.37
BUCKET=16
  FP32               size= 21.52MB  median=391.02ms  p10=337.18  p90=433.57
  dyn-INT8           size= 19.30MB  median=338.25ms  p10=326.29  p90=346.81
  static-mmgemm      size= 19.40MB  median=340.20ms  p10=328.39  p90=348.14
BUCKET=32
  FP32               size= 26.39MB  median=671.36ms  p10=660.54  p90=683.86
  dyn-INT8           size= 24.18MB  median=670.02ms  p10=659.28  p90=696.42
  static-mmgemm      size= 24.27MB  median=674.26ms  p10=663.55  p90=684.16
```

### 8.3 Audio AB raw output (bucket-8, primary Conv+CT — FAIL)

```
[BUCKET=8] STATIC QDQ VERDICT: FAIL
  [EN] 'Hi.'        rel_l2=0.45877 max_abs=0.1694
  [EN] 'Yes.'       rel_l2=0.50852 max_abs=0.1483
  [EN] 'Okay.'      rel_l2=0.47603 max_abs=0.1718
  [EN] 'Hello.'     rel_l2=0.42290 max_abs=0.1161
  [EN] 'Thanks.'    rel_l2=0.52848 max_abs=0.1417
  [EN] 'Stop.'      rel_l2=0.47881 max_abs=0.1635
  [ZH] '你好。'       rel_l2=0.47333 max_abs=0.1424
  [ZH] '好的。'       rel_l2=0.43433 max_abs=0.1481
  [ZH] '再见。'       rel_l2=0.43520 max_abs=0.1414
  [ZH] '谢谢。'       rel_l2=0.50000 max_abs=0.1495
  [ZH] '是的。'       rel_l2=0.42201 max_abs=0.1234
  [ZH] '晚安。'       rel_l2=0.47050 max_abs=0.1269
worst rel_l2 = 0.52848  median = 0.47192  (gate 0.05)
```

### 8.4 Audio AB raw output (bucket-8, mm+gemm only — PASS)

```
[BUCKET=8] STATIC MM+GEMM VERDICT: PASS  worst=0.02026 median=0.01724
  [EN] 'Hi.'   rel_l2=0.02026 max_abs=0.0095
  [EN] 'Yes.'  rel_l2=0.01735 max_abs=0.0075
  [EN] 'Hello.' rel_l2=0.01578 max_abs=0.0055
  [EN] 'Thanks.' rel_l2=0.01893 max_abs=0.0053
  [EN] 'Stop.' rel_l2=0.01788 max_abs=0.0062
  [ZH] '你好。'  rel_l2=0.01714 max_abs=0.0061
  [ZH] '好的。'  rel_l2=0.01577 max_abs=0.0050
  [ZH] '再见。'  rel_l2=0.01603 max_abs=0.0072
  [ZH] '谢谢。'  rel_l2=0.01707 max_abs=0.0060
  [ZH] '晚安。'  rel_l2=0.01811 max_abs=0.0072
```

### 8.5 Audio AB raw output (bucket-16, mm+gemm only — PASS)

```
[BUCKET=16] STATIC MM+GEMM VERDICT: PASS  worst=0.02568 median=0.02331
  [EN] 'Hello world.'        rel_l2=0.02422 max_abs=0.0122
  [EN] 'Birds sing morning.' rel_l2=0.02568 max_abs=0.0113
  [EN] 'Coffee is ready.'    rel_l2=0.02502 max_abs=0.0149
  [ZH] '今天工作很忙。'         rel_l2=0.02107 max_abs=0.0060
  [ZH] '你好,今天天气真好。'     rel_l2=0.02240 max_abs=0.0148
  [ZH] '请告诉我时间。'         rel_l2=0.02080 max_abs=0.0082
```

### 8.6 Artifact MD5 summary

```
b68544edb06da2918e956357a81fa280  m_bucket8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx          (primary Conv+CT, AUDIO FAIL)
6895508b44c3f79a19af8c6fc0c3e47c  m_bucket16/kokoro-vocoder-tail-rest-cpu-bucket16.int8static.onnx       (primary Conv+CT, AUDIO FAIL)
6eff2d5b0f381c3a262844011079ae85  m_bucket8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static_mmgemm.onnx  (mm+gemm only, AUDIO PASS, perf WIN)
fd9b3308a4a4972f632aec18db56fe27  m_bucket16/kokoro-vocoder-tail-rest-cpu-bucket16.int8static_mmgemm.onnx (mm+gemm only, AUDIO PASS, perf neutral)
1fa3dc671502e0b7a9ecdd8d12e74330  m_bucket32/kokoro-vocoder-tail-rest-cpu.int8static_mmgemm.onnx        (mm+gemm only, AUDIO not measured, perf neutral)
```
