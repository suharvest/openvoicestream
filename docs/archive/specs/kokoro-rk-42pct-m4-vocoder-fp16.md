# Kokoro RK NPU 42% — M4 Vocoder Front-Half FP16 RKNN Report

Status: **M4 BLOCKED — parity gate FAIL (2026-05-23)**. All three rewriter paths and the native (no-rewrite) path were built and parity-tested on wsl2-local against the bucket-32 M1 boundary. No build crashed; the front-half RKNN engine compiles cleanly. The **max-abs-diff ≤0.03** gate cannot be met without either (a) a higher-precision Sin replacement than the existing polynomial rewriters, (b) a boundary regression to spec §5 fallback (`resblocks.0-1`, user decision required), or (c) a relaxed gate justified by end-to-end audio MOS rather than per-tensor max-abs.

Per the M4 spec instruction "failure → STOP and report; do not auto-fallback", this report stops at the parity-FAIL boundary. Steps 6–8 of the M4 work order (radxa 50-shot stability, HF upload, manifest integration) were **not executed**.

## 1. Artifacts produced (on wsl2-local)

Work directory: `/home/harve/kokoro-analysis/m4-vocoder/`

| Artifact | Size | md5 | Purpose |
|---|---:|---|---|
| `kokoro-vocoder-front-half.onnx` | 68 MiB | `0d2ec599ee2cf4e9013ed7fcfd481327` | Step 1: front-half subgraph extracted from CPU tail ONNX, output `/decoder/generator/Add_5_output_0 [1,256,4200]` |
| `kokoro-vocoder-tail-rest-cpu.onnx` | 26 MiB | `68c62e751d7c21bc1763950be04c1e90` | Step 3: CPU tail-rest (input = Add_5, output = audio) |
| `kokoro-vocoder-front-half.rewritten.onnx` | 68 MiB | `f016bcf3d5dcbf7423ae916eef43820c` | Step 2 v1: clipped-Sin polynomial + Pow2→Mul + IN→primitive (via `fix_kokoro_tail_rknn_ops.py`) |
| `kokoro-vocoder-front-half.sinrr.onnx` | 68 MiB | `e5be8fe6ffc4364486c53d2b05d35db4` | Range-reduced Sin polynomial only |
| `kokoro-vocoder-front-half.sinrr-pow2.onnx` | 68 MiB | `afd0e6f721d36a9938f2d74a6a3887ce` | + Pow2→Mul |
| `kokoro-vocoder-front-half.rewritten-v2.onnx` | 68 MiB | `53b50cf3595538da47461b3fc9240403` | Step 2 v2: range-reduced Sin + Pow2 + IN (full v2 rewrite) |
| `rk3588/kokoro-vocoder-front-half.fp16.rknn` | 44 MiB | `3d7afd431977a736b983ddd907438e1f` | Step 4 v1: RKNN FP16 from rewritten v1 (clipped-Sin) |
| `rk3588/kokoro-vocoder-front-half.native.fp16.rknn` | 41 MiB | `cfa959fdcd9c7c69eed1cd404366847e` | Native FP16 RKNN from the **original** ONNX (no rewriter) — RKNN-toolkit2 2.3.0 + rk3588 + opt level 0 |

The RKNN engine built cleanly in all variants — no `Sin`/`Pow`/`InstanceNorm` fallback warnings, no register-overflow warnings, no `E` (error) lines. Build log filtered with `grep -iE '^E |^W |fallback|register|error|fail' | grep -v config` returns empty for the native build.

## 2. Parity gate

Eight real samples were reused from `kokoro-bucket-32-int8-front/quant_dataset/` (bucket-32 prefix outputs `/MatMul_1_output_0 [1,512,210]` + `/Slice_2_output_0 [1,128]`). For each sample:

1. Run **CPU decoder-front ONNX** to obtain `/decoder/decode.3/Mul_output_0 [1,512,420]` (the front-half RKNN input)
2. Run **CPU original front-half ONNX** to obtain the FP32 reference
3. Run **RKNN simulator** (toolkit `init_runtime(target=None)`) on each candidate engine
4. Compare against the FP32 reference: `max-abs-diff`, `mae`, `rel_l2`

### 2a. v1 (clipped-Sin polynomial) — `kokoro-vocoder-front-half.fp16.rknn`

```
sample        rewriter (rew vs orig)               RKNN total (rknn vs orig)         RKNN only (rknn vs rew)
sample000     mae=4.49  max=52.63  rel_l2=0.4546   mae=4.49  max=52.62  rel_l2=0.4546   mae=0.010 max=0.120 rel_l2=0.0010
sample001     mae=5.57  max=93.79  rel_l2=0.5885   mae=5.57  max=93.77  rel_l2=0.5885   mae=0.009 max=0.134 rel_l2=0.0009
sample002     mae=5.61  max=89.60  rel_l2=0.6671   mae=5.61  max=89.70  rel_l2=0.6672   mae=0.015 max=0.241 rel_l2=0.0017
sample003     mae=4.99  max=70.19  rel_l2=0.5716   mae=4.99  max=70.21  rel_l2=0.5716   mae=0.010 max=0.120 rel_l2=0.0011
sample004     mae=5.72  max=69.14  rel_l2=0.5812   mae=5.72  max=69.18  rel_l2=0.5811   mae=0.011 max=0.137 rel_l2=0.0010
sample005     mae=6.26  max=83.05  rel_l2=0.5863   mae=6.26  max=83.08  rel_l2=0.5863   mae=0.015 max=0.194 rel_l2=0.0013
sample006     mae=4.35  max=83.00  rel_l2=0.5733   mae=4.35  max=83.00  rel_l2=0.5732   mae=0.012 max=0.138 rel_l2=0.0015
sample007     mae=7.07  max=93.91  rel_l2=0.4917   mae=7.07  max=93.88  rel_l2=0.4917   mae=0.012 max=0.182 rel_l2=0.0008

AGGREGATE
  rewriter (rew vs orig):       max-abs worst = 93.91  rel_l2 worst = 0.6671
  RKNN total (rknn vs orig):    max-abs worst = 93.88  rel_l2 worst = 0.6672   ← GATE FAIL (>>0.03)
  RKNN only (rknn vs rew):      max-abs worst =  0.24  rel_l2 worst = 0.0017   ← RKNN itself is fine
```

The entire error budget is consumed by the clipped-Sin polynomial — RKNN reproduces the rewritten model to rel_l2 ≈ 0.001 (well within RKNN-only budget). Saved at `parity_report.json`.

### 2b. v2 (range-reduced Sin) — `kokoro-vocoder-front-half.rewritten-v2.onnx` (ORT vs original ORT)

```
sample        rew-v2 vs orig
sample000     mae=1.179 max=13.030 rel_l2=0.1148
sample001     mae=1.222 max=15.126 rel_l2=0.1217
sample002     mae=1.240 max=14.332 rel_l2=0.1414
sample003     mae=1.334 max=13.121 rel_l2=0.1456
sample004     mae=1.453 max=14.890 rel_l2=0.1397
sample005     mae=1.474 max=14.165 rel_l2=0.1282
sample006     mae=1.189 max=15.877 rel_l2=0.1538
sample007     mae=2.175 max=23.009 rel_l2=0.1461

WORST  max=23.01  rel_l2=0.1538   ← Still well above 0.03 gate.
```

Range-reduce Sin alone cuts error 4x vs clipped-Sin but is still ~500× above the gate. RKNN build for v2 was not executed because the rewriter-only error already exceeds the gate.

### 2c. Rewriter isolation: which step is the source?

Comparing intermediate rewrite stages against the original ORT output (`isolate_rewriter.py`):

| Stage | Worst max-abs-diff | Worst rel_l2 |
|---|---:|---:|
| A. range-reduced Sin only | 23.01 | 0.1538 |
| B. + Pow2→Mul | 23.01 | 0.1538 |
| C. + InstanceNorm→primitive (= v2) | 23.01 | 0.1538 |

**A == B == C exactly.** Pow2 and IN rewrites are bit-identical to original ORT. **100% of the rewriter error originates from the Sin polynomial replacement.**

### 2d. Native FP16 (no rewriter) — `kokoro-vocoder-front-half.native.fp16.rknn`

```
sample        native FP16 vs ORT FP32
sample000     mae=0.0101 max=0.0831 rel_l2=0.000962
sample001     mae=0.0095 max=0.1028 rel_l2=0.000922
sample002     mae=0.0142 max=0.1921 rel_l2=0.001587
sample003     mae=0.0099 max=0.0877 rel_l2=0.001055
sample004     mae=0.0111 max=0.0968 rel_l2=0.001049
sample005     mae=0.0145 max=0.1287 rel_l2=0.001241
sample006     mae=0.0114 max=0.1585 rel_l2=0.001410
sample007     mae=0.0137 max=0.1336 rel_l2=0.000898

WORST  max=0.1921  rel_l2=0.001587   ← GATE FAIL on max-abs (0.19 > 0.03)
                                       BUT rel_l2 0.0016 << spec §6 ideal 0.01.
```

Native FP16 RKNN engine cleared the rel_l2 ≤ 0.01 spec §6 criterion (1.6e-3 vs 1e-2 budget) but **violates the M4 sub-gate of max-abs-diff ≤ 0.03** by a factor of ~6×. Mean-abs-diff is ~0.01, so the 0.19 max is a small set of peak-bin outliers — likely at the steepest slopes of `sin(α·x)` in the Snake activation, which is intrinsically phase-sensitive (spec §6 notes "单点 max 因相位敏感可适度放宽" but that language is for the **whole-pipeline** acceptance, not the M4 sub-gate).

Saved at `parity_report_native.json`.

## 3. Why the polynomial Sin rewriters fail at the front-half boundary

Probe of Sin-op input ranges over a real sample (`probe_sin_range.py`):

```
/decoder/generator/m_source/l_sin_gen/Transpose_3_output_0   min=  16.31  max=49748.95   |x|/π = 15835.58
/decoder/generator/noise_res.0/Mul_output_0                  min= -14.96  max=   15.07   |x|/π =     4.80
/decoder/generator/noise_res.0/Mul_2_output_0                min=  -9.58  max=   13.85   |x|/π =     4.41
... (22 more Snake-activation Sin sites, all |x|/π ∈ [4.0, 9.1])
/decoder/generator/resblocks.1/Mul_10_output_0               min= -28.49  max=   28.65   |x|/π =     9.12
```

The clipped-Sin polynomial in `fix_kokoro_tail_rknn_ops.py` is valid on `[-π, π]`. The `m_source` sinusoidal-source generator drives Sin inputs ≈ 50000 (≈15800·π) — totally outside any 7th-order polynomial's valid range — and the 25 Snake-activation Sin sites in `noise_res.0 + resblocks.0-2` routinely hit ±9π. Clipping at ±π zeros out almost the entire dynamic range and produces the observed max-abs ≈ 93 (≈ peak of `sin²(αx)` term scaled by the resblock chain).

`rewrite_onnx_sin_poly_range_reduce.py` does floor-based period reduction (`x' = x - 2π·⌊(x+π)/(2π)⌋`), which yields a true-period |x'| ≤ π — but the 7th-order Taylor series still has ≈5% rel error at |x'| approaching π. With 25 Sin sites, an `x² · sin(αx)` Snake term, and downstream amplification through three resblock kernels, the per-Sin 5% explodes to per-tensor max ≈ 23.

The native RKNN path eliminates the Sin approximation entirely (RKNN 2.3.0 + rk3588 supports `Sin`/`Pow`/`InstanceNormalization` natively at FP16 in this dim range) and reaches rel_l2 1.6e-3 — the residual is the pure FP16-vs-FP32 quantization error, intrinsic to the FP16 datatype.

## 4. Path forward (decisions required from the operator)

This is the decision boundary the M4 work-order's "STOP on FAIL" rule pushes back to the user:

### Option A — accept native FP16, relax the M4 sub-gate

The native engine passes spec §6 acceptance criteria (`rel_l2 ≤ 0.01`, with 1.6e-3 actual). The 0.03 max-abs-diff was a **sub-gate** in spec §4 M4 row. If end-to-end audio MOS (the spec §6 "音质 parity" path) is the operative quality bar, the 0.19 peak-bin difference is likely below audibility — FP16 vocoders are routinely used in production. **Recommended next test**: build end-to-end `prefix → decoder-front int8 → native vocoder-front FP16 RKNN → CPU tail-rest` on radxa, run the 50-prompt smoke set, compare audio waveform / MOS / spectral distortion vs the current 17%-NPU baseline. If audio parity holds, ship the native build and rewrite the M4 gate language to bound max-abs by a phase-sensitive percentile (e.g., 99th-percentile abs-diff ≤ 0.03) rather than max.

### Option B — invest in a better Sin approximation

A higher-order Chebyshev-minimax polynomial over `[-π, π]` (10th/12th order) or a LUT-based Sin (256-entry table + linear interpolation) would push range-reduce Sin parity well under 1% rel, which after compounding through 25 sites should land around max-abs 0.05–0.1 — still above 0.03 but in the ballpark where FP16 quant noise dominates. This requires writing a new rewriter (`rewrite_onnx_sin_chebyshev.py` or `rewrite_onnx_sin_lut.py`), or if RKNN supports `Erf` natively, using an `erf`-based identity. Cost: 1–2 engineer-days.

### Option C — spec §5 fallback: tighter boundary

Cut at `ups.0 + resblocks.0-1` (omit `resblocks.2`). Sin sites drop from 25 → 17, may reduce compounded error proportionally. NPU compute share drops from ~16.8% to ~12% (target ~37% total). Requires re-doing M1 boundary discovery (new clean cut tensor before resblocks.2 isn't directly available — would need to manually inject an Add to merge the partial path on CPU). Cost: 1 engineer-day.

### Option D — keep status quo (17% NPU)

The current shipped `rk_manifest 2026-05-23` already passes RTF 0.673 and 50-shot smoke. Defer M4 indefinitely; revisit if/when a better Sin replacement lands in RKNN-toolkit upstream (>2.3.0).

## 5. EVIDENCE

### 5.1 Artifact md5

```text
$ md5sum /home/harve/kokoro-analysis/m4-vocoder/*.onnx /home/harve/kokoro-analysis/m4-vocoder/rk3588/*.rknn
0d2ec599ee2cf4e9013ed7fcfd481327  kokoro-vocoder-front-half.onnx
53b50cf3595538da47461b3fc9240403  kokoro-vocoder-front-half.rewritten-v2.onnx
f016bcf3d5dcbf7423ae916eef43820c  kokoro-vocoder-front-half.rewritten.onnx
afd0e6f721d36a9938f2d74a6a3887ce  kokoro-vocoder-front-half.sinrr-pow2.onnx
e5be8fe6ffc4364486c53d2b05d35db4  kokoro-vocoder-front-half.sinrr.onnx
68c62e751d7c21bc1763950be04c1e90  kokoro-vocoder-tail-rest-cpu.onnx
3d7afd431977a736b983ddd907438e1f  rk3588/kokoro-vocoder-front-half.fp16.rknn
cfa959fdcd9c7c69eed1cd404366847e  rk3588/kokoro-vocoder-front-half.native.fp16.rknn
```

### 5.2 v1 (clipped-Sin) RKNN build log (key section, ANSI-stripped)

```text
I rknn-toolkit2 version: 2.3.0
I Loading 100%|███████████████████| 178/178 [00:00<00:00, 12316.45it/s]
W load_onnx: The config.mean_values is None, zeros will be set for input 0!
W load_onnx: The config.std_values is None, ones will be set for input 0!
W load_onnx: The config.mean_values is None, zeros will be set for input 1!
W load_onnx: The config.std_values is None, ones will be set for input 1!
I rknn building ...
I rknn building done.
output=/home/harve/kokoro-analysis/m4-vocoder/rk3588/kokoro-vocoder-front-half.fp16.rknn
```

No fallback / register-overflow / Sin-unsupported warnings. The mean/std warnings are benign (we feed already-normalised tensors).

### 5.3 Native RKNN build log (key section)

```text
$ sed 's/\x1b\[[0-9;]*m//g' /tmp/build_native.log | grep -iE '^E |^W |fallback|register|error|fail' | grep -v config | head -30
(empty)
```

```text
I rknn building ...
I rknn building done.
output=/home/harve/kokoro-analysis/m4-vocoder/rk3588/kokoro-vocoder-front-half.native.fp16.rknn
```

### 5.4 v1 parity raw print

(See section 2a above; recorded verbatim in `parity_report.json`.)

### 5.5 Native parity raw print

```text
I Target is None, use simulator!
Found 8 samples
[sample000] mae=0.010087 max=0.083054 rel_l2=0.000962
[sample001] mae=0.009497 max=0.102757 rel_l2=0.000922
[sample002] mae=0.014204 max=0.192127 rel_l2=0.001587
[sample003] mae=0.009879 max=0.087711 rel_l2=0.001055
[sample004] mae=0.011105 max=0.096794 rel_l2=0.001049
[sample005] mae=0.014517 max=0.128716 rel_l2=0.001241
[sample006] mae=0.011410 max=0.158520 rel_l2=0.001410
[sample007] mae=0.013712 max=0.133648 rel_l2=0.000898

=== NATIVE FP16 RKNN (no rewrite) ===
max-abs-diff worst: 0.192127  (gate ≤0.03)
rel_l2 worst      : 0.001587  (informational ≤0.01 ideal)
FAIL
```

### 5.6 Rewriter isolation

```text
   sample |       A: sinrr only                |       B: +pow2                     |       C: +IN (=v2)
sample000 | mae=1.1793 max=13.0298 rl=0.1148   | mae=1.1793 max=13.0298 rl=0.1148   | mae=1.1793 max=13.0298 rl=0.1148
sample007 | mae=2.1750 max=23.0094 rl=0.1461   | mae=2.1750 max=23.0094 rl=0.1461   | mae=2.1750 max=23.0094 rl=0.1461

Worst (max-abs): A=23.0094  B=23.0094  C=23.0094   →  Pow2 + IN rewrites contribute zero error.
```

### 5.7 Sin-op input range probe (first sample, all 25 Sin sites)

```text
Found 25 Sin ops
  /decoder/generator/m_source/l_sin_gen/Transpose_3_output_0   min=   16.31 max=49748.95  OUT(x15835.58π)
  /decoder/generator/noise_res.0/Mul_output_0                  min=  -14.96 max=   15.07  OUT(x4.80π)
  /decoder/generator/noise_res.0/Mul_2_output_0                min=   -9.58 max=   13.85  OUT(x4.41π)
  /decoder/generator/noise_res.0/Mul_4_output_0                min=  -13.32 max=   12.74  OUT(x4.24π)
  /decoder/generator/noise_res.0/Mul_6_output_0                min=  -20.87 max=   17.87  OUT(x6.64π)
  /decoder/generator/noise_res.0/Mul_8_output_0                min=  -17.00 max=   13.47  OUT(x5.41π)
  /decoder/generator/noise_res.0/Mul_10_output_0               min=  -18.29 max=   18.54  OUT(x5.90π)
  /decoder/generator/resblocks.0/Mul_output_0                  min=  -18.68 max=   18.61  OUT(x5.95π)
  /decoder/generator/resblocks.0/Mul_2_output_0                min=  -10.25 max=   14.83  OUT(x4.72π)
  /decoder/generator/resblocks.0/Mul_4_output_0                min=  -19.03 max=   15.97  OUT(x6.06π)
  /decoder/generator/resblocks.0/Mul_6_output_0                min=  -14.13 max=   12.80  OUT(x4.50π)
  /decoder/generator/resblocks.0/Mul_8_output_0                min=  -16.15 max=   19.07  OUT(x6.07π)
  /decoder/generator/resblocks.0/Mul_10_output_0               min=  -12.31 max=   12.56  OUT(x4.00π)
  /decoder/generator/resblocks.1/Mul_output_0                  min=  -13.75 max=   13.46  OUT(x4.38π)
  /decoder/generator/resblocks.1/Mul_2_output_0                min=  -12.48 max=   13.01  OUT(x4.14π)
  /decoder/generator/resblocks.1/Mul_4_output_0                min=  -25.56 max=   17.18  OUT(x8.14π)
  /decoder/generator/resblocks.1/Mul_6_output_0                min=  -18.12 max=   18.16  OUT(x5.78π)
  /decoder/generator/resblocks.1/Mul_8_output_0                min=  -16.22 max=   13.91  OUT(x5.16π)
  /decoder/generator/resblocks.1/Mul_10_output_0               min=  -28.49 max=   28.65  OUT(x9.12π)
  /decoder/generator/resblocks.2/Mul_output_0                  min=  -12.83 max=   15.07  OUT(x4.80π)
  /decoder/generator/resblocks.2/Mul_2_output_0                min=  -18.52 max=   14.60  OUT(x5.89π)
  /decoder/generator/resblocks.2/Mul_4_output_0                min=  -16.19 max=   16.52  OUT(x5.26π)
  /decoder/generator/resblocks.2/Mul_6_output_0                min=  -21.26 max=   20.71  OUT(x6.77π)
  /decoder/generator/resblocks.2/Mul_8_output_0                min=  -14.38 max=   13.79  OUT(x4.58π)
  /decoder/generator/resblocks.2/Mul_10_output_0               min=  -21.56 max=   20.94  OUT(x6.86π)

Max |x| across all Sin inputs: 49748.953  (clip range: π=3.142)
```

### 5.8 docker logs / device runtime checks

Not applicable — M4 stopped at parity gate before any radxa deployment. No container or device-side state was modified.

### 5.9 Before/after / production impact

Not applicable — no rk_manifest or profile changes were committed (those are M6 scope and explicitly out of M4). The current production manifest `rk3588-kokoro-hybrid-2026-05-23` (17%-NPU hybrid) is unchanged and continues to be the shipping path.

## 6. Deferred work-order items

The following M4 work-order steps were **not executed** because of the parity FAIL stop:

- Step 6: radxa 50-shot real-device sanity (no engine to validate)
- Step 7: HF model repo upload (no validated artifact)
- Step 8: this commit lands the report only — no manifest/profile changes

## 7. Risks carried forward

1. **The native FP16 engine is shippable as a quality experiment** if Option A is accepted, but the radxa-side runtime hasn't yet been smoke-tested against bucket-32 4200-dim Sin/Pow/InstanceNorm — there is residual risk that the simulator passes but on-device returns `None` (the historical risk noted in spec §2 and M1 §3). This is the next experimental step under Option A.
2. **The 50000-magnitude Sin input from the `m_source` generator** is a known Kokoro architectural artifact (sinusoidal source-noise generator). Even native RKNN must evaluate `sin(50000)` in FP16 (where 50000 is representable but the lost LSBs near π-period boundaries are ≈ 2π·2⁻⁸ ≈ 0.025 — close to the rel_l2 floor seen here).
3. **bucket-64 will exceed 8191 time-dim** (front-half time = 9560) — guard in M6 dispatch still required if buckets > 32 ship.

---

EVIDENCE summary status: all md5s verified, parity numbers raw-printed, build logs filtered for errors (none found in native build), rewriter isolation matrix included, Sin input-range diagnostic included. No production state changed.
