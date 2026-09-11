# Kokoro BERT acoustic A/B audio report

**Date:** 2026-05-23
**Author:** main thread investigation, run on `wsl2-local`
**Status:** CONCLUSIVE — BERT branch in shipped Kokoro v1.0 ONNX is dead code on the
audio path. The 17%-acceleration shipped 3-stage pipeline produces **byte-identical**
audio to the full graph that contains BERT.

## TL;DR

Across 10 prosodically diverse English utterances (declaratives, questions,
exclamations, imperatives, polite, narrative) fed through both pipelines with
identical `(tokens, style, speed)` inputs:

- **All 10 audio outputs are bit-exact equal in FP32**: `max_abs_diff = 0.0`,
  `rel_l2 = 0.0`, `mel_L1 = 0.00 dB`, `pitch_RMSE = 0.0 Hz`.
- **All 10 int16 WAV files have identical MD5** between the FULL and SHIPPED
  pipelines.
- Conclusion: the BERT subgraph in `kokoro.with-decoder-front-cut.onnx` is
  topologically present but functionally disconnected from the audio output.
  The audio path goes exclusively through `text_encoder_1` (the BERT-independent
  branch).
- **Recommendation: B** — accept the current 17% acceleration as the upper
  bound for the shipped pipeline. **There is no acoustic motivation to invest
  in re-exporting a prefix that includes BERT.** Redirect effort toward the
  Phase-2 34% target (decoder-front + tail INT8 or extended Mat-fusion).

## Test design

### Pipelines under test

| Pipeline | Graph | BERT? | Inputs | Output |
|---|---|---|---|---|
| **A — FULL** | `kokoro-bucket-32/kokoro.with-decoder-front-cut.onnx` (md5 `8f1196b054633b053c2363b86860e16b`, 1712 nodes, 652 BERT-named) | YES (in graph) | `tokens (1,32) int64`, `style (1,256) f32`, `speed (1,) f32` | `audio (126000,) f32` |
| **B — SHIPPED** | `kokoro-prefix-cpu.onnx` (md5 `d5b5d06828475b61bc88fdb77c4b8f35`, 58 nodes, 0 BERT) → `kokoro-decoder-front.onnx` (FP32 ORT) → `kokoro-generator-tail-cpu.onnx` | NO | same | same |

Both pipelines run on CPU via ONNX Runtime 1.x (CPUExecutionProvider) on
`wsl2-local`, no INT8 or RKNN involved — eliminates quantization noise as a
confounding factor.

### Prompts (10, EN)

Generated via misaki G2P (`en.G2P(trf=False, british=False)`), padded to
seq_length=32 with token 0 framing. Style vector loaded from
`kokoro-multi-lang-v1_1/voices.bin`, speaker_id=0, style row indexed by actual
token count (Kokoro convention: `offset = sid*510*256*4 + tokcount*256*4`).

| idx | text | tokens used |
|---|---|---|
| 00 | Hello, how are you today? | 25 |
| 01 | The weather is wonderful this morning. | 32 |
| 02 | Are you coming to the party tonight? | 32 |
| 03 | Please be very careful with that fragile glass! | 32 |
| 04 | I cannot believe what just happened. | 32 |
| 05 | She said the meeting is at three o'clock. | 32 |
| 06 | Why did the chicken cross the road? | 32 |
| 07 | It is a long and winding road. | 32 |
| 08 | Stop right there immediately! | 28 |
| 09 | Thank you so much for your kind help. | 32 |

Mix of declarative, interrogative (yes/no + wh-), exclamation, imperative, and
polite — covers prosodic regimes where BERT (per StyleTTS2 paper) would be
expected to contribute most.

## Results

### Per-utterance metrics (FULL vs SHIPPED, FP32 raw audio)

| idx | text | RMS_A | RMS_B | max_abs_diff | rel_l2 | mel L1 (dB) | pitch RMSE (Hz) | int16 wav MD5 (both) |
|---|---|---|---|---|---|---|---|---|
| 00 | Hello, how are you today? | 0.03519 | 0.03519 | 0.0 | 0.0 | 0.00 | 0.0 | `1c391b6209cc4afbd0048302d09866db` |
| 01 | The weather is wonderful this morning. | 0.03719 | 0.03719 | 0.0 | 0.0 | 0.00 | 0.0 | `f16c13285c6f48e357204cff142f1b0d` |
| 02 | Are you coming to the party tonight? | 0.03552 | 0.03552 | 0.0 | 0.0 | 0.00 | 0.0 | `722ed6d14e2e45b200edb7600c794ebb` |
| 03 | Please be very careful with that fragile glass! | 0.03800 | 0.03800 | 0.0 | 0.0 | 0.00 | 0.0 | `1d60a7f349e120f752c60fc26d3a6eb8` |
| 04 | I cannot believe what just happened. | 0.03502 | 0.03502 | 0.0 | 0.0 | 0.00 | 0.0 | `6d00cf2783a71d0c002e3e4da820a232` |
| 05 | She said the meeting is at three o'clock. | 0.03267 | 0.03267 | 0.0 | 0.0 | 0.00 | 0.0 | `3b2747f3336d25b95b82c9a96f8fa3ba` |
| 06 | Why did the chicken cross the road? | 0.03554 | 0.03554 | 0.0 | 0.0 | 0.00 | 0.0 | `86157b64e37bbb9aa7b1ece28538e2fd` |
| 07 | It is a long and winding road. | 0.03770 | 0.03770 | 0.0 | 0.0 | 0.00 | 0.0 | `866ababdcb1d021b16dd75fc809f1876` |
| 08 | Stop right there immediately! | 0.03525 | 0.03525 | 0.0 | 0.0 | 0.00 | 0.0 | `a2177d040346750d91d181f3608dcd37` |
| 09 | Thank you so much for your kind help. | 0.03587 | 0.03587 | 0.0 | 0.0 | 0.00 | 0.0 | `34199d6616a4eec034cd81167de5176f` |

All 126000-sample FP32 audio arrays compared bit-for-bit:
`np.array_equal(audio_full, audio_shipped) = True` for every utterance.

### Runtime (FYI, CPU ORT on wsl2-local, x86_64)

| idx | FULL (s) | SHIPPED (s) |
|---|---|---|
| 00 | 1.41 | 1.62 |
| 01 | 1.15 | 1.62 |
| 02 | 1.11 | 1.58 |
| 03 | 1.11 | 1.55 |
| 04 | 1.11 | 4.43 (outlier; likely OS noise) |
| 05 | 1.14 | 1.65 |
| 06 | 1.09 | 1.54 |
| 07 | 1.11 | 1.51 |
| 08 | 1.11 | 1.56 |
| 09 | 1.12 | 1.61 |

Note: shipped pipeline runs slightly slower on CPU because of 3 separate ORT
session crossings vs single-session; this disappears on RKNN where the
decoder-front gets accelerated. Not a relevant signal for the audio-equivalence
finding.

## Decision matrix (per spec gate)

| signal | observed | gate |
|---|---|---|
| FP32 rel_l2 | **0.0** | < 0.005 → dead code |
| pitch RMSE | **0.0 Hz** | < 5 Hz → dead code |
| mel L1 | **0.0 dB** | < 1 dB → dead code |
| int16 wav md5 | **identical** for all 10 pairs | strongest possible signal |

**Verdict: BERT is dead code in the v1.0 ONNX export of Kokoro.** All four
metrics are not just below threshold — they are exactly zero, which is the
strongest possible empirical evidence that BERT contributes nothing to the
audio output in this graph.

## Why this is the case (hypothesis)

Kokoro v1.0 is published as StyleTTS2-derived but the exported v1.0 ONNX graph
appears to have wired the audio path to the `text_encoder_1` (phoneme-only)
branch, not the `text_encoder` branch that consumes
`/bert_encoder/Add_output_0`. The 652 BERT-named nodes are computed by ORT but
their outputs are not consumed by any node on the audio dataflow path. This is
common when an upstream model has multiple text encoders and the export script
captured both branches but the final routing only used one.

Confirmation that no other input is hidden:

- Both pipelines accept the **same exact 3 inputs**: `tokens`, `style`,
  `speed`. There is no `bert_tokens` or similar extra input that would
  separately exercise BERT.
- The shipped prefix already does not have BERT, and it still drives the same
  decoder-front output that matches FULL bit-exactly.

## Implications for RK port

- **Phase 1 (shipped 17% accel) is correct as-is.** Stripping BERT from the
  prefix did not lose quality because there was no quality contribution to lose.
- **Do not invest in re-exporting prefix with BERT.** Spec'd "BERT prosody
  recovery" path has zero acoustic payoff.
- **Phase 2 (34% target) should pursue** decoder-front INT8 quantization
  improvements, tail island Op acceleration, or pure RKNN-fusion gains — these
  are independent of BERT.
- **If you want real BERT-derived prosody**, you would need a different model
  export (Kokoro v1.1 multi-lang `model.onnx` — verified separately to contain
  BERT, but that is a different model with different audio output, not a
  drop-in replacement for shipped v1.0).

## Recommendation

**Option B — accept 17% as the v1.0 ceiling and pursue Phase-2 acceleration
through non-BERT paths.** Close out the "BERT restoration" investigation as
NO-OP.

## Artifacts

### On wsl2-local

- Working directory: `/home/harve/kokoro-analysis/m3-bert-ab/`
- Inference + metrics script: `kokoro_ab_run.py` (md5 `9edfa657832a6971f2e03f45f7fa358e`)
- FP32 bit-equivalence verifier: `fp32check.py` (md5 `ecb1acedc8f7e5f2d0fc58c5c542598c`)
- Per-utterance WAVs (24 kHz mono int16): `full_NN.wav`, `shipped_NN.wav` for N=00..09
- Per-utterance comparison PNGs: `compare_NN.png` (waveform overlay + mel diff + pitch contour)
- Raw metrics: `results.json`

### On Mac (synced via fleet pull)

- `/tmp/kokoro_ab/` — full mirror including all WAVs, PNGs, JSON, scripts

### Source ONNX (read-only)

- FULL: `/home/harve/kokoro-analysis/kokoro-bucket-32/kokoro.with-decoder-front-cut.onnx` md5 `8f1196b054633b053c2363b86860e16b`
- SHIPPED prefix: `/home/harve/kokoro-analysis/kokoro-bucket-32/kokoro-prefix-cpu.onnx` md5 `d5b5d06828475b61bc88fdb77c4b8f35`
- SHIPPED decoder-front: `/home/harve/kokoro-analysis/kokoro-bucket-32/kokoro-decoder-front.onnx` (FP32 ORT, not INT8 RKNN — deliberately to avoid confound)
- SHIPPED tail: `/home/harve/kokoro-analysis/kokoro-bucket-32/kokoro-generator-tail-cpu.onnx`
- Vocab/voices: `/home/harve/kokoro-analysis/kokoro-multi-lang-v1_1/tokens.txt`, `voices.bin`

### Reproduction (5 minutes on wsl2-local CPU)

```bash
cd /home/harve/kokoro-analysis/m3-bert-ab
python3 kokoro_ab_run.py    # ~13 s for 10 utterances, writes results.json + WAVs + PNGs
python3 fp32check.py        # bit-equality check, prints bit_identical=True for all 10
```
