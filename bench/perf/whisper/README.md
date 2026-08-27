# Whisper cross-platform benchmark harness

Companion report: [`docs/perf/whisper-cross-device-20260827.md`](../../../docs/perf/whisper-cross-device-20260827.md)

Runs Whisper on Hailo-8 / RK3588 / RK3576 / Jetson / Raspberry Pi CPU over one corpus with one scorer.

## Design constraint

**Devices emit only transcripts and per-stage timings; scoring runs once, on one machine.** This is deliberate: installing jiwer/cn2an/opencc separately on each device invites metric drift, and metric drift is fatal to a cross-device comparison. Device-side dependencies are therefore kept minimal — numpy plus each platform's own runtime, no torch and no librosa.

The corpus is `bench/perf/corpus` (SHA256-pinned) and scoring re-implements `bench/perf/runners.py`.

## Files

| File | Purpose | Device-side dependencies |
|---|---|---|
| `rknn_whisper_run.py` | RK3588 / RK3576, RKNN encoder plus either an RKNN or a CPU ONNX decoder | numpy, rknnlite, onnxruntime (hybrid mode) |
| `hailo_corpus_bench.py` | Hailo-8, HEF encoder plus CPU ONNX decoder, with VAD segmentation | numpy, hailo_platform, onnxruntime |
| `wcpp_corpus_run.py` | Jetson, wraps whisper.cpp CUDA and parses its per-stage timings | stdlib plus numpy |
| `trt_whisper_run.py` | Jetson, bare TensorRT three-engine pipeline | tensorrt, cuda-python, numpy |
| `cmp_engine_precision.py` | Diffs every TensorRT engine against onnxruntime | tensorrt, cuda-python, onnxruntime, numpy |
| `score_all.py` | Scoring: CER/WER plus RTF and TTFT | jiwer, cn2an, opencc |

## Usage

```bash
# RK: --encoder_duration must match the window the .rknn was converted at.
# It is not a free knob.
python3 rknn_whisper_run.py --corpus corpus --lang en \
  --encoder model/whisper_encoder_base_20s.rknn --decoder onnx_dec \
  --vocab-dir model --encoder_duration 20 --all-cores \
  --label rk3588-hybrid-en --out results/hybrid_en.json

# Jetson whisper.cpp: do not pass -np, it also disables whisper_print_timings
python3 wcpp_corpus_run.py --corpus corpus --lang en \
  --bin ./whisper.cpp/build/bin/whisper-cli --model models/ggml-base.bin \
  --label orin-nano-wcpp-base-en --out results/wcpp_base_en.json

# Scoring
python3 score_all.py 'results/*.json'
```

Passing a `.rknn` file to `--decoder` runs the all-NPU path; passing a directory runs the optimum ONNX decoder on the CPU with a KV cache. The latter measured better on both axes.

## Three guards, on by default

Audio without enough content makes Whisper skip EOS and then repeat or confabulate. The failure mode is unrelated to the vendor and fires on both zero-padding (a short utterance filling Hailo's fixed window) and truncation (a tail chunk after RK 10 s segmentation), so these are defaults rather than per-platform patches:

1. **Duration-proportional token budget**, `min(cap, duration×8+12)`. A fixed cap guards against the crash, not against the runaway — Whisper's position table holds 448 entries and exceeding it raises `idx=448 out of data bounds`.
2. **Similarity-based repetition cleanup**. Hailo's own `clean_transcription` tests substring containment only, so it misses self-paraphrase, and it splits on `.` and `?` alone, so Chinese with no sentence punctuation passes straight through. This uses a difflib ratio plus CJK sentence enders.
3. **Intra-sentence loop guard**. n-gram detection of a phrase repeating ≥3 times — sentence-level dedup cannot see a loop like `by Llew, by Llew, ...` inside one sentence.

## `trt_whisper_run.py`: build the base encoder with `--bf16`

**And do not also pass `--fp16`.**

`trtexec --fp16` on the base encoder ONNX yields an engine whose output scores cosine **0.826** against onnxruntime, deterministic run-to-run — fp16 kernel selection (5 exponent bits), not a race. `--bf16` restores fp32's exponent range, scores **0.9996**, and gives error rates **identical to fp32** while the encoder costs 12.53 ms against fp32's 39.1 ms.

**Passing `--fp16 --bf16` together does nothing**: TRT picks fp16 throughout and the engine comes out bit-identical to the pure fp16 one (same cosine, same maxabsdiff, same std). Do not expect it to select bf16 for the layers that need the range.

**The defect is deceptive**: the bad encoder still emits a plausible-looking tensor, so the decoder produces fluent English that drifts off-topic and stops early — indistinguishable from a KV-cache bug by inspection. tiny's 30 s and 10 s fp16 engines and both decoder engines are fine (cosine 0.9999), so **this is model-specific**.

**Make an onnxruntime diff a standing step in the engine build process** rather than inferring correctness from the precision flag.

```bash
# base encoder: bf16, no --fp16
trtexec --onnx=enc_base_30s.onnx --bf16 --shapes=input_features:1x80x3000 \
        --saveEngine=enc_base_30s_bf16.plan
# tiny encoder: fp16 is fine
trtexec --onnx=enc_tiny_30s.onnx --fp16 --shapes=input_features:1x80x3000 \
        --saveEngine=enc_tiny_30s.plan
# verify either way — cosine >= 0.999 or the engine is not usable
python3 cmp_engine_precision.py
```
