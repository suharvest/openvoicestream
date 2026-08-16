# Nemotron-3.5-ASR v0.10 validation

Status: evaluated on 2026-08-16; not adopted into a production profile.

## Decision

The TensorRT steady-state performance gate passed, but the multilingual
quality gate failed. Chinese CER was 26.03% versus the 10% limit and the Qwen
v0.10 incumbent's 6.58%. English WER was 6.33%, passing the absolute 8% limit
but trailing Qwen's 4.88%. No Nemotron engine or runtime image was published.

The official Transformers reference reproduced the same result: 26.34%
Chinese CER and 6.33% English WER. TensorRT matched 17 of 20 transcripts
exactly; the remaining three differed by only a few Chinese characters, and
TensorRT's aggregate CER was slightly better. The HF-to-ONNX-to-TensorRT
conversion parity gate therefore passed. The Chinese failure is a checkpoint
or evaluation-domain limitation, not a conversion regression.

Nemotron remains an experimental lane outside the qualified v0.10 release
lock. This evaluation does not invalidate or modify the published release.

## Pinned identity

- Model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- HF revision: `1c8deaecc64b91f034d73e08dd8b64625eb3395d`
- Model SHA-256: `9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174`
- Platform: JetPack 6.2, CUDA 12.6, TensorRT 10.3, SM87, `MAXN_SUPER`
- Corpus: 20 pinned Chinese/English WAVs
- Prompt IDs: `zh-CN=4`, `en-US=0`

The checkpoint supports cache-aware streaming, but the v0.10 experimental
Edge-LLM runner tested here is offline batch 1 and does not expose streaming
session state. This result makes no streaming, TTFD, cancellation, or
concurrency claim.

## Results

| Path | Chinese CER | English WER | Performance |
|---|---:|---:|---|
| TensorRT v0.10 FP16 | 26.03% | 6.33% | RTF 0.035 / 0.030 on two representative clips |
| Transformers FP32 | 26.34% | 6.33% | parity reference on Spark |
| Qwen v0.10 incumbent | 6.58% | 4.88% | same corpus and normalization |

The RTF threshold was 0.08. Cold one-process wall time was about 6.7 seconds
because it includes process startup and engine deserialization, so it is not a
service TTFD measurement.

Machine-readable evidence is in
`bench/perf/baselines/edgellm-v010-nemotron-asr-orin-nano-evaluation.json`.
Reproducible runners are `bench/perf/nemotron_asr_offline_gate.py` and
`bench/perf/nemotron_asr_hf_reference.py`.
