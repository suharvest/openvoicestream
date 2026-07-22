# Paraformer RK3576 Streaming A/B - 2026-06-08

Device: `cat-remote` RK3576  
Base container image: `openvoicestream:rk-slim-fresh-20260603`  
Validation image: `openvoicestream:rk-paraformer-v2v-prepare-20260608`  
Corpus: `bench/perf/corpus`, `short`, 5 zh + 5 en files  
Artifact: `paraformer-hybrid` with RKNN encoder prefix buckets, CPU encoder
suffix ONNX, and RKNN decoder:

- initial: `encoder_prefix_to_block30.400.fp16.rknn`
- added in this run: `40`, `80`, `160`, `240` frame FP16 buckets

## Finding

The original stream path was not usable because it mixed:

- per-utterance fbank CMVN recomputed on growing partial audio, so the same
  frame changed feature values across chunks;
- decoder cache reuse across chunks while encoder context changed.

This caused duplicated/missing tokens and heavy repeated compute.

## RK3576 Results

| Mode | zh CER | en WER | en CER no-space | Mean feed ms | Mean prepare ms | Mean finalize ms | Mean total compute ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline, utterance CMVN | 7.6% | not measured | 8.8% | 0 | not measured | 2140 / 2253 | 2140 / 2253 |
| original stream | 48.3% | 100.0% before spacing fix | 41.6% | 8642 / 10487 | not measured | 2207 / 2124 | 10849 / 12612 |
| stream, CMVN none | 28.4% | not rerun | not rerun | 8799 | not measured | 2104 | 10903 |
| stream, CMVN none + batch-final decoder | 9.4% | 100.0% before spacing fix | 9.8% | 7905 / 9357 | not measured | 1959 / 1976 | 9864 / 11333 |
| stream, CMVN none + batch-final decoder + 2.0s process cadence, 400 bucket only | 9.4% | 21.7% | 8.9% | 2243 / 2467 | 1984 / 1993 | 0.18 / 0.14 | 4226 / 4461 |
| same, 80/160/240/400 buckets | 9.4% | 21.7% | 8.9% | 549 / 635 | 845 / 872 | 0.29 / 0.24 | 1394 / 1508 |
| same, 40/80/160/240/400 buckets, `/asr/stream` realtime | 9.4% | 34.6% | 11.4% | wall 3597 / 3995 | hidden by speech | 861 / 869 EOS-to-final | n/a |
| same + `/asr/stream` prepare control, 500 ms lead | 9.4% | 34.6% | 11.4% | wall 3596 / 3995 | hidden by VAD lead | 326 / 347 EOS-to-final | 829 / 851 prepare-to-final |

## `/v2v/stream` Dialogue-Path Results

Measured on `cat-remote` RK3576, short corpus, 250 ms chunks, client EOS.
`total` is the client-observed time from final audio/EOS to ASR final result.

| Mode | zh CER | zh total mean / p50 / p95 | en WER | en total mean / p50 / p95 | prepare-to-final mean |
|---|---:|---:|---:|---:|---:|
| hybrid encoder + CPU decoder, client EOS | 12.1% | 860 / 821 / 989 ms | 21.7% | 871 / 876 / 994 ms | n/a |
| hybrid encoder + CPU decoder, `asr_prepare`, 500 ms lead | 12.1% | 323 / 281 / 453 ms | 21.7% | 392 / 361 / 499 ms | zh 825 ms, en 893 ms |
| hybrid encoder + RKNN decoder, client EOS | 12.1% | 548 / 507 / 653 ms | 21.7% | 579 / 510 / 712 ms | n/a |
| hybrid encoder + RKNN decoder, `asr_prepare`, 500 ms lead | 12.1% | 105 / 107 / 135 ms | 21.7% | 165 / 126 / 265 ms | zh 606 ms, en 667 ms |

Measured on `radxa` RK3588 with RK3588-specific prefix buckets and RKNN
decoder, same image/code path and corpus:

| Mode | zh CER | zh total mean / p50 / p95 | en WER | en total mean / p50 / p95 | prepare-to-final mean |
|---|---:|---:|---:|---:|---:|
| hybrid encoder + RKNN decoder, client EOS | 12.1% | 424 / 450 / 524 ms | 21.7% | 477 / 470 / 525 ms | n/a |
| hybrid encoder + RKNN decoder, `asr_prepare`, 500 ms lead | 12.1% | 100 / 91 / 138 ms | 21.7% | 98 / 96 / 121 ms | zh 602 ms, en 599 ms |

Current acceleration split:

- RKNN/NPU: encoder prefix buckets
  (`encoder_prefix_to_block30.{40,80,160,240,400}.fp16.rknn`).
- CPU/ONNXRuntime: encoder suffix (`encoder_suffix_from_block30.onnx`) and
  decoder (`decoder-rknn.onnx`) for the original validated profile.
- RKNN/NPU decoder is usable when `decoder.400x40.fp16.rknn` is present in the
  hybrid RKNN directory. For this validation it was copied from
  `/home/cat/models/paraformer-streaming/rknn/decoder.400x40.fp16.rknn` into
  `/home/cat/models/paraformer-hybrid/rknn/rk3576/`.
- There is no GPU acceleration path on RK for this stack. ONNXRuntime logs GPU
  device discovery warnings, but the remaining ONNX sessions are CPUExecutionProvider.
  The RK acceleration path is RKNN/NPU.

Full RKNN encoder validation:

- `rk3576-paraformer-full-rknn-matcha` successfully loaded full encoder buckets
  `encoder.{40,80,160,400}.fp16.rknn` and RKNN decoder.
- Warmup failed with `Paraformer RKNN encoder warmup produced invalid output`.
- A direct 40-frame encoder probe on `cat-remote` showed finite values in only
  `652 / 8704` effective-frame encoder elements. The invalid values occur inside
  the real audio frames, not only padding. This full encoder artifact is not
  safe to use on RK3576 without reconversion or graph repair.

ORT threading A/B on zh short, `/v2v/stream`, 500 ms prepare lead:

| ORT intra/inter threads | total mean / p50 / p95 | prepare-to-final mean / p50 / p95 | Result |
|---|---:|---:|---|
| default | 323 / 281 / 453 ms | 825 / 782 / 954 ms | best |
| 1 / 1 | 2118 / 1953 / 2686 ms | 2620 / 2454 / 3189 ms | much slower |
| 2 / 1 | 838 / 766 / 1078 ms | 1343 / 1266 / 1593 ms | slower |
| 4 / 1 | 484 / 482 / 575 ms | 985 / 983 / 1077 ms | slower than default |

This confirms the remaining CPU suffix/decoder work is substantial and benefits
from ORT's default multithreading. The next material performance step is not
thread tuning; it is moving more of the suffix/decoder path to RKNN or exporting
full encoder buckets that avoid the CPU suffix.

Raw result files:

- `bench/perf/results/v2v_stream_remote_20260608-174744-806463-p50056-4029c59f.json`
- `bench/perf/results/v2v_stream_remote_20260608-174816-733794-p51092-0f69cb61.json`
- `bench/perf/results/v2v_stream_remote_20260608-174853-791459-p52317-d21a0111.json`
- `bench/perf/results/v2v_stream_remote_20260608-174930-972653-p53512-7d491b78.json`
- `bench/perf/results/v2v_stream_remote_20260609-091320-020359-p48423-e939ef16.json`
- `bench/perf/results/v2v_stream_remote_20260609-091354-919794-p49522-1074431b.json`
- `bench/perf/results/v2v_stream_remote_20260609-091428-139064-p50578-9a351661.json`
- `bench/perf/results/v2v_stream_remote_20260609-091501-834856-p51655-49b19185.json`
- `bench/perf/results/v2v_stream_remote_20260609-102818-204724-p19394-c5474029.json`
- `bench/perf/results/v2v_stream_remote_20260609-102859-141197-p20726-5f00be98.json`
- `bench/perf/results/v2v_stream_remote_20260609-102941-629668-p22010-fc44b2ee.json`
- `bench/perf/results/v2v_stream_remote_20260609-103022-845003-p23276-1b24c854.json`

English spacing is now restored in `decode_ids()`, so WER is usable again.
`en CER no-space` is still useful to compare content accuracy independent of
word-boundary formatting.

## Current Patch Direction

Added optional RKNN Paraformer controls:

- `PARAFORMER_FBANK_CMVN=none`
- `PARAFORMER_STREAM_DECODE=batch_final`
- `PARAFORMER_STREAM_PROCESS_SEC=2.0`

These are set in RK3576/RK3588 Paraformer profiles. Defaults remain compatible:
existing deployments keep `utterance` CMVN, incremental stream decode, and
0.67s process cadence unless the profile opts in.

`decode_ids()` now restores English word spaces while keeping CJK tokens joined
and BPE `@@` subwords merged. RK3576 validation:

- English short 5: WER 21.7%, no-space char CER 8.9%.
- Chinese short 1 sanity: CER 0.0%, no added spaces.

RK3576 small buckets were generated on WSL2 with an isolated `uv` environment
to avoid the host user-site torch/NCCL conflict:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  UV_PYTHON_PREFERENCE=only-system \
  uv run --isolated \
    --with rknn-toolkit2==2.3.2 \
    --with 'setuptools<81' \
    --with 'onnx==1.16.1' \
    python /tmp/convert_paraformer_prefix_buckets.py \
      --onnx /tmp/paraformer-hybrid/paraformer-hybrid/encoder_prefix_to_block30.onnx \
      --out-dir /tmp/paraformer-hybrid/paraformer-hybrid/rknn \
      --target rk3576 \
      --precision fp16 \
      --frames 40 80 160 240
```

For dialogue-style clients, `/asr/stream` now accepts a JSON control frame:

```json
{"type": "prepare"}
```

The server drains queued audio, runs `stream.prepare_finalize()` in the
background, and keeps the WebSocket open. A later empty binary frame or EOU
returns the cached final text. On RK3576, sending `prepare` 500 ms before EOS
reduced short-corpus EOS-to-final from ~0.86s to ~0.33s without changing Chinese
CER. The total ASR prepare-to-final compute remains ~0.83-0.85s; the gain is
that ~500 ms is hidden under frontend/external VAD hangover.

The same early-prepare mechanism is now wired into the main `/v2v/stream`
path in a reusable way:

- `voxedge.engine.ASRSessionManager` owns
  `prepare_finalize_for_generation(generation)`, guarded by the same
  generation and worker-op serialization rules as `finalize_with_status()`.
- `server/core/asr_session_manager.py` is back to a plain voxedge re-export;
  no application-local subclass/hack is needed.
- `/v2v/stream` accepts `{"type":"asr_prepare"}` plus compatible aliases
  (`prepare`, `pre_eou`, `prepare_finalize`), schedules same-generation
  prepare, and waits for that same task before finalizing.
- `bench/perf/perf.py v2v-stream` now has `--prepare-lead-ms` and records
  `prepare_to_final_ms`, so the dialogue path can be A/B tested with the same
  client-EOS lead-time model.

Local validation passed for the generic voxedge manager and seeed v2v wiring.
RK3576 `/v2v/stream` device validation passed with the updated voxedge package.
Two container compatibility fixes were needed:

- `deploy/docker/Dockerfile.rk.voxedge-patch` puts
  `/opt/speech/third_party/rkvoice-stream` before `/opt/speech` in
  `PYTHONPATH`, so the image uses the project source copy rather than an older
  installed `rkvoice_stream` package.
- `ParaformerRKNNBackend.create_stream()` now accepts optional
  `stream_options`, matching the shared ASR backend interface used by voxedge.

## Remaining Performance Work

The stream path is now accuracy-usable on short utterances. With
`prepare_finalize()` the final response call itself is effectively free
(~0.2-0.3 ms). Adding 80/160/240 frame buckets drops RK3576 prepare from
~2.0s to ~0.85-0.87s and total compute from ~4.2-4.5s to ~1.4-1.5s for the
short-dialog corpus. A 40-frame bucket was also generated but did not improve
the measured `/asr/stream` EOS-to-final latency on this corpus; 1.0s process
cadence was also tested and did not improve latency versus 2.0s.

Next steps:

1. Upload or package the RK3576/RK3588 small buckets with the `paraformer-hybrid`
   artifact sets, including `decoder.400x40.fp16.rknn`, so deployment gets the
   same performance improvement without device-local test copies. Manifest sets:
   `rk3576-paraformer-hybrid-rknn-decoder-2026-06-09` and
   `rk3588-paraformer-hybrid-rknn-decoder-2026-06-09`.
2. Generate and validate a production RK image/tag with the updated voxedge
   package and refreshed `rkvoice-stream` source, replacing the temporary patch
   image used here.
3. If ASR stop-to-final still needs to drop further, the remaining ASR bottleneck
   is now the CPU encoder suffix. Full encoder RKNN is not currently usable on
   RK3576 because the existing artifact emits invalid values on real frames; the
   next model-side task is reconverting or splitting the suffix into smaller
   RKNN-safe islands.

## Full RKNN encoder — CONCLUSIVE root cause + verdict (2026-06-16, cat-remote)

Exhaustive follow-up confirms the full-encoder RKNN is **not viable on RK3576 and
not fixable by conversion settings** — only by model-side surgery.

- **Root cause = intrinsic fp16 dynamic-range overflow** in the deep encoder
  residual stream. Per-block probe on real audio (zh `0.wav`): residual magnitude
  grows monotonically through the 49 blocks and first breaches fp16 max (65504) at
  **encoder block 31's residual Add** (`/encoder/encoders/encoders.31/Add_1`), with
  full collapse (~11% of elements → Inf/NaN) by block 35. This is exactly the
  boundary the HYBRID split uses (NPU prefix-to-block30 + CPU/ONNX suffix-from-
  block30) — which is why hybrid is finite and full-fp16 is not.
- **No precision/opt-level recipe works on RK3576** (rknn-toolkit2 2.3.2, builds on
  aarch64 via `uv run --isolated`):
  - fp16, optimization_level=0 → still non-finite (rules out RKNN graph fusion).
  - bf16 → conversion fails: `unsupported tensor dtype in Sub … per-layer mul` → SIGSEGV.
  - tf32 → `Can not support request type: tfloat32`.
  RK3576 NPU is **fp16-only** for this graph, and fp16 cannot hold the block-31+
  residual range.
- **Verdict: keep HYBRID** (production-correct, the only finite config). The full
  variant's only fix is model-side range control before conversion (scale/normalize/
  clamp residual adds in blocks 31–48, or split the suffix into a higher-headroom
  sub-graph) — not worth it vs the working CPU suffix. Leaf
  `asr.paraformer_rknn.rk3576.full` is marked ⛔ NOT VIABLE.
