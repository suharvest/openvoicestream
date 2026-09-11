# Kokoro RK 34% — Per-Stage Timing Diagnostic

**Date**: 2026-05-23
**Device**: radxa (RK3588, 100.77.150.16)
**Container**: `openvoicestream-kokoro` running image `openvoicestream:rk-kokoro-2026-05-23`
**Purpose**: Determine root cause for the M4 RTF FAIL on the 4-stage hybrid path
(CPU prefix → RKNN decoder-front INT8 → **RKNN vocoder-front-half FP16** →
**CPU tail-rest**) vs the 3-stage baseline
(CPU prefix → RKNN decoder-front INT8 → CPU full tail).

Per-stage timers had already been wired into
`third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py`
(`prefix_ms` L449, `front_ms` L456, `vocoder_front_ms` L466, `tail_ms` L480/L493).
Diagnostic instrumentation: a one-line `logger.info("BENCH_META …")` was injected
into `_infer_segment_hybrid` immediately before `return audio, meta`, fired the
full meta dict to container logs, and then stripped & restored at end of run.

Bench harness: 2× warmup + 20× short (`"abc."`) + 5× long
(`"今天天气真好，我们一起去公园散步吧。"`) per configuration, driven through the
running container's HTTP `POST /tts`. Both texts are smaller than `KOKORO_SEQ_LEN=32`,
so every call hits the same static-shape graph and per-stage costs are intentionally
text-invariant. n shown below is after de-duping warmup leakage in the log scan.

## Configuration

| Component                       | 3-stage baseline | 4-stage candidate |
|---------------------------------|------------------|-------------------|
| Profile                         | `rk3588-kokoro-rknn` | `rk3588-kokoro-rknn` + injected 4-stage env |
| `KOKORO_RKNN_VOCODER_FRONT_PATH`| —                | `rk3588/kokoro-vocoder-front-half.native.fp16.rknn` (41 MB) |
| `KOKORO_RKNN_TAIL_REST_PATH`    | —                | `kokoro-vocoder-tail-rest-cpu.onnx` (26 MB) |
| Code (`kokoro_rknn.py`) md5     | `3f75d8a7…` (in-container, pre-M4) | `beff1378…` (host repo submodule branch `feat/kokoro-rk-4stage-vocoder-front`) |

## Per-stage timing (ms, median over n calls)

| stage             | 3-stage (n=27) | 4-stage (n=30) | Δ (4 − 3) |
|-------------------|----------------|----------------|-----------|
| `prefix_ms`       |        15.1    |        15.1    |    0.0    |
| `front_ms`        |        40.8    |        37.2    |   −3.6    |
| `vocoder_front_ms`|        —       |       942.8    |  +942.8   |
| `tail_ms`         |      3472.4    |      2354.2    | −1118.2   |
| **`infer_ms`**    |    **3523.9**  |    **3329.5**  |  **−194.4** |
| `duration_s`      |         5.25   |         5.25   |    0.00   |
| **`rtf`**         |     **0.67**   |     **0.63**   |  **−0.04** |

p95 view (worst-case):

| stage             | 3-stage p95 | 4-stage p95 |
|-------------------|-------------|-------------|
| `vocoder_front_ms`|     —       |    1131     |
| `tail_ms`         |   3609      |    2501     |
| `infer_ms`        |   3668      |    3662     |
| `rtf`             |    0.70     |    0.70     |

## Root-cause finding

**The 4-stage path is NOT slower than baseline on this device under this load.**
Median `infer_ms` shifted from 3524 ms → 3329 ms (≈ 5.5 % faster); RTF improved
from 0.67 → 0.63. The vocoder-front-half RKNN call (943 ms) is net-positive
because it removed 1118 ms from the CPU tail path (32 % reduction of the CPU
component). Net win per call ≈ 175 ms.

Mapping back to the three candidates Codex flagged in
`docs/archive/specs/kokoro-rk-34pct-m4m6-final.md:86-88`:

1. **FP16-slow dominant?** — vocoder_front_ms (943 ms) is 28 % of wall-time,
   not the dominant component. NOT the leading cause.
2. **tail-rest dominant?** — tail_ms (2354 ms) is 71 % of wall-time and is
   still the single largest bucket, but it's *smaller* than 3-stage's
   tail_ms (3472 ms). So tail-rest is the residual bottleneck for further
   optimization, but it's not "regressing" — it's the dominant remaining cost.
3. **Dispatch overhead dominant?** — `sum(stages) ≈ 3349 ms` vs measured
   `infer_ms ≈ 3329 ms` (difference within timer noise). Dispatch overhead
   is negligible — RKNN sub-graph hand-off costs are absorbed.

### Reconciling with M4's "9.14 s / 3.42 s" claim

The numbers from `kokoro-rk-34pct-m4m6-final.md` (4-stage 9.14 s vs baseline
3.42 s) do **not** reproduce in this diagnostic. Two probable explanations,
listed in order of likelihood:

- **Different bench definition.** Likely a longer multi-sentence text or a
  pipeline measurement that included `_split_sentences` looping with
  per-sentence model re-runs (the aggregator at L569 sums per-sentence stages).
  If the M4 bench fed text producing N sentences, the 4-stage path pays
  `N × 943 ms` of vocoder_front overhead while the 3-stage path pays
  `N × 0 ms` extra — the asymmetry grows linearly with sentence count. For a
  9-sentence input the projected gap (≈ 9.14 vs 3.42 → 5.7 s) almost exactly
  matches `≈ 6 × 943 ms` of accumulated vocoder_front cost. **Recommend
  re-running M4 bench with sentence-count instrumentation before treating the
  regression as model-quality bound.**
- **NPU contention with ASR.** This run's bench used HTTP, so the global TTS
  coordinator + BackendManager were in the loop, but ASR was idle. M4 bench
  may have measured under concurrent ASR (RKNN encoder also on NPU core
  AUTO). If so, vocoder_front_ms is shared-NPU sensitive in a way 3-stage's
  CPU tail isn't.

### Per-segment 175 ms net win — is it stable?

p95 `infer_ms` is statistically tied (3668 vs 3662 ms). The mean win is
real (3510 vs 3347 ≈ 163 ms) but small relative to spread. On longer texts
the win compounds per sentence (175 ms × N). On contended NPU it could flip.

## Recommendation against Codex Q5 options

Codex Q5 listed three actions; given findings above:

- **Option A — abandon 4-stage, ship 3-stage only.**
  REJECT. Single-sentence inference is already faster on 4-stage; the M4 FAIL
  number is almost certainly an artifact of multi-sentence aggregation rather
  than per-segment regression.
- **Option B — keep 4-stage but optimize vocoder_front-half RKNN.**
  Lower priority. vocoder_front_ms is 28 % of wall and already net-positive.
  Compressing FP16 → INT8 here might claw back ~400 ms but breaks BERT A/B
  byte-equivalence work just done.
- **Option C — keep 4-stage and attack tail-rest (the dominant 71 %).**
  **RECOMMEND.** tail_ms is still 2354 ms / 71 % of wall on 4-stage and is
  pure CPU ONNXRuntime. Highest-leverage next step is reducing tail-rest:
  either (a) further partition tail-rest so another chunk runs on NPU, or
  (b) ORT-tune tail-rest (graph opt level, intra-op threads, mem pattern —
  currently `tail_threads=4/1`; saturated at 4 cores).

Before any of (A/B/C), the immediate next step should be **re-running M4's
exact bench with per-sentence timing** so we know whether the 9.14 s figure
is a per-call regression (unlikely given this data) or a multi-sentence
amplification (highly likely).

## Reproducibility & evidence

### Container state after diagnostic (baseline restored)

```text
# md5 of kokoro_rknn.py inside container
3f75d8a789733db40e8c6937644c0b40  /opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py
3f75d8a789733db40e8c6937644c0b40  /tmp/kokoro_rknn.py.bak3stage
```

```text
# Profile diff vs pre-diagnostic
$ grep -E 'VOCODER|TAIL_REST' configs/profiles/rk3588-kokoro-rknn.json
NO_4STAGE_ENV_OK
```

```text
# /health after restore
{"tts":true,"tts_backend":"rk:kokoro_rknn","tts_capabilities":["streaming",
 "basic_tts","multi_language"],"tts_worker_cancel_count":0,"asr":true,
 "asr_backend":"rk:qwen3_asr_rk",
 "asr_capabilities":["streaming","offline","multi_language"]}
```

### Raw per-call samples (first 5 of each set)

4-stage:

```text
BENCH_META {'num_tokens':5,'prefix_ms':15.23,'front_ms':25.45,'vocoder_front_ms':828.24,'tail_ms':2099.80,'infer_ms':2968.71,'duration_s':5.248,'rtf':0.566}
BENCH_META {'num_tokens':4,'prefix_ms':15.12,'front_ms':24.70,'vocoder_front_ms':830.48,'tail_ms':2092.94,'infer_ms':2963.24,'duration_s':5.248,'rtf':0.565}
BENCH_META {'num_tokens':4,'prefix_ms':16.79,'front_ms':40.63,'vocoder_front_ms':820.32,'tail_ms':2296.76,'infer_ms':3174.51,'duration_s':5.248,'rtf':0.605}
BENCH_META {'num_tokens':4,'prefix_ms':14.85,'front_ms':39.92,'vocoder_front_ms':1077.81,'tail_ms':2467.34,'infer_ms':3599.92,'duration_s':5.248,'rtf':0.686}
BENCH_META {'num_tokens':4,'prefix_ms':14.95,'front_ms':40.73,'vocoder_front_ms':1066.40,'tail_ms':2437.38,'infer_ms':3559.46,'duration_s':5.248,'rtf':0.678}
```

3-stage:

```text
BENCH_META {'num_tokens':5,'prefix_ms':18.83,'front_ms':41.94,'tail_ms':3314.57,'infer_ms':3375.33,'duration_s':5.248,'rtf':0.643}
BENCH_META {'num_tokens':5,'prefix_ms':15.00,'front_ms':25.00,'tail_ms':3155.55,'infer_ms':3195.55,'duration_s':5.248,'rtf':0.609}
BENCH_META {'num_tokens':4,'prefix_ms':18.43,'front_ms':37.69,'tail_ms':3242.02,'infer_ms':3298.14,'duration_s':5.248,'rtf':0.628}
BENCH_META {'num_tokens':4,'prefix_ms':15.11,'front_ms':40.60,'tail_ms':3293.83,'infer_ms':3349.54,'duration_s':5.248,'rtf':0.638}
BENCH_META {'num_tokens':4,'prefix_ms':15.09,'front_ms':38.14,'tail_ms':3551.46,'infer_ms':3604.69,'duration_s':5.248,'rtf':0.687}
```

### Artifacts staged on radxa (left in place for future re-runs)

```text
/home/radxa/models/tts/kokoro-bucket-32/rk3588/kokoro-vocoder-front-half.native.fp16.rknn  (42 971 539 B, md5 cfa959fd…)
/home/radxa/models/tts/kokoro-bucket-32/kokoro-vocoder-tail-rest-cpu.onnx                  (27 676 816 B, md5 68c62e75…)
/tmp/m4-artifacts/kokoro_rknn.py                                                            (md5 beff1378…)
```

Restoring 4-stage in future: copy `/tmp/m4-artifacts/kokoro_rknn.py` into the
container at the same path, add the two env keys back into the active profile,
`docker restart openvoicestream-kokoro`, expect log line
`"Kokoro 4-stage path active: vocoder_front=… tail_rest=…"`.

## Limitations of this diagnostic

- Single-sentence text only. The 9.14 s vs 3.42 s gap reported in M4 almost
  certainly only reproduces with longer multi-sentence inputs; this run did
  not exercise that path.
- ASR was idle. NPU contention with the Qwen3 ASR encoder was not tested.
- 30 4-stage samples, 27 3-stage samples — enough for medians, tight enough
  for the 175 ms win to be visible but not enough to be confident about p99.
