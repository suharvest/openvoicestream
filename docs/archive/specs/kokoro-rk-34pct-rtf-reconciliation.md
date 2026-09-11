# Kokoro RK 34% 4-stage RTF Measurement Reconciliation

**Date**: 2026-05-23
**Hardware**: radxa (RK3588), Tailscale `100.77.150.16`
**Container**: `openvoicestream-kokoro`, image `openvoicestream:rk-kokoro-2026-05-23`
**Text fixture**: `"abc."` → 5.2481 s audio @ 24 kHz s16le mono (251908 bytes)

## Context

Prior measurements of the 4-stage Kokoro hybrid path produced contradictory
RTFs:

| Method | Source | 4-stage RTF | 3-stage RTF | Verdict |
|---|---|---|---|---|
| File mtime span (n=50 each) | M4 agent | **1.74** | **0.66** | 4-stage 163% slower |
| In-process Python loop | timing diagnostic | **0.63** | 0.67 | 4-stage 5.5% faster |

This document settles the contradiction by measuring HTTP `/tts/stream` —
the actual production wire path — under both configurations.

## Method

`/tmp/http_rtf_bench.py` (also pushed to radxa) POSTs JSON
`{"text":"abc."}` to `http://127.0.0.1:8621/tts/stream` and measures TTFB
(time-to-first-byte) and wall time per request. Stream is raw int16 PCM @
24 kHz; `audio_dur = nbytes / 2 / 24000`. RTF = `wall / audio_dur`.
n = 100 measured + 5 discarded warmups per config.

## Results

### 3-stage HTTP baseline (n=100, container `openvoicestream-kokoro` stock, no `KOKORO_RKNN_VOCODER_FRONT_PATH` env)

```
SUMMARY {"label": "3stage", "n": 100,
         "ttfb_ms_median": 3460.83, "ttfb_ms_p95": 3679.94,
         "wall_ms_median": 3461.75, "wall_ms_p95": 3680.83,
         "audio_s_median": 5.2481,
         "rtf_median": 0.6596, "rtf_p95": 0.7014, "rtf_mean": 0.6565}
```

Every request returned the full 251908 audio bytes. RTF tightly clustered
0.62–0.71. Matches the in-process diagnostic 3-stage number (0.67) within
margin.

### 4-stage HTTP attempt

After recreating the container with
`KOKORO_RKNN_VOCODER_FRONT_PATH=/opt/kokoro-rknn/kokoro-vocoder-front-half.native.fp16.rknn`
+ `KOKORO_RKNN_TAIL_REST_PATH=/opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.onnx`,
and patching `kokoro_rknn.py` from `/tmp/m4-artifacts/`, the container
startup log shows:

```
[I] kokoro_rknn: Kokoro 4-stage path active:
    vocoder_front=/opt/kokoro-rknn/kokoro-vocoder-front-half.native.fp16.rknn
    tail_rest=/opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.onnx
```

But all 100 HTTP requests returned **4 bytes** in ~2.5 ms with container
error:

```
[E] app.main: tts/stream synthesis failed for sentence='abc.'
TypeError: rkvoice_stream.backends.tts.kokoro_rknn.KokoroRKNNBackend.synthesize_stream()
           got multiple values for keyword argument 'speaker_id'
```

This is **not** a 4-stage perf bug. It is a signature/caller contract
break: the M4-version `kokoro_rknn.py` (md5
`beff1378356c30d06cc0462d4149fe66`) declares
`synthesize_stream(self, text, speaker_id=0, ...)` but the RK TTS wrapper
in `/opt/speech/app/backends/rk/tts.py` calls it as

```python
self._inner.synthesize_stream(
    text=text,
    speaker_id=speaker_id,    # positional kwarg
    speed=speed, pitch_shift=pitch_shift,
    **kwargs,                 # `kwargs` still contains speaker_id passed
                              # from main.py via voice_kwargs
)
```

so `speaker_id` arrives twice → TypeError on every request, generator
raises before yielding any audio.

### Post-restoration 3-stage check

After restoring stock `kokoro_rknn.py` (md5
`3f75d8a789733db40e8c6937644c0b40`) by recreating the container without
4-stage env, the **same TypeError still fires** on every HTTP request:

```
[001] MEAS ttfb=    3.1ms wall=     3.1ms audio=0.0001s bytes=4 rtf=37.47
[002] MEAS ttfb=    2.6ms wall=     2.7ms audio=0.0001s bytes=4 rtf=31.96
```

This is the same bug. Yet the **first** 3-stage run on the same image
md5 produced 251908-byte valid audio (RTF 0.66). The only explanation: the
container that produced the first 3-stage run had its `kokoro_rknn.py`
module loaded into Python memory **before** something modified the
on-disk file. Recreating the container forces fresh module load from
disk, exposing the bug. Either:

1. A prior diagnostic agent hot-patched a working `kokoro_rknn.py` in
   that container's overlay layer but did not commit the change back to
   `/tmp/m4-artifacts/`, then "restored" with `docker cp` of the buggy
   stock file (which had been clobbered by something else) — and the
   running Python interpreter kept the *previously loaded* working
   module in memory.
2. The image's stock `kokoro_rknn.py` and the calling wrapper
   `app/backends/rk/tts.py` are simply incompatible in this image build
   — every freshly-created container is broken — and prior measurement
   sessions worked by accident of hot-loaded modules.

Either way, **the on-disk code currently shipped in the image cannot
serve a real HTTP `/tts/stream` request with `voice_kwargs` containing
`speaker_id`**.

## Reconciliation of the two prior numbers

| Number | What it really measured |
|---|---|
| M4 mtime RTF 1.74 (4-stage) | Includes a non-perf cost: probably mtime span over a process that retried after errors, or measured wall clock of a workflow that included unrelated overhead (file IO, container restart, multi-process sync). Not a fair HTTP-equivalent RTF. |
| M4 mtime RTF 0.66 (3-stage) | Coincidentally close to the **valid** HTTP measurement (0.660 median this report) — likely because 3-stage code path was actually executing successfully in that run, and mtime span tracked it accurately. |
| In-proc diagnostic 0.63 (4-stage) | Bypassed `synthesize_stream` wrapper and called the backend's internal compute functions directly. Reflects raw 4-stage compute cost, **not** end-to-end HTTP RTF. |
| In-proc diagnostic 0.67 (3-stage) | Same bypass. Matches HTTP 3-stage (0.66) because no wrapper-level breakage hid the compute. |

Neither M4 mtime nor in-process Python loop tested the actual production
HTTP wire path with the M4 `kokoro_rknn.py` artifact. The HTTP wire path
**cannot run** with the current artifact because of the kwarg duplication
bug. The 4-stage perf claim is unverifiable end-to-end until the
signature is fixed.

## Verdict

**NO-GO — cannot promote 4-stage to default profile.**

Rationale:
- Cannot run a full HTTP A/B against 3-stage because the 4-stage
  artifact (and apparently the stock 3-stage on disk in this image)
  errors on every real request.
- The 4-stage RTF "wins" reported elsewhere came from code paths that
  bypass the wrapper. Production traffic does not bypass the wrapper.
- Even if the M4 4-stage `synthesize_stream` signature is patched to
  pop `speaker_id` from kwargs, an actual HTTP A/B must be rerun to
  validate the perf claim before promotion.

## Next actions (not done in this task — out of scope)

1. Fix `app/backends/rk/tts.py:generate_streaming` to `kwargs.pop("speaker_id", None)` (and confirm `synthesize_stream` wrapper at line 169 likewise) before forwarding to `self._inner.synthesize_stream`. This is a real bug regardless of 3-stage vs 4-stage; first-container success was an accident of stale-module hot-load.
2. Or fix `kokoro_rknn.py:synthesize_stream` to accept `**kwargs` and silently drop duplicate `speaker_id`.
3. Re-run this HTTP A/B (`/tmp/http_rtf_bench.py` on radxa, `python3 /tmp/http_rtf_bench.py {label} 100 5`) once the signature is fixed.
4. Decision gate: 4-stage HTTP RTF median must be ≤ 0.66 (3-stage HTTP measurement from today) to promote.

## Container restore state

- Container `openvoicestream-kokoro`: stock image, no 4-stage env.
- On-disk `kokoro_rknn.py`: md5 `3f75d8a789733db40e8c6937644c0b40` (image stock).
- Host bind dir `/home/radxa/models/tts/kokoro-bucket-32/`: now contains
  both `kokoro-vocoder-front-half.native.fp16.rknn` (M4 4-stage front
  half, md5 `cfa959fdcd9c7c69eed1cd404366847e`) and
  `kokoro-vocoder-tail-rest-cpu.onnx` (md5 `68c62e751d7c21bc1763950be04c1e90`),
  left in place for fast re-test after signature fix.
- M4 artifacts still in `/tmp/m4-artifacts/` on radxa.
- `/health` returns `tts:true, tts_backend:rk:kokoro_rknn`.
- **Important caveat**: HTTP `/tts/stream` synthesis currently fails
  with the kwarg-duplication TypeError. This pre-existed this
  measurement session; the very first 3-stage run that produced valid
  audio appears to have been served by a stale in-memory Python module
  in the prior container, not the on-disk code.

## Raw evidence

3-stage successful HTTP run (last 5 of 100):
```
[095] MEAS ttfb= 3741.7ms wall=  3742.6ms audio=5.2481s bytes=251908 rtf=0.7131
[096] MEAS ttfb= 3585.3ms wall=  3586.2ms audio=5.2481s bytes=251908 rtf=0.6833
[097] MEAS ttfb= 3491.7ms wall=  3492.7ms audio=5.2481s bytes=251908 rtf=0.6655
[098] MEAS ttfb= 3461.2ms wall=  3462.1ms audio=5.2481s bytes=251908 rtf=0.6597
[099] MEAS ttfb= 3321.8ms wall=  3322.7ms audio=5.2481s bytes=251908 rtf=0.6331
```

4-stage failing HTTP run (last 5 of 100, all return 4 bytes):
```
[095] MEAS ttfb=    2.6ms wall=     2.6ms audio=0.0001s bytes=4 rtf=30.81
[096] MEAS ttfb=    2.5ms wall=     2.5ms audio=0.0001s bytes=4 rtf=30.25
[097] MEAS ttfb=    2.5ms wall=     2.5ms audio=0.0001s bytes=4 rtf=29.69
[098] MEAS ttfb=    2.4ms wall=     2.4ms audio=0.0001s bytes=4 rtf=29.28
[099] MEAS ttfb=    2.5ms wall=     2.5ms audio=0.0001s bytes=4 rtf=30.09
```

Container error log (4-stage attempt and post-restore 3-stage check):
```
[E] app.main: tts/stream synthesis failed for sentence='abc.'
TypeError: rkvoice_stream.backends.tts.kokoro_rknn.KokoroRKNNBackend.synthesize_stream()
           got multiple values for keyword argument 'speaker_id'
```
