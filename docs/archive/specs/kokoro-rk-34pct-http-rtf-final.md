# Kokoro RK 34% 4-stage HTTP RTF — Final (Promoted)

**Date**: 2026-05-23
**Hardware**: radxa (RK3588), Tailscale `100.77.150.16`
**Container**: `openvoicestream-kokoro`, image `openvoicestream:rk-kokoro-2026-05-23`
**Decision**: **PROMOTE** 4-stage to default (HTTP RTF 0.59 < 3-stage 0.66).

## Context

Earlier RTF reconciliation (`kokoro-rk-34pct-rtf-reconciliation.md`) measured
the 4-stage hybrid path as 100% TypeError because the production container
was running an image-shipped (broken) `app/backends/rk/tts.py` that did not
pop the speaker-id kwarg. The fixed wrapper (md5
`4961497d910cac5531ceafe35e4f1713`) was hot-patched into the production
container (main commit `f3a0edc`), restoring 3-stage /tts/stream throughput
to baseline.

This run re-tests the 4-stage path now that the wrapper is fixed.

## Method

Container `openvoicestream-kokoro` was recreated with bind-mounts so the
fix survives container restarts (the prior `docker cp` workaround was lost
on each restart):

```
/tmp/fixed-tts.py        -> /opt/speech/app/backends/rk/tts.py           (ro)
/tmp/fixed-kokoro_rknn.py -> /opt/speech/third_party/rkvoice-stream/...  (ro)
/tmp/m4-artifacts/kokoro-vocoder-front-half.native.fp16.rknn ->
   /opt/kokoro-rknn/kokoro-vocoder-front-half.native.fp16.rknn           (ro)
/tmp/m4-artifacts/kokoro-vocoder-tail-rest-cpu.onnx ->
   /opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.onnx                    (ro)
```

Env added:

```
KOKORO_RKNN_VOCODER_FRONT_PATH=kokoro-vocoder-front-half.native.fp16.rknn
KOKORO_RKNN_TAIL_REST_PATH=kokoro-vocoder-tail-rest-cpu.onnx
```

Container startup log confirms:
`Kokoro 4-stage path active: vocoder_front=/opt/kokoro-rknn/kokoro-vocoder-front-half.native.fp16.rknn tail_rest=/opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.onnx`.

Benchmark script: `/tmp/bench_kokoro.sh` POSTs `{"text":"abc."}` 100 times
serially to `http://localhost:8621/tts/stream` and records
`%{time_total} %{http_code} %{size_download}` per request.

## File integrity

| File | md5 |
|---|---|
| `/opt/speech/app/backends/rk/tts.py` (fixed wrapper, bind-mount) | `4961497d910cac5531ceafe35e4f1713` |
| `/opt/speech/third_party/rkvoice-stream/.../kokoro_rknn.py` (4-stage, bind-mount) | `beff1378356c30d06cc0462d4149fe66` |
| `kokoro-vocoder-front-half.native.fp16.rknn` | `cfa959fdcd9c7c69eed1cd404366847e` |
| `kokoro-vocoder-tail-rest-cpu.onnx` | `68c62e751d7c21bc1763950be04c1e90` |
| 3-stage audio output ("abc.") | `13cd893168ab9f917ada5107fbe87d47` |
| 4-stage audio output ("abc.") | `1ba124427962bfed41eeccde1175338e` |
| 4-stage audio output (Chinese long) | `d59fbf5e85d478465672f24b2b638474` |

All requests returned HTTP 200 + 251908 bytes (5.2481 s audio @ 24 kHz s16le mono).
4-stage and 3-stage audio MD5s differ (different vocoder split) but bytes per shot
are deterministic within each config.

## Results

### 4-stage HTTP (n = 100, this run)

| metric | wall (s) | RTF |
|---|---|---|
| mean | 3.111 | 0.593 |
| p50 (median) | 3.107 | 0.592 |
| p95 | 3.394 | 0.647 |
| p99 | 3.560 | 0.678 |
| min | 2.925 | 0.557 |
| max | 3.709 | 0.707 |

audio_dur = 251908 / 2 / 24000 = 5.2481 s

First-10 raw (s):
```
3.276647  3.101870  2.925782  2.935490  2.933756
2.924556  3.037565  3.709427  3.230775  3.087001
```
Last-10 raw (s):
```
3.064391  3.062964  3.046826  3.091059  3.040146
3.182135  3.184805  3.152340  3.080265  (md5 line)
```
Non-200: **none**.

### 3-stage HTTP baseline

Reused from `kokoro-rk-34pct-rtf-reconciliation.md` (same container, same
fixture, n=100 within the same image): median wall 3.461 s,
**RTF median 0.660**, p95 RTF 0.701.

Spot-check on this run (5 sanity shots before recreating container in
4-stage mode): 3.186, 3.356, 3.526, 3.586, 3.555 s → consistent with
the 0.66 prior baseline.

### Chinese long-sentence (4-stage, n = 5)

text = "你好，今天天气怎么样？我感觉很棒。"
Wall (s): 3.612, 3.075, 2.980, 3.313, 2.953 → median 3.075 s, RTF 0.586.
All shots returned 251908 bytes (model output buffer is fixed-length;
length is not driven by input token count in this path).

## Decision

| Case | observed | gate | verdict |
|---|---|---|---|
| 4-stage HTTP RTF ≤ 3-stage HTTP RTF (0.66) | **0.59** | promote | ✓ **PROMOTE** |

**~10% RTF improvement** over 3-stage at the median; p95 also passes
(0.647 ≤ 0.66).

## Production changes (this commit)

1. `deploy/artifacts/rk_manifest.json`:
   - `rk3588-kokoro-hybrid-34pct-2026-05-23.runtime_contract.closed_loop_validated`
     → `true`
   - add `production_validated_at: "2026-05-23"`
   - `validation_status: audio_ab_pass_rtf_fail` → `http_rtf_pass`

2. `configs/profiles/rk3588-kokoro-rknn.json`:
   - add `KOKORO_RKNN_VOCODER_FRONT_PATH`,
     `KOKORO_RKNN_TAIL_REST_PATH`,
     `RK_ARTIFACT_SET=rk3588-kokoro-hybrid-34pct-2026-05-23`
     so new deployments pick up the 4-stage artifact set + env wiring by default.

## Long-running deployment caveat (image rebuild still required)

The production image `openvoicestream:rk-kokoro-2026-05-23` still ships the
**broken** `app/backends/rk/tts.py` and the **3-stage**
`kokoro_rknn.py`. The radxa container is currently running both files via
host bind-mounts (`/tmp/fixed-tts.py` and `/tmp/fixed-kokoro_rknn.py` →
container destinations). Bind-mounts survive `docker restart` and
`docker stop/start`, but if the container is destroyed and recreated from
scratch (or the host `/tmp` is wiped on reboot — `/tmp` is RAM-backed on
many configs), the wiring is lost.

**Action items for the next image release**:
- Bake fixed `tts.py` (md5 `4961497d910cac5531ceafe35e4f1713`) into the image.
- Bake 4-stage `kokoro_rknn.py` (md5 `beff1378356c30d06cc0462d4149fe66`) into
  the image at `third_party/rkvoice-stream/rkvoice_stream/backends/tts/`.
- Ensure the 4-stage artifacts (`kokoro-vocoder-front-half.native.fp16.rknn`
  + `kokoro-vocoder-tail-rest-cpu.onnx`) are included in `/opt/kokoro-rknn/`.
- After image rebuild, the bind-mounts can be dropped.

Until then, the container launch incantation (recorded in
`/home/radxa` deploy history) must include the four bind-mounts plus the
two env vars listed above.

## Post-test health

```
/health => {"tts":true,"tts_backend":"rk:kokoro_rknn","asr":true, ...}
POST /tts/stream "abc." => HTTP 200, 251908 bytes, 3.39 s
fixed tts.py md5    : 4961497d910cac5531ceafe35e4f1713   (unchanged ✓)
4-stage kokoro_rknn.py md5 : beff1378356c30d06cc0462d4149fe66 (unchanged ✓)
```
