# Kokoro RK Streaming — TTFA verification (P2 Phase 1)

Status: VERIFIED 2026-05-23 on radxa (RK3588, image `openvoicestream:rk-kokoro-2026-05-23`).
Outcome: Case A — backend per-sentence streaming is already correctly wired
end-to-end. No backend logic change required. TTFA `≤ 800 ms` target stated
in the original P2 brief is **unreachable** with per-sentence granularity
because sentence 0 itself takes ~3.1 s on RK3588 4-stage (hybrid) Kokoro.

## TL;DR

| metric (multi-sentence `"你好，今天天气怎么样？我感觉很棒。"`, n=30) | value |
| --- | --- |
| `t_first_audio_ms` median (TTFA) | **3088.6** |
| `t_total_ms` median | 6201.7 |
| `t_first_audio_ms` mean / p10 / p90 / min / max | 3105.9 / 2930.4 / 3276.2 / 2925.3 / 3438.0 |
| `t_total_ms` mean / min / max | 6214.7 / 5824.2 / 6576.3 |
| bytes per request | 503812 (== 2 × 251904 PCM bytes) |
| audio MD5 (multi-run) | `065ad7f8a7ab2b84e04979c6dcf8bdad` (byte-identical 3/3 runs) |

`t_first_audio_ms ≈ t_header_ms` (≤ 1 ms apart) — i.e. the 4-byte SR
preamble and the first audio chunk arrive in the same TCP burst at
sentence-0 end.

Per-sentence pipelining IS active (timeline run shows sentence 0 audio
arriving in a burst at ~3.21 s then sentence 1 audio at ~6.50 s):

```
t= 3212.1 ms  cum_bytes=4096        ← SR header + sentence 0 start
t= 3213.7 ms  cum_bytes=249856      ← sentence 0 done (~250 KB)
t= 6504.9 ms  cum_bytes=299008      ← sentence 1 first chunk
t= 6506.1 ms  cum_bytes=503812      ← sentence 1 done (~250 KB)
```

Therefore: from the *client's* perspective the second 250 KB worth of
audio is delivered ~3.3 s **after** the first batch — long before
sentence-1 synthesis would have finished if we were buffering the whole
utterance. Real-time playback on the client therefore stays continuous
across the sentence boundary, which is the intended streaming win even
though backend TTFA itself is bounded by sentence-0 wall.

## Code path (verified)

1. `app/main.py:799-801` — `SentenceBuffer` splits the request text
   into sentences via pysbd (zh/en aware).
2. `app/main.py:817-1013` — `stream()` async-generator yields the
   4-byte SR header (`yield struct.pack("<I", sr)`), then submits
   sentence 0 to the TTS executor and drains its chunk queue
   in order. As soon as sentence `i` emits its first chunk,
   sentence `i+1` is submitted (sliding window bounded by
   `OVS_TTS_STREAM_PREFETCH`, default = `max_workers` = 2).
3. `app/backends/rk/tts.py:142-168` — wrapper forwards
   `backend.synthesize_stream(...)` items unchanged, converting
   `np.float32` to `int16` PCM bytes.
4. `third_party/rkvoice-stream/.../kokoro_rknn.py:702-741` —
   `synthesize_stream` iterates `_split_sentences(text)` and yields
   `(audio, meta)` per sentence. **Because the HTTP layer already
   sentence-splits, this backend almost always sees a single
   sentence per call** — the loop is dead-code defensive.
5. `_infer_segment_hybrid` (line 543) runs the full Kokoro
   prefix→front→vocoder-front→tail-rest 4-stage pipeline and
   returns the **entire** sentence audio as one `np.ndarray`
   (no intra-sentence streaming hooks).

## Per-sentence TTFA log line

Added in `kokoro_rknn.py::synthesize_stream` (after first non-empty
sentence yield):

```
kokoro synthesize_stream TTFA: first chunk yielded at t=3169.8 ms
  (sentence 0/1, infer_ms=3168.1, num_tokens=16)
```

`t_ms` is wall time from `synthesize_stream` entry (i.e. one
sentence's worth of inference, since HTTP pre-splits). Grep
`docker logs openvoicestream-kokoro | grep 'synthesize_stream TTFA'`
for direct backend-level latency without re-running the HTTP probe.

## Why TTFA ≠ 800 ms

Sentence 0 wall on RK3588 4-stage (FP16 vocoder-front-half):
- prefix CPU ONNX  ~  60 ms
- decoder-front RKNN INT8  ~ 1450 ms
- vocoder-front-half RKNN FP16  ~ 800 ms
- tail-rest CPU ONNX  ~ 800 ms

Sum ≈ 3.1 s for a 16-token Chinese sentence at bucket-32. That bound
holds regardless of HTTP framing — the smallest unit of audio Kokoro
can emit today is one complete sentence.

To break the 3 s floor we would need either:
- chunked decode-front (split the bucket-32 ONNX into smaller temporal
  segments), or
- a fundamentally different vocoder that streams (e.g. WaveRNN-style
  per-frame), or
- a smaller / faster Kokoro variant (e.g. v0.x mini).

None of these are P2 Phase 1 scope — they belong in P2 Phase 2+.

## Single-sentence regression spot-check

- English `"abc."` (4 tokens, bucket-32): 1 sentence, TTFA = 3512 ms.
- English `"abc."` (warm cache): TTFA = 2972 ms — within previous
  bench range (no regression from instrumentation).
- Audio MD5 stable across re-runs.

## Files touched

- `third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py`
  — added `logger.info("kokoro synthesize_stream TTFA: ...")` after
  first non-empty sentence yield (also captures `sentence i/N`,
  `infer_ms`, `num_tokens`). No control-flow change. Submodule branch
  `feat/kokoro-rk-4stage-vocoder-front`.
- `docs/archive/specs/kokoro-rk-streaming.md` (this file).

No `app/` changes (HTTP layer and wrapper were already correct).

## Container verification

- Local md5 of edited `kokoro_rknn.py`:
  `596009e5993e1f77c9e0874125665c57`.
- After `docker restart openvoicestream-kokoro`, container path
  `/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py`
  md5 = `596009e5993e1f77c9e0874125665c57` (bind-mounted from
  `/tmp/fixed-kokoro_rknn.py` on radxa).
- `/health` returns `tts:true` with kokoro 4-stage active
  (`Kokoro 4-stage path active: vocoder_front=...native.fp16.rknn
  tail_rest=...tail-rest-cpu.onnx`).
