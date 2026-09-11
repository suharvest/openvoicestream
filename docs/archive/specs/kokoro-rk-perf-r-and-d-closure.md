# Kokoro RK NPU Performance R&D — Closure Summary (2026-05-23)

> Status: **CLOSED**. Original spec target (42% NPU → revised to 34% after BERT A/B verification) achieved. All identified follow-up optimization paths systematically evaluated; remaining paths are NO-GO (hardware-bound) or low ROI / high risk.

## What shipped

Production runs the 4-stage hybrid pipeline with 3-bucket dynamic router:

```
tokens → misaki G2P → CPU prefix → RKNN decoder-front INT8 → RKNN vocoder-front-half FP16 → CPU tail-rest → audio
                                                  ↑
                                       bucket-8 / bucket-16 / bucket-32 routed by n_tokens
```

### Quantitative outcomes (start vs final)

| Metric | Start | Final | Improvement |
|---|---|---|---|
| Short EN TTFA p50 | 3.1s | **0.75s** | 4.1× |
| Short ZH TTFA p50 | broken (4 B silent fail) | **0.70s** | fixed + 4.4× |
| Mid EN/ZH TTFA p50 | 3.1s | **1.6s** | 2× |
| Long ZH multi-sentence | broken/garbage audio | works correctly | fixed |
| Chinese language support | 0% (no G2P) | full (misaki) | wired |
| HTTP RTF (long sentences) | 0.66 | **0.59** | 10% |
| Per-stage timing | unknown | instrumented | TTFA log shipped |

### Production state

- HF mirror complete at `harvestsu/seeed-local-voice-rk-artifacts/blob/main/rk3588/kokoro-hybrid-v1/{bucket8,bucket16,bucket-32}/`
- Production image **rebuilt 2026-05-23 as `openvoicestream:rk-kokoro-2026-05-23-rebuilt`** — self-contained (misaki ZH G2P stack, fixed `app/backends/rk/tts.py`, submodule `65b9a13` `kokoro_rknn.py`, all 14 active bucket artifacts + `tokens.txt` baked in). No host bind-mounts of model files or `.py` overrides required. Pre-rebuild image `openvoicestream:rk-kokoro-2026-05-23` retained on host as the rollback target.
- Deployment runbook: `docs/runbooks/kokoro-rk-deploy.md`.
- 3-bucket router env-gated; missing env vars fallback to baseline gracefully.

## R&D paths systematically evaluated

### Delivered

- **M1-M6** per original spec (boundary discovery, BERT FP16 RKNN, vocoder front-half FP16 RKNN, manifest integration, RTF gate)
- **P1** Chinese silent-fail bug — misaki G2P wired
- **P2 Phase 1** per-sentence streaming verification — already worked, added TTFA instrumentation
- **P2 Phase 2** dynamic bucket router (8/16/32) — landed
- **P3** full reproduction documentation + HF mirror
- **P7b (bucket-8 only)** tail-rest ORT static MM+Gemm INT8 — bucket-8 TTFA p50 671 ms (vs P7a 747 ms → -10 %), bucket-16/32 untouched (no perf delta from static at longer sequences). Audio gate PASS (rel_l2 0.020, gate 0.05). Shipped 2026-05-23 as 3-tier env-gated default (static > dynamic > FP32). Bucket-16/32 static QDQ built but not wired (perf-neutral; would only add artifact complexity). Conv/ConvTranspose static QDQ confirmed audio-incompatible on this vocoder (structural NO-GO — see negative findings).

### Negative findings (NO-GO with evidence)

| Path | Root cause | Reference |
|---|---|---|
| BERT FP16 RKNN integration | BERT bit-exact dead code in Kokoro v1.0 ONNX (10/10 A/B byte-identical) | `kokoro-bert-ab-audio-report.md` (commit 3b18517) |
| P2 Phase 3a — bucket-8/16 tail-rest on NPU | REGTASK bit-width overflow on bucket-16 (silent zero output); bucket-8 NPU 13% slower than CPU (ConvTranspose unfavorable on RKNPU) | this doc + agent a45ad569b09af4a8b report |
| P2 Phase 3b — bucket-32 chunked tail-rest | Same ConvTranspose CPU-bound constraint as 3a; chunking adds complexity without fixing root cause | dependency on 3a |
| P4 codex Q4 — decoder-front single-segment extension | RKNN one-graph-one-precision constraint: INT8 full-segment fails audio gate (M4 history rel_l2 0.07-0.41); FP16 full-segment slows decoder-front by 25-35ms vs 10ms dispatch savings → net loss | this doc + codex eval |
| P7a tail-rest ORT dynamic INT8 (MatMul+Gemm) | Audio gate PASS (worst rel_l2 0.018, gate 0.05); deployed env-gated. But measured TTFA delta ±5% (run-to-run noise) vs spec-projected -15-20%. Tail-rest is Conv-dominated, not MatMul-dominated → wrong target. Shipped as opt-in baseline-equivalent default; future work needs static QDQ with calibration covering Conv ops. | `kokoro-rk-tail-rest-int8.md` |
| P7b tail-rest static QDQ on **Conv + ConvTranspose** (the spec target) | Universal audio gate FAIL: worst rel_l2 0.42–0.55 (10× over 0.05 gate), uniform across every text and bucket. 1D iSTFT magnitude path is exquisitely sensitive to per-channel activation quantization noise on the upsample Conv stack. Structural blocker — would need quantization-aware fine-tuning or a different vocoder. **Spec's 10-25% TTFA improvement target is not achievable at INT8 with this vocoder architecture.** | `kokoro-rk-tail-rest-int8-static.md` §3 |

### Parked (low ROI / high risk / quality-not-perf)

| Path | Reason | If revisited |
|---|---|---|
| INT8 tail-rest (generator-rest-preexp historical route) | Calibration data required; vocoder INT8 historically fails audio gate; payoff unclear | Needs explicit user authorization + a fresh audio-gate first design |
| Pipeline parallelism (spec §7 overlap) | Already at sentence-level pipelining in HTTP layer; deeper overlap requires model architecture changes | Outside Kokoro RKNN scope |
| Smaller model (Kokoro nano) | Model selection decision, not perf optimization | User strategic choice |
| bucket-64/128 long-sentence support | Quality (truncation fix), not perf; bucket-64 vocoder time-dim 9560 > 8191 NPU limit, needs CPU vocoder fallback | When long-sentence quality becomes a priority |

## Why TTFA cannot drop below current numbers (hardware bound)

For each bucket, TTFA equals sentence-0 wall-time of the 4-stage pipeline. Within that:

- prefix + decoder-front: small (~50ms total across all buckets)
- vocoder-front-half RKNN FP16: bucket-scaled, dominated by Sin/InstanceNorm/AdaIN at fixed compute density
- **tail-rest CPU ORT: ~71% of wall time for bucket-32, ConvTranspose-heavy and not RKNPU-friendly at this output time-dim**

To break below the current floor would require either:

1. **Smaller model** (architectural change, not perf optimization)
2. **Sub-sentence streaming** (single-call audio chunking inside vocoder-front or tail-rest, requires ONNX re-export + overlap-add — high engineering cost)
3. **INT8 tail-rest with audio gate validation** (uncertain payoff)
4. **Different vocoder architecture** without dilated ConvTranspose

All are R&D commitments significantly larger than the optimizations already delivered.

## Side effects discovered & resolved

- **Production image `openvoicestream:rk-kokoro-2026-05-23` was built 5 min before commit 6155ebe** — shipped with broken `app/backends/rk/tts.py` (missing `speaker_id` pop). Hot-patched into running containers; resolved by image rebuild 2026-05-23 (`openvoicestream:rk-kokoro-2026-05-23-rebuilt`, commit f3a0edc + this rebuild commit).
- **bind-mount deployment → resolved by image rebuild 2026-05-23**: misaki + bucket artifacts + `tokens.txt` are now baked into the rebuilt image. `docker compose --force-recreate` no longer drops them. See `docs/runbooks/kokoro-rk-deploy.md`.

## Files of record

Documentation chain (all committed to main):

1. `kokoro-rk-npu-42pct.md` (7bd7228, target spec, revised in bd2b053)
2. `kokoro-rk-42pct-m1-boundary-report.md` (e50dc7b)
3. `kokoro-rk-42pct-m2-bert-fp16.md` (50463cb)
4. `kokoro-bert-ab-audio-report.md` (3b18517) — BERT dead-code decisive evidence
5. `kokoro-rk-42pct-m4-vocoder-fp16.md` (add7ddf)
6. `kokoro-rk-34pct-m4m6-final.md` (f482832)
7. `kokoro-rk-34pct-rtf-reconciliation.md` (654ec72)
8. `kokoro-prod-image-stale-2026-05-23.md` (f3a0edc)
9. `kokoro-rk-34pct-http-rtf-final.md` (c7302f9)
10. `kokoro-rk-zh-mid-sentence-silent-fail-diag.md` (afd834f)
11. `kokoro-rk-zh-fix-misaki.md` (e2665d0)
12. `kokoro-rk-streaming.md` (d014370)
13. `kokoro-rk-bucket8-ttfa.md` (735b5e9)
14. `kokoro-rk-bucket16-mid-ttfa.md` (ca9332c)
15. `kokoro-rk-34pct-reproduction-guide.md` (0ab706b, updated 05aca9d)
16. `docs/specs/README.md` (0ab706b)
17. **This doc (closure)**

Submodule `third_party/rkvoice-stream` final state: branch `feat/kokoro-rk-4stage-vocoder-front`, HEAD `65b9a13` (post P7b bucket-8 static MM+Gemm INT8 + bucket-16 router + misaki ZH G2P).

Main repo final state: HEAD `93a279b` (post image rebuild + ZH regression fix + scripts/p7b reproduction). Subsequent doc-polish commits update closure refs; pull main for latest.

## Next user-decision gate

If perf R&D should resume, the only paths with non-zero potential are:

1. **INT8 tail-rest experiment** (need audio gate validated calibration data — high R&D effort, uncertain payoff)
2. **Sub-sentence streaming** (ONNX re-export of vocoder with frame-level inputs — significant model engineering)
3. **Pivot to a different TTS model** (Kokoro nano or successor) — out of scope of this Kokoro RK pipeline

All require explicit user kickoff. R&D for the current spec is closed.
