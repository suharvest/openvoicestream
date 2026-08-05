# TensorRT Edge-LLM v0.9.1 model and concurrency validation matrix

Status: v0.9.1 release and production concurrency gates complete; final Orin
NX production profile rechecked 2026-08-05.

The inventory sections below preserve the pre-migration baseline. Subsequent
sections supersede that baseline with fresh v0.9.1 engine, quality,
concurrency, cancellation, and co-residency evidence.

The final deployed profile is Qwen3-ASR + Matcha + Qwen3.5-4B GDN/MTP 8K
(4K is an optional payload using the same runtime image). Its production
contract is stable N=1 per pipeline with model-specific higher concurrency
only where `/v1/capabilities` advertises it. A synchronized ASR + LLM + TTS
overlap run completed in 0.358 / 1.592 / 0.380 seconds respectively; both
containers remained healthy with zero restarts and no OOM. The final cold
start and rollback evidence is in
`docs/validation/edgellm-v091-release-checkpoint-20260803.md`.

## Scope and result vocabulary

The upgrade gate covers every Edge-LLM-dependent model found in the repository
or on `orin-nx`, plus SenseVoiceSmall because it is an explicit regression
requirement even though it uses standalone TensorRT rather than Edge-LLM.

- `available-reference`: an old v0.8.0/v0.9.0 artifact exists and may be used
  only as a provenance, quality, or performance reference.
- `missing-v0.9.1`: a clean engine/plugin/worker built from official v0.9.1 is
  not yet present.
- `blocked`: a required sidecar, harness, or model artifact is absent.
- `pending`: the runnable gate is known but has not been run.
- `pass` may be written only after raw logs and machine-readable results exist
  under the v0.9.1 evidence root.

Old TensorRT engines are not runtime-compatible evidence for v0.9.1. Every
v0.9.1 row requires a fresh engine and worker/plugin provenance record.

## Device inventory

### GDN / LLM

| Variant | Existing asset on `orin-nx` | Concurrency encoded by asset | Inventory result |
|---|---|---:|---|
| Qwen3.5-4B AWQ, vanilla GDN | `/home/harvest/edgellm-workspace/qwen35-4b-awq/engines-v080-gdn` (3.3 GiB: `llm.engine`, embeddings, external INT4 FFN weights) | `max_batch_size=1` | `available-reference`; current stopped service uses this path |
| Qwen3.5-4B AWQ, GDN+MTP | Historical scripts point to `/home/harvest/edgellm-workspace/qwen35-4b-awq/engines-v080-gdn-mtp-t4` | historical T4 draft settings, not current | `blocked`: directory is absent |
| HF source | `/home/harvest/edgellm-workspace/qwen35-4b-awq/hf_src/model.safetensors` (4.58 GB) | N/A | reusable read-only source |

The stopped container is
`edge-llm-chat-service:v0.8.0-gdn-mtp-merged`, but its active `.env` explicitly
says `vanilla GDN (NO MTP spec-decode)` and sets only
`EDGELLM_ENGINE_DIR=/workspace/qwen35-4b-awq/engines-v080-gdn`. The image
contains the runtime toggle, but no configured or surviving draft engine.
Therefore the name `gdn-mtp-merged` must not be treated as proof that the
deployed service was running MTP.

### Voice models

| Model / variant | Existing reference assets | Encoded ceiling | Inventory result |
|---|---|---:|---|
| Qwen3-ASR 0.6B INT4, b1 | `/home/harvest/project/v090-engines/qwen3-asr/llm`, shared audio encoder under `qwen3-asr/audio/audio` | 1 | `available-reference` |
| Qwen3-ASR 0.6B INT4, b2 | `/home/harvest/project/v090-engines/qwen3-asr-b2/llm` (`llm.engine` 555 MB) plus shared audio encoder (381 MB) | 2 | `available-reference`; intended v0.9.1 max gate is N=2 |
| Qwen3-TTS CustomVoice INT4 | `/home/harvest/project/v090-engines/qwen3-tts-customvoice/{talker,code_predictor,code2wav}` | talker batch 2 | `available-reference`; nine named speakers/language rows in config |
| Qwen3-TTS Base FP16 | `/home/harvest/project/v090-engines/qwen3-tts-base/{talker,code_predictor,code2wav}` plus `speaker_encoder.fp16.engine` | talker batch 2 | `available-reference`; external speaker embedding/voice-clone path |
| SparkTTS 0.5B BF16 | `/home/harvest/project/v090-engines/sparktts/llm_bf16` plus shared BiCodec and speaker decoder | 2 | `available-reference` |
| SparkTTS 0.5B W4A16 | `/home/harvest/project/v090-engines/sparktts/llm_w4a16` plus shared BiCodec and speaker decoder | 2 | `available-reference`; repository default |
| MOSS-TTS-Nano FP16 TRT | `/home/harvest/moss-mix1-bundle/engines` (five TTS plans, shared data, tokenizer) | worker/profile ceiling 2 | `blocked`: no `codec_decode_step.plan` exists anywhere under `/home/harvest`; codec ONNX sidecars exist separately under `/home/harvest/moss-onnx-bundle-trtfix` |
| SenseVoiceSmall FP16 TRT | profile/leaf exists; engine is supposed to be generated from `sense-voice-encoder.scaled.fixed.onnx` | 1 | `blocked`: neither ONNX nor `sensevoice.plan` was found on `orin-nx`; also not an Edge-LLM worker |

Relevant v0.9.0 worker references exist in
`/home/harvest/project/v090-bake/opt/bin/` and
`/home/harvest/project/v090-opt/bin/`, including
`qwen3_asr_worker`, `qwen3_tts_streaming_worker`, `spark_tts_worker`, and
`moss_tts_nano_worker`. They are reference binaries only and must not be mixed
with a v0.9.1 plugin.

The vendored worker source additionally builds
`native/edgellm_voice_worker/qwen3_tts_worker.cpp`. It is an older Qwen TTS
entrypoint, not another model family. The v0.9.x product profiles select the
upstream-style `qwen3_tts_streaming_worker`; the divergence audit must classify
the older binary as retained compatibility, obsolete, or still required rather
than silently testing both and double-counting one model.

The repository has a chronology conflict that the new gate must resolve:
`BENCHMARKS.md` (2026-07-04) calls CustomVoice N=1 by design, while
`docs/deploy-v090.md` (2026-07-07) and
`configs/profiles/jetson-edgellm-v090-n2.json` claim a later N=2 CustomVoice
deployment. Treat N=2 as required but unproven on v0.9.1.

## Existing corpus, goldens, and harnesses

### Reusable data and evidence

- General ASR/TTS corpus and SHA-locked manifests:
  `bench/perf/corpus/`.
- General historical comparison gate:
  `bench/perf/baselines/baseline.json` and `bench/perf/gate.py`.
- Qwen3 TTS mel/PCM goldens:
  `third_party/jetson-voice-engine/tests/golden_mels/`.
- Remote CustomVoice v0.8 WAV golden:
  `/home/harvest/tensorrt-edgellm-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/golden-v080/customvoice_en_golden.wav`.
- Remote v0.9 voice samples:
  `/home/harvest/project/v090-engines/out-audio/`.
- Remote SparkTTS W4A16 samples and ASR result:
  `/home/harvest/sparktts-quant/int4_wavs/` and
  `/home/harvest/sparktts-quant/asr_result_int4.json`.
- Remote GDN/MTP historical logs:
  `/home/harvest/asr_v080_e2e/`, `/home/harvest/mtp-sweep/`, and
  `/home/harvest/mtp-t4-progress/`.

Historical output is a comparison reference, not a v0.9.1 pass.

### Harness readiness

| Harness | What it proves | v0.9.1 readiness |
|---|---|---|
| `bench/perf/gdn_sse_abort_recovery.py` | 50-cycle SSE abort, immediate next request, health and interval-log scan | checked in and run 50/50 against the v0.9.1 audit server; stability passed, but recovery TTFT remained +354 ms and no server-side cancellation acknowledgement exists |
| `bench/perf/qwen_asr_n2_service.py` | distinct-WAV/language N=2, recovery and N+1 admission | executed on the fresh v0.9.1 ASR service: 50/50 pairs and recovery passed with exact transcripts; the third request queues instead of returning the desired 4429 admission response |
| `bench/perf/tts_n2_cancel_isolation.py` | cancel A while B completes, output isolation and immediate recovery | executed for the qualified Base, CustomVoice, Spark, and MOSS paths; model-specific outcomes are recorded below, while the final Matcha profile exposes its own limit through `/v1/capabilities` |
| `bench/perf/v2v_concurrency_probe.py` | simultaneous `/v2v/stream` ASR windows and N+1 admission | runnable through service; uses one WAV for all clients, so it does not prove zh/en isolation |
| `bench/asr_n2_streaming.py` | bare-worker N=2, distinct WAVs, saturation, slot recovery | not directly runnable on v0.9.x: it requires the old mel-settings/mel-filters worker CLI |
| `bench/tts_n2_harness.py` and `_lang.py` | bare Qwen TTS N=2, overlap, saturation, PCM capture | hard-coded to v0.7 paths; must be parameterized before v0.9.1 use |
| `bench/tts_n2_seq_check.py` | sequential control for concurrency-only corruption | same hard-coded path limitation |
| `bench/tts_n2_asr_loopback.sh` | ASR intelligibility check on N=2 WAVs | usable after the producer paths are updated |
| `bench/perf/stability_tts_n2_common.py` | HTTP N=1 baseline, N=2 burst, pre/post PCM MD5, CUDA log scan | used by the qualified model-specific gates; retained as the common HTTP regression harness |
| `bench/perf/stress_cancel_n1.py` | repeated client disconnect/cancel recovery | N=1 only; does not prove cancellation isolation |
| `bench/perf/stress_moss_tts_n2.py` | MOSS N=2 basic, burst, parity, mixed-length | executed with the complete six-plan MOSS stack; N=2 and 50-cycle cancel/recovery passed |
| `bench/perf/smoke_moss_tts_backend.py` | MOSS preload/stream/shutdown | executed against the complete v0.9.1 MOSS backend |

The remaining items are optimization or API-semantics work, not blockers for
the qualified production profile:

1. Native simultaneous GDN contexts still reproduce the Myelin
   `already loaded binary graph` failure; production intentionally uses the
   guarded N=1/singleflight path.
2. The ASR N+1 admission path queues the third request instead of returning
   the desired 4429 response, although N=2 execution and recovery pass.
3. The final ASR + Matcha + LLM overlap gate is complete. A translator-inclusive
   four-service orchestrator remains optional coverage for a different profile.

## Per-model acceptance matrix

All tests use deterministic seeds where supported, record absolute start/end
times to prove overlap, and scan the full test interval for `CUDA`, `TensorRT`,
`Myelin`, assertion, illegal-memory-access, and worker-exit errors.

| Model | N=1 gate | N=max gate | Cancel isolation | 50-shot gate | Quality gate | Current state |
|---|---|---|---|---|---|---|
| Qwen3.5 GDN | deterministic + sampled + 512-token direct runs pass at 35.10-39.24 tok/s; 20/20 sequential server requests pass | **FAIL**: two simultaneous clients reproduce Myelin already-loaded graph; one receives no content | 50/50 abort -> immediate-next cycles pass, but recovery TTFT is +354 ms vs identical idle prompt and server has no explicit disconnect cancel | PASS for serial abort stability; zero interval-log hits | response schema/content sane; direct throughput matches/exceeds v0.8 ~35 tok/s | `partial`: N=1/direct/serial pass; simultaneous server concurrency fails |
| Qwen3.5 GDN+MTP | fresh base/draft engines pass and expose draft counters | wrapper serves both clients safely through singleflight | same 50-cycle recovery gate | 50/50 recovery | max 3/3 drafts accepted; +22.98% throughput, +20.9 MiB | `pass` for fresh engine and guarded service; native simultaneous Myelin limitation remains guarded |
| Qwen3-ASR INT4 | fresh direct worker exact on three Chinese samples | N=2 actual `/asr` service, distinct inputs and positive overlap | immediate post-pair recovery | 50/50 pairs and 50/50 recovery, zero cross-talk/errors | all 100 concurrent transcripts exact | `partial`: engine/worker/service N=2 pass; third request queues instead of required 4429 admission response |
| Qwen3-TTS CustomVoice INT4 | fresh official-v0.9.1-exported talker, named speakers, ZH/EN ASR loopback | N=2 shared-engine output interleaves and matches solo PCM | cancel tripped; immediate recovery byte-identical | 50/50 N=2 pairs, every pair interleaved and isolated | four ZH/EN SenseVoice roundtrips similarity 1.0; non-silent, unclipped | `pass` for fresh CuTe engine/worker N=2; service-layer N+1 admission gate remains |
| Qwen3-TTS Base INT4 | fresh external embedding from fixed speaker encoder; 3 ZH + 2 EN strict ASR roundtrips | N=2 overlaps; both PCM digests match their solo baselines | cancel A, continue B, immediate B recovery; byte-identical | 50/50 uninterrupted rounds; zero CUDA/TRT hits | reference/output speaker-encoder cosine 0.9328-0.9475; production 512-frame Code2Wav passed 10 alternating ASR/TTS swaps and 20 overlapping GDN requests with restart `0→0` | `pass`; independent N=2 remains isolated, while production uses N=1 plus exclusive ASR/TTS residency swapping and the default 512-frame engine alongside active GDN |
| SparkTTS BF16 | fresh v0.9.1 CuTe engine; controllable + clone ZH/EN pass | true N=2 interleaving; deterministic A matches solo PCM | cancel A, keep B, immediate recovery | 50/50 rounds, all keep/recovery PCM byte-identical; worker rc=0 | four ZH/EN SenseVoice roundtrips similarity 1.0; TTFA/RTF ZH 0.700/0.767, EN 0.641/0.762 | `pass`; fresh export, engine, worker, N=2, cancellation, recovery, clone, and semantic gates |
| SparkTTS W4A16 | fresh v0.9.1 CuTe engine; controllable + clone ZH/EN pass | true N=2 interleaving; deterministic A matches solo PCM | cancel A, keep B, immediate recovery | 50/50 rounds, all keep/recovery PCM byte-identical; worker rc=0 | four ZH/EN SenseVoice roundtrips similarity 1.0; TTFA/RTF ZH 0.455/0.507, EN 0.412/0.494 | `pass`; preferred production Spark variant; product precision extension remains local debt |
| MOSS-TTS-Nano TRT | fresh six-plan stack, three prompts | local extended worker shows true N=2 overlap and distinct request output | cancel A observed after one chunk while B continues; immediate recovery passes | 50/50 N=2 cancel/recovery rounds, worker rc=0, zero CUDA hits | concurrent pair non-silent/distinct; SenseVoice similarity 0.9565/0.9545; three solo samples exact | `pass` with local dispatcher/cancel extension; unextended NVIDIA tree has no MOSS model target and is not a bug-PR candidate |
| SenseVoiceSmall TRT | fresh 492,466,348-byte FP16 plan, finite logits | N=1 only | N/A | not yet full corpus | exact transcripts on fresh MOSS samples and CustomVoice ZH/EN roundtrips | `partial`: source and plan recovered; targeted quality pass, full 50-file corpus remains |

For every N=2 TTS row, the hard gates are:

1. each concurrent PCM is equal to its own solo reference when the worker is
   deterministic, otherwise it meets the same ASR/speaker-quality threshold;
2. request IDs and output bytes never cross;
3. A and B inference windows overlap;
4. cancellation of A cannot truncate, corrupt, or delay B beyond the declared
   ratio;
5. the next request succeeds after the pair;
6. 50 paired rounds produce zero CUDA/TRT/assertion errors;
7. peak unified memory and second-slot incremental memory are recorded;
8. shared-engine memory saving is compared with the prior approximately
   1284 MB result.

## Runnable command templates

Run only after the parent task has assigned a unique v0.9.1 service/container
and evidence root. Do not reuse the production container name or old engine
directories.

```bash
cd /home/harvest/project/seeed-local-voice
export V091_EVIDENCE=/home/harvest/validation/edgellm-v091-official-20260724T0418Z/model-concurrency
export VOICE_URL=http://127.0.0.1:8621
export VOICE_CONTAINER=seeed-voice-v091-audit
mkdir -p "$V091_EVIDENCE"
```

The directory creation above is a future execution command, not an action
performed by this inventory task.

### Service-level Qwen ASR overlap and admission

```bash
python3 bench/perf/v2v_concurrency_probe.py \
  --url ws://127.0.0.1:8621 \
  --wav bench/perf/corpus/short/zh_short_01.wav \
  --n 2 --label qwen3-asr-v091-n2 \
  >"$V091_EVIDENCE/qwen3-asr-n2-overlap.log" 2>&1

python3 bench/perf/v2v_concurrency_probe.py \
  --url ws://127.0.0.1:8621 \
  --wav bench/perf/corpus/short/zh_short_01.wav \
  --n 3 --label qwen3-asr-v091-oversubscribe \
  >"$V091_EVIDENCE/qwen3-asr-n3-admission.log" 2>&1
```

The service-level distinct-WAV gate has no dependency on the historical
mel-settings/mel-filters worker CLI:

```bash
python3 bench/perf/qwen_asr_n2_service.py \
  --base-url "$VOICE_URL" \
  --wav-a bench/perf/corpus/short/zh_short_01.wav \
  --language-a Chinese --expect-a '震惊|母亲' \
  --wav-b bench/perf/corpus/short/en_short_01.wav \
  --language-b English --expect-b 'white smoke|plant' \
  --rounds 50 --check-oversubscribe \
  --server-log "$V091_EVIDENCE/qwen3-asr/cute/server-interval.log" \
  --output "$V091_EVIDENCE/qwen3-asr/cute/n2-distinct-wav.json"
```

Repeat the command against the fallback service and write under
`qwen3-asr/fallback/`. It fails unless both transcripts match their own
language row, both HTTP request windows overlap, the immediate recovery
request succeeds after every pair, at least one request in the final N=3
probe is rejected with 429/4429, and the supplied interval log is clean. HTTP
window overlap is admission evidence; retain worker timestamps or trace logs
alongside this JSON when claiming true GPU compute overlap.

### Generic HTTP N=2 TTS stability gate

Qwen CustomVoice example:

```bash
python3 -c 'from bench.perf.stability_tts_n2_common import main_entry; raise SystemExit(main_entry("qwen3_tts_customvoice", ("trt_edge_llm","qwen3"), "OVS_TTS_WORKER_CONCURRENCY", None))' \
  --base-url "$VOICE_URL" --bursts 50 --warmup 3 \
  --container "$VOICE_CONTAINER" \
  --output-dir "$V091_EVIDENCE/qwen3-tts-customvoice"
```

Use the same command with:

- Base: label `qwen3_tts_base`, expected substrings
  `("trt_edge_llm","qwen3")`;
- SparkTTS: label `sparktts`, expected substrings `("spark",)`;
- MOSS: label `moss_tts_nano`, expected substrings `("moss",)`.

This common gate covers PCM stability and 50 paired bursts, but not
cancel-one/continue-one. Run the checked-in cancellation gate for every
concurrent TTS row:

```bash
python3 bench/perf/tts_n2_cancel_isolation.py \
  --base-url "$VOICE_URL" --rounds 50 \
  --payload-a-json '{"speaker":"vivian","language":"Chinese"}' \
  --payload-b-json '{"speaker":"serena","language":"English"}' \
  --server-log "$V091_EVIDENCE/qwen3-tts-customvoice/cute/server-interval.log" \
  --capture-dir "$V091_EVIDENCE/qwen3-tts-customvoice/cute/cancel-bodies" \
  --output "$V091_EVIDENCE/qwen3-tts-customvoice/cute/cancel-isolation.json"
```

For Qwen Base, SparkTTS, and MOSS, keep the command and replace only the
model-specific payload fields and evidence path. Use
`--no-require-byte-equal` only for a backend that is intentionally
nondeterministic and pair the JSON with its ASR/speaker-quality result. The
gate always captures all of B, requires A's close to occur before B completes,
then executes and validates an immediate full recovery request. Repeat it in
both CuTe and fallback columns.

### MOSS standalone gates

After staging the missing codec plan and sidecars in an isolated v0.9.1 model
root:

```bash
MOSS_MAX_SLOTS=2 python3 bench/perf/stress_moss_tts_n2.py \
  --mode basic --max-slots 2
MOSS_MAX_SLOTS=2 python3 bench/perf/stress_moss_tts_n2.py \
  --mode parity --max-slots 2
MOSS_MAX_SLOTS=2 python3 bench/perf/stress_moss_tts_n2.py \
  --mode mixed --max-slots 2
MOSS_MAX_SLOTS=2 python3 bench/perf/stress_moss_tts_n2.py \
  --mode burst --rounds 50 --max-slots 2 \
  >"$V091_EVIDENCE/moss-n2-50.log" 2>&1
```

### Mixed-service gate

With the unique v0.9.1 voice container, v0.9.1 LLM audit service, and restored
translator all healthy, launch one request to each path in the same shell
interval:

```bash
curl --fail --silent --show-error \
  -F audio=@bench/perf/corpus/short/zh_short_01.wav \
  "$VOICE_URL/asr" >"$V091_EVIDENCE/mixed-asr.json" &
curl --fail --silent --show-error \
  -H 'content-type: application/json' -d '{"text":"并发语音验证"}' \
  "$VOICE_URL/tts/stream" >"$V091_EVIDENCE/mixed-tts.bin" &
curl --fail --silent --show-error \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-4B-AWQ","messages":[{"role":"user","content":"只回答：并发正常"}],"max_tokens":16}' \
  http://127.0.0.1:8000/v1/chat/completions \
  >"$V091_EVIDENCE/mixed-llm.json" &
curl --fail --silent --show-error \
  -H 'content-type: application/json' \
  -d '{"text":"你好世界","src_lang":"zho_Hans","tgt_lang":"eng_Latn"}' \
  http://127.0.0.1:9001/translate \
  >"$V091_EVIDENCE/mixed-translator.json" &
wait
```

Wall-clock timestamps, `tegrastats`, health before/after, and container logs
must be captured around this command. Successful HTTP responses without
recorded overlap and memory headroom do not pass the mixed-service gate.

### GDN commands

The isolated v0.9.1 audit used the exact fresh engine, plugin, and pybind under
`/home/harvest/validation/edgellm-v091-official-20260724T0418Z`. Because the
official CLI exposes only required `--model` even though the API supports
prebuilt engines, `125-gdn-v091-audit-server-no-preload.py` instantiated
official `LLM(engine_dir=...)` and bound only `127.0.0.1:18091`.

The executed abort gate was:

```bash
python3 bench/perf/gdn_sse_abort_recovery.py \
  --base-url http://127.0.0.1:18091 \
  --model v091-official-gdn-audit \
  --rounds 50 --abort-after-events 1 \
  --abort-max-tokens 256 --recovery-max-tokens 16 \
  --timeout 120 \
  --max-next-delay-ms 100 \
  --server-log \
    /home/harvest/validation/edgellm-v091-official-20260724T0418Z/131-abort-server.log \
  --output \
    /home/harvest/validation/edgellm-v091-official-20260724T0418Z/133-abort50-result.json
```

Result: 50/50 rounds passed, every health check returned 200, immediate-next
delay was 0.012-0.026 ms, and the mandatory interval-log scan had zero hits.
Recovery TTFT was 420.77-423.73 ms (mean 421.09 ms), versus 66.60 ms for ten
idle runs of the identical prompt. Since official `generate_stream()` has no
consumer-disconnect `channel.cancel()` and only joins its worker for up to five
seconds, this is a stability pass but not proof of prompt cancellation or
immediate compute release.

The harness explicitly closes the raw SSE response, records the delay before
the next request starts, consumes that recovery stream through `[DONE]`, and
checks health after every cycle. It fails for missing interval logs or
CUDA/Myelin/TensorRT/assertion/worker-exit signatures. Do not substitute a
hand-written `curl | head` loop: it cannot reliably prove the client
disconnected, that the next request started immediately, or that all 50
cycles remained healthy. Server-side request lifecycle timestamps in
`server.log` remain the evidence that the aborted generation unwound; the
client JSON alone cannot observe that internal event.

The GDN server row does **not** pass concurrency. Twenty sequential requests
passed, but two simultaneously started clients reproduced
`Myelin ... Called with an already loaded binary graph`; one client completed
while the other received HTTP 200 plus `[DONE]` with zero content. The server
remained healthy and accepted a subsequent request, so this is a shared
runtime/context concurrency defect.

GDN+MTP is no longer blocked. Fresh `spec_base.engine` and
`spec_draft.engine` artifacts generated `2 plus 2 equals 4` and improved the
measured short-prompt generation throughput from 39.24 to 48.26 tok/s
(+22.98%). Its isolated HTTP wrapper passed two clients through the deliberate
single-flight gate and 50 abort/recovery cycles. GDN/MTP also overlaps safely
with Base TTS N=1. It must not run concurrently with both Base TTS slots:
GDN itself completed correctly in that experiment, but one TTS cancel/keep
round exceeded 45 seconds.

Mixed service evidence is positive for the tested pair: the v0.9.1 audit
server and original translator completed simultaneous functional requests
with 835.57 ms overlap. Peak RAM was 9,058/15,656 MiB, leaving 6,598 MiB
headroom, and both health endpoints remained healthy. The audit server was
then stopped and the original services were restored for the remainder of the
hardware audit. On 2026-07-25 the final production cutover selected the same
validated v0.9.1 GDN+MTP image and engines on port 8000 while retaining the
translator on port 9001. Warmup, real completion, translator health, and the
two-client single-flight gate all passed.

## CuTe versus fallback matrix

CuTe is a build/runtime axis, not a GDN-only microbenchmark. Each supported
voice row must be exercised under both columns before retiring the local
fallback patches.

| Gate | Official CuTe column | Fallback column |
|---|---|---|
| Provenance | exact official v0.9.1 SHA; record packaged artifact hash, architecture, and build CUDA | exact v0.9.1 plus only the explicitly retained fallback patch set |
| Build | `ENABLE_CUTE_DSL=ON`; first try official sm_87 package, then clean local sm_87 generation if toolkit-incompatible | `ENABLE_CUTE_DSL=OFF`; no stale CuTe artifact may be loaded |
| Load/link | plugin and every required worker load with no CUDA symbol/toolkit mismatch | Qwen/Spark workers link and load the tiled FP16/GEMV path |
| Correctness | same model-specific solo/concurrent quality gates | same gates against the same prompts/seeds |
| Concurrency | N=max, cancellation isolation, 50 paired rounds | identical workload |
| Performance | TTFT/TTFA, RTF/tok/s, peak unified memory | same metrics and deltas |
| GPU fbank | verify Qwen audio online GPU fbank only when official build reports the feature enabled | mark not applicable with `ENABLE_CUTE_DSL=OFF`, not failed |

Required rows in both columns:

- Qwen3.5 GDN vanilla and, once rebuilt, GDN+MTP;
- Qwen3-ASR INT4 N=1/N=2;
- Qwen3-TTS CustomVoice INT4 N=1/N=2;
- Qwen3-TTS Base INT4 N=1/N=2;
- SparkTTS BF16 and W4A16 N=1/N=2.

MOSS and SenseVoice should still run regression/coexistence gates, but a
feature that does not call the Edge-LLM CuTe path is recorded `not applicable`
with static call-path evidence.

Retire local patches `v090-sparktts-0001..0003` only if the official CuTe
column passes the complete matrix above. A packaged-artifact load test, CuTe
microbenchmark, or GDN-only success is insufficient.

## Evidence layout and closeout checklist

```text
/home/harvest/validation/edgellm-v091-official-20260724T0418Z/
  provenance/
  gdn/
  qwen3-asr/{cute,fallback}/
  qwen3-tts-customvoice/{cute,fallback}/
  qwen3-tts-base/{cute,fallback}/
  sparktts/{bf16,w4a16}/{cute,fallback}/
  moss/
  sensevoice/
  mixed-services/
```

Each leaf must contain:

- exact command and environment allowlist;
- source SHA, dirty status, build flags, plugin/worker/engine hashes;
- engine config and TensorRT/CUDA runtime provenance;
- request inputs, seeds, start/end timestamps, responses or PCM hashes;
- health before and after;
- full log scan result;
- `tegrastats` raw log and summarized peak/headroom;
- a machine-readable verdict that names every failed or not-applicable gate.

Closeout result:

1. Fresh GDN+MTP engines passed direct, HTTP, abort/recovery, performance, and
   production-cutover gates.
2. Fresh MOSS and SenseVoice engines passed quality regression.
3. Fresh ASR passed b1/b2 WAV-ingest, distinct-language N=2, and 50-round
   service gates.
4. CustomVoice INT4, Base INT4, and fresh SparkTTS BF16/W4A16 passed N=2
   isolation, cancellation, immediate recovery, and uninterrupted 50/50
   rounds. Spark also passed controllable/clone and four strict semantic
   roundtrips.
5. The original MOSS loop was conclusively bounded to N=1. The local
   dispatcher/cancel extension subsequently passed 50/50 true N=2 rounds and
   is now the required production worker for this model.
6. ASR/TTS residency swapping, LLM+Base N=1, and LLM+translator coexistence
   passed. The production 512-frame Code2Wav gate completed 10/10 swaps and
   20 overlapping GDN requests without a restart or OOM. The failed
   LLM+Base N=2 experiment is retained as a scheduling boundary.
   `jetson-edgellm-v091-qwen3ttsbase-isolated-n2` exposes the validated Base
   N=2 capacity only when GDN is stopped; the normal Base production profile
   intentionally remains N=1.
7. The v0.9.1 CuTe release path is accepted. Fallback tests remain retirement
   gates for individual fallback patches, not blockers for the CuTe production
   cutover.
8. Production ports 8000/8621/9001 are healthy with v0.9.1 GDN+MTP, the
   default-512 voice runtime, and the existing translator. The v0.8
   compose/image remain the rollback path.
