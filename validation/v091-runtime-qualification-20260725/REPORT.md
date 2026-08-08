# TensorRT Edge-LLM v0.9.1 normalized runtime qualification

Date: 2026-07-25/26  
Device: Orin NX, JetPack 6.2, CUDA 12.6, TensorRT 10.3, SM87  
Candidate image:
`seeed-local-voice:v0.9.1-edgellm-runtime-normalized-d52d973-20260725`
(`sha256:13f8b69ed37ad1238afdb0116b003d36e0a32102555aa0c0f636e168b42222d9`)

## r2 qualification update (2026-07-25/26)

The rebuilt r2 runtime image is
`seeed-local-voice:v0.9.1-edgellm-runtime-r2-20260725`
(`sha256:74c34eb765223c6e0a59c72d2215a19e030dfc852f6820504e6d6fe575f14938`).
The current source heads are outer
`021112eda3207a57ae91056f24d198303574b555`, engine overlay
`4b28dd24bcf5c5240f604b7d6be78ad39d9c5e2f`, and NVIDIA upstream
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`.

- The release artifact staging gate passed with 171 payload files and
  19,711,158,973 bytes. All 29 engine sidecars and all 62 deploy-required
  paths passed. `published_to_hf` remains false.
- The preceding manifest was stale: it omitted the 12 required engine,
  sidecar, and config paths for Base b1/b2 KV1536 even though those files
  already existed. The r2 finalizer now inventories them.
- Missing-required-file and stale-sidecar negative fixtures both failed
  closed. The independent `SHA256SUMS -c` pass succeeded.
- The old artifact set's manifest, SHA envelope, provenance, and MOSS worker
  hashes are unchanged. The corresponding r2 control files and worker have
  independent inodes. Same-filesystem hard links limited incremental disk
  allocation to 765,952 bytes while preserving the old rollback set.
- Base N=1 passed ASR true streaming 3/3, TTS HTTP streaming 3/3, and strict
  cancel-to-recovery 20/20. ASR first partial was 1001.4-1001.6 ms before EOS;
  TTS TTFA was 2195.0-2270.0 ms with RTF 0.610-0.631; cancellation recovery
  was 2140.2-2146.9 ms with non-empty 24 kHz PCM.
- With the original Qwen3.5-4B GDN/MTP service resident, the product N=1
  sequence ASR true-streaming -> GDN SSE -> TTS passed 10/10. Total latency
  was 6713.2-6773.5 ms; GDN TTFT was 70.2-84.5 ms; TTS TTFA was
  2099.6-2127.7 ms.
- Valid pairwise overlap passed: ASR/GDN 3/3 with 135.2-135.5 ms request
  overlap, and TTS/GDN 3/3 with 283.3-283.9 ms overlap. Both containers
  remained healthy with restart count 0 and no CUDA, TensorRT, or Myelin
  error hits.
- Three independent ASR + TTS + GDN requests are not admitted by the Base
  N=1 profile. ASR occupied the aggregate speech session and TTS returned
  HTTP 429 with `Retry-After: 5` in 3/3 attempts. This is the configured
  aggregate limiter, not a CUDA crash. The earlier pairwise co-residency test
  must not be described as true three-way request concurrency.
- During the product N=1 co-residency gate, whole-system RAM peaked at
  15,212/15,656 MB, swap at 1,761/7,828 MB, and GR3D at 99%. This is stable
  but leaves little physical-memory headroom.
- Base isolated N=2 passed with the actual ASR and TTS workers both using the
  b2 KV1536 engines and `--max_slots 2`. ASR true-streaming two-lane isolation
  passed 3/3 with 5.084-5.091 s overlap; both lanes emitted partials before
  EOS and returned their own correct final transcript.
- Base isolated TTS N=2 passed 3/3 with valid 24 kHz non-empty PCM, distinct
  lane hashes, and true overlap. The lazy-load round had about 11.53 s TTFA;
  the two hot rounds had 587.7-596.4 ms TTFA, 4.186-4.211 s overlap, and
  approximately 1.012-1.032 RTF.
- Base isolated cancellation passed 20/20. Every keep lane completed 288,000
  PCM bytes while the other lane was disconnected. Recovery completed in
  4.954-4.978 s, always before the strict 15 s deadline, and overlapped the
  uninterrupted keep lane by 4.826-4.880 s.
- The isolated N=2 container remained healthy with restart count 0 and no
  CUDA/TensorRT/Myelin errors. Whole-system RAM peaked at 12,610/15,656 MB,
  swap at 659 MB, and GR3D at 99%; the fully resident container settled at
  about 3.96 GiB.
- CustomVoice N=1 core synthesis passed with three distinct built-in voices
  (Serena, Ryan, and Ono Anna) across Chinese and English. All outputs were
  non-empty 24 kHz PCM with distinct hashes. Hot TTFA was
  2346.7-2352.1 ms and RTF was 0.585-0.589. Disconnect followed by recovery
  passed 10/10 in 3006.4-3014.7 ms. The real worker used the CustomVoice
  talker/code-predictor engines without `--max_slots 2`.
- The CustomVoice N=1 container remained healthy with restart count 0.
  Whole-system RAM peaked at 9,463/15,656 MB, swap at 655 MB, and GR3D at
  99%; the fully resident container settled at about 2.42 GiB.
- CustomVoice is not yet a complete profile pass: the baked VoxEdge backend
  incorrectly advertises `VOICE_CLONE` and `supports_voice_cloning=true`.
  `/tts/clone` forwarded an external embedding to the CustomVoice worker,
  which returned `handleAudioGeneration failed`, surfaced as HTTP 500.

Local evidence:
`validation/v091-runtime-qualification-20260725/device-evidence/`
(`r2-artifact-staging`, `r2-base-n1`, `r2-triple-overlap-n1`, and
`r2-product-n1-co-residency`, `r2-base-isolated-n2`, and
`r2-customvoice-n1`).

## Promotion decision

Not promoted. The original production voice image remains the rollback-safe
active target while the blocking items below are fixed and a new image is
built. GDN and translator remain on their original production images.

## Rollback safety

- Full private `docker inspect` snapshots were captured for
  `seeed-voice-v091`, `edge-llm-chat-service`, and `translator`.
- Raw inspect and reconstructed env files are mode 0600 and are excluded from
  distributable checksums. Only redacted snapshots may be published.
- A real rehearsal switched to the candidate, passed Base N=1, restored the
  old voice container, and recovered `/readyz` in under two minutes.
- Restored old voice image:
  `seeed-local-voice:v0.9.1-edgellm-runtime-20260725-0b8d966`;
  health passed and restart count remained zero.

## Passing runtime gates

- Base N=1 + ASR + Qwen3.5-4B GDN/MTP co-residency: 3/3 overlap rounds
  passed. TTS was about 3.70 s, ASR about 0.49 s, GDN about 0.32 s; both
  speech workers remained resident and GDN did not restart.
- ASR true streaming N=1 works when
  `EDGE_LLM_ASR_STREAM_MODE=worker`: 10 partials, first partial 1002.4 ms
  before EOS, final result present, EOS-to-final 1327.7 ms.
- Base isolated ASR N=2: 3/3 rounds passed; all 6 lanes emitted partials
  before EOS; first partial 501-1002 ms and true overlap 4.93-5.10 s.
- Base TTS HTTP chunk N=1: 200, 4-byte little-endian sample-rate header
  followed by raw PCM chunks, 24 kHz, TTFA 2609.9 ms; disconnect followed by
  recovery passed. This endpoint is chunked binary HTTP, not SSE.
- Base TTS isolated N=2: both concurrent streams returned valid 24 kHz PCM.
- CustomVoice N=1 while GDN was resident: valid 24 kHz streaming output;
  disconnect and immediate recovery passed.
- Qwen3.5-4B GDN/MTP kernel inference and SSE correctness passed without
  CUDA/Myelin/runtime errors.
- A diagnostic CMake-managed relink of the pre-codec-fix MOSS worker against
  the exact ORT 1.23.2
  headers/library passed semantic `ldd -r`, referenced only
  `OrtGetApiBase@VERS_1.23.2`, and passed `--help`. The candidate worker SHA256
  was `f5de6809ebccdb92eb9e171d24ec892c5f965224df472e994a39f0c936d1995a`.
  This is ABI/runtime evidence only and is not a release candidate; the final
  worker must be rebuilt from the explicit-codec-directory source.
  The normalized and formal MOSS worker/runtime source hashes are identical,
  while the prebuilt v0.9.1 core and device-link hashes remained unchanged.
- MOSS direct N=1/N=2/cancellation passed with the ORT 1.23.2 worker and a
  validation-only codec compatibility root: true N=2 overlap, distinct
  non-silent 48 kHz stereo outputs, and 3/3 cancel/keep/recovery rounds with
  no CUDA/TensorRT error hits. The worker advertised two slots, concurrent
  dispatch, cooperative cancellation, prompt-template loading, and voice
  cloning.
- The normalized v0.9.1 Spark worker passed direct W4A16 and BF16 N=1/N=2
  gates. Both N=2 variants interleaved chunks. W4A16 also passed voice clone
  and 3/3 cancel/keep/recovery rounds without CUDA errors. W4A16 N=1 TTFA was
  420-458 ms versus 642-695 ms for BF16, making W4A16 the better default.
- Voice, GDN, translator rollback and final health probes returned HTTP 200.

## Blocking findings

1. The Base profile in the tested image defaults ASR streaming to accumulate
   mode. It emitted only a final message (`partial_count=0`). Adding
   `EDGE_LLM_ASR_STREAM_MODE=worker` made true streaming pass. The outer
   profile fix is commit `c0adbd1`; it must be included in the rebuilt image.
2. Base TTS HTTP N=2 cancellation has a service-layer race: two streams
   produced valid audio, but after closing a long request after first audio,
   the immediate recovery returned HTTP 200 with zero PCM. This is a failure,
   not a successful empty synthesis.
3. MOSS startup in the tested image exited 3 because its packaged worker was
   linked against `OrtGetApiBase@VERS_1.20.0`, while the image contains ORT
   1.23.2 and no `libonnxruntime.so.1` link. The corrected worker now passes
   direct runtime qualification, but a rebuilt image must still package it,
   create the SONAME link, and run the semantic build gate. Independently,
   the MOSS decode loader expects
   `codec_decode_step.plan` and `codec_browser_onnx_meta.json` directly under
   `MOSS_ENGINE_DIR`, while the staged files are under its `codec/`
   subdirectory. The final profile/artifact layout must make that contract
   explicit; `MOSS_CODEC_ONNX_DIR` only covers clone encoding.
   The thin-image build must validate the exact release worker copied to
   `/opt/edgellm-v091/bin`, not the unrelated legacy
   `/opt/jv-workers/moss_tts_nano_worker`; deployment preflight must repeat
   the semantic gate after the artifact set is mounted.
4. GDN two-client SSE returned correct results in 3/3 rounds, but token
   overlap was 0/3. The service still provides one active request plus
   queueing; it is not true continuous batching.
5. Spark W4A16/BF16 and the normalized `spark_tts_worker` pass direct runtime
   gates, but no v0.9.1 service profile is bundled. Spark remains a local
   worker/export/service extension rather than an upstream-provided feature.
6. Independent review of the candidate TTS HTTP release gate rejected it:
   the HTTP/body validation, metrics label, dual-executor, and legacy
   quarantine cases are now covered, but the recovery path can still accept a
   successful response that completes after its absolute deadline. The static
   MOSS test still asserts the legacy worker path; the expected worker SHA is
   caller-supplied rather than bound to release metadata; the runtime
   read-only artifact mount can replace the build-gated worker without a
   second SHA/ABI preflight; and the codec required-file contract omits the
   browser metadata and clone encoder model/data.
7. The fresh formal build selected an SM87 CUTE artifact generated with CUDA
   13.2.78 / CUTLASS DSL 4.6.0 and enabled
   `f16_moe,ffpa,gdn,gemm,int4_fp16_gemm,ssd`. On the JetPack 6.2 CUDA 12.6
   host it failed compiling the generated F16-MoE header because
   `cudaLibrary_t` is unavailable. The independently clean six-PR build that
   passed selected CUDA 12.6.68 / CUTLASS DSL 4.5.1 and only
   `gdn,gemm,ssd`. Both trees have the same `cmake/CuteDsl.cmake` SHA, so the
   failure is an artifact selection/layout mismatch, not a generic MOSS or
   ONNX Runtime failure. A final clean replay must use the compatible SM87
   artifact set; reusing an old core build is not acceptable for promotion.
8. A strict replay of inner `692ffc7` stopped before configuration because
   patch 0041 no longer applied after patch 0031's explicit codec-directory
   constructor change. This proves the current 7+35 patch stack is not yet
   reproducible. Patch 0041 must be minimally rebased, its lock metadata
   updated, and the replay restarted from a new empty source/build path.
9. The r2 runtime's baked
   `voxedge.backends.jetson.trt_edge_llm_tts.TRTEdgeLLMTTSBackend`
   unconditionally includes `VOICE_CLONE` in `capabilities` and has no
   CustomVoice-specific `supports_voice_cloning` property. For
   `model_id=qwen3-tts-customvoice`, external embedding synthesis is not
   supported by the engine and currently fails as HTTP 500. The minimal fix
   belongs in the VoxEdge backend: remove `VOICE_CLONE`, report
   `supports_voice_cloning=false`, and fail clone/enrollment before worker
   dispatch when the active model is CustomVoice. Rebuild the VoxEdge wheel
   and runtime image, then repeat CustomVoice N=1 before testing its N=2
   profile.
   **Resolved in r3:** outer commit `ef27c98` embeds committed VoxEdge wheel
   0.0.5a0 (SHA256
   `c39a68b36e8d62c1cf74443131eca93ace0e3de9bd890858c13a2dc3cf05a037`)
   from VoxEdge commit `8e043d3`. A fresh context built image
   `seeed-local-voice:v0.9.1-edgellm-runtime-r3-cvfix-ef27c98-20260726`
   (`sha256:04f3e582da5975f636105e16ff8824a664454692dd921fad958fcea1e0de2bee`).
   Installed-file SHA, revision label, static CustomVoice-false/Base-true
   assertions, and mounted MOSS runtime gates passed. CustomVoice N=1 then
   passed end to end: both clone endpoints returned capability-aware HTTP
   400, three built-in voices produced distinct 24 kHz PCM, and
   cancel/recovery passed 10/10.
10. CustomVoice N=2 dual inference is real, but strict multi-sentence
    cancellation is not yet release-safe. ASR b2 dual streaming passed 1/1.
    Different-voice dual TTS passed 3/3 with the actual CustomVoice worker
    using `--max_slots 2`; hot TTFA was 4047-4057 ms and overlap was about
    4.89 s. In cancel/keep/recovery, round 1 passed in 10.91 s but round 2
    missed the absolute 15 s deadline after seven 429 responses and a read
    timeout; the keep stream's next sentence then encountered worker
    `pool_saturated`.

    A bounded single-sentence diagnostic passed 5/5, with every keep stream
    completing 864,000 PCM bytes. It also exposed a repeatable service-layer
    delay: HTTP disconnect was detected immediately, but Python
    `GeneratorExit` and `WorkerIO.cancel` were not reached until
    5.82-5.84 s later, after the executor's blocked generator produced its
    next chunk. Recovery was accepted about 0.21-0.23 s after that and
    completed in 9.83-9.86 s. The disconnect watcher currently calls
    `close()` on a sync generator executing in another thread; it cannot
    promptly interrupt the blocking WorkerIO queue read. Cancellation must
    be propagated as an explicit event/token into VoxEdge WorkerIO, with
    `cancel_ack` handled as non-terminal and the service admission token held
    until the worker emits terminal `cancelled`. This is a local
    service/VoxEdge integration fix, not an upstream TensorRT engine change.

## MOSS codec deploy contract

`codec_browser_onnx_meta.json` declares the clone encoder as
`moss_audio_tokenizer_encode.onnx`; that ONNX's protobuf external-data
location is exactly `moss_audio_tokenizer_encode.data`. The metadata also
maps both decode ONNX variants to `moss_audio_tokenizer_decode_shared.data`.
The validated TRT deployment directory contains the metadata, decode plan,
plan metadata, shared decode data, encoder ONNX, and encoder external data.
All six files should be included in the release required-file contract.

## Evidence root

Device:
`/home/harvest/validation/v091-runtime-qualification-d52d973-20260725`

Raw private snapshots and `state/voice.env` must not be uploaded.

## r5 integrated cancellation qualification (2026-07-26)

- Integrated source is outer commit `6e83cf0`, VoxEdge commit `fff4e47`, and
  wheel SHA256
  `31f82fba9e13c5cfeaced6b8027d842e2b343d07d361c8986591b6325ed03148`.
  The fresh r5 runtime image is
  `seeed-local-voice:v0.9.1-edgellm-runtime-r5-prefetch-cancel-6e83cf0-20260726`
  (`sha256:5782618cbe5ce13f34789c98d5fabd44dca113aea90bfc8f50fd8d2a9d86c9dc`).
  Image static gates, installed-wheel content verification, MOSS static gate,
  and mounted ORT 1.23.2 semantic gate passed.
- CustomVoice N=2 multi-sentence cancel/keep/recovery passed 20/20 without
  relaxing the 15-second recovery deadline. Every keep stream completed
  3,087,360 PCM bytes. Recovery elapsed time was 10,850.2-10,999.2 ms, with
  exactly one expected immediate HTTP 429 per round followed by HTTP 200.
- Raw stderr contains 20 disconnects, 20 `WorkerIO.cancel` sends, 20
  `cancel_ack` events with `tripped=True`, and 20 consumed terminal
  `reason=cancelled` events. Disconnect-to-ack was 75-99 ms and
  disconnect-to-terminal was 184-205 ms. There were no `pool_saturated`,
  traceback, ERROR, CUDA, TensorRT, or Myelin hits. Container restart count
  remained zero and final readiness was `ready`.
- This resolves the r4 cancellation blocker: r4 retained the HTTP session
  lease behind a queued multi-sentence prefetch job and missed the second
  round's deadline at 15,185.2 ms. r5 does not wait on a prefetch executor job
  that has not started, while still draining every job that has begun using
  the backend. VoxEdge additionally cancels while waiting to acquire its
  internal WorkerIO semaphore.
- A separate N=2 residency blocker remains. Loading CustomVoice TTS after ASR
  caused a global OOM and the kernel killed `qwen3_asr_worker`. A subsequent
  ASR N=2 switch-back request passed functionally, but caused another global
  OOM that killed `qwen3_tts_streaming_worker`. The service stayed healthy and
  restarted neither PID 1 nor the container, but this is OOM-driven mutual
  eviction, not controlled model switching and not ASR+CustomVoice
  co-residency. During the TTS gate RAM reached 15,268/15,656 MB with about
  1,781/7,828 MB swap.

Local r5 evidence:
`validation/v091-runtime-qualification-20260725/device-evidence/r5-customvoice-n2-multisentence/`.

## r6 MOSS concurrency qualification and Spark provenance audit (2026-07-26)

- The fresh r6 image is
  `seeed-local-voice:v0.9.1-edgellm-runtime-r6-moss-n2-b11ada3-20260726`
  (`sha256:31b71218a2d87696a31b676df2913287947f76458f824e2f38ea0a2913db2ef9`).
  Its exact outer source is `b11ada3e8da4f6e83b06fe6320251479bd857f68`,
  VoxEdge source is `f738123cdef13f774b8e6c55cc32f9dca8dba8ec`,
  and the embedded wheel SHA256 is
  `7cb2d067ee0796f9f4ce49437242ee56b82eaf1cbd414f55ff136d6341c6490e`.
  Static image gates and the mounted MOSS ORT semantic gates passed.
- MOSS N=1 and clone passed at 48 kHz. The original two-request gate passed
  20/20 with distinct PCM and overlapping HTTP request lifetimes, and the
  worker was actually launched with `--max-slots=2`. That gate is not
  sufficient evidence of true two-lane execution: one request can remain
  open while its work is queued behind the other.
- A stricter cancel/keep/recovery gate started the cancel request 50 ms before
  a long keep request, closed it after the first PCM, and required recovery
  PCM to arrive before the keep request ended. Cancellation itself worked:
  first cancel audio arrived in 102.9 ms, disconnect raised the cancel flag,
  worker `cancel_ack tripped=True` arrived about 97 ms later, and terminal
  `reason=cancelled` followed. The keep stream returned 7,403,520 valid PCM
  bytes.
- The strict true-N=2 condition failed. Immediate recovery returned HTTP 429;
  its retry returned HTTP 200, but first PCM arrived 83.4 ms *after* the keep
  request ended. Recovery took 8,431.0 ms. This proves the r6 MOSS path still
  serializes useful synthesis despite two advertised slots. Production must
  retain MOSS N=1; advertising it as true N=2 would be unsafe. The container
  remained healthy, restart count was zero, `OOMKilled=false`, the runtime
  error scan had no hits, and the whole test window had no kernel error
  entries.
- Spark shared-engine provenance remains fail-closed. The retained configs
  record original ONNX MD5
  `f5ec96fae85be28099d43118a3b709a5` for BiCodec and
  `1654b353f50c0d6f63c3c72508d56f47` for the speaker decoder. A complete
  `/home/harvest` scan hashed 9,191 ONNX paths and found zero matches.
  There is no `BiCodec/` checkpoint directory and no Spark-TTS source checkout.
  The only old source paths in the TRT 10.3 build logs point under the deleted
  `/home/harvest/project/v090-export/` tree. The retained current export
  scripts are auditable, but cannot regenerate either ONNX without the
  official Spark-TTS repo revision and the complete
  `Spark-TTS-0.5B/BiCodec` checkpoint.
- Consequently no new Spark shared engines or final artifact set were
  created, and no Spark service was started. The old engines were not
  overwritten. A deployable Spark artifact must be rebuilt into a new
  JP6.2/TRT10.3 output directory from those missing inputs, then carry real
  input/output hashes, commands, source revisions, device/runtime versions,
  timestamps, numerical/shape gates, and `PROVENANCE.md`. It must not depend
  on `/home/harvest/project/v090-engines`.

Local evidence:
`validation/v091-runtime-qualification-20260725/device-evidence/r6-moss-cancel-firstpcm-strict/`
and
`validation/v091-runtime-qualification-20260725/device-evidence/spark-shared-input-audit/`.
