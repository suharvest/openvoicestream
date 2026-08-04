# TensorRT Edge-LLM v0.9.1 release checkpoint

Date: 2026-08-04
Decision: model-level r5/v5 promoted on Orin NX and published

## Model-level production cutover (2026-08-04)

This section supersedes the aggregate artifact-set publication sequence below.
The runtime images contain no model engines. Each model has one immutable
Hugging Face repository/revision recorded in
`deploy/artifacts/v091-release-lock.json`; a profile only composes those model
sources into a service. Downloads use `HF_ENDPOINT=https://hf-mirror.com`, are
SHA-256/size checked, safely extracted into staging, and atomically installed
into persistent model storage.

- Speech image:
  `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-jp62-trt103-edgellm-v091-20260804-r5`.
  Registry digest: `sha256:b1d9db8d0e61344dc02367bb0114fd6889335f21638cea790e3f795d8226ce5c`
  (674,375,463 bytes). Stable tag:
  `jetson-jp62-trt103-edgellm-v091`.
- LLM image:
  `sensecraft-missionpack.seeed.cn/solution/edge-llm-chat-service:v0.9.1-gdn-mtp-8k-20260804-v5`.
  Registry digest: `sha256:0ec928901a020cd9e67078d2b32837acc28137bc0c3dbfc5b08798e2133efc98`
  (143,149,319 bytes). Stable tag: `v0.9.1-gdn-mtp-8k`.
- Production speech profile: `jetson-edgellm-v091-matcha`, port 8621,
  model volume `speech-models-v091`.
- Production LLM: Qwen3.5-4B GDN/MTP, 8K input/KV, port 8000, artifact
  `harvestsu/qwen3.5-4b-gdn-mtp-jetson-artifacts@90e24bbebc46134d63cff35afc1aced1684faed4`.

An empty-cache boot downloaded ASR (1,827,532,800 bytes), Matcha
(376,135,680 bytes), and GDN/MTP (3,876,147,200 bytes) from the mirror and
installed all three from their fixed revisions. The first speech attempt
exposed two release defects: a stale aggregate-set startup preflight in the
device build context, and an unconditional legacy 26-file Qwen downloader in
`LANGUAGE_MODE=multilanguage`. The build context was reconciled with the
committed model-level launcher, and commit `9623899` makes explicit Qwen model
sources bypass the legacy aggregate path. The focused routing/model-level
suite passes 24/24; runtime packaging passes 9/9.

Final production acceptance passed on Orin NX:

- `/v1/models` reports Matcha and Qwen3-ASR with canonical IDs and aliases;
- `/v1/capabilities` reports Matcha voice `0`/`Default`, speed control,
  streaming, and ASR streaming capability;
- `/v1/audio/speech` returns valid 16 kHz mono WAV with HTTP chunked transfer;
- the generated speech round-trip transcribed exactly as
  `正式服务迁移验证。`;
- Qwen3.5-4B returned `迁移通过`;
- simultaneous Matcha TTS and GDN requests both completed and both services
  remained healthy; Qwen3-ASR stayed resident;
- steady Docker memory was about 2.08 GiB for speech and 2.36 GiB for GDN.

Rollback containers remain stopped and intact as
`seeed-voice-v091-pre-r5` and `edge-llm-chat-service-pre-v5`.

## Final r5/r12 qualification

The earlier r6/Spark-blocker sections below are retained as investigation
history. They are superseded by the final qualification in this section.

- Final artifact set:
  `orin-nx-edgellm-v091-jp62-trt103-sm87-20260803-r5`.
- Artifact root on the target:
  `/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260803-r5/v091`.
- Payload: 202 files, 19,919,345,439 bytes, 31 verified engine sidecars.
- Independent `sha256sum -c` passed for all 202 files.
- Rollback r2 remains untouched.
- Final runtime source: outer `084f774`, engine overlay `7cbfa84`, NVIDIA
  v0.9.1 `7f061f21`.
- Immutable runtime image:
  `seeed-local-voice:v0.9.1-edgellm-runtime-r12-084f774-20260803`, image ID
  `sha256:d5845b4a0d516929a7023da16dd3e8736c3433b783812b7af7050d9857cbe452`.
- Stable device tag:
  `seeed-local-voice:v0.9.1-edgellm-runtime-r12-20260803`.

The r12 image was built from a clean transferred source context. No runtime
source file, profile, sidecar, or Python module was bind-mounted over the
image during its qualification.

SparkTTS W4A16 HTTP results on the clean image:

| Gate | Result |
| --- | --- |
| N=1, two rounds | TTFA 372.2–423.6 ms; total 1.686–1.739 s |
| N=2, three rounds | 6/6 complete; TTFA 704.2–724.9 ms; total 2.703–3.052 s |
| cancel A / continue B / recovery | 3/3; B 209,280 PCM bytes; recovery 200 |

The production Base profile was then restored and the Qwen3.5-4B GDN+MTP
container started beside it. Three complete co-residency rounds passed:

- every TTS request overlapped a real GDN request;
- every ASR request overlapped a real GDN request;
- deterministic 24 kHz mono Base output, 132,480 frames per round;
- transcript: `默认五百一十二长度的语音合成和大语言模型可以稳定共同运行。`;
- ASR latency 362–367 ms; TTS request time 3.509–3.537 s;
- GDN restart count remained zero;
- final ASR and TTS residency was `both` and both containers were healthy.

Device evidence:
`/home/harvest/validation/v091-final-r12-clean-20260803`.

The final patch boundary is explicit and mechanically checked: seven exact
minimal upstream bug-fix candidates are applied first, followed by 35 sparse
local product-extension patches. The relevant outer/inner contract suite
passes 84/84. SparkTTS, MOSS, Qwen3-TTS service protocols and device scheduling
remain local product functionality; only generic bugs belong in the existing
NVIDIA issue/PR queue.

## Strict three-model overlap qualification (2026-08-04)

The previous co-residency gate overlapped GDN separately with ASR and TTS. A
stricter gate now starts all three requests together and requires the GDN token
stream to span both the first ASR partial and the first TTS audio chunk. It
also requires the ASR and TTS useful-output windows to cross, correct outputs,
sub-500 ms request-start skew, healthy containers, and zero restarts.

The unmodified Base profile correctly rejected the second voice session. A
temporary `max_concurrent_sessions=2` alone was still clamped to one because
the resolver used `min(asr.max_concurrent, tts.max_concurrent)`. That aggregate
confused same-modality fan-out with one ASR plus one TTS request. The local
resolver now supports an explicit, device-qualified
`execution_policy.cross_modal_overlap=true`; it adds the independent ASR and
TTS capacities without increasing either worker's own slot count.

Qwen3-ASR + Qwen3-TTS Base + Qwen3.5-4B GDN passed 3/3 qualification and
5/5 soak rounds with zero voice/GDN restarts. ASR first partial was
1.252 seconds and GDN TTFT was 111–167 ms. However Base TTS TTFA rose to
7.53–7.68 seconds under the long GDN stream, so this is packaged as the opt-in
`jetson-edgellm-v091-qwen3ttsbase-triple` profile and is not the low-latency
default.

The original MOSS profile (`exclusive`, lazy TTS) could not keep ASR active:
MOSS and GDN completed, but the ASR WebSocket was closed on every triple
round. With `concurrent`, ASR/MOSS preload, and
`EDGE_LLM_ASR_STREAM_MODE=worker`, all three workers fit beside GDN. The strict
gate passed 3/3 qualification and 10/10 soak rounds with zero restarts:

- ASR first partial: 1.252 seconds;
- MOSS TTFA: 222–227 ms in soak (308 ms first qualification warmup);
- GDN TTFT: 162–167 ms in soak;
- strict three-request overlap: 3.39–4.86 seconds;
- post-soak memory: voice 2.398 GiB, GDN 614 MiB by Docker accounting.

MOSS therefore becomes the preferred profile for low-latency simultaneous
ASR + TTS + GDN on this Orin NX. Evidence is under
`/home/harvest/validation/v091-r12-strict-triple-20260804`.

### Clean r13 image confirmation

The concurrency changes were packaged into immutable image
`seeed-local-voice:v0.9.1-edgellm-runtime-r13-28a648e-20260804`, image ID
`sha256:660d7ebbad22c83707f5b529e41ad3e8969af31e4f2771c0779c26ac73c1801c`.
Its OCI source revision is
`28a648ebe858ab6d59829fb8273eccdb0df9e1d1`; the packaged MOSS worker remains
the release-gated binary with SHA-256
`9d114d8390e684c8876e2ef9e20e28ee6d4ec6ce18b81df5da3ba64c8f057deb`.
Neither clean-image run bind-mounted source code or profiles.

The MOSS profile passed 3/3 strict rounds: ASR first partial was
1.252 seconds, TTS TTFA was 224.7–265.0 ms, GDN TTFT was 162.1–207.3 ms,
and useful three-request overlap was 3.51–4.38 seconds. The voice and GDN
containers remained at zero restarts; post-gate memory was 2.053 GiB and
240.7 MiB respectively.

The opt-in Base triple profile also passed 3/3 strict rounds: ASR first
partial was 1.252 seconds, TTS TTFA was 7.67–7.77 seconds, GDN TTFT was
131.1–163.2 ms, and useful three-request overlap was 7.81–7.88 seconds.
Both containers again remained at zero restarts. This confirms correctness,
but also confirms that Base is not the low-latency triple choice.

Clean-image reports are `r13-moss-clean-triple3.json` and
`r13-base-clean-triple3.json` in the evidence directory above.

### Base low-latency profile correction

The r13 Base profiles had accidentally omitted the already-qualified HTTP
streaming parameters. The service therefore fell back to a 50-frame first
chunk and 97-frame subsequent chunks instead of the product's low-latency
7/10 configuration. This was a profile regression, not a Base engine or
speaker-embedding limitation.

Restoring `streaming_profile=low_latency`, first chunk 7, subsequent/max chunk
10, and disabling adaptive growth produced the following Orin NX A/B while
the GDN service remained resident:

- Base N=1, GDN idle: 5/5 complete; warm TTFA 355–356 ms (416 ms first round),
  compared with about 1.96 seconds before the correction;
- ASR + Base + long-streaming Qwen3.5-4B GDN: 3/3 strict rounds; Base TTFA
  887–994 ms, compared with 7.67–7.77 seconds before the correction;
- ASR first partial remained 1.252 seconds and GDN TTFT remained 157–166 ms;
- useful three-request overlap remained 7.83–7.86 seconds and both containers
  remained at zero restarts.

The remaining triple-load penalty is GPU scheduling contention, but it is
about half a second over the warm Base-only TTFA rather than the previous
multi-second delay. Evidence files are `r13-base-lowlatency-single5.json` and
`r13-base-lowlatency-triple3.json` in the same device evidence directory.

The correction is packaged without bind mounts in immutable image
`seeed-local-voice:v0.9.1-edgellm-runtime-r14-8f9084e-20260804`, image ID
`sha256:7f9ff7903e0aae0f16c74af1e6e70b46aff1b1a85e2f819796380fa07afd4c6a`,
with OCI source revision
`8f9084e77e1a86778d8e94e937590537d74d8006`. The clean r14 triple gate
repeated the result 3/3 at 883–981 ms Base TTFA. After restoring the formal
service name and conservative default profile, five single-request rounds
again produced 356 ms warm TTFA, the HTTP cancellation/recovery gate passed,
and the voice/GDN containers remained at zero restarts. Evidence files are
`r14-base-clean-lowlatency-triple3.json`,
`r14-production-base-single5.json`, and
`r14-production-base-cancel-recovery.json`.

## Reproducible source boundary

- NVIDIA TensorRT Edge-LLM v0.9.1:
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6`.
- Product integration branch: `codex/edgellm-v091-upstream-audit`.
- Engine overlay branch: `codex/edgellm-v091-production-migration`.
- VoxEdge source used by the qualified r6 image:
  `f738123cdef13f774b8e6c55cc32f9dca8dba8ec`.
- VoxEdge wheel SHA-256:
  `7cb2d067ee0796f9f4ce49437242ee56b82eaf1cbd414f55ff136d6341c6490e`.

The post-r6 source adds two release corrections that still require a cached
thin-image rebuild:

1. package the self-contained `jetson-edgellm-v091-sparktts` profile;
2. emit metadata for both Spark shared engines and refuse model downloads
   unless `HF_ENDPOINT=https://hf-mirror.com`.
3. fetch only the required Spark BiCodec checkpoint inputs with pinned source
   SHA `2f1ea9082400547242641f5271b6f941c9f439d1` and model revision
   `642071559bfc6346c2359d19dcb6be3f9dd8a05d`.

## Qualified image and runtime findings

The latest device-built image is
`seeed-local-voice:v0.9.1-edgellm-runtime-r6-moss-n2-b11ada3-20260726`
(`sha256:31b71218a2d87696a31b676df2913287947f76458f824e2f38ea0a2913db2ef9`).
Its static image gates and mounted MOSS ONNX Runtime 1.23.2 ABI gates passed.

- Base N=1, true-streaming ASR, TTS, cancellation recovery, GDN co-residency
  and rollback rehearsal passed in the earlier complete gate.
- Base isolated N=2 passed ASR and TTS overlap, isolation and cancellation.
- CustomVoice N=1 passed. Its unsupported external embedding clone API now
  fails before worker dispatch instead of returning HTTP 500.
- CustomVoice TTS-only N=2 cancellation passed 20/20, but ASR N=2 plus
  CustomVoice N=2 co-residency caused kernel OOM eviction on the 16 GB device.
  It is therefore an isolated qualification profile, never the production
  multi-model default.
- MOSS r6 N=1 and clone passed. Two HTTP requests completed 20/20 with the
  worker launched at two slots. A stricter useful-work overlap gate did not
  pass its first captured round: recovery first PCM arrived 83.4 ms after the
  long keep stream ended. Cancellation, PCM validity, recovery deadline,
  container health and runtime error scan all passed. Until a repeatable
  first-PCM overlap gate passes, MOSS production remains N=1.
- Qwen3.5-4B GDN returns correct results to two clients but the measured token
  overlap is zero. The service is singleflight/queued and must not be called
  official continuous batching.

## Spark release blocker

The v0.9.1 Spark W4A16/BF16 LLM engines and native worker passed direct N=1,
N=2, clone and cancellation gates. W4A16 remains the latency default.

The shared BiCodec and speaker-decoder engines are not publishable yet. The
retained files have configuration and numerical records but no trustworthy
source/build provenance. Their recorded source ONNX MD5 values are:

- BiCodec: `f5ec96fae85be28099d43118a3b709a5`;
- speaker decoder: `1654b353f50c0d6f63c3c72508d56f47`.

A complete device scan found neither matching ONNX files nor the original
Spark-TTS source/checkpoint. The old engines remain untouched and outside the
final artifact set. The recovery route now pins the official source and model
revisions above and downloads only `BiCodec/config.yaml` and
`BiCodec/model.safetensors` through hf-mirror. Those inputs have not yet been
downloaded or rebuilt on the target device.

## Remaining publication sequence

1. Keep r2 rollback and the r5 artifact tree immutable.
2. Publish r14 only after the owner confirms the exact container registry
   repository/tag. Verify the pulled remote image ID and OCI revision.
3. Publish the r5 artifact set only after the owner confirms the exact model
   repository. For Hugging Face upload, use `HF_HUB_DISABLE_XET=1`, upload one
   file at a time, and verify every remote hash before marking it published.
4. Deploy the conservative Base profile by default. Select
   `jetson-edgellm-v091-moss` for low-latency simultaneous ASR + TTS + GDN,
   or explicitly select `jetson-edgellm-v091-qwen3ttsbase-triple` when Base
   voice characteristics matter more than its measured triple-load latency.

No additional NVIDIA issue or PR is authorized by this sequence. Existing
bug PRs remain preparation/maintenance work; model features and product
protocols stay out of the bug queue.
