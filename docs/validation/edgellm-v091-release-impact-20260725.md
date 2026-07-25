# TensorRT-Edge-LLM v0.9.1 release impact

Date: 2026-07-25  
Official base: NVIDIA `v0.9.1`,
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`

## Decision summary

v0.9.1 is a useful migration target and reduces our patch debt, but it is not
a pure-official deployment on JetPack 6.2 / TensorRT 10.3. Four generic
compatibility fixes remain required: the pre-10.7 stream reader, pre-10.8 FP4
guard, CuTe CUDA shim/final-link propagation, and mask-scoped FMHA cubin
loading.

The release does **not** add vLLM-style continuous/inflight batching. The
production GDN service must keep its singleflight admission gate: active LLM
generation is N=1 and the second HTTP request waits.

## Gemma 4 audio

The core library now contains real Gemma 4 audio models, runners, builders,
and kernels. The precise support matrix matters:

| Model | Official modalities in v0.9.1 | Current project status |
|---|---|---|
| Gemma 4 E2B / E4B | text, image, audio; paired MTP | Candidate for an Orin experiment; not built or qualified |
| Gemma 4 Unified 12B | text, image, audio | Core path exists; Orin memory and latency are unqualified |
| Gemma 4 31B | text, image only | No official audio support |
| Gemma 4 26B-A4B NVFP4 | text, image only | No audio, and NVFP4 is not an Orin deployment path |

Therefore the answer to “can Gemma 4 audio be used?” is:

- **library capability: yes** for E2B, E4B, and Unified 12B;
- **our Orin production service: not yet**.

Before product enablement, a pinned model must pass export, engine build,
audio end-to-end quality, memory, latency, and co-residency gates on the
16 GB Orin NX. The official experimental server also does not currently list
Gemma 4 in its VLM model-type routing, so the OpenAI/Python service path is not
yet demonstrably plug-and-play.

Official evidence:

- supported-model matrix:
  <https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html>
- v0.9.1 changelog:
  <https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.9.1/CHANGELOG.md>
- server model routing:
  <https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.9.1/experimental/server/engine.py#L66-L75>

## Continuous batching

v0.9.1 supports a **static batch**: one call may contain a fixed list of
request slots, and completed slots can be compacted. It does not admit a new
HTTP request into a running decode batch or refill a completed slot.

The official Python generation path iterates prompts and constructs
`request.requests = [req]`. `--max-batch-size` controls engine capacity; it is
not a dynamic HTTP scheduler. Paged KV prefill and paged XQA are kernel
infrastructure and do not change that scheduling contract. The standard
attention plugin path still disables paged KV, while the current KV manager
allocates a fixed tensor from maximum batch and sequence sizes.

Hardware evidence agrees with the source:

- official server: 20/20 sequential requests passed, while true concurrent
  clients reproduced Myelin `already loaded binary graph` and one empty
  response;
- product singleflight: 50/50 requests were correct, but 0/50 pairs had token
  overlap; TTFT p50/p95 was 142/210 ms.

The production conclusion remains **LLM active N=1, queued N=2**. TensorRT-LLM
Executor has continuous batching, but it is a different runtime from
TensorRT-Edge-LLM and must not be used as evidence for this deployment.

Official source evidence:

- Python request construction:
  <https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.9.1/experimental/server/engine.py#L794-L926>
- runtime batch/eviction loop:
  <https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.9.1/cpp/runtime/llmInferenceRuntime.cpp#L525-L578>
  and
  <https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.9.1/cpp/runtime/llmInferenceRuntime.cpp#L1883-L2024>
- standard attention paged-KV flag:
  <https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.9.1/cpp/plugins/attentionPlugin/attentionPlugin.cpp#L367-L375>
- current KV allocation contract:
  <https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.9.1/cpp/runtime/kvCacheManager.h#L36-L55>

## Benefits to this project

1. Official SM80+ CuTe INT4-FP16 GEMM/GEMV lets the TTS Talker prefer the
   upstream kernel. The fixed local GEMM/GEMV implementation can move toward
   rollback-only status.
2. GDN plus MTP improved measured single-stream generation from 39.2397 to
   48.2558 token/s (`+22.98%`) with only 20.9 MiB additional peak unified
   memory. This is throughput, not request concurrency.
3. Online GPU filter-bank preprocessing and WAV/PCM ingestion simplify
   Qwen3-ASR/Omni integration. The v0.9.1 WAV-boundary ASR b2 gate passed
   50/50, although it does not automatically replace our streaming mel/VAD
   frontend.
4. Speculative-decode routing/sampling, logprobs/logit-bias, and runtime
   synchronization/deserialization fixes reduce server and stability debt.
5. DFlash/DDTree for Qwen3/Qwen3.5 is a future Orin optimization candidate,
   subject to a separate draft-model, quality, memory, and latency gate.
6. Gemma 4 audio expands the future model choice, but is not part of this
   release's production acceptance.

Features without direct present value on Orin include Qwen3-TTS CodePredictor
FP8 (Orin has no FP8 deployment path) and paged-KV kernels as a substitute for
continuous batching. Official Qwen3-TTS still covers CustomVoice; Base remains
our extension.

## Rebuilt runtime artifact

The release runtime image was rebuilt on the Orin NX from product revision
`0b8d966923456fdf19e8287a7f109c76b6bb2c9c`:

```text
tag:     seeed-local-voice:v0.9.1-edgellm-runtime-20260725-0b8d966
digest:  sha256:b5f31b3d7a124ce7d68378a7fc880432dae2191cac2822538428ca5a11a69a95
arch:    linux/arm64
size:    597403779 bytes content size; 2.03 GB shown by docker image ls
log:     /home/harvest/validation/v091-runtime-build-20260725-0b8d966/docker-build.log
log sha: 66e49c091eb9a7ecc385297b60f5e4a99acf22c69ba9e18e8ba27c70073b2e5a
```

Build-time and in-image Python compilation passed. The image contains the TTS
HTTP cancellation fix and the qualified low-latency
`first=7 / chunk=10 / adaptive=0` configuration.

### Orin runtime gate

The old voice container was retained, stopped, and renamed
`seeed-voice-v091-rollback-b1kv1536`; it was not deleted. The new image then
replaced the production voice container on port 8621.

Validation before restoring GDN:

- container reached `healthy` with `RestartCount=0`;
- ASR recognized the existing Chinese quality WAV exactly, reporting
  161 ms end-to-end inference and 155 ms worker time;
- warmed TTS HTTP streaming completed 2/2, with 745.4 ms mean TTFA and
  0.622 mean RTF for the short smoke prompt;
- a forced HTTP stream abort incremented the real worker cancel counter from
  0 to 1;
- the immediate post-cancel full TTS stream completed with 945.9 ms TTFA and
  0.623 RTF.

GDN was restarted only after voice load and warm-up, avoiding the simultaneous
cold-start memory pressure observed in the earlier migration run.

Final co-resident state:

- `seeed-voice-v091`: running, healthy, restart count 0;
- `edge-llm-chat-service`: running, healthy, restart count 0, MTP enabled;
- `translator`: running and healthy.
