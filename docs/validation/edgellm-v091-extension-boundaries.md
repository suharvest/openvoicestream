# TensorRT Edge-LLM v0.9.1 local extension boundaries

Date: 2026-07-25

This note separates NVIDIA TensorRT Edge-LLM changes from Seeed model and
service extensions. It is the decision boundary for patch retirement and
upstream proposals; it does not replace the detailed patch-debt matrix.

## Short answer

MOSS, SparkTTS, Qwen3-TTS Base, and the product's N=2 behavior are not merely
export scripts.

| Area | Export / quantization | NVIDIA C++ runtime, kernels, or build | Local worker / service | Upstream direction |
| --- | --- | --- | --- | --- |
| Qwen3-TTS Base | Yes | Yes: external speaker conditioning and related runtime path | Yes: embedding transport, clone API, admission and cancellation | Split the small Base export enablement from the larger feature work |
| Qwen3-TTS CustomVoice additions | Yes: language metadata | Yes: language conditioning and cancellation hook | Yes: named-speaker policy, capability reporting and HTTP lifecycle | Feature proposal only after model-quality evidence; service policy stays local |
| MOSS-TTS-Nano | Yes | Yes: model runtime, kernels, ORT codec integration and CMake target | Yes: JSON worker, slot dispatcher, clone, cancellation and HTTP integration | Keep local; it is a model-support proposal, not a bug-fix PR |
| SparkTTS | Yes: mixed precision and W4A16 | Yes: BF16 residual path, INT4 output dtype kernels/plugin and shared contexts | Yes: worker protocol, style/clone registry, admission and cancellation | Keep local until generalized; not suitable as a small bug-fix PR |
| ASR / TTS N=2 | Engine batch/profile selection | Yes for lane/session ownership and shared-engine execution | Yes: request IDs, slot pools, semaphores, aggregate limits and disconnect cleanup | Product architecture; upstream only small independently reproducible runtime bugs |
| GDN parallel requests | Official engine/runtime path | Official v0.9.1 currently serializes the tested two-client workload | Local wrapper queues requests | Do not claim continuous batching until token-generation overlap is measured |

## What remains tied to NVIDIA TensorRT Edge-LLM

The local model extensions compile against and modify the upstream source
tree. They depend on TensorRT Edge-LLM tensor layouts, runtime ownership,
plugins, CUDA kernels, engine serialization, and CMake targets. An official
version change can therefore break them even when the Python product service
does not change.

- Base is the smallest extension, but only its CLI export guard is close to
  export-only. The usable product feature also needs speaker-encoder output,
  external speaker embedding conditioning in the runtime, and a worker
  transport for that embedding.
- MOSS adds a model runtime and CUDA/TensorRT execution path that NVIDIA
  v0.9.1 does not provide. Its codec also adds an ONNX Runtime 1.23.2 ABI and
  artifact-layout contract.
- SparkTTS adds a mixed-precision model path and W4A16 support that touches
  exporter configuration, model construction, ONNX types, INT4 kernels and
  the TensorRT plugin. The native worker is only the outermost layer.
- Native N=2 uses local lane allocation, per-slot execution contexts,
  shared-engine ownership, cooperative cancellation and worker dispatch.
  Official engine batching alone does not provide the product's HTTP
  concurrency semantics.

## What is product-only

The following behavior should not be proposed to NVIDIA as TensorRT
Edge-LLM fixes:

- JSON-line worker request and event schemas;
- VoxEdge backend capability declarations and voice registries;
- HTTP 400/429/503 mapping, `Retry-After`, request prefetch and disconnect
  handling;
- aggregate ASR/TTS session ceilings and co-residency policy;
- profile paths, artifact auto-download policy, manifests and deployment
  layouts;
- the decision that a 16 GB Orin NX uses N=1 for stable multi-model
  co-residency while preserving isolated N=2 qualification profiles.

These are still required production code. Moving them out of the upstream
patch series reduces rebase debt without deleting the functionality.

## Upstream submission rule

Only minimal, model-independent bugs with a standalone reproducer belong in
the current NVIDIA issue/PR set. The prepared bug candidates cover build,
ABI, loader and correctness defects. Base, MOSS, SparkTTS and the product
worker protocols remain feature work and are intentionally excluded from
those PRs.

For a future feature proposal:

1. define a public model/runtime API independent of the local JSON worker;
2. add export, engine-build, numerical/quality and cancellation tests;
3. demonstrate the feature on exact official source without unrelated
   product patches;
4. submit it on NVIDIA's feature cadence only after product ownership and
   maintenance expectations are agreed.

## Deployment policy for this migration

- The production profile is stable N=1 multi-model co-residency.
- N=2 is enabled only by an explicit profile with model-specific evidence.
- Qwen3-TTS Base N=2 is an isolated, no-GDN mode.
- Qwen3-TTS CustomVoice N=2 is TTS-only qualification on the 16 GB device:
  loading ASR N=2 and CustomVoice N=2 together caused kernel OOM eviction.
- MOSS and SparkTTS N=2 require their final-image service gates and memory
  evidence before their deployment scope is finalized.

