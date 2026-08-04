# NVIDIA TensorRT-Edge-LLM v0.9.1 upstream bug PR review notes

> Historical preparation notes plus the current review mapping. Bug PRs #118
> and #145-#149 were already submitted under the owner's earlier explicit
> authorization. Do not create another PR, push revisions, comment, label, or
> otherwise mutate upstream state without a new explicit confirmation.

Upstream base: `v0.9.1`
(`7f061f21f0a581ba234a1e233c9315b89d8e47d6`).

## 1. Qwen3-ASR MRoPE export

Submitted review: [PR #146](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/146).

Local branch: `codex/upstream-v091-fix-asr-mrope`

Proposed title:
`Fix Qwen3-ASR export config losing MRoPE semantics`

Draft body:

The Qwen3-ASR checkpoint can expose `mrope_section` while retaining
`rope_type: linear`. Passing that value through to the Edge-LLM runtime config
means the C++ rope collector does not select the MRoPE path, and the exported
engine produces degraded transcription. For `model_type == qwen3_asr`, this
change normalizes `type` and `rope_type` to `mrope` when `mrope_section` is
present, while preserving the remaining rope parameters and leaving other
model types untouched.

Validation to attach before submission:

- focused exporter unit test;
- bad/good exported config diff;
- fresh-engine transcript comparison;
- batch-1 and batch-2 device results.

Cleanup required: remove the Claude co-author/session trailers and historical
local-document reference from the current commit message.

## 2. TensorRT 10.3 stream reader compatibility

Submitted review: [PR #147](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/147).

Local branch: `codex/upstream-v091-fix-trt103-stream-reader`

Proposed title:
`Support streamed engine reads before TensorRT 10.7`

Draft body:

The runtime currently relies on the newer TensorRT stream-reader interface.
JetPack 6.2 ships TensorRT 10.3, where deserialization needs an
`IStreamReader`-compatible adapter. This patch adds the version-bounded
compatibility implementation and keeps the newer path unchanged.

Validation to attach before submission:

- clean TRT 10.3 build;
- real engine deserialization;
- read, seek, EOF, and error-path unit coverage;
- a newer-TensorRT build proving the existing path is unchanged.

## 3. FP4 enum guard

Submitted review: [PR #145](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/145).

Local branch: `codex/upstream-v091-fix-trt103-fp4-guard`

Proposed title:
`Guard TensorRT FP4 data type use on pre-10.8 headers`

Draft body:

`nvinfer1::DataType::kFP4` is unavailable in the TensorRT 10.3 headers used by
JetPack 6.2, so the plugin fails at compile time even when FP4 is not selected.
This change guards the FP4-specific cases at the TensorRT version where the
enum became available; supported data types and newer TensorRT behavior remain
unchanged.

Validation to attach before submission:

- clean TRT 10.3 plugin build;
- clean newer-TensorRT plugin build;
- FP4-path smoke test on a runtime that supports FP4.

## 4. CuTe final-link propagation

Submitted review: existing [PR #118](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/118).

Local branch: `codex/upstream-v091-fix-cute-final-link`

Proposed title:
`Propagate CuTe CUDA driver link requirements to final targets`

Draft body:

CuTe-generated objects require their CUDA driver/shim link requirements to
reach the final executable or shared-library target. Keeping those
requirements private allows intermediate compilation to succeed but produces
an unresolved final link/load on JetPack 6.2. This change propagates the
required link options from the CuTe CMake helper.

Validation to attach before submission:

- clean final link on JetPack 6.2;
- runtime load of the resulting plugin/worker;
- a build without the local shim;
- split the generic propagation fix from any local shim implementation.

## 5. Context-FMHA mask-scoped cache

Submitted review: [PR #148](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/148).

Local branch: `codex/upstream-v091-fix-fmha-mask-cache`

Proposed title:
`Scope context-FMHA cubin loading and cache entries by mask type`

Draft body:

Context-FMHA kernel selection depends on mask type, but the current cubin
loading/cache path can reuse state without that distinction. This change
threads the mask type through loading and scopes cached state accordingly, so
one attention configuration cannot silently reuse another configuration's
kernel selection.

Validation to attach before submission:

- non-custom-mask device-positive engine tests;
- mixed mask-type cache-order regression;
- true custom-mask negative test proving unsupported kernels still fail
  loudly.

This candidate remains blocked on the negative test.

## 6. Checkpoint destination dtype preservation

Submitted review: [PR #149](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/149).

Local branch: `codex/upstream-v091-fix-checkpoint-dtype`

Proposed title:
`Preserve declared half dtype when loading FP32 checkpoint tensors`

The original `0033` mail patch depended on earlier local loader changes. This
branch is the extracted clean-base replacement: it inspects an existing
parameter/buffer destination, casts FP32/BF16/FP16 sources only when that
destination declares FP16/BF16, preserves deliberately FP32 scale buffers,
and retains the legacy BF16 fallback for destinations that do not yet exist.

Syntax compilation and `git diff --check` pass. Its CPU tests are present but
still need execution in a working PyTorch environment; the current WSL
PyTorch install cannot load because of an unrelated
`ncclCommWindowDeregister` symbol mismatch.

## Not a PR

The GDN Myelin simultaneous-context failure has no minimal library fix. Keep
it as a reproduction-only issue draft, also requiring explicit owner approval.

## Local verification

The six listed local branches remain immutable evidence branches based on the
exact upstream SHA and pass `git diff --check`; the submitted PR branches are
tracked separately. Read-only refresh on 2026-08-05 confirms #118 and
#145-#149 are open, non-draft, `MERGEABLE`, and `BLOCKED` only on upstream
review, with no reported checks. No upstream write was performed by this
refresh.
