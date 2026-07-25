# TensorRT-Edge-LLM upstream issue and PR plan

Date: 2026-07-25  
Target upstream: <https://github.com/NVIDIA/TensorRT-Edge-LLM>  
Reproduction base: `v0.9.1` /
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`

## Submission guard

This document began as preparation only. The owner subsequently authorized
creating the six bug issues and directly opening minimal linked PRs for issues
that exist. The resulting upstream state is recorded below. Further unrelated
upstream writes still require explicit owner confirmation.

NVIDIA's contribution rules require every bug fix or change to begin with an
issue that is reviewed and approved by TensorRT-Edge-LLM engineers before
code review. Each PR should address one concern, normally target `main`, use
the repository template and conventional-commit naming, pass pre-commit, add
tests, and contain DCO sign-off.

The existing local candidate branches are evidence branches based on v0.9.1.
They are **not submission-ready**:

- replay the minimal change on current upstream `main`;
- confirm the bug is still present on `main`;
- replace `codex@local.invalid` authorship where applicable;
- recreate the final commit with the real contributor identity and
  `git commit --signoff`;
- run `pre-commit run --all-files` plus the focused tests;
- do not include Seeed product code, paths, model artifacts, or patch-series
  numbering.

## Recommended split

Use one issue and one PR for each independent root cause. Do not create an
umbrella “JetPack 6.2 compatibility” issue: it would mix unrelated API,
plugin, linker, and kernel-registry defects.

| Order | Issue | Follow-up | Template | State |
|---:|---|---|---|---|
| 1 | [#140](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/140): TRT 10.3 build fails because v0.9.1 unconditionally uses `IStreamReaderV2` | [PR #147](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/147): version-gated streamed reader | C++ Runtime | open and mergeable; CI/device validation pending |
| 2 | [#141](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/141): TRT 10.3 build fails because the FP4 plugin references `DataType::kFP4` | [PR #145](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/145): guard FP4-only code before TRT 10.8 | C++ Runtime | open and mergeable; CI/device validation pending |
| 3 | [#142](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/142): Qwen3-ASR export preserves `rope_type=linear` and silently disables MRoPE | [PR #146](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/146): normalize runtime MRoPE config plus regression test | Python Export | open and mergeable; end-to-end validation pending |
| 4 | [#143](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/143): FMHA loader eagerly loads an unrelated custom-mask cubin and fails with `INVALID_IMAGE` | [PR #148](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/148): scope load/cache by requested mask type | C++ Runtime | open and mergeable; CUDA tests pending |
| 5 | [#117](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/117): CuTe compatibility shim is not propagated to final consumers | existing [PR #118](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/118); related [PR #103](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/103) | C++ Runtime | PR #118 refreshed on v0.9.1 and mergeable; CI/Orin validation pending; do not duplicate |
| 6 | [#144](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/144): `_set_tensor` installs FP32 checkpoint tensors into declared half modules | [PR #149](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/149): destination-dtype cast plus CPU tests | Python Export | open and mergeable; upstream CI pending |

The six changes are code-independent. On the JetPack 6.2 test machine, a
complete build may require issues 1, 2, and 5 to be fixed together. That is a
**validation dependency**, not justification for a stacked or combined PR.
Each PR should still compile/test its own affected unit independently, and
cross-link the other issues only in the full-device test note.

Duplicate search on current upstream `main` found that group 5 already had
the exact issue and two overlapping PRs. Only groups 1, 2, 3, 4, and 6 were
newly filed on 2026-07-25.

## Issue 1: streamed reader on TensorRT 10.3

Proposed title:

```text
[Bug][Runtime]: v0.9.1 fails to build on JetPack 6.2 because IStreamReaderV2 requires TensorRT 10.7
```

Use `Bug report - C++ Runtime`.

Issue body essentials:

- Impact: build blocker for an NVIDIA-supported Orin NX / JetPack 6.2 stack.
- Actual: `cpp/common/trtUtils.cpp` derives from
  `nvinfer1::IStreamReaderV2`; TensorRT 10.3 headers expose only
  `IStreamReader`.
- Expected: use `IStreamReaderV2` on TensorRT 10.7+ and retain the legacy
  streamed reader contract on older supported TensorRT.
- Minimal reproduction: clean v0.9.1, the exact CMake command, and the first
  compiler diagnostic. Do not attach the 40-patch product stack.
- System: Orin NX 16 GB, aarch64, SM87, JetPack 6.2 / L4T 36.4.3,
  CUDA 12.6, TensorRT 10.3, Release build.

Proposed PR title:

```text
fix(runtime): support streamed engine reads before TensorRT 10.7
```

PR relationship and scope:

```text
Fixes #<issue-1>
```

- only `cpp/common/trtUtils.*` and focused read/seek/error tests;
- preserve the 10.7+ `IStreamReaderV2` path unchanged;
- demonstrate compile plus real engine deserialization on TRT 10.3;
- compile the modern branch against TRT 10.7+.

Prepared evidence: `codex/upstream-v091-fix-trt103-stream-reader`,
`44da5b3bb580911f5450a8a943d9106503b10afe`.

## Issue 2: FP4 enum on TensorRT 10.3

Proposed title:

```text
[Bug][Runtime]: FP4 plugin source does not compile with TensorRT versions before 10.8
```

Use `Bug report - C++ Runtime`.

Issue body essentials:

- Actual: the plugin references `nvinfer1::DataType::kFP4`, which is absent
  from TensorRT 10.3/10.7 headers.
- Expected: FP4-only capability remains compiled on TensorRT 10.8+, while
  older TensorRT builds compile without advertising FP4.
- Show the minimal translation-unit/compiler failure independently from the
  streamed-reader failure.

Proposed PR title:

```text
fix(plugins): guard FP4 data type before TensorRT 10.8
```

PR relationship and scope:

```text
Fixes #<issue-2>
```

- one plugin source plus a version-matrix compile test;
- TRT 10.3 negative-capability test;
- TRT 10.8+ test proving FP4 behavior is unchanged.

Prepared evidence: `codex/upstream-v091-fix-trt103-fp4-guard`,
`8829601185f9ba9ce2a1ab80b5c1a358f76eaab1`.

## Issue 3: Qwen3-ASR MRoPE export

Proposed title:

```text
[Bug][Export]: Qwen3-ASR export can preserve rope_type=linear and disable MRoPE at runtime
```

Use `Bug report - Python Export Pipeline`.

Issue body essentials:

- Input checkpoint contains `rope_type=linear`, `factor=1.0`, and
  `mrope_section`.
- Exported `llm/config.json` preserves `linear`; the C++ runtime therefore
  selects neither its `mrope` path nor the default-with-sections path.
- Actual: engine builds, but ASR output degrades, making this a silent
  correctness bug.
- Expected: the Qwen3-ASR factory emits the runtime's canonical MRoPE
  semantics.
- Attach a tiny config fixture and bad/good exported JSON; link quality
  evidence without uploading proprietary audio.

Proposed PR title:

```text
fix(export): preserve Qwen3-ASR MRoPE runtime semantics
```

PR relationship and scope:

```text
Fixes #<issue-3>
```

- exporter factory/config normalization plus a focused unit test;
- no product ASR worker, audio protocol, or model artifact;
- show exact transcript recovery and N=2 regression evidence only as
  validation results.

Prepared evidence: `codex/upstream-v091-fix-asr-mrope`,
`b7ac34f7cf051646fa48bed1eec347a1b7b7158e`.

## Issue 4: FMHA mask-unaware cubin loading

Proposed title:

```text
[Bug][Runtime]: context FMHA loads unrelated mask-type cubins and can fail with CUDA_ERROR_INVALID_IMAGE
```

Use `Bug report - C++ Runtime`.

Issue body essentials:

- A non-custom-mask Qwen3.5 build on SM87 loads all matching dtype/SM cubins.
- Two unused head-size-256 `CUSTOM_MASK` cubins were produced for CUDA 12.8.
- JetPack 6.2 / CUDA 12.6 rejects those unused images with `INVALID_IMAGE`,
  preventing a valid non-custom-mask engine build.
- Expected: load/cache only cubins eligible for the requested
  dtype/SM/mask contract; an actually requested unsupported custom mask must
  still fail loudly.

Proposed PR title:

```text
fix(attention): scope FMHA cubin loading by mask type
```

PR relationship and scope:

```text
Fixes #<issue-4>
```

- the three FMHA loader/plugin files and focused registry-selection tests;
- positive non-custom-mask engine build;
- negative true-custom-mask test showing no silent fallback.

Prepared evidence: `codex/upstream-v091-fix-fmha-mask-cache`,
`e275a1068e82737fa075ea014c0a1bcdee0498a9`.

## Issue 5: CuTe shim final-link propagation

Update on 2026-07-25: the existing PR
[#118](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/118) was rebuilt on
v0.9.1 `main` as two signed-off minimal commits. It changes only
`CMakeLists.txt` and `cmake/CuteDsl.cmake` (`+19/-3`), preserves
`CUDA_DRIVER_LIB` as `PRIVATE`, and is now reported by GitHub as mergeable.
Focused CMake property checks and `git diff --check` pass. Native Orin
compilation and repository CI remain pending.

Proposed title:

```text
[Bug][Build]: CuTe CUDA compatibility shim is not linked into final runtime consumers
```

Use `Bug report - C++ Runtime`.

Issue body essentials:

- A locally generated SM87 CuTe artifact correctly selects
  `trt_edgellm_cutedsl_cudart_shim`.
- The plugin links, but `_edgellm_runtime` and `llm_inference` omit the
  selected shim and retain unresolved `cudaLibrary*` symbols.
- Expected: the chosen compatibility target and wrap/driver requirements
  propagate to every final consumer.
- Show final link lines and `nm` evidence; avoid proposing an old DSL pin as
  the fix.

Proposed PR title:

```text
fix(cmake): propagate CuTe compatibility link requirements
```

PR relationship and scope:

```text
Fixes #<issue-5>
```

- generic `CuteDsl.cmake` target propagation only;
- no local CUDA shim implementation or Jetson-specific path;
- test both shim-selected and native CUDA-library-API configurations.

Prepared evidence: `codex/upstream-v091-fix-cute-final-link`,
`c84f7664639a68196229f544208ae8ed22c2f720`.

## Issue 6: checkpoint destination dtype

Proposed title:

```text
[Bug][Export]: checkpoint loader can replace declared FP16/BF16 parameters with FP32 tensors
```

Use `Bug report - Python Export Pipeline`.

Issue body essentials:

- `_set_tensor` converts FP16/BF16 source tensors but raw-assigns FP32
  sources.
- Reproducer: load an FP32 checkpoint tensor into a declared FP16 module;
  after loading, the parameter unexpectedly becomes FP32.
- Impact: mixed-dtype models, extra memory, export/build dtype mismatch.
- Expected: floating-point checkpoint tensors are converted to the declared
  destination dtype; non-floating and already matching tensors retain their
  intended behavior.

Proposed PR title:

```text
fix(checkpoint): cast floating tensors to the declared destination dtype
```

PR relationship and scope:

```text
Fixes #<issue-6>
```

- loader change plus CPU-only FP32-to-FP16, FP32-to-BF16, already-matching,
  and non-floating regression tests;
- no SparkTTS model support or mixed-precision feature wiring.

Prepared evidence: `codex/upstream-v091-fix-checkpoint-dtype`,
`cebc1542ab90000f3ba5e8b5f60daaeb5f71f9a6`.

This candidate should be submitted last: first rerun its CPU tests in a clean
PyTorch environment. The previous WSL PyTorch installation had an unrelated
NCCL symbol-load failure.

## Issue-only reproducers

These must not be bundled with the six fix PRs:

1. Qwen3.5 native simultaneous contexts cause Myelin
   `already loaded binary graph`. File a runtime issue only after reducing it
   to the official server/runtime, with one N=1 control and one N=2 failure.
   Do not propose a speculative scheduler patch.
2. Gemma 4 audio core support exists but the experimental server does not
   route Gemma 4 as a VLM. File only after an official E2B/E4B C++ audio path
   passes and the same artifact fails specifically at server routing.

Continuous batching is a feature request, not a bug PR. Qwen3-TTS Base, MOSS,
HTTP cancellation/chunking, worker protocols, and device recipes are product
extensions and remain out of the upstream bug queue.

## Submission sequence after owner approval

For each candidate, sequentially:

1. search open and closed upstream issues again to avoid a duplicate;
2. reproduce on current `main`;
3. fill the matching official issue template with the smallest public
   reproducer and exact environment;
4. present the final issue text to the owner;
5. create only the approved issue;
6. wait for NVIDIA engineer acknowledgement/approval;
7. rebase the minimal fix on current `main`, add focused tests, run
   pre-commit, and create a signed-off commit;
8. present the final diff and PR text to the owner;
9. push to the contributor fork and open the approved PR;
10. put `Fixes #<issue>` in the PR body and cross-link validation-only
    dependencies without combining their code.
