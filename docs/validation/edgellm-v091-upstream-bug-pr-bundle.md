# NVIDIA TensorRT-Edge-LLM v0.9.1 bug PR bundle

## Submission policy

Target upstream:
`https://github.com/NVIDIA/TensorRT-Edge-LLM.git`.

Base:
`v0.9.1` / `7f061f21f0a581ba234a1e233c9315b89d8e47d6`.

This bundle is preparation only. Do not push branches to NVIDIA, create a
GitHub pull request, or open an upstream issue without explicit owner
confirmation.

Model features and product extensions are deliberately excluded from the bug
queue: Qwen3-TTS Base enablement, external speaker embeddings, MOSS support
and its N=2 dispatcher, SparkTTS mixed precision, worker JSON protocols,
device recipes, and differentiated streaming chunks.

## Candidate order

| Order | Local patch | Proposed PR | State before owner review |
|---|---|---|---|
| 1 | `0034` | Fix Qwen3-ASR export config losing MRoPE semantics | Strongest candidate. Clean v0.9.1 reproduction, bad/good exported config evidence, fresh engine, exact transcripts, N=2 and 50-round service evidence exist. Add a small exporter unit test and remove historical wording from the commit message. |
| 2 | `0038` | Guard TensorRT `DataType::kFP4` on pre-10.8 headers | Small compile bug. TRT 10.3 device build passes. Before submission, compile the same clean commit against a newer TensorRT to prove FP4 behavior is unchanged. |
| 3 | `0037` | Use `IStreamReader` before TensorRT 10.7 | Generic compatibility bug. TRT 10.3 build and real-engine deserialization pass. Reduce the 93-line compatibility implementation if NVIDIA prefers a smaller adapter and add read/seek/error-path coverage. |
| 4 | `0033` | Cast FP32 checkpoint tensors to the declared FP16/BF16 destination | The patch-stack mail patch is not independently applicable, but a minimal clean-base replacement is now prepared on `codex/upstream-v091-fix-checkpoint-dtype`. Syntax and `diff --check` pass. Run its CPU tests in a working PyTorch environment before owner review; the current WSL PyTorch install fails to load because of an unrelated NCCL symbol mismatch. |
| 5 | `0040` | Scope context-FMHA cubin loading/cache by mask type | Device-positive evidence exists for non-custom-mask SM87 engines. Block submission until a true custom-mask negative test proves unsupported kernels still fail loudly. |
| 6 | `0039` | Propagate CuTe CUDA driver/shim link requirements to final targets | Real JP6.2 final-link/load failure and fixed build evidence exist. Split the generic CMake propagation from the local shim implementation before review. |

## Reproduction-only upstream defect

Qwen3.5 GDN native simultaneous contexts reproduce a Myelin
`already loaded binary graph` failure. The product service is safe through
singleflight and passed 50 abort/recovery cycles, but there is not yet a
minimal library fix. Prepare this as a self-contained NVIDIA issue/reproducer,
not as a speculative PR:

- exact official v0.9.1 SHA and engine config;
- two simultaneous client timeline;
- one successful N=1 control;
- failing log signature and process health;
- proof that the failure is independent of MTP;
- no Seeed service wrapper or product scheduler code.

Do not submit the issue without the same explicit owner confirmation required
for PRs.

## Clean-base preparation status

The following local-only branches were prepared from the exact official base.
They have not been pushed anywhere:

- `codex/upstream-v091-fix-asr-mrope`
- `codex/upstream-v091-fix-trt103-stream-reader`
- `codex/upstream-v091-fix-trt103-fp4-guard`
- `codex/upstream-v091-fix-checkpoint-dtype`
- `codex/upstream-v091-fix-cute-final-link`
- `codex/upstream-v091-fix-fmha-mask-cache`

Patches `0034`, `0037`, `0038`, `0039`, and `0040` apply independently to
the official SHA. The original `0033` mail patch does not, so the separate
minimal branch above replaces it for upstream review. All branch and PR work
remains local pending owner confirmation.

## Evidence attached to review

- ASR: `evidence/asr/quality-linear-fail.json`,
  `quality-mrope-wav.json`, `n2-service-50-wav.json`.
- MOSS is intentionally absent from this upstream bug bundle.
- Runtime/device matrix:
  `docs/validation/edgellm-v091-model-concurrency-matrix.md`.
- Patch classification:
  `docs/validation/edgellm-v091-patch-debt-matrix.md`.
- Full migration evidence:
  `docs/validation/edgellm-v091-production-migration.md`.

## Owner review checklist

For each approved candidate:

1. start from a clean branch at the exact official SHA;
2. apply only the candidate diff and its focused test;
3. remove local paths, product names, session links, co-author automation
   metadata, and patch-series numbering;
4. show clean `git diff --check` and the smallest relevant build/test matrix;
5. include the minimal reproducer, observed result, expected result, platform,
   and version boundary in the PR body;
6. show that no model feature or product protocol change is included;
7. present the final commit and PR text to the owner;
8. push or submit only after explicit confirmation.
