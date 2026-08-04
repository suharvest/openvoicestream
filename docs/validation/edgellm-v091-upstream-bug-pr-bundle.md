# NVIDIA TensorRT-Edge-LLM v0.9.1 bug PR bundle

## Submission policy

Target upstream:
`https://github.com/NVIDIA/TensorRT-Edge-LLM.git`.

Base:
`v0.9.1` / `7f061f21f0a581ba234a1e233c9315b89d8e47d6`.

This bundle began as preparation only. The owner subsequently authorized
creating the bug issues and directly opening their minimal linked PRs. The
submitted state is recorded below; unrelated upstream writes remain outside
that authorization.

The proposed one-issue/one-PR decomposition, titles, templates, dependency
rules, and submission order are in
`docs/validation/edgellm-v091-upstream-issue-pr-plan.md`.

Issue creation status on 2026-07-25:

- stream reader: NVIDIA/TensorRT-Edge-LLM
  [#140](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/140);
- FP4 guard:
  [#141](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/141);
- Qwen3-ASR MRoPE:
  [#142](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/142);
- FMHA mask-scoped loading:
  [#143](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/143);
- checkpoint destination dtype:
  [#144](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/144).

Linked PR status on 2026-07-25:

- [#147](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/147) closes #140;
- [#145](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/145) closes #141;
- [#146](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/146) closes #142;
- [#148](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/148) closes #143;
- [#149](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/149) closes #144;
- refreshed [#118](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/118)
  closes the pre-existing #117.

All six PRs are open, based on v0.9.1 `main`, and reported as mergeable.
GitHub currently reports no upstream CI checks for these fork branches.

Status rechecked read-only on 2026-08-05: issues #117 and #140-#144 remain
open; PRs #118 and #145-#149 remain open and `MERGEABLE`. GitHub reports
`mergeStateStatus=BLOCKED` for all six because they have not received the
required upstream review/merge authorization, not because of branch
conflicts. No checks are currently reported. No issue, PR, branch, comment,
or label was changed during this recheck.

CuTe propagation was not filed again: existing issue
[#117](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/117) is already
linked to the author's open
[PR #118](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/118), and
[PR #103](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/103) overlaps the
same propagation root cause. PR #118 was refreshed onto v0.9.1 `main` on
2026-07-25 and is now mergeable; PR #103 remains overlapping. Do not create a
third PR.

Model features and product extensions are deliberately excluded from the bug
queue: Qwen3-TTS Base enablement, external speaker embeddings, MOSS support
and its N=2 dispatcher, SparkTTS mixed precision, worker JSON protocols,
device recipes, and differentiated streaming chunks.

## Candidate order

| Order | Local patch | Proposed PR | State before owner review |
|---|---|---|---|
| 1 | `0034` | [PR #146](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/146): fix ASR MRoPE normalization | Submitted as a one-line production change plus two focused tests. Open and mergeable; model export/build/inference remains pending. |
| 2 | `0038` | [PR #145](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/145): guard TensorRT `DataType::kFP4` before 10.8 | Submitted as one file `+8/-0`. TRT macro compile/preprocess matrix passes. Open and mergeable; real SDK build remains pending. |
| 3 | `0037` | [PR #147](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/147): use `IStreamReader` before TensorRT 10.7 | Submitted as one file `+29/-8`, retaining the V2 device path on 10.7+. Open and mergeable; real 10.3/10.7 build remains pending. |
| 4 | `0033` | [PR #149](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/149): preserve checkpoint destination dtype | Submitted with seven CPU matrix tests; before `3 failed, 4 passed`, after `7 passed`. Open and mergeable. |
| 5 | `0040` | [PR #148](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/148): scope context-FMHA cubin loading/cache by mask type | Submitted with explicit mask arguments at every call site and no custom-mask fallback. Open and mergeable; CUDA positive/negative tests remain pending. |
| 6 | upstream patch `0001` | Existing [PR #118](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/118): propagate CuTe shim/wrap requirements through static targets | Refreshed on v0.9.1 as a two-file `+19/-3` minimal fix and now mergeable. Generic propagation only; `CUDA_DRIVER_LIB` remains private. Normalized Orin product A/B built all product targets without downstream `0039`; final link retained wrap/CuTe/shim/libcuda and `ldd -r` passed. Upstream CI remains pending. |

## Local consumption state

The exact seven commits behind PR #118 and #145–149 are now vendored and
byte-locked in the engine overlay. Local duplicate patches `0033`, `0034`,
`0037`, `0038`, and `0040` are retired. Local `0009` retains only its
BF16Linear tied-weight extension. Local `0039` is also retired and is not an
upstream candidate: Orin product A/B proved its downstream
`CUDA_DRIVER_LIB` PUBLIC edge redundant. PR #118 remains the minimal generic
shim/wrap fix proposed upstream.

This is a source-consumption change only; it does not imply that NVIDIA has
merged any PR. The build continues to start from pure official v0.9.1 and
applies the locked commits deterministically.

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

Gemma 4 audio has a second integration-level candidate that must remain a
reproducer until hardware validation: v0.9.1 contains Gemma 4 audio
model/runner/builder paths, but `experimental/server/engine.py` does not list
Gemma 4 in `_VLM_MODEL_TYPES`. First prove that an official E2B/E4B audio
engine works through the C++ path and fails only at server routing; only then
prepare the smallest routing/test fix. Do not turn this into a feature request
or submit it without owner confirmation.

The following are explicitly **not** NVIDIA bug PRs:

- product TTS HTTP disconnect/cancellation and chunk scheduling;
- continuous batching as a new scheduler feature;
- Qwen3-TTS Base, MOSS, or product worker protocols;
- DFlash enablement and other model roadmap requests.

## Clean-base preparation status

The following local-only branches were prepared from the exact official base.
They have not been pushed anywhere:

| Local branch | Prepared commit |
|---|---|
| `codex/upstream-v091-fix-asr-mrope` | `b7ac34f7cf051646fa48bed1eec347a1b7b7158e` |
| `codex/upstream-v091-fix-trt103-stream-reader` | `44da5b3bb580911f5450a8a943d9106503b10afe` |
| `codex/upstream-v091-fix-trt103-fp4-guard` | `8829601185f9ba9ce2a1ab80b5c1a358f76eaab1` |
| `codex/upstream-v091-fix-checkpoint-dtype` | `cebc1542ab90000f3ba5e8b5f60daaeb5f71f9a6` |
| `codex/upstream-v091-fix-cute-final-link` | `c84f7664639a68196229f544208ae8ed22c2f720` |
| `codex/upstream-v091-fix-fmha-mask-cache` | `e275a1068e82737fa075ea014c0a1bcdee0498a9` |

The prepared minimal changes corresponding to former local `0034`, `0037`,
`0038`, and `0040`, plus the independent PR #118 branch, apply to the official
SHA. Retired `0039` is deliberately excluded from upstream review. The
original `0033` mail patch does not apply independently, so the separate
minimal branch above replaces it for upstream review. All branch and PR work
remains local pending owner confirmation. A fresh WSL audit verified every
branch has merge-base
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`, passes `git diff --check`, and
has no configured upstream tracking branch.

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
