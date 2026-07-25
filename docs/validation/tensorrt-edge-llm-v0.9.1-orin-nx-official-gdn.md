# TensorRT Edge-LLM v0.9.1 — Orin NX official/GDN validation

Date: 2026-07-24  
Device: `orin-nx` (JetPack 6.2, TensorRT 10.3, CUDA 12.6, SM87)  
Upstream revision: `v0.9.1` / `7f061f21f0a581ba234a1e233c9315b89d8e47d6`

## Outcome

The unmodified v0.9.1 source does not build or run end-to-end on the documented
JetPack 6.2 stack. A clean build succeeds after four narrowly scoped compatibility
changes:

1. TensorRT `< 10.7` stream reader fallback.
2. TensorRT `< 10.8` guard for `DataType::kFP4`.
3. Static CuTe DSL CUDA-runtime shim/driver/wrap propagation.
4. Context FMHA cubin loading scoped by `ContextAttentionMaskType`.

The first three changes produce a clean plugin and tools build, and the Python
export of Qwen3.5-4B-AWQ succeeds. The fourth change is required because v0.9.1
ships two SM87 custom-mask FMHA cubins compiled for CUDA 12.8; the original
loader eagerly loads all FP16/SM87 cubins even when the Qwen plugin explicitly
sets `enable_vision_block_attention=0`.

## FMHA root cause

- All four generated SM87 XQA cubins relevant to Qwen head-size 256 load through
  the CUDA Driver API successfully.
- Nine ordinary source-packaged SM87 context-FMHA cubins also load.
- Exactly two source-packaged cubins fail `cuModuleLoadData` with
  `CUDA_ERROR_INVALID_IMAGE`:
  - `fmha_v2_flash_attention_fp16_fp32_64_128_S_q_k_v_256_custom_mask_sm87.cubin.cpp`
  - `fmha_v2_flash_attention_fp16_fp32_64_16_S_q_k_v_256_custom_mask_sm87.cubin.cpp`
- The failing cubins have a different CUDA ELF ABI marker (`...0141` versus
  `...0133` for every loadable SM87 cubin).
- The official `kernelSrcs/fmha_v2/README.md` says these SM87 cubins are
  generated using CUDA 12.8 from TensorRT-LLM commit `0ffa77a...` plus
  `gen_fmha_cubin.patch`. It provides no CUDA 12.6 generation recipe.
- Official v0.9.0 (`1ac0f2b99642045125e1c5ac7b109434ba3b36c7`) contains no
  `custom_mask_sm87` cubin counterparts. No older counterpart exists in the
  device's v0.8 source/assets.

The semantic fix keys the FMHA loader cache by data type, SM, and mask type.
CAUSAL/SLIDING Qwen instances therefore never load CUSTOM_MASK modules.
Vision-block CUSTOM_MASK instances still request that mask type explicitly and
still fail loudly if their required CUDA 12.8 cubin is invalid; errors are not
ignored.

## GDN status

Fresh model export and plugin creation instantiate the causal-convolution,
gated-delta-net, and Int4 plugins successfully. The original failure occurred
later in `AttentionPlugin`, not in GDN. With mask-scoped FMHA loading, the fresh
engine build completes successfully (`llm.engine`, 1,019,531,860 bytes).

Direct inference against that fresh engine passes:

| Mode | Result | Generation throughput | Peak unified memory |
|---|---:|---:|---:|
| deterministic short | rc=0, 9 tokens | 39.24 tok/s | 2,372.4 MiB |
| sampled | rc=0, 128 tokens | 35.40 tok/s | 2,374.7 MiB |
| long | rc=0, 512 tokens | 35.10 tok/s | 2,378.5 MiB |

The official experimental Python server also loads the exact fresh engine and
pybind successfully when the plugin is loaded by `PyLLMRuntime`, which is the
official load order. Preloading the plugin globally before importing the
pybind extension is invalid: duplicate CUDA registration/symbol interposition
caused `initializeMRopeCosSinKernel` occupancy lookup to report
`invalid device function`. The pybind and successful C++ binary contain the
same SM87 MRope cubin (SHA-256 `a298e4c8...`); removing only the non-official
preload made the server healthy.

Runtime results are mixed:

- 20/20 sequential SSE requests completed. Mean TTFT was 68.64 ms and mean
  end-to-end time was 113.27 ms.
- A true simultaneous two-client test failed. Both HTTP responses began and
  overlapped for 18.26 ms, but one stream completed with 96 content events
  while the other returned HTTP 200 plus `[DONE]` with zero content. The
  server logged the historical failure exactly:
  `Myelin ... Called with an already loaded binary graph`, followed by
  `Failed to execute base model for prefill step`.
- The server remained healthy and a subsequent single request returned the
  expected `recovered` response. This is therefore a concurrency-safety defect,
  not a permanent process crash.
- In a fresh server interval, the repository's deterministic abort harness
  passed 50/50 abort -> immediate-next cycles with zero CUDA, TensorRT, Myelin,
  assertion, or worker-exit log hits. Immediate-next delay was
  0.012-0.026 ms and every health check returned 200.
- Abort recovery has a latency concern: mean recovery TTFT was 421.09 ms,
  versus 66.60 ms for ten idle runs of the identical `recovery-ok` prompt
  (approximately +354.49 ms). The server log has no explicit cancellation
  acknowledgement, so the evidence proves stable recovery but does not prove
  prompt cancellation without a cleanup/synchronization wait.
- The v0.9.1 audit server and the original CUDA translator completed
  simultaneous functional requests with 835.57 ms overlap. Peak tegrastats RAM
  was 9,058/15,656 MiB, leaving 6,598 MiB headroom; both health endpoints
  remained healthy.

Official GDN+MTP support is statically applicable: the checkpoint has
`text_config.mtp_num_hidden_layers=1` and 34 `mtp.*` weight tensors. It was not
built in this run. The required isolated MTP export repeats the full MTP base
and adds `mtp_draft/`, then requires both `spec_base.engine` and
`spec_draft.engine`. The existing vanilla export and engine already occupy
3.9 GiB and 3.3 GiB. With only 16 GiB free and a mandatory 4 GiB restart floor,
a second full export, two engines, and TensorRT build temporaries cannot be
bounded safely; no MTP build was started.

The upgrade gate therefore remains blocked: vanilla GDN inference and serial
abort recovery pass, but the official server is not safe for simultaneous
clients, and production-equivalent MTP is not yet built or tested.

## Evidence on Orin NX

Evidence root:
`/home/harvest/validation/edgellm-v091-official-20260724T0418Z`

- `92-v091-jp62-combined-candidate.patch`: first three compatibility changes.
- `94-cutedsl451-propagation-clean-rebuild.log` and `95-...result.txt`: clean
  successful build.
- `99-v091-qwen35-awq-export.log` and `100-...result.txt`: successful fresh
  export.
- `107-v091-engine-retry-pluginpath.log`: original AttentionPlugin invalid-image
  failure with exact plugin path.
- `109-sm87-xqa-context-fmha-direct-load.txt`: direct Driver API cubin matrix.
- `110-mask-scoped-fmha-incremental-build.log` and `111-...result.txt`: successful
  three-file incremental build.
- `113-v091-engine-mask-scoped-retry.log` and `114-...result.txt`: successful
  single fresh engine build, rc=0.
- `115-mask-scoped-fmha-loader.patch` and `116-...summary.txt`: final patch and
  source/probe state.
- `117-*` through `120-*`: deterministic, sampled, and long direct-inference
  outputs/profiles and summary.
- `121-*` through `127-*`: isolated server launch provenance, exact pybind
  comparison, the invalid preload reproducer, and healthy official-load-order
  server.
- `129-server-smoke-result.json`, `126-server-no-preload.log`: 20 sequential
  passes and the simultaneous-client Myelin failure.
- `133-abort50-result.json`, `131-abort-server.log`,
  `133-abort50-tegrastats.log`: clean 50-cycle abort/recovery gate and raw
  memory evidence.
- `136-coexist-result.json`, `136-coexist-tegrastats.log`: functional
  v0.9.1-server plus translator overlap.
- `137-mtp-feasibility.txt`: MTP artifact and free-space assessment.
- `140-original-services-restored.txt`: final container identities, images,
  ports, health responses, and closed audit port.

The original containers were restored unchanged. `edge-llm-chat-service`
(`8f2af667...`, image ID `sha256:af219111...`) is healthy on port 8000 and
`translator` (`1f30af4b...`, image ID `sha256:c47bde5b...`) is healthy on
port 9001. Both retain `unless-stopped`; the audit port 18091 is closed.
