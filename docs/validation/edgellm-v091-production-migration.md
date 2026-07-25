# TensorRT Edge-LLM v0.9.1 production migration log

Date: 2026-07-25  
Branch: `codex/edgellm-v091-upstream-audit`  
Status: production cutover and whole-device voice gate complete; runtime
image rebuilt and deployed locally; external artifact/image publication
awaits explicit destination approval

## Decision

The product migration target is official TensorRT Edge-LLM v0.9.1 plus the
smallest required local patch stack. The v0.9.0 deployment remains the
rollback target until fresh v0.9.1 engines, model quality, concurrency, and
co-residency gates pass.

The retained rollback target is the v0.8 image
`edge-llm-chat-service:rollback-v080-20260724`; the previously planned v0.9.0
rollback is no longer the active production fallback.

The voice-service rollback container is additionally retained as
`seeed-voice-v091-rollback-b1kv1536`. The active rebuilt voice image is
`seeed-local-voice:v0.9.1-edgellm-runtime-20260725-0b8d966` with digest
`sha256:b5f31b3d7a124ce7d68378a7fc880432dae2191cac2822538428ca5a11a69a95`.

The previous audit conclusion that disk space was insufficient referred to
the peak temporary storage needed for a fresh export plus base and MTP draft
engines while preserving rollback assets. It was not a statement that MTP
cannot run on a 16 GB Orin NX. This migration run is authorized to remove
confirmed rebuildable audit/cache artifacts and continue the full build.

## Baseline audit carried into migration

- Pure official v0.9.1 does not compile/run unchanged on JP6.2/TRT 10.3 due to
  streamed-reader, FP4, CuTe artifact/link, and FMHA cubin-selection gaps.
- A 40-patch non-active candidate applies cleanly to exact official v0.9.1 and
  builds for SM87 with CuTe OFF and ON.
- Fresh vanilla GDN direct and serial-server validation passed, while two
  simultaneous official-server clients reproduced the Myelin already-loaded
  graph failure.
- Old v0.9.0 voice engines passed bounded ABI/mechanism smoke with candidate
  workers. Those results are not fresh v0.9.1 release evidence.
- MOSS and SenseVoice were blocked by missing local build assets.

## Migration checkpoints

### Repository source/build integration

Inner repository branch:
`third_party/jetson-voice-engine@codex/edgellm-v091-production-migration`.
Normalized patch-stack commit: `b9ca87d`; provenance hardening:
`4693afc`; pinned source head: `6361606`. Exact patch/series/LOCK/checksum
inputs are marked `-text` so Git cannot rewrite release bytes. Exact upstream
mail patches additionally use `-whitespace`; release gate scripts are pinned
to LF so they remain executable in `core.autocrlf=true` checkouts.

- `UPSTREAM_PIN` now selects exact official v0.9.1 SHA
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6`.
- The active build path first validates and applies seven byte-locked exact
  commits from NVIDIA PR #118 and #145–149, then applies the explicit sparse
  36-patch local product series. Generic duplicates `0033`, `0034`, `0037`,
  `0038`, and `0040` are retired. Local `0009` keeps only BF16Linear
  tied-weight support; local `0039` keeps only the CUDA-driver propagation
  residual pending final-link A/B validation.
- Both series have SHA-256 sidecars. The proposed-upstream `LOCK` additionally
  records official repo URL, PR, commit, parent, tree, and stable patch-id.
  v080/v090 files remain rollback/history and are not referenced by v0.9.1.
- The three legacy placeholder manifests now express exact v0.9.1/SM87/JP6.2
  contracts and require provenance plus SHA-256 sidecars.
- The device engine entry point now produces separate ASR b1 and b2 engines,
  includes Base TTS speaker-encoder export/build, and exposes distinct GDN base
  and GDN+MTP modes. The MTP mode intentionally fails unless an executable
  `EDGELLM_MTP_BUILD_SCRIPT` is supplied; no unverified upstream CLI was
  invented.
- Four additive v0.9.1 profiles, three v0.9.1 leaves, and a fail-loud
  `/opt/edgellm-v091` Docker overlay were added. They contain no v080/v090
  runtime paths.

Apply-only verification used a fresh generated checkout cloned from the
preserved exact-v0.9.1 local source:

```text
UPSTREAM_PIN: 7f061f21f0a581ba234a1e233c9315b89d8e47d6
vendored format-patch comparison: 7/7 byte-exact to NVIDIA PR refs
forward replay: 7/7 proposed-upstream + 36/36 local product
reverse replay: 36/36 local + 7/7 proposed-upstream
post-reverse: official tracked tree + addon-only untracked tree
build.sh --apply-only: pass
generated-tree Python py_compile and git diff --check: pass
full-build manifest gate: TOML parse + upstream/local source hash verification
negative gates: missing manifest, stale hash/entry, LOCK/SHA order-set mismatch
official objects: 7/7 parent/tree/patch-id verified
core.autocrlf=true fresh clone: locked bytes and integrity pass
```

The already-staged 41-patch artifact set is immutable historical evidence and
was not rewritten. The normalized source identity requires its own Orin build,
runtime regression, artifact prefix, and manifest before publication.

Local validation:

```text
bash -n build.sh build-engines-for-device.sh: pass
profile JSON parse: 4/4 pass
leaf YAML parse: 3/3 pass
overlay/profile pytest: 35 passed
leaf registry pytest: 69 passed
composition boot pytest (clean env): 11 passed
outer and inner git diff --check: pass
```

At this checkpoint, v0.9.1 was the active repository build contract only. Fresh
engine builds, complete runtime gates, isolated release image validation,
production cutover, and rollback soak remain pending and must be appended
below.

### Device-preflight contract correction

The first device preflight found build contracts that could otherwise produce
false progress. The active integration now fails early as follows:

- CustomVoice INT4 requires a reviewed executable driver and preserves its
  revision/provenance with the talker engine; the nonexistent
  `tts-int4-drivers` path is no longer referenced.
- ASR b1/b2 defaults are restored to production-equivalent 1024 input / 1536 KV
  capacity; long context requires an explicit opt-in.
- SparkTTS has separate BF16 and W4A16 entry points and artifact roots. Both
  require a passing 32-token gate plus driver revision/provenance.
- `audio_build` is explicitly built and checked.
- quant/export runs a CUDA compatibility probe and rejects CUDA-major mismatch
  such as torch CUDA 13 on the JetPack 6.2 CUDA 12.6 driver.

### Orin NX cleanup and rollback checkpoint

Device baseline: Orin NX 16 GB, JetPack 6.2 / L4T 36.4.3, CUDA 12.6, and
TensorRT 10.3.

- Before cleanup: approximately 16 GB free.
- The running v0.8 image was retained under immutable rollback tag
  `edge-llm-chat-service:rollback-v080-20260724`.
- The original chat-service and translator container inspect data, health
  state, and effective environment were captured before stopping them.
- Six stale July 4 demo processes backed by
  `/home/harvest/project/v090-export` were stopped. They are not production
  services and will not be restored.
- Only verified rebuildable v0.9.1 audit/build workspaces, the stale 22 GB
  v0.9.0 export workspace, and two empty anonymous Docker volumes were
  removed. Accepted/release/rollback assets, model stores, the candidate
  source tree, and the local SM87 CuTe build were preserved.
- After cleanup: 41 GB free (43,064,995,840 bytes).

Device evidence root:
`/home/harvest/validation/edgellm-v091-gdn-mtp-migration-20260724T103611Z`.

### Fresh v0.9.1 GDN plus MTP engines

Release candidate root:
`/home/harvest/edgellm-workspace/v091-release-candidate-qwen35-4b-awq-mtp-20260724T103611Z`.

The model was freshly exported with official v0.9.1 MTP structure and
external INT4 FFN weights. The export contains distinct `llm` and
`mtp_draft` ONNX roles with `spec_decode_type=mtp`.

Both TensorRT engines then built and serialized successfully:

| Engine | Build time | Size | SHA-256 prefix |
| --- | ---: | ---: | --- |
| `spec_base.engine` | 192.185 s | 978 MB | `0e82d742` |
| `spec_draft.engine` | 101.205 s | 339 MB | `b65b5cb7` |

TensorRT reported approximately 3,460 MiB peak GPU allocator use for each
build. The complete engine directory is 3.7 GB and the device retained
approximately 33 GB free afterward. There were no build errors. TensorRT 10.3
reported non-fatal default `output_dtype` plugin-attribute and DLA-profile
fallback warnings.

The missing `llm_inference` binary was supplied by building only that existing
CMake target in the preserved CuTe build tree; neither engine was rebuilt.
Passing the MTP spec directory directly to the official runtime then produced
the deterministic answer `2 plus 2 equals 4.`.

Measured on that prompt:

| Metric | Vanilla v0.9.1 | MTP v0.9.1 | Delta |
| --- | ---: | ---: | ---: |
| Generation throughput | 39.2397 tok/s | 48.2558 tok/s | +22.98% |
| Peak unified memory | 2,372.4 MiB | 2,393.26 MiB | +20.9 MiB |

MTP generated 9 tokens over 3 iterations and accepted an average of 3.0 draft
tokens per iteration. With `draftStep=3`, this test reached the configured
maximum acceptance. Prefill throughput was 393.49 tok/s.

The isolated HTTP wrapper passed two simultaneous clients (2/2 HTTP 200,
non-empty SSE output, and `[DONE]`) through its intentional single-flight
gate. A 50-cycle abort/recovery test completed without a failed recovery;
post-abort recovery TTFT averaged 4,651.7 ms and health remained HTTP 200.
The original chat-service and translator were then restored healthy.

### Fresh voice-engine preflight

The device has enough space for a sequential migration, but not for retaining
all downloaded checkpoints, quantization intermediates, ONNX trees, and final
engines simultaneously. A 4 GB free-space floor will be enforced and only
redownloadable/rebuildable intermediates will be removed after their release
artifacts have passed their gates.

Ready or fully sourced:

- MOSS: the complete TRT-fixed TTS and audio-tokenizer ONNX bundle is present,
  including the formerly missing tokenizer decode-step graph. Six plans are
  expected.
- SenseVoice: canonical scaled/fixed ONNX plus its MVN, embedding, and BPE
  assets have an exact acquisition/build recipe.

The Qwen3 ASR, CustomVoice, Base, and Spark checkpoints are not cached on the
device. Their exact revisions were recovered from provenance and together
require approximately 10.84 GB of downloads. Existing v0.9.0 engines and ONNX
trees remain rollback/quality references, not fresh-v0.9.1 evidence.

Preflight also found migration-script defects that must be corrected before
execution:

- CustomVoice INT4 referenced two driver scripts absent from both the
  candidate source and device. The active path must fail loudly or use an
  explicit reviewed driver hook.
- The unmodified official v0.9.1 exporter rejects Qwen3-TTS Base with
  `Only Qwen3-TTS CustomVoice checkpoints are supported`, and the previously
  referenced `tensorrt-edgellm-export-audio` speaker-encoder CLI does not
  exist. The build contract now requires an explicit Base driver and a
  separately provenanced speaker-encoder ONNX directory instead of claiming
  these are official v0.9.1 export stages.
- Conversely, the exact official v0.9.1 exporter successfully exported the
  pinned CustomVoice checkpoint
  `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice@85e237c12c027371202489a0ec509ded67b5e4b5`
  in FP16. Talker, code predictor, and code2wav all passed full external-data
  loading and `onnx.checker`; required payload sizes are 887,226,368,
  157,351,936, and 228,196,352 bytes. This establishes the official FP16
  baseline while leaving the production INT4 talker behind the reviewed
  `EDGELLM_TTS_INT4_DRIVER`.
- The removed temporary export environment used a CUDA 13 PyTorch build on a
  CUDA 12.6 driver. GPU quantization must run in a proven compatible
  environment; a CUDA initialization probe is mandatory.
- ASR had drifted from the production-equivalent 1024 input / 1536 KV limits
  to 4096 / 4096. The first migration build will preserve 1024 / 1536;
  long-context remains a separate opt-in experiment.
- Spark described only one default precision. The migration contract requires
  explicit BF16 and W4A16 outputs with independent provenance and token gates.
- The v0.9.1 `audio_build` target must be built before audio engine generation.

These contracts were corrected before device execution; no voice build is
credited as migrated by bypassing them.

### Fresh Qwen3-TTS Base export

The Base extension was exported in a WSL worktree based on exact official
v0.9.1 SHA `7f061f21f0a581ba234a1e233c9315b89d8e47d6`. The only exporter
behavioral change was to allow the already-implemented Talker,
CodePredictor, and Code2Wav component paths for `tts_model_type=base`.

Inputs:

- Base model:
  `Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992436eae1d760afd27aff78a71d676296fc`;
- INT4 Talker stage 2:
  `wip/native-int4-talker@ff2318e66525365b2ed9f55811bf5d2381280ed8`;
- export driver SHA-256 prefix: `6d986a`;
- WSL output:
  `/home/harve/project/edgellm-v091-base-export-int4-20260725`.

All three graphs loaded their external data and passed `onnx.checker`. The
Talker has 1,643 nodes, including 196 `Int4GroupwiseGemmPlugin` and 28
`AttentionPlugin` nodes. Selected artifact hashes:

| Component | Artifact | SHA-256 |
| --- | --- | --- |
| Talker | `model.onnx.data` | `0aa0695193a24f3d4a63f06bd50f4568540cb14372521c2bd3242f1b592e8fbe` |
| CodePredictor | `model.onnx.data` | `829a59622ed1e521628f268147d34fdad5645c9ef5daebfe383f2f1de94c7603` |
| Code2Wav | `model.onnx.data` | `2e8c88b6145a0e1ec2746e7d8cfbcb88af60547f2487be457866e092a50a34d4` |

The transferred Orin NX copies matched these hashes. This establishes that
the official v0.9.1 internal component exporters support Base and that the
unconditional CLI guard was the sole blocker for these three stages.
Speaker-encoder export and external-embedding transport remain the deliberate
Seeed extension boundary.

### Fresh Qwen3-TTS Base engines on Orin NX

With the production chat and translator containers stopped, all four Base
engines were built from the fresh artifacts above:

| Engine | Build result | Size | SHA-256 |
| --- | --- | ---: | --- |
| INT4 Talker | batch 2, input/KV 4096 | 239 MiB | `406fa2ff986e665083fbe93e328668da6fd3a9610006293d8669ae9a50bb47bd` |
| CodePredictor | batch 2, input/KV 4096 | 183 MiB | `5944225edf48d6bfa5930c181ab7d7c235d4c20f212cbe3a5b0e139a95c6ee54` |
| Code2Wav | official dynamic profile | 224 MiB | `f9b8014fc0dd5f374360bf571f813edefe014f43ea98b2e9024e5a3e7a69e1a6` |
| speaker encoder | FP16, mel time 10/555/2000 | 19 MiB | `755a023e9e80ee00c44480317d4a65fadafff4e1b3d809850c835a199a07282f` |

The clean Talker build initially failed with `Plugin not found`; explicitly
loading the Edge-LLM plugin before invoking `llm_build` fixed the issue. The
production builder now applies this requirement consistently instead of
depending on a previously loaded process.

Code2Wav built in 1,438.32 seconds. TensorRT skipped tactics requiring more
than the available unified memory, selected a feasible tactic, and serialized
successfully. Peak TensorRT allocator use was 7,500 MiB GPU and 2,953 MiB CPU.

The source speaker encoder
(`544da28fd7463a3fd722d54630fe39b730376b6aee850ca66845ee981e87a40f`)
contains an exporter-generated terminal `If`. TensorRT 10.3 rejects its
statically mismatched branch annotations `[1,1024]` and `[1,1024,1]`. The
versioned compatibility transform replaces that conditional with its
batch-one `Squeeze(axis=2)` branch. ONNX Runtime comparison at mel lengths 10,
555, and 2000 was bit-exact (`max_abs=0`) before the transformed graph
(`e29fe675083396136b90e6be6baef7f7362de3a2f17e16b0e48c5efee1054906`)
was built. The resulting engine loaded and executed successfully at all three
profile points; typical length 555 measured approximately 2.77 ms GPU
latency.

Build evidence:
`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/evidence/tts-base-int4-20260725/build`.

### Fresh Qwen3-TTS Base runtime, quality, and concurrency

The reference WAV was processed by the fixed speaker encoder with the official
24 kHz magnitude-STFT and Slaney-normalized mel pipeline. The rebuilt
1024-element FP32 embedding has L2 10.5675. The formal concurrency and mixed
service gates use this rebuilt embedding.

Three ordinary Chinese prompts and two English prompts passed SenseVoice
roundtrip with normalized similarity 1.0. A Chinese numeral prompt was
transcribed as the semantically equivalent Arabic digits and is retained as
evidence but not counted as one of the three strict string-normalized passes.
After the production containers were restored, an additional Base N=1 sample
using the rebuilt embedding transcribed exactly as
`新版基准音色验证通过`, proving the current encoder/embedding path while the
production LLM and translator remained co-resident.

Speaker-encoder cosine similarity between the real reference and the five
Base outputs was 0.9328-0.9475 across Chinese and English. This is useful
quantitative timbre evidence, but not a calibrated speaker-verification
score: one CustomVoice Serena control scored 0.9511 while a Dylan control
scored 0.9230. Product acceptance should therefore retain the audio samples
for listening and must not turn this cosine range into an identity threshold.

The INT4 Base worker then passed the direct two-slot gate:

- full simultaneous N=2 requests overlapped;
- each concurrent PCM SHA-256 was byte-identical to its own single-request
  baseline and distinct from the other prompt;
- in every round, request A was cancelled after streaming began, request B
  completed byte-identically to its baseline, and an immediate B recovery was
  also byte-identical;
- 50/50 uninterrupted rounds passed with zero CUDA/TensorRT stderr hits.

Mixed-service results define the supported scheduling boundary:

| Workload | Result |
| --- | --- |
| Base N=2 active + SenseVoice load/inference | pass; transcript similarity 1.0 |
| Base N=1 active + GDN/MTP | pass; GDN answered `2 plus 2 equals 4`, mixed and recovery TTS PCM matched baseline, and execution windows overlapped |
| Base N=2 active + GDN/MTP | not supported concurrently; GDN completed correctly, but TTS cancel/keep round 32 exceeded the 45 s gate |

The last row is a capacity/scheduling failure rather than silent corruption.
Production must cap Base to N=1 while a GDN request is active, or serialize
GDN against the Base N=2 pool. Base N=2 remains independently validated.

Runtime evidence:
`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/evidence/tts-base-int4-20260725/runtime`.

### Fresh v0.9.1 MOSS engines and quality gate

All six MOSS TensorRT plans were rebuilt on the Orin NX against TensorRT 10.3
and the v0.9.1 worker/plugin:

- global prefill and decode-step: FP32;
- local decoder, cached step, and fixed sampled frame: FP16;
- codec decode-step: FP32.

The mixed precision is deliberate. A blanket FP16 build had previously
produced silent audio while still returning success. Every new plan
deserialized successfully. The release was promoted atomically to:

`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/engines/moss`

Three Chinese synthesis samples passed:

| Sample | Audio duration | TTFA | Wall time | RMS | Clipping |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.32 s | 120 ms | 1,006 ms | 0.03628 | 0 |
| 2 | 1.68 s | 94 ms | 407 ms | 0.06059 | 0 |
| 3 | 4.40 s | 95 ms | 993 ms | 0.05157 | 0 |

The samples were additionally transcribed by the freshly rebuilt SenseVoice
engine below. All three matched their requested Chinese text exactly
(normalized similarity 1.0), closing the silent/noise false-success gap.

The original v0.9.1 MOSS JSONL loop was confirmed synchronous: a second
request could not emit `ready` before the first request completed, and a
cancel message was not consumed during generation. This is a product model
extension rather than an NVIDIA v0.9.1 regression.

The local extension now uses a fixed two-thread dispatcher, mutex-protected
JSONL output, request-ID scoped cancellation state, an explicit 4429
saturation response, and a short preprocessing lock around the shared
SentencePiece/ORT codec objects. On Orin NX it passed 50/50 rounds of true
N=2 overlap, cancel-A/continue-B, and immediate recovery. Every cancellation
was observed after one chunk, the worker returned zero, and stderr contained
no CUDA error. The initial concurrent pair was non-silent and distinct;
SenseVoice semantic loopback scored 0.9565 and 0.9545 normalized similarity.

Production may therefore advertise MOSS N=2 only with the extended worker
whose capability event reports `max_slots=2`, `concurrent_dispatch=true`, and
`cooperative_cancel=true`. Evidence:
`evidence/moss/runtime-smoke/moss-v091-n2-50.json`. The earlier negative
capability report remains as proof of the limitation in the unextended
product worker.

Evidence:
`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/evidence/moss`.

### Fresh SenseVoice engine and quality gate

The scaled/fixed Jetson ONNX was pinned to
`harvestsu/sensevoice-rknn@3dedec6aeb8b9c541d573d0b87237f64d1691f5c`.
The 937,596,352-byte ONNX has SHA-256
`ebfdbe962ce2fc5707821792cce39251e210bbf45c7482f3151765ae2da33df9`,
matching its authoritative repository object.

The project model downloader's production build path generated an FP16
TensorRT 10.3 plan with a 3 GiB workspace:

```text
size: 492,466,348 bytes
sha256: 1426038843761e0cf8571772fecdc22d770074ae40f0c3ff4389cc47384fa568
input:  speech FLOAT [1,344,560]
output: encoder_out FLOAT [1,344,25055]
deserialize: pass
```

While the production chat and translator services were co-resident, inference
on the three fresh MOSS samples took 123.50 ms, 54.05 ms, and 122.08 ms.
Every logit tensor was finite and each transcript exactly matched the source
text, so the former Chinese FP16 overflow concern did not reproduce.

Release:
`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/engines/sensevoice`.

Evidence:
`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/evidence/sensevoice`.

### Official ASR quantization/export environment

Quantization and ONNX export use an RTX 3060 WSL2 host only as a build host;
the Orin NX remains the required TensorRT build, runtime, quality, concurrency,
and co-residency target.

The host checkout is a clean exact official v0.9.1 tree at
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`. Its environment uses Python
3.12, PyTorch 2.12.0+cu130, and NVIDIA ModelOpt 0.44.0. A CUDA probe passed
on compute capability 8.6 with 12 GB VRAM. Quantization uses the pinned
`Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0`
checkpoint and the official `tensorrt-edgellm-quantize llm` CLI with
`int4_awq` and 512 samples.

An upstream packaging defect was found independently of the model path:
official v0.9.1 declares Python `>=3.10` while pinning `numpy==2.4.6`, which
requires Python 3.11 or newer. A universal `uv sync` therefore cannot resolve
the advertised Python 3.10 split. The migration uses Python 3.12 without
modifying the official source; this is a focused upstream PR candidate.

The official quantization completed in 357.8 seconds:

```text
quantized checkpoint: 887 MiB
config.json sha256:
  9711fff05648b0c31739e0ad6e8984b25f69b25f3a3716833e779d978e3ef962
model.safetensors sha256:
  00dc1094fb439b9122f95af8a77b2c41ebd2b9bc94575458b8d4dac66d4f851f
```

The unmodified official `tensorrt-edgellm-export` CLI then exported both
thinker and audio roles with FP8 embedding. The thinker config identifies
Edge-LLM `0.9.1`, `qwen3asrthinker`, FP16 KV cache, and the expected audio
token IDs. The output occupies approximately 1.1 GiB:

| Role/file | SHA-256 |
| --- | --- |
| LLM config | `6c3a4928b202a5be36a3281a82091db51c68a8cd5f3956f2541088dd515b941a` |
| LLM ONNX | `19d9d3e468089b4e2fd867a8b1396a4bd89557bf55186905465821f593583d1d` |
| LLM external data | `1a5a56aa62b1dda08f549ddba75f36cdcd9db3daef2b97871c5a11e254730e20` |
| FP8 embedding | `d36c095aa89c56d4a28e264fbb167439e6d54a7a0c6b8aa025eea24f44259974` |
| Audio config | `35910552df484d9d2d36192d362947b90eae6b1036c642a0a74806dc6d33695f` |
| Audio ONNX | `62a38322519e49cd3c088554ac3384dfddbe4a6e82c4f099f6fcd31903fed000` |
| Audio external data | `b1bac041ca95d5f90702b82ef47f69f71a341c9fc404778755f38d2994f7c3ec` |

The source and final Orin copies pass full ONNX external-payload loading and
`onnx.checker`. A directory-level direct transfer briefly overlapped a later
single-file retry and truncated the audio external-data file. The gate caught
the mismatch before a release build was credited. After all transfer
processes exited, the file was retransferred, remained stable at 378,011,648
bytes, and passed the full checker on Orin.

Using the v0.9.1 CuTe build on Orin NX, both production-capacity thinker
engines built successfully:

| Engine | Batch | Input / KV capacity | Build time | Peak TRT GPU |
| --- | ---: | ---: | ---: | ---: |
| thinker-b1 | 1 | 1024 / 1536 | 53.15 s | 580 MiB |
| thinker-b2 | 2 | 1024 / 1536 | 44.40 s | 580 MiB |

The official `audio_build` target was compiled from the same v0.9.1 CuTe tree.
The audio engine then built in 64.41 seconds and deserialized successfully.

### Fresh v0.9.1 ASR runtime and N=2 service gate

The unmodified official export contains a Qwen3-ASR configuration defect:
`rope_scaling.type` and `rope_scaling.rope_type` are emitted as `linear` even
though `mrope_section` is present. The resulting engine builds, but the
official worker fails at runtime because `mropeCosSinOut` is not supplied.
Normalizing both fields to `mrope` changes the exported config SHA-256 from
`6c3a4928b202a5be36a3281a82091db51c68a8cd5f3956f2541088dd515b941a`
to
`c5f622ec6fb9a02ce21ab28277cd9244ab28eafe9ab97f1827fc2b37d438aa41`.
This is already represented by the candidate export patch and remains a
high-value upstream PR.

The corrected release engines are:

| Engine | Batch | Size | SHA-256 |
| --- | ---: | ---: | --- |
| thinker | 1 | 555,099,444 bytes | `2be3dcfda6e6cdb2dfb9370bef6fba865ff4593189821dbb4a4b123ac7a4f2d6` |
| thinker | 2 | 555,099,444 bytes | `8cb75878263bb12506c97e7ba68c2f831dae3ba04c12d19d8597e03d58ec9d06` |
| audio encoder | shared | 377,790,212 bytes | `78ceb1a384dab396530d9fac669dfb19982c8ed0476cb809bf21ac5368fd56c8` |

Direct v0.9.1 worker inference on all three fresh MOSS samples produced the
exact requested Chinese text (normalized similarity 1.0). Measured latencies
were 856.24 ms, 100.22 ms, and 218.07 ms; the first result includes cold
initialization.

The service-layer migration exposed a second version boundary. The old
voxedge backend sent host-generated `mel.safetensors`, which is the v0.8
protocol; the v0.9.1 worker expects WAV/PCM. That mismatch consistently timed
out after 60 seconds. Running the current voxedge WAV-ingest path with
`EDGELLM_REQUEST_AUDIO_WAV=1` fixed the boundary without changing the worker
or engine.

On the actual FastAPI `/asr` route, the corrected batch-2 stack passed:

- 50/50 paired rounds with two different WAVs and HTTP 200 for both;
- exact transcript identity in all 100 concurrent responses;
- positive request-window overlap in every round (hot overlap approximately
  117-120 ms);
- 50/50 immediate post-pair recovery requests;
- zero CUDA, TensorRT, worker-exit, assertion, or cross-talk log hits.

A separate three-client observation found that `/asr` queues the third request
inside `WorkerIO` and eventually returns HTTP 200 instead of rejecting it.
The two short requests completed in about 124 ms and the queued requests in
about 438 ms. This is a product admission-control gap, not an engine-capacity
failure; the N+1 rejection gate remains open.

Two additional old product-layer issues were fixed or isolated:

- an ASR-only profile was incorrectly clamped to N=1 by the absent TTS
  backend's conservative default. The active resolver now treats an absent
  modality as the neutral `+inf` capability while preserving N=1 for a
  declared backend whose capability resolution fails;
- the old engine resolver deletes an existing engine when its private
  `.meta.json` sidecar is absent. Fresh v0.9.1 validation therefore used
  explicit versioned paths and skipped that resolver; formal image packaging
  must generate provenance before enabling resolver ownership.

Release:
`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/engines/asr`.

Evidence:
`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/evidence/asr`.

Targeted product tests pass: 68 capability/config tests and 4 concurrency
harness tests. The original `edge-llm-chat-service` and `translator`
containers were restored healthy after the isolated gate. After verifying the
release hashes and confirming no process or container used the sources, the
14 GB temporary service copy and approximately 3.1 GB of duplicate ASR engine
work directories were removed. The release, ONNX inputs, evidence, rollback
image, and model stores remain; device free space is 26 GB.

### Official Qwen3-TTS export and Orin NX build checkpoint

The exact official v0.9.1 checkout was also tested against the two Qwen3-TTS
model families, without relabeling historical ONNX as a fresh export.

The official v0.9.1 CLI does not expose Base export. It rejects the pinned Base
checkpoint before component export with:

```text
Only Qwen3-TTS CustomVoice checkpoints are supported.
Got tts_model_type='base'.
```

This is conclusive for the public CLI, but not yet for the internal Talker,
CodePredictor, and Code2Wav export stages. Those stages must be invoked directly
on the untouched official checkout before deciding whether the guard is the
only exporter blocker.

The earlier migration script additionally referenced a nonexistent
`tensorrt-edgellm-export-audio` command for the Base speaker encoder. The
device build contract now fails closed unless it receives both a reviewed
Base export driver and an explicitly versioned external speaker-encoder ONNX
bundle. Base therefore remains a local integration rather than an official
v0.9.1 path. The upstream split and acceptance gate are recorded in
`docs/specs/edgellm-v091-qwen3-tts-base-upstream-plan.md`.

A dedicated Base extension driver is now present at
`third_party/jetson-voice-engine/engine-overlay/drivers/export-qwen3-tts-base-v091.sh`.
It is wired for the production INT4 target and will not run without an exact
v0.9.1 checkout, immutable Base and stage-2 revisions, the Base exporter
extension, complete ONNX external-data validation, and generated provenance.
No Base engine is credited until this driver runs on WSL and its fresh outputs
pass the Orin acceptance gate.

In contrast, unmodified official v0.9.1 successfully exported the pinned
CustomVoice checkpoint
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice@85e237c12c027371202489a0ec509ded67b5e4b5`
in FP16. Full external-data loading and `onnx.checker` passed for all three
components:

| Component | External payload | Tensor count |
| --- | ---: | ---: |
| talker | 887,226,368 bytes | 254 |
| code predictor | 157,351,936 bytes | 46 |
| code2wav | 228,196,352 bytes | 231 |

The transfer archive SHA-256 was
`edf9919b7a2e25b166fd4ce8bf0fa681da70699b97ab718bb40ecbb6625ed233`.
The WSL and Orin copies matched exactly. After successful extraction and
payload verification, the redundant 2.07 GB Orin archive was removed; the WSL
copy remains recoverable. Available device space increased from 21 GB to
23 GB.

With the production and translator containers stopped, the v0.9.1
SM87/CUDA-12.6 CuTe build produced:

| Engine | Capacity | Build result | Build time | Peak TRT GPU |
| --- | --- | --- | ---: | ---: |
| talker FP16 | batch 2, input/KV 4096 | pass, 862 MiB engine | 92.04 s | 846 MiB |
| code predictor FP16 | batch 2, input/KV 4096 | pass, 183 MiB engine | 17.73 s | 180 MiB |
| code2wav FP16 | official default profile, code length 1/300/2000 | pass, 234,379,004-byte engine | 1,528.01 s | 7,500 MiB |

The talker output also contains the official tokenizer, processed chat
template, and projection/embedding sidecars. The code predictor contains its
codec embeddings and LM heads. Both builds completed without a TensorRT,
CUDA, or CuTe error.

The first code2wav invocation lost its client socket after several minutes
and left no engine, so it is not credited as a pass. A second invocation used
a persistent device log and explicit exit-code file. It completed with exit
code zero after approximately 25.5 minutes. TensorRT skipped tactics needing
11.25-16.875 GiB of device memory, selected a feasible tactic, and serialized
the engine normally. Its final activation-memory plan is 2,949,122,048 bytes.
The long silent interval is therefore expected tactic profiling rather than a
hang.

Fresh engine hashes:

| Engine | Bytes | SHA-256 |
| --- | ---: | --- |
| talker | 903,779,444 | `6ffd81e5ee3b5127abc5049ee77ae458b52ef8420ca003b7fde381d5c036472e` |
| code predictor | 191,571,628 | `01db396b976ad52a500a0fcd699ffe085dee94f80a806a8461b691b0a22cdea6` |
| code2wav | 234,379,004 | `7a38d97c760dc2b9bde2bf3f9bfd62ec4ffb8939fc1d0c26a0dce6f674439a85` |

### Official CustomVoice FP16 runtime gate

A one-slot v0.9.1 CuTe worker loaded the fresh engines in 6.63 seconds. Four
quality prompts completed with no CUDA error, worker exit, clipping, or empty
audio:

| Prompt | Frames | Duration | PCM RMS | Peak | SenseVoice similarity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chinese 1 | 46 | 3.68 s | 1,020.9 | 16,975 | 1.0 |
| Chinese 2 | 77 | 6.16 s | 1,556.7 | 11,368 | 1.0 |
| English 1 | 33 | 2.64 s | 1,369.9 | 7,368 | 1.0 |
| English 2 | 59 | 4.72 s | 1,936.0 | 12,800 | 1.0 |

The normalized SenseVoice transcripts exactly matched all four requested
texts. Two named speakers (`serena` and `dylan`) produced different PCM
digests for the same prompt. Language-explicit Chinese and language-omitted
Chinese were byte-identical because Chinese is the worker default; this is
expected and is not evidence that the language row is ignored. The separate
Chinese/English roundtrips prove both requested language paths.

Cooperative cancellation tripped after streaming began, emitted a cancelled
terminal event, and an immediate follow-up request reproduced the known-good
PCM digest. The worker then exited cleanly with zero CUDA errors.

The same FP16 engines cannot open two full slots on this 16 GB Orin NX. During
the N=2 preload the kernel OOM killer terminated
`qwen3_tts_streaming_worker` (`total-vm` approximately 18.3 GB). N=1 is
functionally correct, but FP16 is not a production concurrency candidate.
Production N=2 therefore still requires the version-matched INT4 talker
driver/artifact; this official FP16 path is the upstream correctness and CuTe
baseline only.

Evidence:
`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/evidence/tts-customvoice-fp16`.

### CustomVoice INT4 debt reduction and N=2 gate

The historical INT4 quantization is not coupled to the v0.8 runtime. A
preserved CustomVoice stage-2 checkpoint from
`wip/native-int4-talker@ff2318e` was passed directly to the unmodified
official v0.9.1 `tensorrt-edgellm-export --components talker` command. Export
completed successfully. Full external-data loading and `onnx.checker` passed
on the resulting 1,643-node graph, which contains the expected
`Int4GroupwiseGemmPlugin` nodes.

This narrows the production INT4 debt substantially:

- runtime, worker, plugin ABI, model config, and ONNX exporter can all use
  official v0.9.1;
- the local remainder is the ModelOpt quantization plus checkpoint key
  reassembly recipe;
- the recipe must be moved out of the old fork branch into the reviewed
  `jetson-voice-engine/recipes` layer with pinned ModelOpt/model revisions and
  provenance output.

The exported transfer archive is 890,583,040 bytes with SHA-256
`0b6518ed09e59c2838448912675701b2211110dcf3abbb6c76becced275dc336`.
The controlled Fleet transfer verified its MD5 and the Orin copy matched the
source SHA-256. After extraction, the duplicate Orin archive was removed; the
WSL copy remains recoverable.

The v0.9.1 CuTe builder accepted all exported INT4 and attention plugin nodes
and produced a batch-2 talker in 53.36 seconds. Peak TensorRT builder memory
was 2,308 MiB GPU and 2,054 MiB CPU:

```text
engine bytes: 250,250,388
engine sha256:
  fd07a3d0b73a6ee5057c3b4f96bf4613e039ba6f1292e6f226cbb2e80ed5145b
```

The production precision mix (INT4 talker plus the official v0.9.1 code
predictor and code2wav engines) then passed:

- worker ready with `max_slots=2`;
- four Chinese/English quality prompts, all non-silent and unclipped;
- two named speakers with distinct output;
- four SenseVoice roundtrips with normalized similarity 1.0;
- true N=2 chunk interleaving;
- each concurrent PCM byte-identical to its matching single-request baseline;
- cooperative cancel plus immediate byte-identical recovery;
- 50/50 additional N=2 stress rounds, every round interleaved and isolated;
- worker exit code zero and no CUDA error.

The 50-round evidence is:
`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/evidence/tts-customvoice-int4/n2-50-stress.json`.

The old harness reports one expected false failure when comparing explicit
Chinese with omitted language: the outputs are identical because Chinese is
the default language. Explicit Chinese and English prompts both passed exact
ASR roundtrip, so this does not indicate a missing conditioning row.

Two temporary ASR validation uvicorn processes on ports 8621/8622 and their
remaining 1.2 GB RSS worker were stopped before the TTS measurement. They were
not production containers. The `edge-llm-chat-service` and `translator`
containers were restored after the engine/runtime gate and both reported
healthy.

The local focused migration suite was rerun outside the filesystem sandbox
because its mock HTTP tests bind loopback sockets: all 89 selected
overlay/capability/config/concurrency tests passed.

### SparkTTS BF16 and W4A16 fresh v0.9.1 gate

SparkTTS was re-exported in an independent WSL worktree based on the exact
official v0.9.1 commit. All 40 candidate patches applied to the isolated
export branch, so this exercise did not silently fall back to the v0.9.0
source tree. Both exported ONNX graphs load their external data and pass
`onnx.checker`:

| Variant | External-data SHA-256 | Nodes |
| --- | --- | ---: |
| BF16 | `bab8bbc373633185075b5c28533b169225b68e609afe3ef183366cd6589096c4` | 905 |
| W4A16 | `6178438ad7a67e6f52a4044a9f0e33f248288880c5795cf6fa8cd67e11de2569` | 977 |

The v0.9.1 CuTe builder on Orin NX produced fresh engines:

| Variant | Engine bytes | Engine SHA-256 | Build time |
| --- | ---: | --- | ---: |
| BF16 | 1,029,536,796 | `0dbef23eeada0cbd7adbe1e5e3f3357bf9bc77d293a1cadb1554272d6bc9099d` | 84 s |
| W4A16 | 559,714,300 | `258bc1e4921d03bd4dcb7fb303069cca83818f5c1e847ba61acd2f6ea7c77d16` | 71 s |

The BiCodec decoder and speaker decoder are standalone TensorRT 10.3 audio
engines and do not contain an Edge-LLM runtime. They were therefore retained
as independent, version-compatible inputs while the Spark language-model
engines and worker were rebuilt from v0.9.1.

Both variants passed the complete device gate:

- controllable Chinese and English synthesis, with non-silent PCM and worker
  exit code zero;
- clone mode produced output distinct from controllable mode;
- true shared-engine N=2 interleaving, with the deterministic A output
  byte-identical to its solo baseline;
- 50/50 cancel-A/continue-B/immediate-recovery rounds, with every kept and
  recovery output byte-identical to baseline and no CUDA/TensorRT error;
- independent SenseVoice TensorRT roundtrip on both languages: all four
  normalized transcript similarities were 1.0.

W4A16 is the preferred production Spark variant. Its observed TTFA/RTF was
0.455 s/0.507 for Chinese and 0.412 s/0.494 for English, versus BF16
0.700 s/0.767 and 0.641 s/0.762. This validates the current mixed-precision
extension on v0.9.1; it does not prove that the product-specific precision
patch belongs upstream unchanged.

Evidence:
`/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/evidence/sparktts-20260725`.

### Production GDN+MTP cutover

On 2026-07-25 the Orin NX production LLM service was switched from the v0.8
vanilla-GDN container to the validated v0.9.1 GDN+MTP release candidate. The
translator remained online. The original compose file, v0.8 image, engine
directory, and rollback tag were not modified.

The cutover uses
`deploy/docker-compose.edgellm-v091-cutover.yml` as a second compose file. It
selects:

- image `edge-llm-chat-service:v0.9.1-gdn-mtp-20260725`, an immutable
  production tag for the validated audit image ID `9677a8050dce`;
- exact upstream base
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6`;
- `/workspace/v091-release-candidate-qwen35-4b-awq-mtp-20260724T103611Z/engines`;
- `spec_base.engine` SHA-256
  `0e82d742fca769fab6dc34c54909c4d8d63b49419a72b97348e6b6cca0b4f38f`;
- `spec_draft.engine` SHA-256
  `b65b5cb749d5403a3c3801833aa7fc160a541a1820a2206bba5cabb93639cba1`;
- runtime plugin SHA-256
  `82bb174986b6968db67a523aea0bda43b83ba06684be7f7bb9e4e38eb3105adf`;
- strict provenance checking, max input 2048, KV capacity 4096, and one active
  server request with queueing.

The production container loaded both fresh engines, selected the MTP decoder,
captured its CUDA graphs, passed entrypoint inference warmup, and became
Docker-healthy. Client-side gates then passed:

- `/internal/readyz`: ready, one slot available;
- translator `/health`: `nllb-200-distilled-600M` on CUDA;
- real OpenAI-compatible completion: exact response `production-v091-ready`;
- two simultaneous SSE clients: both non-empty and complete, one active plus
  one waiting observed, then capacity returned to idle;
- no CUDA, TensorRT, Myelin, assertion, or container restart error.

After the SparkTTS build/runtime gate temporarily stopped both GPU services,
the same production compose overlay and translator container were restored.
The post-restore black-box gate again produced exact
`production-v091-ready`; two SSE clients both completed while the capacity
endpoint observed `active=1` and `waiting=1`, then returned to idle. Both
containers were healthy, the LLM restart count remained zero, memory usage was
9,309/15,656 MiB with 6,054 MiB available, and 14 GB of disk remained.

This is deliberate single-flight service concurrency. Direct simultaneous use
of multiple official runtime contexts remains unsupported because it
reproduces the Myelin already-loaded graph defect. Voice scheduling is also
bounded: Base TTS N=2 is supported independently, while active GDN+MTP may
overlap only Base TTS N=1. GDN must be serialized against the Base N=2 pool.

### Production Base Code2Wav 512 and residency-swap gate

The official Code2Wav builder defaults to a maximum code length of 2000. Its
TensorRT profile required approximately 2.95 GiB of activation memory and
caused the 16 GB Orin NX kernel to kill GDN when Base TTS became resident,
even after the ASR worker had been fully unloaded. The production default is
therefore 512 code frames (approximately 41 seconds at 12.5 Hz), while the
2000-frame engine remains a separate, explicit full-range artifact.

The fresh v0.9.1 build used `min/opt/max = 1/128/512` and produced:

| Property | Result |
|---|---|
| engine SHA-256 | `a2d2db1a9cad8b38a0d551448c5cd0792aa6a1a8a79eda3d5cc3ad1ebac3560b` |
| engine size | 235,202,796 bytes |
| TensorRT activation memory | 754,976,768 bytes |
| TensorRT weights memory | 228,102,048 bytes |
| build peak GPU allocator use | 5,760 MiB |
| profile written to config | `min=1`, `opt=128`, `max=512` |

The production Base profile also uses `LAZY_TTS=1` and an exclusive
ASR/TTS coordinator. Startup preloads only ASR. The first TTS request unloads
ASR before loading Base; the next ASR request unloads Base before reloading
ASR. This is model residency swapping inside one service, not simultaneous
ASR and TTS residency.

The deployed runtime image ID is
`sha256:a0ac71b44c6d513781b9921d7d08cda64de9df2c8d369ea782878a26b91d2e4f`.
With GDN+MTP and the translator healthy, the service-level gate passed:

- a real Chinese ASR input returned `今天天气真好。`;
- the first Base request loaded
  `/opt/edgellm-v091/engines/tts_base_code2wav_512` and generated valid
  24 kHz, mono, 16-bit PCM WAV;
- the generated WAV looped back through ASR with the intended sentence
  preserved apart from a minor homophone/punctuation difference;
- 10/10 alternating TTS→ASR rounds passed;
- every TTS and ASR phase overlapped a real GDN completion request (20
  overlapping GDN requests total);
- all ten WAVs were deterministic with SHA-256
  `801ac45fa9c255cfe87b2dacba63a15bdcab47fc5cd5b260f61bdfad6f0fdcb5`;
- GDN restart count remained `0→0`, `OOMKilled=false`, and both services
  remained healthy;
- final residency was ASR-only, and the voice container restart policy was
  promoted to `unless-stopped`.

Machine-readable device evidence:
`/home/harvest/validation/v091-prod512-hotswap-gdn-10.json`.

The release artifact now contains 143 payload files (17,275,482,249 bytes)
and 25 verified engine sidecars. Full `sha256sum -c SHA256SUMS` passed on both
Orin NX and the WSL publication staging copy. The manifest remains
`published_to_hf=false` until the explicit external upload is approved and
remotely verified.

Rollback remains one command using only the original compose file:

```text
docker compose -f docker-compose.yml up -d
```

This recreates `edge-llm-chat-service` from the preserved v0.8 `.env` and
rollback image without deleting the v0.9.1 engines or evidence.

The rollback path was exercised after the default-512 production gate, not
only inspected. Voice was stopped, the original compose plus preserved `.env`
recreated the chat service from
`edge-llm-chat-service:v0.8.0-gdn-mtp-merged` /
`edge-llm-chat-service:rollback-v080-20260724` (both resolve to image
`sha256:af219111ef86d0c955e5795fc3e1e92c124ba920632681b83c046fd60bc88b11`).
It loaded `/workspace/qwen35-4b-awq/engines-v080-gdn`, became healthy with
restart count zero, and returned exact `rollback-v080-ready`. The translator
remained healthy throughout.

The v0.9.1 compose overlay then recreated GDN+MTP from image
`sha256:9677a8050dce7a391fb20ac3242746b0b57d39ae6ee79978f71152b066539d20`.
After it became healthy, the default-512 voice service was restarted and a
fresh WAV again returned `今天天气真好。`; ports 8000, 8621, and 9001 were
all healthy and the restored GDN restart count remained zero.
