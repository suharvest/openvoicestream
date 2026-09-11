# Qwen3-TTS Base on TensorRT Edge-LLM v0.9.1

## Status

Qwen3-TTS Base is a Seeed-maintained extension on top of the exact official
TensorRT Edge-LLM v0.9.1 commit
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`.

The fail-loud export entry point is
`third_party/jetson-voice-engine/engine-overlay/drivers/export-qwen3-tts-base-v091.sh`.
It accepts only a checkout whose Git HEAD is the exact official pin, requires
immutable model and INT4 stage-2 revisions, checks all three exported ONNX
graphs with external data loaded, and writes driver/provenance hashes. It also
refuses to run if the official CustomVoice-only guard is still present.

It must not be described as an official v0.9.1 feature until the corresponding
changes are accepted upstream. The extension may reuse the official v0.9.1
Talker, CodePredictor, and Code2Wav export/build paths, but it also needs an
external speaker embedding and a separately versioned speaker encoder.

The extension has now passed the Orin NX device gate with fresh v0.9.1
artifacts: all four engines built, 3 Chinese and 2 English strict content
roundtrips passed, independent N=2 passed 50/50 cancel/isolation/recovery
rounds, and the production services were restored healthy. The supported
mixed scheduling boundary is GDN+Base N=1; GDN must be serialized against the
Base N=2 pool.

## Current evidence

- The official CLI rejects the pinned Base checkpoint before component export:

  ```text
  Only Qwen3-TTS CustomVoice checkpoints are supported.
  Got tts_model_type='base'.
  ```

- This proves that the public v0.9.1 CLI does not support Base. A follow-up
  export from exact official SHA `7f061f21` with only the entry guard relaxed
  proved that all three existing internal component exporters already process
  Base:

  - Talker passed external-data loading and `onnx.checker`; the graph has
    1,643 nodes, including 196 `Int4GroupwiseGemmPlugin` and 28
    `AttentionPlugin` nodes.
  - CodePredictor and Code2Wav both passed external-data loading and
    `onnx.checker`.
  - The immutable Base model revision is
    `Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992436eae1d760afd27aff78a71d676296fc`.
  - The INT4 stage-2 source is
    `wip/native-int4-talker@ff2318e66525365b2ed9f55811bf5d2381280ed8`.

  Therefore the unconditional CustomVoice-only CLI guard is the sole blocker
  for the three model components. Speaker-encoder export remains separate.
- The old Base integration has hardware evidence for the speaker-conditioning
  semantics. The Qwen3-TTS Python reference inserts the raw speaker-encoder
  output in the same Talker prefill slot used by a preset speaker embedding.
  The old Orin NX path passed three content roundtrips using a real 1024-element
  FP32 embedding.
- The speaker encoder is not exported by an official
  `tensorrt-edgellm-export-audio` command in v0.9.1. Its ONNX, preprocessing
  contract, model revision, hash, and TensorRT build therefore remain part of
  this extension.

## Patch split

Do not send the current four-patch chain as one pull request.

### PR 1: stage-aware Base component export

Scope:

- `tensorrt_edgellm/scripts/export.py`
- Python exporter tests only

Change the hard CustomVoice-only check only after the unmodified internal
component exporters are proven against the pinned Base checkpoint. The public
behavior should be explicit:

- allow Base for the components that are actually implemented;
- state that the speaker encoder is not included;
- preserve the current fail-loud behavior for a requested end-to-end bundle
  that cannot be made runnable;
- do not include local branch names, Seeed paths, or historical commit IDs in
  the warning.

Candidate `0035` has been reduced from the original probe to an upstream-neutral
change: it allows only the known `base` type, retains a hard error for unknown
TTS variants, and explicitly says that component export does not include the
speaker encoder. It is applied in the WSL worktree
`codex/v091-base-extension`; no upstream PR has been opened.

Required tests:

1. CustomVoice behavior is unchanged.
2. Base reaches each supported component exporter.
3. Unsupported component combinations fail with an actionable error.
4. Exported configs retain `tts_model_type=base`.
5. ONNX external data loads and `onnx.checker` passes.

### PR 2: preserve TTS language metadata

Scope:

- `tensorrt_edgellm/scripts/export.py`
- `tensorrt_edgellm/quantization/qwen3_omni.py`
- Python config roundtrip tests

Candidate `0036` is independent of Base. It preserves
`codec_language_id` and `codec_think_id` through direct and standalone
quantized Talker export. It is a generic correctness fix for CustomVoice and
should be submitted separately.

Required tests:

1. direct FP16 config preserves the language map;
2. standalone INT4 config preserves the map and `codec_think_id`;
3. a model without the optional keys receives no synthetic values;
4. patching an already-correct config is idempotent.

### PR 3: optional external speaker embedding in the runtime

Scope:

- `TalkerGenerationRequest`
- Qwen3-TTS preamble construction
- focused C++/CUDA tests

Candidate `0025` carries the core generic feature. Before submission it needs:

- a model-neutral field name and documented FP32 `[talkerHiddenSize]`
  contract;
- finite-value and exact-size validation;
- a test proving the external vector replaces the speaker row without adding
  `ttsPad`;
- a regression test proving an omitted vector is byte-identical to the
  existing named-speaker path;
- concurrent-runtime and cancellation coverage;
- a clear decision for requests that also specify `speakerName` or
  `speakerId` (prefer fail-loud over implicit precedence).

The implementation may keep one device buffer per runtime instance. It must
not introduce a mutable global buffer or share a request-owned host pointer
past the asynchronous copy boundary.

### Local adapter: worker transport

Candidate `0027` should remain local initially. Base64-encoded little-endian
FP32 is a product transport choice, not the runtime feature itself. The current
decoder also skips invalid characters and truncates trailing bytes, so it is
not suitable for upstream review as written.

If upstream asks for an example adapter, submit it only after adding:

- strict Base64 validation and a decoded byte count divisible by four;
- exact hidden-size validation before request dispatch;
- NaN/Inf rejection;
- rejection of simultaneous named-speaker and external-embedding inputs;
- protocol and malformed-input tests.

### Separate extension: speaker encoder

The speaker encoder is not part of the Edge-LLM exporter today. Keep it in the
companion repository until its source and license provenance are complete.
The release contract must include:

- pinned Qwen3-TTS model revision;
- exporter source revision;
- `speaker_encoder.onnx` plus external data if present;
- SHA-256 for every file;
- input `mel [1, time, 128]` FP32 and output `[1024]` FP32;
- the exact 24 kHz, magnitude-STFT, Slaney-normalized mel preprocessing
  contract;
- TensorRT version, target SM, build command, engine hash, and provenance.

Only after this is reproducible should a separate upstream proposal add an
official speaker-encoder exporter or example.

## Device acceptance gate

The Base extension can be enabled in the v0.9.1 product profile only after a
fresh build on Orin NX passes:

1. Base Talker, CodePredictor, Code2Wav, and speaker encoder built from recorded
   revisions, with no relabeled v0.8/v0.9.0 engine;
2. real-reference voice clone with at least three Chinese and two English
   content roundtrips;
3. speaker similarity evidence in addition to ASR content correctness;
4. N=1 baseline, configured maximum concurrency, simultaneous-start overlap,
   per-request PCM isolation, and cancellation/recovery;
5. 50 concurrent rounds with zero CUDA/TensorRT errors;
6. ASR+Base TTS and GDN+Base TTS mixed-service memory gates;
7. clean worker shutdown and original production services restored healthy.

FP16 must not be advertised as N=2 merely because the profile contains an N=2
leaf. If FP16 exceeds the 16 GB device budget, the production Base profile must
use a freshly exported, versioned INT4 Talker and record the quantization
driver provenance, as already required for CustomVoice.

## Immediate verification order

1. On the untouched official WSL checkout, invoke the internal Base component
   exporters to determine whether the CLI guard is the only exporter blocker.
2. If successful, export a fresh Base INT4 Talker from the preserved stage-2
   checkpoint with the official v0.9.1 exporter and check the ONNX.
3. Locate and verify the speaker-encoder source/ONNX provenance; do not use an
   unversioned historical engine as release evidence.
4. Build all four engines on Orin NX in versioned directories.
5. Run the device acceptance gate.
6. Reduce the local series to the verified runtime extension plus the local
   transport/encoder layer, then prepare PR 1-3 as independent commits.
