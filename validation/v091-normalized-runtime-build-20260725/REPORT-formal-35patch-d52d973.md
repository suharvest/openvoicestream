# TensorRT-Edge-LLM v0.9.1 normalized runtime validation

Date: 2026-07-25  
Device: Orin NX (`aarch64`)

## Formal source provenance

- Outer head: `3d0ce7e5f995cc569f86d6d2354ee0a675dcd3d3`
- Inner head: `d52d973f4a69951831ce7de0b5eee2b5ecf81006`
- Inner gitlink in the outer head: `d52d973f4a69951831ce7de0b5eee2b5ecf81006`
- Official upstream pin: `7f061f21f0a581ba234a1e233c9315b89d8e47d6`
- Official upstream tree: `3c3550839468342d36d57c22f09f38841b01c256`
- Patch stack: 7 exact proposed-upstream patches plus 35 local product patches
- Outer bundle SHA-256: `b14661b98d123fff0ac2fb68ac2eaf78c27256887af7c4294ef9a6b14b26d372`
- Inner bundle SHA-256: `b894c609d1391a56b6afc6c47d3537eb3fc4eba3c0ac27711f19b8a23b5fe9bf`

Both bundles passed `git bundle verify` on the device. The formal patch-stack
gate and `build.sh --apply-only` passed against the exact official pin.

## Formal source versus preserved B source

The formal 35-patch materialization was compared with the preserved B source,
which was created earlier by reversing patch 0039 only. All comparison gates
matched:

- Binary diff SHA-256, both sides:
  `32f24d15bcb094d41a937f0b3d11fa0b0b907dd6c7f97e7bea9871a93fe88a58`
- Non-gitlink tracked-file archive SHA-256, both sides:
  `7ca945f58173719e8cd041b21bb21233d00069b59aa914b9535b348cf6a14183`
- Git index SHA-256, both sides:
  `6cb24e1c429a6f114740290d01d05cc1ead9308d4e952567b3e75838ade41a51`
- Changed-path lists matched byte for byte.

Gitlinks are covered by the index comparison and are deliberately excluded
from the tracked-file tar comparison to prevent recursive traversal into
submodule worktrees.

## Preserved B product artifacts

The following final-linked B artifacts passed `ldd -r` without an undefined
symbol and were copied into the formal validation directory:

- `moss_tts_nano_worker`:
  `e733b739601959401b161eb1c88c4003543f2c9f06b05766856a5006a472e9fa`
- `qwen3_asr_worker`:
  `eca09cce30496e7892c119d6806586163b831a84420f5b6f5f4d2d323838ac6f`
- `qwen3_tts_streaming_worker`:
  `cf83addd893a4f230802816f284c8c3fe7ad6150f85d3b746f06c247472f04ce`
- `spark_tts_worker`:
  `fd76e67eb36169d4c53eba229b239a63cb70d7f3549c3147f1a97d61e1ec1efe`
- `libNvInfer_edgellm_plugin.so.1.0`:
  `8d02067937e9b8362cf91f3bc8d997c73378bb3827000930e045194d23428afa`

The B build also completed final links for `llm_inference`. Link commands retain
`--wrap=_cudaLaunchKernelEx` and use the full CUDA 12.6 / CuTe DSL 4.5.1 SM87
artifact.

## Patch 0039 A/B conclusion

Changing the CUDA driver dependency from `PUBLIC` to `PRIVATE` did not break
the final link or `ldd -r` checks for the plugin, `llm_inference`, Qwen3 TTS,
MOSS, Qwen3 ASR, or Spark TTS products. This supports retiring patch 0039 from
the formal product stack.

This conclusion is intentionally limited to product build, final-link, and
dynamic-symbol validation. A GDN engine launch using the B binary was not part
of this gate, so this report does not claim an additional B-specific GDN
runtime-launch proof.

## New runtime image

- Tag:
  `seeed-local-voice:v0.9.1-edgellm-runtime-normalized-d52d973-20260725`
- Image ID and local repository digest:
  `sha256:13f8b69ed37ad1238afdb0116b003d36e0a32102555aa0c0f636e168b42222d9`
- Size: `595708732` bytes
- In-image Python compile smoke: PASS

The image was not pushed and production was not switched to it.

## Production safety

Production stayed online during the formal gate and image build. Final state:

- `seeed-voice-v091`: running, healthy, restart count 0
- `edge-llm-chat-service`: running, healthy, restart count 0
- `translator`: running, healthy, restart count 0
- Voice readiness endpoint: PASS
- LLM health endpoint: PASS
- Translator health endpoint: PASS

## Device paths

- Validation root:
  `/home/harvest/validation/v091-formal-35patch-d52d973-20260725`
- Formal outer source:
  `/home/harvest/project/seeed-local-voice-v091-formal-3d0ce7e-20260725`
- Formal materialized TensorRT-Edge-LLM source:
  `/home/harvest/project/TensorRT-Edge-LLM-v091-formal-d52d973-20260725`
- Preserved B source:
  `/home/harvest/project/TensorRT-Edge-LLM-v091-normalized-b364b06-no0039-20260725`
- Preserved B build:
  `/home/harvest/build/TensorRT-Edge-LLM-v091-normalized-b364b06-B-no0039`

No source/build directory, artifact, or image named above was deleted.
