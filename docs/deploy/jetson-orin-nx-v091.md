# Jetson Orin NX: TensorRT Edge-LLM v0.9.1

This is the production path for the qualified three-model composition:

- Qwen3-ASR 0.6B
- Matcha TTS
- Qwen3.5-4B GDN/MTP, 8K context

The two containers contain runtime code only. Each selected model is fetched
from its own immutable Hugging Face revision through `hf-mirror.com` and is
verified before it is atomically installed into a persistent volume. Switching
speech profiles changes only the required model downloads; it does not rebuild
the runtime image. The final OCI-provenance LLM runtime image is
`sensecraft-missionpack.seeed.cn/solution/edge-llm-chat-service:v0.9.1-gdn-mtp-runtime-20260804-v13`
(`sha256:3c3e9235efb1ab5c0eac69f47e494a7d03fd381fce83320771e9328801a02116`,
143,229,815 bytes). It carries service revision
`85965efe31a1b1f377a97f4e9be41405bc67737c` and the official v0.9.1 upstream
pin `7f061f21f0a581ba234a1e233c9315b89d8e47d6`.

## Install

```bash
deploy/install.sh --target orin-nx --pull --verify
```

Auto-detection selects the same path when `/proc/device-tree/model` identifies
an Orin NX. Speech is served on port `8621`; the OpenAI-compatible LLM endpoint
is served on port `8000`.

The default speech profile is `jetson-edgellm-v091-matcha`. Other v0.9.1 speech
profiles select Base, CustomVoice, MOSS, Spark, or Matcha independently. Model
repositories, immutable revisions, payload hashes, and sizes are locked in
`deploy/artifacts/v091-release-lock.json`.

The default LLM compose is the 8K profile. The same model-level runtime also
has a 4K profile, selected explicitly after its public HF commit is available:

```bash
EDGELLM_ENGINE_PROFILE=4k \
EDGELLM_4K_ENGINE_REVISION=<published-hf-commit> \
deploy/install.sh --target orin-nx --pull --verify
```

The 4K and 8K payload locks are respectively
`06273e358a579590bb8344b451aa35c89983cd99401339fb1858d61af4dbd107` and
`9208e46d61a4f1440ac68a312e35dde3d04b88edf0e4ee12b32210e7190d3325`. Until
the corresponding revision markers are replaced with real HF commit IDs,
startup fails closed by design.

## Verify

```bash
deploy/verify.sh --url http://127.0.0.1:8621 --tts-smoke --roundtrip
curl -fsS http://127.0.0.1:8621/v1/models
curl -fsS http://127.0.0.1:8621/v1/capabilities
curl -fsS http://127.0.0.1:8000/v1/models
```

`POST /v1/audio/speech` streams PCM/WAV chunks from the selected TTS backend.
Voice, speed, cloning, streaming, and concurrency support are discoverable from
`/v1/capabilities`; clients must not assume every TTS model has the same voice
features.

## Roll back

Speech and LLM rollback are separate operations.

To return the speech/API service to the previous generic Jetson release, use
its independent model volume:

```bash
deploy/install.sh --target jetson --pull --verify
```

The qualified LLM rollback target is the preserved v0.8 image
`edge-llm-chat-service:rollback-v080-20260724` (image ID
`sha256:af219111ef86d0c955e5795fc3e1e92c124ba920632681b83c046fd60bc88b11`)
with `/workspace/qwen35-4b-awq/engines-v080-gdn`. Use the preserved v0.8
compose/environment described in
`docs/validation/edgellm-v091-production-migration.md`; it was exercised on
Orin NX and returned `rollback-v080-ready` with restart count zero. The image
and its 3.3 GB host engine cache were rechecked on 2026-08-05 before the final
v0.9.1 cutover.

Do **not** use `v0.9.1-gdn-mtp-8k-20260804-v5` as a rollback target. It is a
superseded development image whose old schema-v2 verifier rejects the final
cache's additional `PROVENANCE.md` file. It is retained only as build history.

The v0.9.1, v0.8, 4K, and 8K caches are deliberately not shared. Do not rename
or copy an engine cache between runtime or context profiles.

## Rebuild policy

Published images are the primary installation path. A source rebuild must stage
every runtime binary listed in the release lock into
`deploy/artifacts/v091-release-gate/`; missing or mismatched files fail the
Docker build. Model engines remain outside the image.

The 4K Qwen3.5 payload has passed the Orin NX build/hash gates, but remains
publication-pending until its immutable HF revision is recorded. Do not use the
older July 2048-input/4096-KV candidate or relabel it as a 4K-input build.
