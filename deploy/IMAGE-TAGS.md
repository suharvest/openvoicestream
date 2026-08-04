# Image Tag → Commit Mapping

Reproducible record of registry image tags built for `seeed-local-voice`.

The voxedge wheel (`deploy/wheels/`) and worker binaries (`deploy/jetson-workers/`)
are **gitignored**, so reproducibility relies on the recorded voxedge commit below
plus the seeed commit. Rebuild the wheel from the recorded voxedge commit
(`uv build --wheel`) and stage the same worker binaries.

| Tag | Seeed commit | voxedge commit | Date | Build host | Registry digest |
|-----|-------------|----------------|------|-----------|-----------------|
| `prod-unified-v8` | `9bad68d99d2c20e0448c6b958f1302b35756829f` | `02e4f0bbe46e4c0cb6513c396cc83aab652ade65` | 2026-06-03 | recomputer-desktop | `sha256:910e298e9b5bf3643133c070618588164f104f9b326cc3f16de8200f5c760f5a` |
| `prod-unified-v9` | `9bad68d99d2c20e0448c6b958f1302b35756829f` | `02e4f0b+mossfix(afdef16)` | 2026-06-17 | seeed-orin-nx | `sha256:bf355e8af0e214be2c76681313f5e2ff590b0f6f692679c5f27a3b6e77c5ac22` |
| `jetson-jp62-trt103-edgellm-v091-20260804-r5` | release lock `orin-nx-edgellm-v091-jp62-trt103-sm87-20260803-r5` | `f738123` (0.0.5a0, superseded by next reproducible rebuild) | 2026-08-04 | seeed-orin-nx | `sha256:b1d9db8d0e61344dc02367bb0114fd6889335f21638cea790e3f795d8226ce5c` |
| `edge-llm-chat-service:v0.9.1-gdn-mtp-8k-20260804-v5` | `634855b` runtime artifact commits (superseded) | n/a | 2026-08-04 | seeed-orin-nx | `sha256:0ec928901a020cd9e67078d2b32837acc28137bc0c3dbfc5b08798e2133efc98` |
| `edge-llm-chat-service:v0.9.1-gdn-mtp-runtime-20260804-v12` | model-neutral v0.9.1 runtime, service `2dee2993c3193628f135ca96419b149de567062a` | `voxedge==0.0.6a1` | 2026-08-04 | seeed-orin-nx | `sha256:4b0929562f6b68714ade695abe81df8a6c6d9f4042d6093080374f56f9155c38` (143,229,804 bytes) |

**prod-unified-v8** — single UNIFIED image serving both conversation modes via a
runtime flag: flag-OFF = client-loop pass-through; flag-ON = server-loop
(`voxedge.engine.conversation.ConversationEngine._handle_tool_advertise`,
conversation.py:481). Built from `Dockerfile.jetson.slim` (now `deploy/docker/archive/`) target `final-slim`,
`LANGUAGE_MODE=multilanguage`. Models are HF-fetched at runtime (not baked).

**prod-unified-v9** — OVERLAY on `prod-unified-v8`: reinstalls voxedge with ONLY
the moss `channels=1` stereo→mono downmix cherry-pick (`MossTtsNanoBackend._stereo_to_mono_s16le`)
+ adds the combined `jetson-qwen3asr-moss-nx` profile (Qwen3 ASR via
`QWEN3_ARTIFACT_SET` env + MOSS TTS via `required_engines`, `OVS_TTS_CHANNELS=1`).
Built via overlay Dockerfile on seeed-orin-nx (`/home/seeed/moss-slv-build/`),
not a full rebuild. Verified: downmix present (mono_hex 9600 for stereo[100,200]),
profile parses (asr=jetson.trt_edge_llm, tts=jetson.moss_tts_nano, moss_channels=1).
