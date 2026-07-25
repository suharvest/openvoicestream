# TensorRT Edge-LLM v0.9.1 Orin NX voice validation

Date: 2026-07-24  
Device: `orin-nx` (Jetson Orin NX 16 GB, JetPack 6.2 / L4T R36.4.3)  
TensorRT / CUDA: 10.3.0.30 / 12.6  
Remote evidence:
`/home/harvest/validation/edgellm-v091-voice-candidate-20260724T0925Z`

## Result boundary

This run proves that the v0.9.1 candidate source and the carried voice patches
build for SM87, and that the fallback binaries can load the existing v0.9.0
voice engines without a worker crash or CUDA/TensorRT error in the executed
smoke gates.

The old-engine runs below are **ABI and mechanism gates only**. They are not
v0.9.1 engine quality, performance, or release acceptance evidence. Fresh
v0.9.1 engines and the full 50-pair/cancel-isolation matrix remain required
before any model row can be promoted to a v0.9.1 pass.

## Provenance and isolation

- Official v0.9.1 source SHA:
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6`.
- Candidate source SHA after the 40-patch series:
  `f4cf5c0235ac00e0467ddac6ceb943bd3058f597`.
- Candidate source SHA after adding the vendored worker source:
  `50def0691221b0421d72679709457265af645226`.
- The 40 candidate patches applied 40/40 without conflict.
- Locked recursive submodules:
  - NVTX `f71a0342a464b8580ac8573e4349086a631c3992`
  - googletest `7917641ff965959afae189afb5f052524395525c`
  - nlohmannJson `22db828de4e24818599931dca17e0f111e1e895f`
- Existing v0.8/v0.9 source trees, engine directories, production model
  volumes, compose configuration, and reference outputs were not modified.
- The two production containers were recorded before the run and stopped with
  authorization. Their full IDs, images, restart policy, ports, and mounts are
  retained in `provenance/original-services.txt` in the remote evidence root.
- Free disk stayed at 16 GiB after the fallback build and smoke gates, above
  the required 4 GiB floor. No prune, package upgrade, or model deletion was
  performed.

## Build correction and submodule finding

The first fallback configure did not use the official AArch64 toolchain and
would have compiled XQA/FMHA for the wrong architecture set. It was stopped
before compilation and preserved as rejected evidence. The accepted builds
use:

```text
-DCMAKE_TOOLCHAIN_FILE=<source>/cmake/aarch64_linux_toolchain.cmake
-DAARCH64_BUILD=ON
-DCMAKE_CUDA_ARCHITECTURES=87
-DEMBEDDED_TARGET=jetson-orin
-DCUDA_CTK_VERSION=12.6
-DTRT_PACKAGE_DIR=/usr
```

The first accepted fallback compile stopped because
`3rdParty/nlohmannJson/include/nlohmann/json.hpp` was absent. This was a clean
clone initialization issue, not a candidate patch failure. Initializing the
release-pinned submodules fixed it; no submodule was advanced to a remote
branch head.

## Fallback build (`ENABLE_CUTE_DSL=OFF`)

The plugin, core-dependent workers, MOSS worker, and builder completed with
return code 0. The final build-rule audit found four SM87 CUDA flag records and
zero SM80/86/89 records.

| Artifact | SHA-256 |
|---|---|
| `libNvInfer_edgellm_plugin.so` | `7ab35db702c725b29477e47b6441ae2b3b3a1b094b65c792d3be8da6608447d0` |
| `qwen3_tts_streaming_worker` | `9aa4c4a9cc8325711d84d13f2f96f5f63624998275e105f4f125d9e7f7501e4d` |
| `moss_tts_nano_worker` | `775ef45a310daaa3884ff19beb3550e20473baa5451dec0489a70f626fdf8ff1` |
| `llm_build` | `e3ffc19f160ae574fd38e3c1721b02f0d4e881d19d8b1df5c85d86fe74b872bd` |
| `qwen3_asr_worker` | `d31d47ef68df692328913f04adc72f9d4bea2b82c1fe6016a7e00fd508e2a115` |
| `spark_tts_worker` | `6d2e7c594948d18fdd1ed2b00d9a5842fdf023481f635e3220665913044daaf1` |

The standalone ASR/Spark worker device-link audit found three SM87 records and
zero SM80/86/89 records. Its generated `link.txt` files reference the current
fallback build's `cpp/libedgellmCore.a`; no v0.9.0 build/library path appears
in any worker link command.

## CuTe artifact and build path

The official packaged SM87 archive is CUDA 13.2.78 / cutlass-dsl 4.6.0, so it
was not used on JetPack 6.2. The accepted CuTe path uses an SM87 artifact
generated on this device with CUDA 12.6:

| Field | Value |
|---|---|
| architecture | `aarch64`, `sm_87` |
| CUDA | `12.6.68` |
| cutlass-dsl | `4.5.1` |
| groups | `gdn`, `gemm`, `ssd` |
| library SHA-256 | `2252e293801f1fd505a7a2c40ed9e58987c3c000559485147b2adb5a23f41722` |
| metadata SHA-256 | `5fd23c06136225b26ee51c0f2a8a3bdf5383a12b6f2fbad676cc15ef5f411dbf` |

The `ENABLE_CUTE_DSL=ALL` configure selected this exact artifact, enabled the
GDN/SSD groups and all eight supplied Ampere GEMM variants, and generated
artifact references in the target link rules.

The CuTe plugin, core, Qwen streaming worker, MOSS worker, and builder all
completed with return code 0. Final build-rule/link audits found 26 SM87
records, zero SM80/86/89 records, eight artifact-library references, and eight
CUDA 12.6 `_cudaLaunchKernelEx` shim-wrap references.

| CuTe artifact | SHA-256 |
|---|---|
| `libNvInfer_edgellm_plugin.so` | `82bb174986b6968db67a523aea0bda43b83ba06684be7f7bb9e4e38eb3105adf` |
| `qwen3_tts_streaming_worker` | `e07e7ab51932b217a20046f07ffd628aa0809814367b2cb26f27b1a4343a2a90` |
| `moss_tts_nano_worker` | `7a6036a1d8c1142570cb6144f25fd226dd3418005b4493499c6d17fe5c80112a` |
| `llm_build` | `e3ffc19f160ae574fd38e3c1721b02f0d4e881d19d8b1df5c85d86fe74b872bd` |
| `qwen3_asr_worker` | `0d4a1aeddad638ecd295605ad0fad6eb0ec2fd7f907f59feb546f3ae046fe15a` |
| `spark_tts_worker` | `0e0f886cd188fbbf45a3ecc058e6fbdf58b7a7aebacc600212607afbe4b32008` |

The CuTe standalone workers link the current CuTe core and the same SM87
artifact. Their audit found six SM87 records, zero SM80/86/89 records, and
three artifact-library link references.

## Fallback old-engine ABI/mechanism gates

All engine directories in this section are the read-only v0.9.0 references.
No result in this table is a fresh v0.9.1 model pass.

| Model / gate | Result |
|---|---|
| Qwen3-ASR INT4, stream + one-shot | Worker rc 0; 6/6 punctuation-normalized exact transcripts |
| Qwen3-TTS CustomVoice, ZH/EN and conditioning | Worker rc 0; all smoke checks passed |
| CustomVoice N=2 | Both streams completed and were interleaved |
| CustomVoice cancel/recovery | `cancel_ack`, terminal `cancelled`, then immediate request completed |
| Qwen3-TTS Base, five N=1 requests | Worker rc 0; all completed; two reference embeddings produced distinct output |
| SparkTTS BF16 basic | rc 0; ZH/EN completed; no CUDA errors |
| SparkTTS BF16 N=2 | rc 0; both completed; streams interleaved |
| SparkTTS W4A16 basic | rc 0; ZH/EN completed; no CUDA errors |
| SparkTTS W4A16 N=2 | rc 0; both completed; streams interleaved |

The CustomVoice cancellation probe covers cancellation and immediate recovery
in one worker. It does not yet prove the stricter “cancel A while B continues
unchanged” requirement. Base N=2, ASR distinct-input N=2, all 50-pair runs,
and 50-pair cancellation isolation remain open.

## CuTe old-engine ABI/mechanism gates

- Qwen3-ASR stream + one-shot: worker rc 0 and 6/6
  punctuation-normalized exact transcripts.
- Qwen3-TTS CustomVoice: worker rc 0; ZH/EN, conditioning, N=2 interleave,
  cancellation, and immediate recovery all passed with no reported CUDA
  error.
- CuTe SparkTTS BF16/W4A16 basic and N=2 were not run. Service restoration had
  already started when this remaining hard-range request arrived; per the
  closeout instruction the production services were not stopped a second
  time.

These remain old-engine ABI/mechanism gates, not fresh v0.9.1 model passes.

## Blocked model rows

- MOSS-TTS-Nano: the v0.9.1 worker builds, but
  `codec_decode_step.plan` is absent. No runtime pass is claimed.
- SenseVoiceSmall: the required ONNX and generated `sensevoice.plan` are
  absent. No runtime pass is claimed.
- Fresh v0.9.1 ASR/TTS/Spark engines: not generated in this run; old-engine
  ABI evidence cannot substitute for them.

## Closeout

The original containers were restarted without recreation:

- `edge-llm-chat-service`: original full ID
  `8f2af667531bddabd79edc17bf44b13a7cdc4e6e2fe51e4e49efc03fc2446ee3`,
  original image `edge-llm-chat-service:v0.8.0-gdn-mtp-merged`, healthy;
  `http://127.0.0.1:8000/health` returned healthy.
- `translator`: original full ID
  `1f30af4b97bdbab84390c0a15e956d1a8043e58374d456e8fa9a7595a69e2b4a`,
  original image
  `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:translator-cuda-jetson-v2`,
  healthy; `http://127.0.0.1:9001/health` returned ok.

Both retain restart policy `unless-stopped`. Final free disk was 16 GiB. Raw
container identity, health polling, and endpoint bodies are under
`closeout/` in the remote evidence root.
