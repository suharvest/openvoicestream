# TensorRT-Edge-LLM v0.9.1 Base N=1 residency validation

Date: 2026-07-25
Device: Jetson Orin NX 16GB, JetPack 6.2, CUDA 12.6, TensorRT 10.3, SM87

Publication status was completed on 2026-08-05; see
`deploy/artifacts/v091-release-lock.json` for the final immutable revisions.

## Outcome

The v0.9.1 migration had accidentally rebuilt the Qwen3-TTS Base Talker and
CodePredictor with the upstream general-purpose limits:

- `max_batch_size=2`
- `max_input_len=4096`
- `max_kv_cache_capacity=4096`

The product limits already validated on v0.8 are restored for the production
N=1 artifact:

- `max_batch_size=1`
- `max_input_len=1024`
- `max_kv_cache_capacity=1536`
- Code2Wav `max_code_len=512`

The N=2 build is now an explicit opt-in (`TTS_MAX_BATCH_SIZE=2`) and uses
separate `-b2` engine directories.

## New device artifacts

| Component | SHA-256 | Size |
|---|---|---:|
| Base Talker N=1 input1024/KV1536 | `27357dc0ee91fb6a52e8952861f5bce540de0c1c76df4086dfa23332e94e017e` | 246,036,444 bytes |
| Base CodePredictor N=1 input1024/KV1536 | `0e9ac804d24a49d9307e7e1a831e219e31c82d21e7738a55b851fdc704990265` | 190,964,252 bytes |

Device paths:

- `engines/tts_base_talker_b1_kv1536`
- `engines/tts_base_code_predictor_b1_kv1536`

The old 4096-context artifacts remain present as rollback inputs and were not
overwritten.

## Residency and functional gates

Resident services:

- Qwen3-ASR 0.6B, batch 1
- Qwen3-TTS Base 0.6B, batch 1
- Qwen3.5-4B AWQ + MTP

Both speech workers remained alive for every gate.

### Sequential N=1

- One functional round: pass
  - TTS: 2.56 s
  - ASR: 0.35 s, exact transcript
  - GDN: 0.29 s
- 10 rounds: pass
- 50-round soak: pass
  - TTS: 2.44–2.54 s
  - ASR: 0.28–0.37 s
  - GDN: 0.22–0.28 s
  - voice restart/OOM: `0 / false`
  - GDN restart/OOM: `0 / false`
  - observed minimum `MemAvailable`: about 103 MiB

### Controlled component overlap

The gate overlaps TTS with one GDN request, then ASR with another GDN request.

- 1 round: pass
- 10 rounds: pass
- exact ASR transcript and byte-identical WAV across rounds
- voice restart/OOM: `0 / false`
- GDN restart/OOM: `0 / false`
- overlapped TTS: about 3.65–3.73 s
- overlapped ASR: usually 0.44–0.46 s
- overlapped GDN: usually 0.25–0.34 s

Compared with sequential execution, overlap increases TTS latency by about 49%.
The device is stable in this bounded test but has little physical-memory
headroom and relies heavily on swap. N=2 must remain opt-in until it passes a
separate long soak and latency gate.

## Product changes

- The Base production profile preloads and keeps ASR/TTS resident.
- Talker and CodePredictor use the new N=1 input1024/KV1536 artifacts.
- The production compose memory limit is raised from 7 GiB to the validated
  8 GiB audit limit.
- CustomVoice and MOSS retain their existing exclusive/lazy policies.

## Production image and final gate

Production image:

- tag: `seeed-local-voice:v0.9.1-edgellm-runtime-20260725-b1kv1536`
- image digest:
  `sha256:0242800eea03919bc089aff62556a82ad78256f1c04933836d52e63fb49bd2da`

The image was deployed as `seeed-voice-v091` on port 8621. The final
production gate ran 10 controlled overlap rounds against the deployed
container:

- every ASR transcript matched exactly
- every TTS WAV was byte-identical to the reference
- ASR and TTS remained resident at the end of every round
- voice and GDN restart counts remained zero
- neither container reported an OOM

`N=1` is the production stability baseline: one active request per model. It
does not mean the three models must be swapped or run one at a time. ASR, TTS,
and Qwen3.5-4B remain resident together; TTS/GDN and ASR/GDN overlap also
passed. Within the voice service, ASR and TTS requests stay serialized by the
current worker capability contract.

For the low-latency objective, sequential component scheduling remains the
best baseline because forced overlap raised TTS latency by roughly 49%.
Additional concurrency is therefore a throughput option to be qualified
separately, not a replacement for the stable N=1 profile.

At the time of this historical validation, external publication and registry
upload were still pending explicit approval.
