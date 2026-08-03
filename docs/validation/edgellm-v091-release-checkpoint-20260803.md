# TensorRT Edge-LLM v0.9.1 release checkpoint

Date: 2026-08-03  
Decision: not yet promoted or published

## Reproducible source boundary

- NVIDIA TensorRT Edge-LLM v0.9.1:
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6`.
- Product integration branch: `codex/edgellm-v091-upstream-audit`.
- Engine overlay branch: `codex/edgellm-v091-production-migration`.
- VoxEdge source used by the qualified r6 image:
  `f738123cdef13f774b8e6c55cc32f9dca8dba8ec`.
- VoxEdge wheel SHA-256:
  `7cb2d067ee0796f9f4ce49437242ee56b82eaf1cbd414f55ff136d6341c6490e`.

The post-r6 source adds two release corrections that still require a cached
thin-image rebuild:

1. package the self-contained `jetson-edgellm-v091-sparktts` profile;
2. emit metadata for both Spark shared engines and refuse model downloads
   unless `HF_ENDPOINT=https://hf-mirror.com`.

## Qualified image and runtime findings

The latest device-built image is
`seeed-local-voice:v0.9.1-edgellm-runtime-r6-moss-n2-b11ada3-20260726`
(`sha256:31b71218a2d87696a31b676df2913287947f76458f824e2f38ea0a2913db2ef9`).
Its static image gates and mounted MOSS ONNX Runtime 1.23.2 ABI gates passed.

- Base N=1, true-streaming ASR, TTS, cancellation recovery, GDN co-residency
  and rollback rehearsal passed in the earlier complete gate.
- Base isolated N=2 passed ASR and TTS overlap, isolation and cancellation.
- CustomVoice N=1 passed. Its unsupported external embedding clone API now
  fails before worker dispatch instead of returning HTTP 500.
- CustomVoice TTS-only N=2 cancellation passed 20/20, but ASR N=2 plus
  CustomVoice N=2 co-residency caused kernel OOM eviction on the 16 GB device.
  It is therefore an isolated qualification profile, never the production
  multi-model default.
- MOSS r6 N=1 and clone passed. Two HTTP requests completed 20/20 with the
  worker launched at two slots. A stricter useful-work overlap gate did not
  pass its first captured round: recovery first PCM arrived 83.4 ms after the
  long keep stream ended. Cancellation, PCM validity, recovery deadline,
  container health and runtime error scan all passed. Until a repeatable
  first-PCM overlap gate passes, MOSS production remains N=1.
- Qwen3.5-4B GDN returns correct results to two clients but the measured token
  overlap is zero. The service is singleflight/queued and must not be called
  official continuous batching.

## Spark release blocker

The v0.9.1 Spark W4A16/BF16 LLM engines and native worker passed direct N=1,
N=2, clone and cancellation gates. W4A16 remains the latency default.

The shared BiCodec and speaker-decoder engines are not publishable yet. The
retained files have configuration and numerical records but no trustworthy
source/build provenance. Their recorded source ONNX MD5 values are:

- BiCodec: `f5ec96fae85be28099d43118a3b709a5`;
- speaker decoder: `1654b353f50c0d6f63c3c72508d56f47`.

A complete device scan found neither matching ONNX files nor the original
Spark-TTS source/checkpoint. The old engines remain untouched and outside the
final artifact set. The committed export scripts and build route can rebuild
them once the exact official source revision and complete
`Spark-TTS-0.5B/BiCodec` checkpoint are restored.

## Remaining release sequence

1. Restore an Orin NX and a download/build host to Fleet.
2. Run `fleet bootstrap <device> --profile edge-mirror`, then verify both the
   current shell and `bash -c` see `HF_ENDPOINT=https://hf-mirror.com` before
   any model download.
3. Obtain the pinned Spark source/checkpoint through hf-mirror or ModelScope,
   rebuild both shared engines into a new directory, and pass ONNX/PyTorch/TRT
   numerical, shape and provenance gates.
4. Create a new immutable artifact set; do not modify r2. It must contain all
   Spark runtime dependencies under `engines/sparktts-shared/` and pass full
   manifest/SHA/negative gates.
5. Build the cached final thin image from the current source boundary and run
   MOSS and Spark service-level N=1/N=2/cancel gates plus the production Base
   N=1 whole-device gate.
6. Repeat rollback, then cut over only to the stable N=1 production profile.
7. Upload the verified artifact set and image. For Hugging Face upload, use
   `HF_HUB_DISABLE_XET=1` and commit one file at a time; mirror endpoints are
   download-only. Verify remote hashes before setting `published_to_hf=true`.

No additional NVIDIA issue or PR is authorized by this sequence. Existing
bug PRs remain preparation/maintenance work; model features and product
protocols stay out of the bug queue.
