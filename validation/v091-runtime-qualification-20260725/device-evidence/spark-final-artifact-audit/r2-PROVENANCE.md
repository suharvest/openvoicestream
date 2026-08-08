# TensorRT-Edge-LLM v0.9.1 Orin NX r2 release candidate

- Artifact set: `orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2`
- NVIDIA upstream: `https://github.com/NVIDIA/TensorRT-Edge-LLM.git`
- Upstream tag/SHA: `v0.9.1` / `7f061f21f0a581ba234a1e233c9315b89d8e47d6`
- Outer source SHA: `021112eda3207a57ae91056f24d198303574b555`
- Engine overlay SHA: `4b28dd24bcf5c5240f604b7d6be78ad39d9c5e2f`
- Formal replay diff SHA-256: `32f24d15bcb094d41a937f0b3d11fa0b0b907dd6c7f97e7bea9871a93fe88a58`
- Upstream bug-fix candidates: 7 ordered patches, series SHA-256 `9fe8677a4c6886a4b7d8d253a245b6d95509d26a7fe8072eb0de6ed83633c733`
- Product/model overlay: 35 ordered patches, series SHA-256 `a358dcafc5d3fc70468a01e21c97236a1a6664644d26f635ed6ee6934d3be3d5`
- Platform: Jetson Orin NX 16GB, SM87, JetPack 6.2, L4T R36.4.3, CUDA 12.6, TensorRT 10.3.0.30, aarch64
- MOSS worker SHA-256: `9d114d8390e684c8876e2ef9e20e28ee6d4ec6ce18b81df5da3ba64c8f057deb`

The r2 payload reuses immutable validated engine files from the preceding local
artifact set through same-filesystem hard links. Its worker and all control
files have independent inodes. `manifest.json` inventories every payload file,
including the Base b1/b2 KV1536 engines added after the preceding manifest was
generated. `SHA256SUMS` covers every payload file and is generated atomically by
the finalizer with `published_to_hf=false`.

MOSS and SparkTTS are local model extensions and are intentionally excluded
from NVIDIA bug-fix PRs. Upstream PR candidates remain local and must not be
submitted without explicit owner confirmation.
