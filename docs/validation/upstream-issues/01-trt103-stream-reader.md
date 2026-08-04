## Describe the bug

TensorRT-Edge-LLM at `v0.9.1` / current `main`
(`7f061f21f0a581ba234a1e233c9315b89d8e47d6`) does not compile against
TensorRT 10.3, which is the TensorRT version shipped with JetPack 6.2.

`cpp/common/trtUtils.cpp` unconditionally derives `FileStreamReaderV2` from
`nvinfer1::IStreamReaderV2`. That interface was introduced in TensorRT 10.7;
TensorRT 10.3 exposes `nvinfer1::IStreamReader` instead. Compilation therefore
stops at `cpp/common/trtUtils.cpp:155` before engine deserialization can be
tested.

Impact: blocker for building the unmodified project on the documented Orin /
JetPack 6.2 software stack.

### Steps/Code to reproduce bug

On a Jetson Orin NX with JetPack 6.2:

**Build configuration:**

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=OFF
cmake --build build -j"$(nproc)"
```

The compile reaches `cpp/common/trtUtils.cpp:155`, where the TensorRT 10.3
headers have no `nvinfer1::IStreamReaderV2` definition.

Source:
<https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/7f061f21f0a581ba234a1e233c9315b89d8e47d6/cpp/common/trtUtils.cpp#L155>

### Expected behavior

The runtime should retain `IStreamReaderV2` on TensorRT 10.7 and newer, while
using the legacy `IStreamReader` contract on older supported TensorRT
versions. Both paths should deserialize the same engine and propagate read
errors.

## System information (Edge Device)

- Platform: NVIDIA Jetson Orin NX 16 GB
- Software release: JetPack 6.2 / L4T 36.4.3
- CPU architecture: aarch64
- GPU compute capability: SM87
- Total device memory: 16 GB
- Build type: Release
- Library versions:
  - TensorRT Edge-LLM: `7f061f21f0a581ba234a1e233c9315b89d8e47d6`
  - CUDA: 12.6
  - TensorRT: 10.3
  - C++ compiler: GCC 11.4
- CMake options:
  - `CMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake`
  - `EMBEDDED_TARGET=jetson-orin`
  - `TRT_PACKAGE_DIR=/usr`

A narrowly version-gated fix and real-engine deserialization validation are
available for a follow-up PR after the issue is approved.
