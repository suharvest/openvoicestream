## Describe the bug

The fused NVFP4 GEMM/all-reduce plugin in `v0.9.1` / current `main`
(`7f061f21f0a581ba234a1e233c9315b89d8e47d6`) unconditionally references
`nvinfer1::DataType::kFP4`. The enum is not available before TensorRT 10.8,
so compiling all plugins fails with the TensorRT 10.3 headers shipped in
JetPack 6.2 even when no FP4 model or engine is requested.

This report is specifically about the remaining unguarded references in
`fusedNvfp4GemmAllReducePlugin.cpp`. PR #82 previously guarded a different
`kFP4` use in `cpp/common/tensor.cpp`; it was closed without merging, and the
current main source still contains the two plugin references below.

Related: <https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/82>

### Steps/Code to reproduce bug

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

The plugin compilation encounters `DataType::kFP4` at:

- <https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/7f061f21f0a581ba234a1e233c9315b89d8e47d6/cpp/plugins/fusedNvfp4GemmAllReducePlugin/fusedNvfp4GemmAllReducePlugin.cpp#L216-L223>

TensorRT 10.3 and 10.7 do not define that enum.

### Expected behavior

FP4-specific format cases should be compiled only when the TensorRT headers
provide `DataType::kFP4` (TensorRT 10.8+). Builds against older TensorRT
versions should still compile the non-FP4 cases and must not advertise FP4
support. Behavior on TensorRT 10.8+ should remain unchanged.

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

A minimal version guard is available for a follow-up PR, with compile
coverage planned for both TensorRT `<10.8` and `>=10.8`.
