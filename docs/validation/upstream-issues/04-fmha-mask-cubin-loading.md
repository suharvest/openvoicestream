## Describe the bug

The context-FMHA loader in TensorRT-Edge-LLM `v0.9.1` / current `main`
initially filters and caches cubins only by data type and SM. It eagerly calls
`cuModuleLoadData` for every matching cubin before the requested attention
mask type participates in function selection.

On JetPack 6.2 / CUDA 12.6 / SM87, a non-custom-mask Qwen3.5 engine build
therefore attempts to load two unrelated `CUSTOM_MASK` cubins that were
generated with CUDA 12.8. Both fail with `CUDA_ERROR_INVALID_IMAGE`, blocking
an otherwise supported causal/non-custom-mask engine.

The two failing, unused cubins are:

```text
fmha_v2_flash_attention_fp16_fp32_64_128_S_q_k_v_256_custom_mask_sm87.cubin.cpp
fmha_v2_flash_attention_fp16_fp32_64_16_S_q_k_v_256_custom_mask_sm87.cubin.cpp
```

Nine non-custom-mask SM87 context-FMHA cubins and all required Qwen XQA
cubins load successfully on the same system.

### Steps/Code to reproduce bug

Build the unmodified project for SM87 and run `llm_build` for a Qwen3.5
configuration with vision/custom-mask attention disabled:

```bash
./build/examples/llm/llm_build \
  --onnxDir ./qwen35-onnx \
  --engineDir ./qwen35-engine \
  --maxBatchSize 1
```

The build reaches context-FMHA initialization and fails while loading one of
the unused `custom_mask_sm87` modules with
`CUDA_ERROR_INVALID_IMAGE`.

Current load/cache implementation:
<https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/7f061f21f0a581ba234a1e233c9315b89d8e47d6/cpp/kernels/contextAttentionKernels/contextFMHARunner.cpp#L193-L250>

### Expected behavior

The loader should load and cache only cubins eligible for the requested data
type, SM, and attention mask contract. A causal/non-custom-mask engine should
not load unrelated custom-mask modules. If `CUSTOM_MASK` is actually
requested and its required cubin is unsupported, the request should continue
to fail loudly rather than silently falling back.

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

A mask-scoped loader/cache change has completed a fresh non-custom-mask
engine build and direct inference. A follow-up PR will include focused
cache-order coverage and a true custom-mask fail-loud regression.
