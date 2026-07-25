## Describe the bug

Exporting the public Qwen3-ASR checkpoint with TensorRT-Edge-LLM `v0.9.1` /
current `main` can produce a runtime LLM config with:

```json
{
  "rope_scaling": {
    "type": "linear",
    "rope_type": "linear",
    "factor": 1.0,
    "mrope_section": [24, 20, 20]
  }
}
```

The checkpoint contains MRoPE sections, but the current normalization helper
does not canonicalize `linear + mrope_section` to MRoPE. The C++ runtime does
not select its MRoPE path for this exported config. The engine can build, but
runtime execution then lacks `mropeCosSinOut`, producing a silent
configuration/correctness failure instead of a clean export error.

### Steps/Code to reproduce bug

**Installation method:**

Built from source at
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`.

**Model:**

```text
Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0
```

**Export flow:**

```bash
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-ASR-0.6B \
  --output_dir ./qwen3-asr-int4 \
  --quantization int4_awq

tensorrt-edgellm-export llm \
  --model_dir ./qwen3-asr-int4 \
  --output_dir ./qwen3-asr-onnx
```

Inspect the generated LLM `config.json`. Both `rope_scaling.type` and
`rope_scaling.rope_type` remain `linear` while `mrope_section` is present.

Current normalization:
<https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/7f061f21f0a581ba234a1e233c9315b89d8e47d6/tensorrt_edgellm/checkpoint/checkpoint_utils.py#L54-L72>

### Expected behavior

For Qwen3-ASR, an exported rope config containing `mrope_section` should use
the canonical MRoPE semantics expected by the C++ runtime. The exporter
should preserve all unrelated rope parameters, leave other model families
unchanged, and be idempotent for already-canonical MRoPE configs.

## System information (x86 Host with GPU)

- Container used: source virtual environment
- OS: WSL2 / Linux
- CPU architecture: x86_64
- GPU: NVIDIA GeForce RTX 3060
- GPU memory: 12 GB
- Number of GPUs: 1
- Library versions:
  - Python: 3.12
  - TensorRT Edge-LLM: `7f061f21f0a581ba234a1e233c9315b89d8e47d6`
  - CUDA: 13.0 build environment
  - PyTorch: 2.12.0+cu130
  - ModelOpt: 0.44.0

A focused config-level regression test and minimal exporter fix are available
for a follow-up PR after issue approval. Device validation used a freshly
built SM87 engine; only the exported rope config changed.
