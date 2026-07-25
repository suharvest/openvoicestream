## Describe the bug

`tensorrt_edgellm.checkpoint.loader._set_tensor` in `v0.9.1` / current
`main` casts a BF16 source tensor to FP16, but otherwise directly replaces the
destination parameter or buffer with the checkpoint tensor.

Consequently, loading an FP32 checkpoint tensor into a module whose existing
parameter is declared FP16 or BF16 changes that parameter to FP32. This
creates a mixed-dtype model, increases memory use, and can later produce
unexpected ONNX/export or engine-build dtypes.

### Steps/Code to reproduce bug

**Installation method:**

Built from source at
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`.

Minimal CPU reproduction:

```python
import torch
from torch import nn
from tensorrt_edgellm.checkpoint.loader import _set_tensor

model = nn.Linear(4, 4, bias=False).to(dtype=torch.float16)
assert model.weight.dtype == torch.float16

checkpoint_weight = torch.ones_like(model.weight, dtype=torch.float32)
assert _set_tensor(model, "weight", checkpoint_weight)

# Actual: torch.float32
print(model.weight.dtype)
```

Current implementation:
<https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/7f061f21f0a581ba234a1e233c9315b89d8e47d6/tensorrt_edgellm/checkpoint/loader.py#L402-L439>

### Expected behavior

Floating checkpoint tensors should be converted to an existing destination's
declared FP16/BF16 dtype before assignment. Already matching tensors should
remain unchanged, and deliberately FP32 buffers such as scale tensors must
remain FP32. Non-floating tensors and destinations that do not yet exist
should retain their current behavior.

## System information (x86 Host with GPU)

- Installation: source checkout
- OS: Linux / WSL2
- CPU architecture: x86_64
- TensorRT Edge-LLM:
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6`
- Python: 3.12
- PyTorch: 2.12.0+cu130

A minimal loader-only fix is prepared separately from any model-specific
mixed-precision changes. The follow-up PR will include CPU tests for
FP32-to-FP16, FP32-to-BF16, already-matching, FP32 destination, and
non-floating behavior.
