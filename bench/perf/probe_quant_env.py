#!/usr/bin/env python3
"""Fail-loud probe for the x86 GPU quantization/export environment."""

from __future__ import annotations

import json


def main() -> int:
    import modelopt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA is unavailable")
    device = torch.device("cuda:0")
    left = torch.randn((256, 256), device=device, dtype=torch.float16)
    right = torch.randn((256, 256), device=device, dtype=torch.float16)
    result = left @ right
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise RuntimeError("CUDA FP16 matrix probe produced non-finite output")
    properties = torch.cuda.get_device_properties(device)
    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "modelopt": getattr(modelopt, "__version__", "unknown"),
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
        "matrix_probe": "pass",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
