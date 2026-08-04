#!/usr/bin/env python3
"""Print ONNX graph I/O shapes and operator counts without loading weights."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import onnx


def value_info(value: onnx.ValueInfoProto) -> dict[str, object]:
    tensor = value.type.tensor_type
    dims: list[int | str | None] = []
    for dim in tensor.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return {"name": value.name, "dtype": tensor.elem_type, "shape": dims}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    model = onnx.load_model(args.model, load_external_data=False)
    operators = collections.Counter(node.op_type for node in model.graph.node)
    print(
        json.dumps(
            {
                "model": str(args.model.resolve()),
                "inputs": [value_info(value) for value in model.graph.input],
                "outputs": [value_info(value) for value in model.graph.output],
                "node_count": len(model.graph.node),
                "operators": dict(sorted(operators.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
