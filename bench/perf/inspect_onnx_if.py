#!/usr/bin/env python3
"""Print ONNX If nodes and the input/output shapes of both branch graphs."""

from __future__ import annotations

import argparse

import onnx


def shape(value: onnx.ValueInfoProto) -> list[str | int]:
    dims = value.type.tensor_type.shape.dim
    return [dim.dim_value or dim.dim_param or "?" for dim in dims]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    model = onnx.load(args.model, load_external_data=False)
    for node in model.graph.node:
        if node.op_type != "If":
            continue
        print(f"node={node.name!r} inputs={list(node.input)} outputs={list(node.output)}")
        for attribute in node.attribute:
            if attribute.type != onnx.AttributeProto.GRAPH:
                continue
            graph = attribute.g
            print(f"  branch={attribute.name}")
            for value in graph.input:
                print(f"    input {value.name}: {shape(value)}")
            for value in graph.output:
                print(f"    output {value.name}: {shape(value)}")
            print(f"    nodes={len(graph.node)}")
            for branch_node in graph.node:
                print(
                    f"      {branch_node.op_type} {branch_node.name!r} "
                    f"in={list(branch_node.input)} out={list(branch_node.output)}"
                )
        needed = set(node.input)
        visited: set[str] = set()
        producers = {output: producer for producer in model.graph.node for output in producer.output}
        while needed:
            output = needed.pop()
            if output in visited:
                continue
            visited.add(output)
            producer = producers.get(output)
            if producer is None:
                continue
            print(
                f"  condition {producer.op_type} {producer.name!r} "
                f"in={list(producer.input)} out={list(producer.output)}"
            )
            needed.update(producer.input)


if __name__ == "__main__":
    main()
