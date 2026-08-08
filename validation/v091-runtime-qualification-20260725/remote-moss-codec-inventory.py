#!/usr/bin/env python3
import json
import os

codec = (
    "/home/harvest/edgellm-artifacts/"
    "orin-nx-edgellm-v091-jp62-trt103-sm87-20260725/"
    "v091/engines/moss/codec"
)
meta_path = os.path.join(codec, "codec_browser_onnx_meta.json")
with open(meta_path, encoding="utf-8") as stream:
    meta = json.load(stream)

print(f"meta={meta_path}")
print(f"top_keys={sorted(meta)}")
print(f"files={meta.get('files')!r}")
print(f"external_data_files={meta.get('external_data_files')!r}")
print(f"codec_files={sorted(os.listdir(codec))!r}")

encode = meta["files"]["encode"]
encode_path = os.path.normpath(os.path.join(codec, encode))
assert os.path.commonpath([codec, encode_path]) == codec
print(f"encode_relative={encode}")
print(f"encode_absolute={encode_path}")
print(f"encode_size={os.path.getsize(encode_path)}")

with open(encode_path, "rb") as stream:
    onnx_data = stream.read()

candidate_files = [
    name
    for name in os.listdir(codec)
    if os.path.isfile(os.path.join(codec, name))
    and name not in {os.path.basename(encode_path), os.path.basename(meta_path)}
]
referenced = [
    name
    for name in candidate_files
    if name.encode("utf-8") in onnx_data
]
print(f"onnx_external_data_references={sorted(referenced)!r}")

try:
    import onnx
except ImportError:
    print("onnx_parser=unavailable")
else:
    model = onnx.load_model(encode_path, load_external_data=False)
    external_locations = sorted({
        entry.value
        for tensor in model.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    })
    print(f"onnx_parser=onnx-{onnx.__version__}")
    print(f"onnx_external_locations={external_locations!r}")
