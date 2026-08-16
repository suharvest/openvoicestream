#!/usr/bin/env python3
"""Run the official Transformers Nemotron ASR path on a pinned local checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.audio_utils import load_audio


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_weight_identity(model_dir: Path) -> list[dict[str, Any]]:
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise RuntimeError(f"no safetensors weights found in {model_dir}")
    return [
        {"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
        for path in weights
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    # The reference gate must never silently fetch a different checkpoint.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["files"][: args.limit]
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForRNNT.from_pretrained(
        args.model_dir,
        local_files_only=True,
        device_map="auto",
    )
    model.eval()
    sampling_rate = processor.feature_extractor.sampling_rate

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for entry in entries:
            audio_path = args.corpus_root / entry["filename"]
            actual_sha = sha256(audio_path)
            if actual_sha != entry["sha256"]:
                raise RuntimeError(f"{entry['id']}: WAV SHA mismatch: {actual_sha}")
            audio = load_audio(str(audio_path), sampling_rate=sampling_rate)
            language = "zh-CN" if entry["lang"] == "zh" else "en-US"
            inputs = processor(audio, sampling_rate=sampling_rate, language=language)
            inputs = inputs.to(model.device, dtype=model.dtype)
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            started = time.perf_counter()
            generated = model.generate(**inputs, return_dict_in_generate=True)
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            transcript = processor.decode(generated.sequences[0], skip_special_tokens=True)
            rows.append(
                {
                    "id": entry["id"],
                    "lang": entry["lang"],
                    "language_prompt": language,
                    "wav": str(audio_path),
                    "wav_sha256": actual_sha,
                    "duration_s": entry["duration_s"],
                    "text": transcript,
                    "elapsed_ms": elapsed_ms,
                    "rtf": elapsed_ms / 1000.0 / entry["duration_s"],
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    payload = {
        "schema_version": 1,
        "implementation": "transformers_AutoModelForRNNT",
        "model_dir": str(args.model_dir),
        "model_weights": model_weight_identity(args.model_dir),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(model.device),
            "dtype": str(model.dtype),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
