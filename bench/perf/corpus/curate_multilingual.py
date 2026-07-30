#!/usr/bin/env python3
"""Curate multilingual smoke corpus.

Pulls 2 utterances (1 short + 1 long) per language from FLEURS into
`bench/perf/corpus/multilingual/<lang>/`. Languages chosen to cover what
multilang preset users typically ask about (ja/es/ko/de/fr).

After running this, append the new entries to manifest.json under a
'multilingual' section (script writes a separate multilingual_manifest.json
to avoid polluting the main 20-file corpus).
"""
from __future__ import annotations
import json, sys, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_MANIFEST = ROOT / "multilingual_manifest.json"

LANGS = {
    "ja": "ja_jp",   # Japanese
    "es": "es_419",  # Spanish (Latin America; most edge users)
    "ko": "ko_kr",   # Korean
    "de": "de_de",   # German
    "fr": "fr_fr",   # French
}
BANDS = {"short": (1.5, 4.0), "long": (8.0, 16.0)}


# Peak-normalization target, in units of full scale (0.89 ~= -1 dBFS). Same
# value and same reason as curate_public_corpus.py: raw FLEURS levels vary by
# ~60 dB across speakers and languages, and an ASR front-end reads a very quiet
# utterance as silence and returns an empty transcript. That already invalidated
# one round of perf numbers on the zh/en corpus before it was caught, and this
# script emits five more languages from the same source.
PEAK_TARGET = 0.89


def peak_normalize(arr):
    """Scale a float waveform so its absolute peak sits at PEAK_TARGET.

    No-op for all-silent input (nothing to normalize against).
    """
    import numpy as np
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.abs(arr).max()) if arr.size else 0.0
    if peak <= 0.0:
        return arr
    return arr * (PEAK_TARGET / peak)


def pcm16_from_float(arr):
    """fleurs gives float32 in [-1, 1]; peak-normalize, then convert to int16 PCM.

    Pure gain: sample count, duration and sample rate are untouched. Clipping is
    kept as a safety net for material already above full scale (normalization
    brings loud input down, so it should never trigger).
    """
    import numpy as np
    clipped = np.clip(peak_normalize(arr), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def write_wav(path: Path, samples_int16, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples_int16.tobytes())


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install 'datasets<3' soundfile numpy")

    entries = []
    for lang_short, fl in LANGS.items():
        print(f"\n=== {lang_short} ({fl}) ===")
        picked = {"short": None, "long": None}
        for split in ("test", "train"):
            if all(picked.values()):
                break
            try:
                ds = load_dataset("google/fleurs", fl, split=split,
                                  streaming=True, trust_remote_code=True)
            except Exception as e:
                print(f"  load failed: {e}")
                continue
            for sample in ds:
                arr = sample["audio"]["array"]
                sr  = sample["audio"]["sampling_rate"]
                dur = len(arr) / sr
                raw = (sample.get("raw_transcription")
                       or sample.get("transcription") or "").strip()
                if not raw:
                    continue
                for cat, (lo, hi) in BANDS.items():
                    if picked[cat] is None and lo <= dur <= hi:
                        picked[cat] = {"samples": pcm16_from_float(arr), "sr": sr,
                                       "dur": round(dur, 2), "transcript": raw}
                        print(f"  + {cat}: dur={dur:.2f}s '{raw[:50]}...'")
                        break
                if all(picked.values()):
                    break
        for cat, item in picked.items():
            if not item:
                print(f"  ! missing {lang_short}_{cat}")
                continue
            file_id  = f"{lang_short}_{cat}_01"
            filename = f"multilingual/{lang_short}/{file_id}.wav"
            path = ROOT / filename
            write_wav(path, item["samples"], item["sr"])
            entries.append({
                "id": file_id, "filename": filename,
                "lang": lang_short, "category": cat,
                "duration_s": item["dur"],
                "transcript": item["transcript"],
                "sha256": "",
            })

    OUT_MANIFEST.write_text(json.dumps(
        {"version": 1,
         "description": "Multilingual smoke corpus (ja/es/ko/de/fr). FLEURS subset, CC BY 4.0.",
         "audio_spec": {"sample_rate": 16000, "channels": 1, "bit_depth": 16, "format": "wav"},
         "files": entries},
        indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(entries)} entries to {OUT_MANIFEST}")
    print("Next:")
    print("  python fetch.py --recompute-hashes --manifest multilingual_manifest.json")
    print("  hf upload datasets/harvestsu/seeed-local-voice-perf-corpus  multilingual/  multilingual/  --repo-type dataset")


if __name__ == "__main__":
    main()
