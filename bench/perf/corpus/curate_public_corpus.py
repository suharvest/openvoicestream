#!/usr/bin/env python3
"""Curate the perf corpus from Google FLEURS (CC BY 4.0).

Why FLEURS instead of AISHELL/LibriSpeech:
- Streamable via HuggingFace datasets — no 19 GB tarball pull
- Same source for zh + en (cmn_hans_cn / en_us), removes cross-dataset bias
- Native 16 kHz mono, no re-encoding needed
- Apache-compatible CC BY 4.0 license; we just need to credit the source

Run ONCE on a dev box (Mac), then upload the resulting tarball to Seeed CDN
so every device pulls the same bytes via `fetch.py --from cdn`.

Usage:
  pip install datasets soundfile
  python curate_public_corpus.py
  python fetch.py --recompute-hashes   # then commit manifest.json
"""
from __future__ import annotations
import json, sys, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"

# Target counts per (lang, category)
TARGETS = {
    ("zh", "short"): 5,
    ("zh", "long"):  5,
    ("en", "short"): 5,
    ("en", "long"):  5,
}

# Duration band per category (seconds). FLEURS train tends to have utterances
# slightly above 3.5s, so we widen the short upper bound to fill 5/5.
BANDS = {
    "short": (1.5, 4.0),
    "long":  (10.0, 20.0),
}


def _normalize_transcript(text: str, lang: str) -> str:
    """FLEURS zh transcripts are space-separated by character ('我 们 都'); rejoin.
    Also strip outer whitespace."""
    text = text.strip()
    if lang == "zh":
        # remove ALL spaces between Chinese chars
        text = "".join(text.split())
    return text

# FLEURS dataset IDs
FLEURS_LANG = {"zh": "cmn_hans_cn", "en": "en_us"}


# Peak-normalization target, in units of full scale (0.89 ~= -1 dBFS).
#
# WHY: ASR front-ends are level-sensitive — a VAD/energy gate sees a very quiet
# utterance as silence and the recognizer returns an empty transcript. Raw FLEURS
# recording levels vary by ~60 dB across speakers/languages (measured on this
# corpus: en_short_01 peaked at -49 dBFS while zh_short_01 peaked at -3 dBFS),
# which made the quiet English files transcribe as empty strings and silently
# invalidated the perf numbers. Normalizing every emitted WAV to one peak makes
# the corpus level-consistent, and it is a pure gain change: sample count,
# duration and sample rate are untouched.
#
# 0.89 rather than 1.0 leaves ~1 dB of headroom so the int16 rounding cannot
# wrap, while staying loud enough for any ASR gate.
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

    Clipping is retained as a safety net for anything already above full scale
    (normalization brings loud material down, so this should never trigger).
    """
    import numpy as np
    clipped = np.clip(peak_normalize(arr), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def write_wav(path: Path, samples_int16, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int16.tobytes())


def curate_lang(lang: str, picked: dict[tuple, list]) -> None:
    """Stream FLEURS for a language; fill `picked` lists until TARGETS satisfied."""
    from datasets import load_dataset
    fl = FLEURS_LANG[lang]
    print(f"\n=== FLEURS {fl} ===")
    # Try test split first, then dev, then train, until quotas met
    for split in ("test", "dev", "train"):
        if all(len(picked[(lang, cat)]) >= TARGETS[(lang, cat)] for cat in ("short", "long")):
            break
        print(f"  -- split={split} --")
        try:
            ds = load_dataset("google/fleurs", fl, split=split, streaming=True, trust_remote_code=True)
        except Exception as e:
            print(f"  load_dataset failed: {e}")
            continue
        for sample in ds:
            arr = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            dur = len(arr) / sr
            raw = sample.get("raw_transcription") or sample.get("transcription") or ""
            transcript = _normalize_transcript(raw, lang)
            if not transcript:
                continue
            # find which category this fits
            for cat, (lo, hi) in BANDS.items():
                if lo <= dur <= hi and len(picked[(lang, cat)]) < TARGETS[(lang, cat)]:
                    picked[(lang, cat)].append({
                        "samples": pcm16_from_float(arr),
                        "sr": sr,
                        "dur": round(dur, 2),
                        "transcript": transcript,
                    })
                    print(f"  + {lang}_{cat} #{len(picked[(lang, cat)])}: dur={dur:.2f}s  '{transcript[:40]}...'")
                    break
            if all(len(picked[(lang, cat)]) >= TARGETS[(lang, cat)] for cat in ("short", "long")):
                break


def main():
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        sys.exit("pip install datasets soundfile  (and re-run)")

    picked: dict[tuple, list] = {k: [] for k in TARGETS}
    for lang in ("zh", "en"):
        curate_lang(lang, picked)

    # Write WAVs + update manifest
    manifest = json.loads(MANIFEST.read_text())
    by_id = {e["id"]: e for e in manifest["files"]}

    for (lang, cat), items in picked.items():
        for i, item in enumerate(items, 1):
            file_id = f"{lang}_{cat}_{i:02d}"
            if file_id not in by_id:
                print(f"  ! manifest has no entry for {file_id}, skipping")
                continue
            entry = by_id[file_id]
            path = ROOT / entry["filename"]
            write_wav(path, item["samples"], item["sr"])
            entry["transcript"]  = item["transcript"]
            entry["duration_s"]  = item["dur"]
            print(f"  wrote {entry['filename']:30s}  {item['dur']:.2f}s  {item['transcript'][:40]}...")

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Summary ===")
    for (lang, cat), items in picked.items():
        target = TARGETS[(lang, cat)]
        status = "ok" if len(items) >= target else "SHORT"
        print(f"  {lang}_{cat}: {len(items)}/{target}  {status}")

    print("\nNext:")
    print("  python fetch.py --recompute-hashes")
    print("  git add -p manifest.json")
    print("  tar czf models-perf-corpus.tar.gz -C bench/perf/corpus short long")
    print("  # upload models-perf-corpus.tar.gz to Seeed CDN")
    print("\nCredit (BibTeX in your README):")
    print('  Conneau et al. 2022, "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech"')
    print("  CC BY 4.0 — https://huggingface.co/datasets/google/fleurs")


if __name__ == "__main__":
    main()
