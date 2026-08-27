#!/usr/bin/env python3
"""Same corpus, same output schema — but through the voxedge WhisperASR backend.

The other runners in this directory drive each platform's runtime directly.
This one goes through the shipped backend, so what it measures is what a
deployment actually gets: the backend's silence-based segmentation, its
degeneration guards, and its joiner, rather than the harness's fixed-hop split
with overlap stitching.

Emits the schema ``score_all.py`` reads, so a backend run and a harness run can
be scored side by side by one copy of the scoring code.
"""
import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, f"{path}: {w.getframerate()} Hz"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "zh"])
    ap.add_argument("--encoder-kind", required=True, choices=["hailo", "rknn", "tensorrt"])
    ap.add_argument("--encoder", required=True, help=".hef / .rknn / .plan")
    ap.add_argument("--decoder-dir", required=True, help="optimum ONNX export dir")
    ap.add_argument("--vocab-dir", required=True)
    ap.add_argument("--window-s", type=float, required=True,
                    help="must equal the window the encoder graph was built at")
    ap.add_argument("--padding-cutoff-s", type=float, default=0.0)
    ap.add_argument("--decoder-threads", type=int, default=0)
    ap.add_argument("--all-cores", action="store_true")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from voxedge.backends.whisper import WhisperASR, WhisperASRConfig

    be = WhisperASR(WhisperASRConfig(
        encoder_kind=args.encoder_kind,
        encoder_path=args.encoder,
        decoder_dir=args.decoder_dir,
        vocab_dir=args.vocab_dir,
        window_s=args.window_s,
        language=args.lang,
        padding_cutoff_s=args.padding_cutoff_s,
        decoder_threads=args.decoder_threads,
        all_cores=args.all_cores,
    ))
    t0 = time.perf_counter()
    be.preload()
    print(f"preload {(time.perf_counter() - t0):.1f}s", flush=True)

    manifest = json.loads((Path(args.corpus) / "manifest.json").read_text(encoding="utf-8"))
    files = [f for f in manifest["files"] if f["lang"] == args.lang]

    rows = []
    for i, f in enumerate(files, 1):
        audio = load_wav(Path(args.corpus) / f["filename"])
        dur = len(audio) / SAMPLE_RATE
        t = time.perf_counter()
        res = be.transcribe_array(audio, args.lang)
        wall_ms = (time.perf_counter() - t) * 1000
        m = res.meta
        # RTF the way the other runners define it: inference only, so the
        # mel front end is excluded and the numbers stay comparable.
        infer_ms = m["encoder_ms"] + m["decoder_ms"]
        row = dict(
            id=f["id"], lang=args.lang, category=f["category"], duration_s=dur,
            n_chunks=m["chunks"], seg_mode="silence-split",
            encoder_ms=round(m["encoder_ms"], 1), decoder_ms=round(m["decoder_ms"], 1),
            preprocess_ms=round(wall_ms - infer_ms, 1),
            infer_ms=round(infer_ms, 1),
            ttft_ms=m["ttft_ms"], n_tokens=0,
            rtf=round(infer_ms / 1000 / dur, 4),
            rtf_wall=round(wall_ms / 1000 / dur, 4),
            segments_dropped=m["segments_dropped"],
            ref=f.get("eval_transcript") or f["transcript"], hyp=res.text,
        )
        rows.append(row)
        print(f"[{i}/{len(files)}] {f['id']} {dur:.2f}s chunks={m['chunks']} "
              f"enc={m['encoder_ms']:.1f} dec={m['decoder_ms']:.1f} rtf={row['rtf']:.4f} "
              f"ttft={row['ttft_ms']}\n     {res.text}", flush=True)

    Path(args.out).write_text(json.dumps(dict(
        label=args.label, window_s=args.window_s, variant="base",
        decoder="onnx-kvcache-cpu", driver="voxedge.backends.whisper",
        rows=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    be.unload()
    print("BACKEND RUN COMPLETE")


if __name__ == "__main__":
    main()
