#!/usr/bin/env python3
"""whisper.cpp (CUDA) corpus runner for Jetson.

Emits the same row schema as rknn_whisper_run.py so one scorer covers every
platform. Per-stage timings come from whisper.cpp's own whisper_print_timings,
which reports encode and decode separately — the same split measured on the
NPU platforms.
"""
import argparse, json, re, subprocess, time
from pathlib import Path

TIMING = {
    "mel": re.compile(r"mel time\s*=\s*([\d.]+)\s*ms"),
    "sample": re.compile(r"sample time\s*=\s*([\d.]+)\s*ms"),
    "encode": re.compile(r"encode time\s*=\s*([\d.]+)\s*ms"),
    "decode": re.compile(r"decode time\s*=\s*([\d.]+)\s*ms"),
    "batchd": re.compile(r"batchd time\s*=\s*([\d.]+)\s*ms"),
    "prompt": re.compile(r"prompt time\s*=\s*([\d.]+)\s*ms"),
    "total": re.compile(r"total time\s*=\s*([\d.]+)\s*ms"),
}


def parse_timings(stderr):
    out = {}
    for k, rx in TIMING.items():
        m = rx.search(stderr)
        out[k] = float(m.group(1)) if m else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    files = [f for f in manifest["files"] if f["lang"] == args.lang]

    def run_one(wav):
        # NOT -np: cli.cpp gates whisper_print_timings behind !no_prints, and the
        # per-stage encode/decode numbers parsed below come from exactly that block.
        cmd = [args.bin, "-m", args.model, "-f", str(wav), "-l", args.lang,
               "-t", str(args.threads), "-nt"]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = (time.perf_counter() - t0) * 1000
        return p.stdout.strip(), parse_timings(p.stderr), wall, p.returncode, p.stderr

    first = corpus / files[0]["filename"]
    for _ in range(args.warmup):
        run_one(first)

    rows = []
    for i, f in enumerate(files, 1):
        wav = corpus / f["filename"]
        text, t, wall, rc, err = run_one(wav)
        if rc != 0:
            print(f"[{i}/{len(files)}] {f['id']} FAILED rc={rc}\n{err[-500:]}")
            continue
        # without -np, stdout still holds only the transcript (progress and
        # timings go to stderr), but strip any stray bracketed timestamps.
        hyp = " ".join(re.sub(r"\[[^\]]*\]", " ", text).split())
        enc = t["encode"]
        dec = t["decode"] + t["batchd"] + t["prompt"]
        infer = enc + dec
        dur = f["duration_s"]
        row = dict(id=f["id"], lang=args.lang, category=f["category"], duration_s=dur,
                   n_chunks=1, seg_mode="whisper.cpp-internal-30s",
                   encoder_ms=round(enc, 1), decoder_ms=round(dec, 1),
                   preprocess_ms=round(t["mel"], 1),
                   infer_ms=round(infer, 1), wall_ms=round(wall, 1),
                   # whisper.cpp does not expose a first-token timestamp; encode
                   # plus one decode step is the closest honest proxy and is
                   # labelled as such rather than presented as a measured TTFT.
                   ttft_ms=round(enc + t["sample"], 1), ttft_is_proxy=True,
                   n_tokens=0,
                   rtf=round(infer / 1000 / dur, 4),
                   rtf_wall=round(wall / 1000 / dur, 4),
                   ref=f.get("eval_transcript") or f["transcript"], hyp=hyp)
        rows.append(row)
        print(f"[{i}/{len(files)}] {f['id']} {dur:.2f}s enc={enc:.1f} dec={dec:.1f} "
              f"rtf={row['rtf']:.3f} wall={wall:.0f}ms\n     {hyp}", flush=True)

    Path(args.out).write_text(json.dumps(
        dict(label=args.label, window_s=30, variant=Path(args.model).stem,
             decoder="whisper.cpp-ggml-cuda", rows=rows),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("WCPP RUN COMPLETE")


if __name__ == "__main__":
    main()
