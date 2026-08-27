#!/usr/bin/env python3
"""Run the seeed-local-voice perf corpus through the Hailo Whisper pipeline.

Metrics match bench/perf/runners.py so the numbers drop into the cross-device table:
  - zh -> CER, en -> WER, with cn2an number normalisation + punctuation strip
  - RTF   = total inference time / audio duration
  - TTFT  = encoder time + first decoded token of the FIRST chunk
Long-form (> encoder window) uses silero-VAD segmentation with a hard cap,
which is the streaming-shaped solution, not a naive fixed cut.
"""
import argparse, json, os, sys, time, wave, tempfile
from pathlib import Path
import numpy as np
import librosa

ROOT = Path(__file__).resolve().parent
EW = Path("/home/harvest/hailo-whisper-bench/edge_whisper")
sys.path.insert(0, str(EW / "benchmarking"))
sys.path.insert(0, str(EW / "inference"))

# ---------------- scoring (mirrors bench/perf/runners.py) ----------------
_PUNCT_TABLE = str.maketrans("", "",
    "，。！？、；：“”‘’（）《》【】「」『』" + "·～—-…,.!?:;\"'()<>[]{}/")
_CN_NUM_RE = None

def _normalize_numbers(text, lang):
    if lang != "zh" or not text:
        return text
    try:
        import cn2an
    except ImportError:
        return text
    import re
    global _CN_NUM_RE
    if _CN_NUM_RE is None:
        _CN_NUM_RE = re.compile(r"[零一二三四五六七八九十百千万亿两]+")
    def _repl(m):
        s = m.group(0)
        try:
            return str(cn2an.cn2an(s, "smart"))
        except Exception:
            return s
    return _CN_NUM_RE.sub(_repl, text)

_T2S = None
def _to_simplified(text):
    """Whisper emits Traditional for zh; every other backend in the matrix emits
    Simplified. Reported as a SEPARATE column, never silently folded in."""
    global _T2S
    if _T2S is None:
        try:
            from opencc import OpenCC
            _T2S = OpenCC("t2s")
        except Exception:
            _T2S = False
    return _T2S.convert(text) if _T2S else text

def _normalize_for_match(text, lang):
    text = _normalize_numbers(text, lang)
    s = text.translate(_PUNCT_TABLE).lower().strip()
    return " ".join(s.split())

def compute_error_rate(reference, hypothesis, lang):
    if not reference or not hypothesis:
        return 1.0
    import jiwer
    ref = _normalize_for_match(reference, lang)
    hyp = _normalize_for_match(hypothesis, lang)
    return jiwer.cer(ref, hyp) if lang == "zh" else jiwer.wer(ref, hyp)

# ---------------- segmentation ----------------
_VAD_SESS = None
def _silero_session():
    """silero-vad ships the ONNX weights but its python package hard-imports
    torchaudio, whose arm64 native lib does not load against this torch build.
    Drive the ONNX graph directly instead."""
    global _VAD_SESS
    if _VAD_SESS is not None:
        return _VAD_SESS
    import onnxruntime as ort, importlib.util
    # find_spec locates the package WITHOUT executing its __init__ (which would
    # pull in torchaudio and blow up on this arm64 torch build)
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("silero_vad package not installed")
    d = Path(list(spec.submodule_search_locations)[0]) / "data"
    for name in ("silero_vad.onnx", "silero_vad_16k_op15.onnx"):
        f = d / name
        if f.exists():
            so = ort.SessionOptions()
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
            names = {i.name: i for i in sess.get_inputs()}
            _VAD_SESS = (sess, names)
            return _VAD_SESS
    raise FileNotFoundError(f"no silero onnx under {d}")

def silero_speech_timestamps(audio, sr, threshold=0.5, min_silence_ms=200, speech_pad_ms=60):
    sess, names = _silero_session()
    win = 512 if sr == 16000 else 256
    ctx_n = 64 if sr == 16000 else 32
    state = np.zeros((2, 1, 128), dtype=np.float32)
    h = np.zeros((2, 1, 64), dtype=np.float32); c = np.zeros((2, 1, 64), dtype=np.float32)
    ctx = np.zeros(ctx_n, dtype=np.float32)
    probs = []
    for i in range(0, len(audio) - win + 1, win):
        cur = audio[i:i + win].astype(np.float32)
        if "state" in names:
            # v5/v6 ONNX expects [context | frame]; feeding the bare frame makes
            # every probability ~0 and the VAD silently finds no speech at all.
            chunk = np.concatenate([ctx, cur])[None, :]
            out = sess.run(None, {"input": chunk, "state": state,
                                  "sr": np.array(sr, dtype=np.int64)})
            probs.append(float(np.asarray(out[0]).reshape(-1)[0])); state = out[1]
            ctx = cur[-ctx_n:]
        else:  # v4 layout: bare frame + h/c
            out = sess.run(None, {"input": cur[None, :], "h": h, "c": c,
                                  "sr": np.array(sr, dtype=np.int64)})
            probs.append(float(np.asarray(out[0]).reshape(-1)[0])); h, c = out[1], out[2]
    if not probs:
        return []
    frame_s = win / sr
    min_sil_frames = max(1, int(min_silence_ms / 1000 / frame_s))
    segs, start, silence = [], None, 0
    for i, p in enumerate(probs):
        if p >= threshold:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_sil_frames:
                segs.append((start * frame_s, (i - silence + 1) * frame_s)); start = None; silence = 0
    if start is not None:
        segs.append((start * frame_s, len(probs) * frame_s))
    pad = speech_pad_ms / 1000
    dur = len(audio) / sr
    return [{"start": max(0.0, s - pad), "end": min(dur, e + pad)} for s, e in segs]

def energy_timestamps(audio, sr, top_db=35):
    intervals = librosa.effects.split(audio, top_db=top_db)
    return [{"start": a / sr, "end": b / sr} for a, b in intervals]

def vad_chunks(audio, sr, max_s, overlap_s=1.0):
    """Speech-aware chunking: split at silence, pack up to max_s.
    Falls back to a hard sliding split for a single speech run longer than max_s."""
    mode = "vad"
    try:
        ts = silero_speech_timestamps(audio, sr)
    except Exception as e:
        print(f"[warn] silero VAD unavailable ({e}); using energy VAD")
        ts, mode = energy_timestamps(audio, sr), "energy"
    if not ts:
        return fixed_chunks(audio, sr, max_s, overlap_s), "fixed"
    chunks, cur_start, cur_end = [], None, None
    for seg in ts:
        s, e = seg["start"], seg["end"]
        if cur_start is None:
            cur_start, cur_end = s, e
            continue
        if e - cur_start <= max_s:
            cur_end = e
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_start is not None:
        chunks.append((cur_start, cur_end))
    out = []
    for s, e in chunks:
        if e - s <= max_s:
            out.append(audio[int(s * sr):int(e * sr)])
        else:
            out.extend(fixed_chunks(audio[int(s * sr):int(e * sr)], sr, max_s, overlap_s))
            mode = mode + "+slide"
    return out, mode

def fixed_chunks(audio, sr, max_s, overlap_s):
    step = max(0.5, max_s - overlap_s)
    n = len(audio)
    out, pos = [], 0
    while pos < n:
        out.append(audio[pos:pos + int(max_s * sr)])
        if pos + int(max_s * sr) >= n:
            break
        pos += int(step * sr)
    return out

def clean_transcription(text, lang, thr=0.55):
    """Repetition cleanup. Hailo's upstream version (hailo-apps .../postprocessing.py)
    only catches exact substring containment, which misses the common case where the
    model paraphrases itself ("...from the plant." / "...more than a plan for..."), and
    it only splits on '.'/'?' so Chinese runaway passes through untouched. Extended
    with CJK sentence enders and a similarity test."""
    import re, difflib
    sentences = [x for x in re.split(r"(?<=[.?!。？！])\s*", text) if x.strip()]
    join = "" if lang == "zh" else " "
    uniq = []
    for sentence in sentences:
        n = sentence.lower().strip()
        for u in uniq:
            nu = u.lower().strip()
            if n and nu and (n in nu or nu in n
                             or difflib.SequenceMatcher(None, nu, n).ratio() >= thr):
                return join.join(uniq)
        uniq.append(sentence.strip())
    return join.join(uniq)


def merge_texts(parts, lang):
    """Drop the repeated head that overlap produces (longest suffix/prefix match)."""
    merged = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not merged:
            merged = p
            continue
        best = 0
        for k in range(min(len(merged), len(p), 40), 2, -1):
            if merged[-k:] == p[:k]:
                best = k
                break
        joined = merged + p[best:]
        merged = joined if lang == "zh" else (merged + " " + p[best:]).replace("  ", " ")
    return merged

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir containing manifest.json + short/ long/")
    ap.add_argument("--encoder_hef_file", default=None)
    ap.add_argument("--encoder_onnx_file", default=None)
    ap.add_argument("--decoder_onnx_dir", default=None)
    ap.add_argument("--decoder_hef_file", default=None)
    ap.add_argument("--decoder_assets_path", default=str(EW / "assets/hailo/decoder_assets"))
    ap.add_argument("--encoder_duration", type=int, default=10)
    ap.add_argument("--padding_cutoff_delta", type=float, default=1.0)
    ap.add_argument("--variant", default="tiny")
    ap.add_argument("--decode-max-tokens", type=int, default=None,
                    help="Raise the decode cap (default 32 for tiny) — ONLY valid with an "
                         "ONNX decoder; the fixed-sequence HEF decoder cannot exceed its "
                         "compiled length, so this is ignored (and reported) for --decoder_hef_file")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lang", choices=["zh","en"], required=True,
                    help="one language per process: a second VDevice in the same process "
                         "hits HAILO_OUT_OF_PHYSICAL_DEVICES on Hailo-8")
    ap.add_argument("--timestamps", action="store_true",
                    help="use the decoder prompt WITHOUT <|notimestamps|> and stop at the "
                         "end of the first timestamped segment (ONNX decoder only)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    files = [f for f in manifest["files"] if f["lang"] == args.lang]
    window_s = args.encoder_duration - args.padding_cutoff_delta  # effective usable window

    results, pipelines = [], {}

    def get_pipeline(lang):
        if lang in pipelines:
            return pipelines[lang]
        mod_name = ("bench_zh" if lang == "zh" else "bench_en") + ("_ts" if args.timestamps else "")
        import importlib
        mod = importlib.import_module(mod_name)
        p = mod.BenchmarkingPipeline(
            encoder_hef_path=args.encoder_hef_file,
            encoder_onnx_path=args.encoder_onnx_file,
            decoder_onnx_dir=args.decoder_onnx_dir,
            decoder_hef_path=args.decoder_hef_file,
            decoder_assets_path=args.decoder_assets_path if args.decoder_hef_file else None,
            variant=args.variant,
            encoder_duration=args.encoder_duration,
        )
        if args.decode_max_tokens:
            if args.decoder_hef_file:
                print(f"[note] --decode-max-tokens ignored: HEF decoder is fixed at "
                      f"{p.decoding_sequence_length} tokens")
            else:
                print(f"[note] decode cap {p.decoding_sequence_length} -> {args.decode_max_tokens}")
                p.decoding_sequence_length = args.decode_max_tokens
        pipelines[lang] = (p, mod)
        return pipelines[lang]

    def run_chunk(pipe, mod, chunk, sr):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp = tf.name
        pcm = np.clip(chunk, -1.0, 1.0)
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
        try:
            mel = mod.get_mel_spectrogram(
                tmp, target_duration=pipe.encoder_target_duration,
                padding_cutoff_delta=args.padding_cutoff_delta,
                format_4d=pipe.encoder_format_4d, is_nhwc=pipe.encoder_is_nhwc)
            txt, enc_ms, dec_ms = pipe.get_transcript(mel, return_timing=True)
            tok = getattr(pipe, "last_token_times", None)
            first_tok = float(tok[0]) if tok else float("nan")
            return txt, enc_ms, dec_ms, first_tok
        finally:
            os.unlink(tmp)

    # warmup per language actually used
    for lang in [args.lang]:
        pipe, mod = get_pipeline(lang)
        wav = corpus / next(f["filename"] for f in files if f["lang"] == lang)
        a, sr = librosa.load(str(wav), sr=16000, mono=True)
        for _ in range(args.warmup):
            run_chunk(pipe, mod, a[:int(window_s * sr)], sr)

    for i, f in enumerate(files, 1):
        wav = corpus / f["filename"]
        lang = f["lang"]
        ref = f.get("eval_transcript") or f["transcript"]
        pipe, mod = get_pipeline(lang)
        audio, sr = librosa.load(str(wav), sr=16000, mono=True)
        dur = len(audio) / sr

        if dur <= window_s:
            chunks, seg_mode = [audio], "single"
        else:
            chunks, seg_mode = vad_chunks(audio, sr, window_s)

        if args.decode_max_tokens and not args.decoder_hef_file:
            # A fixed cap doubles as a runaway-generation brake: whisper often fails to
            # emit EOS on the zero-padding a short utterance needs to fill the fixed
            # encoder window. Bound tokens by audio length instead of a magic constant.
            pipe.decoding_sequence_length = int(max(16, min(args.decode_max_tokens, dur * 8 + 12)))

        parts, enc_tot, dec_tot, ttft = [], 0.0, 0.0, None
        t0 = time.perf_counter()
        for ci, ch in enumerate(chunks):
            txt, enc_ms, dec_ms, first_tok = run_chunk(pipe, mod, ch, sr)
            parts.append(txt)
            enc_tot += enc_ms; dec_tot += dec_ms
            if ci == 0:
                ttft = enc_ms + first_tok
        wall_ms = (time.perf_counter() - t0) * 1000
        hyp_raw = merge_texts(parts, lang)
        hyp = clean_transcription(hyp_raw, lang)
        hyp_s = _to_simplified(hyp) if lang == "zh" else hyp
        err_raw = compute_error_rate(ref, hyp_raw, lang)
        err = compute_error_rate(ref, hyp, lang)
        err_s = compute_error_rate(ref, hyp_s, lang) if lang == "zh" else err
        infer_ms = enc_tot + dec_tot
        row = dict(id=f["id"], lang=lang, category=f["category"], duration_s=dur,
                   n_chunks=len(chunks), seg_mode=seg_mode,
                   encoder_ms=round(enc_tot, 1), decoder_ms=round(dec_tot, 1),
                   infer_ms=round(infer_ms, 1), wall_ms=round(wall_ms, 1),
                   ttft_ms=round(ttft, 1) if ttft else None,
                   rtf=round(infer_ms / 1000 / dur, 4),
                   rtf_wall=round(wall_ms / 1000 / dur, 4),
                   decode_cap=pipe.decoding_sequence_length,
                   err=round(err, 4), err_simplified=round(err_s, 4),
                   err_raw=round(err_raw, 4),
                   ref=ref, hyp=hyp, hyp_raw=hyp_raw)
        results.append(row)
        print(f"[{i}/{len(files)}] {f['id']} {dur:.2f}s chunks={len(chunks)}({seg_mode}) "
              f"enc={enc_tot:.1f} dec={dec_tot:.1f} rtf={row['rtf']:.3f} "
              f"ttft={row['ttft_ms']} err={err:.3f}(raw {err_raw:.3f} t2s {err_s:.3f})\n     hyp: {hyp}", flush=True)

    def agg(pred):
        rows = [r for r in results if pred(r)]
        if not rows:
            return None
        return dict(n=len(rows),
                    err=round(float(np.mean([r["err"] for r in rows])), 4),
                    err_raw=round(float(np.mean([r["err_raw"] for r in rows])), 4),
                    err_simplified=round(float(np.mean([r["err_simplified"] for r in rows])), 4),
                    rtf=round(float(np.mean([r["rtf"] for r in rows])), 4),
                    ttft_ms=round(float(np.mean([r["ttft_ms"] for r in rows if r["ttft_ms"]])), 1),
                    encoder_ms=round(float(np.mean([r["encoder_ms"] for r in rows])), 1),
                    decoder_ms=round(float(np.mean([r["decoder_ms"] for r in rows])), 1))

    summary = {g: agg(lambda r, g=g: (r["lang"], r["category"]) == tuple(g.split("_")))
               for g in ("zh_short", "zh_long", "en_short", "en_long")}
    out = dict(label=args.label, config=vars(args), window_s=window_s,
               summary=summary, rows=results)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== SUMMARY:", args.label, "=====")
    for g, s in summary.items():
        if s:
            print(f"{g:9s} n={s['n']} err={s['err']*100:.2f}% (raw {s['err_raw']*100:.2f}% t2s {s['err_simplified']*100:.2f}%) "
                  f"RTF={s['rtf']:.3f} TTFT={s['ttft_ms']:.0f}ms enc={s['encoder_ms']:.1f} dec={s['decoder_ms']:.1f}")
    print("CORPUS BENCH COMPLETE")

if __name__ == "__main__":
    main()
