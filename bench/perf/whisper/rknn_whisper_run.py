#!/usr/bin/env python3
"""Whisper on Rockchip NPU — corpus runner.

Emits raw transcripts + per-stage timings as JSON. Scoring happens off-device so
that every platform in the comparison goes through one copy of the scoring code
(bench/perf/runners.py) instead of a per-device reimplementation.

Front end is a numpy port of the official demo's torch mel so the boards need
neither torch nor librosa; it is bit-comparable with
rknn_model_zoo/examples/whisper/python/whisper.py.
"""
import argparse, json, time
from pathlib import Path
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 20           # default; overridden by --encoder_duration
MAX_LENGTH = CHUNK_LENGTH * 100
N_MELS = 80

EOT = 50257
SOT = 50258
TASK_TRANSCRIBE = 50359
NO_TIMESTAMPS = 50363
TIMESTAMP_BEGIN = 50364
LANG = {"en": 50259, "zh": 50260}


# ---------------- mel front end (numpy port of the demo's torch code) ----------------
def _hann(n):
    # torch.hann_window default is periodic=True
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)


def _stft_mag2(audio):
    """|STFT|^2 matching torch.stft(n_fft, hop, hann, center=True, pad_mode='reflect')."""
    pad = N_FFT // 2
    x = np.pad(audio.astype(np.float64), (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    frames = x[idx] * _hann(N_FFT)[None, :]
    spec = np.fft.rfft(frames, n=N_FFT, axis=-1)          # (frames, 201)
    return (np.abs(spec) ** 2).T                          # (201, frames)


def log_mel_spectrogram(audio, filters, max_frames=None):
    # Pad the *waveform* to the full window before the STFT, the way whisper and
    # the official demo do. Zero-padding the finished mel instead leaves 0.0 in
    # the tail, while the mel of digital silence is about -0.58 -- i.e. the
    # encoder gets shown a constant that never occurs in training.
    if max_frames is not None:
        n_samples = max_frames * HOP_LENGTH
        a = np.zeros(n_samples, dtype=np.float64)
        n = min(len(audio), n_samples)
        a[:n] = audio[:n]
        audio = a
    mag = _stft_mag2(audio)[:, :-1]                       # demo drops the last frame
    mel = filters @ mag
    log_spec = np.log10(np.clip(mel, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def pad_or_trim(mel):
    """Only trims now — padding happens on the waveform in log_mel_spectrogram."""
    if mel.shape[1] >= MAX_LENGTH:
        return np.ascontiguousarray(mel[:, :MAX_LENGTH])
    out = np.zeros((N_MELS, MAX_LENGTH), dtype=np.float32)
    out[:, :mel.shape[1]] = mel
    return out


# ---------------- vocab (Rockchip ships id->token text, base64 for zh) ----------------
def read_vocab(path):
    """Verbatim behaviour of the demo's read_vocab: split on the FIRST space."""
    vocab = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                key, value = parts[0], ""
            else:
                key, value = parts[0], parts[1]
            vocab[key] = value
    return vocab


def _b64_index(c):
    if "A" <= c <= "Z":
        return ord(c) - ord("A")
    if "a" <= c <= "z":
        return ord(c) - ord("a") + 26
    if "0" <= c <= "9":
        return ord(c) - ord("0") + 52
    if c == "+":
        return 62
    if c == "/":
        return 63
    raise ValueError(f"bad base64 char {c!r}")


def base64_decode(encoded_string):
    """Verbatim port of the demo's hand-rolled decoder. The stdlib one is NOT a
    drop-in: this returns a single space the moment it meets '=', which is how
    the zh vocab encodes a word break."""
    if not encoded_string:
        return ""
    out = bytearray(len(encoded_string) // 4 * 3)
    i = oi = 0
    while i < len(encoded_string):
        if encoded_string[i] == "=":
            return " "
        out[oi] = (_b64_index(encoded_string[i]) << 2) + ((_b64_index(encoded_string[i + 1]) & 0x30) >> 4)
        if i + 2 < len(encoded_string) and encoded_string[i + 2] != "=":
            out[oi + 1] = ((_b64_index(encoded_string[i + 1]) & 0x0F) << 4) + ((_b64_index(encoded_string[i + 2]) & 0x3C) >> 2)
            if i + 3 < len(encoded_string) and encoded_string[i + 3] != "=":
                out[oi + 2] = ((_b64_index(encoded_string[i + 2]) & 0x03) << 6) + _b64_index(encoded_string[i + 3])
                oi += 3
            else:
                oi += 2
        else:
            oi += 1
        i += 4
    # Upstream returns the whole pre-sized buffer, so short decodes carry trailing
    # NULs. Invisible when printed to a terminal; scored as insertions. Truncate.
    return out[:oi].decode("utf-8", errors="replace")


# ---------------- models ----------------
class RknnModel:
    def __init__(self, path, core_mask=None):
        from rknnlite.api import RKNNLite
        self.m = RKNNLite()
        assert self.m.load_rknn(path) == 0, f"load_rknn failed: {path}"
        kw = {}
        if core_mask is not None:
            kw["core_mask"] = core_mask
        assert self.m.init_runtime(**kw) == 0, f"init_runtime failed: {path}"

    def run(self, inputs):
        return self.m.inference(inputs=inputs)

    def release(self):
        self.m.release()


def run_encoder(enc, mel):
    t = time.perf_counter()
    out = enc.run([mel[None, :, :]])[0]
    return out, (time.perf_counter() - t) * 1000


def run_decoder_rknn(dec, enc_out, vocab, lang_token, max_new=448):
    """Verbatim port of the demo's fixed-12-slot sliding-window decode."""
    max_tokens = 12
    tokens = [SOT, lang_token, TASK_TRANSCRIBE, NO_TIMESTAMPS] * (max_tokens // 4)
    pop_id = max_tokens
    out_txt, token_times = "", []
    t_all = time.perf_counter()
    for _ in range(max_new):
        t0 = time.perf_counter()
        logits = dec.run([np.asarray([tokens], dtype="int64"), enc_out])[0]
        nxt = int(logits[0, -1].argmax())
        token_times.append((time.perf_counter() - t0) * 1000)
        tokens.append(nxt)
        if nxt == EOT:
            tokens.pop(-1)
            break
        if nxt > TIMESTAMP_BEGIN:
            continue
        if pop_id > 4:
            pop_id -= 1
        tokens.pop(pop_id)
        out_txt += vocab.get(str(nxt), "")
    dec_ms = (time.perf_counter() - t_all) * 1000
    return out_txt, dec_ms, token_times


class OnnxDecoder:
    """whisper-base int8 decoder with a real KV cache, on CPU.

    The RKNN decoder is a fixed 12-slot sliding window with no KV cache: every
    step re-runs the whole graph and the model only ever sees its last ~8 tokens.
    This keeps full history and reuses the cache, the same split that won on
    Hailo-8. encoder_sequence_length is a dynamic axis, so the 20 s RKNN encoder
    output feeds straight in with no re-export.
    """

    def __init__(self, d, threads=0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.init = ort.InferenceSession(str(Path(d) / "decoder_model.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.past = ort.InferenceSession(str(Path(d) / "decoder_with_past_model.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.past_in = {i.name for i in self.past.get_inputs()}

    # Whisper's decoder position table is 448 entries. Exceeding it is not a
    # tuning choice — onnxruntime raises "idx=448 out of data bounds" and the run
    # dies. Cap below it and truncate instead of crashing.
    MAX_POSITIONS = 448

    def decode(self, enc_out, vocab, lang_token, max_new=None):
        enc = np.ascontiguousarray(enc_out.astype(np.float32))
        if enc.ndim == 2:
            enc = enc[None]
        forced = [SOT, lang_token, TASK_TRANSCRIBE, NO_TIMESTAMPS]
        cap = self.MAX_POSITIONS - len(forced) - 1
        max_new = cap if max_new is None else min(max_new, cap)
        out_txt, token_times, kv = "", [], None
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        outs = self.init.run(None, {"input_ids": np.asarray([forced], dtype=np.int64),
                                    "encoder_hidden_states": enc})
        token_times.append((time.perf_counter() - t0) * 1000)
        names = [o.name for o in self.init.get_outputs()]
        logits = outs[0]
        kv = {n.replace("present", "past_key_values"): v
              for n, v in zip(names[1:], outs[1:])}
        kv = {k: v for k, v in kv.items() if k in self.past_in}
        nxt = int(logits[0, -1].argmax())
        for _ in range(max_new):
            if nxt == EOT:
                break
            if nxt <= TIMESTAMP_BEGIN:
                out_txt += vocab.get(str(nxt), "")
            t0 = time.perf_counter()
            feed = {"input_ids": np.asarray([[nxt]], dtype=np.int64)}
            feed.update(kv)
            if "encoder_hidden_states" in self.past_in:
                feed["encoder_hidden_states"] = enc
            outs = self.past.run(None, feed)
            token_times.append((time.perf_counter() - t0) * 1000)
            names = [o.name for o in self.past.get_outputs()]
            logits = outs[0]
            for n, v in zip(names[1:], outs[1:]):
                k = n.replace("present", "past_key_values")
                if k in self.past_in:
                    kv[k] = v
            nxt = int(logits[0, -1].argmax())
        return out_txt, (time.perf_counter() - t_all) * 1000, token_times


def finalize(raw, lang):
    txt = raw.replace("Ġ", " ").replace("<|endoftext|>", "").replace("\n", "")
    return base64_decode(txt) if lang == "zh" else txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True,
                    help=".rknn file, or a directory holding the optimum ONNX decoder pair")
    ap.add_argument("--vocab-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dec-threads", type=int, default=0,
                    help="onnxruntime intra-op threads for the CPU decoder (0 = default)")
    ap.add_argument("--encoder_duration", type=int, default=20,
                    help="the RKNN encoder's COMPILED fixed window, in seconds. This is not "
                         "a free knob: it must match the .rknn that was converted, and any "
                         "value below the longest clip forces the chunking path on.")
    ap.add_argument("--all-cores", action="store_true",
                    help="RK3588 has 3 NPU cores; bind one model across all of them")
    args = ap.parse_args()

    global CHUNK_LENGTH, MAX_LENGTH
    CHUNK_LENGTH = args.encoder_duration
    MAX_LENGTH = CHUNK_LENGTH * 100

    core = None
    if args.all_cores:
        from rknnlite.api import RKNNLite
        core = RKNNLite.NPU_CORE_0_1_2

    vd = Path(args.vocab_dir)
    filters = np.loadtxt(vd / "mel_80_filters.txt", dtype=np.float32).reshape((80, 201))
    vocab = read_vocab(vd / (f"vocab_{args.lang}.txt"))
    lang_token = LANG[args.lang]

    manifest = json.loads((Path(args.corpus) / "manifest.json").read_text(encoding="utf-8"))
    files = [f for f in manifest["files"] if f["lang"] == args.lang]

    enc = RknnModel(args.encoder, core)
    use_onnx_dec = not args.decoder.endswith(".rknn")
    dec = OnnxDecoder(args.decoder, args.dec_threads) if use_onnx_dec else RknnModel(args.decoder, core)
    decode = ((lambda e, mx=None: dec.decode(e, vocab, lang_token, max_new=mx)) if use_onnx_dec
              else (lambda e, mx=None: run_decoder_rknn(dec, e, vocab, lang_token,
                                                        max_new=mx or 448)))

    import wave
    def load_wav(p):
        with wave.open(str(p), "rb") as w:
            assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    # A window narrower than the corpus's longest clip turns segmentation from
    # optional into required: truncated audio makes the decoder skip EOS and run
    # to the position-table limit. 20 s covered everything; 10 s does not.
    def split_for_window(audio):
        limit = int(CHUNK_LENGTH * SAMPLE_RATE)
        if len(audio) <= limit:
            return [audio]
        hop = int((CHUNK_LENGTH - 1.0) * SAMPLE_RATE)   # 1 s overlap
        return [audio[i:i + limit] for i in range(0, len(audio), hop)
                if len(audio[i:i + limit]) > SAMPLE_RATE // 2]

    def clean_repetition(text, lang, thr=0.55):
        import re, difflib
        # 1. drop whole-segment non-speech annotations: (music), [silence], ...
        text = re.sub(r"[\(\[][^)\]]{0,40}[\)\]]", " ", text)
        # 2. sentence-level self-paraphrase truncation
        sents = [x for x in re.split(r"(?<=[.?!。？！])\s*", text) if x.strip()]
        join = "" if lang == "zh" else " "
        uniq = []
        for s in sents:
            n = s.lower().strip()
            for u in uniq:
                nu = u.lower().strip()
                if n and nu and (n in nu or nu in n
                                 or difflib.SequenceMatcher(None, nu, n).ratio() >= thr):
                    sents = uniq; break
            else:
                uniq.append(s.strip()); continue
            break
        out = join.join(uniq)
        # 3. loop guard: a phrase repeating >=3x inside one sentence is a decoder
        #    loop and is invisible to sentence-level dedup ("by Llew, by Llew, ...").
        units = list(out) if lang == "zh" else out.split()
        best_cut = None
        for p in range(1, 13):
            if len(units) < 3 * p:
                break
            for i in range(len(units), 3 * p - 1, -1):
                if units[i-3*p:i-2*p] == units[i-2*p:i-p] == units[i-p:i]:
                    cut = i - 2 * p
                    while cut - p >= 0 and units[cut-p:cut] == units[cut:cut+p]:
                        cut -= p
                    best_cut = cut if best_cut is None else min(best_cut, cut)
                    break
        if best_cut is not None:
            units = units[:best_cut]
        joined = ("" if lang == "zh" else " ").join(units)
        return " ".join(joined.split()) if lang != "zh" else joined.strip()

    def merge(parts, lang):
        out = ""
        for p in (x.strip() for x in parts):
            if not p:
                continue
            if not out:
                out = p; continue
            best = 0
            for k in range(min(len(out), len(p), 40), 2, -1):
                if out[-k:] == p[:k]:
                    best = k; break
            out = out + p[best:] if lang == "zh" else (out + " " + p[best:])
        return out

    first = load_wav(Path(args.corpus) / files[0]["filename"])
    for _ in range(args.warmup):
        m = pad_or_trim(log_mel_spectrogram(first, filters, MAX_LENGTH))
        e, _ = run_encoder(enc, m)
        decode(e, 8)

    rows = []
    for i, f in enumerate(files, 1):
        audio = load_wav(Path(args.corpus) / f["filename"])
        dur = len(audio) / SAMPLE_RATE
        chunks = split_for_window(audio)
        parts, enc_tot, dec_tot, ttft = [], 0.0, 0.0, None
        t_pre_tot = 0.0
        for ci, ch in enumerate(chunks):
            t0 = time.perf_counter()
            mel = pad_or_trim(log_mel_spectrogram(ch, filters, MAX_LENGTH))
            t_pre_tot += (time.perf_counter() - t0) * 1000
            enc_out, enc_ms = run_encoder(enc, mel)
            # Bound tokens by how much audio this chunk actually holds. A fixed
            # cap is a crash guard, not a runaway guard: a near-silent tail chunk
            # will happily generate to the limit.
            budget = int(max(16, min(220, len(ch) / SAMPLE_RATE * 8 + 12)))
            raw, dec_ms, tt = decode(enc_out, budget)
            parts.append(finalize(raw, args.lang))
            enc_tot += enc_ms; dec_tot += dec_ms
            if ci == 0:
                ttft = enc_ms + (tt[0] if tt else 0.0)
        hyp = clean_repetition(merge(parts, args.lang), args.lang)
        row = dict(id=f["id"], lang=args.lang, category=f["category"], duration_s=dur,
                   n_chunks=len(chunks), seg_mode=f"window-{CHUNK_LENGTH}s"
                   + ("" if len(chunks) == 1 else "+1s-overlap"),
                   encoder_ms=round(enc_tot, 1), decoder_ms=round(dec_tot, 1),
                   preprocess_ms=round(t_pre_tot, 1),
                   infer_ms=round(enc_tot + dec_tot, 1),
                   ttft_ms=round(ttft, 1) if ttft else None,
                   n_tokens=0,
                   rtf=round((enc_tot + dec_tot) / 1000 / dur, 4),
                   ref=f.get("eval_transcript") or f["transcript"], hyp=hyp)
        rows.append(row)
        print(f"[{i}/{len(files)}] {f['id']} {dur:.2f}s chunks={len(chunks)} "
              f"enc={enc_tot:.1f} dec={dec_tot:.1f} rtf={row['rtf']:.4f} "
              f"ttft={row['ttft_ms']}\n     {hyp}", flush=True)

    Path(args.out).write_text(json.dumps(
        dict(label=args.label, window_s=CHUNK_LENGTH, variant="base",
             decoder=("onnx-int8-kvcache-cpu" if use_onnx_dec else "rknn-12slot-sliding"),
             rows=rows), ensure_ascii=False, indent=2),
        encoding="utf-8")
    enc.release()
    if not use_onnx_dec:
        dec.release()
    print("RKNN RUN COMPLETE")


if __name__ == "__main__":
    main()
