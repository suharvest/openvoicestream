#!/usr/bin/env python3
"""Whisper on Jetson via bare TensorRT — corpus runner.

Three engines, mirroring the split that wyoming-whisper-trt uses (the only
open-source Whisper-TRT project with a real KV cache), except that optimum's
export already gives the split for free:

  encoder.plan   mel -> encoder hidden states
  prefill.plan   decoder_model.onnx      : forced prompt -> logits + self KV
                                           + cross KV (computed ONCE per utterance)
  step.plan      decoder_with_past.onnx  : one token -> logits + grown self KV

Cross-attention K/V are bound straight from the prefill engine's device buffers
into the step engine; self-attention K/V ping-pong between two device buffers per
(layer, kind). Nothing goes back to the host inside the decode loop.

No onnxruntime, no torch, no torch2trt: tensorrt + cuda-python only, both of
which ship with JetPack.

BUILD THE ENCODER ENGINE AS FP32 FOR whisper-base.
`trtexec --fp16` on the base encoder ONNX yields an engine whose output has
cosine 0.8104 against onnxruntime (fp32 rebuild: 0.999999). It is deterministic
run-to-run, so this is fp16 kernel selection, not a race. The failure is
deceptive: the bad encoder still emits a plausible-looking tensor, so the
decoder greedily produces fluent English that drifts off-topic and stops early —
indistinguishable from a KV-cache bug by inspection. tiny/30s and tiny/10s fp16
engines are fine (cosine 0.9999), as are both decoder engines, so this is
model-specific. Numerically diff every engine against onnxruntime before
trusting it.
"""
import argparse, json, time, wave
from pathlib import Path
import numpy as np
import tensorrt as trt
from cuda import cudart

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
EOT, SOT, TASK_TRANSCRIBE, NO_TIMESTAMPS = 50257, 50258, 50359, 50363
TIMESTAMP_BEGIN = 50364
LANG = {"en": 50259, "zh": 50260}


def _chk(err, msg=""):
    if isinstance(err, tuple):
        err, *rest = err
    else:
        rest = []
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error {err} {msg}")
    return rest[0] if len(rest) == 1 else (rest or None)


# ---------------- mel front end (numpy; matches the RK/torch reference) -------
def _hann(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)


def log_mel(audio, filters, max_frames):
    # Pad the *waveform* to the full window before the STFT, the way whisper
    # does. Zero-padding the finished mel instead leaves 0.0 in the tail, while
    # the mel of digital silence is (log10(1e-10) clipped to max-8 + 4)/4 ~= -0.6
    # -- i.e. the encoder was being shown 27 s of a constant that never occurs
    # in training. Same reason ls.max() must be taken over the padded window.
    n_samples = max_frames * HOP_LENGTH
    a = np.zeros(n_samples, dtype=np.float64)
    n = min(len(audio), n_samples)
    a[:n] = audio[:n]
    pad = N_FFT // 2
    x = np.pad(a, (pad, pad), mode="reflect")
    nf = 1 + (len(x) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(nf)[:, None]
    spec = np.fft.rfft(x[idx] * _hann(N_FFT)[None, :], n=N_FFT, axis=-1)
    mag = (np.abs(spec) ** 2).T[:, :-1]
    mel = filters @ mag
    ls = np.log10(np.clip(mel, 1e-10, None))
    ls = np.maximum(ls, ls.max() - 8.0)
    ls = ((ls + 4.0) / 4.0).astype(np.float32)
    return np.ascontiguousarray(ls[:, :max_frames])


# ---------------- TensorRT plumbing ----------------
class Engine:
    """One .plan + its execution context, with device buffers owned per tensor."""

    def __init__(self, path, stream):
        with open(path, "rb") as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"failed to deserialize {path}"
        self.ctx = self.engine.create_execution_context()
        self.stream = stream
        self.names = [self.engine.get_tensor_name(i)
                      for i in range(self.engine.num_io_tensors)]
        self.is_input = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
                         for n in self.names}
        self.dtype = {n: trt.nptype(self.engine.get_tensor_dtype(n)) for n in self.names}
        self.dev = {}
        self.nbytes = {}
        # cudaMemcpyAsync reads the host buffer after h2d() has returned, so the
        # staging array (which ascontiguousarray may have just allocated for a
        # dtype cast) has to outlive the call.
        self.staged = {}

    def shape_of(self, n):
        return tuple(self.ctx.get_tensor_shape(n))

    def alloc(self, name, shape):
        nb = int(np.prod(shape)) * np.dtype(self.dtype[name]).itemsize
        if self.dev.get(name) is not None and self.nbytes.get(name, 0) >= nb:
            return self.dev[name]
        if self.dev.get(name) is not None:
            _chk(cudart.cudaFree(self.dev[name]))
        ptr = _chk(cudart.cudaMalloc(nb), f"malloc {name}")
        self.dev[name] = ptr
        self.nbytes[name] = nb
        return ptr

    def bind(self, name, ptr):
        self.ctx.set_tensor_address(name, int(ptr))

    def h2d(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=self.dtype[name])
        self.staged[name] = arr
        ptr = self.alloc(name, arr.shape)
        _chk(cudart.cudaMemcpyAsync(ptr, arr.ctypes.data, arr.nbytes,
                                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                    self.stream))
        self.bind(name, ptr)
        return ptr

    def d2h(self, name):
        shape = self.shape_of(name)
        out = np.empty(shape, dtype=self.dtype[name])
        _chk(cudart.cudaMemcpyAsync(out.ctypes.data, self.dev[name], out.nbytes,
                                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                    self.stream))
        return out

    def d2h_ptr(self, ptr, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        _chk(cudart.cudaMemcpyAsync(out.ctypes.data, ptr, out.nbytes,
                                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                    self.stream))
        return out

    def profile_max(self, name):
        return tuple(self.engine.get_tensor_profile_shape(name, 0)[2])

    def run(self):
        ok = self.ctx.execute_async_v3(stream_handle=self.stream)
        if not ok:
            raise RuntimeError("execute_async_v3 failed")

    def sync(self):
        _chk(cudart.cudaStreamSynchronize(self.stream))


class TrtWhisper:
    def __init__(self, enc_plan, prefill_plan, step_plan, n_layers=6):
        self.stream = _chk(cudart.cudaStreamCreate())
        self.enc = Engine(enc_plan, self.stream)
        self.pre = Engine(prefill_plan, self.stream)
        self.stp = Engine(step_plan, self.stream)
        self.L = n_layers

    # --- encoder -------------------------------------------------------
    def encode(self, mel):
        name = next(n for n in self.enc.names if self.enc.is_input[n])
        inp = mel[None, :, :]
        if len(self.enc.engine.get_tensor_shape(name)) == 4:
            inp = inp[:, :, None, :]
        self.enc.ctx.set_input_shape(name, inp.shape)
        self.enc.h2d(name, inp)
        out_name = next(n for n in self.enc.names if not self.enc.is_input[n])
        self.enc.alloc(out_name, self.enc.shape_of(out_name))
        self.enc.bind(out_name, self.enc.dev[out_name])
        t = time.perf_counter()
        self.enc.run(); self.enc.sync()
        return out_name, (time.perf_counter() - t) * 1000

    # --- decoder -------------------------------------------------------
    def _kv_bufs(self, slot_nbytes):
        """Two device buffers per (layer, kind), sized for the engine's longest
        legal past. Reallocated only if a larger engine is ever loaded."""
        if getattr(self, "_kv", None) is not None and self._kv_nbytes >= slot_nbytes:
            return self._kv
        for pair in (getattr(self, "_kv", None) or {}).values():
            for ptr in pair:
                _chk(cudart.cudaFree(ptr))
        self._kv = {(l, k): [_chk(cudart.cudaMalloc(slot_nbytes), "kv"),
                             _chk(cudart.cudaMalloc(slot_nbytes), "kv")]
                    for l in range(self.L) for k in ("key", "value")}
        self._kv_nbytes = slot_nbytes
        return self._kv

    def decode(self, enc_out_ptr, enc_shape, vocab, lang_token, max_new=200):
        L = self.L
        forced = np.array([[SOT, lang_token, TASK_TRANSCRIBE, NO_TIMESTAMPS]], dtype=np.int64)
        token_times = []

        t0 = time.perf_counter()
        self.pre.ctx.set_input_shape("input_ids", forced.shape)
        self.pre.ctx.set_input_shape("encoder_hidden_states", enc_shape)
        self.pre.h2d("input_ids", forced)
        self.pre.bind("encoder_hidden_states", enc_out_ptr)
        for n in self.pre.names:
            if not self.pre.is_input[n]:
                self.pre.alloc(n, self.pre.shape_of(n))
                self.pre.bind(n, self.pre.dev[n])
        self.pre.run(); self.pre.sync()
        token_times.append((time.perf_counter() - t0) * 1000)

        logits = self.pre.d2h("logits"); self.pre.sync()
        nxt = int(logits[0, -1].argmax())

        # Cross-attention KV is constant for the whole utterance: bind the
        # prefill engine's own device buffers into the step engine. Safe because
        # stp.alloc() is never called for those names, so stp never frees a
        # pointer it does not own, and pre only reallocates on a shape growth
        # that cannot happen while the encoder window is fixed.
        for l in range(L):
            for kind in ("key", "value"):
                src = f"present.{l}.encoder.{kind}"
                dst = f"past_key_values.{l}.encoder.{kind}"
                self.stp.ctx.set_input_shape(dst, self.pre.shape_of(src))
                self.stp.bind(dst, self.pre.dev[src])

        # Self-attention KV: two device buffers per tensor, ping-ponged. The step
        # engine writes present = past ++ new, so this step's `present` IS next
        # step's `past` -- swapping the two pointers is the whole cache update
        # and no KV ever crosses the host boundary.
        kv_dtype = self.stp.dtype["past_key_values.0.decoder.key"]
        item = np.dtype(kv_dtype).itemsize
        pmax = self.stp.profile_max("past_key_values.0.decoder.key")[2]
        b, H, past_len, D = self.pre.shape_of("present.0.decoder.key")
        bufs = self._kv_bufs(b * H * (pmax + 1) * D * item)
        for l in range(L):
            for kind in ("key", "value"):
                src = f"present.{l}.decoder.{kind}"
                nb = int(np.prod(self.pre.shape_of(src))) * item
                _chk(cudart.cudaMemcpyAsync(
                    bufs[(l, kind)][0], self.pre.dev[src], nb,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, self.stream))
        self.pre.sync()

        out_txt = ""
        for _ in range(max_new):
            if nxt == EOT or past_len > pmax:
                break
            if nxt <= TIMESTAMP_BEGIN:
                out_txt += vocab.get(str(nxt), "")
            t = time.perf_counter()
            ids = np.array([[nxt]], dtype=np.int64)
            self.stp.ctx.set_input_shape("input_ids", ids.shape)
            for l in range(L):
                for kind in ("key", "value"):
                    past, pres = bufs[(l, kind)]
                    self.stp.ctx.set_input_shape(
                        f"past_key_values.{l}.decoder.{kind}", (b, H, past_len, D))
                    self.stp.bind(f"past_key_values.{l}.decoder.{kind}", past)
                    self.stp.bind(f"present.{l}.decoder.{kind}", pres)
            self.stp.h2d("input_ids", ids)
            self.stp.alloc("logits", self.stp.shape_of("logits"))
            self.stp.bind("logits", self.stp.dev["logits"])
            self.stp.run(); self.stp.sync()
            logits = self.stp.d2h("logits"); self.stp.sync()
            for pair in bufs.values():
                pair.reverse()
            past_len += 1
            nxt = int(logits[0, -1].argmax())
            token_times.append((time.perf_counter() - t) * 1000)
        return out_txt, sum(token_times), token_times


def read_vocab(path):
    vocab = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ")
            vocab[parts[0]] = parts[1] if len(parts) >= 2 else ""
    return vocab


def _b64i(c):
    if "A" <= c <= "Z": return ord(c) - 65
    if "a" <= c <= "z": return ord(c) - 97 + 26
    if "0" <= c <= "9": return ord(c) - 48 + 52
    return 62 if c == "+" else 63


def base64_decode(s):
    if not s: return ""
    out = bytearray(len(s) // 4 * 3); i = oi = 0
    while i < len(s):
        if s[i] == "=": return " "
        out[oi] = (_b64i(s[i]) << 2) + ((_b64i(s[i+1]) & 0x30) >> 4)
        if i + 2 < len(s) and s[i+2] != "=":
            out[oi+1] = ((_b64i(s[i+1]) & 0x0F) << 4) + ((_b64i(s[i+2]) & 0x3C) >> 2)
            if i + 3 < len(s) and s[i+3] != "=":
                out[oi+2] = ((_b64i(s[i+2]) & 0x03) << 6) + _b64i(s[i+3]); oi += 3
            else: oi += 2
        else: oi += 1
        i += 4
    return out[:oi].decode("utf-8", errors="replace")


def finalize(raw, lang):
    t = raw.replace("Ġ", " ").replace("<|endoftext|>", "").replace("\n", "")
    return base64_decode(t) if lang == "zh" else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--encoder-plan", required=True)
    ap.add_argument("--prefill-plan", required=True)
    ap.add_argument("--step-plan", required=True)
    ap.add_argument("--vocab-dir", required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    max_frames = args.window * 100
    vd = Path(args.vocab_dir)
    filters = np.loadtxt(vd / "mel_80_filters.txt", dtype=np.float32).reshape((80, 201))
    vocab = read_vocab(vd / f"vocab_{args.lang}.txt")
    man = json.loads((Path(args.corpus) / "manifest.json").read_text(encoding="utf-8"))
    files = [f for f in man["files"] if f["lang"] == args.lang]

    w = TrtWhisper(args.encoder_plan, args.prefill_plan, args.step_plan, args.layers)

    def load(p):
        with wave.open(str(p), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    def run_one(audio):
        mel = log_mel(audio, filters, max_frames)
        out_name, enc_ms = w.encode(mel)
        shape = w.enc.shape_of(out_name)
        txt, dec_ms, tt = w.decode(w.enc.dev[out_name], shape, vocab, LANG[args.lang])
        return txt, enc_ms, dec_ms, tt

    first = load(Path(args.corpus) / files[0]["filename"])
    for _ in range(args.warmup):
        run_one(first)

    rows = []
    for i, f in enumerate(files, 1):
        audio = load(Path(args.corpus) / f["filename"])
        dur = len(audio) / SAMPLE_RATE
        txt, enc_ms, dec_ms, tt = run_one(audio)
        hyp = finalize(txt, args.lang)
        row = dict(id=f["id"], lang=args.lang, category=f["category"], duration_s=dur,
                   n_chunks=1, seg_mode=f"trt-single-window-{args.window}s",
                   encoder_ms=round(enc_ms, 1), decoder_ms=round(dec_ms, 1),
                   infer_ms=round(enc_ms + dec_ms, 1),
                   ttft_ms=round(enc_ms + (tt[0] if tt else 0), 1),
                   n_tokens=len(tt),
                   rtf=round((enc_ms + dec_ms) / 1000 / dur, 4),
                   ref=f.get("eval_transcript") or f["transcript"], hyp=hyp)
        rows.append(row)
        print(f"[{i}/{len(files)}] {f['id']} {dur:.2f}s enc={enc_ms:.2f} dec={dec_ms:.1f} "
              f"rtf={row['rtf']:.4f} ttft={row['ttft_ms']:.1f} tok={len(tt)}\n     {hyp}",
              flush=True)

    Path(args.out).write_text(json.dumps(
        dict(label=args.label, window_s=args.window, backend="tensorrt-fp16",
             decoder="trt-prefill+cached-step", rows=rows),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRT RUN COMPLETE")


if __name__ == "__main__":
    main()
