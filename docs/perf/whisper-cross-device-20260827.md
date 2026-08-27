# Whisper across edge platforms — measured (2026-08-27)

Whisper measured on Hailo-8, RK3588, RK3576, Jetson Orin Nano and Raspberry Pi 5 CPU, to answer two product questions: which board and which configuration for **conversation** (short utterances, latency-bound) and for **transcription** (long audio, throughput and accuracy).

## Summary

**Both NPU vendors ship a default configuration that is among the worst options on their own hardware.** Hailo and Rockchip both split Whisper into an encoder and a decoder graph and put both on the NPU — and **neither NPU decoder has a KV cache**. Whisper's decoder is autoregressive, so without a cache every step recomputes the whole sequence. Offloading the encoder is a real win; offloading the decoder is a net loss.

Moving the decoder to a CPU ONNX graph with a real KV cache makes every board **both faster and more accurate**: RK3588 English long-form goes from 10.44% to **7.58%** WER while RTF drops from 0.149 to **0.061**.

**Window length is the conversation/transcription trade-off lever, and it is configurable.** Same RK3588, same model, same code, only the encoder window changed from 20 s to 10 s: TTFT 301 ms → **124 ms** (2.4×) and short-utterance WER 13.37% → **11.37%**; the cost is that long audio must now be segmented, taking long-form WER from 7.58% to 11.40%.

**No board fixes Chinese.** The best Chinese result here is 15.99% CER; the Paraformer path already deployed on RK3588 scores **2.6%** on the same audio. That is Whisper's own ceiling, not a hardware gap. The one configuration change that helps materially is Hailo tiny/10s → base/5s (long-form 50.22% → **34.08%**), and even that is still an order of magnitude behind.

**A TensorRT engine's numerical correctness cannot be inferred from its precision flag.** The whisper-base encoder built with `--fp16` scores cosine **0.826** against onnxruntime, while the tiny encoders and both decoder engines built the same way are fine. A bad engine raises no error — it just makes everything downstream produce fluent, wrong text. Diff every engine against onnxruntime.

---

## Method

**Corpus**: this repository's own `bench/perf/corpus` — 20 human recordings from Google FLEURS (CC BY 4.0), 5 each of zh/en × short/long, SHA256-pinned so every device consumes byte-identical audio.

**Scoring** re-implements `bench/perf/runners.py` line for line: cn2an numeral normalisation, the same punctuation table, jiwer CER for Chinese and WER for English. **Devices emit only transcripts and per-stage timings; scoring runs once, on one machine.** Installing jiwer/cn2an/opencc separately on each device invites metric drift, and metric drift is fatal to a cross-device comparison.

**Metrics**:

- **RTF** = inference time ÷ audio duration, measured offline
- **TTFT** = encoder time + first decoded token
- **t2s** = CER after Traditional→Simplified normalisation. Whisper emits Traditional; every other ASR backend in the fleet emits Simplified. Reported separately, never folded into the main column.

> **Two metric definitions, do not mix them.** RTF here is the offline "inference ÷ duration". The **Finalize RTF** in `docs/performance-comparison.md` is time *after* the speaker stops, measured through the streaming server. They are not the same quantity, and TTFT has no counterpart in that matrix.

---

## Results

Five files per group, mean after warm-up.

| Board | Configuration | en short | en long | zh short (t2s) | zh long (t2s) | TTFT | RTF (long) |
|---|---|---|---|---|---|---|---|
| **Orin Nano** | base / 30s / **bare TensorRT (bf16)** | 11.37% | 9.19% | 57.03 (45.97) | 31.47 (16.71) | **18 ms** | **0.008** |
| Orin Nano | base / 30s / whisper.cpp CUDA | 13.59% | 8.59% | 58.62 (46.66) | 30.75 (**15.99**) | 216 ms\* | 0.023 |
| Orin Nano | tiny / 30s / whisper.cpp CUDA | **7.30%** | 12.26% | 48.75 (39.49) | 37.21 (27.15) | 196 ms\* | 0.019 |
| **Hailo-8 + Pi 5** | tiny / 10s / hybrid | **10.95%** | 21.58% | 52.98 (50.31) | 58.13 (50.22) | 60 ms\*\*\* | 0.029 |
| **Hailo-8 + Pi 5** | **base / 5s** / hybrid | 13.81% | **19.03%** | 57.28 (**42.59**) | 48.90 (**34.08**) | 81 ms | 0.057 |
| **RK3588** | base / 20s / **hybrid** | 13.37% | **7.58%** | 52.00 (44.94) | 32.32 (**19.63**) | 301 ms | 0.061 |
| RK3588 | base / 20s / all-NPU *(vendor default)* | 15.37% | 10.44% | 51.09 (44.04) | 29.83 (19.78) | 318 ms | 0.149 |
| **RK3588** | base / **10s** / hybrid | **11.37%** | 11.40% | 55.32 (40.63) | 48.77 (37.72) | **124 ms** | 0.072 |
| **RK3576** | base / 20s / **hybrid** | 17.81%\*\* | **7.58%** | 44.94 | 32.32 (20.32) | 366 ms | 0.104 |
| RK3576 | base / 20s / all-NPU *(vendor default)* | 15.37% | 10.44% | 41.54 | 29.83 (19.78) | 294 ms | 0.146 |
| **RK3576** | base / **10s** / hybrid | **11.37%** | 10.40% | 52.59 (41.54) | 58.24 (42.97) | 149 ms | 0.108 |
| Pi 5 CPU only | tiny / 10s / ONNX | 10.16% | 23.73% | 50.30 (47.23) | 57.74 (36.94) | 156 ms | 0.043 |

\*\*\* The Hailo rows were re-measured after the mel front end was unified to the numpy port and with `warmup=5`. **The previously published TTFT of 38.7 ms was an underestimate**: first-token was recorded as 15.0 ms then, but with more warm-up it settles at 28–48 ms — this is not a warm-up artefact, the 15 ms figure itself was the outlier. Mechanically, prefill has to consume the full encoder output and produce every cross-attention K/V, so ~30 ms is the plausible number while the encoder alone takes 24 ms.

\*\* RK3576's 17.81% on en short against RK3588's 13.37% comes **entirely from 1 of the 5 files**: on `en_short_03` RK3576 emits `Erasmith had` where RK3588 emits `Aerosmith have`, two words that move that file from 0.111 to 0.333 and the group mean by 4.4 points. Long-form is **7.58% on both, bit-identical**. This is fp16 numerics differing slightly between the two chips and greedy decoding amplifying a near-tie into a different token — **not a capability difference**.

\* Jetson whisper.cpp TTFT is a **proxy**: whisper.cpp exposes no first-token timestamp, so this is encode plus one sample step. Every other TTFT in the table is measured per token. The two are not directly comparable.

### Pick by scenario

| Scenario | Choice | Basis |
|---|---|---|
| **Conversation / latency** | Latency first → Orin Nano + TensorRT (bf16); cost first → Hailo-8 + Pi 5 (tiny/10s for English, base/5s for Chinese) | TTFT **18 ms** vs **60/81 ms** — Jetson is 3–4.5× faster. English short-utterance accuracy is comparable across all three (11.37 / 10.95 / 13.81). Hailo's value is costing an order of magnitude less while still being fast enough. |
| **Transcription / throughput + Chinese** | Orin Nano + bare TensorRT (bf16) | RTF **0.008** — an hour of audio in ~29 s; Chinese long-form 16.71% (whisper.cpp 15.99%, same band) |
| **English long-form / cost** | RK3588 hybrid | **7.58%**, level with the Jetson's 9.19% on far cheaper silicon |

---

## Cross-platform findings

### 1. Neither NPU can decode

| Platform | NPU decoder shape | Consequence |
|---|---|---|
| Hailo-8 | Fixed 32-token sequence, full attention inside it | Everything past 28 generated tokens is **truncated** — Chinese sentences stop mid-clause |
| RK3588 / RK3576 | **12-slot sliding window**, unbounded length | The decoder only ever sees its last ~8 tokens; long sentences lose the thread |
| Jetson (GGML / TRT) | Real KV cache | Not affected |

Measured decoder time per utterance (English long-form mean):

```
Hailo-8   HEF decoder      680 ms   (42 ms/token, no KV cache)
Hailo-8   CPU KV-cache     159 ms   (first token ~30 ms, then ~8 ms/token)
RK3588    RKNN decoder    1391 ms
RK3588    CPU KV-cache     426 ms
Orin Nano CUDA            134 ms
```

**Those absolute numbers do not compare across platforms** — the models differ. The Pi 5 / Hailo rows run whisper-**tiny** (4 layers, d384); the RK rows run **base** (6 layers, d512), a **2.67×** larger decoder. Normalised per token and then per model size (English short):

| CPU / accelerator | Model | ms/token | tiny-equivalent |
|---|---|---|---|
| Pi 5 CPU (A76 ×4) | tiny | 13.02 | 13.02 |
| Pi 5 CPU (A76 ×4) | base | 13.77 | **5.16** |
| **RK3588 CPU (A76 ×4 + A55 ×4)** | base | 15.62 | **5.86** |
| **RK3576 CPU (A72)** | base | 38.16 | **14.31** |
| Orin Nano GPU (TRT bf16) | base | 3.14 | 1.18 |

Three things fall out:

- **RK3588's CPU is about 2.2× faster than the Pi 5's** (5.86 vs 13.02). RK3588's larger absolute decoder time is only because it runs base.
- **RK3576 is 2.4× slower than RK3588** (14.31 vs 5.86). The A72/A76 generation gap shows up clearly in autoregressive decoding, which is small serial matrix work. This is also the direct evidence for "the gap between the two RK boards is entirely CPU, not NPU" — their encoder times are nearly identical.
- **On the same Pi 5, base normalises better than tiny** (5.16 vs 13.02). Same runtime, same CPU — a larger model should not be more efficient per parameter. The plausible reading is that tiny's matrices (d384) are small enough that the CPU is bound by scheduling overhead rather than arithmetic, which is consistent with the decoder being bandwidth/overhead-bound.

Encoders are not slow on either NPU (Hailo 24 ms, RK 250 ms). The whole problem is autoregressive decoding.

### 1b. Hailo tiny/10s vs base/5s: choose by language, not by "which is better"

Hailo ships exactly two encoder HEFs: tiny with a 10 s window and base with a **5 s** window. Bigger model, half the window — the two effects pull in opposite directions, so it has to be measured:

| | tiny / 10s | base / 5s |
|---|---|---|
| en short | **10.95%** | 13.81% |
| en long | 21.58% | **19.03%** |
| zh short (t2s) | 50.31% | **42.59%** |
| zh long (t2s) | 50.22% | **34.08%** |
| TTFT | **60 ms** | 81 ms |
| RTF (long) | **0.029** | 0.057 |
| Chunks per long file | 1–2 | 2–3 |

**base/5s wins Chinese by a wide margin**: −7.7 points on short, **−16.1 on long**. The direction matches OpenAI's FLEURS table (base 34.1 vs tiny 40.5 CER), and notably **it wins even though the window is halved and long files now need three chunks** — on Chinese, model capacity buys far more than the shorter window costs. English goes the other way and by much less. The price is TTFT 60 → 81 ms and roughly double the RTF.

**So route by language on Hailo**: English short commands through tiny/10s (lowest latency), Chinese through base/5s. This is the only place in this report where a configuration change materially improves Chinese — and 34.08% is still far behind Paraformer's 2.6% on the same corpus, so it does not change the overall conclusion.

### 2. The window is a product setting, not a vendor property

Shipped windows differ by 6×: Hailo tiny **10s** (9s usable — the mel front end crops one second to avoid boundary hallucination), Hailo base **5s**, Rockchip **20s**, whisper.cpp **30s** (Whisper's native).

This looked like a fixed platform property. It is not. On one RK3588:

| | 10s window | 20s window |
|---|---|---|
| en short | **11.37%** | 13.37% |
| **TTFT** | **124 ms** | 301 ms |
| en long | 11.40% | **7.58%** |
| zh long (t2s) | 37.72% | **19.63%** |

The short-utterance half is free: **2.4× better TTFT and slightly better accuracy**. The entire cost lands on long audio — once the window is shorter than the longest clip, segmentation becomes mandatory.

**So the setting is really "window + whether segmentation is required", and exposing only the first half is misleading.** Hailo's 9 s effective window is not a defect; it pins the trade-off at the conversation end. Rockchip's 20 s pins it at the transcription end.

### 3. Audio without enough content makes Whisper skip EOS

This failure mode appeared three times on three different paths, with no relation to the vendor:

| Trigger | Symptom | Why it had not shown up |
|---|---|---|
| Hailo: a short utterance zero-padded to fill the fixed 10 s window | Repeats and confabulates; en short WER 10.95% → **70.16%** | The hard-coded `cap=32` had been absorbing it |
| RK 10s: a tail chunk with too little content after segmentation | `by Llew, by Llew, ...` until the position table runs out | At 20 s the whole corpus fit in one window |
| RK 10s: a tail chunk that is nearly all silence | Emits `(dramatic music)` / `[silence]` | Same |

**Whisper's decoder position table holds 448 entries.** Exceeding it is not a quality degradation, it is a crash (`idx=448 out of data bounds` from onnxruntime). A hard-coded token cap looks like a length limit but is simultaneously standing in for EOS — **the vendor demos' default configuration has been hiding this**.

The harness therefore makes three guards default rather than per-platform patches:

1. **Duration-proportional token budget**, `min(cap, duration×8+12)` — a fixed cap guards against the crash, not against the runaway
2. **Similarity-based repetition cleanup** — Hailo's own `clean_transcription` tests substring containment only, so it misses self-paraphrase (`...from the plant.` / `...more than a plan for...`), and it splits on `.` and `?` alone, so Chinese with no sentence punctuation passes straight through
3. **Intra-sentence loop guard** — n-gram detection of a phrase repeating ≥3 times; sentence-level dedup cannot see a loop inside one sentence

---

## The published ceiling, for reference

The corpus comes from FLEURS, which is also where the Whisper paper reports per-language accuracy, so the two are directly comparable (arXiv 2212.04356, Appendix D.2.4, Table 13). The paper's own normalisation note settles the units: *"we put a space between every letter for the languages that do not use spaces to separate words, namely Chinese, Japanese, Thai, Lao, and Burmese, effectively measuring the character error rate instead"* — so the Chinese column is CER, on the same convention used here.

| Model | English WER | Chinese CER | On Hailo-8 | On Rockchip |
|---|---|---|---|---|
| tiny | 12.4% | **40.5%** | yes, 10 s window | no |
| base | 8.9% | **34.1%** | yes, 5 s window | yes, 20 s window |
| small | 6.1% | 20.8% | no | no |
| large-v2 | 4.2% | 14.7% | no | no |

Our pure-CPU baseline (fp32 encoder, no NPU, no runaway) scored 10.16% English and 50.30% Chinese against the published 12.4% and 40.5% — five files per group against the full test split, so the agreement is directional. It is close enough to settle the point: **40–50% Chinese CER is whisper-tiny's designed accuracy, not a quantisation or hardware artefact**.

`small`, at 20.8%, is the first size where Chinese becomes arguable, and neither Hailo-8 nor Rockchip ships one. Even large-v2's 14.7% would still trail the Paraformer path already deployed on RK3588 (2.6%).

**Route by language, not by board.** English through Whisper wherever the latency budget points; Chinese through the Chinese-native models each platform already runs.

---

## Jetson: bare TensorRT vs TensorRT-Edge-LLM

**Use bare TensorRT. Do not put Whisper into edge-llm, and do not use ONNX Runtime's TensorRT EP.**

| | edge-llm | ORT + TRT EP | bare TensorRT |
|---|---|---|---|
| Architectural fit | ❌ **Zero** occurrences of cross-attention in the tree; its ASR path is prefix injection into a decoder-only LLM | ✅ | ✅ cross-attention is ordinary ops in TRT |
| Dependencies | already present | ❌ heavy; partitions by subgraph and falls back to the CUDA EP | ✅ `python3-libnvinfer` + `trtexec` + `cuda-python` **already on the device, nothing new** |
| Work required | implement cross-attention in NVIDIA's upstream runtime and `llm_build`, on top of 7 upstream + 35 local patches | low | write the KV-cache device-memory management |

Qwen3-ASR's thinker feeds audio as **prefix tokens** into a decoder-only LLM; Whisper's decoder attends to the encoder output at every layer. **That is not adding a model, it is adding a class of attention to the runtime.** edge-llm's slot pool, streaming worker and N=2 support are better spent on Qwen3-ASR (5.3% Chinese) than on a model whose Chinese is unusable.

### ⚠️ The base fp16 encoder engine is numerically broken; bf16 is the fix

**An engine built from the whisper-base encoder ONNX with `trtexec --fp16` produces output with cosine 0.826 against onnxruntime.** It is deterministic run-to-run (maxdiff 0.0), so this is fp16 kernel selection rather than a race — fp16 has 5 exponent bits, and Whisper's encoder has high-dynamic-range spots in the residual accumulation and the softmax denominator. (We hit the same class of bug on SparkTTS: a `down_proj` output channel reached ~230k against fp16's 65504 ceiling.)

**bf16 fixes it at almost no cost in speed** — bf16 keeps fp32's 8 exponent bits and only gives up mantissa, 10 bits down to 7:

| Build | cosine vs onnxruntime | Encoder latency | End-to-end |
|---|---|---|---|
| `--fp16` | **0.826** | 10.75 ms | ❌ content drifts |
| **`--bf16`** | **0.9996** | **12.53 ms** | ✅ **bit-identical error rates to fp32** |
| `--fp16 --bf16` | **0.826** | 10.46 ms | ❌ see below |
| (no flag, fp32) | 1.000000 | 39.1 ms | ✅ |

bf16 recovers **3.1× of encoder speed** over fp32 (39.1 → 12.53 ms) and takes end-to-end TTFT from 46 ms to **18 ms**, with error rates on all four en/zh groups **identical to fp32**.

**⚠️ Do not pass `--fp16 --bf16` together.** With both flags TRT picks fp16 throughout and the resulting engine is **bit-identical** to the pure fp16 one (same cosine, same maxabsdiff, same std) — bf16 is simply not used. This is counter-intuitive; one expects offering more precision options to let TRT choose per layer. **Pass `--bf16` alone.**

**The defect is deceptive.** The bad encoder still emits a plausible-looking tensor — mean and variance in the right range — so the decoder greedily produces fluent English that drifts off-topic and stops early. By inspection it is indistinguishable from a KV-cache bug, and that is where the investigation went first. It was only settled by feeding the *same* encoder output to both onnxruntime's and TRT's decoders and finding **per-token argmax identical**, which moved the suspicion from the decoder to the encoder.

**It is model-specific.** Built the same way with `--fp16`, `enc_tiny_30s` (cosine 0.999864), `enc_tiny_10s` (0.999937) and both decoder engines are fine. So neither "fp16 is fine" nor "fp16 is unusable" generalises — **diff every engine against onnxruntime**, and treat that as a standing step in the engine build process (script in `bench/perf/whisper/`).

**Not located**: which layer overflows. `trtexec --fp16 --precisionConstraints=obey --layerPrecisions=` pinning LayerNorm/Softmax to fp32 and bisecting would find it; the remaining upside is only 12.53 → 10.75 ms, so it was not pursued.

### TensorRT encoder micro-benchmarks (numerically verified engines only)

`trtexec`, end-to-end latency including H2D/D2H (together under 0.3 ms):

| Encoder | Precision | TensorRT | whisper.cpp GGML CUDA | Ratio |
|---|---|---|---|---|
| base / 30s | **bf16** (fp16 unusable) | **12.53 ms** | 124.3 ms | **9.9×** |
| tiny / 30s | fp16 | **5.29 ms** | 103.7 ms | **19.6×** |
| tiny / 10s | fp16 | **1.61 ms** | — (Hailo-8 HEF is 23.8 ms) | 14.8× vs Hailo |

**Note that whisper.cpp already enables flash attention by default** (`cli.cpp:79`, default `true`), so this gap is not a missing optimisation.

### The decoder is bandwidth-bound, not compute-bound

whisper-base needs roughly 113 MFLOP per token (6 layers × ~10 MFLOP plus the vocabulary projection, 512×51865×2 ≈ 53 MFLOP). At 8 TFLOPS that is **0.014 ms**.

But every token must read the decoder weights once: 40M parameters × fp16 = **80 MB**, and Orin Nano has 8 GB of LPDDR5 at 68 GB/s → a **1.2 ms/token bandwidth floor**.

| | per token |
|---|---|
| Compute floor | 0.014 ms |
| **Memory bandwidth floor** | **1.2 ms** |
| TRT measured (`trtexec`, KV=16, encoder 1500 frames) | **2.63 ms** |
| whisper.cpp GGML CUDA measured | ~5 ms |

Measured lands at 2.2× the bandwidth floor and 188× away from the compute floor. This explains a counter-intuitive observation elsewhere in this report: **the Pi 5 CPU decoder runs at 8 ms/token and the Jetson GPU at 5–6 ms/token — orders of magnitude apart in compute, 1.4× apart in practice**, because both are waiting on memory.

TRT's benefit is therefore very asymmetric: **an order of magnitude on the encoder, about 2× on the decoder and already close to a hard floor**. Pushing the decoder further needs quantisation (int8 → a 0.6 ms floor), batching (read the weights once, serve several requests) or CUDA Graphs (removing launch overhead).

### Full TRT pipeline

Three engines (encoder / prefill / cached-step), following the split `Jonah-May-OSS/wyoming-whisper-trt` uses — the only open-source Whisper-TRT project with a real KV cache — except that optimum's export already provides that split. Cross-attention K/V are produced once by prefill and bound straight into the step engine as device pointers; self-attention K/V ping-pong between two device buffers per layer. Nothing returns to the host inside the decode loop.

| Orin Nano base/30s | bare TensorRT (bf16) | whisper.cpp CUDA |
|---|---|---|
| en short | **11.37%** | 13.59% |
| en long | 9.19% | 8.59% |
| zh long (t2s) | 16.71% | 15.99% |
| **TTFT** | **18 ms** (measured) | 216 ms (proxy) |
| **RTF (long)** | **0.008** | 0.023 |
| encoder | **12.5 ms** | 124 ms |
| decoder | 40–132 ms | 77–218 ms |

Accuracy is level within the noise of five files, but **TTFT is 9–16× better and RTF 3× better**, and the TRT figure is a measured first token rather than a proxy. **18 ms is clearly better than Hailo-8's 60 ms**, and that is at a 30 s window — a 10 s window would go lower still (`enc_tiny_10s` fp16 is verified and takes 1.61 ms), but that was **not measured and is not part of the conclusion**.

---

## Which Whisper each platform offers

| Platform | Variants | Window | Source |
|---|---|---|---|
| Hailo-8 / 8L / 10H | tiny, base (tiny.en on 10H only) | tiny **10s** / base **5s** | Hailo's own `Hailo-Application-Code-Examples/runtime/python/speech_recognition/app/download_resources.py`. **Not** the downloader in `ktomanek/edge_whisper` — its `--hw-arch hailo8` is an empty option, the `FILES` dict only has 8L and 10H |
| RK3562/3566/3568/**3576**/**3588**/RV1126B | base | **20s** | `airockchip/rknn_model_zoo/examples/whisper`, natively takes `--task en\|zh`. The official usage lists only `fp`, not i8 |
| Jetson | any | 30s (whisper.cpp) / any (self-built TRT) | No official TRT path. **The arm64 CTranslate2 on PyPI is CPU-only** (`get_cuda_device_count()` returns 0) |

**TTS: Hailo officially supports none.** A staff response states "Hailo currently doesn't support any TTS models", and the GenAI Model Zoo's 12 models contain no TTS entry. TTS on Raspberry Pi + Hailo runs on the CPU.

---

## Reproducing

Device runners and the scorer are in `bench/perf/whisper/`:

```bash
# RKNN (RK3588 / RK3576): --encoder_duration must match the window the .rknn was converted at
python3 rknn_whisper_run.py --corpus corpus --lang en \
  --encoder model/whisper_encoder_base_20s.rknn --decoder onnx_dec \
  --vocab-dir model --encoder_duration 20 --all-cores \
  --label rk3588-hybrid-en --out results/hybrid_en.json

# whisper.cpp CUDA (Jetson): do not pass -np, it also disables whisper_print_timings
python3 wcpp_corpus_run.py --corpus corpus --lang en \
  --bin ./whisper.cpp/build/bin/whisper-cli --model models/ggml-base.bin \
  --label orin-nano-wcpp-base-en --out results/wcpp_base_en.json

# Scoring — run once, on one machine
python3 score_all.py 'results/*.json'
```

Model conversion:

```bash
# RKNN: must convert on x86, and the toolkit version must match the device runtime
# (both boards run rknnlite 2.3.0)
python convert.py whisper_encoder_base_20s.onnx rk3588 fp out.rknn

# TensorRT: the base encoder needs bf16, tiny is fine on fp16 — verify either way
trtexec --onnx=enc_base_30s.onnx --bf16 --shapes=input_features:1x80x3000 --saveEngine=enc.plan
python3 cmp_engine_precision.py     # cosine >= 0.999, or the engine is not usable
```

---

## Gotchas

### Layout and shape

**The exported ONNX input rank must match what the runtime feeds, and a mismatch is not reported.** The encoder exported for Hailo is 4D NCHW `[1,80,1,1000]`; Rockchip's official one is 3D `[1,80,2000]`. When the element counts happen to match (80×1000), the ONNX conversion is legal and **rknn-lite raises nothing at runtime** — it reinterprets the buffer as 4D. The only symptom is the decoder emitting `(chiming)` / `(chewing)` and similar non-speech annotations.

This is the same class of failure as the RK matcha vocos case (shape mismatch with matching byte count → silent reinterpretation, −22 dB with no error). **There is no error message for this class of bug; only end-to-end semantic verification finds it.** The export script now takes `--input_rank {3,4}` to make the choice explicit.

Isolation method: verify the same ONNX end-to-end with onnxruntime on a dev machine *before* spending a conversion round on it.

### Quantisation

**`quantize_dynamic` output is an ORT-specific format and TensorRT rejects it** — `checkDynamicQuantizeLinear` / `checkMatMulInteger`. The int8 ONNX we use for the CPU decoder cannot be fed to TRT; an int8 TRT decoder would need explicit QDQ plus a calibration set, which is a different export path entirely.

**Rockchip's `convert.py` cannot actually produce a usable i8 model**: `rknn.build(do_quantization=do_quant)` is called with no `dataset=`, so there is no calibration data. That, not a platform limitation, is why the official usage lists only `fp`.

### TensorRT

**Do not infer numerical correctness from the precision flag** — see the bf16 section above. Prefer `--bf16` for overflow-class problems (it keeps fp32's exponent range and gives up only mantissa), and never pass `--fp16 --bf16` together.

### Vocabulary and decoding

**Rockchip's `read_vocab` splits on the FIRST space** (not the last), and **`base64_decode` must be their hand-rolled version** — it returns a single space the moment it meets `=`, which is how that vocab encodes a word break, so `base64.b64decode` changes the semantics.

**Their `base64_decode` has a defect we fixed**: `bytearray(len//4*3)` is sized to an upper bound but returned whole, so short decodes carry trailing `\x00`. Invisible on a terminal, scored as insertions. It needs `out[:oi]`.

### Hardware

**Binding RK3588's three NPU cores with `NPU_CORE_0_1_2` gains nothing** (encoder 260 ms vs RK3576's dual-core 246 ms, marginally slower) → this encoder is not NPU-compute-bound.

**RK3588 and RK3576 are equivalent on this workload**: English transcripts are **byte-identical**; 3 of 10 Chinese files differ in small ways (Traditional/Simplified choice, one comma).

**Hailo-8 grants `/dev/hailo0` to a single process** — anything else holding it returns `HAILO_OUT_OF_PHYSICAL_DEVICES (74)`; **two VDevices inside one process collide the same way**, so English and Chinese runs must be separate processes.

**The Hailo harness segfaults on exit.** After all files complete and the result JSON is written, the process exits with **rc=139 (SIGSEGV)** inside Hailo VDevice `release()`. It does not affect the data, but it breaks `&&` chains — use `; echo rc=$?`.

### Front end

**Whisper pads the *waveform* to the window length and then computes the mel**, not the other way round. The mel of digital silence is about **−0.58, not 0.0**, so zero-padding the finished mel shows the encoder a constant that never occurs in training. Our numpy port had this wrong on the RK path; the Hailo path used upstream `audio_utils` (which pads the waveform) and was correct.

A/B on RK3588 after the fix:

| | before | after |
|---|---|---|
| 20s en short | 13.37% | 13.37% |
| 20s en long | 6.33% | 7.58% |
| 20s zh long (t2s) | 20.47% | 19.63% |
| **10s en long** | **14.26%** | **11.40%** |

**Only the 10 s English long-form change is outside the noise (−2.9 points)**, which matches the mechanism: a wrong padding value only affects the padded part, and at 20 s the whole corpus fits in one window while at 10 s the tail chunks are mostly padding. The ±1 point moves at 20 s are noise on five files — comparing transcripts line by line, the content is nearly identical, differing by a leading space.

**The mel front end can be pure numpy**; the boards need neither torch nor librosa. Verified against `torch.stft` at max|diff| ~1e-5, mean ~1e-7, and the filterbank is **bit-identical** to `librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)` (max|diff| 0.0) — Rockchip's shipped `mel_80_filters.txt` is exactly that matrix. Dropping torch freed 622 MB on the Pi.

### TTFT is sensitive to the prefill implementation

TTFT = encoder + first token. In a two-stage decoder (prefill + cached step) the first token is the **most** expensive step, not the cheapest: prefill consumes the full encoder output and produces every cross-attention K/V. Measured at 28–48 ms on Hailo, where the encoder alone takes 24 ms.

An early run measured 15 ms and that figure reached the document; re-measuring with more warm-up showed it was an outlier. **Report the first-token distribution rather than a single mean, and state the warm-up count.**

### Toolchain

**`optimum` 1.27.0 breaks against recent torch**: `ImportError: cannot import name '_attention_scale' from torch.onnx.symbolic_opset14`. Pin `torch==2.6.0` + `transformers==4.49.0`.

---

## Through the shipped backend

Everything above was measured with the per-platform harness runners, which drive each vendor runtime directly. This section measures the same corpus through `voxedge.backends.whisper` — the backend a deployment actually loads. Result JSONs are in `bench/perf/whisper/results_backend/`, the harness baselines in `results_harness/`.

The two differ in exactly one place: **the harness cuts long audio at a fixed hop and stitches the overlapping transcripts; the backend cuts at silence and concatenates** (reusing `voxedge.audio.segment`, which the RK and TRT-Edge-LLM backends already use for their own fixed-context decoders).

### When the audio fits one window, the two are numerically identical

| RK3588, group | harness | backend | chunks |
|---|---|---|---|
| 20s en short | 13.37% | 13.37% | 1 |
| 20s en long | 7.58% | 7.58% | 1 |
| 20s zh short | 52.00% | 52.00% | 1 |
| 20s zh long | 32.32% | 32.32% | 1 |
| 10s en short | 11.37% | 11.37% | 1 |
| 10s zh short | 55.32% | 55.32% | 1 |

Six groups, thirty files, not one digit of difference. Everything from the mel front end through the encoder to the decoder is doing the same arithmetic in both paths, so any remaining gap is attributable to segmentation alone.

### Where they segment, cutting at silence wins

| RK3588 10s, group | harness (fixed hop + overlap) | backend (silence) |
|---|---|---|
| en long | 11.40% | **10.44%** |
| zh long | 48.77% | **42.14%** |

### The Chinese number is a bug this run found

The first backend run scored 49.13% on `zh_long` — *worse* than the harness. One file explained it: `zh_long_03` came back as `…上下文語經中找到找到找到…×18…並能针对特定问题…`, a repetition run sitting in the middle of the segment with correct transcript on both sides.

voxedge's degeneration guard had two anchors — a period explaining the whole segment, and one anchored at the tail — and a run with content on both sides matches neither. Adding a third anchor (`voxedge` 0f7eb7b) took `zh_long` from 49.13% to **42.14%**, past the harness baseline.

Two properties of that fix are worth stating, because a guard that over-fires is worse than one that under-fires:

- **The 20 s configuration did not move at all** — 32.32% before and after, byte for byte. That configuration never degenerated, and the new anchor left it alone.
- The interior anchor has no coverage check to fall back on, so its repeat thresholds are stricter than the whole-segment ones: 6 units, or 12 for a single character. `对对对`, `very very very`, `no no no no` and 排比 structures survive, with a test for each.

### Jetson: two independently built bf16 engines agree exactly

| | en short | en long | zh short | zh long | encoder | TTFT |
|---|---|---|---|---|---|---|
| Orin NX, base/30s | 13.59% | 9.19% | 56.12% | 35.39% | 11.4 ms | 58-83 ms |
| Orin Nano, base/30s | 13.59% | 9.19% | 56.12% | 35.39% | 13.1-13.5 ms | 88-112 ms |

The error rates are identical across all four groups. The two `.plan` files were built separately — Orin Nano's during the harness round, Orin NX's freshly with `trtexec --bf16` — from the same ONNX. Given that the fp16 build of this graph fails *silently* (cosine 0.826, fluent output that drifts off-topic), two independent builds agreeing on twenty files is the evidence that the bf16 recipe is reproducible rather than one lucky engine.

Only the speed differs, which is what a device comparison should show.

### RK3576: correct transcripts, destroyed timings at 20 s

`cat-remote` dropped off the network mid-run and looked like a crash. It was not one — `uptime` showed 65 days when it came back, and all four result files were on disk, including the one the run appeared never to produce. **The box kept working; only the link died.**

Accuracy is fine and matches the other RK board:

| RK3576, group | 10 s | 20 s | RK3588 same config |
|---|---|---|---|
| en short | 11.37% | 17.81% | 11.37% (10 s) |
| en long | 10.44% | 10.44% | 10.44% (10 s) |
| zh short | 52.59% | 44.94% | 55.32% / 52.00% |
| zh long | 45.68% | 32.32% | 42.14% / 32.32% |

English at a 10 s window is **identical to RK3588 to the digit**. Chinese is not, which is consistent with the earlier finding that the two boards agree byte-for-byte on English and diverge on Chinese.

The timings at 20 s are another matter:

| group | RTF 10 s | RTF 20 s |
|---|---|---|
| en short | 0.134 | 1.755 |
| en long | 0.095 | 6.365 |
| zh short | 0.178 | 137.5 |
| zh long | 0.114 | 46.2 |

Worst single files: `zh_short_04` spent **2,057,339 ms** in the decoder (34 minutes) and `zh_long_01` 2,391,377 ms (40 minutes), against 0.5-1.3 s for the same files at 10 s.

**It was memory, and the kernel log says so.** The board has 7.9 GB of RAM, **no swap**, and was running an unrelated voice service holding 2.85 GB RSS:

```
[8月27日 18:24:59] python3 invoked oom-killer: ... global_oom
[8月27日 18:24:59] Out of memory: Killed process 696509 (python3)
                   anon-rss:1645212kB shmem-rss:1539232kB      ← 3.2 GB
```

The container restarted at 18:25:16, and the benchmark's last output file was written at **18:25** — it had been stalled for tens of minutes and finished the minute the OOM killer freed 3.2 GB.

The split between the two stages says where the cost landed: the encoder slowed 5-15×, the decoder three orders of magnitude. The encoder is the part on the NPU. The decoder is a 315 MB ONNX graph that onnxruntime memory-maps and touches on **every** autoregressive step, so once the page cache holding those weights is under reclaim pressure, each step re-reads them from slow storage. Doubling the window doubles the encoder output the decoder cross-attends over, and its working set with it.

**Two lessons that outlive this board.** A CPU ONNX decoder is a page-cache resident, not a fixed allocation — `free` looking healthy before a run says nothing about whether its weights stay resident during one. And a benchmark sharing a board with a service is measuring the pair, not the board: here the benchmark did not merely get slow numbers, it took down a production container.

### Hailo-8: the backend beats the vendor harness on short-form, and segmentation decides long-form

The backend was measured against the same corpus the harness table above uses, so these are directly comparable.

| | harness | backend |
|---|---|---|
| tiny/10s en short | 10.95% | **7.30%** |
| base/5s en short | 13.81% | **11.59%** |
| base/5s en long | 19.03% | **14.60%** |
| tiny/10s en long | **21.58%** | 52.37% |

Getting there took three fixes, and none of them are Hailo-specific — all three were in shared code, and all three were found only by running on hardware.

**The decoder on this path never emits EOS.** It transcribes correctly and then repeats the sentence until the token budget runs out. `en_short_01` came back as the right sentence four times over, scoring 168%. Two separate holes in the degeneration guard let it through:

1. **The repeat threshold was set for single words.** Spaced-language text needed 6 repeats before anything collapsed, because "no no no no" is ordinary English. A 10-word sentence repeated 4 times sat one short of that bar. Thresholds now scale with unit length — 6+ words need 3 repeats. English short-form: **168.19% → 7.30%** on tiny.
2. **Capping tokens changed the shape of the failure without removing it.** 32 tokens holds about two and a half short sentences, and "two full repeats plus a started third" counts as 2, which lands inside the standing rule that two repeats always survive. The partial now counts, for long units only — which is what keeps "I love you. I love you. I love you so much" intact.

**Frame energy cannot segment a 4-second window.** After Hailo's boundary guard the base HEF leaves 4 s of usable audio, and continuous speech essentially never goes quiet inside 4 s, so the energy splitter fell back to a hard cut mid-phrase at nearly every boundary. Switching to VAD, which decides on speech rather than loudness:

| base/5s, en long | energy | VAD |
|---|---|---|
| en_long_01 | 84.2% | 10.5% |
| en_long_02 | 18.2% | 15.2% |
| en_long_03 | 0.0% | 0.0% |
| en_long_04 | 25.0% | 18.8% |
| en_long_05 | 33.3% | 28.6% |
| **mean** | 32.15% | **14.60%** |

Four of five files improved, none got worse, and the spread collapsed from 0-84.2 to 0-28.6.

**The tiny/10s comparison, by contrast, is noise** — and saying so matters more than the 52.37% in the table. Per-file it went 5.3→10.5, 12.1→45.5, 70.0→55.0, 18.8→93.8, 95.2→57.1: two better, three worse, swings of up to 75 points in both directions. At a 10 s window these 10-11 s files split into exactly two chunks either way, so there is a single cut and its placement is close to a coin flip; at a 5 s window there are three or four cuts and better placement compounds. **Do not read the tiny/10s long-form row as evidence that VAD hurts.**

### The controlled re-run separates the two cleanly

Stopping that one container (`docker stop`, the other two left running) took available memory from 990 MB to 3747 MB. Re-running the same two configurations on the same board:

| group | RTF contended | RTF alone | error rate contended | error rate alone |
|---|---|---|---|---|
| en short | 1.755 | **0.200** | 17.81% | 17.81% |
| en long | 6.365 | **0.105** | 10.44% | 10.44% |
| zh short | 137.5 | **0.262** | 44.94% | 44.94% |
| zh long | 46.19 | **0.126** | 32.32% | 32.32% |

Encoder 4319 ms → 245 ms, decoder 64341 ms → 914 ms, minimum available memory 3700 MB throughout, and no new OOM in `dmesg`. The container was restarted afterwards and came back healthy.

**Every error rate is identical to the digit; every timing moved by 8× to 525×.** Contention destroyed the timings and left the transcripts untouched — which is what "a slow decode is still a correct decode" means, stated as a measurement rather than an assumption.

**Use the uncontended row.** RK3576 at a 20 s window is RTF 0.105-0.262, in the same band as its own 10 s figures and as RK3588. The pathological column is a record of what a shared board measures, not of what this board does.

---

## Through the OVS server: the number the other tables do not have

Everything above measures the backend in-process. This section drives a live
OpenVoiceStream server over `WS /asr/stream` with `bench/perf/perf.py asr`, on
RK3588 with the `rk3588-whisper-10s` profile, same corpus, 5 samples per group.

| group | Finalize RTF p50 | **EOS→Final p50** | CER/WER p50 |
|---|---|---|---|
| en short | 0.120 | **461 ms** | 10.0% |
| en long | 0.092 | **1013 ms** | 10.5% |
| zh short | 0.139 | 486 ms | 53.3% |
| zh long | 0.097 | 1314 ms | 38.6% |

The error rates match the in-process figures (11.37% / 10.44% English at the same
window), which is the check worth having: the server path adds no accuracy of its
own.

**EOS→Final is the figure this backend should actually be judged on, and it does
not exist in-process.** Whisper has no streaming state, so nothing is emitted
until finalize — the user sees nothing for 461 ms after a short utterance and a
full second after a long one. For comparison, the README's table has Paraformer
on Orin NX at 58 ms EOS-to-audio for the whole voice-to-voice loop. Whisper here
is a transcription backend that can be used conversationally, not the reverse.

Chinese was measured with `WHISPER_LANGUAGE=zh` pinned at the container. It has
to be pinned: this backend declares no LANGUAGE_ID capability and does not honour
a per-request language, so it decodes with the configured token and reports that
truthfully. A first attempt without pinning decoded Mandarin audio as English and
scored 200-446% — a meaningless number rather than a low score, and one worth
naming so nobody records it as Whisper's Chinese accuracy.

### The deployment path had a silent defect, and only running it found this

`WHISPER_LANGUAGE=zh` on the container did nothing at first. `profile_loader`
keeps a hand-maintained list of env prefixes a profile may not overwrite, and
`WHISPER_` was not in it, so the profile's `en` silently replaced the operator's
`zh`. Nothing raised, nothing was logged, and the only symptom was Mandarin
coming back as English — which reads as a broken model long before it reads as an
ignored environment variable.

Provisioning, by contrast, worked first time: an empty model directory, the
profile applied, 360 MB fetched from `harvestsu/whisper-edge` through hf-mirror,
and `whisper-rknn` serving 116 s after container start.

---

## Known limitations

- **Five files per group; one file moving shifts a group mean by 5–10 points.** Three independent instances in this round: on Orin Nano tiny (7.30%) beats its own base (13.59%) on the strength of a single file, `en_short_05`; RK3576 and RK3588 differ by 4.4 points on English short entirely because of two words in `en_short_03`; and `zh_short_05` flipping Traditional/Simplified opens a 9.5-point gap. **Read the magnitudes and the bands, never the ranking.**
- **Jetson whisper.cpp TTFT is a proxy** and is not comparable to the measured TTFTs in the same column.
- **The other ASR backends' numbers in `docs/performance-comparison.md`** (Paraformer 2.6% and so on) were measured 2026-05-13, on different dates with different images, each platform running its own model. Same audio bytes and the same scoring function, everything else different — **read the order of magnitude only**.
- Not covered: the tiny variants on Jetson (orin-nano lacked the disk for a second set of decoder engines — tiny is 4 layers / d384 and cannot reuse base's), and int8 RKNN.
- **RK3576's first 20 s pass was taken under contention** with an unrelated 2.85 GB service on a board with no swap, and is kept only as a record of that failure mode. The uncontended re-run is the comparable one.
- **Hailo long-form rests on five files with per-file swings up to 75 points.** The base/5s VAD improvement is consistent across files; the tiny/10s difference is not, and is reported as noise.
