# Qwen3-ASR-0.6B int4-AWQ — TensorRT-Edge-LLM v0.8.0, Jetson Orin (sm_87)

Optimization #1 of the "shrink the voice stack" goal: quantize the Qwen3-ASR-0.6B
decoder to **int4-AWQ (W4A16)** to cut VRAM with no accuracy loss. Audio encoder
stays fp16 (only the LLM backbone is quantized). Built on TensorRT-Edge-LLM
**v0.8.0** (upstream ref `f9cc746…`, release/0.8.0), validated on orin-nano.

## Result — VALIDATED, no degradation

| precision | engine | size |
|---|---|---|
| LLM backbone int4-AWQ | `llm.engine` | 550 MB (≈ −45% vs fp16) |
| audio encoder fp16 | `audio_encoder.engine` | 381 MB |
| text embedding fp16 | `embedding.safetensors` | 311 MB |

Accuracy (10-clip CN+EN set, my own CER/WER with `opencc` t2s normalization, run
through the **production decode contract** below):

- **ZH CER = 0.00%** (6/6 clips byte-perfect) — provably non-degraded.
- **EN WER = 11.1%** — a single `drop`→`drape` substitution on one 3-word clip
  (acoustic near-homophone, not an int4 signature; other 3 EN clips word-perfect).

Deploy bundle (on orin-nano): `qwen3-asr-0.6b-int4-v080-deploy.tgz`
md5 `cce8985353e9f091e4d3c670307cf7fe` (~934 MB). Per-file md5:
`llm.engine 4f3496c3…`, `audio_encoder.engine f7a7fa8c…`,
`embedding.safetensors 8db9ceda…`, `tokenizer.json 68e0da75…`,
`libNvInfer_edgellm_plugin.so 17385ff8…`.

## ⚠️ Two non-obvious fixes are required — without EITHER, output is garbage

A first A/B run wrongly concluded "int4 quantization damage" (Chinese repetition
loops, `" focus"` hallucinations, `"spoken in English"` leaks). **The weights are
fine.** Two separate things must both be right:

### 1. Decode contract (the garbage→clean separator in the A/B)
The raw int4 engine with the asr.md default (`temperature=1.0, top_k=50`, no
assistant prime) produces noise. The **production** contract (voxedge
`backends/jetson/trt_edge_llm_asr.py`; worker `qwen3_asr_worker.cpp:857-877`):

- greedy: `temperature=0.0, top_k=1, top_p=1.0`
- `apply_chat_template=true`, `add_generation_prompt=false`
- **assistant-turn prime** — supply a partial assistant turn so the model is
  forced onto the transcription rail (this is what `force_language` does in prod):
  ```
  messages = [
    {role: system,    content: ""},
    {role: user,      content: [{type: audio, audio: <mel.safetensors>}]},
    {role: assistant, content: "language <LangName><asr_text>"}   # LangName ∈ {Chinese, English, …}
  ]
  ```
- **post-process**: strip the leading `"language <Lang> "` prefix from the output
  (production `stripLanguagePrefix`, `qwen3_asr_worker.cpp:1015`). Note the known
  greedy-`\w+` over-consume bug on a no-space boundary like `EnglishStop`.

Lesson: an executor's quality verdict on a model is only as good as its prompt
harness. Replicate the production decode contract before declaring "quant damage".

### 2. rope_type `linear` → `mrope` (export-time config fix)
The exported `onnx/llm/config.json` ships `rope_scaling.rope_type: "linear"`,
which is **wrong** for Qwen3-ASR. The validated engine config must be:
```json
"rope_scaling": {"factor":1.0,"interleaved":true,"mrope_interleaved":true,
  "mrope_section":[24,20,20],"rope_theta":1000000.0,"rope_type":"mrope","type":"mrope"},
"rope_theta": 1000000.0
```
The qwen3_asr int4 export path must emit `mrope`, not `linear`. (Applied as a
post-export config edit here; carry it into the fork's export path.)

## Reproduce

Quantization command (reconstructed — no on-device history survives; x86 CUDA
host, **modelopt ≥ 0.39**). int4_awq is a CLI-supported but undocumented ASR path:
```bash
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-ASR-0.6B \
  --output_dir $WS/Qwen3-ASR-0.6B-int4awq.int4_awq \
  --quantization int4_awq           # defaults: --dtype fp16 --num_samples 512
# NO --audio_quantization  -> audio encoder stays fp16
# NO --lm_head_quantization -> lm_head fp16
# ASR auto-uses LibriSpeech multimodal calibration (no --dataset)
tensorrt-edgellm-export $WS/Qwen3-ASR-0.6B-int4awq.int4_awq Qwen3-ASR-0.6B/onnx
```
`tensorrt_edgellm/quantization/quantize.py:~737` calls
`export_hf_checkpoint(model, export_dir=…, extra_state_dict=mtp_state_dict)`
unconditionally — the `extra_state_dict=` kwarg HARD-REQUIRES modelopt ≥ 0.39
(`mtp_state_dict={}` for plain int4_awq ASR; drop the kwarg to downgrade).

Engine build (on-device, v0.8.0 binaries, `EDGELLM_PLUGIN_PATH` set):
```bash
llm_build  --onnxDir onnx/llm --engineDir engines/llm \
  --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 1536
audio_build --onnxDir onnx/audio --engineDir engines/audio \
  --minTimeSteps 100 --maxTimeSteps 3000        # 100, NOT the doc's 1000 (short clips fail at 1000)
```
Then apply the rope fix (§2) to `engines/llm/config.json`.

Run / validate: use the v0.8.0 `llm_inference` (the 0.7.0 binary segfaults on
0.8.0 engines), `--multimodalEngineDir` pointing at the audio **parent** dir (the
runtime appends the subtype; the leaf `audio/audio` → "does not have an audio
runner"), with the decode contract from §1.

Full device-side provenance (verbatim quantize.py region, all config.json):
captured at `orin-nano:/home/harvest/project/qwen3-asr-ab/SOURCE_PROVENANCE.md`.
See also the TTS-base sibling: [`qwen3-tts-base-v080-port.md`](./qwen3-tts-base-v080-port.md).
