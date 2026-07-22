# Handoff v2: CustomVoice Qwen3-TTS Talker Low-Bit Quantization (W8A16 / W8A8 / W4A16)

**Date**: 2026-06-12
**Status**: BLOCKED — no real-path low-bit quantization of the CustomVoice talker has ever
worked; FP16 is the only EOS-valid precision. Five distinct quant schemes exhausted.
**Supersedes / continues**: `third_party/jetson-voice-engine/engine-overlay/addon/docs/known-issues/w8a16-talker-handoff.md` (v1, 2026-05-25).
**Goal that motivated this**: run the talker engine at <16-bit to save memory so Orin NX 16 GB
can co-resident ASR+TTS+LLM, and ideally make **W8A16 the default CustomVoice precision on
Jetson**. 2026-06-12 update: FP16 CustomVoice with a 1024-token Talker KV cap now passes a clean
Orin Nano ASR-ready + TTS inference smoke, but it is not a wide-margin Nano default because the run
used swap.

---

## TL;DR for the next engineer

1. The CustomVoice Qwen3-TTS talker (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, 28 decoder layers,
   hidden=1024) **breaks codec-EOS prediction under every low-bit weight quantization we have
   tried**. Symptom: greedy decode never samples the codec-EOS token (id **2150**) and runs to
   the frame cap (200 frames / 16 s, or 4096 frames / 511 chunks on the v0.7.1 worker) emitting
   hallucinated audio. **FP16 on the identical worker always terminates correctly** (28/42/38
   frames, byte-exact ASR).
2. **The "it worked in 0.7.1 / 5/26" memory is real but was preload-masked.** There is a genuine
   `goal_status met:true` at 2026-05-26 04:42 (session `8e644b32`, line 2817) showing W8A16 →
   radxa ASR `今天天气真不错哦。`. **But that smoke ran with
   `QWEN3_TTS_PRELOAD_TALKER_EMBEDS=/tmp/ref_talker_embeds_15row.bin`**, which injects a fixed
   15-row reference talker embedding and **bypasses the talker's autoregressive generation
   entirely** — so the audio was correct regardless of W8A16 quality. The W8A16 talker's real
   generation was never validated. (Subagent `a5e30e7f119fd6d5b`.)
3. **Validation rule, non-negotiable**: any W8A16/W8A8/W4 "pass" MUST be a **no-preload** run
   (`env -u QWEN3_TTS_PRELOAD_TALKER_EMBEDS`), greedy, with **bounded frame count + codec-EOS
   firing + ASR-correct audio**. A correct WAV with preload set proves nothing.
4. **Reusable assets produced** (do not redo): the v0.7.1 `W8A16LinearPlugin` is now **ported into
   the v0.8.0 runtime** (so a v0.8.0 worker can load a W8A16 engine), a full-EOS-coverage
   calibration driver, an AWQ ONNX rewriter, a SmoothQuant QDQ path, and a no-preload EOS smoke
   harness. See "Reusable infrastructure" below.
5. **2026-06-12 correction after re-checking the remembered "stable precision" path**: the older
   `talker_decode_w8a16_outputk.engine` did use W8A16, but its TensorRT boundary was **FP32**:
   `inputs_embeds`, every `past_key_*` / `past_value_*`, `logits`, `last_hidden`, and every
   `new_past_key_*` / `new_past_value_*` binding are `DataType.FLOAT`. The current v0.8 CustomVoice
   W8A16 engine is a different contract: `inputs_embeds`, combined `past_key_values_*`,
   `hidden_states`, and `present_key_values_*` are all `DataType.HALF` (only `logits` is FLOAT).
   This is the main migration gap. Do not treat "rollback selected nodes to FP16" as equivalent to
   the old stable route.
6. **Next high-value path**: make v0.8 consume a FP32-boundary Talker engine, or port the old
   split-KV FP32 Talker runner contract forward. Plain v0.8 `AttentionPlugin`/`W8A16LinearPlugin`
   cannot do this by config alone: both plugins currently accept HALF activations/KV boundaries.

---

## The complete failure matrix

All runs: greedy (`talker_temperature=0, talker_top_k=1, predictor_temperature=0,
predictor_top_k=1`), **no preload**, v0.8.0 worker `qwen3_tts_worker` md5 `37035ecc`, prompts
`今天天气真不错` / `我们今天一起去公园散步看看风景吧` / `the weather is really nice today`.

| # | Scheme | Bits | Act quant | Scale granularity | Calib | Realization | Result |
|---|--------|------|-----------|-------------------|-------|-------------|--------|
| 1 | naive max-abs W8A16 | W8 | none (A16) | per-output | none | W8A16LinearPlugin | 200 frames, no EOS |
| 2 | AWQ W8A16 (flawed) | W8 | none | per-output (collapsed) | `max_new=8` (missed EOS region) | W8A16LinearPlugin | 200 frames, no EOS |
| 3 | AWQ W8A16 (corrected) | W8 | none, weight-only | per-output (collapsed) | full-EOS (`max_new=128`, reaches eos 2150) | W8A16LinearPlugin | 200 frames, no EOS |
| 4 | SmoothQuant W8A8 | W8 | **int8 per-tensor** | per-channel weight | full-EOS | **native TRT QDQ (no plugin)** | 200 frames, no EOS |
| 5 | AWQ W4A16 groupwise | **W4** | none | **true per-group (block 128)** | full-EOS | **int4GroupwiseGemmPlugin (native groupwise)** | 200 frames, no EOS |
| — | **FP16 control** | 16 | — | — | — | — | **28/42/38 frames, EOS ✓, ASR byte-exact** |

**Axes swept and eliminated**: bit-width (4 and 8), weight-only vs weight+activation, per-output
vs per-channel vs per-group scales, AWQ vs SmoothQuant, naive vs full-EOS-coverage calibration,
custom plugin vs native TRT int8 vs groupwise plugin. Every combination fails identically; FP16 is
always clean on the same worker. This rules out: the worker, the ONNX export, preload, the custom
plugin, the AWQ scale-collapse hypothesis, and calibration coverage. **What remains is that
quantizing the 28 decoder-body linears — by any scheme — perturbs the hidden state enough to
suppress the EOS logit.** The heads (`codec_head`/`node_linear_196`, `lm_head`) were kept FP16 in
all attempts; that alone is insufficient (already known from v1 handoff).

---

## Why each "success" in the history was not real

- **2026-05-25 (v1 handoff)**: naive W8A16 explicitly documented as EOS-runaway (4096 frames). AWQ
  recommended but never completed. The older `talker_decode_w8a16_outputk.engine` "success" is a
  **different, older Qwen3-TTS model** (not CustomVoice); its source ONNX is deleted and
  deserialization was noted as failing.
- **2026-05-26 04:42 (`8e644b32` L2817, `met:true`)**: W8A16 + radxa ASR `今天天气真不错哦。` —
  **ran with `QWEN3_TTS_PRELOAD_TALKER_EMBEDS=/tmp/ref_talker_embeds_15row.bin`** (subagent
  `a5e30e7f119fd6d5b` evidence). Preload bypasses talker generation → correct audio is from the
  fixed reference, not the W8A16 talker. **This is the run people remember as "跑通".**
- **Eager PyTorch "cos 0.9983 / EOS logit 3.99 vs 4.03"**: not reproducible end-to-end; almost
  certainly modelopt's internal per-layer AWQ-reconstruction metric, not a head-logit comparison.
  Do not treat as evidence the quant is lossless.
- **A later eager pre-check (this session)** comparing FP16 vs quant head-logits returned cos≈0 /
  top1=0% for *both* W4 and a known-good W8 control → the eager harness itself was invalid (it
  compared two independently-quantized instances / mismatched inputs). **Eager logit comparison on
  this speech model is unreliable; trust the on-device no-preload smoke instead.**

---

## Reusable infrastructure (already built — reuse, don't rebuild)

On **orin-nx** unless noted. (`seeed-orin-nx` is real production — never touch it.)

| Asset | Path | Notes |
|-------|------|-------|
| **W8A16LinearPlugin ported into v0.8.0** | `~/project/edgellm-v080/build/libNvInfer_edgellm_plugin.so` md5 `7d3fabe24661a0ce47b03e71b29bd6c5` | v0.7.1 plugin was already `IPluginV3` → trivial port. Registers W8A16LinearPlugin; all v0.8.0 plugins (GDN/int4/attention) retained; FP16 unbroken. Backup: `…so.1.0.bak` md5 `90e4dddddd6d9924ccc2fa5a9f477758`. **A v0.8.0 worker can now load a W8A16 engine.** |
| W8A16 plugin/kernel source | `~/project/edgellm-v080/cpp/plugins/w8A16LinearPlugin/`, `cpp/kernels/w8A16LinearKernels/` | Kernel is **per-output only** (`w8A16Linear.cu:302` rejects non-`kPerOutput`; `:306` group_size only 0/1/k). `scale_mode`/`group_size` plugin fields exist but no groupwise kernel path. |
| int4 groupwise plugin (native groupwise) | `~/project/edgellm-v080/cpp/plugins/int4GroupwiseGemmPlugin/` | 3 inputs: act kHALF, packed int4 weights kINT8 `[N/2,K]`, per-group scales kHALF `[K/group,N]`; attrs gemm_n/gemm_k/group_size. Used by Qwen3.5-4B-AWQ. |
| AWQ ONNX rewriter | `third_party/jetson-voice-engine/engine-overlay/addon/scripts/quantize_onnx_matmul_w8a16_awq.py` | Bakes `pre_quant_scale` as `Mul` nodes + per-output int8 weight. **Collapses per-group amax to per-output** (~line 139) because W8A16LinearPlugin is per-output. |
| int4 DQ→plugin fold | `tensorrt_edgellm/llm_models/layers/int4_gemm_plugin.py:581` `int4_dq_gemm_to_plugin(graph)` | Folds `DequantizeLinear(int4, per-group scales, block_size)→MatMul` into `Int4GroupwiseGemmPlugin`. Standalone-runnable. |
| Full-EOS-coverage calibration driver | wsl2-local `~/w8a16-calib/calib_talker_awq2.py` (+ `_sq` SmoothQuant, `_w4` variants) | modelopt 0.43/0.44, talker = `Qwen3TTSTalkerForConditionalGeneration`, 196 linears, head excludes, calibration that **reaches natural EOS** (`max_new>=128`). Includes the 5 transformers-5.3 compat patches needed to load the model. |
| SmoothQuant QDQ rewriter | wsl2-local `~/w8a16-calib/quantize_onnx_matmul_w8a8_qdq.py` | Emits native QDQ (Quantize/DequantizeLinear) int8; TRT builds int8×int8 natively (no plugin). Note `mtq.export_onnx` does NOT exist in modelopt 0.44 and torch.onnx can't reproduce the KV-cache+AttentionPlugin graph — that's why a rewriter is used. |
| int4 groupwise emitter | wsl2-local `~/w8a16-calib/quantize_onnx_matmul_w4_groupwise.py` + `int4_fold_standalone.py` | Per-group int4 DQ (no collapse) + AWQ `Mul(inv_pqs)` + fold. (Needs `onnxslim`, `gs 0.6.1`.) |
| No-preload EOS smoke driver | orin-nx `~/w8a16-port/drive_worker.py` | Drives the simple-CLI `qwen3_tts_worker` (4 dir flags), greedy, `env -u QWEN3_TTS_PRELOAD_TALKER_EMBEDS`. |
| FP16 talker source ONNX | orin-nx `~/qwen3-tts-onnx-minimal-fix/llm/model.onnx` (+ `.data`) md5 `7e182cf65b8639e5e10ad2a9f303f0cb` | The quant input. ("minimal-fix" = the 3-bug-fixed FP16 talker that works real-path.) |
| FP16 b1 / b2 engines (WORK) | `~/tensorrt-edgellm-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-v080-tts/talker` (b1) and `…-tts-b2/talker` (b2, maxBatch=2) | b1 EOS-clean (28/42/38 frames). b2 verified for N=2 batch-lane. config.json `max_batch_size` 1 vs 2. |
| FP16 kv1024 engine (WORK; new default candidate) | orin-nx `~/qwen3-tts-fp16-kv1024-prodonnx/talker/llm.engine` md5 `ef16cc7e8cbd5045578eb4268e0c42b9` | Built from the production v0.8 CustomVoice ONNX, not the older `minimal-fix` ONNX. `builder_config.max_input_len=max_kv_cache_capacity=1024`. No-preload greedy gates passed: `今天天气真不错` → 42 frames, ASR exact; `我们今天一起去公园散步看看风景吧` → 46 frames, ASR exact. |
| Orin Nano kv1024 staging smoke (WORK; narrow margin) | orin-nano `~/kv1024-test/models/talker_direct` + v0.8 worker/plugin copied direct from orin-nx | Clean no-preload TTS: `今天天气真不错` → 42 frames, 3.36 s, `audio_complete=true`. ASR-ready + TTS dual-open: ASR ready in 7438 ms; TTS ready in 9417 ms; TTS still 42 frames / 3.36 s / `audio_complete=true`; memory after TTS 4.3 GiB used, 2.9 GiB available, swap 912 MiB. The container on Nano was not current; this was a standalone staging validation. |
| Built (broken) quant engines for inspection | `~/w8a16-port/corrected/` (AWQ-W8 461 MB), `~/w8a16-port/sq/` (W8A8 499 MB), `~/w8a16-port/w4g/engine_out/llm.engine` (W4 groupwise 246 MB) | All run-away; kept for diffing. |
| Selective-attention-projection rollback probe | `~/w8a16-port/select_attnproj_fp16_0612a/` | Rolled back 112 q/k/v/o W8A16 nodes to FP16 MatMul (`W8=84`, `MatMul=113`). Built and ran, but still capped at 200 frames / 16 s; ASR `嗯嗯嗯嗯`. This is evidence that FP16 rollback is not the remembered stable route. |
| Old W8A16 output-k reference | `~/qwen3-models/engines/orin-nx/highperf/talker_w8a16_outputk/talker_decode_w8a16_outputk.engine` md5 `267e8fdfc782172c0df0eac8a92a04af` | Deserializes with v0.7.1 plugin. Binding dtypes are FP32 split-KV (`past_key_i` / `past_value_i`, `new_past_key_i` / `new_past_value_i`) plus FP32 `inputs_embeds`, `logits`, and `last_hidden`. This is the concrete artifact behind the remembered W8A16 + stable-boundary path. |
| v0.8.0 worker (no-preload + N=2 capable) | `~/project/v080-worker-build/build_v080/workers/qwen3_tts_worker` md5 `37035ecc` | The smoke worker. |

Build entrypoint (engine from a quantized ONNX): `~/project/edgellm-v080/build/examples/llm/llm_build
--onnxDir <dir-with-config.json> --engineDir <out> --maxBatchSize 1 --maxInputLen 8192
--maxKVCacheCapacity 8192` with `EDGELLM_PLUGIN_PATH` = the ported `.so`. Gotcha: stage `config.json`
in the onnx dir; patch its `builder_config.max_input_len/max_kv_cache_capacity` to 8192 or the
worker rejects the engine.

---

## Reproduce the failure / validate a fix (the only gate that counts)

```bash
# on orin-nx — no preload, greedy, real autoregressive path
EDGELLM_PLUGIN_PATH=~/project/edgellm-v080/build/libNvInfer_edgellm_plugin.so \
env -u QWEN3_TTS_PRELOAD_TALKER_EMBEDS \
python3 ~/w8a16-port/drive_worker.py \
  --worker ~/project/v080-worker-build/build_v080/workers/qwen3_tts_worker \
  --talker <CANDIDATE_TALKER_ENGINE_DIR> \
  --code-predictor ~/tensorrt-edgellm-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-v080-tts/code_predictor \
  --code2wav    ~/tensorrt-edgellm-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-v080-tts/code2wav \
  --tokenizer   ~/tensorrt-edgellm-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-v080-tts/tokenizer \
  --text "今天天气真不错" --talker-temperature 0 --talker-top-k 1
# PASS: done event frames ~26-30 (NOT 200), EOS fires; then ASR the WAV (sherpa paraformer) → 今天天气真不错
# FAIL: frames=200, audio_s=16.0, garbage. Always run the FP16 b1 talker as the control.
```

---

## What to try next (only if pursuing W8A16 further; otherwise ship FP16)

Ranked by expected value:

1. **Restore the old stable precision boundary in v0.8**. The confirmed-good W8A16 output-k engine is
   not merely "some layers FP16"; it is a FP32 split-KV Talker contract. To validate this route under
   v0.8, either:
   - port the old `past_key_i` / `past_value_i` FP32 Talker runner path forward and adapt
     `qwen3OmniTTSRuntime` to feed FP32 `inputs_embeds` / read FP32 `last_hidden`; or
   - extend the v0.8 LLM builder/runtime/plugins so Talker `inputs_embeds`, KV cache, and
     `hidden_states` can be FP32 (or at least a proven BF16 variant, but prior notes say BF16-KV and
     full-BF16 did not recover reference codes).
   Source constraints to fix: `AttentionPlugin::supportsFormatCombination` currently requires HALF
   Q/K/V and HALF/FP8 KV; `W8A16LinearPlugin::supportsFormatCombination` requires HALF activation and
   HALF output; `qwen3OmniTTSRuntime.cpp` allocates Talker loop buffers as HALF.
2. **Per-layer EOS-logit diagnostic, then selective high precision**. If FP32-boundary migration is
   too invasive, instrument the talker forward to dump, at the natural sentence end, the EOS-token
   (2150) logit and the winning logit for FP16 vs quantized. Prefer FP32/BF16 candidates over FP16
   rollback: a 2026-06-12 q/k/v/o FP16 rollback probe still capped at 200 frames.
3. **QAT / distillation** of a quantization-friendly talker, or a smaller talker checkpoint. Days–
   weeks; only worth it if Orin Nano CustomVoice TTS is a hard requirement.
4. **Accept FP16** (recommended for product until the FP32-boundary port exists). FP16 is the only
   v0.8 CustomVoice precision currently EOS-valid; FP16 N=2 batch-lane is
   verified on Orin NX. Use FP16 on nx; Nano gets a different/lighter TTS backend (matcha/kokoro)
   rather than CustomVoice. Keep the ported W8A16 plugin as a ready asset for a future
   quantization-friendly talker.

**Do NOT** repeat any row of the failure matrix (naive / AWQ-W8 / AWQ-W8-fullcalib / W8A8-SmoothQuant
/ W4-groupwise), and **do NOT** evaluate any candidate with preload set.

---

## References

- v1 handoff: `third_party/jetson-voice-engine/engine-overlay/addon/docs/known-issues/w8a16-talker-handoff.md`
- Original CustomVoice W8A16 session (preload "PASS" at L2817): `~/.claude/projects/-Users-harvest-project-seeed-local-voice/8e644b32-8e6e-4441-9d31-a67425e8faff.jsonl`; W8A16 subagent `a5e30e7f119fd6d5b`.
- v0.7.1 fork (plugin/kernel source): `mac:/Users/harvest/project/TensorRT-Edge-LLM`, `orin-nx:~/project/v071-build/TensorRT-Edge-LLM`.
- modelopt configs: `INT4_AWQ_CFG` (W4 groupwise, awq_lite, input_quantizer disabled), `INT8_SMOOTHQUANT_CFG` (W8A8). modelopt 0.43/0.44 has **no** `INT8_AWQ_CFG`.
- Failure signature constant: codec-EOS token id **2150**; cap 200 frames (v0.8.0) / 4096 frames / 511 chunks (v0.7.1).
</content>
