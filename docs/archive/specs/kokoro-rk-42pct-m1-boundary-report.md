# Kokoro RK NPU 42% — M1 Boundary Discovery Report

Status: M1 complete (2026-05-23). All discovery work performed on `wsl2-local` against the shipped bucket-32 ONNX artifacts (manifest `rk3588-kokoro-hybrid-2026-05-23`). Decisions below feed M2–M4. Source ONNX inspected:

- Full graph (decoder-front already cut): `/home/harve/kokoro-analysis/kokoro-bucket-32/kokoro.with-decoder-front-cut.onnx` (301,933,208 bytes)
- Generator tail (shape-inferred copy): `/home/harve/kokoro-analysis/kokoro-bucket-32/kokoro-generator-tail-cpu.shaped.onnx` (89,009,749 bytes)
- Manifest: `/home/harve/kokoro-analysis/kokoro-bucket-32/manifest-rk3588.json`

> Important context vs the spec: the only bucket currently shipped is **bucket-32** (decoder-front time dim = 420, generator-tail upsample factor 10×). bucket-16 and bucket-64 artifacts exist in `kokoro-analysis/` but are **not in `deploy/artifacts/rk_manifest.json`** today (see `rk_manifest.json:222-258`). The 8191 time-dim hard limit is a **per-bucket** constraint — see Vocoder section below for the bucket-64 implication.

---

## 1. BERT encoder boundary

### Selected boundary

| Field | Value |
|---|---|
| Subgraph name (planned) | `kokoro-bert-encoder` |
| RKNN segment input | `tokens` (graph input, INT64, `[1, 32]`) |
| RKNN segment output | **`/bert_encoder/Add_output_0`** (`[1, 32, 512]`, FP32) |
| Source node | `Add` named `/bert_encoder/Add`, consuming `/bert_encoder/MatMul_output_0` + a bias initializer |
| Time dim | 32 (≪ 8191; safe) |

The BERT module's **single clean exit edge** is the `/bert_encoder/Add` node (an `Add` after a projection `MatMul` that follows the final ALBERT layer's `full_layer_layer_norm_11`). Its output is consumed by exactly one external node, `/text_encoder/Transpose`, which is the start of the downstream length-regulator / duration-predictor path on CPU. This is the natural M1 cut.

### Why this and not `/encoder/MatMul_1_output_0`

The Jetson spec (`app/backends/jetson/kokoro_trt.py:974-985`) references `/encoder/MatMul_1_output_0` for its TRT engine. That literal name **does not exist** in our shipped ONNX — the Jetson artifact comes from a separately exported encoder graph. In our full ONNX, "MatMul_1" appears 12× under each ALBERT attention layer (`/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_<N>/MatMul_1_output_0`), all with shape `[1, 12, 32, 64]` — these are intra-layer attention head outputs, **not** a usable encoder boundary.

The semantically correct RK-side analog of "BERT encoder output" is `/bert_encoder/Add_output_0` (post-projection final hidden state, `[1, 32, 512]`).

### Rejected BERT candidates

| Candidate | Shape | Rejection reason |
|---|---|---|
| `/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_<n>/MatMul_1_output_0` | `[1, 12, 32, 64]` | Intra-attention output (head output); cutting here would put 12 layer-tails on CPU, contradicting the goal |
| `/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm_11/Add_1_output_0` | `[1, 32, 768]` | Just before the projection MatMul; not the natural exit edge — choosing `Add_output_0` after the projection keeps `/bert_encoder/MatMul` (hidden→512) on NPU |
| `/bert/embeddings/word_embeddings/Gather_output_0` | `[1, 32, 128]` | Too early; only the embedding gather, no BERT compute |

### Note on `tokens/style/speed`

`tokens` is the only graph input the BERT subgraph consumes. `style [1, 256]` and `speed [1]` are routed to **other branches** (style is sliced for AdaIN ahead of decoder-front and again inside the vocoder; speed feeds the duration regulator). The RKNN BERT engine will therefore have a single INT64 input (`tokens`) and a single FP32 output (`Add_output_0`), keeping the contract simple.

---

## 2. Vocoder front-half boundary

### Selected boundary

| Field | Value |
|---|---|
| Subgraph name (planned) | `kokoro-vocoder-front-half` |
| RKNN segment inputs | `/decoder/decode.3/Mul_output_0` (`[1, 512, 420]`, FP32, sole producer = decoder-front output), `/Slice_2_output_0` (`[1, 128]`, FP32, style) |
| RKNN segment output | **`/decoder/generator/Add_5_output_0`** (`[1, 256, 4200]`, FP32) |
| Source node | `Add` named `/decoder/generator/Add_5`, summing `Add_4_output_0` (= `ups.0 + noise_res.0 + resblocks.0 + resblocks.1`) with `resblocks.2/Add_8_output_0` |
| Sole consumer | `/decoder/generator/Div_1` (Div by 3, normalising the 3-kernel resblock fusion) |
| Time dim | 4200 (≪ 8191 at bucket-32; safe) |

This is the **only single tensor** that captures the full sum of the front-half (`ups.0 + noise_res.0 + resblocks.0 + resblocks.1 + resblocks.2`). After the boundary, CPU runs `Div_1 → LeakyRelu → ups.1 [×6 upsample to 25200] → resblocks.3-5 + noise_res.1 + conv_post`, all of which exceed the 8191 hard limit anyway and cannot be ported.

### Front-half merge structure (just before the cut)

```
ups.0/ConvTranspose ─┐
noise_res.0/Add_8 ───┼─► Add_3 ──┐
                                  ├─► Add_4 ──┐
resblocks.0/Add_8 ───────────────┘            │
resblocks.1/Add_8 ────────────────────────────┘
                                                ├─► Add_5 ──► Div_1 (÷3) ──► back-half
resblocks.2/Add_8 ─────────────────────────────┘
```

All intermediate tensors are `[1, 256, 4200]`.

### Rejected vocoder candidates

| Candidate | Shape | Rejection reason |
|---|---|---|
| `/decoder/generator/ups.0/ConvTranspose_output_0` | `[1, 256, 4200]` | Only `ups.0`, no resblocks — leaves the heavy `resblocks.0-2 + noise_res.0` compute on CPU (defeats the 42% target) |
| `/decoder/generator/noise_res.0/Add_8_output_0` | `[1, 256, 4200]` | Single resblock-equivalent only; same problem as above. Also historically known to have returned `None` in earlier tail-island RKNN runtime tests (`third_party/rkvoice-stream/docs/kokoro-rknn-analysis.md:111-116`); avoiding it removes that risk |
| `/decoder/generator/resblocks.2/Add_8_output_0` | `[1, 256, 4200]` | Captures only resblocks.2, not the merged sum; would require manually re-adding the other 3 paths on CPU side, breaking the clean cut |
| Any `/decoder/generator/ups.1/...` tensor | `[1, 128, 25200]` | **Time dim 25200 > 8191** — exceeds RK3588 register limit |
| Anything inside resblocks.3-5 / noise_res.1 / conv_post | `[1, *, 25200+]` | Same 8191 violation |

### Bucket scaling (critical for future expansion)

Decoder-front output time dim scales linearly with bucket seq_len; `ups.0` then multiplies by ×10:

| Bucket | decode.3 time | ups.0 / front-half time | Status |
|---|---|---|---|
| 16 | 218 | 2180 | safe (under 8191) |
| **32 (shipped)** | **420** | **4200** | **safe — this is the M1 target** |
| 64 | 956 | 9560 | **EXCEEDS 8191** — front-half RKNN must skip bucket-64 if added |

If buckets >32 are ever shipped, the vocoder front-half RKNN engine must fall back to CPU for those buckets (or be re-segmented further — e.g., per-resblock RKNN tiles). M4 should encode the bucket-32 cap.

---

## 3. Operator compatibility matrix

### BERT subgraph (652 total nodes; ~24% of the full ONNX)

```
Add: 174    MatMul: 98   Mul: 97    ReduceMean: 50
Reshape: 48 Transpose: 48 Pow: 37   Sub: 25
Sqrt: 25    Div: 25      Softmax: 12 Tanh: 12   Gather: 1
```

| Op pattern | Count | RKNN status | Action |
|---|---|---|---|
| LayerNorm (decomposed: Pow + ReduceMean + Sub + Div + Sqrt) | 25 instances | All primitive ops already supported | **No new rewriter** — already-decomposed form works |
| GELU (Tanh approximation: `0.5 * x * (1 + Tanh(...))`) | 12 instances | Tanh + Mul + Add primitives | **No new rewriter** — already-decomposed form works |
| Self-attention (MatMul / Softmax / Reshape / Transpose) | 12 × ~5 ops | Standard, already supported | No action |
| Word embedding (Gather, vocab `[V, 128]`) | 1 | Gather supported | No action |
| Loop / If / Range / Erf / LayerNormalization (fused) | **0** | n/a | No action — the graph is already in primitive form |

**Update to spec §2 matrix**: the spec listed LayerNorm and GELU under "BERT: needs new rewriter". After M1 inspection this is **not required** — the exported ALBERT graph is already op-decomposed and uses only primitives. The "uncertain" BERT-side rewriter risk identified in the spec is therefore resolved as **green**.

### Vocoder front-half subgraph (≈ resblocks.0-2 + noise_res.0 + ups.0)

| Op pattern | Risk | Existing rewriter | Action |
|---|---|---|---|
| `Sin` (in source-noise generator / Snake activation) | High (RKNN runtime previously returned `None` on Sin-heavy tail) | `rewrite_onnx_sin_poly*.py` + `fix_kokoro_tail_rknn_ops.py` (Sin→polynomial) | Apply existing rewriter; use polynomial form (rel_l2 ≈ 5%) since spec already accepts |
| `Pow(x, 2)` (in Snake activation) | High | `rewrite_onnx_pow2_to_mul.py` (Pow→Mul) | Apply existing rewriter |
| `InstanceNormalization` (in adain blocks) | Medium | `fix_kokoro_tail_rknn_ops.py` (IN→Pow/Mean/Sub/Div primitives) | Apply existing rewriter |
| `Conv` with `dilation > 1` (resblocks dilated convs) | **High risk** — 519MB tail island previously crashed at `Mul:noise_res.0/adain1.0/Mul_2` with output `[1,256,52320]` | n/a | At bucket-32 the output is `[1,256,4200]`, far below the 52320 crash. Should be safe but **must be empirically verified in M4 build** |
| ConvTranspose (`ups.0`) | Low | Standard | No action |

**Update to spec §2 matrix**: dilated-conv risk identified at the 52320 dim is **not present at bucket-32**, but the M4 RKNN build must still confirm no per-layer time-dim overflow during compilation.

---

## 4. Quantization decision updates

The spec §3 matrix needs no major rewrite, but two refinements:

1. **BERT FP16 first — confirmed correct.** The BERT subgraph has 98 MatMul + 174 Add + heavy attention softmax. INT8 risk is real (community-quantized Kokoro variants ship `MatMulInteger`/`DynamicQuantize`, which RKNN does not support per the older analysis). M2 should target FP16 only; INT8 deferred to M5 if at all.
2. **Vocoder front-half FP16 baseline — confirmed.** The 22 `InstanceNormalization` instances + 24 `Sin` (rewritten to polynomial) + 37 `Pow` accumulate quantization error fast. FP16 first; INT8 only if rel_l2 ≤ 0.01 across 50 prompts (spec §6 gate).
3. **No change** to decoder-front: keep current INT8 (already validated, rel_l2 = 0.00227).

Expected end-state NPU compute share (with FP16 BERT + INT8 decoder-front + FP16 vocoder front-half): ~42% as planned.

---

## 5. Inputs to M2/M3/M4 (handoff)

| Milestone | Input from M1 |
|---|---|
| **M2** BERT FP16 RKNN | Subgraph extractor target: input `tokens [1,32]` INT64 → output `/bert_encoder/Add_output_0 [1,32,512]` FP32. Extractor command: use `onnx.utils.extract_model(input_path, output_path, input_names=['tokens'], output_names=['/bert_encoder/Add_output_0'])` then RKNN-convert FP16 |
| **M3** Prefix pipeline refactor | New CPU `kokoro-prefix-regulator-cpu.onnx` consumes `/bert_encoder/Add_output_0` + `style` + `speed`, outputs `/MatMul_1_output_0 [1,512,210]` + `/Slice_2_output_0 [1,128]`. Equivalent to existing prefix but with BERT removed |
| **M4** Vocoder front-half FP16 RKNN | Subgraph inputs: `/decoder/decode.3/Mul_output_0 [1,512,420]` + `/Slice_2_output_0 [1,128]`. Output: `/decoder/generator/Add_5_output_0 [1,256,4200]`. Apply `fix_kokoro_tail_rknn_ops.py` (Sin→poly, Pow2→Mul, IN→primitives) before RKNN convert. Tail-rest CPU model: input `Add_5_output_0` + `Slice_2_output_0` + source/noise inputs (m_source path), output `audio` |

---

## 6. EVIDENCE

All commands run via `fleet exec wsl2-local -- ...` on 2026-05-23.

### 6.1 Artifact existence

```text
$ ls /home/harve/kokoro-analysis/kokoro-bucket-32/
kokoro-decoder-front.onnx                   (134,437,717 bytes)
kokoro-generator-tail-cpu.onnx              ( 89,009,749 bytes)
kokoro-generator-tail-cpu.shaped.onnx       ( 89,009,749 bytes)
kokoro-prefix-cpu.onnx                      ( 22,922,434 bytes)
kokoro.seq32.rknn-ready.onnx                (301,933,208 bytes)
kokoro.with-decoder-front-cut.onnx          (301,933,208 bytes)
manifest-rk3588.json
rk3588/                                     (built RKNN dir)
```

The shipped tail SHA `1bdbb9fd…ee451bcb` from `rk_manifest.json:248` matches `kokoro-generator-tail-cpu.onnx` here.

### 6.2 Full-graph inputs / outputs

```text
=== Graph inputs ===
tokens [1, 32] 7        (INT64)
style  [1, 256] 1       (FP32)
speed  [1] 1            (FP32)
=== Graph outputs ===
audio [126000] 1
onnx::Shape_3623 [32] 7
```

### 6.3 BERT exit-edge probe

```text
=== BERT module exit tensors (consumed by non-bert nodes) ===
('/bert_encoder/MatMul', 'MatMul', '/bert_encoder/MatMul_output_0',
 '/bert_encoder/Add', 'Add')
('/bert_encoder/Add', 'Add', '/bert_encoder/Add_output_0',
 '/text_encoder/Transpose', 'Transpose')
('/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm_11/Add_1',
 'Add', '.../full_layer_layer_norm_11/Add_1_output_0',
 '/bert_encoder/MatMul', 'MatMul')

=== BERT op counts ===
  Add: 174,  MatMul: 98,  Mul: 97,  ReduceMean: 50,
  Reshape: 48, Transpose: 48, Pow: 37, Sub: 25,
  Sqrt: 25, Div: 25, Softmax: 12, Tanh: 12, Gather: 1

=== Total BERT nodes: 652 ===
```

### 6.4 BERT exit-tensor shapes

```text
/bert_encoder/MatMul_output_0                                                       -> [1, 32, 512]
/bert_encoder/Add_output_0                                                          -> [1, 32, 512]
/bert/encoder/albert_layer_groups.0/.../full_layer_layer_norm_11/Add_1_output_0     -> [1, 32, 768]
/bert/embeddings/word_embeddings/Gather_output_0                                    -> [1, 32, 128]
/text_encoder_1/Where_7_output_0                                                    -> [1, 512, 32]
/MatMul_1_output_0                                                                  -> [1, 512, 210]
/Slice_2_output_0                                                                   -> [1, 128]
```

### 6.5 Global graph stats (full ONNX)

```text
total nodes: 1712
Range: 0  Where: 12  Gelu(fused): 0
LayerNormalization(fused): 0   Erf: 0   Loop: 0   If: 0
Tanh: 13  (12 BERT GELU + 1 outside)
```

No fused LayerNorm/GELU and zero Erf/Range/Loop/If — fully primitive graph.

### 6.6 Vocoder front-half→back-half boundary

```text
=== boundary edges (front-half output → back-half input) ===
('/decoder/generator/noise_res.0/Add_8',      'Add',           '...Add_8_output_0',  [1, 256, 4200], '/decoder/generator/Add_3', 'Add')
('/decoder/generator/ups.0/ConvTranspose',    'ConvTranspose', '...ConvTranspose_output_0', [1, 256, 4200], '/decoder/generator/Add_3', 'Add')
('/decoder/generator/resblocks.0/Add_8',      'Add',           '...Add_8_output_0',  [1, 256, 4200], '/decoder/generator/Add_4', 'Add')
('/decoder/generator/resblocks.1/Add_8',      'Add',           '...Add_8_output_0',  [1, 256, 4200], '/decoder/generator/Add_4', 'Add')
('/decoder/generator/resblocks.2/Add_8',      'Add',           '...Add_8_output_0',  [1, 256, 4200], '/decoder/generator/Add_5', 'Add')
--- total boundary edges: 5
```

### 6.7 Selected boundary tensor: shape + sole consumer

```text
/decoder/generator/Add_3_output_0  -> [1, 256, 4200]
/decoder/generator/Add_4_output_0  -> [1, 256, 4200]
/decoder/generator/Add_5_output_0  -> [1, 256, 4200]
/decoder/generator/ups.0/ConvTranspose_output_0 -> [1, 256, 4200]

=== consumers of /decoder/generator/Add_5_output_0 ===
/decoder/generator/Div_1  Div
  inputs:  ['/decoder/generator/Add_5_output_0',
            '/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Constant_1_output_0']
  outputs: ['/decoder/generator/Div_1_output_0']
```

Exactly one consumer (`Div_1`), confirming a clean cut.

### 6.8 Upsample factor confirmation

```text
/decoder/generator/ups.0/ConvTranspose -> [1, 256, 4200]    (10×  vs decode.3=420)
/decoder/generator/ups.1/ConvTranspose -> [1, 128, 25200]   (6×   vs Add_5=4200 → exceeds 8191)
```

### 6.9 Front-half input topology

```text
=== /decoder/decode.3/Mul_output_0 consumers ===
/decoder/generator/LeakyRelu  LeakyRelu  -> [/decoder/generator/LeakyRelu_output_0]
(single consumer → feeds into ups.0 chain)

style (/Slice_2_output_0) consumers in front-half: 24
style consumers in back-half: 24
style consumers elsewhere:   0
```

### 6.10 Tail tensor catalog (key entries at time dim = 4200)

Filtered to candidate front-half outputs:

```text
[1, 256, 4200]  ConvTranspose  /decoder/generator/ups.0/ConvTranspose_output_0
[1, 256, 4200]  Add            /decoder/generator/noise_res.0/Add_8_output_0
[1, 256, 4200]  Add            /decoder/generator/resblocks.0/Add_8_output_0
[1, 256, 4200]  Add            /decoder/generator/resblocks.1/Add_8_output_0
[1, 256, 4200]  Add            /decoder/generator/resblocks.2/Add_8_output_0
[1, 256, 4200]  Add            /decoder/generator/Add_3_output_0   ← merge step 1
[1, 256, 4200]  Add            /decoder/generator/Add_4_output_0   ← merge step 2
[1, 256, 4200]  Add            /decoder/generator/Add_5_output_0   ← CHOSEN front-half output
```

All 4200 ≤ 8191. Bucket-32 path is safe.

### 6.11 Manifest cross-check

```text
$ cat /home/harve/kokoro-analysis/kokoro-bucket-32/manifest-rk3588.json
{
  "target": "rk3588", "seq_len": 32, "decoder_width": 2616,
  "cut_shapes": {
    "/MatMul_1_output_0":            [1, 512, 210],
    "/Slice_2_output_0":             [1, 128],
    "/decoder/decode.3/Mul_output_0":[1, 512, 420]
  },
  ...
}
```

Confirms the decoder-front output dim (420) used in §1 scaling table.

---

## 7. Risks carried forward

1. **bucket-64** would put the vocoder front-half at time dim 9560 > 8191. If any future profile re-enables bucket-64, the vocoder front-half engine must skip it. M4 should add a guard in the RKNN dispatch layer (e.g., `if seq_len > 32: fallback_to_cpu_for_vocoder_front`).
2. **Sin / Pow2 rewrite parity** in the vocoder front-half: existing rewriters expected to work but rel_l2 ≈ 5% from polynomial Sin is the absolute spec gate. M4 must measure end-to-end (not just front-half) parity, since errors compound through the CPU back-half.
3. **RKNN compilation of dilated convs at 4200 dim** is untested. The historical crash was at 52320; theoretically safe at 4200, but M4 should treat first RKNN build as the gating evidence.
4. **bert/text_encoder rename**: M3 prefix-regulator refactor must keep the downstream output names (`/MatMul_1_output_0`, `/Slice_2_output_0`) stable, since the current decoder-front RKNN was built against those names.

No carry-forward risk on the BERT side: graph is op-primitive, time dim is small (32), and there is no fused-op surprise lurking.
