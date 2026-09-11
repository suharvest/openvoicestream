# Kokoro RK NPU 推至 42% R&D spec

> Status: **CLOSED 2026-05-23** — 34% NPU shipped (target revised from 42% after BERT bit-exact dead-code A/B verification at commit `3b18517`). 4-stage hybrid pipeline + 3-bucket dynamic router + misaki ZH G2P all in production (final commits 735b5e9 / ca9332c / e2665d0 / 05aca9d). TTFA short EN/ZH p50 0.70-0.75s (4.1× / 4.4× improvement), mid EN/ZH p50 1.6s (2×), HTTP RTF 0.59 (10% gain). All identified follow-up paths systematically evaluated and either delivered, NO-GO (with evidence), or parked (require explicit user kickoff). **For the full closure report, paths evaluated, and decision rationale see `docs/archive/specs/kokoro-rk-perf-r-and-d-closure.md` (commit 4e2d6ff)**. Reproduction guide: `docs/archive/specs/kokoro-rk-34pct-reproduction-guide.md`.
>
> Historical context: original 42% budget assumed BERT contributed 23.7% of compute; that compute exists in the graph but is not on the audio path. M2 BERT FP16 RKNN artifact (commit `50463cb`) is retained as a reference/diagnostic build but is not wired into the runtime — it would only relocate dead computation to the NPU. M3 (prefix split) is closed as NO-OP per the same finding.

## §1 Subgraph boundary selection

目标不是重试 full graph RKNN。full graph 已知会撞 RK3576/RK3588 单层时间维寄存器上限：文档记录 NPU time-dim 上限为 `8191`，Kokoro vocoder 内部达到 `13081`，上采样后更高，因此这是硬件维度问题，不是普通 op support 问题（`third_party/rkvoice-stream/docs/kokoro-rknn-analysis.md:9-18`）。当前已验证 topology 是 `CPU prefix -> RKNN decoder-front -> CPU generator tail`，其中 prefix 输出 `/MatMul_1_output_0` 和 `/Slice_2_output_0`，decoder-front 输出 `/decoder/decode.3/Mul_output_0`（`third_party/rkvoice-stream/docs/kokoro-rknn-analysis.md:68-70`；`third_party/rkvoice-stream/models/tts/kokoro/export_kokoro_decoder_front.py:24-27`）。

**A. BERT encoder boundary**：local repo 没有可直接 inspect 的 `prefix.onnx` 文件，find 只看到 manifest/config JSON，未发现 `.onnx/.rknn` artifact；因此 RK prefix 内部 BERT 输出 tensor name 不能从本机 ONNX metadata 确认。设计采用 Jetson split 作为候选边界：`tokens/style/speed` 进入 encoder engine，encoder 后由 CPU length regulator 接续；Jetson runtime 明确执行 `encoder -> CPU length regulator -> decoder`（`app/backends/jetson/kokoro_trt.py:966-985`）。其 encoder-to-length boundary 在 Jetson artifact 中是 `/encoder/MatMul_1_output_0`，后续 decoder 还消费 `/decoder/decoder/F0_conv/Conv_output_0`、`/decoder/decoder/N_conv/Conv_output_0`、`/decoder/decoder/Unsqueeze_output_0` 和 `style`（`app/backends/jetson/kokoro_trt.py:974-985`）。RK spec 决策：新增 `kokoro-bert-encoder.onnx/rknn`，候选输入为 `tokens`，候选输出为 `/encoder/MatMul_1_output_0`；`style`、`speed` 保留在 CPU prefix-regulator 侧，避免把 duration/length regulator 的 shape-sensitive path 一起推入 RKNN。落地前必须用 `inspect_onnx_tensors.py` 按 substring 查 BERT/encoder 节点和 shape。

**B. Vocoder front-half boundary**：当前 CPU tail 输入已确认是 `/decoder/decode.3/Mul_output_0` 和 `/Slice_2_output_0`，输出 `audio`（`third_party/rkvoice-stream/models/tts/kokoro/export_kokoro_decoder_front.py:115-130`）。目标 front-half 是 `resblocks.0-2 + noise_res.0`，算力占 16.8%，文档判定"能上 RKNN"；back-half `resblocks.3-5 + noise_res.1` 占 50.2%，因 dim=13081 不能上 RKNN（`third_party/rkvoice-stream/docs/kokoro-rknn-analysis.md:24-26`）。`probe_kokoro_tail_splits.py` 支持从 tail 输入抽任意 candidate tensor（`third_party/rkvoice-stream/models/tts/kokoro/probe_kokoro_tail_splits.py:6-12`）。已知候选包括 `/decoder/generator/ups.0/ConvTranspose_output_0` 和 `/decoder/generator/noise_res.0/Add_8_output_0`，但 `noise_res.0` 曾在真实 runtime 返回 `None`，需谨慎（`third_party/rkvoice-stream/docs/kokoro-rknn-analysis.md:111-116`）。最终决策：先用 `probe_kokoro_tail_splits.py` 找到所有 `resblocks.2` consumer 与第一个 `resblocks.3` input；若 exact tensor 无法稳定导出，则采用保守 front-half boundary 到 `/decoder/generator/noise_res.0/Add_8_output_0`。

**目标 topology**（基于 M1 boundary report + ground-truth inspect 2026-05-23 修正）：

```
tokens, style, speed
   |
   | tokens ─► [RKNN bert-encoder FP16, M2]──► /bert_encoder/Add_output_0 [1,32,512]
   |                                                                            |
   |                                                                            ▼
   | style, speed ─────────────────────────────► [CPU residual prefix ORT, M3]
   |                                                  (text_encoder + duration predictor)
   |                                              outputs: /MatMul_1_output_0 [1,512,210]
   |                                                       /Slice_2_output_0  [1,128]
   |                                                                            |
   ▼                                                                            ▼
                              [RKNN decoder-front INT8]  ← existing validated, unchanged
                                          | output: /decoder/decode.3/Mul_output_0
                                          ▼
                              [RKNN vocoder-front-half FP16, M4: ups.0 + resblocks.0-2 + noise_res.0]
                                          | output: /decoder/generator/Add_5_output_0 [1,256,4200]
                                          ▼
                              [CPU vocoder-back-half: resblocks.3-5 + noise_res.1 + post/ISTFT]
                                          |
                                       audio
```

**Ground truth 备注**：当前 shipped `kokoro-prefix-cpu.onnx`（22MB，sha `549fece0…`）包含 776 个 `bert*`-named 节点（整个 BERT + text_encoder + predictor），从 `tokens/style/speed` 走到 `/MatMul_1_output_0 + /Slice_2_output_0`。BERT 当前是 CPU。RKNN decoder-front（40MB INT8）实际 inputs 只有 `[/MatMul_1_output_0, /Slice_2_output_0]`、不含 BERT 权重（kmodel.decoder.* only）。要让 M2 BERT FP16 RKNN 真正接入，必须把现有 CPU prefix 在 `/bert_encoder/Add_output_0` 处再切一刀：上半 = BERT（由 M2 RKNN 替代），下半 = `residual prefix CPU`（text_encoder + predictor，输入 `/bert_encoder/Add_output_0 + style + speed`，输出维持原 ABI）。decoder-front 和 tail 不动。

## §2 Operator compatibility pre-assessment

**BERT side**：full graph 之前在 text encoder reshape/constant-fold 点失败，包括 `/text_encoder_1/Transpose_output_0_rs` shape mismatch（`third_party/rkvoice-stream/docs/kokoro-rknn-analysis.md:56-62`）。`fix_kokoro_rknn.py` 已处理 `Loop/If/Sequence`、`Range`、`RandomNormalLike/RandomUniformLike`（`:64-69`、`:155-164`、`:314-322`）。BERT 特有的不友好算子 LayerNorm、GELU、Embedding/Gather、Rotary positional Range、Where 尚无专用 rewriter；若任一算子顽固，需新写 BERT rewriter 或采用更窄的 encoder 切分。

**Vocoder front-half side**：`noise_res.0` 含 AdaIN/InstanceNormalization/Snake（`Sin/Pow`），之前导致 RKNN runtime `None`。`fix_kokoro_tail_rknn_ops.py` 明确 rewrite `Sin` 为多项式、`Pow(x,2)` 为 `Mul(x,x)`、`InstanceNormalization` 为 primitive ops（`:8-12`、`:143-158`）。量化精度对比：clip 多项式 rel_l2≈5%，range-reduced Sin≈0.1% 但较慢（`third_party/rkvoice-stream/docs/kokoro-rknn-analysis.md:281-296`）；决策：使用 `Pow2->Mul`，不使用近似 Sin，除非 parity 通过。

**Dilated conv 风险**：前一次 519MB tail 在真机 RK3588 崩于 `Mul:/decoder/generator/noise_res.0/adain1.0/Mul_2`，输出 `[1,256,52320]`（`third_party/rkvoice-stream/docs/kokoro-rknn-analysis.md:72-75`）。任何候选 tensor 的 time dim 若超 8191，在 boundary discovery 阶段即拒绝，不进入 RKNN build。

**现有 rewriter 覆盖矩阵：**

| 算子 | BERT | vocoder front-half | 现有 rewriter |
|---|:---:|:---:|---|
| Sin | 低风险 | 高风险 | `rewrite_onnx_sin_poly*.py`、`fix_kokoro_tail_rknn_ops.py` |
| Pow(x,2) | 低风险 | 高风险 | `rewrite_onnx_pow2_to_mul.py` |
| InstanceNorm | 无 | 有 | `fix_kokoro_tail_rknn_ops.py` |
| LayerNorm | 有 | 无 | 无（需新写） |
| GELU | 有 | 无 | 无（需新写） |
| Gather/Embedding | 有 | 无 | 无（需评估） |
| Loop/If/Range | 有 | 无 | `fix_kokoro_rknn.py` |

## §3 Quantization strategy

**校准数据**：使用 `dump_kokoro_tail_inputs.py` 走真实 tokenizer + prefix ORT + RKNN decoder-front，逐 SLV prompt dump `.npy` 中间 tensor（`:50-97`）。选 128-256 个 prompt，覆盖短英文、中文、标点符号、混合语言。存储格式：per-segment `.npy` + JSON manifest，不使用纯 random calibration（`quantize_rknn_probe.py` 的随机策略对 tail 验证已显示误导性）。

**量化决策矩阵：**

| Segment | 建议精度 | 理由 |
|---|---|---|
| BERT encoder | FP16 first | BERT int8 常见质量劣化；社区 quantized ONNX（MatMulInteger/DynamicQuantize 方案）RKNN 不兼容（`kokoro-rknn-analysis.md:171-176`） |
| vocoder front-half | FP16 baseline；int8 实验 | 之前 selective int8 rel_l2=0.07~0.41，超出标准（`kokoro-rknn-analysis.md:334-346`） |
| decoder-front | INT8（保持现状） | 已验证 rel_l2=0.00227，RTF 0.655（`kokoro-rknn-analysis.md:186-195`） |

## §4 Implementation milestones

| 里程碑 | 产出物 | Verification gate | 预计耗时 |
|---|---|---|---|
| M1 Boundary discovery | `boundary-report.md`：确认 BERT 和 vocoder front-half 输出 tensor name + shape | 无候选 time dim >8191；`inspect_onnx_tensors.py` / `probe_kokoro_tail_splits.py` 输出中可引用 exact name | 半天 |
| M2 BERT FP16 RKNN | `kokoro-bert-encoder.onnx`、`rk3588/kokoro-bert-encoder.fp16.rknn`、extractor script | ORT vs RKNN hidden max-abs-diff ≤0.02；RK3588 inference 非 None | 1 天 |
| M3 Prefix split + 4-stage wiring | (a) `kokoro-bert-only.onnx`（`tokens → /bert_encoder/Add_output_0`）— 用于校验 M2 BERT RKNN parity；(b) `kokoro-prefix-residual-cpu.onnx`（`/bert_encoder/Add_output_0 + style + speed → /MatMul_1_output_0 + /Slice_2_output_0`）；(c) `kokoro_rknn.py` `_preload_hybrid` / `_infer_segment_hybrid` 改为 4 段流水（CPU tokenize → BERT RKNN → residual prefix ORT → decoder-front RKNN → tail ORT），新增第二个 RKNN handle | 端到端 audio vs 旧三段路径 max-abs-diff ≤0.02、rel_l2 ≤0.005；`/MatMul_1_output_0` + `/Slice_2_output_0` ABI 不变；新 pipeline 5 段总时长 ≤ 旧 3 段（NPU 抵消多段 dispatch） | 1-2 天 |
| M4 Vocoder front-half FP16 RKNN | `kokoro-vocoder-front-half.onnx`、`rk3588/kokoro-vocoder-front-half.fp16.rknn`、`kokoro-vocoder-tail-rest-cpu.onnx` | front-half 输出 max-abs-diff ≤0.03；50 真实输入 RKNN 无 None | 2 天 |
| M5 INT8 experiments (optional) | `bert.int8.rknn`（如可行）、`vocoder-front-half.int8.rknn` | 全 audio max-abs-diff ≤0.05 AND rel_l2 ≤0.01；否则保持 FP16 | 1 天 |
| M6 Manifest/profile integration | 追加 `rk_manifest.json:240-258` 后方的新 artifact 条目；更新 `rk3588-kokoro-rknn.json` profile | `abc.` RTF ≤0.673；profile env 序列化正常 | 半天 |

## §5 Failure rollback / partial-success paths

| 失败场景 | Fallback | NPU 占比目标 |
|---|---|---|
| BERT int8 掉点 | 保留 BERT FP16 | ~42%（BERT FP16 + decoder-front int8 + vocoder-front FP16） |
| BERT 整段失败（顽固 LayerNorm/GELU） | 只上 vocoder front-half；BERT 留 CPU | ~34%（17% decoder-front + 16.8% vocoder front）≈ 35% bucket |
| Vocoder front-half dilated conv 真机崩 | 进一步切至 resblocks.0-1 | ~30%（前半仅 ups.0+resblocks.0-1） |
| Vocoder front-half + BERT 双败 | 回滚到现有 decoder-front INT8 only | ~17-22%（现状） |

## §6 Acceptance criteria

**音质 parity**：全流水线 audio vs CPU 全图参考，max-abs-diff ≤0.05，rel_l2 ≤0.01。decoder-front 历史数据：mae=0.00528、max=0.0900、rel_l2=0.00227（`kokoro-rknn-analysis.md:83-90`）——新段应满足相同 rel_l2 门槛，单点 max 因相位敏感可适度放宽。

**性能**：`abc.` smoke RTF 不能超过 0.673；当前 int8 decoder-front 路径记录 RTF 0.655（`kokoro-rknn-analysis.md:186-195`）。长文 benchmark TBD，但必须包含 smallest-fit bucket 行为（fixed seq512 会生成 65.17s 超长 audio，扭曲 latency 评估，`kokoro-rknn-analysis.md:94-105`）。

**稳定性**：50 次连发 `synthesize("abc.")` + 50 次混合 SLV prompt，无 RKNN `None`，无 zero audio，无 process crash。

**内存/dispatch**：多段 RKNN dispatch overhead 可控——端到端 RTF 改善或持平；前一次 tail island 实验证明孤立段 overhead 可抹杀算力收益（`kokoro-rknn-analysis.md:197-212`）。

## §7 Future optional optimizations（本期 scope 外）

- Vocoder back-half chunked NPU inference：time-axis tiling + receptive field overlap-add + boundary de-clicking。现有文档警告当前固定 tail 不能直接切片，需新图 + 精确 Conv/ConvTranspose padding + InstanceNorm/AdaIN stats + source/noise 一致性处理（`kokoro-rknn-analysis.md:378-387`）。
- `generator-rest-preexp` RKNN productionization：当前已发现 RKNN 比 CPU 慢且有 REGTASK/fallback 警告，需先解 bug 再讨论生产化（`kokoro-rknn-analysis.md:236-254`）。
