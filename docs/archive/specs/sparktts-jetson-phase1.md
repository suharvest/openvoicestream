# SparkTTS-0.5B → Jetson (TensorRT-Edge-LLM) 适配 · Phase-1 Spec

状态：DRAFT · 2026-06-25 · 作者：CTO 线程
前置：BiCodec decoder 去风险 spike **PASS**（见 memory `sparktts_bicodec_decoder_trt_spike_2026_06_24`）

---

## 1. 背景与目标

SparkTTS（SparkAudio/Spark-TTS-0.5B，Apache-2.0）= **Qwen2.5-0.5B LLM backbone + BiCodec 单码本声学 codec**。
评估结论：backbone 与现有 Qwen3-TTS talker 同族零风险；最大风险（BiCodec vocoder 无官方 ONNX）已用 spike 证实可干净上 TRT fp16（ONNX vs PyTorch max_abs 9e-6，TRT fp16 vs PyTorch 0.023，无 plugin）。

**Phase-1 目标（一句话）**：在 orin-nx 上跑通 **controllable 模式**（无参考音频，gender/pitch/speed 标签控制音色）的 text→speech 全链路，产出**可懂、有声、ASR 可验**的中文+英文音频，并把它接进 voxedge 的 backend/leaf/profile 体系，达到"可在 dev 设备演示"的程度。

**非目标（明确排除，留后续）**：
- voice-clone（zero-shot）—— 拖入 wav2vec2 XLSR-53 + ECAPA-TDNN 重依赖，Phase-2。
- W8A16/int4 量化 —— Phase-1 只 fp16。（注意已知风险：Qwen3-TTS-CustomVoice 的 W8A16 talker EOS 破，SparkTTS 是否同病未知，留 Phase-2 单独验。）
- 流式低延迟首块优化 —— Phase-1 先做非流式正确性，流式列为 Phase-1b 拉伸目标。
- 生产部署（seeed-orin-nx）—— Phase-1 只在 dev orin-nx，**绝不碰 live arm 栈**。

设备：导出 = wsl2-local（x86_64+GPU）；构建+测试 = orin-nx（dev Jetson，非生产，语音栈可停）。

---

## 2. 架构拆解与数据流

controllable 模式下 LLM **自己生成 global+semantic token**（CoT 顺序：属性 token → global(32) → semantic(T)），**不需要任何参考音频分析链**（wav2vec2/ECAPA 只在 clone 时用于从参考音频反推 token）。

```
                       ┌─────────────────────────────────────────────┐
 text + attr labels →  │  ① Qwen2.5-0.5B LLM (autoregressive)         │
 (gender/pitch/speed)  │     vocab 扩展: +global(4096 FSQ) +semantic  │
                       │     (8192) +control tokens                   │
                       └───────────────┬─────────────────────────────┘
                                       │ token stream (CoT)
                          ┌────────────┴────────────┐
                          │ ② host-side token split  │  纯 Python/numpy
                          │  global_ids(32) ──────────┼──► FSQ codebook gather
                          │  semantic_ids(T) ─┐       │      → d_vector f32[1,1024]
                          └───────────────────┼───────┘   (绕开 einx.vmap，见 §4.3)
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       │  ③ BiCodec decoder (TRT fp16, 已 spike PASS) │
                       │     in: semantic_ids int64[1,T], d_vector    │
                       │     out: wav f32[1,1,T*320] @16kHz           │
                       └──────────────────────┬──────────────────────┘
                                              │ wav 16kHz
                                              ▼  (resample → 主链 24k/16k 按需)
```

三个组件的状态：
| # | 组件 | 状态 | Phase-1 工作量 |
|---|------|------|--------------|
| ① | Qwen2.5-0.5B LLM | 未做（核心决策见 §4） | 大头 |
| ② | host token split + FSQ→d_vector | spike 已验等价(max_abs 0) | 小，移植即可 |
| ③ | BiCodec decoder TRT engine | **spike PASS** | 小，工程化 |

decoder 边界（spike 已锁定，来自 `Spark-TTS/sparktts/models/bicodec.py:170 detokenize`）：
`DecoderWrapper(semantic_tokens int64[B,T], d_vector f32[B,1024]) → wav f32[B,1,L]`，
内部 = quantizer.detokenize(F.embedding) → prenet(feat_decoder Vocos/ConvNeXt) → +d_vector → wave_generator(Snake-GAN)。`remove_weight_norm` 在 load 时自动调用。

---

## 3. 核心架构决策：LLM 运行时怎么落地

**SparkTTS 不能直接复用 `qwen3_tts_worker`**：现有 `voxedge/backends/jetson/trt_edge_llm_tts.py` 围绕 explicit-KV talker **+ CodePredictor + Code2Wav** 三段构建；SparkTTS **没有 CodePredictor**（单码本，semantic 直接出自 LLM），vocoder 签名也不同 `(semantic, d_vector)→wav`。强塞会污染那条生产路径。

三条候选路径：

| 选项 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A. 复用 edge-llm LLM runtime** | SparkTTS LLM 走现有 TRT-Edge-LLM 的 Qwen LLM 引擎，token-streaming 模式吐 raw special-token id | 复用成熟 KV/decode | 现有 server 面向 text chat，需要 raw-token-id 出口；要确认能吐 >151k 扩展 vocab 的 id |
| **B. 新建 C++ worker** `spark_tts_worker` | 抄 qwen3 talker explicit-KV decode 机制 + BiCodec vocoder（去掉 CP） | 最贴现有基建、流式/perf 都在 C++ | C++ 工作量大，迭代慢 |
| **C. Python decode loop** | Python 侧驱动 LLM TRT 引擎逐 token 解码 + vocoder TRT | 快速验正确性、纯 Python 好调 | Python decode 慢，非生产 perf |

**建议（分阶段，先解耦"正确性"与"性能/基建"）**：

- **Phase-1a（正确性优先）**：LLM 用 **PyTorch/transformers 直接在 orin-nx 上跑**（0.5B fp16 完全跑得动），vocoder 用 spike 的 TRT engine。目标只回答一个问题：**"LLM 生成的 token → 我的 host-split → TRT vocoder" 这条线产出的音频可懂吗？** 这把"LLM 上 TRT"的风险**完全隔离**出去，先证 e2e 通路 + 音质。
- **Phase-1b（基建/性能）**：把 LLM 迁到 TRT-Edge-LLM。**优先试选项 A**（reuse edge-llm runtime，最省力），A 不通再退选项 C（Python TRT decode）做过渡。选项 B（C++ worker）推迟到性能/流式真正成为瓶颈、且方案 A/C 验证过模型正确性之后再投入。

> 决策理由：spike 已证 vocoder 没问题；现在最该先回答的是"controllable token 流喂进去音频对不对"，这与 LLM 跑在哪无关。用 PyTorch LLM 把这个问题一天内拍死，再谈把 LLM 塞进 TRT 的工程。

### 3.1 决策已定（2026-06-25）：Phase-1b 走 **Option A，且用标准 LLM runtime（非 qwen3OmniTTSRuntime）**

代码核查确认 **edge-llm 原生支持普通 Qwen2.5 文本 LLM**，Option A 比 Explore 初判干净得多：
- 未注册 model_type 回退通用 `CausalLM`（`TensorRT-Edge-LLM/tensorrt_edgellm/model.py:139` → `models/default/modeling_default.py:485`）。
- default decoder 参数化覆盖 Qwen2/Qwen3 两架构差异：`attention_bias`（Qwen2=True，`config.py:517` 从 config.json 读）、`has_qk_norm`（Qwen2=False，`config.py:450` 扫 safetensors 自动探测）；RoPE scaling 注释直接点名 Qwen2（`config.py:290`）。
- 标准 LLM 导出现成：`scripts/export.py:479` → `llm/model.onnx`（吐 `logits[B,T,vocab]`）。
- **独立通用 LLM 运行时 `cpp/runtime/llmInferenceRuntime.cpp/.h`**（即跑 Qwen3.5-4B chat 那条），自回归 decode 吐 token id。

**关键纠正**：SparkTTS talker = 普通 Qwen2.5 LLM，**无需 CodePredictor**（单码本，Phase-1a 已证），**根本不用碰 `qwen3OmniTTSRuntime`/CP**。Explore 标红的"改 C++ runtime 跳过 CP"风险不存在。

**Phase-1b 路径（锁定）**：
```
SparkTTS Qwen2.5-0.5B(扩展vocab) → 标准 edge-llm 导出 → llm engine
  → llmInferenceRuntime 自回归 decode 吐 token（现成 runtime）
  → 正则/映射 bicodec_global(32)+semantic(T)   ← Phase-1a 已证契约
  → host-split FSQ→d_vector → BiCodec TRT vocoder(已建)  → wav
```
bicodec token 本就在 tokenizer vocab 内，decode 成文本再正则即可（官方 `SparkTTS.py:217-229` 就这么做），新写 C++ 几乎为零，剩工程串接 + voxedge 集成。
（选项 B/C 退役为兜底：若 llmInferenceRuntime 的 token-id 出口或扩展 vocab/EOS 有意外，再退 Python decode 过渡。）

---

## 4. 工作分解（Phase-1a → 1b）

### Phase-1a：e2e 正确性（orin-nx，LLM=PyTorch + vocoder=TRT）

1. **vocoder 工程化**（基于 spike 产物）
   - 导出脚本固化到 `jetson-voice-engine/scripts/export_sparktts_bicodec_decoder.py`（WSL 跑）：
     - 输入边界 = §2 的 DecoderWrapper；`remove_weight_norm` 走 checkpoint 默认。
     - **必须带 dynamic T 的 shape profile**（`--minShapes/optShapes/maxShapes` 给 semantic_tokens 与输出 L），否则 trtexec 把 T 冻成 1（spike 实证坑）。建议 T ∈ [min 8, opt 200, max 600]（600≈12s@50TPS）。
   - orin-nx 上 trtexec build fp16 engine + sidecar（config.json 含采样率/upsample/codebook 维度）。
   - 产物上传 HF artifacts repo（参考 `harvestsu/qwen3-edgellm-jetson-artifacts` 模式，新建 `sparktts/` 前缀目录）。
2. **host token split + FSQ→d_vector**（§4.3）固化成 `voxedge/backends/jetson/_sparktts_util.py`（纯 numpy/torch，无 CUDA，import 干净）。
3. **参考实现验证脚本** `bench/sparktts/e2e_pytorch_ref.py`（orin-nx）：
   - SparkTTS 官方 PyTorch 全链路跑一组固定文本（中/英各 ≥5 句，含 controllable 三档 pitch/speed）→ 存参考 wav + 中间 token。
   - 我方链路（PyTorch LLM → host-split → **TRT vocoder**）跑同输入 → 比对。
4. **验收 1a**：见 §5。

### Phase-1b：LLM 上 TRT + voxedge 集成

5. **LLM → TRT-Edge-LLM**（选项 A 优先）：
   - 确认 edge-llm 能加载 SparkTTS LLM（扩展 vocab 的 Qwen2.5-0.5B）并以 raw-token-id 模式解码（贪婪/带采样）。
   - 失败则退选项 C（Python TRT decode loop）。
   - 比对：TRT-LLM token 流 vs PyTorch token 流（理想贪婪逐 token 一致；fp16 采样下比 e2e 音频可懂度）。
6. **voxedge backend**：新建 `voxedge/backends/jetson/sparktts_trt.py`
   - `@dataclass SparkTTSConfig`（**无任何模块作用域 os.environ.get**，全字段构造注入 —— 见 `trt_edge_llm_tts.py:1-90` 的硬规矩 + memory `trt_edge_llm_tts_env_staleness`）。
   - 实现 `TTSBackend` 契约（capability: BASIC_TTS + MULTI_LANGUAGE；STREAMING 留 1b 拉伸）。
   - in-process TRT（参考 `matcha_trt.py`/`kokoro_trt.py` 的延迟 import onnxruntime/tensorrt 模式）；若走 C++ worker 才复用 `worker_io.py` 的 JSON-line + WorkerIO。
7. **leaf 注册**：`configs/leaves/sparktts-0p5b.yaml`（N=1 起步；shared sub-leaf 放 vocoder/tokenizer，参考 `qwen3-tts-nx.yaml` 结构）+ `models.yaml` 加逻辑模型 `sparktts-0p5b`，`default_precision.jetson: fp16`（precision 是 leaf 属性，翻一行即换 §见 leaf_config_refactor memory）。
8. **profile + config builder**：`configs/profiles/jetson-sparktts-0p5b-nx.json`（填 `SPARKTTS_*` env + `required_engines`）+ `server/core/voxedge_backend_config.py::build_sparktts_trt_config(profile, env)`（env→Config，所有字段显式传）。
9. **验收 1b**：见 §5。

### 4.3 host-side global-FSQ 处理（spike 已定方案）

global token 的 FSQ 查表（`get_codes_from_indices` 的 `einx.get_at`→`torch.vmap`）**导不进 ONNX**（RuntimeError: Unsupported value kind: Tensor）。spike 已验绕法：**留在 host 侧**用原始 `speaker_encoder.detokenize`（一次性 32-token gather）算出 `d_vector f32[1,1024]` 再喂 vocoder —— split 边界 vs 原始 `detokenize` max_abs=0.0 完全等价。Phase-1 直接移植该函数到 `_sparktts_util.py`；可选优化（Phase-2）= 重写成 plain `F.embedding` 彻底去 einx 依赖。
**注意 d_vector 是 1024 维**（非常被误传的 256）。

---

## 5. 验收标准（PASS/FAIL，必须带原始数字）

**Phase-1a PASS（全部满足）**：
- [ ] TRT vocoder vs PyTorch vocoder：同 token 输入 max_abs ≤ 0.05（spike 已 0.023，工程化后不应退化）。
- [ ] e2e（PyTorch LLM → TRT vocoder）中文 ≥5 句、英文 ≥5 句：输出 wav **非零能量**（RMS > 0.01）且 **faster-whisper 回转可懂**（CER 中文 ≤0.10 / EN WER ≤0.15，对照输入文本）。**禁用过期 Groq 做参考**（memory 教训）。
- [ ] controllable 三档（low/mid/high pitch 或 speed）产出音频**可听出差异**（基频/时长统计 + 人耳 spot-check）。

**Phase-1b PASS（全部满足）**：
- [ ] LLM-on-TRT token 流相对 PyTorch：贪婪解码逐 token 一致，或采样下 e2e 音频满足 1a 的可懂度阈值。
- [ ] 经 voxedge backend（leaf+profile 解析、非裸脚本）跑通同样 ≥10 句，CER/WER 达标。
- [ ] 显存：orin-nx 上 LLM+vocoder 常驻 `peak_unified_mb` 实测（**真机测，不信估算**），≤ orin-nx 预算且与现有栈共存不 OOM。
- [ ] 单 utterance 端到端延迟（非流式）实测记录（baseline，不设硬阈值）。
- [ ] backend 导入在无 CUDA 机器（Mac/CI）干净（onnxruntime/tensorrt 延迟 import）。

每个 gate 报告必含：原始命令输出、产物 md5、faster-whisper 转写原文、before/after。

---

## 6. 风险与未决问题

| 风险 | 等级 | 缓解 |
|------|------|------|
| 选项 A（edge-llm 吐 raw token id）不支持扩展 vocab / 非文本输出 | 中 | Phase-1a 用 PyTorch LLM 先解耦；A 不通退选项 C |
| SparkTTS LLM 扩展 vocab embedding 在 edge-llm export 的兼容性 | 中 | 先小步验 LLM 单独导出/解码，再接 vocoder |
| controllable 模式 token 契约（attr/global/semantic 边界、特殊 token id）解析错 | 中 | 严格对照官方 PyTorch 推理代码的 token 拼装；§4.3 host-split 用官方函数 |
| W8A16 talker EOS（Qwen3-TTS 踩过） | 低(Phase-1 fp16) | Phase-2 单独验，不进 Phase-1 |
| 流式 vocoder（conv 需 stateful 滑窗） | 低(1b 拉伸) | 非流式先过；流式参考 qwen3 stateful code2wav 静态化套路（但 BiCodec 无 ScatterND/ISTFT/RVQ 更简单） |
| 中文支持质量（SparkTTS 训练以中英为主，验） | 中 | 验收强制中英双语 CER/WER |

未决问题（实施前需答）：
1. SparkTTS LLM checkpoint 的 tokenizer/special-token 映射在哪、controllable prompt 模板的确切格式？（读官方 `cli/inference.py` / `sparktts/models/audio_tokenizer.py`）
2. edge-llm 现有 Qwen LLM 引擎能否配置成 raw-token-id 输出？（问 jetson-voice-engine 维护面）
3. SparkTTS 多音色是靠 controllable 属性组合，还是需要预置 speaker prompt？Phase-1 选哪几个"演示音色"？

---

## 7. 交付物清单

- `jetson-voice-engine/scripts/export_sparktts_bicodec_decoder.py`（+ LLM 导出脚本，1b）
- `voxedge/backends/jetson/sparktts_trt.py` + `_sparktts_util.py`
- `configs/leaves/sparktts-0p5b.yaml` + `models.yaml` 条目
- `configs/profiles/jetson-sparktts-0p5b-nx.json`
- `server/core/voxedge_backend_config.py::build_sparktts_trt_config`
- `bench/sparktts/e2e_pytorch_ref.py` + 验收报告（1a / 1b 各一）
- HF artifacts：`sparktts/` 引擎 + sidecar

---

## 8. 执行计划（派发顺序）

1. **派执行体 · Phase-1a**：vocoder 工程化导出 + host-split 移植 + PyTorch-LLM e2e 参考验证（orin-nx + WSL）。带 §硬护栏（禁破坏性操作、只 dev 设备、EVIDENCE 段）。
2. 主线程评审 1a 数字 → 决定 LLM 运行时（选项 A/C）。
3. **派执行体 · Phase-1b**：LLM 上 TRT + voxedge 全集成 + leaf/profile + e2e gate。
4. 主线程自验关键数字（CER/WER/显存）→ 决定是否进 Phase-2（clone / W8A16 / 流式）。
