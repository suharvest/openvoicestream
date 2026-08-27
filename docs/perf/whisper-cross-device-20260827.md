# Whisper 边缘平台横向实测（2026-08-27）

在 Hailo-8、RK3588、RK3576、Jetson Orin Nano、树莓派 5 CPU 五个平台上实测 OpenAI Whisper，回答两个产品问题：**对话场景**（短句、看延迟）和**转录场景**（长音频、看吞吐与精度）分别该选哪块板、跑什么配置。

## 结论摘要

**两家 NPU 厂商的官方默认配置，在各自硬件上都是最差选项之一。** Hailo 和 Rockchip 都把 Whisper 拆成 encoder + decoder 两张图全部放 NPU，而两家的 **NPU decoder 都没有 KV cache**。Whisper 的 decoder 是自回归的，没有 cache 就每步重算整个序列。encoder 上 NPU 是真收益，decoder 上 NPU 是净损失。

把 decoder 换成 CPU 上带 KV cache 的 ONNX 图之后，**每块板都是又快又准**：RK3588 英文长句 WER 10.44% → **7.58%**，RTF 0.149 → **0.061**。

**窗口长度是对话/转录之间的取舍杠杆，而且是可配置的。** 同一块 RK3588、同一个模型、同一套代码，只把 encoder 窗口从 20 秒换成 10 秒：TTFT 297 ms → **124 ms**（2.4×），短句 WER 13.37% → **11.37%**；代价是长音频必须切块，长句 WER 7.58% → 11.40%。

**中文换板子救不了。** 全场最好的中文成绩是 15.99% CER，而同一批语料上 RK3588 已部署的 Paraformer 是 **2.6%**。这是 Whisper 自身的水位，不是硬件差距。

**TensorRT 引擎的数值正确性不能凭精度标志推断。** whisper-base 的 encoder 用 `--fp16` 建出的引擎，输出与 onnxruntime 的 cosine 只有 **0.8104**，而同样 fp16 的 tiny encoder 和两个 decoder 引擎都正常。坏引擎不会报错，只会让下游解出"通顺但不对"的文本——排查时极易误判成 KV cache 问题。**每个引擎都要对着 onnxruntime 做数值对拍。**

---

## 方法

**语料**：仓库自带的 `bench/perf/corpus`，20 条真人录音（Google FLEURS，CC BY 4.0），中英各 5 条短句 + 5 条长句，sha256 钉死，各设备消费完全相同的字节。

**评分**：逐行复刻 `bench/perf/runners.py` —— cn2an 中文数字归一 + 同一张标点表 + jiwer（中文 CER / 英文 WER）。**设备只吐转录文本和分段计时，打分统一在一台机器上做一次**，避免每台设备各装一套依赖导致口径漂移。

**指标**：

- **RTF** = 推理耗时 ÷ 音频时长（离线口径）
- **TTFT** = encoder 耗时 + 第一个 token 的耗时
- **t2s** = 简繁归一后的 CER。Whisper 输出繁体，而 fleet 里其他 ASR 后端都输出简体；不折进主列，单独给。

> **两个口径不能混用。** 本文的 RTF 是离线的「推理耗时 ÷ 时长」；`docs/performance-comparison.md` 里的 **Finalize RTF** 是「说话人停止后」经流式服务测的，两者不是同一个量。TTFT 在既有矩阵里没有对应项。

---

## 实测结果

每组 5 条，warmup 后取均值。

| 板子 | 配置 | en 短 | en 长 | zh 短 (t2s) | zh 长 (t2s) | TTFT | RTF(长) |
|---|---|---|---|---|---|---|---|
| **Hailo-8 + Pi5** | tiny / 10s / hybrid | **10.95%** | 21.58% | 52.98 (50.31) | 58.13 (50.22) | 60 ms\*\*\* | 0.029 |
| Pi5 纯 CPU | tiny / 10s / ONNX | **10.16%** | 23.73% | 50.30 (47.23) | 57.74 (36.94) | 156 ms | 0.043 |
| **RK3588** | base / 20s / **hybrid** | 13.37% | **7.58%** | 52.00 (44.94) | 32.32 (**19.63**) | 301 ms | 0.061 |
| RK3588 | base / 20s / 全 NPU（厂商默认） | 15.37% | 10.44% | 51.09 (44.04) | 29.83 (19.78) | 318 ms | 0.149 |
| **RK3588** | base / **10s** / hybrid | **11.37%** | 11.40% | 55.32 (40.63) | 48.77 (37.72) | **124 ms** | 0.072 |
| **RK3576** | base / 20s / **hybrid** | 17.81%\*\* | **7.58%** | 44.94 | 32.32 (20.32) | 366 ms | 0.104 |
| **RK3576** | base / **10s** / hybrid | **11.37%** | 10.40% | 52.59 (41.54) | 58.24 (42.97) | 149 ms | 0.108 |
| RK3576 | base / 20s / 全 NPU（厂商默认） | 15.37% | 10.44% | 41.54 | 29.83 (19.78) | 294 ms | 0.146 |
| **Orin Nano** | base / 30s / **裸 TensorRT (bf16)** | **11.37%** | 9.19% | 57.03 (45.97) | 31.47 (16.71) | **18 ms** | **0.008** |
| Orin Nano | base / 30s / whisper.cpp CUDA | 13.59% | 8.59% | 58.62 (46.66) | 30.75 (**15.99**) | 216 ms\* | 0.023 |
| Orin Nano | tiny / 30s / whisper.cpp CUDA | **7.30%** | 12.26% | 48.75 (**39.49**) | 37.21 (27.15) | 196 ms\* | **0.019** |

\*\* RK3576 的 en 短句 17.81% 与 RK3588 的 13.37% 之差，**全部来自 5 条里的 1 条**：`en_short_03` 上 RK3576 出 `Erasmith had`、RK3588 出 `Aerosmith have`，两词之差使该条 err 0.333 vs 0.111，摊到 5 条均值就是 4.4 点。长句两板 **7.58% 逐位相同**。这是 fp16 数值在两颗芯片上的微小差异被贪婪解码在近似打平处放大的结果，**不是能力差异**。

\*\*\* Hailo 这行是 mel 前端统一为 numpy 版、warmup=5 后重测的。**此前发布的 38.7 ms TTFT 是低估**：当时首 token 记为 15.0 ms，而加大 warmup 后它稳定在 28–48 ms——不是预热不足，15 ms 那个值本身是异常。机理上 prefill 要处理 4 个 prompt token + 完整 encoder 输出并算出全部 cross-attention KV，30 ms 量级才合理。

\* Jetson 的 TTFT 是**代理值**：whisper.cpp 不暴露首 token 时刻，取 encode + 一次 sample。其余各行的 TTFT 是逐 token 实测。两者不可直接并排比较。

### 按场景选型

| 场景 | 选择 | 依据 |
|---|---|---|
| **对话 / 低延迟** | 延迟优先 → Orin Nano + TensorRT(bf16)；成本优先 → Hailo-8 + Pi5 | TTFT **18 ms** vs **60 ms**，Jetson 快 3.2 倍。短句精度两者同档（11.37% vs 10.95%）。Hailo 的价值在成本低一个量级而 60 ms 对对话仍够用 |
| **转录 / 吞吐 + 中文** | Orin Nano + 裸 TensorRT(bf16) | RTF **0.008**，一小时音频约 29 秒跑完；中文长句 16.71%（whisper.cpp 为 15.99%，同档） |
| **英文长句 / 性价比** | RK3588 hybrid | en 长句 **7.58%**，与 Jetson 的 9.19% 同档而硬件成本低得多 |

---

## 三个跨平台机制性发现

### 1. NPU decoder 没有 KV cache，两家表现形式不同

| 平台 | NPU decoder 形态 | 后果 |
|---|---|---|
| Hailo-8 | 固定 32 序列，窗内全注意力 | 超过 28 个新 token **硬截断**，中文长句从句中断开 |
| RK3588 / RK3576 | **12 槽滑动窗口**，长度不限 | decoder 只看得到最近约 8 个 token，长句失去上下文 |
| Jetson（GGML / TRT） | 真 KV cache | 无此问题 |

实测每 token 耗时（英文长句均值）：

```
Hailo-8   HEF decoder      680 ms/句   （42 ms/token，无 KV cache）
Hailo-8   CPU KV-cache     159 ms/句   （首 token ~30ms，后续 ~8 ms/token）
RK3588    RKNN decoder    1391 ms/句
RK3588    CPU KV-cache     426 ms/句
Orin Nano CUDA            134 ms/句
```

encoder 两家都不慢（Hailo 23.8 ms、RK 250 ms），问题全在自回归解码。

### 2. 窗口是产品配置项，不是厂商属性

各家出厂窗口差 6 倍：Hailo tiny **10s**（有效 9s，mel 前端裁掉 1s 防边界幻觉）、Hailo base **5s**、Rockchip **20s**、whisper.cpp **30s**（Whisper 原生）。

原以为这是各平台的固有属性，实测证明**可以自己选**。同一块 RK3588 上：

| | 10s 窗口 | 20s 窗口 |
|---|---|---|
| en 短句 | **11.37%** | 13.37% |
| **TTFT** | **124 ms** | 301 ms |
| en 长句 | 11.40% | **7.58%** |
| zh 长句 (t2s) | 37.72% | **19.63%** |

短句这半是纯赚：**TTFT 快 2.4 倍，精度还略好**。代价全在长音频——窗口一旦小于最长音频，就必须切块。

**所以窗口的语义是「窗口 + 是否需要分段」这一对，不能只暴露前者。** Hailo 的 9 秒有效窗口不是缺陷，是把这个取舍钉死在了对话那一端；RK 的 20 秒钉在转录那一端。

### 3. 内容不足的音频会让 Whisper 不吐 EOS

这个失败模式在三条不同路径上各出现一次，与硬件厂商无关：

| 触发路径 | 现象 | 之前为何没暴露 |
|---|---|---|
| Hailo：短语音补零填满 10s 固定窗口 | 复读并胡编，en 短句 WER 10.95% → **70.16%** | 写死的 `cap=32` 一直在兜底 |
| RK 10s：切块后尾块内容不足 | `by Llew, by Llew, ...` 一路到位置表上限 | 20s 窗口时全部语料单窗装下 |
| RK 10s：尾块几乎全静音 | 多吐 `(dramatic music)` / `[silence]` | 同上 |

**Whisper 的 decoder 位置表只有 448 项**，超出不是精度下降而是直接崩（onnxruntime 报 `idx=448 out of data bounds`）。写死的 token 上限看着是长度限制，实际同时在替 EOS 兜底——**官方 demo 的默认配置一直掩盖着这个问题**。

因此这套 harness 把三个防护做成默认，而不是某个平台的补丁：

1. **按时长定 token 预算** `min(cap, 时长×8+12)`——固定上限只防崩溃，不防失控
2. **相似度复读清理**——Hailo 官方的 `clean_transcription` 只判子串包含，抓不到自我改述（`...from the plant.` / `...more than a plan for...`）；且只切 `.` `?`，中文无句读直接漏过
3. **句内循环守卫**——按 n-gram 检测重复 ≥3 次的短语并截断，句级去重看不见句内循环

---

## 官方精度天花板（对照）

语料取自 FLEURS，而 Whisper 论文正是在 FLEURS 上报告逐语言精度，可直接对照（arXiv 2212.04356 附录 D.2.4 表 13）。论文自述：*"we put a space between every letter for the languages that do not use spaces to separate words, namely Chinese, Japanese, Thai, Lao, and Burmese, effectively measuring the character error rate instead"* —— 所以中文那列虽然表头写 WER，实际是 CER，与本文口径一致。

| 模型 | 英文 WER | 中文 CER | Hailo-8 有无 | Rockchip 有无 |
|---|---|---|---|---|
| tiny | 12.4% | **40.5%** | ✅ 10s 窗口 | ❌ |
| base | 8.9% | **34.1%** | ✅ 5s 窗口 | ✅ 20s 窗口 |
| small | 6.1% | 20.8% | ❌ | ❌ |
| large-v2 | 4.2% | 14.7% | ❌ | ❌ |

我们实测的纯 CPU 基线（fp32 encoder，无 NPU，无失控）英文 10.16% / 中文 50.30%，对照官方全集的 12.4% / 40.5%——每组仅 5 条 vs 全测试集，属方向性吻合，足以说明**中文 40~50% 是 Whisper-tiny 的设计水位，不是量化或硬件问题**。

`small`（20.8%）才是中文勉强可用的第一档，而 Hailo-8 和 Rockchip 都没有 small 的构建。即便 large-v2 的 14.7% 也仍差于 RK3588 上已部署的 Paraformer（2.6%）。

**结论：语言路由应该按语言分流，而不是按板子分流。** 英文走 Whisper，中文走各平台已有的中文原生模型（Paraformer / SenseVoice / Qwen3-ASR）。

---

## Jetson：TensorRT vs TensorRT-Edge-LLM

评估在 Jetson 上跑 Whisper 该用哪条栈。

**结论：走裸 TensorRT，不要塞进 edge-llm，也不要用 ONNX Runtime 的 TensorRT EP。**

| | edge-llm | ORT + TRT EP | 裸 TensorRT |
|---|---|---|---|
| 架构匹配 | ❌ 全仓**零处** cross-attention；ASR 走 prefix 注入 + 逐层 KV cache，是 decoder-only LLM 语义 | ✅ | ✅ cross-attention 在 TRT 里就是普通算子 |
| 依赖 | 已有 | ❌ 依赖重，按子图切分回落 CUDA EP | ✅ `python3-libnvinfer` + `trtexec` + `cuda-python` **设备上已有，零新增** |
| 改动面 | 要在 NVIDIA 上游 runtime + `llm_build` 里实现 cross-attn，叠在现有 7 上游 + 35 本地补丁之上 | 低 | 需自写 KV cache 显存管理 |

Qwen3-ASR 的 thinker 是把音频当**前缀 token** 喂进 decoder-only LLM；Whisper 的 decoder 每层都要 attend 到 encoder 输出。**这不是加个模型，是给 runtime 加一类注意力。** edge-llm 的 slot pool / 流式 worker / N=2 应该花在 Qwen3-ASR（中文 5.3%）上，而不是花在中文不可用的 Whisper 上。

### ⚠️ base 的 fp16 encoder 引擎数值是坏的，解法是 bf16

**`trtexec --fp16` 从 whisper-base 的 encoder ONNX 构建出的引擎，输出与 onnxruntime 参照的 cosine 只有 0.826**。TRT 输出是 run-to-run 确定的（maxdiff 0.0），所以不是竞态或陈旧缓冲区，是 fp16 kernel 选型的精度问题——fp16 只有 5 位指数，而 Whisper encoder 的残差累加和 attention softmax 分母都是高动态范围位置。（我们在 SparkTTS 上踩过同类坑：down_proj 输出某通道到 ~230k，远超 fp16 的 65504 上限。）

**bf16 解决了它**，且几乎不付出速度代价——bf16 的指数位和 fp32 一样是 8 位，只是尾数从 10 位降到 7 位：

| 构建 | cosine vs onnxruntime | encoder 延迟 | 端到端精度 |
|---|---|---|---|
| `--fp16` | **0.826** | 10.75 ms | ❌ 内容漂移 |
| **`--bf16`** | **0.9996** | **12.53 ms** | ✅ **与 fp32 逐位相同** |
| `--fp16 --bf16` | **0.826** | 10.46 ms | ❌ 见下 |
| （无 flag，fp32） | 1.000000 | 39.1 ms | ✅ |

bf16 相对 fp32 **拿回 3.1× encoder 速度**（39.1 → 12.53 ms），端到端 TTFT 46 → **18 ms**，而 en/zh 四组的 err 与 fp32 **完全相同**。

**⚠️ 不能同时给 `--fp16 --bf16`。** 两个 flag 一起给时，TRT 全程选了 fp16 kernel，产出的引擎与纯 fp16 **逐位相同**（cosine、maxabsdiff、std 三个值一模一样），等于没配 bf16。这与直觉相反——一般会以为多给几个精度选项让 TRT 自选更优。**只给 `--bf16`。**

**这个缺陷极具欺骗性**：坏 encoder 的输出仍是一个「看起来正常」的张量（均值/方差量级都对），decoder 会照常贪心解出语法通顺的英文，只是内容漂移、提前 EOT——外观上和「KV cache 没累积」一模一样。最初的排查方向因此被带偏，直到把同一份 encoder 输出分别喂给 onnxruntime 和 TRT 的 decoder、发现**逐 token argmax 完全一致**，才把嫌疑从 decoder 转到 encoder。

**是模型特异的**：同样用 `--fp16`，`enc_tiny_30s`（cosine 0.999864）和 `enc_tiny_10s`（0.999937）以及两个 decoder 引擎都正常。所以既不能说「fp16 一律没事」，也不能说「fp16 一律不能用」——**每个引擎都要对着 onnxruntime 做一次数值对拍**，这应该是 TRT 构建流程里的常设步骤（脚本见 `bench/perf/whisper/`）。

### TensorRT encoder 微基准（数值已验证的引擎）

`trtexec`，端到端延迟（含 H2D/D2H，两者合计 < 0.3 ms）：

| encoder | 精度 | TensorRT | whisper.cpp GGML CUDA | 倍数 |
|---|---|---|---|---|
| base / 30s | **bf16**（fp16 不可用） | **12.53 ms** | 124.3 ms | **9.9×** |
| tiny / 30s | fp16 | **5.29 ms** | 103.7 ms | **19.6×** |
| tiny / 10s | fp16 | **1.61 ms** | —（Hailo-8 HEF 为 23.8 ms） | 14.8× vs Hailo |

base 用 bf16 后 **9.9×**，tiny 用 fp16 **19.6×**。**注意 whisper.cpp 默认已开 flash-attention**（`cli.cpp:79` 默认 `true`），所以这不是漏开优化造成的差距。

### decoder 是带宽瓶颈，不是算力瓶颈

whisper-base 每 token 约 113 MFLOP（6 层 × ~10 MFLOP + 词表投影 512×51865×2 ≈ 53 MFLOP）。Orin Nano 按 8 TFLOPS 算，理论 **0.014 ms**。

但每生成一个 token 必须把 decoder 权重读一遍：40M 参数 × fp16 = **80 MB**，Orin Nano 是 8GB LPDDR5 / 68 GB/s → **1.2 ms/token 的带宽硬下限**。

| | 每 token |
|---|---|
| 算力下限 | 0.014 ms |
| **显存带宽下限** | **1.2 ms** |
| TRT 实测（`trtexec`，KV=16，encoder 1500 帧） | **2.63 ms** |
| whisper.cpp GGML CUDA 实测 | ~5 ms |

实测落在带宽下限的 2.2 倍，离算力下限差 188 倍。这解释了一个反直觉的观察：**Pi5 CPU 的 decoder 是 8 ms/token，Jetson GPU 是 5–6 ms/token——算力差几十倍，decoder 只差 1.4 倍**，因为两边都在等内存。

所以 TRT 的收益在两端极不对称：**encoder 白捡一个数量级，decoder 只有约 2 倍且已贴近硬下限**。要再压 decoder 只有三条路：量化（int8 → 0.6 ms 下限）、批处理（权重读一次服务多请求）、CUDA Graph（打掉 launch 开销）。

### 完整 TRT 管线实测

三个引擎（encoder / prefill / cached-step），架构参照 `Jonah-May-OSS/wyoming-whisper-trt`，而 optimum 的两张图天然对应该拆分。cross-attention KV 由 prefill 产出后直接绑 device 指针给 step 引擎；self-attention KV 用每层两块显存 ping-pong，全程不回 host。

| Orin Nano base/30s | 裸 TensorRT | whisper.cpp CUDA |
|---|---|---|
| en 短句 | **11.37%** | 13.59% |
| en 长句 | 9.19% | 8.59% |
| zh 长句 (t2s) | 16.71% | 15.99% |
| **TTFT** | **18 ms**（实测） | 167–285 ms（代理值） |
| **RTF（长）** | **0.008** | 0.023 |
| encoder | **12.5 ms** | 124 ms |
| decoder | 40–132 ms | 77–218 ms |

精度两边在 5 条样本的噪声内持平，但 **TTFT 快 9–16 倍、RTF 快 3 倍**，且 TRT 的 TTFT 是逐 token 实测而非代理值。**18 ms 已明显优于 Hailo-8 的 60 ms**，而且这是在 30 秒窗口下拿到的——换 10 秒窗口（`enc_tiny_10s` fp16 已验证数值可用、1.61 ms）还会更低，但**未测，不写进结论**。

## 各平台可用的 Whisper

| 平台 | 变体 | 窗口 | 来源 |
|---|---|---|---|
| Hailo-8 / 8L / 10H | tiny、base（tiny.en 仅 10H） | tiny **10s** / base **5s** | Hailo 官方 `Hailo-Application-Code-Examples/runtime/python/speech_recognition/app/download_resources.py`。**注意不是** `ktomanek/edge_whisper` 里那个 downloader——它的 `--hw-arch hailo8` 是空选项，`FILES` 字典里只有 8L 和 10H |
| RK3562/3566/3568/**3576**/**3588**/RV1126B | base | **20s** | `airockchip/rknn_model_zoo/examples/whisper`，原生带 `--task en\|zh`。官方 usage 只列 `fp`，不列 i8 |
| Jetson | 任意 | 30s（whisper.cpp）/ 任意（自建 TRT） | 无官方 TRT 方案。**PyPI 上 arm64 的 CTranslate2 是 CPU-only**（`get_cuda_device_count()` 返回 0） |

**TTS：Hailo 官方明确零支持。** 员工原话 "Hailo currently doesn't support any TTS models"，且 GenAI Model Zoo 的 12 个模型里无任何 TTS 条目。树莓派 + Hailo 上 TTS 只能跑 CPU。

---

## 复现

设备侧 runner 与打分脚本在 `bench/perf/whisper/`：

```bash
# RKNN（RK3588 / RK3576）：--encoder_duration 必须与 .rknn 编译时的窗口一致
python3 rknn_whisper_run.py --corpus corpus --lang en \
  --encoder model/whisper_encoder_base_20s.rknn --decoder onnx_dec \
  --vocab-dir model --encoder_duration 20 --all-cores \
  --label rk3588-hybrid-en --out results/hybrid_en.json

# whisper.cpp CUDA（Jetson）：不要传 -np，它会关掉 whisper_print_timings
python3 wcpp_corpus_run.py --corpus corpus --lang en \
  --bin ./whisper.cpp/build/bin/whisper-cli --model models/ggml-base.bin \
  --label orin-nano-wcpp-base-en --out results/wcpp_base_en.json

# 打分（在一台机器上统一做）
python3 score_all.py 'results/*.json'
```

模型转换：

```bash
# RKNN：必须在 x86 上转，且 toolkit 版本要对齐设备 runtime（两块板都是 rknnlite 2.3.0）
python convert.py whisper_encoder_base_20s.onnx rk3588 fp out.rknn

# TensorRT
trtexec --onnx=encoder.onnx --fp16 --shapes=input_features:1x80x3000 --saveEngine=enc.plan
```

---

## 踩坑清单

### 布局与形状

**导出的 ONNX 输入 rank 必须和运行时喂的一致，不匹配时不会报错。** 为 Hailo 导出的 encoder 是 4D NCHW `[1,80,1,1000]`，Rockchip 官方的是 3D `[1,80,2000]`。两者元素总数相同（80×1000）时，ONNX 转换阶段合法、**rknn-lite 运行时也不报错**，它会按 4D 重新解释缓冲区。唯一症状是 decoder 吐 `(chiming)` / `(chewing)` 这类非语音标注。

这与 RK matcha vocos 那次是同一类失败（尺寸不匹配但字节数对得上 → 静默重解释，−22 dB 无声无息）。**这类 bug 没有任何错误信息，只能靠端到端语义验证发现。** 导出脚本已加 `--input_rank {3,4}` 显式区分。

隔离方法：先在开发机上用 onnxruntime 跑同一份 ONNX 验证转录正确，再花时间做平台转换。

### 量化

**`quantize_dynamic` 产出的是 ORT 专用格式，TensorRT 拒收。** 报 `checkDynamicQuantizeLinear` / `checkMatMulInteger`。我们给 CPU decoder 用的那份 int8 ONNX 无法直接喂给 TRT；要做 int8 TRT decoder 得走显式 QDQ + 校准数据集，是完全不同的导出路线。

### 词表与解码

**Rockchip 的 `read_vocab` 按第一个空格切**（不是最后一个），**`base64_decode` 必须用官方那个手写版**——它遇到 `=` 直接返回单个空格，中文 vocab 靠这个编码词边界，换 `base64.b64decode` 语义就变了。

**官方 `base64_decode` 有一个我们修掉的缺陷**：`bytearray(len//4*3)` 按上界分配却整个返回，短解码尾部带 `\x00`。打印到终端看不见，进评分会被当成插入错误。需要 `out[:oi]`。

### 硬件

**RK3588 开三核 `NPU_CORE_0_1_2` 没有收益**（encoder 260 ms vs RK3576 双核 246 ms，反而略慢）→ 这个 encoder 不是 NPU 算力瓶颈。

**RK3588 与 RK3576 在此负载上等价**：英文输出**逐字节相同**，中文 10 条里 3 条有细微差异（简繁选择、一个逗号）。zh 短句那个 41.54% vs 51.09% 的差距几乎全来自单条 `zh_short_05` 的简繁翻转，**不能解读成「RK3576 中文更好」**。在赢家配置里 decoder 跑在 CPU，所以两块板的差距（426 ms vs 786 ms）**全部来自 CPU（A76 vs A72），与 NPU 无关**。

**Hailo-8 单进程独占 `/dev/hailo0`**，别的进程占着就报 `HAILO_OUT_OF_PHYSICAL_DEVICES (74)`；**同一进程内建两个 VDevice 也会撞**，所以中英文必须分进程跑。

### TensorRT

**不要凭 `--fp16` 标志推断数值正确性。** whisper-base 的 encoder 用 fp16 建出的引擎 cosine 只有 0.8104，而同样 fp16 的 tiny encoder 和两个 decoder 都是好的。**每个引擎都要对着 onnxruntime 做一次数值对拍**。这类缺陷不会报错，只会让下游产出"通顺但不对"的结果。

**`quantize_dynamic` 的 int8 ONNX 喂不进 TRT**（见上文「量化」）。

### Hailo harness 退出时 segfault

十条语料全部跑完、结果 JSON 正常写出之后，进程以 **rc=139（SIGSEGV）** 退出。是 Hailo VDevice 在 `release()` 阶段的问题，**不影响已产出的数据**，但会让 shell 的 `&&` 链断掉。脚本里要用 `; echo rc=$?` 而不是 `&&` 串联。

### TTFT 口径对 prefill 实现敏感

TTFT = encoder + 首 token。两阶段 decoder（prefill + cached step）的首 token 是**最贵的一步**，不是最便宜的：prefill 要吃完整 encoder 输出并算出全部 cross-attention KV。Hailo 上实测 28–48 ms，而 encoder 本身只要 24 ms。

早期一次测得 15 ms 并被写进文档，事后加大 warmup 复测证明那是异常值。**报告 TTFT 时应给出首 token 的分布而不是单点均值**，并说明 warmup 次数。

### mel 补零位置

**Whisper 是先把波形补零到窗口长度、再算 mel**，不是算完 mel 再给 mel 尾部补 0。数字静音的 mel 值约为 **−0.58 而不是 0.0**，后者等于给 encoder 喂了一段训练分布外的常数。

本项目最初的 numpy 移植（RK 那条链）犯了这个错，Hailo 那条链因为直接用了上游 `audio_utils`（在**波形**上 `_pad_or_trim`）而没有问题。

修正后在 RK3588 上 A/B：

| | 修正前 | 修正后 |
|---|---|---|
| 20s en_short | 13.37% | 13.37% |
| 20s en_long | 6.33% | 7.58% |
| 20s zh_long (t2s) | 20.47% | 19.63% |
| **10s en_long** | **14.26%** | **11.40%** |

**只有 10s 的英文长句是超出噪声的真实改善（−2.9 点）**，符合机理——补零值错误只影响被补零的部分，而 20s 窗口下全部语料单窗装下、几乎不补零，10s 切块后的尾块补零最多。20s 那几处 ±1 点的摆动在 5 条样本上属噪声（逐条对比转录，内容几乎一致，仅首部空格之差），不应解读为「修正让 20s 变差」。

### 工具链

**whisper.cpp 不要传 `-np`**：它会连 `whisper_print_timings` 一起关掉（`cli.cpp:1350` 的 `if (!params.no_prints)`），而分阶段的 encode/decode 耗时正是从那里解析的。

**mel 前端可以纯 numpy 实现**，不必在板子上装 torch/librosa：与 `torch.stft` 实测 max|diff| ~1e-5、mean ~1e-7。

**`optimum` 1.27.0 撞新版 torch**：`ImportError: cannot import name '_attention_scale' from torch.onnx.symbolic_opset14`。钉 `torch==2.6.0` + `transformers==4.49.0`。

---

## 已知限制

- **每组仅 5 条，一条摆动就动 5~10 个百分点。** 本轮有三个独立实例：Orin Nano 上 tiny（7.30%）赢过 base（13.59%），差别只在 `en_short_05` 一条；RK3576 与 RK3588 的 en 短句差 4.4 点，全部来自 `en_short_03` 的两词之差；中文上 `zh_short_05` 的简繁翻转拉开 9.5 点。**只能看量级和分档，不能看排名。**
- **Jetson 的 TTFT 是代理值**，与其余各行的实测 TTFT 不可直接比较。
- **`docs/performance-comparison.md` 里其他 ASR 后端的数字**（Paraformer 2.6% 等）测于 2026-05-13，不同日期、不同镜像、每个平台跑各自的模型。同一批音频、同一个评分函数，但其余条件都不同——**只能看数量级**。
- 未覆盖：Orin NX（磁盘满）、Hailo base 的 5s 窗口、RK3576 的 10s 窗口、int8 量化的 RKNN。
