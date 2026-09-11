# Kokoro RK 量化路线收口 + 方法学纠错（2026-08-28）

> Status: **量化路线 CLOSED**。战线 A（NPU 侧量化）与战线 B（tail-rest CPU 侧量化）全部走到底。
> 本文同时纠正三处既有文档的错误结论，并给出一条方法学硬规则。
> 所有 wall time 均为 **radxa RK3588 真机、容器 `openvoicestream` 内 ORT、四线程**（= 生产口径）。

## TL;DR

| 结论 | 状态 |
|---|---|
| tail-rest 做 INT8/混合位宽量化 | **关闭**——渐近线 0.0517 就在 0.05 门线上方 |
| vocoder-front-half 做量化 | **关闭**——可量化的 Conv 全家只占该阶段 19.8%，落到全局 3% |
| 既有文档里的 x86 微基准时延 | **全部作废**——ARM 比那台 x86 慢 9.5× |
| 新发现的靶子：Sin CPU fallback | 未做，值 ~10% |
| 唯一未关闭的杠杆 | **0.05 这个门本身**（从未验证过可听性） |

## 0. 方法学硬规则（本轮最贵的一课）

**RK 的 CPU 侧时延结论必须在 radxa 上测。x86 微基准只能用来比精度。**

同一个 ONNX（`kokoro-vocoder-tail-rest-cpu.onnx`，md5 `68c62e75…`）：

| 机器 | 线程 | 耗时 |
|---|---|---|
| wsl2-local（x86_64） | 1 | 687 ms |
| radxa（RK3588 Cortex-A76） | 1 | **6559 ms** |
| radxa | 4 | **2169 / 2170 / 2176 ms**（三次独立复测） |

架构差 **9.5×**，线程加速 **3.1×**。radxa 四线程 2169 ms 对上生产埋点 `tail_ms` 2100 ms —— harness 可信。

`kokoro-rk-tail-rest-int8-static.md:112-114,127` **自己就写了这条 caveat**（"wsl2-local CPU performance does NOT directly map to RK3588 ARM… Validate on RK3588 before shipping"），但 P7b 和本轮第一次实验都原样继承了 x86 口径并据此下了性能结论。

**根因不是某个人疏忽，是仓库里两套口径混放且无标注**：`third_party/rkvoice-stream/docs/kokoro-rknn-analysis.md` 的表格一直是 radxa 真机数据，`docs/specs/kokoro-rk-tail-rest-int8*.md` 的表格是 x86 数据，读者无从区分。

**要求**：本仓库任何时延表格必须在表头注明测量机器与线程数。

测量方法（可直接复用）：

```bash
~/.rpty/bin/fleet transfer wsl2-local:/path/x.onnx radxa:/home/radxa/x.onnx
~/.rpty/bin/fleet exec radxa -- 'docker cp /home/radxa/x.onnx openvoicestream:/tmp/x.onnx \
  && docker exec openvoicestream python3 /tmp/bench_tail.py /tmp/x.onnx 4'
```

> ⚠️ `fleet transfer` 有时报 `verify: SKIP (md5 unavailable on remote)`，**此时校验没有真的发生**。曾拿到字节数正确（18154152）但 md5 完全不同的文件，ORT 直接 `INVALID_PROTOBUF`。断点续传残留会污染目标文件。**重传前先 `rm -f` 目标，且必须手工 md5 对拍。**

## 1. 真机基准（radxa，四线程）

流水线（长句，5.25s 音频）：prefix 15ms → decoder-front INT8 (NPU) 40ms → vocoder-front-half FP16 (NPU) 830ms → **tail-rest (CPU ORT) 2169ms** → 合计 3175ms，RTF 0.59。

tail-rest 占 **68%**。

## 2. 战线 A：vocoder-front-half —— 量化是错的杠杆

用 C 程序直调 `librknnrt.so` 的 `rknn_init(RKNN_FLAG_COLLECT_PERF_MASK)` → `rknn_run` → `rknn_query(RKNN_QUERY_PERF_DETAIL)` 拿到真机逐层表（合计 755ms，对生产埋点 830ms）：

| OpType | Target | n | 耗时 | 占比 |
|---|---|---|---|---|
| **Sin** | **CPU** | 24 | 314 ms | **41.6%** |
| Transpose | NPU | 42 | 152 ms | 20.1% |
| Conv | NPU | 38 | 84 ms | 11.1% |
| exNorm | NPU | 21 | 60 ms | 7.9% |
| ConvAdd | NPU | 13 | 55 ms | 7.3% |
| Mul / Add | NPU | 148 | 61 ms | 8.2% |
| Pow | NPU | 24 | 13.5 ms | 1.8% |
| exConvTransposePad | CPU | 1 | 11 ms | 1.4% |

**by target: CPU 43.1% / NPU 56.9%。**

### 2.1 纠错：`kokoro-rk-34pct-m4m6-final.md:90` 的结论不成立

原文称"NPU 的 native-FP16 性能在这个 Sin/Pow/InstanceNorm-heavy 子图上不如 ORT-CPU"。

**实际是 24 个 Sin 被 RKNN 编译器 fallback 回 CPU，从来没在 NPU 上跑过。** 原结论是拿"整段比 CPU 慢"倒推出的一个关于 NPU 硬件性能的判断，缺少 per-op target 证据。42 个 Transpose（20.1%）是 CPU/NPU 来回切换的 layout 税。

**通用教训：任何"某段上了 NPU 反而更慢"的观察，必须先查 per-op target 有没有 fallback，再谈硬件性能。**

### 2.2 为什么量化在这里不值得

可量化的 Conv 全家（Conv + ConvAdd + exConvTransposePad）合计 **19.8% / 150 ms**。即便 INT8 给到 3×，省 100 ms，落到 3175 ms 全流水线是 **3%**。

真正的靶子是 Sin + 其 layout 税 = **61.7% / 466 ms**。从算子名（`noise_res.0/Sin`、`resblocks.N/Sin`）加 24 Sin 配 24 Pow 判断是 **Snake 激活** `x + sin²(αx)/α`。

M4 当年试过 clipped-Sin 多项式改写让它上 NPU，parity 挂在 rel_l2 0.45–0.67，但 `kokoro-rk-42pct-m4-vocoder-fp16.md:52` 写明**误差全部来自多项式本身，RKNN 复现改写后的模型只有 0.001**——是数值近似问题，不是硬件问题。

**估算收益 ~10%**（RTF 0.59 → 0.53，长句 TTFA 3.1s → 2.8s），天花板 16%。锚点：同一批张量上逐元素算子在 NPU 上 0.56 ms/op（由 24 个 Pow 的 13.5 ms 得出），Sin 在 CPU 上 13.1 ms/op。

未做。前提是找到能过 parity 门的 sin 近似（范围规约 + 高阶多项式，或 `sin²(αx) = (1-cos(2αx))/2` 换形式）。

## 3. 战线 B：tail-rest —— 三条路全部证伪

图结构（417 节点）：Add 93 / Mul 80 / Slice 51 / **Conv 28** / **Sin 27** / **Pow 26** / Reshape 25 / **Gemm 24** / **InstanceNormalization 22** / **ConvTranspose 3** / LeakyRelu 2 / Exp 1。

注意：量化只覆盖 Conv/ConvTranspose/MatMul/Gemm，**Sin/Pow/InstanceNorm 全程 FP32**。这解释了 A8 全量量化只能拿到 −49%（1098 vs 2170 ms）——约一半的图根本不可量化。

音频门：`rel_l2 ≤ 0.05`，held-out 样本 `sample_0200–0211`（标定只用前 200 个）。

### 3.1 选层排除（H1）

BitTTS（Interspeech 2025, arXiv:2506.03515）指出量化 vocoder 必须排除最靠近波形输出的卷积。

| 排除集 | rel_l2 | 四线程 |
|---|---|---|
| 全量 A8 (`n0`) | 0.573 | 1155 ms |
| 排 iSTFT 两个 ConvTranspose | 0.579 | — |
| **只排 `conv_post`** (`cp`) | **0.278** | **1098 ms** |
| 排 9 层 (`n9`) | 0.224 | 1491 ms |

- 排 iSTFT 的两个 ConvTranspose **完全无效**（0.573→0.579）——它们无辜
- `conv_post` 是真正的敏感点，排它一层**误差砍半且零速度成本**（它输出幅度谱后紧接 `Exp`，误差被放大成乘性）
- 再往回排 7 层只从 0.269 到 0.224，wall time 却涨 393 ms——**误差弥散在整个 Conv 栈**

### 3.2 标定方法（H2）

Percentile 99.999：0.219（比 MinMax 好 18%）；Percentile 99.99：0.405（截太狠）；**Entropy 产出与 MinMax 逐字节相同的 md5**（日志确认算法真跑了，88 tensor / 128 bins）——等于无操作。标定样本 48 vs 200 产物 md5 相同，样本量不是变量。

### 3.3 逐 tensor 混合 A8/A16（H3）

API：`quantize_static(extra_options={'TensorQuantOverrides': …, 'UseQDQContribOps': True})`，配合 `onnxruntime/quantization/execution_providers/qnn/mixed_precision_overrides_utils.py:33` 的 `MixedPrecisionTensorQuantOverridesFixer` 插入 u8↔u16 转换对。

**敏感度排序方法**（`p7d/sens.py`）：单 tensor QDQ 注入——在 FP32 图里**只**给一个 tensor 插一对 Q/DQ，其余全 FP32，量端到端 rel_l2。这个指标**天然包含下游误差放大**。

> **SQNR 单独用会排错**：`Mul_output_0` 局部 SQNR 37.1 dB（85 个 tensor 里算干净的），端到端伤害却排第 5。局部信噪比不含放大因子。

> **坑**：注入的 Q/DQ 必须插在 producer 节点**后面一位**；append 到 node 列表末尾会破坏拓扑序，`onnx.checker` 全部 ValidationError。

**敏感度空间规律**：top-16 几乎全部落在 `resblocks.4` / `resblocks.5`（最后两组、时间分辨率最高的残差块）+ `ups.1/ConvTranspose` 及紧贴它的 `LeakyRelu_1` / `Mul_output_0`。早期低分辨率的 `noise_res.*` / `m_source.*` 排到 32 名以后。**误差集中在上采样链末端的高分辨率段。**

**K 扫描**（底座 = 排 `conv_post`，top-K 提到 A16）：

| K | rel_l2 | 四线程 |
|---|---|---|
| FP32 | — | 2170 ms |
| 0 | 0.278 | 1098 ms |
| 2 | 0.237 | 1178 ms |
| 4 | 0.209 | 1227 ms |
| 8 | 0.173 | 1334 ms |
| 16 | 0.125 | 1522 ms |
| 32 | 0.082 | 1844 ms |

边际成本约 **23 ms / 每个 A16 tensor**。**性能门全过，音频门全不过。**

### 3.4 为什么不可能过：渐近线论证

这条曲线的渐近点（K→85，即全 A16 + 排 `conv_post`）就是已测的 `n2_u16` = **0.0517**，而门是 **0.05**。

> **rel_l2 曲线的下界本身就在门线上方。混合位宽只是在 0.278 和 0.0517 之间插值，K 取多少都碰不到 0.05。**

唯一真正过门的 `n9_u16`（A16 **且**排 9 层，0.0498）代价 2246 ms ≈ FP32 的 2170 ms——速度全吐回去。

**误差不是被少数几个 tensor 绑架的；A8→A16 的收益（0.573→0.069）是全图弥散摊出来的。** 这与 3.1 的"选层救不了"是同一现象的两面。

### 3.5 副产物：A16 覆盖严格优于排层

`k8`（1334 ms / 0.173）比 `n9`（1491 ms / 0.224）**又快 157 ms 又准**。以后要在精度-速度曲线上腾挪，用 A16 覆盖，不要用排层退回 FP32。

## 4. 已排除的其他方向

| 方向 | 结论 | 证据 |
|---|---|---|
| 换 FFT-based iSTFT 替代 ConvTranspose | 收益可忽略 | iSTFT 段仅 **7 ms / 1961 ms ≈ 0.36%**（radxa 真机，`third_party/rkvoice-stream/docs/kokoro-rknn-analysis.md:236-241`） |
| tail-rest 整体上 NPU | 比 CPU 慢 | 同表：RKNN FP16 3464 ms、INT8 2656 ms vs CPU 1961 ms |
| 等价缩放（SmoothQuant / CLE） | **关闭**，但理由与原推断不同 | 残差**在激活侧**（A16/W-fp32 = 0.103 vs A-fp32/W8 = 0.049），方向正确；关闭是因为搬运目标端（权重侧 0.049）紧贴 0.05 的门、零余量。CLE 另受阻于 Snake 非标量等变。见 §5.2 |

## 5. 未闭合项

### 5.1 【最高优先】0.05 这个门本身

0.05 是 P7b 定的**张量级代理指标**，**从未验证过它对应可听的劣化**。M4 有放宽先例（per-tensor 0.03 → 端到端音频门）。

杠杆很大：

| 门 | 可用配置 | 收益 |
|---|---|---|
| 0.10 | `k16`（1522 ms） | 省 648 ms = **全局 20%** |
| 0.09 | `k32`（1844 ms） | 省 326 ms = 全局 10% |
| 0.05（现状） | 无 | 全部路径关闭 |

**下一步应做 `k16` / `k32` 的 AB 听音，而不是继续调量化配置。**

### 5.2 等价缩放（SmoothQuant / CLE）—— 已判定，关闭

SmoothQuant 式的 `X·W = (X/s)·(s·W)` 把量化难度从激活搬到权重。方向上对准了我们的症状（激活爆、权重空闲：P7a 权重-only INT8 就是 rel_l2 0.01）。可折点：22 个 InstanceNorm(AdaIN) 的 gamma 来自 Gemm，缩放可折进 Gemm 权重行。

**判定实验已做（2026-08-28）**，三个点（同一 fake-quant 注入口径，同 gate、同 held-out）：

| 变体 | rel_l2 |
|---|---|
| A16 / W-fp32（纯激活侧残差） | **0.1030** |
| A-fp32 / W8（纯权重侧残差） | **0.0490** |
| A8 / W-fp32（对照） | 0.2846 |

**归因：残差在激活侧**，激活 0.103 是权重 0.049 的两倍多，A16 远未榨干。等价缩放瞄准的正是主要矛盾——原先"残差在权重侧、故方向错误"的推断**被证伪**。

**但仍然关闭，理由改为落点已满**：搬运的目标端（权重侧）单独就是 0.0490，紧贴 0.05 的门。等价缩放的理想极限是把激活误差压到零、只剩权重误差 ≈ 0.049，**零余量**；真实 SmoothQuant 只能收回部分差距，落点必然更高。

> ⚠️ **方法学警告：这三个数不能跨口径相加。** 8 bit 下激活-only 0.2846 vs 真实 A8W8 0.573（远低）；16 bit 下激活-only 0.1030 vs 真实 A16W8 0.0691（**反而更高**，自相矛盾）。本注入 harness 对全部 85 个 tensor 无差别插 QDQ，是**穷尽上界**，而 ORT 真实管线会丢弃/融合部分 QDQ。`0.103 + 0.049 = 0.069` 不成立，误差是非线性交互。**唯一站得住的是组内比较**（激活 ≫ 权重）；上面"理想极限 0.049"的推算跨了口径，信心须打折。

排除项：标定样本 48 与 200 产出逐位相同的模型（md5 `e91fe8bd…`），"held-out 被裁剪"的解释不成立。

经典 CLE 基本用不了：全图只有 2 个 LeakyRelu，而 Snake `x+sin²(αx)/α` 不满足正标量等变性。

实现口径（复现用）：u16 的 Q/DQ 必须走 `com.microsoft` 域——模型 opset 为 ai.onnx 14，标准域 `QuantizeLinear` 的 uint16 支持始于 opset 21。权重侧 fake-quant 为 per-channel 对称 int8（`scale = max|w|/127`），轴按 ORT 约定 Conv→0 / ConvTranspose→1 / MatMul→1 / Gemm→transB 时 0 否则 1。脚本 `p7d/attrib.py`。

### 5.3 Sin CPU fallback（战线 A 的真靶子）

见 §2.2。值 ~10%，需要能过 parity 门的 sin 近似。

## 6. 纠正的既有文档

| 文档 | 原文 | 更正 |
|---|---|---|
| `kokoro-rk-34pct-m4m6-final.md:90` | "NPU native-FP16 在此子图上不如 ORT-CPU" | Sin 从没上过 NPU，是 CPU fallback。见 §2.1 |
| `kokoro-rk-perf-r-and-d-closure.md` | "RKNN one-graph-one-precision constraint" | 不成立。rknn-toolkit2 2.3.2 `rknn/api/rknn.py:84-86` 一手确认支持 `w8a8/w8a16/w16a16i/w16a16i_dfp/w4a16`、`normal/mmse/kl_divergence/gdq`、`hybrid_quantization_step1(proposal=True)`、`build(auto_hybrid=True)`、`accuracy_analysis()`。混合精度自 v1.1.0 即有 |
| `kokoro-rk-tail-rest-int8-static.md` §2a | bucket-32 "killed before completion" | 有误。`qstatic_b32.log` 显示 `OK in 297.2s`，产物一直在盘上。当年"bucket-32 音频未测"的真实原因是磁盘上没有 bucket-32 的 prefix/front ONNX，文本驱动 gate 报 `tokens Expected: 16` |
| `kokoro-rk-tail-rest-int8*.md` 全部时延表 | x86 单线程数字 | 作废，见 §0。rel_l2 一列仍有效 |

## 7. 产物与复现

**wsl2-local** `/home/harve/kokoro-analysis/`：
- `p7c/` — 选层与标定实验（`gate_npz.py` 是 gate 脚本）
- `p7d/` — 混合位宽实验（`sens.py` 敏感度、`qmix.py` 构建、`sens.json` 完整 85 条排序）

**radxa** `/home/radxa/`：
- `perf_probe3.c` / `perf_probe3` — RKNN 逐层 perf 探针（C 直调 librknnrt）
- `bench_tail.py` — ORT wall time harness（容器内 `/tmp/bench_tail.py`）
- `{n0,n9,cp,k2,k4,k8,k16,k32}.onnx`

**RKNN 探针要点**（rknn-toolkit2 在 aarch64 装不上：依赖 `onnxoptimizer==0.3.8` 无 aarch64 wheel；rknn-toolkit-lite2 没有 `eval_perf()`）：

```c
rknn_init(&ctx, model, size, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
/* 输入必须喂 fp32，runtime 自转 fp16；pass_through=1 配 attr.size 会报
   "param input size(430080) < model input size(860160)" */
rknn_run(ctx, NULL);
rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
```

**未变更**：本轮全程只读生产。未动 radxa 容器、compose、env、`/opt/`，未发布任何产物到 HF。
