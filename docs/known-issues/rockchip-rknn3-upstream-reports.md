# 待报上游 Rockchip 的两个 RKNN3 / RKLLM 缺陷

两条都已完整诊断、可复现，且**无法在本地绕过** —— 需要上游修。本文档是提交给
Rockchip 时可以直接引用的材料。

平台：Radxa ROCK 5T（RK3588 host）+ RK1828 / RM182X PCIe NPU 加速卡，
PCIe ID `[1d87:182a]`，BDF `0001:11:00.0`。工具链 RKNN3-toolkit / RKLLM3 **V1.0.4**。
主机内核 `6.1.84-8-rk2410`。诊断日期 2026-07-30。

---

## 缺陷 1：`rknn-smi` 在该平台完全失效 → EP 显存与健康零观测

### 症状

所有信息类子命令都失败，**包括以 root 执行、以及 EP 完全空闲时**：

```
$ rknn-smi info            → Failed to initialize rknnsmi
$ rknn-smi info -t memory  → Failed to initialize rknnsmi
$ rknn-smi info -l         → Failed to initialize rknnsmi
```

`/var/log/rknn-smi.log` 里记录的是：

```
[ERROR] [rk_nsmi.c:109] cannot open lock file /tmp/rknnsmi.lock (errno: 13)
[ERROR] [rk_nsmi.c:152] rknnsmi init failed to acquire lock
```

### 为什么那条错误信息是误导性的

按顺序排除过：

| 试验 | 结果 |
|---|---|
| 删除 `/tmp/rknnsmi.lock` 后重试 | 锁文件被重新创建（`radxa:radxa 644`，0 字节），**仍然失败** |
| 检查文件属性 | `lsattr` 干净，无 immutable |
| 以 owner 身份 `open(O_RDWR)` | **成功**（Python 实测）→ 权限本身没问题 |
| 检查设备节点 | `/dev/pcie-rkep-0001:11:00.0` 是 `crw------- root root` → 非 root 必然 EACCES，**这曾是最像的解释** |
| **以真 root 执行**（`id -u` 确认为 0） | **仍然 `Failed to initialize rknnsmi`，exit 255** |
| EP 空闲时（停掉占用 EP 的服务后）执行 | **同样失败** → 不是"EP 忙"的已知行为 |

所以 `rk_nsmi.c:109` 报的 `cannot open lock file (errno 13)` **不是真实失败点** ——
root 也失败，且锁文件可被正常打开。

### 疑似真因：出厂即存在的 host / EP 固件版本不匹配

`/var/log/rknn-smi.log` 的 PCIe 握手段记录：

```
rc_cc_version = 30301     (host RC runtime v3.03.01)
ep_cc_version = 30201     (EP firmware   v3.02.01)
```

而**现装固件已经是 V1.0.4 安装包自带的最新版**：
`/lib/firmware/rknn3_rk1820.img`，md5 `37ca5e4e1ac9fbb8479a042a35574760`，
与 SDK V1.0.4 installer 内的固件一致。即 **V1.0.4 自身就把 RC runtime 配成 30301、
EP firmware 配成 30201**，且没有更新的 EP 固件可刷 → 无法在本地消除这个 skew。

值得注意的是 `librknnrt3` 的推理路径**不受影响**：Qwen3-4B 在同一张卡上稳定运行
（TTFT 132 ms、81.5 tok/s、8192 上下文）。只有 SMI 这条码路踩到了。

### 影响（这是最要紧的部分）

**EP 的显存占用和健康状态没有任何观测手段。** 后果：

1. **容量规划只能靠试。** 单 EP 只有 5120 MB，判断"还能不能再装一个模型"过去是靠
   `rknn-smi info | grep '/ 5120'`。现在唯一的信号是"模型加载成不成"。
2. **而这个信号本身有破坏性。** 反复失败的 model load 会把 EP 从 8 核退化到 4 核
   （需干净重启恢复）→ **被迫用一个有副作用的方法去探测本该只读的信息**。
3. **健康/wedge 判定失去主要手段。** 区分"firmware 握手偶发失败，重试即可"与
   "EP 已退化，必须干净重启"过去依赖 `rknn-smi info` 显示的 `RK1828 Online` 与核数。
   现在只能盲着做重启决定 —— 而这张卡上跑的是生产服务。
4. 温度 / 功耗 / `pcie_err` / `work_mode` / `prefill_mode` 全部不可见。

仍可用的替代观测（不足以做容量规划）：`lspci`、
`rknn3_transfer_proxy devices`（EP 仍正常枚举为 `0001:11:00.0ptr PCIE`）、
`systemctl is-active rknn3.service`、以及功能探针（模型 init 成功 ⇒ 显存足够）。

### 请求

1. 修复 `rknn-smi` 在 `rc_cc_version` / `ep_cc_version` 不一致时的初始化路径，
   或至少让它**报出真实失败原因**而不是一条误导性的 lock-file 错误。
2. 提供可刷的、与 V1.0.4 的 RC runtime 匹配的 EP 固件（30301）。
3. 若上述都不可行：提供**任何**只读的 EP 显存查询通路（sysfs、ioctl、API 皆可）。
   没有它，单 EP 平台上的多模型部署无法安全规划。

---

## 缺陷 2：RKLLM v1.2.3 无法在 EMBED 输入路径上复用 KV / prompt cache

### 背景

Qwen3-ASR 的解码器以 **EMBED 输入**方式驱动（音频编码器输出的 embedding 直接喂给
RKLLM，而非 token id）。其 prompt 布局为：

```
[system prefix][<|audio_start|>][AUDIO EMBED × N 帧][<|audio_end|>]
[指令][<|im_end|>][<|im_start|>assistant …][<asr_text>][prefix_text]
```

因为每次解码都必须重新 prefill 整段音频 embedding（`keep_history=0`），
流式识别的每一次 partial 与 final 都要付一遍完整 prefill 代价。

### 四条路全部走不通（V1.0.4 / RKLLM v1.2.3，RK3588，实测）

| 尝试 | 结果 |
|---|---|
| `rkllm_clear_kv_cache(keep=n)` + EMBED | **RoPE 位置被重置为 0** → 位置错配，输出错误 |
| `rkllm_save_prompt_cache`，输入为 EMBED | **cache 文件根本不生成** |
| `rkllm_save_prompt_cache`，输入为 TEXT | cache 生成成功（2.3 MB / 14 token 状态），但 load 后以 EMBED + `keep_history=1` 运行 → **输出错误**：模型继续 TEXT 上下文（"You are a helpful."）而不是处理音频。根因是 **TEXT 模式与 EMBED 模式的 KV 内部状态不同，RoPE 位置跨模态不兼容** |
| `rkllm_load_prompt_cache` + EMBED + `keep_history=0` | cache **被静默丢弃**，零加速 |

（这四条的结论与复现细节已固化在 rkvoice-stream 的
`backends/asr/qwen3/engine.py` 的注释块中，标注 `Tested on: RKLLM v1.2.3, RK3588`。
相应的 `precompute_prefix_kv()` 实现是写好的，因为试不通而被一行开关关闭。）

### 影响

- 直接可量化的损失：仅固定 prefix 那 15 个 token 就是**每次解码额外 90–130 ms**
  的 prefill。音频 embedding 那部分（一个 4 秒句子约 50 帧）更大，且完全无法摊销。
- **这不是 ASR 专属问题**：任何在 RK 平台上使用 embedding 输入的 LLM 应用
  （多模态、投影层前置、自定义编码器）都会撞上同一面墙。

### 请求

按 `engine.py` 注释中已记录的结论：**需要 RKLLM API 侧支持跨模态 cache，或提供一个
接受 EMBED 格式 cache 的 `rkllm_load_prompt_cache` 变体。** 在此之前该优化在 RK 平台
上不可实现（注：同一优化在 NVIDIA Jetson / TensorRT 路径上可用，解码提速 3–6 倍，
所以这确实是平台能力差距而非我们的实现问题）。

---

## 提交状态

未提交。渠道：Rockchip 官方支持，或 `airockchip/rknn-llm` /
`airockchip/rknn3-toolkit` 的 GitHub issue。
