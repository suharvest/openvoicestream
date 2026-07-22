# voxedge 后端实机性能基准 (2026-05-31)

新 voxedge 代码路径(server-loop OFF = 纯后端 passthrough，等价 main 行为）在 4 台实机上的各后端性能。目的：① 验证 voxedge 间接层未引入 perf 回归；② 为各后端建立新基线。

**测量方法**
- Harness: `bench/perf/run_on_device.sh <device> -- <asr|tts|v2v>`，client 跑在设备本地（去掉网络延迟 = 设备内禀数）。
- **ASR 口径 = `eos_to_final`**（强制 eos 后到 final 文本的真后端算力）。`tfd`(~503ms Jetson) 含 VAD/segment 常量，**不是**后端延迟。offline 型后端（paraformer）`eos_to_final` 退化（~0.5ms），真算力看 `asr_finalize_compute`。
- **TTS 口径 = TTFA**（首音频帧）+ **RTF**（稳态合成 / 音频时长，越低越快）。TTS 必验**非静音**（RMS 能量 / ASR 回环），字节非空 ≠ 有声。
- **v2v = EOS→Audio**（端到端，LLM delay=0）。

---

## ASR 后端（finalize 延迟）

| 后端 | 设备 | 平台 | finalize p50/p95 | RTF | 备注 |
|---|---|---|---|---|---|
| **trt_edgellm** (qwen3 ASR) | orin-nx | Jetson 16GB | **142 / 144 ms** | — | eos→final，最快 |
| **paraformer_trt** | orin-nx | Jetson 16GB | 251 ms (compute¹) | — | offline，eos→final 退化 |
| **paraformer_trt** | orin-nano | Jetson 8GB | 667 / 743 ms | — | 8GB 内存带宽受限 |
| **qwen3_asr_rk** (RKNN enc + RKLLM dec) | radxa | RK3588 | zh 1260 / en 1387 ms | ~1.7 | NPU，RTF>1 模型内禀 |
| **qwen3_asr_rk** | cat-remote | RK3576 | zh 1885 / 2006 ms | ~1.5 | NPU |

¹ paraformer eos→final 退化（~0.5ms），真算力 = `asr_finalize_compute` (VAD-endpoint→final) 250.8ms。

## TTS 后端（TTFA / RTF）

| 后端 | 设备 | 平台 | TTFA p50 | RTF p50 | 非静音 |
|---|---|---|---|---|---|
| **matcha_trt** | orin-nx | Jetson 16GB | **4.7 ms** | **0.017** | ✓ RMS 3422 |
| **matcha_trt** | orin-nano | Jetson 8GB | ~7-9 ms | 0.027 | ✓ |
| **kokoro_trt** | orin-nano | Jetson 8GB | ~9 ms | 0.041 | ✓ RMS 1402 |
| **matcha_rknn** (ORT acoustic + RKNN vocos) | radxa | RK3588 | — | 0.09 | ✓ |
| **matcha_rknn** | cat-remote | RK3576 | ~14 ms | 0.16 | ✓ RMS ~5000 |
| **kokoro_rknn** 3-stage (17% NPU) | radxa | RK3588 | — | 0.43 | ✓ RMS ~3000 |
| **kokoro_rknn** 4-stage (34% NPU) | radxa | RK3588 | — | 0.38 | ✓ RMS ~3000 |
| **moss_tts_nano** | orin-nx | Jetson 16GB | **137 ms** | — | ✓ RMS 1526² |
| **qwen3_tts** (highperf) | orin-nx | Jetson 16GB | **531 ms** | **0.688** | ✓ RMS 2281³ |

² moss **已修好并跑通**（2026-06-01）：model_id 修复（6217b2c）让 backend+worker 启动；**ORT ABI 阻塞已解 = 把 C++ worker 重链到 ORT 1.23.2**（旧 worker built against `VERS_1.20.0`，标准镜像装 onnxruntime 1.23.2 找不到符号）。重链后 `nm` 确认只剩 `VERS_1.23.2`，用标准 ORT 1.23.2 实测合成 `今天天气真不错。` → **TTFA 137ms / wall 746ms / RMS 1526 非静音**，与历史 ~157ms 同量级。**moss 现进统一镜像**（无需 ORT-1.20 专用镜像）。新 worker（sha256 b09f7f8d，562KB）已上 HF + 更新 deploy/jetson-workers/MANIFEST.json + docs/runbooks/moss-tts-nano-deployment.md。engine_resolver 的 files-as-list 崩 + 无 .meta 误删引擎两 bug 已修（97a9b9f）。
³ qwen3_tts **FP16 CustomVoice 路径已跑通**。旧 W8A16 成功记录后来确认是 `QWEN3_TTS_PRELOAD_TALKER_EMBEDS` 掩盖了 talker 自回归生成；v0.8 no-preload gate 下 W8A16 到 200-frame cap 且 codec EOS 不触发，不能作为生产精度。TTFA/RTF = 直连 `/tts/stream` 实测（首个 PCM chunk，含 talker prefill + 首个 code2wav chunk），24kHz mono，RMS 998–3393 真实人声。历史启动修复仍适用：(a) customvoice worker 并发参数需匹配实际 worker CLI；(b) voxedge `trt_edge_llm_tts.py:_ensure_worker` 要透传 explicit-KV talker flags。

## v2v 端到端（EOS→Audio）

| 组合 | 设备 | EOS→Audio p50 | 备注 |
|---|---|---|---|
| qwen3asr + matcha_trt | orin-nx | ~146 ms | ASR 主导 |
| qwen3asr + qwen3_tts (highperf) | orin-nx | ~673 ms⁵ | ASR finalize 142ms + TTS 首音 531ms |
| paraformer + matcha_trt | orin-nx | ~257 ms | |
| paraformer + matcha_trt | orin-nano | 640-704 ms | 8GB |
| qwen3_asr_rk + matcha_rknn | radxa | 308 ms | |
| qwen3_asr_rk + kokoro_rknn 3-stage | radxa | 515 ms | |
| qwen3_asr_rk + kokoro_rknn 4-stage | radxa | 518 ms | |
| qwen3_asr_rk + matcha_rknn | cat-remote | **OOM** | RK3576 8GB：RKLLM decoder(795MB)+KV+matcha 共驻爆 8GB⁴ |

⁴ 见 `asr_worker_kv_overflow_long_audio` memory：cat-remote cutover 应 gate 在 ASR KV-cap 修复之后。
⁵ qwen3asr+qwen3_tts EOS→Audio：bench harness `v2v` 的 `tts_tfd` 只计到 4-byte sample-rate header（~5ms，**非真实首音**），故复合值由 ASR finalize(142ms) + 直连 TTS 实测首音(531ms) 相加得 ~673ms。harness v2v 同时受 session-limiter 背靠背 4429 拖累（10 样本 7 个 timeout，已设 `OVS_MAX_CONCURRENT_SESSIONS=3` 缓解）。

---

## 覆盖状态

| | trt_edgellm | paraformer | qwen3_asr_rk | matcha | kokoro | moss | qwen3_tts |
|---|---|---|---|---|---|---|---|
| orin-nx 16G | ✅ | ✅ | — | ✅ trt | 缺模型 | ✅ 137ms | ✅ |
| orin-nano 8G | 缺 artifact | ✅ | — | ✅ trt | ✅ trt | ⏳ | n/a |
| radxa RK3588 | — | — | ✅ | ✅ rknn | ✅ 3+4stage | — | — |
| cat RK3576 | — | — | ✅ | ✅ rknn | (matcha 线) | — | — |

✅ 有数 · ⏳ 修复已就绪待重跑/build · 🚫 硬阻塞 · — 该平台无此后端

## 结论
- **voxedge 间接层无 perf 回归**：matcha_trt RTF 0.017 / trt_edgellm finalize 142ms 与迁移前同量级。
- **qwen3_tts 补全（2026-05-31, orin-nx；2026-06-12 复核）**：FP16 CustomVoice no-preload 路径 EOS 正常；W8A16 no-preload 复核失败，不能继续作为 highperf 默认。镜像 `openvoicestream:jetson-voxedge-profile3`（customvoice 基底 overlay + 新 voxedge wheel 6e019cce）。启动注意项见脚注³。
- **moss 已修好（2026-06-01）**：C++ worker 重链 ORT 1.23.2 → 标准镜像实测 TTFA 137ms/RMS 1526 非静音，进统一镜像。新 worker 上 HF。
- **profiling 副产出多个真 bug**：#40 base concurrency dict、Dockerfile wheel-name、rk3588-34% profile 路径降级、moss model_id、session-limiter 泄漏(#41)、agent/SLV server-loop 引号、config 漂移；本轮新增：voxedge TTS 未透传 explicit-KV talker flags、engine_resolver `files`-as-list 崩溃 + 无 `.meta` 时误删引擎。

> 数据来源：2026-05-31 4 设备 voxedge profiling sweep。复跑见 `bench/perf/results/`。
