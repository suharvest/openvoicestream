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
| **moss_tts_nano** | — | Jetson | _pending_ | _pending_ | model_id 修复后待重跑² |
| **qwen3_tts** (highperf) | — | Jetson NX | _pending_ | _pending_ | 缺 talker 引擎，defer³ |

² moss 经 #40 (base concurrency_capability) + model_id 修复后真机可启动；perf 待用含修复的镜像重跑（引擎已 staged 在 orin-nano）。历史 ORT/TRT 路径 TTFA 已知 ~157ms。
³ qwen3_tts 需离线 build talker `llm.engine`（只有 safetensors 源）；perf 前序已知。

## v2v 端到端（EOS→Audio）

| 组合 | 设备 | EOS→Audio p50 | 备注 |
|---|---|---|---|
| qwen3asr + matcha_trt | orin-nx | ~146 ms | ASR 主导 |
| paraformer + matcha_trt | orin-nx | ~257 ms | |
| paraformer + matcha_trt | orin-nano | 640-704 ms | 8GB |
| qwen3_asr_rk + matcha_rknn | radxa | 308 ms | |
| qwen3_asr_rk + kokoro_rknn 3-stage | radxa | 515 ms | |
| qwen3_asr_rk + kokoro_rknn 4-stage | radxa | 518 ms | |
| qwen3_asr_rk + matcha_rknn | cat-remote | **OOM** | RK3576 8GB：RKLLM decoder(795MB)+KV+matcha 共驻爆 8GB⁴ |

⁴ 见 `asr_worker_kv_overflow_long_audio` memory：cat-remote cutover 应 gate 在 ASR KV-cap 修复之后。

---

## 覆盖状态

| | trt_edgellm | paraformer | qwen3_asr_rk | matcha | kokoro | moss | qwen3_tts |
|---|---|---|---|---|---|---|---|
| orin-nx 16G | ✅ | ✅ | — | ✅ trt | 缺模型 | ⏳ | ⏳ |
| orin-nano 8G | 缺 artifact | ✅ | — | ✅ trt | ✅ trt | ⏳ | n/a |
| radxa RK3588 | — | — | ✅ | ✅ rknn | ✅ 3+4stage | — | — |
| cat RK3576 | — | — | ✅ | ✅ rknn | (matcha 线) | — | — |

✅ 有数 · ⏳ 修复已就绪待重跑/build · — 该平台无此后端

## 结论
- **voxedge 间接层无 perf 回归**：matcha_trt RTF 0.017 / trt_edgellm finalize 142ms 与迁移前同量级。
- **profiling 副产出 7 个真 bug**（全已修提交）：#40 base concurrency dict、Dockerfile wheel-name、rk3588-34% profile 路径降级、moss model_id、session-limiter 泄漏(#41)、agent/SLV server-loop 引号、config 漂移。
- **缺口**：moss（修复已就绪，重跑出数即可补全本表）、qwen3_tts（需 build talker 引擎）。

> 数据来源：2026-05-31 4 设备 voxedge profiling sweep。复跑见 `bench/perf/results/`。
