# voxedge-engine Overlay 抽取清单 (extraction inventory)

> 状态: 只读分析产出。本文档不改 fork 任何文件、不 build、不 commit、不创建 overlay 仓。
> 对象: `/Users/harvest/project/TensorRT-Edge-LLM` @ 分支 `v071/customvoice-product`
> (NVIDIA 官方仓 fork, push remote = `github.com/suharvest/TensorRT-Edge-LLM`)。
> 背景: `docs/archive/specs/voxedge-engine-overlay-and-artifacts.md` §2 + §2A (addon/patch 纪律 + (a)/(b) 分流)。
> 配套: fork 内未跟踪 `docs/DIVERGENCE.md` 已并入本清单 (见各节 + §6 草表)。

---

## 0. Base commit (UPSTREAM_PIN)

| 项 | 值 |
|---|---|
| **merge-base (base)** | `364769036fc83351d9d0aac4cc064a8e56a83178` |
| 上游 tag | `v0.7.1` (= `3647690`, `Merge pull request #90 … dev-release/0.7.1`) |
| 分析 HEAD | `893ba2a` (`refactor(runtime): extract generic SlotPool<TSlot>, use in TTS worker`) |
| commits ahead | 90 (`v0.7.1..HEAD`) |
| HEAD 直接 descend v0.7.1 | 是 (无上游 tag 后的 commit 混入), clean rebaseable |

> `UPSTREAM_PIN` = `364769036fc83351d9d0aac4cc064a8e56a83178`,`upstream.remote` = `https://github.com/NVIDIA/TensorRT-Edge-LLM.git`。
> 注意: 任务 brief 提到的 `qwen3-tts-highperf-runtime-w8a16` 分支名不存在为本地/远程 ref;**活分支是 `v071/customvoice-product`**。

### diff 统计

`git diff --name-status <base>...HEAD`: **A=40, M=27, D=0**。整体高度 *additive*
(DIVERGENCE.md 记 13,555 insertions vs 336 deletions),大多数新功能落在隔离的新文件,
这是 overlay 抽取的理想姿态——多数进 `addon/` 纯拷,只少量改上游需 `patches/`。

---

## 1. Addon 清单 (A = 上游不存在的新文件 → `addon/<相对路径>`)

按 spec §2 分组列全 40 个新增文件。Isolation 全部为 ISO (独立新文件)。

### 1.1 `cpp/kernels/` — CUDA kernel (6)

| addon 路径 | 用途 | 分类 |
|---|---|---|
| `cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.cu` | MOSS linear-attention KV append kernel | (b) |
| `cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.h` | 同上 header | (b) |
| `cpp/kernels/qwen3TtsCpKernels/qwen3TtsCpKernels.cu` | Qwen3-TTS code-predictor (CP) kernel | (b) |
| `cpp/kernels/qwen3TtsCpKernels/qwen3TtsCpKernels.h` | 同上 header | (b) |
| `cpp/kernels/w8A16LinearKernels/w8A16Linear.cu` | W8A16 量化 linear kernel (talker AWQ) | (b) |
| `cpp/kernels/w8A16LinearKernels/w8A16Linear.h` | 同上 header | (b) |

### 1.2 `cpp/runtime/` — runtime (3)

| addon 路径 | 用途 | 分类 |
|---|---|---|
| `cpp/runtime/mossTtsNanoRuntime.cpp` | MOSS-TTS-Nano 推理 runtime (1276 行) | (b) |
| `cpp/runtime/mossTtsNanoRuntime.h` | 同上 header | (b) |
| `cpp/runtime/slotPool.h` | 通用 `SlotPool<TSlot>` 模板 (N=2 并发槽位池) | (b) + pr_candidate |

### 1.3 `cpp/multimodal/` — multimodal runner (2)

| addon 路径 | 用途 | 分类 |
|---|---|---|
| `cpp/multimodal/statefulCode2WavRunner.cpp` | 有状态流式 Code2Wav runner (per-slot 状态) | (b) |
| `cpp/multimodal/statefulCode2WavRunner.h` | 同上 header | (b) |

### 1.4 `cpp/plugins/` — TRT plugin (2)

| addon 路径 | 用途 | 分类 |
|---|---|---|
| `cpp/plugins/w8A16LinearPlugin/w8A16LinearPlugin.cpp` | W8A16 linear TRT plugin (455 行) | (b) |
| `cpp/plugins/w8A16LinearPlugin/w8A16LinearPlugin.h` | 同上 header | (b) |

### 1.5 `cpp/workers/` — IPC worker (2)

| addon 路径 | 用途 | 分类 |
|---|---|---|
| `cpp/workers/moss_tts_nano_worker.cpp` | MOSS JSON-line 流式 worker (822 行) | (b) |
| `cpp/workers/build_moss_worker.sh` | MOSS worker 构建脚本 | (b) → 也可视 `addon/scripts/` |

### 1.6 `examples/omni/` + `examples/llm/` — example/spike (5)

| addon 路径 | 用途 | 分类 |
|---|---|---|
| `examples/omni/qwen3_tts_streaming_worker.cpp` | Qwen3-TTS JSON-line 流式 worker (896 行, `--max_slots`, cancel) | (b) |
| `examples/llm/spike_m1_append_prefill_embeds.cpp` | 流式 ASR spike: prefill embeds 追加 | (b) |
| `examples/llm/spike_m2_session_lifecycle.cpp` | ASR session 生命周期 spike | (b) |
| `examples/llm/spike_m35_audio_runner_split.cpp` | audioRunner 拆分 spike | (b) |
| `examples/llm/spike_m36_empirical_lcs.cpp` | 经验 LCS dedup spike | (b) |

### 1.7 `scripts/` — export/quantize (3)

| addon 路径 | 用途 | 分类 |
|---|---|---|
| `scripts/quantize_onnx_matmul_w8a16.py` | ONNX matmul W8A16 量化 | (b) |
| `scripts/quantize_onnx_matmul_w8a16_awq.py` | ONNX matmul W8A16-AWQ 量化 | (b) |
| `scripts/create_qwen3_vocoder50_wrapper.py` | Qwen3 vocoder50 包装器导出 | (b) |

### 1.8 `tests` / `unittests` — 测试 (5)

| addon 路径 | 用途 | 分类 |
|---|---|---|
| `unittests/mossLinearKvKernelsTests.cu` | MOSS KV kernel gtest | (b) |
| `unittests/mossTtsNanoSmokeMain.cpp` | MOSS runtime smoke main | (b) |
| `experimental/server/tests/__init__.py` | server test 包初始化 | (a) cand |
| `experimental/server/tests/test_cache_messages_branch.py` | system-prompt cache 分支测试 | (a) cand |
| `experimental/server/tests/test_tool_call_stream_parser.py` | tool_call 流式 parser 测试 | (b) (绑 tool-calling 特性) |

### 1.9 `docs/` — 文档/部署脚本 (10) — *非 build 输入, 进 overlay 文档区*

| addon 路径 | 用途 | 分类 |
|---|---|---|
| `docs/deploy-container/README.md` | AWQ Orin 部署容器说明 | (b) |
| `docs/deploy-container/STATUS.md` | 部署状态记录 | (b) |
| `docs/deploy-container/build_qwen3_awq_on_orin.sh` | Orin 上 build AWQ engine | (b) |
| `docs/deploy-container/build_server_bindings_on_orin.sh` | Orin 上 build server bindings | (b) |
| `docs/deploy-container/export_qwen3_awq_on_wsl.sh` | WSL 导出 AWQ ONNX | (b) |
| `docs/deploy-container/package_qwen3_awq_onnx.sh` | 打包 AWQ ONNX | (b) |
| `docs/deploy-container/qwen_processed_chat_template.json` | chat template 配置 | (b) |
| `docs/deploy-container/run_qwen3_awq_inference.sh` | AWQ 推理脚本 | (b) |
| `docs/deploy-container/serve_qwen3_awq_http.sh` | AWQ HTTP 服务脚本 | (b) |
| `docs/known-issues/qwen35-orin-nx-oom.md` | Qwen3.5 Orin NX OOM 已知问题 | (b) |
| `docs/known-issues/w8a16-talker-handoff.md` | W8A16 talker handoff 已知问题 | (b) |
| `experimental/server/launch.sh` | server 启动脚本 | (a) cand |

> 注: docs/deploy-container/* 实为 build/export/package 配方,迁移时应归 `addon/scripts/qwen3/`
> (export/build/quantize) 与 overlay 文档区,而非 runtime 镜像。

**Addon 小结**: 共 40 文件。绝大多数 (b) 自留 (产品模型/量化/并发/worker);
仅 server test scaffolding + `launch.sh` 标 (a) candidate (随 SSE/server 通用化一起上游)。

---

## 2. Patches 清单 (M = 改上游已有文件 → `patches/NNNN-*.patch`)

27 个 M 文件,按主题分组并给建议 patch 名 + 意图摘要 (读 `git diff <base>...HEAD -- <file>` 得)。
Isolation 标 CORE (改上游核心) / 边缘。

### 主题 A — `0001-orin-tegra-build-compat.patch` (build 兼容)

| M 文件 | 改了什么 | ISO |
|---|---|---|
| `CMakeLists.txt` | auto-detect Tegra (`/etc/nv_tegra_release`) → 默认 `EMBEDDED_TARGET=jetson-orin`; aarch64 保留用户 `CMAKE_CUDA_ARCHITECTURES`; CUDA driver lib HINTS | CORE 小 |
| `cmake/CuteDsl.cmake` | sm_87 Orin 重新启用 CuTe DSL GEMM; link CUDA driver lib + `--wrap=_cudaLaunchKernelEx` | CORE 小 |
| `cpp/CMakeLists.txt` | link `dl`/cuBLAS/cuBLASLt; 注册新 plugin/kernel (W8A16 / MOSS / CP) | CORE 小 |

**分类: (a) 上游化候选** (Orin/Tegra 原生 build 修复, 对所有 Jetson 用户有益)——但 `cpp/CMakeLists.txt`
里混了 (b) 私有 kernel/plugin 注册,抽 patch 时需**拆开**: build-compat 部分提 PR,私有注册留 overlay。

### 主题 B — `0002-weight-streaming-budget.patch` (Orin 统一内存 weight streaming)

| M 文件 | 改了什么 | ISO |
|---|---|---|
| `cpp/builder/builderUtils.cpp` | engine build 时 set `kWEIGHT_STREAMING` flag (Orin NX 15GB 统一内存避免一次性 pin 全权重) | CORE |
| `cpp/runtime/llmRuntimeUtils.cpp` | 新增 `parseWeightStreamingBudgetEnv` / `applyWeightStreamingBudget` (`EDGELLM_WEIGHT_STREAMING_BUDGET` env) | CORE (多为新函数) |
| `cpp/runtime/llmRuntimeUtils.h` | 上述声明 | CORE 小 |
| `cpp/runtime/llmEngineRunner.cpp` | runner 构造时 apply weight-streaming budget | CORE |
| `cpp/runtime/llmEngineRunner.h` | shared-engine 构造器 (ASR slot-pool D1) + budget 注释 | CORE |
| `cpp/runtime/eagleDraftEngineRunner.cpp` | EAGLE draft runner 同样 apply budget | CORE 小 |

**分类: (a) 上游化候选 + pr_candidate** (通用 TRT 10 weight-streaming, Jetson 统一内存通用收益)。
*注意* `llmEngineRunner.*` 里的 **shared-engine 构造器 (ASR slot-pool D1)** 是 (b) 并发逻辑,
与 weight-streaming (a) 混在同文件——抽 patch 时按 hunk 拆: budget = (a), shared-engine ctor = (b)。

### 主题 C — `0003-asr-streaming-session.patch` (流式 ASR session — 最 conflict-prone)

| M 文件 | 改了什么 | ISO |
|---|---|---|
| `cpp/multimodal/audioRunner.cpp` | chunked-prefill / MRope streaming-ASR session init (213+/112−) | CORE 重 |
| `cpp/multimodal/audioRunner.h` | streaming-ASR session 接口 | CORE |
| `cpp/multimodal/multimodalRunner.h` | `initMRopeCosSinCache` streaming-ASR session 接口 | CORE |
| `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp` | `beginAsrSession`/`endAsrSession`, session 生命周期 (629+/50−) | CORE 重 |
| `cpp/runtime/llmInferenceSpecDecodeRuntime.h` | 上述声明 + slot 状态 (333+) | CORE 重 |
| `cpp/runtime/llmRuntimeUtils.cpp` | (部分) `generateMultimodalIndices` 加 `audioIndexBase` | CORE |

**分类: (b) 自留** (产品核心: 边缘流式 ASR worker 路径)。DIVERGENCE.md 标这是 fork 里
**最易冲突的 CORE 编辑**,rebase 时重点守护。

### 主题 D — `0004-tts-slotpool-concurrency.patch` (TTS N=2 并发 / slot-pool 接线)

| M 文件 | 改了什么 | ISO |
|---|---|---|
| `cpp/runtime/qwen3OmniTTSRuntime.cpp` | per-slot 状态 ownership, SlotPool 接线, CP 输出 (618+/50−) | CORE 重 |
| `cpp/runtime/qwen3OmniTTSRuntime.h` | slot-pool 状态/接口 (149+) | CORE |
| `cpp/multimodal/code2WavRunner.cpp` | `QWEN3_TTS_CODE2WAV_PROFILE` env profiling hook + 计时 | CORE 小 |

**分类: (b) 自留** (边缘并发护城河, 绑 worker 模型, 非上游 request 路径)。

### 主题 E — `0005-customvoice-language-conditioning.patch` (CustomVoice 语言条件 9-row prefix)

| M 文件 | 改了什么 | ISO |
|---|---|---|
| `cpp/kernels/talkerMLPKernels/talkerMLPKernels.cu` | language_id codec 条件 + 9-row prefix layout; CuTe DSL disabled 时 `ELLM_CHECK` | CORE |
| `cpp/kernels/talkerMLPKernels/talkerMLPKernels.h` | 上述 + dedup language member | CORE |
| `examples/omni/qwen3_tts_inference.cpp` | per-request `language` 字段 + user→assistant role coerce | 边缘 (example) |
| `experimental/llm_loader/export_all_cli.py` | 导出 `codec_language_id` map (CustomVoice 9-row prefix) | 边缘 (export) |

**分类: (b) 自留** (CustomVoice 产品模型专属, 上游无此 checkpoint)。

### 主题 F — `0006-sse-disconnect-cancel.patch` (SSE/客户端断开崩溃修复) — **(a) 已 PR-pending**

| M 文件 | 改了什么 | ISO |
|---|---|---|
| `experimental/server/api_server.py` | `_watch_disconnect` asyncio 监视器轮询 `request.is_disconnected()`, 断开时 `channel.cancel()` → 取消 TRT stream (api_server.py 改 899+/30−, 但本主题仅其断开监视器 hunk, ~L585-957) | CORE |
| `experimental/server/engine.py` | `LLM.generate_stream()` finally 块调 `channel.cancel()` 后再 `worker.join()` (cooperative cancel) | CORE |

**分类: (a) 可上游化, 已 PR-pending** — 详见 §3。

### 主题 G — `0007-server-openai-api.patch` (OpenAI 兼容 server 增强, *混合*)

| M 文件 | 改了什么 | ISO |
|---|---|---|
| `experimental/server/api_server.py` | (其余 hunk) tool_call 早发 delta, `/v1/cache/system_prompt`, `/v1/warmup`, SSE cache metrics, MTP 启动断言 | CORE |
| `experimental/server/engine.py` | (其余) system-prompt cache, warmup KV (含 draft KV), EAGLE engine-dir fallback | CORE |
| `docs/source/user_guide/examples/experimental-server.md` | server 端点文档 | 边缘 (doc) |

**分类: 混合**。通用部分 (cache metrics `ca36892`, MTP/EAGLE robustness `87a958c` 减 tool-calling,
warmup KV reuse `12953d5`) = **(a) candidate**;tool-calling 早发 (`c952043`) 偏产品 = **(b)**。
**抽 patch 时逐 commit/hunk 拆**,不要 bulk 上游。

### 主题 H — `0008-build-misc.patch` (杂项 build/example 注册 + gitignore)

| M 文件 | 改了什么 | ISO |
|---|---|---|
| `examples/omni/CMakeLists.txt` | 注册 `qwen3_tts_streaming_worker` target | CORE 小 → (b) |
| `examples/llm/CMakeLists.txt` | 注册 spike_m1/m2/m35/m36 targets | CORE 小 → (b) |
| `.gitignore` | 加 `.DS_Store` / `._*` (macOS 噪音) | 琐碎 → (a) cand 或丢弃 |

**分类**: example/worker 注册 = (b) (随对应 addon worker);`.gitignore` macOS 行可上游或本地丢弃。

**Patches 小结**: 27 M 文件归 8 个建议 patch 主题。其中
**(a) 上游候选**: A (build-compat, 需拆私有注册), B (weight-streaming, 需拆 shared-engine ctor),
F (SSE 断开, 已 PR-pending), G 的通用子集。
**(b) 自留**: C (ASR session), D (TTS slot-pool), E (CustomVoice language), G 的 tool-calling 子集,
H 的 worker/spike 注册。

---

## 3. SSE/client-disconnect 修复定位 (= (a), 已 PR-pending)

证据 (`git show 0898b5f` + diff grep):

- **commit**: `0898b5f fix(server): cancel TRT stream channel on client disconnect`
  (Author suharvest, 2026-05-22)。
- **根因** (commit message 原文): client 中途断开 `/v1/chat/completions` (canonical trigger =
  voice-agent barge-in) → C++ runtime worker 继续生成 token → 下一请求与仍在跑的 worker
  race 共享 engine context → 崩溃 `[TensorRT] Error Code 1: Myelin (Called with an already
  loaded binary graph.)`。
- **修复涉及文件 (2)**:
  1. `experimental/server/api_server.py` — 新增 `_watch_disconnect()` asyncio task 轮询
     `request.is_disconnected()`,断开时 `channel.cancel()` 把 cancel 传进 TRT runtime;
     diff 命中行 `api_server.py:585-957` (`_watch_disconnect`, `disconnected` flag,
     `finish_reason="cancelled"`)。
  2. `experimental/server/engine.py` — `LLM.generate_stream()` finally 块在 `worker.join()`
     前调 `channel.cancel()` (cooperative, 在 streaming.cpp 解码循环迭代边界检查)。
- **处置**: category **(a)**, `pr_candidate: true`, upstream PR **pending**
  (项目 memory: 用户指派他人提交, **禁止自动提交**)。merge 后从 `patches/` 退役 + bump
  `UPSTREAM_PIN` 越过合并点。
- 与之配对的 **worker 侧 cancel** (`{"type":"cancel"}` 在 `qwen3_tts_streaming_worker.cpp`,
  commit `776cd03`) 绑产品 worker = **(b) 自留**,不随 SSE-fix 上游。

---

## 4. MOSS 专项节 — "首个 overlay 迁移范例" (category b)

MOSS-TTS-Nano 是 spec §2A 指定的首个 overlay 迁移范例 (产品模型, 上游不收)。整条链:
adapter@voxedge (`voxedge/backends/jetson/moss_tts_nano.py`) → worker/kernel/runtime@overlay
→ engine@HF → manifest@voxedge。引擎侧足迹如下,**全部 ISO 纯新增** (最干净一类,只 CMake 注册算 CORE)。

### 4.1 Addon 文件 (纯新增 → `addon/`)

| addon 路径 | 角色 |
|---|---|
| `cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.cu` / `.h` | linear-attention KV append CUDA kernel |
| `cpp/runtime/mossTtsNanoRuntime.cpp` / `.h` | MOSS 推理 runtime (1276+271 行) |
| `cpp/workers/moss_tts_nano_worker.cpp` | JSON-line 流式 worker (822 行, SentencePiece + codec_encode + chunk emit) |
| `cpp/workers/build_moss_worker.sh` | worker 构建脚本 (→ 也可归 `addon/scripts/moss/`) |
| `unittests/mossLinearKvKernelsTests.cu` | KV kernel gtest (5/5 PASS on orin-nx) |
| `unittests/mossTtsNanoSmokeMain.cpp` | runtime smoke main |

(关联 export 脚本在产品 repo `vendor/moss-tts-nano/export_*` — 见 spec §5,迁 `addon/scripts/moss/`。)

### 4.2 Patch 触点 (改上游 → `patches/`, 最小)

| M 文件 | MOSS 相关 hunk | 建议归 |
|---|---|---|
| `cpp/CMakeLists.txt` | 注册 `mossLinearKvKernels` kernel + `mossTtsNanoRuntime` 编译进 edgellmCore | 主题 A patch 的 MOSS 子 hunk |

> MOSS 的 CORE 触点**仅** `cpp/CMakeLists.txt` 一处注册 (build glob 或显式 add)。
> DIVERGENCE.md 评: "cleanest category — almost entirely isolated new files, low rebase risk"。
> 迁移做法 (spec §2 抽取法): checkout 上游@pin → 拷上述 addon 文件 → apply 一个最小
> `0004-moss-cmake-register.patch` (仅 MOSS 注册行) → build → 比对 worker 二进制名/checksum。

### 4.3 DIVERGENCE.md 条目 (见 §6 草表 `moss-tts-nano-port`)

category **b** (产品模型, 上游不收), pr_candidate **false**。

---

## 5. (a)/(b) 分流总表

| # | 主题 / 范围 | 类别 | pr_candidate | 理由 |
|---|---|---|---|---|
| A | Orin/Tegra build-compat (CMake auto-detect, CuTe sm_87) | (a) | true | 通用 Jetson 原生 build 修复; 但需拆出混入的私有 kernel 注册 |
| B | weight-streaming budget (`EDGELLM_WEIGHT_STREAMING_BUDGET`) | (a) | true | 通用 TRT10 统一内存路径; 需拆出 shared-engine ctor (b) |
| C | 流式 ASR session (audioRunner / specDecodeRuntime) | (b) | false | 产品核心边缘 ASR worker, 上游无此路径 |
| D | TTS slot-pool N=2 并发 (qwen3OmniTTS / code2Wav) | (b) | false | 边缘并发护城河, 绑 worker 模型 |
| E | CustomVoice 语言条件 9-row prefix (talkerMLP) | (b) | false | CustomVoice 产品 checkpoint 专属 |
| F | **SSE/client-disconnect 取消** (api_server + engine) | **(a)** | **true** | **上游 Myelin 崩溃 bug, barge-in 触发; PR pending** |
| G | OpenAI server 增强 (cache metrics / MTP-EAGLE robustness / warmup) | (a) 子集 | true | 通用 server robustness; tool-calling 早发子集为 (b) |
|   | └ tool_call 早发 delta | (b) | true | 偏产品 tool-calling, 但可上游候选 |
| H | example/worker/spike CMake 注册 + .gitignore | (b)/琐碎 | — | worker 注册随 addon; gitignore macOS 行可丢/可上游 |
| W8A16 | 量化 kernel/plugin/scripts (主题穿插 A/B) | (b) | false | 产品 talker AWQ W8A16 on Orin |
| MOSS | MOSS kernel/runtime/worker (§4) | (b) | false | 首个 overlay 迁移范例; 产品模型 |
| cancel | worker 侧 `{"type":"cancel"}` (qwen3_tts_streaming_worker) | (b) | false | 绑产品 worker; 与 SSE-fix 区分 |

**统计**: addon **40 文件** (38 (b) + 2 (a)-cand server scaffolding);
patches **8 主题组** (27 M 文件)。
**(a) 上游候选 X = 4 主题** (A build-compat, B weight-streaming, F SSE-disconnect, G server 通用子集);
**(b) 自留 Y = 5 主题** (C ASR, D TTS-slotpool, E CustomVoice, G tool-calling, H worker 注册)
+ 全部 W8A16 / MOSS / worker-cancel addon。

> 拆 patch 纪律 (spec §2A): A、B、G 三主题是 **(a)/(b) 混在同文件**,抽 `patches/` 时必须
> **按 hunk/commit 拆**,(a) 部分单独成 patch 提 PR,(b) 部分留 overlay,避免上游 merge 后重复 apply 冲突。

---

## 6. DIVERGENCE.md 草表 (spec §2A 增强模板, 已填)

> 迁移时提升到 overlay 根 `voxedge-engine/DIVERGENCE.md`。每条按模板填 category/pr_candidate/
> upstream_pr/retirement/rationale/scope/validation/last_replay。

```text
# voxedge-engine DIVERGENCE.md
upstream:       github.com/NVIDIA/TensorRT-Edge-LLM (Apache 2.0)
UPSTREAM_PIN:   364769036fc83351d9d0aac4cc064a8e56a83178   (= tag v0.7.1)
fork_branch:    v071/customvoice-product  (HEAD 893ba2a, 90 commits ahead)

(a)/(b) 判定准则:
  1. 上游自己代码里的正确性 bug → (a)。
  2. 我们产品/模型/路线图专属特性或性能路径 → (b)。
  3. 拿不准 → 默认 (b) + pr_candidate:true,下次 review 重评。

## orin-tegra-build-compat
category:     a
pr_candidate: true
upstream_pr:  (none yet — file after splitting private kernel-registration hunks out)
retirement:   merged → drop patch, bump UPSTREAM_PIN past <commit>
rationale:    Jetson 原生 build: auto-detect Tegra → EMBEDDED_TARGET=jetson-orin; re-enable CuTe DSL GEMM sm_87; preserve user CUDA arch on aarch64
scope:        patches=0001-orin-tegra-build-compat.patch (CMakeLists.txt, cmake/CuteDsl.cmake, cpp/CMakeLists.txt 的 build-compat hunk;私有 kernel/plugin 注册另留)
validation:   hardware (orin-nx/orin-nano build)
last_replay:  — (initial extraction)

## weight-streaming-budget
category:     a
pr_candidate: true
upstream_pr:  (none yet)
retirement:   merged → drop patch, bump UPSTREAM_PIN
rationale:    Orin 15GB 统一内存: build kWEIGHT_STREAMING flag + EDGELLM_WEIGHT_STREAMING_BUDGET env runtime budget,避免一次性 pin 全权重
scope:        patches=0002-weight-streaming-budget.patch (builderUtils.cpp, llmRuntimeUtils.{cpp,h}, llmEngineRunner.{cpp,h} budget hunk, eagleDraftEngineRunner.cpp);shared-engine ctor (slot-pool D1) 拆给 (b)
validation:   hardware (Orin NX engine build + runtime)
last_replay:  —

## sse-disconnect-fix
category:     a
pr_candidate: true
upstream_pr:  PENDING (do NOT auto-submit — user assigns filer)
retirement:   merged → drop 0006 patch, bump UPSTREAM_PIN past merge
rationale:    /v1/chat/completions client 断开 (barge-in) → worker 继续生成 → 下请求 race 共享 engine ctx → Myelin "already loaded binary graph" 崩溃。修: api_server 加 _watch_disconnect 轮询 is_disconnected → channel.cancel();engine.py generate_stream finally 协作取消
scope:        patches=0006-sse-disconnect-cancel.patch (experimental/server/api_server.py L585-957, experimental/server/engine.py generate_stream finally)
validation:   server e2e + voice-agent barge-in repro (memory: trt_edge_llm_sse_disconnect_pr)
last_replay:  — (commit 0898b5f, 2026-05-22)

## server-openai-api (mixed)
category:     a (generic subset) / b (tool-calling subset)
pr_candidate: true
upstream_pr:  (none yet — split per-hunk before filing)
retirement:   a-subset merged → drop those hunks
rationale:    OpenAI 兼容 server: cache metrics (ca36892), MTP startup assertion + EAGLE engine-dir fallback (87a958c), warmup KV reuse incl draft (12953d5) = 通用;tool_call 早发 delta (c952043) = 偏产品
scope:        patches=0007-server-openai-api.patch (experimental/server/api_server.py 余 hunk, engine.py 余 hunk, docs/.../experimental-server.md);addon=experimental/server/tests/*, experimental/server/launch.sh
validation:   experimental/server/tests/* unit
last_replay:  —

## asr-streaming-session
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    边缘流式 ASR worker 核心: chunked-prefill / MRope session, beginAsrSession/endAsrSession 生命周期。最 conflict-prone CORE 编辑,rebase 重点守护
scope:        patches=0003-asr-streaming-session.patch (audioRunner.{cpp,h}, multimodalRunner.h, llmInferenceSpecDecodeRuntime.{cpp,h}, llmRuntimeUtils.cpp audioIndexBase hunk);addon=examples/llm/spike_m1/m2/m35/m36.cpp + examples/llm/CMakeLists.txt
validation:   hardware ASR roundtrip + perf gate; spike gtest
last_replay:  —

## tts-slotpool-concurrency
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    TTS N=2 并发护城河: 通用 SlotPool<TSlot> + per-slot 状态 ownership + statefulCode2WavRunner;绑产品 worker 模型
scope:        addon=cpp/runtime/slotPool.h, cpp/multimodal/statefulCode2WavRunner.{cpp,h}, examples/omni/qwen3_tts_streaming_worker.cpp;patches=0004-tts-slotpool-concurrency.patch (qwen3OmniTTSRuntime.{cpp,h}, code2WavRunner.cpp profiling hook, llmEngineRunner shared-engine ctor)
validation:   hardware N=2 burst (30/30 0 CUDA err, audio MD5 byte-identical)
last_replay:  —

## customvoice-language-conditioning
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    CustomVoice checkpoint 专属: language_id codec 条件 + 9-row prefix layout (vs 上游 8-row)
scope:        patches=0005-customvoice-language-conditioning.patch (talkerMLPKernels.{cu,h}, examples/omni/qwen3_tts_inference.cpp per-request language, experimental/llm_loader/export_all_cli.py codec_language_id)
validation:   hardware (radxa SenseVoice ASR byte-exact transcribe)
last_replay:  —

## w8a16-quantization
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    Qwen3-TTS talker AWQ W8A16 on Orin: 量化 kernel + TRT plugin + ONNX 量化脚本 (engine 435MB, −49.5% vs FP16)
scope:        addon=cpp/kernels/w8A16LinearKernels/w8A16Linear.{cu,h}, cpp/plugins/w8A16LinearPlugin/w8A16LinearPlugin.{cpp,h}, scripts/quantize_onnx_matmul_w8a16{,_awq}.py, scripts/create_qwen3_vocoder50_wrapper.py;patches=W8A16 注册 hunk in cpp/CMakeLists.txt + cpp/runtime plugin-select wiring
validation:   hardware (radxa ASR semantic match), docs/known-issues/w8a16-talker-handoff.md
last_replay:  —

## moss-tts-nano-port   ← 首个 overlay 迁移范例
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    MOSS-TTS-Nano 产品模型移植 (linear-attention KV kernel + stateful runtime + JSON-line worker)。上游无此模型。TTFA 157ms,19× faster than ORT。最干净一类: 几乎全 ISO 新文件
scope:        addon=cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.{cu,h}, cpp/runtime/mossTtsNanoRuntime.{cpp,h}, cpp/workers/moss_tts_nano_worker.cpp, cpp/workers/build_moss_worker.sh, unittests/mossLinearKvKernelsTests.cu, unittests/mossTtsNanoSmokeMain.cpp;patches=0004-moss-cmake-register.patch (cpp/CMakeLists.txt MOSS 注册 hunk only)
validation:   unittests/mossLinearKvKernelsTests (5/5 PASS), mossTtsNanoSmokeMain; hardware 3 zh prompts CER=0, N=2 byte-identical
last_replay:  —

## worker-cancel-protocol
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    worker 侧 {"type":"cancel","id":...} 协作取消 (cancelMap + atomic flag)。绑产品 worker IPC,与上游可上游的 server 侧 SSE-fix (sse-disconnect-fix) 区分
scope:        包含在 tts-slotpool worker addon (examples/omni/qwen3_tts_streaming_worker.cpp cancel hunk)
validation:   stress 200+ early-break, audio MD5 byte-identical
last_replay:  — (commit 776cd03)
```

---

## 7. 抽取执行提示 (给后续 P5 overlay 实施者)

1. `UPSTREAM_PIN` 写 `364769036fc83351d9d0aac4cc064a8e56a83178`;`upstream.remote` = NVIDIA URL。
2. addon: `git diff --name-status <base>...HEAD | awk '$1=="A"{print $2}'` → 40 文件拷进 `addon/<path>`。
3. patches: 按 §2 八主题逐主题 `git diff <base>...HEAD -- <files> > patches/NNNN-topic.patch`;
   **A/B/G 三组混合主题必须按 hunk 拆 (a)/(b)**,否则上游 merge 后重复 apply。
4. 验证可复现: clone 上游@PIN → 拷 addon → 按序 apply patches → `build.sh` → 比对 worker
   二进制名/checksum (qwen3_asr_worker / qwen3_tts_streaming_worker / moss_tts_nano_worker /
   libNvInfer_edgellm_plugin.so)。
5. MOSS 作首例 (§4): 最少 CORE 触点 (仅 cpp/CMakeLists.txt 注册),先跑通它验证 overlay 流程。
6. (a) 类 (A/B/F/G-subset) 提 PR 后从 patches 退役 + bump pin 越过合并点。**SSE-fix PR 禁止自动提交**。
```
