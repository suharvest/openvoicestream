# TensorRT-Edge-LLM v0.9.0 — C++ TTS runtime/worker re-port spec（P2）

> codex 独立设计（2026-07-03，session 019f279e-0d54-7ea3-aeeb-82adce18de7b），主线程审阅收录。
> 基线：我们的 `integration/v080-sparktts`（v0.8.0 + 30 commits）→ 上游 tag `v0.9.0`（`1ac0f2b`）。
> Python 量化侧已单独 rebase 完成：fork 分支 `integration/v090-sparktts` @ `d74e35b`（22 picks + `mixed_precision`→`bf16_residual` 改名）。本 spec 只覆盖 C++ 侧剩余 8 个 patch 的处置。
> ⚠️ 待 P0 orin-nx 真机 spike 实证交叉验证后定稿（尤其 ①②④ 三个结论）。

## A. API 映射

- **构造**：旧 worker slot0 路径构造、slot1 借 `ICudaEngine*` 构造（`integration/v080-sparktts:examples/omni/qwen3_tts_streaming_worker.cpp:773,:779`；旧 ctor `cpp/runtime/qwen3OmniTTSRuntime.h:96,:116`）。v0.9.0 只有路径构造（`v0.9.0:cpp/runtime/qwen3OmniTTSRuntime.h:100`，实现 `.cpp:137`），内部改 `EngineExecutor/StepPreparer/TensorMap`（`.cpp:278-285,:308-316`）。共享 ctor 需 RE-PORT。
- **submit**：旧 `ttsRuntime.handleAudioGeneration(request, response, stream)`（worker `:637-638`）→ v0.9.0 **同名 API 保留**（`v0.9.0:...h:195-202`）。调用形状可保留。
- **流式取数**：旧 `codecChunkFrames/subsequentChunkFrames/onAudioChunkReady(batchIdx,isFinal)`（旧 h `:171-180`；worker `:607-633`）→ 上游原生 `streamingChunkFrames/onChunkReady(codes,isFinal)`（`v0.9.0:...h:135-145`），runtime 按 batch 装 handlers（`.cpp:1328-1349`），final flush `isFinal=true`（`:1807-1812`）。**上游原生覆盖 chunk 回调；首包/后续差异化 chunk size 未覆盖**。
- **cancel**：旧 `cancelMap`+`shouldCancel`（worker `:301-333,:609`）。v0.9.0 grep 无 cancel 机制 → REDESIGN。
- **slot 管理**：旧 `SlotPool` CAS acquire/release（`slotPool.h:22-39,:139-181`），饱和返回 4429（worker `:920-928`）。v0.9.0 无 worker/slotPool，仅单 runtime 内 per-batch state（`v0.9.0:...h:394-404`）。

## B. 逐功能处置

| # | Patch | 处置 | 要点 | 预估 |
|---|---|---|---|---|
| 1 | `10b338d` streaming worker | RE-PORT worker/slotPool；runtime 内 chunk 回调手术 **DROP**（改用原生 `streamingChunkFrames/onChunkReady`）。overlap vocoder/slot worker 在 worker 侧重做（旧 `:575-602`） | 350-500 行 |
| 2 | `ba9ecdb` 外部 speaker embedding | RE-PORT。`speakerName/speakerId` 只解析为 token id（`v0.9.0:...cpp:1151-1164,:2140-2150`），**不能承载任意向量**。落点：`TalkerGenerationRequest` 加 `speakerEmbedding`；`projectToTalkerInput` 加参（新签名 `...h:690-691`）；`invokeAssistantPreamble` 加外部 row-6 源（旧证据 `integration:...cpp:1017-1043`） | — |
| 3 | `12ee383`+`c48c0de` CV 9-row | RE-PORT。v0.9 kernel 仍固定 8-row（`v0.9.0:cpp/kernels/talkerMLPKernels.h:97-121`；`.cu:541-565`）。request 加 `language`、config 读 `codec_language_id`、`prepareTalkerInput` 解析 lang（旧 `.cpp:1299-1325`）、`projectToTalkerInput` 算 9-row（旧 `:990-1043`） | 120-180 行 |
| 4 | `873ca22` fp8 text_embedding | RE-PORT。v0.9 直接取 `textEmbedTensors[0]` 无 FP8 scales 分类（`v0.9.0:...cpp:201-214`）；旧分类/scales `integration:...cpp:172-199`，lookup `:957,:1285` | 80-120 行 |
| 5 | `8de933f` N<=0 guard | RE-PORT。v0.9 只 guard `seqLen==0`（`.cpp:1138-1148`）；旧 guard `integration:...cpp:1000-1009` | ~10 行 |
| 6 | `21119ec` shared-engine ctor | RE-PORT/重设计共享方式。v0.9 只有 `unique_ptr<EngineExecutor>`（`...h:571-585`），需新增借用/共享只读 engine 入口；**待验证 EngineExecutor 是否暴露底层 engine** | 120-220 行 |
| 7 | `64185fa` cancel 协议 | REDESIGN。保留 worker `cancelMap`；v0.9 request 增 `shouldCancel` 或 decode loop 每帧检查；取消时标记 batch finished → emitter `flushFinal()` → `isFinal=true` 作流结束 → worker 发 `cancelled`。插入点 decode loop（`.cpp:1770-1796`）+ final flush（`:1807-1812`） | — |

## 四个关键问题结论

1. **上游 per-batch 独立流 ≠ 我们的 per-slot N=2**。上游是单 `handleAudioGeneration` 调用内的 batch 隔离（per-batch state `...h:394-404`、per-batch handler `:423-441`、batched prefill stash/repack `.cpp:1236-1274`）；我们是多 runtime/stream/thread/code2wav 的 worker 级并发隔离（worker `:381-399`）。**上游覆盖 batch 内流，不覆盖 worker 并发槽 → SlotPool 保留。**
2. **`speakerName/speakerId` 不能承载 voice-clone 任意 embedding**（预置 token/name 映射）。注入点 = `projectToTalkerInput` 调 `invokeAssistantPreamble` 前（`.cpp:1162-1164,:828-830`）。
3. **9-row 插入点** = standalone TTS 的 `prepareTalkerInput→projectToTalkerInput→invokeAssistantPreamble` 链（`.cpp:1213-1227,:807-830`）；Omni segment 路径另有 `buildTalkerPrefillFromSegments`（`:2339-2341`），旧代码未传播 language（`integration:...cpp:2383-2387`）。
4. **worker 协议仍传文本，但必须包成 `messages`**（`...h:129-130`；runtime 内部 `applyChatTemplate+encode` `.cpp:1213-1223`）。旧 worker 本来就包 assistant Message（worker `:365-374`），改动小。不要直接喂 token；要喂 token 只有 Omni API 的 `textTokenIds`（`...h:213-215`）。

## C. Milestones（orin-nx 验证序）

- **M1 最小 worker 跑通**：worker/slotPool/CMake 用原生 `streamingChunkFrames/onChunkReady`；`max_slots=1` 文本输入。验收：ready/chunk/is_final/done 协议齐、音频能量校验可播放、零 runtime 改动。
- **M2 runtime 小补丁**：外部 embedding、9-row、N<=0、FP8（`qwen3OmniTTSRuntime.{h,cpp}` + `talkerMLPKernels.{h,cu}`）。验收：Base speaker_embedding_b64、CustomVoice language、空文本失败、FP8 加载各一条冒烟。
- **M3 并发/取消/共享权重**：shared EngineExecutor ctor、worker cancel、N=2。验收：两路并发不串音；cancel_ack→cancelled 且末 chunk `is_final=true`；显存低于双反序列化；连续 50 次无死锁。

## P0 真机实证补充（2026-07-03，orin-nx spike，spec 据此定稿）

- **CuTe-DSL 阻断（JP6.2/CUDA 12.6，仅纯上游）**。纯上游 0.9.0 talker MLP 硬依赖 CuTe-DSL GEMM 无回退（`talkerMLPKernels.cu:341`；`ENABLE_CUTE_DSL=OFF` 时 MLP 是 no-op → 1 帧 EOS 假成功）；预编译 sm_87 DSL 产物为 CUDA 13 构建，r540 驱动 `cuLibraryLoadData→200` 拒载。**我们的历史修复已在 P1 合入 v090 分支**：`e024b2c`（原 26a4a69，自研 tiled FP16 GEMM 路由 `invokeTalkerMLP/invokeLinearLayer` 的 #else 分支，OFF 可用、零 cuBLAS 依赖、生产验证过）+ `07d3f4f`（M=1 GEMV）+ `d18bfa3`（cudart shim）。**首选路径 = ENABLE_CUTE_DSL=OFF + 自研 kernel**；M1 需 grep 确认 0.9.0 无绕过这两个入口的新 CuTe 调用点（有则补路由）。spike 的 cuBLAS ABI 替换（`orin-nx:~/project/spike-v090/shim/cublas_gemm_override.cu`，CUTE_DSL=ALL + 链接替换 `_mlir_gemm_ampere_*`）仅作备选，两方案解决同一问题勿并存。
- 四问实证：①流式跑通（`--streaming --chunkFrames`，TTFC≈21ms，chunk=1 时 TTFPA 115ms，RVQ 与非流式 bit-exact）②batch=2 真独立（batched prefill+joint decode+per-batch EOS，实测两路交错；但采样参数取 requests[0]、静态 batch 无 continuous batching → SlotPool 结论不变）③speaker 预置名单实锤（config.json speaker_id map，未知名 fallback 默认）④messages 唯一入口实锤，`apply_chat_template:false` 不可用（无条件剥 3 头+5 尾 token）。
- 质量红旗：上游流式 CLI WAV 拼装饱和削波（peak=32767，疑 fp16 缩放 bug；RVQ 正确 → 我们 worker 自己的 PCM 路径需自行能量验证，勿用上游 CLI 拼装）；0.8.0 ONNX × 0.9.0 runtime 英文生成长度失控（521~4066 帧漂移）→ **P3 必须用 0.9.0 导出链重导 ONNX，跨版本引擎仅限机制验证**。
- code2wav 引擎构建需 ~5GB 空闲显存：16GB Orin NX 上先 `docker stop edge-llm-chat-service`（完事 start 回来）。
- spike 产物可复用：引擎 `orin-nx:~/project/spike-v090/engines/`（talker b2/cp/code2wav，CV 0.6B），M1 冒烟用 ZH 文本（该引擎 ZH 行为正常）。

## 遗留（不在本 spec，另行处理）

- SSE disconnect fix（`49c94ff`）：对 v0.9.0 `experimental/server` 重新实现（上游仍无 disconnect 处理），可顺势准备上游 PR（用户派人提交，禁自动提交）。
- MOSS-TTS-Nano worker（`3c6c263`）：自建文件重编适配新 Tensor/runtime util API。
- P1 交接项：examples/omni/CMakeLists.txt 的 streaming worker 构建块随 #1 一起补回；旧 checkpoint config.json 的 `mixed_precision` key 需改 `bf16_residual`（P3 导出前）。
