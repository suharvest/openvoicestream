# 智能家居语音助手交付方案（RK3588 + Home Assistant）

> 状态：设计稿，2026-07-30。
> 标注约定：**【已有】** = 已存在且已验证；**【需新建】** = 本方案的新增工作量；
> **【实测】** = 本文档中的数字来自真机测量；**【估算】** = 推算，未验证。

## 1. 交付物概览

给客户两样东西，加一个明确的第二阶段：

| # | 交付物 | 状态 |
|---|---|---|
| 一 | 一组开箱可跑的镜像 + 一个 compose：包含全部语音能力（ASR / TTS / LLM / 对话编排） | 部分已有，见 §4 |
| 二 | Home Assistant 集成示例 + 开发说明：如何用我们的框架控制智能家居 | **【需新建】**，见 §5 |
| 第二阶段 | Wyoming 协议适配层，让我们的 ASR/TTS 直接插进 HA Assist | 设计见 §7 |

**给客户的沟通口径**：先基于我们本地这套（我们当大脑）跑起来验证效果，同时告知 Wyoming 适配我们也在做 —— 两条路不是替代关系，是**互补**，客户可以先用前者验证体验，后续按自己的 HA 使用深度选择迁移或并存。

## 2. 两种集成形态，为什么先推第一种

| | 形态 A：我们当大脑（**先推**） | 形态 B：HA 当大脑（Wyoming，第二阶段） |
|---|---|---|
| 谁做意图理解 | 我们的 LLM，自由对话 | HA Assist 的 intent 匹配 |
| HA 的角色 | 执行器（被我们通过 REST API 调用） | 大脑；我们退为 ASR/TTS 供应商 |
| 客户已有 automation | 用不上（除非显式暴露成工具） | **全部保留** |
| 我们的差异化 | 全部发挥：本地 LLM、878ms 嘴到耳、真流式、自由对话 | 流式优势**同样能发挥**（见 §7），但意图能力交给 HA |
| 适合谁 | 想要"能聊天的音箱"、HA 用得浅 | HA 深度用户，automation 是资产 |

先推 A 的理由：能立刻展示我们的核心能力（低延迟 + 本地 LLM + 自由对话），而 B 更像"把我们变成一个更好的零件"。

## 3. 架构：为什么 HA 工具必须走 remote tool

这不是选择，是当前实现的约束：

- 服务端（voxedge 引擎）**没有**"直接注册一个外部 HTTP 工具"的通路。所有工具执行路径都假定**执行方是设备客户端**。
- 因此 HA 的 REST 调用必须由一个**设备侧客户端**发起，该客户端通过 `tool_advertise` 把工具 schema 广告给服务端 LLM 循环，服务端选中后用 `tool_call` 把执行代理回客户端（协议详见 `docs/api/v2v-stream.md` 的 Remote tools 一节）。
- **已有生产先例**：`voice_rebot_arm` 就是这个形状（server-loop 客户端 + advertise 工具控制机械臂）。HA app 是同一模式换个执行目标，**不是新架构**。

```
  麦克风 ──┐
           │  ws://voice:8621/v2v/stream
   ┌───────▼────────────┐   tool_advertise / tool_call / tool_result
   │  agent (HA app)    │◄──────────────────────────────┐
   │  · 麦克风 / 扬声器  │                               │
   │  · HA 工具实现      │──── HTTP ───► Home Assistant  │
   └────────────────────┘      REST API   (客户已有)     │
                                                        │
   ┌────────────────────────────────────────────────────▼──┐
   │  voice service (:8621)                                │
   │  · Qwen3-ASR  (RK3588 NPU: RKNN encoder + RKLLM 解码)  │
   │  · matcha TTS (Vocos 在 NPU / 声学模型 ORT CPU)        │
   │  · V2V server loop（对话编排 + 工具循环）              │
   └───────────────────────┬───────────────────────────────┘
                           │ OpenAI 兼容 /v1/chat/completions
                  ┌────────▼─────────┐
                  │  LLM (:1828)     │  Qwen3-4B @ 8K ctx
                  │  RK1828 加速卡    │  （PCIe NPU，独立 5GB）
                  └──────────────────┘
```

## 4. 交付物一：镜像组

**固定为我们验证过的配置**：ASR = Qwen3-ASR，TTS = matcha。不提供后端选择，减少客户踩坑面。

### 4.1 铁律：模型产物永不进镜像

镜像只含代码 + 运行时库，模型按客户配置在**首次启动时拉取**到 named volume。

**【已有】** 这个机制已经在跑：`Dockerfile.rk` 的分层注释原文就是 `NO model/engine/voice artifacts`，配合 compose 里的：

```yaml
RK_ARTIFACT_AUTO_DOWNLOAD=1
RK_ARTIFACT_REPO_ID=harvestsu/seeed-local-voice-rk-artifacts   # HF 仓库
RK_ARTIFACT_MANIFEST=/opt/speech/deploy/artifacts/rk_manifest.json
RK_ARTIFACT_SET=rk3588-multilang-2026-05-17                     # 按配置选产物集
```

新增的 LLM 镜像照同一套做，不发明第二种机制。

### 4.2 三个镜像

| 镜像 | 内容 | 状态 |
|---|---|---|
| `seeed-local-voice:rk-*` | ASR + TTS + V2V server loop | **【已有】** 需重烘一次以固化两个修复（见 §4.3） |
| `edge-llm-rk1828:*` | RKNN3/RKLLM3 运行时 + server 模式 binary + OpenAI 兼容 shim | **【需新建】** —— 当前只有 Jetson 的 `edge-llm-chat-service`，RK 侧完全没有 |
| `ovs-agent:rk-*` | agent + HA app（麦克风/扬声器 + HA 工具） | **【需新建】** —— 只有 Jetson 的 agent overlay，RK 侧没有 |

客户视角：一个 `docker compose up -d`，三个服务。

### 4.3 语音镜像必须重烘的两个修复

当前生产设备上这两个修复是 **bind-mount 的热补丁**，不在任何镜像里 —— 不固化则下次重烘静默回退：

1. **matcha vocos 加共享 NPU 锁**。ASR 的 RKLLM 解码器以 `npu_core_num=3` 占满三个核，而 matcha 的 Vocos 钉在 core 0 且原先不取锁 → 核隔离救不了，必须加锁。
2. **TTS 文本路径剥离 Markdown**。LLM 回答语音轮次时仍会输出 Markdown，句子切分器会切出 `**`、`：**` 这类纯标记碎片，后端只能返回 0.1 秒静音（实测 40 分钟内 72 次）。

配套的部署侧参数（属于同一个修复的另一半）：

```yaml
ASR_NPU_CORE_MASK=NPU_CORE_1     # 绝不能用 NPU_CORE_AUTO
VAD_ENDPOINT_SILENCE_MS=400
mem_limit: 12g                    # 16GB 板子，7500m 是从 8GB RK3576 profile 继承来的
OVS_V2V_ENGINE=voxedge            # 与下一行缺一不可
OVS_V2V_SERVER_LOOP=1
EDGE_LLM_BASE_URL=http://127.0.0.1:1828/v1
EDGE_LLM_MODEL=Qwen3-4B
OVS_V2V_SYSTEM_PROMPT=...         # 不设则 LLM 按 chat 助手回答：~330 字 Markdown → 87 秒音频
OVS_V2V_LLM_MAX_TOKENS=96
```

### 4.4 LLM 镜像设计

| | |
|---|---|
| 镜像内 | RKNN3/RKLLM3 V1.0.4 运行时（**与语音镜像的 RKNN2 是两套，不可共用**）+ server 模式 `rknn_qwen3_demo` + FastAPI OpenAI shim |
| 运行时拉取 | `Qwen3-4B.rknn` / `.weight` / `.embed.bin` / `.tokenizer.gguf`（共 ~3.2GB，**四个文件是配对产物，必须整套用**） |
| 配置项 | `RK1828_MAX_CONTEXT`（默认 **8192**）、`RK1828_CORE_MASK=0xff`、端口 1828 |
| 暴露 | `/v1/chat/completions`（SSE 流式）、`/v1/models`、`/health` |
| 需要 | `privileged` + 挂 `/dev/pcie-rkep-0001:11:00.0` |

**已知实现要点**（做的时候别重新踩）：
- stdout 只许有协议帧，init 前先 `dup(STDOUT)→frame_fd` + `dup2(stderr, stdout)`
- 每个请求后无条件 `RKNN3_KVCACHE_CLEAR_ALL`，含错误路径（否则脏 KV 累积溢出 max_ctx）
- SSE generator 里**不要持 worker 锁** —— 客户端中途断开会让锁泄漏 + 帧流 desync；要由专门线程持锁并无条件 drain 到 EOS

### 4.5 宿主机前提（必须写进交付文档）

**镜像不是自包含的** —— RK1828 的驱动和固件在宿主机：

1. `pcie_rkep.ko` 需装载；`rknn3.service` 在开机时向 EP 重刷固件。容器只能访问一张**已被宿主机初始化好**的卡。
2. **`.ko` 重启会失效，需要做持久化**（已知坑）。
3. 验收判据：`lspci` 能看到 `[1d87:182a]`、`systemctl is-active rknn3.service` 为 active、`/dev/pcie-rkep-0001:11:00.0` 存在。

→ 交付需附一个**宿主机 bringup 脚本**，不能只给镜像。

### 4.6 单 EP 资源约束

RK1828 只有一个 5120 MB 上下文，**大模型互斥**。Qwen3-4B @ 8192 ctx 占用【估算】约 3,640 MB（权重 2,432 MB + KV 8192×144KB=1,208 MB）= 71%。含义：**这张卡给了 LLM 就不能同时跑别的大模型**（例如 RK1828 版的 TTS）。这也是为什么 TTS 走 RK3588 自己的 NPU 而不是这张卡。

## 5. 交付物二：HA 集成

**现状：零 HA 支持** —— 全仓库 `homeassistant` / `hass` 零命中。全部新建。

### 5.1 新增 app：`agent/ovs_agent/apps/home_assistant/`

沿用现有 app 结构（`companion_robot` 是最小样板）：

```
home_assistant/
  app.py         # 继承 MultiModeApp，注册 HA 工具
  config.yaml    # slv_url / llm / system_prompt / HA 连接参数
  ha_tools.py    # @tool 实现，调 HA REST API
  README.md      # 开发说明（交付物的核心）
```

### 5.2 工具集设计

用 `@tool` 装饰器，从 Python 类型标注自动生成 OpenAI schema（不用手写 JSON Schema）：

| 工具 | 作用 | 备注 |
|---|---|---|
| `list_devices(area?)` | 列可控设备 | 必须**过滤 + 摘要**，不能把几百个 entity 全塞进 prompt |
| `turn_on(device)` / `turn_off(device)` | 开关 | 中文名 → entity_id 的映射在工具内做 |
| `set_brightness(device, percent)` | 调光 | |
| `set_temperature(value)` | 温控 | |
| `get_state(device)` | 查状态 | |
| `call_service(domain, service, entity_id, data?)` | 通用逃生口 | 给高级用户，schema 宽松 |

**设计要点**：
- **实体命名是成败关键**。LLM 看到的是工具 schema 和描述，不是 HA 的 entity_id。要在 app 侧维护「口语名 ↔ entity_id」映射（从 HA 的 `friendly_name` 和 area 自动构建 + 允许 config 覆盖）。
- `preamble_text`（如"好的，正在开灯。"）—— 外部 API 有明显耗时时先出声，避免用户以为没听见。
- `response_mode`：开关类用 `template`（跳过 LLM 第二轮，延迟最低）；查询类用 `await`。

### 5.3 五个已知会绊到人的坑（写进 README）

| 坑 | 说明 |
|---|---|
| `httpx` 必须 `trust_env=False` | 否则走系统代理，够不到局域网 HA（已踩过两次） |
| 结果字段是 `ok` 不是 `success` | 帧级字段名 |
| `call_id` 要先读 `call_id` 再读 `id` | 先读 `id` 曾致 15 秒卡死 |
| **必须压在 15 秒内** | `timeout_s` 实际没被 advertise 出去，服务端用 15s 兜底；客户端自己默认 30s，会超过服务端耐心 |
| 没开 server loop 时 advertise 静默失效 | 只打 warning，不回错误帧 |

### 5.4 验证方式

**已在做**：起一个**真的 HA 实例**（Docker）+ 长效 token + 若干可控实体，让交付给客户的示例代码是**真验证过的**，而不是"看起来能跑"。交付物含一份可复现的验证记录。

## 6. 实测基线（可写进计划书的数字）

全部【实测】于 Radxa ROCK 5T（RK3588 + RK1828），`/v2v/stream` server-loop，中文短句，n=5：

| 指标 | 值 |
|---|---|
| **嘴到耳（停止说话 → 首个音频字节）p50** | **878–895 ms** |
| ├ ASR final | 340 ms |
| ├ LLM（TTFT + 到首个小句） | 277 ms |
| └ 首句合成 | 260 ms |
| LLM TTFT / 吞吐（Qwen3-4B @ 8192 ctx） | **132 ms / 81.5 tok/s** |
| ASR + TTS NPU 共驻 40 分钟 | **零 RKNN 故障** |
| 说话打断（真重叠）3 次 | 3/3 重叠成功且打断句识别正确，零故障 |

## 7. 第二阶段：Wyoming 适配层

### 7.1 关键事实更正：Wyoming **是流式的**，两个方向都是

（早期判断有误，已查协议源头更正。）

**ASR（我们 → HA）**：`transcript-start` / `transcript-chunk`（`text`：部分转写结果）/ `transcript-stop`；`transcript` 保留作向后兼容。
**TTS（HA → 我们）**：`synthesize-start` / `synthesize-chunk`（`text`：**要合成的文本的一部分**）/ `synthesize-stop`；我们回 `synthesize-stopped` 表示最终音频已发完，音频本体走 `audio-chunk`。

**能力声明字段**（HA 靠它决定走不走流式）：
- ASR program：`supports_transcript_streaming`（bool，"program can stream transcript chunks"）
- TTS program：`supports_synthesize_streaming`（bool，"program can stream text chunks"）

### 7.2 为什么这对我们特别有利

`synthesize-chunk` 的语义是「文本的一部分」→ **HA 会把 LLM 的 token 增量流给我们**，我们边收文本边出音频。这跟我们已经在跑的模型**完全同构**：

| 我们已有 | Wyoming 事件 |
|---|---|
| `asr_partial`（true_streaming） | `transcript-chunk` |
| `asr_final` | `transcript-stop` / `transcript` |
| `LowLatencyTTSBuffer` 小句切分（CJK 15/24/40 字，默认已开） | `synthesize-chunk` 流入，按小句合成 |
| 小句 PCM 增量下发 | `audio-chunk` |

**结论**：走 Wyoming **不会**让我们退化成"非流式的 ASR/TTS 供应商" —— 260 ms 的合成段、310 ms 的 ASR final、小句级 TTFA 都能带过去。这让形态 B 从"降级方案"变成真正可竞争的形态。

### 7.3 实现形状

**独立的薄适配容器**，不塞进语音镜像：
- 零侵入，语音镜像不动，两条路可并存
- 对内调我们的 **WS `/v2v/stream`**（流式）而**不是** HTTP `POST /asr`（一次性）—— 否则外面流式、里面阻塞，白瞎
- 必须实现**流式变体**并在 `info` 里声明上述两个 `supports_*` 字段；同时保留非流式 `transcribe` / `synthesize` 以兼容旧版 HA，靠 `info` 协商

只做 ASR 和 TTS 两类服务。唤醒词不做（HA 侧已有 openWakeWord）。

### 7.4 仍待查证

- HA 侧从哪个版本开始发 `synthesize-chunk`，旧版兼容策略的具体分支条件。

## 8. 前提、风险与未验证项

| 项 | 状态 |
|---|---|
| RK1828 加速卡 | **本地 LLM 的硬性前提**。没有这张卡则 LLM 必须走外部（局域网另一台或云端），语音能力不受影响 |
| 宿主机驱动/固件 | 见 §4.5，必须附 bringup 脚本；`.ko` 持久化是已知坑 |
| LLM 模型产物托管 | 正在上传到我们自己的 HF 仓库。**不采用第三方仓库**（`t-firefly/qwen3-4b-rknn3-rk1828` 字节不同、来源无文档、且会让客户依赖第三方账号），仅留作交叉验证 |
| `rknn-smi` 整机失效 | RK1828 的显存/健康**无任何观测手段**（root 也失败，疑固件版本 skew）。只能靠"模型加载成不成"这个二值信号。已知限制，需报上游 |
| NPU 利用率 | RK3588 侧 devfreq `load` 恒为 100（空闲也是），同样无可用观测面 |
| 16K 上下文 | 【估算】占 EP 92%，未验证。8192 已【实测】通过 |
| 多轮 + server-loop 的压测工具 | 现有 harness 在该组合下会报误导性的空转写，已加显式拒绝守卫；需要独立修 |

## 9. 里程碑建议

| 阶段 | 内容 | 阻塞项 |
|---|---|---|
| M1 | LLM 产物上传 HF + LLM 镜像封装 | 进行中 |
| M2 | 语音镜像重烘（固化两个修复） | 依赖代码合并 |
| M3 | `home_assistant` app + 工具集，对真 HA 实例验证 | HA 测试实例（进行中） |
| M4 | agent 镜像（RK）+ 三服务 compose + 宿主机 bringup 脚本 | M1–M3 |
| M5 | 开发说明文档（客户照着能加自己的工具） | M3 |
| M6（第二阶段） | Wyoming 适配容器 | 独立，可并行 |
