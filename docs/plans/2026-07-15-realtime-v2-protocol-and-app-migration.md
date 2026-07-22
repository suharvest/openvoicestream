# Realtime V2 统一协议与应用迁移计划

日期：2026-07-15

状态：`feat/realtime-v2-protocol` 实施中；本地纵向链路完成，云 relay 已进入可测试阶段

## 0. 当前实施进度（2026-07-16）

已完成第一批：

- `/v2v/stream` 通过 `seeed.realtime.v2` WebSocket subprotocol 协商 V2，未协商的
  连接迁移期继续走 V1。
- `session.created → session.update → session.updated` 握手、binary PCM 格式协商，
  以及 server-managed response 能力的真实校验。
- 本地级联和 voxedge 路径统一输出 input-buffer、transcription、
  `response.created`、`response.output_audio.done`、`response.done`。
- 共享 Agent 默认使用 V2，并把 wire lifecycle 收敛为稳定的应用 hook；
  `on_assistant_done` 等待 response 终止和本地 playback drain。
- 浏览器公共客户端、TTS-only 示例和 reBot compose 已接入 V2。
- 已补协议单元测试、真实 WebSocket 握手、VAD/转写、取消状态和播放 drain 回归。

已完成第二批：

- 工具 wire 已对齐 canonical function-call lifecycle；voxedge 内部旧事件只存在于
  Gateway adapter，不再暴露给 V2 应用。
- 新增原子 `x_v2v.response.speak`，替代应用依赖 `text + tts_flush` 两帧组合。
- Agent 和浏览器在打断时统计实际已播放 PCM 时长，并发送
  `conversation.item.truncate`；本地级联明确确认安全 no-op。
- 新增 `x_v2v.conversation.reset`，共享 Agent 同时清理本地 Session 和 Gateway 状态。

已完成第三批骨架：

- 根据 2026-07 官方协议把控制面更新为 OpenAI Realtime GA 当前字段：
  `output_modalities`、`audio.*.format.rate` 和扁平 function tools；旧形状仅保留输入兼容。
- 新增 OpenAI/Qwen provider adapter 与 Gateway 双向 WebSocket relay：binary PCM ↔
  provider Base64 audio、session/event 映射、OpenAI 16→24 kHz 输入重采样。
- compose 已支持仅通过 `OVS_REALTIME_PROVIDER=local|openai|qwen` 切换后端，云端
  凭证只进入 Gateway，不进入机器人应用。
- capability 会真实声明云端差异；当前不会把 generative prompt 冒充 direct speak，
  也不会把 Qwen 未确认的 truncate 宣称为已支持。

仍在后续批次：

- OpenAI/Qwen 带真实凭证的连通性、长会话和故障注入验收。
- 云模式 deterministic direct-speak TTS side channel，以及 Qwen truncate/reset
  能力的供应商版本验证或 Gateway 历史模拟。
- Reachy Voice、SO-ARM 外部仓库的依赖/镜像/preset 升级及真机验收。

## 1. 目标

把现有 `/v2v/stream` 从本地 ASR/TTS 专用事件协议升级为统一的
Realtime V2 协议，使同一个设备应用只连接 Seeed V2V Gateway，通过服务端配置
切换以下后端，而不改麦克风、播放器、UI、机器人动作和工具执行代码：

- 本地级联：ASR → text LLM → TTS
- OpenAI Realtime 端到端语音模型
- Qwen Realtime 端到端语音模型

协议不是逐字复制某个云厂商的 WebSocket，而是采用三层结构：

1. **统一控制面**：使用 OpenAI Realtime GA 风格的 session、input buffer、
   conversation item、response 和 tool 生命周期。
2. **边缘音频数据面**：继续用 WebSocket binary 传输 PCM，避免 Base64 的带宽和
   CPU 开销。
3. **Provider adapter**：本地级联、OpenAI、Qwen 的事件名、音频格式、VAD、工具
   和取消行为全部在 Gateway 内归一化。

应用不能按 `provider` 写业务分支；只能根据 `session.capabilities` 做能力降级。

## 2. 关键协议决策

### 2.1 `tts_done` 不是简单改名

当前 `tts_done` 同时表达“服务端不再生成音频”和“本轮回复完成”。V2 拆成：

1. `response.output_audio.done`：不会再发送本 response 的音频字节。
2. `response.done`：整个 response 进入
   `completed | cancelled | failed | incomplete` 终态。
3. 本地 playback drained：设备扬声器队列真正排空；这是 Agent 本地状态，不是
   provider 事件。

`on_assistant_done` 继续作为稳定的应用层 hook，但其定义固定为“response 已终止，
且启用 playback drain 时本地音频也已经播完”。机械臂提示音、清历史、自动休眠
都应依赖这个 hook，而不是直接依赖 wire event。

### 2.2 推荐事件映射

| 当前事件/行为 | Realtime V2 | 备注 |
|---|---|---|
| `config` | `session.update` | 服务端先发 `session.created`，应用等待 `session.updated` |
| binary 上行 PCM | 隐式 `input_audio_buffer.append` | 音频格式来自 session，不再藏在 frame 中 |
| `asr_eos` | `input_audio_buffer.commit` | 手动提交一个输入 item |
| 无 | `input_audio_buffer.clear` | 清除未提交输入 |
| `vad_event:speech_start/end` | `input_audio_buffer.speech_started/stopped` | 作为打断和 UI 的低延迟信号 |
| `asr_endpoint` | `input_audio_buffer.committed` | 明确 input item id |
| `asr_partial` | `conversation.item.input_audio_transcription.delta` | 转写是辅助信息，不是原生模型理解的唯一真值 |
| `asr_final` | `conversation.item.input_audio_transcription.completed` | 用 item/event id 去重 |
| 无 | `response.created` | 建立 response 生命周期和相关 ID |
| binary 下行 PCM | 隐式 `response.output_audio.delta` | 保留 binary PCM |
| `tts_sentence_done` | `x_v2v.tts_sentence_done` | 可选优化事件，核心状态机不得依赖 |
| `tts_done` | `response.output_audio.done` + `response.done` | 分离音频完成和回复完成 |
| `abort` | `response.cancel` | 指定 `response_id` |
| 无 | `conversation.item.truncate` | 按实际播放毫秒数裁掉用户未听到的回复 |
| `tool_advertise` | `session.update.session.tools` | 本地扩展字段放 `x_v2v` |
| `tool_call` | function-call output item / arguments events | 保留 `call_id` |
| `tool_result` | `conversation.item.create(function_call_output)` | 再按模式触发 `response.create` |
| 字符串 `error` | 结构化 `error` | 包含 code、message、param、event_id |

### 2.3 Server-loop 变成协议语义

不再让应用通过隐式环境变量决定“谁运行 LLM”。使用 session 配置：

```json
{
  "type": "session.update",
  "session": {
    "type": "realtime",
    "output_modalities": ["audio"],
    "audio": {
      "input": {
        "turn_detection": {
          "type": "server_vad",
          "create_response": true,
          "interrupt_response": true
        }
      }
    }
  }
}
```

- `create_response: true`：Gateway 自动产生回复；适合本地 server-loop 和云端端到端。
- `create_response: false`：只提交/转写输入，客户端显式 `response.create`；适合需要在
  回复前注入视觉上下文的 Reachy。
- `interrupt_response: true`：speech started 自动取消当前 response。

现有 `OVS_V2V_SERVER_LOOP` 和 `OVS_AGENT_SERVER_LOOP` 只作为迁移期配置输入，最终
收敛为 session 语义，避免服务端和 Agent 两边开关不一致。

### 2.4 音频格式和 binary 约束

删除“首个下行 binary frame 的 4-byte sample-rate header”。采样率、声道、位深和
字节序放在 `session.created/session.updated`：

```json
{
  "audio": {
    "input": {"format": {"type": "audio/pcm", "rate": 16000,
                           "channels": 1, "endianness": "little"}},
    "output": {"format": {"type": "audio/pcm", "rate": 24000,
                            "channels": 1, "endianness": "little"}}
  }
}
```

裸 binary frame 没有 `response_id`，V2 初版必须保证每个 WebSocket 同时最多一个
active audio response。若未来允许并发，需要增加 binary envelope，至少携带
stream id、sequence 和 timestamp。

### 2.5 打断必须同步上下文

标准流程：

1. `response.cancel(response_id)`。
2. 立即清空/锁死旧 generation 的本地播放队列。
3. 上报实际播放位置：
   `conversation.item.truncate(item_id, content_index, audio_end_ms)`。
4. 最终只接收一个 `response.done(status=cancelled)`。

否则云端 conversation 会认为用户听到了实际没有播放的后半段回复。

### 2.6 会话、工具和关联 ID

所有事件至少支持：

- `event_id`
- `session.id`
- `response.id`
- `item.id`
- `call_id`
- `output_index`
- `content_index`

工具列表通过 `session.update` 下发；工具变化后再次更新 session。工具执行结果使用
`function_call_output` conversation item。现有 `timeout_s`、`preamble_text`、
`completion_text`、`dispatch_mode` 等本地字段保留在 `x_v2v` 命名空间。

Gateway 还需声明 capabilities，例如：

```json
{
  "binary_audio": true,
  "function_calling": true,
  "conversation_truncate": true,
  "input_transcription": true,
  "direct_speak": true
}
```

## 3. 共享框架迁移工作

三个被审查应用都复用了 `ovs_agent`，因此协议迁移应首先在框架层完成，而不是在
每个机器人项目中复制 adapter。

### 3.1 `SLVClient` / protocol types

主要文件：

- `agent/ovs_agent/protocol.py`
- `agent/ovs_agent/slv_client.py`
- `agent/ovs_agent/app_base.py`
- `agent/ovs_agent/audio_io.py`

工作：

1. 用共享 schema/类型代替 Server 和 Agent 两份手工字符串常量。
2. 连接后读取 `session.created`，发送 `session.update`，等待 `session.updated`，再开放
   mic pump，避免配置尚未生效就丢入首段语音。
3. 增加 `SessionCreated/Updated`、转写 item、`ResponseCreated`、
   `ResponseOutputAudioDone`、`ResponseDone`、function-call、结构化 error 类型。
4. 用 session audio format 设置播放器采样率，删除首 binary frame header 解析。
5. 记录 active response/item/content index 和 audio playback cursor。
6. `abort()` 改为 `cancel_response()` + truncate；取消应幂等，并丢弃旧 response 的晚到
   binary 音频。
7. `asr_eos()` 改为 commit；补 clear 和显式 `response.create`。
8. 保持应用 hook 稳定：`on_user_partial`、`on_user_utterance`、
   `on_assistant_sentence_start`、`on_assistant_sentence`、
   `on_assistant_done`。
9. `on_assistant_done` 只在 response 终态且 playback drain 满足时广播一次。
10. 工具注册/执行映射到 session tools 和 function-call item；断线后重新发送完整
    session 配置和工具表。

### 3.2 保留一个统一的 `speak(text)` 应用能力

当前多个应用使用 `send_text()` + `flush_tts()` 做“不经过 LLM 的直接播报”。标准
端到端模型的 `response.create` 不保证逐字照读，因此不能机械替换。

在 Agent API 保留：

```python
await app.speak(text, conversation="none")
```

Gateway adapter 的实现策略：

- 本地级联：直接进入 TTS。
- 有独立 TTS 能力的云 provider：调用其 TTS 通道。
- 只有 Realtime 语音模型：使用隔离的 response 请求逐字播报，并标记为
  `direct_speak`；能力不足时明确返回结构化错误，不能静默让模型改写安全提示。

`direct_speak` 是 Seeed 扩展能力，但对机械臂故障提示和提示音顺序是必要的。

### 3.3 测试

至少增加以下协议契约测试：

- session handshake/config ack
- auto response 与 manual response
- output-audio done、response done、playback drain 三阶段顺序
- cancel 只产生一个 cancelled response done
- cancel 后旧 response PCM 不进入新播放队列
- truncate 使用实际播放毫秒数
- reconnect 后 tools/session config 恢复
- function call/result 多轮
- direct speak 不污染 conversation
- 本地级联/OpenAI mock/Qwen mock 对同一 canonical event trace 的 parity
- 未知扩展事件可忽略，未知核心事件可观测告警

## 4. Reachy Voice 迁移审查

> 实施状态（2026-07-16）：已在 `../clawd-reachy-mini` 的
> `feat/realtime-v2-migration` 分支完成应用迁移。默认启用 Realtime V2
> manual server-loop；视觉上下文按 `session.update → response.create` 顺序注入，
> 情绪改为 `play_emotion` 结构化工具，远端/本地 conversation reset、播放 drain
> 和 0.2.0 vendored Agent wheel 已接入。活跃测试结果：138 passed，7 skipped
>（硬件/GStreamer 条件跳过）。尚未完成 Jetson 真机和云 provider 凭证验收。

项目：`../clawd-reachy-mini`（仓库/包已更名为 `reachy-voice`；活跃代码为
`src/reachy_voice/`，不包含已退休的 `legacy/reachy_claw/`）

### 4.1 当前结构

- `src/reachy_voice/conversation.py` 通过 `CompanionRobotApp` 使用共享
  `ovs_agent`，应用本身没有解析 V2V wire JSON。
- 当前明确选择 **client-loop**：本地 edge LLM 生成 token，再调用
  `slv.send_text()` 送入本地 TTS。
- `_TtsTagFilter` 在 token 进入 TTS 前剥离 `[happy]` 一类标签，同时触发机器人动作。
- 每个用户 utterance 前动态更新 system prompt，包含视觉/访客上下文。
- Dashboard 的 `asr_partial/asr_final` 是应用内部事件，不是 wire event；可以保留，
  由新的框架 hook 继续驱动。
- 生产镜像基于 `reachy-claw:slv-v7`，开发依赖还固定到仓库内
  `openvoicestream_agent-0.1.0` wheel，因此只改主仓库 Agent 不会自动进入 Reachy。

### 4.2 必做迁移

1. 更新/重打 `openvoicestream-agent` wheel，更新 `uv.lock`，并替换生产镜像中的
   ovs_agent；最好停止依赖不透明的 `reachy-claw:slv-v7` 旧框架层，或至少构建明确
   的 Realtime V2 base tag。
2. `build_ovs_config()` 从旧的 flat `slv_config` 迁移为 session audio、VAD、tools 和
   response 配置；播放器输出采样率由 session 协商，不固定假设。
3. 云端模式不能继续走当前 client-loop。Reachy 更适合
   `create_response:false`：等待 transcription completed，更新本轮视觉上下文，再发
   `response.create`。这样不会在动态 prompt 更新前被 server VAD 抢先生成回复。
4. 把 `session.reset()` 的访客 idle reset 映射为 Gateway conversation reset/new
   session；不能只清本地 history。
5. 重新设计情绪控制。原生 speech-to-speech 已经生成音频，无法再通过
   `_TtsTagFilter` 删除将被朗读的标签。推荐把情绪/动作改成 function tools 或独立
   structured side-channel；assistant transcript delta 只能用于展示和兜底触发，不应
   作为可靠动作控制。
6. motion tool schemas 通过 `session.update.tools` 注册；断线和动作列表变化后重发。
7. 更新 client-loop 专用测试，增加 local/cloud 两种 provider trace；硬件运动、视觉、
   Dashboard 逻辑不需要重写。

### 4.3 工作量判断

**中到高**。普通收听/播放迁移主要由 `ovs_agent` 吸收，但动态视觉 prompt 和情绪
标签是 Reachy 专属的架构变化。若先保留本地 client-loop，只升级 wire protocol，
工作量中等；若要求同一个 Reachy 应用立即支持原生云端 E2E，必须完成第 3～5 项。

## 5. 当前仓库 reBot Arm 迁移审查

项目入口：

- `agent/ovs_agent/apps/voice_rebot_arm/`
- `deploy/docker-compose.jetson-rebot.yml`
- `agent/Dockerfile.rebot-arm`

独立的 `../reBotArm_control_py` 是 B601-DM 电机、运动学和通信驱动库，不解析 V2V，
不需要协议迁移。变化集中在语音 Agent、工具协议和部署。

### 5.1 当前结构和发现

- 应用继承 `MultiModeApp`，wire 处理完全在共享 `SLVClient/BaseApp`。
- reBot 的动作和抓取通过本地 tool registry 执行，适合云端模型调用本地工具。
- speech 服务 compose 开了 `OVS_V2V_SERVER_LOOP=1`，但 Agent 侧读取的是
  `OVS_AGENT_SERVER_LOOP`/config `server_loop`，当前 reBot compose 没有设置它。
  因而“服务端支持 server-loop”和“Agent 实际使用 server-loop”存在配置不一致风险；
  V2 应通过 session ack 消除此类双开关。
- `GraspPlugin.on_assistant_done()` 依赖回复/播放结束后播放 ready tone。
- 抓取失败播报直接调用 `slv.send_text()` + `flush_tts()`。
- ArmPlugin 默认在每轮结束清理本地 LLM history，以提升小模型工具可靠性。

### 5.2 必做迁移

1. 升级共享 Agent 后重建 `agent/Dockerfile.rebot-arm` 镜像。
2. 删除/降级双端 server-loop 环境开关，Agent session 明确设置
   `create_response:true`，并验证 `session.updated` 返回实际模式。
3. 动作和 grasp tools 通过 session tools 注册；`tool_call/tool_result` 改为标准
   function-call item，保留超时和安全策略在 `x_v2v`。
4. `GraspPlugin` 的 ready tone 继续依赖稳定的应用级 `on_assistant_done`，不能直接绑
   `response.done`，否则本地仍有缓冲 PCM 时提示音会抢播。
5. 失败原因播报改用统一 `app.speak()`；测试 direct speak 的 response 不进入普通
   conversation，也不会被云模型改写。
6. `clear_history_on_turn_end` 改为 Gateway conversation reset；云端模式下清本地
   `session.history` 没有作用。
7. 动作列表动态变化时，除更新本地 registry 外，还要触发一次异步
   `session.update.tools`，并等待/记录 ack。
8. reBot 的音频和 barge-in 测试改用 response/item id，并增加“机械臂动作运行期间
   用户打断只停语音、不误停已提交安全动作”的回归测试。

### 5.3 工作量判断

**中等**。硬件驱动、IK、视觉抓取和动作实现基本不动；主要是共享框架升级、工具
协议、direct speak、清历史语义和部署配置。它比 Reachy 少动态 prompt/情绪标签
问题，但安全回归要求更高。

## 6. SO-ARM 方案迁移审查

项目：`../sensecraft-solutions/solutions/respeaker_flex_soarm`

运行应用源码实际来自本仓库：

- `agent/ovs_agent/apps/voice_arm/`
- `agent/ovs_agent/plugins/actuator_actions.py`

SenseCraft solution 在构建时把本仓库 `agent/` 复制进 voice-arm 镜像，因此 solution
目录本身主要负责镜像、配置、部署和文档。

### 6.1 当前结构

- VoiceArmApp 继承 `MultiModeApp`，同样由共享框架处理 wire events。
- 当前 solution 没有显式设置 Agent 的 server-loop 开关，默认仍可能走 client-loop；
  Dockerfile 中“server-loop mode”的注释主要指 server-side VAD/不加载本地 VAD，
  不能当作 LLM loop 已经在服务端运行的证据。
- ArmPlugin 在 `on_assistant_done` 清本地 history；该 workaround 针对本地
  Qwen3-4B-AWQ 多轮工具调用退化。
- 动作列表可以运行时变化，工具 registry 会重新注册。
- solution 当前固定部署本地 seeed-voice、edge-llm 和 voice-arm 三个服务，并假设
  16 kHz 输入/输出。

### 6.2 应用层必做迁移

1. 与 reBot 共用新的 `ovs_agent`、session handshake、response lifecycle 和工具
   function-call 映射。
2. `clear_history_on_turn_end` 在本地 provider 下可以保留当前策略；云 provider 下必须
   调 Gateway conversation reset，且等当前 response/playback 完成后执行。
3. 动作录制/修改后触发 `session.update.tools`，不能只改本地 registry。
4. 输出采样率从 session 读取并让 AudioIO 重采样，删除 16 kHz 固定假设。
5. 如果 SO-ARM 后续增加直接故障播报，也统一走 `app.speak()`。

### 6.3 Solution 包和部署必做迁移

1. 同步新版 `agent/` 到 solution build context，重建并发布新的 voice-arm image tag；
   更新 `solution.yaml`/compose 引用，不能继续复用包含旧 protocol.py 的镜像。
2. 为 local/openai/qwen 增加部署 preset 或 provider 参数。API key、model、region、
   endpoint 只注入 Gateway，不能放到机器人应用或前端。
3. 云端 preset 下 edge-llm 应变成可选服务；Gateway 仍保留，负责二进制 PCM、采样率、
   tool proxy 和 provider adapter。
4. 更新两份 `agent.yaml.tmpl`、runtime env、compose、guide、KNOWN_ISSUES 和 mock
   testing 文档，删除旧 `asr_final/tts_done` wire 语义描述。
5. 给一键部署验证增加 provider health/capability 检查；凭证错误要在部署或
   `session.created/error` 阶段清楚暴露。
6. 本地和云端 preset 都跑同一套 scripted audio + fake arm regression，确认工具动作、
   单轮清历史、打断和播放完成顺序一致。

### 6.4 工作量判断

**中等到高**。VoiceArm Python 应用本身改动小于 Reachy，但 solution 打包需要新增
provider 配置、凭证输入、条件部署、本地/云端验收矩阵和镜像发布，交付面更广。

## 7. 跨项目迁移汇总

| 项目 | 共享框架升级 | 应用专属改动 | 部署/制品 | 总体 |
|---|---:|---:|---:|---:|
| Reachy Voice | 高度复用 | 动态视觉 prompt、情绪控制重构 | wheel + base/image 更新 | 中到高 |
| reBot Arm | 高度复用 | direct speak、完成提示音、云端清历史 | 重建 reBot Agent image | 中等 |
| SO-ARM solution | 高度复用 | 动态工具同步、云端清历史 | presets、凭证、条件服务、镜像发布 | 中到高 |

协议升级后，三个应用的麦克风采集、wake word、播放设备、机器人 SDK、动作实现、
视觉算法和 Dashboard 业务层大部分都不需要迁移。真正的公共主干是
`SLVClient + BaseApp + Gateway adapters`。

## 8. 推荐实施顺序

### Phase 0：冻结规范

1. 定义 Realtime V2 JSON Schema、binary contract、扩展命名空间和 capabilities。
2. 明确 `direct_speak`、conversation reset 和 tool update 的统一应用 API。
3. 明确 breaking endpoint/version，例如 `/v2v/stream` + WebSocket subprotocol
   `seeed.realtime.v2`，避免旧客户端误连后静默失效。

### Phase 1：Gateway + 共享 Agent

1. session handshake、input buffer、response lifecycle。
2. binary PCM 格式协商。
3. cancel/truncate/playback drain。
4. tools/function-call。
5. direct speak 和 conversation reset。
6. 本地级联 adapter parity 测试。

### Phase 2：先迁移 reBot Arm

reBot 的应用结构最接近目标 server-managed tool loop，适合作为第一台真实机器人
验收工具、打断和 direct speak；同时安全测试必须完整。

### Phase 3：迁移 SO-ARM

复用 reBot 验证过的 ArmPlugin 路径，再完成 SenseCraft solution 镜像、presets 和
一键部署改造。

### Phase 4：迁移 Reachy

先升级普通对话链路，再单独完成动态视觉上下文和情绪/动作结构化控制，最后验证
local/OpenAI/Qwen 三 provider 切换。

### Phase 5：云 provider adapter

同一 canonical trace 分别跑 OpenAI 和 Qwen adapter。应用端不得出现 provider 名称
分支；provider 差异只允许存在 Gateway 配置、adapter 和 capabilities 中。

## 9. 验收标准

1. 三个应用只改 Gateway provider 配置即可切换 local/OpenAI/Qwen。
2. 应用代码不解析任何 provider 原始 event 名称。
3. 每个 response 恰好一个终态 `response.done`。
4. `on_assistant_done` 在实际播放 drain 后恰好触发一次。
5. 打断后 200 ms 内不再播放旧 response 音频，conversation 截断到实际播放位置。
6. 工具 call id 在断线、超时、取消和多轮调用中不串线。
7. 机械臂 direct speak 不污染历史，安全提示不被模型改写。
8. Reachy 的视觉上下文在 response 创建前生效，情绪控制不依赖朗读文本标签。
9. SO-ARM 一键部署对 local/cloud preset 都能做 provider health 和 capability 验证。
10. 旧客户端连接 V2 时收到明确版本错误，而不是无声卡住。

## 10. 当前最重要的风险

- 只把 `tts_done` 改成 `response.done`，会继续混淆服务端生成完成和本地播放完成。
- 原生 E2E 下继续依赖 Reachy 的文本标签过滤，会出现标签被读出或动作丢失。
- 只取消音频不 truncate，云端后续上下文会包含用户没听到的回复。
- 云端模式仍只清本地 history，SO-ARM/reBot 的单轮策略将失效。
- 继续复制 Server/Agent protocol constants，会让镜像和源码版本再次漂移。
- 不保留 direct speak，会让机械臂安全/故障播报变成非确定性的模型生成。
- 当前 SO-ARM/reBot 关于 server-loop 的注释、服务端开关和 Agent 实际模式并不完全
  等价；V2 必须用握手确认最终生效配置。
