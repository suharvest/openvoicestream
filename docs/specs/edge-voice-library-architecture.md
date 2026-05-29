# Edge-Voice Library —— 架构设计 Spec (v1, 待评审)

> 状态:**DRAFT,待主线程/用户评审**。2026-05-29。
> 目标:把 seeed-local-voice 产品栈抽离成一个**通用的、面向边缘设备的高性能实时语音对话库**,开源发布(Apache 2.0)。
> 本文档由主线程综合 codex 两轮只读设计 + 代码核实结论定稿草案。`file:line` 引用为撰写时快照,实施前需复核。

---

## 0. 定位与边界(已拍板)

- **定位**:"Pipecat for the edge" —— 边缘原生、本地推理优先的实时语音对话库。差异化护城河是**边缘性能工程**(低延迟 TTFA、小 GPU 上 N=2 并发不崩、热插拔不漏显存),不是编排胶水。
- **边界 = 窄而深**:首版只官方保证少数几条调好的黄金路径(带性能 SLA + benchmark gate);其他设备/模型开放插口但"性能自负"(`support_level="self_service"`)。
- **范围 = open-core 三层**(见 §0.1):核心引擎(护城河)+ 通用 agent 层(可复用)+ 示例。具体产品(如 VoiceArm)是下游消费者,不进开源库。
- **发布**:开源,Apache 2.0。

### 0.1 Open-core 分层与产品关系(2026-05-29 拍板)

```
voxedge (开源, Apache 2.0)
├── ① 核心引擎       边缘语音(ASR/VAD/TTS + 实时编排 + Transport)← 护城河,性能 SLA
├── ② agent 层       通用 agent 能力(LLM 回路/工具调用/记忆/对话模式)← 可复用,从产品反哺
└── examples/        ② 的精简演示(注册工具→语音调用,领域无关)

        ▲ pip install voxedge
        │
VoiceArm (独立 repo, 产品, license 自定)  —— voxedge 的下游消费者,不在开源库内
└── 只含机械臂独有的工具 + 硬件集成 + 领域逻辑,其余全 import voxedge
```

- **核心 = ①(引擎)**,是护城河和库的身份;**② agent 层是核心之上的可复用层,不是护城河**(编排能力 Pipecat 也有)。对外 headline 是引擎,不是"全栈 agent 框架"——避免在 Pipecat 的广度主场竞争。
- **② 从真实产品反哺**:VoiceArm 里凡通用的能力(LLM 回路、`@tool` 机制、记忆、对话模式、领域无关工具)抽离沉到 ②;VoiceArm 只留独有的(臂工具/硬件/领域逻辑),`pip install voxedge` 引用 ②。第二、三个"语音控制 XX"产品同样薄薄一层接上。
- **抽离边界口诀**:提到"设备/硬件/领域"的 → 留产品 repo;只关于"语音/LLM/工具调用/记忆"机制的 → voxedge ②。现状大部分已划好:`agent/openvoicestream_agent/` ≈ 通用层 ②,`agent/apps/*` ≈ 具体应用(示例或下游)。
- **open-core 收益**:放心开源 voxedge(基础设施)立社区口碑,同时把 VoiceArm(差异化产品/可商业化)留在独立 repo,两者经 pip 依赖解耦,产品价值不被白送。

### 黄金路径(首版官方保证)
| 路径 | ASR/TTS 后端 | profile |
|---|---|---|
| Jetson Orin + TRT-Edge-LLM | `jetson.trt_edge_llm` | `configs/profiles/jetson-multilang-highperf.json` |
| RK3588 + RKNN | `rk.asr` / `rk.tts` | `configs/profiles/rk3588-kokoro-rknn.json` |
| RPi5 + Hailo | **不存在,标 roadmap** | 接口预留,不阻塞首版 |

---

## 1. 关键认知:架构在进程/算力上也已分离(区别于 §0.1 的库分层)

> 注意:§0.1 讲的是**库/包分层**(核心①/agent②/示例);本节讲**运行时算力/进程切分**,用 A/B/C 标号避免混淆。

代码核实(2026-05-29)纠正了"这是个单体"的初始假设。运行时实际是分离的:

```
app/                          ← A. GPU 语音引擎(稀缺资源,slot-pool N=2 上限)
   ↑ Transport (WS / InProcess)
agent/openvoicestream_agent/  ← C. agent 大脑(轻,纯编排,每对话一实例)
   └── 调用 B. LLM 引擎(EdgeLLM /v1/chat/completions,本身也吃 GPU)
```

边缘上有三块算力:**A** ASR/TTS 引擎(GPU)**B** LLM 引擎(GPU)**C** agent 大脑(轻)。**稀缺的是 A/B**,C 无所谓。(映射到 §0.1 库分层:A = 核心①,C 的通用部分 = agent 层②,B 是可插拔后端。)

### 已存在、可复用(不是造抽象,是打磨)
- **`agent/openvoicestream_agent/llm/base.py`** —— `LLMBackend` ABC + `LLMEvent`(kind: `text`/`tool_call_delta`/`finish`)+ `stream_events`/`stream`/`warmup`/`aclose` 生命周期,**已完整实现**。实现:`edge_llm.py`(EdgeLLM)、`openai_compat.py`、`noop.py`。
- **`agent/openvoicestream_agent/tools/`** —— `registry.py` + `runner.py` + `builtin.py`,工具调用框架。
- **`agent/openvoicestream_agent/modes/`** —— chat / interpreter / monologue / transcribe。
- `session.py` / `state.py` / `event_bus.py`(编排)、`slv_client.py`(WS 连 app/)、`wake_source` / `translator` / `plugins`。

→ "LLM + 工具进库"大部分已做完。库化主要工作 = **剥 app/ 内核 + 打包 + 文档 + 开源 carve-out**。

---

## 2. 内核剥离(app/ 层,仍是主要工作)

V2V 编排目前埋在 `app/main.py` 的 `/v2v/stream` handler(约 `2472-3593`),需剥成可 import 的 `ConversationEngine` / `Session`,不依赖 FastAPI。

**留在 Transport/HTTP 适配层(不搬)**:
- 路由装饰 + handler 入口 `app/main.py:2472-2473`
- auth / `ws.accept()` / admission gate / BackendManager WS 登记 `app/main.py:2509-2550`
- 首帧 config 读取解析 `app/main.py:2593-2608`
- WS 专属错误码(1003/1011/4429)映射 → adapter 侧策略;engine 只 emit 类型化错误(如 `pool_saturated`)`app/main.py:3342-3356, 3470-3495`

**搬进编排内核**:
- backend / VAD / TTS buffer 组装 `app/main.py:2672-2745`
- per-conn state / generation counter / endpoint flags `app/main.py:2751-2791`
- `dispatcher()` 音频/控制帧状态机 `app/main.py:2814-2984`
- `asr_out_task()` partial/finalize/timeout/generation gate `app/main.py:2992-3205`
- `tts_out_task()` sentence queue/barge-in cancel/watchdog/deadline `app/main.py:3207-3432`
- task 编排 + cleanup `app/main.py:3434-3575`
- per-utterance 生命周期已可复用:`app/core/asr_session_manager.py`(IDLE/ACTIVE/FINALIZING/CANCELLING/ERROR_REBUILD + generation + bounded cancel + worker restart)

**提议包结构**:
```
edge_voice/
  engine/     ConversationEngine, Session, turn state
  backends/   ASRBackend, TTSBackend, VADBackend, LLMBackend ABCs (LLM 复用 agent/llm)
  transport/  InProcessTransport(默认), WebSocketTransport, (WebRTC/mic stubs)
  agent/      复用现有 openvoicestream_agent: llm/tools/modes
  profiles/   黄金路径 capability 声明
  bench/      regression gate 包装 (bench/perf/gate.py)
```
`ConversationEngine.__init__(backends, profile, tool_registry, timeouts)` 只收解析后配置,**不直读 env**。`Session.run(transport)` 驱动 dispatcher/asr/tts。

---

## 3. 后端抽象基类(现状 vs 目标)

- **ASRBackend** —— public API 已在 `app/core/asr_backend.py:94-147`(`name`/`capabilities`/`create_stream`/`transcribe`/`unload`/`concurrency_capability`),`ASRStream` 在 `36-91`。**差距**:`create_asr_backend()` 暗引 `current_profile()`(`asr_backend.py:163-178`)、TRT ASR `_load_config()` 直读 `EDGE_LLM_ASR_*`/`ASR_*` env(`trt_edge_llm_asr.py:257-354`)→ 改 config 注入。
- **TTSBackend** —— `app/core/tts_backend.py:31-142`。目标流式签名 `generate_streaming(text, *, language, speaker, cancel_token) -> Iterator[bytes]`。**差距**:`OVS_TTS_MODEL_ID`、worker concurrency、artifact resolver 都从 env 取(`tts_backend.py:49-65`、`trt_edge_llm_tts.py:595-623`)→ profile 注入。
- **VADBackend** —— 现 `app/core/vad.py:90-109` + `create_vad():244-257`。目标 `create_session(sr, silence_ms) -> VADSession`,`process(samples) -> "speech_start"|"speech_end"|None`。**差距**:`SILERO_VAD_ONNX_PATH` env(`vad.py:67-73`)→ 注入。
- **LLMBackend** —— **已存在**(`agent/openvoicestream_agent/llm/base.py`),直接采用为库的标准接口。

### 3.1 Backend 代码归属:repo 边界(2026-05-29 拍板)

模型适配代码按**语言/构建重量 + 迭代节奏 + 依赖方向**切分,确保 **voxedge 永远是纯 Python、可 `pip install`、不需 CUDA 也能装**。

**决策口诀**:
> 重型(C++/CUDA/需构建/ABI 特定)→ 引擎 repo,ship 预构建产物。
> 薄胶水(Python/实现 ABC/IPC 串联)→ voxedge,作为可选 extra。
> 模型产物 → 不进 code repo,HF 下载 / release artifact。

**三层 repo 结构**:
```
voxedge (Python 库, 开源)
├── 核心:ASRBackend/TTSBackend/VADBackend/LLMBackend ABC + 编排
└── 薄 Python 适配器(仓库内可选 extra):voxedge[trt] / voxedge[rknn] ...
        │ 经 IPC(JSON-line)调
        ▼
TensorRT-Edge-LLM fork (C++ 引擎 repo, 已存在, Apache 2.0)
├── runtime + kernels + C++ JSON-line worker + build.sh + ONNX/engine 脚本
└── ship 预构建二进制 / release artifact(不进 pip 源码)

模型权重 / engine 产物 → HF 下载 or release artifact(无 code repo)
```

**"worker 代码"分两层**(关键):
- **C++ worker 进程**(真正串联引擎、干活的)→ 引擎 repo(fork),跟引擎一起编译 + ship 预构建。源 of truth = TRT-Edge-LLM fork(非 submodule)。
- **Python worker-driver / adapter**(`app/backends/jetson/trt_edge_llm_*.py` + `_ipc.py`,实现 ABC + 经 IPC 驱动 worker)→ voxedge,作为 `voxedge[trt]` extra。

**适配器形态决定(已拍板)**:**仓库内可选 extra**(`voxedge[trt]`、`voxedge[rknn]`),装对应 extra 才拉重依赖;ABC + 适配器在同一 repo lockstep 同步,匹配"窄而深、你自己掌控 2-3 条黄金路径"。**预留**:定义 plugin entry-point 机制,以便将来第三方 / 新设备适配器可独立成包接入,不用 fork voxedge。

**新 port 一个模型的归属**:C++ kernel/worker → fork;Python adapter → voxedge `[新后端]` extra;权重 → 下载。

---

## 4. LLM + 工具调用闭环(库内自动,不再丢给客户端)

`asr_final` → LLM(带 tools)→ TTS 的回路在库内闭环:
1. `asr_manager.finalize_with_status()` 过 generation gate → `asr_final` `app/main.py:3130-3197`
2. `Session` 把 user text 入 history,调 `ToolRunner.stream_with_tools()`
3. `ToolRunner` 从 LLM stream 检出 `tool_call_delta` → 执行本地 tool → 追加 `role:tool` → continuation(`agent/.../tools/runner.py`)
4. LLM text delta 灌 `TTSBuffer.add()` → sentence queue(等价现 `CLIENT_TEXT` path `app/main.py:2953-2960`)
5. `tts_out_task` 消费队列、`generate_streaming`、发 sample-rate header + `tts_started` + PCM `app/main.py:3304-3361`

**库一等组件**:`LLMEvent`/`LLMBackend`/`ToolRegistry`/`ToolRunner`/`ConversationEngine`。**留产品层**:app 专属 action(如 `set_mode`)。

---

## 5. Transport 抽象(两模式,默认同进程)

```python
class Transport(ABC):
    def recv_audio(self) -> AsyncIterator[bytes]: ...
    def send_audio(self, chunk: bytes) -> None: ...
    def recv_event(self) -> AsyncIterator[dict]: ...
    def send_event(self, event: dict) -> None: ...
    def close(self, code=None, reason=None) -> None: ...
```
现状映射:`ws.receive()`→`recv_audio/recv_event`(`app/main.py:2817-2833, 2944-2952`)、`ws.send_json()`→`send_event`(`2795-2801`)、`ws.send_bytes()`→`send_audio`(`2802-2807`)、`ws.close()`→`close`(`3537-3547`)。

- **`InProcessTransport`(默认)**:大脑↔引擎零 IPC,单设备最低延迟。多对话 = 一进程 N 个 Session 共享 slot-pool。
- **`WebSocketTransport`**:进程隔离,引擎做共享 appliance 支持多客户端/分布式;保留现 `agent/.../slv_client.py` 路径。超 N 上限走 4429 `pool_saturated`(已存在)。

---

## 6. 性能保证机制(窄而深)

- 黄金路径在 `edge_voice/profiles/` 用 typed declaration 表达,含 `support_level`(`official` / `self_service`)。
- gate 包装现 `bench/perf/gate.py`:按设备 baseline JSON 比对,regression 则 nonzero exit(`gate.py:1-27, 130-188`)。官方路径 = baseline 存在 + strict gate PASS。
- 并发 SLA 复用 `concurrency_capability()` 的 `min(asr, tts)` ceiling(`app/core/capability_resolver.py:136-153`)。
- **开源信誉钩子**:发布真实设备 benchmark 表(TTFA / N=2 / 内存)+ 可复现脚本。

---

## 7. SDK 打包 + quickstart

```python
import asyncio
from edge_voice.engine import ConversationEngine
from edge_voice.profiles import load_profile
from edge_voice.backends import create_backends
from edge_voice.transport import InProcessTransport   # 默认零 IPC
from edge_voice.tools import ToolRegistry

tools = ToolRegistry()

@tools.tool(description="Return local time.")
def time_now() -> dict:
    import datetime
    return {"now": datetime.datetime.now().isoformat()}

engine = ConversationEngine(
    backends=create_backends(load_profile("jetson_orin_trt_edge_llm")),
    tool_registry=tools,
    multi_utterance=True,
)
# 单设备:本地 mic / in-process;或 WebSocketTransport(ws) 做共享 appliance
asyncio.run(engine.run(InProcessTransport(mic=..., speaker=...)))
```
**后向兼容**:`app/main.py` 路由保留,内部委托 `ConversationEngine.run(WebSocketTransport(ws))`;协议常量保留 `app/core/v2v.py:30-54`;现 Docker 产品零改动。

---

## 8. 分阶段落地路线

| Phase | 内容 | 验证 | 因 agent/ 已存在的变化 |
|---|---|---|---|
| **P1** 内核剥离(零行为变化) | `edge_voice/engine` + backend ABCs;`app/main.py` 保路由委托 engine | 现有 V2V bench + unit test 同协议输出(`v2v.py:30-54`) | LLM/tools/modes 不动,直接 wrap `agent/` |
| **P2** `trt_edge_llm_asr` 迁新 ABC | config 注入去 env 耦合(`trt_edge_llm_asr.py:257-354`) | slot-N `concurrency_capability()` + streaming bench gate | — |
| **P3** Transport 抽出 | `InProcessTransport` + `WebSocketTransport`;close/error 映射移 adapter | cancel cleanup + close 路径 e2e | InProcess 为新增默认路径 |
| **P4** 开源 carve-out + 发布 | 私有物剥离、license 处理、Apache 2.0、CI bench gate | 全 official 设备 gate PASS | 依赖 license 审计结论(进行中) |

每 Phase 可独立验证、不破坏现生产部署;回滚策略:保留 env→config shim / legacy transport facade。

---

## 9. 开源 carve-out(审计完成 2026-05-29)

**阻塞项已清除:TRT-Edge-LLM fork 是 Apache 2.0**(NVIDIA,`/Users/harvest/project/TensorRT-Edge-LLM/LICENSE:190` + SPDX header,上游公开 v0.7.1)。Apache §2 允许 fork+改+再分发(含商用)。**→ 全栈开源可行**(不需退到"引擎私有")。fork 内 3rdParty 全是 permissive 子模块(googletest BSD-3 / nlohmann-json MIT / NVTX Apache-BSD)。模型权重运行时下载、不进 repo,license 风险天然在库外。**未扫到任何 secret / 私有 IP / 凭据。**

### A. 可直接 Apache 2.0(自研核心)
`app/` 编排器 + `app/core/`(asr/tts 抽象、capability framework、hot-swap、model_downloader)、`app/backends/` 集成胶水、整个 `agent/openvoicestream_agent/`(llm/transport/tools/modes/vad)、`services/translator/` 代码、`bench/` harness、`configs/profiles/`、`deploy/`(已参数化)。

### B. 开源前必须处理(发布 checklist)
- [ ] **relicense MIT → Apache 2.0**:现 `LICENSE` 是 MIT(Copyright Harvest Su)+ README badge,单一版权人改 Apache 2.0 合法,但 `LICENSE` 和 README badge 都要改。
- [ ] **修 `third_party/qwen3-edgellm-jetson/LICENSE` 错误署名**:现写 `MIT … Copyright (c) 2025 Artur Skowronski`(占位/错署),实为你自己的 repo `suharvest/qwen3-edgellm-jetson`,发布前订正。(`third_party/rkvoice-stream` 已正确 Apache 2.0)
- [ ] **加 `THIRD-PARTY-NOTICE` / `MODELS.md`**:逐个列模型 license,**显著标注 NLLB 权重非商用、Qwen3-TTS checkpoint license 需用户自查**。
- [ ] **Apache §4(b) change-notice**:fork 里被你 patch 的文件加"changed by"变更说明,保留 NVIDIA 版权头 + 传播上游 NOTICE。
- [ ] **剥离内部文档**:`agent/HANDOFF*.md`、`docs/specs/` 下 39 个内部实现/debug 史(如 `cute-dsl-*-bug-investigation.md`、`prod-hardening-week*.md`);未 track 的 `CONTINUOUS_DEBUG_NOTES.md` / `tools_probe/` 不要加入。
- [ ] **预构建二进制**:`patches/sherpa-onnx-lib/*.so`(aarch64)、`voices/af_cute.bin` —— 优先提供重建脚本而非直接塞 blob。

### C. 不进 repo —— 用户自备 / 预构建
所有模型权重(已是下载模式)、**NLLB 权重(CC-BY-NC 4.0 非商用)**、**Qwen3-TTS checkpoint(license 待核)**、aarch64 `.so` blob(改为 release artifact / 重建说明)。

### 需法务确认的灰色地带
1. ~~Qwen3-TTS/ASR checkpoint~~ —— **已确认 Apache 2.0(用户 2026-05-29),非问题**。
2. NLLB 非商用:确认只发 translator 代码(不含权重)可接受,并把权重下载做成 opt-in + 非商用警告。**首版可考虑直接把 NLLB 翻译列为可选/roadmap,绕开非商用风险。**
3. `af_cute.bin`(kvoicewalk 派生 Kokoro embedding)是否干净继承 Apache。
4. NVIDIA 商标(Apache §6):库命名/品牌不得暗示 NVIDIA 背书。

---

## 待确认/未决
1. **库命名 / 品牌**(开源对外名)。
2. TRT fork 能否开源(审计中,决定 §0 范围是"全栈开源"还是"框架开源+引擎插件")。
3. RPi5+Hailo backend 何时做(roadmap)。
