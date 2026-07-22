# VoxEdge — 定位、分层与传播规划

> 配套文档:`consolidation-plan.md`(代码整合计划)、`code-structure/`(代码结构分析)。
> 本文解决三件事:① 对外怎么讲清"这是什么"(定位 + 命名分层);② 底层 TensorRT-Edge-LLM 依赖该 fork 还是 patch;③ 怎么做成有传播性、吸引不同层次开发者。

---

## 一、一句话定位

**VoxEdge = 面向边缘设备的开源实时语音 AI 栈。任何边缘芯片,端侧运行,低延迟,可对话。**

- 英文 tagline:*"Real-time voice AI for any edge device. On-device, low-latency, open."*
- 类比锚点(让人秒懂):**"边缘版的 Pipecat / LiveKit Agents"**——但端侧高性能 + 多后端,不依赖云。
- 一句话故事:**"VoxEdge 让实时语音在任何边缘设备上跑;OpenVoiceStream 是基于它的开箱即用应用。"**

---

## 二、分层架构(对外清晰:三层,只有一个品牌)

**核心原则:对外只有一个品牌 = VoxEdge(引擎层)。其余一律用"关系"来描述,不再起并列品牌名。**
这是为了消除 "Voice Stream / Voice Engine" 这类并列命名造成的"到底用哪个"的困惑——两个长得像的名字 = 没人分得清层级。

| 层 | 是什么 | 代码 | 对外叫法 | 受众 |
|---|---|---|---|---|
| **引擎层** | 可嵌入的高性能实时语音管线(ASR→LLM→TTS、打断、轮次管理、后端抽象) | `voxedge` | **VoxEdge**(品牌本身,`pip install voxedge`) | 系统/算法开发者 |
| **后端层** | 各设备的模型执行后端(不是产品,是插件) | TRT-Edge-LLM(Jetson)/ RK(RKNN)/ sherpa(RPi) | "**VoxEdge backends**" | 选设备的开发者 |
| **应用层** | 基于引擎搭的开箱即用应用 + 示例(不是并列品牌) | OpenVoiceStream + demos | "**built with VoxEdge**"(参考应用/示例) | 应用开发者 |

- 品牌只压在**引擎**上(它才是差异化:高性能 + 多后端 + 端侧)。
- 一键 Docker 部署的那个 = "**VoxEdge 参考应用**(OpenVoiceStream)";机械臂语音、实时翻译等 = "**VoxEdge 应用示例**"。
- **不要**把应用层当成跟引擎并列的产品去推。

**两层受众,两套卖点:**
- 后端/引擎层 → 有开发能力的系统开发者:*"不管你什么设备,都能跑一个高性能实时语音库"*。卖点 = **性能数字 + 多后端覆盖 + 量化深度**。
- 应用层 → 应用开发者:*"在我们这儿能很方便搭各种语音应用"*。卖点 = **一键部署 + demo gallery + SDK**。

**落到代码 = 3 层 5 个职责单元 / 6 物理仓(与整合计划 B.1 一致;RK = rkvoice-stream + rkvoice-engine 两仓算一个后端职责):**

| 层 | 仓库 | 职责 | 对外叫法 |
|---|---|---|---|
| TOP 应用 | `seeed-local-voice` | server + agent apps + demos | "VoxEdge 参考应用/示例" |
| MIDDLE 引擎 | `voxedge` | 管线 + 后端抽象 + Python 适配 | **VoxEdge**(品牌) |
| BOTTOM 后端 | `TensorRT-Edge-LLM`(fork) | Jetson runtime+导出 source-of-truth,pin v0.8.0,薄可上游 patch | "Jetson backend" |
| BOTTOM 后端 | `jetson-voice-engine` | 纯自研不可上游的 Jetson 功能/overlay/recipes | (后端 overlay) |
| BOTTOM 后端 | `rkvoice-stream`/`rkvoice-engine` | RK NPU 后端 | "RK backend" |

- 仓库数基本不变(~5),变的是**每仓只干一件事 + 删跨仓重复**。
- **fork vs jetson-voice-engine 切干净**:可上游 bug → fork(薄 patch,合并即归零);纯自研功能 → jetson-voice-engine。
- 99% 终端用户**不碰任何源码仓**,只 `docker run` + 自动拉 HF 预编译引擎。详见整合计划 B.1。

---

## 三、TensorRT-Edge-LLM 依赖:fork vs patch 的解法

**痛点(你说的):** fork = 每次上游升级迁移税大;patch = 理解成本高、构建复杂。

**解法 —— 关键洞察:我们的增值不是 fork runtime,而是"产出制品的工具"。** 据此把底层拆成三块:

1. **上游 TRT-Edge-LLM = 固定版本依赖 + 一组薄的、编号的 patch(分两类,见整合计划 B.1 三类变更模型)**
   - **C1 上游 bug 修复**(如 `N≤0` prefill guard):可上游、**合并即归零**,理解成本最低。
   - **C2 本地 runtime 扩展**(如 fp8-embed wiring、cuBLAS-free tiled GEMM):我们加的 runtime 能力,暂不/不可上游,**长期维护型 patch**,但仍只在 fork 落地(fork = runtime 唯一 source-of-truth),不会出现第二份手写源。
   - 两类都在 fork;jve 侧的 overlay patch 从 fork **自动生成**,非手写。

2. **我们的 recipes / model-zoo = 量化+导出工具(int4-AWQ + fp8),在上游之上跑,不 fork runtime**
   - 物理位置 = **`jetson-voice-engine/recipes/`**(class C3),只调 fork 的 export API、pin 一个 fork commit,**不改 runtime**。
   - 我们的模型导出跟官方不一样,是因为这是**我们额外的优化步骤**——它干净地独立成一层,而不是把 runtime 分叉。
   - 终端用户想自己量化才会碰这层;大多数人不碰。

3. **预编译优化引擎放 HF(已经有 3 个 bundle)= 终端用户实际消费的东西**
   - 99% 用户**不编译、不看 fork、不看 patch**:`docker run` + 按配置自动从 HF 拉对应的 int4 引擎。
   - 已上传:`harvestsu/qwen3-asr-0.6b-int4-v080`、`qwen3-tts-0.6b-base-jetson-trtllm-int4fp8`、`qwen3-tts-0.6b-customvoice-jetson-trtllm-int4fp8`。

> **一句话:** "fork vs patch 的理解成本只对贡献者存在;终端用户看到的永远是『拉引擎 → 跑』。" 所以——patch 集保持小且可上游、recipes 独立成层、引擎预编译分发。

**和代码现状的衔接(AST 分析发现的):** 现在同一批 feature 被编码了两遍——`jetson-voice-engine` 上 8 个 v0.7.1 patch ⟷ fork v0.8.0 已原生;worker 概念在 3 个源码树重复(fork `examples/omni`、jetson-voice-engine `native/edgellm_voice_worker`、seeed `deploy/asr-worker-v080`)。
→ 整合方向:**fork 的 v0.8.0 = 唯一 source-of-truth**;overlay(jetson-voice-engine)重新 pin 到 v0.8.0、删掉已原生的冗余 patch;worker 收敛到一个树,其余变成 build-input。int4 导出 driver(现只在 `wip/native-int4-talker`)并进 canonical 分支作为 recipes 层。

---

## 四、传播策略

**双受众漏斗 + "wow → run → build" 内容弧:**

1. **Hook(所有人)** — 一个端侧实时 demo,最好是**机械臂听语音实时执行、全程无云**。制造"哇,这怎么做到的"的瞬间。
2. **→ 系统开发者** — "在你自己的 Jetson / Pi / RK 上,一条命令跑起来" + **benchmark 页**(Orin Nano RTF<1、TTFA 0.2s、int4 省 ~960MB、多路并发)。**性能就是护城河。**
3. **→ 应用开发者** — "N 行代码搭你自己的" + **demo gallery**(语音机械臂、实时翻译、实时字幕) + SDK quickstart。

**传播必备资产(也是 README/文档要补的):**
- 杀手级 **landing README**:简化架构图(上面那三层) + 性能表 + demo GIF/视频 + **真·一条命令** quickstart。
- **benchmarks 文档**——我们性能确实硬,要把数字摆出来。
- **app gallery**(机械臂 / 翻译 / 字幕,每个配一段视频)。
- 各层**清晰的贡献路径**("贡献一个后端" / "搭一个应用")。
- **一键 Docker + 按配置自动拉模型**(整合计划 workstream D)——这是把"看视频的人"转化成"用户"的关键;没有它,漏斗会漏。

**性能指标(可对外的 hero 数字,数据出处见 `benchmarks-dataset.md`):**
> ⚠️ 只列**已验证 + 有出处**的数字。带 ⚠️ 的为 experimental,**不进 hero/README/Show HN**,仅内部追踪。发布前每个数字须补 repro 元数据(设备/profile/engine md5),见数据集 repro 要求。

| 模型 | 精度 | 关键指标 | 状态 |
|---|---|---|---|
| Qwen3-TTS-0.6B base | int4+fp8 | **RTF 0.44 / TTFA 0.21s**(Orin Nano)、−1.06GB/实例、CER==fp16 | ✅ 已验(N=1;N=2 待 gate) |
| Qwen3-ASR-0.6B | int4-AWQ | ZH CER 0% / EN WER ~11%(短指令)、引擎 −45% | ✅ 解码契约验(未过 production worker) |
| Qwen3-TTS-0.6B CustomVoice | int4+fp8 | ASR 可懂、talker −660MB | ⚠️ experimental(无 RTF/TTFA、未过 worker) |
| 覆盖 | — | 多后端(Jetson/RK/RPi)、任意 ASR×TTS 组合 | ✅ |

**启动序列(建议):**
- **准备期**:README/landing + benchmark + 1–2 个旗舰 demo 视频 + 一键部署可跑。
- **软发布**:目标社区——Hacker News(Show HN)、Reddit(r/LocalLLaMA、r/embedded、r/selfhosted)、X/Twitter、相关 Discord、NVIDIA Jetson / Rockchip 社区。
- **内容分发**:短视频(机械臂 wow,面向所有人) + 技术博客(int4 端侧优化的工程深度,钓系统开发者) + quickstart 教程(钓应用开发者)。
- **定位锚点反复用**:"Pipecat for the edge" / "ollama for voice agents",一句话让人记住。

---

## 五、落地(和整合计划衔接)

- 本文的分层 → 落到代码 canonical repos:`seeed-local-voice`=Apps、`voxedge`=Engine、`TensorRT-Edge-LLM` fork=后端 source-of-truth(见整合计划 A/B)。
- 命名:对外只推 **VoxEdge**;`OpenVoiceStream` 改称"VoxEdge 参考应用"。
- 执行顺序接整合计划:C(统一到 v0.8.0)→ D(一键部署 + 自动拉模型)→ E(README/landing + 性能表 + demo gallery)。

**发布前 license / model-card 结论(已调研,2026-06-21):**

| 模型 | License | 商用 | 重分发 engine | 处理 |
|---|---|---|---|---|
| Qwen3-TTS-Base / Qwen3-ASR / Qwen3-4B | Apache-2.0 | ✅ | ✅ | 附 LICENSE/NOTICE 即可 |
| Kokoro-82M | Apache-2.0 | ✅ | ✅ | 同上 |
| Matcha-TTS(代码) | MIT | ✅ | ✅ | 保留版权 |
| NVIDIA TensorRT-Edge-LLM(fork base) | Apache-2.0 | ✅ | ✅ | 标注我们的修改文件 |
| **Paraformer / SenseVoice** | FunASR Model License v1.1(阿里自定义,非 Apache) | ✅ | ✅(有条件) | **必须署名来源 + 随附 MODEL_LICENSE 原文**;含失权条款 |
| **NLLB-200**(翻译) | **CC-BY-NC-4.0** | ❌ **非商用** | 仅非商用 | **不打进默认包**;翻译做可选组件、只发 recipe、文档标非商用 + 给可商用替代 |
| **Qwen3-TTS-CustomVoice**(一等公民) | Apache-2.0(随 Qwen 官方) | ✅ | ✅ | **跟随 Qwen 官方 CV license,我们不增不改**:附 Qwen 的 LICENSE/NOTICE 原样分发,使用条款以 Qwen model card 为准,我们不自加额外限制或免责门槛。与 Base 同档处理。 |
| **MOSS-TTS-Nano** | 元数据 Apache-2.0,但正文有"LICENSE 未发布前视为未授权"兜底语 | ?待核 | ?待核 | 合并前核对仓库根 LICENSE 文件是否落地 Apache 全文;未落地则暂不打包 engine |

→ 发布前出 `LEGAL_AND_LICENSE.md`:第一档(Apache/MIT)随包;FunASR 两个附署名+协议;**NLLB 剥成可选非商用组件**;**CustomVoice 跟随 Qwen 官方 license 原样分发(不自加限制)**;MOSS 核实 LICENSE 文件。

**Launch readiness checklist(全绿才发 Show HN / Reddit):**
- [ ] `docker compose up` / install.sh 在**干净设备**上一键跑通(对应 plan D)。
- [ ] HF / mirror 自动拉模型成功 + checksum 校验(对应 plan D4)。
- [ ] demo 视频里用的 **exact profile 真实存在**且可复现。
- [ ] 每个对外 hero 数字有 repro 元数据(设备/profile/engine md5),benchmark profile 真实存在。
- [ ] license matrix 完成,无未澄清的可分发性风险。

> ⚠️ **数据纪律(codex 审核结论):** 对外只用**已验证 + 有出处 + 有 repro 元数据**的数字。当前 CustomVoice int4 仅 experimental(无 RTF/TTFA、未过 streaming worker),**不进 hero/README/Show HN**;ASR int4 是裸解码契约验证、未过 production worker;Base int4+fp8 的 N=2 仍待 gate(只有 fp16 N=2 实测)。详见 `benchmarks-dataset.md` GAPS + repro 要求。

---

## 六、竞品传播复盘(已补:打法验证)

> 全文 `competitor-research.md`(10 个项目 × 6 维度,带源 URL)。下面是对第四节传播策略的**实证校准**。

**对标定位被证实成立:**
- **"Pipecat for the edge" 锚点直接可用** —— Pipecat 几乎全是云后端,"端侧/本地/量化"正是它给不了的对比维度 = VoxEdge 护城河。tagline 直接用 `VoxEdge — Pipecat for the edge.`
- **最像的功能竞品是 sherpa-onnx**,但它是"组件库"不是"可对话成品"(无 LLM/打断/轮次/开箱即用 app)。VoxEdge 差异化叙事 = **"成品语音 agent + 机械臂 wow"**,并补 sherpa 缺的 Show HN 首发与一键 demo。

**七条被验证的成功要素(可迁移):**
1. **首发主战场 = Show HN,标题放可量化钩子**(数字/对标/设备)。范例分数:llama.cpp 1311 / RealtimeVoiceChat 524 / whisper.cpp 399 / Pipecat 346。
2. **"X for Y" 类比锚**是最高杠杆的一句话定位。
3. **降低"亲自试"门槛**:live demo > `curl|sh`/`pip` > `docker compose up`。VoxEdge 至少做到 `docker compose up`(呼应 workstream D)。
4. **"runs on X device" 设备奇观**是端侧独有爆点(whisper.cpp 的 iPhone、llama.cpp 的 MacBook)。VoxEdge 的 Jetson/RK/Pi + **机械臂**是天然弹药。
5. **两类内容并行打两层受众**:设备奇观录屏(应用开发者,情绪)+ 硬核 benchmark 对比表(系统开发者,数字)。whisper.cpp vs faster-whisper 正是这两条路线范本。
6. **生态兼容做长期护城河**:OpenAI-compatible API + 多后端可插拔,首屏强调。
7. **可信度信号 > 营销辞藻**:真机可复现硬数字(与项目内部 perf 纪律一致)、支持矩阵、谦逊工程语气。

**首屏 README 优先级(被验证的排序):** ① emoji headline + "Pipecat for the edge" 副标 → ② 一条延迟硬数字 + 设备名("end-to-end <XXXms on Jetson Orin Nano 8GB") → ③ 机械臂 demo 视频置顶 → ④ 一行 `docker compose up`(视觉焦点) → ⑤ 设备×后端支持矩阵(来自 `benchmarks-dataset.md` 表A) → ⑥ "what you can build" 场景(表B)。**架构图移第二屏。空泛卖点删。**

**机械臂 demo 怎么拍(本赛道最稀缺 wow):** 一镜到底"说话→机械臂立即动→全程离线(拔网线/飞行模式入镜)";屏角叠加实时延迟计时;演示**打断(barge-in)**;入镜真实硬件型号牌;30–60s 主视频 + 2–3min how-it-works(含 `docker compose up` 复现)。

**benchmark 呈现(打系统受众):** faster-whisper 式对比表——VoxEdge(int4/fp8) vs 全精度 vs 云,纵轴 = 端到端延迟/TTFA/显存/**是否离线**/**每分钟成本**。"$0/min 离线 vs 云 per-minute 计费"是 VoxEdge 独占维度。数据全部出自 `benchmarks-dataset.md`,真机可复现附命令。

**四条反面教训(规避):**
- sherpa-onnx:功能最强但"组件库"无爆点、慢热 → VoxEdge 必须以成品 agent + demo 破局。
- faster-whisper:硬数字极强但无 demo/无首发 → "墙内开花"被低估 → 光有数字不够,必做首发 + 配 demo。
- LiveKit:借势 OpenAI 双刃剑 → 护城河叙事落在自身能力(边缘性能+量化+多设备),别押单一合作方。
- 几乎所有高传播项目首屏都**不放大架构图** → 首屏只留"定位+数字+demo+一行命令"。

**渠道顺序:** Show HN(标题用延迟数字/设备奇观 + 第一人称痛点开场)→ 同步 r/LocalLLaMA + r/selfhosted + r/robotics + Jetson/RPi 设备社区 → X 机械臂短视频争取 Jetson/Seeed KOL 转发 → 进 NVIDIA Jetson / Awesome-edge-AI 列表(第二曲线)→ "每设备/每后端一条 demo"维持复利小爆点。
