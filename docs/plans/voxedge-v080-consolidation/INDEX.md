# VoxEdge v0.8.0 整合 — 执行入口 (INDEX)

> 给**执行者**的总入口。本目录是 VoxEdge 边缘语音栈统一到 TensorRT-Edge-LLM **v0.8.0** + 开源传播的完整规划。
> 经 3 轮 codex 独立审核 + 真机实证 + 多 agent 调研收敛。最后更新 **2026-06-21**。

---

## 0. 怎么读(顺序)

| # | 文档 | 作用 | 谁先读 |
|---|---|---|---|
| 1 | **本 INDEX** | 状态 / 决策 / 环境 / 从哪开始 | 所有人 |
| 2 | [consolidation-plan.md](consolidation-plan.md) | 主计划:现状图 A、目标架构 B(+B.1 五仓)、工作流 C0–C7 / D / E / F(测试 gate)/ G(清理)/ H(回归策略)+ dispatch 护栏 | 执行者必读 |
| 3 | [c0-patch-ownership-manifest.md](c0-patch-ownership-manifest.md) | **迁移 day-1 依据**:20 项 runtime/patch 归属(C1/C2/C3)+ 双写风险 + 源vs封装边界 | 动 fork/jve 前必读 |
| 3b | [c2-repin-spec.md](c2-repin-spec.md) | **C2 可执行 spec**(codex 设计,带 file:line):逐 patch 处置 + 重生成命令 + md5 验收点 + prefix-multiturn 不可复现新发现 | 做 C2 时必读 |
| 4 | [benchmarks-dataset.md](benchmarks-dataset.md) | 性能/能力唯一真相(表A性能 + 表B最佳实践 + GAPS + repro 元数据) | 写 README/BENCHMARKS 前 |
| 5 | [positioning-and-propagation.md](positioning-and-propagation.md) | 对外定位/分层/传播/license 矩阵 | 做发布/文档时 |
| 6 | [competitor-research.md](competitor-research.md) | 10 个同类开源项目传播复盘 | 做发布时 |
| 7 | [code-structure/](code-structure/) | 五仓 AST 结构分析(在正确分支 @ ref 上做的) | 需要代码定位时 |

**角色分工(沿用 CTO/spec/executor 三段式):** 主线程定 spec 边界 + 串联 + 自验关键数字;codex 出设计/审核(只读带 file:line);general-purpose 照 spec 实施(build/deploy/test)。**执行体 prompt 必带护栏**(见 plan「DISPATCH NOTES」+ 「Preflight 防误触」)。

---

## 1. 当前状态(2026-06-21)

### 🔑 重大状态修正(2026-06-21,执行中发现)
> 本节 INDEX 原写「C1–H 全未开始」**已过时**。两轮只读侦察查清:**v0.8.0 迁移其实已在 `jetson-voice-engine` 的 `feat/edgellm-v080-migration` 分支(已 push,31 commits)做了约 80%** 并留完整验收链(到 v080-0026)+ 自带回归 harness。已落地且 orin-nx 实机验收过(2026-06-10):ASR streaming Phase1-5、CustomVoice 9-row(feat 3e0bb0b)、MOSS port(8e0dcf0)、TTS N=2 batch-lane(9002b04=maxBatchSize=2)、full ASR→LLM→TTS pipeline、**real serve gate PASS**、一键 compose。
>
> **但有三大真实缺口(=真正剩余工作的核心):**
> 1. **C2 未做**:`engine-overlay/UPSTREAM_PIN` 至今仍 v0.7.1(连迁移分支也没改);feat 分支的 v0.8.0 工件是**绕过 overlay** 直接 build 的。C2=把 overlay 机制正式对齐 v0.8.0(见 [c2-repin-spec.md](c2-repin-spec.md)),是合并 main 的前置。
> 2. **C3 未做**:fork 无 ASR worker 唯一源;N≤0 guard 从未落地(仅 LOG_ERROR)。
> 3. **prefix-multiturn/N>1 streaming 路径不可复现**:依赖未发布的 `asr-b2` 引擎 → 当前可复现/可合并的 v0.8.0 = plain `asr`+accumulate(已发 HF + serve-gate 验证)。详见 c2-repin-spec §7。
>
> **执行策略(verify-first,零衰退):** 先在 orin-nx 用分支自带 harness(`~/project/seeed-local-voice/bench/regression/run_v080_regression.py` + goldens `v071-edgellm`)固化 canonical v0.8.0(plain asr+accumulate)为基线 → 再 C2 re-pin → 每改一处对基线复跑 harness + md5 对账(vs v080-0017 HF manifest)证明零衰退。RK 不在这条 Jetson 链(分支 Jetson-only)。

### ✅ 已完成(执行者不要重做)
- **N>1 能力全实机验证(2026-06-21):** ASR N=2+streaming through-service gate PASS(orin-nx);TTS N=2 slot-pool int4 staggered gate PASS(orin-nano,~4GB fits 8G/16G);TTS N=2 shared-engine gate PASS(orin-nano,2nd slot +1.6GB 省~436MB)。fork 整合分支 `suharvest/port/qwen3-tts-base-v080-n1n2`(a361221)=Base+streaming worker+shared-engine。jve `feat/c2-repin-v080`(4a0b837)pin 指它,build.sh 编 shared worker。worker:c00a0752(独立)/190178f6(shared)。引擎:asr-b2 4122dfcc(HF)、talker-b2 f7339e02(已存 orin-nx)、int4 talker(HF base int4fp8 245.9MB)。
- **数据丢失隐患全清零:**
  - `tensorrt-edge-llm`(小写副本)的 4 个生产关键 commit(SlotPool 抽取 / ASR-assistant prefix / N-instance slot-pool worker / shared-engine ctor)→ 已 push 到 `suharvest/v071/customvoice-product`(FF `1668470..893ba2a`)。
  - int4 export driver 分支:`wip/native-int4-talker`(ff2318e)+ `wip/asr-int4-decoder`(c80bcc0)**都在 suharvest fork**;未提交 working 文件(`quantize_talker_stage1_bigcalib.py` + `cv-int4-derisk/*`)已拉回 Mac。
- **全仓本地备份**:`~/project-backups/20260621-133543/`(6.6GB,97 bundle 含全部未推送 + 35 diff + untracked 内容 + 本规划文档 + wsl2-recovered/ 的 int4 drivers)。
- **stateful code2wav × N=2** 真机+源码定论:flag 是 no-op(没接线,非不安全),收益增量,**backlog**(详见 benchmarks-dataset GAPS#10 + c0 manifest W2 旁注)。
- **E(README/BENCHMARKS/runbook 对外文档)= DONE(2026-06-21):** 用已验 N>1 数字落地对外文档 ——
  - `BENCHMARKS.md`(repo 根,新建):v0.8.0 N>1 表(ASR N=2 streaming gate v080-0023、TTS N=2 int4 slot-pool + shared-engine VRAM、零衰退、int4 vs fp16 talker),每行带设备/日期/gate。
  - `README.md`:加 Performance「v0.8.0 Concurrency (N>1)」段 + Key Features + Changelog 条目 + **统一后端结构**说明(recipes/HF_ARTIFACTS/docs/AGENTS 各后端平起,fork vs self-authored 已 DIVERGENCE 说明)。
  - `docs/deploy-v080-n1n2.md`(新建,D 折入):拉镜像 `v0.8.0-n1n2-rebake` + 引擎(HF / 挂载隔离卷)+ profile `jetson-edgellm-v080-n2`(session-gate 三件套 LAZY_TTS=1 + OVS_TTS_WORKER_CONCURRENCY=2 + OVS_MAX_CONCURRENT_SESSIONS=2)+ `docker run` 示例 + int4/shared-engine 选用。明确**不部署 seeed-orin-nx 生产**。
  - `benchmarks-dataset.md`:GAPS#1/#3(N=2 int4 待 gate)关闭 + 表 A 补 N=2 实测;详见该文件「v0.8.0 N>1 实测(2026-06-21)」段。
  - 注:C7(镜像 bake)/D(compose 实改)/RK 链仍按各自 workstream 推进,本轮只交付 E 文档(纯 Mac,不 build/部署)。
- **模型 license 调研完成**(见 positioning §六 license 矩阵)。
- 规划经 3 轮 codex 审 + must-fix 全闭合。

### ⏳ 待执行(本计划的工作流,尚未开工)
- C0 manifest 已产出(本目录);**C1–C7 / D / E / F / G / H 全部待执行**。
- 两个外部确认:核 MOSS-TTS-Nano 仓库根 LICENSE 文件是否落地 Apache 全文;(CV license 已定=跟随 Qwen 官方不改)。

---

## 2. 决策记录(都已拍板,执行者按此办,勿再议)

| 决策 | 结论 | 依据 |
|---|---|---|
| **迁移范围(2026-06-21)** | **含 N>1/streaming**(用户定):不止最小可复现 generic 路径,要把并发(N=2 ASR + maxBatchSize=2 TTS)+ 流式(prefix-rollback)纳入本次迁移并产线化。后果:**必须重建丢失的 asr-b2 引擎 + 发布到 HF(补可复现缺口) + 重证 maxBatchSize=2 共享 Talker 并发安全(F5) + 复活 C2a-cont 暂 drop 的 streaming 栈**。 | 用户 AskUserQuestion |
| **shipped vs R&D 真相(2026-06-21)** | 已发布 v0.8.0(v080-0017 manifest)= 最小 generic 路径(base f9cc746 + v080-0007 customvoice + v080-0008 cutedsl + generic workers);N>1/streaming 当年是 R&D、未进 shipped 二进制、asr-b2 引擎已丢失。本次迁移要把 R&D 提升为产线。 | C2a-cont(v080-0016/0017/0019 + Dockerfile 实证) |
| **Base vs CustomVoice 拆变体(2026-06-21)** | customvoice 9-row langId 与 Base speaker-encoder kernel 在 talker prefill 冲突(8-row vs 9-row,不能同一二进制)→**Base 与 CV 是两个 build 变体**。本次合 **Base TTS N>1**(已验证);**CV 作独立一等变体**(同 slot-pool/shared-engine N>1 worker + CV int4 引擎 + v080-0007 patch + 独立 build,后续)。 | 用户 AskUserQuestion |
| **TTS N=2 = slot-pool(非 batch-lane)(2026-06-21)** | 生产走 slot-pool(独立 lane,staggered 友好,与 ASR SessionLaneManager 一致);batch-lane(lockstep,短等长)不做。shared-engine ctor 已实现(权重共享~1×)。 | 用户问 staggered + 实测 |
| **shared-engine ctor(2026-06-21)** | 净新实现(commit a361221,v071 无有效版);N=2 2nd slot 仅 +1.6GB(context/KV 非二次权重),vs 独立省~436MB/slot。整合进 fork 分支 port/qwen3-tts-base-v080-n1n2。 | orin-nano gate 实测 |
| 最终仓库结构 | 3 层 **5 职责单元 / 6 物理仓**(RK=2 仓算 1 后端);对外只一个品牌 VoxEdge | plan B.1 |
| fork vs jetson-voice-engine | 三类变更模型:C1 上游bug + C2 本地runtime扩展 → **源在 fork**;C3 overlay/recipes → jve。**任何 runtime 改动只在 fork 落地,jve patch 自动生成** | plan B.1 / c0 manifest |
| int4/fp8 export drivers 归属 | **`jetson-voice-engine/recipes/`**(不放 fork),pin fork commit,只调 fork export API | plan C6 / c0 |
| CustomVoice | **一等公民**(用户定):走 int4,移植 P5 9-row 条件到 v0.8.0,**DROP W1 w8a16**(EOS 破不可行);过 F4+F5-CV 后可进默认 | plan C1b/B / c0 §4 |
| stateful code2wav N>1 | **backlog**(没接线,收益增量,RTF 已<1 性价比低) | benchmarks GAPS#10 |
| 默认精度翻转 | **C1 只加 opt-in,C1b 才翻默认**,且挂 F-gate 全绿;CustomVoice 默认前过 F4+F5-CV | plan C1/C1b |
| 开源 license | 主体 Apache/MIT 可开+商用;**NLLB=CC-BY-NC 剥成可选非商用**;FunASR(Paraformer/SenseVoice)需署名+附协议;**CV 跟随 Qwen 官方不加料**;MOSS 待核 LICENSE | positioning §六 |
| N≤0 prefill guard | **尚未落地**(分支里只有 empty-fullText LOG_ERROR),需新写并提交 fork(C1) | plan C3(a) / c0 §4 |
| stateful 等性能优化定性 | B2(M=1 GEMV)/A2(streaming worker)= C2(我们路线,可日后 PR) | c0 §4 |
| qwen3asr_rk | 外部库(非我们),**移出整合范围,不删不并** | plan G |

---

## 3. 环境与访问(执行者必须遵守)

### 设备(用 fleet,完整路径见下)
- **可用(profiling,可 stop/测)**:`orin-nano`(v0.8.0 build 树 `~/project/edgellm-v080-build`,ENABLE_CUTE_DSL=OFF)、`orin-nx`、`wsl2-local`(RTX,有 modelopt,做 int4/fp8 导出;flap-prone,跑长任务先 mask 定时器+tmux)。
- **🚫 绝对不碰**:`seeed-orin-nx`(生产机械臂栈)+ 生产 `speech-models` 卷。测试一律用隔离卷 `~/comp-e2e-models`。
- Fleet 全路径(子 shell alias 不生效):`uv run --project ~/project/_hub python ~/project/_hub/fleet.py`。**flags(--sudo/--timeout/--json)放设备名之前**;`--sudo` 用于 apt/systemctl/docker/写 $HOME 外。坑:zsh 里别把 fleet 路径塞进带空格的变量(词分裂)。

### fork remote(git push 高危,看清楚)
- fork 工作区里 **`origin` = NVIDIA 官方仓**!推到我们 fork 必须用 **`suharvest`**(或 `origin-claude`,同 URL)。**绝不 `git push origin`**。
- canonical fork = `https://github.com/suharvest/TensorRT-Edge-LLM.git`。v0.8.0 主分支 `port/qwen3-tts-base-v080`;int4 drivers `wip/native-int4-talker` + `wip/asr-int4-decoder`;CV/v071 在 `v071/customvoice-product`。

### HF 预编译产物(终端用户消费的东西)
`harvestsu/qwen3-asr-0.6b-int4-v080`、`harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8`、`harvestsu/qwen3-tts-0.6b-customvoice-jetson-trtllm-int4fp8`。

### 备份与已恢复资产
- 全仓备份:`~/project-backups/20260621-133543/`(bundles/ + uncommitted/ + untracked-content/ + scratch/ + planning-docs/ + wsl2-recovered/)。
- 恢复方式见该目录 `MANIFEST.txt`。

---

## 4. 从哪开始(day-1,无破坏性)

1. **建 C0 patch 清单的可执行版**:照 [c0-patch-ownership-manifest.md](c0-patch-ownership-manifest.md),逐个 v0.7.1 patch 对 v0.8.0 base 跑 `git apply --check`,确认哪些已被上游吸收(归零)、哪些需保留。
2. **C6 收尾(recovery 已做)**:把已恢复的 int4 drivers 落进 `jetson-voice-engine/recipes/`,写 recipes README + pin fork commit。
3. 之后按 plan 执行顺序:**C(统一 v0.8.0)→ D(一键部署)→ E(README/BENCHMARKS/Recipes)**,每步过对应 F-gate(H 节回归策略是硬约束:F 是各 workstream 的 exit-criteria,不是收尾)。

**所有 build/deploy/test 派发必带**:EVIDENCE 段(md5 + 原始验证输出 + before/after + `docker logs|grep -iE error|crash|fail`)、Fleet 规则块、Preflight 防误触、禁破坏性操作(见 plan「DISPATCH NOTES」)。
