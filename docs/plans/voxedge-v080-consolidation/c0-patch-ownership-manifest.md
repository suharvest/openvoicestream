# C0 — Patch Ownership Manifest (v0.8.0 整合)

> 整合计划 `consolidation-plan.md` workstream **C0** 的交付物。
> 三类模型:C1 上游bug(源=fork,可上游,合并归零)/ C2 本地runtime扩展(源=fork,长期维护)/ C3 overlay·编排·recipes(源=jve)。
> 原则:**任何 C++/CUDA runtime 改动的"源"都在 fork;jve overlay patch 是从 fork 自动生成的"封装",不是第二份手写源。**

## 0. 最重要的总体判定(先读)

- **jve 当前 pin 在 v0.7.1**(`UPSTREAM_PIN=364769…`=NVIDIA v0.7.1,`fork_branch=v071/customvoice-product`,90 commits ahead),**不是 v0.8.0**。
- 目标 = fork `port/qwen3-tts-base-v080`(相对 `origin/release/0.8.0` 仅 **6 commit / 9 文件**)。
- **本质:re-pin 到 v0.8.0 + 删除冗余封装,而非把 v0.7.1 的 8 个 patch 照搬过去。** v0.7.1 时代多数 patch 在 v0.8.0 base 已原生。
- **int4 native kernels/plugin = 上游 v0.8.0 原生**;我们的 int4 工作只剩 **export drivers(C3)**,不需要任何 native int4 overlay。

## 1. 归属清单主表

| # | 改动 | owner | class | source (file) | packaging (jve) | 可上游 | removal cond | gate | engines | 证据 |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Base speaker-encoder(external-embedding)移植 v0.8.0 | fork | C2 | `qwen3OmniTTSRuntime.{cpp,h}` | re-gen from port | 否 | 长期 | F-TTS e2e | tts-base | `ba9ecdb` |
| A2 | streaming worker(base,N>1 slot-pool)移植 v0.8.0 | fork | C2 | `examples/omni/qwen3_tts_streaming_worker.cpp`+`slotPool.h`(md5 `fe7c0736`) | jve addon copy **md5 `0038bbd6`=v0.7.1 过期** | 否 | 长期 | F-N2 | tts-base | `10b338d` |
| A3 | cutedsl cudart shim link 进 omni exes | fork | C1 | `examples/omni/CMakeLists.txt` | 0001 对应 hunk | 是 | 上游→删 | F-build | all sm_87 | `867b74d` |
| B1 | **cuBLAS-free tiled FP16 GEMM fallback** | fork | **C1** | `talkerMLPKernels.cu:278-490` | N/A | **是**(上游 #else 仅 LOG_ERROR,平台正确性补全) | 上游→归零 | F-build+TTS | tts sm_87 | `26a4a69` |
| B2 | M=1 warp-per-column GEMV | fork | C2 | `talkerMLPKernels.cu:330-401` | N/A | 偏否(可 PR) | 长期 | F-perf | tts sm_87 | `50b8670` |
| B3 | **fp8 text_embedding wiring**(5 lookup 点) | fork | **C2** | `qwen3OmniTTSRuntime.cpp:172-199,753,1067,1988,1998,2172` | N/A | 否 | 长期 | F-TTS+VRAM | tts-base | `873ca22` |
| B4 | fp8 embedding **export driver** | fork | **C3** | `scripts/quantize_text_embedding_fp8.py` | ⚠️ jve 异名 `scripts/quantize_embedding_safetensors_fp8.py`(§3 双写) | recipe | 随 model 重导 | F-export | tts-base | `873ca22` |
| B5 | export.py 放开 non-CustomVoice(Base 导出) | fork | C3 | `tensorrt_edgellm/scripts/export.py:1665+` | recipe | 是 | 上游→删 | F-export | tts-base | port diff |
| C-int4-rt | int4 native kernels/plugin | **上游原生** | — | `cpp/kernels/int4GroupwiseGemmKernels/*`,`plugins/int4{GroupwiseGemm,Moe}Plugin/*` | N/A | — | 不归我们 | — | all | base ls-tree |
| C-int4-drv | int4 talker/CP 量化 **export drivers** | fork `wip/native-int4-talker` | **C3** | `quantize_talker_stage1.py`/`stage2_export.py`/`quantize_cp_stage1.py`/`cp_stage2_export.py` | **应迁 jve recipes** | recipe | 随 model 重导 | F8 | tts int4 | `ff2318e` |
| D1 | **ASR stripLangTag 英文首词修复** | seeed(源)→**应回 fork** | **C1** | `seeed/deploy/asr-worker-v080/qwen3_asr_worker.cpp`(md5 `13b34dc`)+`.patch` | jve `native/.../qwen3_asr_worker.cpp`(异代 md5 `7885f2e`) | 是(纯字符串清洗 bug) | 落回 fork+上游→归零 | F-ASR | asr | seeed README+patch |
| P1 | 0001 orin-tegra-build-compat | jve | C1+C2 混 | fork CMake | `patches/0001` | 部分 | hunk 拆 | F-build | all | DIVERGENCE |
| P2 | 0002 weight-streaming-budget | jve | C1+C2 混 | fork builderUtils/llmRuntimeUtils | `patches/0002` | budget→PR;shared-engine ctor 留 | hunk 拆 | F-build+rt | all | DIVERGENCE |
| P3 | 0003 asr-streaming-session | jve | C2 | fork audioRunner/specDecodeRuntime | `patches/0003` | 否 | 长期 | F-ASR | asr | DIVERGENCE |
| P4 | 0004 tts-slotpool-concurrency | jve | C2 | fork qwen3OmniTTSRuntime+slotPool | `patches/0004` | 否(N=2 护城河) | 长期 | F-N2 | tts | DIVERGENCE |
| P5 | 0005 customvoice-language-conditioning(9-row) | jve→**源回 fork** | C2 | fork talkerMLPKernels(9-row prefix) | `patches/0005`(re-gen) | 否 | **用户定 CV=一等公民 → 必须移植到 v0.8.0**;长期维护 | F4 + F5-CV | tts-CV | DIVERGENCE |
| P6 | 0006 server-sse-disconnect + openai-api | jve | C1+C2 混 | fork api_server.py/engine.py | `patches/0006` | SSE-fix→PR(**禁止自动提交**);tool-call 留 | hunk 拆 | F-server | server | `0898b5f` |
| P7 | 0007 server-openai-api-docs | jve | C3(doc) | fork docs | `patches/0007` | 是 | 随 P6 上游 | doc | — | DIVERGENCE |
| P8 | 0008 build-misc-example-registration | jve | C3 | fork CMake 注册 | `patches/0008` | 否 | 长期 | F-build | all | DIVERGENCE |
| W1 | w8a16 quant kernel+plugin(v0.7.1) | jve | C2 | jve addon `w8A16LinearKernels/*`,`w8A16LinearPlugin/*` | addon | 否 | **DROP:CV w8a16 已确认不可行(EOS 破),不进 v0.8.0;CV 走 int4 而非 w8a16** | — | tts W8A16 | DIVERGENCE |
| W2 | MOSS runtime / stateful code2wav(v0.7.1) | jve | C2 | jve addon `mossTtsNanoRuntime.*`,`statefulCode2WavRunner.*` | addon | 否 | 长期 | — | moss-tts | addon ls + N=2 实测 md5 `cccc41a6` |

> **W2 旁注 — stateful code2wav × N>1(2026-06-21 调研 closed,backlog):** 当前 N>1 不走 stateful 的根因是 **"没接线"**(`StatefulCode2WavRunner` 在 worker 里只 `#include`+unused 形参,从未实例化;slot-pool 每 slot 只构造 stateless `Code2WavRunner`;flag 仅进 ready 元数据)。**不是安全墙** —— 历史 concurrent-reset 崩溃只在"共享单实例",runner 状态全实例私有,**per-slot 独立 stateful 实例原理安全**。收益 = 省 stateless 每 chunk 25 帧左上下文重算(Code2Wav ~81-87ms,export gate max_abs 5e-6),但 Talker/CP 才是延迟主项 → 增量非量变,仅 chunk 显著变小/超长流式才显著。要启用约 7 项(接线 per-slot stateful+per-slot stream/per-req reset/增量喂码/禁 async/显存 gate/ConvTranspose 相位质量门/CMake GLOB rerun),最大风险=ConvTranspose 相位 click。**判定:incremental headroom,非 blocker,当前 RTF<1 性价比低 → 放 backlog**,等"更小 chunk / 更低 TTFA"成硬需求再启用(届时已铺好 ~80%)。

## 2. v0.8.0 base 已原生 / 该删的冗余封装

【事实】① int4 native runtime 在 `origin/release/0.8.0` 已原生(port 未改却存在)→ 不需任何 native int4 overlay,只留 export drivers。② jve addon `qwen3_tts_streaming_worker.cpp`(md5 `0038bbd6`)= fork v0.7.1 副本 ≠ port/v080(`fe7c0736`)→ re-pin 后此封装作废,必须从 port 重新生成。
【推断,需 re-pin 后 `git apply --check` 逐 patch 验证】v0.7.1 的 `0001 build-compat`/`0002 weight-streaming`/`0006 SSE` 的部分 hunk 很可能已被 v0.8.0 吸收 → **整合执行体第一步 = 逐 patch 对 v0.8.0 tree base-apply,确认哪些归零**。

## 3. 双写风险(只认 fork 为源)

1. **ASR worker(最高危)**:≥5 个非 git 副本,三处并存且异代 —— seeed `deploy/asr-worker-v080`(md5 `13b34dc`,含 strip 修复)、jve `native/edgellm_voice_worker`(md5 `7885f2e`,异代用 runStreamingHop)、**fork 根本没有 asr_worker**。→ **必须先把 stripLangTag 修复落回 fork `examples/omni/qwen3_asr_worker.cpp` 建立唯一源**,seeed/jve 都降为封装。
2. **streaming TTS worker**:fork port(`fe7c0736`)vs jve addon(`0038bbd6` v0.7.1)→ 只能从 fork 重新 vendor,禁手改 jve 副本。
3. **fp8 embedding export driver 异名分叉**:fork `quantize_text_embedding_fp8.py` vs jve `quantize_embedding_safetensors_fp8.py` → diff 确认是否同逻辑;若是,删 jve 那份只认 fork。

## 4. 需要拍板的归属(决策点)

1. **N≤0 prefill guard**:【事实】port/v080 与 v071 分支里**都没有**显式 `N<=0` 守卫,只有 `empty fullText` 的 LOG_ERROR(`qwen3OmniTTSRuntime.cpp:1655`)→ **这是尚未落地的计划项,不是已存在的 patch**。判 C1(防垃圾音频正确性防御,源在 fork),需**新写**。
2. **CustomVoice 在 v0.8.0 是否保留 → ✅ 已定(用户 2026-06-21):CV = 一等公民。** 走 int4(非 w8a16)。后果:**P5(9-row CV 条件)必须移植到 v0.8.0**(源回 fork,C2);**W1(w8a16)DROP**(EOS 破不可行);需补 **F4(CV streaming worker+EOS+perf)+ F5-CV(CV N=2)** gate;license 跟随 Qwen 官方 CV(我们不增不改,无额外门槛)。
3. **B2(M=1 GEMV)/A2(streaming worker)定性**:技术上可上游(性能优化通用),按"我们路线/暂不上游"判 C2;若要上游改 C1。

## 关键 EVIDENCE
- v0.8.0 port delta(`git log origin/release/0.8.0..port/qwen3-tts-base-v080`)= 6 commit:`873ca22`(fp8 embed)/`50b8670`(M=1 GEMV)/`26a4a69`(cuBLAS-free GEMM)/`867b74d`(cutedsl shim)/`ba9ecdb`(Base speaker-encoder)/`10b338d`(streaming worker)。9 文件。
- jve `UPSTREAM_PIN=364769…`=NVIDIA v0.7.1(**非 v0.8.0**);`fork_branch=v071/customvoice-product`(90 ahead);patches 0001–0008。
- worker md5:fork port=`fe7c0736`;fork v071=`0038bbd6`;**jve addon=`0038bbd6`(过期)**;jve asr=`7885f2e`;seeed asr=`13b34dc`。
- rkvoice-engine:DIVERGENCE 写 `self-authored, NO upstream fork` → 完全独立,不参与 fork↔jve 归属。
