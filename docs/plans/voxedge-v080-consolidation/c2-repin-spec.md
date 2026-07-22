# C2 — Re-pin jetson-voice-engine overlay 到 v0.8.0（可执行 spec）

> 由 codex 独立设计 + 主线程综合（2026-06-21）。执行体照此开工。
> 配套：[c0-patch-ownership-manifest.md](c0-patch-ownership-manifest.md)、[consolidation-plan.md](consolidation-plan.md) §C2、[INDEX.md](INDEX.md)。

## 0. 核心认知（先读，否则会误解整个迁移）

`feat/edgellm-v080-migration` 分支**已经做了约 80% 的 v0.8.0 集成**（ASR streaming Phase1-5、CustomVoice 9-row、MOSS port、TTS N=2、full pipeline + serve gate PASS on orin-nx 2026-06-10），**但它是绕过 overlay 系统做的**：直接用 fork `f9cc746`(v0.8.0 base) + v080 patch 在 Jetson 上 build，baked 进镜像 `:v0.8.0-edgellm-20260610e`。**`engine-overlay/UPSTREAM_PIN` 从未更新，至今仍是 v0.7.1 `364769…`（连迁移分支也没改）。**

**所以 C2 = 把 overlay 机制本身正式对齐到 v0.8.0**，让 `build.sh`（clone UPSTREAM_PIN → copy addon → apply patches）能在 v0.8.0 base 上复现出与 feat 分支/HF 发布一致的引擎。这是合并 main 的必要前置。

## 1. 正确 Base
- **UPSTREAM_PIN 应 = `f9cc74623d95d7acf1addab6026b9d410ba81f52`**（NVIDIA `origin/release/0.8.0` 顶，"Merge #101 dev-release/0.8.0"）。
- fork `port/qwen3-tts-base-v080` = base 上 6 commit / 9 文件：`873ca22`(fp8-embed) `50b8670`(M=1 GEMV) `26a4a69`(cuBLAS-free GEMM) `867b74d`(cutedsl shim) `ba9ecdb`(Base speaker-encoder) `10b338d`(streaming worker)。

## 2. 逐 Patch 处置表

| Patch | 处置 | 关键细节 |
|---|---|---|
| P1 `0001` orin-tegra-build-compat | **clean-apply 保留** | `patches/0001:1,47,83`（CMake/CuteDSL/plugin 注册）；可拆 build-compat 与 plugin 两段 |
| P2 `0002` weight-streaming-budget | **重基** | v0.8.0 `builderUtils.cpp:256-264` 变动，旧 patch 落 `:266`；定位新行号再 apply |
| P3 `0003` asr-streaming-session | **废弃→改用 feat 分支 worker** | v0.8.0 `audioRunner.cpp:506-587` 与 fork port 等同；旧 patch 重写整个 session runtime 已过时。[VERIFY: v080-0024/0026 增量-KV/prefix patch 是否需并入] |
| P4 `0004` tts-slotpool | **废弃→替换为 fork `10b338d` streaming worker** | 旧 slot-pool 在 `qwen3OmniTTSRuntime.cpp:1059` 冲突；fork worker md5 `fe7c0736`，overlay 旧版 `0038bbd6`（过期） |
| P5 `0005` customvoice-9row | **直接采用 feat 分支，不重做** | 来源 feat `3e0bb0b` = `engine-overlay/patches/v080-0007-customvoice-language-conditioning.patch`，覆盖 `qwen3OmniTTSRuntime`/`talkerMLPKernels`/`export.py`。`experimental/llm_loader` 路径 v0.8.0 已移，不能对 v0.7.1 路径重做 |
| P6 `0006` server-sse | **重基** | `api_server.py:33` 因 server 重写冲突；拆 SSE-fix / tool-call 两段（SSE→可上游，**禁自动提交**） |
| P7 `0007` server docs | **随 P6 重基或 drop** | 与 P6 绑定 |
| P8 `0008` build-misc-example-registration | **clean-apply 保留** | `patches/0008:15-78`；C3 hook 点 |

## 3. 重新生成 patches/addon（执行体命令序列）

```bash
# Step 1 — 从 fork port 对 v0.8.0 base 生成 patch 集
cd ~/project/TensorRT-Edge-LLM
git fetch suharvest                       # 绝不 push origin(=NVIDIA)
git switch port/qwen3-tts-base-v080
git format-patch --no-stat -o /tmp/jve-v080-patches \
  f9cc74623d95d7acf1addab6026b9d410ba81f52..HEAD     # 6 patch

# Step 2 — 验证 9 文件覆盖
git diff --name-status f9cc746..port/qwen3-tts-base-v080
#   预期: talkerMLPKernels/*, qwen3OmniTTSRuntime.{h,cpp}, slotPool.h,
#         examples/omni/*, quantize_text_embedding_fp8.py, export.py

# Step 3 — addon vendor：
#   新增 qwen3_tts_streaming_worker.cpp ← fork port 10b338d (md5 fe7c0736)
#   采用 qwen3_asr_worker.cpp / qwen3_tts_worker.cpp ← feat 分支(见 §4 md5)
#   MOSS v080-0011 patch + CustomVoice v080-0007 patch ← feat 分支

# Step 4 — 写 pin
echo "f9cc74623d95d7acf1addab6026b9d410ba81f52" > engine-overlay/UPSTREAM_PIN
```
`build.sh` 契约不变（clone pin → copy addon → apply patches in order）。

## 4. 直接采用 feat 分支成果（带 md5，避免重复劳动）
- `native/edgellm_voice_worker/qwen3_asr_worker.cpp` md5 `ab09b992…`（feat `9775f35`/`dae9002`/`c6bf483`/`e5f0999` 累积）
- `native/edgellm_voice_worker/qwen3_tts_worker.cpp` md5 `2e2de126…`
- `deploy/docker/worker_io.voxedge-patch.py` md5 `a8ff1b7c…`
- MOSS v0.8.0 patch（feat `8e0dcf0`，DIVERGENCE replay PASS）
- CustomVoice 9-row patch（feat `3e0bb0b`）

**不采用**：overlay 旧 `addon/.../qwen3_tts_streaming_worker.cpp`（md5 `0038bbd6` v0.7.1）→ 换 fork port `fe7c0736`。

## 5. ⚠️ 验收判据修正（2026-06-21 实测）：md5-match 不可用 → 功能等价

**实测发现：TRT 引擎序列化非确定性。** 同一 ONNX + 同一 flags + 同一 TRT 10.3.0.30，重建 talker-b2 引擎得 md5 `041b83ff` ≠ 原 `f7339e02`（config 功能相同，~3.6KB tactic-table 抖动）。引擎/二进制**不是 byte-可复现**的（TRT 按设备计时选 kernel tactic，嵌入 timing-cache 状态）。

**所以下面这张「md5 应一致」表只能用于「下载完整性校验」(确认从 HF 拉的工件没损坏)，不能作为「re-pin 重建复现」的判据。** re-pin / 重建的**正确 no-regression 判据 = 功能/行为等价**：①重建产物能干净编译/构建 ②部署后过 gate（ASR：CER 对基线快照不退；TTS：phase5b M5 concurrent==solo RVQ hash + audio 可懂 + 0 CUDA error；N=2：30-burst 无崩）③对 `v080-c2-before-20260621` 基线快照不衰退。**已存在的 canonical 引擎（HF/设备上 md5 已知的那份）应直接复用，不要为了"复现"去重建引擎**（重建只会得到功能等价但 byte 不同的新引擎，反而要重新 gate）。

### 下表 = 下载完整性参考（NOT 重建判据）
来源 `v080-0017-hf-artifacts-manifest.md`（repo `harvestsu/seeed-local-voice-artifacts`，base `sm87-trt10.3-jp6.2/v0.8.0/`，**Orin NX / TRT10.3.0.30 / CUDA12.6.68 / JP6.2** 产物，34 文件 7.577GB）：

| 产物 | 期望 md5 |
|---|---|
| plugin `libNvInfer_edgellm_plugin.so` | `8f004bb4…` |
| `qwen3_asr_worker` binary | `5ddbcdf7…` |
| `qwen3_tts_worker` binary | `22216e8d…` |
| `moss_tts_nano_worker` binary | `6a03bdf5…` |
| ASR audio encoder engine | `ede676fb…`（min-chunk-1 重 stage 版） |
| ASR LLM engine | `b133dff2…` |
| TTS talker engine | `471d36d8…` |
| CP engine | `baff21ea…` |
| code2wav engine | `566c389e…` |

⚠️ md5 是 **Orin NX** 上的 build 产物；**Orin Nano / 不同 CUDA 会不同** → 在哪台 build 就对哪台的 serve-gate 记录核对。

## 6. 次序 + 风险
- **次序**：C2(re-pin overlay) → C3(fork 落 N≤0 guard + ASR worker 回 fork 唯一源) → C4 → C5。
- **N≤0 guard**：fork port 与 feat 分支**都没有**（仅 `qwen3OmniTTSRuntime.cpp:1655` 的 empty-fullText LOG_ERROR）。C3 必须**单独新写**提交 fork，**不能在 C2 顺带写**。
- **ASR worker 唯一源**：fork port **不含** asr_worker；feat 分支是当前最新来源。C2 阶段只是 vendor feat 当前快照；C3 才正式 cherry-pick/移植进 fork 建唯一源。执行体第一步先锁定 feat 分支 ASR worker 的最终 commit hash + md5。
- **TTS N=2 设计变更**（feat `9002b04`：maxBatchSize=2 共享 Talker，取代 slot-pool 复制）→ 并发安全性与旧路线不同，F5 gate 须重证（30-burst 无 CUDA error + md5 byte-identical）。

## 7. ⚠️ 新发现：prefix-multiturn / N>1 streaming 路径不可复现（必须先解决再上生产）
- 唯一在 orin-nx 的 v0.8.0 语音镜像 `:v0.8.0-edgellm-20260611-prefix-multiturn` 依赖一个 **`asr-b2`（batch-2 streaming）引擎**，该引擎**设备上没有、HF 也从未发布**（v080-0017 manifest 只发了 plain `asr`）。它是 2026-06-10 之后的临时 build，已丢失。
- 含义：① **canonical / 可复现 / 可合并的 v0.8.0 = plain `asr` + accumulate 模式**（= `:…-20260610e` 镜像路径，引擎已在设备 + 已发 HF + serve-gate 验证）。C2 基线与验收都走这条。② prefix-multiturn / N>1 ASR streaming 是更靠后的增量特性，**其引擎未归档/未发布 → 当前不可复现 → 不能按现状合并**。需在 C4 之后重 build asr-b2 引擎 + 发 HF + 单独基线，才能让该路径进生产。
