# Edge-Voice Stack — v0.8.0 Consolidation & Layering Plan

**Status:** REVIEW + PLANNING (read-only). No changes executed. Hand each workstream (C/D/E/F/G) to a separate executor.
**Author:** main-thread review, 2026-06-21.
**Scope guardrails for executors:** NEVER touch production `seeed-orin-nx` (live robot-arm stack). Test env = `orin-nx` / `orin-nano` / `wsl2-local`. Fork-vs-submodule policy: upstream-bug fixes → the TensorRT-Edge-LLM fork; purely-our-features → `jetson-voice-engine`.

Referenced artifacts (do not duplicate — read these):
- `~/project/seeed-local-voice/ARCHITECTURE.md`, `docs/CONFIGURATION.md`, `docs/REPRODUCE.md`, `deploy/IMAGE-TAGS.md`
- Memory: `qwen3tts_base_v080_port_2026_06_19.md`, `customvoice_talker_int4_eos_fail_2026_06_20.md`, `qwen3asr_int4_validation_forcelanguage_2026_06_20.md`, `leaf_config_refactor_w8a16_goal_2026_06_12.md`, `trunk_consolidation_2026_06_18.md`, `edgellm_v080_jetson_cutedsl_cuda126_2026_06_19.md`, `trt_edge_llm_fork_path.md`
- HF: `harvestsu/qwen3-asr-0.6b-int4-v080`, `harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8`, `harvestsu/qwen3-tts-0.6b-customvoice-jetson-trtllm-int4fp8`, `harvestsu/qwen3-edgellm-jetson-artifacts` (engine sets), `harvestsu/seeed-local-voice-artifacts`

---

## EXECUTIVE SUMMARY

The intended 3-layer architecture **already exists** in code (agent = `seeed-local-voice/agent/ovs_agent`; orchestration = `voxedge`; backends = `jetson-voice-engine` submodule + the TensorRT-Edge-LLM fork + `rkvoice-stream` + sherpa). The consolidation work is **not a rewrite — it is reconciliation and version-alignment**:

1. **Version skew is the core problem.** The `jetson-voice-engine` submodule is pinned to **TensorRT-Edge-LLM v0.7.1** (`engine-overlay/UPSTREAM_PIN` = `364769036…` / release/0.7.1), but every shipped v0.8.0 runtime, worker, engine, and the 3 new int4 models live OUTSIDE that submodule — on fork branches (`port/qwen3-tts-base-v080@873ca22`, `wip/native-int4-talker`), the `edgellm-v080-migration` scratch repo, overlay images, and HF. Production runs older overlay images. The submodule must be re-pinned to a v0.8.0 base and the divergent v0.8.0 patches (streaming worker, fp8-embed wiring, N≤0 prefill guard, M=1 GEMV, cuBLAS-free tiled GEMM, stateful-code2wav static-shape) folded into the canonical overlay per the fork-vs-submodule policy.
2. **The new int4 models are validated but not wired into the deploy registry.** Qwen3-ASR int4 (ZH CER 0%/EN WER 11%), Qwen3-TTS base int4+fp8 (RTF 0.44, TTFA 0.21s, −1.06GB), Qwen3-TTS CustomVoice int4+fp8 (−960MB/instance) are on HF and ASR-verified, but `configs/leaves/*` still point at fp16/fp8embed-only engine sets. Wiring them in is the highest-leverage, lowest-risk win.
3. **Duplicate checkouts** (`qwen3-edgellm-jetson` ≡ `jetson-voice-engine` same HEAD; `voxedge-engine` deprecated into `jetson-voice-engine`; `tensorrt-edge-llm`/`_qwen3tts-port` = stale fork clones; `seeed-local-voice-v080` = stale migration branch checkout) should be deprecated to one canonical repo per layer.
4. **One-click deploy + auto-pull machinery exists** (leaf/composition + `model_downloader` + `install.sh` + per-device compose) but the new int4 leaves and a profile-picker UX need finishing.

**Recommended path:** (A) wire the 3 int4 HF bundles into the leaf registry + a new v0.8.0 composition profile per device (low risk, big win) → (B) re-pin the submodule to a v0.8.0 base and consolidate the divergent v0.8.0 patches into one overlay (the structural fix) → (C) finish the one-click profile-picker + auto-pull UX → (D) docs/perf-table overhaul → (E) test-env build+smoke gate → (F) deprecate the duplicate checkouts. Each is independently dispatchable.

---

## A. CURRENT-STATE MAP

> "TRT ver" = which TensorRT-Edge-LLM version the component targets. Statuses: **LIVE** (production path), **migration** (v0.8.0 work-in-progress, not yet prod), **legacy** (older but referenced), **dup** (duplicate checkout), **scratch** (working dir).

| Project (path under ~/project) | Layer | Role | Branch | TRT ver | Status | Recommendation |
|---|---|---|---|---|---|---|
| **seeed-local-voice** [main @4532ed0] | TOP (agent + product) | The product: `server/` FastAPI, `agent/ovs_agent`, `configs/`, `deploy/`. THIS is "OpenVoiceStream". | main | drives v0.7.1(prod)→v0.8.0(target) | **LIVE / canonical** | KEEP — canonical top layer |
| seeed-local-voice-v080 [feat/edgellm-v080-migration @1fed2f8] | TOP | Stale checkout of an old v080 migration branch (2026-06-10). | feat/edgellm-v080-migration | v0.8.0 (early) | **dup / stale** | DEPRECATE — work merged to main per trunk_consolidation; delete checkout after confirming no unmerged commits |
| local_voice [main @2026-01-14] | — | Unrelated web app (backend/frontend/database) — not the edge voice stack. | main | n/a | **unrelated** | LEAVE — out of scope (different product) |
| sensecraft_voice [no git] | — | Non-git scratch dir. | — | n/a | **scratch** | FLAG for user — confirm contents, likely deletable |
| **voxedge** [main @b783037] | MIDDLE (orchestration) | "Pipecat for the edge": engine/conversation loop, backend ABCs, jetson/rk/sherpa backends, transport. Pure-python. **Has the shipped base-speaker injection (b783037).** | main | backend-agnostic; drives v0.8.0 worker via CLI | **LIVE / canonical** | KEEP — canonical middle layer |
| voxedge-engine [main @03664b0] | (was BOTTOM) | Thin overlay over NVIDIA TRT-Edge-LLM. **Self-deprecated 2026-06-01: "merged into jetson-voice-engine engine-overlay".** | main | v0.7.1 pin | **legacy / deprecated** | DEPRECATE — content lives in jetson-voice-engine/engine-overlay; archive repo |
| **jetson-voice-engine** [main @3750ea9] | BOTTOM (Jetson backend) | Submodule of seeed-local-voice (`third_party/jetson-voice-engine`). Overlay = UPSTREAM_PIN + addon/ + patches/ + DIVERGENCE.md. Renamed FROM qwen3-edgellm-jetson. | main | **v0.7.1 (UPSTREAM_PIN=364769036)** | **LIVE (prod) — but STALE version** | KEEP as canonical Jetson backend repo; **MUST re-pin to v0.8.0** (workstream C) |
| qwen3-edgellm-jetson [main @3750ea9] | BOTTOM | **Identical HEAD + same remote** as jetson-voice-engine (old name, stale clone). | main | v0.7.1 | **dup** | DEPRECATE — delete checkout (it IS jetson-voice-engine pre-rename) |
| **TensorRT-Edge-LLM** [v071/customvoice-product @893ba2a] | BOTTOM (the fork) | The real fork the user works from (per `trt_edge_llm_fork_path` memory). Holds the v0.8.0 TTS runtime work on branches: `port/qwen3-tts-base-v080@873ca22` (fp8-embed + streaming worker + base), `v071/customvoice-product`, `wip/w8a16`. **Source-of-truth for the TTS worker + runtime patches.** | currently v071/customvoice-product; v0.8.0 work on port/* | mixed (per branch) | **LIVE source-of-truth (fork)** | KEEP — canonical fork. Consolidate v0.8.0 branches; ensure `wip/native-int4-talker` (int4 drivers) is pushed here (see risk below) |
| tensorrt-edge-llm [v071/customvoice-product @893ba2a] | BOTTOM | Lowercase duplicate checkout of the fork, same HEAD. | v071/customvoice-product | v0.7.1 | **dup** | DEPRECATE — redundant with TensorRT-Edge-LLM |
| _qwen3tts-port [wip/fp8-embedding @873ca22] | BOTTOM | Underscore-prefixed scratch clone of the fork on the fp8-embed branch. | wip/fp8-embedding | v0.8.0 | **dup / scratch** | DEPRECATE — same content as fork port branch; delete after confirming pushed |
| edgellm-v080-migration [feat/edgellm-v080-migration @c01059b] | BOTTOM (scratch) | v0.8.0 migration scratch: holds v0.8.0 worker sources (qwen3_asr_worker.cpp/qwen3_tts_worker.cpp), N=2 patch, turnkey compose, pipeline driver. Worker shim source (b4ffaa41 lineage). | feat/edgellm-v080-migration | v0.8.0 | **scratch / migration** | HARVEST then DEPRECATE — extract the v0.8.0 worker sources + compose into jetson-voice-engine overlay, then archive |
| rkvoice-stream [main @9fe452c] | BOTTOM (RK backend) | Submodule `third_party/rkvoice-stream`. RK runtime. | main | n/a (RKNN) | **LIVE / canonical** | KEEP — canonical RK backend |
| rkvoice-engine [main @1f133f3] | BOTTOM (RK) | RK conversion pipeline (MIT, open-sourced). Companion to rkvoice-stream. | main | n/a | **LIVE (tooling)** | KEEP — RK conversion tooling repo |
| qwen3asr_rk [main @2026-03-06] | BOTTOM (RK) | Qwen3-ASR RK conversion scripts (convert_as_qwen3.py). | main | n/a | **legacy/tooling** | KEEP or fold into rkvoice-engine — confirm if still referenced |
| jetson-qwen3-speech [main @2026-04-08] | BOTTOM | Older Jetson Qwen3 speech cpp/python + wheels; predates the overlay model. | main | pre-v0.7 era | **legacy** | DEPRECATE/ARCHIVE — superseded by jetson-voice-engine overlay |
| jetson-voice [no git] | — | Non-git dir (likely NVIDIA's old jetson-voice). | — | n/a | **legacy/scratch** | FLAG — likely external reference, deletable |
| jetson-llm-benchmark [detached HEAD] | — | Benchmark scratch. | HEAD (detached) | n/a | **scratch** | LEAVE/ARCHIVE |
| jetson-sim [no git] | — | Isaac grasp sim scratch (per memory). | — | n/a | **scratch** | LEAVE — robot-arm sim, out of voice scope |
| frigate-on-jetson [main @2025-08] | — | Unrelated (video). | main | n/a | **unrelated** | LEAVE — out of scope |
| seeed-solutions-hub [feat/solution-seo @eb00a30] | — | Solutions website/hub. | feat/solution-seo-fields | n/a | **unrelated** | LEAVE — out of scope |

### Canonical repo per layer/backend (the source-of-truth decision)

| Layer / backend | Canonical repo | Source-of-truth note |
|---|---|---|
| TOP (agent + product) | **seeed-local-voice** [main] | The product. Branding "OpenVoiceStream". Holds `configs/`, `deploy/`, agent apps. |
| MIDDLE (orchestration) | **voxedge** [main] | Pure-python lib; flows into images as a built wheel (`deploy/wheels/voxedge-*.whl`). |
| BOTTOM — Jetson overlay | **jetson-voice-engine** [main] (submodule `third_party/jetson-voice-engine`) | Overlay (UPSTREAM_PIN + addon/ + patches/). **Holds purely-our-features.** Currently pinned v0.7.1 — must move to v0.8.0. |
| BOTTOM — the fork (upstream-bug fixes + TTS worker/runtime) | **TensorRT-Edge-LLM** [fork, suharvest remote] | **TTS worker + runtime patches source-of-truth** (NOT the submodule — historical reconciliation point). Upstream-bug fixes land here; the overlay's UPSTREAM_PIN points at the corresponding base. |
| BOTTOM — RK | **rkvoice-stream** (runtime, submodule) + **rkvoice-engine** (conversion) | RKNN path. |
| BOTTOM — sherpa/RPi/CPU | inside **voxedge** `backends/sherpa/` + CDN model_downloader | No separate repo. |

**Reconciliation rule (TTS worker source-of-truth):** the TTS streaming worker and runtime kernels are authored on the **TensorRT-Edge-LLM fork** (`port/qwen3-tts-base-v080`). The `jetson-voice-engine` overlay carries them as **addon/ files or patches** referenced from its `manifests/` so a build reconstructs the exact worker. Do NOT let the two drift: the overlay's `UPSTREAM_PIN` + `patches/` must reproduce the fork branch's worker byte-for-byte (validate via the worker md5s recorded in memory).

---

## B. TARGET ARCHITECTURE (clean 3 layers)

```
 TOP    seeed-local-voice  (product / "OpenVoiceStream")
        ├─ server/      FastAPI: HTTP+WS API, backend registry, hot-reload, profiles
        ├─ agent/       ovs_agent apps (voice_arm, voice_rebot_arm, multi_mode…)
        ├─ configs/     profiles (JSON) + leaf composition (YAML)  ← one-click selection
        └─ deploy/      Dockerfiles + compose + voxedge wheel + install.sh
                 │ imports voxedge.*                    │ git submodules
                 ▼                                      ▼
 MIDDLE voxedge  (orchestration, pure-python)     BOTTOM third_party/
        ├─ engine/    conversation loop                ├─ jetson-voice-engine  (Jetson overlay)
        ├─ backends/  ASR/TTS/VAD/LLM ABCs +           │     UPSTREAM_PIN=v0.8.0 + addon/ + patches/
        │   jetson/ rk/ sherpa/                         │     reconstructs the TRT worker at build
        ├─ transport/ InProcess + WebSocket             └─ rkvoice-stream       (RK runtime)
        └─ capabilities/
                 │ jetson backends shell out to the native worker
                 ▼
        TensorRT-Edge-LLM fork (suharvest)  ── source-of-truth for the TTS worker + runtime
            release base = v0.8.0;  branch port/qwen3-tts-base-v080 (streaming worker, fp8-embed,
            base speaker-encoder backport, M=1 GEMV, cuBLAS-free tiled GEMM, N≤0 guard);
            wip/native-int4-talker (int4-AWQ drivers).  → folded into jetson-voice-engine overlay.
```

**How the fork + submodule + edgellm-v080-migration patches relate (one source-of-truth):**
- The **fork** (`TensorRT-Edge-LLM`) is where C++/CUDA runtime + worker authoring happens. v0.8.0 work = `port/qwen3-tts-base-v080` (+ int4 drivers on `wip/native-int4-talker`).
- The **submodule** (`jetson-voice-engine`) is the *overlay* that pins an upstream NVIDIA commit and re-applies the fork's deltas as `addon/`+`patches/`. **Re-pin UPSTREAM_PIN from v0.7.1 → the NVIDIA v0.8.0 base**, then regenerate `patches/` from the fork's v0.8.0 branch so a clean `build.sh` reproduces the shipped worker.
- **edgellm-v080-migration** is *scratch* that proved out the v0.8.0 workers (`qwen3_tts_worker.cpp` CLI-shim b4ffaa41, `qwen3_asr_worker.cpp` 8cf0b8df) + N=2 patch + turnkey compose. **Harvest those into the overlay, then archive the scratch repo.** It is NOT a long-term home.

**Precision policy (target defaults, v0.8.0):**
- Qwen3-ASR (Jetson): **int4-AWQ decoder + fp16 audio encoder** (validated; replaces fp16 default).
- Qwen3-TTS base (Jetson): **int4-AWQ talker + int4 CP + fp8 text_embedding + fp16 code2wav** (strictly better than fp16: RTF 0.44 vs 0.69, −1.06GB).
- Qwen3-TTS CustomVoice (Jetson): **int4-AWQ talker + fp8 embedding + fp16 CP/code2wav** (talker 906→246MB; int4 RESOLVED — earlier "EOS-broken" verdict was a wrong-tokenizer harness bug, see `customvoice_talker_int4_eos_fail` memory). **用户定 CV = 一等公民(2026-06-21):走 int4(非 w8a16,w8a16 EOS 破已 DROP),P5 9-row CV 条件移植到 v0.8.0。** 默认翻转前过 F4 + F5-CV + license 免责声明(音色肖像权)。当前 production CV 仍 fp16 → int4 经 F4/F5-CV 验证后可进默认。
- These flip via `configs/leaves/models.yaml default_precision[jetson]` once leaves reference the int4 artifact paths.

### B.1 最终 canonical 仓库 = 3 层 5 个职责单元(6 个物理仓)(权威版)

整合**不减仓库数**,而是给每仓定唯一职责、删跨仓重复编码。对外只有一个品牌(VoxEdge=引擎层),其余用"关系"描述。
> 注:RK 后端是 **2 个物理仓**(`rkvoice-stream` 运行时 + `rkvoice-engine` 模型/工具),逻辑上算 1 个"RK 后端"职责单元。故 = 5 职责单元 / 6 物理仓。

| 层 | 仓库 | 唯一职责 | 对外叫法 |
|---|---|---|---|
| **TOP 应用** | `seeed-local-voice` | FastAPI server + ovs_agent apps + demos(机械臂/翻译/字幕)+ configs/deploy | "VoxEdge 参考应用/示例" |
| **MIDDLE 引擎** | `voxedge` | 实时管线(turn_driver/transport/capabilities)+ 后端 ABC + 各后端 Python 适配 | **VoxEdge**(品牌,`pip install voxedge`) |
| **BOTTOM 后端** | `TensorRT-Edge-LLM`(fork, suharvest) | Jetson C++ runtime + 导出工具 **source-of-truth**;pin v0.8.0;只带**可上游**的薄 patch | "Jetson backend" |
| **BOTTOM 后端** | `jetson-voice-engine` | 我们自己的、**不可上游**的 Jetson 功能/overlay/worker 编排 + 量化 recipes driver | (后端 overlay) |
| **BOTTOM 后端** | `rkvoice-stream` / `rkvoice-engine` | RK NPU 后端(RKLLM runtime) | "RK backend" |

> RPi(sherpa)无独立仓 —— 在 voxedge `backends/sherpa/` 内,无 C++ 后端,不占仓位。

**fork vs jetson-voice-engine 职责切分 —— 三类变更模型(消除"fp8 wiring 双写"歧义):**

关键原则:**任何 C++/CUDA runtime 改动都只在 fork 落地(fork 是 runtime 的唯一 source-of-truth)**;jve 不独立手写 runtime 代码,只把 fork 的 delta 机械地 re-apply 成 overlay patch。据此把变更分三类:

| 类 | 定义 | source-of-truth | 上游性 | 例 |
|---|---|---|---|---|
| **C1 上游 bug 修复** | 修上游 runtime 的 bug(C++/CUDA) | **fork**(打 `upstreamable` 标) | 推上游,合并即归零 | `N≤0` prefill guard |
| **C2 本地 runtime 扩展** | 我们加的 runtime 能力(C++/CUDA),暂不/不可上游 | **fork**(维护型 patch) | 不归零,长期维护 | fp8-embed wiring、cuBLAS-free tiled GEMM、M=1 GEMV |
| **C3 overlay/编排/recipes** | 非 runtime:构建胶水、worker spawn 配置、Python 导出 driver、profile | **jetson-voice-engine** | n/a | int4 导出 driver、overlay patch 生成、deploy compose |

→ **fp8-embed wiring 归为 C2 类 = 只在 fork 维护**;jve 侧的"overlay patch"是从 fork **自动生成**的(不是第二份手写源),因此不存在双写。worker **二进制**从 fork 源码编译,jve 只负责打包,**绝不**保留第二份 runtime 源。

**变更映射(现状 → 最终,治"同一 feature 编码两遍"):**
1. `jetson-voice-engine` 现 pin v0.7.1,扛 8 个 patch + fp8 product patch —— 这些在 fork v0.8.0 已原生 → **重 pin 到 v0.8.0,删已原生的冗余 patch**(workstream C2)。
2. worker 源码现散 3 处(fork `examples/omni`、jve `native/edgellm_voice_worker`、seeed `deploy/asr-worker-v080`)→ **收敛到 fork 一棵树**,另两处变 build-input。
3. int4/fp8 导出 driver 现孤立在 fork `wip/native-int4-talker` → 迁到 **`jetson-voice-engine/recipes/`**(class C3 recipes,**不放 fork**),pin 一个 fork commit;它们只调 fork 的 export API(`_export_talker` 已原生 detect W4A16),不改 runtime。fork 只留 native int4 kernels/plugin + export API。(对应 C6;统一了早先"推到 fork"的措辞。)
4. 终端用户消费的不是任何源码仓,而是 **HF 预编译 int4 引擎**(已传 3 bundle):`docker run` + 按 config 自动拉。
5. `voxedge-engine`(已自废,2026-06-01 并入 jve)、`qwen3-edgellm-jetson`(旧名)→ 留 README 指针归档(workstream G)。

---

## C. v0.8.0 CONSOLIDATION WORKSTREAM (ordered)

**Goal:** every production-path component on TensorRT-Edge-LLM v0.8.0. Dispatch as 2–3 sub-workstreams; C1 (config/registry) is independent of C2/C3 (repo/build).

| # | Step | Repo / path | Dependencies | Risk | Effort |
|---|---|---|---|---|---|
| **C0** | **✅ DONE — patch ownership manifest 已产出 → `c0-patch-ownership-manifest.md`(20 项全分类 + 双写风险 + 待拍板项)。关键发现:(1) jve 当前 pin v0.7.1 非 v0.8.0,整合本质=re-pin+删冗余封装(非照搬 8 patch);(2) int4 native kernels/plugin 上游 v0.8.0 已原生,我们只需 export drivers(C3);(3) **fork 里根本没有 ASR worker 源** → stripLangTag 修复(现仅在 seeed/jve 副本)必须先落回 fork 建唯一源;(4) **N≤0 guard 尚未落地**(分支里只有 empty-fullText LOG_ERROR)→ 是计划新写项非已存在 patch。生成 manifest(原始任务): 列出每个 patch:`patch id / owner repo(fork or jve)/ class(C1上游bug \| C2本地runtime扩展 \| C3 overlay-recipes)/ source-of-truth(源码在哪)vs packaging(overlay patch 在哪,自动生成)/ upstreamability / removal condition / validation gate / affected engines`。**关键:对每个 runtime 改动显式区分"源"(fork)与"封装"(jve 自动生成的 overlay patch),执行体只改"源",绝不手改"封装"。** 必须逐项拍死归属的两可项(codex 终审标出):`N≤0 prefill guard`(防御性 runtime,两可于 C1/C2)、`fp8-embed wiring`(C2,源在 fork,jve patch 自动生成)、`worker 源 vs 封装`(源=fork、打包=jve)、`CuTe-DSL→cuBLAS tiled GEMM fallback`(sm_87 平台兼容,两可于 C1/C2)、`ASR stripLangTag 英文首词修复`(C++ worker,此前未纳入任何 class)、`int4 native kernels/plugin(fork) vs export drivers(jve recipes)`(跨 C2/C3,F8 同依赖两处)。 | jetson-voice-engine `DIVERGENCE.md` + fork branch notes | none | **MED(防双写,关键前置)** | 0.5–1d |
| C1 | **Wire the 3 int4 HF bundles into the leaf registry — opt-in only, DO NOT flip defaults.** Update `configs/leaves/qwen3-tts-base.yaml`/`qwen3-asr-nx.yaml` to ADD int4 leaf variants (int4 talker+CP+fp8 embed / int4 decoder) pointing `artifacts.repo`/`files` at the HF int4 bundles; add a `qwen3-tts-customvoice` int4 leaf marked **experimental**. Create explicit **opt-in profiles** (`jetson-edgellm-v080-int4-*`). **`models.yaml default_precision[jetson]` 保持 fp16 不变**——默认翻转是单独一步(见 C1b),挂在 F-gate 全绿之后。 | seeed-local-voice `configs/leaves/*`, `configs/profiles/*` | none (config-only) | LOW | 0.5–1d |
| **C1b** | **Flip default precision to int4 — gated, NOT in C1.** 只有当对应 F-gate 全绿才翻 `default_precision[jetson]`:ASR int4 → 需 **F2b**(过 production worker + N=2,F2a 已绿);TTS base int4+fp8 → 需 **F5**(int4+fp8 N=2 burst/MD5);**CustomVoice int4(一等公民,用户定)→ 需 F4(streaming worker+EOS+全链 perf)+ F5-CV(CV N=2),全绿后可进默认**(license 跟随 Qwen 官方,不设额外门槛);在此之前 experimental opt-in。 | seeed-local-voice `configs/leaves/models.yaml` | C1, F2b/F4/F5/F5-CV | **MED(改默认=改生产路径)** | 0.5d |
| C2 | **Re-pin the Jetson overlay to v0.8.0.** Change `jetson-voice-engine/engine-overlay/UPSTREAM_PIN` from `364769036…`(v0.7.1) to the NVIDIA `release/0.8.0` base commit; regenerate `patches/` from the fork `port/qwen3-tts-base-v080`; update `addon/` with the v0.8.0 streaming worker + plugins; refresh `manifests/` + `DIVERGENCE.md`. Bump the submodule pointer in seeed-local-voice. | jetson-voice-engine (canonical); seeed-local-voice submodule bump | C-fork-branch-consolidation | **HIGH** (build reproducibility; shared plugin blast radius — base TTS + ASR int4 + CustomVoice share the plugin) | 3–5d |
| C3 | **Land the 2 pending runtime fixes — both ON THE FORK (per C0 manifest).** (a) **N≤0 prefill guard** (defensive — fixes silent garbled output on tokenizer mismatch) = **class C1 上游bug**. ⚠️ **C0 finding: NOT yet in any fork branch** — only an uncommitted local edit on orin-nx (`.prefillguard.bak`); the branches have just an `empty fullText` LOG_ERROR (`qwen3OmniTTSRuntime.cpp:1655`). **Action: write/recover it and COMMIT to fork** as the single source, tag upstreamable. (b) **fp8 text_embedding wiring** (`mTextEmbeddingScales` at 5 call sites, commit 873ca22) = **class C2 本地runtime扩展** → already on fork port branch, **stays in fork**. jve's overlay patch for it is **auto-generated from the fork**, NOT a separate hand-written copy (no double-write). (c) **Establish the ASR worker source in the fork (C0 finding: fork has NO asr_worker).** Land `qwen3_asr_worker.cpp` **with the stripLangTag fix** (currently only in seeed `deploy/asr-worker-v080` md5 `13b34dc` + a stale jve copy `7885f2e`) into fork `examples/omni/` as the single source = **class C1**; seeed/jve copies become generated packaging. | fork TensorRT-Edge-LLM (source); jve/seeed copies generated | C0, C2 | MED | 1–2d |
| C4 | **Harvest edgellm-v080-migration workers into the overlay.** Move `qwen3_tts_worker.cpp` (CLI-shim b4ffaa41), `qwen3_asr_worker.cpp` (8cf0b8df), the N=2 patch, and the turnkey compose into jetson-voice-engine addon/patches + manifests. Validate reconstructed worker md5 matches the recorded values. | jetson-voice-engine ← edgellm-v080-migration | C2 | MED | 1–2d |
| C5 | **Reconcile divergent overlay branches.** Memory cites deployed v0.8.0 runtime on `edgellm-v080 @7b6ac36` + fp8-embed only on the fork port branch (873ca22) + a separate `~/project/edgellm-v080` build tree. Pick ONE: the jetson-voice-engine overlay reconstructed from the fork. Verify the reconstructed engine set matches the HF `highperf-v080/` bundle. | jetson-voice-engine, fork | C2,C3,C4 | MED | 1d |
| C6 | **Relocate the int4/fp8 export drivers to `jetson-voice-engine/recipes/` (class C3 recipes, NOT the fork).** ✅ **Recovery DONE 2026-06-21** — driver branches are SAFE on suharvest fork: `wip/native-int4-talker` (ff2318e, TTS talker+CP `quantize_talker_stage1.py`/`stage2_export.py`/`quantize_cp_stage1.py`/`cp_stage2_export.py`) was already pushed; `wip/asr-int4-decoder` (c80bcc0, ASR int4) **now pushed**. Uncommitted working files (`quantize_talker_stage1_bigcalib.py` + `cv-int4-derisk/*` fp16/CV scripts+logs) **pulled to Mac** `~/project-backups/20260621-133543/wsl2-recovered/`. These thin drivers CALL the fork export API (`tensorrt_edgellm/scripts/`, `_export_talker` W4A16 auto-detect), do NOT touch C++/CUDA runtime → recipes layer. **Remaining (consolidation, not data-loss):** land them in `jetson-voice-engine/recipes/` with a pinned fork commit + recipes README; keep fork minimal (native int4 kernels/plugin + export API only). Reproducibility validated by F8. | jetson-voice-engine `recipes/` (drivers) + fork (pinned, unchanged) | none | LOW (recovery done; relocation remains) | 0.5d |
| C7 | **Rebake the v0.8.0 production-path image (overlay model).** Per memory, the composition image is an OVERLAY on a prefix base (`composition-e2e-v080w4/w5`): COPY v0.8.0 workers (TTS b4ffaa41 + ASR 8cf0b8df) + plugin (7d3fabe) + updated leaves. Bake with the int4 engine set. `deploy/jetson-workers` MUST stay at production-original (it is a shared build input — leaving v0.8.0 workers there mis-bakes flat-prod images). | seeed-local-voice deploy + jetson-voice-engine | C1–C5 | **HIGH** (image bake; shared build input) | 1–2d |

**Ordering:** C6 + C1 can start immediately in parallel. C2→C3→C4→C5 is the repo/build chain. C7 last. Do NOT deploy C7 to seeed-orin-nx — bake + test on orin-nx/orin-nano.

**Known traps (from memory — do not re-derive):**
- v0.7.0 `tensorrt-edge-llm/build/llm_inference` SEGFAULTS on v0.8.0 engines; use the v0.8.0 binary (`~/project/edgellm-v080-build/build/...`).
- Shared `speech-models` volume: a v0.8.0 Qwen3-ASR engine and the prod v0.7.0 ASR worker CANNOT co-exist (version-mismatch refusal). Clean e2e MUST use an isolated volume / host dir (`~/comp-e2e-models`).
- ASR audio encoder must be the **minchunk1** build (opt-profile min-dim 1) or batch=1 prewarm fails.
- CuTe-DSL GEMM is NOT viable on sm_87 (`invalid device function`); build `ENABLE_CUTE_DSL=OFF`, use the cuBLAS-free tiled FP16 GEMM. `-rdc` incremental device-link is broken → clean `rm -rf build` after any `.cu` change.

---

## D. ONE-CLICK DEPLOY + AUTO MODEL-PULL

**What exists today:**
- `deploy/install.sh` — auto-detects target (jetson/rk3576/rk3588/rpi via `/etc/nv_tegra_release`, `/dev/rknpu`, device-tree), selects the compose file, `--pull --verify`.
- `deploy/docker-compose.yml` (+ `.rk.yml`, `.rpi.yml`, `.radxa.yml`) — `OVS_PROFILE=… docker compose up -d`; models auto-download on first start into a `speech-models` volume.
- `server/core/model_downloader.py` — CDN tarball pull (matcha/paraformer/kokoro) + `QWEN3_ARTIFACT_MANIFEST` → HF pull for Qwen3 engines.
- Leaf composition: `server/core/leaf_composition.py` + `composition_boot.py` → a `composition` profile selects leaves, and the **leaf-union drives the HF pull** per config (Slice 1–3 DONE, commits 00e4d6e/c70a67b/dee93e4).

**Target UX (genuinely one-click):**
```
# Pick a profile, everything else is automatic:
OVS_PROFILE=jetson-edgellm-v080-int4 deploy/install.sh --pull --verify
#   → install.sh detects Jetson → selects compose → pulls image
#   → container boot: composition_boot resolves the profile's leaves
#   → leaf-union → model_downloader auto-pulls EXACTLY the int4 bundles for that config from HF
#   → engine_resolver no-op (leaf env sets paths) → prewarm → /readyz green
```

| # | Step | Repo / path | Dep | Risk | Effort |
|---|---|---|---|---|---|
| D1 | **Add v0.8.0 int4 composition profiles** — one per device/use-case: `jetson-edgellm-v080-int4-nx` (ASR int4 + TTS base int4), `…-customvoice`, `…-nano`. Use the `composition:{device,asr,tts}` form (cf. `jetson-qwen3-composition-nx.json`) so leaf-union auto-pull engages. | seeed-local-voice `configs/profiles/` | C1 | LOW | 0.5d |
| D2 | **Register the int4 HF bundles as pullable leaves.** Ensure `model_downloader._ensure_qwen3_artifacts_via_hf` receives the leaf-union `required_files` for the int4 bundles (the non-empty-override path, model_downloader.py ~:333/350). Confirm the fat-script path doesn't shadow the override (known boundary). | seeed-local-voice `server/core/model_downloader.py` + leaves | C1,D1 | MED (the fat-script override boundary) | 1d |
| D3 | **Profile-picker UX.** Add a `--profile` flag / interactive menu to `install.sh` that lists the v0.8.0 profiles with one-line descriptions + device fit (RAM ceiling from leaf `peak_unified_mb` + device `base_reservation_mb`), so the user picks by use-case not by memorizing profile names. Validate the pick against device RAM via `validate_composition` before `up`. | seeed-local-voice `deploy/install.sh` | D1 | LOW | 1d |
| D4 | **First-boot auto-pull verification + artifact-source fallback.** Add to `verify.sh`: assert the leaf-union pulled the expected int4 files (md5/size), engines loaded, /readyz green, TTS smoke + TTS→ASR roundtrip. **Define artifact source priority: HF → mirror → local cache**, and on each pull **print the actual source + checksum**; if mirror hasn't synced a new bundle (orin-nx can't reach hf.co directly), fall back to local cache and **fail loudly with the missing md5** rather than silently booting a wrong/old engine. | seeed-local-voice `deploy/verify.sh` + `model_downloader.py` | D1,D2 | MED (mirror lag) | 0.5d |
| D5 | **Compose env hygiene.** The compose still references `QWEN3_EDGELLM_JETSON_ROOT=/opt/qwen3-edgellm-jetson` (old submodule name). Update to the renamed `jetson-voice-engine` path + the new int4 env keys; keep backward-compat defaults. | seeed-local-voice `deploy/docker-compose*.yml` | C7 | LOW | 0.5d |

**UX principle:** the user picks ONE thing (a profile name describing a use-case, e.g. "Orin Nano — fast multilingual dialogue, int4"), and `install.sh` + composition_boot + leaf-union + model_downloader do detection → image pull → per-config model pull → prewarm → health, with a single `--verify` proving it.

---

## E. README / DOCS OVERHAUL + PERF METRICS

| # | Step | Path | Risk | Effort |
|---|---|---|---|---|
| E1 | **Top-level README**: update the model badges/tables to include the 3 new int4 v0.8.0 models; add a "v0.8.0" section; fix the registry-namespace note; update Quick-Start to the new int4 profiles. | seeed-local-voice `README.md` | LOW | 0.5d |
| E2 | **ARCHITECTURE.md is STALE** — it still describes `voxedge-engine` as the engine repo and the submodule as `qwen3-edgellm-jetson`. Rewrite the "three repositories" section: voxedge-engine deprecated → `jetson-voice-engine` (renamed) is the overlay submodule; fork = TTS worker source-of-truth. | seeed-local-voice `ARCHITECTURE.md` | LOW | 0.5d |
| E3 | **Add perf/VRAM/quality tables** (numbers below) to README + a `docs/perf/` page + the per-model leaf header comments + (optionally) sync the HF READMEs (the ASR + CustomVoice HF cards lack RTF/TTFA — add from memory). | seeed-local-voice README + docs/perf + HF cards | LOW | 0.5d |
| E4 | **Per-project READMEs**: jetson-voice-engine README still says UPSTREAM_PIN v0.7.1 — update post-C2; voxedge README; deprecate voxedge-engine / qwen3-edgellm-jetson READMEs with a pointer to canonical. | jetson-voice-engine, voxedge, voxedge-engine | LOW | 0.5d |
| E5 | **`BENCHMARKS.md`(全量矩阵)** — 渲染表 A(设备×后端×模型×精度→RTF/TTFA/VRAM/CER)+ 方法论口径。数据**唯一来源** = `benchmarks-dataset.md`(已汇总,带 file 出处)。每个数字保留出处脚注,`—` 单元格列进「待测」。 | seeed-local-voice `BENCHMARKS.md` | LOW | 0.5d |
| E6 | **Recipes 选择器页**(用例×设备→推荐栈+能力)— 渲染表 B,每行末尾给「一键部署 profile」链接,闭环到 workstream D 的 `--profile`。这是把"看 benchmark 的人"转成"用户"的转化点。 | seeed-local-voice `docs/recipes.md`(或 landing) | LOW | 0.5d |

### E 节·benchmark/recipes 呈现策略(single-source → 按受众多视图)

**一个数据源 → 四个视图,绝不各处手抄数字**:

```
benchmarks-dataset.md   ← 唯一真相(表A性能 + 表B最佳实践 + 方法论 + GAPS)
        │
        ├─▶ Landing README "hero 数字"  → 给所有人:只挑**已验证+有出处**的(Orin Nano Base int4+fp8 RTF 0.44 / TTFA 0.21s【N=1】/ int4 省 ~1GB/实例;MOSS-TTS-Nano N=2 byte-identical【已验】)。⚠️ Base int4+fp8 的 N=2 待 F5,**不得**作为 hero;CustomVoice int4 experimental 不进 hero。
        ├─▶ BENCHMARKS.md (E5)          → 给系统开发者:全量表A + 口径,数字即护城河
        ├─▶ Recipes 选择器 (E6)         → 给应用开发者:表B,"我要做X用什么"→ 直给 profile + 一键命令
        └─▶ HF model cards / leaf 头注释 → 给消费引擎的人:单模型那一行 perf(E3 已覆盖,补 RTF/TTFA)
```

**三条规则**:
1. **数字只在数据集文件改一次**,四个视图都从它生成/引用 —— 杜绝 README 和 HF card 数字打架。
2. **每个数字带出处**(memory/spec/commit),对外文档保留方法论口径段(贪婪解码、finalize 口径、N=verified 定义),否则"性能数字"不可信。
3. **Recipes → 一键部署闭环**:表 B 每行的"推荐栈"必须对应一个真实存在的 leaf/profile 组合;选择器输出 = `docker run ... --profile <X>`,直接落到 workstream D。没有这个闭环,漏斗会漏。

**受众-视图-卖点对应**(呼应定位文档第四节双受众漏斗):
| 视图 | 受众 | 卖点 | 落点 |
|---|---|---|---|
| Landing hero | 所有人 | "哇,端侧能这么快" | README 顶部 + demo GIF |
| BENCHMARKS.md | 系统/算法开发者 | 全量性能 + 量化深度 + 多后端覆盖 | repo 根 |
| Recipes 选择器 | 应用开发者 | "我的用例→直接能跑的栈" | docs/landing + 闭环 profile |
| HF card / leaf 注释 | 拉引擎的人 | 这个引擎多大多快多准 | HF + configs/leaves |


**Perf/quality/VRAM tables to add** (sources: memory files + HF cards; ✱ = from memory, HF card omits):

*Qwen3-ASR-0.6B int4-AWQ (v0.8.0, Jetson)* — `harvestsu/qwen3-asr-0.6b-int4-v080`
| Metric | Value |
|---|---|
| ZH CER | 0.00% (6/6 exact, opencc-normalized) |
| EN WER | ~11% (single substitution on a 3-word clip) |
| LLM engine | ~525 MB (int4-AWQ / W4A16) |
| Audio encoder | ~364 MB (fp16) |
| Embedding params | ~297 MB |
| Plugin | ~43 MB |
| Decode contract | greedy (temp 0.0, top_k 1, top_p 1.0) + force_language prime |

*Qwen3-TTS-0.6B BASE int4+fp8 (v0.8.0, Orin Nano)* — `harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8`
| Metric | Value |
|---|---|
| RTF | 0.44 |
| TTFA (warm) | 0.21 s |
| Talker | 907 MB → 245 MB (int4-AWQ) |
| CodePredictor | 191 MB → 98 MB (int4-AWQ) |
| text_embedding | 622 MB → 320 MB (fp8 e4m3) |
| Code2Wav | fp16 (INT8 not viable: TRT10.3 Conv1D QDQ) |
| VRAM | ≈ −1.06 GB / instance |
| EN WER / ZH CER | 0% / 1.7% (== fp16 baseline) |
| Concurrency | N=2 **verified at fp16** (KV cap 1536, 5.6GB free); int4+fp8 N=2 **pending F5 gate** — do NOT cite as a verified hero number |

*Qwen3-TTS-0.6B CustomVoice int4+fp8 (v0.8.0, Orin NX)* — `harvestsu/qwen3-tts-0.6b-customvoice-jetson-trtllm-int4fp8`
| Metric | Value |
|---|---|
| Talker | 246 MB (int4-AWQ) |
| text_embedding | 321 MB (fp8 e4m3, per-group-128) |
| Code predictor | ~182 MB (fp16) |
| Code2wav | ~224 MB (fp16) |
| Sidecars | 6.3 MB embedding + 12.6 MB text_projection |
| VRAM | ≈ −960 MB / instance vs all-fp16 |
| Quality | A/B vs fp16, ASR-verified intelligible ZH+EN |
| Safety | hard N≤0 prefill guard (fails loud on tokenizer mismatch) |
| ✱RTF/TTFA | not yet measured through streaming worker — add after F |

---

## F. TEST / COMPILE VERIFICATION (test env only — NOT seeed-orin-nx)

> **F 是各 workstream 的 exit-criteria,不是收尾阶段。** C1b(翻默认)、C2(re-pin)、C7(bake)在标 DONE 前必须先过对应 F-gate:C1b→F2b/F4/F5、C2→F1+F8、C7→F6。**不允许 C/D 标完成后再补跑 F。**

**Devices:** build host = **orin-nano** (`~/project/edgellm-v080-build`, ENABLE_CUTE_DSL=OFF) for Jetson; **wsl2-local** (RTX, has modelopt 0.39) for int4/fp8 export; **orin-nx** for NX-class memory/perf (free it of co-resident services first — it is a profiling device, the arm stack is on seeed-orin-nx). Use `fleet` (full path in subshells).

| # | Component | Where | Command (sketch) | Pass gate |
|---|---|---|---|---|
| F1 | v0.8.0 engine + worker build | orin-nano | `cd ~/project/edgellm-v080-build && rm -rf build && cmake -DENABLE_CUTE_DSL=OFF … && make qwen3_tts_streaming_worker qwen3_asr_worker llm_inference NvInfer_edgellm_plugin` | binaries link; plugin .so produced; record md5 |
| F2 | Qwen3-ASR int4 validation — **two sub-gates** | orin-nano / orin-nx | **F2a (decode contract, DONE):** `llm_inference` w/ `temp=0,top_k=1,top_p=1`, `apply_chat_template`, assistant primed `"language <Lang><asr_text>"`, `--engineDir=<llm> --multimodalEngineDir=<audio_PARENT>`, audio min-dim minTimeSteps=100. **F2b (PENDING, required before C1b):** run int4 through the **production `qwen3_asr_worker` streaming path** (not just `llm_inference`) + **N=2 slot-pool**. | F2a: ZH CER 0% (6-clip, opencc-norm), EN ≤1 acoustic word. F2b: same accuracy through real worker + N=2 both slots rc=0, no OOM |
| F3 | Qwen3-TTS base int4+fp8 synth | orin-nano | worker + `EDGELLM_PLUGIN_PATH` + talker/cp/code2wav int4 engines + `speaker_embedding_b64` (precomputed) → wav; faster-whisper round-trip | clean EOS; intelligible (faster-whisper, NOT Groq — key expired); RTF≈0.44/TTFA≈0.21 |
| F4 | Qwen3-TTS CustomVoice int4 synth | orin-nx | worker + **CORRECT CV tokenizerDir** (the wrong-tokenizer trap — assistant role must be non-empty) + N≤0 guard build | "你好"/"今天天气怎么样" intelligible (faster-whisper); EOS clean (NOT maxFrames); prefill ≥12 rows |
| F5 | N=2 parallel TTS (base int4+fp8) | orin-nano | 2 slots, `--max_slots 2`, KV cap 1536 | both slots rc=0, no OOM, ≥5GB free (base on 8GB) |
| F5-CV | N=2 parallel TTS (**CustomVoice int4**, 一等公民) | orin-nx | 2 slots `--max_slots 2`, CV checkpoint + P5 9-row conditioning ported to v0.8.0 | both slots rc=0, no OOM, EOS clean, intelligible (faster-whisper) |
| F6 | Composition e2e (config → leaf-union pull → engines → /asr+/tts) | orin-nx (isolated volume `~/comp-e2e-models`, NOT the prod speech-models volume) | overlay image `composition-e2e-v080w*` + `OVS_PROFILE=<int4 profile>`, no config overrides | /asr accurate transcript; /tts non-silent wav; prewarm batch=1 OK |
| F7 | voxedge unit + engine-inprocess (no GPU) | mac / any | `cd ../voxedge && pytest voxedge/tests/test_engine_inprocess.py -q`; `pytest server/tests/test_leaf_composition.py -q` | green (byte-equivalent contract) |
| F8 | int4 export reproducibility | wsl2-local | run `jetson-voice-engine/recipes/quantize_talker_stage1.py` + `stage2_export.py` (recipes calling the **pinned fork** export API, after C6 relocates them) → ONNX → engines match HF md5 | export reproduces shipped int4 engines byte-for-byte (engine md5 == HF bundle) |

**Gates carried from memory:** bytes≠speech (always ASR-verify audio); md5 byte-gate is invalid for sampling paths (use greedy or ASR check); opencc t2s-normalize before CER; faster-whisper not Groq.

**F 节盲区(codex 终审标出,F1–F8 未覆盖,补进 H 节回归策略):** ① license/model-card 分发风险 gate ② artifact mirror-lag/source-fallback 自动化 ③ README/HF/leaf benchmark 数字一致性 CI ④ worker sampling 参数生效验证(CV 曾被 harness 采样误导)⑤ ASR `stripLangTag` 英文首词回归(生产级 bug)⑥ clean-device 一键 install 全链路 ⑦ 逐组合真机 e2e(35 leaf × 设备)。(⑧ stateful×N=2 已实机 closed:flag 在 slot-pool 是 no-op,无收益无风险,见数据集 GAPS#10 + manifest W2 — 无需 gate。)

---

## H. MIGRATION REGRESSION STRATEGY (防衰退 — 迁移前置)

> 迁移面:生产路径 v0.7.1 overlay→v0.8.0、引擎 fp16→int4/fp8、worker 源 3处→1处、export driver→recipes。本节是"不衰退"的硬约束,与 F 节(单点验证)互补:F 验"新东西能跑",H 验"老行为没退"。

### H1. Golden 基线(迁移前必须先冻结)
迁移**任何**生产路径前,先抓基线包存 `golden/v080-migration/<date>/<profile>/`:
- **音频**:输入 wav + 输出 wav + md5 + 采样参数(N=x verified 的 MD5 音频门定义见数据集方法论)。
- **CER/WER**:ASR 用 greedy + force_language scaffold + opencc t2s;当前基线 ZH CER 0% / EN WER 11.11%。
- **TTFA/RTF**:warm TTFA、RTF=wall/audio;Base int4+fp8 基线 RTF 0.44 / TTFA 0.21s(Orin Nano)。
- **VRAM**:engine size + 运行峰值;Base −1.06GB/实例、CV −960MB/实例。
- **md5 钉死**:engine md5 + **worker md5** + plugin md5(教训:ASR worker 8cf0b8df 靠 md5 钉死才发现镜像漂移)。

### H2. 等价性判据(分两类,别用错 gate)
| 场景 | gate | 阈值 |
|---|---|---|
| greedy TTS / N=2 slot 输出 / worker rebuild / export 复现 | **byte-identical** | md5 完全一致(MOSS 30/30、F8 engine md5==HF) |
| sampling TTS | **质量等价**(md5 无效) | faster-whisper round-trip 可懂 + EOS clean |
| ASR int4 vs fp16 | **质量等价** | ZH CER ≤ 0.5% / EN WER ≤ 15% / 无 language leak/loop |
| TTS Base int4+fp8 vs fp16 | **质量等价** | RTF ≤ 0.50 / TTFA ≤ 0.25s / CER·WER 不劣于 fp16 |
| CustomVoice int4(F4 前 experimental) | **质量等价** | faster-whisper 可懂 + EOS clean + prefill ≥ 12 rows |

### H3. 分层测试矩阵(各层迁移各跑什么)
| 层 | 内容 | gate | GPU |
|---|---|---|---|
| unit | leaf composition、engine-inprocess 契约 | F7 全绿、byte-equivalent | 无(进 CI) |
| worker 协议 | ASR/TTS worker CLI、force_language scaffold、**stripLangTag**、N=2 streaming | 准确率同 F2a;EOS clean;**英文首词不丢** | Jetson |
| engine 数值 | llm_inference/worker 直测、engine/plugin md5、export 复现 | F1 md5 + F8 engine md5==HF | Jetson + wsl2 |
| e2e composition | config→leaf-union pull→engines→/asr+/tts | F6(隔离卷) | Orin NX |
| N=2 | TTS slot pool、ASR streaming slot-pool、VRAM/竞态 | F5 + F2b 两 slot rc=0 无 OOM | Orin Nano/NX |

### H4. 回归触发点(改 X → 必跑 Y)
- **换 plugin** → F1 + F3/F4 + F6(C2 blast radius:base TTS + ASR int4 + CV 共享 plugin)。
- **re-pin overlay** → F1 + F8 + F6 + **全 worker md5 回归**。
- **换引擎精度** → ASR:F2b;TTS base:F3+F5;CV:F4+F5;默认翻转挂 C1b gate。
- **换 worker 源** → worker 协议层 + F6 + md5 钉死(ASR firstword 事故教训)。
- **换 artifact pull 源/mirror** → D4 checksum + F6 first-boot,missing md5 必须 fail loudly。
- **改 leaf/profile** → F7 + F6。
- **改 tokenizer/chat-template** → F4/CV smoke(wrong tokenizer→N≤0→垃圾音频)。

### H5. CI 化(防"忘了跑")
- **进 CI(无 GPU)**:F7(`pytest voxedge/tests/test_engine_inprocess.py` + `server/tests/test_leaf_composition.py`)+ benchmark id/repro 元数据存在性校验。
- **真机手动**:F1–F6/F8(全需 Jetson/wsl2 GPU)。
- **防忘**:C1b/C2/C7 的 Done 条件写成 **gate-blocking**;PR 模板强制粘贴 F-gate 编号 + 设备 + engine md5 + worker md5 + artifact checksum(否则视为未验证)。

### H6. 复用现有资产(别新造)
| 资产 | 路径 | 复用为 |
|---|---|---|
| `smoke_tts_multiturn` | `bench/perf/smoke_tts_multiturn.py` | F3/F4 TTS 多轮 smoke(已含 faster-whisper round-trip) |
| `perf gate.py` | `bench/perf/gate.py` | TTFA/RTF/VRAM gate 收敛入口(口径见数据集) |
| `test_engine_inprocess` | `voxedge/tests/` | 直接进 CI(F7) |
| `test_leaf_composition` | `server/tests/` | leaf 契约,进 CI |
| `toolcall_stability_bench` | (repo) | server-loop 稳定性回归(机械臂用例) |
| `grasp_cycle_check` | (voice_rebot_arm) | 机械臂 demo/toolcall 回归入口(也是传播 hook) |
| `sim_pump` | `agent/tests/sim_pump.py` | agent 侧音频/事件泵确定性模拟 |
| faster-whisper round-trip | F3/F4 | **唯一** TTS 质量 gate(Groq key 已过期,禁用) |

---

## G. DEPRECATION / CLEANUP LIST (flag for user — DO NOT DELETE)

| Item | Type | Why redundant | Confirm-before-delete |
|---|---|---|---|
| `~/project/qwen3-edgellm-jetson` | dup checkout | Same HEAD + remote as `jetson-voice-engine` (the pre-rename name). ⚠️ AUDIT: 0 unpushed but **20 uncommitted changes**; and prod compose + `model_downloader.py` still use `QWEN3_EDGELLM_JETSON_ROOT=/opt/qwen3-edgellm-jetson` env path. | review/discard the 20 changes; update env var or keep a compat alias BEFORE rename; then archive |
| `~/project/tensorrt-edge-llm` | dup checkout | Lowercase clone of the canonical fork `TensorRT-Edge-LLM`. ✅ **RESOLVED 2026-06-21: the 4 commits (893ba2a SlotPool / 7b52dc6 ASR-assistant prefix / 07b500d N-instance slot-pool worker / 766530b shared-engine ctor) PUSHED to `suharvest/v071/customvoice-product` (fast-forward `1668470..893ba2a`, unpushed now 0).** 3 untracked (docs/DIVERGENCE.md, docs/IPC-CONTRACT.md, uv.lock) backed up. Still hardcoded by many jetson-voice-engine `scripts/*.sh` + docs as `~/project/tensorrt-edge-llm`. | **Remaining: rebase all `~/project/tensorrt-edge-llm` path refs → `TensorRT-Edge-LLM` (or env var), then archive.** (data-loss risk cleared) |
| `~/project/_qwen3tts-port` | **git worktree** | Worktree of fork @ wip/fp8-embedding (873ca22 = same as fork port branch); no direct refs. | low risk; remove via `git worktree remove` (NOT `rm -rf`) after confirming 873ca22 pushed |
| `~/project/seeed-local-voice-v080` | **git worktree** | Worktree on feat/edgellm-v080-migration, 0 unpushed; per trunk_consolidation feat==main; no refs. | low risk; `git worktree remove` |
| `~/project/edgellm-v080-migration` | scratch | v0.8.0 worker/compose scratch — HARVEST into overlay (C4) then archive. | after C4 lands its content in jetson-voice-engine |
| `~/project/voxedge-engine` | deprecated repo | Self-deprecated 2026-06-01 (merged into jetson-voice-engine). | archive (keep README pointer) |
| `~/project/jetson-qwen3-speech` | legacy | Pre-overlay Jetson Qwen3 speech; superseded. | confirm nothing imports it |
| `~/project/sensecraft_voice` | non-git scratch (284MB) | Referenced by seeed `docs/specs/ovs-punct-speaker-handoff.md` as PoC for punctuation/speaker capability. | archive (not delete) + repoint the doc; or keep |
| `~/project/jetson-voice` | non-git empty shell (12KB, only `.claude`+`.DS_Store`) | No refs, empty. | safe to remove |
| `~/project/jetson-qwen3-speech` | legacy git repo (f689f95, 2026-04) | 0 unpushed; 1 historical doc mention in jetson-voice-engine perf doc. | low risk; archive |
| `~/project/jetson-llm-benchmark` (detached HEAD) | scratch | Benchmark scratch, 11 uncommitted, no refs. | archive |
| `~/project/qwen3asr_rk` (external) | **external lib (HF qzxyz, 22GB)** | AUDIT: **NOT ours** — different maintainer, ZERO code overlap with rkvoice-engine (it's a runtime consumer, rkvoice-engine is a model producer); NO canonical code reference (only a hardcoded test path in rkvoice-stream). | **DECISION: out of our consolidation scope — KEEP as-is, do NOT fold into rkvoice-engine, no action.** Remove from this list. |
| Fork stale branches: `qwen3tts-issue-artifacts-20260516`, `official-*`, `pr-*` | branch cruft | Investigation/PR-prep branches. | user confirms which to prune |
| seeed-local-voice untracked dirs: `_tts_rate_demo/`, `agent/_grasp_frames*/`, `_ik_envelope_new.csv` | scratch artifacts | Demo/sim scratch, not tracked. | gitignore or remove |

**Untracked in-tree (decide tracked-vs-ignore):** `configs/profiles/jetson-edgellm-v080-moss.json`, `deploy/jetson/Dockerfile.edgellm-moss-nx`, `deploy/asr-worker-v080/` — these are the v0.8.0 deploy basis; **track them** (they belong in the C/D workstreams), don't delete.

---

## DISPATCH NOTES FOR EXECUTORS

- Each of C/D/E/F/G is independently dispatchable. Suggested parallelism: **C1+C6 (now)** ∥ **C2→C3→C4→C5 (chain)**; **D1–D5 after C1**; **E after C2**; **F throughout (gate each C/D step)**; **G last (needs user confirm)**.
- Every build/deploy dispatch MUST carry the EVIDENCE + no-destructive guardrails (md5s, raw verification output, before/after, `docker logs | grep -iE error|crash|fail`) and the Fleet rules block. NEVER `docker compose down` the prod project; use `up -d <service>` for image swaps. NEVER touch seeed-orin-nx.
- **Preflight 防误触(每个 deploy/test 任务模板首行强制执行):** 先 `hostname`/`fleet status` 打印目标 → **若命中 `seeed-orin-nx` 立即 STOP**;**拒绝**操作 prod compose project 名;deploy/test 必须用 isolated volume(`~/comp-e2e-models`),**拒绝**挂 prod `speech-models` 卷。preflight 不过不准往下走。
- ✅ **Data-loss risks CLEARED 2026-06-21:** (1) `tensorrt-edge-llm` 4 unpushed commits → pushed to `suharvest/v071/customvoice-product`; (2) int4 export drivers → `wip/native-int4-talker` + `wip/asr-int4-decoder` both on suharvest, uncommitted working files pulled to Mac backup. **No outstanding data-loss risk.** Remaining C6 work is pure consolidation (relocate drivers into `jetson-voice-engine/recipes/`), not loss-prevention.
- **Path-hardcode debt (audit-found):** many `jetson-voice-engine/scripts/*.sh` + docs hardcode `~/project/tensorrt-edge-llm` (lowercase); prod compose + `model_downloader.py` use `QWEN3_EDGELLM_JETSON_ROOT=/opt/qwen3-edgellm-jetson`. Rebase refs / add compat alias before any rename or delete. Worktrees (`_qwen3tts-port`, `seeed-local-voice-v080`) → `git worktree remove`, never `rm -rf`.
```