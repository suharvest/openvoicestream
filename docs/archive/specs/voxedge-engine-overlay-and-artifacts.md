# voxedge Engine-Overlay + Artifact 架构 spec

> 状态:DRAFT,待用户确认后分期实施。
> 产出:codex 只读分析四个 repo(seeed-local-voice / voxedge / TensorRT-Edge-LLM@v071/customvoice-product / rkvoice-stream)+ 主线程补充「分流策略」节(§2A)。
> 目标:把边缘语音栈理成四层 —— **轻库 voxedge + 重引擎 overlay + HF artifact + 产品服务** —— 干掉死副本、用 overlay 最小化对上游侵入、artifact 上 HF 用 manifest 引用。

---

## 1. 四层最终拓扑

```text
Layer 4 Product/Profile/Deploy
  /Users/harvest/project/seeed-local-voice
  FastAPI service, profiles, Docker images, device install policy
      depends on
Layer 3 voxedge Library Runtime
  /Users/harvest/project/voxedge/voxedge
  backend ABCs, orchestration moat, thin backend adapters, runtime artifact manifest
      depends on prebuilt artifacts / optional backend runtimes, NOT product app.*
Layer 2 Runtime Artifacts
  Hugging Face model repos + manifest references
  .engine/.so/.rknn/.rkllm/.onnx/data sidecars, SHA-pinned, downloaded/installed by profile
      produced by
Layer 1 Engine Build/Overlay Repos
  proposed: voxedge-engine overlay for TensorRT-Edge-LLM
  proposed: rkvoice-engine / rkvoice-stream overlay symmetry for RKNN/RKLLM conversion
  upstream sources are pinned inputs, NOT vendored source copies
```

| Layer | Physical location | Verified facts |
|---|---|---|
| Product | `seeed-local-voice` | FastAPI service `server/main.py:10` (import) / `:44` (app); `fastapi>=0.115.0` at `app/requirements.txt:1`; structure at `README.md:510-515`. |
| Library | `voxedge/voxedge` | Package `voxedge` `pyproject.toml:18-21`; deps pure Python/numpy-only `pyproject.toml:27-33`; open-core extraction `README.md:5-8`. |
| Library moat | `voxedge/voxedge/engine` | 2064 行:conversation 1016 / asr_session_manager 430 / capability_resolver 238 / tts_buffer 167 / coordinator 138 / concurrency_capability 62。 |
| Product→library dep | `seeed-local-voice/server/core` | ASR registry → `voxedge.backends.*` `server/core/asr_backend.py:150-154`;TTS registry `server/core/tts_backend.py:145-153`;factories build voxedge config `asr_backend.py:176-198` / `tts_backend.py:174-201`。 |
| 无 library→product 依赖 | `voxedge/voxedge` | 无活的 `from app`/`import app`,残留仅注释(`voxedge/backends/rk/artifacts.py:5-10`、`voxedge/backends/jetson/trt_edge_llm_ipc.py:6-18`、`voxedge/backends/rk/tts.py:11-20`)。 |
| Engine fork | `TensorRT-Edge-LLM` | branch `v071/customvoice-product`;remote fetch=NVIDIA upstream / push=suharvest;新增 `cpp/workers/moss_tts_nano_worker.cpp`、`cpp/workers/build_moss_worker.sh`、`cpp/runtime/mossTtsNanoRuntime.cpp`、`examples/omni/qwen3_tts_streaming_worker.cpp`。 |
| RK runtime | `rkvoice-stream` | `rkvoice_stream/{backends,engine,runtime}` + `models/` 转换脚本(`models/tts/kokoro/export_kokoro_rknn.py:1-6`、`models/asr/paraformer/convert_paraformer_rknn.py:1-7`)。 |

**依赖规则**:`Product → voxedge → optional runtime/artifacts`;`voxedge` 不得 import `app.*`;engine overlay 产 artifact 经 manifest 消费,不经 Python import。**确认无环。**

---

## 2. Overlay 仓设计:`voxedge-engine`

**建议**:新建薄 overlay 仓 `voxedge-engine`;**不 vendor 整棵 NVIDIA 源码树**,只装上游 pin、patches、addon、build 包装、divergence 文档、构建复现 manifest。

```text
voxedge-engine/
  UPSTREAM_PIN          # NVIDIA TensorRT-Edge-LLM 精确 commit
  upstream.remote       # https://github.com/NVIDIA/TensorRT-Edge-LLM.git
  patches/  0001-*.patch
  addon/    cpp/{workers,runtime,kernels}/  examples/omni/  scripts/  tests/
  build.sh
  manifests/ qwen3-tts-highperf-sm87.toml  qwen3-asr-sm87.toml  customvoice-v071.toml
  DIVERGENCE.md
```

**Pin 机制**:`UPSTREAM_PIN` 存 NVIDIA commit(观测到的 merge-base = `364769036fc83351d9d0aac4cc064a8e56a83178`);`upstream.remote` 存 NVIDIA URL;build 时 clone/fetch 上游@pin 到 workdir → 拷 `addon/` → apply `patches/*.patch` → build。

**addon vs patch 纪律**:

| 改动类型 | 去向 | 规则 |
|---|---|---|
| 上游不存在的新文件 | `addon/<相对路径>` | checkout 后拷入。现有:`cpp/workers/moss_tts_nano_worker.cpp`、`cpp/workers/build_moss_worker.sh`、`cpp/runtime/mossTtsNanoRuntime.cpp`、`cpp/runtime/slotPool.h`、`examples/omni/qwen3_tts_streaming_worker.cpp`。 |
| 必须改上游已有文件 | `patches/NNNN-*.patch` | 最小、可审、可重放。现有改动:`CMakeLists.txt`、`cpp/CMakeLists.txt`、`examples/omni/CMakeLists.txt`、`cpp/runtime/qwen3OmniTTSRuntime.cpp`、`experimental/server/engine.py`。 |
| build/export/package 脚本 | `addon/scripts/` | 现有产品 third-party 脚本已类 overlay 行为(`third_party/qwen3-edgellm-jetson/scripts/export_qwen3_asr_onnx.sh:1-5`、`package_qwen3_artifacts.py:1-6`)。 |
| divergence 文档 | `DIVERGENCE.md` | fork 已有未跟踪 `docs/DIVERGENCE.md`,提升到 overlay 根。 |

**从现有 fork 抽取 addon/patches 的方法**:
1. base = `git merge-base HEAD origin/main` → `364769036...`。
2. 新文件:`git diff --name-status <base>...HEAD | awk '$1=="A"{print $2}'` → 拷进 `addon/<path>`。
3. 改动文件:`... | awk '$1=="M"{print $2}'` → 按主题生成 `git diff <base>...HEAD -- <file> > patches/NNNN-topic.patch`。
4. 验证可复现:clone 上游@PIN → 拷 addon → apply patches → `build.sh` → 比对产物文件名/checksum。
5. 此后禁止直接编辑上游 checkout;新特性起于 `addon/`,改上游需走 patch 审查。

**build.sh 契约**:

| 项 | 行为 |
|---|---|
| 输入 | `UPSTREAM_PIN`、上游 URL、`addon/`、`patches/`、所选 build manifest、target `sm`、CUDA/TRT 版本、模型源 ref。 |
| 输出 | worker 二进制(`qwen3_asr_worker`/`qwen3_tts_worker`/`moss_tts_nano_worker`)、plugin `.so`(`libNvInfer_edgellm_plugin.so`)、`.engine`、sidecar 校验。当前 Docker 从 `deploy/jetson-workers/` 拷到 `/opt/jv-workers`、`/opt/edgellm-bin`(`deploy/docker/Dockerfile.jetson:85-100`)。 |
| HF push | 按 canonical layout staging + 写 SHA 清单 + push HF(`package_qwen3_artifacts.py:1-6`;下载验证 `deploy_qwen3_artifacts.py:1-5`)。 |
| build manifest | 记上游 commit、patch 列表、addon digest、export/quant 参数、target device、产物名、HF repo/rev/SHA。 |

**RK 对称处理**:`rkvoice-stream` 当前 runtime + 大量 `models/` 转换脚本混在一起。建议拆:runtime Python 包仍 `rkvoice-stream` 可安装;转换/build overlay 持 `models/` 脚本 + RKNN/RKLLM build manifest + HF push manifest。产品 RK 下载器已假设 `.rknn/.rkllm/tokenizer/config/lexicon` 是 manifest 描述的外部 artifact(`server/core/rk_artifacts.py:1-6`)。

---

## 2A. 分流策略:upstream-PR vs 自留 patch（主线程补充,贯穿 §2）

overlay **不是 divergence 垃圾堆,而是有主动上游贡献回路的受控分叉**。每条 divergence(无论 addon 还是 patch)**必须分类**:

| 类别 | 定义 | 处置 | 范例 |
|---|---|---|---|
| **(a) 可上游化** | 通用修复 / 上游自身代码路径里的 bug,对所有用户有益,非我们产品独有 | 给 NVIDIA 上游**提 PR**;**合并后从 `patches/` 退役**(bump `UPSTREAM_PIN` 越过合并点,删该 patch,避免重复 apply 冲突) | **SSE/client-disconnect 崩溃修复**(`/v1/chat/completions` 客户端断开后 Myelin "already loaded binary graph" 崩溃,voice-agent barge-in 触发)——已 PR-pending,典型 (a) |
| **(b) 自留** | 我们路线图上、上游近期不做的,我们先做了;产品/模型/性能路径专属 | 长期留在 overlay 的 `addon/`+`patches/`,**不提 PR**(或上游无意接收) | MOSS-TTS-Nano 移植(我们的产品模型)、slot-pool N=2 并发(边缘并发护城河) |

**(a)/(b) 判定准则**(写进 `DIVERGENCE.md` 顶部):
1. 这是上游**自己代码里的正确性 bug** 吗?→ **(a)**。
2. 这是**我们产品/模型/路线图专属**的特性或性能路径吗?→ **(b)**。
3. **拿不准 → 默认 (b) 先携带**,但标 `pr_candidate: true`,下次 review 重新评估(宁可多背一会,不要贸然认为上游会收)。
4. 边界例子:`sizeof(half)` KV buffer dtype ABI 修复——若是上游通用 bug 则 (a),若仅 MOSS FP32 IO 特有则 (b);需逐条判,默认 (b)+pr_candidate。

**build.sh + CI 配合**:`build.sh` 对 (a)(b) 两类 patch 都 apply;CI 加一条检查——当某 (a) 类 patch 对应的上游 PR **已 merge**,告警提示"下次 bump pin 时退役该 patch",防止"上游已有 + 本地 patch 再打"的重复冲突。

**增强版 `DIVERGENCE.md` 条目模板**(在 §2 基础模板上加分类字段):

```text
## <topic>  (e.g. moss-tts-nano-port / sse-disconnect-fix / slot-pool-concurrency)
category:        a | b
pr_candidate:    true | false        # b 类里仍可能上游化的,标 true 待评估
upstream_pr:     <url + status>      # a 类必填;merged 后触发退役
retirement:      merged → drop patch, bump UPSTREAM_PIN past <commit>   # a 类
rationale:       <为什么 diverge:runtime need / device constraint / perf result>
scope:           addon=<files>  patches=<NNNN-*.patch>
validation:      unit / hardware / audio-ASR-TTS gate
last_replay:     <上次 rebase 上游的日期 + 冲突/决策>
```

> MOSS 是「首个 overlay 迁移范例」:适配器已在 voxedge(`voxedge/backends/jetson/moss_tts_nano.py`),引擎侧(worker/runtime/kernel)目前就地在 fork,需按 §2 抽取法迁进 `addon/`(纯新增)+ 最小 `patches/`(注册/CMake 挂载),并在 DIVERGENCE.md 记为 **category b**(产品模型,上游不收)。整条链:adapter@voxedge → worker/kernel@overlay → engine@HF → manifest@voxedge。

---

## 3. voxedge 运行时 artifact manifest

**现状(已部分存在,可复用)**:
- `deploy/artifacts/MANIFEST.md:1-5` 声明大 artifact 不入 git、优先 HF;Jetson Qwen3 在 `harvestsu/qwen3-edgellm-jetson-artifacts`(`:7-13`);RK 在 `harvestsu/seeed-local-voice-rk-artifacts` + `rk_manifest.json`(`:41-47`)。
- 产品 RK 下载器读 `RK_ARTIFACT_MANIFEST`/`RK_ARTIFACT_REPO_ID` → 取 `rk_manifest.json` → 逐文件下载 + SHA 校验(`server/core/rk_artifacts.py:52-70`、`:143-162`)。
- **voxedge 已有 env-free RK artifact adapter** `RKArtifactConfig`(`voxedge/backends/rk/artifacts.py:54-77`)+ `ensure_rk_artifacts(config)`(`:171-221`)。→ manifest 模式 RK 侧有雏形,泛化即可。

**建议**:通用运行时 artifact identity 移到 `voxedge/voxedge/artifacts/manifest.json`(JSON 优先,因现有 deploy manifest 都是 JSON、产品已有 JSON 加载;TOML 增加 parser 问题;Python 模块 manifest 不适合外部更新的元数据)。

**schema**:`schema_version` / `backend_key`(如 `rk.tts`) / `artifact_ref`(如 `rk3588-kokoro-hybrid-34pct-2026-05-23`) / `device`(`jetson-orin-sm87`/`rk3588`) / `precision`(`fp16`/`int8`/`w4a16`) / `hf_repo`+`revision` / `sha256` + `file_list[].sha256` / `file_list[].path`(**artifact 相对路径,非 install root**) / `file_list[].source_path`(HF 源路径) / `runtime_contract`(env/profile 约束)。字段都有现成参照(`deploy/artifacts/rk_manifest.json:1-8/222/242-257` 等)。

**下载 helper**:`voxedge/voxedge/artifacts/download.py`,接口 `resolve_artifact(artifact_ref, install_root, manifest_path=None, hf_endpoint=...) -> ArtifactInstallResult`,env-free(仿 `ensure_rk_artifacts`)。**install-path 留 profile**:manifest 存相对 `path`,install root 由产品/profile 选(现 RK 已从 config/spec 解析 root,`voxedge/backends/rk/artifacts.py:195-206`)。

**config 接法**:voxedge config dataclass 加 `artifact_ref: str | None = None`;产品/profile 设 `artifact_ref`/`artifact_install_root`/env override,builder 映射进 config(契合 `voxedge_backend_config.py:1-13` 的 env/profile→config 翻译层定位)。

**RK Kokoro 根治范例(对应 voice-pack 静音 bug)**:当前 RK Kokoro artifact set 只列 ONNX/RKNN(`rk_manifest.json:240-258`、`284-315`),**不含 `style.npy`**。Required fix:`file_list` 必须列**runtime 需要的每个文件含 `style.npy`/voice pack sidecar`;若缺,image build 或启动 preflight **必须 fail**,而不是 TTS 静默返回静音。`[当前 style.npy 覆盖 UNVERIFIED — 需手工核 RK Kokoro backend + artifact tree]`。

---

## 4. 产品清理实施步骤

> 注:本节描述的 `app/backends/jetson/` 死副本即 restructure 前的旧布局。产品服务目录已 `app/`→`server/`,真实后端实现现已迁入 `voxedge` 包(`voxedge.backends.jetson.*`)。以下路径保留 `app/backends/jetson/` 写法以描述「待退役的旧副本」语义;实际清理时按 `server/` + `voxedge` 现状对应。

**死副本边界**:registry 已指 voxedge(`asr_backend.py:150-154`/`tts_backend.py:145-153`),factory 先 build voxedge config(`:176-198`/`:174-201`);但 `app/backends/jetson/` 仍有全套副本(`trt_edge_llm_asr/tts.py`、`kokoro_trt.py`、`matcha_trt.py`、`moss_tts_nano.py`、`paraformer_trt.py`、`qwen3_trt.py`、`trt_edge_llm_ipc.py`、`__init__.py`)。**Caveat**:测试/脚本仍 import 这些副本(`app/tests/test_trt_edge_llm_asr.py:9`、`test_trt_edge_llm_tts.py:20`、`test_kokoro_trt.py:34`、`scripts/verify_product_explicit_kv_tts.py:8`)——清理前先 repoint/退役它们。

**唯一活的跨界 import(要消除)**:`build_trt_edge_llm_asr_config()` 从 `app.backends.jetson.trt_edge_llm_ipc` 取部署路径常量(`voxedge_backend_config.py:97-104`);`build_trt_edge_llm_tts_config()` 同(`:754-764`)。源模块定义 env 派生 root(`app/backends/jetson/trt_edge_llm_ipc.py:32-47`)。

**迁移全集**(每个常量 file:line 见 spec 表;摘要):`ASR_BINARY`/`ASR_WORKER_BINARY`/`ASR_ENGINE_DIR`/`ASR_AUDIO_ENC_DIR`/`ASR_PLUGIN_PATH`/`TTS_BINARY`/`PLUGIN_PATH`(部署路径)+ `resolve_tts_{talker,code_predictor,tokenizer,code2wav,worker_binary}_dir`(env resolver)+ `qwen3_runtime_profile`(profile resolver)。定义在 `app/backends/jetson/trt_edge_llm_ipc.py:55-269` 各处。

**移动目标**:新建 `server/core/deploy_paths.py`,把产品/profile 部署路径逻辑(`EDGE_LLM_BASE`/`EDGE_LLM_BUILD_DIR`/`OVS_WORKER_BUILD`/binary/plugin/engine dir + resolver)移入;通用 helper(`run_binary`、safetensors、mel 常量)留 `voxedge.backends.jetson.trt_edge_llm_ipc`(已 env-free,`voxedge/.../trt_edge_llm_ipc.py:6-23`)。

**repoint**:`voxedge_backend_config.py:98` 与 `:755` 的 `from server.backends.jetson.trt_edge_llm_ipc import (...)` → `from server.core.deploy_paths import (...)`;之后删 `app/backends/jetson/trt_edge_llm_ipc.py`。

**删除清单**(repoint 测试后):`app/backends/jetson/{__init__,trt_edge_llm_asr,trt_edge_llm_tts,trt_edge_llm_ipc,kokoro_trt,matcha_trt,moss_tts_nano,paraformer_trt,qwen3_trt}.py`。

**测试触点**:更新 registry 测试(`test_backend_factory.py:57-72`、`:105-116` 仍注入旧模块路径);保留 config fidelity 测试(`test_voxedge_backend_config.py:1-20/33-45/363-410/578-585`);跑 voxedge dep 测试(`voxedge/tests/test_dep_checks.py:20-35/75-132`)。

---

## 5. export 脚本迁移清单

| 当前位置 | 迁往 | 备注 |
|---|---|---|
| `vendor/moss-tts-nano/export_moss_tts_browser_onnx.py` | `voxedge-engine/addon/scripts/moss/` | Jetson/MOSS 导出,非产品 runtime |
| `vendor/moss-tts-nano/export_hf_to_tts_onnx.py` | `voxedge-engine/addon/scripts/moss/` | 依赖 HF 源/导出环境 |
| `scripts/build_engine_bundle.py` | `voxedge-engine/addon/scripts/packaging/` | build 复现 + artifact 打包属 overlay |
| `scripts/p7b/quantize_static_p7b.py` | RK overlay(Kokoro tail)否则 engine overlay `addon/scripts/quantization/` | 产 Kokoro tail-rest ONNX,偏 RK |
| `scripts/p7b/quantize_static_mmgemm_p7b.py` | 同上 | 配 build manifest 命名产物 |
| `third_party/qwen3-edgellm-jetson/scripts/export_qwen3_asr_onnx.sh` | `voxedge-engine/addon/scripts/qwen3/` | overlay 持导出配方 |
| `third_party/qwen3-edgellm-jetson/scripts/export_qwen3_tts_onnx.sh` | `voxedge-engine/addon/scripts/qwen3/` | 同上 |
| `third_party/qwen3-edgellm-jetson/scripts/package_qwen3_artifacts.py` | `voxedge-engine/addon/scripts/packaging/` | 产 runtime manifest payload |
| `third_party/qwen3-edgellm-jetson/scripts/deploy_qwen3_artifacts.py` | 拆:通用下载 helper→`voxedge/voxedge/artifacts/`;build-only verifier 留 overlay | runtime 用户调库 helper,非 third-party 脚本 |
| `third_party/qwen3-edgellm-jetson/scripts/*` | `voxedge-engine/addon/scripts/qwen3/` 按 export/build/quantize/verify 分组 | 对 third_party 的相对依赖变 overlay-relative |
| `third_party/rkvoice-stream/models/*` | `rkvoice-engine/addon/models/` | runtime `rkvoice_stream/` 与转换脚本分离 |

---

## 6. 分期实施(低风险 → 高)

| Phase | Risk | Work | Gate | Verification |
|---|---:|---|---|---|
| P0 inventory | Low | 冻结现状证据:registry/import/manifest/脚本清单 | 无 | `rg from app` in voxedge;查 registry |
| P1 deploy-path 抽取 | Low | 建 `server/core/deploy_paths.py`,repoint `voxedge_backend_config.py:98/755` | 不需设备 | `test_voxedge_backend_config.py` |
| P2 死副本清理 | Medium | repoint 测试/脚本 → 删 `app/backends/jetson/*.py` | P1 | factory 测试更新 + app/voxedge 单测 |
| P3 voxedge manifest | Medium | 加 `voxedge/voxedge/artifacts/manifest.json` + helper + config `artifact_ref` | HF 可只读起步,本地 file manifest 测 | 单测 local download/sha,仿 `rk_artifacts.py:143-162` |
| P4 HF artifact push | Medium/High | 发 Jetson/RK artifact 完整 file_list(含 Kokoro style/voice sidecar) | 先建 HF repo + 上传 | 每文件 SHA 校验;缺文件 startup fail |
| P5 overlay 抽取 | High | 建 `voxedge-engine`,从 fork 抽 addon/patches + build manifest | clean 上游 pin + build host | clone@pin + apply overlay + build,比对产物 |
| P6 RK 对称 overlay | High | RK 转换/build 脚本拆出 runtime 包 + 发 RK build manifest | RK build host/WSL2 + 设备验 | RKNN 转换需 WSL2/x86 + 设备验 |
| P7 设备镜像 cutover(#27) | Highest | 镜像重建装 voxedge + 按 manifest 拉 artifact + 停拷 build 树 | P3-P6 完成 | Jetson/RK 硬件 smoke:/health、/tts/stream、ASR/TTS roundtrip |

**manifest 对镜像 cutover 的影响**:当前 Jetson 镜像 bake `deploy/jetson-workers` + 整棵 `third_party/qwen3-edgellm-jetson`(`Dockerfile.jetson:85-100`);RK 镜像 bake app/scripts/整棵 `third_party/rkvoice-stream`/runtime libs/Kokoro artifact(`Dockerfile.rk:85-107`)。cutover 后镜像只 bake:runtime OS deps + 产品 app/profile + `pip install voxedge[...]` + runtime libs + manifest bootstrap;模型/engine 由 manifest 拉 + SHA 校验 + 装进 profile 选定路径 —— **直接根治"手工 COPY 漏 `style.npy`"那类静音 bug**。

---

## 7. 风险 + 一次真实上游更新演练

**风险**:
- 隐藏的测试/脚本对死副本的依赖(`test_trt_edge_llm_asr.py:9` 等)——repoint/退役后才能删。
- 路径语义漂移:`voxedge_backend_config.py:10-17` 承诺 byte-identical env/默认行为,抽 deploy-path 必须保持。
- artifact 完整性:RK Kokoro manifest 未含 `style.npy`(`[UNVERIFIED]`),加 preflight。
- overlay patch sprawl:fork 当前改了多个上游文件,无 addon/patch 纪律则上游 bump 成本高。
- build/runtime 耦合:Docker 当前把 build 树(`third_party/*`、`scripts/`)带进 runtime 镜像(`Dockerfile.jetson:77-90`、`Dockerfile.rk:85-93`)。

**一次上游更新**:
1. 改 `UPSTREAM_PIN` → 新上游 commit。
2. clean worktree:clone NVIDIA 上游,checkout 新 pin。
3. 拷 `addon/`。
4. 按序 apply `patches/*.patch`。
5. patch 冲突 → 在上游文件解,只重生成那个 patch,DIVERGENCE.md 记冲突/决策。
6. `build.sh --manifest manifests/qwen3-tts-highperf-sm87.toml`。
7. 产 worker/plugin/engine + checksum 清单。
8. 上传 HF,artifact 不入代码仓。
9. 更新 `voxedge/voxedge/artifacts/manifest.json` 的 rev/SHA/file_list。
10. 跑库单测 → 产品 config 测 → 硬件 smoke/roundtrip。
11. **(a) 类退役**:若本轮上游已 merge 某 (a) 类 PR,从 `patches/` 删该 patch、确认 pin 已越过合并点(§2A)。

---

## 附:实施顺序建议(主线程)

- **立即可做(纯本地、低风险、不碰设备)**:P0 + P1 + P2 —— 干掉死副本 + deploy-path 迁 profile,消除"循环感"。这部分可先做、单测护栏足够。
- **需要 HF 准备**:P3 + P4 —— manifest 化 + 推 artifact(RK 侧有雏形可复用)。
- **重活/需 build host**:P5 + P6 —— overlay 抽取(MOSS 作首例)。
- **绑 cutover #27**:P7 —— 镜像重建,顺带根治 voice-pack 静音。
