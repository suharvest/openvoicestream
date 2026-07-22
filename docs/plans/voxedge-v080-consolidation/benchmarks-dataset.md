# VoxEdge — 性能 & 能力数据集(single source of truth)

> 这是把散落在 memory/specs/leaf 里的所有量化指标汇成的**唯一权威数据集**,带 `file` 出处。
> 用途:渲染对外的 `BENCHMARKS.md`(全量矩阵)、Landing hero 数字、Recipes 选择器。
> 扫描出处:`/Users/harvest/.claude/projects/.../memory/*.md` + `docs/specs/` + `configs/leaves`。
> 口径见文末「方法论」。`—` = 该数尚未测(见 GAPS)。

---

## 表 A — 性能矩阵(设备 × 后端 × 模型 × 精度)

| 设备 | 芯片 | 后端 | 模型 | 精度 | RTF | TTFA/TTFT | VRAM/引擎大小 | 准确率(CER/WER) | N(已验并发) | 出处 |
|---|---|---|---|---|---|---|---|---|---|---|
| Jetson Orin **Nano** | SM_87 | TRT-EdgeLLM | Qwen3-TTS-0.6B-Base | fp16 | 0.69 | 0.54s(warm) | talker 907MB;全链 ~1.3GB;N=2 需 5.6GB free | 可懂(ASR验) | **2**(fp16 实测 burst) | qwen3tts_base_v080_port |
| Jetson Orin **Nano** | SM_87 | TRT-EdgeLLM | Qwen3-TTS-0.6B-Base | int4-AWQ+fp8 | **0.44** | **0.21s**(warm) | talker **245MB**;整体 −1.06GB/实例 | ZH CER 1.7% / EN WER 0% | 1(int4+fp8 **N=2 待 gate**,见 GAPS) | qwen3tts_base_v080_port |
| Jetson Orin Nano | SM_87 | TRT-EdgeLLM | Qwen3-TTS-0.6B-CustomVoice ⚠️experimental | int4+fp8 | —(无记录,待 streaming-worker 实测) | — | talker 246MB;−960MB/实例 | 可懂(ASR smoke,CN+EN) | — | customvoice_talker_int4_eos_fail |
| Jetson Orin NX | SM_87 | TRT-EdgeLLM | Qwen3-ASR-0.6B | int4-AWQ | — | finalize 68–125ms | LLM 550MB;encoder fp16 381MB | **ZH CER 0% / EN WER 11%**(单词声学近音;**裸 llm_inference 契约验,未过 production worker**) | —(int4 N≥2 **无记录**) | qwen3asr_int4_validation |
| Jetson Orin NX | SM_87 | TRT-EdgeLLM | Qwen3.5-4B-AWQ | GDN+MTP | — | TTFT 2.28→**0.60s**(prefix-cache 3.8×) | 3.87GB 引擎 | bench 100%(p50 1.16s) | 1+ | qwen35_mtp_migration_prefix_cache |
| Jetson Orin NX | SM_87 | TRT | MOSS-TTS-Nano | FP32 | — | **157ms**(19× vs ORT) | — | CER 0(3 prompt) | ≥1 | moss_tts_nano_trt |
| Jetson Orin Nano | SM_87 | TRT | MOSS-TTS-Nano | FP32 | — | 290ms(N=1) | 5.53GB peak / 7.4GB 总 | CER 0 | **2**(30/30 byte-identical) | moss_tts_nano_trt / capability_framework_orin_nano |
| Jetson Orin Nano | SM_87 | Matcha-TTS | Matcha | — | — | N=2 slow-client TTFA 1.4–2.2× | — | md5 byte-identical | 2 | tts_n2_phase_b_stability / capability_framework_orin_nano |
| RK3588 | RK3588 NPU | RKNN | Kokoro(4-stage hybrid) | INT8/FP16 | **0.59**(HTTP) | bucket-8 0.79s / 16 1.62s / 32 3.1s | 41MB vocoder-front + 26MB tail(CPU) | — | 1 | kokoro_rk_perf_closure |
| RK3588 | RK3588 NPU | RKNN | Paraformer hybrid + Matcha | — | — | — | — | — | — | leaf_config_refactor(注册项) |
| RK3588 | RK3588 NPU | RKNN | Qwen3-RKNN-ASR | w8a8 | — | — | — | — | — | leaf_config_refactor(注册项) |
| RK3576 | RK3576 NPU | RKNN | Paraformer(full-encoder) | fp16 | — | — | — | ⛔ NaN/Inf(block31 溢出,NOT VIABLE) | — | leaf_config_refactor |
| RK3576 | RK3576 NPU | RKNN | Paraformer(hybrid, block30 split) | — | — | — | — | — | — | leaf_config_refactor |
| Raspberry Pi 4/5 | Cortex-A | sherpa-onnx | Paraformer / SenseVoice | FP32(CPU) | — | — | — | — | ≥1 | sensevoice_offline_3platform |
| Mac | CPU | sherpa-onnx | Paraformer / SenseVoice | FP32(CPU) | — | — | — | — | ≥1 | sensevoice_offline_3platform |
| (任意 Jetson) | SM_87 | TRT-EdgeLLM | Qwen3-ASR v0.8.0 streaming | fp16 | — | decode 68–125ms/hop;长音频 10.4s→3.2s(6.1×) | — | CER 0.0000–0.025(final 0.0) | — | asr_incremental_kv_12 |

---

## 表 B — 最佳实践 / 能力矩阵(用例 × 推荐栈)

| 用例 | 推荐设备 | ASR | TTS | LLM | 特殊能力 | 出处 |
|---|---|---|---|---|---|---|
| 实时本地语音对话(双语低延迟) | Orin NX | Paraformer(TRT)/ Qwen3-ASR(prefix) | Matcha(TRT 轻量) | Qwen3.5-4B-AWQ GDN+MTP(0.60s TTFT) | 双语自动识别、prefix-cache 3.8×、工具循环 | qwen35_mtp / qwen3asr_int4 |
| 高质量多语 TTS(表现力语音) | Orin Nano/NX | Qwen3-ASR int4 | **Qwen3-TTS 0.6B Base**(int4+fp8;N=2 fp16 已验,int4+fp8 N=2 待 gate) | — | 零样本音色(speaker-encoder 预算嵌入)、int4 talker 245MB | qwen3tts_base_v080_port |
| 语音克隆 / 自定义音色 | Orin NX | Qwen3-ASR | **Qwen3-TTS CustomVoice**(int4+fp8) | — | 9 内置说话人 + 自定义音色,−960MB/实例 | customvoice_talker_int4 |
| 超轻量语音(预算边缘) | RPi 4/5 | Paraformer(sherpa CPU) | MOSS-TTS-Nano(ORT)/ Matcha(sherpa) | — | 纯 CPU ONNX,无需 GPU | sensevoice_offline_3platform |
| NPU 加速多语(RK) | RK3588 | Paraformer hybrid / Qwen3-RKNN(w8a8) | **Kokoro**(4-stage hybrid,RTF 0.59) | — | NPU 加速 TTS、misaki ZH G2P、3-bucket 动态路由 | kokoro_rk_perf_closure |
| 机械臂 / server-loop(工具调用) | Orin NX | Qwen3-ASR(streaming,CER 0%) | Matcha(N=1)/ Qwen3-TTS base talker(N=2 fp16 已验) | Qwen3.5-4B-AWQ(server-loop) | server-loop(LLM+工具在服务端,agent 执行)、voice_arm | qwen35_mtp / qwen3tts_base_v080_port |
| 纯 ASR | 任意 Jetson/RK/RPi | Qwen3-ASR(TRT/RKNN/sherpa) | — | — | 增量 KV(长音 6.1×)、回滚最终 byte-exact、prefix-cache | asr_incremental_kv_12 |
| 纯 TTS | Orin Nano | — | MOSS-TTS-Nano(TRT)/ Kokoro | — | MOSS N=2 8GB byte-identical | moss_tts_nano_trt |
| 双语 ASR(中英) | Orin NX / RK3588 | Qwen3-ASR(auto detect) | — | — | force_language scaffold(贪婪解码 prime)、流式 partial+final | qwen3asr_int4_validation |
| 标点 + 声纹(opt-in) | Orin NX | Qwen3-ASR | Matcha | — | CT-Transformer 标点 + CAM++ 声纹,关闭时零开销 | ovs_punct_speaker_capabilities |
| 实时翻译 / 字幕 | Orin NX | Qwen3-ASR | (字幕路线不切 TTS) | NLLB(CT2,14–35× CPU) | live_caption / simul_interpret 双 app | nllb_translator_slim_cuda_jetson |

---

## 表 C — v0.8.0 N>1 实测(2026-06-21,全部真机 + byte-identical gate + 0 CUDA)

> 这是本轮迁移把「R&D 并发/流式」提升为产线后跑出的实测。N=2 = 已验上限。
> 工件锚点:fork tip `port/qwen3-tts-base-v080-n1n2` @ `7142a30`;镜像 `seeed-local-voice:v0.8.0-n1n2-rebake`;
> workers asr `5ebd436b` / tts shared `190178f6`;引擎 int4 talker(HF base int4fp8 245.9MB)/ asr-b2 `4122dfcc` / talker-b2 `f7339e02`。

| 维度 | 设备 | gate | 结果 | 出处 |
|---|---|---|---|---|
| **ASR N=2 streaming** | Orin NX | v080-0023(through-service) | 单会话 9 partials→final CER **0.105**(offline 同 clip ~0.05);N=2 zh/en 隔离无串话;5 并发→2 进 3 拒 `4429 too_many_sessions`;**0 CUDA** | edgellm-v080-migration/docs/plans/v080-0023-* + 任务记录 |
| **TTS N=2 int4 slot-pool** | Orin Nano | staggered(G1/G2/G3) | G1 staggered(B 不被 A 阻塞)/ G2 byte-identical(concurrent==solo)/ G3 4429 全 PASS;系统 RAM ~**4GB**(tegrastats peak 5718/7620,baseline 1703);worker RSS 908MB;无 OOM,fits 8G/16G;int4 talker **245.9MB** vs fp16 **903MB**(−73%) | 任务记录(staggered gate) |
| **TTS N=2 shared-engine** | Orin Nano | shared-engine gate | N=1 peak 3805MB→N=2 **5385MB**,2nd slot 仅 **+1.6GB**(context/KV 非二次权重),vs 独立省 **~436MB**;byte-identical(A `154f7880` / B `1a5324be` concurrent==solo);**0 CUDA** | 任务记录 / INDEX §2 shared-engine ctor |
| **TTS M5 spike** | Orin NX | phase5b | concurrent==solo RVQ hash byte-exact + audio md5 byte-exact + **0 CUDA** | 任务记录(phase5b M5) |
| **v0.8.0 vs v0.7.1 baseline** | Orin NX | ASR `--check` | **17/20 PASS**;英文+干净中文全过;多条优于 golden(zh_long_01 0.080→0.043);3 FAIL = 高基线 CER 硬 clip 的 abs-tolerance gate 脆性(非衰退) | seeed-local-voice/bench/regression/baselines/v080-c2-before-20260621/(工件在 ~/project/edgellm-v080-migration) |

对外视图:repo 根 `BENCHMARKS.md`(本表的外向子集)+ runbook `docs/deploy-v080-n1n2.md`。

---

## GAPS(尚无数据,需补测)

1. ~~**Qwen3-TTS Base int4+fp8 N=2**~~ — ✅ **CLOSED 2026-06-21**:int4 slot-pool N=2 staggered gate(G1/G2/G3)全 PASS,~4GB RAM fits 8G/16G;另有 shared-engine N=2(+1.6GB/省 436MB)byte-identical。见**表 C**。
2. **Qwen3-ASR int4 过 production worker + N≥2** — 现仅裸 `llm_inference` 生产解码契约验证(CER 0% / WER 11%);未过真实 `qwen3_asr_worker` streaming 路径,也无 int4 N≥2 记录(对应 plan F2/F3)。
3. **Qwen3-TTS CustomVoice int4 全链 perf** — 仅 ASR smoke 可懂 + talker 246MB;**无 RTF/TTFA 记录**,未过 streaming worker(对应 plan F4)。⚠️ experimental,不进默认/hero。
4. **Jetson AGX Orin** — 整套 VRAM/N 上限全 TBD(leaf 注册标 TBD-measure)。
5. **Orin Nano + LLM 同驻** — 真实显存争用场景未测。
6. **Qwen3-TTS W8A16** — CustomVoice 确认不可行(EOS 破);base W8A16 从未成功(prior "success" = 反序列化失败的假阳性)。
7. **SenseVoice RK/Jetson** — 只做了 RPi Phase1 适配,RTF/TTFA 未测。
8. **RK3576 Kokoro / Qwen3-ASR(w4a16g128)** — 待启动,无可行性证据。
9. **N>2 并发** — 任何设备都没测(N=2 是已验上限)。
10. **stateful code2wav v0.8.0 — 已实机+源码定论(2026-06-21)。根因 = "没接线"(非"不安全")。** worker 里 `StatefulCode2WavRunner` 只被 `#include` + 作为 unused 形参传入,**从未实例化**;slot-pool 每 slot 只构造 stateless `Code2WavRunner`,flag 仅进 `ready` 元数据 → 实测 N=2 下 =1/=0 输出逐字节相同(md5 `cccc41a6…`,两 slot rc=0)。**hazard 澄清**:历史 concurrent-reset 崩溃只存在于"共享单实例",stateful runner 状态全实例私有(`reset` 只 memset 自己缓冲)→ **per-slot 各自独立 stateful 实例原理上安全**。**收益**:stateful 省掉 stateless 每 chunk 25 帧左上下文重算(Code2Wav 总耗时 ~81-87ms,export gate max_abs 5e-6),但 Talker/CP 生成才是延迟主项 → 收益**增量非量变**,只在 chunk 显著变小/超长流式时才显著。**结论:backlog**(当前 RTF<1 性价比低;要启用约 7 项小改,最大风险 = ConvTranspose 相位对齐 click)。真实 env 名 = `EDGE_LLM_TTS_STATEFUL_CODE2WAV`。详见 manifest W2。
11. **35 leaf 组合 × 设备 的逐组合真机 e2e** — 仅 62 单测绿,无逐组合真机端到端。

---

## 方法论(口径,对外文档要附)

- **ASR 延迟** = finalize(audio-end→final),不含 VAD 400ms 等待(常量,与 decode 重叠)。CER/WER 用贪婪(top_k=1, temp=0)+ force_language scaffold,复刻生产解码契约(top_k=50 采样是 harness bug)。
- **TTS 延迟** = TTFA(warm,prefill+首 chunk)。RTF = wall-clock / 音频时长。N=2 slow-client TTFA 1.4–5× 惩罚是 memory-bandwidth-bound,非 bug。
- **N=x verified** = 真机 burst 压测 + MD5 音频门 + 零 CUDA/race。N=2 同时覆盖 TTS talker batch-lane 与 ASR streaming slot-pool。
- **prefix-cache TTFT** 2.28→0.60s 仅在 Qwen3.5-4B-AWQ GDN+MTP 实测,收益 per-model/per-prompt。

### repro 元数据(对外 hero/README 数字的硬门槛 + 已填明细)

规则:任何要进 Landing hero / README / Show HN 的数字,**必须**附下列 `repro` 块,否则不可对外。下面已为**当前 hero-eligible 行**逐行填好;非 hero 行标 `repro: pending`。

**① Qwen3-TTS Base int4+fp8 — RTF 0.44 / TTFA 0.21s(hero-eligible,N=1)**
```yaml
repro:
  metric: {RTF: 0.44, TTFA_s: 0.21, scope: N=1}   # N=2 待 F5,不在此 hero
  device: jetson-orin-nano-8gb
  chip: sm_87
  power: {nvpmodel: MAXN_SUPER, jetson_clocks: locked}
  build: {ENABLE_CUTE_DSL: OFF, kv: {maxInputLen: 1024, maxKVCacheCapacity: 1536}}
  engines: {talker: int4-AWQ 245MB, code_predictor: int4-AWQ, text_embedding: fp8-e4m3 320MB, code2wav: fp16}
  plugin_md5: 7d3fabe                # NvInfer_edgellm_plugin
  worker_md5: 6bc2d7db               # qwen3_tts_streaming_worker(precision-agnostic;须 build 时复核)
  profile: jetson-edgellm-v080-qwen3ttsbase
  hf_bundle: harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8
  input: precomputed speaker_embedding_b64;~12词中/英句
  sampling: {temp: ref-embedding, note: "不可降 talker_temperature(EOS runaway)"}
  accuracy_check: faster-whisper round-trip(NOT Groq=key过期);CER opencc t2s-normalize 后计
  source: qwen3tts_base_v080_port_2026_06_19.md
```

**② MOSS-TTS-Nano — N=2 byte-identical / TTFA 157ms(hero-eligible)**
```yaml
repro:
  metric: {TTFA_ms: 157(orin-nx) / 290(orin-nano N=1), N2: "30/30 byte-identical", peak_mb: 5530}
  device: jetson-orin-nano-8gb / orin-nx
  fork_commit: 3c6c263
  profile: jetson-moss-tts-nano-trt
  gate: MD5 音频门 + 零 CUDA/race
  source: moss_tts_nano_trt_production_ready.md
```

**③ Qwen3-ASR int4 — ZH CER 0% / EN WER 11%(条件 hero:须注明"裸解码契约,未过 production worker")**
```yaml
repro:
  metric: {ZH_CER: "0%", EN_WER: "~11%(1 acoustic word)", scope: "llm_inference 解码契约;F2b worker+N=2 PENDING"}
  device: jetson-orin-nano
  decode: {temp: 0, top_k: 1, top_p: 1, prime: "language <Lang><asr_text>", chat_template: applied}
  engine: int4-AWQ LLM ~525MB + audio-encoder fp16(minchunk1 build)
  clip_set: 6-clip(opencc t2s-normalized)
  hf_bundle: harvestsu/qwen3-asr-0.6b-int4-v080
  spec_commit: 1fce82e
  source: qwen3asr_int4_validation_forcelanguage_2026_06_20.md
```

**未填(repro: pending,不得对外):** CustomVoice int4(experimental)、Kokoro RK、Matcha、Qwen3.5-4B GDN+MTP、所有 GAPS 行。

→ **后续落地**:把本数据集转机器可读 YAML(每行带上述 `repro` 块),markdown 由脚本生成,CI 校验 README/HF/leaf 引用的 benchmark id 存在(plan E 节"single-source→多视图"的真正形态)。当前手写 markdown 为过渡,但 hero 行的 repro 已先行填齐。
