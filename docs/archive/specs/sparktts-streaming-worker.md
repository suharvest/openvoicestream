# SparkTTS 流式 C++ Worker + 并发 + voxedge/OVS 集成 · Spec

状态：DRAFT · 2026-06-25 · 作者：CTO 线程
前置：Phase-1 GOAL PASS（LLM 在 edge-llm bf16-hybrid TRT 上端到端 ASR 通过，见 `sparktts-jetson-phase1.md` + memory `sparktts_bicodec_decoder_trt_spike_2026_06_24`）

---

## 0. 目标

把 SparkTTS 从"一次性 CLI e2e"做成**常驻流式 C++ worker**，接进 voxedge/OVS 服务层，**支持 N>1 并发**。worker 写在 `~/project/jetson-voice-engine/native/edgellm_voice_worker/`（与 qwen3_tts_worker 同处），链接 edge-llm 库（`~/project/TensorRT-Edge-LLM` 分支 `feat/bf16-hybrid`）。

## 1. 仓库分层（确认）
- `jetson-voice-engine`（github suharvest，**我们的**）：worker C++（`native/edgellm_voice_worker/`）+ export 脚本 + build/recipes/deploy。
- `TensorRT-Edge-LLM`（NVIDIA fork，`feat/bf16-hybrid`）：edge-llm 库本体（runtime/plugin/AttentionPlugin + bf16-hybrid）。worker 链接它。
- `voxedge`（sibling）：backend 抽象 `voxedge/backends/jetson/`（新增 `sparktts_trt.py` + `_sparktts_util.py`）。
- `seeed-local-voice`：`configs/leaves` + `configs/profiles` + `server/core/voxedge_backend_config.py` + OVS 服务层。

## 2. Worker 架构

单进程持 **LLM engine（mixed-precision）+ BiCodec vocoder engine**，shared engine + per-slot execution context。

```
stdin JSON-line 请求(text + gender/pitch/speed + 生成参数 + stream/chunk 控制)
  → 构造 controllable prompt
  → [slot] LLM 自回归 decode (StreamChannel 逐 token)
       ├─ 收集前 32 个 bicodec_global token → FSQ→d_vector[1,1024]  ← §4 待定放哪
       └─ 之后 bicodec_semantic token 增量到达
  → [slot] BiCodec vocoder(d_vector + semantic chunk) → 16kHz PCM16
  → stdout JSON-line: event=chunk{audio_b64/bytes, chunk_index, samples, is_final, ...}
```

### 关键 file:line 参考（现有 qwen3 worker）
- engine 加载/持有：`qwen3_tts_worker.cpp:279-297, 375-379`
- 请求字段：`:223-241, 319-333`；chunk 响应字段：`:412-456, 199-220`
- generate-then-chunk（旧 per-frame 回调已移除）：`:261-264, 550-575`
- voxedge 端契约：`worker_io.py` 写请求 `:104-157`、按 id 多路复用 `:163-183`、yield 到 done `:206-244`、cancel `:246-272`
- LLM 流式：`StreamChannel`（`streaming.h:82-88, 159-196`，`streaming.cpp:357-391`，`llmInferenceRuntime.cpp:548-564, 725-740`）。**`onTokenGenerated` 只在 prefill 触发**（`:1175-1183`，`vanillaDecoder.cpp:122-132` decode 阶段不调）→ 用 StreamChannel。

## 3. 流式 BiCodec Vocoder
- **S3 原型 = overlap-chunk**：50Hz token，upsample 320× → 16kHz；12-16 token/块≈240-320ms，左 8-16 token overlap 丢弃，时域交叉淡入消咔哒。边界有 artifact。
- **生产 = stateful 导出（推荐）**：参考 `scripts/qwen3_tts_code2wav_stateful_export.py`（conv pending buffer 作 I/O state tensor `:48-76, 121-144`，chunked-concat vs 全段 parity gate `:307-339`，dynamic shape profile `:355-377`）。BiCodec conv-upsample[8,5,4,2] 无 ScatterND，比 qwen3 更简单；state buffer = causal conv left-pad，固定尺寸，**天然 per-slot 隔离**。
- **前置约束**：global 32 token 必须全收齐 + d_vector 算完，vocoder 才能起步（~640ms@50tok/s）。

## 4. ⚠️ host-split (FSQ→d_vector) 放哪 —— S1 必须定（codec 偏差修正）

codex 初稿建议放 Python（套 Qwen3 "speaker embedding 预传入" 模式）。**但 SparkTTS controllable 模式 global token 是 LLM 现场生成的**，d_vector 必须在 LLM 吐出 32 global **之后**算 → 放 Python 要 worker→Python→worker 往返，对流式 worker 别扭。
**推荐：在 worker(C++) 内收齐 32 global 后算 d_vector。** `speaker_encoder.detokenize`(global[1,1,32]→d_vector[1,1024]) 是个小网络（FSQ dequant + decoder），可做成**独立小 TRT engine** 由 worker 调，或 C++ 重实现。S1 先确认这条（spike 的 Python `sparktts_host_split.py` 是参考实现，需移到 worker 侧）。
（若证实 d_vector 可由纯 codebook gather 得到则更简单；spike 已知 FSQ 查表 = einx.vmap，需在 C++/engine 复现该 gather + decoder。）

## 5. 并发设计（一等要求）

### 5.1 voxedge 契约
- `WorkerIO` `asyncio.Semaphore(concurrency)` 限流（`worker_io.py:58-72`）。
- `SparkTTSConfig.worker_concurrency:int=1` → worker `--max_slots`（参考 `trt_edge_llm_tts.py:746-782, 100-120`）。
- `ConcurrencyCapability`（`concurrency_capability.py`，注册 `voxedge_backend_config.py:487-517`）；leaf `concurrency` 字段分 session 槽。

### 5.2 TRT 并发硬约束
execution context 非线程安全 → **shared engine + N contexts**（省显存，同 qwen3 talker N=2，slot-pool `qwen3_tts_streaming_worker.cpp:743-787`）。
- **每 slot 独立**：`IExecutionContext`（LLM + vocoder 各一）、KV cache、vocoder state tensors、I/O buffer、CUDA stream。
- **共享只读**：engine 权重、tokenizer、plugin、FSQ/speaker decoder 权重。

### 5.3 per-slot 状态隔离（硬性，绕开历史坑）
memory `tts_n2_throughput_investigation` / `tts_n2_phase_b_stability_landed`：StatefulCode2WavRunner 共享可变状态 → N=2 concurrent reset → illegal memory access → 被迫 max_workers=1 → per-slot tensor pool（8a286ce）才修。
**SparkTTS 原则**：vocoder state（conv pending buffer / hidden state）+ 所有可变 buffer **per-slot 独立分配**，禁止单例/类级可变状态。`SlotPool<SparkTTSSlot>`，每 slot 全生命周期持 LLM context + vocoder context + state buffer。**S1 架构就要体现**，不能后补。

### 5.4 显存（Orin NX 16GB，N=2 估算，需 tegrastats 实测）
LLM engine ~1.63GB(shared) + vocoder ~0.19GB(shared) + LLM context 2×~0.3GB + vocoder state 2×~20-50MB + OS ~2-3GB ≈ **5-6GB，可行**。N=3 需实测，不预先承诺。leaf 分 n1/n2（参考 `qwen3-tts-nx.yaml` talker batch lane b1/b2）。

### 5.5 流式×并发
每 slot 独立 decode loop + 独立 vocoder state；`WorkerIO` 已按 request_id 多路复用（`worker_io.py:163-183`），契约无需改。

## 6. voxedge / OVS 集成
- `SparkTTSConfig`（类比 `TRTEdgeLLMTTSConfig:100-186`）：`worker_binary, plugin_path, llm_engine_dir, tokenizer_dir, bicodec_engine_dir, speaker_decoder_engine(或 fsq model), sample_rate=16000, first_chunk_tokens, chunk_tokens, max_tokens, temperature, top_k, top_p, worker_concurrency=1, extra_env`。**禁止模块级 `os.environ.get`**（坑 `trt_edge_llm_tts_env_staleness`），env 读取在 `preload()`。
- `SparkTTSBackend`（TTSBackend）：`name/model_id/capabilities/sample_rate/preload/unload/_generate_streaming_impl/_synthesize_impl`（参考 `trt_edge_llm_tts.py:597-622, 909-1190`，`moss_tts_nano.py:130-160, 262-427`），复用 `WorkerIO`。capability：BASIC_TTS + MULTI_LANGUAGE + STREAMING + ConcurrencyCapability。
- leaf `configs/leaves/sparktts-0p5b.yaml`（n1/n2 + shared sub-leaf 放 vocoder/speaker-decoder/tokenizer）+ `models.yaml` 逻辑模型（precision=mixed_bf16，leaf 属性）。
- profile `configs/profiles/jetson-sparktts-0p5b-nx.json`（`SPARKTTS_*` env + required_engines）。
- `voxedge_backend_config.py::build_sparktts_trt_config(profile,env)` 注册进 `_TTS_CONFIG_BUILDERS`。

## 7. Build
- `native/edgellm_voice_worker/CMakeLists.txt` 加 `spark_tts_worker` target，链接 `voiceWorkerUtils edgellmCore commonLibraryExt`（参考 `:180-192`），`EDGE_LLM_BASE` 指向 `feat/bf16-hybrid` 构建产物。orin-nx build `ENABLE_CUTE_DSL=OFF`/CUDA12.6/sm_87。部署走薄 overlay。

## 8. 分阶段里程碑

| 阶段 | 目标 | 验收 |
|---|---|---|
| **S1** | worker 骨架 + 非流式 N=1；定 §4 d_vector 放法 | JSONL ready/done/error 正确；WAV md5 与 CLI Spark e2e 一致（或 ASR 等价）；无 leak；per-slot 资源结构就位 |
| **S2** | LLM StreamChannel 逐 token | 32 global 收齐触发 d_vector；semantic 增量；cancel 中止 decode 无挂起 |
| **S3** | 分块 vocoder（overlap-chunk → stateful） | overlap：TTFA < 全段；stateful：parity max_abs<1e-3；per-slot state 不共享 |
| **S4** | voxedge/OVS 集成 + N=2 真机 | leaf resolve；`/tts` stream 16kHz PCM16；**N=2 MD5 byte-identical 30 burst 0 CUDA error**（对齐 qwen3/moss N=2 验收）；tegrastats 显存实测 |

## 9. 风险
- `onTokenGenerated` decode 不触发 → 用 StreamChannel（必要时 patch `vanillaDecoder.cpp:128`）。
- StatefulCode2WavRunner 并发 reset 历史坑 → per-slot state 隔离 S1 就体现。
- BiCodec stateful export causal padding 建模错 → parity gate 失败。
- §4 d_vector 现场推导（非预传入）→ S1 必须定 worker 内算。
- bf16 分支 engine 与 orin-nx CUDA12.6/TRT10.3 link → 预先验。
- N=2 VRAM 估算 → tegrastats 实测；N=3 不承诺。
