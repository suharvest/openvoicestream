# SparkTTS Voice-Clone (zero-shot 音色克隆) · 设计 Spec

状态：DESIGN · 2026-06-25 · 作者：CTO 调研线程
前置：controllable 模式全链路已 SHIPPED（LLM 上 edge-llm bf16/fp16 mixed TRT + 常驻 C++ streaming worker `spark_tts_worker` + N=2 + voxedge/OVS 集成）。本 spec 只设计 **clone 模式接入**，不实现/不导出/不 build。
源码确认：SparkTTS spike 在 wsl2-local `~/spike-sparktts/Spark-TTS`，下面所有 file:line 均为该路径下源码。

---

## 0. 结论先行（TL;DR）

1. **clone 的 d_vector 来源与 controllable 完全相同** —— 都是 `global tokens(32) → speaker_encoder.detokenize → d_vector[1024]`（`sparktts/models/bicodec.py:184`）。唯一区别：controllable 的 global token 由 LLM 现场生成；clone 的 global token 由**参考音频经 ECAPA+FSQ 分析得到**。==> **现有 `sparktts_speaker_decoder.fp32.engine` 直接复用，零改动。**
2. **clone 多出的是一条"参考音频→token"的分析链**，三个模块：wav2vec2-XLSR-53（~300M，提语义特征）+ BiCodec semantic encoder（ConvNeXt/Vocos + FVQ）+ BiCodec global tokenizer（mel → ECAPA-TDNN + Perceiver + ResidualFSQ）。
3. **关键设计洞察成立：分析链是"一次性 enrollment"**，不在合成热路径上，无低延迟要求。==> **推荐方案：enrollment 时在 host(wsl2-local) 用 PyTorch 跑一次分析链，产出 `{global_ids[32], (可选)ref_semantic_ids[Tr], ref_text}`，缓存为 voice profile（小 JSON/npy，~数百字节~KB）。Jetson 合成时只读缓存的 token，不跑 wav2vec2/ECAPA。** Jetson 上**完全不需要**为分析链导出/部署任何额外 engine，显存零增量。
4. **接入难度：低-中。** worker 只需加 clone 分支（新 prompt 模板 + 接收 host 传入的 global/ref-semantic token，跳过 controllable 属性 token）。d_vector 计算路径、BiCodec vocoder、LLM engine 全部复用。最大的"新工作"是 host enrollment 工具（纯 PyTorch，照搬 `BiCodecTokenizer.tokenize`）+ voxedge 的 voice-registry 表面。
5. **阶段**：P1 host enrollment 工具（PyTorch，产 token 缓存）→ P2 worker clone 分支（prompt + token 注入）→ P3 voxedge voice-registry + OVS 注册/选择 API → P4（可选）分析链上 Jetson TRT（仅当需要"设备端自助注册"才做，否则永不需要）。

---

## 1. clone 契约（读 SparkTTS 源码确认）

### 1.1 clone prompt 模板 — `cli/SparkTTS.py:53 process_prompt`

clone 有**两种策略**（由是否提供 `prompt_text` 即参考音频转写决定）：

**策略 A · text + ref-global only（无转写，`prompt_text=None`）** — `SparkTTS.py:95-104`：
```
<|task_tts|>
<|start_content|> {text} <|end_content|>
<|start_global_token|> {global_tokens} <|end_global_token|>
```
- `global_tokens` = `"".join(f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze())`（`SparkTTS.py:74-76`），即参考音频的 **32 个 global token**。
- LLM 在此 prefix 后**自回归生成 semantic token**（音色由 global prefix 决定）。

**策略 B · transcript + ref-global + ref-semantic（有转写，`prompt_text` 给定）** — `SparkTTS.py:79-94`：
```
<|task_tts|>
<|start_content|> {prompt_text}{text} <|end_content|>          ← 转写拼在目标文本前
<|start_global_token|> {global_tokens} <|end_global_token|>
<|start_semantic_token|> {ref_semantic_tokens}              ← 注意：无 <|end_semantic_token|>，作为生成前缀
```
- `ref_semantic_tokens` = `"".join(f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze())`（`SparkTTS.py:80-82`），即参考音频的 semantic token，作为 **LLM 续写前缀**（in-context cloning：模型先"看到"参考音频的 semantic 序列，再续写目标文本的 semantic）。
- 这是更高保真的克隆策略（同时给声学前缀），但 prompt 更长、首块延迟更高。

**对比 controllable** — `process_prompt_control` (`SparkTTS.py:110`) 用的是 `<|task_controllable_tts|>` + `<|start_style_label|>` 属性 token；clone 用 `<|task_tts|>`（`TASK_TOKEN_MAP["tts"]`）+ global/semantic token。**两个是不同 task token，互不干扰，天然共存。**

### 1.2 token 解析回收（生成端，已在 worker 实现）

`SparkTTS.py:216-228`：生成结果用正则 `bicodec_semantic_(\d+)` 抽 semantic id；**clone 模式不重新抽 global**（global 来自参考音频，已知），直接复用 `process_prompt` 返回的 `global_token_ids`（`SparkTTS.py:191-193, 231-234`）。worker 现已按 token-id range 解析（`spark_tts_worker.cpp:163-166`），clone 下只需**不依赖 LLM 吐 global，而是用 host 传入的 global**。

### 1.3 detokenize / d_vector 来源 — **与 controllable 同一条路径**

`audio_tokenizer.py:132 detokenize` → `bicodec.py:171 detokenize`：
```python
z_q      = self.quantizer.detokenize(semantic_tokens)          # bicodec.py:183
d_vector = self.speaker_encoder.detokenize(global_tokens)      # bicodec.py:184  ← 关键
x = self.prenet(z_q, d_vector); x = x + d_vector.unsqueeze(-1); wav = self.decoder(x)
```
`speaker_encoder.detokenize` (`speaker_encoder.py:107-112`)：
```python
zq = self.quantizer.get_output_from_indices(indices)   # ResidualFSQ 反查码本
x  = zq.reshape(B, -1)                                  # [B, 128*32]
d_vector = self.project(x)                              # nn.Linear(128*32 → 1024)
```
**这正是我们已导出的 `sparktts_speaker_decoder.fp32.engine`**（worker `runSpeakerDecoder` global[1,1,32]→d_vector[1024]，`spark_tts_worker.cpp:295-312`）。==> clone 的 d_vector **不需要新建任何东西**：参考音频的 global_ids[32] 喂进同一个 speaker_decoder engine 即得 d_vector。

**这是整个接入最重要的复用点。**

---

## 2. 分析链（参考音频 → token）的完整链路

入口 `BiCodecTokenizer.tokenize(audio_path)`（`audio_tokenizer.py:119-130`）：

```
wav(16k, vol-norm) ──┬─► extract_wav2vec2_features ──► feat[1, T, 1024]  ─┐
 (process_audio,     │   (audio_tokenizer.py:85-99)                       │
  audio_tokenizer.py:72)                                                  ▼
                     │                                    BiCodec.tokenize (bicodec.py:151)
ref_wav(ref clip) ───┴─► mel_transformer(ref_wav) ──► mel[1,128,Tm] ──────┤
 (get_ref_clip,                                                           │
  audio_tokenizer.py:57)                                                  ▼
                              z = encoder(feat.T)          → semantic_tokens = quantizer.tokenize(z)   [1, Tr]  (8192 码本)
                              global = speaker_encoder.tokenize(mel.T)                                  [1,1,32] (FSQ 4^6=4096)
```

### 2.1 wav2vec2-XLSR-53 用法 — `audio_tokenizer.py:85-99`
- 模型：`Wav2Vec2Model.from_pretrained(".../wav2vec2-large-xlsr-53")`（`audio_tokenizer.py:52`），`output_hidden_states=True`。~300M 参数 transformer。
- **特征是第 11/14/16 层 hidden_states 的平均**（`audio_tokenizer.py:95-97`）：`feats_mix = (h[11]+h[14]+h[16])/3`。
- 输入：16kHz 单声道 wav（经 `Wav2Vec2FeatureExtractor` 归一化，`audio_tokenizer.py:87-93`）。输出：`feat[1, T_frames, 1024]`，T_frames ≈ 音频秒数 × 50（wav2vec2 50Hz 帧率）。dtype fp32。

### 2.2 BiCodec semantic encoder + quantizer — `bicodec.py:162-166`
- `z = self.encoder(feat.transpose(1,2))`：Encoder = Vocos/ConvNeXt 栈（config `encoder`: input_channels 1024, vocos_dim 384, 12 layers, out 1024）。输入 `feat[1,1024,T]`，输出 `z[1,1024,T]`。
- `semantic_tokens = self.quantizer.tokenize(z)`：`FactorizedVectorQuantize`（单码本，**codebook_size 8192**, codebook_dim 8, L2-normalize, config `quantizer`）。输出 `semantic_tokens[1, T]` int（每帧一个 0-8191 的 id），50Hz token 率。

### 2.3 BiCodec global tokenizer（ECAPA + Perceiver + FSQ）— `bicodec.py:167`, `speaker_encoder.py:100-105`
- `mel = mel_transformer(ref_wav)`（`bicodec.py:163`）：torchaudio MelSpectrogram，**128 mel**, n_fft 1024, win 640, hop 320, 16k（config `mel_params`）。
- `global_tokens = speaker_encoder.tokenize(mel.transpose(1,2))`（`bicodec.py:167`），内部（`speaker_encoder.py:100-105`）：
  - `ECAPA_TDNN_GLOB_c512(feat_dim=128, embed_dim=1024)` → features（`speaker_encoder.py:55-57, 102`）
  - `PerceiverResampler(dim=128, dim_context=512*3, num_latents=32)` → 32 个 latent（`speaker_encoder.py:58-60, 103`）
  - `ResidualFSQ(levels=[4,4,4,4,4,4], num_quantizers=1, dim=128)` → indices（`speaker_encoder.py:61-67, 104`）。
- 输出 **global_tokens `[1, 1, 32]`，每个 ∈ [0, 4095]**（4^6=4096 FSQ 码本，token_num=32）。

### 2.4 ref clip 截取 — `audio_tokenizer.py:57-70`
global tokenizer 只用 `ref_segment_duration` 秒的 ref clip（config，对齐到 `latent_hop_length`），不足则 tile 重复。semantic encoder 用全段 wav。==> enrollment 时参考音频建议 **3-15s 干净单人语音**（足够覆盖 ref clip + 给 semantic 前缀）。

### 2.5 各模块 shape/dtype 汇总（"需在导出时验证" 标注的是 spike 未实跑的）

| 模块 | 输入 | 输出 | 参数量 | 备注 |
|------|------|------|--------|------|
| wav2vec2-XLSR-53 | wav fp32 [1,L@16k] | hidden×25 层 → mix[11,14,16] → feat[1,T,1024] | ~300M | 最重 |
| BiCodec encoder | feat[1,1024,T] fp32 | z[1,1024,T] | ~小（12 层 ConvNeXt） | |
| FVQ.tokenize | z[1,1024,T] | semantic[1,T] int | 码本 8192×8 | |
| mel_transformer | ref_wav[1,Lref] | mel[1,128,Tm] | 0（torchaudio） | |
| speaker_encoder.tokenize | mel.T[1,Tm,128] | global[1,1,32] int∈[0,4095] | ~小（ECAPA c512） | |

---

## 3. 分析链架构：放哪 / 怎么跑（含推荐）

### 3.1 三个候选（题面 a/b/c）评估

| 方案 | 描述 | 显存账(Jetson) | 难度 | 评价 |
|------|------|---------------|------|------|
| **(a) 全上 Jetson TRT** | wav2vec2/encoder/ECAPA 都导 TRT，设备端实时分析参考音频 | wav2vec2 ~300M fp16≈600MB + encoder/ECAPA ~数百MB；常驻挤占 LLM/vocoder 显存（Orin Nano 8GB 紧张） | **高** | 仅当要"设备端自助注册"才值得。wav2vec2 是标准 transformer，**大概率可上 TRT**（无奇异算子），但 25 层 hidden_states 输出 + 取 11/14/16 层需自定义导出 wrapper；ECAPA(TDNN/SE/attentive-pool)、Vocos encoder 导出**需逐一验证**（"需验证"）。投入产出比差。 |
| **(b) host(wsl2-local/服务端) Python 常驻** | 分析链跑在有 GPU 的 host，合成时 Jetson 远程请求 d_vector/token | Jetson 0 增量 | 中 | 引入"合成依赖外部 host"的运行时耦合，违背边缘自洽。除非已有中心服务，否则不推荐。 |
| **(c) enrollment 一次性算好缓存** ★ | 注册音色时（离线/低频）在 host 跑一次分析链，产出 token 缓存；Jetson 合成只读缓存 | Jetson 0 增量 | **低** | **推荐。** 与"clone 是一次性 enrollment"洞察完全吻合。 |

### 3.2 推荐：方案 (c) enrollment-时-缓存

**理由**：
- 分析链不在合成热路径，无延迟约束 —— 一次注册算几百 ms 完全可接受。
- 产出物极小（global 32×int + ref_semantic Tr×int + ref_text 字符串），可序列化成 voice profile 文件随镜像/卷分发或 OVS 上传。
- Jetson 合成时显存零增量，不碰 8GB Orin Nano 的紧张预算，不动已 SHIPPED 的 LLM/vocoder 布局。
- 分析链留在 PyTorch（`BiCodecTokenizer` 原样可用），**无需任何 TRT 导出**，规避 wav2vec2/ECAPA/Vocos-encoder 三个未验证的导出风险。

**enrollment 在哪跑**：wsl2-local（x86+GPU，已是 SparkTTS 导出机）。提供一个 CLI/函数：
```
enroll(ref_wav, prompt_text|None) → VoiceProfile{
    voice_id, global_ids[32], ref_semantic_ids[Tr]|None, ref_text|None, sr, src_audio_md5
}
```
直接调 `BiCodecTokenizer.tokenize`（`audio_tokenizer.py:119`）取 `(global_token_ids, semantic_token_ids)`。

**何时才需要 (a)**：未来若产品要"用户在 Jetson 设备上对麦录音→即时注册音色"，再把分析链上 TRT。届时 wav2vec2 是主成本，可考虑 int8/小一号 wav2vec2，单列 spike。**当前不做。**

---

## 4. worker / backend / voxedge / OVS 接入改动点

### 4.1 worker（`spark_tts_worker.cpp`）改动

现状：worker 只做 controllable（`buildControllablePrompt` 写死 `<|task_controllable_tts|>` + 属性 token，`spark_tts_worker.cpp:149-156`），global 从 LLM 输出解析（`:586-589`），32 个到齐触发 speaker_decoder（`:601-610`）。

clone 分支需要：
1. **新请求字段**（stdin JSON）：`mode:"clone"`、`global_ids:[32 ints]`、`ref_semantic_ids:[ints]|null`、`ref_text:string|null`。controllable 仍走 `mode:"controllable"`(默认)。
2. **新 prompt 构造** `buildClonePrompt(text, global_ids, ref_semantic_ids, ref_text)`：拼 §1.1 的 A/B 模板（用 `<|task_tts|>` + `<|bicodec_global_i|>` ... ；策略 B 再加 `<|bicodec_semantic_i|>` 前缀）。token 文本→id 由 LLM tokenizer 处理（与 controllable 同，raw prompt 不套 chat template）。
3. **d_vector 来源切换**：clone 下**不等 LLM 吐 global**，而是收到请求即用传入的 `global_ids[32]` 调 `runSpeakerDecoder`（复用 `:295`，零改动），立刻 d_vector-ready。consumer 循环对 clone **只解析 semantic token**（global range 的 token 在 clone prompt 里是输入前缀，理论上不会再被生成；若出现可忽略）。
4. **semantic 前缀处理（策略 B）**：LLM 会把 `ref_semantic_ids` 当作已生成前缀续写。worker 解析 LLM 输出 semantic 时，需**跳过 prompt 里已注入的 Tr 个 ref-semantic**（只取新生成的 semantic 喂 vocoder），否则会把参考音频的内容也合成出来。==> 记录 prompt 中 ref-semantic 数量，consumer 解析时偏移（"需验证"：确认 edge-llm streaming 输出是否包含 prompt token 的回显；若 worker 走的是"只吐新生成 token"则无需偏移）。
5. 其余（streaming overlap-chunk vocode `:425`、cancel、N=2 slot pool）**完全复用**。

工作量：~1 个新函数 + 1 个分支 + consumer 小改，**中低**。

### 4.2 backend（`voxedge/backends/jetson/sparktts_trt.py`）改动
1. `SparkTTSConfig` 新增 clone 相关默认（可选，多数 clone 参数随请求来）。
2. `_build_request`：当 kwargs 带 `voice_profile`（已加载的 VoiceProfile）时，发 `mode:"clone"` + `global_ids`/`ref_semantic_ids`/`ref_text`，**不发 gender/pitch/speed**；否则走现有 controllable 分支。
3. `_parse_style` 之上加 voice 路由：`speaker`/`voice` 值若是已注册 voice_id（命中 registry）→ clone；否则尝试解析为 `gender_pitch_speed` controllable spec（现状）。
4. 新增 voice-registry 加载（见 §4.3）。`capabilities` 可加一个 `VOICE_CLONE`/`SPEAKER_REF` cap（若 `TTSCapability` 有；否则文档标注）。

### 4.3 voxedge voice-registry（新组件）
- 一个轻量 registry：`voice_id → VoiceProfile`（global_ids/ref_semantic_ids/ref_text/sr/md5）。
- 存储：profile 文件（JSON+npy 或单 JSON 内嵌 int 数组，体积 KB 级）放在卷/目录（如 `/opt/models/sparktts-0p5b/voices/<voice_id>.json`），backend preload 时扫描加载。
- 与 controllable 共存：controllable speaker spec（`female_moderate_high`）和 clone voice_id 同走 `speaker`/`voice` 字段，registry 命中即 clone，未命中按 controllable 解析。命名建议给 clone voice_id 加前缀（如 `clone:alice`）避免与 `gender_*` 撞。

### 4.4 OVS / 产品表面
1. **注册 API**（enrollment）：`POST /voices`（multipart：参考音频 + 可选 transcript + voice_id）→ 后端在 host 跑分析链（§3.2）→ 写 VoiceProfile → 返回 `voice_id`。注：分析链需 GPU host；若 OVS 跑在 Jetson 而 enrollment 想在 Jetson 做，则触发 §3 方案(a) 讨论 —— **首版建议 enrollment 走 host 工具/离线，OVS 只做"上传已生成的 profile"或代理到 host**。
2. **合成选音色**：现有合成请求的 `voice`/`speaker` 字段填 `voice_id` 即可（registry 路由），无需新参数。
3. 列表/删除：`GET /voices`、`DELETE /voices/<id>`（操作 registry 目录）。

---

## 5. 复用 vs 新建清单

| 组件 | clone 是否复用 | 说明 |
|------|---------------|------|
| LLM engine（edge-llm mixed-precision TRT） | ✅ 复用，零改动 | 同一 `<|task_tts|>` task token 在 SparkTTS 词表内；只是 prompt 前缀不同 |
| `sparktts_speaker_decoder.fp32.engine`（global→d_vector） | ✅ 复用，零改动 | clone 的 d_vector 与 controllable 同路径（§1.3） |
| BiCodec vocoder engine（semantic+d_vector→wav） | ✅ 复用，零改动 | |
| C++ worker 框架（slot pool / cancel / overlap-chunk streaming / N=2） | ✅ 复用 | 只加 clone 分支 |
| **wav2vec2-XLSR-53 runtime** | ❌ 新建（host PyTorch） | 仅 enrollment 用，不上 Jetson |
| **BiCodec semantic encoder + FVQ.tokenize** | ❌ 新建（host PyTorch） | 仅 enrollment 用 |
| **BiCodec global tokenizer（ECAPA+Perceiver+FSQ.tokenize）** | ❌ 新建（host PyTorch） | 仅 enrollment 用 |
| **host enrollment 工具** | ❌ 新建 | 薄封装 `BiCodecTokenizer.tokenize` |
| **voxedge voice-registry** | ❌ 新建 | 轻量 |
| **OVS 注册/选择 API** | ❌ 新建 | |

==> **Jetson 设备端无任何新 engine**（推荐方案下）。新建的全是 host enrollment 工具 + 软件层 registry/API。

---

## 6. 分阶段里程碑 + 难度/风险 + 显存账

| 阶段 | 内容 | 难度 | 风险 | 显存增量(Jetson) |
|------|------|------|------|------------------|
| **P0 · 验证克隆质量(spike)** | host 上跑原版 SparkTTS clone（PyTorch 全栈），确认 0.5B + 我们的参考音频克隆可懂、像；A/B 两策略对比 | 低 | 模型本身克隆能力（非工程）；0.5B clone 保真度未知（"需验证"） | 0（host） |
| **P1 · host enrollment 工具** | CLI/函数：ref_wav(+transcript) → VoiceProfile(global_ids/ref_semantic/ref_text)。直接调 `BiCodecTokenizer.tokenize`。序列化格式定稿 | 低 | 低（照搬源码） | 0 |
| **P2 · worker clone 分支** | `buildClonePrompt` + `mode:clone` 字段 + d_vector 走传入 global + semantic 前缀偏移处理 | 中 | semantic 前缀回显偏移（§4.1.4，需验证 edge-llm 是否回显 prompt token）；clone prompt 更长→KV/首块延迟 | 0（复用现有 engine） |
| **P3 · voxedge registry + OVS API** | registry 加载/路由 + 注册/选择/列表/删除 API + 与 controllable 共存路由 | 中 | API 设计 & enrollment 跑在哪（host vs 设备）；命名冲突 | 0 |
| **P4（可选）· 分析链上 Jetson TRT** | 仅当需"设备端自助注册"。wav2vec2/encoder/ECAPA 导 TRT | 高 | wav2vec2 25层 hidden 导出 wrapper；ECAPA/Vocos-encoder 算子可导性（"需验证"）；wav2vec2 ~600MB fp16 常驻显存 | **wav2vec2 ~600MB + encoder/ECAPA 数百MB**（Orin Nano 8GB 需谨慎，可能要按需 load/unload） |

**显存账小结**：P1-P3（推荐路径）**Jetson 显存零增量**，clone 与 controllable 共用全部 engine。只有 P4 才引入分析链显存（~1GB 量级），且仅在选择设备端注册时。

---

## 7. 与 controllable 模式的共存设计

1. **task token 天然隔离**：controllable=`<|task_controllable_tts|>`，clone=`<|task_tts|>`（`SparkTTS.py:146` vs `:84/97`）。同一 LLM engine、同一 worker、同一 vocoder 处理两者，靠 prompt 区分。
2. **请求路由**：worker `mode` 字段（默认 `controllable` 保持向后兼容）。backend 按 `voice`/`speaker` 是否命中 registry 决定：命中→clone(发 token)，未命中→controllable(发 gender/pitch/speed)。
3. **d_vector 路径统一**：两模式都经 speaker_decoder engine（controllable 用 LLM 生成的 global，clone 用参考音频的 global）。
4. **N=2 / streaming / cancel** 对两模式一致，slot 内无模式相关持久状态（仍满足 §5.3 per-slot 隔离）。
5. **容量**：clone profile 缓存独立于 controllable，互不影响；同一 worker 进程可任意混跑 controllable 与多个 clone voice_id 的请求。

---

## 8. 待验证清单（动手前需确认，勿臆测）

- [ ] **P0 克隆质量**：SparkTTS-0.5B 对我们目标音色的 zero-shot 克隆保真度（像不像 / 可懂）。是否需要策略 B（带 ref-semantic 前缀）才达标。
- [ ] **edge-llm streaming 是否回显 prompt token**：决定 worker 是否需对 ref-semantic 前缀做偏移（§4.1.4）。读 edge-llm StreamChannel 语义或实测。
- [ ] **ref_segment_duration / latent_hop_length / volume_normalize 具体值**：在 BiCodec config.yaml（`audio_tokenizer.py:59-62, 77`），enrollment 工具需读，影响参考音频最短时长。
- [ ] **clone prompt 长度 vs KV 容量**：策略 B 的 ref-semantic 前缀（Tr 可达数百）+ ref_text 占 KV，确认不超 worker 的 KV/maxGenerateLength 预算（`spark_tts_worker.cpp:534`）。
- [ ] **（仅 P4）** wav2vec2 25 层 hidden_states 输出的 TRT 导出可行性；ECAPA/Vocos-encoder 算子可导性。

---

## 9. P0 验证结果（DONE · 2026-06-25 · wsl2-local PyTorch host）

参考音频 = SparkTTS 仓自带 `Spark-TTS/example/prompt_audio.wav`（9.95s 中文女声广告，转写见 `infer.sh` 的 `prompt_text`）。
官方 clone 路径：`cli.SparkTTS.SparkTTS.inference`（`SparkTTS.py:158`）→ `process_prompt`（`:53`，A/B 策略由 `prompt_text` 决定）。
目标文本：3 ZH + 3 EN 新内容（与参考不同）；策略 A（无转写）+ B（带转写）各跑一遍，每句生成新克隆 wav。

**可懂度**（faster-whisper small 回转，CER 中/WER 英；中文 OpenCC t2s 归一）：

| 策略 | mean CER (ZH) | mean WER (EN) | mean cos(ref,clone) |
|------|---------------|---------------|---------------------|
| A（无转写） | **0.0185** | **0.0000** | **0.9224** |
| B（带转写+ref-semantic 前缀） | 0.0333 | 0.0256 | 0.9011 |

全部远低于阈值 0.10/0.15。少数非零是 whisper 近音错认（桌子→珠子、red→rent），非合成失真；听感对照确认音频清晰可懂。

**音色相似度**：clone-vs-ref d_vector cosine 逐句 0.88–0.94；**随机基线 cos(ref, random-global)=0.291**。克隆显著高于基线 → 明显像参考。
（d_vector 路径 = global_ids[32] → `speaker_encoder.detokenize`，即合成侧 `sparktts_speaker_decoder` engine 同一路径，§1.3。）

**结论：SparkTTS-0.5B zero-shot 克隆可用** —— 可懂 + 像参考。策略 A 在本组上略优于 B 且 prompt 更短/首块更快，**首版推荐策略 A**；策略 B 保留作高保真选项。
产物：`~/spike-sparktts/p0_clone_validate.py`、`p0_asr_validate.py`、`p0_out/`（12 wav + `p0_manifest.json` + `p0_asr_summary.json`）。

---

## 10. 附录 · VoiceProfile 格式（P1 DONE，P2 消费契约）

P1 host enrollment 工具 `~/spike-sparktts/enroll_voice.py`：`ref_wav (+--ref-text) --voice-id` → `BiCodecTokenizer.tokenize`（`audio_tokenizer.py:119`）→ 产出 VoiceProfile，存为 **npz（数值数组）+ json（元数据/索引）** 一对。

**每个 voice_id 落两个文件**（`voice_id` 中 `:` / `/` 替换为 `_`）：

`<out_dir>/<safe_id>.json`（路由 + 元数据，source of truth）：
```jsonc
{
  "voice_id": "clone:demo_zh_female",   // registry key；建议 clone: 前缀避免与 controllable gender_* 撞
  "global_ids": [3363, 2367, ... 32 个],// 32 ints ∈[0,4095] → 喂 sparktts_speaker_decoder engine 得 d_vector
  "ref_text": "...转写...|null",        // 策略 B 用；null → 策略 A
  "sample_rate": 16000,
  "source_meta": { "ref_wav_basename","ref_wav_md5","ref_dur_s","ref_sr_in",
                   "model_dir","enrolled_at","tool","format_version":1 },
  "d_vector_dim": 1024,
  "ref_semantic_len": 497,             // 策略 B 前缀长度（KV 预算参考，§4.1.4/§8）
  "npz_file": "<safe_id>.npz",
  "npz_md5": "..."
}
```
`<out_dir>/<safe_id>.npz`（数值数组）：
- `global_ids`  int32[32]
- `ref_semantic_ids`  int32[Tr]（策略 B 前缀；策略 A 可忽略）
- `d_vector`  f32[1024]（便利副本；= global_ids 过 `speaker_encoder.detokenize`，可由 global_ids 重算，存它省一次推理）

**P2 加载参考实现**：`enroll_voice.load_profile(json_path)` → 读 json + sibling npz，返回 `global_ids`/`ref_semantic_ids`/`d_vector` 已物化的 dict。voxedge voice-registry（§4.3）即按此加载：命中 voice_id → 发 `mode:clone` + `global_ids`（+ 策略 B 的 `ref_semantic_ids`/`ref_text`）。

**自洽验证（DONE）**：`p1_selfcheck.py` 确认 ① npz/json 往返一致；② 入册 `global_ids` 与 P0 克隆时参考音频提取的 token **逐位相同**；③ 缓存 d_vector == `global_ids` 经官方 `speaker_encoder.detokenize` 重算（cos=1.000000，max_abs 5.5e-7，仅 fp32 cuda↔cpu 舍入）→ 缓存 d_vector 即合成时所用，无漂移。
产物：`enroll_voice.py`、`voices/clone_demo_zh_female.{json,npz}`、`p1_selfcheck.py`。
