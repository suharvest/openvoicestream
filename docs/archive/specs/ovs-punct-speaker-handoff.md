# Handoff — 给 OVS 新增「标点」+「声纹 embedding」两个可选能力

> 写给接手实施的 agent。**本次目标**:在 OpenVoiceStream(OVS)上新增两个 feature-flag 控制的能力 —— ① 标点,② 声纹向量输出(embedding)。默认关闭,可手动开启;OVS 保持无状态,匹配/身份逻辑留给使用端。
> 范围只在 OVS 仓库。本文件是设计与背景的最终结论,**不含代码**,实施前请按下方"实施前必读"再校验一遍 file:line。

---

## 0. 一句话结论

OVS 加两个可选能力,**~3 天工作量,全程无状态**。标点纯文本后处理;声纹只在 **finalize** 时对该句音频跑一次 CAM++,**只输出 embedding**,由使用端去匹配注册库得到身份。两者各一个开关,**默认 off(行为与现在完全一致),可手动开**。

---

## 1. 背景(为什么做这件事)

涉及三套东西,逐步推导到这个结论:

- **sensecraft 语音系统**(`/Users/harvest/project/sensecraft_voice`,4 个独立 GitLab 仓库):边缘采集(voice-client)→ ASR(asr-service,Sherpa-ONNX)→ 云端后端(voice-service,存录音/识别/关键词/门店/用户)→ Web 后台。顶层 README 已写:`/Users/harvest/project/sensecraft_voice/README.md`。
  - 现有 **asr-service** 是三模型拼装:SenseVoice(ASR)+ CT-Transformer(标点)+ **CAM++/3D-Speaker(声纹)**。模型与配置见 `sensecraft-asr-service/config.json`(speaker 段在 :52,标点在 :44,asr 在 :37)。
- **Omi**(`/Users/harvest/project/omi`,BasedHardware 开源可穿戴):评估过用 Omi 设备做采集端。结论:Omi 设备只采音+Opus 编码,**转写在它后端用 Deepgram(云、收费、数据出境)**。Omi App 有 `custom` STT(`app/lib/services/devices/`、`app/lib/models/stt_provider.dart`)可指向自有 STT。Omi 的 BLE GATT 协议是公开标准(Service `19B10000-...`,音频包 `[2B包号][1B index][Opus]`),第三方设备可伪装接入。**Omi App/云代码不在我们仓库**,我们只在 voice-service 放宽过 MAC 校验(commit `67ac9db`)。
- **OVS / OpenVoiceStream**(`/Users/harvest/project/seeed-local-voice`,GitHub `Seeed-Solution/openvoicestream`):更产品化的本地流式 ASR+TTS,多后端(Jetson TRT / RK RKNN / RPi sherpa),稳定 HTTP/WS API,有一键部署。**计划用 OVS 当统一引擎、逐步替代 asr-service**。

**OVS 当前缺口 vs asr-service**:OVS 的 ASR 文本本身带标点(SenseVoice/Paraformer 自带),但**没有独立标点模块、没有声纹**。本次就是补这两块。声纹模型直接复用 asr-service 里现成的 CAM++,标点复用现成的 CT-Transformer。

> 详细的多轮评估推导(Omi 三条接入路径、OVS vs asr-service 契约对比、声纹/说话人术语澄清)已在对话里;关键结论都收敛进了本文件第 2-5 节。

---

## 2. 已锁定的设计决策(用户已拍板)

1. **声纹只在 finalize 时跑一次**(按句粒度,不逐帧、不在 partial 做)。同理标点也只在 final 时做。
2. **同步输出**:speaker_embedding 直接放进 final payload,不搞 async 补发(延迟可忽略,CAM++ 几秒音频 CPU 上几十 ms)。
3. **OVS 只发 embedding,不做匹配**:OVS 无状态,只吐声纹向量;注册库 / 余弦匹配 / 阈值 / 身份(speaker_id/name)**全在使用端**(voice-service 业务侧,它有 DB)。
4. **后端/设备无关**:标点(CT-Transformer)、声纹(CAM++)都是小 ONNX,用**通用 onnxruntime-CPU** 跑,**不要绑 sherpa 后端**(sherpa 只在 RPi/Mac profile 有;Jetson 用 TRT、RK 用 RKNN —— 见 `server/core/asr_backend.py:178-183` 与 `configs/profiles/*.json` 的 `asr_backend` 字段)。这样 RPi/RK/Jetson 通吃。
5. **两个 feature flag,默认关**:沿用 OVS 既有惯例(VAD 就是 `OVS_VAD_BACKEND` env 默认 + `?vad=` query 覆盖)。默认 off → 现有 `/asr/stream`、`/v2v/stream` 行为不变;**懒加载**,关着就不加载模型、不吃内存。

因为决策 3,**"辨认 vs 盲分(diarization)"对 OVS 不再有影响**(OVS 都只吐向量),该问题挪到使用端、可后续独立决定,**不阻塞本次 OVS 实施**。

---

## 3. OVS 侧最终交付范围(契约)

```
新增离线端点:
  POST /punctuate            text  → 带标点 text
  POST /speaker/embedding    audio → { embedding_b64, embedding_model, dim }

流式增强(/asr/stream 与 /v2v/stream 的 final):
  ?punctuate=true          → final.text 内联标点
  ?speaker_embedding=true  → final 同步带 embedding 字段

开关:
  全局 env 默认 off:   OVS_PUNCT / OVS_SPEAKER_EMB  (命名待定)
  单连接 query 覆盖:   ?punctuate=true / ?speaker_embedding=true
  懒加载:关闭则不加载对应模型

final payload 示例:
  关:  {type:"final", text:"你好世界"}
  开:  {type:"final", text:"你好,世界。",
        speaker_embedding:"<base64>", embedding_model:"campplus_sv", dim:192}
```

**embedding 是跨服务契约 —— 必须带元数据**(`embedding_model` + `dim` + 是否归一化)。一旦换声纹模型,旧注册向量全失效需重注册;使用端靠 `embedding_model` 版本号检测。

---

## 4. 实施前必读:关键 file:line(动手前再核验一遍)

> **⚠ 2026-06-09 已核验**:`server/main.py` 自 handoff 撰写后整体下移了 **约 +10 行**(顶部新增了导入/定义)。下方行号**已更新为当前真实值**,但实施前仍以 grep 符号为准。其它文件(`asr_backend.py`/`model_downloader.py`/`profile_loader.py`/`config.json`)行号准确无漂移。

均在 `/Users/harvest/project/seeed-local-voice`:

- **端点注册风格**(FastAPI 装饰器,照抄):
  - `server/main.py:2211` `POST /asr`(离线,UploadFile + `_asr_impl` 在 :2222)← **新离线端点照这个抄**
  - `server/main.py:2267` `WS /asr/stream`(`asr_stream()` 在 :2268)
  - `server/main.py:2890` `WS /v2v/stream`(`v2v_stream()` 在 :2891)
  - `server/main.py:1213` `GET /asr/capabilities`
  - API key 校验:`Depends(_require_api_key)`,定义 `server/main.py:168`;会话限流 `acquire_http(...)` `:2217-2218`(`acquire_http` 本体在 `server/core/session_limiter.py:267`)
- **流式 finalize / final 文本生成点**(标点和声纹的注入位置):
  - `_unpack_finalize_result` 本体在 `server/main.py:703`(未漂移);VAD endpoint finalize 调用在 :2454/:2510 一带,`final_text` 多处生成(:2454/:2459/:2483/:2510/:2514)→ 拿到 `final_text` 后,**标点在发送前插一行;声纹在此处对该段音频跑 CAM++**
  - 其它 final 点:`:2454`(force endpoint)、`:2478-2493`(显式 EOS,空字节 `if len(data) == 0` 在 :2478、`stream.finalize` :2482、payload :2484);v2v 的 final 在 `:3812-3878`(`finalize_with_status` :3812、server 回 `asr_final` :3850-3877)
  - partial 在 `:2545`(`stream.get_partial()`;**不要在 partial 上做标点/声纹**)
- **流式音频"段缓冲"——本次唯一的管线改动**:
  - 现状:`server/main.py:2498` `stream.accept_waveform()` 把音频喂 ASR 后**main.py 层不再保留 PCM**;VAD 在 `:2501-2543`(stream reset 在 :2543)。
  - 需加:收 chunk 时顺手 `seg += pcm`,**finalize 后 `seg.clear()`**。不是带淘汰的 ring buffer,就是"一句一攒、finalize 一清"。边界仅三处需清空:`reset`/force-endpoint、断连、每次 finalize 后。
- **后端工厂 / 模型加载框架**(可复用):
  - `server/core/asr_backend.py:178-183` `_ASR_REGISTRY`(profile→backend 映射) + `:191-226` `create_asr_backend()`(profile 驱动注册表)
  - `server/core/model_downloader.py:26-42` `MODELS` 注册表 + `:121-168` `ensure_models()`(CDN 下载/解压,启动即加载;三语模式 zh_en/en/shared + profile 路由)——可把标点/声纹模型登记进去走自动下载
  - profile 机制 `server/core/profile_loader.py:153` `current_profile()`(返回 `_CURRENT_PROFILE`)
- **现状 grep 确认(2026-06-09 复核仍成立)**:OVS 现在**没有任何**标点独立模块、没有 CAM++/3D-Speaker/diarization 代码;`punctuation` 关键字命中的都是文本分割逻辑(`v2v.py`、`trt_edge_llm_tts.py`),`speaker` 关键字唯一来源是 `server/core/tts_speakers.py`(TTS 发音人 preset + embedding 类型,与声纹无关)。**从零加**。
- **VAD feature-flag 惯例(照抄此 env默认+query覆盖 模式)**:
  - `server/main.py:717` `_default_vad_backend()`(:719 读 `OVS_VAD_BACKEND` env)
  - `asr_stream` 入参 `vad: Optional[str] = None`(:2272),初始化 `:2328` `vad_backend = vad if vad is not None else _default_vad_backend()`,创建 `:2346` `vad_mod.create_vad(...)`
  - **新增的 `?punctuate=` / `?speaker_embedding=` 两个开关照这套写**:连接入参 default=None → 缺省时取 env(`OVS_PUNCT`/`OVS_SPEAKER_EMB`)→ query 显式给值时覆盖。
- **最小客户端参考**:`tests/stream_test.py:20-56`(连 `ws://localhost:8621/asr/stream` 发 PCM、空字节 EOF、收 JSON)。

OVS 服务端口 **8621**,FastAPI+Uvicorn。流式输入契约:二进制 PCM16/16k/mono,空字节结束。

## 5. 要搬过来的两个模型(来自 sensecraft asr-service)

在 `/Users/harvest/project/sensecraft_voice/sensecraft-asr-service`(行号 2026-06-09 已核验):

- **标点**:CT-Transformer,`config.json:42-47`(嵌在 `recognition` 段) → `models/punctuation/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx`(路径在 :44)
- **声纹**:CAM++/3D-Speaker,`config.json:52-68`(`speaker` 段) → `models/speaker/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`(路径在 :54,`max_speakers: 10` 在 :62)。**注:192 维是模型架构属性,不是 config 字段**(OVS 侧契约里 `dim:192` 要硬写或从输出 shape 探)。注册参数(`max_speakers` 等)**不搬进 OVS**,只搬模型+提向量。
- **直拉 URL**(`download_models.sh` 已核验,全来自 HuggingFace `csukuangfj` / GitHub k2-fsa):
  - 标点(:59):`https://huggingface.co/csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/resolve/main/model.onnx`
  - 声纹(:48):`https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`
  - (参考)ASR SenseVoice(:32):`https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09/.../model.onnx`;VAD silero(:68):k2-fsa GitHub release
  - → 登记进 `model_downloader.py:26-42` `MODELS` 时,可直接用这两个 resolve URL(注意 OVS 设备多走 HF 镜像,见全局 `HF_ENDPOINT` 约定)。

> 提向量/标点可用 sherpa-onnx 的 Python API(`OfflinePunctuation`、`SpeakerEmbeddingExtractor`),但 **sherpa-onnx 不一定在 Jetson/RK 镜像里**。为跨设备一致,**优先用 onnxruntime 直接推这两个小 ONNX**,做成与 ASR 后端解耦的独立模块;确认各目标镜像都含 onnxruntime-CPU。

## 6. 使用端(voice-service 业务侧)责任 —— 本次不做,但要写进对接说明

- 收 final 的 `speaker_embedding` → 和注册库逐个余弦相似度 → `max>阈值 ? speaker_id : unknown` → 写 `recordings.speaker_id/speaker_name`
- 注册(enroll):注册音频也过 OVS 的 `POST /speaker/embedding` 取同源向量入库
- 阈值调参、未注册判定、"辨认 vs 盲分"取向 —— 都在使用端决定
- 待确认(不阻塞 OVS):voice-service 里 `speaker_id/speaker_name` 现在实际怎么填(从 `recordings` 表存 name 看大概率是**注册式辨认**,需要注册库)。可查 `sensecraft-voice-service/pkg/controller/recording/recording.go` 与 `pkg/db/model/recordings.go`。

## 7. 实施建议路线(降风险)

1. **先非流式独立端点**(`/punctuate`、`/speaker/embedding`)跑通,验证模型 + onnxruntime 跨设备(0.5-1.5 天)
2. **再折进流式**:标点内联(改 final 点,~0.5 天);声纹加段缓冲 + finalize 提向量(~1-1.5 天)
3. **开关 + 懒加载 + 跨硬件确认**(env 默认 off,query 覆盖)
4. 边界测试:多句连续、reset、断连、barge-in(v2v)下段缓冲正确清空

## 8. 旁支待办(不属本次,记录备查)

- asr-service 有未合并分支 `feature/origin/omi-remote`(commit `77d7b61`,2025-12-03):给 asr-service 加了 `internal/remote`(上报云端)+`internal/llm`+`internal/audio_cache`。半落地、与 voice-client 职责重叠,半年没动。需另行决定合并/废弃。
- sensecraft 各仓库 `config.yaml`/`.env.production` 有**明文凭据**入 git(MySQL/MinIO/Dify/OpenAI key)—— 安全债,已记在顶层 README「待改进项」。**本 handoff 不含任何凭据**。

---

## Suggested skills(下个 session 建议调用)

- **`dev-flow`** / 派 `Agent(subagent_type="general-purpose")`:实际写 OVS 端点代码 + build + 测试,按"主线程规划 → 派执行体实施 → 校验"流程。给执行体的 prompt 必须含护栏(禁破坏性操作、指定唯一构建入口、EVIDENCE 段)。
- **`edge-computing-optimize` / `device-gotchas`**:把标点/声纹 ONNX 跑到 RK/Jetson/RPi 时查对应平台坑(onnxruntime EP、ARM CPU 推理、HEF/RKNN 转换若需 NPU 加速)。
- **`voice-agent-e2e`**:给开了标点/声纹的流式链路搭无人值守端到端回归(scripted audio → WS event probe),验证 final payload 字段正确、开关 on/off 行为。
- **`fleet`**:需要在真实 RPi/RK/Jetson 上验证跨设备推理时,先 `fleet status` 找在线设备再部署。
- **`codex:codex-rescue`**:流式段缓冲 + VAD 生命周期若调试卡住(barge-in/reconnect 边界),派 codex 出第二意见。

## 关键路径索引

| 用途 | 路径 |
|---|---|
| OVS 仓库(本次主战场) | `/Users/harvest/project/seeed-local-voice` |
| OVS 端点入口 | `server/main.py`(:2201 /asr, :2257 /asr/stream, :2880 /v2v/stream) |
| OVS 模型框架 | `server/core/{asr_backend,model_downloader,profile_loader}.py` |
| 标点+声纹模型来源 | `/Users/harvest/project/sensecraft_voice/sensecraft-asr-service`(`config.json`, `download_models.sh`) |
| sensecraft 顶层 README | `/Users/harvest/project/sensecraft_voice/README.md` |
| 使用端(匹配责任) | `/Users/harvest/project/sensecraft_voice/sensecraft-voice-service` |
| Omi(参考,非本次) | `/Users/harvest/project/omi` |
