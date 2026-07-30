# 智能家居语音助手 · 部署与开发指南（RK3588 + Home Assistant）

在 Radxa ROCK 5T（RK3588）上跑一套**全本地**的语音助手，用自然语言控制已有的
Home Assistant。ASR、TTS、LLM 全部在设备上，不依赖云端。

**实测性能**（真机，中文短句，n=5）：

| 指标 | 值 |
|---|---|
| **停止说话 → 听到第一声（嘴到耳）p50** | **845–895 ms** |
| ├ ASR 出终稿 | ~300–340 ms |
| ├ LLM 到第一个小句 | ~277 ms |
| └ 首句合成 | ~260 ms |
| LLM 吞吐（Qwen3-4B @ 8K 上下文） | TTFT 132 ms / 81.5 tok/s |

---

## 0. 硬件与前置条件

| | |
|---|---|
| 主板 | Radxa ROCK 5T 或同等 RK3588，**16 GB 内存**（8 GB 不够） |
| **本地 LLM** | **需要 RK1828 / RM182X M.2 加速卡**。没有这张卡，ASR 和 TTS 照常工作，但 LLM 必须指向外部（局域网另一台机器，或任何 OpenAI 兼容的云端） |
| 麦克风 / 扬声器 | 任何 ALSA 能识别的设备 |
| 磁盘 | 首次启动要拉约 4 GB 模型，**预留 10 GB 以上** |
| Home Assistant | 你已有的实例，**从这台设备网络可达** |

### 0.1 RK1828 加速卡：宿主机准备（镜像不是自包含的）

加速卡的**内核驱动和固件在宿主机上**，容器只能访问一张已被宿主机初始化好的卡。
仓库里带了核验脚本：

```bash
sudo deploy/scripts/rk1828-host-bringup.sh --check    # 只读核验
sudo deploy/scripts/rk1828-host-bringup.sh            # 顺手补上模块加载与持久化
```

9 项全绿才继续。脚本**不会**编译内核模块（那需要 RM182X SDK + 内核头文件，见
`services/rk1828-llm/BUILD.md`），缺了会明确告诉你。

三条会咬人的事，脚本里也写了：

- **模块加载必须持久化**，否则重启后卡就消失，LLM 服务启动失败且看不出原因
- 卡有**独立 12V 供电**。漆黑 + 风扇不转 = 没通电；主电源口和风扇口紧挨着，容易接错
- **绝不要跑 `rknn-smi reset`** —— 会把卡推进一个连宿主机重启都可能救不回的状态，而且卡不随宿主机断电

### 0.2 单卡独占：一次只能驻一个大模型

RK1828 只有**一个约 5 GB 的上下文**。Qwen3-4B @ 8192 token 估算占用约 3.6 GB，
所以**这张卡给了 LLM，就不能同时跑别的大模型**（比如 RK1828 版的 TTS）。
本方案的 TTS 跑在 RK3588 自己的 NPU 上，正是为此。

> ⚠️ 这张卡的**显存和健康状态没有任何观测手段** —— `rknn-smi` 在该平台整机失效
> （root 也失败、卡空闲时也失败，疑为出厂的 host/EP 固件版本不匹配）。唯一的信号
> 是"模型加载成不成"。而**失败的加载会把卡从 8 核退化到 4 核**，所以不要靠反复
> 试加载来探容量。

---

## 1. 拿到 Home Assistant 的长效令牌

HA 网页 → 左下角头像 → **安全** → **长期访问令牌** → 创建。**只显示一次**，复制保存。

然后**在这台设备上**（不是你的笔记本上）验证地址可达 —— 这是最常见的失败原因：

```bash
curl -sI http://<你的HA地址>:8123/    # 期望 200 或 405，不是 timeout
```

---

## 2. 启动

```bash
cd <仓库根目录>
cat > deploy/.env <<'EOF'
HA_BASE_URL=http://192.168.1.10:8123
HA_TOKEN=<你的长效令牌>
EOF

docker compose -f deploy/docker-compose.radxa-ha.yml up -d
```

三个服务：

| 服务 | 端口 | 干什么 |
|---|---|---|
| `llm` | 1828 | Qwen3-4B 在 RK1828 卡上，OpenAI 兼容接口 |
| `speech` | 8621 | Qwen3-ASR + matcha TTS 在 RK3588 NPU 上，对话编排 |
| `agent` | — | 麦克风/扬声器客户端，执行 Home Assistant 控制 |

**任何镜像里都没有模型。** 每个服务首次启动时按自己的配置拉取到 named volume，
所以第一次启动会久（LLM 要拉约 3.2 GB，健康检查的宽限期设了 900 秒）。

进度观察：

```bash
docker compose -f deploy/docker-compose.radxa-ha.yml logs -f llm      # 下载 + 加载
curl -s http://127.0.0.1:1828/health                                   # LLM 就绪
curl -s http://127.0.0.1:8621/readyz                                   # 语音服务就绪
docker logs ovs-agent --tail 30                                        # 看 HA 连上没有
```

agent 日志里应该出现 `[ha] connected to http://… — N controllable devices`。

如果国内网络拉不动 HF，在 `.env` 里加 `HF_ENDPOINT=https://hf-mirror.com`（镜像站
可能滞后于源站）。

---

## 3. 验证

对着麦克风说：**"客厅灯打开"**。预期：约 0.9 秒后听到"好的"，灯亮。

分段排查：

```bash
# LLM 单独测（TTFT / 吞吐）
python3 - <<'EOF'
import json,urllib.request
r=urllib.request.urlopen(urllib.request.Request(
  "http://127.0.0.1:1828/v1/chat/completions",
  data=json.dumps({"messages":[{"role":"user","content":"你好"}],"max_tokens":32}).encode(),
  headers={"Content-Type":"application/json"}))
print(json.load(r)["choices"][0]["message"]["content"])
EOF

# 全链路延迟（仓库自带）
uv run --with websocket-client --with soundfile --with numpy \
  python bench/perf/measure_v2v_unified.py --host 127.0.0.1:8621 \
  --wav bench/perf/corpus/short/zh_short_01.wav --language Chinese --runs 5 --server-loop
```

---

## 4. 加你自己的设备控制工具

这是**主要的扩展点**。完整说明在
`agent/ovs_agent/apps/home_assistant/README.md`，这里只给要点。

内置七个工具：`list_devices` / `turn_on` / `turn_off` / `set_brightness` /
`set_cover_position` / `get_state` / `call_service`（通用逃生口）。

加一个新工具 = 往 `ha_tools.py` 加一个函数：

```python
@_r.tool(
    description="启动扫地机器人清扫。room 是房间名，例如 客厅。",
    preamble_text="好的，开始打扫。",   # 工具一开始执行就先出声
    response_mode="template",
    completion_text="已经开始打扫了。",  # 跳过 LLM 第二轮，延迟最低
)
def start_vacuum(room: str) -> dict:
    try:
        ha = _ha()
        d = ha.resolve(room, domains=("vacuum",))
    except ResolveError as e:
        return _fail(str(e), e.candidates)     # 让 LLM 拿 candidates 重试
    try:
        ha.call_service("vacuum", "start", {"entity_id": d.entity_id})
    except Exception as e:
        return _fail(f"启动 {d.name} 失败: {e}")
    return {"ok": True, "device": d.name}
```

**schema 从类型标注自动生成**，不用手写 JSON Schema。改完重启 `agent` 即可：

```bash
docker compose -f deploy/docker-compose.radxa-ha.yml up -d --build agent
```

### 关于设备名（值得单独看）

**Home Assistant 会把非拉丁实体名转写成拼音 `entity_id`**：

```
客厅灯   → light.ke_ting_deng
客厅窗帘 → cover.ke_ting_chuang_lian
```

所以**不要靠解析 `entity_id` 判断设备是什么** —— 唯一可靠的人类可读标签是
`friendly_name`。框架的匹配全部基于它，并且做了口语容错（`把客厅的灯` 能匹配到
`客厅灯`）。匹配不到或有歧义时**不会瞎猜**，而是返回 candidates 让 LLM 反问 ——
关错房间的灯比多问一句糟糕得多。

想让效果好，**在 HA 里把实体名起清楚**（"客厅主灯"而不是"Light 1"）比改代码有效得多。

---

## 5. 排查

| 症状 | 原因 |
|---|---|
| agent 日志 `[ha] cannot reach ...` | 地址从**设备上**不通，或令牌过期。用 §1 的 curl 验 |
| agent 日志 `[ha] ha_base_url / ha_token not configured` | `.env` 没被读到，或变量名写错 |
| 能对话但从不控制设备 | `speech` 的 `OVS_V2V_SERVER_LOOP` 没开。这个和 `OVS_V2V_ENGINE` **缺一不可**，少一个就退化成"只会复述的语音服务" |
| LLM 容器起不来，日志 `no /dev/pcie-rkep-*` | 宿主机没准备好，回到 §0.1 |
| LLM 容器 `MODEL_SETUP fail` / `ACK_FAIL` | 卡被别的大模型占着（检查 `systemctl is-active tts-radxa`），或上下文设得太大。**最多重试两次**，之后干净重启宿主机 —— 反复失败会让卡退化 |
| 回复很长、把符号念出来 | `OVS_V2V_SYSTEM_PROMPT` 被覆盖成空了。不设的话模型会按聊天助手风格输出 Markdown（实测：一个短问句能合成出 **87 秒**音频） |
| 首次启动很久没反应 | 在拉 4 GB 模型。`docker logs -f ovs-llm` 看进度 |

---

## 6. 调优

| 想要 | 怎么做 |
|---|---|
| 更长的对话上下文 | `.env` 里 `RK1828_MAX_CONTEXT=16384`。8192 是**实测过**的；16384 估算占卡的 92%，**未验证**，失败就退回 8192（且只试一次） |
| 更短的回复 | 降 `OVS_V2V_LLM_MAX_TOKENS`（默认 96） |
| 更快的断句 | 降 `VAD_ENDPOINT_SILENCE_MS`（默认 400）。太低会把停顿误判成说完 |
| 换 LLM | 改 `EDGE_LLM_BASE_URL` 指向任何 OpenAI 兼容端点，并相应改 `EDGE_LLM_MODEL`。不需要 RK1828 卡 |
| 换音色 | `TTS_DEFAULT_SID`；可用音色见 `GET /tts/speakers` |

**不要动的两个**：

- `MATCHA_USE_ORT=1` —— 这是 RK3588 上**唯一验证过闭环**的配方，改成 0 走全 RKNN
  路径会让 TTS→ASR 闭环失败
- `ASR_NPU_CORE_MASK=NPU_CORE_1` —— matcha 把自己的 Vocos 钉在 core 0 并给 ASR
  编码器留了 core 1。改成 `NPU_CORE_AUTO` 会**静默作废**这个隔离，让两个共驻的
  RKNN 上下文可能落到同一个核

---

## 7. 已验证到什么程度

**已真机验证**：
- 全链路延迟与并发安全性（ASR + TTS 共驻 40 分钟零 RKNN 故障；说话打断真重叠 3/3
  通过，打断句仍正确识别）
- 七个 HA 工具逐个对真实 Home Assistant 实例生效（light / switch / cover 三个域，
  中文名 + 拼音 entity_id），以及三条错误路径（歧义名 / 不存在的设备 / 超范围参数）
- 宿主机 bring-up 脚本 9 项检查

**尚未验证**：从麦克风到 HA 的完整链路只在分段层面验证过，未在一台装好麦克风的
整机上连续跑过。首次整机联调时优先看 `docker logs ovs-agent`。

---

## 8. 另一条路：Wyoming（兼容选项）

如果你已经重度使用 HA 的 Assist 和自动化，不希望把"大脑"交给我们，我们也在做
**Wyoming 协议适配** —— 把我们的 ASR/TTS 直接作为 HA Assist 的语音服务，
意图理解仍由 HA 负责，你现有的自动化全部保留。

这是**兼容选项，不是替代**。取舍：TTS 侧我们的流式优势能完整保留（首音延迟主要由
这一段决定），但 ASR 侧的流式中间结果 HA 当前不消费。适合"自动化是资产"的用户；
想要自由对话则走本文档这条路。
