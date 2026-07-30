# HomeAssistantApp — 用语音控制 Home Assistant

把这套语音栈接到已有的 Home Assistant，让用户可以直接说「把客厅的灯调暗一点」。
这个 app 同时是**可跑的示例**和**你自己扩展的模板** —— 加自己的设备/服务，就是往
`ha_tools.py` 里加一个 `@tool` 函数。

## 它在整套系统里的位置

```
  麦克风 ──┐
           │  ws://voice:8621/v2v/stream
   ┌───────▼────────────┐   tool_advertise / tool_call / tool_result
   │  这个 app          │◄──────────────────────────────┐
   │  · 麦克风 / 扬声器  │                               │
   │  · HA 工具实现      │──── HTTP ───► Home Assistant  │
   └────────────────────┘      REST API                 │
                                                        │
   ┌────────────────────────────────────────────────────▼──┐
   │  语音服务 (:8621)  ASR + TTS + 对话编排（含 LLM 工具循环）│
   └───────────────────────┬───────────────────────────────┘
                  ┌────────▼─────────┐
                  │  LLM (:1828)     │
                  └──────────────────┘
```

**这个 app 是客户端。** 它持麦克风，连上语音服务后把工具的 schema
`tool_advertise` 上去；服务端 LLM 决定要开灯时，通过 `tool_call` 把执行**代理回这里**，
由这里去调 HA 的 REST API，再用 `tool_result` 把结果送回去。

**这个设计的好处**：LLM 从来不需要你的 HA 令牌，也不需要能访问你的家庭网络。
凭据和网络访问都留在设备本地。

## 快速开始

### 1. 拿一个 Home Assistant 长效令牌

HA 网页 → 左下角头像 → **安全** → **长期访问令牌** → 创建令牌。复制保存（只显示一次）。

### 2. 确认 HA 地址「从设备上」能通

这是最常见的坑：填了一个只在你笔记本上能解析的地址。**在跑 agent 的那台设备上**验：

```bash
curl -sI http://<你的HA地址>:8123/    # 期望 200 或 405，不是 timeout
```

### 3. 配置

令牌走环境变量，不要写进文件：

```bash
export HA_BASE_URL=http://192.168.1.10:8123
export HA_TOKEN=<你的长效令牌>
```

其余在 `config.yaml` 的 `metadata:` 段。**注意必须放在 `metadata` 下面** ——
`load_config` 会**静默丢弃**未知的顶层 key（只打一条日志），所以顶层写
`ha_base_url` 会无声无息地失效。

### 4. 服务端必须开 server loop

语音服务那侧需要（**两个缺一不可**）：

```
OVS_V2V_ENGINE=voxedge
OVS_V2V_SERVER_LOOP=1
EDGE_LLM_BASE_URL=http://127.0.0.1:1828/v1
EDGE_LLM_MODEL=Qwen3-4B
```

没开 server loop 的话，`tool_advertise` 会被**静默忽略**（服务端只打一条 warning，
不回错误帧），表现就是「语音能听懂，但从不控制任何设备」。

## 内置工具

| 工具 | 说明 |
|---|---|
| `list_devices()` | 可控设备清单（名称/类型/状态）。LLM 遇到没见过的设备名时先调这个 |
| `turn_on(device)` | 开。窗帘→打开，门锁→解锁 |
| `turn_off(device)` | 关。窗帘→关闭，门锁→上锁 |
| `set_brightness(device, percent)` | 灯亮度 0–100 |
| `set_cover_position(device, percent)` | 窗帘开合 0–100 |
| `get_state(device)` | 读状态 |
| `call_service(domain, service, entity_id, data_json)` | 逃生口：直接调任意 HA service |

## 加你自己的工具

```python
# ha_tools.py
@_r.tool(
    description="启动扫地机器人清扫指定房间。room 是房间名，例如 客厅。",
    preamble_text="好的，开始打扫。",   # 工具一开始执行就先出声
    response_mode="template",
    completion_text="已经开始打扫了。",  # 跳过 LLM 第二轮，延迟最低
)
def start_vacuum(room: str) -> dict:
    """room 用中文房间名。"""
    try:
        ha = _ha()
        d = ha.resolve(room, domains=("vacuum",))
    except ResolveError as e:
        return _fail(str(e), e.candidates)      # 让 LLM 拿着 candidates 重试
    try:
        ha.call_service("vacuum", "start", {"entity_id": d.entity_id})
    except Exception as e:
        return _fail(f"启动 {d.name} 失败: {e}")
    return {"ok": True, "device": d.name}
```

要点：

- **schema 是从类型标注自动生成的**，不用手写 JSON Schema。支持
  `str/int/float/bool/list/dict/Literal[...]/Optional[T]`；未知类型会静默降级成 string
- **永远返回 dict，永远不要抛异常**。抛出去就是一个不透明的失败；返回
  `{"ok": False, "error": ...}` 才能让 LLM 说点有用的、并且能重试
- `description` 是**给 LLM 看的**，写清楚参数含义和取值范围
- 加了新工具要在 `__all__` 里带上

### `response_mode` 怎么选

| 值 | 行为 | 用在 |
|---|---|---|
| `await`（默认） | 等结果，再让 LLM 说第二轮 | 查询类（`get_state`、`list_devices`） |
| `template` | 跳过 LLM 第二轮，直接念 `completion_text`。**延迟最低** | 开关类（结果只有成功/失败，不需要 LLM 组织语言） |
| `parallel` | 函数须 ~200ms 内返回，LLM 第二轮和副作用并行 | 耗时的物理动作 |

`preamble_text` 对外部 API 特别值得设 —— 不设的话用户说完话会先听到一段静音，
会以为没被听见。

## 设备名怎么匹配的（重要）

**Home Assistant 会把中文实体名转写成拼音 `entity_id`**：

```
客厅灯   → light.ke_ting_deng
客厅窗帘 → cover.ke_ting_chuang_lian
卧室空调 → switch.wo_shi_kong_diao
```

所以**永远不要靠解析 `entity_id` 判断设备是什么** —— 唯一可靠的人类可读标签是
`friendly_name` 属性。`ha_client.py` 的匹配全部基于它。

匹配分四档，每档只在**唯一命中**时才算成功，否则继续往下：

1. 原样 `entity_id`（LLM 可以把 `list_devices` 的结果直接传回来）
2. 归一化后精确匹配 —— 会去掉「的/把/请/帮我」这类口语填充词，所以
   **「把客厅的灯」能匹配到「客厅灯」**
3. 双向包含 —— 「客厅灯光」⊃「客厅灯」，「客厅」⊂「客厅灯」
4. 字符重叠度打分，且必须明显领先第二名

**匹配不到或有歧义时不会瞎猜**，而是返回 candidates：

```json
{"ok": false, "error": "'客厅' 匹配到多个设备，需要说得更具体",
 "candidates": ["客厅灯", "客厅电视", "客厅窗帘"]}
```

系统提示词里已经写了「如果工具返回了 candidates，用那些名字向用户确认」。
**关错房间的灯比多问一句糟糕得多。**

### 哪些实体会被暴露

默认只暴露 `light / switch / fan / cover / climate / media_player / lock / scene / script`。

**`input_boolean` 默认排除** —— 模板灯和模板开关通常由一个 `input_boolean` 支撑，
包含这个域会让每个设备**出现两次**（一次 `light.x`，一次背后的 helper），LLM 就有两个
同样合理的目标。确实要直接控 helper（比如「假期模式」开关）的话：

```yaml
metadata:
  ha_extra_domains: ["input_boolean"]
```

另外自动过滤掉：状态为 `unavailable`/`unknown` 的、带 `entity_category`
的（那些是设置界面用的配置/诊断实体）。要藏特定实体用
`ha_exclude_entity_ids`。

## 五个会绊到人的坑

| 坑 | 说明 |
|---|---|
| **`httpx` 必须 `trust_env=False`** | 否则会读取环境里的 `HTTP(S)_PROXY`，把局域网地址也走代理，够不到 HA。`ha_client.py` 里已经设了 —— 你自己加的 HTTP 调用也要设 |
| **必须压在 15 秒内** | per-tool 的 `timeout_s` **不会**通过 advertise 传到服务端，所以真正生效的是服务端的 ~15s 默认值。客户端自己默认 30s，**会超过服务端的耐心** |
| **字段是 `ok` 不是 `success`** | 帧级字段名 |
| **`call_id` 要先读 `call_id` 再读 `id`** | 两个键都会被接受，但先读 `id` 曾导致 15 秒卡死 |
| **没开 server loop 时 advertise 静默失效** | 只打 warning，不回错误帧。症状是「能听懂但从不控制设备」 |

## 排查

| 症状 | 原因 |
|---|---|
| 日志 `[ha] cannot reach ...` | URL 从设备上不通，或令牌过期。用上面第 2 步的 curl 验 |
| 日志 `[ha] ha_base_url / ha_token not configured` | 环境变量没传进容器，或写在了 `metadata` 之外 |
| 能对话但从不调工具 | 服务端 `OVS_V2V_SERVER_LOOP` 没开；或 `tools_enabled` 不是 true（框架默认 false） |
| LLM 老是挑错设备 | 实体的 `friendly_name` 有歧义。在 HA 里改名，或用 `ha_exclude_entity_ids` 收窄 |
| 回复很长、念出符号 | 服务端 `OVS_V2V_SYSTEM_PROMPT` 没设。不设的话模型会按 chat 助手风格回答 Markdown |

## 已验证范围

工具层已对**真实 Home Assistant 实例**逐项验证（6 个可控实体，覆盖
light/switch/cover 三个域，中文名 + 拼音 entity_id）：

- `list_devices` / `turn_on` / `turn_off` / `set_brightness` /
  `set_cover_position` / `get_state` / `call_service` 全部真实改变了设备状态
- 三条错误路径行为正确：歧义名返回 candidates、不存在的设备返回 candidates、
  超范围参数被挡下

**尚未端到端联跑**：从麦克风到 HA 的完整语音链路依赖已部署的语音服务 + LLM。
参考链路实测数据：嘴到耳 p50 878–895 ms（ASR 340 + LLM 277 + 合成 260），
LLM TTFT 132 ms / 81.5 tok/s @ 8K 上下文。
