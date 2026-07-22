# ovs_agent 开发者示例 / Developer Examples

面向想基于 `ovs_agent` 框架写自己语音应用的开发者，三个单文件示例，
目标是**复制即改即跑**（copy → tweak → run）。

Three self-contained, copy-paste-friendly examples for developers building
their own voice apps on top of the `ovs_agent` framework.

| 示例 / Example | 演示什么 / What it shows | 需要 LLM? |
|---|---|---|
| [`minimal_echo_app.py`](minimal_echo_app.py) | 「20 行级」最小应用：subclass `BaseApp` + 覆写 `on_user_utterance`，把用户的话说回去。Minimal app: one hook, echo back what the user said. | 否 / No（`llm_backend: noop`） |
| [`voice_tools_app.py`](voice_tools_app.py) | 语音工具调用闭环：`@default_registry.tool` 注册工具 + `tools_enabled` / allowlist；说「现在几点」→ LLM 调 `time_now` → 语音播报。Voice tool-calling loop with a builtin + a custom tool. | 是 / Yes |
| [`custom_mode_app.py`](custom_mode_app.py) | 自定义 `AppMode`（复读机模式）注册进 `MultiModeApp`，用 `default_mode` / 内置 `set_mode` 工具切换。Custom AppMode registered into MultiModeApp with mode switching. | 否 / No |

## 前置条件 / Prerequisites

1. **一台在跑 SLV 语音服务的设备**（Jetson / RPi / RK / 本机），暴露
   `/v2v/stream` WebSocket（默认端口 8621）。所有示例通过
   `--slv-url` 参数或 `SLV_URL` 环境变量指向它。
   A running SLV voice service exposing `ws://<device>:8621/v2v/stream`.
2. **本机麦克风 + 扬声器**（示例在你开发机上采音/放音，音频经 WS 往返
   SLV 设备）。Local mic + speaker on the machine running the example.
3. 仅 `voice_tools_app.py` 需要：**一个 OpenAI 兼容的 LLM 服务**
   （edge-llm / vLLM / Ollama 均可），用 `--llm-base-url` / `LLM_BASE_URL`
   指向。Only the tools example needs an OpenAI-compatible LLM endpoint.

### 安装依赖 / Install

```bash
# 推荐：uv（仓库标准做法）
cd agent && uv sync

# 或 pip 可编辑安装 / or editable pip install
pip install -e agent/
```

## 运行方式 / How to run

### 方式 A：直接 python 跑（推荐上手）

示例文件自带 `__main__` 入口，在 `agent/` 的 uv 环境里直接跑：

```bash
cd agent
uv run python ../examples/agent/minimal_echo_app.py --slv-url ws://<device>:8621/v2v/stream

SLV_URL=ws://<device>:8621/v2v/stream \
LLM_BASE_URL=http://<device>:8000/v1 \
uv run python ../examples/agent/voice_tools_app.py

uv run python ../examples/agent/custom_mode_app.py --slv-url ws://<device>:8621/v2v/stream
```

### 方式 B：走 `ovs-agent run` CLI

CLI 按约定动态 import `ovs_agent.apps.<name>.app` 里的 `App` 符号
（见 `agent/ovs_agent/cli.py`），所以把示例复制进 apps 包即可：

```bash
mkdir -p agent/ovs_agent/apps/my_echo
touch agent/ovs_agent/apps/my_echo/__init__.py
cp examples/agent/minimal_echo_app.py agent/ovs_agent/apps/my_echo/app.py
# 可选：agent/ovs_agent/apps/my_echo/config.yaml 放 YAML 配置
cd agent && uv run ovs-agent run my_echo
```

每个示例文件头部注释里有对应的 `config.yaml` 字段清单。
Each example's header comment lists the YAML keys for CLI mode.

### 脱机自检 / Offline self-check

三个示例都支持 `--check`：只构造 Config + App（含工具/模式注册校验），
不连 SLV、不开麦克风，用于验证环境装好了：

```bash
cd agent
uv run python ../examples/agent/minimal_echo_app.py --check
uv run python ../examples/agent/voice_tools_app.py --check
uv run python ../examples/agent/custom_mode_app.py --check
```

## 常见坑 / Common pitfalls

- **macOS 麦克风权限**：首次运行时终端（Terminal/iTerm）会弹麦克风授权；
  拒绝过的话去 系统设置 → 隐私与安全性 → 麦克风 重新勾选，否则 ASR
  永远收不到声音（表现为说话无任何反应）。
  Grant microphone permission to your terminal on macOS.
- **SLV_URL 写错**：必须是 WebSocket 地址且带路径
  `ws://<host>:8621/v2v/stream`（不是 `http://`，不要漏 `/v2v/stream`）。
  连不上时先 `curl http://<host>:8621/healthz` 确认服务活着。
- **工具示例没配 LLM**：`voice_tools_app.py` 的工具循环由 LLM 驱动，
  LLM 端点不通时会走 15s 首 token 超时后报错 —— 先确认
  `curl <LLM_BASE_URL>/models` 可达。
- **听得到自己的回声**：开发机外放 + 无 AEC 的麦克风会让 TTS 播报再次
  触发 ASR（自己跟自己聊）。用耳机，或看 `Config` 里
  `mic_drop_while_speaking` / `energy_gate_*` 等选项。
- **`sounddevice` 导入报错**：需要 PortAudio（macOS wheel 自带；Linux 上
  `apt install libportaudio2`）。

## 进一步阅读 / Further reading

- `agent/README.md` — 框架总览与硬性约定（invariants）
- `docs/agent/tool-usage.md` — 工具调用完整指南（allowlist 语义、
  `ctx` 注入、async 工具、超时、response_mode）
- `agent/ovs_agent/app_mode.py` — `AppMode` / `ModeContext` / `ModeManager` 源码
- `agent/ovs_agent/apps/` — 更多真实 app（translator / live_caption /
  voice_arm / companion_robot …）
