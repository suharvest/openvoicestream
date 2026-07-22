"""语音工具调用闭环（Voice tool-calling loop）示例。

演示链路：说「现在几点」→ 流式 ASR → LLM 决定调用 `time_now` 工具 →
工具结果喂回 LLM → LLM 组织回答 → 流式 TTS 播报。

注册了两个工具（tools）：
  * time_now    —— 直接复用内置工具（agent/ovs_agent/tools/builtin.py），
                   import ovs_agent.tools 时已自动注册到 default_registry；
  * set_timer   —— 自定义玩具工具（假实现，只回话不真正计时），
                   演示 @tool 装饰器 + 类型注解自动派生 JSON Schema。

前置条件（Prerequisites）：
  * SLV 语音服务在跑（默认 ws://localhost:8621/v2v/stream）
  * 一个 OpenAI 兼容的 LLM 服务（edge-llm / vLLM / Ollama 等，
    默认 http://localhost:8000/v1）—— 工具调用必须有 LLM

运行方式（Two ways to run）：

  方式 A — 直接 python:
      cd agent && uv sync
      SLV_URL=ws://<device>:8621/v2v/stream \
      LLM_BASE_URL=http://<device>:8000/v1 \
      uv run python ../examples/agent/voice_tools_app.py

  方式 B — ovs-agent CLI（复制进 apps/ 包 + YAML 配置）:
      mkdir -p agent/ovs_agent/apps/my_tools
      touch agent/ovs_agent/apps/my_tools/__init__.py
      cp examples/agent/voice_tools_app.py agent/ovs_agent/apps/my_tools/app.py
      # 再写 agent/ovs_agent/apps/my_tools/config.yaml：
      #   slv_url: ws://<device>:8621/v2v/stream
      #   llm_base_url: http://<device>:8000/v1
      #   llm_model: qwen2.5-3b-instruct
      #   tools_enabled: true
      #   tools_default_allowlist: [time_now, set_timer]
      cd agent && uv run ovs-agent run my_tools

自检（dry-run）：
      uv run python ../examples/agent/voice_tools_app.py --check
"""
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from ovs_agent import Config
from ovs_agent.apps.multi_mode.app import MultiModeApp

# import ovs_agent.tools 的副作用：内置工具 time_now / set_mode 已注册进
# default_registry（见 agent/ovs_agent/tools/__init__.py 尾部的副作用 import）。
from ovs_agent.tools import default_registry


# ── 自定义工具：@tool 装饰器 ──────────────────────────────────────────
# 参数 JSON Schema 由类型注解自动派生（seconds: int → {"type":"integer"}，
# 无默认值 → required）。description（或函数 docstring）会成为 LLM 看到的
# 工具说明 —— 写清楚触发语义，LLM 才知道什么时候调它。
#
# response_mode 的选择（三种，见 agent/ovs_agent/tools/registry.py Tool 注释）：
#   * "await"（默认）  —— 等工具执行完，把结果喂给 LLM 第二轮，由 LLM 组织
#                        回答（能把 seconds 等参数复述进回答里）。适合
#                        查询类 / 需要 LLM 解读结果的工具。本例用这个。
#   * "parallel"       —— 工具体必须 ~200ms 内快速返回 {"started": True}，
#                        LLM 第二轮与真实副作用（机械臂运动等）并行。
#                        适合耗时的物理动作。
#   * "template"       —— 跳过 LLM 第二轮，直接播报注册时固定的
#                        completion_text。回复延迟最低，但内容固定。
#                        例：@default_registry.tool(response_mode="template",
#                            completion_text="好的，计时器已设置。")
@default_registry.tool(
    description=(
        "Set a countdown timer for the given number of seconds. "
        "设置一个倒计时，例如用户说“设一个十秒的计时器”。"
    ),
)
def set_timer(seconds: int) -> dict[str, Any]:
    """假实现（demo stub）：不真正计时，只把参数回给 LLM 让它播报确认。"""
    return {
        "success": True,
        "seconds": seconds,
        "note": "demo stub — no real timer was started",
    }


# MultiModeApp 自带 ChatMode：每句用户话音走
# ModeContext.run_default_dialogue_turn → stream_with_tools（LLM ↔ 工具循环，
# 见 agent/ovs_agent/app_mode.py）。工具调用能力挂在这条链路上，所以这里直接
# 复用 MultiModeApp，不需要写任何转发代码。
class App(MultiModeApp):
    pass


def build_config(slv_url: str, llm_base_url: str, llm_model: str) -> Config:
    return Config(
        slv_url=slv_url,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        # ── 工具调用开关（默认 False，纯聊天不受影响）──
        tools_enabled=True,
        # allowlist 语义（见 docs/agent/tool-usage.md）：
        #   非空列表 → 只暴露列出的工具；空列表/不设 → 暴露所有已注册工具。
        # 显式列出更安全：内置的 set_mode 等不会被误触发。
        tools_default_allowlist=["time_now", "set_timer"],
        # 每轮用户话音最多几次 LLM ↔ 工具往返（防失控）。
        tools_max_iterations=5,
        system_prompt=(
            "你是一个简洁的中文语音助手。可以调用工具查询当前时间、设置计时器。"
            "回答要短，适合语音播报。"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--slv-url",
        default=os.environ.get("SLV_URL", "ws://localhost:8621/v2v/stream"),
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("LLM_MODEL", "qwen2.5-3b-instruct"),
    )
    parser.add_argument("--check", action="store_true", help="只构造不运行")
    args = parser.parse_args()

    cfg = build_config(args.slv_url, args.llm_base_url, args.llm_model)
    app = App(cfg)
    if args.check:
        names = default_registry.list_names()
        assert "time_now" in names and "set_timer" in names, names
        print(f"OK: tools registered = {names}")
        return 0

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
