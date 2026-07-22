"""自定义 AppMode（custom mode）示例 —— 复读机模式（Parrot mode）。

AppMode 是策略模式（Strategy pattern）扩展点：BaseApp/MultiModeApp 拥有
管线（mic → ASR → 事件分发 → TTS 播放），而「一句用户话音进来之后做什么」
由当前激活的 AppMode 决定。内置模式有 chat / interpreter / monologue /
transcribe（见 agent/ovs_agent/modes/），本例再注册一个自己的 ParrotMode。

Mode 与 App 的区别（何时用哪个）：
  * 只想改「每句话的处理逻辑」且希望能在运行时切来切去 → 写 AppMode；
  * 想改管线本身（接自定义硬件、改状态机）→ subclass BaseApp。

mode 切换机制（两条路）：
  1. 启动默认：Config.default_mode 指定开机进入哪个模式（本例用这条）；
  2. 运行时切换：开启工具调用后，LLM 可调内置工具 `set_mode`
     （agent/ovs_agent/tools/builtin.py）按用户口头指令切模式，
     也可通过 dashboard / 代码里 `app.modes.switch("parrot")` 切。

运行方式（Two ways to run）：

  方式 A — 直接 python:
      cd agent && uv sync
      uv run python ../examples/agent/custom_mode_app.py \
          --slv-url ws://<device>:8621/v2v/stream

  方式 B — ovs-agent CLI（复制进 apps/ 包）:
      mkdir -p agent/ovs_agent/apps/my_parrot
      touch agent/ovs_agent/apps/my_parrot/__init__.py
      cp examples/agent/custom_mode_app.py agent/ovs_agent/apps/my_parrot/app.py
      # agent/ovs_agent/apps/my_parrot/config.yaml 里写：
      #   slv_url: ws://<device>:8621/v2v/stream
      #   llm_backend: noop
      #   default_mode: parrot
      cd agent && uv run ovs-agent run my_parrot

自检（dry-run）：
      uv run python ../examples/agent/custom_mode_app.py --check
"""
from __future__ import annotations

import argparse
import asyncio
import os

from ovs_agent import Config
from ovs_agent.app_mode import AppMode, ModeContext
from ovs_agent.apps.multi_mode.app import MultiModeApp


class ParrotMode(AppMode):
    """复读机：把用户的话原样说回去，不经过 LLM。

    AppMode 子类唯一必须实现的是 on_user_utterance；其余 hook
    （enter / exit / on_assistant_done / preprocess_user_text）都有
    no-op 默认实现，按需覆写（见 agent/ovs_agent/app_mode.py:AppMode）。

    类属性是声明式默认值，部署时可用 config.mode_overrides["parrot"]
    覆盖（如换 system_prompt / temperature —— 本模式不用 LLM，用不上）。
    """

    # name 是注册/切换用的 key（set_mode / default_mode 都认它），必填。
    name = "parrot"
    display_name = "复读机"
    icon = "🦜"
    description = "把用户说的话原样复述一遍，不调用 LLM"

    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        """Mode 版本的 hook：多了 ctx（依赖注入包，每次调用新建）。

        ctx 上有 config / slv / llm / session / translator / mode_manager 等
        （见 ModeContext dataclass）。两个常用的高层方法：
          * ctx.speak(text)  —— 直接把现成文本送 TTS（不走 LLM、不进对话
            历史）。复读 / 模板播报用它。
          * ctx.run_default_dialogue_turn(text) —— 标准 LLM 对话轮
            （ChatMode 就是这一行）。
        """
        await ctx.speak(f"你说：{text}")

    async def enter(self, ctx: ModeContext) -> None:
        """切入本模式时回调一次（可用来播报提示音/初始化资源）。"""
        # 这里保持安静；如需切换提示可: await ctx.speak("复读机模式已开启。")


class App(MultiModeApp):
    """在 MultiModeApp 的内置模式（chat/interpreter/...）之外追加自定义模式。

    注册后 ParrotMode 会出现在 mode 列表里（dashboard 可见），
    并可被内置工具 set_mode 或 config.default_mode 选中。
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.modes.register(ParrotMode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--slv-url",
        default=os.environ.get("SLV_URL", "ws://localhost:8621/v2v/stream"),
    )
    parser.add_argument("--check", action="store_true", help="只构造不运行")
    args = parser.parse_args()

    cfg = Config(
        slv_url=args.slv_url,
        # 复读机不需要 LLM。注意：这样内置 chat 模式将不可用 —— 想同时
        # 保留 chat，请配置真实的 llm_backend/llm_base_url 而不是 noop。
        llm_backend="noop",
        # 开机直接进入自定义模式（MultiModeApp.run 里调 modes.start(default_mode)）。
        default_mode="parrot",
    )
    app = App(cfg)
    if args.check:
        assert app.modes.get("parrot") is not None
        print("OK: ParrotMode registered; default_mode =", cfg.default_mode)
        return 0

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
