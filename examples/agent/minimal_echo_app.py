"""最小语音应用（Minimal voice app）— 复述用户说的话。

「20 行级」核心：subclass BaseApp + 覆写一个 hook。麦克风采集（mic pump）、
SLV WebSocket 连接、VAD、TTS 播放全部由 BaseApp 托管，你只需要决定
「听到一句话之后做什么」。

运行方式（两种任选其一，Two ways to run）：

  方式 A — 直接用 python 跑（推荐上手）:
      cd agent && uv sync          # 首次安装依赖（installs deps once）
      uv run python ../examples/agent/minimal_echo_app.py \
          --slv-url ws://<device>:8621/v2v/stream

  方式 B — 走 ovs-agent CLI（把文件复制进 apps/ 包）:
      mkdir -p agent/ovs_agent/apps/my_echo
      touch agent/ovs_agent/apps/my_echo/__init__.py
      cp examples/agent/minimal_echo_app.py agent/ovs_agent/apps/my_echo/app.py
      # CLI 动态 import `ovs_agent.apps.<name>.app` 并找 `App` 符号
      # （见 agent/ovs_agent/cli.py:_load_app_class）
      cd agent && uv run ovs-agent run my_echo

不改代码自检（dry-run，无需真实 SLV 服务）:
      uv run python ../examples/agent/minimal_echo_app.py --check
"""
from __future__ import annotations

import argparse
import asyncio
import os

# ovs_agent 顶层就 re-export 了 BaseApp / Config（见 agent/ovs_agent/__init__.py）
from ovs_agent import BaseApp, Config


class EchoApp(BaseApp):
    """听到什么就说什么（加个前缀）的回声应用。

    BaseApp 已经替你做完的事（What BaseApp does for you）:
      * 持久 WebSocket 连到 SLV `/v2v/stream`（multi_utterance，跨轮不断连）
      * mic pump：后台 task 持续把麦克风 PCM 推给 SLV 做流式 ASR
      * VAD / 端点检测：一句话说完（静音）后 SLV 回一个 `asr_final` 事件
      * TTS 音频回传：SLV 合成的 PCM 自动送本机扬声器播放
      * barge-in：你说话时自动打断正在播放的 TTS

    你唯一要实现的 hook 就是 on_user_utterance —— 它在每个 `asr_final`
    （用户一句完整话）到达时被调用一次。
    """

    async def on_user_utterance(
        self, text: str, detected_language: str | None = None
    ) -> None:
        """每句用户话音的处理入口（hook 时机：ASR final 之后）。

        签名必须与 BaseApp.on_user_utterance 一致 —— 框架调用时会传
        detected_language 关键字参数（ASR 识别出的语言名，如 "Chinese"；
        后端不支持语言识别时为 None）。
        """
        # send_text: 把文本流式推给 SLV；SLV 在服务端做分句 + TTS 合成。
        await self.slv.send_text(f"你刚才说：{text}")
        # flush_tts: 本轮文本发完，让 SLV 把缓冲里不足一句的尾巴也合成掉。
        await self.slv.flush_tts()


# ovs-agent CLI 按约定查找模块里的 `App` 符号（方式 B 需要这个别名）。
App = EchoApp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--slv-url",
        default=os.environ.get("SLV_URL", "ws://localhost:8621/v2v/stream"),
        help="SLV /v2v/stream WebSocket 地址（也可用环境变量 SLV_URL）",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只做配置 + 应用构造自检，不连接 SLV / 不开麦克风",
    )
    args = parser.parse_args()

    # 回声应用不需要 LLM —— 用 noop 后端跳过 LLM 初始化
    # （合法取值见 agent/ovs_agent/app_base.py:_build_llm）。
    cfg = Config(slv_url=args.slv_url, llm_backend="noop")

    app = EchoApp(cfg)
    if args.check:
        print(f"OK: EchoApp constructed (slv_url={cfg.slv_url})")
        return 0

    try:
        # run() 里完成：连 SLV → 起 mic pump → 事件分发循环（Ctrl-C 退出）。
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
