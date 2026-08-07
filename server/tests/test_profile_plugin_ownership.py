"""老 profile 必须能压过镜像烘焙的 EDGE_LLM_ASR_PLUGIN_PATH。

背景（2026-08-07 实测）：v091 运行时镜像把 EDGE_LLM_ASR_PLUGIN_PATH 写成镜像
ENV，指向 /opt/edgellm-v091 那份 31MB 插件。而 EDGE_LLM_ 是 operator 前缀，
profile_loader 把镜像 ENV 视为 operator 所有、永不覆盖，于是 7 个上一代
profile 声明的 /opt/edgellm-bin 插件（70MB，与它们那代引擎配套）根本没生效。
TensorRT 的 engine plan 按名字+版本查 plugin creator，版本不匹配 → ASR worker
静默退出（stderr 为空），表现为 "ASR worker failed to start: "。

修法是让这些 profile 用现成的 profile_owned_env 机制声明拥有该键。

破坏链路验证：把任一 profile 的 profile_owned_env 去掉后，
test_legacy_profiles_own_asr_plugin_path 会失败。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

PROFILES_DIR = Path(__file__).resolve().parents[2] / "configs" / "profiles"

# 上一代（非 v091）且使用 trt_edge_llm ASR 的 profile —— 它们的插件路径必须
# 压过镜像默认值才能配上自己那代的引擎。
LEGACY_TRT_ASR_PROFILES = [
    "jetson-qwen3asr-matcha-nx",
    "jetson-qwen3asr-matcha",
    "jetson-qwen3-composition-nx",
    "jetson-multilang-official",
    "jetson-multilang-highperf",
    "jetson-multilang-highperf-nx",
    "jetson-qwen3asr-moss-nx",
]


def _load(name: str) -> dict:
    return json.loads((PROFILES_DIR / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", LEGACY_TRT_ASR_PROFILES)
def test_legacy_profiles_own_asr_plugin_path(name: str) -> None:
    prof = _load(name)
    env = prof.get("env") or {}
    assert "EDGE_LLM_ASR_PLUGIN_PATH" in env, f"{name} 未声明 ASR 插件路径"
    owned = set(prof.get("profile_owned_env") or [])
    assert "EDGE_LLM_ASR_PLUGIN_PATH" in owned, (
        f"{name} 声明了 EDGE_LLM_ASR_PLUGIN_PATH 却没列入 profile_owned_env；"
        f"EDGE_LLM_ 是 operator 前缀，镜像 ENV 会永久压过它"
    )


def test_owned_keys_must_exist_in_env() -> None:
    """profile_owned_env 只能列自己确实提供了值的键 —— 否则会把该键从
    operator 保护里摘出来却不给替代值，等于凭空清掉一个生效配置。"""
    bad = []
    for path in PROFILES_DIR.glob("*.json"):
        prof = json.loads(path.read_text(encoding="utf-8"))
        env = prof.get("env") or {}
        for key in prof.get("profile_owned_env") or []:
            if key not in env:
                bad.append(f"{path.name}: owned={key} 但 env 里没有该键")
    assert not bad, "\n".join(bad)


def test_profile_owned_env_actually_overrides_image_default(monkeypatch) -> None:
    """端到端：模拟镜像 ENV 已设 v091 插件，加载老 profile 后须变成老插件。"""
    image_default = "/opt/edgellm-v091/libNvInfer_edgellm_plugin.so"
    monkeypatch.setenv("EDGE_LLM_ASR_PLUGIN_PATH", image_default)

    # operator 快照在 import 时取，必须在设好 env 之后重新加载模块
    import importlib

    import server.core.profile_loader as pl
    pl = importlib.reload(pl)
    assert "EDGE_LLM_ASR_PLUGIN_PATH" in pl._OPERATOR_KEYS, "前置条件不成立"

    name = "jetson-qwen3asr-matcha-nx"
    want = (_load(name)["env"])["EDGE_LLM_ASR_PLUGIN_PATH"]
    assert want != image_default, "测试选错了 profile：两者本就相同"

    pl.apply_profile(name)
    assert os.environ["EDGE_LLM_ASR_PLUGIN_PATH"] == want, (
        "profile 未能压过镜像默认值"
    )
