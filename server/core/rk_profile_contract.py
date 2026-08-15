"""Runtime contract for the production Qwen3-ASR + Matcha RK profiles.

The RK compose files used to repeat a subset of profile settings.  Because
``profile_loader`` intentionally preserves operator-owned environment values,
those repeated values could silently win over the selected profile (notably
``NPU_CORE_AUTO`` and an 800 ms endpoint debounce on RK3576).  This module is
the small, dependency-free contract used by startup/health/capability code to
make that drift observable.

It deliberately reports only bounded configuration values; no credentials or
artifact paths are included in the status payload.
"""

from __future__ import annotations

import os
from typing import Mapping


RK_RELEASE_PROFILES = frozenset(
    {
        "rk3576-default",
        "rk3576-multilang",
        "rk3588-default",
        "rk3588-multilang",
    }
)

_COMMON_EXPECTED: dict[str, str] = {
    "ASR_MAX_NEW_TOKENS": "64",
    "ASR_FINAL_STOP_ON_PUNCT": "1",
    "ASR_FINAL_STOP_MIN_CHARS": "8",
    "ASR_FINAL_STOP_MIN_CHUNKS": "2",
    "ASR_NPU_CORE_MASK": "NPU_CORE_1",
    "QWEN3_ASR_CHUNK_CONFIRM": "0",
    "QWEN3_ASR_STREAM_MODE": "true_streaming",
    "QWEN3_ASR_STREAM_TRUE": "1",
    "QWEN3_ASR_TRUE_ROLL_SEC": "5",
    "QWEN3_ASR_TRUE_PARTIAL_TOKENS": "8",
    "QWEN3_ASR_TRUE_PARTIAL_INTERVAL_MS": "1500",
    "QWEN3_ASR_TRUE_PARTIAL_WARMUP": "2",
    "QWEN3_ASR_FRONTEND_EOU_MIN_AUDIO_S": "2.5",
    "VAD_ENDPOINT_SILENCE_MS": "400",
    "MATCHA_USE_ORT": "1",
    "MATCHA_MODEL_SEQ_LEN": "80",
    "MATCHA_MIN_MEL_FRAMES": "72",
    "MATCHA_STREAM_CHUNK_MS": "40",
}

_PLATFORM_EXPECTED: dict[str, dict[str, str]] = {
    # RK3576 regressed when the final RKLLM decode was moved to the async
    # endpoint path; keep its measured synchronous close-out recipe.
    "rk3576": {
        "QWEN3_ASR_VAD_FINAL_ASYNC": "0",
        "VOCOS_FRAMES": "600",
    },
    # RK3588 is the platform where async endpoint finalization was measured to
    # improve dialogue latency.
    "rk3588": {
        "QWEN3_ASR_VAD_FINAL_ASYNC": "1",
        "VOCOS_FRAMES": "256",
    },
}

_PROFILE_KEYS = (*_COMMON_EXPECTED, "QWEN3_ASR_VAD_FINAL_ASYNC")


def _profile_name(profile: Mapping[str, object] | None) -> str:
    if not profile:
        return ""
    return str(profile.get("name") or "").strip()


def is_release_profile(profile: Mapping[str, object] | None) -> bool:
    """Return whether *profile* is one of the four production Qwen3 RK profiles."""
    return _profile_name(profile) in RK_RELEASE_PROFILES


def _device_for(name: str, profile_env: Mapping[str, object], env: Mapping[str, str]) -> str | None:
    value = profile_env.get("RK_PLATFORM") or env.get("RK_PLATFORM")
    if value:
        return str(value)
    if name.startswith("rk3576-"):
        return "rk3576"
    if name.startswith("rk3588-"):
        return "rk3588"
    return None


def runtime_status(
    profile: Mapping[str, object] | None,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Compare a selected RK release profile with the effective process env.

    ``required`` is false for all other profiles so existing Jetson/RPi and
    legacy RK profiles retain their current contract.  For a release profile,
    ``verified`` is true only when both the JSON profile and the process env
    contain the complete, validated true-streaming/NPU/Matcha set.
    """
    name = _profile_name(profile)
    profile_env_raw = (profile or {}).get("env") or {}
    profile_env = profile_env_raw if isinstance(profile_env_raw, Mapping) else {}
    actual = os.environ if env is None else env

    if name not in RK_RELEASE_PROFILES:
        return {
            "required": False,
            "profile": name or None,
            "device": _device_for(name, profile_env, actual),
            "contract": "not_applicable",
            "verified": True,
            "settings": {},
            "missing_profile": [],
            "missing_runtime": [],
            "mismatches": {},
        }

    device = _device_for(name, profile_env, actual)
    expected = dict(_COMMON_EXPECTED)
    expected.update(_PLATFORM_EXPECTED.get(device or "", {}))
    keys = (*_PROFILE_KEYS, "VOCOS_FRAMES")

    missing_profile = [key for key in keys if str(profile_env.get(key, "")).strip() == ""]
    missing_runtime = [key for key in keys if str(actual.get(key, "")).strip() == ""]
    mismatches: dict[str, dict[str, str]] = {}
    for key, wanted in expected.items():
        profile_value = str(profile_env.get(key, ""))
        actual_value = str(actual.get(key, ""))
        if profile_value and profile_value != wanted:
            mismatches[key] = {
                "expected": wanted,
                "profile": profile_value,
                "runtime": actual_value,
            }
        elif actual_value and actual_value != wanted:
            mismatches[key] = {
                "expected": wanted,
                "profile": profile_value or wanted,
                "runtime": actual_value,
            }

    settings = {key: str(actual.get(key, "<missing>")) for key in keys}
    return {
        "required": True,
        "profile": name,
        "device": device,
        "contract": "rk-qwen3-true-streaming-matcha-v1",
        "verified": not missing_profile and not missing_runtime and not mismatches,
        "settings": settings,
        "missing_profile": missing_profile,
        "missing_runtime": missing_runtime,
        "mismatches": mismatches,
    }


def format_failure(status: Mapping[str, object]) -> str:
    """Format a short actionable startup/readiness diagnostic."""
    details: list[str] = []
    if status.get("missing_profile"):
        details.append("missing profile keys=" + ",".join(map(str, status["missing_profile"])))
    if status.get("missing_runtime"):
        details.append("missing runtime keys=" + ",".join(map(str, status["missing_runtime"])))
    mismatches = status.get("mismatches") or {}
    if mismatches:
        details.append("mismatches=" + ",".join(map(str, mismatches)))
    return (
        f"RK profile {status.get('profile')!r} does not satisfy "
        f"{status.get('contract')}: " + ("; ".join(details) or "unknown drift")
    )
