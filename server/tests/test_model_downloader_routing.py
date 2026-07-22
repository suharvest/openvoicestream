"""Tests for server.core.model_downloader profile-driven routing.

Regression: orin-nano 2026-05-25 silent Qwen3-skip bug. When OVS_PROFILE
selected a Qwen3 ASR profile but the environment had LANGUAGE_MODE=zh_en
pre-set, ensure_models() routed by language_mode and skipped
_ensure_qwen3_artifacts(). After fix, routing is profile-driven first
(asr_backend/tts_backend) and language_mode-driven second; both UNION.
"""

from __future__ import annotations

from unittest.mock import patch

from server.core import model_downloader


def _no_op_download(url, dest_dir):  # pragma: no cover - safety net
    raise AssertionError(
        f"unexpected real download attempt: {url} -> {dest_dir}"
    )


def test_profile_with_trt_edge_llm_triggers_qwen3_even_when_lang_mode_zh_en(
    tmp_path, monkeypatch,
):
    """Profile asr_backend=jetson.trt_edge_llm must call _ensure_qwen3_artifacts
    even when language_mode='zh_en' (the legacy zh_en path would otherwise
    never call it). This is the core orin-nano regression fix."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "asr_backend": "jetson.trt_edge_llm",
            "tts_backend": "jetson.matcha_trt",
        },
    )
    # Patch the symbol the function actually calls (module-level lookup).
    with patch.object(
        model_downloader, "_ensure_qwen3_artifacts"
    ) as mock_qwen3, patch.object(
        model_downloader, "_download_and_extract", side_effect=_no_op_download,
    ):
        # Pretend zh_en assets (matcha + paraformer) are already present
        # so no actual download fires. language_mode='zh_en' unions in
        # the legacy zh_en requirements alongside profile-driven matcha.
        for sub, files in (
            ("matcha-icefall-zh-en", ("model-steps-3.onnx", "tokens.txt", "lexicon.txt")),
            ("paraformer-streaming", ("encoder.onnx", "tokens.txt")),
        ):
            d = tmp_path / sub
            d.mkdir()
            for f in files:
                (d / f).write_text("x")

        model_downloader.ensure_models(
            language_mode="zh_en", model_dir=str(tmp_path),
        )

    mock_qwen3.assert_called_once()


def test_no_profile_zh_en_legacy_path_does_not_call_qwen3(tmp_path, monkeypatch):
    """Backward-compat: no profile + LANGUAGE_MODE=zh_en must NOT trigger
    Qwen3 (legacy behaviour for users who never opted into profiles)."""
    from server.core import profile_loader
    monkeypatch.setattr(profile_loader, "current_profile", lambda: {})
    with patch.object(
        model_downloader, "_ensure_qwen3_artifacts"
    ) as mock_qwen3, patch.object(
        model_downloader, "_download_and_extract", side_effect=_no_op_download,
    ):
        # Pretend both zh_en models exist so we don't trip the downloader.
        for sub, files in (
            ("matcha-icefall-zh-en", ("model-steps-3.onnx", "tokens.txt", "lexicon.txt")),
            ("paraformer-streaming", ("encoder.onnx", "tokens.txt")),
        ):
            d = tmp_path / sub
            d.mkdir()
            for f in files:
                (d / f).write_text("x")

        model_downloader.ensure_models(
            language_mode="zh_en", model_dir=str(tmp_path),
        )

    mock_qwen3.assert_not_called()


def test_moss_profile_triggers_moss_provision(tmp_path, monkeypatch):
    """A profile with tts_backend=jetson.moss_tts_nano must fire the MOSS
    provisioner (#47 unified-entry dispatch), even though MOSS does not go
    through the MODELS/CDN tarball mechanism."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {"asr_backend": None, "tts_backend": "jetson.moss_tts_nano"},
    )
    with patch.object(model_downloader, "_ensure_moss_artifacts") as mock_moss, \
            patch.object(model_downloader, "_ensure_qwen3_artifacts"), \
            patch.object(
                model_downloader, "_download_and_extract", side_effect=_no_op_download
            ):
        # multilanguage mode + moss-only TTS: no matcha/paraformer required.
        model_downloader.ensure_models(
            language_mode="multilanguage", model_dir=str(tmp_path),
        )

    mock_moss.assert_called_once()


def test_asr_artifact_manifest_routes_to_edgellm_and_not_qwen3(tmp_path, monkeypatch):
    """A trt_edge_llm profile that declares `asr_artifact_manifest` must go to
    the flat-manifest v090 provisioner and must NOT also fire the qwen3 path —
    including via the language_mode='multilanguage' branch, which is exactly
    what the v090 profile sets (LANGUAGE_MODE=multilanguage)."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "asr_backend": "jetson.trt_edge_llm",
            "tts_backend": "jetson.moss_tts_nano",
            "asr_artifact_manifest": "deploy/artifacts/edgellm_v090_manifest.json",
        },
    )
    with patch.object(model_downloader, "_ensure_edgellm_v090_artifacts") as mock_v090, \
            patch.object(model_downloader, "_ensure_qwen3_artifacts") as mock_qwen3, \
            patch.object(model_downloader, "_ensure_moss_artifacts"), \
            patch.object(
                model_downloader, "_download_and_extract", side_effect=_no_op_download
            ):
        model_downloader.ensure_models(
            language_mode="multilanguage", model_dir=str(tmp_path),
        )

    mock_v090.assert_called_once_with("deploy/artifacts/edgellm_v090_manifest.json")
    mock_qwen3.assert_not_called()


def test_trt_edge_llm_without_manifest_field_keeps_qwen3_path(tmp_path, monkeypatch):
    """No `asr_artifact_manifest` → unchanged legacy behaviour: qwen3 fires and
    the v090 provisioner stays out of it."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "asr_backend": "jetson.trt_edge_llm",
            "tts_backend": "jetson.moss_tts_nano",
        },
    )
    with patch.object(model_downloader, "_ensure_edgellm_v090_artifacts") as mock_v090, \
            patch.object(model_downloader, "_ensure_qwen3_artifacts") as mock_qwen3, \
            patch.object(model_downloader, "_ensure_moss_artifacts"), \
            patch.object(
                model_downloader, "_download_and_extract", side_effect=_no_op_download
            ):
        model_downloader.ensure_models(
            language_mode="multilanguage", model_dir=str(tmp_path),
        )

    mock_v090.assert_not_called()
    assert mock_qwen3.call_count >= 1


def test_v090_profile_json_dispatches_to_edgellm(tmp_path, monkeypatch):
    """End-to-end on the real profile file: loading the shipped
    jetson-edgellm-v090-moss.json routes ASR provisioning to the v090
    manifest, never to qwen3."""
    import json
    from pathlib import Path

    from server.core import profile_loader

    profile = json.loads(
        (Path(__file__).resolve().parents[2]
         / "configs" / "profiles" / "jetson-edgellm-v090-moss.json").read_text()
    )
    monkeypatch.setattr(profile_loader, "current_profile", lambda: profile)
    with patch.object(model_downloader, "_ensure_edgellm_v090_artifacts") as mock_v090, \
            patch.object(model_downloader, "_ensure_qwen3_artifacts") as mock_qwen3, \
            patch.object(model_downloader, "_ensure_moss_artifacts") as mock_moss, \
            patch.object(
                model_downloader, "_download_and_extract", side_effect=_no_op_download
            ):
        model_downloader.ensure_models(
            language_mode=profile["env"]["LANGUAGE_MODE"], model_dir=str(tmp_path),
        )

    mock_v090.assert_called_once_with("deploy/artifacts/edgellm_v090_manifest.json")
    mock_qwen3.assert_not_called()
    mock_moss.assert_called_once()


def test_non_moss_profile_does_not_trigger_moss(tmp_path, monkeypatch):
    """A non-MOSS profile (Qwen3 ASR + Matcha TTS) must NOT fire the MOSS
    provisioner."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "asr_backend": "jetson.trt_edge_llm",
            "tts_backend": "jetson.matcha_trt",
        },
    )
    with patch.object(model_downloader, "_ensure_moss_artifacts") as mock_moss, \
            patch.object(model_downloader, "_ensure_qwen3_artifacts"), \
            patch.object(
                model_downloader, "_download_and_extract", side_effect=_no_op_download
            ):
        # matcha already present so no real download fires.
        d = tmp_path / "matcha-icefall-zh-en"
        d.mkdir()
        for f in ("model-steps-3.onnx", "tokens.txt", "lexicon.txt"):
            (d / f).write_text("x")
        model_downloader.ensure_models(
            language_mode="multilanguage", model_dir=str(tmp_path),
        )

    mock_moss.assert_not_called()
