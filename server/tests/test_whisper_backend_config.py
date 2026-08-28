"""env → WhisperASRConfig, and the capability probe that reads it."""
from __future__ import annotations

import pytest

from server.core import voxedge_backend_config as vbc

_ENV = {"WHISPER_ENCODER_PATH": "/opt/models/whisper/enc.rknn", "MODEL_DIR": "/opt/m"}


def test_encoder_path_is_required():
    # Without it the backend would happily construct and fail at preload with
    # an unrelated file-not-found on the default path.
    with pytest.raises(ValueError, match="WHISPER_ENCODER_PATH"):
        vbc.build_whisper_asr_config("rknn", env={})


@pytest.mark.parametrize("kind,window,cutoff", [
    # Each default is the window the shipped graph for that path was compiled
    # at, plus that path's boundary guard — not a tuning preference.
    ("hailo", 10.0, 1.0),
    ("rknn", 10.0, 0.0),
    ("tensorrt", 30.0, 0.0),
])
def test_per_path_window_and_cutoff_defaults(kind, window, cutoff):
    cfg = vbc.build_whisper_asr_config(kind, env=_ENV)
    assert (cfg.window_s, cfg.padding_cutoff_s) == (window, cutoff)
    assert cfg.encoder_kind == kind


def test_directories_default_under_model_dir():
    # MODEL_DIR still works as a root; the layout underneath is the one the
    # downloader writes, and the decoder family follows WHISPER_VARIANT.
    cfg = vbc.build_whisper_asr_config("rknn", env={**_ENV, "WHISPER_VARIANT": "base10"})
    assert cfg.vocab_dir == "/opt/m/whisper"
    assert cfg.decoder_dir == "/opt/m/whisper/decoder/base"


def test_env_overrides_win():
    cfg = vbc.build_whisper_asr_config("rknn", env={
        **_ENV,
        "WHISPER_WINDOW_S": "20",
        "WHISPER_LANGUAGE": "zh",
        "WHISPER_DECODER_THREADS": "4",
        "WHISPER_ALL_CORES": "1",
        "WHISPER_VOCAB_DIR": "/data/vocab",
    })
    assert cfg.window_s == 20.0
    assert cfg.language == "zh"
    assert cfg.decoder_threads == 4
    assert cfg.all_cores is True
    assert cfg.vocab_dir == "/data/vocab"


def test_an_unparseable_window_raises_rather_than_defaulting():
    # The window selects the encoder's compiled shape, and rknn-lite does not
    # validate it — a silent fallback returns plausible nonsense instead.
    with pytest.raises(ValueError, match="WHISPER_WINDOW_S"):
        vbc.build_whisper_asr_config("rknn", env={**_ENV, "WHISPER_WINDOW_S": "wide"})


@pytest.mark.parametrize("spec,kind", [
    ("hailo.whisper", "hailo"),
    ("rk.whisper", "rknn"),
    ("jetson.whisper_trt", "tensorrt"),
])
def test_spec_dispatch(monkeypatch, spec, kind):
    monkeypatch.setenv("WHISPER_ENCODER_PATH", _ENV["WHISPER_ENCODER_PATH"])
    assert vbc.build_config_for_spec(spec, "asr", {}).encoder_kind == kind


def test_capability_probe_reads_the_encoder_kind(monkeypatch):
    """The probe builds a stub without __init__, so it must not need hardware.

    Hailo hands /dev/hailo0 to one process, so its path needs a cross-backend
    device mutex; a TRT engine shares the GPU with the TTS stack and does not.
    """
    monkeypatch.setenv("WHISPER_ENCODER_PATH", _ENV["WHISPER_ENCODER_PATH"])
    from voxedge.backends.whisper import WhisperASR

    hailo = vbc.concurrency_capability_for_spec("hailo.whisper", WhisperASR, "asr", {})
    trt = vbc.concurrency_capability_for_spec("jetson.whisper_trt", WhisperASR, "asr", {})
    assert hailo.requires_exclusive_device is True
    assert trt.requires_exclusive_device is False
    assert hailo.max_concurrent == 1 and hailo.is_stateful is False
