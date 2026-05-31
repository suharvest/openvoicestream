import importlib

import numpy as np


def test_kokoro_profile_only_requires_kokoro_model(monkeypatch, tmp_path):
    from app.core import model_downloader, profile_loader

    model_root = tmp_path / "models"
    kokoro_dir = model_root / "kokoro-multi-lang-v1_0"
    kokoro_dir.mkdir(parents=True)
    for name in ("model.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt"):
        (kokoro_dir / name).write_bytes(b"ok")

    monkeypatch.setattr(
        profile_loader,
        "_CURRENT_PROFILE",
        {"tts_backend": "jetson.kokoro_trt"},
    )
    monkeypatch.setattr(model_downloader, "_patch_kokoro_voices", lambda _model_dir: None)

    def fail_download(*_args, **_kwargs):
        raise AssertionError("Kokoro profile should not download zh_en models")

    monkeypatch.setattr(model_downloader, "_download_and_extract", fail_download)

    model_downloader.ensure_models("zh_en", str(model_root))


def test_kokoro_default_sid_overrides_compose_default(monkeypatch):
    monkeypatch.setenv("TTS_DEFAULT_SID", "0")
    monkeypatch.setenv("KOKORO_DEFAULT_SID", "52")

    import voxedge.backends.jetson.kokoro_trt as kokoro_trt

    kokoro_trt = importlib.reload(kokoro_trt)
    assert kokoro_trt.DEFAULT_SPEAKER_ID == 52


def test_kokoro_stream_split_preserves_spaces(monkeypatch):
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend

    backend = KokoroTRTBackend.__new__(KokoroTRTBackend)
    monkeypatch.setattr(backend, "_text_to_token_ids", lambda text: list(text.replace(" ", "")))

    segments = backend._split_stream_text(
        "This is a deliberately long validation sentence", max_tokens=12
    )

    assert segments
    assert all("  " not in segment for segment in segments)
    assert " ".join(segments) == "This is a deliberately long validation sentence"


def test_kokoro_bucket_selection():
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend

    backend = KokoroTRTBackend.__new__(KokoroTRTBackend)
    backend._split_engines = {"decoder": object()}
    backend._split_long_engines = {"decoder": object()}
    # Per-call ctx rework: ctxs are passed in as kwargs instead of being
    # backend state; tests pass empty dicts (engine identity is what matters).
    split_ctxs = {"decoder": object()}
    split_long_ctxs = {"decoder": object()}

    assert backend._select_split_bucket(
        256, split_ctxs=split_ctxs, split_long_ctxs=split_long_ctxs
    )[0] is backend._split_engines
    assert backend._select_split_bucket(
        257, split_ctxs=split_ctxs, split_long_ctxs=split_long_ctxs
    )[0] is backend._split_long_engines
    assert backend._select_split_bucket(
        512, split_ctxs=split_ctxs, split_long_ctxs=split_long_ctxs
    )[0] is backend._split_long_engines

    backend._split_long_engines = {}
    try:
        backend._select_split_bucket(
            257, split_ctxs=split_ctxs, split_long_ctxs={}
        )
    except ValueError as exc:
        assert "outside available TRT buckets" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing long bucket")


def test_kokoro_synthesize_segments_instead_of_truncating(monkeypatch):
    from voxedge.backends.jetson import kokoro_trt
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend

    backend = KokoroTRTBackend.__new__(KokoroTRTBackend)
    backend._runtime_mode = "split_generator"
    backend._hybrid_max_seq_len = 10
    monkeypatch.setattr(backend, "_text_to_token_ids", lambda text: list(range(20)))
    monkeypatch.setattr(backend, "_split_stream_text", lambda text, max_tokens: ["first", "second"])

    def fake_one(text, speaker_id=None, speed=None):
        samples = np.ones(240, dtype=np.float32) * (0.1 if text == "first" else 0.2)
        return kokoro_trt._samples_to_wav(samples, kokoro_trt.SAMPLE_RATE), {
            "num_tokens": 6,
            "infer_ms": 1.5,
        }

    monkeypatch.setattr(backend, "_synthesize_one", fake_one)

    wav, meta = backend.synthesize("too long")

    assert len(wav) > 44
    assert meta["segments"] == 2
    assert meta["truncated"] is False
    assert meta["num_tokens"] == 12
