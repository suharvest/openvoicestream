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


# NOTE: dropped — voxedge kokoro_trt has no module-level DEFAULT_SPEAKER_ID
# (env-free). KOKORO_DEFAULT_SID/TTS_DEFAULT_SID → config.default_speaker_id is
# covered in app/tests/test_voxedge_backend_config.py.


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


# NOTE: dropped — the synthesize-segments-instead-of-truncating algorithm moved
# to voxedge with the env-free migration and is covered byte-for-byte in
# voxedge/tests/test_kokoro_synth_segments.py.
