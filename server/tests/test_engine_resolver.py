import json

from server.core import engine_resolver


def test_hf_bundle_metadata_covers_all_extracted_engines(tmp_path, monkeypatch):
    engine_dir = tmp_path / "models" / "demo" / "engines"
    encoder = engine_dir / "encoder.plan"
    decoder = engine_dir / "decoder.plan"

    def fake_fetch_manifest(model_id):
        assert model_id == "demo"
        return {
            "files": {
                "engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz": {
                    "sha256": "unused",
                    "size": 123,
                }
            }
        }

    def fake_download_and_extract_tarball(rel_path, dest_dir, expected_sha256=None):
        assert rel_path == "models/demo/engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz"
        assert expected_sha256 == "unused"
        dest_dir.mkdir(parents=True, exist_ok=True)
        encoder.write_bytes(b"encoder")
        decoder.write_bytes(b"decoder")
        (dest_dir / "._decoder.plan").write_bytes(b"macos metadata")

    from server.core import hf_artifacts

    monkeypatch.setattr(hf_artifacts, "fetch_manifest", fake_fetch_manifest)
    monkeypatch.setattr(
        hf_artifacts,
        "download_and_extract_tarball",
        fake_download_and_extract_tarball,
    )

    spec = engine_resolver.EngineSpec(
        model_id="demo",
        engine_file="encoder.plan",
        engine_path=encoder,
        env_var="ENC_ENGINE",
        onnx_input=None,        hf_only=True,
        required=True,
    )
    host = engine_resolver.HostSignature("87", "10.3", "6.2", "12.6")

    assert engine_resolver._try_hf_resolve(spec, host)

    encoder_meta = json.loads((engine_dir / "encoder.plan.meta.json").read_text())
    decoder_meta = json.loads((engine_dir / "decoder.plan.meta.json").read_text())
    assert encoder_meta["source"] == "hf_bundle"
    assert decoder_meta["source"] == "hf_bundle"
    assert encoder_meta["host"] == host.to_dict()
    assert decoder_meta["host"] == host.to_dict()
    assert not (engine_dir / "._decoder.plan.meta.json").exists()


def test_hf_bundle_resolve_requires_extra_files(tmp_path, monkeypatch):
    engine_dir = tmp_path / "models" / "demo" / "engines"
    encoder = engine_dir / "encoder.plan"

    def fake_fetch_manifest(model_id):
        return {
            "files": {
                "engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz": {
                    "sha256": "unused",
                    "size": 123,
                }
            }
        }

    def fake_download_and_extract_tarball(rel_path, dest_dir, expected_sha256=None):
        dest_dir.mkdir(parents=True, exist_ok=True)
        encoder.write_bytes(b"encoder")

    from server.core import hf_artifacts

    monkeypatch.setattr(hf_artifacts, "fetch_manifest", fake_fetch_manifest)
    monkeypatch.setattr(
        hf_artifacts,
        "download_and_extract_tarball",
        fake_download_and_extract_tarball,
    )

    spec = engine_resolver.EngineSpec(
        model_id="demo",
        engine_file="encoder.plan",
        engine_path=encoder,
        env_var="ENC_ENGINE",
        onnx_input=None,        hf_only=True,
        required=True,
        extra_files=["engines/missing-runtime.onnx"],
    )
    host = engine_resolver.HostSignature("87", "10.3", "6.2", "12.6")

    assert not engine_resolver._try_hf_resolve(spec, host)


def test_hf_resolve_tolerates_files_as_list(tmp_path, monkeypatch):
    """A moss-style manifest with ``files`` as a LIST (not the bundle dict)
    must not crash _try_hf_resolve — it should report no bundle match."""
    from server.core import hf_artifacts

    monkeypatch.setattr(
        hf_artifacts,
        "fetch_manifest",
        lambda mid: {"files": [{"path": "engines/x.plan", "sha256": "y"}]},
    )
    spec = engine_resolver.EngineSpec(
        model_id="moss", engine_file="e.plan", engine_path=tmp_path / "e.plan",
        env_var="E", onnx_input=None,
        hf_only=True, required=True,
    )
    host = engine_resolver.HostSignature("87", "10.3", "6.2", "12.6")
    assert engine_resolver._try_hf_resolve(spec, host) is False  # no AttributeError


def test_resolve_one_keeps_unverified_engine_when_resolve_fails(tmp_path, monkeypatch):
    """An existing offline engine is adopted and gets a current sidecar."""

    engine_dir = tmp_path / "models" / "demo" / "engines"
    engine_dir.mkdir(parents=True)
    engine = engine_dir / "encoder.plan"
    engine.write_bytes(b"pre-staged-engine")  # exists, NO .meta sidecar

    def unexpected_hf(*args, **kwargs):
        raise AssertionError("existing engine migration must not access HF")

    monkeypatch.setattr(engine_resolver, "_try_hf_resolve", unexpected_hf)

    spec = engine_resolver.EngineSpec(
        model_id="demo", engine_file="encoder.plan", engine_path=engine,
        env_var="ENC", onnx_input=None,
        hf_only=True, required=True,
    )
    host = engine_resolver.HostSignature("87", "10.3", "6.2", "12.6")
    engine_resolver._resolve_one(spec, host, force_rebuild=False)
    assert engine.exists()
    assert engine.read_bytes() == b"pre-staged-engine"
    meta = json.loads((engine_dir / "encoder.plan.meta.json").read_text())
    assert meta["host"] == host.to_dict()
    assert meta["source"] == "existing_install_migration"
    assert meta["engine_sha256"] == engine_resolver._sha256_file(engine)


def test_resolve_one_migrates_pre_platform_sidecar(tmp_path, monkeypatch):
    engine = tmp_path / "encoder.plan"
    engine.write_bytes(b"legacy-engine")
    host = engine_resolver.HostSignature("87", "10.3", "6.2", "12.6")
    old_host = host.to_dict()
    old_host.pop("platform")
    engine_resolver._meta_path(engine).write_text(json.dumps({
        "host": old_host,
        "engine_sha256": engine_resolver._sha256_file(engine),
        "onnx_sha256": "old-onnx-sha",
        "source": "hf_bundle",
    }))

    def unexpected_hf(*args, **kwargs):
        raise AssertionError("matching legacy sidecar must migrate offline")

    monkeypatch.setattr(engine_resolver, "_try_hf_resolve", unexpected_hf)
    spec = engine_resolver.EngineSpec(
        model_id="demo", engine_file=engine.name, engine_path=engine,
        env_var="ENC", onnx_input=None, hf_only=True, required=True,
    )

    engine_resolver._resolve_one(spec, host, force_rebuild=False)

    meta = json.loads(engine_resolver._meta_path(engine).read_text())
    assert meta["host"] == host.to_dict()
    assert meta["onnx_sha256"] == "old-onnx-sha"
    assert meta["source"] == "existing_install_migration"


def test_resolve_one_rejects_mismatched_legacy_sidecar(tmp_path, monkeypatch):
    import pytest

    engine = tmp_path / "encoder.plan"
    engine.write_bytes(b"legacy-engine")
    host = engine_resolver.HostSignature("87", "10.3", "6.2", "12.6")
    engine_resolver._meta_path(engine).write_text(json.dumps({
        "host": {
            "sm": "72", "trt_version": "8.5", "jp_version": "5.1",
            "cuda_version": "11.4",
        },
        "engine_sha256": engine_resolver._sha256_file(engine),
    }))
    monkeypatch.setattr(engine_resolver, "_try_hf_resolve", lambda *a, **k: False)
    spec = engine_resolver.EngineSpec(
        model_id="demo", engine_file=engine.name, engine_path=engine,
        env_var="ENC", onnx_input=None, hf_only=True, required=True,
    )

    with pytest.raises(RuntimeError, match="no valid local engine"):
        engine_resolver._resolve_one(spec, host, force_rebuild=False)


def test_entry_kind_classification():
    """Kind attribution for required_engines env_vars (bundled-profile split)."""
    from server.core.engine_resolver import _entry_kind
    assert _entry_kind("PARAFORMER_ENC_ENGINE") == "asr"
    assert _entry_kind("SENSEVOICE_TRT_MODEL_DIR") == "asr"
    assert _entry_kind("EDGE_LLM_ASR_ENGINE_DIR") == "asr"
    assert _entry_kind("KOKORO_SPLIT_GENERATOR_ENGINE") == "tts"
    assert _entry_kind("MATCHA_SPLIT_ESTIMATOR_ENGINE") == "tts"
    assert _entry_kind("VOCOS_ENGINE") == "tts"
    assert _entry_kind("EDGE_LLM_TTS_TALKER_DIR") == "tts"
    # MOSS is a TTS engine family — must be scoped out of ASR reloads (else a
    # kind=asr reload of a MOSS bundle provisions/validates the MOSS TTS engines).
    assert _entry_kind("MOSS_ENGINE_DIR") == "tts"
    assert _entry_kind("MOSS_CODEC_ONNX_DIR") == "tts"
    assert _entry_kind("MOSS_RESOLVED_PREFILL") == "tts"
    # Shared / ambiguous → None (validated for every kind)
    assert _entry_kind("QWEN3_ARTIFACT_ROOT") is None
    assert _entry_kind("MODEL_DIR") is None


def test_kind_scoped_entry_filter():
    """A kind-scoped resolve of a paraformer+kokoro bundle keeps only that
    modality's engines (+ shared) — the other kind is never provisioned, so an
    offline reload can't hang fetching the paired backend's engines."""
    from server.core.engine_resolver import _entry_kind
    entries = [
        {"env_var": "PARAFORMER_ENC_ENGINE"},
        {"env_var": "KOKORO_SPLIT_GENERATOR_ENGINE"},
        {"env_var": "QWEN3_ARTIFACT_ROOT"},  # shared → kept for both
    ]

    def keep(kind):
        return [e["env_var"] for e in entries
                if _entry_kind(e.get("env_var", "")) in (None, kind)]

    assert keep("asr") == ["PARAFORMER_ENC_ENGINE", "QWEN3_ARTIFACT_ROOT"]
    assert keep("tts") == ["KOKORO_SPLIT_GENERATOR_ENGINE", "QWEN3_ARTIFACT_ROOT"]
