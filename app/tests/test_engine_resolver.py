import json

from app.core import engine_resolver


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

    from app.core import hf_artifacts

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
        onnx_input=None,
        build_script=None,
        build_env={},
        hf_only=True,
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

    from app.core import hf_artifacts

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
        onnx_input=None,
        build_script=None,
        build_env={},
        hf_only=True,
        required=True,
        extra_files=["engines/missing-runtime.onnx"],
    )
    host = engine_resolver.HostSignature("87", "10.3", "6.2", "12.6")

    assert not engine_resolver._try_hf_resolve(spec, host)


def test_hf_resolve_tolerates_files_as_list(tmp_path, monkeypatch):
    """A moss-style manifest with ``files`` as a LIST (not the bundle dict)
    must not crash _try_hf_resolve — it should report no bundle match."""
    from app.core import hf_artifacts

    monkeypatch.setattr(
        hf_artifacts,
        "fetch_manifest",
        lambda mid: {"files": [{"path": "engines/x.plan", "sha256": "y"}]},
    )
    spec = engine_resolver.EngineSpec(
        model_id="moss", engine_file="e.plan", engine_path=tmp_path / "e.plan",
        env_var="E", onnx_input=None, build_script=None, build_env={},
        hf_only=True, required=True,
    )
    host = engine_resolver.HostSignature("87", "10.3", "6.2", "12.6")
    assert engine_resolver._try_hf_resolve(spec, host) is False  # no AttributeError


def test_resolve_one_keeps_unverified_engine_when_resolve_fails(tmp_path, monkeypatch):
    """An engine staged without a .meta sidecar must NOT be deleted before a
    replacement is secured — deleting first then failing loses it (that is how
    moss_tts_prefill.plan got destroyed)."""
    import pytest
    from app.core import hf_artifacts

    engine_dir = tmp_path / "models" / "demo" / "engines"
    engine_dir.mkdir(parents=True)
    engine = engine_dir / "encoder.plan"
    engine.write_bytes(b"pre-staged-engine")  # exists, NO .meta sidecar

    def no_manifest(model_id):
        raise hf_artifacts.ArtifactError("no manifest")

    monkeypatch.setattr(hf_artifacts, "fetch_manifest", no_manifest)

    spec = engine_resolver.EngineSpec(
        model_id="demo", engine_file="encoder.plan", engine_path=engine,
        env_var="ENC", onnx_input=None, build_script=None, build_env={},
        hf_only=True, required=True,
    )
    host = engine_resolver.HostSignature("87", "10.3", "6.2", "12.6")
    with pytest.raises(RuntimeError):  # hf_only + no bundle -> resolution fails
        engine_resolver._resolve_one(spec, host, force_rebuild=False)
    # ...but the unverified engine must SURVIVE the failed resolve.
    assert engine.exists()
    assert engine.read_bytes() == b"pre-staged-engine"
