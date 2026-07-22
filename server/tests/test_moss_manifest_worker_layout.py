"""Pin the on-disk layout the MOSS C++ worker actually requires.

``moss_tts_nano_worker`` resolves every artifact relative to the directory it
is given as ``--engine-dir``; ``--codec-onnx-dir`` does not move the codec
metadata or the codec plan. The manifest used to declare those two under
``codec_onnx/``, which is where they live on HF, so a clean provision left
``engines/`` short of two files and the worker died at startup with

    Missing MOSS codec metadata: .../engines/codec_browser_onnx_meta.json
    Failed to open TensorRT engine: .../engines/codec_decode_step.plan

This went unnoticed for a long time because the development Jetson had those
two bridged by hand-made symlinks inside ``engines/``. Any host provisioned
from scratch -- i.e. every real deployment -- had neither.

The lesson these tests encode: "provisioner reported ready" is not the same as
"the worker can start". Marking a file the worker needs as ``optional`` turns a
404 into a silent skip, and the failure then surfaces ten seconds later as an
opaque ``worker_exit / stdout eof``.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST = (
    Path(__file__).resolve().parents[2] / "deploy" / "artifacts" / "moss_manifest.json"
)

# Everything moss_tts_nano_worker opens relative to --engine-dir.
WORKER_REQUIRES_IN_ENGINES = {
    "moss_tts_prefill.plan",
    "moss_tts_decode_step.plan",
    "moss_tts_local_decoder.plan",
    "moss_tts_local_cached_step.plan",
    "moss_tts_local_fixed_sampled_frame.plan",
    "moss_tts_global_shared.data",
    "moss_tts_local_shared.data",
    "tokenizer.model",
    # Codec pair: stored under codec_onnx/ on HF, but the worker looks for both
    # in engines/, so the manifest has to relocate them via source_path.
    "codec_decode_step.plan",
    "codec_browser_onnx_meta.json",
    # Prompt templates. Without this the worker warns "prompt_templates missing
    # in browser_poc_manifest.json -- generation will likely be incorrect".
    "browser_poc_manifest.json",
    "tts_browser_onnx_meta.json",
}


def _files() -> list[dict]:
    return json.loads(MANIFEST.read_text())["files"]


def _model_root_entries() -> dict[str, dict]:
    return {
        f["path"]: f for f in _files() if f.get("dest", "model_root") == "model_root"
    }


def test_every_file_the_worker_opens_lands_in_engines() -> None:
    entries = _model_root_entries()
    landed = {
        Path(p).name for p in entries if Path(p).parent.as_posix() == "engines"
    }
    missing = sorted(WORKER_REQUIRES_IN_ENGINES - landed)
    assert not missing, (
        "the worker resolves these relative to --engine-dir but the manifest "
        f"does not land them in engines/: {missing}"
    )


def test_relocated_entries_keep_a_source_path() -> None:
    """A path the worker needs may differ from where HF stores it.

    When it does, ``source_path`` must carry the remote location, otherwise the
    provisioner builds a URL that 404s.
    """
    for path, entry in _model_root_entries().items():
        name = Path(path).name
        if name in {"codec_decode_step.plan", "codec_browser_onnx_meta.json"}:
            assert entry.get("source_path", "").startswith("codec_onnx/"), (
                f"{path} is relocated into engines/ but has no codec_onnx/ "
                f"source_path; the download URL would 404"
            )


def test_worker_inputs_are_not_optional() -> None:
    """``optional`` turns a missing file into a silent skip.

    The provisioner then logs "MOSS artifacts ready" and the real failure shows
    up much later as an unexplained worker exit.
    """
    silently_skippable = sorted(
        path
        for path, entry in _model_root_entries().items()
        if entry.get("optional") and Path(path).name in WORKER_REQUIRES_IN_ENGINES
    )
    assert not silently_skippable, (
        f"these are worker inputs and must fail loudly if absent: {silently_skippable}"
    )
