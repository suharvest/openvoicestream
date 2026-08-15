from __future__ import annotations

import hashlib
import io
import tarfile

from server.core import model_downloader


def _tar_gz_with_file(name: str, contents: bytes) -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(contents)
        tar.addfile(info, io.BytesIO(contents))
    return archive.getvalue()


def test_verified_archive_download_uses_module_tempfile_on_cold_start(
    tmp_path, monkeypatch
):
    """A clean model volume must reach verified extraction without scoping errors."""
    payload = _tar_gz_with_file("matcha-icefall-zh-en/tokens.txt", b"token\n")

    monkeypatch.setattr(model_downloader.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        model_downloader.urllib.request,
        "urlopen",
        lambda _request, timeout: io.BytesIO(payload),
    )

    model_downloader._download_and_extract(
        "https://example.invalid/models-matcha.tar.gz",
        str(tmp_path),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
    )

    assert (tmp_path / "matcha-icefall-zh-en" / "tokens.txt").read_bytes() == b"token\n"
