"""Tests for GET /admin/backend/loadable.

The endpoint globs ``<repo>/configs/profiles/*.json`` and asks each manager to
preview its own half of every profile, classifying each as loadable /
unloadable / invalid. We redirect the glob to a temp profiles dir (by pointing
``backend_manager.__file__`` at a fake repo root) and stub both managers plus
``find_missing_artifacts`` so classification is fully controlled.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class _FakeMgr:
    """Stand-in manager: ``_load_profile_kind`` returns a tagged preview,
    except for names in ``bad`` which raise (→ classified invalid)."""

    def __init__(self, kind: str, bad: set[str] | None = None):
        self._kind = kind
        self.name = kind  # real BackendManager exposes .name = "tts"/"asr"
        self._bad = bad or set()

    def _load_profile_kind(self, ref: str) -> dict:
        if ref in self._bad:
            raise ValueError(f"boom:{ref}")
        return {"name": ref, "kind": self._kind}


@pytest.fixture
def loadable_env(monkeypatch, tmp_path):
    # Fake repo root so the endpoint globs OUR controlled profile set.
    core = tmp_path / "server" / "core"
    core.mkdir(parents=True)
    fake_file = core / "backend_manager.py"
    fake_file.write_text("# fake")
    prof = tmp_path / "configs" / "profiles"
    prof.mkdir(parents=True)
    for stem in ("p-good", "p-missing", "p-bad"):
        (prof / f"{stem}.json").write_text("{}")

    from server.core import backend_manager as bm
    from server.core import profile_loader as pl

    monkeypatch.setattr(bm, "__file__", str(fake_file))

    # tts: p-bad fails to LOAD → invalid; p-missing missing artifacts.
    tts = _FakeMgr("tts", bad={"p-bad"})
    # asr: p-bad LOADS fine but lacks its ASR engine → unloadable (not invalid).
    asr = _FakeMgr("asr")
    monkeypatch.setattr(bm, "tts_manager", lambda: tts)
    monkeypatch.setattr(bm, "asr_manager", lambda: asr)

    def fake_missing(preview: dict, kind: str | None = None) -> list[dict]:
        name = preview.get("name")
        if name == "p-missing":
            return [{"env_var": "SOME_ENGINE", "path": "/nope"}]
        if kind == "asr" and name == "p-bad":
            return [{"env_var": "ASR_ENGINE", "path": "/nope2"}]
        return []

    monkeypatch.setattr(pl, "find_missing_artifacts", fake_missing)
    return tmp_path


def _client(admin_allowed: bool):
    from server.main import app
    from server.core.admin_auth import require_admin

    if admin_allowed:
        async def _allow():
            return None
        app.dependency_overrides[require_admin] = _allow
    return app, require_admin


def test_loadable_classifies_per_kind(loadable_env):
    app, require_admin = _client(admin_allowed=True)
    try:
        c = TestClient(app)
        r = c.get("/admin/backend/loadable")
    finally:
        app.dependency_overrides.pop(require_admin, None)

    assert r.status_code == 200, r.text
    body = r.json()

    # TTS side: p-good loadable, p-missing unloadable, p-bad invalid.
    assert body["tts"]["loadable"] == ["p-good"]
    assert [u["name"] for u in body["tts"]["unloadable"]] == ["p-missing"]
    assert body["tts"]["unloadable"][0]["missing"] == [
        {"env_var": "SOME_ENGINE", "path": "/nope"}
    ]
    assert [i["name"] for i in body["tts"]["invalid"]] == ["p-bad"]
    assert "boom:p-bad" in body["tts"]["invalid"][0]["error"]

    # ASR side: same profile set, DIFFERENT verdicts — p-bad is loadable-parse
    # but missing its ASR engine (unloadable), and p-good is loadable.
    assert body["asr"]["loadable"] == ["p-good"]
    assert {u["name"] for u in body["asr"]["unloadable"]} == {"p-missing", "p-bad"}
    assert body["asr"]["invalid"] == []


def test_loadable_requires_admin(loadable_env):
    # No dependency override → non-loopback TestClient host, OVS_ADMIN_KEY unset
    # → require_admin refuses with 403.
    app, _ = _client(admin_allowed=False)
    c = TestClient(app)
    r = c.get("/admin/backend/loadable")
    assert r.status_code == 403, r.text
