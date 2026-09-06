"""Cover every cell of configs/matrix/language_device.yaml and the error paths.

The matrix is the contract: a (language, device) pair resolves to exactly one
profile that exists on disk, Chinese never falls back to Whisper, and an
unsupported pair fails loudly instead of picking something plausible.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "resolve_profile.py"
MATRIX_PATH = REPO_ROOT / "configs" / "matrix" / "language_device.yaml"
PROFILES_DIR = REPO_ROOT / "configs" / "profiles"

sys.path.insert(0, str(REPO_ROOT / "tools"))

import resolve_profile as rp  # noqa: E402

MATRIX = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
LANGUAGES = MATRIX["languages"]
DEVICES = MATRIX["devices"]
CELLS = MATRIX["cells"]

DEVICE_IDS = sorted(DEVICES)
GROUPS = sorted({c["group"] for c in CELLS})


def cell_for(device: str, group: str) -> dict:
    matches = [c for c in CELLS if c["device"] == device and c["group"] == group]
    assert len(matches) == 1, f"expected exactly one cell for {device}/{group}, got {len(matches)}"
    return matches[0]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


# --------------------------------------------------------------- matrix shape


def test_matrix_covers_every_device_group_pair():
    assert len(CELLS) == len(DEVICE_IDS) * len(GROUPS)
    for device in DEVICE_IDS:
        for group in GROUPS:
            cell_for(device, group)


def test_language_catalog_matches_rk_runtime_size():
    # The catalogue is deliberately the narrower RK runtime list, not Qwen3's
    # advertised 52 languages (spec section 5 risk note).
    assert MATRIX["language_catalog_size"] == 30
    assert len(LANGUAGES) == 30
    assert LANGUAGES["zh"]["group"] == "zh"
    assert LANGUAGES["en"]["group"] == "en"


def test_every_cell_has_a_status_and_evidence():
    for cell in CELLS:
        assert cell["status"] in {"measured", "untested", "unsupported"}, cell
        assert cell.get("evidence"), f"{cell['device']}/{cell['group']} has no evidence"
        assert all(str(item).strip() for item in cell["evidence"])


def test_supported_cells_name_a_profile_that_exists():
    for cell in CELLS:
        if cell["status"] == "unsupported":
            assert cell["ovs_profile"] is None
            assert cell.get("reason"), f"{cell['device']}/{cell['group']} must say why"
            continue
        profile = cell["ovs_profile"]
        assert (PROFILES_DIR / f"{profile}.json").is_file(), profile


def test_supported_profiles_declare_both_asr_and_tts_backends():
    for cell in CELLS:
        if cell["status"] == "unsupported":
            continue
        data = json.loads((PROFILES_DIR / f"{cell['ovs_profile']}.json").read_text())
        assert data["asr_backend"], cell["ovs_profile"]
        assert data["tts_backend"], cell["ovs_profile"]


def test_planned_alternatives_reference_real_profiles():
    for cell in CELLS:
        planned = cell.get("planned_alternative")
        if not planned:
            continue
        assert planned.get("blocked_on"), cell["device"]
        for key in ("asr_profile", "tts_profile"):
            name = planned.get(key)
            if name:
                assert (PROFILES_DIR / f"{name}.json").is_file(), name


# ------------------------------------------------------------ every cell resolves


@pytest.mark.parametrize("device", DEVICE_IDS)
@pytest.mark.parametrize("group", GROUPS)
def test_every_cell_resolves_as_declared(device, group):
    cell = cell_for(device, group)
    language = next(code for code, meta in LANGUAGES.items() if meta["group"] == group)

    if cell["status"] == "unsupported":
        with pytest.raises(rp.ResolveError) as excinfo:
            rp.resolve(language, device, matrix_path=MATRIX_PATH, profiles_dir=PROFILES_DIR)
        assert excinfo.value.code == rp.EXIT_UNSUPPORTED
        return

    env, resolved_cell, warnings = rp.resolve(
        language, device, matrix_path=MATRIX_PATH, profiles_dir=PROFILES_DIR
    )
    assert resolved_cell == cell
    assert env["OVS_PROFILE"] == cell["ovs_profile"]
    assert env["OVS_MATRIX_STATUS"] == cell["status"]
    assert env["LANGUAGE"] == language
    assert env["TTS_LANGUAGE"] == language
    assert env["ASR_LANGUAGE"] == language
    # Single session, single mono lane on every cell (spec section 3).
    assert env["OVS_MAX_CONCURRENT_SESSIONS"] == "1"
    assert env["OVS_AUDIO_MONO"] == "1"
    if cell["status"] == "untested":
        assert warnings and warnings[0].startswith("WARN:")
    else:
        assert warnings == []


@pytest.mark.parametrize("language", sorted(LANGUAGES))
def test_every_language_routes_to_exactly_one_profile_per_device(language):
    for device in DEVICE_IDS:
        try:
            env, _, _ = rp.resolve(
                language, device, matrix_path=MATRIX_PATH, profiles_dir=PROFILES_DIR
            )
        except rp.ResolveError as exc:
            assert exc.code == rp.EXIT_UNSUPPORTED
            continue
        assert env["OVS_PROFILE"]


# ------------------------------------------------------------------ error paths


def test_chinese_on_rpi5_is_refused_and_never_falls_back_to_whisper():
    with pytest.raises(rp.ResolveError) as excinfo:
        rp.resolve("zh", "rpi5", matrix_path=MATRIX_PATH, profiles_dir=PROFILES_DIR)
    assert excinfo.value.code == rp.EXIT_UNSUPPORTED
    assert "not supported" in str(excinfo.value)
    # The refusal must name Whisper only to rule it out, never as a fallback.
    assert "must never silently fall back to Whisper" in str(excinfo.value)


def test_no_cell_routes_chinese_to_a_whisper_profile():
    for cell in CELLS:
        if cell["group"] != "zh" or cell["status"] == "unsupported":
            continue
        data = json.loads((PROFILES_DIR / f"{cell['ovs_profile']}.json").read_text())
        assert "whisper" not in str(data["asr_backend"]).lower()
        assert "whisper" not in cell["ovs_profile"]


def test_unknown_language_exits_3():
    with pytest.raises(rp.ResolveError) as excinfo:
        rp.resolve("klingon", "rk3576", matrix_path=MATRIX_PATH, profiles_dir=PROFILES_DIR)
    assert excinfo.value.code == rp.EXIT_UNKNOWN


def test_unknown_device_exits_3():
    with pytest.raises(rp.ResolveError) as excinfo:
        rp.resolve("zh", "rk9999", matrix_path=MATRIX_PATH, profiles_dir=PROFILES_DIR)
    assert excinfo.value.code == rp.EXIT_UNKNOWN


def test_missing_matrix_file_exits_3(tmp_path):
    with pytest.raises(rp.ResolveError) as excinfo:
        rp.resolve("zh", "rk3576", matrix_path=tmp_path / "nope.yaml")
    assert excinfo.value.code == rp.EXIT_UNKNOWN


def test_missing_profile_json_exits_4(tmp_path):
    with pytest.raises(rp.ResolveError) as excinfo:
        rp.resolve("zh", "rk3576", matrix_path=MATRIX_PATH, profiles_dir=tmp_path)
    assert excinfo.value.code == rp.EXIT_MISSING_PROFILE


def test_language_tag_and_case_are_normalised():
    for raw in ("zh", "ZH", "zh-CN", "zh_CN", " zh "):
        env, _, _ = rp.resolve(raw, "rk3576", matrix_path=MATRIX_PATH, profiles_dir=PROFILES_DIR)
        assert env["OVS_PROFILE"] == "rk3576-default"


def test_device_alias_normalisation():
    env, _, _ = rp.resolve("zh", "ORIN-NX", matrix_path=MATRIX_PATH, profiles_dir=PROFILES_DIR)
    assert env["OVS_PROFILE"] == "jetson-qwen3asr-matcha-nx"


# ------------------------------------------------------------------------- CLI


def test_cli_measured_cell_is_quiet_and_exits_0():
    proc = run_cli("--language", "zh", "--device", "rk3576")
    assert proc.returncode == 0, proc.stderr
    assert "OVS_PROFILE=rk3576-default" in proc.stdout
    assert "WARN" not in proc.stderr


def test_cli_untested_cell_warns_but_exits_0():
    proc = run_cli("--language", "en", "--device", "orin_nx")
    assert proc.returncode == 0, proc.stderr
    assert "OVS_PROFILE=jetson-qwen3asr-matcha-nx" in proc.stdout
    assert "WARN" in proc.stderr


def test_cli_unsupported_cell_exits_2_and_emits_nothing():
    proc = run_cli("--language", "zh", "--device", "rpi5")
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "not supported" in proc.stderr


def test_cli_ignores_the_posix_locale_LANGUAGE_variable():
    # LANGUAGE is POSIX's locale fallback list ("en_US:en"). It must never
    # select a profile; only OVS_LANGUAGE may.
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--device", "rk3576"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={"PATH": "/usr/bin:/bin", "LANGUAGE": "en_US:en"},
    )
    assert proc.returncode == 3
    assert proc.stdout == ""


def test_cli_reads_ovs_language_and_ovs_device_from_env():
    proc = subprocess.run(
        [sys.executable, str(TOOL)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={"PATH": "/usr/bin:/bin", "OVS_LANGUAGE": "zh", "OVS_DEVICE": "rk3588"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "OVS_PROFILE=rk3588-default" in proc.stdout


def test_cli_missing_arguments_exit_3():
    proc = subprocess.run(
        [sys.executable, str(TOOL)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 3
    assert "required" in proc.stderr


def test_cli_json_format():
    proc = run_cli("--language", "zh", "--device", "rk3588", "--format", "json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["profile"] == "rk3588-default"
    assert payload["status"] == "measured"
    assert payload["env"]["TTS_LANGUAGE"] == "zh"


def test_cli_write_env_produces_a_sourceable_file(tmp_path):
    target = tmp_path / "nested" / "ovs.env"
    proc = run_cli("--language", "ja", "--device", "rk3588", "--write-env", str(target))
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in target.read_text().splitlines() if line]
    assert all("=" in line and " " not in line.split("=", 1)[0] for line in lines)
    assert "OVS_PROFILE=rk3588-kokoro-rknn" in lines
