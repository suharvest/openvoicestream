# Unified RK `final-slim` packaging

The current candidate is one unified `final-slim` runtime derived from the
receipt-bound `convonly-base`. RK resources, model profiles, and optional JA
dictionary data are mounted or selected at runtime. The image includes the
four small JA frontend wheels (`pyopenjtalk`, `fugashi`, `jaconv`, and
`mojimoji`) from the locked receipt; `unidic` and `unidic-lite` Python
packages and dictionary data remain external. The historical `final-convonly`,
`final-ja`, and legacy target interfaces remain documented below.

# ConvOnly source build with external device inputs

Application code is buildable from a clean clone with initialized submodules;
the ARM64 Rockchip runtime libraries and model bundles remain explicit external
release inputs and are checked before build/deploy.
The default `final-slim`, `final-thick`, `base`, and `final-ja` interfaces remain
legacy interfaces. `convonly-base` starts at `runtime-os`; no ancestor copies
the rkvoice source tree or installs it editable. `final-convonly` adds the
staged model bundle. No image build or device rollout has been qualified yet.

## Required frozen inputs

- `Dockerfile.rk` and `.dockerignore`.
- `requirements.kokoro-convonly.lock` (Python 3.11, ARM64 Linux closure).
- `requirements.kokoro-build.lock` (hash-pinned tools for pure-Python sdists).
- initialized `third_party/rkvoice-stream`; Docker builds its wheel in the
  disposable `rkvoice-wheel-builder` stage with `uv build`.
- published `voxedge==0.0.13a0`; if the release is absent
  from the configured mirror, publication must happen before image build.
- official `en_core_web_sm` 3.8.0 wheel downloaded by Dockerfile with SHA256
  `075db178346c39942ad96b96bdfa391685274549cb9dfc52e7ce213b59aebb59`.
- Frozen `server/`, `configs/`, `voices/`, `deploy/artifacts/` snapshots with
  dirty-source status and per-file hashes.
- externally staged `deploy/rk-runtime/` library inventory and authenticated
  manifest; a missing directory must stop the build before deployment.
- `deploy/kokoro-convonly-artifacts/` target bundles and inventories, only for
  `final-convonly` (not the dependency-only `convonly-base` qualification).

The manifest/source snapshots must be frozen together after implementation
stops. The shared `server/requirements.txt` published
VoxEdge pin is intentionally not changed or installed by this dedicated stage.

## Dependency verification

The core R package still supports Python 3.10; the optional Kokoro runtime
and this release target are qualified only for Python 3.11. ORT 1.29.0 matches
the observed production container, and RKNN Lite is pinned to 2.3.2.
Misaki 0.9.4 `en.G2P(trf=False)` uses `tok2vec` and `tagger`; its broad `en`
extra additionally declares curated-transformers, which this closure does
not require. EN/ZH use the explicitly listed subset and the frozen POS wheel.

Create an empty Python 3.11 venv and install the default dependency lock with
`uv pip sync --require-hashes`. Build the RKVoice wheel with `uv build`, then
install the application wheels with `--no-deps`. Run `check_kokoro_packages.py --applications` and
`uv pip check`. The checker performs real US/GB/ZH G2P, ORT/RKNN imports,
rejects editable installs and forbidden default packages, and checks local
application versions. No source checkout belongs on its Python import path.

## Explicit Japanese provisioning

Default EN/ZH excludes `pyopenjtalk`, `fugashi`, `jaconv`, `mojimoji`,
`unidic-lite` and full `unidic`. R's `kokoro-ja` extra and the separate
`requirements.kokoro-ja.lock` opt into the first five packages. Do not install
the broad `misaki[ja]` extra, which selects full UniDic.

Misaki 0.9.4 `JAG2P()` defaults to Cutlet, whose constructor imports fugashi
and creates `Tagger()`. The production image sets `MECABRC` from
`KOKORO_JA_DICDIR`, so the mounted dictionary is selected without importing a
dictionary package or embedding its data. pyopenjtalk
is still required by the module-level import, but this default route never
uses its separate Open JTalk dictionary. No inference-time download is needed.
Do not silently switch to `JAG2P(version='pyopenjtalk')`, which is a different
phoneme route and would require separately provisioned Open JTalk data.

Mount the external dictionary at `KOKORO_JA_DICDIR` and set `MECABRC` to its
`mecabrc`; the image carries the frontend code but no dictionary data. Install
the JA hash lock explicitly into the isolated installation and run
`check_kokoro_packages.py --japanese --applications`, then `uv pip check`.
The test asserts default Cutlet selection, real Japanese phonemes and the
selected dictionary path, and records per-file dictionary SHA256 values.
ARM64 pyopenjtalk and mojimoji may build from pinned source distributions;
the current optional JA venv test used build isolation. A frozen JA image
also needs receipt-bound built extension wheels/toolchain provenance; the
legacy `final-ja` target is not evidence of that clean JA image qualification.

## Recorded checkpoint

Fresh ARM64 Python 3.11 installation: 112 packages including the POS wheel,
real EN/GB/ZH frontend calls and `uv pip check` passed. Explicit JA addition:
117 packages, default `JAG2P()` emitted `koɲɲiʨiβa, sekai.` and selected the
lite dictionary; `uv pip check` passed. These are dependency/G2P checks, not
model synthesis, final application-wheel, server-startup or image acceptance.
Raw evidence is in the workspace's
`evidence-kokoro-remaining-20260903/clean-package-repair/` directory.
