# #27 Device Image Cutover — Dockerfile Change Draft

Status: **DRAFT for review** (no Dockerfiles changed). Goal: rebuild device
images so they run the post-`voxedge`-migration code (latest `app/` + `voxedge`
installed). Registry already points at the new code paths because the migration
is committed — the image just needs the new `app/` baked in **plus the `voxedge`
package installed**. No registry change required.

Scope of this draft: local analysis only. No build / deploy / device / HF.

---

## 0. TL;DR of recommended changes

| Image | Change | Why |
|---|---|---|
| RK + Jetson (both) | Add `voxedge` install step | `app/` now imports `voxedge.*` (main.py, voxedge_backend_config.py) — image fails without it |
| RK | Add `prometheus_client` to pip step | #24: image predates the dep, runtime crashes without it |
| RK | Bake kokoro voice-pack `default.npy` + `style.npy` into `/opt/kokoro-rknn/` | Fixes silence; active RK_ARTIFACT_SET on device is the **matcha** set, so the manifest downloader never pulls the kokoro npy |
| Jetson | Add `voxedge` install step (only change needed) | worker/engine artifacts already baked |

**voxedge install method chosen: ② prebuilt wheel COPY + pip install.** (rationale §2.)

---

## 1. Current-state summary (key lines)

### 1.1 `deploy/docker/Dockerfile.rk`
- Base: `debian:12-slim` (`:22`).
- Python deps: single `pip install` from PyPI mirror, **no requirements.txt / uv** — deps inlined (`:60-83`). Misaki + RK NPU + sherpa-onnx + fastapi etc. **No `prometheus_client`.**
- App COPY: `app/`, `configs/`, `deploy/artifacts/`, `voices/`, `scripts/`, `third_party/rkvoice-stream/`, `deploy/rk-runtime/` (`:87-93`).
- third_party handling: `rkvoice-stream` copied then `pip install -e` (`:107`).
- Artifact bake (kokoro): `COPY deploy/kokoro-artifacts/bucket-{8,16,32}/ → /opt/kokoro-bucket-{8,16}/` and **bucket-32 → `/opt/kokoro-rknn/`** (`:102-104`). Staged on host build-time only (dir not committed).
- RK runtime libs: `deploy/rk-runtime/ → /opt/rk-runtime/` + symlinks (`:93`, `:110-113`).
- ASR/other models: NOT baked — downloaded at first start by `server/core/model_downloader.py → rk_artifacts.ensure_rk_artifacts()` per manifest (`deploy/artifacts/rk_manifest.json`).
- Build context: **repo root** (`docker build -f deploy/docker/Dockerfile.rk … .`, `:14`).
- Entry: `uvicorn server.main:app` (`:135`), `PYTHONPATH=/opt/speech`.

### 1.2 `deploy/docker/Dockerfile.jetson`
- Base: `arm64v8/ubuntu:22.04`, Python 3.10; CUDA/TRT libs mounted from host at runtime (`:27-41`).
- Python deps: single `pip3 install` from mirror, inlined (`:44-59`). Includes `transformers<5`, `cuda-python`, `onnxruntime`, `piper-phonemize`. **No `prometheus_client`** (the **slim** variant has it, `:54` of `.slim`).
- App COPY: `app/`, `configs/`, `voices/`, `scripts/`, `entrypoint.jetson.sh` (`:78-83`).
- third_party / worker handling:
  - `deploy/jetson-workers/ → /opt/jv-workers/` (`:88`)
  - `third_party/qwen3-edgellm-jetson/ → /opt/qwen3-edgellm-jetson/` (`:89`)
  - workers chmod + plugin `.so` `cp` into `/opt/edgellm` + `/opt/edgellm-bin` (`:93-100`).
- Artifact bake: worker binaries + plugin `.so` baked (`:88-100`); CustomVoice v0.7.1 bin/plugin/embeds baked (`:110-118`). Models/engines downloaded on first start by `model_downloader._ensure_qwen3_artifacts()` (`:91-92`).
- Build context: **repo root** (`:12`).
- Entry: `entrypoint.jetson.sh` → sources `/etc/openvoicestream.env`, resolves `OVS_PROFILE`, `exec`s `uvicorn server.main:app` (`:138-139`). `PYTHONPATH=/opt/speech:/usr/lib/python3.10/dist-packages` (`:129`).

### 1.3 `deploy/docker/Dockerfile.jetson.slim`
- Same as `.jetson` but: `--no-compile` + purge `tests/`/`__pycache__`/`.pyc` (`:38-58`), **already lists `prometheus_client`** (`:54`), and hardlink/symlink-dedupes the plugin `.so` instead of `cp` (`:88-95`). All `voxedge` and RK changes below apply symmetrically here.

---

## 2. How `voxedge` enters the image (KEY)

### Facts
- `voxedge` is a **separate sibling repo**: `/Users/harvest/project/voxedge` (NOT under `seeed-local-voice/third_party/`, NOT a submodule).
- It is **pure-Python, `py3-none-any`** (core dep: `numpy` only; heavy backend deps are optional extras that are intentionally empty placeholders / `aarch64`-gated and ship from engine repos, not from voxedge). See `voxedge/pyproject.toml`.
- A **prebuilt universal wheel already exists**: `voxedge/dist/voxedge-0.0.1a0-py3-none-any.whl` (151 KB).
- `app/` hard-depends on it: `server/main.py` (`from voxedge.engine.* / voxedge.transport.*`) and `server/core/voxedge_backend_config.py` (imports `voxedge.backends.jetson.* / .rk.* / .sherpa.*`). Without `voxedge` installed, the image still **boots** (imports are mostly lazy/inside functions) but the v2v path + backend config break — must be installed.
- Build runs **on the device** (orin-nx recon confirmed local build), and the build context is the **repo root** of `seeed-local-voice`. A sibling repo is **outside** that context — `COPY ../../voxedge` is **illegal** (Docker forbids `..` escapes from context).

### Two candidate approaches
- **① Add voxedge source to the build context** (e.g. `COPY voxedge/ …` after vendoring it into the repo, then `pip install ./voxedge`). Rejected: requires either changing the build context root (breaks every other `COPY app/…` relative path) or vendoring the whole voxedge source tree into seeed-local-voice (couples the two repos, defeats the open-core split, and drags voxedge's `tests/`, `.venv`, egg-info unless carefully filtered).
- **② Build voxedge wheel, COPY the wheel, `pip install` it.** ✅ **CHOSEN.**

### Why ② (wheel COPY)
1. The wheel is **already built** and is `py3-none-any` — no per-arch rebuild, installs identically on RK (debian/x86-ish glibc) and Jetson (aarch64).
2. Keeps the two repos decoupled: only one 151 KB file enters the context, not the whole source tree.
3. Pure-Python + `numpy` already present in both images → no extra heavy deps pulled. Install **without extras** (`pip install voxedge-….whl`, NOT `voxedge[jetson]`/`[rk]`) because the extras' real runtime deps (tensorrt, rknn, sherpa, qwen3_speech_engine) are already satisfied by the image's own inlined deps / baked workers; the extras are placeholders/aarch64-gated and would at best be redundant, at worst try to pull wheels that don't exist on PyPI.
4. Deterministic & offline-friendly on-device build (no network fetch of voxedge from an index).

### Wiring (build-time staging)
The wheel must be **inside the build context** (repo root) at build time. Recommended: stage it under a build-only dir, mirroring how `deploy/kokoro-artifacts/` is staged but not committed:

```
# build host (device), before docker build:
mkdir -p deploy/wheels
cp /path/to/voxedge/dist/voxedge-0.0.1a0-py3-none-any.whl deploy/wheels/
```

(Add `deploy/wheels/` to `.gitignore` like `deploy/kokoro-artifacts/`, since the wheel is a release artifact, not source. Alternatively commit a pinned wheel if reproducibility-by-checkout is wanted — that's a policy call for review.)

### Draft Dockerfile fragment — both images
Place **after** the app `COPY` block (so app code is present) but it can go right after the main pip step. Minimal:

```dockerfile
# ── voxedge (edge-native voice library) ───────────────────────────────
# Pure-Python py3-none-any wheel (numpy-only core). Installed WITHOUT extras:
# heavy backend deps (tensorrt / rknn / sherpa / qwen3_speech_engine) are
# already provided by this image's own deps + baked workers. See
# docs/archive/specs/cutover-dockerfile-draft.md §2.
COPY deploy/wheels/voxedge-0.0.1a0-py3-none-any.whl /tmp/voxedge.whl
RUN pip install --no-cache-dir --index-url ${PIP_INDEX} /tmp/voxedge.whl \
    && rm -f /tmp/voxedge.whl
```

- RK image uses `pip` (venv on PATH); Jetson uses `pip3` — match the existing image's pip invocation.
- For `.slim`, append the same `__pycache__`/`.pyc` purge tail if matching its size discipline (optional — voxedge is tiny).

---

## 3. RK image — extra changes

### 3.1 `prometheus_client` (#24)
Add to the existing `pip install` list at `Dockerfile.rk:60-83`. Single new line:

```diff
         'addict==2.4.0' \
         regex \
+        prometheus_client \
     && ldconfig
```

(The slim Jetson already has it; add to **RK** here. Non-slim Jetson `.jetson` does **not** list it — see §4 note.)

### 3.2 Kokoro voice-pack → `/opt/kokoro-rknn/` (fix silence)

**Root-cause of why the manifest path does NOT cover this:**
- The voice-pack `default.npy` / `style.npy` (sha `e844b0d3…`) are registered **only** in the `rk3588-kokoro-hybrid-*` artifact sets of `deploy/artifacts/rk_manifest.json` (lines `:260-269`, `:328-…`), mapped to `opt/kokoro-rknn/default.npy` / `style.npy`.
- But the device's active set defaults to **`RK_ARTIFACT_SET=rk3576-multilang-2026-05-17`** (`deploy/docker-compose.rk.yml:53`) — the **matcha** set, which contains **no** kokoro npy.
- `ensure_rk_artifacts()` only downloads files of the **single selected set**. So when running a matcha (or default) RK_ARTIFACT_SET, the kokoro voice-pack is never fetched → kokoro TTS reads missing/empty style vectors → **silent audio**.

**Simplest reliable fix — bake the two npy into `/opt/kokoro-rknn/`.**

The Dockerfile already does `COPY deploy/kokoro-artifacts/bucket-32/ → /opt/kokoro-rknn/` (`:104`). Two options:

- **3.2.a (preferred, zero new COPY):** ensure the build-time staging of `deploy/kokoro-artifacts/bucket-32/` **includes** `default.npy` and `style.npy` (from HF `harvestsu/seeed-local-voice-rk-artifacts/rk3588/kokoro-hybrid-v1/`). Then the existing `:104` COPY lands them at `/opt/kokoro-rknn/default.npy` + `style.npy` automatically. This is a **staging instruction**, not a Dockerfile change — document it in the build runbook.
- **3.2.b (explicit, self-documenting):** add a dedicated COPY so the requirement is visible in the Dockerfile and independent of bucket-32 staging contents:

```dockerfile
# Kokoro voice-pack (style vectors). Baked because the active RK_ARTIFACT_SET
# on device is the matcha set, which does not carry these — without them
# kokoro TTS emits silence. Source: HF rk3588/kokoro-hybrid-v1/{default,style}.npy
# (sha256 e844b0d3200a28b47a2472d4052c2696ac639197aa0ab06bb3fb708781b5a09d).
COPY deploy/kokoro-artifacts/voice-pack/default.npy /opt/kokoro-rknn/default.npy
COPY deploy/kokoro-artifacts/voice-pack/style.npy   /opt/kokoro-rknn/style.npy
```

Recommend **3.2.b** for explicitness (survives bucket-32 re-staging changes). Place right after the existing bucket COPYs (`:104`).

> Alternative (NOT recommended for cutover): switch the device's `RK_ARTIFACT_SET` to a `rk3588-kokoro-hybrid-*` set so the downloader pulls the npy. That changes runtime behavior/model selection, not just the image, and is out of scope for an image-cutover. Baking is the narrow fix.

---

## 4. Jetson image — extra changes

- **Only required change: install `voxedge`** (§2 fragment). Worker binaries, plugin `.so`, CustomVoice artifacts, and engine downloader are all already wired (`:88-118`).
- **Note for review (not strictly part of #27):** non-slim `Dockerfile.jetson` does **not** list `prometheus_client` while `.slim` does (`:54`). If the device runs the **non-slim** Jetson image and the metrics endpoint is exercised, add `prometheus_client` to `Dockerfile.jetson:44-59` for parity. Flagged, not assumed.

---

## 5. Risk / gotchas

1. **Build context / sibling repo (highest):** `voxedge` lives outside the `seeed-local-voice` build context. `COPY ../../voxedge` is impossible. The wheel **must be staged into the repo tree** (`deploy/wheels/`) before `docker build`. The on-device build runbook must include the `cp …whl deploy/wheels/` step, or the build fails on missing COPY source. This is the single most likely cutover failure.
2. **Wheel version pinning:** fragment hardcodes `voxedge-0.0.1a0`. If voxedge bumps version, the COPY source path drifts. Mitigate with a glob-free explicit filename (current) + a runbook check, or stage as a fixed name (`cp …whl deploy/wheels/voxedge.whl` and COPY that fixed name).
3. **Install WITHOUT extras:** do **not** `pip install voxedge[jetson]` / `[rk]`. The extras' real heavy deps (tensorrt, cuda-python, rknn-toolkit-lite2, sherpa-onnx, qwen3_speech_engine) are either already in the image or `aarch64`-gated / non-PyPI; pulling extras risks a failed/duplicate install. Plain `pip install <wheel>` is correct because every heavy runtime import in voxedge backends is **deferred (lazy)**, so the package imports fine with only `numpy`.
4. **numpy version:** voxedge core wants `numpy>=1.24`. RK pins `numpy<2` (`Dockerfile.rk:61`); Jetson installs unpinned `numpy`. `>=1.24,<2` satisfies voxedge → OK. No conflict, but verify the resolved numpy in the final image isn't downgraded below 1.24 by another dep.
5. **RK voice-pack staging dependency:** if choosing 3.2.a (reuse bucket-32), the silence fix silently regresses if a future bucket-32 staging drops the npy. 3.2.b (explicit COPY) is more robust but needs its own `deploy/kokoro-artifacts/voice-pack/` staging dir present at build time.
6. **PYTHONPATH ordering:** both images set `PYTHONPATH=/opt/speech…`. `voxedge` installs into site-packages (not `/opt/speech`), so it's importable regardless — no PYTHONPATH change needed. Confirmed app imports `voxedge` as an installed package, not a vendored path.
7. **`.slim` parity:** apply §2 voxedge fragment to `Dockerfile.jetson.slim` too, or the slim image cutover breaks. All three Dockerfiles need the voxedge step; only RK needs prometheus + voice-pack.
8. **No registry change needed:** confirmed — migration is code-committed; baking new `app/` + installing voxedge is sufficient. Registry tag/repo untouched.

---

## 6. Change checklist (for the eventual edit, after review)

- [ ] Stage `voxedge` wheel → `deploy/wheels/voxedge-0.0.1a0-py3-none-any.whl` (+ `.gitignore`).
- [ ] `Dockerfile.rk`: add voxedge COPY+install fragment; add `prometheus_client` to pip list; add voice-pack COPY (3.2.b) + stage `deploy/kokoro-artifacts/voice-pack/{default,style}.npy`.
- [ ] `Dockerfile.jetson`: add voxedge COPY+install fragment. (Optional: add `prometheus_client`.)
- [ ] `Dockerfile.jetson.slim`: add voxedge COPY+install fragment (+ purge tail if desired).
- [ ] Update on-device build runbook with the wheel + voice-pack staging steps.
