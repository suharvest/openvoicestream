# NVIDIA Isaac Sim 4.x — Host Assessment: `wsl2-local`

Date: 2026-06-14 · Probe-only (no install performed) · Device: `wsl2-local` (WSL2 on Windows host)

## Verdict: **GO** — via the **container route (Route B)**

RTX 3060 (12 GB) + WSL2 GPU passthrough working + Docker with the NVIDIA container
runtime, CDI, and `nvidia-container-toolkit 1.19.0` already installed. The Isaac Sim
container ships everything missing on the host (NVIDIA Vulkan ICD, GL libs, Python 3.10).
The pip-wheel route is **disfavored** because the host runs Python 3.12 / Ubuntu 24.04,
neither of which matches Isaac Sim 4.x's official support matrix (Python 3.10, Ubuntu
20.04/22.04). The container sidesteps both.

---

## Facts table

| Fact | Value | Source | Isaac Sim 4.x requirement | OK? |
|---|---|---|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12288 MiB VRAM | pre-gathered | RTX (RTX 2070+ / ≥8 GB rec.) | ✅ |
| GPU driver (Win host) | 591.74 | pre-gathered | ≥ 525 (RTX), 537+ for WSL | ✅ |
| WSL2 kernel | 5.15.167.4-microsoft-standard-WSL2 | pre-gathered | WSL2 GPU passthrough | ✅ |
| Disk `/` | 193 GB free of 1007 GB (80% used) | pre-gathered | ~30–50 GB (image + assets) | ✅ |
| `$HOME` volume | same `/dev/sdc`, 193 GB free | `df -h ~` | same big volume (not overlay) | ✅ |
| RAM | 31 GiB total, ~30 GiB free | pre-gathered | ≥ 32 GB rec. (16 GB min) | ⚠️ borderline (31≈32) |
| OS | **Ubuntu 24.04.4 LTS (noble)** | `/etc/os-release` | 20.04 / 22.04 official | ⚠️ newer than supported |
| Python | **3.12.3** (no `python3.10`) | `python3 --version` | **3.10** for pip wheels | ❌ for Route A |
| uv | 0.11.14 | `uv --version` | n/a | ✅ |
| Vulkan loader | `libvulkan.so.1.3.275` present | `ls` | needed | ✅ |
| NVIDIA Vulkan ICD | **absent** (no `nvidia_icd.json`; only mesa/lvp/virtio/intel/radeon/nouveau ICDs) | `ls /usr/share/vulkan/icd.d/` + `find` | NVIDIA ICD needed for renderer | ❌ on host → container provides it |
| `vulkaninfo` / `glxinfo` | not installed | `which` | diagnostic only | — |
| WSL GPU device `/dev/dxg` | **present** (`crw-rw-rw-`) | `ls -la /dev/dxg` | required for WSL GPU | ✅ |
| WSL libs | `libnvidia-encode`, `libnvidia-ml.so.1`, `libnvidia-gpucomp.so.590.52.01`, `nvidia-smi` under `/usr/lib/wsl/lib` | `ls` | driver passthrough present | ✅ |
| Docker | **29.5.2** | `docker info` | needed for Route B | ✅ |
| NVIDIA container runtime | **registered** (`Runtimes: ... nvidia runc`; CDI `nvidia.com/gpu=all`) | `docker info` | required for `--gpus all` | ✅ |
| nvidia-container-toolkit | **1.19.0-1** (+ libnvidia-container-tools / libnvidia-container1) | `dpkg -l` | required for Route B | ✅ |
| User in `docker` group | **yes** (`harve ... docker`) | `id -nG` | docker run without sudo | ✅ |
| Existing Isaac/Omniverse | **none** found in `$HOME` | `ls ~`, `find`, `pip list` | fresh install | ✅ |
| CUDA toolkit (`nvcc`) | absent | `nvcc --version` | not required (ships own) | ✅ |

---

## Reasoning + caveats

- **RTX-capable + passthrough confirmed.** RTX 3060 with `/dev/dxg` present and the WSL
  driver libs in `/usr/lib/wsl/lib` mean the RTX/OptiX path is available. Driver 591.74 is
  well above the WSL minimum.
- **12 GB VRAM caveat.** 12 GB is comfortable for robotics scenes (single/few robots,
  warehouse-scale), USD assets, RTX real-time rendering, and Isaac Lab RL with modest
  parallel envs. It is **not** enough for very large multi-robot scenes, dense photoreal
  environments, or high parallel-env counts in Isaac Lab — expect to cap `num_envs` and
  scene complexity. For headless RL/synthetic-data workloads this is fine; for heavy
  rendering, budget accordingly.
- **RAM 31 GiB is right at the recommended floor** (32 GB). Workable; avoid running other
  memory-heavy stacks concurrently during large scene loads.
- **OS / Python mismatch is the deciding factor.** Ubuntu 24.04 + Python 3.12 fall outside
  Isaac Sim 4.x's official matrix. Route A (pip wheels) needs Python **3.10** specifically;
  a `uv venv --python 3.10` would have to download a 3.10 interpreter and the wheels run on
  an unsupported distro. The container pins a known-good Ubuntu 22.04 + Python 3.10 + the
  NVIDIA Vulkan ICD internally, neutralizing both mismatches. **→ Route B.**
- **Host has no NVIDIA Vulkan ICD.** The Isaac Sim renderer needs a Vulkan ICD pointing at
  the NVIDIA driver; the host only has software/other-vendor ICDs. Inside the official
  container, NVIDIA ships `nvidia_icd.json` + GL/Vulkan libs and the container toolkit
  injects the WSL driver — so rendering works in-container without touching the host.

---

## Recommended route: **B — Container** (`nvcr.io/nvidia/isaac-sim:4.x.x`)

Everything Route B needs is already installed (docker 29.5, nvidia runtime, CDI,
toolkit 1.19, user in docker group). No one-time host setup required.

### Exact run command (DO NOT RUN until approved)

First pull (~ approx **15–25 GB** image, one-time):

```bash
# Pull (run on wsl2-local). Replace 4.x.x with the chosen tag, e.g. 4.5.0
docker pull nvcr.io/nvidia/isaac-sim:4.5.0
```

Headless run (livestream / no X window), GPU + asset cache mounted to the big volume:

```bash
docker run --name isaac-sim --rm --gpus all \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  --network=host \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  nvcr.io/nvidia/isaac-sim:4.5.0 \
  ./runheadless.sh -v
```

Notes:
- `--gpus all` works because the nvidia runtime + CDI are registered (confirmed).
- The cache `-v` mounts under `~/docker/isaac-sim/...` keep the multi-GB asset/shader
  caches on the 193 GB `/dev/sdc` volume (NOT inside the container layer). First headless
  run still downloads **several GB of Omniverse assets/shaders** into these caches.
- `--network=host` exposes the WebRTC/livestream ports for the Omniverse Streaming Client.
- `nvcr.io` may require `docker login nvcr.io` with an NGC API key if the tag is gated.

### Route A — pip wheels (NOT recommended here; documented for completeness)

Blocked by host Python 3.12 + Ubuntu 24.04. To use it you'd first stand up a Python 3.10
venv (one-time): `uv` can fetch a 3.10 interpreter.

```bash
# One-time: 3.10 venv (uv downloads CPython 3.10)
uv venv --python 3.10 ~/isaacsim-venv
source ~/isaacsim-venv/bin/activate
# Wheels: ~10–15 GB download; assets pulled on first run
uv pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
# Headless first launch
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y PRIVACY_CONSENT=Y
isaacsim isaacsim.exp.full --no-window
```

Caveat: running on unsupported Ubuntu 24.04 / glibc-noble is unvalidated by NVIDIA and may
hit Vulkan ICD / GLIBC issues the container avoids. Prefer Route B.

---

## WSL2-specific headless gotchas

- **Vulkan ICD selection.** The host lacks an NVIDIA Vulkan ICD. In the container, NVIDIA's
  ICD is present and the toolkit injects the WSL driver — no host change needed. If you ever
  go host-native (Route A), you must install/point a `nvidia_icd.json` at the WSL driver
  (`/usr/lib/wsl/lib`), otherwise the renderer falls back to llvmpipe (software) or fails.
- **Headless / no window.** Use `runheadless.sh` (container) or `--no-window` (pip). Connect
  with the **Omniverse Streaming Client** over the host network (WebRTC). There is no local
  display in WSL2 by default.
- **`ACCEPT_EULA=Y` + `PRIVACY_CONSENT=Y`** are required to run unattended; without them the
  first launch blocks on an interactive prompt.
- **Run as non-root inside container if possible.** The default image runs as root; the cache
  mounts above map to `/root/...`. If you switch to a non-root user, repoint the cache mounts
  to that user's home, or asset caches won't persist.
- **Asset cache location / disk impact.** Omniverse caches land in `~/.cache/ov`,
  `~/.cache/nvidia/GLCache`, `~/.nv/ComputeCache`, and `~/.nvidia-omniverse` (host-native), or
  the mounted `~/docker/isaac-sim/cache/*` (container). These grow to **several GB** and the
  first headless run downloads assets/shaders — keep them on the 193 GB `/dev/sdc` volume,
  never in the container's writable layer.
- **First run is slow + downloads.** Expect a multi-GB asset/shader download and long shader
  compile on the first headless boot; subsequent boots reuse the caches.
- **`/dev/dxg` is the GPU passthrough device** — confirmed present. If a future WSL/driver
  update breaks GPU access, check `/dev/dxg` and `nvidia-smi` inside WSL first.

---

## Next command to run when the user approves install

**Route B (recommended):**
```bash
# on wsl2-local — pulls ~15–25 GB, one-time
docker pull nvcr.io/nvidia/isaac-sim:4.5.0
mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache} \
         ~/docker/isaac-sim/{logs,data,documents}
# then the docker run command above
```

**Route A (only if a host-native install is specifically required):**
```bash
uv venv --python 3.10 ~/isaacsim-venv && source ~/isaacsim-venv/bin/activate
uv pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
```

> Verify the exact `4.x.x` tag against the current NGC catalog before pulling; 4.5.0 is the
> latest 4.x line as of writing. Confirm whether `docker login nvcr.io` is needed for the tag.

---

## EVIDENCE — raw outputs

### `cat /etc/os-release`
```
PRETTY_NAME="Ubuntu 24.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.4 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
UBUNTU_CODENAME=noble
```

### `python3 --version` (+ which python3.10 / uv)
```
Python 3.12.3
<no python3.10 on PATH>
uv 0.11.14 (x86_64-unknown-linux-gnu)
```

### `vulkaninfo --summary | head` + ICDs
```
which vulkaninfo  → not installed (no output)
ls /usr/share/vulkan/icd.d/:
asahi_icd.json
gfxstream_vk_icd.json
intel_hasvk_icd.json
intel_icd.json
lvp_icd.json
nouveau_icd.json
radeon_icd.json
virtio_icd.json
( NO nvidia_icd.json )
libvulkan.so loader present: /usr/lib/x86_64-linux-gnu/libvulkan.so.1.3.275
/dev/dxg: crw-rw-rw- 1 root root 10, 127  (GPU passthrough device present)
```

### `docker info | grep -i runtime` (+ version + toolkit)
```
Docker version 29.5.2, build 79eb04c
  cdi: nvidia.com/gpu=all
 Runtimes: io.containerd.runc.v2 nvidia runc
 Default Runtime: runc
dpkg nvidia-container:
ii  libnvidia-container-tools        1.19.0-1  amd64
ii  libnvidia-container1:amd64       1.19.0-1  amd64
ii  nvidia-container-toolkit         1.19.0-1  amd64
ii  nvidia-container-toolkit-base    1.19.0-1  amd64
user groups: harve adm cdrom sudo dip plugdev users docker
```

### `df -h ~`
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdc       1007G  764G  193G  80% /
```

### Existing Isaac/Omniverse + nvcc
```
ls ~ | grep -iE 'isaac|omni'  → (none)
find ~ -maxdepth 3 -iname '*isaac*'  → (none)
pip list | grep -i isaac  → (none)
nvcc --version  → (not installed)
```

---

**Markdown written to:** `/Users/harvest/project/seeed-local-voice/docs/sim/isaac_host_assessment.md`

---

## Pull result — 2026-06-14 (`wsl2-local`)

**Status: PULLED + import-smoke PASS. Renderer (Vulkan) BLOCKED on WSL2 — see below.**

### Tag pulled
`nvcr.io/nvidia/isaac-sim:4.5.0` — **anonymous pull, no NGC login required**
(`docker manifest inspect` returned a valid manifest without auth).

- Final pull status: `Status: Downloaded newer image for nvcr.io/nvidia/isaac-sim:4.5.0`
- Digest: `sha256:c2f47dc82a7714af08d3766efe80ac9d084c2b37b5d0dfbd074797ec56390fc7`
- On disk: `nvcr.io/nvidia/isaac-sim:4.5.0   c2f47dc82a77   22.6GB`
- Proxy/auth steps needed: **none** (daemon reached nvcr.io directly; no `~/.docker/config.json` change, no `docker login`).

### Smoke results
- Container runs, GPU visible in-container: `GPU 0: NVIDIA GeForce RTX 3060` (`nvidia-smi -L`).
- Bundled Python: **3.10.15**.
- `from isaacsim import SimulationApp` → **`SIMAPP_IMPORTABLE`** (import succeeds).
  - NOTE: use `python.sh <scriptfile>`, **NOT** `python.sh -c "..."`. On this build `-c`
    force-boots the full kit renderer first (which crashes on the Vulkan issue below) before
    running the inline code. A script file runs the import cleanly.
- Full `SimulationApp({"headless": True})` init → **FAILS** (segfault) with:
  `[carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER` /
  `vkCreateInstance failed. Vulkan 1.1 is not supported`.

### Renderer blocker (must resolve before scene build / RTX rendering)
- GPU **compute** passthrough works (`/dev/dxg` present, `nvidia-smi` OK, CUDA libs injected).
- GPU **Vulkan/rendering** does NOT: the host's `/usr/lib/wsl/lib` has compute libs
  (`libnvidia-ml`, `libnvidia-encode`, `libnvidia-gpucomp`) but **no NVIDIA Vulkan driver lib**
  (`libnvidia-vulkan` / `libGLX_nvidia.so.0`) and **no Vulkan ICD JSON**. The nvidia container
  runtime therefore injects no working Vulkan ICD; the in-container `nvidia_icd.json` points at
  `libGLX_nvidia.so.0` which is incompatible under WSL → `ERROR_INCOMPATIBLE_DRIVER`.
- Likely fixes for the next task (try in order): (1) update the Windows NVIDIA driver to a build
  that ships the WSL Vulkan passthrough lib in `/usr/lib/wsl/lib`, then bind-mount `/usr/lib/wsl`
  into the container; (2) confirm `dzn`/dxg Vulkan is present on host (`vulkaninfo` not installed
  here). Headless **synthetic-data / physics-only** workflows that don't need RTX rendering may
  still run, but any rendered scene needs Vulkan fixed first.

### Recommended `docker run` for the scene-build task
```bash
docker run --name isaac-sim --rm --gpus all \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  --network=host \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  nvcr.io/nvidia/isaac-sim:4.5.0 \
  ./runheadless.sh -v
# Cache dirs already created on device: ~/docker/isaac-sim/{cache/{kit,ov,pip,glcache,computecache},logs,data,documents}
# ⚠️ Will hit the Vulkan ERROR_INCOMPATIBLE_DRIVER above until the WSL Vulkan driver is resolved.
# For python entry use a script file: ./python.sh /path/to/script.py  (NOT python.sh -c).
```

