# Isaac Sim 4.5.0 Pull + Headless GPU Smoke (wsl2-local)

Date: 2026-06-14
Device: `wsl2-local` (WSL2, NVIDIA GeForce RTX 3060, host driver 591.74)

## Summary

- **Pull: COMPLETE.** `nvcr.io/nvidia/isaac-sim:4.5.0` fully present, 22.6 GB.
- **Python: OK** (`ISAAC_PY_OK` printed).
- **GPU/CUDA visible in container: OK** — `nvidia-smi -L` shows the RTX 3060.
- **`SimulationApp` import: FAILED (segfault).** Root cause is **NOT** the image — it is a
  WSL2 host gap: the NVIDIA **Vulkan** driver/ICD is not exposed to WSL, so Isaac Sim's
  renderer init crashes (`Vulkan 1.1 is not supported`). CUDA works (via `/dev/dxg`),
  Vulkan does not.

## Evidence (raw)

### `docker images | grep isaac-sim`
```
nvcr.io/nvidia/isaac-sim:4.5.0      c2f47dc82a77   22.6GB
```

### `nvidia-smi -L` from inside the container (`--gpus all`)
```
GPU 0: NVIDIA GeForce RTX 3060 (UUID: GPU-fd916fee-82e8-3a13-a35f-6933ecf385c2)
```

### Smoke stdout — `python.sh -c "import isaacsim; from isaacsim import SimulationApp"`
`ISAAC_PY_OK` printed (Python interpreter works). Kit extensions begin startup, then the
Vulkan renderer init fails and the process segfaults during `from isaacsim import SimulationApp`:
```
[ext: omni.kit.renderer.init-0.0.0] startup
[Warning] [carb.windowing-glfw.plugin] GLFW initialization failed.
[Warning] [omni.platforminfo.plugin] failed to open the default display.  Can't verify X Server version.
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed. Vulkan 1.1 is not supported, or your driver requires an update.
[Error] [gpu.foundation.plugin] carb::graphics::createInstance failed.
...
[Fatal] [carb.crashreporter-breakpad.plugin] ... libpython3.10.so ... kit!_start
Segmentation fault (core dumped)
```
`SIMAPP_IMPORTABLE` was **not** reached.

### Root-cause probe on WSL host
```
# /usr/lib/wsl/lib NVIDIA vulkan lib:
NO_WSL_VULKAN_LIB              # only libnvidia-encode/gpucomp/ml/ngx/opticalflow present, no libnvidia-vulkan*
# /dev/dxg:
crw-rw-rw- 1 root root 10, 127 /dev/dxg     # present -> CUDA path works
# host nvidia-smi:
NVIDIA GeForce RTX 3060, 591.74
# NVIDIA Vulkan ICD:
NO_NVIDIA_ICD                  # /usr/share/vulkan/icd.d has only asahi/intel/radeon/nouveau/llvmpipe/virtio/gfxstream, no nvidia_icd.json
```

## Interpretation

The image is good and CUDA is fully wired through WSL. Isaac Sim's `SimulationApp`
eagerly boots the Omniverse Kit renderer (Vulkan) even in headless mode; this WSL2 host
provides no NVIDIA Vulkan ICD (`nvidia_icd.json`) and no `libnvidia-vulkan*` under
`/usr/lib/wsl/lib`, so `vkCreateInstance` returns `ERROR_INCOMPATIBLE_DRIVER` and Kit
crashes. This is an environment/driver provisioning gap, not a pull or image defect.

### To unblock the renderer (for the later scene-build task — out of scope here)
Provide an NVIDIA Vulkan ICD to the container. Typical WSL2 fixes:
- Update the **Windows** NVIDIA driver to a version that ships the WSL Vulkan stack, so
  `libnvidia-vulkan*` / `dzn`/`d3d12` Vulkan support appears under `/usr/lib/wsl/lib`.
- Install `mesa-vulkan-drivers` + `vulkan-tools` in WSL and verify `vulkaninfo --summary`
  reports a device with `apiVersion >= 1.1` before retrying Isaac Sim.
- Then mount the host Vulkan loader/ICD into the container (`-v /usr/lib/wsl/lib:/usr/lib/wsl/lib`,
  `VK_ICD_FILENAMES=...`) or rely on the NVIDIA Container Toolkit's vulkan capability.

Verify with, inside the container:
`vulkaninfo --summary` should list the RTX 3060 with `apiVersion 1.1+` before `SimulationApp` will init.

## Recommended `docker run` line (for the later scene-build task)

```bash
docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -v ~/isaac-cache/kit:/root/.cache \
  -v ~/isaac-cache/ov:/root/.nvidia-omniverse \
  nvcr.io/nvidia/isaac-sim:4.5.0 \
  /isaac-sim/python.sh /path/to/your_script.py
```

Notes:
- Entrypoint enforces EULA even for `ls`; always pass `ACCEPT_EULA=Y` (and
  `OMNI_KIT_ACCEPT_EULA=YES`). For non-Kit commands use `--entrypoint`.
- Cache mounts on `~/isaac-cache/{kit,ov}` persist shader/asset caches across runs.
- **Renderer (Vulkan) currently non-functional on this WSL2 host** — any script that
  constructs `SimulationApp` will segfault until the Vulkan ICD gap above is resolved.
  CUDA-only / non-renderer workloads are fine.
- `/isaac-sim` layout confirmed: `python.sh`, `isaac-sim.sh`, `apps/`, `exts/`, `kit/`,
  `extension_examples/`, etc.

## Pull + smoke result

Independently re-verified by the orchestrating thread on 2026-06-14 (not just trusting the
sub-agent summary). All outputs below are raw from `wsl2-local`.

**Final image tag + size**
```
$ docker images | grep isaac-sim
nvcr.io/nvidia/isaac-sim:4.5.0      c2f47dc82a77       22.6GB         7.34GB
$ ps -ef | grep 'docker pull nvcr.io/nvidia/isaac-sim' | grep -v grep || echo NO_PULL_PROC
NO_PULL_PROC
$ df -h ~ | tail -1
/dev/sdc       1007G  785G  172G  83% /
```
Pull COMPLETE, process gone, 172 GB free on /dev/sdc. Image ID `c2f47dc82a77`, 22.6 GB.

**GPU visible in container (`--gpus all`)**
```
$ docker run --rm --gpus all -e ACCEPT_EULA=Y --entrypoint nvidia-smi nvcr.io/nvidia/isaac-sim:4.5.0 -L
GPU 0: NVIDIA GeForce RTX 3060 (UUID: GPU-fd916fee-82e8-3a13-a35f-6933ecf385c2)
```
(Default entrypoint enforces EULA; use `--entrypoint` for non-Kit commands like `nvidia-smi`.)

**Import-only smoke (raw, with cache mounts)**
- `ISAAC_PY_OK` printed; Kit boots all extensions; Warp initializes CUDA on the RTX 3060
  (`"cuda:0" : "NVIDIA GeForce RTX 3060" (12 GiB, sm_86)`); URDF importer loads; then:
```
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed. Vulkan 1.1 is not supported, or your driver requires an update.
...
Segmentation fault (core dumped)        # process exit code 139
```
- `SIMAPP_IMPORTABLE` NOT reached. Confirmed root cause = WSL2 NVIDIA Vulkan ICD gap
  (`/usr/lib/wsl/lib` has NO `libnvidia-vulkan*`; `/usr/share/vulkan/icd.d/` has only
  asahi/intel/radeon/nouveau/llvmpipe/virtio/gfxstream — NO `nvidia_icd.json`). CUDA path
  (via `/dev/dxg`) works; renderer does not. Image/pull are NOT at fault.

**Recommended `docker run` line (later scene-build task)**
```bash
docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -v ~/isaac-cache/kit:/root/.cache \
  -v ~/isaac-cache/ov:/root/.nvidia-omniverse \
  nvcr.io/nvidia/isaac-sim:4.5.0 \
  /isaac-sim/python.sh /path/to/your_script.py
```
Blocker for any `SimulationApp`-based scene: provision the WSL2 NVIDIA Vulkan stack first
(Windows driver update exposing `libnvidia-vulkan*`, or mount an ICD), then confirm
`vulkaninfo --summary` shows the RTX 3060 at apiVersion >= 1.1 inside the container.
