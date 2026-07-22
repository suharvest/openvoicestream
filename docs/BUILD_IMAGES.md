# Building OpenVoiceStream Images

How to build & publish the container images after the 2026-06 restructure
(`app/`→`server/`, `openvoicestream_agent`→`ovs_agent`, SO-ARM driver moved
into its app). One-time reference for whoever cuts the images.

There are **three** images in a full SO-ARM voice deployment:

| image | what it is | source | build host |
|---|---|---|---|
| `…/seeed-local-voice:<dev>-slim` | the **voice service** (ASR/TTS/LLM-loop server) | `seeed-local-voice/server/` + `voxedge` wheel | the target arm64 device |
| `…/voice-arm:<tag>` | the **agent** (SO-ARM voice app) | `seeed-local-voice/agent/` | Jetson (l4t base) |
| `…/edge-llm-chat-service:<tag>` | the LLM container | (not built here — pull from registry) | — |

### v0.9.0 + MOSS, engines on demand

`deploy/docker/Dockerfile.jetson.edgellm-v090-ondemand` builds the image the
`jetson-edgellm-v090-moss` profile expects. Build context is the **repo root**,
and the voxedge wheel has to be sitting there first:

```bash
cp ../voxedge/dist/voxedge-0.0.5a0-py3-none-any.whl .
docker build -f deploy/docker/Dockerfile.jetson.edgellm-v090-ondemand \
  -t sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.9.0-ondemand-<date> .
```

It bakes no engines: the ASR pair + worker + plugin (1.29 GB) come from
`models/edgellm-v090-asr/` on the artifact repo at first boot, the MOSS engines
(1.93 GB) from `models/moss-tts-nano/`. That keeps the image near 1.2 GB instead
of the 6.2 GB of the baked builds, and stops shipping the 1.39 GB of qwen3-TTS
engines this profile never loads.

Because the engines arrive at runtime, `/opt/edgellm-v090` and `/opt/models`
must be writable, and a China-network host needs `HF_ENDPOINT=https://hf-mirror.com`.

Registry: `sensecraft-missionpack.seeed.cn/solution/…` (push from a machine
that's `docker login`'d — the Mac is; the profiling devices are **not**, so
device-built images relay through the Mac: `docker save` → transfer → `docker
load` → `docker push`).

---

## 1. Server slim images (Jetson / RK / RPi)

Slim images bake **no** models/engines — they pull artifacts from Hugging Face
(or the Seeed CDN) on first start. Three device Dockerfiles, each with a
`final-slim` target.

**Prereq — stage the voxedge wheel** into the build context (the Dockerfiles
`COPY deploy/wheels/voxedge-*.whl`):
```bash
cd voxedge && uv build --wheel
cp dist/voxedge-0.0.2a0-py3-none-any.whl ../seeed-local-voice/deploy/wheels/
```

**Build** (run on the matching arm64 device, in the `seeed-local-voice/` build
context; `--sudo` docker on devices):
```bash
# Jetson (Orin) — multilanguage default
docker build -f deploy/docker/Dockerfile.jetson.slim --target final-slim \
  --build-arg LANGUAGE_MODE=multilanguage \
  -t sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-slim-<date> .

# Rockchip (RK3588)
docker build -f deploy/docker/Dockerfile.rk --target final-slim \
  -t sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-slim-<date> .

# Raspberry Pi (CPU / sherpa)
docker build -f deploy/docker/Dockerfile.rpi --target final-slim \
  -t sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rpi-slim-<date> .
```
> After `app→server`, the Dockerfiles `COPY server/ …` and run
> `uvicorn server.main:app`. (`final-thick` bakes engines offline instead.)

**Validate** before pushing (fresh empty models volume so the HF-pull path runs):
```bash
docker run -d --name slim-test --runtime nvidia \
  -e OVS_PROFILE=jetson-qwen3asr-matcha-nx -e HF_ENDPOINT=https://hf-mirror.com \
  -p 18080:8000 -v slim_test_models:/opt/models <image>
# expect logs: "resolved N engine(s)", then  GET /readyz → 200 {"status":"ready"}
docker rm -f slim-test && docker volume rm slim_test_models
```
First boot downloads engines from HF (Jetson ~6 min; the ~1.1 GB qwen3
`llm.engine` dominates). Sizes (content): Jetson ~1.16 GB, RK ~223 MB, RPi ~570 MB.

---

## 2. voice-arm agent image

The SO-ARM voice agent. **Post-migration the code lives in
`seeed-local-voice/agent/`** (the `ovs_agent` framework + `ovs_agent/apps/voice_arm/`
holding the SO-ARM driver). It is **no longer cloned from the public repo** —
build it from a `seeed-local-voice` checkout.

- **Build context**: `seeed-local-voice/agent/`. The Dockerfile should:
  ```dockerfile
  FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0
  COPY agent/ /opt/agent/
  RUN pip install /opt/agent           # installs ovs_agent + its apps
  # + openwakeword models, audio deps, the deploy entrypoint
  ```
  (Drop the old `prepare-build.sh` / `.ovs-cache/` / `ovs_agent_src/` — those
  vendored a clone of the public agent and are obsolete.)
- **Deploy glue stays** in `app_collaboration/solutions/respeaker_flex_soarm/`:
  `solution.yaml`, `docker-compose.yml`, `devices/`, `default_config/`
  (`actions.yaml` pose library + `prompt.yaml`), `entrypoint.sh`. The image
  reads its config from a host-mounted volume; defaults ship in `default_config/`.
- **Launch**: the container runs `ovs-agent run voice_arm` (the `ovs-agent` CLI
  command is unchanged by the package rename). `voice_arm` is discovered as
  `ovs_agent.apps.voice_arm.app:App`.
- **A new motor** = a new app under `ovs_agent/apps/` with its own driver that
  calls `register_actuator(...)`; no image/framework change.

```bash
docker build -f deploy/docker/Dockerfile -t \
  sensecraft-missionpack.seeed.cn/solution/voice-arm:<tag> seeed-local-voice/
docker push sensecraft-missionpack.seeed.cn/solution/voice-arm:<tag>
```

---

## 3. Deploy

`docker compose -f app_collaboration/solutions/respeaker_flex_soarm/assets/docker/docker-compose.yml up -d`
brings up the three containers (`seeed-voice`, `edge-llm`, `voice-arm`). Pin
each image via the compose `${*_IMAGE}` env vars. SO-ARM connects over USB
serial (`/dev/ttyACM0`); the reSpeaker Flex is the mic array.

> **Production note**: changes to the robot (`seeed-orin-nx`) need explicit
> authorization + an image backup + rollback tag — never `docker compose down`
> the whole project; use `docker compose up -d <service>` to swap one image.
