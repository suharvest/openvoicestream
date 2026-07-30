# Building the RK1828 LLM service image

OpenAI-compatible LLM endpoint backed by the RK1828 PCIe NPU accelerator card.
The repo already had an LLM chat service for Jetson (`edge-llm-chat-service`);
this is the Rockchip counterpart, which did not exist.

The RK1828 uses **RKNN3 / RKLLM3 V1.0.4** — a different toolchain from the
RKNN2 in the voice image. The two cannot share a base image.

## What is in git and what is not

| | |
|---|---|
| In git | `rk1828_llm_server.py` (the OpenAI shim), `patches/Qwen3-cpp-main.cc` (the server-mode worker source), `artifacts.json`, `entrypoint.sh`, `Dockerfile` |
| **Not** in git | `deploy/rk1828-runtime/` — the compiled worker binary + `librknn3_api.so`. Same policy as `deploy/rk-runtime/`: a staged build artifact with a `MANIFEST.json` for provenance |
| **Never** anywhere | the model artifacts (3.2 GB). Pulled at runtime, see below |

## Step 1 — build the worker binary (on an RK1828 host)

The worker is a server-mode conversion of the RKNN3 model-zoo Qwen3 demo: init
once, then read framed requests on stdin and stream token frames on stdout. The
upstream demo is a one-shot CLI that takes the prompt as `argv`.

`patches/Qwen3-cpp-main.cc` is the converted source. Drop it over the model zoo's
copy and build **only** via the zoo's own script:

```bash
cd ~/rk1828/rknn3-model-zoo
cp <repo>/services/rk1828-llm/patches/Qwen3-cpp-main.cc examples/Qwen3/cpp/main.cc
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3
```

**Never call cmake/make directly** — a bare cmake build produces ABI-incompatible
artifacts.

Protocol implemented by the converted worker (do not redesign it):

* argv: `<model_dir> [--core-mask <hex>] [--max-context <n>] [--device-id <id>] -`
  — the trailing `-` selects server mode; the original 6-arg one-shot path still works.
* stderr: `READY 1` handshake plus all diagnostics.
* stdin: one request per line, `<max_new_tokens>\t<escaped prompt>`.
* stdout: `[uint32 LE len][utf8 token]` per token, `[uint32 LE 0xFFFFFFFE]` as the
  per-request EOS sentinel.

Three implementation details that are load-bearing:

1. **stdout must carry only protocol frames.** `dup(STDOUT)→frame_fd` then
   `dup2(stderr, stdout)` **before** model init — the RKNN runtime prints during
   init, and any of that text on the frame channel desyncs the reader into a
   multi-gigabyte read.
2. **`rknn3_session_clear_kvcache(..., RKNN3_KVCACHE_CLEAR_ALL)` after EVERY
   request, including error and empty-prompt paths.** A skipped clear leaves dirty
   KV that accumulates until it overflows `max_context` and aborts at runtime.
3. **Do not hold the worker lock inside the SSE generator.** If a client
   disconnects mid-stream the generator is abandoned still holding the lock with
   unread frames queued; the next request then blocks forever, or desyncs onto the
   previous request's tokens. A dedicated thread must own the lock and always
   drain through the EOS sentinel regardless of whether the HTTP consumer is
   still there. (Same class as the `wanted 1852143441 bytes` desync noted in
   rkvoice-stream's `docs/rk1828-package-integration.md` §12.2.)

## Step 2 — stage the runtime

```bash
mkdir -p deploy/rk1828-runtime/lib
# from the device, after a successful build:
#   install/rk3588_linux_aarch64/rknn_Qwen3_demo/rknn_qwen3_demo
#   install/rk3588_linux_aarch64/rknn_Qwen3_demo/lib/librknn3_api.so
```

Layout matters: the binary's rpath is `$ORIGIN/lib`, so `librknn3_api.so` must sit
in a `lib/` directory **next to** the binary. Write a `MANIFEST.json` recording
sizes, md5s, the SDK version (V1.0.4) and the source SHA of `main.cc`, the same
way `deploy/rk-runtime/MANIFEST.json` does.

## Step 3 — build the image

Build context is the **repo root**, because the runtime is staged under `deploy/`:

```bash
docker build -f services/rk1828-llm/Dockerfile -t edge-llm-rk1828:$(date +%Y%m%d) .
```

Native arm64. Resulting image is ~200 MB (no models).

## Runtime

Models are pulled on first start into `RK1828_MODEL_DIR` — mount a named volume
so a container replacement does not re-download 3.2 GB.

| Env | Default | Notes |
|---|---|---|
| `RK1828_ARTIFACT_AUTO_DOWNLOAD` | `1` | `0` = expect models already present |
| `RK1828_ARTIFACT_REPO_ID` | `harvestsu/seeed-local-voice-rk-artifacts` | |
| `RK1828_ARTIFACT_PREFIX` | `rk1828/opt/llm/qwen3-4b` | matches the repo's SoC-prefixed layout |
| `HF_ENDPOINT` | `https://huggingface.co` | set a mirror behind the firewall; mirrors may lag |
| `RK1828_MAX_CONTEXT` | `8192` | verified; a runtime parameter, no re-export needed |
| `RK1828_CORE_MASK` | `ff` | 8 cores |
| `RK1828_PORT` | `1828` | |

The entrypoint verifies file SIZES, not just existence — a half-finished download
would otherwise be treated as present and fail model init with something far less
obvious. It also refuses to start on an incomplete set, because the four files are
a matched export and a mismatched pair surfaces as a firmware-level `ACK_FAIL`
that looks like a hardware fault.

## Host prerequisites — the image is NOT self-contained

The card's kernel driver and firmware live on the host:

* `pcie_rkep` module loaded. **It does not persist across reboot** — make it
  durable on the host.
* `rknn3.service` active; it reflashes the EP firmware at host boot.

Acceptance before starting the container:

```bash
lspci | grep 182a                      # expect 0001:11:00.0 ... Device 182a
systemctl is-active rknn3.service      # expect active
ls /dev/pcie-rkep-*                    # expect the char device
```

The container needs `privileged: true` and the char device. The entrypoint fails
fast with a pointed message if the device is not visible.

## Single-EP exclusivity

The card has ONE ~5 GB context, so **large models are mutually exclusive**.
Qwen3-4B at 8192 tokens uses an estimated ~3,640 MB of 5,120 MB (weights 2,432 MB
plus KV 8192 × 144 KB = 1,208 MB). Consequences:

* An RK1828-hosted TTS cannot run at the same time as this. That is why the voice
  image runs TTS on the RK3588's own NPU.
* There is **no way to measure EP memory**: `rknn-smi` is non-functional on this
  host (fails as root, fails when the EP is idle — suspected host/EP firmware
  version skew, `rc_cc_version=30301` vs `ep_cc_version=30201`). Model-load
  success is the only available signal, and **failed loads degrade the EP from 8
  cores to 4**, so do not probe by trial and error.
* **Never run `rknn-smi reset`** — it can wedge the card into a boot state that a
  host reboot may not recover, and the card has its own 12 V supply so it does not
  power-cycle with the host.
