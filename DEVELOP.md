# Development setup

> For the full picture of how the three repos fit together, read
> [ARCHITECTURE.md](ARCHITECTURE.md). This file is the practical "get a dev box
> working" checklist.

## The `voxedge` dependency (open-core split, 2026-05-30)

The voice library was extracted into its own repo at `../voxedge`
(`/Users/harvest/project/voxedge`). The product (`server/` + `agent/`) imports
`voxedge.*` but does **not** vendor it, so a fresh checkout has no `voxedge` on
`sys.path` and will fail to import until you install it.

### Local dev — editable install

```bash
scripts/dev-setup.sh        # installs voxedge editable + server reqs + agent[dev]
```

or by hand:

```bash
uv pip install -e ../voxedge
```

`import voxedge` then resolves to the standalone repo. The product's backend
registry (`server/core/asr_backend.py` / `tts_backend.py`) points at
`voxedge.backends.*`; `server/core/voxedge_backend_config.py` builds each
backend's config from env/profile (voxedge backends are env-free).

> **The agent now imports voxedge too (turn-driver unification, 2026-06).**
> `agent/ovs_agent/tools/runner.py` imports `voxedge.engine.turn_driver` at module
> load — both loop modes share one pump — so `voxedge` is a declared dep in
> `agent/pyproject.toml` (`[tool.uv.sources]` editable path). Bare `voxedge` is
> numpy-only, so this is cheap. **Deployment note:** the agent *images*
> (`voice-rebot-arm`, `voice-arm`) must therefore also ship voxedge; the
> production server-loop deployment still runs an older agent image without it
> (server-loop never calls the agent's pump), so rolling this to a device is a
> separate image rebuild — see `docs/plans/turn-driver-unification.md`.

### Deployment (docker) — voxedge comes from PyPI

The images do **not** bind-mount voxedge and do **not** carry a wheel from this
repo. Every device Dockerfile installs an exact published version:

```dockerfile
ARG VOXEDGE_VERSION=0.0.9a0
RUN pip install "voxedge==${VOXEDGE_VERSION}"
```

The same version is pinned in `server/requirements.txt` and used by CI. **Bump
all of them together** — `server/requirements.txt`, every Dockerfile's
`VOXEDGE_VERSION`, `.github/workflows/ci.yml`, and `EXPECTED_VERSION` in
`scripts/build_voxedge_wheel.sh` — otherwise the build-time version check fails.
`deploy/IMAGE-TAGS.md` is a record of images already built; it is history, not a
pin to update.

**Publishing is a prerequisite, not a follow-up.** If the pin names a version
that is not on PyPI yet, Docker and CI fail on purpose rather than falling back
to an older build. Publish first, then merge the pin.

#### Trying a voxedge change on hardware before publishing

```bash
scripts/build_voxedge_wheel.sh   # -> deploy/wheels/voxedge-<version>.whl
```

This is a **qualification wheel only**: install it onto a device by hand to test
an unreleased change. `deploy/wheels/` is git-ignored — **never commit the
wheel**. Commit the voxedge change in its own repo, publish, then bump the pin
here.

`voxedge.BUILD.txt` beside the wheel records the source SHA and a clean/dirty
flag, so you can always tell which voxedge commit a qualification wheel came
from. Build from a committed tree: a dirty tree is flagged and not reproducible.

## Running things locally

- **No-GPU smoke (mock backends):** see ARCHITECTURE.md → "Run it locally".
- **Server:** `python -m uvicorn server.main:app --port 8000`
- **Agent:** `ovs-agent run multi_mode --config <cfg.yaml>`

## Tests

```bash
pytest tests/                              # server integration tests (~175)
uv run --project agent pytest agent/tests/ # agent framework tests (~660; agent has its own venv)
( cd ../voxedge && pytest )                # library tests (~225, mock-based, no GPU)
```

> Agent tests run in the agent's own venv (`agent/.venv`), so use
> `uv run --project agent`. `agent/tests/e2e/` needs a live SLV and the
> `rebot`/`arm` optional extras (cv2/onnxruntime); skip those off-device.
