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

### Deployment (docker) — wheel install

The images do **not** bind-mount voxedge. Every device Dockerfile
(`deploy/docker/Dockerfile.jetson{,.slim}`, `Dockerfile.rk`, `Dockerfile.rpi`)
`pip install`s a pre-built wheel staged at
`deploy/wheels/voxedge-0.0.1a0-py3-none-any.whl`. The thin "diff" images
(`Dockerfile.*.voxedge-patch`) `--force-reinstall` just that wheel onto a base
image for fast Python-only iteration.

**The wheel is committed to git** (a ~200KB pure-Python build artifact of
`../voxedge`). That's deliberate: the device images have no `git` (so a
`pip install git+https://…` isn't an option) and we don't publish to PyPI, so
committing the wheel means a fresh checkout can build/deploy any image with **no
"rebuild the wheel first" step**. `deploy/wheels/` is otherwise git-ignored;
only the wheel + its `voxedge.BUILD.txt` are tracked (see `.gitignore`).

**When `../voxedge` changes, rebuild AND commit the wheel:**

```bash
scripts/build_voxedge_wheel.sh        # regenerates wheel + voxedge.BUILD.txt
git add -f deploy/wheels/voxedge-0.0.1a0-py3-none-any.whl deploy/wheels/voxedge.BUILD.txt
git commit -m "deps(voxedge): rebuild wheel @<sha>"
```

`voxedge.BUILD.txt` records the source git SHA / dirty flag / build date, so you
can always tell which voxedge commit the committed wheel came from (and CI/review
can assert it matches `../voxedge`).

> The wheel filename is version-stable (`0.0.1a0`, pinned in
> `voxedge/pyproject.toml`) so the Dockerfiles never need editing on a rebuild;
> provenance lives in `voxedge.BUILD.txt`, not the filename.

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
