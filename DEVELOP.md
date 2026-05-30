# Development setup

## voxedge dependency (P3 split, 2026-05-30)

The voice library was extracted into its own repo at `../voxedge`
(`/Users/harvest/project/voxedge`). The product depends on it; install editable:

```bash
uv pip install -e ../voxedge        # or the absolute path
```

`import voxedge` then resolves to the standalone repo. The product's backend
registry (`app/core/asr_backend.py` / `tts_backend.py`) points at
`voxedge.backends.*`; `app/core/voxedge_backend_config.py` builds each backend's
config from env/profile (voxedge backends are env-free).

### Deployment (docker) — TODO, deferred
The jetson images currently expect the voice library under `/opt/speech/voxedge`
(previously the in-repo `voxedge/`). Until the image build is updated to
`pip install`/COPY the `voxedge` package, bind-mount or copy `../voxedge/voxedge`
to `/opt/speech/voxedge`. Track this before the next deploy.
