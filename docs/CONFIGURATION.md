# Configuration

> **TL;DR — the one path you should use:** pick a **profile** with `OVS_PROFILE`.
> A profile fully describes a device's ASR/TTS backends, model paths, and
> execution policy. You almost never set individual model env vars by hand —
> the profile sets them for you. Everything below is detail for when you need it.

## The model: profiles set env, env can override

Configuration is plain environment variables, but you rarely write them
directly. A **profile** (`configs/profiles/<name>.json`) is a bundle of env
defaults plus the backend selection. At startup `server/core/profile_loader.py`
applies the chosen profile's `env` block, then individual env vars you set
yourself take precedence.

```
 OVS_PROFILE=jetson-zh-en
        │
        ▼  profile_loader applies configs/profiles/jetson-zh-en.json
 ┌──────────────────────────────────────────────┐
 │ asr_backend / tts_backend  →  which voxedge   │
 │ env: { LANGUAGE_MODE, MODEL_DIR, … }          │
 │ execution_policy           →  coordinator mode │
 │ required_engines           →  artifact check   │
 └──────────────────────────────────────────────┘
        │
        ▼  operator-set env vars win (snapshotted at import)
 effective configuration
```

### Precedence (highest first)

1. **Operator/shell/`.env` env vars** — snapshotted at import as "operator-owned"
   and never overwritten by a profile (`_snapshot_operator_keys`).
2. **Profile `env` block** — applied for any key the operator did not set.
3. **Hardcoded defaults** in the backend modules.

> **Guardrail:** a handful of *steering* keys (the ones that pick a backend or
> model family) are treated as **hard-mismatch**: if your env disagrees with the
> selected profile, the server fails loudly at boot rather than silently letting
> "operator wins" pick a broken combo. This is intentional — it prevents
> "profile says Qwen3 ASR but env forced Paraformer paths" foot-guns.

## Choosing a profile

```bash
OVS_PROFILE=jetson-zh-en        # selection: file stem under configs/profiles/
# or OVS_PROFILE_JSON=/abs/path/to/custom.json for an out-of-tree profile
```

The compose files in `deploy/` set `OVS_PROFILE` per device. The full list of
shipped profiles is in the README; a profile looks like:

```json
{
  "name": "jetson-zh-en",
  "asr_backend": "jetson.paraformer_trt",
  "tts_backend": "jetson.matcha_trt",
  "execution_policy": { "mode": "serialized", "shared_resource": "gpu" },
  "env": { "LANGUAGE_MODE": "zh_en", "MODEL_DIR": "/opt/models", … },
  "required_engines": [ … ]   // checked / auto-downloaded at boot
}
```

To add a device variant, copy the closest profile and edit `env` +
`{asr,tts}_backend`. You do not touch Python.

## Operator-facing env vars

These are the knobs you might actually set. **Everything else** (the
`PARAFORMER_*`, `MATCHA_*`, `MOSS_*`, `RK_ARTIFACT_*`, `SENSEVOICE_*`, … model
paths) is **profile-managed — leave it alone** unless you are authoring a
profile.

| Var | Purpose | Typical |
|---|---|---|
| `OVS_PROFILE` | Select a device profile (file stem) | `jetson-zh-en` |
| `OVS_PROFILE_JSON` | Select an out-of-tree profile by path | _(unset)_ |
| `LANGUAGE_MODE` | Language family | `zh_en`, `en`, `multilanguage` |
| `MODEL_DIR` | Root for downloaded model artifacts | `/opt/models` |
| `OVS_MAX_CONCURRENT_SESSIONS` | Session admission ceiling | `1`–`2` |
| `OVS_V2V_ENGINE` | `/v2v` orchestrator: `voxedge` enables the library engine | `voxedge` |
| `OVS_V2V_SERVER_LOOP` | Run the LLM+tool loop server-side (needs `OVS_V2V_ENGINE=voxedge`) | `1` |
| `OVS_AUTO_DOWNLOAD_ARTIFACTS` | Fetch missing engines/models at boot | `1` |
| `HF_ENDPOINT` | HuggingFace mirror (devices behind GFW) | `https://hf-mirror.com` |
| `LAZY_TTS` | Defer TTS model load until first request | `0`/`1` |
| `EDGE_LLM_BASE_URL` | External LLM service for the conversation loop | host:port |

> **The two flags that trip everyone up:** the server-loop voice path needs
> **both** `OVS_V2V_ENGINE=voxedge` **and** `OVS_V2V_SERVER_LOOP=1`. One without
> the other silently falls back to the legacy path.

## Leaf composition (optional, opt-in — ignore unless you need it)

There is a second, newer config layer under `configs/leaves/` ("leaf
composition": `devices.yaml`, `models.yaml`, per-device ASR/TTS memory budgets).
It composes a backend stack from `capability × backend × device × concurrency`
and validates the result fits the device memory budget.

**It is OFF by default.** Flat profiles are the supported path. Leaf composition
only activates when a profile opts in (a `composition` key) and is gated by
`server/core/composition_boot.py`. If you are not deliberately working on it,
treat `configs/leaves/` as not-present. It exists to eventually replace
hand-maintained per-device memory tuning; until then, **do not mix the two** —
pick a flat profile.
