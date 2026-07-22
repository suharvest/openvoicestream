# seeed-local-voice — TOP layer (server + "OpenVoiceStream" agent)

**Analyzed ref:** `main` @ **4532ed0** ("feat(rk1828): Gemma-4 multimodal …").
**AST tool:** Python `ast` module (custom extractor) over the canonical source packages
(`server/`, `agent/ovs_agent/` — excluding the vendored `.venv`, `third_party/`, `vendor/` which inflate the raw .py count to ~7k).

Two products live here:
1. **the server** (`server/`) — the WebSocket/HTTP voice service that loads ASR/TTS/LLM backends.
2. **the agent** (`agent/ovs_agent/`, package "OpenVoiceStream / ovs_agent") — the device-side wake→capture→tool-calling app.

---

## 1. `server/core/` — backend registry + composition + provisioning (the heart for consolidation)

```
backend_manager.py        BackendManager[T] lifecycle (start/shutdown/reload/acquire, WS drain, hot-swap)
asr_backend.py            ASRBackend facade ABC + _ASR_REGISTRY + create_asr_backend()
tts_backend.py            TTSBackend facade ABC + _TTS_REGISTRY + create_tts_backend()
edge_llm_backend.py       EdgeLLMBackend(LLMBackend)  (HTTP/SSE client to the edge-LLM service)
voxedge_backend_config.py build_*_config(profile=...) — builds each voxedge backend's config from the profile
leaf_composition.py       the leaf/device/model registry + CompositionPlan (pure, no I/O)
composition_boot.py       gated glue: apply a profile's `composition` block → env (no-op for flat profiles)
model_downloader.py       ensure_models(...) — profile-driven artifact pull (+ bundle suppression)
profile_loader.py / profile_selector.py / engine_resolver.py / capability_resolver.py
concurrency_capability.py session_limiter.py / asr_session_manager.py / coordinator.py / v2v.py
tts_service.py / tts_runtime.py / tts_speakers.py / speaker_embedding.py / punctuation.py / vad.py
rk_runtime.py / rk_artifacts.py / moss_artifacts.py / qwen3_artifact_downloader.py / hf_artifacts.py
gpu_watchdog.py / metrics.py / admin_auth.py / api_auth.py / deploy_paths.py / worker_io.py
```

### Backend registry — the dispatch tables (asr_backend.py / tts_backend.py)

**`_ASR_REGISTRY: Dict[str, (module_path, class_name)]`** (asr_backend.py:178):
| key | module | class |
|---|---|---|
| `jetson.trt_edge_llm` | voxedge.backends.jetson.trt_edge_llm_asr | TRTEdgeLLMASRBackend |
| `jetson.paraformer_trt` | voxedge.backends.jetson.paraformer_trt | ParaformerTRTBackend |
| `jetson.sensevoice_trt` | voxedge.backends.jetson.sensevoice_trt | SenseVoiceTRTBackend |
| `cpu.sherpa_asr` | voxedge.backends.sherpa.asr | SherpaASRBackend |
| `rk.asr` | (resolved via build_rk_asr_config; RK backend) | |

**`_TTS_REGISTRY`** (tts_backend.py:145):
| key | module | class |
|---|---|---|
| `jetson.trt_edge_llm` | voxedge.backends.jetson.trt_edge_llm_tts | TRTEdgeLLMTTSBackend |
| `jetson.matcha_trt` | voxedge.backends.jetson.matcha_trt | MatchaTRTBackend |
| `jetson.kokoro_trt` | voxedge.backends.jetson.kokoro_trt | KokoroTRTBackend |
| `jetson.qwen3_trt` | voxedge.backends.jetson.qwen3_trt | Qwen3TRTBackend |
| `jetson.moss_tts_nano` | voxedge.backends.jetson.moss_tts_nano | MossTtsNanoBackend |
| `cpu.sherpa` | voxedge.backends.sherpa.tts | SherpaTTSBackend |
| `rk.tts` | (build_rk_tts_config; RK backend) | |

**Dispatch flow** (`create_{asr,tts}_backend()`):
1. read `ASR_BACKEND`/`TTS_BACKEND` env (a registry key).
2. import `(module_path, class_name)` lazily from voxedge.
3. `if spec == "...": config = voxedge_backend_config.build_*_config(profile=current_profile())` then `cls(config=config)`.

> **Structural fact for consolidation:** seeed owns NO backend *implementations* — every registry value points into
> voxedge. The only seeed-side per-backend code is the `build_*_config()` mappers in `voxedge_backend_config.py`
> and the facade ABCs. RK backends (`rk.asr`/`rk.tts`) are resolved via config builders rather than a static
> registry tuple — slightly asymmetric vs the jetson/cpu entries (worth normalizing in consolidation).

### `BackendManager[T]` (backend_manager.py:102) — lifecycle
- `__init__/start/shutdown`, `state()`, `backend_name()`, `profile_name()`, `is_ready()`, `get_backend_unsafe()`,
  `acquire()` (async ctx mgr, raises `backend_unavailable`), `register_ws`/`unregister_ws`,
  `_wait_for_http_drain`, `_force_close_ws_sessions`, **`reload(...)`** (hot-swap; closes WS 1012, dry-run pre-check).
- Module funcs: `init_backend_managers(...)`, `tts_manager()`, `asr_manager()`. `BackendState` enum.

### Leaf/composition config system
- **`leaf_composition.py`** — pure registry loaded from `configs/leaves/*.yaml`:
  `Leaf`, `Artifacts`, `DeviceSpec`, `ModelSpec`, `ResolvedLeaf`, `CompositionPlan`, `Registry`
  (`device_class`, `resolve_precision` — explicit leaf precision wins else model `default_precision[class]`).
  Funcs: `load_registry`, `validate_composition` (memory headroom / unknown-or-unbuilt leaf / illegal capability pairing),
  `resolve_env` (merge leaf < overrides). Errors: `CompositionError`, `RegistryError`.
- **`composition_boot.py`** — `selected_leaf_ids(composition)`, `apply_composition(profile)`:
  gated behind a profile `composition` block; **flat profiles are byte-for-byte unchanged (strict no-op)**.
  os.environ values WIN over leaf-derived (operator-owned env precedence).
- **`configs/leaves/`** (11 yaml): `models.yaml`, `devices.yaml`, and per-leaf:
  `qwen3-asr-nx`, `qwen3-tts-nx`, `qwen3-tts-base`, `moss-tts-nano`, `matcha-tts`, `kokoro-tts`,
  `sensevoice-asr`, `paraformer-asr`. **Precision is a leaf attribute** (flip models.yaml to switch).
- **`configs/profiles/`** — flat profiles (e.g. the new untracked `jetson-edgellm-v080-moss.json`).

### `model_downloader.ensure_models(...)`
- Routing: **profile-driven first** (profile.asr_backend/tts_backend), language_mode second.
- `_BUNDLE_MODEL_BACKEND` map suppresses over-fetch (a Kokoro profile must not pull Matcha; a Qwen3 profile
  must not pull Paraformer). `_ensure_qwen3_artifacts`, `_download_and_extract`, `_detect_tar_mode`.

## 2. `server/main.py` + entrypoints
- `server/main.py` — FastAPI/WS app; wires `init_backend_managers`, the `/asr` `/tts` `/v2v` endpoints,
  the server-loop (`OVS_V2V_SERVER_LOOP=1`, `OVS_V2V_ENGINE=voxedge` → `voxedge.engine.turn_driver.run_turn`).
- 90+ tests in `server/tests/` cover registry/composition/hot-swap/v2v race conditions.

## 3. `agent/ovs_agent/` — the "OpenVoiceStream" device agent (105 .py)

```
ovs_agent/
  cli.py app_base.py app_mode.py        entrypoints (wake→capture→dispatch app shells)
  session.py state.py event_bus.py      session/state machine
  audio_io.py vad.py wake_source.py     audio capture + wake-word
  slv_client.py protocol.py             talks to the seeed server (SLV = server local voice) over WS
  config.py plugin.py
  llm/                                  local LLM client adapters
  tools/  runner.py + @tool decorators  TOOL-CALLING pump  → imports voxedge turn_driver
  modes/  wake_sources/  audio/  translator/  streaming_translate/
  actuators/                            Actuator ABC + empty factory (arms register in apps/)
  apps/   voice_arm/ voice_rebot_arm/ companion_robot/ live_caption/ simul_interpret/ translator/ multi_mode/
  plugins/ (+ static/)
```

### `tools/runner.py` — the tool-calling shim (KEY cross-layer edge)
- `from voxedge.engine.turn_driver import run_turn` (line 33).
- `class ToolCallCtx`, `_ToolCallAcc`, `_AgentRegistryAdapter`, `_AgentMessageSink`, `_TokenTextSink`, `_ShimLLM`.
- `async def stream_with_tools(...)` — thin shim over `run_turn`; the agent's local tool pump and the server's
  server-loop now share the SAME `run_turn` (unification). `_open_stream`, `_wrap_tool_{started,completed}`,
  `_wrap_completion_text`.
- Actuators (SO-ARM, reBot B601-DM) self-register in `apps/voice_{arm,rebot_arm}/`; a new motor = a new app.

## 4. `deploy/` — the deployment/overlay structure (NOT a build system)
```
deploy/docker/      Dockerfile.jetson(.slim) + many `*-overlay` / `*-patch` Dockerfiles:
                    voxedge-latest-overlay, agent-latest-overlay, asr-multiturn-overlay,
                    rebot-{graspangle,inject,micgate}-overlay, slv-{asrfix,preroll}-overlay,
                    tts-phase-b-overlay, v2v-multiturn-overlay …  (Dockerfile.rk / .rpi too)
deploy/asr-worker-v080/  qwen3_asr_worker.cpp + stripLangTag-firstword-fix.patch  (the ASR C++ worker source kept here)
deploy/artifacts/   manifests (kokoro_trt / moss / rk)
deploy/docker-compose.{jetson-rebot,radxa,rk,rpi}.yml + docker/_patch_*.py
```
> Overlays patch CHANGED files onto a baked production base image (canonical Dockerfile lacks openwakeword/entrypoint
> → cannot full-rebuild). Production config is `.tmpl`-rendered, not baked. Deploy wheels are gitignored (pip-installed).

## 5. Cross-project dependency edges (who imports voxedge)
| seeed file | imports from voxedge |
|---|---|
| `server/core/edge_llm_backend.py` | `backends.base.{LLMBackend, LLMEvent}` |
| `server/core/voxedge_backend_config.py` | builds configs for jetson/sherpa/rk backends |
| `server/core/asr_session_manager.py` | re-exports `engine.asr_session_manager` |
| `server/core/{punctuation,speaker_embedding}.py` | re-export `capabilities.*` (single source of truth) |
| `server/main.py` | `engine.turn_driver.run_turn`, `transport.base.WebSocketTransport`, tts/qwen3/kokoro/matcha modules |
| **`agent/ovs_agent/tools/runner.py`** | **`engine.turn_driver.run_turn`** (top layer → middle layer, direct) |

The backend impl modules (`backends/jetson/*`, `backends/sherpa/*`) are imported **lazily** via the
registry tuples — they are never top-level imports in seeed (keeps the server importable without CUDA).
