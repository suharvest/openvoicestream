# Voice rebot-arm app (`voice_rebot_arm`)

Voice agent for the reBot B601-DM arm: same wake-word → ASR → LLM
tool-calling → arm wiring as `voice_arm`, with the `rebot_arm` actuator
backend, a force-control gripper, camera-guided grasp (Phase B), and a live
dashboard.

Source: `app.py` (`VoiceRebotArmApp` / `App`) · `config.yaml` · `actions.yaml`
· `rebot_actuator.py` · `grasp_plugin.py` / `grasp_service.py` /
`perception/` · `dashboard_plugin.py`

## What's in this app

- **rebot_arm actuator** — serial channel to the B601-DM; late attach works
  (plug the arm in after the agent starts), disconnects are tracked by
  generation, a dead gripper is surfaced instead of silently printed.
- **Camera-guided grasp** (Phase B) — `GraspPlugin` loads camera + perception
  (onnxruntime, camera SDK) lazily on first grasp; the LLM dispatches a grasp
  and the perception pipeline closes the loop. The LLM round-trip on the
  dispatch result is deliberately skipped — the grasp result is spoken, not
  re-reasoned.
- **Dashboard** — arm state + grasp feedback over the shared dashboard bus.

## Deployment matrix

| Target | Compose | Notes |
|---|---|---|
| Jetson (rebot demo) | [`deploy/docker-compose.jetson-rebot.yml`](../../../../deploy/docker-compose.jetson-rebot.yml) | live `/dev` mount (not a snapshot) so the arm can attach after start; GPU onnxruntime for perception |

Robustness fixes are covered by `agent/tests/test_arm_late_attach.py`,
`test_arm_tool_response_mode.py`, `test_gripper_health_surfaced.py` (landed
via #72).

## Run

```bash
uv run ovs-agent run voice_rebot_arm --config agent/ovs_agent/apps/voice_rebot_arm/config.yaml
```

Requires: SLV speech service, an OpenAI-compatible LLM, the B601-DM on USB
(its serial channel resolves like the SO-ARM's port), and a camera for grasp
mode.

## Measured results

Not measured at app level. Command-acceptance rate and wake→motion latency
are unrecorded; do not quote SLV backend benchmarks as app-level numbers.

## Known limitations

- Grasp perception loads its first model lazily — expect a one-time delay on
  the first grasp after boot.
- English trigger set in the demo prompt keeps the system prompt within the
  4K headroom budget; adding triggers costs that budget.
