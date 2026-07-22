# MTP-on vs vanilla-GDN unified-memory delta (orin-nx, v0.8.0 Qwen3.5-4B)
Started: measurement in progress

## Host guard
Linux 5.15.148-tegra aarch64 / orinnx — PASS

## Steps
- [ ] baseline snapshot docker ps
- [ ] stop edge-llm-chat-service + seeed-voice (leave translator)
- [ ] BASE_MB
- [ ] vanilla run + VAN_PEAK_MB
- [ ] MTP run + MTP_PEAK_MB
- [ ] DELTA
- [ ] RESTORE

## Baseline snapshot (raw)
docker ps before:
- seeed-voice Up (8621) prod-unified-v8
- translator Up (healthy, 9001) translator-cuda-jetson-v2  [LEAVE RUNNING = constant baseline]
- edge-llm-chat-service Up healthy (8000) openai-compat-v6
- industrial-security-demo Restarting (PRE-EXISTING, not touched)
- mtp-throwaway Exited(137) [leftover OOM from prior phase — removing]

health 8000: speculative_decoding:false (current prod LLM, engines-8192 — NOT our test)
health 8621: seeed-voice tts/asr OK

disk: 233G/220G used 3.1G free 99% (engines mounted, no copy)

To restore: docker start edge-llm-chat-service seeed-voice

## BASE_MB
tegrastats raw (translator-only baseline):
RAM 4629/15656MB x5 (identical)
BASE_MB = 4629  (total 15656)

## VANILLA (engines-v080-gdn, MTP off)
NOTE: image needs host lib mounts (libnvinfer.so.10) — replicated prod mounts:
  /host-libs /host-cuda /host-cudla /host-nvidia-libs + LD_LIBRARY_PATH
/health: {"status":"healthy","model":"/engines","speculative_decoding":false}  <-- spec=false OK
6 inferences (short x2, long-400tok, code, medium x2), 237 tegrastats@500ms samples
VAN_PEAK_MB = 7945
Vanilla footprint = 7945 - 4629 = 3316 MB

## MTP (engines-v080-gdn-mtp, MTP on, baked 1/3/4)
symlinks created: llm.engine->eagle_base.engine, config.json->base_config.json
env: EDGELLM_SPEC_DECODE_ENGINE_DIR=/engines, EDGELLM_MTP_ENABLED=1
/health: {"status":"healthy","model":"/engines","speculative_decoding":true}  <-- spec=true OK
SAME 6 inferences, 237 tegrastats@500ms samples
MTP_PEAK_MB = 11228
MTP footprint = 11228 - 4629 = 6599 MB

## DELTA
DELTA = MTP_footprint - vanilla_footprint = 6599 - 3316 = 3283 MB = 3.21 GB
Interpretation: ~3.2GB extra — FAR ABOVE the 0.4-0.7GB estimate.
(eagle_draft.engine 355MB + draft FFN 36MB on disk, but runtime: draft model
 weights + separate draft KV cache + verify-tree buffers add ~3.2GB unified.)

## RESTORE — pending

## RESTORE — DONE
docker start edge-llm-chat-service seeed-voice
seeed-voice Up (healthy soon), edge-llm-chat-service Up (healthy)
/health 8000: {"status":"healthy","model":"/workspace/Qwen3-4B-AWQ/engines-8192","speculative_decoding":false}  (prod LLM back)
/health 8621: tts:matcha_trt asr:paraformer_trt OK
translator Up 29h (never touched). industrial-security-demo restart-loop PRE-EXISTING (never touched).
MTP dir symlinks removed (returned to original state; real engines untouched).
No rm/down-v of prod, no rebuild/push/commit, no real-prod (seeed-orin-nx never touched).

## FINAL
BASE_MB        = 4629
VAN_PEAK_MB    = 7945  -> vanilla footprint 3316 MB
MTP_PEAK_MB    = 11228 -> MTP footprint     6599 MB
DELTA          = 3283 MB = 3.21 GB  (>> 0.4-0.7GB estimate)
