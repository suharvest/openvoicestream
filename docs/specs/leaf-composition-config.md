# Leaf-and-Composition Config Architecture

Status: DESIGN (2026-06-12). Owner: main thread. Supersedes the device-level
flat `artifact_set` bundling for artifact provisioning + profile assembly.

## Problem

Today "what to pull" is baked into **device-level bundles**: the
`QWEN3_ARTIFACT_MANIFEST` artifact_set (e.g. `orin-nx-highperf-2026-05-14`)
carries a flat `required_files` list that mixes ASR + TTS engine files in one
unit (`third_party/jetson-voice-engine/deploy/artifacts/qwen3_manifest.json`).
Consequences:

- Selecting a TTS variant (e.g. concurrency b1↔b2 talker) is entangled with the
  ASR pull — there is no structural guarantee that "choosing TTS doesn't change
  the ASR artifacts."
- Precision (FP16 vs W8A16), concurrency (N=1 vs N=2), and device are not
  first-class axes; they are implicit in which hand-authored bundle you pick.
- Per-device "optimal default" and composition constraints (combined memory,
  max concurrency) live nowhere — invalid combos fail at runtime (OOM/crash)
  instead of at compose time.

## Design: leaves (atomic) + composition (thin)

### Leaf — the atomic provisioning unit

A **leaf** declares everything needed to run ONE capability with ONE backend on
ONE device at ONE concurrency. Keyed by the tuple:

```
(capability, backend, device, concurrency)
```

Schema (registry entry):

```yaml
id: tts.qwen3_tts.orin-nx.n2
capability: tts                      # asr | tts | vad | ...
backend: jetson.trt_edge_llm         # the voxedge backend id
device: orin-nx                      # device class
concurrency: 2                       # the N this leaf is built for
model: qwen3-tts-customvoice         # logical model identity
precision: fp16                      # LEAF ATTRIBUTE (fp16 | w8a16 | ...)
artifacts:                           # files this leaf contributes to the pull
  repo: harvestsu/qwen3-edgellm-jetson-artifacts
  files:
    - engines/orin-nx/highperf/talker_fp16_b2/talker_decode.engine
    # talker is the ONLY file that differs between n1 and n2; cp/code2wav/
    # tokenizer are shared and declared by a SHARED sub-leaf (see below).
runtime_env:                         # env this leaf contributes to the composed config
  EDGE_LLM_TTS_TALKER_DIR: ${ROOT}/tts/talker_b2
  EDGE_LLM_TTS_TALKER_ENGINE: ${ROOT}/engines/orin-nx/highperf/talker_fp16_b2/talker_decode.engine
  EDGE_LLM_TTS_TALKER_BACKEND: qwen3_tts_explicit_kv
resources:
  peak_unified_mb: 9057              # measured; used by composition memory validation
```

Notes:
- **Shared sub-artifacts**: code_predictor / code2wav / tokenizer are identical
  across TTS concurrency leaves. Model them as a `shared` leaf the concrete leaf
  `requires:` so the union resolver de-dups; do not copy file lists.
- **Concurrency need not change the engine** for every backend. ASR slot-pool
  shares one engine across sessions → `asr.*.n1` and `asr.*.n2` resolve to the
  SAME engine files (only runtime session ceiling differs). TTS batch-lane DOES
  change the engine (b1 maxBatch=1 vs b2 maxBatch=2). The leaf abstraction
  absorbs this: each leaf just declares its own files; whether N changed them is
  a per-backend fact, not a special case in the resolver.

### Composition — the profile

A profile becomes a thin selection over leaves for a device + overrides:

```yaml
device: orin-nx
asr: asr.qwen3_asr.orin-nx.n2
tts: tts.qwen3_tts.orin-nx.n1        # user picks n1 or n2
vad: vad.silero.any.n1
overrides:                           # last-word env, same precedence as today
  OVS_VAD_SILENCE_MS: "400"
```

Resolver behavior:
1. **Pull** = union of selected leaves' `artifacts.files` (de-dup; shared
   sub-leaves pulled once). ASR leaf files are invariant under any TTS choice —
   structural, not enforced by convention.
2. **Env** = merge of leaves' `runtime_env`, then `overrides` on top, then live
   env (env > profile > leaf-default, preserving current precedence in
   `voxedge_backend_config.py`).
3. **Validate composition constraints** (see below) BEFORE provisioning.

### Composition constraints + per-device defaults

A device registry declares RAM + the recommended default combo:

```yaml
devices:
  orin-nx:   { unified_mb: 15656, default: { asr: ...n2, tts: ...n1 } }
  orin-nano: { unified_mb: 7864,  default: { asr: ...n2, tts: ...n1 } }
```

Validator (compose-time, fail fast with a clear message):
- `sum(selected leaf peak_unified_mb)` ≤ device `unified_mb * headroom` →
  e.g. FP16 TTS N=2 (9057 MB) + ASR N=2 only validates on nx, REJECTED on nano.
- total concurrency ceiling per device.
- legal backend pairings (e.g. can't pick two TTS backends).
- Unknown/unbuilt leaf id → explicit error ("no leaf
  tts.qwen3_tts.orin-nano.n2 — not built"), never a silent fallback.

## Precision as a leaf attribute (W8A16-future-default)

`precision` is a leaf attribute, NOT a profile/composition concern. A logical
model declares its **default precision per device class**:

```yaml
models:
  qwen3-tts-customvoice:
    default_precision: { jetson: fp16 }   # TODAY
    # FUTURE: flip jetson -> w8a16 once a quality-valid CustomVoice W8A16 quant
    # exists. One-line change here re-resolves every Jetson CustomVoice TTS leaf
    # to w8a16; profiles/compositions are untouched.
```

Leaf ids stay precision-agnostic (`tts.qwen3_tts.orin-nx.n2`); the registry
resolves the concrete precision via the model's `default_precision[device_class]`
(overridable per-leaf for A/B). This makes the stated intent — *"future W8A16
directly replaces FP16 as the CustomVoice Jetson default"* — a single-locus
change, with no churn in profiles, the resolver, or the worker.

W8A16 is weight-only / batch-agnostic, so a future W8A16 b2 is build-flag-only
ONCE a valid CustomVoice W8A16 ONNX exists. The current blocker is purely the
quant quality (naive max-abs W8A16 of the CustomVoice talker breaks EOS); the
existing `talker_decode_w8a16_outputk.engine` is for an OLDER Qwen3-TTS model
and its source ONNX is gone. Tracked separately; not a blocker for this refactor.

## The immediate N=2 deliverable, mapped onto leaves

- New leaf `tts.qwen3_tts.orin-nx.n2` → **FP16 b2** (verified engine
  `engines-v080-tts-b2/talker`, peak 9057 MB, nx-only). Upload b2 talker to the
  artifact repo as a TTS-only sub-artifact; ASR files unchanged.
- Existing `tts.qwen3_tts.orin-nx.n1` → FP16 b1 (current behavior).
- Worker: the N=2-capable binary (md5 37035ecc; N=1 fast-path byte-identical) is
  shipped regardless of leaf — concurrency is a `--max_slots` runtime arg already
  wired (`trt_edge_llm_tts.py:807`). Selecting the n1 leaf + `--max_slots 1`
  reproduces today's path exactly.
- Profile knob: user picks the tts leaf (n1/n2); default = n1 (opt-in), nx-only
  for n2 (validator rejects on nano).

## Migration (incremental, not big-bang)

1. Introduce the leaf registry + union/env resolver + composition validator +
   device registry. Cover qwen3 ASR, qwen3 TTS (b1/b2), matcha to start.
2. Re-express `jetson-multilang-highperf-nx.json` as a composition over leaves;
   keep the old flat manifest path working until parity is proven.
3. Wire the FP16-b2 n2 leaf + upload + worker binary; deploy + verify N=2 greedy
   on orin-nx (byte-clean both lanes under temp=0/topK=1).
4. Fold in RK / RPi profiles.
5. Retire the flat `artifact_set` once all profiles are compositions.

## Open questions for grounding (codex)

- Exact integration points (file:line) where the flat manifest pull + profile
  env application happen today, and the minimal seam to insert the leaf resolver
  without breaking the live matcha path.
- Whether the leaf registry should live as YAML under `configs/leaves/` or be
  generated from the existing manifests; backward-compat shim plan.
- Where composition validation should run in the boot sequence (before
  `model_downloader`/`engine_resolver`).
- Risk: live production (matcha) must not regress during migration.
