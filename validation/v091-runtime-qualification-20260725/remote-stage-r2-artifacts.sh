#!/usr/bin/env bash
set -euo pipefail

OLD=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725
FINAL=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2
TEMP=/home/harvest/edgellm-artifacts/.orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2.staging-021112e
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
INPUTS="$VALIDATION/r2-staging-inputs"
EVIDENCE="$VALIDATION/results/r2-artifact-staging"
QUARANTINE="$VALIDATION/quarantine"
FINALIZER=/home/harvest/project/seeed-local-voice-v091-r2-021112e-20260726/scripts/finalize_edgellm_v091_artifact.py

mkdir -p "$EVIDENCE" "$QUARANTINE"

for container in seeed-voice-v091 edge-llm-chat-service translator; do
  status="$(docker inspect -f '{{.State.Status}}' "$container")"
  if [[ "$status" != exited ]]; then
    echo "refusing staging while $container is $status" >&2
    exit 1
  fi
done

[[ "$(stat -c %d "$OLD")" == "$(stat -c %d "$(dirname "$FINAL")")" ]]
[[ -f "$OLD/v091/manifest.json" ]]
[[ -f "$FINAL/v091/bin/moss_tts_nano_worker" ]]
[[ ! -e "$TEMP" ]]

date --iso-8601=seconds > "$EVIDENCE/started-at.txt"
df -B1 /home/harvest/edgellm-artifacts > "$EVIDENCE/df-before.txt"
sha256sum \
  "$OLD/v091/manifest.json" \
  "$OLD/v091/SHA256SUMS" \
  "$OLD/v091/PROVENANCE.md" \
  "$OLD/v091/bin/moss_tts_nano_worker" \
  > "$EVIDENCE/old-control-worker-before.sha256"
stat -c '%d %i %s %n' \
  "$OLD/v091/manifest.json" \
  "$OLD/v091/SHA256SUMS" \
  "$OLD/v091/PROVENANCE.md" \
  "$OLD/v091/bin/moss_tts_nano_worker" \
  > "$EVIDENCE/old-control-worker-before.stat"

partial_quarantine="$QUARANTINE/partial-r2-before-staging-021112e"
[[ ! -e "$partial_quarantine" ]]
mv "$FINAL" "$partial_quarantine"

mkdir "$TEMP"
cp -al "$OLD/." "$TEMP/"

worker_tmp="$TEMP/v091/bin/.moss_tts_nano_worker.tmp-$$"
cp "$partial_quarantine/v091/bin/moss_tts_nano_worker" "$worker_tmp"
chmod 0755 "$worker_tmp"
mv -f "$worker_tmp" "$TEMP/v091/bin/moss_tts_nano_worker"

manifest_tmp="$TEMP/v091/.manifest.json.input-$$"
provenance_tmp="$TEMP/v091/.PROVENANCE.md.input-$$"
cp "$INPUTS/r2-manifest-template.json" "$manifest_tmp"
cp "$INPUTS/r2-PROVENANCE.md" "$provenance_tmp"
mv -f "$manifest_tmp" "$TEMP/v091/manifest.json"
mv -f "$provenance_tmp" "$TEMP/v091/PROVENANCE.md"

python3 "$FINALIZER" "$TEMP/v091" --published-to-hf false \
  | tee "$EVIDENCE/finalizer.json"
(cd "$TEMP/v091" && sha256sum -c SHA256SUMS) \
  > "$EVIDENCE/SHA256SUMS.check.txt"

python3 "$INPUTS/finalizer-negative-gates.py" \
  "$FINALIZER" "$VALIDATION/negative-gates" \
  > "$EVIDENCE/finalizer-negative-gates.json"

python3 - "$TEMP/v091" "$INPUTS/qwen3_manifest.json" \
  > "$EVIDENCE/required-files-gate.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
qwen = json.loads(Path(sys.argv[2]).read_text())
key = "orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2"
required = qwen["artifact_sets"][key]["required_files"]
missing = [path for path in required if not (root / path).is_file()]
codec = [
    "engines/moss/codec/codec_decode_step.plan",
    "engines/moss/codec/codec_decode_step.plan.meta.json",
    "engines/moss/codec/codec_browser_onnx_meta.json",
    "engines/moss/codec/moss_audio_tokenizer_decode_shared.data",
    "engines/moss/codec/moss_audio_tokenizer_encode.onnx",
    "engines/moss/codec/moss_audio_tokenizer_encode.data",
]
spark = [
    "engines/sparktts-w4a16/llm.engine",
    "engines/sparktts-w4a16/token-gate.json",
    "engines/sparktts-bf16/llm.engine",
    "engines/sparktts-bf16/token-gate.json",
]
result = {
    "required_count": len(required),
    "missing": missing,
    "codec_complete": all((root / path).is_file() for path in codec),
    "spark_complete": all((root / path).is_file() for path in spark),
}
print(json.dumps(result, indent=2, sort_keys=True))
if missing or not result["codec_complete"] or not result["spark_complete"]:
    raise SystemExit(1)
PY

sha256sum -c "$EVIDENCE/old-control-worker-before.sha256" \
  > "$EVIDENCE/old-control-worker-after.check.txt"

old_worker_inode="$(stat -c %i "$OLD/v091/bin/moss_tts_nano_worker")"
new_worker_inode="$(stat -c %i "$TEMP/v091/bin/moss_tts_nano_worker")"
old_manifest_inode="$(stat -c %i "$OLD/v091/manifest.json")"
new_manifest_inode="$(stat -c %i "$TEMP/v091/manifest.json")"
old_sums_inode="$(stat -c %i "$OLD/v091/SHA256SUMS")"
new_sums_inode="$(stat -c %i "$TEMP/v091/SHA256SUMS")"
old_provenance_inode="$(stat -c %i "$OLD/v091/PROVENANCE.md")"
new_provenance_inode="$(stat -c %i "$TEMP/v091/PROVENANCE.md")"
[[ "$old_worker_inode" != "$new_worker_inode" ]]
[[ "$old_manifest_inode" != "$new_manifest_inode" ]]
[[ "$old_sums_inode" != "$new_sums_inode" ]]
[[ "$old_provenance_inode" != "$new_provenance_inode" ]]

mv "$TEMP" "$FINAL"

stat -c '%d %i %s %n' \
  "$FINAL/v091/manifest.json" \
  "$FINAL/v091/SHA256SUMS" \
  "$FINAL/v091/PROVENANCE.md" \
  "$FINAL/v091/bin/moss_tts_nano_worker" \
  > "$EVIDENCE/r2-control-worker-after.stat"
sha256sum \
  "$FINAL/v091/manifest.json" \
  "$FINAL/v091/SHA256SUMS" \
  "$FINAL/v091/PROVENANCE.md" \
  "$FINAL/v091/bin/moss_tts_nano_worker" \
  > "$EVIDENCE/r2-control-worker-after.sha256"
sha256sum -c "$EVIDENCE/old-control-worker-before.sha256" \
  > "$EVIDENCE/old-control-worker-post-rename.check.txt"
du -s "$OLD" "$FINAL" > "$EVIDENCE/du-deduplicated-kib.txt"
df -B1 /home/harvest/edgellm-artifacts > "$EVIDENCE/df-after.txt"
date --iso-8601=seconds > "$EVIDENCE/completed-at.txt"

echo "R2_ARTIFACT_STAGING_PASS"
