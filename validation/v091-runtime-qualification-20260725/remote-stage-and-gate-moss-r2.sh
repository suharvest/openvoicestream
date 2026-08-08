#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
result_root="$validation/results/strict-formal-r2"
artifact_root=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2/v091
source_worker=/home/harvest/build/TensorRT-Edge-LLM-v091-formal-r2-4b28dd2-clean-20260726/examples/omni/moss_tts_nano_worker
staged_worker="$artifact_root/bin/moss_tts_nano_worker"
engine=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725/v091/engines/moss
codec="$engine/codec"
ort=/opt/onnxruntime-linux-aarch64-1.23.2
harness="$validation/scripts/moss_worker_n2_gate.py"
result="$validation/results/moss-direct-r2-no-compat-n1-n2-cancel-3.json"
stdout_log="$validation/logs/moss-direct-r2-no-compat.stdout.log"
stderr_log="$validation/logs/moss-direct-r2-no-compat.stderr.log"

test -x "$source_worker"
test "$(sha256sum "$source_worker" | cut -d' ' -f1)" = \
  9d114d8390e684c8876e2ef9e20e28ee6d4ec6ce18b81df5da3ba64c8f057deb
test ! -e "$staged_worker"
test -s "$engine/tokenizer.model"
test -s "$codec/codec_decode_step.plan"
test -s "$codec/codec_decode_step.plan.meta.json"
test -s "$codec/codec_browser_onnx_meta.json"
test -s "$codec/moss_audio_tokenizer_decode_shared.data"
test -s "$codec/moss_audio_tokenizer_encode.onnx"
test -s "$codec/moss_audio_tokenizer_encode.data"
test -s "$harness"
test "$(docker inspect -f '{{.State.Status}}' seeed-voice-v091)" = exited
test "$(docker inspect -f '{{.State.Status}}' edge-llm-chat-service)" = exited
test "$(docker inspect -f '{{.State.Status}}' translator)" = exited

mkdir -p "$artifact_root/bin"
install -m 0755 "$source_worker" "$staged_worker"
test "$(sha256sum "$staged_worker" | cut -d' ' -f1)" = \
  9d114d8390e684c8876e2ef9e20e28ee6d4ec6ce18b81df5da3ba64c8f057deb
{
  printf 'staged_at=%s\n' "$(date -Is)"
  printf 'source=%s\n' "$source_worker"
  printf 'destination=%s\n' "$staged_worker"
  stat -c 'mode=%a size=%s uid=%u gid=%g' "$staged_worker"
  sha256sum "$staged_worker"
} >"$result_root/moss-worker-r2-stage.txt"

env \
  LD_LIBRARY_PATH="$ort/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  ldd -r "$staged_worker" >"$result_root/moss-worker-r2.ldd-r.txt" 2>&1
if grep -E 'not found|undefined symbol|version .* not found' \
  "$result_root/moss-worker-r2.ldd-r.txt"; then
  exit 51
fi
nm -D --with-symbol-versions "$staged_worker" \
  >"$result_root/moss-worker-r2.nm-dynamic.txt"
grep 'OrtGetApiBase@VERS_1.23.2' "$result_root/moss-worker-r2.nm-dynamic.txt"
! grep -q 'OrtGetApiBase@VERS_1.20' "$result_root/moss-worker-r2.nm-dynamic.txt"
env \
  LD_LIBRARY_PATH="$ort/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  "$staged_worker" --help >"$result_root/moss-worker-r2.help.txt" 2>&1

# Deliberately pass the real engine root plus its explicit codec subdirectory.
# No compatibility root or root-level codec symlink is supplied.
env \
  LD_LIBRARY_PATH="$ort/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  python3 "$harness" \
  --worker "$staged_worker" \
  --engine-dir "$engine" \
  --codec-onnx-dir "$codec" \
  --rounds 3 \
  --timeout 90 \
  --output "$result" \
  >"$stdout_log" 2>"$stderr_log"

test -s "$result"
python3 -m json.tool "$result" >"$result.pretty"
python3 - "$result" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))
assert d["ready"]["max_slots"] == 2, d
assert d["ready"]["concurrent_dispatch"] is True, d
assert d["ready"]["cooperative_cancel"] is True, d
assert d["ready"]["engine_dir"].endswith("/v091/engines/moss"), d
assert d["baseline_a"]["bytes"] > 0 and d["baseline_b"]["bytes"] > 0, d
assert d["initial_pair"]["overlap"] is True, d
assert d["initial_pair"]["outputs_distinct"] is True, d
assert d["rounds_passed"] == d["rounds_requested"] == 3, d
assert all(row["ok"] for row in d["rounds"]), d
assert d["worker_returncode"] == 0, d
assert not d["stderr_error_hits"], d
PY

sha256sum \
  "$result_root/moss-worker-r2-stage.txt" \
  "$result_root/moss-worker-r2.ldd-r.txt" \
  "$result_root/moss-worker-r2.nm-dynamic.txt" \
  "$result_root/moss-worker-r2.help.txt" \
  "$result" \
  "$result.pretty" \
  >"$result_root/MOSS-R2-DIRECT-EVIDENCE.SHA256SUMS"
sha256sum -c "$result_root/MOSS-R2-DIRECT-EVIDENCE.SHA256SUMS"
cat "$result_root/moss-worker-r2-stage.txt"
cat "$result.pretty"
