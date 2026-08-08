#!/usr/bin/env bash
set -u

ROOT=/home/harvest
OUT=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725/results/spark-shared-input-audit
mkdir -p "$OUT"

{
  echo "== configs =="
  cat /home/harvest/project/v090-engines/sparktts/bicodec_decoder_dynT.config.json
  cat /home/harvest/project/v090-engines/sparktts/sparktts_speaker_decoder.config.json
  echo
  echo "== legacy engine hashes =="
  sha256sum \
    /home/harvest/project/v090-engines/sparktts/bicodec_decoder_dynT.fp16.engine \
    /home/harvest/project/v090-engines/sparktts/sparktts_speaker_decoder.fp32.engine
  echo
  echo "== candidate ONNX files =="
  timeout 300 find "$ROOT" -type f -name '*.onnx' -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -n
  echo
  echo "== candidate build scripts/configs/logs by filename or TRT tokens =="
  timeout 300 find "$ROOT" -type f \
    \( -iname '*bicodec*' -o -iname '*speaker*decoder*' -o -iname '*sparktts*' \) \
    \( -name '*.py' -o -name '*.sh' -o -name '*.log' -o -name '*.txt' -o -name '*.json' -o -name 'Dockerfile*' \) \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort
  echo
  echo "== files containing recorded ONNX md5 =="
  timeout 300 grep -RIl \
    -e 'f5ec96fae85be28099d43118a3b709a5' \
    -e '1654b353f50c0d6f63c3c72508d56f47' \
    "$ROOT" 2>/dev/null
  echo
  echo "== git repositories =="
  timeout 300 find "$ROOT" -type d -name .git -printf '%h\n' 2>/dev/null | sort
} >"$OUT/inventory.txt" 2>"$OUT/inventory.stderr.log"

# Hash only ONNX candidates after the inventory, so a large tree cannot hide a
# matching source file merely because its filename differs.
timeout 900 find "$ROOT" -type f -name '*.onnx' -print0 2>/dev/null \
  | xargs -0 -r md5sum >"$OUT/onnx-md5.txt" 2>"$OUT/onnx-md5.stderr.log"

grep -E \
  '(^| )(f5ec96fae85be28099d43118a3b709a5|1654b353f50c0d6f63c3c72508d56f47) ' \
  "$OUT/onnx-md5.txt" >"$OUT/onnx-md5-matches.txt" || true

cat "$OUT/onnx-md5-matches.txt"
