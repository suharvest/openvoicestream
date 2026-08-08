#!/usr/bin/env bash
set -u

echo "== BiCodec directories =="
find /home/harvest -type d \( -name BiCodec -o -iname '*spark-tts*' -o -iname '*sparktts-0.5b*' \) -print 2>/dev/null | sort
echo "== checkpoint marker files =="
find /home/harvest -type f \
  \( -path '*/BiCodec/*' -o -path '*/models--SparkAudio--Spark-TTS-0.5B/*' \) \
  -printf '%s %p\n' 2>/dev/null | sort -n
echo "== Spark source roots =="
find /home/harvest -type f \
  \( -path '*/sparktts/models/audio_tokenizer.py' \
     -o -path '*/sparktts/models/*' \
     -o -name 'cli.py' \) \
  -printf '%p\n' 2>/dev/null | grep -i spark | sort
echo "== legacy export/build references =="
grep -RIn \
  -e 'bicodec_decoder_dynT.onnx' \
  -e 'sparktts_speaker_decoder.onnx' \
  -e 'SPARKTTS_MODEL_DIR' \
  -e 'Spark-TTS-0.5B' \
  /home/harvest/project /home/harvest/*.log 2>/dev/null | head -n 2000
