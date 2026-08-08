#!/usr/bin/env bash
set -euo pipefail

VOICE=seeed-voice-v091-r2-canary-base-n1-021112e
GDN=edge-llm-chat-service
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
EVIDENCE="$VALIDATION/results/r2-triple-overlap-n1"
INPUTS="$VALIDATION/r2-base-n1-inputs"
SINCE=2026-07-25T15:33:00-04:00

docker inspect -f '{{.Name}} {{.State.Status}} {{.State.Health.Status}} {{.RestartCount}}' \
  "$VOICE" "$GDN" > "$EVIDENCE/post-state.txt"
curl -fsS http://127.0.0.1:18621/readyz > "$EVIDENCE/voice-ready-post.json"
curl -fsS http://127.0.0.1:8000/health > "$EVIDENCE/gdn-health-post.json"
docker stats --no-stream "$VOICE" "$GDN" > "$EVIDENCE/docker-stats-post.txt"
docker top "$VOICE" > "$EVIDENCE/voice-top-post.txt"
docker top "$GDN" > "$EVIDENCE/gdn-top-post.txt"
docker logs --since "$SINCE" "$VOICE" > "$EVIDENCE/voice-runtime.stdout.log" 2> "$EVIDENCE/voice-runtime.stderr.log"
docker logs --since "$SINCE" "$GDN" > "$EVIDENCE/gdn-runtime.stdout.log" 2> "$EVIDENCE/gdn-runtime.stderr.log"
python3 "$INPUTS/scan-runtime-errors.py" \
  "$EVIDENCE/voice-runtime.stdout.log" \
  "$EVIDENCE/voice-runtime.stderr.log" \
  "$EVIDENCE/gdn-runtime.stdout.log" \
  "$EVIDENCE/gdn-runtime.stderr.log" \
  > "$EVIDENCE/runtime-error-scan.json"
date --iso-8601=seconds > "$EVIDENCE/post-captured-at.txt"
echo "TRIPLE_FAILURE_POST_EVIDENCE_PASS"
