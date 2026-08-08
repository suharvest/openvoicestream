#!/usr/bin/env bash
set -euo pipefail

validation_root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
umask 077
mkdir -p \
  "$validation_root/snapshot" \
  "$validation_root/logs" \
  "$validation_root/results" \
  "$validation_root/scripts"

for container in seeed-voice-v091 edge-llm-chat-service translator; do
  docker inspect "$container" \
    >"$validation_root/snapshot/${container}.inspect.full.json"
  docker inspect -f '{{json .State}}' "$container" \
    >"$validation_root/snapshot/${container}.state.json"
  docker inspect -f '{{json .NetworkSettings.Networks}}' "$container" \
    >"$validation_root/snapshot/${container}.networks.json"
  docker inspect \
    -f '{{.Image}} {{.Config.Image}} {{.RestartCount}} {{json .State.Health}}' \
    "$container" \
    >"$validation_root/snapshot/${container}.identity-health.txt"
  docker inspect \
    -f '{{range .Mounts}}{{.Type}}|{{.Source}}|{{.Destination}}|{{.RW}}{{println}}{{end}}' \
    "$container" \
    >"$validation_root/snapshot/${container}.mounts.txt"
  docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$container" |
    sed -E \
      's/^([^=]*(TOKEN|KEY|SECRET|PASS|CREDENTIAL)[^=]*)=.*/\1=<REDACTED>/I' \
      >"$validation_root/snapshot/${container}.env.redacted.txt"
done

chmod -R go-rwx "$validation_root"
docker ps \
  --filter name=seeed-voice-v091 \
  --filter name=edge-llm-chat-service \
  --filter name=translator \
  --format '{{.Names}}|{{.Image}}|{{.ID}}|{{.Status}}|{{.Ports}}' \
  >"$validation_root/snapshot/docker-ps.txt"
free -h >"$validation_root/snapshot/free-before.txt"
df -h / >"$validation_root/snapshot/df-before.txt"

curl -sS -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8621/readyz \
  >"$validation_root/snapshot/voice-ready.http" || true
curl -sS -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8001/health \
  >"$validation_root/snapshot/llm-health.http" || true
curl -sS -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8002/health \
  >"$validation_root/snapshot/translator-health.http" || true

printf 'snapshot_root=%s\n' "$validation_root"
cat "$validation_root/snapshot/docker-ps.txt"
printf 'voice='
cat "$validation_root/snapshot/voice-ready.http"
printf 'llm='
cat "$validation_root/snapshot/llm-health.http"
printf 'translator='
cat "$validation_root/snapshot/translator-health.http"
