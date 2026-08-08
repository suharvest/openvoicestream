#!/usr/bin/env bash
set -euo pipefail

validation_root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
new_image=seeed-local-voice:v0.9.1-edgellm-runtime-normalized-d52d973-20260725

docker image inspect "$new_image" >"$validation_root/snapshot/new-image.inspect.json"
docker image inspect \
  -f '{{.Id}}|{{json .RepoDigests}}|{{.Size}}|{{json .Config.Entrypoint}}|{{json .Config.Cmd}}' \
  "$new_image" >"$validation_root/snapshot/new-image.identity.txt"
docker inspect \
  -f '{{json .HostConfig}}' \
  seeed-voice-v091 >"$validation_root/snapshot/seeed-voice-v091.hostconfig.json"
docker inspect \
  -f '{{json .Config.Labels}}' \
  seeed-voice-v091 >"$validation_root/snapshot/seeed-voice-v091.labels.json"
docker inspect \
  -f '{{json .Config.ExposedPorts}}' \
  seeed-voice-v091 >"$validation_root/snapshot/seeed-voice-v091.exposed-ports.json"

printf 'new_image='
cat "$validation_root/snapshot/new-image.identity.txt"
printf 'voice_network_mode='
docker inspect -f '{{.HostConfig.NetworkMode}}' seeed-voice-v091
printf 'voice_restart='
docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' seeed-voice-v091
printf 'voice_profile='
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' seeed-voice-v091 |
  sed -n 's/^OVS_PROFILE=//p'
