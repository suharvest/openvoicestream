#!/usr/bin/env bash
set -euo pipefail

root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725/results/strict-formal-r2
log=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725/logs/strict-formal-r2-resume.log
pid_file="$root/RESUME.pid"
status="$root/RESUME.status"

test ! -s "$pid_file"
rm -f "$status"
nohup bash /tmp/v091-remote-strict-formal-r2-resume.sh >"$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_file"
printf 'pid=%s\nlog=%s\n' "$pid" "$log"
