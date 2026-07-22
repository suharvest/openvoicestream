#!/usr/bin/env bash
# Run the Demo Gallery frontends LOCALLY (localhost = secure context, so the
# browser mic works without HTTPS) while talking to a REMOTE SLV backend.
#
# ── Configure here (the one place) ──────────────────────────────────────────
SLV_HOST="${SLV_HOST:-100.82.225.102}"   # device running the SLV server (Tailscale/LAN IP)
SLV_PORT="${SLV_PORT:-18765}"            # SLV port on that device
SSH_USER="${SSH_USER:-harvest}"          # ssh user for the tunnel
SLV_ADMIN_KEY="${SLV_ADMIN_KEY:-}"       # admin key (model-switch panel); empty = panel read-only
# Presentation labels + switch allowlist (optional):
DEMO_ASR_MODEL_ID="${DEMO_ASR_MODEL_ID:-}"          # e.g. qwen3-asr-0.6b (ASR pill label; SLV has no asr model_id)
DEMO_SWITCH_PROFILES="${DEMO_SWITCH_PROFILES:-}"    # e.g. jetson-edgellm-v090-qwen3ttsbase,...  (empty = platform filter)
# Whether to tunnel the SLV port to localhost (recommended: WS also stays proxy-free).
USE_TUNNEL="${USE_TUNNEL:-1}"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="/tmp/slv-demos-local-logs"; mkdir -p "$LOGDIR"

if [ "$USE_TUNNEL" = "1" ]; then
  # Point the demos at localhost so both the HTTP proxy AND the browser WS go
  # through the tunnel (no system proxy / Clash interference, no HTTPS needed).
  pkill -f "ssh -f -N.*${SLV_PORT}:localhost:${SLV_PORT}" 2>/dev/null || true
  ssh -f -N -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -L "${SLV_PORT}:localhost:${SLV_PORT}" "${SSH_USER}@${SLV_HOST}"
  SLV_URL="http://localhost:${SLV_PORT}"
  echo "tunnel: localhost:${SLV_PORT} -> ${SSH_USER}@${SLV_HOST} (pid $(pgrep -f "ssh -f -N.*${SLV_PORT}" | head -1))"
else
  # Direct: HTTP proxy bypasses the system proxy (trust_env=False); the browser
  # WS goes to the device IP (needs your proxy to allow that IP, e.g. Tailscale).
  SLV_URL="http://${SLV_HOST}:${SLV_PORT}"
fi
echo "SLV_URL = ${SLV_URL}"

cd "$HERE"
uv sync -q
pkill -f "slv-demos-local.*backend/main" 2>/dev/null || true
pkill -f "uvicorn gallery.backend.main" 2>/dev/null || true
sleep 1

export SLV_URL SLV_ADMIN_KEY DEMO_ASR_MODEL_ID DEMO_SWITCH_PROFILES
export DEMO_PROFILES_DIR="${DEMO_PROFILES_DIR:-$(cd "$HERE/../configs/profiles" && pwd)}"

SLV_URL="$SLV_URL" SLV_ADMIN_KEY="$SLV_ADMIN_KEY" DEMO_PROFILES_DIR="$DEMO_PROFILES_DIR" \
DEMO_SWITCH_PROFILES="$DEMO_SWITCH_PROFILES" DEMO_ASR_MODEL_ID="$DEMO_ASR_MODEL_ID" PORT=8700 \
  nohup uv run python -m uvicorn gallery.backend.main:app --host 127.0.0.1 --port 8700 \
  > "$LOGDIR/gallery.log" 2>&1 &

for pair in "asr-caption:8701" "tts-playground:8702" "v2v-chat:8703" "diarization:8704" "voice-clone:8705"; do
  app="${pair%%:*}"; port="${pair##*:}"
  SLV_URL="$SLV_URL" PORT="$port" nohup uv run python "$app/backend/main.py" \
    > "$LOGDIR/$app.log" 2>&1 &
done

sleep 10
echo "── health ──"
for p in 8700 8701 8702 8703 8704 8705; do
  printf "  localhost:%s " "$p"
  curl -s -m 5 "http://127.0.0.1:$p/healthz" >/dev/null 2>&1 && echo ok || echo DOWN
done
echo
echo "open  http://localhost:8700"
