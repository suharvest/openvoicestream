#!/usr/bin/env bash
# Provision a fresh Linux+RTX Isaac Sim 4.5 instance for reBot grasp simulation.
# One command to switch devices and continue testing.
#
# Usage:  bash sim/provision_isaac.sh <SSH_HOST> <SSH_PORT> [SSH_USER]
#   e.g.  bash sim/provision_isaac.sh ssh5.vast.ai 16636 root
#
# Prereqs on the instance: the nvcr.io/nvidia/isaac-sim:4.5.0 image
# (so /isaac-sim/python.sh exists) + an RTX GPU. SSH key already authorized.
# See docs/sim/isaac_sim_repro_runbook.md for the full runbook.
set -euo pipefail

HOST="${1:?usage: provision_isaac.sh <SSH_HOST> <SSH_PORT> [SSH_USER]}"
PORT="${2:?missing SSH_PORT}"
USER="${3:-root}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH="ssh -p ${PORT} -o StrictHostKeyChecking=accept-new ${USER}@${HOST}"
SCP="scp -P ${PORT} -o StrictHostKeyChecking=accept-new"

echo "==> [1/5] verify instance reachable + Isaac image present"
$SSH "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader && ls /isaac-sim/python.sh && echo IMG_OK" \
  | grep -q IMG_OK || { echo "FAIL: no Isaac image / GPU at ${USER}@${HOST}:${PORT}"; exit 1; }

echo "==> [2/5] bundle sim assets + grasp pipeline"
cd "$REPO_ROOT"
tar czf /tmp/rebot_sim_bundle.tar.gz \
  sim/rebot_b601dm_urdf sim/calib sim/linux_smoke.py \
  docs/sim/isaac_bridge_spec.md \
  agent/ovs_agent/apps/voice_rebot_arm/perception \
  agent/ovs_agent/apps/voice_rebot_arm/tools/synthetic_grasp_harness.py \
  agent/ovs_agent/apps/voice_rebot_arm/tools/artifacts/ik_envelope_b601dm.csv

echo "==> [3/5] transfer + unpack on instance"
$SCP /tmp/rebot_sim_bundle.tar.gz "${USER}@${HOST}:/root/"
$SSH "cd /root && tar xzf rebot_sim_bundle.tar.gz && cp sim/linux_smoke.py /root/linux_smoke.py && echo UNPACKED"

echo "==> [4/5] install deps (numpy<2 is mandatory; pin/opencv pull numpy2 which breaks Isaac ABI)"
$SSH "/isaac-sim/python.sh -m pip install -q pin opencv-python-headless && /isaac-sim/python.sh -m pip install -q 'numpy<2' && \
      /isaac-sim/python.sh -c 'import numpy,cv2,pinocchio; print(\"DEPS\",numpy.__version__,cv2.__version__,pinocchio.__version__)'"

echo "==> [5/5] headless render smoke (1-3 min on first run: shader/asset cache)"
$SSH "cd /isaac-sim && ./python.sh /root/linux_smoke.py 2>&1 | grep -iE 'SIMAPP_OK|RENDER_60_OK|WORLD_RESET_OK|WORLD_STEP_RENDER_30_OK|CLOSED_OK|EXCEPTION'"

cat <<EOF

==> DONE. If you saw the 5 *_OK markers above, the platform is ready.
    Assets on instance: /root/sim/  +  /root/agent/ovs_agent/apps/voice_rebot_arm/
    Bridge spec:        /root/docs/sim/isaac_bridge_spec.md
    Build bridge under: /root/sim_bridge/   (see runbook §5)
    Remember to:  vastai destroy instance <CONTRACT_ID>   when done (billing).
EOF
