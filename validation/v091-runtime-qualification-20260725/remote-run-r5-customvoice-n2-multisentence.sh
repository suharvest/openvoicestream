#!/usr/bin/env bash
set -euo pipefail
root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725/r2-customvoice-inputs
source "$root/remote-r5-customvoice-n2-env.sh"
export CANARY_EVIDENCE=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725/results/r5-customvoice-n2-multisentence
export SKIP_CORE=1
exec "$root/remote-run-customvoice-n2-helper.sh"
