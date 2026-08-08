#!/usr/bin/env bash
set -euo pipefail

root=/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/evidence/sparktts-20260725
find "$root" -maxdepth 4 -type f -printf '%p|%s\n' | sort
