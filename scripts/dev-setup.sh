#!/usr/bin/env bash
# One-shot local dev setup. The product imports `voxedge` from the sibling
# repo ../voxedge — without this editable install, `import voxedge` fails and
# server/agent won't start. Run this once after cloning.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOXEDGE_SRC="${VOXEDGE_SRC:-$(cd "${REPO_ROOT}/../voxedge" 2>/dev/null && pwd || true)}"

if [[ -z "${VOXEDGE_SRC}" || ! -f "${VOXEDGE_SRC}/pyproject.toml" ]]; then
  echo "ERROR: voxedge source not found next to this repo (expected ../voxedge)." >&2
  echo "       Clone it:  git clone <voxedge-repo-url> ${REPO_ROOT}/../voxedge" >&2
  echo "       Or point:  VOXEDGE_SRC=/abs/path/to/voxedge scripts/dev-setup.sh" >&2
  exit 1
fi

echo "==> installing voxedge (editable) from ${VOXEDGE_SRC}"
uv pip install -e "${VOXEDGE_SRC}"

echo "==> installing server requirements"
uv pip install -r "${REPO_ROOT}/server/requirements.txt"

echo "==> installing agent (editable, dev extra)"
uv pip install -e "${REPO_ROOT}/agent[dev]"

echo "==> verifying imports"
python -c "import voxedge; from voxedge.backends.mock import MockASR; print('voxedge OK:', voxedge.__file__)"
echo "Done. Next: see ARCHITECTURE.md → 'Run it locally (no GPU)'."
