#!/usr/bin/env bash
# RK1828 host bring-up / verification.
#
# WHY THIS SCRIPT EXISTS
#   The LLM container is NOT self-contained. The RK1828 accelerator card's kernel
#   driver and firmware live on the HOST; a container can only talk to a card the
#   host has already initialised. Handing someone just an image and a compose file
#   leaves them with a service that fails at model-load time for reasons that look
#   like a hardware fault.
#
# Run with --check to verify only (safe, read-only), or with no argument to also
# load and persist the module.
#
#   sudo deploy/scripts/rk1828-host-bringup.sh --check
#   sudo deploy/scripts/rk1828-host-bringup.sh
#
# This script does NOT build the kernel module. Building it needs the RM182X SDK
# plus kernel headers; see services/rk1828-llm/BUILD.md. If the module is not
# installed this script says so and stops.
set -uo pipefail

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

PCI_ID="1d87:182a"
MODULE="pcie-rkep"          # hyphen for modprobe / modules-load.d
MODULE_LSMOD="pcie_rkep"    # underscore as lsmod reports it
FIRMWARE="/lib/firmware/rknn3_rk1820.img"
PERSIST="/etc/modules-load.d/pcie-rkep.conf"

# /sbin is not always on PATH under sudo or in a non-login shell.
PATH="/sbin:/usr/sbin:${PATH}"

fail=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; fail=1; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }

if [ "$(id -u)" != "0" ] && [ "${CHECK_ONLY}" = "0" ]; then
  echo "ERROR: run as root (or use --check for a read-only verification)" >&2
  exit 1
fi

step "1. card present on the PCIe bus"
if lspci -nn 2>/dev/null | grep -qi "${PCI_ID}"; then
  ok "$(lspci -nn | grep -i "${PCI_ID}" | head -1)"
else
  bad "no ${PCI_ID} on the PCIe bus."
  echo "       This is HARDWARE, not software. The card has its OWN 12 V supply —"
  echo "       a dark card with a still fan means it is not powered. Check that"
  echo "       the 12 V lead is on the main power pins, not the fan header (they"
  echo "       are adjacent and easy to confuse)."
fi

step "2. kernel module installed"
if modinfo "${MODULE}" >/dev/null 2>&1; then
  ok "$(modinfo "${MODULE}" | awk '/^filename:/{print $2}')"
else
  bad "module ${MODULE} not installed for kernel $(uname -r)."
  echo "       Build + install it from the RM182X SDK (pcie-rkep driver source)"
  echo "       — see services/rk1828-llm/BUILD.md. Nothing below will work."
fi

step "3. kernel module loaded"
if lsmod | grep -q "^${MODULE_LSMOD}"; then
  ok "$(lsmod | grep "^${MODULE_LSMOD}")"
elif [ "${CHECK_ONLY}" = "1" ]; then
  bad "${MODULE_LSMOD} not loaded (re-run without --check to load it)"
else
  echo "  loading ${MODULE} ..."
  if modprobe "${MODULE}"; then ok "loaded"; else bad "modprobe ${MODULE} failed"; fi
fi

step "4. module load persisted across reboot"
# The module does NOT come back by itself; without this the card disappears
# after any reboot and the LLM service fails to start with no obvious cause.
if [ -f "${PERSIST}" ] && grep -q "${MODULE}" "${PERSIST}" 2>/dev/null; then
  ok "${PERSIST} -> $(tr -d '\n' < "${PERSIST}")"
elif [ "${CHECK_ONLY}" = "1" ]; then
  bad "not persisted; re-run without --check to write ${PERSIST}"
else
  echo "${MODULE}" > "${PERSIST}"
  ok "wrote ${PERSIST}"
fi

step "5. EP firmware present"
if [ -f "${FIRMWARE}" ]; then
  ok "${FIRMWARE} ($(stat -c %s "${FIRMWARE}") bytes)"
else
  bad "${FIRMWARE} missing — installed by the RKNN3 arm64 installer from the SDK."
fi

step "6. rknn3.service (reflashes the EP firmware at boot)"
if systemctl is-active --quiet rknn3.service; then
  ok "active"
else
  bad "not active: $(systemctl is-active rknn3.service 2>&1)"
  echo "       Without it the EP has no firmware and every model init fails."
fi
if systemctl is-enabled --quiet rknn3.service 2>/dev/null; then
  ok "enabled (starts at boot)"
else
  if [ "${CHECK_ONLY}" = "1" ]; then
    bad "not enabled; the card will be dead after a reboot"
  else
    systemctl enable rknn3.service >/dev/null 2>&1 && ok "enabled" || bad "could not enable"
  fi
fi

step "7. character device visible (what the container needs)"
if compgen -G '/dev/pcie-rkep-*' >/dev/null; then
  for d in /dev/pcie-rkep-*; do ok "$(ls -l "$d")"; done
else
  bad "no /dev/pcie-rkep-* — the container cannot reach the card."
fi

step "8. EP not already occupied by another large model"
# The card has ONE ~5 GB context. Qwen3-4B at 8192 tokens uses an estimated
# ~3.6 GB, so a second large model cannot be resident. tts-radxa is the usual
# culprit on our own boards.
if systemctl is-active --quiet tts-radxa.service 2>/dev/null; then
  warn "tts-radxa.service is ACTIVE and holds the EP with an RK1828 TTS model."
  echo "       It cannot co-reside with the LLM. Stop it before starting the LLM:"
  echo "         systemctl stop tts-radxa"
  echo "       (In this delivery TTS runs on the RK3588's own NPU, so it is not needed.)"
else
  ok "no known EP-holding service active"
fi

step "9. observability caveat (not a failure)"
warn "rknn-smi is non-functional on this platform: it fails for info /"
echo "       info -t memory / info -l, as root, and even with the EP idle."
echo "       Suspected host/EP firmware version skew shipped by V1.0.4"
echo "       (rc_cc_version=30301 vs ep_cc_version=30201), with no newer EP"
echo "       firmware to flash. Consequence: EP memory and health have NO"
echo "       observability — model-load success is the only signal."
echo "       NEVER run 'rknn-smi reset': it can wedge the card into a boot state"
echo "       a host reboot may not recover, and the card does not power-cycle"
echo "       with the host. Repeated FAILED model loads also degrade the EP from"
echo "       8 cores to 4, so do not probe capacity by trial and error."

echo
if [ "${fail}" = "0" ]; then
  echo "RESULT: host is ready for the RK1828 LLM container."
  exit 0
fi
echo "RESULT: host is NOT ready — fix the FAIL items above." >&2
exit 1
