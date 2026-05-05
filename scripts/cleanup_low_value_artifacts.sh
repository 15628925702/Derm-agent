#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN="${DRY_RUN:-1}"

log() {
  printf '%s\n' "$1"
}

remove_path() {
  local target="$1"
  if [[ ! -e "${target}" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[dry-run] remove ${target}"
    return 0
  fi
  rm -rf "${target}"
  log "[removed] ${target}"
}

log "[info] project root: ${PROJECT_ROOT}"
log "[info] dry run     : ${DRY_RUN}"
log "[info] preserving valuable runs:"
log "  - ${PROJECT_ROOT}/outputs/smoke_cycles/medium_signal_no_ablation_10h_v1"
log "  - ${PROJECT_ROOT}/outputs/smoke_cycles/reuse3h_signal_check_20260328T141307Z"
log "  - ${PROJECT_ROOT}/backups/*"

LOW_VALUE_PATHS=(
  "${PROJECT_ROOT}/outputs/smoke_cycles/stage34_probe_30m_20260328T163440Z"
)

for path in "${LOW_VALUE_PATHS[@]}"; do
  remove_path "${path}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  log "[done] dry-run complete. Re-run with DRY_RUN=0 to actually clean."
else
  log "[done] low-value artifact cleanup complete."
fi
