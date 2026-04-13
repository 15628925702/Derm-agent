#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-medium_signal_no_ablation_v1}"
BACKUP_ROOT="${BACKUP_ROOT:-${PROJECT_ROOT}/backups}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BACKUP_NAME="${BACKUP_NAME:-longrun_assets_${RUN_ID}_${STAMP}}"
TARGET_DIR="${BACKUP_ROOT}/${BACKUP_NAME}"
MAKE_TARBALL="${MAKE_TARBALL:-0}"

mkdir -p "${BACKUP_ROOT}"
mkdir -p "${TARGET_DIR}"

copy_if_exists() {
  local source_path="$1"
  local target_path="$2"
  if [[ -e "${source_path}" ]]; then
    mkdir -p "$(dirname "${target_path}")"
    cp -r "${source_path}" "${target_path}"
    echo "[ok] copied: ${source_path} -> ${target_path}"
  else
    echo "[warn] missing, skipped: ${source_path}"
  fi
}

copy_if_exists "${PROJECT_ROOT}/state/split_states" "${TARGET_DIR}/state/split_states"
copy_if_exists "${PROJECT_ROOT}/state/policy" "${TARGET_DIR}/state/policy"
copy_if_exists "${PROJECT_ROOT}/outputs/checkpoints" "${TARGET_DIR}/outputs/checkpoints"
copy_if_exists "${PROJECT_ROOT}/outputs/smoke_cycles/${RUN_ID}" "${TARGET_DIR}/outputs/smoke_cycles/${RUN_ID}"

MANIFEST_PATH="${TARGET_DIR}/backup_manifest.json"
cat > "${MANIFEST_PATH}" <<EOF
{
  "backup_name": "${BACKUP_NAME}",
  "run_id": "${RUN_ID}",
  "created_at_utc": "${STAMP}",
  "project_root": "${PROJECT_ROOT}",
  "paths": {
    "split_states": "${TARGET_DIR}/state/split_states",
    "policy_state": "${TARGET_DIR}/state/policy",
    "checkpoints": "${TARGET_DIR}/outputs/checkpoints",
    "smoke_cycle_run": "${TARGET_DIR}/outputs/smoke_cycles/${RUN_ID}",
    "run_local_checkpoints": "${TARGET_DIR}/outputs/smoke_cycles/${RUN_ID}/checkpoints"
  }
}
EOF

echo "[ok] manifest: ${MANIFEST_PATH}"

if [[ "${MAKE_TARBALL}" == "1" ]]; then
  TARBALL_PATH="${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz"
  tar -czf "${TARBALL_PATH}" -C "${BACKUP_ROOT}" "${BACKUP_NAME}"
  echo "[ok] tarball: ${TARBALL_PATH}"
fi

echo "[done] backup directory: ${TARGET_DIR}"
