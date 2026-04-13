#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash scripts/start_detached_run.sh <run-script> [script-args...]

Example:
  RUN_ID=medium_signal_no_ablation_10h_v2 \
  bash scripts/start_detached_run.sh scripts/run_medium_signal_no_ablation_10h.sh

This launcher detaches the run from the current terminal using nohup, writes a
pid file and a launcher log, and prints the exact monitor commands to use.
EOF
  exit 1
fi

RUN_SCRIPT_INPUT="$1"
shift || true

if [[ "${RUN_SCRIPT_INPUT}" = /* ]]; then
  RUN_SCRIPT="${RUN_SCRIPT_INPUT}"
else
  RUN_SCRIPT="${PROJECT_ROOT}/${RUN_SCRIPT_INPUT}"
fi

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "[error] run script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  chmod +x "${RUN_SCRIPT}"
fi

RUN_ID="${RUN_ID:-}"
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(basename "${RUN_SCRIPT}" .sh)_$(date -u +%Y%m%dT%H%M%SZ)"
fi

STATE_DIR="${PROJECT_ROOT}/state/run_launchers"
LOG_DIR="${PROJECT_ROOT}/outputs/launcher_logs"
mkdir -p "${STATE_DIR}" "${LOG_DIR}"

PID_FILE="${STATE_DIR}/${RUN_ID}.pid"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    echo "[error] run appears to already be active: run_id=${RUN_ID} pid=${EXISTING_PID}" >&2
    echo "[hint] inspect log: ${LOG_FILE}" >&2
    exit 1
  fi
fi

COMMAND=("${RUN_SCRIPT}" "$@")
printf -v COMMAND_STR '%q ' "${COMMAND[@]}"

(
  cd "${PROJECT_ROOT}"
  nohup env RUN_ID="${RUN_ID}" OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" bash -lc "${COMMAND_STR}" > "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
)

PID="$(cat "${PID_FILE}")"

cat <<EOF
[ok] detached run started
run_id   : ${RUN_ID}
pid      : ${PID}
pid file : ${PID_FILE}
log file : ${LOG_FILE}

monitor:
  tail -f ${LOG_FILE}
  ps -p ${PID} -o pid=,etime=,%cpu=,%mem=,cmd=

stop:
  kill ${PID}
EOF
