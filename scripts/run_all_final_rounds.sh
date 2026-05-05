#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SERVER_CHECK_TIMEOUT="${SERVER_CHECK_TIMEOUT:-180}"
SERVER_WAIT_RETRIES="${SERVER_WAIT_RETRIES:-120}"
SERVER_WAIT_INTERVAL_SECONDS="${SERVER_WAIT_INTERVAL_SECONDS:-10}"
SERVER_CHAT_CHECK_RETRIES="${SERVER_CHAT_CHECK_RETRIES:-20}"
SERVER_CHAT_CHECK_INTERVAL_SECONDS="${SERVER_CHAT_CHECK_INTERVAL_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_FINAL_ROUND="${CLEAN_FINAL_ROUND:-1}"

PAD_BOOTSTRAP_COUNT="${PAD_BOOTSTRAP_COUNT:-48}"
PAD_COMPARE_LIMIT="${PAD_COMPARE_LIMIT:-80}"
ISIC_BOOTSTRAP_COUNT="${ISIC_BOOTSTRAP_COUNT:-48}"
ISIC_COMPARE_LIMIT="${ISIC_COMPARE_LIMIT:-80}"
SCIN_BOOTSTRAP_COUNT="${SCIN_BOOTSTRAP_COUNT:-48}"
SCIN_COMPARE_LIMIT="${SCIN_COMPARE_LIMIT:-80}"
SD198_BOOTSTRAP_COUNT="${SD198_BOOTSTRAP_COUNT:-48}"
SD198_COMPARE_LIMIT="${SD198_COMPARE_LIMIT:-80}"
XIANGYA_BOOTSTRAP_COUNT="${XIANGYA_BOOTSTRAP_COUNT:-50}"
XIANGYA_COMPARE_LIMIT="${XIANGYA_COMPARE_LIMIT:-13}"
HAM_BOOTSTRAP_COUNT="${HAM_BOOTSTRAP_COUNT:-64}"
HAM_COMPARE_LIMIT="${HAM_COMPARE_LIMIT:-100}"

TOTAL_BOOTSTRAP=$((PAD_BOOTSTRAP_COUNT + ISIC_BOOTSTRAP_COUNT + SCIN_BOOTSTRAP_COUNT + SD198_BOOTSTRAP_COUNT + XIANGYA_BOOTSTRAP_COUNT + HAM_BOOTSTRAP_COUNT))
TOTAL_COMPARE=$((PAD_COMPARE_LIMIT + ISIC_COMPARE_LIMIT + SCIN_COMPARE_LIMIT + SD198_COMPARE_LIMIT + XIANGYA_COMPARE_LIMIT + HAM_COMPARE_LIMIT))
TOTAL_CASE_STEPS=$((TOTAL_BOOTSTRAP + TOTAL_COMPARE * 2))

export SERVER_CHECK_TIMEOUT
export SERVER_WAIT_RETRIES
export SERVER_WAIT_INTERVAL_SECONDS
export SERVER_CHAT_CHECK_RETRIES
export SERVER_CHAT_CHECK_INTERVAL_SECONDS
export DRY_RUN

cd "${PROJECT_ROOT}"

echo "[all-final] clean final round : ${CLEAN_FINAL_ROUND}"
echo "[all-final] dry run           : ${DRY_RUN}"
echo "[all-final] bootstrap total   : ${TOTAL_BOOTSTRAP}"
echo "[all-final] compare total     : ${TOTAL_COMPARE}"
echo "[all-final] compare target ops: $((TOTAL_COMPARE * 2)) (baseline + agent)"
echo "[all-final] total case steps  : ${TOTAL_CASE_STEPS}"
echo "[all-final] server timeout    : ${SERVER_CHECK_TIMEOUT}"
echo "[all-final] model wait        : ${SERVER_WAIT_RETRIES} x ${SERVER_WAIT_INTERVAL_SECONDS}s"
echo "[all-final] chat wait         : ${SERVER_CHAT_CHECK_RETRIES} x ${SERVER_CHAT_CHECK_INTERVAL_SECONDS}s"
echo "[all-final] plan:"
echo "  1. PAD-UFES-20   bootstrap=${PAD_BOOTSTRAP_COUNT} compare=${PAD_COMPARE_LIMIT}"
echo "  2. ISIC2019      bootstrap=${ISIC_BOOTSTRAP_COUNT} compare=${ISIC_COMPARE_LIMIT}"
echo "  3. SCIN grouped  bootstrap=${SCIN_BOOTSTRAP_COUNT} compare=${SCIN_COMPARE_LIMIT}"
echo "  4. SD-198 grouped bootstrap=${SD198_BOOTSTRAP_COUNT} compare=${SD198_COMPARE_LIMIT}"
echo "  5. Xiangya SFT   bootstrap=${XIANGYA_BOOTSTRAP_COUNT} compare=${XIANGYA_COMPARE_LIMIT}"
echo "  6. HAM10000      bootstrap=${HAM_BOOTSTRAP_COUNT} compare=${HAM_COMPARE_LIMIT}"

if [[ "${CLEAN_FINAL_ROUND}" == "1" ]]; then
  echo "[all-final] cleaning old outputs/final_round directories"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] rm -rf outputs/final_round/{pad20,isic2019,scin,sd198,xiangya_sft,ham10000}"
  else
    rm -rf \
      "${PROJECT_ROOT}/outputs/final_round/pad20" \
      "${PROJECT_ROOT}/outputs/final_round/isic2019" \
      "${PROJECT_ROOT}/outputs/final_round/scin" \
      "${PROJECT_ROOT}/outputs/final_round/sd198" \
      "${PROJECT_ROOT}/outputs/final_round/xiangya_sft" \
      "${PROJECT_ROOT}/outputs/final_round/ham10000"
  fi
fi

run_one() {
  local ordinal="$1"
  local label="$2"
  local bootstrap_count="$3"
  local compare_limit="$4"
  local script="$5"
  echo "[all-final] >>> ${ordinal}/6 ${label} bootstrap=${bootstrap_count} compare=${compare_limit}"
  BOOTSTRAP_COUNT="${bootstrap_count}" \
  COMPARE_LIMIT="${compare_limit}" \
  bash "${script}"
  echo "[all-final] <<< ${ordinal}/6 ${label} completed"
}

run_one "1" "PAD-UFES-20" "${PAD_BOOTSTRAP_COUNT}" "${PAD_COMPARE_LIMIT}" "${PROJECT_ROOT}/scripts/run_pad20_final_round.sh"
run_one "2" "ISIC2019" "${ISIC_BOOTSTRAP_COUNT}" "${ISIC_COMPARE_LIMIT}" "${PROJECT_ROOT}/scripts/run_isic2019_final_round.sh"
run_one "3" "SCIN grouped" "${SCIN_BOOTSTRAP_COUNT}" "${SCIN_COMPARE_LIMIT}" "${PROJECT_ROOT}/scripts/run_scin_final_round.sh"
run_one "4" "SD-198 grouped" "${SD198_BOOTSTRAP_COUNT}" "${SD198_COMPARE_LIMIT}" "${PROJECT_ROOT}/scripts/run_sd198_final_round.sh"
run_one "5" "Xiangya SFT" "${XIANGYA_BOOTSTRAP_COUNT}" "${XIANGYA_COMPARE_LIMIT}" "${PROJECT_ROOT}/scripts/run_xiangya_sft_final_round.sh"
run_one "6" "HAM10000" "${HAM_BOOTSTRAP_COUNT}" "${HAM_COMPARE_LIMIT}" "${PROJECT_ROOT}/scripts/run_ham10000_final_round.sh"

echo "[ok] all final rounds completed."
