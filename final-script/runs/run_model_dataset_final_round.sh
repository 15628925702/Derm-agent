#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required, e.g. llama, hulumed, dermatollama, medgemma, skinvl, qwen}"
DATASET_ID="${DATASET_ID:?DATASET_ID is required, e.g. pad20, isic2019, scin, sd198, xiangya_sft, ham10000}"
POLICY_VARIANT="${POLICY_VARIANT:-final_round_20260424_manual}"

source "${PROJECT_ROOT}/final-script/configs/common.env"
source "${PROJECT_ROOT}/final-script/configs/final_assets_registry.env"
source "${PROJECT_ROOT}/final-script/configs/model_servers.env"

model_key="$(printf '%s' "${MODEL_KEY}" | tr '[:upper:]' '[:lower:]')"
dataset_id="$(printf '%s' "${DATASET_ID}" | tr '[:upper:]' '[:lower:]')"

case "${model_key}" in
  qwen)
    CLIENT_BASE_URL="${CLIENT_BASE_URL:-${QWEN_SERVER_BASE_URL}}"
    CLIENT_API_KEY="${CLIENT_API_KEY:-${QWEN_SERVER_API_KEY}}"
    CLIENT_MODEL="${CLIENT_MODEL:-${QWEN_SERVER_MODEL}}"
    ;;
  medgemma)
    CLIENT_BASE_URL="${CLIENT_BASE_URL:-${MEDGEMMA_SERVER_BASE_URL}}"
    CLIENT_API_KEY="${CLIENT_API_KEY:-${MEDGEMMA_SERVER_API_KEY}}"
    CLIENT_MODEL="${CLIENT_MODEL:-${MEDGEMMA_SERVER_MODEL}}"
    ;;
  skinvl)
    CLIENT_BASE_URL="${CLIENT_BASE_URL:-${SKINVL_SERVER_BASE_URL}}"
    CLIENT_API_KEY="${CLIENT_API_KEY:-${SKINVL_SERVER_API_KEY}}"
    CLIENT_MODEL="${CLIENT_MODEL:-${SKINVL_SERVER_MODEL}}"
    ;;
  llama)
    CLIENT_BASE_URL="${CLIENT_BASE_URL:-${LLAMA_SERVER_BASE_URL}}"
    CLIENT_API_KEY="${CLIENT_API_KEY:-${LLAMA_SERVER_API_KEY}}"
    CLIENT_MODEL="${CLIENT_MODEL:-${LLAMA_SERVER_MODEL}}"
    ;;
  hulumed)
    CLIENT_BASE_URL="${CLIENT_BASE_URL:-${HULUMED_SERVER_BASE_URL}}"
    CLIENT_API_KEY="${CLIENT_API_KEY:-${HULUMED_SERVER_API_KEY}}"
    CLIENT_MODEL="${CLIENT_MODEL:-${HULUMED_SERVER_MODEL}}"
    ;;
  dermatollama)
    CLIENT_BASE_URL="${CLIENT_BASE_URL:-${DERMATOLLAMA_SERVER_BASE_URL}}"
    CLIENT_API_KEY="${CLIENT_API_KEY:-${DERMATOLLAMA_SERVER_API_KEY}}"
    CLIENT_MODEL="${CLIENT_MODEL:-${DERMATOLLAMA_SERVER_MODEL}}"
    ;;
  *)
    echo "[error] unsupported MODEL_KEY=${MODEL_KEY}" >&2
    exit 2
    ;;
esac

RUN_GROUP="${RUN_GROUP:-${model_key}_manual_policy_final_round}"
MODEL_STATE_ROOT="${MODEL_STATE_ROOT:-${PROJECT_ROOT}/state/model_runs/${RUN_GROUP}}"
MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/model_runs/${RUN_GROUP}}"
DATASET_STATE_ROOT="${MODEL_STATE_ROOT}/${dataset_id}/split_states"
DATASET_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}/${dataset_id}"

case "${POLICY_VARIANT}" in
  final_round_20260424_manual|manual|heuristic)
    case "${dataset_id}" in
      pad20|pad_ufes_20|pad)
        DATASET_ID="pad20"
        DATASET_LABEL="PAD-UFES-20"
        DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/pad_ufes_20}"
        SOURCE_SPLIT_STATE_ROOT="${PROJECT_ROOT}/state/split_states"
        COMPARE_SPLIT_JSON="${COMPARE_SPLIT_JSON:-${PROJECT_ROOT}/outputs/overnight_eval_bundle/split_sanity.json}"
        COMPARE_POLICY_CONFIG="${COMPARE_POLICY_CONFIG:-${PROJECT_ROOT}/state/policy/current_stable_policy.json}"
        BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${PROJECT_ROOT}/scripts/bootstrap_pad20_train_cases.sh}"
        BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-48}"
        COMPARE_LIMIT="${COMPARE_LIMIT:-80}"
        COMPARE_CASE_OFFSET="${COMPARE_CASE_OFFSET:-0}"
        ;;
      isic2019|isic)
        DATASET_ID="isic2019"
        DATASET_LABEL="ISIC2019"
        DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/isic2019}"
        SOURCE_SPLIT_STATE_ROOT="${PROJECT_ROOT}/state/dataset_adaptation/isic2019_v1/split_states"
        COMPARE_SPLIT_JSON="${COMPARE_SPLIT_JSON:-${PROJECT_ROOT}/outputs/dataset_adaptation/isic2019_v1/isic2019_split.json}"
        COMPARE_POLICY_CONFIG="${COMPARE_POLICY_CONFIG:-${PROJECT_ROOT}/state/dataset_adaptation/isic2019_v1/policy/current_stable_policy.json}"
        BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${PROJECT_ROOT}/scripts/bootstrap_isic2019_train_cases.sh}"
        BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-48}"
        COMPARE_LIMIT="${COMPARE_LIMIT:-80}"
        COMPARE_CASE_OFFSET="${COMPARE_CASE_OFFSET:-0}"
        ;;
      scin)
        export DERMAGENT_SCIN_LABEL_SPACE_ID="${DERMAGENT_SCIN_LABEL_SPACE_ID:-scin_grouped}"
        DATASET_ID="scin"
        DATASET_LABEL="SCIN grouped"
        DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/scin/official_mirror}"
        SOURCE_SPLIT_STATE_ROOT="${PROJECT_ROOT}/state/dataset_adaptation/scin_v1/split_states"
        COMPARE_SPLIT_JSON="${COMPARE_SPLIT_JSON:-${PROJECT_ROOT}/outputs/dataset_adaptation/scin_v1/scin_split.json}"
        COMPARE_POLICY_CONFIG="${COMPARE_POLICY_CONFIG:-${PROJECT_ROOT}/state/dataset_adaptation/scin_v1/policy/current_stable_policy.json}"
        BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${PROJECT_ROOT}/scripts/bootstrap_scin_train_cases.sh}"
        BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-48}"
        COMPARE_LIMIT="${COMPARE_LIMIT:-80}"
        COMPARE_CASE_OFFSET="${COMPARE_CASE_OFFSET:-4278}"
        ;;
      sd198)
        export DERMAGENT_SD198_LABEL_SPACE_ID="${DERMAGENT_SD198_LABEL_SPACE_ID:-sd198_grouped}"
        DATASET_ID="sd198"
        DATASET_LABEL="SD-198 grouped"
        DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/sd198/sd-198}"
        SOURCE_SPLIT_STATE_ROOT="${PROJECT_ROOT}/state/dataset_adaptation/sd198_v1/split_states"
        COMPARE_SPLIT_JSON="${COMPARE_SPLIT_JSON:-${PROJECT_ROOT}/outputs/dataset_adaptation/sd198_v1/sd198_split.json}"
        COMPARE_POLICY_CONFIG="${COMPARE_POLICY_CONFIG:-${PROJECT_ROOT}/state/dataset_adaptation/sd198_v1/policy/current_stable_policy.json}"
        BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${PROJECT_ROOT}/scripts/bootstrap_sd198_train_cases.sh}"
        BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-48}"
        COMPARE_LIMIT="${COMPARE_LIMIT:-80}"
        COMPARE_CASE_OFFSET="${COMPARE_CASE_OFFSET:-0}"
        ;;
      xiangya_sft|xiangya)
        DATASET_ID="xiangya_sft"
        DATASET_LABEL="Xiangya SFT"
        DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/sft数据}"
        SOURCE_SPLIT_STATE_ROOT="${PROJECT_ROOT}/state/dataset_adaptation/xiangya_sft_v1/split_states"
        COMPARE_SPLIT_JSON="${COMPARE_SPLIT_JSON:-${PROJECT_ROOT}/outputs/dataset_adaptation/xiangya_sft_v1/xiangya_sft_split.json}"
        COMPARE_POLICY_CONFIG="${COMPARE_POLICY_CONFIG:-${PROJECT_ROOT}/state/dataset_adaptation/xiangya_sft_v1/policy/current_stable_policy.json}"
        BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${PROJECT_ROOT}/scripts/bootstrap_xiangya_sft_train_cases.sh}"
        BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-50}"
        COMPARE_LIMIT="${COMPARE_LIMIT:-13}"
        COMPARE_CASE_OFFSET="${COMPARE_CASE_OFFSET:-0}"
        ;;
      ham10000|ham)
        DATASET_ID="ham10000"
        DATASET_LABEL="HAM10000"
        DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/ham10000}"
        SOURCE_SPLIT_STATE_ROOT="${PROJECT_ROOT}/state/dataset_adaptation/ham10000_v2/split_states"
        COMPARE_SPLIT_JSON="${COMPARE_SPLIT_JSON:-${PROJECT_ROOT}/outputs/dataset_adaptation/ham10000_v2/ham10000_split.json}"
        COMPARE_POLICY_CONFIG="${COMPARE_POLICY_CONFIG:-${PROJECT_ROOT}/state/dataset_adaptation/ham10000_v2/policy/current_stable_policy.json}"
        BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${PROJECT_ROOT}/scripts/bootstrap_ham10000_train_cases.sh}"
        BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-64}"
        COMPARE_LIMIT="${COMPARE_LIMIT:-100}"
        COMPARE_CASE_OFFSET="${COMPARE_CASE_OFFSET:-0}"
        ;;
      *)
        echo "[error] unsupported DATASET_ID=${DATASET_ID}" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "[error] unsupported POLICY_VARIANT=${POLICY_VARIANT}; use final_round_20260424_manual" >&2
    exit 2
    ;;
esac

POLICY_ROOT="$(dirname "${COMPARE_POLICY_CONFIG}")"
SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT:-${DATASET_STATE_ROOT}}"
ROUND_ROOT="${ROUND_ROOT:-${DATASET_OUTPUT_ROOT}/round}"
BOOTSTRAP_OUTPUT_DIR="${BOOTSTRAP_OUTPUT_DIR:-${DATASET_OUTPUT_ROOT}/bootstrap}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-${DATASET_OUTPUT_ROOT}/compare}"
RUN_MODE="${RUN_MODE:-${model_key}_${DATASET_ID}_manual_policy_bootstrap}"
COMPARE_POLICY_LABEL="${COMPARE_POLICY_LABEL:-${DATASET_LABEL} 2026-04-24 manual final-round policy on ${CLIENT_MODEL}}"
COMPARE_DATA_SPLIT="${COMPARE_DATA_SPLIT:-test}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] mkdir -p ${SPLIT_STATE_ROOT} ${BOOTSTRAP_OUTPUT_DIR} ${COMPARE_OUTPUT_DIR} ${ROUND_ROOT}"
else
  mkdir -p "${SPLIT_STATE_ROOT}" "${BOOTSTRAP_OUTPUT_DIR}" "${COMPARE_OUTPUT_DIR}" "${ROUND_ROOT}"
fi

if [[ ! -e "${SPLIT_STATE_ROOT}/train/cognition_state.json" && -d "${SOURCE_SPLIT_STATE_ROOT}" ]]; then
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] seed isolated split state from ${SOURCE_SPLIT_STATE_ROOT} to ${SPLIT_STATE_ROOT}"
  else
    echo "[setup] seeding isolated split state from ${SOURCE_SPLIT_STATE_ROOT}"
    cp -a "${SOURCE_SPLIT_STATE_ROOT}/." "${SPLIT_STATE_ROOT}/"
  fi
fi

echo "[model-final] model key      : ${model_key}"
echo "[model-final] client model   : ${CLIENT_MODEL}"
echo "[model-final] dataset        : ${DATASET_LABEL}"
echo "[model-final] policy variant : ${POLICY_VARIANT}"
echo "[model-final] policy config  : ${COMPARE_POLICY_CONFIG}"
echo "[model-final] split state    : ${SPLIT_STATE_ROOT}"
echo "[model-final] output root    : ${DATASET_OUTPUT_ROOT}"

export DATASET_ID
export DATASET_LABEL
export DATA_ROOT
export POLICY_ROOT
export SPLIT_STATE_ROOT
export COMPARE_SPLIT_JSON
export COMPARE_POLICY_CONFIG
export BOOTSTRAP_SCRIPT
export BOOTSTRAP_COUNT
export BOOTSTRAP_OUTPUT_DIR
export RUN_MODE
export COMPARE_LIMIT
export COMPARE_CASE_OFFSET
export COMPARE_DATA_SPLIT
export COMPARE_OUTPUT_DIR
export COMPARE_POLICY_LABEL
export ROUND_ROOT
export CLIENT_BASE_URL
export CLIENT_API_KEY
export CLIENT_MODEL
export STOP_ON_ERROR

bash "${PROJECT_ROOT}/scripts/run_dataset_final_round.sh"
