#!/usr/bin/env bash
#
# Re-run the six Qwen final-round datasets with the old final-round frozen
# assets and dataset workflow routing only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dermagent-6x6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/qwen_final_round_asset_rerun_$(date -u +%Y%m%dT%H%M%SZ)}"
QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:8000/v1}"
QWEN_MODEL="${QWEN_MODEL:-Qwen2.5-VL-7B-Instruct}"
QWEN_API_KEY="${QWEN_API_KEY:-EMPTY}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-120}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-1}"
CASE_LIMIT_OVERRIDE="${CASE_LIMIT_OVERRIDE:-}"
RUN_DATASETS="${RUN_DATASETS:-pad20 isic2019 scin sd198 xiangya_sft ham10000}"

mkdir -p "${OUTPUT_ROOT}/asset_roots" "${OUTPUT_ROOT}/logs"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [qwen-asset-rerun] $*" | tee -a "${OUTPUT_ROOT}/rerun.log"
}

activate_env() {
  # shellcheck disable=SC1091
  source /home/zhongnan/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
}

require_qwen() {
  curl -s -m 5 -H "Authorization: Bearer ${QWEN_API_KEY}" "${QWEN_BASE_URL}/models" >/dev/null
}

stage_assets() {
  local dataset="$1"
  local final_dataset="$2"
  local compare_dir
  compare_dir="$(find "${PROJECT_ROOT}/outputs/final_round/${final_dataset}/compare" -maxdepth 1 -type d -name 'compare_agent_vs_qwen_*' | sort | tail -1)"
  if [[ -z "${compare_dir}" ]]; then
    log "ERROR: missing old final-round compare dir for ${final_dataset}"
    return 1
  fi
  local frozen_root="${compare_dir}/frozen_state"
  local asset_root="${OUTPUT_ROOT}/asset_roots/${dataset}"
  rm -rf "${asset_root}"
  mkdir -p "${asset_root}/test"
  ln -s "${frozen_root}/experience" "${asset_root}/test/experience"
  ln -s "${frozen_root}/cognition_state.json" "${asset_root}/test/cognition_state.json"
  echo "${frozen_root}"
}

dataset_config() {
  local dataset="$1"
  case "${dataset}" in
    pad20)
      echo "pad20|pad20|${PROJECT_ROOT}/data/pad_ufes_20|${PROJECT_ROOT}/outputs/overnight_eval_bundle/split_sanity.json|80|1953"
      ;;
    isic2019)
      echo "isic2019|isic2019|${PROJECT_ROOT}/data/isic2019|${PROJECT_ROOT}/outputs/dataset_adaptation/isic2019_v1/isic2019_split.json|80|21531"
      ;;
    scin)
      echo "scin|scin|${PROJECT_ROOT}/data/scin/official_mirror|${PROJECT_ROOT}/outputs/dataset_adaptation/scin_v1/scin_split.json|80|4278"
      ;;
    sd198)
      echo "sd198|sd198|${PROJECT_ROOT}/data/sd198/sd-198|${PROJECT_ROOT}/outputs/dataset_adaptation/sd198_v1/sd198_split.json|80|0"
      ;;
    xiangya_sft|xiangya)
      echo "xiangya_sft|xiangya_sft|${PROJECT_ROOT}/data/sft数据|${PROJECT_ROOT}/outputs/dataset_adaptation/xiangya_sft_v1/xiangya_sft_split.json|13|0"
      ;;
    ham10000)
      echo "ham10000|ham10000|${PROJECT_ROOT}/data/ham10000|${PROJECT_ROOT}/outputs/dataset_adaptation/ham10000_v2/ham10000_split.json|100|0"
      ;;
    *)
      log "ERROR: unknown dataset ${dataset}"
      return 1
      ;;
  esac
}

run_dataset() {
  local requested_dataset="$1"
  local cfg dataset final_dataset data_root split_json full_limit case_offset frozen_root limit
  cfg="$(dataset_config "${requested_dataset}")"
  IFS='|' read -r dataset final_dataset data_root split_json full_limit case_offset <<<"${cfg}"
  frozen_root="$(stage_assets "${dataset}" "${final_dataset}")"
  limit="${full_limit}"
  if [[ -n "${CASE_LIMIT_OVERRIDE}" ]]; then
    limit="${CASE_LIMIT_OVERRIDE}"
    if (( limit > full_limit )); then
      limit="${full_limit}"
    fi
  fi

  local label_space_env=()
  case "${dataset}" in
    scin)
      label_space_env=(DERMAGENT_SCIN_LABEL_SPACE_ID="${DERMAGENT_SCIN_LABEL_SPACE_ID:-scin_grouped}")
      ;;
    sd198)
      label_space_env=(DERMAGENT_SD198_LABEL_SPACE_ID="${DERMAGENT_SD198_LABEL_SPACE_ID:-sd198_grouped}")
      ;;
  esac

  log "Running ${dataset}: limit=${limit}, offset=${case_offset}, frozen=${frozen_root}"
  if ((${#label_space_env[@]})); then
    log "Dataset label-space override: ${label_space_env[*]}"
  fi
  env \
    "${label_space_env[@]}" \
    DERMAGENT_SPLIT_STATE_ROOT="${OUTPUT_ROOT}/asset_roots/${dataset}" \
    DERMAGENT_DISABLE_MODEL_WORKFLOW_ROUTING=1 \
    python "${PROJECT_ROOT}/scripts/compare_agent_vs_qwen.py" \
    --data-root "${data_root}" \
    --limit "${limit}" \
    --case-offset "${case_offset}" \
    --data-split test \
    --split-json "${split_json}" \
    --output-dir "${OUTPUT_ROOT}/${dataset}/compare" \
    --policy-config "${frozen_root}/policy.json" \
    --policy-label "qwen final-round frozen-asset rerun (${dataset})" \
    --agent-base-url "${QWEN_BASE_URL}" \
    --agent-api-key "${QWEN_API_KEY}" \
    --agent-model "${QWEN_MODEL}" \
    --client-timeout "${CLIENT_TIMEOUT}" \
    --client-max-retries "${CLIENT_MAX_RETRIES}" \
    --disable-model-workflow-routing \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${dataset}.log"
}

main() {
  activate_env
  log "Output root: ${OUTPUT_ROOT}"
  log "Datasets: ${RUN_DATASETS}"
  if ! require_qwen; then
    log "ERROR: Qwen endpoint is not reachable at ${QWEN_BASE_URL}"
    return 1
  fi
  for dataset in ${RUN_DATASETS}; do
    run_dataset "${dataset}"
  done
  log "Done. Results under ${OUTPUT_ROOT}"
}

main "$@"
