#!/usr/bin/env bash
#
# 6x6 matrix experiment runner: 6 models × 6 datasets, 30-case bootstrap
# Four-GPU parallel execution with per-model port binding
# Progress tracking via summary.tsv, failure-tolerant
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================================
# Configuration
# ============================================================================

BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-30}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
COMPARE_COUNT="${COMPARE_COUNT:-30}"
COMPARE_SEED="${COMPARE_SEED:-42}"

# Model definitions: name, port, GPU, start script
declare -A MODEL_PORTS=(
  [qwen]=8000
  [medgemma]=8010
  [skinvl]=8011
  [llama]=8012
  [hulumed]=8013
  [dermatollama]=8014
)

declare -A MODEL_GPUS=(
  [qwen]=0
  [llama]=1
  [skinvl]=2
  [hulumed]=3
  [medgemma]=0
  [dermatollama]=1
)

declare -A MODEL_SCRIPTS=(
  [qwen]="${SCRIPT_DIR}/start_qwen_server.sh"
  [medgemma]="${SCRIPT_DIR}/start_medgemma_server.sh"
  [skinvl]="${SCRIPT_DIR}/start_skinvl_server.sh"
  [llama]="${SCRIPT_DIR}/start_llama_server.sh"
  [hulumed]="${SCRIPT_DIR}/start_hulumed_server.sh"
  [dermatollama]="${SCRIPT_DIR}/start_dermatollama_server.sh"
)

# Model name mapping (short name -> actual served model name)
declare -A MODEL_NAMES=(
  [qwen]="Qwen2.5-VL-7B-Instruct"
  [medgemma]="medgemma-4b-it"
  [skinvl]="SkinVL-MM"
  [llama]="Llama-3.2-11B-Vision-Instruct"
  [hulumed]="HuluMed-VL-7B-Instruct"
  [dermatollama]="DermatoLlama-3.2-11B-Vision-Instruct"
)

# Dataset definitions: split_id, data_root, policy_root, split_state_root
declare -A DATASET_SPLIT_IDS=(
  [ham10000]="ham10000_balanced_v1"
  [isic2019]="isic2019_contiguous_v1"
  [pad20]="pad_ufes_20_contiguous_v1"
  [scin]="scin_contiguous_v1"
  [sd198]="sd198_balanced_v1"
  [xiangya]="xiangya_sft_grouped_v1"
)

declare -A DATASET_DATA_ROOTS=(
  [ham10000]="${PROJECT_ROOT}/data/ham10000"
  [isic2019]="${PROJECT_ROOT}/data/isic2019"
  [pad20]="${PROJECT_ROOT}/data/pad_ufes_20"
  [scin]="${PROJECT_ROOT}/data/scin"
  [sd198]="${PROJECT_ROOT}/data/sd198"
  [xiangya]="${PROJECT_ROOT}/data/sft数据"
)

MODELS=(qwen llama skinvl hulumed medgemma dermatollama)
DATASETS=(ham10000 isic2019 pad20 scin sd198 xiangya)

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/6x6_matrix_$(date -u +%Y%m%d_%H%M%S)}"
SUMMARY_TSV="${OUTPUT_ROOT}/summary.tsv"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# ============================================================================
# Helper functions
# ============================================================================

log_info() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [INFO] $*" | tee -a "${OUTPUT_ROOT}/runner.log"
}

log_error() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [ERROR] $*" | tee -a "${OUTPUT_ROOT}/runner.log" >&2
}

init_summary_tsv() {
  if [[ ! -f "${SUMMARY_TSV}" ]]; then
    echo -e "model\tdataset\tstatus\tbootstrap_status\tcompare_status\tstart_time\tend_time\tduration_sec\tlog_file" > "${SUMMARY_TSV}"
  fi
}

update_summary() {
  local model="$1"
  local dataset="$2"
  local status="$3"
  local bootstrap_status="$4"
  local compare_status="$5"
  local start_time="$6"
  local end_time="$7"
  local duration="$8"
  local log_file="$9"

  # Remove old entry if exists
  if [[ -f "${SUMMARY_TSV}" ]]; then
    grep -v "^${model}	${dataset}	" "${SUMMARY_TSV}" > "${SUMMARY_TSV}.tmp" || true
    mv "${SUMMARY_TSV}.tmp" "${SUMMARY_TSV}"
  fi

  echo -e "${model}\t${dataset}\t${status}\t${bootstrap_status}\t${compare_status}\t${start_time}\t${end_time}\t${duration}\t${log_file}" >> "${SUMMARY_TSV}"
}

is_port_listening() {
  local port="$1"
  netstat -ltn 2>/dev/null | grep -q ":${port}.*LISTEN" || \
  lsof -i ":${port}" 2>/dev/null | grep -q LISTEN
}

wait_for_model_ready() {
  local model="$1"
  local port="${MODEL_PORTS[$model]}"
  local max_wait=300
  local waited=0

  log_info "Waiting for ${model} on port ${port}..."

  while (( waited < max_wait )); do
    if is_port_listening "${port}"; then
      # Additional health check via /v1/models (401/403 means server is up)
      local response
      response=$(curl -s -m 5 "http://127.0.0.1:${port}/v1/models" 2>&1 || true)
      if [[ -n "${response}" ]] && [[ "${response}" =~ (Unauthorized|object|data|error) ]]; then
        log_info "${model} is ready on port ${port}"
        return 0
      fi
    fi
    sleep 5
    waited=$((waited + 5))
  done

  log_error "${model} did not become ready after ${max_wait}s"
  return 1
}

start_model_server() {
  local model="$1"
  local port="${MODEL_PORTS[$model]}"
  local gpu="${MODEL_GPUS[$model]}"
  local script="${MODEL_SCRIPTS[$model]}"

  if [[ ! -f "${script}" ]]; then
    log_error "Start script not found for ${model}: ${script}"
    return 1
  fi

  log_info "Starting ${model} on GPU ${gpu}, port ${port}"

  # Kill any existing process on this port
  local pids
  pids=$(netstat -ltnp 2>/dev/null | grep ":${port}.*LISTEN" | awk '{print $7}' | cut -d'/' -f1 | sort -u || true)
  if [[ -n "${pids}" ]]; then
    log_info "Killing existing process(es) on port ${port}: ${pids}"
    kill -TERM ${pids} 2>/dev/null || true
    sleep 5
    kill -KILL ${pids} 2>/dev/null || true
  fi

  # Start model server with GPU binding
  CUDA_VISIBLE_DEVICES="${gpu}" \
  HOST="127.0.0.1" \
  FORCE_RESTART=1 \
  bash "${script}" "${PROJECT_ROOT}" >> "${LOG_DIR}/${model}_server.log" 2>&1 &

  local server_pid=$!
  echo "${server_pid}" > "${OUTPUT_ROOT}/${model}_server.pid"

  # Wait for ready
  if ! wait_for_model_ready "${model}"; then
    log_error "Failed to start ${model}"
    return 1
  fi

  return 0
}

stop_model_server() {
  local model="$1"
  local pid_file="${OUTPUT_ROOT}/${model}_server.pid"

  if [[ -f "${pid_file}" ]]; then
    local pid
    pid=$(cat "${pid_file}")
    if kill -0 "${pid}" 2>/dev/null; then
      log_info "Stopping ${model} (pid ${pid})"
      kill -TERM "${pid}" 2>/dev/null || true
      sleep 10
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
  fi
}

run_bootstrap() {
  local model="$1"
  local dataset="$2"
  local split_id="${DATASET_SPLIT_IDS[$dataset]}"
  local data_root="${DATASET_DATA_ROOTS[$dataset]}"
  local port="${MODEL_PORTS[$model]}"
  local model_name="${MODEL_NAMES[$model]}"

  local output_dir="${OUTPUT_ROOT}/${model}/${dataset}/bootstrap"
  local policy_root="${OUTPUT_ROOT}/${model}/${dataset}/policy"
  local split_state_root="${OUTPUT_ROOT}/${model}/${dataset}/split_states"
  local split_json="${OUTPUT_ROOT}/${model}/${dataset}/${dataset}_split.json"

  mkdir -p "${output_dir}" "${policy_root}" "${split_state_root}"

  # Generate split JSON
  cd "${PROJECT_ROOT}"
  python - <<PY
from pathlib import Path
from configs.dataset_splits import write_fixed_split_json
write_fixed_split_json(
    Path("${split_json}"),
    split_id="${split_id}",
    data_root=Path("${data_root}").parent,
)
PY

  # Generate stratified sample indices
  local indices_json="${OUTPUT_ROOT}/${model}/${dataset}/bootstrap_indices.json"
  python "${SCRIPT_DIR}/bootstrap_stratified_30case.py" \
    --split-id "${split_id}" \
    --data-root "$(dirname "${data_root}")" \
    --split-name train \
    --n-samples "${BOOTSTRAP_COUNT}" \
    --seed "${BOOTSTRAP_SEED}" \
    --output "${indices_json}"

  # Read indices
  mapfile -t CASE_INDICES < <(python -c "import json; print('\n'.join(map(str, json.load(open('${indices_json}'))['selected_indices'])))")

  log_info "${model}/${dataset}: Running bootstrap on ${#CASE_INDICES[@]} cases"

  local failed=0
  for idx in "${!CASE_INDICES[@]}"; do
    local case_idx="${CASE_INDICES[$idx]}"
    local current=$((idx + 1))
    log_info "${model}/${dataset}: Bootstrap case ${current}/${#CASE_INDICES[@]} (index=${case_idx})"

    if ! DERMAGENT_POLICY_ROOT="${policy_root}" \
         DERMAGENT_SPLIT_STATE_ROOT="${split_state_root}" \
         python "${SCRIPT_DIR}/debug_single_case.py" \
           --case-index "${case_idx}" \
           --data-root "${data_root}" \
           --output-dir "${output_dir}" \
           --enable-writeback \
           --data-split train \
           --run-mode "${dataset}_bootstrap_30case" \
           --client-base-url "http://127.0.0.1:${port}/v1" \
           --client-api-key "EMPTY" \
           --client-model "${model_name}" \
           --client-timeout 120 \
           >> "${LOG_DIR}/${model}_${dataset}_bootstrap.log" 2>&1; then
      log_error "${model}/${dataset}: Bootstrap failed at case ${case_idx}"
      failed=$((failed + 1))
    fi
  done

  if (( failed > 0 )); then
    log_error "${model}/${dataset}: Bootstrap completed with ${failed} failures"
    return 1
  fi

  log_info "${model}/${dataset}: Bootstrap completed successfully"
  return 0
}

run_compare() {
  local model="$1"
  local dataset="$2"
  local split_id="${DATASET_SPLIT_IDS[$dataset]}"
  local data_root="${DATASET_DATA_ROOTS[$dataset]}"
  local port="${MODEL_PORTS[$model]}"
  local model_name="${MODEL_NAMES[$model]}"

  local output_dir="${OUTPUT_ROOT}/${model}/${dataset}/compare"
  local policy_root="${OUTPUT_ROOT}/${model}/${dataset}/policy"
  local split_state_root="${OUTPUT_ROOT}/${model}/${dataset}/split_states"
  local split_json="${OUTPUT_ROOT}/${model}/${dataset}/${dataset}_split.json"

  mkdir -p "${output_dir}"

  # Promote state from train to val/test
  python "${SCRIPT_DIR}/manage_dataset_experiment_assets.py" promote-state \
    --split-state-root "${split_state_root}" \
    --source-split train \
    --target-splits val,test \
    >> "${LOG_DIR}/${model}_${dataset}_promote.log" 2>&1 || true

  log_info "${model}/${dataset}: Running compare on ${COMPARE_COUNT} test cases"

  if ! DERMAGENT_POLICY_ROOT="${policy_root}" \
       DERMAGENT_SPLIT_STATE_ROOT="${split_state_root}" \
       OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" \
       OPENAI_API_KEY="EMPTY" \
       OPENAI_MODEL="${model_name}" \
       OPENAI_TIMEOUT=120 \
       python "${SCRIPT_DIR}/compare_agent_vs_qwen.py" \
         --data-root "${data_root}" \
         --limit "${COMPARE_COUNT}" \
         --case-offset 0 \
         --data-split test \
         --split-json "${split_json}" \
         --output-dir "${output_dir}" \
         --policy-label "${model} ${dataset} 30case policy" \
         >> "${LOG_DIR}/${model}_${dataset}_compare.log" 2>&1; then
    log_error "${model}/${dataset}: Compare failed"
    return 1
  fi

  log_info "${model}/${dataset}: Compare completed successfully"
  return 0
}

run_model_all_datasets() {
  local model="$1"

  log_info "========== Starting ${model} lane =========="

  # Start model server
  if ! start_model_server "${model}"; then
    log_error "${model}: Failed to start server, skipping all datasets"
    for dataset in "${DATASETS[@]}"; do
      update_summary "${model}" "${dataset}" "SKIPPED" "SKIPPED" "SKIPPED" "" "" "0" ""
    done
    return 1
  fi

  # Run all datasets sequentially for this model
  for dataset in "${DATASETS[@]}"; do
    local start_time
    start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local start_sec
    start_sec=$(date +%s)

    log_info "${model}/${dataset}: Starting experiment"

    local bootstrap_status="PENDING"
    local compare_status="PENDING"
    local overall_status="RUNNING"

    # Bootstrap
    if run_bootstrap "${model}" "${dataset}"; then
      bootstrap_status="OK"
    else
      bootstrap_status="FAILED"
      overall_status="FAILED"
    fi

    # Compare (run even if bootstrap failed, to test baseline)
    if [[ "${bootstrap_status}" == "OK" ]]; then
      if run_compare "${model}" "${dataset}"; then
        compare_status="OK"
      else
        compare_status="FAILED"
        overall_status="FAILED"
      fi
    else
      compare_status="SKIPPED"
    fi

    local end_time
    end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local end_sec
    end_sec=$(date +%s)
    local duration=$((end_sec - start_sec))

    if [[ "${overall_status}" == "RUNNING" ]]; then
      overall_status="OK"
    fi

    update_summary "${model}" "${dataset}" "${overall_status}" "${bootstrap_status}" "${compare_status}" \
      "${start_time}" "${end_time}" "${duration}" "${LOG_DIR}/${model}_${dataset}_*.log"

    log_info "${model}/${dataset}: Completed (${overall_status})"
  done

  # Stop model server
  stop_model_server "${model}"

  log_info "========== Finished ${model} lane =========="
}

# ============================================================================
# Main execution
# ============================================================================

main() {
  log_info "Starting 6x6 matrix experiment"
  log_info "Models: ${MODELS[*]}"
  log_info "Datasets: ${DATASETS[*]}"
  log_info "Bootstrap count: ${BOOTSTRAP_COUNT}"
  log_info "Compare count: ${COMPARE_COUNT}"
  log_info "Output root: ${OUTPUT_ROOT}"

  init_summary_tsv

  # Run models in parallel (4 GPUs, so 4 concurrent models max)
  # We'll use background jobs and wait
  local pids=()
  local model_idx=0

  for model in "${MODELS[@]}"; do
    run_model_all_datasets "${model}" &
    pids+=($!)

    model_idx=$((model_idx + 1))

    # Limit to 4 concurrent models (4 GPUs)
    if (( model_idx % 4 == 0 )); then
      log_info "Waiting for current batch of 4 models to complete..."
      for pid in "${pids[@]}"; do
        wait "${pid}" || true
      done
      pids=()
    fi
  done

  # Wait for remaining models
  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done

  log_info "All experiments completed"
  log_info "Summary: ${SUMMARY_TSV}"

  # Print summary
  echo ""
  echo "========== SUMMARY =========="
  column -t -s $'\t' "${SUMMARY_TSV}"
}

main "$@"
