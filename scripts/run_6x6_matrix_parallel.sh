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
USE_EXISTING_SERVERS="${USE_EXISTING_SERVERS:-0}"
MAX_PARALLEL_MODELS="${MAX_PARALLEL_MODELS:-6}"

# Model definitions: name, port, GPU, start script
declare -A MODEL_PORTS=(
  [qwen]=8000
  [medgemma]=8010
  [skinvl]=8011
  [llama]=8012
  [hulumed]=8013
  [dermatollama]=8014
  [llama_replica]=8022
  [hulumed_replica]=8023
  [dermatollama_replica]=8024
  [skinvl_heavy]=8025
  [skinvl_light]=8026
)

declare -A MODEL_GPUS=(
  [llama_replica]=0
  [dermatollama_replica]=1
  [qwen]=2
  [medgemma]=3
  [skinvl]=4
  [llama]=5
  [hulumed]=6
  [dermatollama]=7
  [hulumed_replica]=4
  [skinvl_heavy]=2
  [skinvl_light]=3
)

declare -A MODEL_SCRIPTS=(
  [qwen]="${SCRIPT_DIR}/start_qwen_server.sh"
  [medgemma]="${SCRIPT_DIR}/start_medgemma_server.sh"
  [skinvl]="${SCRIPT_DIR}/start_skinvl_server.sh"
  [llama]="${SCRIPT_DIR}/start_llama_server.sh"
  [hulumed]="${SCRIPT_DIR}/start_hulumed_server.sh"
  [dermatollama]="${SCRIPT_DIR}/start_dermatollama_server.sh"
  [llama_replica]="${SCRIPT_DIR}/start_llama_server.sh"
  [hulumed_replica]="${SCRIPT_DIR}/start_hulumed_server.sh"
  [dermatollama_replica]="${SCRIPT_DIR}/start_dermatollama_server.sh"
  [skinvl_heavy]="${SCRIPT_DIR}/start_skinvl_server.sh"
  [skinvl_light]="${SCRIPT_DIR}/start_skinvl_server.sh"
)

# Model name mapping (short name -> actual served model name)
declare -A MODEL_NAMES=(
  [qwen]="Qwen2.5-VL-7B-Instruct"
  [medgemma]="medgemma-4b-it"
  [skinvl]="SkinVL-MM"
  [llama]="Llama-3.2-11B-Vision-Instruct"
  [hulumed]="Hulu-Med-7B"
  [dermatollama]="DermatoLlama-full"
  [llama_replica]="Llama-3.2-11B-Vision-Instruct"
  [hulumed_replica]="Hulu-Med-7B"
  [dermatollama_replica]="DermatoLlama-full"
  [skinvl_heavy]="SkinVL-MM"
  [skinvl_light]="SkinVL-MM"
)

declare -A MODEL_OUTPUT_NAMES=(
  [llama_replica]="llama"
  [hulumed_replica]="hulumed"
  [dermatollama_replica]="dermatollama"
  [skinvl_heavy]="skinvl"
  [skinvl_light]="skinvl"
)

declare -A MODEL_DATASETS=()

# Dataset definitions:
# - DATASET_DATA_ROOTS are the runtime roots consumed by dataio loaders.
# - DATASET_SPLIT_ROOTS are the repository data roots consumed by configs/dataset_splits.py.
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

declare -A DATASET_SPLIT_ROOTS=(
  [ham10000]="${PROJECT_ROOT}/data"
  [isic2019]="${PROJECT_ROOT}/data"
  [pad20]="${PROJECT_ROOT}/data"
  [scin]="${PROJECT_ROOT}/data"
  [sd198]="${PROJECT_ROOT}/data"
  [xiangya]="${PROJECT_ROOT}/data"
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

canonical_model_name() {
  local model="$1"
  if [[ -n "${MODEL_OUTPUT_NAMES[$model]+x}" ]]; then
    printf '%s\n' "${MODEL_OUTPUT_NAMES[$model]}"
  else
    printf '%s\n' "${model}"
  fi
}

datasets_for_model() {
  local model="$1"
  if [[ -n "${MODEL_DATASETS[$model]+x}" ]]; then
    printf '%s\n' "${MODEL_DATASETS[$model]}"
  else
    printf '%s\n' "${DATASETS[*]}"
  fi
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

  if [[ "${USE_EXISTING_SERVERS}" == "1" ]]; then
    log_info "Using existing ${model} server on port ${port}"
    wait_for_model_ready "${model}"
    return $?
  fi

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

  if [[ "${USE_EXISTING_SERVERS}" == "1" ]]; then
    log_info "Leaving existing ${model} server running"
    return 0
  fi

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

  STOP_WAIT_SECONDS="${STOP_WAIT_SECONDS:-10}" \
  GPU_RELEASE_WAIT_SECONDS="${GPU_RELEASE_WAIT_SECONDS:-20}" \
  WAIT_SECONDS=30 \
  bash "${SCRIPT_DIR}/switch_model_server.sh" stop "${PROJECT_ROOT}" \
    >> "${LOG_DIR}/${model}_server_stop.log" 2>&1 || true
}

dataset_env_vars() {
  local dataset="$1"
  case "${dataset}" in
    scin)
      printf '%s\n' "DERMAGENT_SCIN_LABEL_SPACE_ID=${DERMAGENT_SCIN_LABEL_SPACE_ID:-scin_grouped}"
      ;;
    sd198)
      printf '%s\n' "DERMAGENT_SD198_LABEL_SPACE_ID=${DERMAGENT_SD198_LABEL_SPACE_ID:-sd198_grouped}"
      ;;
  esac
}

write_dataset_split_json() {
  local dataset="$1"
  local split_id="$2"
  local split_root="$3"
  local split_json="$4"

  DATASET="${dataset}" \
  SPLIT_ID="${split_id}" \
  SPLIT_ROOT="${split_root}" \
  SPLIT_JSON="${split_json}" \
  python - <<'PY'
import json
import os
from pathlib import Path

from configs.dataset_splits import build_fixed_split_payload
from dataio.scin_loader import load_scin_rows

dataset = os.environ["DATASET"]
split_id = os.environ["SPLIT_ID"]
split_root = Path(os.environ["SPLIT_ROOT"])
split_json = Path(os.environ["SPLIT_JSON"])

payload = build_fixed_split_payload(split_id=split_id, data_root=split_root)

if dataset == "scin":
    rows = load_scin_rows(split_root / "scin" / "official_mirror")
    filtered: dict[str, list[int]] = {}
    filtered_ids: dict[str, list[str]] = {}
    for split_name in ("train", "val", "test"):
        raw_range = payload.get(f"{split_name}_range")
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise ValueError(f"SCIN split payload missing {split_name}_range")
        start, end = int(raw_range[0]), int(raw_range[1])
        indices = [
            idx for idx in range(start, end + 1)
            if idx < len(rows) and str(rows[idx].get("original_label", "")).strip()
        ]
        filtered[split_name] = indices
        filtered_ids[split_name] = [str(rows[idx].get("case_id", "")).strip() for idx in indices]
        payload[f"{split_name}_case_indices"] = indices
        payload[split_name] = filtered_ids[split_name]

    ordered = filtered["train"] + filtered["val"] + filtered["test"]
    train_end = len(filtered["train"]) - 1
    val_start = len(filtered["train"])
    val_end = val_start + len(filtered["val"]) - 1
    test_start = val_end + 1
    test_end = test_start + len(filtered["test"]) - 1
    payload["train_range"] = [0, train_end] if filtered["train"] else [0, -1]
    payload["val_range"] = [val_start, val_end] if filtered["val"] else [val_start, val_start - 1]
    payload["test_range"] = [test_start, test_end] if filtered["test"] else [test_start, test_start - 1]
    payload["total_cases"] = len(ordered)
    payload["strategy"] = f"{payload.get('strategy', 'contiguous_by_metadata_index')}_labeled_only"
    payload["notes"] = str(payload.get("notes", "")).rstrip() + " 6x6 matrix runner filters SCIN to labeled/scorable cases."

split_json.parent.mkdir(parents=True, exist_ok=True)
split_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

run_bootstrap() {
  local model="$1"
  local dataset="$2"
  local split_id="${DATASET_SPLIT_IDS[$dataset]}"
  local data_root="${DATASET_DATA_ROOTS[$dataset]}"
  local split_root="${DATASET_SPLIT_ROOTS[$dataset]}"
  local port="${MODEL_PORTS[$model]}"
  local model_name="${MODEL_NAMES[$model]}"
  local output_model
  output_model="$(canonical_model_name "${model}")"

  local output_dir="${OUTPUT_ROOT}/${output_model}/${dataset}/bootstrap"
  local policy_root="${OUTPUT_ROOT}/${output_model}/${dataset}/policy"
  local split_state_root="${OUTPUT_ROOT}/${output_model}/${dataset}/split_states"
  local split_json="${OUTPUT_ROOT}/${output_model}/${dataset}/${dataset}_split.json"

  mkdir -p "${output_dir}" "${policy_root}" "${split_state_root}"

  # Generate split JSON
  cd "${PROJECT_ROOT}"
  write_dataset_split_json "${dataset}" "${split_id}" "${split_root}" "${split_json}"

  # Generate stratified sample indices
  local indices_json="${OUTPUT_ROOT}/${output_model}/${dataset}/bootstrap_indices.json"
  python "${SCRIPT_DIR}/bootstrap_stratified_30case.py" \
    --split-id "${split_id}" \
    --data-root "${split_root}" \
    --split-name train \
    --n-samples "${BOOTSTRAP_COUNT}" \
    --seed "${BOOTSTRAP_SEED}" \
    --output "${indices_json}"

  # Read indices
  mapfile -t CASE_INDICES < <(python -c "import json; print('\n'.join(map(str, json.load(open('${indices_json}'))['selected_indices'])))")
  mapfile -t DATASET_ENV < <(dataset_env_vars "${dataset}")

  log_info "${model}/${dataset}: Running bootstrap on ${#CASE_INDICES[@]} cases"

  local failed=0
  for idx in "${!CASE_INDICES[@]}"; do
    local case_idx="${CASE_INDICES[$idx]}"
    local current=$((idx + 1))
    log_info "${model}/${dataset}: Bootstrap case ${current}/${#CASE_INDICES[@]} (index=${case_idx})"

    if ! env "${DATASET_ENV[@]}" \
         DERMAGENT_POLICY_ROOT="${policy_root}" \
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
  local output_model
  output_model="$(canonical_model_name "${model}")"

  local output_dir="${OUTPUT_ROOT}/${output_model}/${dataset}/compare"
  local policy_root="${OUTPUT_ROOT}/${output_model}/${dataset}/policy"
  local split_state_root="${OUTPUT_ROOT}/${output_model}/${dataset}/split_states"
  local split_json="${OUTPUT_ROOT}/${output_model}/${dataset}/${dataset}_split.json"

  mkdir -p "${output_dir}"
  mapfile -t DATASET_ENV < <(dataset_env_vars "${dataset}")

  # Promote state from train to val/test
  python "${SCRIPT_DIR}/manage_dataset_experiment_assets.py" promote-state \
    --split-state-root "${split_state_root}" \
    --source-split train \
    --target-splits val,test \
    >> "${LOG_DIR}/${model}_${dataset}_promote.log" 2>&1 || true

  log_info "${model}/${dataset}: Running compare on ${COMPARE_COUNT} test cases"

  if ! env "${DATASET_ENV[@]}" \
       DERMAGENT_POLICY_ROOT="${policy_root}" \
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
         --policy-label "${output_model} ${dataset} ${BOOTSTRAP_COUNT}case policy" \
         >> "${LOG_DIR}/${model}_${dataset}_compare.log" 2>&1; then
    log_error "${model}/${dataset}: Compare failed"
    return 1
  fi

  log_info "${model}/${dataset}: Compare completed successfully"
  return 0
}

run_model_all_datasets() {
  local model="$1"
  local output_model
  output_model="$(canonical_model_name "${model}")"
  local model_datasets_raw
  model_datasets_raw="$(datasets_for_model "${model}")"
  local model_datasets=()
  read -r -a model_datasets <<< "${model_datasets_raw}"

  log_info "========== Starting ${model} lane (output=${output_model}, datasets=${model_datasets[*]}) =========="

  # Start model server
  if ! start_model_server "${model}"; then
    log_error "${model}: Failed to start server, skipping all datasets"
    for dataset in "${model_datasets[@]}"; do
      update_summary "${output_model}" "${dataset}" "SKIPPED" "SKIPPED" "SKIPPED" "" "" "0" ""
    done
    return 1
  fi

  # Run assigned datasets sequentially for this model lane.
  for dataset in "${model_datasets[@]}"; do
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

    update_summary "${output_model}" "${dataset}" "${overall_status}" "${bootstrap_status}" "${compare_status}" \
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
  log_info "Use existing servers: ${USE_EXISTING_SERVERS}"
  log_info "Max parallel models: ${MAX_PARALLEL_MODELS}"
  log_info "Output root: ${OUTPUT_ROOT}"

  init_summary_tsv

  # Run models in parallel.
  # We'll use background jobs and wait
  local pids=()
  local model_idx=0

  for model in "${MODELS[@]}"; do
    run_model_all_datasets "${model}" &
    pids+=($!)

    model_idx=$((model_idx + 1))

    if (( model_idx % MAX_PARALLEL_MODELS == 0 )); then
      log_info "Waiting for current batch of ${MAX_PARALLEL_MODELS} models to complete..."
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

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
