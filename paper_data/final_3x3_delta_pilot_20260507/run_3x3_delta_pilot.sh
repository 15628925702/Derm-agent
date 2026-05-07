#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="/data/gh/DermAgent"
PY="/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python"
ENV_BIN="/home/zhongnan/miniconda3/envs/dermagent-6x6/bin"
FINAL_ROOT="${PROJECT_ROOT}/paper_data/final_3x3_delta_pilot_20260507"
LOG_DIR="${FINAL_ROOT}/run_logs"
SERVICE_LOG_DIR="${FINAL_ROOT}/service_logs"
COMPARE_ROOT="${FINAL_ROOT}/compare_reports"
CASE_EXPORT_ROOT="${PROJECT_ROOT}/paper_data/case_level_exports"
mkdir -p "${LOG_DIR}" "${SERVICE_LOG_DIR}" "${COMPARE_ROOT}" "${FINAL_ROOT}/state"

export PATH="${ENV_BIN}:${PATH}"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="EMPTY"
export OPENAI_TIMEOUT=120
export DERMAGENT_REPO_ROOT="${PROJECT_ROOT}"

MASTER_LOG="${LOG_DIR}/runner_$(date -u +%Y%m%dT%H%M%SZ).log"
SUMMARY_TSV="${FINAL_ROOT}/run_summary.tsv"
touch "${MASTER_LOG}"
echo -e "job_id\tmodel\tdataset\tgpu\tport\tstatus\tstarted_at\tfinished_at\tduration_sec\tlog_file" > "${SUMMARY_TSV}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${MASTER_LOG}"
}

trap 'rc=$?; log "runner exiting status=${rc} pid=$$"; exit "${rc}"' EXIT

append_summary() {
  local job_id="$1" model="$2" dataset="$3" gpu="$4" port="$5" status="$6" started="$7" finished="$8" duration="$9" log_file="${10}"
  echo -e "${job_id}\t${model}\t${dataset}\t${gpu}\t${port}\t${status}\t${started}\t${finished}\t${duration}\t${log_file}" >> "${SUMMARY_TSV}"
}

dataset_env_prefix() {
  local dataset="$1"
  if [[ "${dataset}" == "scin" ]]; then
    echo "DERMAGENT_SCIN_LABEL_SPACE_ID=scin_grouped"
  else
    echo ""
  fi
}

data_root_for_dataset() {
  case "$1" in
    scin) echo "${PROJECT_ROOT}/data/scin" ;;
    pad20) echo "${PROJECT_ROOT}/data/pad_ufes_20" ;;
    isic2019) echo "${PROJECT_ROOT}/data/isic2019" ;;
    *) echo "[error] unknown dataset $1" >&2; return 1 ;;
  esac
}

bootstrap_data_root_for_dataset() {
  case "$1" in
    scin) echo "${PROJECT_ROOT}/data/scin/official_mirror" ;;
    pad20) echo "${PROJECT_ROOT}/data/pad_ufes_20" ;;
    isic2019) echo "${PROJECT_ROOT}/data/isic2019" ;;
    *) echo "[error] unknown dataset $1" >&2; return 1 ;;
  esac
}

split_json_for_dataset() {
  echo "${FINAL_ROOT}/splits/$1_final_3x3_delta_pilot_split.json"
}

bootstrap_count_for_dataset() {
  case "$1" in
    isic2019) echo 120 ;;
    scin|pad20) echo 100 ;;
    *) return 1 ;;
  esac
}

eval_count_for_dataset() {
  echo 300
}

model_name_for_key() {
  case "$1" in
    llama) echo "Llama-3.2-11B-Vision-Instruct" ;;
    medgemma) echo "medgemma-4b-it" ;;
    skinvl) echo "SkinVL-MM" ;;
    *) echo "[error] unknown model $1" >&2; return 1 ;;
  esac
}

start_script_for_model() {
  case "$1" in
    llama) echo "${PROJECT_ROOT}/scripts/start_llama_server.sh" ;;
    medgemma) echo "${PROJECT_ROOT}/scripts/start_medgemma_server.sh" ;;
    skinvl) echo "${PROJECT_ROOT}/scripts/start_skinvl_server.sh" ;;
    *) return 1 ;;
  esac
}

bootstrap_script_for_dataset() {
  case "$1" in
    scin) echo "${PROJECT_ROOT}/scripts/bootstrap_scin_train_cases.sh" ;;
    pad20) echo "${PROJECT_ROOT}/scripts/bootstrap_pad20_train_cases.sh" ;;
    isic2019) echo "${PROJECT_ROOT}/scripts/bootstrap_isic2019_train_cases.sh" ;;
    *) return 1 ;;
  esac
}

stop_service() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      sleep 5
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill -9 "${pid}" >/dev/null 2>&1 || true
      fi
    fi
  fi
}

run_job() {
  local job_id="$1" model="$2" dataset="$3" gpu="$4" port="$5"
  local started finished start_sec end_sec duration status
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_sec="$(date +%s)"
  status="OK"

  local model_name data_root bootstrap_data_root split_json bootstrap_count eval_count
  model_name="$(model_name_for_key "${model}")"
  data_root="$(data_root_for_dataset "${dataset}")"
  bootstrap_data_root="$(bootstrap_data_root_for_dataset "${dataset}")"
  split_json="$(split_json_for_dataset "${dataset}")"
  bootstrap_count="$(bootstrap_count_for_dataset "${dataset}")"
  eval_count="$(eval_count_for_dataset "${dataset}")"

  local state_root policy_root split_state_root output_dir bootstrap_dir compare_dir run_name job_log pid_file service_log
  state_root="${FINAL_ROOT}/state/${model}/${dataset}"
  policy_root="${state_root}/policy"
  split_state_root="${state_root}/split_states"
  bootstrap_dir="${state_root}/bootstrap"
  compare_dir="${COMPARE_ROOT}/${model}/${dataset}"
  run_name="final_3x3_delta_pilot_20260507__${model}__${dataset}__eval${eval_count}"
  output_dir="${CASE_EXPORT_ROOT}/${run_name}"
  job_log="${LOG_DIR}/${job_id}_${model}_${dataset}.log"
  service_log="${SERVICE_LOG_DIR}/${job_id}_${model}_${dataset}_gpu${gpu}_port${port}.log"
  pid_file="${FINAL_ROOT}/state/${job_id}_${model}_${dataset}_port${port}.pid"

  mkdir -p "${policy_root}" "${split_state_root}" "${bootstrap_dir}" "${compare_dir}" "${output_dir}"

  {
    echo "[job] id=${job_id} model=${model} dataset=${dataset} gpu=${gpu} port=${port}"
    echo "[job] started=${started}"
    echo "[job] model_name=${model_name}"
    echo "[job] data_root=${data_root}"
    echo "[job] split_json=${split_json}"
    echo "[job] bootstrap_count=${bootstrap_count} eval_count=${eval_count}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    PORT="${port}" \
    LOG_FILE="${service_log}" \
    PID_FILE="${pid_file}" \
    SERVED_MODEL_NAME="${model_name}" \
    OPENAI_API_KEY="EMPTY" \
    FORCE_RESTART=1 \
    WAIT_SECONDS=900 \
    bash "$(start_script_for_model "${model}")" "${PROJECT_ROOT}"

    echo "[job] service ready"

    local env_prefix
    env_prefix="$(dataset_env_prefix "${dataset}")"

    if [[ -n "${env_prefix}" ]]; then
      env ${env_prefix} \
        DATA_ROOT="${bootstrap_data_root}" \
        POLICY_ROOT="${policy_root}" \
        SPLIT_STATE_ROOT="${split_state_root}" \
        OUTPUT_DIR="${bootstrap_dir}" \
        RUN_MODE="${run_name}_memory_build" \
        DATA_SPLIT="train" \
        SPLIT_JSON="${split_json}" \
        START_INDEX=0 \
        COUNT="${bootstrap_count}" \
        CLIENT_BASE_URL="http://127.0.0.1:${port}/v1" \
        CLIENT_API_KEY="EMPTY" \
        CLIENT_MODEL="${model_name}" \
        CLIENT_TIMEOUT=120 \
        CLIENT_MAX_RETRIES=1 \
        STOP_ON_ERROR=0 \
        bash "$(bootstrap_script_for_dataset "${dataset}")"
    else
      DATA_ROOT="${bootstrap_data_root}" \
      POLICY_ROOT="${policy_root}" \
      SPLIT_STATE_ROOT="${split_state_root}" \
      OUTPUT_DIR="${bootstrap_dir}" \
      RUN_MODE="${run_name}_memory_build" \
      DATA_SPLIT="train" \
      SPLIT_JSON="${split_json}" \
      START_INDEX=0 \
      COUNT="${bootstrap_count}" \
      CLIENT_BASE_URL="http://127.0.0.1:${port}/v1" \
      CLIENT_API_KEY="EMPTY" \
      CLIENT_MODEL="${model_name}" \
      CLIENT_TIMEOUT=120 \
      CLIENT_MAX_RETRIES=1 \
      STOP_ON_ERROR=0 \
      bash "$(bootstrap_script_for_dataset "${dataset}")"
    fi

    echo "[job] memory build completed; promoting state to test"
    "${PY}" "${PROJECT_ROOT}/scripts/manage_dataset_experiment_assets.py" promote-state \
      --split-state-root "${split_state_root}" \
      --source-split train \
      --target-splits val,test

    echo "[job] frozen compare starting"
    if [[ -n "${env_prefix}" ]]; then
      env ${env_prefix} \
        DERMAGENT_POLICY_ROOT="${policy_root}" \
        DERMAGENT_SPLIT_STATE_ROOT="${split_state_root}" \
        OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" \
        OPENAI_API_KEY="EMPTY" \
        OPENAI_MODEL="${model_name}" \
        OPENAI_TIMEOUT=120 \
        "${PY}" "${PROJECT_ROOT}/scripts/compare_agent_vs_qwen.py" \
          --data-root "${data_root}" \
          --limit "${eval_count}" \
          --case-offset 0 \
          --data-split test \
          --split-json "${split_json}" \
          --output-dir "${compare_dir}" \
          --policy-label "${run_name} frozen eval" \
          --export-paper-case-data \
          --paper-case-data-dir "${output_dir}" \
          --client-timeout 120 \
          --client-max-retries 1
    else
      DERMAGENT_POLICY_ROOT="${policy_root}" \
      DERMAGENT_SPLIT_STATE_ROOT="${split_state_root}" \
      OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" \
      OPENAI_API_KEY="EMPTY" \
      OPENAI_MODEL="${model_name}" \
      OPENAI_TIMEOUT=120 \
      "${PY}" "${PROJECT_ROOT}/scripts/compare_agent_vs_qwen.py" \
        --data-root "${data_root}" \
        --limit "${eval_count}" \
        --case-offset 0 \
        --data-split test \
        --split-json "${split_json}" \
        --output-dir "${compare_dir}" \
        --policy-label "${run_name} frozen eval" \
        --export-paper-case-data \
        --paper-case-data-dir "${output_dir}" \
        --client-timeout 120 \
        --client-max-retries 1
    fi

    echo "[job] compare completed"
  } >> "${job_log}" 2>&1 || status="FAILED"

  stop_service "${pid_file}"

  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_sec="$(date +%s)"
  duration=$((end_sec - start_sec))
  append_summary "${job_id}" "${model}" "${dataset}" "${gpu}" "${port}" "${status}" "${started}" "${finished}" "${duration}" "${job_log}"
  log "job ${job_id} ${model}/${dataset} finished status=${status} duration=${duration}s log=${job_log}"
}

jobs=(
  "j01 medgemma pad20 0 8100"
  "j02 medgemma scin 1 8101"
  "j03 medgemma isic2019 2 8102"
  "j04 skinvl pad20 3 8103"
  "j05 skinvl scin 4 8104"
  "j06 skinvl isic2019 5 8105"
  "j07 llama pad20 6 8106"
  "j08 llama scin 7 8107"
)
second_wave_job="j09 llama isic2019 4 8108"

log "starting 3x3 delta pilot"
log "runner pid: $$"
log "final root: ${FINAL_ROOT}"
log "first wave: 8 jobs, second wave: llama/isic2019 after skinvl/scin frees gpu 4"

pids=()
skinvl_scin_pid=""
for job in "${jobs[@]}"; do
  read -r job_id model dataset gpu port <<< "${job}"
  log "launching ${job_id}: ${model}/${dataset} on gpu=${gpu} port=${port}"
  run_job "${job_id}" "${model}" "${dataset}" "${gpu}" "${port}" &
  pid="$!"
  pids+=("${pid}")
  if [[ "${job_id}" == "j05" ]]; then
    skinvl_scin_pid="${pid}"
  fi
done

if [[ -n "${skinvl_scin_pid}" ]]; then
  log "waiting for j05 skinvl/scin to free gpu 4 before starting second-wave job"
  wait "${skinvl_scin_pid}" || true
fi

read -r job_id model dataset gpu port <<< "${second_wave_job}"
log "launching ${job_id}: ${model}/${dataset} on gpu=${gpu} port=${port}"
run_job "${job_id}" "${model}" "${dataset}" "${gpu}" "${port}" &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "${pid}" || true
done

log "all 3x3 delta pilot jobs finished"
log "summary: ${SUMMARY_TSV}"
