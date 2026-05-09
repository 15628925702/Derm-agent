#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="/data/gh/DermAgent"
PY="/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python"
ENV_BIN="/home/zhongnan/miniconda3/envs/dermagent-6x6/bin"
FINAL_ROOT="${PROJECT_ROOT}/paper_data/final_6x5_continuous_20260507"
QUEUE_TSV="${FINAL_ROOT}/remaining_queue.tsv"
LOG_DIR="${FINAL_ROOT}/run_logs"
SERVICE_LOG_DIR="${FINAL_ROOT}/service_logs"
COMPARE_ROOT="${FINAL_ROOT}/compare_reports"
CASE_EXPORT_ROOT="${PROJECT_ROOT}/paper_data/case_level_exports"
SUMMARY_TSV="${FINAL_ROOT}/run_summary.tsv"
QUEUE_STATE="${FINAL_ROOT}/queue_state.tsv"
MASTER_LOG="${LOG_DIR}/continuous_runner_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "${LOG_DIR}" "${SERVICE_LOG_DIR}" "${COMPARE_ROOT}" "${FINAL_ROOT}/state" "${FINAL_ROOT}/splits" "${FINAL_ROOT}/manifests"

export PATH="${ENV_BIN}:${PATH}"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="EMPTY"
export OPENAI_TIMEOUT=120
export DERMAGENT_REPO_ROOT="${PROJECT_ROOT}"

touch "${MASTER_LOG}"
echo -e "job_id\tmodel\tdataset\tgpu\tport\tstatus\tstarted_at\tfinished_at\tduration_sec\tlog_file" > "${SUMMARY_TSV}"
echo -e "event_ts\tjob_id\tmodel\tdataset\tgpu\tport\tevent" > "${QUEUE_STATE}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${MASTER_LOG}"
}

record_event() {
  local job_id="$1" model="$2" dataset="$3" gpu="$4" port="$5" event="$6"
  echo -e "$(date -u +%Y-%m-%dT%H:%M:%SZ)\t${job_id}\t${model}\t${dataset}\t${gpu}\t${port}\t${event}" >> "${QUEUE_STATE}"
}

append_summary() {
  local job_id="$1" model="$2" dataset="$3" gpu="$4" port="$5" status="$6" started="$7" finished="$8" duration="$9" log_file="${10}"
  echo -e "${job_id}\t${model}\t${dataset}\t${gpu}\t${port}\t${status}\t${started}\t${finished}\t${duration}\t${log_file}" >> "${SUMMARY_TSV}"
}

gpu_is_free() {
  local gpu="$1"
  local used apps
  used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}' | tail -1)"
  apps="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' || true)"
  [[ -n "${used}" ]] && [[ "${used}" -lt 1024 ]] && [[ -z "${apps}" ]]
}

port_is_free() {
  local port="$1"
  ! ss -ltnp | grep -qE ":${port}\b"
}

port_for_gpu() {
  local gpu="$1"
  echo $((8120 + gpu))
}

model_name_for_key() {
  case "$1" in
    qwen) echo "Qwen2.5-VL-7B-Instruct" ;;
    dermatollama) echo "DermatoLlama-full" ;;
    medgemma) echo "medgemma-4b-it" ;;
    hulumed) echo "Hulu-Med-7B" ;;
    llama) echo "Llama-3.2-11B-Vision-Instruct" ;;
    skinvl) echo "SkinVL-MM" ;;
    *) echo "[error] unknown model $1" >&2; return 1 ;;
  esac
}

start_script_for_model() {
  case "$1" in
    qwen) echo "${PROJECT_ROOT}/scripts/start_qwen_server.sh" ;;
    dermatollama) echo "${PROJECT_ROOT}/scripts/start_dermatollama_server.sh" ;;
    medgemma) echo "${PROJECT_ROOT}/scripts/start_medgemma_server.sh" ;;
    hulumed) echo "${PROJECT_ROOT}/scripts/start_hulumed_server.sh" ;;
    llama) echo "${PROJECT_ROOT}/scripts/start_llama_server.sh" ;;
    skinvl) echo "${PROJECT_ROOT}/scripts/start_skinvl_server.sh" ;;
    *) return 1 ;;
  esac
}

bootstrap_script_for_dataset() {
  case "$1" in
    ham10000) echo "${PROJECT_ROOT}/scripts/bootstrap_ham10000_train_cases.sh" ;;
    isic2019) echo "${PROJECT_ROOT}/scripts/bootstrap_isic2019_train_cases.sh" ;;
    pad20) echo "${PROJECT_ROOT}/scripts/bootstrap_pad20_train_cases.sh" ;;
    scin) echo "${PROJECT_ROOT}/scripts/bootstrap_scin_train_cases.sh" ;;
    sd198) echo "${PROJECT_ROOT}/scripts/bootstrap_sd198_train_cases.sh" ;;
    *) return 1 ;;
  esac
}

data_root_for_dataset() {
  case "$1" in
    ham10000) echo "${PROJECT_ROOT}/data/ham10000" ;;
    isic2019) echo "${PROJECT_ROOT}/data/isic2019" ;;
    pad20) echo "${PROJECT_ROOT}/data/pad_ufes_20" ;;
    scin) echo "${PROJECT_ROOT}/data/scin" ;;
    sd198) echo "${PROJECT_ROOT}/data/sd198" ;;
    *) return 1 ;;
  esac
}

bootstrap_data_root_for_dataset() {
  case "$1" in
    scin) echo "${PROJECT_ROOT}/data/scin/official_mirror" ;;
    sd198) echo "${PROJECT_ROOT}/data/sd198/sd-198" ;;
    *) data_root_for_dataset "$1" ;;
  esac
}

split_json_for_dataset() {
  echo "${FINAL_ROOT}/splits/$1_final_6x5_continuous_split.json"
}

split_id_for_dataset() {
  case "$1" in
    ham10000) echo "ham10000_balanced_v1" ;;
    isic2019) echo "isic2019_contiguous_v1" ;;
    pad20) echo "pad_ufes_20_contiguous_v1" ;;
    scin) echo "scin_contiguous_v1" ;;
    sd198) echo "sd198_balanced_v1" ;;
    *) return 1 ;;
  esac
}

bootstrap_count_for_dataset() {
  case "$1" in
    isic2019) echo 120 ;;
    *) echo 100 ;;
  esac
}

eval_count_for_dataset() {
  echo 300
}

dataset_env_args() {
  case "$1" in
    scin) echo "DERMAGENT_SCIN_LABEL_SPACE_ID=scin_grouped" ;;
    sd198) echo "DERMAGENT_SD198_LABEL_SPACE_ID=sd198_grouped" ;;
    *) echo "" ;;
  esac
}

init_isolated_state() {
  local policy_root="$1" split_state_root="$2"
  POLICY_ROOT="${policy_root}" \
  SPLIT_STATE_ROOT="${split_state_root}" \
  "${PY}" - <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

project_root = Path("/data/gh/DermAgent")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agent.experiment_state import ensure_split_state_paths
from agent.policy_config import load_policy

policy_root = Path(os.environ["POLICY_ROOT"])
split_state_root = Path(os.environ["SPLIT_STATE_ROOT"])
base_policy = project_root / "state" / "policy" / "current_stable_policy.json"
target_policy = policy_root / "current_stable_policy.json"
policy_root.mkdir(parents=True, exist_ok=True)
split_state_root.mkdir(parents=True, exist_ok=True)
if not target_policy.exists():
    shutil.copy2(base_policy, target_policy)
payload = load_policy(target_policy).to_dict()
payload["source_path"] = str(target_policy)
target_policy.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
(policy_root / "manifest.json").write_text(
    json.dumps(
        {
            "stable_policy_id": payload.get("policy_id"),
            "stable_policy_path": str(target_policy),
            "policies": [
                {
                    "policy_id": payload.get("policy_id"),
                    "version": payload.get("version"),
                    "status": payload.get("status", "stable"),
                    "source_path": str(target_policy),
                }
            ],
            "evaluations": [],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
for split in ("train", "val", "test"):
    ensure_split_state_paths(data_split=split, split_state_root=split_state_root, policy_path=target_policy)
print(json.dumps({"policy_path": str(target_policy), "split_state_root": str(split_state_root)}, ensure_ascii=False))
PY
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

  local model_name data_root bootstrap_data_root split_json split_id bootstrap_count eval_count
  model_name="$(model_name_for_key "${model}")"
  data_root="$(data_root_for_dataset "${dataset}")"
  bootstrap_data_root="$(bootstrap_data_root_for_dataset "${dataset}")"
  split_json="$(split_json_for_dataset "${dataset}")"
  split_id="$(split_id_for_dataset "${dataset}")"
  bootstrap_count="$(bootstrap_count_for_dataset "${dataset}")"
  eval_count="$(eval_count_for_dataset "${dataset}")"

  local state_root policy_root split_state_root output_dir bootstrap_dir compare_dir run_name job_log pid_file service_log env_prefix
  state_root="${FINAL_ROOT}/state/${model}/${dataset}"
  policy_root="${state_root}/policy"
  split_state_root="${state_root}/split_states"
  bootstrap_dir="${state_root}/bootstrap"
  compare_dir="${COMPARE_ROOT}/${model}/${dataset}"
  run_name="final_6x5_continuous_20260507__${model}__${dataset}__eval${eval_count}"
  output_dir="${CASE_EXPORT_ROOT}/${run_name}"
  job_log="${LOG_DIR}/${job_id}_${model}_${dataset}.log"
  service_log="${SERVICE_LOG_DIR}/${job_id}_${model}_${dataset}_gpu${gpu}_port${port}.log"
  pid_file="${FINAL_ROOT}/state/${job_id}_${model}_${dataset}_port${port}.pid"
  env_prefix="$(dataset_env_args "${dataset}")"

  mkdir -p "${policy_root}" "${split_state_root}" "${bootstrap_dir}" "${compare_dir}" "${output_dir}"

  (
    set -euo pipefail
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
    echo "[job] initializing isolated policy and split state"
    init_isolated_state "${policy_root}" "${split_state_root}"

    if [[ -n "${env_prefix}" ]]; then
      env ${env_prefix} \
        DATA_ROOT="${bootstrap_data_root}" \
        POLICY_ROOT="${policy_root}" \
        SPLIT_STATE_ROOT="${split_state_root}" \
        OUTPUT_DIR="${bootstrap_dir}" \
        RUN_MODE="${run_name}_memory_build" \
        DATA_SPLIT="train" \
        SPLIT_JSON="${split_json}" \
        SPLIT_ID="${split_id}" \
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
      SPLIT_ID="${split_id}" \
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
  ) >> "${job_log}" 2>&1 || status="FAILED"

  stop_service "${pid_file}"
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_sec="$(date +%s)"
  duration=$((end_sec - start_sec))
  append_summary "${job_id}" "${model}" "${dataset}" "${gpu}" "${port}" "${status}" "${started}" "${finished}" "${duration}" "${job_log}"
  record_event "${job_id}" "${model}" "${dataset}" "${gpu}" "${port}" "finished:${status}"
  log "job ${job_id} ${model}/${dataset} finished status=${status} duration=${duration}s log=${job_log}"
}

read_queue() {
  tail -n +2 "${QUEUE_TSV}" | awk -F '\t' 'NF >= 3 {print $1" "$2" "$3}'
}

log "starting continuous remaining 6x5 queue"
log "runner pid: $$"
log "queue: ${QUEUE_TSV}"
log "ports: gpu index + 8120"

mapfile -t queue < <(read_queue)
next_index=0
declare -A active_pids
declare -A active_desc

while [[ "${next_index}" -lt "${#queue[@]}" || "${#active_pids[@]}" -gt 0 ]]; do
  for gpu in "${!active_pids[@]}"; do
    pid="${active_pids[$gpu]}"
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      wait "${pid}" >/dev/null 2>&1 || true
      log "slot gpu=${gpu} released from ${active_desc[$gpu]}"
      unset 'active_pids[$gpu]'
      unset 'active_desc[$gpu]'
    fi
  done

  for gpu in 0 1 2 3 4 5 6 7; do
    if [[ "${next_index}" -ge "${#queue[@]}" ]]; then
      break
    fi
    if [[ -n "${active_pids[$gpu]:-}" ]]; then
      continue
    fi
    port="$(port_for_gpu "${gpu}")"
    if gpu_is_free "${gpu}" && port_is_free "${port}"; then
      read -r job_id model dataset <<< "${queue[$next_index]}"
      log "launching queued ${job_id}: ${model}/${dataset} on gpu=${gpu} port=${port}"
      record_event "${job_id}" "${model}" "${dataset}" "${gpu}" "${port}" "launched"
      run_job "${job_id}" "${model}" "${dataset}" "${gpu}" "${port}" &
      active_pids[$gpu]="$!"
      active_desc[$gpu]="${job_id}:${model}/${dataset}"
      next_index=$((next_index + 1))
    fi
  done

  sleep 60
done

log "continuous remaining queue finished"
log "summary: ${SUMMARY_TSV}"
