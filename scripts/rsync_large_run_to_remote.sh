#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-100.126.2.24}"
REMOTE_USER="${REMOTE_USER:-zhongnan}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/gh}"
LOCAL_ROOT="${LOCAL_ROOT:-/data/gh}"
COPY_CONDA_ENV="${COPY_CONDA_ENV:-0}"

RSYNC_OPTS=(
  -aH
  --info=progress2
  --partial
  --append-verify
)

PROJECT_EXCLUDES=(
  --exclude='.git/objects/pack/tmp_*'
  --exclude='.pytest_cache/'
  --exclude='__pycache__/'
  --exclude='*/__pycache__/'
  --exclude='.tmp/'
  --exclude='logs/*'
  --exclude='outputs/*'
  --exclude='final-score/final_runs/*'
  --exclude='paper_data/workflow_tuning_runs_*/'
  --exclude='paper_data/workflow_tuning_runs_*_merge_verify/'
  --exclude='paper_data/final_dataset_splits_*/smoke_runs/'
  --exclude='paper_data/final_*/compare_reports/'
  --exclude='paper_data/final_*/state/'
  --exclude='paper_data/final_*/run_logs/'
  --exclude='paper_data/final_*/service_logs/'
)

echo "[sync] creating remote root ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}' '${REMOTE_ROOT}/models'"

echo "[sync] DermAgent project, including data and final split files"
rsync "${RSYNC_OPTS[@]}" "${PROJECT_EXCLUDES[@]}" \
  "${LOCAL_ROOT}/DermAgent/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/DermAgent/"

echo "[sync] model weights"
rsync "${RSYNC_OPTS[@]}" \
  "${LOCAL_ROOT}/models/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/models/"

if [[ "${COPY_CONDA_ENV}" == "1" ]]; then
  echo "[sync] conda environment dermagent-6x6"
  ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p /home/${REMOTE_USER}/miniconda3/envs"
  rsync "${RSYNC_OPTS[@]}" \
    "/home/${REMOTE_USER}/miniconda3/envs/dermagent-6x6/" \
    "${REMOTE_USER}@${REMOTE_HOST}:/home/${REMOTE_USER}/miniconda3/envs/dermagent-6x6/"
fi

echo "[sync] done"
