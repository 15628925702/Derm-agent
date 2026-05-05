#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/hulumed_final.env
cd "$DERMAGENT_REPO_ROOT"

export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$HULUMED_FINAL_GPU_MEMORY_UTILIZATION}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$HULUMED_FINAL_MAX_MODEL_LEN}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-$HULUMED_FINAL_MAX_NUM_SEQS}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$HULUMED_SERVER_MODEL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$HULUMED_SERVER_API_KEY}"

exec bash /root/DermAgent/scripts/start_hulumed_server.sh "$DERMAGENT_REPO_ROOT"
