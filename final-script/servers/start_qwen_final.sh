#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/qwen_final.env
cd "$DERMAGENT_REPO_ROOT"

export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$QWEN_FINAL_GPU_MEMORY_UTILIZATION}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$QWEN_FINAL_MAX_MODEL_LEN}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-$QWEN_FINAL_MAX_NUM_SEQS}"

exec bash /root/DermAgent/scripts/start_qwen_server.sh
