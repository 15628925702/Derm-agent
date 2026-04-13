#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/medgemma_final.env
cd "$DERMAGENT_REPO_ROOT"

export CLIENT_BASE_URL="${CLIENT_BASE_URL:-$MEDGEMMA_SERVER_BASE_URL}"
export CLIENT_API_KEY="${CLIENT_API_KEY:-$MEDGEMMA_SERVER_API_KEY}"
export CLIENT_MODEL="${CLIENT_MODEL:-$MEDGEMMA_SERVER_MODEL}"

exec bash /root/DermAgent/scripts/start_medgemma_server.sh
