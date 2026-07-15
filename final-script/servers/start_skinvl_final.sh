#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/skinvl_final.env"
cd "$DERMAGENT_REPO_ROOT"

bash "${DERMAGENT_REPO_ROOT}/scripts/start_skinvl_server.sh" "$DERMAGENT_REPO_ROOT"
