#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model_servers.env"
source "${SCRIPT_DIR}/../configs/common.env"

TARGET="${1:?usage: switch_model_final.sh <qwen|medgemma|skinvl|llama|hulumed|dermatollama|stop>}"

exec bash "${DERMAGENT_REPO_ROOT}/scripts/switch_model_server.sh" "$TARGET" "$DERMAGENT_REPO_ROOT"
