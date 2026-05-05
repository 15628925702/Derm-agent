#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/model_servers.env
source /root/DermAgent/final-script/configs/common.env

TARGET="${1:?usage: switch_model_final.sh <qwen|medgemma|skinvl|llama|hulumed|dermatollama|stop>}"

exec bash /root/DermAgent/scripts/switch_model_server.sh "$TARGET" "$DERMAGENT_REPO_ROOT"
