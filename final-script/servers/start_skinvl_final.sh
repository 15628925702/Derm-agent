#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/skinvl_final.env
cd "$DERMAGENT_REPO_ROOT"

bash /root/DermAgent/scripts/start_skinvl_server.sh "$DERMAGENT_REPO_ROOT"
