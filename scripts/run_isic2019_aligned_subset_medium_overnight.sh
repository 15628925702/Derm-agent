#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/DermAgent"
DATA_ROOT="${PROJECT_ROOT}/data/isic2019"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/external_eval/isic2019_aligned_subset"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${OUTPUT_BASE}/overnight_medium_${TIMESTAMP}"
MANIFEST_DIR="${RUN_ROOT}/manifests"
COMPARE_DIR="${RUN_ROOT}/comparison_medium"
LOG_PATH="${RUN_ROOT}/run.log"
LATEST_RUN_PATH="${OUTPUT_BASE}/LATEST_OVERNIGHT_RUN.txt"
LATEST_REPORT_PATH="${OUTPUT_BASE}/LATEST_OVERNIGHT_REPORT.txt"
LATEST_SUMMARY_PATH="${OUTPUT_BASE}/LATEST_OVERNIGHT_SUMMARY.txt"

mkdir -p "${MANIFEST_DIR}" "${COMPARE_DIR}"
exec >"${LOG_PATH}" 2>&1

echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[run_root] ${RUN_ROOT}"
echo "[data_root] ${DATA_ROOT}"

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

python - <<'PY'
import json
from dataio.isic2019_aligned_loader import discover_isic2019_aligned_assets

assets = discover_isic2019_aligned_assets("/root/DermAgent/data/isic2019")
print("[dataset_check]")
print(json.dumps(assets.to_dict(), ensure_ascii=False, indent=2))
PY

echo "[manifest] building medium manifest"
python scripts/build_isic2019_aligned_subset_manifest.py \
  --mode medium \
  --per-class 20 \
  --data-root "${DATA_ROOT}" \
  --output-dir "${MANIFEST_DIR}"

MANIFEST_PATH="$(find "${MANIFEST_DIR}" -maxdepth 1 -type f -name 'isic2019_aligned_subset_manifest_medium_*.json' | sort | tail -n 1)"
if [[ -z "${MANIFEST_PATH}" ]]; then
  echo "[error] medium manifest not found under ${MANIFEST_DIR}"
  exit 1
fi

echo "[manifest_path] ${MANIFEST_PATH}"
echo "[compare] starting frozen compare"
python scripts/compare_external_isic2019_aligned_subset.py \
  --manifest "${MANIFEST_PATH}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${COMPARE_DIR}" \
  --label-constraint \
  --client-timeout 1800 \
  --client-max-retries 8

REPORT_PATH="${COMPARE_DIR}/compare_external_isic2019_aligned_subset.json"
SUMMARY_PATH="${COMPARE_DIR}/summary.md"
if [[ ! -f "${REPORT_PATH}" ]]; then
  echo "[error] compare report not found at ${REPORT_PATH}"
  exit 1
fi

printf '%s\n' "${RUN_ROOT}" > "${LATEST_RUN_PATH}"
printf '%s\n' "${REPORT_PATH}" > "${LATEST_REPORT_PATH}"
printf '%s\n' "${SUMMARY_PATH}" > "${LATEST_SUMMARY_PATH}"

echo "[final_report] ${REPORT_PATH}"
echo "[final_summary] ${SUMMARY_PATH}"
echo "[done] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
