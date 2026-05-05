#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-stable_overnight_eval_bundle_${STAMP}}"
RUN_ROOT="${PROJECT_ROOT}/outputs/overnight_eval_bundle/${RUN_ID}"
ISIC_ROOT="${PROJECT_ROOT}/data/isic2019"
ISIC_COUNTS="${ISIC_COUNTS:-MEL=34,BCC=33,NEV=33}"
PADUFES_SPLIT_DATA_ROOT="${PADUFES_SPLIT_DATA_ROOT:-${PROJECT_ROOT}/data}"
PADUFES_EVAL_DATA_ROOT="${PADUFES_EVAL_DATA_ROOT:-${PROJECT_ROOT}/data/pad_ufes_20}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-1800}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-8}"
RUN_PADUFES_VAL="${RUN_PADUFES_VAL:-0}"
LATEST_DIR="${PROJECT_ROOT}/outputs/overnight_eval_bundle"
LATEST_RUN_PATH="${LATEST_DIR}/LATEST_RUN.txt"
LATEST_SUMMARY_PATH="${LATEST_DIR}/LATEST_SUMMARY.json"
LATEST_ISIC_REPORT_PATH="${LATEST_DIR}/LATEST_ISIC100_REPORT.txt"
LATEST_PADUFES_TEST_PATH="${LATEST_DIR}/LATEST_PADUFES_TEST_RESULT.txt"
LATEST_PADUFES_VAL_PATH="${LATEST_DIR}/LATEST_PADUFES_VAL_RESULT.txt"

mkdir -p "${RUN_ROOT}"
mkdir -p "${LATEST_DIR}"

export PYTHONUNBUFFERED=1

FAIL_COUNT=0
declare -a FAIL_TASKS=()
ISIC_REPORT_PATH=""
PADUFES_TEST_RESULT_PATH=""
PADUFES_VAL_RESULT_PATH=""

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

record_failure() {
  local task_name="$1"
  local rc="$2"
  FAIL_COUNT="$((FAIL_COUNT + 1))"
  FAIL_TASKS+=("${task_name}:${rc}")
}

run_checked() {
  local task_name="$1"
  shift
  log "start ${task_name}"
  if "$@"; then
    log "done ${task_name}"
    return 0
  fi
  local rc=$?
  log "fail ${task_name} rc=${rc}"
  record_failure "${task_name}" "${rc}"
  return "${rc}"
}

resolve_stable_split_json() {
  python - <<'PY'
import json
import re
from pathlib import Path

policy_path = Path("/root/DermAgent/state/policy/current_stable_policy.json")
payload = json.loads(policy_path.read_text(encoding="utf-8"))
controller_ckpt = str(payload.get("planner_policy", {}).get("controller_checkpoint_path", "")).strip()
candidates = []
if controller_ckpt:
    match = re.search(r"(.*/outputs/smoke_cycles/[^/]+)/train_runs/[^/]+/", controller_ckpt)
    if match:
        candidates.append(Path(match.group(1)) / "fixed_split.json")
candidates.extend(
    [
        Path("/root/DermAgent/outputs/smoke_cycles/reuse_medium_signal_10h_20260328T173214Z/fixed_split.json"),
        Path("/root/DermAgent/outputs/smoke_cycles/medium_signal_no_ablation_10h_v1/fixed_split.json"),
        Path("/root/DermAgent/outputs/smoke_cycles/medium_signal_no_ablation_v1/fixed_split.json"),
    ]
)
seen = set()
for path in candidates:
    key = str(path)
    if key in seen:
        continue
    seen.add(key)
    if path.exists():
        print(path)
        break
else:
    raise SystemExit("No fixed split JSON found for stable frozen PAD-UFES evaluation.")
PY
}

count_split_cases() {
  local split_json="$1"
  local split_name="$2"
  python - "$split_json" "$split_name" <<'PY'
import json
import sys
from pathlib import Path

split_json = Path(sys.argv[1])
split_name = sys.argv[2]
payload = json.loads(split_json.read_text(encoding="utf-8"))
value = payload.get(split_name)
if isinstance(value, list):
    print(len(value))
    raise SystemExit(0)
if isinstance(value, dict):
    for key in ("case_ids", "selected_case_ids"):
        nested = value.get(key)
        if isinstance(nested, list):
            print(len(nested))
            raise SystemExit(0)
raise SystemExit(f"Unable to resolve split size for `{split_name}` from {split_json}")
PY
}

latest_eval_dir() {
  local root="$1"
  find "${root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

log "run_id=${RUN_ID}"
log "run_root=${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" > "${LATEST_RUN_PATH}"

SPLIT_JSON="${RUN_ROOT}/fixed_split.json"
python - <<PY
from pathlib import Path
from configs.dataset_splits import write_fixed_split_json

output_path = Path("${SPLIT_JSON}")
write_fixed_split_json(output_path, data_root=Path("${PADUFES_SPLIT_DATA_ROOT}"))
print(output_path)
PY
VAL_LIMIT="$(count_split_cases "${SPLIT_JSON}" "val")"
TEST_LIMIT="$(count_split_cases "${SPLIT_JSON}" "test")"

log "stable_split_json=${SPLIT_JSON}"
log "padufes_val_limit=${VAL_LIMIT}"
log "padufes_test_limit=${TEST_LIMIT}"

ISIC_MANIFEST_DIR="${RUN_ROOT}/isic2019_100case/manifests"
ISIC_COMPARE_DIR="${RUN_ROOT}/isic2019_100case/comparison"
PADUFES_TEST_DIR="${RUN_ROOT}/padufes_full_test_eval"
PADUFES_VAL_DIR="${RUN_ROOT}/padufes_full_val_eval"

mkdir -p "${ISIC_MANIFEST_DIR}" "${ISIC_COMPARE_DIR}" "${PADUFES_TEST_DIR}"

if run_checked "isic2019_manifest_100case" \
  python "${PROJECT_ROOT}/scripts/build_isic2019_aligned_subset_manifest_custom.py" \
    --data-root "${ISIC_ROOT}" \
    --class-counts "${ISIC_COUNTS}" \
    --tag "100case" \
    --output-dir "${ISIC_MANIFEST_DIR}"; then
  ISIC_MANIFEST_PATH="$(find "${ISIC_MANIFEST_DIR}" -maxdepth 1 -type f -name 'isic2019_aligned_subset_manifest_100case_*.json' | sort | tail -n 1)"
  if [[ -n "${ISIC_MANIFEST_PATH}" ]]; then
    if run_checked "isic2019_compare_100case" \
      python "${PROJECT_ROOT}/scripts/compare_external_isic2019_aligned_subset.py" \
        --manifest "${ISIC_MANIFEST_PATH}" \
        --data-root "${ISIC_ROOT}" \
        --output-dir "${ISIC_COMPARE_DIR}" \
        --label-constraint \
        --client-timeout "${CLIENT_TIMEOUT}" \
        --client-max-retries "${CLIENT_MAX_RETRIES}"; then
      ISIC_REPORT_PATH="${ISIC_COMPARE_DIR}/compare_external_isic2019_aligned_subset.json"
      if [[ -f "${ISIC_REPORT_PATH}" ]]; then
        printf '%s\n' "${ISIC_REPORT_PATH}" > "${LATEST_ISIC_REPORT_PATH}"
      fi
    fi
  else
    log "fail isic2019_manifest_path_missing"
    record_failure "isic2019_manifest_path_missing" 1
  fi
fi

if run_checked "padufes_full_test_frozen_eval" \
  python "${PROJECT_ROOT}/scripts/run_eval_brief.py" \
    --data-root "${PADUFES_EVAL_DATA_ROOT}" \
    --limit "${TEST_LIMIT}" \
    --case-offset 0 \
    --data-split test \
    --split-json "${SPLIT_JSON}" \
    --output-dir "${PADUFES_TEST_DIR}" \
    --client-timeout "${CLIENT_TIMEOUT}" \
    --client-max-retries "${CLIENT_MAX_RETRIES}"; then
  PADUFES_TEST_EVAL_DIR="$(latest_eval_dir "${PADUFES_TEST_DIR}")"
  if [[ -n "${PADUFES_TEST_EVAL_DIR}" ]]; then
    PADUFES_TEST_RESULT_PATH="${PADUFES_TEST_EVAL_DIR}/result_manifest.json"
    if [[ -f "${PADUFES_TEST_RESULT_PATH}" ]]; then
      printf '%s\n' "${PADUFES_TEST_RESULT_PATH}" > "${LATEST_PADUFES_TEST_PATH}"
    fi
  fi
fi

if [[ "${RUN_PADUFES_VAL}" == "1" ]]; then
  mkdir -p "${PADUFES_VAL_DIR}"
  if run_checked "padufes_full_val_frozen_eval" \
    python "${PROJECT_ROOT}/scripts/run_eval_brief.py" \
      --data-root "${PADUFES_EVAL_DATA_ROOT}" \
      --limit "${VAL_LIMIT}" \
      --case-offset 0 \
      --data-split val \
      --split-json "${SPLIT_JSON}" \
      --output-dir "${PADUFES_VAL_DIR}" \
      --client-timeout "${CLIENT_TIMEOUT}" \
      --client-max-retries "${CLIENT_MAX_RETRIES}"; then
    PADUFES_VAL_EVAL_DIR="$(latest_eval_dir "${PADUFES_VAL_DIR}")"
    if [[ -n "${PADUFES_VAL_EVAL_DIR}" ]]; then
      PADUFES_VAL_RESULT_PATH="${PADUFES_VAL_EVAL_DIR}/result_manifest.json"
      if [[ -f "${PADUFES_VAL_RESULT_PATH}" ]]; then
        printf '%s\n' "${PADUFES_VAL_RESULT_PATH}" > "${LATEST_PADUFES_VAL_PATH}"
      fi
    fi
  fi
fi

FAIL_TASKS_JSON="$(
  printf '%s\n' "${FAIL_TASKS[@]}" | python - <<'PY'
import json
import sys
items = [line.rstrip("\n") for line in sys.stdin if line.rstrip("\n")]
print(json.dumps(items, ensure_ascii=False))
PY
)"

python - <<PY
import json
from pathlib import Path

payload = {
    "run_id": "${RUN_ID}",
    "run_root": "${RUN_ROOT}",
    "stable_split_json": "${SPLIT_JSON}",
    "isic_counts": "${ISIC_COUNTS}",
    "padufes_test_limit": int("${TEST_LIMIT}"),
    "padufes_val_limit": int("${VAL_LIMIT}"),
    "run_padufes_val": ${RUN_PADUFES_VAL},
    "isic_report_path": "${ISIC_REPORT_PATH}",
    "padufes_test_result_path": "${PADUFES_TEST_RESULT_PATH}",
    "padufes_val_result_path": "${PADUFES_VAL_RESULT_PATH}",
    "fail_count": int("${FAIL_COUNT}"),
    "fail_tasks": json.loads('''${FAIL_TASKS_JSON}'''),
}
Path("${RUN_ROOT}/summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
Path("${LATEST_SUMMARY_PATH}").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  exit 1
fi
exit 0
