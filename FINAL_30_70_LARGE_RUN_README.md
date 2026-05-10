# Final 30/70 Large Run

This run uses the `final_20260509` 30/70 split:

- `train` / `final_experience_train`: bootstrap experience-bank build only.
- `test` / `final_compare_test`: held-out comparison only.
- Current scripts cover the five datasets that have final 30/70 split JSONs: `ham10000`, `isic2019`, `pad20`, `scin`, `sd198`.
- Xiangya is not included in `final_20260509`; do not call the result a true 6-dataset run unless a Xiangya final 30/70 split is built first.

The runner defaults to 6 models x 5 final datasets. Bootstrap does not enable the doctor evidence package. Test compare does enable it and exports per-case doctor-facing evidence files.

## Pre-Run Git State

The pre-large-test state is tagged and pushed:

```bash
git tag beforeLargeTest
git push origin merge-final-weak-workflow-tuning
git push origin beforeLargeTest
```

Current pre-run tag points to `a60aae274eb04e3caa3ec37cdc5a2726fb80a3c2`.

## What To Migrate To 100.126.2.24

Required:

- `/data/gh/DermAgent`
- `/data/gh/models`

Usually required if the target machine does not already have the environment:

- `/home/zhongnan/miniconda3/envs/dermagent-6x6`

The model start scripts auto-detect `/data/gh/models`, so keeping the same paths is the least painful route.

Use SSH or `rsync`. Prefer interactive password entry or `SSHPASS` from the shell; do not write passwords into tracked files.

```bash
cd /data/gh/DermAgent
bash scripts/rsync_large_run_to_remote.sh
```

If the target machine lacks the conda env:

```bash
cd /data/gh/DermAgent
COPY_CONDA_ENV=1 bash scripts/rsync_large_run_to_remote.sh
```

Optional non-interactive style, if `sshpass` is installed:

```bash
export SSHPASS='<password>'
sshpass -e bash scripts/rsync_large_run_to_remote.sh
```

## Target Machine Sanity Check

On `100.126.2.24`:

```bash
cd /data/gh/DermAgent
git status --short
git rev-parse HEAD
nvidia-smi
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python -m py_compile \
  scripts/run_final_30_70_6x5_dynamic.py \
  scripts/merge_experience_shards.py \
  scripts/merge_final_compare_reports.py \
  scripts/export_doctor_evidence_packages.py
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python -m pytest \
  tests/test_conservative_fusion.py tests/test_model_workflow_router.py -q
```

## Launch

Use the same `RUN_ID` on both machines. Machine `100.126.2.23` uses `--machine-id 0`; machine `100.126.2.24` uses `--machine-id 1`.

Bootstrap uses GPUs `0..7`. Test compare uses GPUs `0..6` and reserves GPU `7` for the Qwen physician-summary service. This is intentional: the doctor evidence package needs a separate Qwen service during test, and bootstrap must not enable it.

Machine `100.126.2.23`:

```bash
cd /data/gh/DermAgent
export RUN_ID=final_30_70_large_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "paper_data/final_30_70_large_runs/${RUN_ID}"
nohup setsid /home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python \
  scripts/run_final_30_70_6x5_dynamic.py \
  --run-id "${RUN_ID}" \
  --machine-id 0 \
  --machine-count 2 \
  > "paper_data/final_30_70_large_runs/${RUN_ID}/machine_0_launcher.log" 2>&1 &
echo $! > "paper_data/final_30_70_large_runs/${RUN_ID}/machine_0_launcher.pid"
echo "${RUN_ID}"
```

Machine `100.126.2.24`, using the same `RUN_ID`:

```bash
cd /data/gh/DermAgent
export RUN_ID=<same-run-id-from-machine-0>
mkdir -p "paper_data/final_30_70_large_runs/${RUN_ID}"
nohup setsid /home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python \
  scripts/run_final_30_70_6x5_dynamic.py \
  --run-id "${RUN_ID}" \
  --machine-id 1 \
  --machine-count 2 \
  > "paper_data/final_30_70_large_runs/${RUN_ID}/machine_1_launcher.log" 2>&1 &
echo $! > "paper_data/final_30_70_large_runs/${RUN_ID}/machine_1_launcher.pid"
```

Resume uses the same command and same `RUN_ID`; completed shards have `*.DONE.json` markers and are skipped.

## Outputs To Preserve

Per machine:

- `paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/run_manifest.json`
- `paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/reports/`
- `paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/paper_case_exports/`
- `paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/logs/`
- `doctor_evidence_final_30_70_<RUN_ID>/`

The merged per-combo compare report is:

```text
paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/reports/<model>/<dataset>/compare_agent_vs_qwen_final_compare_test_merged.json
```

The paper-ready per-case CSV/JSON/XLSX export is under:

```text
paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/paper_case_exports/<model>/<dataset>/merged/
```

The doctor-facing evidence package is under:

```text
doctor_evidence_final_30_70_<RUN_ID>/<model>/<dataset>/<case_id>.json
doctor_evidence_final_30_70_<RUN_ID>/<model>/<dataset>/<case_id>.md
```

## Monitoring

```bash
tail -f paper_data/final_30_70_large_runs/<RUN_ID>/machine_0/runner.log
find paper_data/final_30_70_large_runs/<RUN_ID>/machine_0 -name '*.DONE.json' | wc -l
nvidia-smi
```

On the second machine replace `machine_0` with `machine_1`.

## Collect Machine 1 Results Back To Machine 0

After both finish:

```bash
rsync -aH --info=progress2 \
  zhongnan@100.126.2.24:/data/gh/DermAgent/paper_data/final_30_70_large_runs/<RUN_ID>/machine_1/ \
  /data/gh/DermAgent/paper_data/final_30_70_large_runs/<RUN_ID>/machine_1/

rsync -aH --info=progress2 \
  zhongnan@100.126.2.24:/data/gh/DermAgent/doctor_evidence_final_30_70_<RUN_ID>/ \
  /data/gh/DermAgent/doctor_evidence_final_30_70_<RUN_ID>/
```

## Engineering Notes

- Bootstrap shards write isolated split-state roots to avoid concurrent JSONL/cognition overwrite.
- After bootstrap, `merge_experience_shards.py` merges raw/tactical/abstract experience and cognition stats.
- Compare shards use the merged train state promoted to `test`.
- The doctor evidence package is only enabled in compare shards.
- Model services use `MAX_NUM_SEQS=1` and conservative context defaults to reduce context/OOM failures.
- Qwen uses `MAX_MODEL_LEN=16384`; MedGemma uses `12288`; Llama/HuluMed/DermatoLlama use `8192`; SkinVL uses its own server defaults.
