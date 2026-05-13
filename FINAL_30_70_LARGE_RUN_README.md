# final_20260509 30/70 全量大实验运行说明

这份说明对应当前整合分支上的大实验调度脚本：

```text
scripts/run_final_30_70_6x5_dynamic.py
```

默认运行的是 `final_20260509` 的 30/70 固定划分：

- `train` / `final_experience_train`：只用于 bootstrap 建经验库。
- `test` / `final_compare_test`：只用于最终对比评测。
- 当前已有 final split 的数据集是 5 个：`ham10000`、`isic2019`、`pad20`、`scin`、`sd198`。
- `xiangya` 目前没有 `final_20260509` 的 30/70 split JSON，所以默认脚本是 `6 models x 5 datasets`，不能把这次直接叫严格的 6 数据集全量，除非先补 Xiangya 的 final split。

## 关键约定

- Bootstrap 阶段不会打开“导出给医生的辅助证据诊断包”。
- Compare 阶段默认 GPU `0,1,2,3,4,5,6,7` 全部跑评测，不单独浪费一张卡跑证据包服务。
- Compare 全部结束并合并 report 后，脚本会进入 posthoc doctor evidence 阶段；这个阶段再用 GPU `0,1,2,3,4,5,6,7` 启动 Qwen 服务并行导出每个 case 的医生证据文本。
- 每个 shard 完成后会写 `*.DONE.json`，断线或终端退出后用同一个命令、同一个 `RUN_ID` 可以续跑。
- 模型服务使用 `MAX_NUM_SEQS=1`，并按模型设置较保守的上下文长度，降低上下文过长和 OOM 风险。

## Git 保护点

大实验开始前已经打 tag 并推送：

```bash
git tag beforeLargeTest
git push origin merge-final-weak-workflow-tuning
git push origin beforeLargeTest
```

当前 `beforeLargeTest` 指向：

```text
a60aae274eb04e3caa3ec37cdc5a2726fb80a3c2
```

## 需要迁移到 100.126.2.24 的目录

必须迁移：

```text
/data/gh/DermAgent
/data/gh/models
```

如果 100.126.2.24 上没有同名 conda 环境，也需要迁移：

```text
/home/zhongnan/miniconda3/envs/dermagent-6x6
```

保持这些路径不变最省事，因为现有模型启动脚本会自动寻找 `/data/gh/models`。

不要把密码写进仓库文件。同步时推荐交互式输入密码；如果确认机器上有 `sshpass`，可以临时在 shell 里用环境变量。

普通同步：

```bash
cd /data/gh/DermAgent
bash scripts/rsync_large_run_to_remote.sh
```

连 conda 环境一起同步：

```bash
cd /data/gh/DermAgent
COPY_CONDA_ENV=1 bash scripts/rsync_large_run_to_remote.sh
```

如果使用 `sshpass`：

```bash
cd /data/gh/DermAgent
export SSHPASS='<这里填本次 ssh 密码>'
sshpass -e bash scripts/rsync_large_run_to_remote.sh
unset SSHPASS
```

## 目标机器启动前检查

在 `100.126.2.24` 上执行：

```bash
cd /data/gh/DermAgent
git status --short
git rev-parse HEAD
nvidia-smi
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python -m py_compile \
  scripts/run_final_30_70_6x5_dynamic.py \
  scripts/merge_experience_shards.py \
  scripts/merge_final_compare_reports.py \
  scripts/generate_doctor_evidence_posthoc.py \
  scripts/export_doctor_evidence_packages.py
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python -m pytest \
  tests/test_conservative_fusion.py tests/test_model_workflow_router.py -q
```

## 双机启动命令

两台机器必须使用同一个 `RUN_ID`。

`100.126.2.23` 使用 `--machine-id 0`：

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

`100.126.2.24` 使用 `--machine-id 1`，并填入同一个 `RUN_ID`：

```bash
cd /data/gh/DermAgent
export RUN_ID=<填 100.126.2.23 打印出来的 RUN_ID>
mkdir -p "paper_data/final_30_70_large_runs/${RUN_ID}"
nohup setsid /home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python \
  scripts/run_final_30_70_6x5_dynamic.py \
  --run-id "${RUN_ID}" \
  --machine-id 1 \
  --machine-count 2 \
  > "paper_data/final_30_70_large_runs/${RUN_ID}/machine_1_launcher.log" 2>&1 &
echo $! > "paper_data/final_30_70_large_runs/${RUN_ID}/machine_1_launcher.pid"
```

续跑仍然用同一条命令和同一个 `RUN_ID`。默认 `--resume` 已开启，已完成 shard 会跳过。

## 输出目录

每台机器各自保留：

```text
paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/run_manifest.json
paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/reports/
paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/paper_case_exports/
paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/logs/
doctor_evidence_final_30_70_<RUN_ID>/
```

每个组合合并后的 compare report：

```text
paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/reports/<model>/<dataset>/compare_agent_vs_qwen_final_compare_test_merged.json
```

给论文留存的逐 case CSV/JSON/XLSX：

```text
paper_data/final_30_70_large_runs/<RUN_ID>/machine_<id>/paper_case_exports/<model>/<dataset>/merged/
```

医生证据包 posthoc 输出：

```text
doctor_evidence_final_30_70_<RUN_ID>/<model>/<dataset>/shard_<id>/<case_id>.json
doctor_evidence_final_30_70_<RUN_ID>/<model>/<dataset>/shard_<id>/<case_id>.md
```

## 监控命令

```bash
tail -f paper_data/final_30_70_large_runs/<RUN_ID>/machine_0/runner.log
find paper_data/final_30_70_large_runs/<RUN_ID>/machine_0 -name '*.DONE.json' | wc -l
nvidia-smi
```

第二台机器把 `machine_0` 换成 `machine_1`。

## 把第二台机器结果收回第一台

两台都结束后，在 `100.126.2.23` 上执行：

```bash
rsync -aH --info=progress2 \
  zhongnan@100.126.2.24:/data/gh/DermAgent/paper_data/final_30_70_large_runs/<RUN_ID>/machine_1/ \
  /data/gh/DermAgent/paper_data/final_30_70_large_runs/<RUN_ID>/machine_1/

rsync -aH --info=progress2 \
  zhongnan@100.126.2.24:/data/gh/DermAgent/doctor_evidence_final_30_70_<RUN_ID>/ \
  /data/gh/DermAgent/doctor_evidence_final_30_70_<RUN_ID>/
```

## 工程实现说明

- Bootstrap shard 使用独立 split-state 根目录，避免多个进程同时写同一份 JSONL/cognition 文件。
- Bootstrap 后由 `scripts/merge_experience_shards.py` 合并 raw/tactical/abstract experience 和 cognition stats。
- Compare 前把合并后的 train state promote 到 `test`。
- Compare shard 只负责评测和论文逐 case 导出，不启动医生证据包 summary。
- Doctor evidence 阶段读取 compare merged report，再用 Qwen 并行生成医生证据文本。
- Qwen 默认 `MAX_MODEL_LEN=16384`，MedGemma 默认 `12288`，Llama/HuluMed/DermatoLlama 默认 `8192`，SkinVL 使用自身服务默认值。
