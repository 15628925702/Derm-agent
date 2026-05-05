# scin_v1

这是 `SCIN` 的第一版 dataset adaptation 实验根。

## 当前状态

已完成：

- 独立 `policy root`
- 独立 `split state root`
- 独立 `output root`
- `scin_split.json`
- `bootstrap_scin_train_cases.sh`

当前默认配置：

- seed policy: `heuristic_with_penalty`
- controller: `heuristic`
- retrieval reranker: `off`
- conservative fusion: `soft`

## 环境切换

```bash
cd /root/DermAgent
source /root/DermAgent/state/dataset_adaptation/scin_v1/experiment.env
```

检查：

```bash
echo $DERMAGENT_POLICY_ROOT
echo $DERMAGENT_SPLIT_STATE_ROOT
```

## split 文件

```bash
/root/DermAgent/outputs/dataset_adaptation/scin_v1/scin_split.json
```

## bootstrap

先确认本地模型服务在线，再执行：

```bash
cd /root/DermAgent
bash scripts/bootstrap_scin_train_cases.sh
```

小规模 smoke bootstrap 示例：

```bash
cd /root/DermAgent
START_INDEX=0 COUNT=8 bash scripts/bootstrap_scin_train_cases.sh
```

## promote

bootstrap 完成后，可将 train state promote 到 val/test：

```bash
python scripts/manage_dataset_experiment_assets.py promote-state \
  --split-state-root /root/DermAgent/state/dataset_adaptation/scin_v1/split_states \
  --source-split train \
  --target-splits val,test
```

## 注意

`SCIN` 当前走的是 full-label 路线，不是 aligned subset。

这意味着：

- 标签空间更大
- baseline / agent 的最终标签命名可能与 ground truth 有较大表述差异
- 后续 compare 前，可能仍需要补更强的 label canonicalization 或 prompt 约束
