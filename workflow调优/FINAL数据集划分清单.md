# Final 数据集划分清单

本清单对应 `final_20260509`，用于个人项目的长期经验库积累与对比验证，不按论文随机评估口径命名。

生成脚本：

```bash
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python scripts/build_final_dataset_splits.py --self-test
```

生成目录：

- `paper_data/final_dataset_splits_20260509/manifest.json`
- `paper_data/final_dataset_splits_20260509/FINAL数据集划分清单.md`
- `paper_data/final_dataset_splits_20260509/self_test_results.json`
- `paper_data/final_dataset_splits_20260509/splits/`

## 划分原则

- 统一比例：`final_compare_test=30%`，`final_experience_train=70%`。
- 全量 case 都用上：每条 case 要么进入经验库训练集，要么进入对比验证集。
- 当前 eval300 作为验证集锚点：每个数据集当前 300-case 全部保留在 `final_compare_test`。
- 扩展方式：在保留 300-case 的基础上，用当前 300-case label 结构和自然全量 label 结构各 50% 的 blend 扩大验证集；稀有 label 默认最多进验证集约 50%，避免经验库训练集被抽空。
- 执行兼容：split JSON 同时提供现有脚本需要的 `train` / `test` / `*_case_indices`，也提供语义别名 `final_experience_train` / `final_compare_test`。

## 数量

| Dataset | 全量 | final_compare_test | final_experience_train | 当前 300 锚点保留 | split JSON |
|---|---:|---:|---:|---:|---|
| ISIC2019 | 25,331 | 7,599 | 17,732 | 300/300 | `paper_data/final_dataset_splits_20260509/splits/isic2019_final_30_70_split.json` |
| HAM10000 | 10,015 | 3,004 | 7,011 | 300/300 | `paper_data/final_dataset_splits_20260509/splits/ham10000_final_30_70_split.json` |
| SD198 | 6,584 | 1,975 | 4,609 | 300/300 | `paper_data/final_dataset_splits_20260509/splits/sd198_final_30_70_split.json` |
| SCIN | 3,061 | 918 | 2,143 | 300/300 | `paper_data/final_dataset_splits_20260509/splits/scin_final_30_70_split.json` |
| PAD20 | 2,298 | 689 | 1,609 | 300/300 | `paper_data/final_dataset_splits_20260509/splits/pad20_final_30_70_split.json` |

## 直接执行方式

对比验证使用 `--data-split test`，即 `final_compare_test`：

```bash
DERMAGENT_DATA_ROOT=/data/gh/DermAgent/data \
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python scripts/compare_agent_vs_qwen.py \
  --data-root /data/gh/DermAgent/data/isic2019 \
  --split-json /data/gh/DermAgent/paper_data/final_dataset_splits_20260509/splits/isic2019_final_30_70_split.json \
  --data-split test \
  --limit 300 --case-offset 0 \
  --output-dir paper_data/final_dataset_splits_20260509/smoke_runs/isic2019_example
```

经验库构建使用同一个 split JSON 的 `train` / `final_experience_train`。不要从 `test` / `final_compare_test` 写回经验。

## 自检结果

`scripts/build_final_dataset_splits.py --self-test` 已通过：

- 5 个 split 均可由现有 `resolve_case_selection` 解析。
- 每个 split 的前 8 个 `train` 与前 8 个 `test` case index 均能通过当前 loader 读回同一 case_id。
- 每个数据集当前 300-case 锚点均保留在 `final_compare_test`。
