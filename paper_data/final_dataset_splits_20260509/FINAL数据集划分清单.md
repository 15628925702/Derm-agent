# Final 数据集划分清单

- 生成时间标识：`final_20260509`
- 统一比例：`final_compare_test=30%`，`final_experience_train=70%`
- 用途：个人项目验证与经验库积累，不按论文随机评估口径命名。
- 规则：全量 case 只进入一个集合；当前 eval300 锚点全部保留在 `final_compare_test`。

| Dataset | 全量 | final_compare_test | final_experience_train | anchor retained | split json |
|---|---:|---:|---:|---:|---|
| isic2019 | 25331 | 7599 | 17732 | 300/300 | `paper_data/final_dataset_splits_20260509/splits/isic2019_final_30_70_split.json` |
| ham10000 | 10015 | 3004 | 7011 | 300/300 | `paper_data/final_dataset_splits_20260509/splits/ham10000_final_30_70_split.json` |
| sd198 | 6584 | 1975 | 4609 | 300/300 | `paper_data/final_dataset_splits_20260509/splits/sd198_final_30_70_split.json` |
| scin | 3061 | 918 | 2143 | 300/300 | `paper_data/final_dataset_splits_20260509/splits/scin_final_30_70_split.json` |
| pad20 | 2298 | 689 | 1609 | 300/300 | `paper_data/final_dataset_splits_20260509/splits/pad20_final_30_70_split.json` |

## 执行方式

对比验证继续使用现有评估脚本，只是把 `--split-json` 换成对应 final split，`--data-split test` 指向 `final_compare_test`。

示例：

```bash
DERMAGENT_DATA_ROOT=/data/gh/DermAgent/data \
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python scripts/compare_agent_vs_qwen.py \
  --data-root /data/gh/DermAgent/data/isic2019 \
  --split-json /data/gh/DermAgent/paper_data/final_dataset_splits_20260509/splits/isic2019_final_30_70_split.json \
  --data-split test \
  --limit 300 --case-offset 0 \
  --output-dir paper_data/final_dataset_splits_20260509/smoke_runs/isic2019_example
```

经验库构建时使用同一个 split JSON 的 `train` / `final_experience_train`，不要从 `test` 写回经验。
