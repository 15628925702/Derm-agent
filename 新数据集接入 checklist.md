# 新数据集接入 Checklist

本文档用于指导 DermAgent 接入一个新的皮肤科数据集，并按当前多数据集规范完成：

- 独立分仓
- 独立 bootstrap
- heuristic-only 评测
- baseline vs agent 对比

---

## 一、明确目标

接入一个新数据集时，目标不是一步到位做到最好，而是按以下顺序推进：

1. 能读数据
2. 能隔离资产
3. 能 bootstrap 专用经验库
4. 能跑 baseline vs agent compare
5. 再看是否需要补该数据集专用规则

---

## 二、数据准备

### 1. 明确数据根目录

需要确认：

- 图像目录
- metadata 文件
- ground truth 文件
- 样本 id / case id 规则

### 2. 确认标签空间

必须明确：

- 原始标签有哪些
- 哪些是恶性
- 哪些是良性
- 是否需要保留原始多分类，还是先做对齐子集

如果是新标签空间，需要在：

- `agent/label_space.py`

新增：

- `LabelSpace`
- `register_label_space(...)`

### 3. 确认 metadata 哪些字段可安全暴露给模型

必须排除：

- 真值标签
- 参考标签
- one-hot label 列
- 任何可直接推出 diagnosis 的字段

如果发现新数据集有特殊标签字段，需要把这些字段加入：

- `agent/state.py` 的 `LEAKY_METADATA_KEYS`

---

## 三、代码接入

### 4. 新建 loader

优先在 `dataio/` 下新增：

- `<dataset>_loader.py`
- 如有需要再加 `<dataset>_schema.py`

loader 至少应提供：

- `load_<dataset>_case_inputs(...)`
- `load_<dataset>_case_input_by_index(...)`

要求：

- `CaseInput.dataset_name` 必须正确
- `CaseInput.label_space_id` 尽量显式传入
- metadata 必须先做去泄漏清洗

### 5. 如有需要，注册到通用 case loader

如果要走通用入口，可在：

- `dataio/case_loader.py`

做路径路由或 loader 注册。

---

## 四、实验资产初始化

### 6. 新建独立实验根

使用：

```bash
python scripts/manage_dataset_experiment_assets.py init \
  --experiment-id <dataset_experiment_id> \
  --base-policy-config /root/DermAgent/state/policy/versions/heuristic_with_penalty.json
```

例如：

```bash
python scripts/manage_dataset_experiment_assets.py init \
  --experiment-id mydataset_v1 \
  --base-policy-config /root/DermAgent/state/policy/versions/heuristic_with_penalty.json
```

### 7. 加载环境

```bash
source /root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/experiment.env
```

检查：

```bash
echo $DERMAGENT_POLICY_ROOT
echo $DERMAGENT_SPLIT_STATE_ROOT
```

确认已经切到新数据集专用根。

---

## 五、构建 split

### 8. 生成或准备 split json

推荐提供一个显式的 split json，至少包含：

- `split_id`
- `train`
- `val`
- `test`

最好还包含：

- `train_case_indices`
- `val_case_indices`
- `test_case_indices`

这样 bootstrap 可以精确按 train split 建库，而不是按连续 index 硬扫。

### 9. 确认 split json 能被 compare 使用

后续 compare 建议统一显式传：

```bash
--split-json /path/to/<dataset>_split.json
```

不要依赖默认 split builder。

---

## 六、bootstrap 专用经验库

### 10. 初始化 train / val / test 分仓

如果是全新实验根，可先执行：

```bash
python - <<'PY'
from pathlib import Path
from agent.experiment_state import ensure_split_state_paths
split_root = Path("/root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/split_states")
for name in ("train", "val", "test"):
    ensure_split_state_paths(data_split=name, split_state_root=split_root)
PY
```

### 11. 只在 train split 上 bootstrap

原则：

- 经验库只在 `train` 上写回
- 不要在 `val/test` 上在线写回

推荐方式：

- 新写一个 `<dataset>_train_bootstrap.sh`
- 或者复用 `debug_single_case.py` 循环调用

必须保证：

- `--data-split train`
- `--enable-writeback`
- 读取的是 train case indices

### 12. bootstrap 完成后 promote 到 val/test

```bash
python scripts/manage_dataset_experiment_assets.py promote-state \
  --split-state-root "$DERMAGENT_SPLIT_STATE_ROOT" \
  --source-split train \
  --target-splits val,test
```

这样后续 frozen eval 才会用一致的 state。

---

## 七、先做小规模验证

### 13. 跑 12 case compare

先小样本验证：

```bash
DERMAGENT_POLICY_ROOT="$DERMAGENT_POLICY_ROOT" \
DERMAGENT_SPLIT_STATE_ROOT="$DERMAGENT_SPLIT_STATE_ROOT" \
python scripts/compare_agent_vs_qwen.py \
  --data-root /path/to/data_root \
  --limit 12 \
  --data-split test \
  --split-json /path/to/split.json \
  --output-dir /path/to/output_dir \
  --policy-config "$DERMAGENT_POLICY_ROOT/current_stable_policy.json"
```

先看：

- top-1
- top-k
- malignant recall
- error rate

### 14. 再跑 24 / 30 case

只有 12 case 看起来不拖后腿，才建议继续放大。

顺序建议：

1. 12 case
2. 24 case
3. 30 case

---

## 八、结果分析

### 15. 看 case-level better / worse

重点不是只看 summary，而是看：

- baseline 正确但 agent 变错的病例
- baseline 错误但 agent 改对的病例

重点关注：

- `selected_evidence` 是否为空
- `override_allowed` 是否为 `false`
- agent 是否仍然把 label 从 baseline 推走

### 16. 判断新数据集属于哪类

常见情况：

- 情况 A：agent 明显正收益  
  说明当前通用规则已适配得不错

- 情况 B：agent 基本持平  
  说明这套框架可迁移，但还需要补数据集特定规则

- 情况 C：agent 拖后腿  
  通常说明需要：
  - 增强 baseline anchoring
  - 降低风险型技能的漂移
  - 新增更适合该标签空间的 specialist

---

## 九、必要时补数据集特定规则

### 17. 判断是否需要新 specialist skill

如果一个新数据集中反复出现新的高频混淆，例如：

- `classA -> classB`
- `classC -> classD`

并且现有 specialist 无法覆盖，就可以考虑新增一个新的 specialist skill。

### 18. 判断是否需要新 confusion cluster

如果高频混淆稳定出现，应在：

- `agent/confusion_clusters.py`

里补该数据集对应的 confusion cluster 和 pair。

### 19. 判断是否需要 metadata field registry

如果该数据集 metadata 结构与现有数据集差异大，需要在：

- `agent/confusion_clusters.py`

里补对应 dataset 的 metadata fields 注册表。

---

## 十、接入完成标准

一个新数据集接入完成，至少要满足：

- 有独立 `policy_root`
- 有独立 `split_state_root`
- 有自己的 loader
- 有显式 `label_space_id`
- 能在 train split 上 bootstrap
- 能 promote 到 val/test
- 能跑 12 / 24 / 30 case compare

如果要认为“效果上可用”，还应满足至少其一：

- agent 明显优于 baseline
- agent 不明显拖后腿，且 case-level 失败模式可解释

---

## 十一、推荐顺序

最推荐的新数据集接入顺序是：

1. loader + label space
2. 独立实验根
3. split json
4. train bootstrap
5. promote-state
6. 12 case compare
7. 24 case compare
8. case-level 分析
9. 必要时补 specialist / confusion cluster / 保守规则

---

## 十二、不要做的事

不要：

- 在默认主仓里给新数据集写经验库
- 用 test split 做 writeback
- 直接把原始标签字段暴露给 prompt
- 不看 case-level 就只凭 summary 下结论
- 还没做小样本验证就直接跑大规模结论实验

---

## 十三、一句话原则

新数据集接入时，优先保证：

- 干净
- 分仓
- 可回退 baseline

然后才追求：

- 更高的 top-1
- 更强的 malignant recall
- 更低的 error rate
