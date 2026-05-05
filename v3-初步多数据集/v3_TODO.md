# v3 任务清单

## P0 — 阻塞多数据集应用（必须先做）

- [x] **HAM10000 混淆对定义**
  在 `agent/confusion_clusters.py` 中调用 `register_confusion_clusters("ham10000", {...})`，定义 `mel_nv`、`bkl_nv`、`mel_bkl` 三个集群

- [x] **HAM10000 重新 bootstrap**
  ham10000_v2 已使用 `ham10000_balanced_v1`（分层split）+ 独立资产仓，经验库有24个case覆盖全部7类标签，无需重新 bootstrap

- [x] **HAM10000 重新评测**
  v3b 确认混淆对正确激活（`mel_nv`/`bkl_nv`/`mel_bkl` 在 complexity_signals 中可见）。10 cases 样本太小无法判断整体效果，需要更大规模评测验证。malignant_recall 与 baseline 持平（0.8）。

---

## P1 — 提升多数据集性能（建议做）

- [x] **增加 per-class metrics**
  `agent/policy_evaluation.py` 新增 `_per_class_metrics()` 函数，`build_policy_summary()` 输出中增加 `per_class` 字段，包含每类的 support/recall/precision/f1

- [x] **`dataio/case_loader.py` 路由改为注册表模式**
  新增 `register_dataset_loader()` / `_LOADER_REGISTRY`，内置 ham10000/isic2019 已预注册，新数据集无需修改 `case_loader.py`

- [x] **HAM10000 metadata 映射**
  `_sanitize_ham10000_metadata()` 新增 `localization->region` 映射和 `diagnosis_confidence`/`has_histopathology` 字段

---

## P2 — 新数据集接入（按需）

- [ ] **DermNet loader**
  专家策划的全球临床照片，图像质量高，接入相对简单
  需要先探索标签分布，再定义标签空间

- [ ] **SCIN loader**
  众包数据，含人口统计学信息（年龄、性别、Fitzpatrick 肤色分级）
  可充分利用 `metadata_consistency_skill`，测试 metadata-rich 场景

- [ ] **SD-198 loader**
  198 类长尾分布，最复杂
  需要先制定标签合并策略（198类 → 粗粒度），或改用 top-k recall 评测

---

## P3 — 长期优化（可选）

- [ ] **经验记录写入时填充 `dataset_name`**
  `memory/experience_transform.py` 中构建 `ExperienceRecord` 时传入 `dataset_name`，完善逻辑标记

- [ ] **跨数据集经验迁移**
  支持从 PAD-UFES-20 经验库中迁移通用 abstract experience 到新数据集

- [ ] **多数据集联合训练 learned components**
  `supervised_controller`, `retrieval_scorer`, `evidence_calibrator` 目前只在 PAD-UFES-20 上训练，联合训练可提升跨数据集泛化能力

- [ ] **自动 metadata schema 对齐**
  自动检测新数据集的 metadata 字段，映射到标准字段（`region`, `age`, `sex` 等）
