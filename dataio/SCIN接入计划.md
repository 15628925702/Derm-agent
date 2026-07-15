# SCIN 接入计划

本文档记录当前 `SCIN` 接入的实际状态、边界和下一步。

## 1. 当前已完成

当前已经完成：

- 新增 [`scin_loader.py`](/root/DermAgent/dataio/scin_loader.py)
- 已注册到 [`case_loader.py`](/root/DermAgent/dataio/case_loader.py)
- 已为 `SCIN` 增加第一版 label space：
  - [`label_space.py`](/root/DermAgent/agent/label_space.py)
- 已为 `SCIN` 增加基础 metadata fields：
  - [`confusion_clusters.py`](/root/DermAgent/agent/confusion_clusters.py)
- 已在固定 split 配置中加入 `SCIN`：
  - [`dataset_splits.py`](/root/DermAgent/configs/dataset_splits.py)
- 已补最小测试：
  - [`test_scin_loader.py`](/root/DermAgent/tests/test_scin_loader.py)

## 2. 当前接入策略

当前采用的是“最小可用接入”策略，而不是一步到位完成完整评测主线。

### 当前 loader 做了什么

- 合并 `scin_cases.csv` 和 `scin_labels.csv`
- 解析 `weighted_skin_condition_label`
- 取 weighted label 的 top-1 作为当前 `original_label`
- 解析最多 3 张图像路径，但当前 `CaseInput.image_path` 只使用第 1 张主图
- 把 body part / texture / symptom / duration 等字段整理进 metadata
- 移除明显泄漏字段，例如：
  - `weighted_skin_condition_label`
  - `dermatologist_skin_condition_on_label_name`

### 当前没有承诺什么

- 还没有完成适合论文主线 compare 的最终 label 子空间
- 还没有完成 `SCIN` 的 dataset-specific confusion clusters
- 还没有完成 `scin_v1` 实验资产初始化和 bootstrap 脚本
- 还没有证明当前 `scin_full` 可以直接公平对接现有 PAD20 / ISIC / HAM 主线评测

## 3. 为什么不能直接拿当前 `SCIN` 跑主线 compare

原因不是 loader 不完整，而是标签空间问题还没收敛：

- `SCIN` 的原始标签非常多，且并非当前仓库主线的固定 dermatology 六分类空间
- 当前 `weighted_skin_condition_label` 涵盖大量炎症性、感染性、肿瘤性和开放类别
- 现有 `Qwen` final diagnosis prompt 与下游评测逻辑，更适合明确、稳定的 label space

所以当前 `SCIN` 接入更准确的定位是：

- “数据已能被 DermAgent 结构化读取”
- 但“评测协议还需要先决定 label space”

## 4. 下一步建议

### 路线 A：先做 `SCIN aligned subset`

这是目前最推荐的路线。

做法：

1. 统计 `SCIN` 高频标签
2. 选一个可评测子集
3. 为该子集注册新的 `label_space_id`
4. 写 `scin_aligned_loader.py` 或在现有 loader 上加 subset 过滤
5. 生成 `split.json`
6. 初始化 `scin_v1`
7. 跑 8-case / 12-case smoke compare

优点：

- 更容易与现有 compare/evaluation 主线兼容
- 更快得到第一轮 agent vs baseline 结果

### 路线 B：直接保留 `scin_full`

做法：

- 用 `scin_full` 作为开放多类数据集
- 继续补更大的 label alias 集合
- 为开放标签评测单独调整 compare / prompt / summary

缺点：

- 工程量更大
- 不能快速复用当前主线 compare 设定

## 5. 当前推荐决策

推荐：

1. 保留当前 `scin_full` loader 作为底层读取层
2. 下一步新增 `SCIN aligned subset` 方案
3. 先用 aligned subset 跑第一轮 smoke compare

也就是说：

- `scin_loader.py` 解决“读数据”
- 下一步要解决的是“怎么评”
