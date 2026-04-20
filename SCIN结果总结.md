# SCIN 结果总结

本文档总结当前工作区中 `SCIN` 数据集接入、评测和结果分析的结论。

## 一、结论先行

当前对 `SCIN` 的总体判断是：

- `SCIN full-label` 已经完成接入，但不适合作为当前主结果线
- `SCIN grouped-label` 是合理的中层评测方式
- 在 `SCIN grouped-label` 下，当前 baseline 已经能识别部分中层疾病家族
- 截至目前，DermAgent 在 `SCIN` 上还**没有显示出超越 direct baseline 的明确增益**

用一句话概括：

> `SCIN` 已经被成功接入，并且 grouped-label 评测表明当前模型能部分识别中层疾病家族，但到目前为止，DermAgent 在 `SCIN` 上还没有证明比 direct baseline 更好；因此 `SCIN` 更适合作为方法外推的探索性数据集，而不是当前主结果支撑线。

## 二、当前已经完成的工作

### 1. 数据接入

已完成：

- `SCIN` 专用 loader
- 注册到通用 `case_loader`
- 支持多图路径读取
- 支持 metadata 清洗与结构化整理
- 支持独立 split
- 支持独立 dataset adaptation root

对应文件包括：

- [`dataio/scin_loader.py`](/root/DermAgent/dataio/scin_loader.py)
- [`dataio/case_loader.py`](/root/DermAgent/dataio/case_loader.py)
- [`configs/dataset_splits.py`](/root/DermAgent/configs/dataset_splits.py)
- [`scripts/bootstrap_scin_train_cases.sh`](/root/DermAgent/scripts/bootstrap_scin_train_cases.sh)
- [`state/dataset_adaptation/scin_v1/experiment.env`](/root/DermAgent/state/dataset_adaptation/scin_v1/experiment.env)

### 2. 标签空间

已完成两层标签空间：

#### `scin_full`

- 对接 `SCIN` 原始 full-label 诊断空间
- 用于 full-label 探索性评测

#### `scin_grouped`

- 将 `SCIN` 原始标签归并到中层标签
- 用于当前更合理的 grouped evaluation

对应文件：

- [`agent/label_space.py`](/root/DermAgent/agent/label_space.py)
- [`agent/scin_full_label_catalog.py`](/root/DermAgent/agent/scin_full_label_catalog.py)

### 3. Prompt / 运行链路适配

已完成：

- `SCIN` full-label 提示
- `related_category` 驱动的候选 shortlist 提示
- 多图输入兼容当前单图上限服务
- 命名过粗时的轻量细化

对应文件：

- [`integrations/openai_client.py`](/root/DermAgent/integrations/openai_client.py)

### 4. 测试

已补：

- `SCIN loader`
- `SCIN split`
- `SCIN label space`
- `SCIN prompt / routing`

对应测试：

- [`tests/test_scin_loader.py`](/root/DermAgent/tests/test_scin_loader.py)
- [`tests/test_dataset_splits_scin.py`](/root/DermAgent/tests/test_dataset_splits_scin.py)
- [`tests/test_openai_client_scin_prompting.py`](/root/DermAgent/tests/test_openai_client_scin_prompting.py)
- [`tests/test_label_space.py`](/root/DermAgent/tests/test_label_space.py)

## 三、full-label 路线的结果与判断

### 1. full-label 初始问题

最早在 `SCIN full-label` 下跑 baseline / agent compare 时，出现了几个问题：

- 预测经常塌缩到少数 broad labels
  - 例如：`Contact Dermatitis`
- `SCIN` ground truth 非常细
- 精确匹配下几乎无法得分

所以 full-label 下最主要的矛盾不是：

- agent 是否有效

而是：

- 当前模型是否能在 `SCIN` 的细粒度标签空间中稳定输出足够精确的标签

### 2. full-label 过程中修过的问题

在 full-label 路线上，已经修过这些问题：

- 输出为空 / 解析失败
- 单图服务与多图数据不兼容
- `SCIN` metadata 噪声过大
- broad label 命名过粗
- `SCIN` family-level 命名等价

这些修复让 full-label 结果从“几乎不可评”变成了“可分析”，但仍然没有让 agent 在 full-label 下形成稳定增益。

### 3. 对 full-label 的最终判断

当前阶段不建议把 `SCIN full-label` 作为：

- 论文主结果线
- agent 有效性的主要支撑证据

更合理的定位是：

- exploratory setting
- 用来观察模型在异构真实世界标签空间中的行为

## 四、grouped-label 路线的结果与判断

### 1. 为什么要做 grouped-label

`SCIN` 的 full-label 太细，而当前模型在这个空间中输出很不稳定。

所以 grouped-label 的作用是：

- 不再要求命中精确 disease name
- 先看模型是否至少抓住中层疾病家族

这能更公平地回答：

- 模型到底是完全不会
- 还是方向大致对，但标签太细

### 2. grouped-label 主要类别

当前 grouped 类别包括：

- `DERMATITIS_ECZEMA`
- `URTICARIA_BITE_FOLLICULITIS`
- `INFECTION_VIRAL_FUNGAL`
- `VASCULAR_PURPURIC`
- `ACNE_ROSACEA_FOLLICULAR`
- `PIGMENT_KERATOSIS_NEVUS`
- `MALIGNANT_PREMALIGNANT`
- `OTHER`

### 3. grouped-label 的关键结果

当前比较有代表性的结果是：

- 在 grouped-label 下，baseline / agent 能明显高于 full-label 表现
- 最新一轮 grouped 测试中：
  - baseline top-1 可达到约 `41.7%`
  - agent top-1 也约为 `41.7%`

这说明：

- grouped 评测方向是有效的
- 当前模型至少能稳定抓住一部分中层类别

### 4. grouped-label 下真正学到的事

在 grouped-label 下，当前模型已经能比较稳定地吃到：

- `DERMATITIS_ECZEMA`

但对于以下中层类仍然明显不稳定：

- `INFECTION_VIRAL_FUNGAL`
- `VASCULAR_PURPURIC`
- `ACNE_ROSACEA_FOLLICULAR`
- `URTICARIA_BITE_FOLLICULITIS`

也就是说，当前模型的默认行为仍然很容易往 dermatitis family 塌缩。

## 五、agent 与 baseline 的对比结论

这是当前最重要的判断。

### 当前看到的事实

无论在 full-label 还是 grouped-label 下：

- agent 与 baseline 的最终预测非常接近
- 很多时候两者输出完全一样
- 在 `SCIN` 上暂时没有看到清晰、稳定的 agent 增益

### 这意味着什么

当前阶段可以说：

- `SCIN` 已经证明“方法能接入并运行在一个更异构、更复杂的数据集上”
- 但还**不能**说 `DermAgent` 在 `SCIN` 上优于 direct baseline

所以 `SCIN` 当前更像：

- 外推验证 / 探索性评测数据集

而不是：

- 当前主结果支撑数据集

## 六、当前最合理的项目表述

如果要把 `SCIN` 放进项目当前阶段的总结里，建议这样表述：

1. `SCIN` 已经完成工程接入
2. `SCIN full-label` 暴露出当前 backbone 在细粒度真实世界标签空间下的明显困难
3. `SCIN grouped-label` 证明当前模型可以部分识别中层疾病家族
4. 但当前 agent 还没有在 `SCIN` 上显示出超越 baseline 的明确收益

## 七、后续建议

### 1. 若目标是论文主线

建议：

- 不把 `SCIN full-label` 作为主结果
- `SCIN grouped-label` 可作为 exploratory / supplementary evidence

### 2. 若目标是继续优化 `SCIN`

后续值得投入的方向是：

- 强化 `VASCULAR_PURPURIC` vs `DERMATITIS_ECZEMA` 的分流
- 强化 `INFECTION_VIRAL_FUNGAL` vs `DERMATITIS_ECZEMA` 的分流
- 继续增加 dataset-specific routing / specialist
- 或考虑更适合 `SCIN` full-label 命名的 backbone / prompt regime

### 3. 若目标是节省时间

当前也完全可以停在这里，把 `SCIN` 定位成：

- “成功接入”
- “grouped-label 可评”
- “尚未证明 agent 增益”

这已经足够支撑当前阶段的项目判断。
