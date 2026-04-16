# DermAgent 参数化组件说明（当前最终状态）

## 结论

三个参数化组件全部有对应的手工规则版本，可以随时通过 policy 配置文件的字段切换，代码路径完整隔离，不需要改代码。

---

## 三个组件的切换方式

| 组件 | 切换字段 | heuristic 路径 | learned 路径 |
|------|----------|---------------|-------------|
| **Controller / Planner Scorer** | `planner_policy.controller_family` | `"heuristic"` → RuleBasedSkillPlanner，不加载 checkpoint | `"learned_supervised"` + `controller_checkpoint_path` → 加载 MLP，与 heuristic 分数混合 |
| **Retrieval Reranker** | `retrieval_policy.enable_learned_retrieval_reranker` | `false` → 完全跳过，返回 None | `true` + `retrieval_reranker_checkpoint_path` → 加载 LearnedRetrievalScorer，重排序 |
| **Evidence Calibrator** | `evidence_policy.calibrator_mode` | `"heuristic"` → 只用规则打分 | `"learned"` 或 `"hybrid"` + `calibrator_checkpoint_path` → MLP 打分，hybrid 模式叠加 |

---

## 当前已有的配置文件

- `state/policy/versions/heuristic_no_penalty.json`：全手工规则，penalty 权重为 0，adaptive budget 关闭，最接近 v0 行为
- `state/policy/versions/heuristic_with_penalty.json`：全手工规则，penalty 和 adaptive budget 开启，代表当前代码 heuristic 默认行为
- `state/policy/current_stable_policy.json`：当前主线，learned_supervised controller + learned reranker + hybrid calibrator

三种配置随时可切换，切换只需改 `--policy-config` 指向对应 JSON 文件。

---

## 系统的两层设计

**基础层（跨场景可部署）**：
- heuristic planner / evidence calibrator（不依赖任何训练数据）
- 固定 skill 语义
- 冻结 Qwen backbone

**可选优化层（站点特异）**：
- learned controller（在本地数据上训练）
- learned retrieval reranker
- hybrid/learned evidence calibrator

这个两层结构是 DermAgent 的核心设计之一。heuristic 层已经能带来相对 direct baseline 的显著提升（10-case 实验：malignant recall 25% → 87.5%），learned 层在此基础上提供可选增益。
