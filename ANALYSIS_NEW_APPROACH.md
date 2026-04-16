# DermAgent 新方向定位说明（当前最终状态）

## 结论

DermAgent 已经是一个"heuristic 层 + optional learned 层"双层框架。  
关于"参数化组件的泛化风险"的问题已经得到解答并落实到实验：  
- 三个参数化组件（Controller、Retrieval Reranker、Evidence Calibrator）均有可用的 heuristic 替代路径，切换只需改 policy JSON 的一个字段。  
- 10-case 实验已验证：heuristic-only 配置下 malignant recall = 87.5%，相比 direct baseline 的 25% 有显著提升。  
- 这证明了"核心价值来自结构化推理设计，不是只靠 learned 组件"。

---

## 一、系统定位

### 基础层（跨医院可部署，零样本）
- heuristic planner（`RuleBasedSkillPlanner`，不加载任何 checkpoint）
- heuristic evidence calibrator（`calibrator_mode: "heuristic"`，纯规则打分）
- 关闭 retrieval reranker（`enable_learned_retrieval_reranker: false`）
- 冻结 Qwen backbone
- 固定 skill 语义

### 可选优化层（站点特异，有本地数据时启用）
- learned controller（`controller_family: "learned_supervised"` + MLP checkpoint）
- learned retrieval reranker（`enable_learned_retrieval_reranker: true` + MLP checkpoint）
- hybrid/learned evidence calibrator（`calibrator_mode: "hybrid"` or `"learned"` + MLP checkpoint）

---

## 二、切换方式

| 组件 | 字段 | heuristic | learned |
|------|------|-----------|---------|
| Controller / Planner Scorer | `planner_policy.controller_family` | `"heuristic"` | `"learned_supervised"` |
| Retrieval Reranker | `retrieval_policy.enable_learned_retrieval_reranker` | `false` | `true` |
| Evidence Calibrator | `evidence_policy.calibrator_mode` | `"heuristic"` | `"learned"` / `"hybrid"` |

切换只改 `--policy-config` 指向的 JSON 文件，不改代码。

---

## 三、已有 policy 配置

- `state/policy/versions/heuristic_no_penalty.json`：全手工规则，penalty 权重 = 0，adaptive budget 关闭（最接近 v0 行为）
- `state/policy/versions/heuristic_with_penalty.json`：全手工规则，penalty 和 adaptive budget 开启
- `state/policy/current_stable_policy.json`：learned controller + learned reranker + hybrid calibrator（主线配置）

---

## 四、实验结果（10-case）

| 配置 | top-1 | top-k | malignant recall | error rate |
|------|-------|-------|-----------------|------------|
| Direct baseline (Qwen only) | 30% | 70% | 25% | 70% |
| Heuristic no-penalty | 50% | **60%** | **87.5%** | 50% |
| Heuristic with-penalty | 50% | 50% | **87.5%** | 50% |

核心结论：heuristic 结构化框架本身就带来了 malignant recall 从 25% → 87.5% 的显著提升。  
50-case 对比实验正在进行中。

---

## 五、论文叙事建议

**应该表述**：
- 框架支持 heuristic 和 learned 两种控制层实现
- Heuristic 层支持零样本部署，是跨医院可迁移的基础
- Learned 层是有数据时的可选优化，不是系统成立的前提
- 核心价值来自结构化推理设计（skill 显式化、分层经验、状态驱动控制、动态 evidence package）

**不应承诺**：
- "learned 组件支持跨医院零样本迁移"（learned 组件在 PAD-UFES-20 上训练，跨数据集泛化未验证）

---

## 六、Workflow 适应叙事的当前状态

框架从架构上支持 workflow-aware 叙事（skill 可组合、experience 分层、state 驱动动态控制），但以下部分尚未完整落地：

| 能力 | 状态 |
|------|------|
| skill 显式化、可组合 | ✅ 已有，skills/ 目录完整 |
| 分层经验（raw/tactical/abstract） | ✅ 已有 |
| state 驱动动态控制 | ✅ 已有（cognition_state + policy_config + planner） |
| 结构化 evidence package | ✅ 已有 |
| heuristic/learned 双路径 | ✅ 已有，可切换 |
| 显式 workflow_context 字段建模 | ❌ 暂无（policy 目前是性能参数，不是医院环境约束） |
| 跨 workflow 场景实验 | ❌ 暂无（可作为后续补充实验） |

workflow_context 代码建模和跨 workflow 实验属于可选增强，不是当前论文主线的前提。
