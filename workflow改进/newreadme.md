# DermAgent 项目说明（当前最终状态）

---

## 一、系统定位

DermAgent 是一个**结构化皮肤科诊断支持 agent 框架**，核心主张是：

> 在冻结同一个视觉语言模型骨干（Qwen）的条件下，结构化、经验驱动的 agent reasoning 能否显著优于 direct prompting？

当前已验证的核心结论：

- **heuristic-only 配置**下，malignant recall 从 25%（direct baseline）提升至 87.5%（10-case 实验）。
- 核心价值来自**结构化推理设计**本身（skill 显式化、分层经验、状态驱动控制、动态 evidence package），而不只是 learned 参数化模块。
- 框架支持 heuristic 和 learned 两种控制层，heuristic 层可零样本部署到新场景。

---

## 二、系统架构

### 三阶段 pipeline

```
[初始感知]
  → Qwen 初始感知（图像 + metadata）
  → 生成初始诊断候选和状态

[Agent 中间层]
  → Planner 选择 skill 序列
  → 执行 skill（morphology、color、differential compare、uncertainty、risk 等）
  → Retrieval 检索相关经验（raw case / tactical / abstract）
  → Evidence calibrator 组织结构化证据包
  → 更新 cognition state

[最终诊断]
  → Qwen 接收结构化 evidence package
  → 输出最终诊断
```

### 两层设计

**基础层（跨医院可部署，零样本）**：
- `RuleBasedSkillPlanner`（不加载 checkpoint）
- Heuristic evidence calibrator（纯规则打分）
- 不启用 retrieval reranker
- 冻结 Qwen backbone

**可选优化层（站点特异，有本地数据时启用）**：
- Learned controller MLP
- Learned retrieval reranker MLP
- Hybrid/learned evidence calibrator

---

## 三、三个参数化组件的切换

| 组件 | 字段 | heuristic | learned |
|------|------|-----------|---------|
| Controller / Planner Scorer | `planner_policy.controller_family` | `"heuristic"` | `"learned_supervised"` |
| Retrieval Reranker | `retrieval_policy.enable_learned_retrieval_reranker` | `false` | `true` |
| Evidence Calibrator | `evidence_policy.calibrator_mode` | `"heuristic"` | `"learned"` / `"hybrid"` |

切换只需改 `--policy-config` 指向对应 JSON 文件，不改代码。

---

## 四、当前 policy 配置文件

| 文件 | 描述 |
|------|------|
| `state/policy/versions/heuristic_no_penalty.json` | 全 heuristic，penalty=0，无 adaptive budget，接近 v0 行为 |
| `state/policy/versions/heuristic_with_penalty.json` | 全 heuristic，penalty 和 adaptive budget 启用 |
| `state/policy/current_stable_policy.json` | 当前主线：learned controller + learned reranker + hybrid calibrator |

---

## 五、实验结果（10-case）

| 配置 | top-1 | top-k | malignant recall | error rate |
|------|-------|-------|-----------------|------------|
| Direct baseline (Qwen only) | 30% | 70% | 25% | 70% |
| Heuristic no-penalty | 50% | **60%** | **87.5%** | 50% |
| Heuristic with-penalty | 50% | 50% | **87.5%** | 50% |

50-case 对比实验进行中。

---

## 六、Skill 列表

| 类别 | Skill |
|------|-------|
| Observation | morphology_analysis, color_pattern_analysis, border_surface_analysis, distribution_analysis |
| Reasoning | differential_compare, contradiction_check, metadata_consistency, temporal_evolution |
| Risk / Uncertainty | malignancy_risk_assessment, uncertainty_assessment |
| Gap / Action | information_gap_detection, escalation_recommendation |

---

## 七、经验系统（三层）

- **Raw case memory**：原始病例记录，站点特异性强
- **Tactical experience**：中层策略经验（"高不确定 + 高风险信号下优先触发 risk skill"）
- **Abstract experience**：跨病例高层诊断原则，可迁移性最强

---

## 八、论文叙事要点

- 核心贡献：结构化推理框架 + 分层经验 + 可审计 evidence package
- heuristic 层：zero-shot deployable，支撑跨医院迁移叙事
- learned 层：site-specific optional optimization，不过度声称跨医院泛化
- 当前 learned 组件在 PAD-UFES-20 上训练，跨数据集泛化有待验证

---

## 九、目录结构说明

```
agent/          - 核心 agent 逻辑（planner、evidence、calibrator、controller）
skills/         - 各原子 skill 实现
memory/         - 经验系统（experience_bank、consolidator）
cognition/      - 认知状态管理
scripts/        - 训练脚本（controller、retrieval scorer、evidence calibrator）
state/policy/   - policy 配置文件
final-script/   - 主线评测脚本
final-score/    - 实验结果输出
```
