# DermAgent v3 进度日志

## 当前状态（2026-04-17）

### 已完成的工作

**v3 分支多数据集支持改造（全部完成）：**
- `agent/confusion_clusters.py` — 注册表模式，HAM10000 集群已注册（mel_nv/bkl_nv/mel_bkl）
- `agent/policy_evaluation.py` — 注册表模式，per-class metrics
- `agent/evaluation_protocol.py` — specialist skills 注册表
- `memory/experience_schema.py` — ExperienceRecord 新增 dataset_name 字段
- `dataio/ham10000_loader.py` — localization->region 映射，diagnosis_confidence 字段
- `dataio/case_loader.py` — 注册表模式
- `agent/planner.py` — PlannerInput 新增 dataset_name，传给 detect_confusion_clusters
- `agent/run_agent.py` — PlannerInput 和 retrieve_bundle 均传入 dataset_name
- `memory/experience_retriever.py` — failure penalty(-2)/success bonus(+1)，confusion_pair_cap=2，dataset_name 传递
- `memory/experience_bank.py` — retrieve_bundle 传入 dataset_name

**文档：**
- `CODE.md` — 所有改动记录
- `v3_TODO.md` — 任务清单（P0/P1 已完成）
- `v3_README.md` — 使用指南
- `v3_multi_dataset_analysis.md` — 多数据集分析文档

---

## 核心问题：Agent 在 HAM10000 上性能差于 Baseline

### 评测结果对比

| 版本 | top1 | topk | malignant_recall |
|------|------|------|-----------------|
| Baseline (Qwen直接) | 0.367 | 0.633 | 0.857 |
| v3c Agent | **0.200** | 0.467 | 0.857 |

per-class（Agent vs Baseline）：
| 类别 | Baseline | Agent |
|------|----------|-------|
| MEL | 1.0 | 1.0 |
| NV | 0.75 | **0.0** |
| AKIEC | 0.6 | **0.0** |
| BCC | 0.2 | 0.2 |
| BKL | 0.0 | 0.25 |
| DF | 0.0 | 0.0 |
| VASC | 0.0 | 0.0 |

### 根本原因分析

**经验检索正常工作**（每个 case 检索到 tact=3, abs=3, raw=2），但检索内容质量极差：

1. **经验库严重偏向 MEL**：277条 tactical experience 中约 175 条（63%）的 confusion_pair 是 `Malignant Melanoma->xxx`（PAD-UFES-20 术语）

2. **diversity cap 没有解决问题**：cap=2 只限制同一 confusion_pair 最多2条，但 `Malignant Melanoma->bcc`、`Malignant Melanoma->nv`、`Malignant Melanoma->df` 是不同的 pair，各自都能进入结果

3. **实际检索到的 confusion_pair 分析**（30 cases）：
   - `Malignant Melanoma->bcc` 出现在 7 个 case 的检索结果中
   - `Malignant Melanoma->nv` 出现在 3 个 case
   - `Malignant Melanoma->df` 出现在 3 个 case
   - `None` 出现在 12 个 case（无匹配）
   - 几乎所有 case 都被推向 MEL 方向

4. **经验库内容问题**：
   - 277 条 tactical experience 全部 `outcome=None`（bootstrap 时未填充）
   - failure penalty 因此无法生效（没有 outcome 字段可读）
   - learning_points 全部为空列表

5. **Qwen 被误导**：经验包里充满 "Malignant Melanoma vs xxx" 的对比，Qwen 在 NV/AKIEC/BCC 等 case 上都预测成了 Malignant Melanoma

### 下一步需要做的事

**方向1（推荐）：修复经验库内容质量**

经验库的 tactical experience 是 bootstrap 时从 PAD-UFES-20 生成的，confusion_pair 全是 PAD-UFES-20 术语。需要：

a) **重新 bootstrap HAM10000 经验库**，用 HAM10000 的 train split 生成新的 tactical experience，confusion_pair 应该是 HAM10000 术语（mel->nv, bkl->nv 等）

b) 或者**清空当前 tactical experience**，只保留 raw_case_memory（24条，覆盖7类），让 agent 主要依赖 raw case 检索

**方向2：降低经验对 Qwen 的影响权重**

检查 `agent/run_agent.py` 中经验包如何被注入到 Qwen prompt，考虑降低 tactical experience 的权重或完全禁用

**方向3：检查 exclusion_reasoning_skill 问题**

之前分析发现 6/24 wrong cases 的正确答案被 exclusion_reasoning_skill 排除，这个问题可能仍然存在

---

## 关键文件路径

- 经验库（live）：`state/dataset_adaptation/ham10000_v2/split_states/test/experience/`
- 经验库（frozen eval）：`outputs/dataset_adaptation/ham10000_v2/compare_test_v3c/compare_agent_vs_qwen_20260417T155851Z/frozen_state/experience/`
- v3c 评测结果：`outputs/dataset_adaptation/ham10000_v2/compare_test_v3c/compare_agent_vs_qwen_20260417T155851Z/result_manifest.json`
- 上一次 30-case 评测（v3）：`outputs/dataset_adaptation/ham10000_v2/compare_test_v3_30cases/`
- HAM10000 split：`outputs/dataset_adaptation/ham10000_v2/ham10000_split.json`
- 实验环境变量：`state/dataset_adaptation/ham10000_v2/experiment.env`

## 运行评测的命令

```bash
cd /root/DermAgent
source state/dataset_adaptation/ham10000_v2/experiment.env && python3 scripts/compare_agent_vs_qwen.py \
  --data-root data/ham10000 \
  --split-json outputs/dataset_adaptation/ham10000_v2/ham10000_split.json \
  --data-split test \
  --limit 30 \
  --output-dir outputs/dataset_adaptation/ham10000_v2/compare_test_v3d \
  --policy-label "v3d_ham10000_fix"
```

## 经验库统计

- tactical_experience: 277 条（全部 outcome=None，63% 是 MEL confusion_pair）
- raw_case_memory: 24 条（覆盖全部7类标签）
- abstract_experience: 20 条

## 待确认的技术问题

1. `memory/experience_retriever.py` 中 `_score_tactical()` 的 failure penalty 是否实际生效？需要确认 outcome 字段的实际结构（当前所有 tactical 的 outcome=None，penalty 无法触发）

2. confusion_pair_cap=2 的 diversity cap 是否在 `_merge_results()` 中正确实现？需要 grep 确认代码已保存

3. 经验包注入 Qwen prompt 的具体方式：查看 `agent/run_agent.py` 中 `_build_evidence_package()` 或类似函数，了解 tactical experience 如何影响最终 prompt
