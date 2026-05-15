# PAD-UFES-20：6个模型与 Hulu-Med Agent+fusion 详细性能统计

生成时间：2026-05-13。

## 数据与口径

- 统计对象：PAD-UFES-20 test split，case-level 记录数 `689`。
- 疾病标签：`MEL`、`BCC`、`SCC`、`ACK`、`NEV`、`SEK`。
- 恶性或高风险：`MEL`、`BCC`、`SCC`、`ACK`；良性：`NEV`、`SEK`。
- Top-1：最终诊断 canonical label 等于 ground truth。Error rate = 1 - Top-1。
- Top-k hit rate：最终诊断加鉴别诊断候选中覆盖 ground truth。
- Malignant recall：ground truth 属于恶性/高风险四类时，最终诊断也落在恶性/高风险集合。也就是说，恶性错成另一种恶性仍算 malignant recall 命中，但不算 Top-1 命中。
- Macro-F1：基于最终诊断的 6 类 one-vs-rest F1 宏平均。单病种表中的 F1 是该病种 one-vs-rest F1；单病种本身没有再做 macro 平均。
- Agent 指的是本次已完成的 `Hulu-Med Agent+fusion` 全量 PAD run，使用 `hulumed_pad20` workflow/fusion；6 个 direct 模型是不走 agent workflow 的最终诊断输出。

## 总体结论

1. **Agent 是综合表现最强的一组**：Top-1、Macro-F1、Top-k 三项都是第一。Top-1 为 45.28%（312/689），只比 DermaLLaMA direct 高 2 个 case，但比 Hulu-Med direct 高 +5.81 pp；Macro-F1 为 37.52%，比最好的 direct Macro-F1 高 +14.84 pp；Top-k 为 82.29%，比最佳 direct Top-k 高 +28.01 pp。
2. **Agent 的安全侧不是第一**：Malignant recall 为 98.17%（537/547），低于 MedGemma direct 的 100.00%、Hulu-Med direct 的 99.82%、LLaMA direct 的 99.09%。不过 MedGemma 的 Top-1 只有 8.42%，更像“几乎全往恶性打”的保守但低准确策略。
3. **Agent 的核心收益来自候选覆盖和长尾类均衡**：Top-k 从 Hulu-Med direct 的 45.14% 升到 82.29%，说明 workflow 很强地把正确答案放进鉴别诊断；Macro-F1 大幅领先，说明它不只是靠 BCC 大类撑准确率。
4. **Agent 的主要短板仍是 ACK、SCC、SEK**：ACK Top-1/Recall 只有 14.81%，大量 ACK 被打成 BCC；SCC Top-1/Recall 26.79%，仍有 37/56 被打成 BCC；SEK Top-1/Recall 5.71%，大量良性 SEK 被打成 BCC。
5. **需要继续保留 safety-first / malignant-preserving fusion 思路**：Agent 比 Hulu-Med direct 多出 9 个恶性/高风险最终错成良性的 case，主要集中在 `hulumed_pad20_guarded_subtype_override` 触发的 benign consensus override。这是当前最需要修的安全侧风险。

## 总体性能表

| Rank(Top-1) | 模型 | N | Top-1 | Error rate | Macro-F1 | Top-k hit | Malignant recall | 关键排名 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Hulu-Med Agent+fusion | 689 | 312/689 (45.28%) | 377/689 (54.72%) | 37.52% | 567/689 (82.29%) | 537/547 (98.17%) | F1#1 / Top-k#1 / Mal#4 |
| 2 | DermaLLaMA direct | 689 | 310/689 (44.99%) | 379/689 (55.01%) | 22.67% | 374/689 (54.28%) | 448/547 (81.90%) | F1#2 / Top-k#2 / Mal#5 |
| 3 | Hulu-Med direct | 689 | 272/689 (39.48%) | 417/689 (60.52%) | 21.68% | 311/689 (45.14%) | 546/547 (99.82%) | F1#3 / Top-k#5 / Mal#2 |
| 4 | LLaMA direct | 689 | 193/689 (28.01%) | 496/689 (71.99%) | 18.75% | 334/689 (48.48%) | 542/547 (99.09%) | F1#5 / Top-k#4 / Mal#3 |
| 5 | Qwen direct | 689 | 181/689 (26.27%) | 508/689 (73.73%) | 19.00% | 374/689 (54.28%) | 407/547 (74.41%) | F1#4 / Top-k#3 / Mal#6 |
| 6 | SkinVL direct | 689 | 76/689 (11.03%) | 613/689 (88.97%) | 7.79% | 91/689 (13.21%) | 258/547 (47.17%) | F1#6 / Top-k#7 / Mal#7 |
| 7 | MedGemma direct | 689 | 58/689 (8.42%) | 631/689 (91.58%) | 2.80% | 309/689 (44.85%) | 547/547 (100.00%) | F1#7 / Top-k#6 / Mal#1 |

### 读表要点

- 如果只看 Top-1，Agent 第一，但和 DermaLLaMA direct 接近：312 vs 310 个正确。
- 如果看 Macro-F1，Agent 优势明显：37.52%，第二名 DermaLLaMA direct 为 22.67%。这说明 Agent 对小类/长尾类更友好。
- 如果看 Top-k，Agent 是断层第一：82.29%，direct 模型最高只有 54.28%。这对“临床鉴别诊断候选是否覆盖真值”非常有价值。
- 如果只看 Malignant recall，MedGemma/Hulu-Med/LLaMA 更高，但 MedGemma 牺牲了几乎全部 Top-1；Hulu-Med direct 是安全侧最接近可用的 direct baseline。

## 各病种性能

说明：单病种 Top-1 与 one-vs-rest Recall 在单标签最终诊断下数值相同，所以表里合并为 `Top-1/Recall`。恶性类的 `安全错误` 表示该类被最终诊断为良性；良性类的 `安全错误` 表示该类被最终诊断为恶性。

### MEL

| 模型 | Support | Top-1/Recall | Top-k | Precision | F1 | FP | FN | 安全错误 | 预测分布 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hulu-Med Agent+fusion | 25 | 19/25 (76.00%) | 23/25 (92.00%) | 50.00% | 60.32% | 19 | 6 | 1 | {"ACK": 2, "BCC": 3, "MEL": 19, "NEV": 1} |
| LLaMA direct | 25 | 17/25 (68.00%) | 19/25 (76.00%) | 25.00% | 36.56% | 51 | 8 | 0 | {"BCC": 4, "MEL": 17, "SCC": 4} |
| Hulu-Med direct | 25 | 13/25 (52.00%) | 19/25 (76.00%) | 52.00% | 52.00% | 12 | 12 | 1 | {"ACK": 2, "BCC": 9, "MEL": 13, "NEV": 1} |
| Qwen direct | 25 | 11/25 (44.00%) | 11/25 (44.00%) | 32.35% | 37.29% | 23 | 14 | 0 | {"BCC": 14, "MEL": 11} |
| SkinVL direct | 25 | 1/25 (4.00%) | 1/25 (4.00%) | 3.57% | 3.77% | 27 | 24 | 0 | {"BCC": 9, "MEL": 1, "SCC": 4, "UNK": 11} |
| DermaLLaMA direct | 25 | 0/25 (0.00%) | 5/25 (20.00%) | 0.00% | 0.00% | 0 | 25 | 22 | {"BCC": 3, "NEV": 22} |
| MedGemma direct | 25 | 0/25 (0.00%) | 0/25 (0.00%) | 0.00% | 0.00% | 0 | 25 | 0 | {"BCC": 1, "SCC": 24} |

### BCC

| 模型 | Support | Top-1/Recall | Top-k | Precision | F1 | FP | FN | 安全错误 | 预测分布 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hulu-Med direct | 250 | 250/250 (100.00%) | 250/250 (100.00%) | 39.31% | 56.43% | 386 | 0 | 0 | {"BCC": 250} |
| DermaLLaMA direct | 250 | 221/250 (88.40%) | 233/250 (93.20%) | 49.66% | 63.60% | 224 | 29 | 21 | {"ACK": 1, "BCC": 221, "NEV": 21, "SCC": 1, "UNK": 6} |
| Hulu-Med Agent+fusion | 250 | 218/250 (87.20%) | 250/250 (100.00%) | 43.51% | 58.06% | 283 | 32 | 1 | {"ACK": 5, "BCC": 218, "SCC": 26, "SEK": 1} |
| LLaMA direct | 250 | 149/250 (59.60%) | 241/250 (96.40%) | 35.48% | 44.48% | 271 | 101 | 0 | {"ACK": 4, "BCC": 149, "MEL": 6, "SCC": 91} |
| Qwen direct | 250 | 138/250 (55.20%) | 248/250 (99.20%) | 33.17% | 41.44% | 278 | 112 | 56 | {"BCC": 138, "MEL": 4, "SCC": 52, "SEK": 56} |
| SkinVL direct | 250 | 67/250 (26.80%) | 82/250 (32.80%) | 57.26% | 36.51% | 50 | 183 | 0 | {"ACK": 1, "BCC": 67, "MEL": 2, "SCC": 19, "UNK": 161} |
| MedGemma direct | 250 | 2/250 (0.80%) | 250/250 (100.00%) | 15.38% | 1.52% | 11 | 248 | 0 | {"BCC": 2, "SCC": 248} |

### SCC

| 模型 | Support | Top-1/Recall | Top-k | Precision | F1 | FP | FN | 安全错误 | 预测分布 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MedGemma direct | 56 | 56/56 (100.00%) | 56/56 (100.00%) | 8.28% | 15.30% | 620 | 0 | 0 | {"SCC": 56} |
| LLaMA direct | 56 | 20/56 (35.71%) | 35/56 (62.50%) | 11.11% | 16.95% | 160 | 36 | 0 | {"BCC": 36, "SCC": 20} |
| Hulu-Med Agent+fusion | 56 | 15/56 (26.79%) | 47/56 (83.93%) | 31.25% | 28.85% | 33 | 41 | 2 | {"ACK": 2, "BCC": 37, "SCC": 15, "SEK": 2} |
| Qwen direct | 56 | 9/56 (16.07%) | 28/56 (50.00%) | 13.85% | 14.88% | 56 | 47 | 12 | {"BCC": 35, "SCC": 9, "SEK": 12} |
| SkinVL direct | 56 | 6/56 (10.71%) | 6/56 (10.71%) | 3.02% | 4.71% | 193 | 50 | 0 | {"BCC": 10, "SCC": 6, "UNK": 40} |
| Hulu-Med direct | 56 | 0/56 (0.00%) | 19/56 (33.93%) | 0.00% | 0.00% | 1 | 56 | 0 | {"BCC": 56} |
| DermaLLaMA direct | 56 | 0/56 (0.00%) | 12/56 (21.43%) | 0.00% | 0.00% | 2 | 56 | 3 | {"BCC": 53, "NEV": 3} |

### ACK

| 模型 | Support | Top-1/Recall | Top-k | Precision | F1 | FP | FN | 安全错误 | 预测分布 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hulu-Med Agent+fusion | 216 | 32/216 (14.81%) | 155/216 (71.76%) | 60.38% | 23.79% | 21 | 184 | 6 | {"ACK": 32, "BCC": 168, "MEL": 3, "NEV": 1, "SCC": 7, "SEK": 5} |
| DermaLLaMA direct | 216 | 16/216 (7.41%) | 32/216 (14.81%) | 94.12% | 13.73% | 1 | 200 | 38 | {"ACK": 16, "BCC": 152, "NEV": 35, "SCC": 1, "SEK": 3, "UNK": 9} |
| Qwen direct | 216 | 13/216 (6.02%) | 62/216 (28.70%) | 100.00% | 11.35% | 0 | 203 | 65 | {"ACK": 13, "BCC": 127, "MEL": 2, "SCC": 2, "SEK": 65, "UNK": 7} |
| LLaMA direct | 216 | 2/216 (0.93%) | 25/216 (11.57%) | 33.33% | 1.80% | 4 | 214 | 5 | {"ACK": 2, "BCC": 163, "NEV": 2, "SCC": 46, "SEK": 3} |
| SkinVL direct | 216 | 2/216 (0.93%) | 2/216 (0.93%) | 16.67% | 1.75% | 10 | 214 | 0 | {"ACK": 2, "BCC": 7, "MEL": 14, "SCC": 116, "UNK": 77} |
| Hulu-Med direct | 216 | 0/216 (0.00%) | 7/216 (3.24%) | 0.00% | 0.00% | 13 | 216 | 0 | {"BCC": 215, "SCC": 1} |
| MedGemma direct | 216 | 0/216 (0.00%) | 0/216 (0.00%) | 0.00% | 0.00% | 0 | 216 | 0 | {"BCC": 4, "SCC": 212} |

### NEV

| 模型 | Support | Top-1/Recall | Top-k | Precision | F1 | FP | FN | 安全错误 | 预测分布 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DermaLLaMA direct | 72 | 70/72 (97.22%) | 70/72 (97.22%) | 34.48% | 50.91% | 133 | 2 | 1 | {"BCC": 1, "NEV": 70, "SEK": 1} |
| Hulu-Med Agent+fusion | 72 | 24/72 (33.33%) | 48/72 (66.67%) | 66.67% | 44.44% | 12 | 48 | 47 | {"ACK": 5, "BCC": 32, "MEL": 10, "NEV": 24, "SEK": 1} |
| Hulu-Med direct | 72 | 8/72 (11.11%) | 15/72 (20.83%) | 61.54% | 18.82% | 5 | 64 | 64 | {"ACK": 5, "BCC": 51, "MEL": 8, "NEV": 8} |
| LLaMA direct | 72 | 3/72 (4.17%) | 9/72 (12.50%) | 42.86% | 7.59% | 4 | 69 | 66 | {"BCC": 28, "MEL": 27, "NEV": 3, "SCC": 11, "SEK": 3} |
| MedGemma direct | 72 | 0/72 (0.00%) | 1/72 (1.39%) | 0.00% | 0.00% | 0 | 72 | 72 | {"BCC": 3, "SCC": 69} |
| Qwen direct | 72 | 0/72 (0.00%) | 3/72 (4.17%) | 0.00% | 0.00% | 0 | 72 | 61 | {"BCC": 50, "MEL": 11, "SEK": 8, "UNK": 3} |
| SkinVL direct | 72 | 0/72 (0.00%) | 0/72 (0.00%) | 0.00% | 0.00% | 0 | 72 | 48 | {"ACK": 8, "BCC": 18, "MEL": 5, "SCC": 17, "UNK": 24} |

### SEK

| 模型 | Support | Top-1/Recall | Top-k | Precision | F1 | FP | FN | 安全错误 | 预测分布 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen direct | 70 | 10/70 (14.29%) | 22/70 (31.43%) | 6.62% | 9.05% | 141 | 60 | 60 | {"BCC": 52, "MEL": 6, "SCC": 2, "SEK": 10} |
| Hulu-Med Agent+fusion | 70 | 4/70 (5.71%) | 44/70 (62.86%) | 30.77% | 9.64% | 9 | 66 | 56 | {"ACK": 7, "BCC": 43, "MEL": 6, "NEV": 10, "SEK": 4} |
| DermaLLaMA direct | 70 | 3/70 (4.29%) | 22/70 (31.43%) | 42.86% | 7.79% | 4 | 67 | 15 | {"BCC": 15, "NEV": 52, "SEK": 3} |
| LLaMA direct | 70 | 2/70 (2.86%) | 5/70 (7.14%) | 25.00% | 5.13% | 6 | 68 | 66 | {"BCC": 40, "MEL": 18, "NEV": 2, "SCC": 8, "SEK": 2} |
| Hulu-Med direct | 70 | 1/70 (1.43%) | 1/70 (1.43%) | 100.00% | 2.82% | 0 | 69 | 65 | {"ACK": 6, "BCC": 55, "MEL": 4, "NEV": 4, "SEK": 1} |
| MedGemma direct | 70 | 0/70 (0.00%) | 2/70 (2.86%) | 0.00% | 0.00% | 0 | 70 | 70 | {"BCC": 3, "SCC": 67} |
| SkinVL direct | 70 | 0/70 (0.00%) | 0/70 (0.00%) | 0.00% | 0.00% | 0 | 70 | 50 | {"ACK": 1, "BCC": 6, "MEL": 6, "SCC": 37, "UNK": 20} |

## Agent 重点分析

### Agent 分病种表现

- **BCC 是 Agent 最强项**：250 个 BCC 中 218 个 Top-1 正确，Recall 87.20%，Top-k 100.00%。但 Precision 只有 43.51%，因为大量 ACK/SCC/NEV/SEK 被吸到 BCC。
- **MEL 表现相对稳**：25 个 MEL 中 19 个 Top-1 正确，Recall 76.00%，Top-k 92.00%；但有 1 个 MEL 被最终诊断为 NEV，这是安全侧不能忽略的错漏。
- **SCC 比之前惨烈的 0% 有明显改善，但仍不够**：56 个 SCC 中 15 个 Top-1 正确，Recall 26.79%，Top-k 83.93%；最大的错误方向是 SCC→BCC（37/56）。这在 malignant recall 上仍算安全命中，但在病种级准确率上是明显短板。
- **ACK 是当前最主要的病种级瓶颈**：216 个 ACK 只有 32 个 Top-1 正确，Recall 14.81%，但 Top-k 71.76%。最大错误方向是 ACK→BCC（168/216），说明正确标签经常在候选里，但最终诊断被 BCC 吸走。
- **SEK 是准确率层面的最大弱项**：70 个 SEK 只有 4 个 Top-1 正确，Recall 5.71%，Top-k 62.86%；43/70 被打成 BCC。这会造成大量良性过度恶性化，压低 Top-1/Macro-F1，但不降低 malignant recall。
- **NEV 中等偏弱**：72 个 NEV 中 24 个 Top-1 正确，Recall 33.33%，Top-k 66.67%；32/72 被打成 BCC，10/72 被打成 MEL，整体偏保守。

### Agent 混淆矩阵（行=真实，列=最终诊断）

| Truth\Pred | MEL | BCC | SCC | ACK | NEV | SEK |
| --- | --- | --- | --- | --- | --- | --- |
| MEL | 19 | 3 | 0 | 2 | 1 | 0 |
| BCC | 0 | 218 | 26 | 5 | 0 | 1 |
| SCC | 0 | 37 | 15 | 2 | 0 | 2 |
| ACK | 3 | 168 | 7 | 32 | 1 | 5 |
| NEV | 10 | 32 | 0 | 5 | 24 | 1 |
| SEK | 6 | 43 | 0 | 7 | 10 | 4 |

### Agent 安全侧错误：恶性/高风险错成良性

| 真实类别 | 最终良性预测 | case 数 |
| --- | --- | --- |
| ACK | SEK | 5 |
| SCC | SEK | 2 |
| ACK | NEV | 1 |
| MEL | NEV | 1 |
| BCC | SEK | 1 |

| case_id | GT | Final | baseline_label | agent_label | consensus_override | fusion reasons |
| --- | --- | --- | --- | --- | --- | --- |
| PAT_1022_115 | ACK | SEK | BCC | BCC | SEK | hulumed_pad20_guarded_subtype_override, use_agent_output |
| PAT_181_833 | ACK | SEK | BCC | ACK | SEK | hulumed_pad20_guarded_subtype_override, use_agent_output |
| PAT_2122_4642 | ACK | NEV | BCC | BCC | NEV | hulumed_pad20_guarded_subtype_override, use_agent_output |
| PAT_236_361 | ACK | SEK | BCC | BCC | SEK | hulumed_pad20_guarded_subtype_override, use_agent_output |
| PAT_490_933 | SCC | SEK | BCC | BCC | SEK | hulumed_pad20_guarded_subtype_override, use_agent_output |
| PAT_595_1142 | ACK | SEK | BCC | BCC | SEK | hulumed_pad20_guarded_subtype_override, use_agent_output |
| PAT_710_1330 | MEL | NEV | NEV | NEV | None | agent_matches_baseline, use_agent_output |
| PAT_76_1039 | SCC | SEK | BCC | BCC | SEK | hulumed_pad20_guarded_subtype_override, use_agent_output |
| PAT_886_1684 | ACK | SEK | BCC | BCC | SEK | hulumed_pad20_guarded_subtype_override, use_agent_output |
| PAT_993_1865 | BCC | SEK | BCC | BCC | SEK | hulumed_pad20_guarded_subtype_override, use_agent_output |

这 10 个恶性/高风险漏诊里：ACK 6 个、SCC 2 个、MEL 1 个、BCC 1 个。9 个来自 `hulumed_pad20_guarded_subtype_override` 路径，典型形态是 baseline/agent label 仍是 BCC/ACK 等恶性或高风险，但 consensus override 把最终诊断推成 SEK/NEV；另 1 个 MEL→NEV 是 baseline 与 agent_label 本身都为 NEV，属于模型/agent 本体没有识别到恶性，而不是 fusion 把恶性改成良性。

### Fusion 方向统计（Agent run）

| 参照字段 | 方向 | case 数 |
| --- | --- | --- |
| baseline_label | 恶性/高风险→良性 | 35 |
| agent_label | 恶性/高风险→良性 | 33 |
| agent_label | 良性→恶性/高风险 | 3 |

这里的方向统计是把 `fusion_decision.baseline_label` 或 `fusion_decision.agent_label` 与最终 `final_diagnosis` 比较，并不等于 ground truth 错误数。总体上，fusion 曾把 baseline_label 的恶性/高风险预测改成良性 35 次，其中 9 次发生在真实恶性/高风险病例上，直接贡献了 malignant recall 下降。

## 是否“最好最厉害”

- **论文主指标若强调综合诊断能力**：推荐把 `Hulu-Med Agent+fusion` 作为主模型，因为 Top-1、Macro-F1、Top-k 全部第一，尤其 Macro-F1/Top-k 的优势不是偶然小差距。
- **论文若强调安全筛查/不能漏恶性**：不能只报 Agent 当前 fusion 结果。Hulu-Med direct 的 malignant recall 更高（99.82% vs 98.17%），且只漏 1 个恶性/高风险；Agent 当前漏 10 个。
- **最合理表述**：Agent 是综合性能最佳，但当前 fusion 存在少量安全侧回退；应新增或启用 malignant-preserving / safety-first fusion 作为安全敏感实验，并与当前 accuracy-first fusion 并列报告。

## 建议

1. 对论文主表：同时报告 Top-1、Macro-F1、Top-k、Malignant recall，不要只报单一指标。Agent 在前三项胜出，安全侧略低，这个故事更完整。
2. 对安全版本：fusion 规则应阻止 “baseline_label 或 agent_label 为恶性/高风险，而最终 consensus_override 为 NEV/SEK” 的降级，除非有非常强的 benign evidence；至少对真实高风险常见混淆 ACK/SCC/BCC 加 malignant-preserving guard。
3. 对 ACK/SCC：不要只看 malignant recall，因为 ACK→BCC、SCC→BCC 在二分类安全上不算漏恶性，但病种能力仍然差。应单独报告病种级 Top-1/Recall 与混淆方向。
4. 对 SEK/NEV：当前 Agent 很保守，良性经常被打成 BCC/MEL。若论文讨论 clinical deployment，应说明这是降低漏恶性风险换来的过诊断/低特异性问题。

## 输出文件

- Overall CSV：`/data/gh/DermAgent/paper_data/pad20_pipeline_runs/pad20_direct_then_hulumed_agent_20260512T_backend/pad20_6models_plus_hulumed_agent_overall_metrics.csv`
- Per-class CSV：`/data/gh/DermAgent/paper_data/pad20_pipeline_runs/pad20_direct_then_hulumed_agent_20260512T_backend/pad20_6models_plus_hulumed_agent_per_class_metrics.csv`
- JSON 明细：`/data/gh/DermAgent/paper_data/pad20_pipeline_runs/pad20_direct_then_hulumed_agent_20260512T_backend/pad20_6models_plus_hulumed_agent_detailed_metrics.json`
