# PAD-UFES-20 各模型 DermAgent 性能分析

更新时间：2026-05-12

## 指标定义

| 指标 | 含义 |
|---|---|
| Top-1 | 最终诊断 canonical label 是否等于真值 label |
| Macro-F1 | 对 `ACK/BCC/MEL/SCC/NEV/SEK` 六类分别计算 one-vs-rest F1 后取平均，用于关注小类和长尾类 |
| Top-k hit rate | 最终诊断加鉴别诊断候选中是否覆盖真值 label |
| Malignant recall | 真值为恶性或高风险类别时，模型是否仍识别为恶性或高风险 |
| 恶性或高风险类别 | `MEL`、`BCC`、`SCC`、`ACK` |
| 良性类别 | `NEV`、`SEK` |

Malignant recall 的核心含义是：`MEL/BCC/SCC/ACK` 真值病例是否被识别为恶性或高风险，而不是被判成 `NEV/SEK`。它衡量的是安全侧漏诊风险，不要求恶性四类内部 subtype 完全正确。

## 数据来源和可比性

| 模型 | 数据来源 | cases | 说明 |
|---|---|---:|---|
| qwen | final 30/70 full test | `689` | 全量 PAD-UFES-20 测试集 |
| medgemma | final 30/70 full test | `689` | 全量 PAD-UFES-20 测试集 |
| skinvl | final 30/70 full test | `689` | 全量 PAD-UFES-20 测试集 |
| hulumed | final 30/70 full test | `689` | 全量 PAD-UFES-20 测试集 |
| llama | eval300 | `300` | 现有 300-case 结果，未跑 PAD20 full |
| dermatollama | eval300 | `300` | 现有 300-case 结果，未跑 PAD20 full |

因此，qwen/medgemma/skinvl/hulumed 四个模型可以直接按 full test 对比；llama/dermatollama 只作为 300-case 参考，不应和 full test 结果做强结论式排序。

## 疾病分布

full 689-case PAD-UFES-20 测试集：

| 子病种 | 类型 | cases | 占比 |
|---|---|---:|---:|
| ACK | 恶性/高风险 | `216` | `31.35%` |
| BCC | 恶性 | `250` | `36.28%` |
| MEL | 恶性 | `25` | `3.63%` |
| SCC | 恶性 | `56` | `8.13%` |
| NEV | 良性 | `72` | `10.45%` |
| SEK | 良性 | `70` | `10.16%` |
| 合计 | - | `689` | `100.00%` |

eval300 PAD-UFES-20 参考集：

| 子病种 | 类型 | cases | 占比 |
|---|---|---:|---:|
| ACK | 恶性/高风险 | `93` | `31.00%` |
| BCC | 恶性 | `107` | `35.67%` |
| MEL | 恶性 | `15` | `5.00%` |
| SCC | 恶性 | `24` | `8.00%` |
| NEV | 良性 | `31` | `10.33%` |
| SEK | 良性 | `30` | `10.00%` |
| 合计 | - | `300` | `100.00%` |

## 各模型总体性能

以下均为 DermAgent 介入后的 agent 端表现。

| 模型 | 数据来源 | cases | Top-1 | Macro-F1 | Top-k hit rate | Malignant recall |
|---|---|---:|---:|---:|---:|---:|
| qwen | full | `689` | `33.67%` | `25.56%` | `70.97%` | `86.47%` |
| medgemma | full | `689` | `34.69%` | `15.63%` | `68.65%` | `99.45%` |
| skinvl | full | `689` | `15.82%` | `12.05%` | `18.00%` | `47.17%` |
| hulumed | full | `689` | `47.17%` | `33.72%` | `82.73%` | `97.99%` |
| llama | eval300 | `300` | `30.33%` | `20.55%` | `51.00%` | `99.16%` |
| dermatollama | eval300 | `300` | `46.67%` | `25.58%` | `62.33%` | `79.92%` |

full test 中，hulumed 的 Top-1、Macro-F1、Top-k 最高；medgemma 的 malignant recall 最高。qwen 的价值主要体现在相对 baseline 的综合提升和恶性漏诊下降，详见 `当前最优结果_qwen_pad20.md`。

## 各疾病类型性能

分病种 Top-1 等价于该病种 one-vs-rest recall；表中的 F1 是该病种 F1，不是整体 Macro-F1。整体 Macro-F1 是六个病种 F1 的平均。

| 模型 | 数据来源 | 子病种 | cases | Top-1 / Recall | F1 | Top-k hit rate | Precision |
|---|---|---|---:|---:|---:|---:|---:|
| qwen | full | ACK | `216` | `20.37%` | `30.88%` | `43.06%` | `63.77%` |
| qwen | full | BCC | `250` | `62.80%` | `45.38%` | `99.60%` | `35.52%` |
| qwen | full | MEL | `25` | `44.00%` | `37.29%` | `48.00%` | `32.35%` |
| qwen | full | SCC | `56` | `16.07%` | `16.51%` | `53.57%` | `16.98%` |
| qwen | full | NEV | `72` | `9.72%` | `17.72%` | `56.94%` | `100.00%` |
| qwen | full | SEK | `70` | `5.71%` | `5.56%` | `91.43%` | `5.41%` |
| medgemma | full | ACK | `216` | `0.00%` | `0.00%` | `39.35%` | `0.00%` |
| medgemma | full | BCC | `250` | `86.80%` | `54.52%` | `100.00%` | `39.74%` |
| medgemma | full | MEL | `25` | `0.00%` | `0.00%` | `80.00%` | `0.00%` |
| medgemma | full | SCC | `56` | `10.71%` | `7.32%` | `89.29%` | `5.56%` |
| medgemma | full | NEV | `72` | `20.83%` | `29.13%` | `40.28%` | `48.39%` |
| medgemma | full | SEK | `70` | `1.43%` | `2.82%` | `55.71%` | `100.00%` |
| skinvl | full | ACK | `216` | `16.20%` | `26.52%` | `16.20%` | `72.92%` |
| skinvl | full | BCC | `250` | `26.80%` | `36.51%` | `32.80%` | `57.26%` |
| skinvl | full | MEL | `25` | `4.00%` | `3.77%` | `4.00%` | `3.57%` |
| skinvl | full | SCC | `56` | `10.71%` | `5.48%` | `10.71%` | `3.68%` |
| skinvl | full | NEV | `72` | `0.00%` | `0.00%` | `0.00%` | `0.00%` |
| skinvl | full | SEK | `70` | `0.00%` | `0.00%` | `0.00%` | `0.00%` |
| hulumed | full | ACK | `216` | `15.28%` | `24.54%` | `75.00%` | `62.26%` |
| hulumed | full | BCC | `250` | `98.00%` | `61.33%` | `100.00%` | `44.63%` |
| hulumed | full | MEL | `25` | `76.00%` | `61.29%` | `92.00%` | `51.35%` |
| hulumed | full | SCC | `56` | `0.00%` | `0.00%` | `87.50%` | `0.00%` |
| hulumed | full | NEV | `72` | `31.94%` | `43.40%` | `58.33%` | `67.65%` |
| hulumed | full | SEK | `70` | `7.14%` | `11.76%` | `62.86%` | `33.33%` |
| llama | eval300 | ACK | `93` | `4.30%` | `8.00%` | `15.05%` | `57.14%` |
| llama | eval300 | BCC | `107` | `62.62%` | `47.18%` | `98.13%` | `37.85%` |
| llama | eval300 | MEL | `15` | `66.67%` | `40.00%` | `80.00%` | `28.57%` |
| llama | eval300 | SCC | `24` | `33.33%` | `16.16%` | `66.67%` | `10.67%` |
| llama | eval300 | NEV | `31` | `3.23%` | `5.88%` | `9.68%` | `33.33%` |
| llama | eval300 | SEK | `30` | `3.33%` | `6.06%` | `10.00%` | `33.33%` |
| dermatollama | eval300 | ACK | `93` | `18.28%` | `29.06%` | `24.73%` | `70.83%` |
| dermatollama | eval300 | BCC | `107` | `90.65%` | `69.04%` | `96.26%` | `55.75%` |
| dermatollama | eval300 | MEL | `15` | `0.00%` | `0.00%` | `40.00%` | `0.00%` |
| dermatollama | eval300 | SCC | `24` | `0.00%` | `0.00%` | `37.50%` | `0.00%` |
| dermatollama | eval300 | NEV | `31` | `74.19%` | `41.44%` | `96.77%` | `28.75%` |
| dermatollama | eval300 | SEK | `30` | `10.00%` | `13.95%` | `53.33%` | `23.08%` |

## 论文分析要点

- full test 里 hulumed 的综合绝对表现最好，尤其 BCC、MEL、Top-k 和 Macro-F1 明显强。
- medgemma 的 malignant recall 接近满分，但 Macro-F1 较低，说明其安全侧很强但 subtype 分布偏向少数类别，ACK/MEL final subtype 命中不足。
- qwen 的绝对 Top-1 不最高，但 DermAgent 对其贡献收益最均衡：Top-1、Macro-F1、Top-k 和 malignant recall 都同步改善。
- skinvl 在 PAD-UFES-20 上绝对表现偏弱，尤其 NEV/SEK final subtype 命中为 0，需要谨慎作为 PAD20 主结果。
- llama 和 dermatollama 当前只有 eval300，适合做补充参考；若论文主表需要完整 6 模型 PAD20 对照，应补跑这两个模型的 full 689-case。

## 逐 case 结果复算说明

本表指标可以从 case-level 结果文件复算。每行应至少包含：

| 字段 | 用途 |
|---|---|
| `case_id` | 病例追踪 |
| `ground_truth_canonical_label` | 真值类别 |
| `ground_truth_malignant_flag` | 真值是否恶性或高风险 |
| `agent_final_diagnosis` / `agent_final_canonical_label` | DermAgent 最终诊断，用于 Top-1、Macro-F1、Recall |
| `agent_differential_diagnoses` / `agent_topk_canonical_labels` | DermAgent 鉴别诊断候选，用于 Top-k |
| `agent_correct` | 已计算 Top-1 hit，可审计 |
| `agent_topk_hit` | 已计算 Top-k hit，可审计 |
| `agent_malignant_recall_hit` | 已计算恶性召回 hit，可审计 |

后续如果要计算任意论文指标，推荐读取逐 case CSV/XLSX/JSONL，优先使用 `agent_final_canonical_label` 和 `agent_topk_canonical_labels`；如果读取旧导出，则先 canonicalize `agent_final_diagnosis` 和 `agent_differential_diagnoses`，再按 `ground_truth_canonical_label` 汇总，而不是只依赖摘要表。
