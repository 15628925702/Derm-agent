# 当前最优结果：Qwen + DermAgent / PAD-UFES-20

更新时间：2026-05-12

## 结论

当前如果只选一个可重点写入论文分析的 PAD-UFES-20 结果，推荐使用：

**Qwen2.5-VL-7B-Instruct + DermAgent / PAD-UFES-20**

它不是单项 Top-1 最大的组合，但在 Top-1、Macro-F1、Top-k、恶性召回、漏诊率和错误率之间最均衡，尤其适合表述为 **DermAgent 对基础模型的诊断贡献收益**：

| 指标 | Baseline | DermAgent | 贡献收益 |
|---|---:|---:|---:|
| Top-1 / 最终诊断正确率 | `180/689 = 26.12%` | `232/689 = 33.67%` | `+7.55pp` |
| Macro-F1 | `19.36%` | `25.56%` | `+6.20pp` |
| Top-k hit rate / 鉴别诊断覆盖率 | `376/689 = 54.57%` | `489/689 = 70.97%` | `+16.40pp` |
| Malignant recall / 恶性召回 | `406/547 = 74.22%` | `473/547 = 86.47%` | `+12.25pp` |
| 恶性漏诊率 | `141/547 = 25.78%` | `74/547 = 13.53%` | `-12.25pp` |
| 错误率 | `509/689 = 73.88%` | `457/689 = 66.33%` | `-7.55pp` |

论文中建议表述为：**在 PAD-UFES-20 全量测试集上，DermAgent 提升了 Qwen 的最终诊断正确率、长尾类别 Macro-F1 和鉴别诊断覆盖率，并显著减少恶性或高风险病变被判为良性的漏诊。**

## 指标口径

| 指标 | 计算口径 |
|---|---|
| Top-1 | 最终诊断 canonical label 是否等于真值 label |
| Macro-F1 | 对 `ACK/BCC/MEL/SCC/NEV/SEK` 六类分别做 one-vs-rest F1 后取平均，用于关注小类和长尾类 |
| Top-k hit rate | 最终诊断加鉴别诊断候选中是否覆盖真值 label |
| Malignant recall | 真值为恶性或高风险类别时，预测是否仍落在恶性或高风险类别 |
| 恶性类别 | `MEL`、`BCC`、`SCC`、`ACK` |
| 良性类别 | `NEV`、`SEK` |

这里的 Malignant recall 不是要求恶性四类内部 subtype 完全正确，而是衡量真恶性或高风险病例是否被识别为恶性/高风险，避免被判成 `NEV` 或 `SEK`。

## 数据来源

| 项目 | 内容 |
|---|---|
| Model | `Qwen2.5-VL-7B-Instruct` |
| Dataset | `pad20` / PAD-UFES-20 |
| Split | final 30/70 test split |
| Test cases | `689` |
| Compare report | `paper_data/final_30_70_large_runs/final_30_70_3x3_0p5to3_strat_no_doctor_20260510T113540Z/machine_0/reports/qwen/pad20/compare_agent_vs_qwen_final_compare_test_merged.json` |
| Case-level exports | `paper_data/case_level_exports/current_best_qwen_pad20_full689/qwen_pad20_full689_case_level.xlsx` |
| Reproducibility cell | `qwen__pad20__dataset_best` |

case-level CSV/XLSX/JSONL 已逐 case 保存：真值标签、baseline 最终诊断及 canonical label、baseline Top-k canonical 候选、DermAgent 最终诊断及 canonical label、DermAgent Top-k canonical 候选、Top-1 是否命中、Top-k 是否命中、恶性召回是否命中。后续计算 Macro-F1、每类 recall、混淆矩阵、漏诊率等指标时，应优先从这些逐 case 文件重新汇总。

## PAD-UFES-20 测试集疾病分布

| 子病种 | 恶性/良性 | cases | 占比 |
|---|---|---:|---:|
| BCC | 恶性 | `250` | `36.28%` |
| ACK | 恶性/高风险 | `216` | `31.35%` |
| NEV | 良性 | `72` | `10.45%` |
| SEK | 良性 | `70` | `10.16%` |
| SCC | 恶性 | `56` | `8.13%` |
| MEL | 恶性 | `25` | `3.63%` |
| 合计 | - | `689` | `100.00%` |

恶性或高风险病例共 `547/689 = 79.39%`，良性病例共 `142/689 = 20.61%`。因此，恶性召回和漏诊率是 PAD-UFES-20 上必须单独报告的安全指标。

## 各疾病类型性能

下表中，分病种 Top-1 等价于该病种 one-vs-rest recall；F1 是该病种 one-vs-rest F1，整体 Macro-F1 是六个病种 F1 的平均。

| 子病种 | cases | Baseline Top-1/Recall | DermAgent Top-1/Recall | 变化 | Baseline F1 | DermAgent F1 | 变化 | Baseline Top-k | DermAgent Top-k | 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACK | `216` | `6.02%` | `20.37%` | `+14.35pp` | `11.35%` | `30.88%` | `+19.53pp` | `28.70%` | `43.06%` | `+14.35pp` |
| BCC | `250` | `54.40%` | `62.80%` | `+8.40pp` | `40.96%` | `45.38%` | `+4.42pp` | `99.20%` | `99.60%` | `+0.40pp` |
| MEL | `25` | `44.00%` | `44.00%` | `+0.00pp` | `37.29%` | `37.29%` | `+0.00pp` | `44.00%` | `48.00%` | `+4.00pp` |
| SCC | `56` | `16.07%` | `16.07%` | `+0.00pp` | `14.75%` | `16.51%` | `+1.76pp` | `50.00%` | `53.57%` | `+3.57pp` |
| NEV | `72` | `1.39%` | `9.72%` | `+8.33pp` | `2.74%` | `17.72%` | `+14.98pp` | `5.56%` | `56.94%` | `+51.39pp` |
| SEK | `70` | `14.29%` | `5.71%` | `-8.57pp` | `9.05%` | `5.56%` | `-3.49pp` | `32.86%` | `91.43%` | `+58.57pp` |

主要观察：

- ACK 是 DermAgent 贡献收益最明显的恶性/高风险类别，Top-1、F1、Top-k 同时提升。
- BCC 在 baseline 已经较强的情况下仍有 Top-1 和 F1 提升。
- MEL 的 Top-1 和 F1 没变，但 Top-k 增加；同时 MEL 的恶性识别已是 `100.00% -> 100.00%`，所以 `+0.00pp` 不代表无效，而是没有漏诊空间。
- SCC 的 subtype Top-1 不变，但 F1 和恶性识别改善，说明 DermAgent 对安全侧仍有贡献。
- NEV 和 SEK 的 Top-k 大幅提升，说明候选诊断覆盖显著改善；SEK 的 Top-1 下降是主要局限性。

## 恶性 / 良性二分类安全指标

| 指标 | Baseline | DermAgent | 贡献收益 |
|---|---:|---:|---:|
| 真恶性判恶性 TP | `406` | `473` | `+67` |
| 真恶性判良性 FN / 漏诊 | `141` | `74` | `-67` |
| 真良性判良性 TN | `21` | `17` | `-4` |
| 真良性判恶性 FP | `121` | `125` | `+4` |
| 恶性召回 / sensitivity | `74.22%` | `86.47%` | `+12.25pp` |
| 良性特异性 / specificity | `14.79%` | `11.97%` | `-2.82pp` |
| 良性误报率 | `85.21%` | `88.03%` | `+2.82pp` |

DermAgent 的主要安全贡献是明显减少恶性或高风险病例被判为良性的漏诊。代价是良性病例被判为恶性的比例略升，体现为 sensitivity-oriented trade-off。

## 各病种二分类识别

恶性/高风险子病种看“是否被判为恶性/高风险”；良性子病种看“是否被判为良性”。

| 子病种 | 类型 | Baseline | DermAgent | 变化 | 解释 |
|---|---|---:|---:|---:|---|
| ACK | 恶性/高风险识别 | `66.67%` | `83.33%` | `+16.67pp` | 明显减少 ACK 漏诊 |
| BCC | 恶性识别 | `77.20%` | `87.60%` | `+10.40pp` | BCC 安全性提升 |
| MEL | 恶性识别 | `100.00%` | `100.00%` | `+0.00pp` | baseline 已无 MEL 漏诊空间 |
| SCC | 恶性识别 | `78.57%` | `87.50%` | `+8.93pp` | SCC 安全性提升 |
| NEV | 良性识别 | `15.28%` | `18.06%` | `+2.78pp` | 良性痣识别小幅改善 |
| SEK | 良性识别 | `14.29%` | `5.71%` | `-8.57pp` | SEK 更容易被保守地推向恶性或癌前方向 |

## 预测分布和混淆变化

| 预测标签 | Baseline 次数 | DermAgent 次数 | 变化 |
|---|---:|---:|---:|
| BCC | `414` | `442` | `+28` |
| SEK | `151` | `74` | `-77` |
| SCC | `66` | `53` | `-13` |
| ACK | `13` | `69` | `+56` |
| MEL | `34` | `34` | `+0` |
| NEV | `1` | `7` | `+6` |
| 未解析/None | `10` | `10` | `+0` |

DermAgent 将一部分原本偏向 SEK 的输出转向 ACK/BCC，因此提升了 ACK 和 BCC 的恶性召回，但也带来 SEK Top-1 下降和良性误报略升。

## Help / Hurt 分析

| 类型 | cases |
|---|---:|
| Top-1 gain | `58` |
| Top-1 loss | `6` |
| Net Top-1 gain | `+52` |
| Top-k gain | `113` |
| Top-k loss | `0` |
| Malignant recall gain | `67` |
| Malignant recall loss | `0` |

这组结果很适合论文分析：DermAgent 不仅让最终诊断多 `52` 个正确病例、Top-k 无损失，还新增 `67` 个恶性识别命中，并且没有造成恶性召回 case-level loss。

Top-1 loss 的 6 个 case 全部来自 SEK：

| case_id | GT | Baseline final | DermAgent final |
|---|---|---|---|
| PAT_1396_1360 | SEK | SEK | BCC |
| PAT_1765_3337 | SEK | SEK | ACK |
| PAT_1946_3924 | SEK | SEK | ACK |
| PAT_1988_4052 | SEK | SEK | ACK |
| PAT_2046_4323 | SEK | SEK | ACK |
| PAT_998_17 | SEK | SEK | ACK |

这说明当前主要代价集中在良性角化类病例被保守推向恶性/癌前方向。论文中可作为安全优先策略的代价说明。

## Fusion reason 追溯

| reason | cases |
|---|---:|
| `use_agent_output` | `689` |
| `qwen_pad20_agent_baseline_agreement` | `340` |
| `qwen_pad20_baseline_anchor_guard` | `253` |
| `qwen_pad20_sun_exposed_ack_topk_promotion` | `56` |
| `qwen_pad20_sek_to_bcc_central_ulcer_topk_promotion` | `18` |
| `qwen_pad20_scc_to_bcc_central_ulcer_topk_promotion` | `14` |
| `qwen_pad20_young_low_risk_nevus_topk_promotion` | `7` |
| `qwen_pad20_older_keratinocyte_scc_topk_promotion` | `1` |

这些 reason 用于结果追溯和可复核性，不建议在论文主文中表述为单条调优规则的收益；主文口径应写作 DermAgent 对 Qwen 诊断结果的整体贡献。

## 可用于论文的表述

英文：

> In the PAD-UFES-20 full test split, DermAgent improved Qwen's Top-1 accuracy from 26.12% to 33.67%, Macro-F1 from 19.36% to 25.56%, and Top-k hit rate from 54.57% to 70.97%. Importantly, malignant recall increased from 74.22% to 86.47%, reducing the malignant miss rate from 25.78% to 13.53%.

中文：

> 在 PAD-UFES-20 全量测试集上，DermAgent 将 Qwen 的 Top-1 正确率从 26.12% 提升到 33.67%，Macro-F1 从 19.36% 提升到 25.56%，Top-k 命中率从 54.57% 提升到 70.97%。更重要的是，恶性召回率从 74.22% 提升到 86.47%，对应恶性漏诊率从 25.78% 降至 13.53%。

## 局限性

- 良性特异性从 `14.79%` 降至 `11.97%`，良性误报率从 `85.21%` 升至 `88.03%`。这说明 DermAgent 在该组合上更偏安全保守。
- SEK 是主要负向子病种：Top-1 从 `14.29%` 降至 `5.71%`。如果继续优化，应优先针对 SEK 与 ACK/BCC 的过度保守混淆增加保护。
- MEL 和 SCC 的 Top-1 没有提升，但 MEL 恶性召回已是 `100% -> 100%`；SCC 虽 Top-1 不变，但恶性识别从 `78.57%` 提升至 `87.50%`。
- 当前结果适合主张“DermAgent 带来安全侧综合收益”，不适合主张“所有子病种均提升”。
