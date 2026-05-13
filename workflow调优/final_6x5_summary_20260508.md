# Final 6x5 Experiment Summary (2026-05-08)

## Scope

- Models: `qwen`, `dermatollama`, `medgemma`, `hulumed`, `llama`, `skinvl`
- Datasets: `ham10000`, `isic2019`, `pad20`, `scin`, `sd198`
- Xiangya was excluded by design.
- Evaluation scale: `300` frozen evaluation cases per completed combo.

## Overall Status

- Total planned combos: `30`
- Fully completed combos: `30`
- Incomplete combos: `0`
- Active `compare_agent_vs_qwen.py` processes at final summary time: `0`

## Post-experiment Tuning Architecture

后续单 workflow 调优采用 memory-side fusion experience 方式：

- workflow-specific final fusion / promotion / override / fallback 规则写入 `memory/fusion_experience/workflow_fusion_decision.py`
- `skills/workflow_fusion_decision.py` 与 `agent/conservative_fusion.py` 仅保留兼容 wrapper，不再新增 workflow-specific 主逻辑
- 每条规则必须精确绑定 `workflow_cell_id`，并补 `tests/test_conservative_fusion.py`

迁移验证已在 `workflow-fusion-skill-migration` 分支完成：旧实现与新 skill 在既有 report `1876` cases 和 live 8-workflow `192` cases 上逐 case fusion decision 对比均为 `mismatch=0`。

## Late-stage Acceleration Note

`llama/ham10000` and `llama/sd198` were finished on `2026-05-08` by preserving existing partial compare records, sharding the remaining cases across 8 GPUs, and then merging the shard outputs back into the canonical compare result.

`medgemma/ham10000` was also closed on `2026-05-08` using an 8-GPU sharded frozen compare rerun against the already-promoted frozen state. Two shard attempts hit a deterministic baseline JSON parse failure on the same case; the successful `r3` rerun used a parser fallback that preserves `raw_text` for diagnosis requests only when JSON repair fails repeatedly.

Relevant artifacts:

- `paper_data/final_6x5_continuous_20260507/llama_tail_accel_20260508/`
- `scripts/merge_compare_partial_and_shards.py`

## Completed 6x5 Matrix

| Model | ham10000 | isic2019 | pad20 | scin | sd198 |
|---|---:|---:|---:|---:|---:|
| `qwen` | complete | complete | complete | complete | complete |
| `dermatollama` | complete | complete | complete | complete | complete |
| `medgemma` | complete | complete | complete | complete | complete |
| `hulumed` | complete | complete | complete | complete | complete |
| `llama` | complete | complete | complete | complete | complete |
| `skinvl` | complete | complete | complete | complete | complete |

## Best Top-1 Gains

| Rank | Model | Dataset | Baseline Top-1 | Agent Top-1 | Delta |
|---|---|---|---:|---:|---:|
| 1 | `medgemma` | `pad20` | 0.0800 | 0.3300 | `+0.2500` |
| 2 | `skinvl` | `scin` | 0.0400 | 0.2533 | `+0.2133` |
| 3 | `llama` | `scin` | 0.1000 | 0.2200 | `+0.1200` |
| 4 | `skinvl` | `pad20` | 0.1067 | 0.2133 | `+0.1067` |
| 5 | `hulumed` | `pad20` | 0.3967 | 0.4800 | `+0.0833` |
| 6 | `skinvl` | `sd198` | 0.4133 | 0.4967 | `+0.0833` |
| 7 | `hulumed` | `sd198` | 0.4233 | 0.4567 | `+0.0333` |
| 8 | `skinvl` | `ham10000` | 0.1167 | 0.1500 | `+0.0333` |

## Positive / Flat / Negative Count

| Bucket | Count |
|---|---:|
| Positive (`delta_top1 > 0`) | `19` |
| Flat (`delta_top1 = 0`) | `8` |
| Negative (`delta_top1 < 0`) | `3` |

## Worst / Flat Top-1 Outcomes

| Model | Dataset | Delta |
|---|---|---:|
| `llama` | `isic2019` | `-0.0067` |
| `medgemma` | `isic2019` | `-0.0033` |
| `qwen` | `isic2019` | `-0.0033` |
| `medgemma` | `ham10000` | `0.0000` |
| many others | multiple | `0.0000` to small positive |

Interpretation: the main strong-gain region remained concentrated in `pad20` and `scin`, while `isic2019` was mostly flat across models.

## Average Delta by Model

| Model | Completed combos | Avg Top-1 delta | Avg Top-k delta | Avg malignant recall delta |
|---|---:|---:|---:|---:|
| `skinvl` | 5 | `+0.0900` | `+0.1160` | `-0.0371` |
| `medgemma` | 5 | `+0.0513` | `+0.0573` | `-0.0008` |
| `llama` | 5 | `+0.0307` | `+0.0397` | `+0.0018` |
| `hulumed` | 5 | `+0.0260` | `+0.1313` | `-0.0042` |
| `dermatollama` | 5 | `+0.0020` | `+0.0227` | `-0.0034` |
| `qwen` | 5 | `+0.0013` | `+0.0573` | `-0.0042` |

Note: `llama` averages here include the two accelerated completions (`ham10000`, `sd198`) and therefore differ from earlier mid-run estimates.

## Average Delta by Dataset

| Dataset | Completed combos | Avg Top-1 delta | Avg Top-k delta | Avg malignant recall delta |
|---|---:|---:|---:|---:|
| `pad20` | 6 | `+0.0783` | `+0.1422` | `-0.0195` |
| `scin` | 6 | `+0.0567` | `+0.0578` | `-0.0385` |
| `sd198` | 6 | `+0.0233` | `+0.0322` | `+0.0111` |
| `ham10000` | 6 | `+0.0094` | `+0.0800` | `+0.0054` |
| `isic2019` | 6 | `+0.0000` | `+0.0394` | `+0.0015` |

## Output Layout

Primary experiment outputs are spread across three run roots:

- `paper_data/final_3x3_delta_pilot_20260507/`
- `paper_data/final_3x3_delta_pilot_20260507_extra_5jobs/`
- `paper_data/final_6x5_continuous_20260507/`

Case-level paper-facing exports live under:

- `paper_data/case_level_exports/`

The latest summary document is this file:

- `paper_data/final_6x5_summary_20260508.md`

Everything is now in a paper-usable completed state.
