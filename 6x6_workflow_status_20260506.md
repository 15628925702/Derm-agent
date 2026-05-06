# 6x6 Workflow Status - 2026-05-06

This is the working map for tuning the 6 model x 6 dataset DermAgent matrix.

Priority order in the current architecture:

1. `model x dataset` workflow cell override
2. model workflow overlay
3. dataset workflow routing
4. default workflow

The qwen row is the only row intentionally tuned so far. The remaining 30 cells should be tuned one by one by adding explicit entries to `MODEL_DATASET_WORKFLOW_PROFILES` in `agent/model_workflow_router.py`.

## Dataset Workflow Routing

These are the dataset-level routes before any model overlay or model x dataset cell:

| Dataset | Label space | Dataset workflow |
|---|---|---|
| `ham10000` | `ham10000_full` | `sparse_lesion_workflow` |
| `isic2019` | `isic2019_full` | `image_archive_full_taxonomy_lesion_workflow` |
| `pad20` | `derm_six` | `clinical_full_taxonomy_lesion_workflow` |
| `scin` | `scin_grouped` | `family_routing_workflow` for diffuse/rash cases, otherwise `coarse_taxonomy_workflow` |
| `sd198` | `sd198_grouped` | `coarse_taxonomy_workflow` |
| `xiangya_sft` | `xiangya_sft_grouped` | `eczematous_family_routing_workflow` |

## Model Overlays

| Model | Current overlay | Tuning status |
|---|---|---|
| `qwen` | explicit qwen model x dataset cells for all 6 datasets | tuned/current-best row |
| `medgemma` | no explicit overlay; dataset workflow only | not tuned |
| `skinvl` | `direct_baseline_workflow`, no retrieval/specialists, conservative fallback | not tuned; only smoke-level evidence |
| `llama` | `conservative_archive_workflow`, conservative fusion | not tuned |
| `hulumed` | no explicit overlay; dataset workflow only | not tuned |
| `dermatollama` | no explicit overlay; dataset workflow only | not tuned |

## Qwen Row - Tuned Line

Current qwen row uses explicit model x dataset cells. For SCIN and SD198, the important fix was forcing grouped label spaces during qwen frozen-asset reruns so routing does not fall back to full taxonomy.

| Cell | Actual workflow | Latest evidence | Tuned? | Notes |
|---|---|---|---|---|
| `qwen x ham10000` | `sparse_lesion_workflow`, cell `qwen__ham10000__dataset_best` | 100 cases: top1 19.00 -> 20.00, topk 27.00 -> 39.00 | yes | Beats baseline; malignant recall tied at 97.73. |
| `qwen x isic2019` | dataset `image_archive_full_taxonomy_lesion_workflow` plus model profile `qwen_isic2019_archive_guard_workflow`, cell `qwen__isic2019__dataset_best` | 50 cases: top1 28.00 -> 32.00, topk 62.00 -> 70.00; 80-case rerun started at `outputs/qwen_isic2019_cell_tune_recheck80_20260506T092150Z` | provisional yes | New guard is winning on 50 cases; wait for 80-case result before freezing claim. |
| `qwen x pad20` | `clinical_full_taxonomy_lesion_workflow`, cell `qwen__pad20__dataset_best` | 80 cases: top1 27.50 -> 38.75, topk 52.50 -> 63.75 | yes | Accuracy wins, but malignant recall dropped 74.19 -> 59.68; later add PAD malignant guard. |
| `qwen x scin` | `family_routing_workflow` and `coarse_taxonomy_workflow`, cell `qwen__scin__grouped_best` | 80 cases: top1 47.92 -> 54.17, topk 47.92 -> 58.33 | yes | Fixed by preserving `scin_grouped`; actual distribution was 55 family + 25 coarse. |
| `qwen x sd198` | `coarse_taxonomy_workflow`, cell `qwen__sd198__grouped_best` | 80 cases: top1 45.00 -> 47.50, topk 62.50 -> 65.00 | yes | Fixed by preserving `sd198_grouped`; topk exceeds old best, top1 is 2 cases below old best. |
| `qwen x xiangya_sft` | `eczematous_family_routing_workflow`, cell `qwen__xiangya_sft__grouped_best` | 13 cases: top1 38.46 -> 46.15, topk 38.46 -> 61.54 | yes, small n | Keep as tuned but sample is only 13. |

## Remaining 30 Cells

The table below uses the latest matrix evidence from `outputs/6x6_64_100_8gpu_dynamic_20260505_175143` where available. "Dataset route + overlay" means no dedicated cell exists yet.

| Cell | Current workflow behavior | Latest evidence | Tuned? | Suggested next action |
|---|---|---|---|---|
| `medgemma x ham10000` | dataset route only | FAILED before compare | no | Fix model bootstrap/server first, then smoke 3-5 cases. |
| `medgemma x isic2019` | dataset route only | FAILED before compare | no | Fix bootstrap/server first. |
| `medgemma x pad20` | dataset route only | FAILED before compare | no | Fix bootstrap/server first. |
| `medgemma x scin` | dataset route only | FAILED before compare | no | Fix bootstrap/server first. |
| `medgemma x sd198` | dataset route only | FAILED before compare | no | Fix bootstrap/server first. |
| `medgemma x xiangya_sft` | dataset route only | 13 cases: top1 38.46 -> 30.77, topk 53.85 -> 53.85 | no | Add conservative cell or direct-baseline-preserving cell. |
| `skinvl x ham10000` | `direct_baseline_workflow` overlay | only 1-case smoke, no signal | no | Run 30-50 cases after server stability check. |
| `skinvl x isic2019` | `direct_baseline_workflow` overlay | only 1-case smoke, no signal | no | Keep low-context path; validate. |
| `skinvl x pad20` | `direct_baseline_workflow` overlay | only 1-case smoke, no signal | no | Validate; likely needs strong baseline preservation. |
| `skinvl x scin` | `direct_baseline_workflow` overlay | only 1-case smoke, no signal | no | Validate grouped routing separately. |
| `skinvl x sd198` | `direct_baseline_workflow` overlay | only 1-case smoke, no signal | no | Validate coarse taxonomy. |
| `skinvl x xiangya_sft` | `direct_baseline_workflow` overlay | only 1-case smoke, no signal | no | Validate small 13-case set. |
| `llama x ham10000` | dataset route + `conservative_archive_workflow` overlay | 100 cases: top1 16.00 -> 16.00, topk 17.00 -> 23.00 | partial | Top1 tie; tune only if top1 must win. |
| `llama x isic2019` | dataset route + `conservative_archive_workflow` overlay | no latest complete compare in selected matrix | no | Run smoke, then 30-50 cases. |
| `llama x pad20` | dataset route + `conservative_archive_workflow` overlay | no latest complete compare in selected matrix | no | Run smoke; watch malformed/empty final. |
| `llama x scin` | dataset route + `conservative_archive_workflow` overlay | 100 cases: top1 11.00 -> 11.00, topk 16.00 -> 16.00 | no | Needs grouped/family short workflow cell. |
| `llama x sd198` | dataset route + `conservative_archive_workflow` overlay | 100 cases: top1 45.00 -> 45.00, topk 54.00 -> 57.00 | partial | Top1 tie; tune coarse taxonomy cell. |
| `llama x xiangya_sft` | dataset route + `conservative_archive_workflow` overlay | no latest complete compare in selected matrix | no | Run 13-case validation. |
| `hulumed x ham10000` | dataset route only | 100 cases: top1 15.00 -> 15.00, topk 19.00 -> 35.00 | partial | Topk wins; add top1-focused conservative correction. |
| `hulumed x isic2019` | dataset route only | 100 cases: top1 41.00 -> 38.00, topk 71.00 -> 79.00 | no | Top1 regression; add baseline anchoring/nevus guard. |
| `hulumed x pad20` | dataset route only | 100 cases: top1 38.00 -> 37.00, topk 45.00 -> 75.00 | no | Top1 regression; protect baseline while keeping topk gains. |
| `hulumed x scin` | dataset route only | 100 cases: top1 45.00 -> 45.00, topk 49.00 -> 50.00 | partial | Top1 tie; small grouped workflow tweak. |
| `hulumed x sd198` | dataset route only | 100 cases: top1 45.00 -> 44.00, topk 52.00 -> 56.00 | no | Top1 regression; use coarse taxonomy conservative cell. |
| `hulumed x xiangya_sft` | dataset route only | 13 cases: top1 38.46 -> 46.15, topk 46.15 -> 61.54 | provisional | Looks good, but small n. |
| `dermatollama x ham10000` | dataset route only | 100 cases: top1 20.00 -> 19.00, topk 24.00 -> 33.00 | no | Top1 regression; preserve baseline unless high-confidence correction. |
| `dermatollama x isic2019` | dataset route only | no latest complete compare in selected matrix | no | Run smoke, then 30-50 cases. |
| `dermatollama x pad20` | dataset route only | 100 cases: top1 43.00 -> 19.00, topk 55.00 -> 54.00 | no | Severe regression; force conservative/direct workflow first. |
| `dermatollama x scin` | dataset route only | 100 cases: top1 22.00 -> 22.00, topk 22.00 -> 22.00 | no | No gain; try short grouped/family route. |
| `dermatollama x sd198` | dataset route only | 100 cases: top1 62.00 -> 62.00, topk 63.00 -> 63.00 | partial | Strong baseline, no agent gain; keep baseline-preserving cell. |
| `dermatollama x xiangya_sft` | dataset route only | no latest complete compare in selected matrix | no | Run 13-case validation. |

## How To Tune The Next Cell

For each bad cell:

1. Add or modify one explicit cell under `MODEL_DATASET_WORKFLOW_PROFILES`.
2. Change only workflow knobs first: allowed skills, disabled skills, retrieval, conservative fusion, fallback behavior, or model profile.
3. Run 3-5 case smoke with frozen evaluation and test writeback disabled.
4. If smoke passes, run 30-50 cases.
5. Promote to "tuned" in this document only when same-case agent top1 is better than baseline, or when an explicit tradeoff is accepted.
