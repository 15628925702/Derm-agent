# DermAgent 数据集来源与疾病分布

统计日期：2026-05-07

本文档整理当前 DermAgent workflow / split / compare 路径中实际用到的数据集。统计直接来自本地 `data/` 元数据与当前 loader / label space 规则；其中 SCIN、SD-198、Xiangya SFT 按 DermAgent 当前实验使用的 grouped label space 汇总。

## 总览

| 数据集 key | 本地元数据 | 当前 label space | 病例 / 图像数 | 当前用途 |
|---|---|---|---:|---|
| `ham10000` | `data/ham10000/HAM10000_metadata.csv` | `ham10000_full` | 10015 | 非 Xiangya 主实验候选 |
| `isic2019` | `data/isic2019/ISIC_2019_Training_GroundTruth.csv` + metadata | `isic2019_full` | 25331 | 非 Xiangya 主实验候选 |
| `pad20` / `pad_ufes_20` | `data/pad_ufes_20/metadata.csv` | `derm_six` | 2298 | 非 Xiangya 主实验候选 |
| `scin` | `data/scin/official_mirror/scin_cases.csv` + labels | `scin_grouped` | 3061 labeled cases | 非 Xiangya 主实验候选 |
| `sd198` | `data/sd198/sd-198/images.txt` + class labels | `sd198_grouped` | 6584 | 非 Xiangya 主实验候选 |
| `xiangya_sft` | `data/sft数据/skin_xiangya.jsonl` | `xiangya_sft_grouped` | 72 | 6x6 路由中保留；最终大实验默认跳过 |

说明：`data/dermnet/` 当前在仓库中存在，但 `dataio/数据集接入设计.md` 明确标为暂缓接入，不属于当前 DermAgent 主流程使用的数据集。

## 来源

| 数据集 | 来源 / 引用 | 本地使用说明 |
|---|---|---|
| HAM10000 | Tschandl, Rosendahl, Kittler, *Scientific Data* 2018, DOI: https://doi.org/10.1038/sdata.2018.161；公开于 ISIC Archive | 使用 HAM10000 metadata 中的 `dx` 七分类 |
| ISIC 2019 | ISIC Challenge 2019 training set：https://challenge.isic-archive.com/data/；本地 `ATTRIBUTION.txt` 说明其聚合来源包括 BCN_20000、HAM10000、MSK | 使用官方 one-hot ground truth 八分类；`UNK` 在训练 ground truth 中为 0 |
| PAD-UFES-20 | Pacheco et al., *Data in Brief* 2020；Mendeley Data DOI: https://doi.org/10.17632/zr7vgbcyr2.1；PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/ | 使用 `diagnostic` 六分类，即 DermAgent 的 `derm_six` |
| SCIN | Google Research / Stanford Medicine SCIN：GitHub https://github.com/google-research-datasets/scin；HF mirror https://huggingface.co/datasets/google/scin | 使用 `weighted_skin_condition_label` 解析出的最高权重皮肤病标签，再映射到 `scin_grouped` |
| SD-198 | Sun et al., *A Benchmark for Automatic Visual Classification of Clinical Skin Disease Images*, ECCV 2016, DOI: https://doi.org/10.1007/978-3-319-46466-4_13；论文 PDF: https://xiaoxiaosun.com/docs/2016-eccv-sd198.pdf | 原始 198 类太碎，当前 workflow 使用 `sd198_grouped` |
| Xiangya SFT | 本地/内部整理的湘雅皮肤科 SFT 数据：`data/sft数据/skin_xiangya.jsonl`；未在仓库中发现公开外部来源记录 | 从 assistant 文本中抽取诊断，再映射到 `xiangya_sft_grouped` |

## HAM10000 分布

当前统计字段：`dx`；总数 `10015`。

| label | n | pct |
|---|---:|---:|
| `nv` | 6705 | 66.95% |
| `mel` | 1113 | 11.11% |
| `bkl` | 1099 | 10.97% |
| `bcc` | 514 | 5.13% |
| `akiec` | 327 | 3.27% |
| `vasc` | 142 | 1.42% |
| `df` | 115 | 1.15% |

标签含义：`nv` melanocytic nevi，`mel` melanoma，`bkl` benign keratosis-like lesions，`bcc` basal cell carcinoma，`akiec` actinic keratoses / intraepithelial carcinoma，`vasc` vascular lesions，`df` dermatofibroma。

## ISIC2019 分布

当前统计字段：`ISIC_2019_Training_GroundTruth.csv` one-hot label；总数 `25331`。

| label | n | pct |
|---|---:|---:|
| `NV` | 12875 | 50.83% |
| `MEL` | 4522 | 17.85% |
| `BCC` | 3323 | 13.12% |
| `BKL` | 2624 | 10.36% |
| `AK` | 867 | 3.42% |
| `SCC` | 628 | 2.48% |
| `VASC` | 253 | 1.00% |
| `DF` | 239 | 0.94% |

`UNK` 是 ISIC2019 label space 的 outlier / unknown 位置，但本地 training ground truth 中未出现正例。

## PAD-UFES-20 / PAD20 分布

当前统计字段：`diagnostic`；总数 `2298`。

| label | n | pct |
|---|---:|---:|
| `BCC` | 845 | 36.77% |
| `ACK` | 730 | 31.77% |
| `NEV` | 244 | 10.62% |
| `SEK` | 235 | 10.23% |
| `SCC` | 192 | 8.36% |
| `MEL` | 52 | 2.26% |

当前 DermAgent 将 PAD20 作为 `derm_six` 六分类 clinical image 数据集。

## SCIN 分布

当前统计口径：仅统计有可解析标签的 labeled cases；`all_rows=5033`，`labeled_rows=3061`。最终实验使用 `scin_grouped`。

| grouped label | n | pct |
|---|---:|---:|
| `DERMATITIS_ECZEMA` | 1359 | 44.40% |
| `URTICARIA_BITE_FOLLICULITIS` | 519 | 16.96% |
| `OTHER` | 470 | 15.35% |
| `INFECTION_VIRAL_FUNGAL` | 319 | 10.42% |
| `VASCULAR_PURPURIC` | 161 | 5.26% |
| `ACNE_ROSACEA_FOLLICULAR` | 135 | 4.41% |
| `MALIGNANT_PREMALIGNANT` | 57 | 1.86% |
| `PIGMENT_KERATOSIS_NEVUS` | 41 | 1.34% |

raw label 高频项用于理解来源分布，不作为最终主表 label space：`Eczema` 505，`Allergic Contact Dermatitis` 498，`Urticaria` 176，`Insect Bite` 154，`Folliculitis` 142，`Drug Rash` 71，`Herpes Simplex` 70，`Psoriasis` 70。

## SD-198 分布

当前统计口径：原始 198 类经 `sd198_grouped` 映射后的 grouped label；总数 `6584`。

| grouped label | n | pct |
|---|---:|---:|
| `DERMATITIS_ECZEMA` | 954 | 14.49% |
| `BENIGN_TUMOR_CYST` | 788 | 11.97% |
| `INFECTION_INFESTATION` | 731 | 11.10% |
| `PIGMENTARY_NEVUS_KERATOSIS` | 725 | 11.01% |
| `HAIR_NAIL_APPENDAGE` | 594 | 9.02% |
| `ACNE_FOLLICULITIS_ROSACEA` | 557 | 8.46% |
| `PAPULOSQUAMOUS_KERATOTIC` | 546 | 8.29% |
| `SUN_DAMAGE_ACTINIC` | 512 | 7.78% |
| `VASCULAR_ULCER_PURPURA` | 488 | 7.41% |
| `MALIGNANT_SKIN_CANCER` | 372 | 5.65% |
| `MUCOSAL_GENITAL_ORAL` | 208 | 3.16% |
| `OTHER` | 109 | 1.66% |

SD-198 原始标签有 198 类，当前不作为 DermAgent 主评测 label space。原始分布大体上每类上限约 60 张，但经过去重/过滤后不同类别数量不完全相同。

## Xiangya SFT 分布

当前统计口径：`xiangya_sft_loader.py` 从中文 assistant 回复中抽取 raw diagnosis，并 canonicalize 到 `xiangya_sft_grouped`；总数 `72`。

| grouped label | n | pct |
|---|---:|---:|
| `ATOPIC_DERMATITIS` | 32 | 44.44% |
| `CONTACT_DERMATITIS` | 27 | 37.50% |
| `ECZEMA_DERMATITIS` | 9 | 12.50% |
| `HAIR_DISORDER` | 1 | 1.39% |
| `HERPETIC_ECZEMA` | 1 | 1.39% |
| `OTHER_INFLAMMATORY` | 1 | 1.39% |
| `PERIORAL_DERMATITIS` | 1 | 1.39% |

raw diagnosis 高频项：`特应性皮炎` 25，`接触性皮炎` 21，未解析 7，`湿疹并感染` 3，`湿疹并继发感染` 3，`传染性湿疹样皮炎` 2。其余 raw diagnosis 每类 1 条。

## 统计复现

主要依据：

- 数据集清单：`README.md`、`configs/dataset_splits.py`、`dataio/case_loader.py`
- label space：`agent/label_space.py`、`agent/sd198_label_catalog.py`
- 统计校验：`final_6x5_experiment_plan_20260507.md` 中的 5 个非 Xiangya 数据集统计，以及本次对本地元数据的重新计数

