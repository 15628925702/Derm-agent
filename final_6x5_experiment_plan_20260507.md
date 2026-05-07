# Final 6x5 Experiment Plan - 2026-05-07

本文档记录五个非 Xiangya 数据集的只读统计、分层抽样建议、6 模型 x 5 数据集最终实验目录和命令模板。当前只做统计与方案，不启动模型服务，不跑完整 6x5。

## 当前分支和工作区状态

- 当前分支：`test-1`
- 基线来源：从原 `skinvlOK` 工作区直接切出，保留既有未提交改动。
- 工作区状态：已有多处修改、删除和未跟踪文件；本轮只新增本文档。
- 关键约束：
  - 跳过 `xiangya_sft`。
  - 必须使用已调好的 `model x dataset` workflow cell。
  - compare 保持 strict frozen evaluation，不加 `--non-strict-frozen-eval`。
  - 不加 `--disable-model-workflow-routing`。
  - 不开启 test writeback；`compare_agent_vs_qwen.py` 的 evaluation protocol 已固定 `enable_writeback=False`。
  - 不默认开启 physician evidence summary。
  - 运行环境使用 `/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python`。

## 已阅读的项目入口

- `docs/项目索引.md`
- `docs/6x6工作流状态_20260506.md`
- `paper_data/README.md`
- `doctor_evidence_package/README.md`
- `scripts/compare_agent_vs_qwen.py`
- `scripts/export_paper_case_data.py`
- `agent/paper_exports.py`
- `agent/model_workflow_router.py`
- `agent/workflow_profiles.py`
- `dataio/case_loader.py`
- `dataio/ham10000_loader.py`
- `dataio/isic2019_loader.py`
- `dataio/scin_loader.py`
- `dataio/sd198_loader.py`

## 数据集规模和图片可用性

统计时间：2026-05-07。SCIN 的 `case_total` 指有 label 的可评估 case；`all_rows=5033`，其中 `labeled_rows=3061`。

| dataset | final label_space | case_total | image case available | image case missing | image refs existing | image refs missing |
|---|---:|---:|---:|---:|---:|---:|
| `ham10000` | `ham10000_full` | 10015 | 10015 | 0 | 10015 | 0 |
| `isic2019` | `isic2019_full` | 25331 | 25331 | 0 | 25331 | 0 |
| `pad20` | `derm_six` | 2298 | 2298 | 0 | 2298 | 0 |
| `scin` | `scin_grouped` | 3061 | 3061 | 0 | 6517 | 1 |
| `sd198` | `sd198_grouped` | 6584 | 6584 | 0 | 6584 | 0 |

SCIN 有 1 个额外图片引用缺失，但所有 3061 个有 label case 至少有 1 张真实存在图片，因此不影响 case-level 评估。

## Label 分布

### HAM10000

| label | n |
|---|---:|
| `nv` | 6705 |
| `mel` | 1113 |
| `bkl` | 1099 |
| `bcc` | 514 |
| `akiec` | 327 |
| `vasc` | 142 |
| `df` | 115 |

### ISIC2019

| label | n |
|---|---:|
| `NV` | 12875 |
| `MEL` | 4522 |
| `BCC` | 3323 |
| `BKL` | 2624 |
| `AK` | 867 |
| `SCC` | 628 |
| `VASC` | 253 |
| `DF` | 239 |

### PAD20

| label | n |
|---|---:|
| `BCC` | 845 |
| `ACK` | 730 |
| `NEV` | 244 |
| `SEK` | 235 |
| `SCC` | 192 |
| `MEL` | 52 |

### SCIN grouped

| grouped label | n |
|---|---:|
| `DERMATITIS_ECZEMA` | 1359 |
| `URTICARIA_BITE_FOLLICULITIS` | 519 |
| `OTHER` | 470 |
| `INFECTION_VIRAL_FUNGAL` | 319 |
| `VASCULAR_PURPURIC` | 161 |
| `ACNE_ROSACEA_FOLLICULAR` | 135 |
| `MALIGNANT_PREMALIGNANT` | 57 |
| `PIGMENT_KERATOSIS_NEVUS` | 41 |

SCIN raw label 高频项：`Eczema` 505，`Allergic Contact Dermatitis` 498，`Urticaria` 176，`Insect Bite` 154，`Folliculitis` 142，`Drug Rash` 71，`Herpes Simplex` 70，`Psoriasis` 70。最终 6x5 以 `scin_grouped` 作为统计和 compare 主 label space。

### SD198 grouped

| grouped label | n |
|---|---:|
| `DERMATITIS_ECZEMA` | 954 |
| `BENIGN_TUMOR_CYST` | 788 |
| `INFECTION_INFESTATION` | 731 |
| `PIGMENTARY_NEVUS_KERATOSIS` | 725 |
| `HAIR_NAIL_APPENDAGE` | 594 |
| `ACNE_FOLLICULITIS_ROSACEA` | 557 |
| `PAPULOSQUAMOUS_KERATOTIC` | 546 |
| `SUN_DAMAGE_ACTINIC` | 512 |
| `VASCULAR_ULCER_PURPURA` | 488 |
| `MALIGNANT_SKIN_CANCER` | 372 |
| `MUCOSAL_GENITAL_ORAL` | 208 |
| `OTHER` | 109 |

SD198 raw label 有 198 类，最终 workflow 使用 `sd198_grouped`，因此主抽样按 grouped label 分层；raw label 分布建议作为 manifest 附表保存。

## 建库集合和 Frozen Evaluation 集合划分建议

建议新增一个最终实验 manifest，而不是复用调参时的 offset。manifest 用显式 `train_case_indices` 和 `test_case_indices`，保证 memory-building set 与 frozen evaluation set 完全互斥，并且 case 顺序固定。

推荐规则：

- 抽样单位：case。
- 图片过滤：只纳入至少 1 张图片真实存在的 case。
- 分层字段：
  - `ham10000`: `dx`
  - `isic2019`: one-hot ground truth label
  - `pad20`: `diagnostic`
  - `scin`: `scin_grouped`
  - `sd198`: `sd198_grouped`
- memory-building set 只进入 `train` split。
- frozen evaluation set 只进入 `test` split。
- 两集合 case_id、case_index 必须无交集。
- 低频类优先保留 test 中最低代表数，再把剩余 case 分配给 memory。
- compare 阶段只读 frozen snapshot，不写回 test split。

建议每个 label 的最低 eval 覆盖：

| dataset | eval 每 label 最低数 | 低频类处理 |
|---|---:|---|
| `ham10000` | 10 | `df`、`vasc` 至少 10；其余按比例补齐。 |
| `isic2019` | 10 | `DF`、`VASC`、`SCC` 至少 10；其余按比例补齐。 |
| `pad20` | 10 | `MEL` 总数 52，建议 eval 10-12，不超过 25%，其余留给 memory。 |
| `scin` | 8 | `PIGMENT_KERATOSIS_NEVUS` 41、`MALIGNANT_PREMALIGNANT` 57，eval 各 8，memory 各至少 10。 |
| `sd198` | 10 | 按 grouped label，`OTHER` 109 也可 eval 10-12。 |

## 建议 case 数

这是成本和统计意义之间的折中版本：每个数据集都覆盖低频类，且总 compare case 数控制在 `6 x 630 = 3780` 个 model-dataset-case。

| dataset | memory-building set | frozen eval set | 6 模型总 eval case |
|---|---:|---:|---:|
| `ham10000` | 240 | 120 | 720 |
| `isic2019` | 300 | 150 | 900 |
| `pad20` | 200 | 120 | 720 |
| `scin` | 180 | 100 | 600 |
| `sd198` | 300 | 140 | 840 |
| **total** | **1220** | **630** | **3780** |

更省成本备选：每个数据集 eval 80，总计 `6 x 400 = 2400`；缺点是 SCIN/SD198 的低频 grouped label 置信区间更宽，论文表格里低频类 recall 会更不稳定。

更强统计备选：每个数据集 eval 160，总计 `6 x 800 = 4800`；成本更高，但分层 CI 更稳。

## 6x5 运行矩阵

| model key | served model name | default port | datasets |
|---|---|---:|---|
| `qwen` | `Qwen2.5-VL-7B-Instruct` | 8000 | `ham10000`, `isic2019`, `pad20`, `scin`, `sd198` |
| `dermatollama` | `DermatoLlama-full` | 8014 | `ham10000`, `isic2019`, `pad20`, `scin`, `sd198` |
| `medgemma` | `medgemma-4b-it` | 8010 | `ham10000`, `isic2019`, `pad20`, `scin`, `sd198` |
| `hulumed` | `Hulu-Med-7B` | 8013 | `ham10000`, `isic2019`, `pad20`, `scin`, `sd198` |
| `llama` | `Llama-3.2-11B-Vision-Instruct` | 8012 | `ham10000`, `isic2019`, `pad20`, `scin`, `sd198` |
| `skinvl` | `SkinVL-MM` | 8011 | `ham10000`, `isic2019`, `pad20`, `scin`, `sd198` |

这些组合在 `agent/model_workflow_router.py` 中都有显式 tuned cell。最终 compare 不应修改 workflow，不应关闭 model workflow routing。

## paper_data 输出目录设计

建议最终实验根目录：

```text
paper_data/final_6x5_experiment_20260507/
  manifests/
    final_6x5_sampling_manifest.json
    final_6x5_run_matrix.json
  splits/
    ham10000_final_stratified_split.json
    isic2019_final_stratified_split.json
    pad20_final_stratified_split.json
    scin_final_stratified_split.json
    sd198_final_stratified_split.json
  state/
    <dataset>/policy/
    <dataset>/split_states/
  compare_reports/
    <model>/<dataset>/
  summaries/
  run_logs/
```

Case-level paper export 目录按每个 model x dataset 单独命名：

```text
paper_data/case_level_exports/final_6x5_20260507__<model>__<dataset>__eval<N>/
```

建议 run name 示例：

```text
final_6x5_20260507__qwen__ham10000__eval120
final_6x5_20260507__skinvl__sd198__eval140
```

注意：当前 `paper_data/.gitignore` 只忽略了 `case_level_exports/**`，还没有忽略 `final_6x5_experiment_20260507/**`。如果最终结果文件很大，建议补一条 gitignore 或只把 manifest/summary 纳入 Git。

## Manifest 方案

建议新增一个小脚本生成 split/manifest，不改 workflow：

```text
scripts/build_final_6x5_manifest.py
```

脚本职责：

- 读取五个 dataset 的 metadata。
- 过滤无图 case。
- 按最终 label_space 分层。
- 生成互斥 memory/test case lists。
- 输出每个 dataset 的 split JSON，包含 `train`, `test`, `train_case_indices`, `test_case_indices`。
- 输出总 `final_6x5_sampling_manifest.json`，记录 seed、label 分布、case_id、case_index、image_path、split、stratum。
- 不写 memory，不跑模型，不改 workflow。

## Compare 命令模板

先设置公共变量：

```bash
cd /data/gh/DermAgent
PY=/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python
DATE_ID=20260507
FINAL_ROOT="paper_data/final_6x5_experiment_${DATE_ID}"
```

单个 model x dataset 模板：

```bash
MODEL_KEY=qwen
DATASET=ham10000
LIMIT=120
PORT=8000
MODEL_NAME="Qwen2.5-VL-7B-Instruct"
DATA_ROOT="data/ham10000"
SPLIT_JSON="${FINAL_ROOT}/splits/${DATASET}_final_stratified_split.json"
RUN_NAME="final_6x5_${DATE_ID}__${MODEL_KEY}__${DATASET}__eval${LIMIT}"

OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" \
OPENAI_API_KEY="EMPTY" \
OPENAI_MODEL="${MODEL_NAME}" \
DERMAGENT_POLICY_ROOT="${FINAL_ROOT}/state/${DATASET}/policy" \
DERMAGENT_SPLIT_STATE_ROOT="${FINAL_ROOT}/state/${DATASET}/split_states" \
"${PY}" scripts/compare_agent_vs_qwen.py \
  --data-root "${DATA_ROOT}" \
  --limit "${LIMIT}" \
  --case-offset 0 \
  --data-split test \
  --split-json "${SPLIT_JSON}" \
  --output-dir "${FINAL_ROOT}/compare_reports/${MODEL_KEY}/${DATASET}" \
  --policy-label "${RUN_NAME} frozen eval" \
  --export-paper-case-data \
  --paper-case-data-dir "paper_data/case_level_exports/${RUN_NAME}" \
  --client-timeout 120 \
  --client-max-retries 1
```

SCIN 和 SD198 需要显式环境变量，虽然 model workflow cell 也会设置；在命令层重复声明能让 split/stat/export 更清楚：

```bash
DERMAGENT_SCIN_LABEL_SPACE_ID=scin_grouped
DERMAGENT_SD198_LABEL_SPACE_ID=sd198_grouped
```

不要加入以下参数或环境：

```text
--disable-model-workflow-routing
--non-strict-frozen-eval
--enable-physician-evidence-summary
DERMAGENT_ENABLE_PHYSICIAN_EVIDENCE_SUMMARY=1
```

## Backfill Export 命令模板

如果 compare 已完成但需要重导 paper case-level 表：

```bash
"${PY}" scripts/export_paper_case_data.py \
  --compare-report-glob "${FINAL_ROOT}/compare_reports/<model>/<dataset>/**/compare_agent_vs_qwen_*.json" \
  --output-dir "paper_data/case_level_exports/final_6x5_${DATE_ID}__<model>__<dataset>__eval<N>" \
  --export-stem "final_6x5_${DATE_ID}__<model>__<dataset>__case_level"
```

## 服务启动前检查

如果需要启动或复用模型服务，先执行只读检查：

```bash
ss -ltnp | grep -E ':(8000|8010|8011|8012|8013|8014|8022|8023|8024|8025|8026)\\b' || true
nvidia-smi
ps -eo pid,ppid,stat,etime,cmd | grep -E 'serve_transformers_openai|start_.*server|vllm|python' | grep -v grep
```

若要停服务，必须先列出将停止的 PID、端口、命令行和理由，确认后再停；不能误停正在运行的大实验。

## Physician Evidence Package

默认最终 6x5 诊断实验不需要对每个 case 开启 physician evidence summary。原因：

- 它额外调用独立 Qwen summary 服务，显著增加成本和服务依赖。
- 它不会改变 final diagnosis。
- 论文主表需要的是 case-level Excel/CSV/JSON/JSONL 导出，`--export-paper-case-data` 已满足。

如需补充材料示例，可以后续只选少量代表 case 单独开启 `--enable-physician-evidence-summary`，不要在完整 6x5 默认开启。

## 风险点

- SCIN 原始 label 极碎，主实验必须固定为 `scin_grouped`；不要用 raw full label 做最终比较主表。
- SD198 原始 198 类很碎，最终 workflow 已按 `sd198_grouped` 调好；主表按 grouped label 统计更合理。
- 自定义 final split manifest 需要和 loader index 精确对齐；生成后必须 dry-run 校验 case_id 顺序。
- `paper_data/final_6x5_experiment_20260507/` 目前不在 `.gitignore` 规则内，完整 compare debug 可能很大。
- 如果 isolated `DERMAGENT_POLICY_ROOT` / `DERMAGENT_SPLIT_STATE_ROOT` 未初始化，strict frozen eval 会失败；应先初始化 dataset-level state，再 compare。
- 不要把 memory-building set 和 test eval set 混用；任何训练/建库只允许发生在 train/memory split。
- 不要使用 tuning 过程中的 40-case offset 当最终论文主结果，除非明确把它定义成 pilot 或 sensitivity analysis。

## 需要确认的问题

1. 是否采用推荐主方案：eval 共 630 cases，6 模型总计 3780 model-dataset-case？
2. 是否允许我新增 `scripts/build_final_6x5_manifest.py` 来生成 final split/manifest？
3. memory-building set 是否真的需要重新建库，还是复用当前 frozen state，仅把 memory set 作为论文方法中的隔离训练/经验候选集合？
4. `paper_data/final_6x5_experiment_20260507/` 是否要加入 `.gitignore`，只保留 manifest/summary 可追踪？
5. 服务端口是否按现有矩阵复用：qwen 8000、medgemma 8010、skinvl 8011、llama 8012、hulumed 8013、dermatollama 8014？
