# DermAgent

DermAgent 是一个面向皮肤科图像诊断的结构化智能体推理框架。它不替代底层视觉语言模型，而是在冻结 backbone 的前提下，把最终诊断前的临床推理组织成可审计的 workflow、技能库、经验库、认知状态、保守融合策略和论文级 case 导出。

当前主问题是：

> 在同一个视觉语言模型 backbone 固定不变、同一批 case frozen evaluation 的条件下，结构化、经验驱动的 agent reasoning 是否能优于 direct baseline prompting？

## 当前状态

截至 2026-05-07，项目已经从早期的单一 Qwen 路线更新为 `6 models x 6 datasets` 的 workflow cell 路线：

- 模型：`qwen`、`dermatollama`、`medgemma`、`hulumed`、`llama`、`skinvl`
- 数据集：`ham10000`、`isic2019`、`pad20`、`scin`、`sd198`、`xiangya_sft`
- 自动调参和最终大实验默认跳过 `xiangya_sft`，除非用户明确点名
- 非 Xiangya 的 5 个数据集是当前论文最终实验的主范围
- 运行时优先使用 `model x dataset` 显式 workflow cell，其次才是 model overlay、dataset workflow 和 default workflow
- 已调好的 cell 默认不要再改；新实验应该复用现有 workflow，不重新调参

详细工作流地图见 [`docs/6x6工作流状态_20260506.md`](docs/6x6工作流状态_20260506.md)。

## 核心边界

DermAgent 的边界必须保持清楚：

- Direct baseline 和 DermAgent 使用同一个底层模型服务。
- Agent 只组织证据、检索经验、执行技能、审计风险和矛盾，并生成结构化证据包。
- 最终诊断责任仍保留给 backbone，不把 agent 中间结论无条件当作最终答案。
- 大多数 tuned cell 使用 conservative fusion、baseline anchor、narrow override 或 malformed fallback，防止 agent final 直接污染结果。
- Frozen evaluation 阶段禁止 test writeback；经验建库必须和 frozen evaluation case 互斥。

因此 DermAgent 不是分类器集合，也不是多模型投票，而是最终诊断前的 structured reasoning scaffold。

## 推理链路

典型一次 DermAgent case 执行包含：

1. Backbone 生成 direct baseline 或初始视觉诊断信息。
2. Workflow router 根据 `model x dataset` cell 选择 workflow profile、label space、技能开关和融合策略。
3. Agent 执行技能选择、经验检索、证据组织、风险审计、矛盾审计和信息缺口分析。
4. Backbone 读取结构化证据包后生成 agent-side final diagnosis。
5. Conservative fusion 根据当前 cell 的规则决定是否接受 agent 输出、回退 baseline、或执行非常窄的 guarded override。
6. Compare runner 在同一批 frozen case 上计算 direct baseline 与 agent 指标，并可导出 paper-facing CSV/JSON/JSONL/XLSX。

核心实现位置：

- [`agent/model_workflow_router.py`](agent/model_workflow_router.py): `MODEL_DATASET_WORKFLOW_PROFILES` 和模型 overlay
- [`agent/workflow_profiles.py`](agent/workflow_profiles.py): dataset workflow routing
- [`agent/conservative_fusion.py`](agent/conservative_fusion.py): baseline anchor、narrow override、malformed fallback
- [`agent/evaluation_protocol.py`](agent/evaluation_protocol.py): split/writeback contamination guard
- [`scripts/compare_agent_vs_qwen.py`](scripts/compare_agent_vs_qwen.py): frozen baseline-vs-agent compare 入口
- [`agent/paper_exports.py`](agent/paper_exports.py): paper-facing case-level 导出

`compare_agent_vs_qwen.py` 的文件名是历史名称；当前它通过 OpenAI-compatible endpoint 和 `OPENAI_MODEL` 支持 Qwen、DermatoLlama、MedGemma、Hulu-Med、Llama、SkinVL 等模型。

## 数据集与默认 Label Space

| 数据集 | 当前 label space | 默认 dataset workflow | 当前用途 |
|---|---|---|---|
| `ham10000` | `ham10000_full` | `sparse_lesion_workflow` | 非 Xiangya 最终实验候选 |
| `isic2019` | `isic2019_full` | `image_archive_full_taxonomy_lesion_workflow` | 非 Xiangya 最终实验候选 |
| `pad20` | `derm_six` | `clinical_full_taxonomy_lesion_workflow` | 非 Xiangya 最终实验候选 |
| `scin` | `scin_grouped` | `family_routing_workflow` / `coarse_taxonomy_workflow` | 非 Xiangya 最终实验候选 |
| `sd198` | `sd198_grouped` | `coarse_taxonomy_workflow` | 非 Xiangya 最终实验候选 |
| `xiangya_sft` | `xiangya_sft_grouped` | `eczematous_family_routing_workflow` | 默认跳过，除非明确要求 |

本地数据通常位于 [`data/`](data/) 下；loader 和 schema 对齐代码位于 [`dataio/`](dataio/)。

## 环境

当前推荐环境是已经配置好的 conda 环境：

```bash
cd /data/gh/DermAgent
conda activate dermagent-6x6
```

在自动化脚本和长跑实验中，优先使用绝对 Python：

```bash
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python
```

旧文档中的 `/root/DermAgent`、`derm-qwen`、单 Qwen final round 口径已经过时。当前工作目录以 `/data/gh/DermAgent` 为准。

## 模型服务

模型服务都走 OpenAI-compatible API。常用启动脚本：

```bash
cd /data/gh/DermAgent
bash scripts/start_qwen_server.sh
bash scripts/start_dermatollama_server.sh
bash scripts/start_medgemma_server.sh
bash scripts/start_hulumed_server.sh
bash scripts/start_llama_server.sh
bash scripts/start_skinvl_server.sh
```

启动或重启服务前必须先检查端口和 GPU 占用，避免误停正在运行的大实验：

```bash
ss -ltnp | grep -E ':(8000|8100|8101|8102|8103|8104|8105|8106|8107|8108|8200)\b' || true
nvidia-smi
ps -eo pid,ppid,stat,etime,cmd | grep -E 'vllm|serve_skinvl|serve_transformers|start_.*server' | grep -v grep
```

如果确实需要停服务，先明确列出将停的 PID，再执行 `kill`。不要用会清空整机任务的粗暴命令。

## Frozen Compare

标准 compare 入口：

```bash
cd /data/gh/DermAgent

DERMAGENT_POLICY_ROOT=paper_data/<run>/state/<model>/<dataset>/policy \
DERMAGENT_SPLIT_STATE_ROOT=paper_data/<run>/state/<model>/<dataset>/split_states \
OPENAI_BASE_URL=http://127.0.0.1:<port>/v1 \
OPENAI_API_KEY=EMPTY \
OPENAI_MODEL=<served_model_name> \
OPENAI_TIMEOUT=120 \
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python scripts/compare_agent_vs_qwen.py \
  --data-root data/<dataset_root> \
  --limit <eval_case_count> \
  --case-offset 0 \
  --data-split test \
  --split-json paper_data/<run>/splits/<dataset>_<run>_split.json \
  --output-dir paper_data/<run>/compare_reports/<model>/<dataset> \
  --policy-label "<run> frozen eval" \
  --export-paper-case-data \
  --paper-case-data-dir paper_data/case_level_exports/<run>__<model>__<dataset> \
  --client-timeout 120 \
  --client-max-retries 1
```

Frozen evaluation 的默认规则：

- 使用固定 `--split-json`
- evaluation 用 `--data-split test`
- 不打开 `--enable-writeback`
- 不在 test split 写经验、技能或认知状态
- case-level paper export 必须打开
- physician evidence summary 默认不打开，除非实验目的明确需要医生可读证据包

## Memory Building

经验建库必须只在 memory-building split 上进行，不能和 frozen eval case 重叠。现有 bootstrap 脚本包括：

```bash
bash scripts/bootstrap_ham10000_train_cases.sh
bash scripts/bootstrap_isic2019_train_cases.sh
bash scripts/bootstrap_pad20_train_cases.sh
bash scripts/bootstrap_scin_train_cases.sh
bash scripts/bootstrap_sd198_train_cases.sh
bash scripts/bootstrap_xiangya_sft_train_cases.sh
```

典型流程是：

1. 用固定 split JSON 选择 train/memory-building cases。
2. Bootstrap 阶段允许 train split writeback 到本次实验隔离的 `policy_root` 和 `split_state_root`。
3. 用 [`scripts/manage_dataset_experiment_assets.py`](scripts/manage_dataset_experiment_assets.py) 将 train 状态 promote 到 val/test 的只读起点。
4. Frozen compare 阶段关闭 writeback，只读使用已 promoted 的 state。

## Paper Data

论文数据统一放在 [`paper_data/`](paper_data/) 下：

- `paper_data/case_level_exports/`: compare 时导出的 paper-facing case-level CSV/JSON/JSONL/XLSX
- `paper_data/final_3x3_delta_pilot_20260507/`: 当前 3 model x 3 dataset delta pilot 的 manifest、split、state、compare reports、logs
- `paper_data/final_6x5_experiment_YYYYMMDD/`: 未来完整 6 model x 5 non-Xiangya 实验建议目录格式

导出说明见 [`paper_data/README.md`](paper_data/README.md)。从已有 compare report 回填导出：

```bash
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python scripts/export_paper_case_data.py \
  --compare-report-glob 'outputs/<run_dir>/**/compare_agent_vs_qwen_*.json' \
  --output-dir paper_data/case_level_exports/<run_name> \
  --export-stem <run_name>_case_level
```

## 当前 3x3 Delta Pilot

为论文最终实验预跑，当前采用“agent 相对 direct baseline 差值大”的组合：

- 模型：`llama`、`medgemma`、`skinvl`
- 数据集：`scin`、`pad20`、`isic2019`
- Eval cases：每个组合 300
- Memory-building cases：`scin=100`、`pad20=100`、`isic2019=120`
- 输出根目录：`paper_data/final_3x3_delta_pilot_20260507/`
- Case export 根目录：`paper_data/case_level_exports/`

监控命令：

```bash
tail -f paper_data/final_3x3_delta_pilot_20260507/run_logs/nohup_runner_20260507T085218Z.log
cat paper_data/final_3x3_delta_pilot_20260507/run_summary.tsv
watch -n 30 nvidia-smi
```

这只是当前实验实例，不应被当作唯一固定实验入口。后续完整 6x5 仍应建立新的 dated run folder 和 manifest。

## Doctor Evidence Package

DermAgent 可以额外生成 doctor-facing evidence package，把 agent 中间证据整理成医生可读摘要。该功能默认关闭，不改变 direct baseline、agent final diagnosis 或 compare 评估逻辑。

医生摘要使用单独 Qwen physician-summary 服务，默认监听 `http://127.0.0.1:8200/v1`：

```bash
cd /data/gh/DermAgent
bash scripts/start_qwen_physician_summary_server.sh
```

单次 compare 开启：

```bash
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python scripts/compare_agent_vs_qwen.py \
  --data-root /path/to/dataset \
  --limit 8 \
  --enable-physician-evidence-summary \
  --physician-evidence-summary-base-url http://127.0.0.1:8200/v1 \
  --physician-evidence-summary-model Qwen2.5-VL-7B-Instruct \
  --physician-evidence-summary-detail detailed
```

完整说明见 [`doctor_evidence_package/README.md`](doctor_evidence_package/README.md)。

## Workflow Evolution 与 Skill 审核

DermAgent 支持离线 workflow evolution 和 skill refinement，但这些机制是 proposal-first，不会自动上线：

- workflow proposal 默认写入 `proposals/workflow_evolution/`，状态为 pending review
- approved proposal 只会复制到 `state/workflow_evolution/approved/`
- runtime 必须显式设置 `DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1` 才会读取 approved proposal
- 自动生成的新 skill 先写入 `skills/pending/`
- 只有人工运行 `scripts/approve_skill.py` 后，pending skill 才会进入正式 `skills/` 并注册

完整说明见 [`workflow_evolution/自进化Workflow说明.md`](workflow_evolution/自进化Workflow说明.md)。

## 仓库结构

| 路径 | 用途 |
|---|---|
| [`agent/`](agent/) | 主推理链路、workflow routing、fusion、evaluation、paper export |
| [`skills/`](skills/) | 原子化临床推理技能和 pending skill 草稿 |
| [`memory/`](memory/) | raw case memory、tactical experience、abstract experience |
| [`cognition/`](cognition/) | 跨病例认知状态 |
| [`dataio/`](dataio/) | 数据集 loader 和 schema 对齐 |
| [`scripts/`](scripts/) | 当前活跃实验脚本、服务启动、bootstrap、compare、导出 |
| [`paper_data/`](paper_data/) | 论文级 case 导出、manifest、final experiment 数据 |
| [`doctor_evidence_package/`](doctor_evidence_package/) | 医生可读证据包说明和示例规范 |
| [`workflow_evolution/`](workflow_evolution/) | 离线 workflow proposal 和 skill refinement |
| [`docs/`](docs/) | 项目索引、6x6 workflow 状态、架构和历史文档 |
| [`tests/`](tests/) | 单测与协议测试 |
| [`final-script/`](final-script/) | 旧 final round 包装入口，保留用于复现旧实验线 |
| [`final-score/`](final-score/) | 旧 final round 分数和导出区 |

项目导航见 [`docs/项目索引.md`](docs/项目索引.md)。

## 测试

常用测试：

```bash
cd /data/gh/DermAgent
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python -m pytest -q tests/test_final_round_scripts.py
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python -m pytest -q tests/test_xiangya_sft_loader.py tests/test_openai_client_scin_prompting.py
```

修改 workflow routing、fusion、export 或 split/writeback guard 后，应优先补跑相关测试或做小规模 dry-run。

## 当前推荐口径

推荐表述：

- DermAgent 是统一 core + model x dataset workflow cells 的结构化推理框架。
- 当前论文最终实验重点是 5 个非 Xiangya 数据集上的 frozen evaluation。
- 已调好的 cell 复用既有 workflow，不在最终实验里重新调参。
- Case-level CSV/JSON/JSONL/XLSX 导出是论文数据准备的默认要求。
- Doctor evidence package 是额外医生可读出口，默认不参与诊断大实验。

不推荐表述：

- 当前默认 backbone 只有 Qwen。
- 运行时只是按 dataset name 开 special switch。
- 所有模型/数据集都强正向。
- Test split 可以用于经验建库或 writeback。
- Agent final 可以绕过 conservative fusion 直接生效。
