# DermAgent

DermAgent 是一个面向皮肤科推理的智能体系统。系统始终保持 `Qwen` 作为唯一最终诊断者，在此基础上增加结构化 skill bank、分层 experience bank、cognition state、evidence aggregation、reflection/writeback、可学习 controller 钩子，以及冻结的论文级评测协议。

项目长期遵守以下原则：

- `Qwen` 始终是唯一最终诊断模型。
- `skill`、`experience`、`cognition` 严格分层、分库。
- reflection / writeback / policy evolution 必须有结构化 evidence 支撑，且可审计。
- 正式评测必须 frozen、fair、reproducible。

## 项目目录结构

- `/root/DermAgent/agent`：planner、retrieval、aggregator、reflection、evaluation、mining、training、export 等主逻辑
- `/root/DermAgent/skills`：基础 skill、specialist skill、skill registry
- `/root/DermAgent/memory`：experience 存储、检索与读写逻辑
- `/root/DermAgent/cognition`：cognition state 定义与更新逻辑
- `/root/DermAgent/configs`：trainable components、policy 等配置
- `/root/DermAgent/state`：运行期可写状态，包括 experience、cognition、policy、server pid
- `/root/DermAgent/data`：数据集输入
- `/root/DermAgent/scripts`：主要 CLI 脚本入口
- `/root/DermAgent/outputs`：执行记录、评测结果、checkpoint、analysis、paper export 等输出
- `/root/DermAgent/design`：设计文档
- `/root/DermAgent/tests`：回归测试与流水线测试

## 环境准备

先克隆项目并进入目录：

```bash
git clone <你的仓库地址> DermAgent
cd DermAgent
```

推荐使用 conda 环境：

```bash
conda env create -f environment.yml
conda activate derm-qwen
```

如果不用 conda，也可以使用 `venv + pip`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

当前验证过的核心依赖版本范围：

- Python 3.10
- `vllm>=0.7,<1`
- `openai>=1.66,<2`
- `pytest>=8.3,<9`

## 本地模型放置方式

`scripts/start_qwen_server.sh` 会按以下顺序寻找本地模型：

1. 环境变量 `MODEL_PATH`
2. `/models/Qwen2.5-VL-7B-Instruct`
3. `/root/models/Qwen2.5-VL-7B-Instruct`
4. `/models` 或 `/root/models` 下第一个匹配 `Qwen*` 的目录

如果你的模型不在默认位置，请显式指定：

```bash
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
```

## 启动本地 Qwen 服务

启动 vLLM 本地服务：

```bash
cd /root/DermAgent
bash scripts/start_qwen_server.sh
```

如果需要强制重启：

```bash
cd /root/DermAgent
FORCE_RESTART=1 bash scripts/start_qwen_server.sh
```

常见自定义启动方式：

```bash
cd /root/DermAgent
MODEL_PATH=/root/models/Qwen2.5-VL-7B-Instruct HOST=127.0.0.1 PORT=8000 bash scripts/start_qwen_server.sh
```

相关日志与 pid 文件：

- `/root/DermAgent/logs/qwen_server.log`
- `/root/DermAgent/state/qwen_server.pid`

## 服务健康检查

先检查模型列表：

```bash
curl -s -H "Authorization: Bearer EMPTY" http://127.0.0.1:8000/v1/models
```

再跑项目内置健康检查：

```bash
cd /root/DermAgent
OPENAI_API_KEY=EMPTY python scripts/check_qwen_server.py --timeout 30
```

## 清理旧测试产物与中间产物

如果你想重新跑一套干净的输出，而不影响代码、skill bank 或 live state，可清空 `outputs/`：

```bash
cd /root/DermAgent
rm -rf outputs/*
mkdir -p outputs
```

这会删除旧的 execution records、评测 manifest、analysis 报告、`outputs/` 下的 checkpoint 和各类中间产物，但不会删除：

- 仓库里的代码与 skill 定义
- `/root/DermAgent/state/experience`
- `/root/DermAgent/state/cognition`
- `/root/DermAgent/state/policy`

## 哪些内容是冻结的，哪些内容会累积

- `skill bank`：存放在仓库中，属于版本化、可审阅的源代码与结构化文本定义
- `experience bank`：累积在 `/root/DermAgent/state/experience`
- `cognition state`：累积在 `/root/DermAgent/state/cognition`
- 正式评测：必须在 frozen mode 下运行，写回关闭，并使用 snapshot isolation

实际使用上：

- 想积累经验库时，用 train/debug 运行并打开 `--enable-writeback`
- 想做公平评测时，用冻结脚本：
  - `scripts/compare_agent_vs_qwen.py`
  - `scripts/run_eval_brief.py`
  - `scripts/run_ablations.py`

## 最小可运行全流程 Smoke Workflow

这是从一个干净的 `outputs/` 目录开始，到重新生成一套新结果的最短已验证链路。

### 1. 先用 train split 的 writeback 跑几条 case，给经验库打底

```bash
cd /root/DermAgent
python scripts/debug_single_case.py --case-index 0 --enable-writeback --data-split train --run-mode train_debug_seed
python scripts/debug_single_case.py --case-index 1 --enable-writeback --data-split train --run-mode train_debug_seed
```

这些运行会生成单病例产物，例如：

- `/root/DermAgent/outputs/PAT_1516_1765/case_execution_record.json`
- `/root/DermAgent/outputs/PAT_1516_1765/evidence_package.json`
- `/root/DermAgent/outputs/PAT_1516_1765/reflection.json`
- `/root/DermAgent/outputs/PAT_1516_1765/state.json`

同时会把结构化经验写回：

- `/root/DermAgent/state/experience/raw_case_memory.jsonl`
- `/root/DermAgent/state/experience/tactical_experience.jsonl`
- `/root/DermAgent/state/experience/abstract_experience.jsonl`

### 2. 把重复出现的经验压缩为 abstract experience 候选

```bash
cd /root/DermAgent
python scripts/consolidate_experiences.py \
  --experience-root /root/DermAgent/state/experience \
  --hard-cases-path /root/DermAgent/outputs/hard_case_mining/hard_cases.jsonl \
  --output-dir /root/DermAgent/outputs/consolidation
```

如果这时还没有 `hard_cases.jsonl`，可以先跑 hard case mining，或者先用 `--dry-run`。在当前小样本 smoke run 中，如果 supporting cases 数量不够，脚本会正常运行，但 `consolidated_count` 可能为 `0`。

### 3. 跑一条真实单病例完整推理链路

```bash
cd /root/DermAgent
python scripts/debug_single_case.py --case-index 0
```

这条链路会经过：

1. Qwen 初始 perception
2. skill retrieval
3. experience retrieval
4. planner/controller 选择
5. skill 执行
6. evidence aggregation
7. Qwen 最终诊断
8. reflection 与可选 writeback

### 4. 在 frozen mode 下比较 direct Qwen 与 full DermAgent

```bash
cd /root/DermAgent
python scripts/compare_agent_vs_qwen.py \
  --limit 1 \
  --case-offset 0 \
  --data-split val \
  --output-dir /root/DermAgent/outputs/comparison \
  --client-timeout 180 \
  --client-max-retries 2
```

本轮已验证生成的示例输出：

- `/root/DermAgent/outputs/comparison/compare_agent_vs_qwen_20260326T121559Z.json`

### 5. 训练 learned components 的 smoke pipeline

```bash
cd /root/DermAgent
python scripts/train_learned_components.py \
  --stages 0,1,2,3,4 \
  --run-id final_flow_smoke \
  --records-root /root/DermAgent/outputs \
  --output-dir /root/DermAgent/outputs/train_runs \
  --checkpoint-out-dir /root/DermAgent/outputs/checkpoints \
  --training-split train \
  --limit 2 \
  --epochs 2 \
  --stage3-data-split val \
  --client-timeout 180 \
  --client-max-retries 2
```

本轮已验证 manifest：

- `/root/DermAgent/outputs/train_runs/final_flow_smoke/train_run_manifest.json`

### 6. 选择最优 checkpoint bundle

```bash
cd /root/DermAgent
python scripts/select_best_checkpoint.py \
  --train-runs-root /root/DermAgent/outputs/train_runs \
  --checkpoints-root /root/DermAgent/outputs/checkpoints \
  --output-dir /root/DermAgent/outputs/checkpoint_selection
```

本轮已验证 selection report：

- `/root/DermAgent/outputs/checkpoint_selection/checkpoint_selection_20260326T122744Z.json`

### 7. 跑冻结评测

```bash
cd /root/DermAgent
python scripts/run_eval_brief.py \
  --data-root /root/DermAgent/data \
  --output-dir /root/DermAgent/outputs/evaluation_protocol \
  --limit 1 \
  --case-offset 0 \
  --data-split test \
  --client-timeout 180 \
  --client-max-retries 2
```

本轮已验证 result manifest：

- `/root/DermAgent/outputs/evaluation_protocol/eval_brief_20260326T122829Z/result_manifest.json`

### 8. 跑论文消融矩阵的 smoke test

```bash
cd /root/DermAgent
python scripts/run_ablations.py \
  --data-root /root/DermAgent/data \
  --output-dir /root/DermAgent/outputs/ablations \
  --mode smoke \
  --limit 1 \
  --case-offset 0 \
  --data-split test \
  --client-timeout 180 \
  --client-max-retries 2
```

本轮已验证 summary：

- `/root/DermAgent/outputs/ablations/ablations_20260326T123015Z/ablation_matrix_summary.json`

### 9. 跑分析脚本并导出论文表格/图表底层数据

```bash
cd /root/DermAgent
python scripts/analyze_skill_helpfulness.py --records-root /root/DermAgent/outputs --output-dir /root/DermAgent/outputs/skill_helpfulness
python scripts/mine_hard_cases.py --records-root /root/DermAgent/outputs --output-dir /root/DermAgent/outputs/hard_case_mining
python scripts/audit_experiment_state.py --records-root /root/DermAgent/outputs/evaluation_protocol --eval-root /root/DermAgent/outputs/evaluation_protocol --fail-on-issues
python scripts/export_paper_tables.py --outputs-root /root/DermAgent/outputs --output-dir /root/DermAgent/outputs/paper_exports/tables
python scripts/export_paper_fig_data.py --outputs-root /root/DermAgent/outputs --output-dir /root/DermAgent/outputs/paper_exports/fig_data
```

本轮已验证输出：

- `/root/DermAgent/outputs/skill_helpfulness/skill_helpfulness_summary.json`
- `/root/DermAgent/outputs/hard_case_mining/summary.json`
- `/root/DermAgent/outputs/analysis/contamination_audit/contamination_audit_20260326T123903Z.json`
- `/root/DermAgent/outputs/paper_exports/tables/paper_tables_manifest.json`
- `/root/DermAgent/outputs/paper_exports/fig_data/paper_fig_data_manifest.json`

## 推荐的完整运行顺序

如果你要从零开始完整跑一遍，建议按这个顺序：

1. 克隆仓库并创建环境
2. 准备本地 Qwen2.5-VL 模型并启动本地服务
3. 跑健康检查
4. 如需干净实验，先清空 `outputs/`
5. 用 train split + `--enable-writeback` 跑一批 debug case，积累 `RawCaseMemory` 与 `TacticalExperience`
6. 定期运行 `scripts/mine_hard_cases.py`
7. 定期运行 `scripts/consolidate_experiences.py`，提炼更稳定的 `AbstractExperience`
8. 运行 `scripts/train_learned_components.py`，训练 controller 与 retrieval scorer
9. 运行 `scripts/select_best_checkpoint.py`，选择稳定候选 bundle
10. 运行冻结评测 `scripts/run_eval_brief.py`
11. 运行冻结消融 `scripts/run_ablations.py`
12. 运行 helpfulness analysis、hard case mining、contamination audit、paper export

## 运行后应该看到的输出目录

- `outputs/<case_id>/`：单病例 debug 产物与 `CaseExecutionRecord`
- `outputs/comparison/`：direct baseline vs DermAgent 的冻结对比结果
- `outputs/train_runs/`：分阶段训练 manifest 与阶段内 checkpoint
- `outputs/checkpoints/`：导出的组件 checkpoint 与 stable bundle
- `outputs/checkpoint_selection/`：checkpoint selection report
- `outputs/evaluation_protocol/`：冻结评测 manifest 与 target records
- `outputs/ablations/`：论文级 ablation manifests 与 summary
- `outputs/skill_helpfulness/`：skill 级 helpfulness 分析结果
- `outputs/hard_case_mining/`：hard case 结果与 summary
- `outputs/analysis/contamination_audit/`：污染审计结果
- `outputs/paper_exports/`：论文表格与图表底层数据
- `state/experience/`：live experience bank
- `state/cognition/`：live cognition state
- `state/policy/`：current stable policy 与版本信息

## 粗略耗时参考

以下耗时来自本机当前 smoke run 的实际观测值，不代表严格上界：

- 启动本地 Qwen 服务：通常 1 到 5 分钟，取决于模型加载与 GPU 状态
- `check_qwen_server.py`：1 分钟内
- `debug_single_case.py` 单病例：大约 30 到 90 秒
- `compare_agent_vs_qwen.py --limit 1`：1 分钟内
- `train_learned_components.py --limit 2 --epochs 2`：约 3 分 30 秒
- `run_eval_brief.py --limit 1`：1 分钟内
- `run_ablations.py --mode smoke --limit 1`：数分钟，因为会串行跑多个 target
- hard case mining / helpfulness analysis / contamination audit / paper export：通常几秒到几十秒

当 case 数量增大后，总耗时主要受 Qwen 推理调用量影响。

## 论文级正式评测规则

当你要产出论文表格或正式结果时，必须保证：

- 使用同一个 Qwen 服务模型
- baseline 与 agent 使用相同 case list
- 评测期间禁止 online writeback
- 必须从 frozen snapshot state 读取
- manifest 中必须记录：
  - Qwen model version
  - prompt/version
  - skill bank version
  - experience bank version
  - cognition version
  - policy version

当前实现中，以下脚本用于保证这些要求：

- `scripts/compare_agent_vs_qwen.py`
- `scripts/run_eval_brief.py`
- `scripts/run_ablations.py`
- `scripts/audit_experiment_state.py`

## 快速验证命令

跑全部测试：

```bash
cd /root/DermAgent
pytest -q tests
```

跑语法编译检查：

```bash
cd /root/DermAgent
python -m compileall agent scripts cognition memory skills
```

## 额外说明

- skill bank 不会在正式评测时自动增长，仍然应该作为仓库内可审阅资产来维护
- experience 的增长来自 writeback 加后续 consolidation，而不是自由文本堆积
- learnable components 只优化 controller / retrieval / calibration 等外围组件，不会训练或修改 Qwen 权重
