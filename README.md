# DermAgent

DermAgent 是一个面向皮肤科图像诊断的结构化智能体推理框架。

它的核心目标不是替代底层视觉语言模型，而是在冻结 backbone 的前提下，把最终诊断前的临床推理过程显式组织为：

- 技能库 `skills`
- 分层经验库 `memory`
- 认知状态 `cognition`
- 结构化证据包 `agent/evidence_package.py`
- 可训练但不改 backbone 的策略层 `agent/policy_*`、`agent/supervised_controller.py`、`agent/retrieval_scorer.py`、`agent/evidence_calibrator.py`

按照论文当前主线设定，DermAgent 的研究问题是：

> 在固定同一个皮肤科视觉语言模型骨干的条件下，结构化、经验驱动的 agent reasoning，是否能优于 direct prompting？

当前仓库的主结果线以 `Qwen` 为核心，扩展验证线以 `SkinVL` 为主。

## 项目定位

DermAgent 采用“三段式”诊断流程：

1. backbone 先做初始感知，生成病灶概述、初始鉴别诊断候选和不确定性线索
2. agent 中间层执行技能选择、经验检索、证据组织与风险/矛盾/信息缺口审计
3. backbone 读取结构化证据包后给出最终诊断

这里有两个非常重要的边界：

- DermAgent 不直接输出最终疾病标签
- 最终诊断责任始终保留给 backbone

所以它不是一个“外挂分类器集合”，也不是“多模型投票器”，而是一个围绕最终诊断前证据构建的 structured reasoning scaffold。

## 论文主张

[`paper/main.tex`](/root/DermAgent/paper/main.tex) 当前表达的主线是：

- 固定视觉语言模型骨干
- 公平比较 direct baseline 与 agent-enhanced pipeline
- 正式评测时关闭在线 writeback
- 使用 frozen state、固定 case list、相同输入可见条件做受控实验

论文摘要里给出的主结果是：

- 同骨干、同 345 个测试病例、冻结评测条件下
- direct Qwen baseline top-1 从 `29.6%` 提升到 `39.7%`
- 恶性召回从 `48.2%` 提升到 `82.2%`
- 错误率从 `70.4%` 降到 `60.3%`

如果你要快速理解这个项目，建议优先读：

1. [`paper/main.tex`](/root/DermAgent/paper/main.tex)
2. [`design/01_overall_architecture.txt`](/root/DermAgent/design/01_overall_architecture.txt)
3. [`design/03_skill_system.txt`](/root/DermAgent/design/03_skill_system.txt)
4. [`design/05_experience_system.txt`](/root/DermAgent/design/05_experience_system.txt)
5. [`design/08_evidence_package.txt`](/root/DermAgent/design/08_evidence_package.txt)
6. [`agent/run_agent.py`](/root/DermAgent/agent/run_agent.py)

## 仓库结构

核心目录如下：

- [`agent`](/root/DermAgent/agent)
  - 主推理链路、planner/controller、evaluation、reflection、evidence package、训练与评测协议
- [`skills`](/root/DermAgent/skills)
  - 原子化临床推理技能，如 morphology、color pattern、differential compare、uncertainty、malignancy risk
- [`memory`](/root/DermAgent/memory)
  - 分层经验系统，包括 raw case memory、tactical experience、abstract experience
- [`cognition`](/root/DermAgent/cognition)
  - 跨病例认知状态
- [`dataio`](/root/DermAgent/dataio)
  - 数据读取与 schema 对齐
- [`scripts`](/root/DermAgent/scripts)
  - 开发与实验脚本，包括服务启动、训练、比较、消融、导表导图
- [`final-script`](/root/DermAgent/final-script)
  - 最终实验入口，收束主线运行方式
- [`final-score`](/root/DermAgent/final-score)
  - 最终实验输出与论文导出结果
- [`paper`](/root/DermAgent/paper)
  - 论文源码、表格、图和论文草稿材料
- [`design`](/root/DermAgent/design)
  - 系统设计文档与技能规范
- [`tests`](/root/DermAgent/tests)
  - 单测与协议测试

## 环境准备

项目当前环境定义见：

- [`environment.yml`](/root/DermAgent/environment.yml)
- [`requirements.txt`](/root/DermAgent/requirements.txt)

推荐使用 conda：

```bash
cd /root/DermAgent
conda env create -f environment.yml
conda activate derm-qwen
```

如果你已经有 Python 3.10 环境，也可以直接：

```bash
cd /root/DermAgent
pip install -r requirements.txt
```

当前依赖重点包括：

- `openai`
- `vllm`
- `pandas`
- `pillow`
- `pydantic`
- `pytest`

## 当前推荐主线

当前项目总入口应理解为：

- 唯一主线：`direct Qwen` vs `agent + Qwen`
- 补充分析：Qwen 线消融、paired statistics、qualitative case study、外部数据集泛化
- 扩展验证：`SkinVL` 迁移与泛化

当前 README 推荐工作流只保留 `Qwen` 主线和 `SkinVL` 扩展线。

## 快速开始

### 0. 先确认资产根目录

主线 `Qwen`、泛化线 `SkinVL` / `MedGemma`，以及任何后续新数据集适配实验，都必须使用各自独立的：

- `DERMAGENT_POLICY_ROOT`
- `DERMAGENT_SPLIT_STATE_ROOT`
- bootstrap / train outputs
- checkpoint exports

从当前版本开始，`run_agent.py` 的经验库读取、cognition 写回与 `debug_single_case.py` 这类底层入口，都会严格跟随这两个环境变量；不再只在上层评测脚本里分仓、而底层写回偷偷落到默认全局仓。

这意味着：

- 使用 `final-script/` 官方入口跑旧主线实验时，命令本身不需要改，因为对应的 `.env` 已经固定好了根目录
- 手动运行底层脚本时，必须先把根目录环境变量设对，否则 writeback / bootstrap 会进入你当前指向的仓

### 1. 启动 Qwen 服务

推荐使用 final-script 的固定入口：

```bash
cd /root/DermAgent
bash final-script/servers/start_qwen_final.sh
```

这个脚本会读取：

- [`final-script/configs/qwen_final.env`](/root/DermAgent/final-script/configs/qwen_final.env)
- [`final-script/configs/common.env`](/root/DermAgent/final-script/configs/common.env)

然后转发到：

- [`scripts/start_qwen_server.sh`](/root/DermAgent/scripts/start_qwen_server.sh)

### 2. 跑主线 compare

```bash
cd /root/DermAgent
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --smoke-10
```

常用模式：

```bash
cd /root/DermAgent
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --medium-40
```

```bash
cd /root/DermAgent
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --full-test
```

脚本位置：

- [`final-script/runs/run_main_qwen_vs_agent_qwen.sh`](/root/DermAgent/final-script/runs/run_main_qwen_vs_agent_qwen.sh)

这个入口会：

1. 等待本地 Qwen 服务就绪
2. 固定 `policy_root` 和 `split_state_root`
3. 调用 [`scripts/compare_agent_vs_qwen.py`](/root/DermAgent/scripts/compare_agent_vs_qwen.py)
4. 将结果写入 `final-score/final_runs/main_qwen_vs_agent_qwen/`

对现有主线旧实验来说，这里最重要的结论是：

- `final-script/runs/run_main_qwen_vs_agent_qwen.sh` 这类官方入口命令保持不变
- 如果你绕过这些入口，直接跑 `scripts/debug_single_case.py`、`scripts/evaluate_policy_candidate.py`、bootstrap 脚本或其他会触发写回的底层命令，必须先显式设置 `DERMAGENT_POLICY_ROOT` 和 `DERMAGENT_SPLIT_STATE_ROOT`

### 3. 产出位置

默认最终实验输出目录定义在：

- [`final-script/configs/common.env`](/root/DermAgent/final-script/configs/common.env)

其中关键路径包括：

- `final-score/final_runs/main_qwen_vs_agent_qwen`
- `final-score/final_runs/ablations_qwen`
- `final-score/final_runs/external_qwen`
- `final-score/final_runs/paired_stats`
- `final-score/final_runs/qual_case_study`
- `final-score/paper_exports`

### 4. SkinVL 扩展线

如果需要跑跨 backbone 扩展验证，当前保留的推荐路线是 `SkinVL`：

```bash
cd /root/DermAgent
bash final-script/servers/start_skinvl_final.sh
```

```bash
cd /root/DermAgent
bash final-script/runs/run_main_skinvl_vs_agent_skinvl.sh
```

相关入口：

- [`final-script/servers/start_skinvl_final.sh`](/root/DermAgent/final-script/servers/start_skinvl_final.sh)
- [`final-script/runs/run_main_skinvl_vs_agent_skinvl.sh`](/root/DermAgent/final-script/runs/run_main_skinvl_vs_agent_skinvl.sh)

这条线更适合作为方法迁移与泛化验证，不替代 `Qwen` 主线。

## 运行模式：手工规则 vs 参数化

DermAgent 支持两种运行模式，通过 `--policy-config` 或 `QWEN_FINAL_POLICY_JSON` 环境变量切换，**不需要改代码**。

### 模式一：手工规则（heuristic，默认推荐）

全部使用规则驱动，不加载任何 learned checkpoint，零样本可部署：

```bash
cd /root/DermAgent
source final-script/configs/final_assets_registry.env

# 使用 heuristic_no_penalty（接近 v0 行为，无 penalty，无 adaptive budget）
export QWEN_FINAL_POLICY_JSON=/root/DermAgent/state/policy/versions/heuristic_no_penalty.json
export OUTPUT_DIR=/root/DermAgent/final-score/final_runs/heuristic_no_penalty
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --smoke-10
```

```bash
# 使用 heuristic_with_penalty（开启 penalty 和 adaptive budget）
export QWEN_FINAL_POLICY_JSON=/root/DermAgent/state/policy/versions/heuristic_with_penalty.json
export OUTPUT_DIR=/root/DermAgent/final-score/final_runs/heuristic_with_penalty
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --smoke-10
```

两个 heuristic policy 文件的差异：

| 配置 | penalty 权重 | adaptive budget | 适用场景 |
|------|-------------|-----------------|---------|
| `heuristic_no_penalty.json` | 0（关闭） | 关闭 | 最接近原始 v0 行为，基线对比 |
| `heuristic_with_penalty.json` | 开启（4.0 / 2.0） | 开启 | 当前 heuristic 推荐默认值 |

### 模式二：参数化 learned（主线配置）

使用 learned controller + learned retrieval reranker + hybrid evidence calibrator：

```bash
cd /root/DermAgent
source final-script/configs/final_assets_registry.env

# 使用主线 learned 配置（final_assets_registry.env 已默认指向此文件）
export QWEN_FINAL_POLICY_JSON=/root/DermAgent/state/policy/current_stable_policy.json
export OUTPUT_DIR=/root/DermAgent/final-score/final_runs/learned_mainline
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --smoke-10
```

或者直接用原始 final-script 入口（已默认读取 learned 主线配置，不需要额外 export）：

```bash
cd /root/DermAgent
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --smoke-10
```

### 三种配置的指标对比（10-case）

| 配置 | top-1 | top-k | malignant recall | error rate |
|------|-------|-------|-----------------|------------|
| Direct baseline (Qwen only) | 30% | 70% | 25% | 70% |
| Heuristic no-penalty | 50% | **60%** | **87.5%** | 50% |
| Heuristic with-penalty | 50% | 50% | **87.5%** | 50% |
| Learned mainline (345-case full) | 39.7% | — | 82.2% | 60.3% |

> heuristic 层本身已带来 malignant recall 25% → 87.5% 的显著提升，learned 在此基础上提供可选增益。

---

## 训练参数化组件

> 训练组件之前，需要先通过 bootstrap 积累足够的执行记录（`outputs/` 目录下的 JSONL），用于提取训练样本。

### 前提：确认训练数据已就绪

```bash
# 确认 controller 训练样本文件存在
ls /root/DermAgent/outputs/controller_training_data/controller_training_examples.jsonl

# 确认 retrieval / evidence calibrator 所需的 execution records 存在
ls /root/DermAgent/outputs/
```

### 1. 训练 Controller（Planner Scorer）

```bash
cd /root/DermAgent
python scripts/train_controller.py \
  --examples-path outputs/controller_training_data/controller_training_examples.jsonl \
  --output-dir state/trainable_components/controller_planner_scorer/candidates \
  --epochs 40 \
  --hidden-dim 128 \
  --lr 1e-3 \
  --selection-profile conservative_sparse
```

训练完成后，checkpoint 保存在：
```
state/trainable_components/controller_planner_scorer/candidates/<run_name>.pt
```

### 2. 训练 Retrieval Reranker

```bash
cd /root/DermAgent
python scripts/train_retrieval_scorer.py \
  --records-root outputs \
  --output-dir state/trainable_components/retrieval_reranker/candidates \
  --epochs 30 \
  --hidden-dim 96 \
  --lr 1e-3
```

训练完成后，checkpoint 保存在：
```
state/trainable_components/retrieval_reranker/candidates/<run_name>.pt
```

### 3. 训练 Evidence Calibrator

```bash
cd /root/DermAgent
python scripts/train_evidence_calibrator.py \
  --records-root outputs \
  --output-dir state/trainable_components/evidence_calibrator/candidates \
  --epochs 20 \
  --hidden-dim 64 \
  --lr 1e-3
```

训练完成后，checkpoint 保存在：
```
state/trainable_components/evidence_calibrator/candidates/<run_name>.pt
```

### 4. 把新 checkpoint 接入 policy

训练好新 checkpoint 后，编辑（或新建）policy JSON，把路径填入对应字段：

```json
{
  "planner_policy": {
    "controller_family": "learned_supervised",
    "controller_checkpoint_path": "/root/DermAgent/state/trainable_components/controller_planner_scorer/candidates/<your_run>.pt"
  },
  "retrieval_policy": {
    "enable_learned_retrieval_reranker": true,
    "retrieval_reranker_checkpoint_path": "/root/DermAgent/state/trainable_components/retrieval_reranker/candidates/<your_run>.pt"
  },
  "evidence_policy": {
    "calibrator_mode": "hybrid",
    "calibrator_checkpoint_path": "/root/DermAgent/state/trainable_components/evidence_calibrator/candidates/<your_run>.pt"
  }
}
```

然后通过环境变量指向这个新文件：

```bash
export QWEN_FINAL_POLICY_JSON=/root/DermAgent/state/policy/versions/<your_new_policy>.json
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --smoke-10
```

### 5. 三个字段的开关对照

| 组件 | 关闭（heuristic） | 开启（learned） |
|------|------------------|----------------|
| Controller | `"controller_family": "heuristic"` | `"controller_family": "learned_supervised"` + `controller_checkpoint_path` |
| Retrieval Reranker | `"enable_learned_retrieval_reranker": false` | `"enable_learned_retrieval_reranker": true` + `retrieval_reranker_checkpoint_path` |
| Evidence Calibrator | `"calibrator_mode": "heuristic"` | `"calibrator_mode": "hybrid"` 或 `"learned"` + `calibrator_checkpoint_path` |

任意组合都可以，例如只开 controller 关 reranker、或只开 calibrator，每个组件独立控制。

---

## 旧实验重跑注意事项

如果你现在要重新启动仓库里原来的旧实验，建议按下面规则执行：

1. 主线 `Qwen` compare、`SkinVL` compare、`MedGemma` compare 这类 `final-script/` 官方入口，命令不用改。
2. 任何带 writeback / bootstrap / 单病例调试 / 候选策略评测的底层脚本，都不要裸跑；先切到对应模型线或实验线的环境变量。
3. 不要在一个 shell 里先 `source` 某条实验线的根目录，再直接去跑另一条线的 bootstrap；换线前重新 `source` 对应 `.env`。
4. 如果要做“新数据集适配实验”，不要复用旧主线的 `state/policy` 与 `state/split_states`；必须另建独立资产仓。

例如，手动进入 `Qwen` 主线环境后再跑底层脚本：

```bash
cd /root/DermAgent
source final-script/configs/qwen_final.env
python scripts/debug_single_case.py --case-index 0 --data-split train --enable-writeback
```

手动进入 `SkinVL` 环境：

```bash
cd /root/DermAgent
source final-script/configs/skinvl_final.env
python scripts/debug_single_case.py --case-index 0 --data-split train --enable-writeback
```

如果你不想手工维护新数据集实验的分仓路径，当前可以使用：

- [`scripts/manage_dataset_experiment_assets.py`](/root/DermAgent/scripts/manage_dataset_experiment_assets.py)

它可以初始化独立 `policy_root / split_state_root / outputs / checkpoints`，并把 bootstrap 后的 `train` 状态安全提升到 `val/test`。

## 多数据集标签空间

从当前版本开始，DermAgent 不再只假设唯一一套固定标签，而是逐步改成支持 dataset-specific label space。

当前设计目标是：

- 不同数据集可以保留各自原始标签
- 不同数据集可以注册各自的 canonical label space
- 不同数据集可以各自建立经验库、训练策略参数、做冻结评测
- 评测、hard-case、experience writeback、policy summary 不再强依赖单一六类标签假设

核心实现入口在：

- [`agent/label_space.py`](/root/DermAgent/agent/label_space.py)

这个模块负责统一处理：

- `canonicalize_label(...)`
- `is_malignant_label(...)`
- `labels_match(...)`
- `label_space_snapshot(...)`
- `register_label_space(...)`

### 当前默认行为

默认仍然保留当前主线的 dermatology six-label behavior：

- `BCC`
- `ACK`
- `NEV`
- `SEK`
- `SCC`
- `MEL`

也就是说，不改任何数据集配置时，主线 `Qwen` 实验与现有 `pad_ufes_20` 逻辑仍然按旧六类方式工作。

### 新数据集接入原则

以后接一个新数据集时，建议不要直接把它硬塞进现有六类逻辑，而是按下面顺序做：

1. 保留原始标签字段。
2. 为该数据集定义自己的 `label_space_id`。
3. 在 [`agent/label_space.py`](/root/DermAgent/agent/label_space.py) 注册该数据集对应的 `LabelSpace`。
4. 在该数据集的 loader/schema 中，把 `dataset_name` 与可选的 `label_space_id` 一起传进 [`CaseInput`](/root/DermAgent/agent/state.py)。
5. 让该数据集的经验库、训练产物和评测结果使用独立资产仓，不与旧主线混用。

推荐的数据结构至少包含：

```json
{
  "original_label": "...",
  "dataset_name": "...",
  "label_space_id": "...",
  "canonical_label": "...",
  "risk_label": "malignant|benign|unknown"
}
```

### 为什么要这样做

这样做的好处是：

- 新数据集不需要反复修改 `evaluation / reflection / hard_case / experience` 的核心逻辑
- 每个数据集都可以有自己稳定的标签空间，而不是到处塞 `if dataset == ...`
- 经验库和后续 consolidation 会知道自己属于哪套标签空间
- 未来做多数据集建库、训练参数和冻结评测时，逻辑边界更清楚，也更容易审计

### 目前已接入这套标签空间逻辑的主链

当前已经开始改造成 dataset-specific label space 的模块包括：

- [`agent/evaluation.py`](/root/DermAgent/agent/evaluation.py)
- [`agent/execution_record.py`](/root/DermAgent/agent/execution_record.py)
- [`agent/reflection.py`](/root/DermAgent/agent/reflection.py)
- [`memory/experience_transform.py`](/root/DermAgent/memory/experience_transform.py)
- [`memory/experience_consolidator.py`](/root/DermAgent/memory/experience_consolidator.py)
- [`agent/hard_case_miner.py`](/root/DermAgent/agent/hard_case_miner.py)
- [`agent/policy_evaluation.py`](/root/DermAgent/agent/policy_evaluation.py)

### 实操建议

如果未来你要针对一个新数据集单独做：

- bootstrap 建经验库
- 训练 controller / retrieval / evidence 相关参数
- 最终在该数据集上做预测与冻结评测

建议把它视为一条独立实验线，同时独立管理：

- `DERMAGENT_POLICY_ROOT`
- `DERMAGENT_SPLIT_STATE_ROOT`
- `outputs`
- `checkpoints`
- `label_space_id`

不要复用旧主线资产仓，也不要默认沿用旧六类标签解释。

### 数据集专用经验库迁移流程

如果你的目标是：

- 为一个新数据集单独建立经验库
- 用这套经验库做 stable agent 迁移测试
- 后续再决定是否训练 learned controller / retrieval

推荐按下面顺序执行。

#### 第一步：初始化独立资产仓

```bash
cd /root/DermAgent

python scripts/manage_dataset_experiment_assets.py init \
  --experiment-id <dataset_experiment_id> \
  --base-policy-config /root/DermAgent/state/policy/current_stable_policy.json
```

这一步会创建独立的：

- `policy_root`
- `split_state_root`
- `outputs`
- `checkpoints`

#### 第二步：进入该实验线环境

```bash
source /root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/experiment.env
```

后续所有 bootstrap / compare / train 命令，都建议在这个环境下执行。

#### 第三步：先在 train split 上 bootstrap 建库

核心原则是：

- 只在 `train` 上开启 writeback
- 只写入当前数据集自己的隔离资产仓
- 不碰旧主线仓

例如：

```bash
cd /root/DermAgent

START_INDEX=0 COUNT=24 \
POLICY_ROOT="$DERMAGENT_POLICY_ROOT" \
SPLIT_STATE_ROOT="$DERMAGENT_SPLIT_STATE_ROOT" \
OUTPUT_DIR="$DERMAGENT_DATASET_EXPERIMENT_OUTPUT_ROOT/bootstrap" \
bash scripts/bootstrap_<dataset>_train_cases.sh
```

如果该数据集使用的是“分层均匀 split”，更推荐按 `train_case_indices` 来 bootstrap，而不是简单用 `START_INDEX=0` 连续取样。通用写法如下：

```bash
cd /root/DermAgent

mapfile -t TRAIN_INDICES < <(
python - <<'PY'
import json
from pathlib import Path
split = Path('/root/DermAgent/outputs/dataset_adaptation/<dataset_experiment_id>/<dataset_split>.json')
data = json.loads(split.read_text(encoding='utf-8'))
for idx in data['train_case_indices'][:24]:
    print(idx)
PY
)

for idx in "${TRAIN_INDICES[@]}"; do
  echo "[bootstrap] case-index=${idx}"
  OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
  OPENAI_API_KEY=EMPTY \
  OPENAI_MODEL=Qwen2.5-VL-7B-Instruct \
  DERMAGENT_POLICY_ROOT="$DERMAGENT_POLICY_ROOT" \
  DERMAGENT_SPLIT_STATE_ROOT="$DERMAGENT_SPLIT_STATE_ROOT" \
  python scripts/debug_single_case.py \
    --case-index "${idx}" \
    --data-root /root/DermAgent/data/<dataset_name> \
    --output-dir "$DERMAGENT_DATASET_EXPERIMENT_OUTPUT_ROOT/bootstrap" \
    --enable-writeback \
    --data-split train \
    --run-mode <dataset_name>_train_bootstrap \
    --client-timeout 180 \
    --client-max-retries 2
done
```

#### 第四步：把 train 状态迁移到 val/test

正式冻结评测不能直接读 train 写回中的在线状态，因此推荐在 bootstrap 之后，把 train 状态复制成 val/test 的冻结状态：

```bash
python scripts/manage_dataset_experiment_assets.py promote-state \
  --split-state-root "$DERMAGENT_SPLIT_STATE_ROOT" \
  --source-split train \
  --target-splits val,test
```

这一步的作用是：

- `train/experience` -> `val/experience`
- `train/experience` -> `test/experience`
- `train/cognition_state.json` -> `val/test`

同时会重写 split-aware metadata，保证后续 frozen evaluation 一致。

#### 第四步半：先确认 split 和 smoke slice 是否合理

对于像 HAM10000 这类容易出现长段同类样本的数据集，强烈建议在 compare 之前先检查 `val[:10]` 和 `test[:10]` 的标签组成，避免出现 “前 10 个全是同一类” 的无效 smoke 切片：

```bash
cd /root/DermAgent

python - <<'PY'
import csv, json
from pathlib import Path
split = Path('/root/DermAgent/outputs/dataset_adaptation/<dataset_experiment_id>/<dataset_split>.json')
meta = Path('/root/DermAgent/data/<dataset_name>/<metadata_csv>')
data = json.loads(split.read_text(encoding='utf-8'))

rows = {}
with meta.open('r', encoding='utf-8', newline='') as f:
    for row in csv.DictReader(f):
        key = row.get('image') or row.get('image_id')
        if key:
            rows[key] = row

label_field = '<label_field_name>'
for name in ['val', 'test']:
    print('===', name, 'first10 ===')
    print([rows[cid][label_field] for cid in data[name][:10]])
PY
```

如果发现 `first10` 仍然严重偏斜，应先修 split，再做 compare。

#### 第五步：先用 stable policy 做 frozen compare

第一轮更推荐先观察：

- `direct baseline`
- `stable agent + dataset-specific experience`

而不是立刻启用 learned candidate。

例如：

```bash
DERMAGENT_SPLIT_STATE_ROOT=/root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/split_states \
DERMAGENT_POLICY_ROOT=/root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/policy \
python scripts/compare_agent_vs_qwen.py \
  --data-root /root/DermAgent/data/<dataset_name> \
  --split-json /root/DermAgent/outputs/dataset_adaptation/<dataset_experiment_id>/<dataset_split>.json \
  --data-split test \
  --output-dir /root/DermAgent/outputs/dataset_adaptation/<dataset_experiment_id>/compare_test \
  --policy-config /root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/policy/current_stable_policy.json \
  --policy-label "<dataset> stable policy"
```

更完整的 `val/test` 对比模板如下：

```bash
cd /root/DermAgent

DERMAGENT_SPLIT_STATE_ROOT=/root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/split_states \
DERMAGENT_POLICY_ROOT=/root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/policy \
python scripts/compare_agent_vs_qwen.py \
  --data-root /root/DermAgent/data/<dataset_name> \
  --split-json /root/DermAgent/outputs/dataset_adaptation/<dataset_experiment_id>/<dataset_split>.json \
  --data-split val \
  --output-dir /root/DermAgent/outputs/dataset_adaptation/<dataset_experiment_id>/compare_val_10 \
  --policy-config /root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/policy/current_stable_policy.json \
  --policy-label "<dataset> stable policy" \
  --limit 10 \
  --client-timeout 180 \
  --client-max-retries 2

DERMAGENT_SPLIT_STATE_ROOT=/root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/split_states \
DERMAGENT_POLICY_ROOT=/root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/policy \
python scripts/compare_agent_vs_qwen.py \
  --data-root /root/DermAgent/data/<dataset_name> \
  --split-json /root/DermAgent/outputs/dataset_adaptation/<dataset_experiment_id>/<dataset_split>.json \
  --data-split test \
  --output-dir /root/DermAgent/outputs/dataset_adaptation/<dataset_experiment_id>/compare_test_10 \
  --policy-config /root/DermAgent/state/dataset_adaptation/<dataset_experiment_id>/policy/current_stable_policy.json \
  --policy-label "<dataset> stable policy" \
  --limit 10 \
  --client-timeout 180 \
  --client-max-retries 2
```

如果 `val/test` 的 10-case smoke 结果趋势不稳定，可以：

1. 先补更多 bootstrap case。
2. 再执行一次 `promote-state`。
3. 再重新跑 `compare_val_10` / `compare_test_10`。

#### 第六步：如果经验库太小，可以继续补 bootstrap

一个常见流程是：

1. 先 bootstrap 12 或 24 个 case
2. 跑一次 `val/test` compare
3. 如果效果不稳定，再补 12 个或更多 case
4. 再次执行 `promote-state`
5. 再跑 compare

也就是说，这条迁移路径本身支持“逐步扩库”：

- `bootstrap more train cases`
- `promote train -> val/test`
- `rerun frozen compare`

#### 第七步：再决定是否训练 learned components

推荐顺序是：

1. 先验证 `stable agent + dataset-specific experience` 是否已经优于 direct baseline
2. 如果已经有明确增益，再考虑 learned controller / retrieval
3. 如果小样本下 candidate 不稳定，不要强行替换 stable heuristic

目前经验表明，很多情况下：

- 专用经验库会先带来收益
- learned parameterization 在小样本新数据集上不一定稳定

所以“先建库，再稳定 compare，最后再训练参数”通常更稳。

## 核心设计

### 1. Skills 不是分类器

DermAgent 中的 skill 是“原子化临床推理动作”，不是最终病种预测器。比如：

- 病灶形态分析
- 颜色模式分析
- 边界与表面审查
- 鉴别比较
- 排除推理
- 恶性风险评估
- 不确定性评估
- 信息缺口检测

对应代码集中在 [`skills`](/root/DermAgent/skills)。

### 2. Experience 是分层的

经验系统不是简单的“相似病例缓存”，而是三层结构：

- raw case memory
- tactical experience
- abstract experience

核心实现位于：

- [`memory/experience_bank.py`](/root/DermAgent/memory/experience_bank.py)
- [`memory/experience_store.py`](/root/DermAgent/memory/experience_store.py)
- [`memory/experience_consolidator.py`](/root/DermAgent/memory/experience_consolidator.py)

### 3. Cognition 和 Policy 分离

项目明确区分：

- experience：系统知道什么
- cognition：系统倾向怎么做
- policy：把这些状态转成当前病例中的控制决策

这样可以在不改 backbone 权重的前提下，单独优化外围策略模块。

### 4. Evidence Package 是中间接口

agent 的终点不是另一个诊断答案，而是结构化证据包。这个接口是 backbone 读取的增强上下文，也是项目最重要的系统边界之一。

相关实现见：

- [`agent/evidence_package.py`](/root/DermAgent/agent/evidence_package.py)
- [`agent/evidence_calibrator.py`](/root/DermAgent/agent/evidence_calibrator.py)

### 5. Frozen Evaluation 是方法的一部分

DermAgent 很强调评测公平性。正式比较时默认要求：

- 相同 backbone
- 相同 case list
- 相同输入可见条件
- 固定 split state
- 关闭 online writeback

这不是单纯工程细节，而是项目回答研究问题的前提。

## 训练与优化

DermAgent 优化的不是 backbone 本身，而是外围三个可训练组件（controller、retrieval reranker、evidence calibrator）。

具体的训练命令和 checkpoint 接入方式见上方"[训练参数化组件](#训练参数化组件)"章节。

核心原则：

- 不训练 Qwen 权重，不做 end-to-end backbone finetuning
- 只对外围策略层做 staged optimization
- 三个组件可独立控制开关，每个组件有独立 checkpoint，heuristic 路径永远可用

相关脚本：

- [`scripts/train_controller.py`](/root/DermAgent/scripts/train_controller.py)
- [`scripts/train_retrieval_scorer.py`](/root/DermAgent/scripts/train_retrieval_scorer.py)
- [`scripts/train_evidence_calibrator.py`](/root/DermAgent/scripts/train_evidence_calibrator.py)
- [`scripts/train_learned_components.py`](/root/DermAgent/scripts/train_learned_components.py)
- [`scripts/evaluate_policy_candidate.py`](/root/DermAgent/scripts/evaluate_policy_candidate.py)

## 测试

仓库已经包含较完整的测试集，覆盖：

- 评测协议
- 污染防护
- trainable components 配置
- policy/root override
- retrieval scorer
- evidence calibrator
- paper exports

运行方式：

```bash
cd /root/DermAgent
pytest
```

如果只想先跑一个小范围：

```bash
cd /root/DermAgent
pytest tests/test_policy_evaluation.py
```

## 开发建议

如果你是第一次接手这个项目，建议按这个顺序熟悉：

1. 先读论文和 design 文档，建立方法边界
2. 再看 [`agent/run_agent.py`](/root/DermAgent/agent/run_agent.py) 和 [`agent/evaluation_protocol.py`](/root/DermAgent/agent/evaluation_protocol.py)
3. 再看 skill、memory、cognition 三层
4. 最后再看 `scripts/` 与 `final-script/` 的实验编排

如果你要继续推进当前主线，优先做这些事：

1. 保持 Qwen 主线与 frozen evaluation 不变
2. 继续围绕 skill selection、experience routing、evidence calibration 做优化
3. 所有正式结果都尽量走 `final-script/` 入口
4. 避免把历史遗留的多 backbone 线路混入当前主 README 叙事

## 相关文档

- 项目主论文：[`paper/main.tex`](/root/DermAgent/paper/main.tex)
- 最终实验执行手册：[`final-script/README.md`](/root/DermAgent/final-script/README.md)
- 最终产出说明：[`final-score/README.md`](/root/DermAgent/final-score/README.md)

## 当前说明

这份 README 以当前项目主线为准，明确聚焦 `Qwen` 同骨干对照、结构化 agent 推理和冻结评测。

项目总说明当前只聚焦 `Qwen` 主线与 `SkinVL` 扩展验证线。
