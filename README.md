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

DermAgent 优化的不是 backbone 本身，而是外围可训练组件。当前代码里主要包括：

- controller / planner scorer
- retrieval reranker / scorer
- evidence calibrator

相关脚本包括：

- [`scripts/train_controller.py`](/root/DermAgent/scripts/train_controller.py)
- [`scripts/train_retrieval_scorer.py`](/root/DermAgent/scripts/train_retrieval_scorer.py)
- [`scripts/train_evidence_calibrator.py`](/root/DermAgent/scripts/train_evidence_calibrator.py)
- [`scripts/train_learned_components.py`](/root/DermAgent/scripts/train_learned_components.py)
- [`scripts/evaluate_policy_candidate.py`](/root/DermAgent/scripts/evaluate_policy_candidate.py)

主张是：

- 不训练 Qwen 权重
- 不做 end-to-end backbone finetuning
- 只对外围策略层做 staged optimization

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
