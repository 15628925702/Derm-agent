# DermAgent

DermAgent 是一个面向皮肤科推理的智能体系统。系统始终保持 `Qwen` 作为唯一最终诊断者，在此基础上增加结构化 skill bank、分层 experience bank、cognition state、evidence aggregation、reflection/writeback、可学习 controller 钩子，以及冻结的论文级评测协议。

项目长期遵守以下原则：

- `Qwen` 始终是唯一最终诊断模型。
- `skill`、`experience`、`cognition` 严格分层、分库。
- reflection / writeback / policy evolution 必须有结构化 evidence 支撑，且可审计。
- 正式评测必须 frozen、fair、reproducible。

## 当前主线策略

当前主线 runtime 默认使用：

- learned controller
- learned retrieval reranker
- hybrid learned evidence calibrator

当前主线 stable policy 位于：

- `/root/DermAgent/state/policy/current_stable_policy.json`

当前主线 evidence calibrator checkpoint 位于：

- `/root/DermAgent/state/trainable_components/evidence_calibrator/candidates/evidence_calibrator_medium.pt`

外部 `HAM10000` 评测默认走保守模式：

- 若未显式指定 `--agent-risk-only-fallback` 或 `--conservative-fusion`
- `scripts/compare_external_ham10000.py` 会默认启用 `conservative-fusion`

## 项目目录结构

- `/root/DermAgent/agent`：planner、retrieval、aggregator、reflection、evaluation、mining、training、export 等主逻辑
- `/root/DermAgent/skills`：基础 skill、specialist skill、skill registry
- `/root/DermAgent/memory`：experience 存储、检索与读写逻辑
- `/root/DermAgent/cognition`：cognition state 定义与更新逻辑
- `/root/DermAgent/configs`：trainable components、policy 等配置
- `/root/DermAgent/state`：运行期可写状态，包括 experience、cognition、policy、trainable component checkpoints、server pid
- `/root/DermAgent/data`：数据集输入
- `/root/DermAgent/scripts`：主要 CLI 脚本入口
- `/root/DermAgent/outputs`：执行记录、评测结果、checkpoint、analysis、paper export 等输出
- `/root/DermAgent/design`：设计文档
- `/root/DermAgent/tests`：回归测试与流水线测试
