# DermAgent 自进化 Workflow 说明

本目录用于离线生成、审核和启用工作流配置提案。

默认情况下，提案不会自动进入运行时；只有在人工审核通过并显式启用后，相关配置才会生效。

## 基本流程

1. 收集比较报告与复核材料
2. 生成候选提案
3. 人工审核提案内容
4. 通过后写入 approved 目录
5. 在需要时显式启用

## 目录说明

- `generate_proposal.py`：根据输入报告生成提案
- `proposal_generator.py`：提案构造逻辑
- `apply_proposal.py`：审核与应用提案
- `runtime.py`：运行时可选加载 approved 提案
- `doctor_experience_template.md`：医生或审核者可填写的辅助说明模板
