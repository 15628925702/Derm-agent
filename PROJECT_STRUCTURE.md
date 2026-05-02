# DermAgent 项目结构文档

## 项目概述

DermAgent 是一个基于视觉-语言模型的智能皮肤病诊断代理系统，具有自进化和元学习能力。

## 目录结构

```
DermAgent/
├── agent/                      # 核心代理模块
│   ├── __init__.py
│   ├── aggregator.py          # 结果聚合器
│   ├── batch_reflection.py    # 批量反思机制
│   ├── planner.py             # 任务规划器
│   ├── state.py               # 状态管理
│   ├── execution_record.py    # 执行记录
│   ├── labels.py              # 标签管理
│   ├── confusion_clusters.py  # 混淆聚类分析
│   ├── hard_case_miner.py     # 困难案例挖掘
│   ├── retrieval_scorer.py    # 检索评分器
│   ├── supervised_controller.py # 监督控制器
│   ├── composite_skill_*.py   # 复合技能相关
│   ├── contamination_guard.py # 污染防护
│   └── ...
│
├── cognition/                  # 认知状态模块
│   ├── __init__.py
│   └── cognition_state.py     # 认知状态管理
│
├── configs/                    # 配置模块
│   ├── __init__.py
│   ├── dataset_splits.py      # 数据集划分配置
│   ├── run_profiles.py        # 运行配置
│   └── policies/              # 策略配置
│
├── dataio/                     # 数据输入输出模块
│   ├── __init__.py
│   ├── case_loader.py         # 案例加载器
│   ├── case_schema.py         # 案例数据模式
│   ├── ham10000_loader.py     # HAM10000 数据集加载器
│   ├── isic2019_loader.py     # ISIC2019 数据集加载器
│   ├── pad_ufes_20_loader.py  # PAD-UFES-20 数据集加载器
│   ├── scin_loader.py         # SCIN 数据集加载器
│   ├── sd198_loader.py        # SD-198 数据集加载器
│   └── ...
│
├── integrations/               # 外部集成模块
│   ├── __init__.py
│   ├── openai_client.py       # OpenAI API 客户端
│   └── transformers_vlm_server.py # Transformers VLM 服务器
│
├── memory/                     # 记忆系统模块
│   ├── __init__.py
│   ├── experience_bank.py     # 经验库
│   ├── experience_consolidator.py # 经验整合器
│   ├── experience_retriever.py # 经验检索器
│   ├── experience_schema.py   # 经验数据模式
│   ├── experience_store.py    # 经验存储
│   ├── experience_transform.py # 经验转换
│   └── experience_writer.py   # 经验写入器
│
├── meta_learning/              # 元学习模块
│   ├── __init__.py
│   └── few_shot_adapter.py    # 少样本适配器
│
├── skills/                     # 技能模块
│   ├── __init__.py
│   ├── base.py                # 技能基类
│   ├── catalog.py             # 技能目录
│   ├── border_surface.py      # 边界表面分析
│   ├── color_pattern.py       # 颜色模式分析
│   ├── distribution.py        # 分布分析
│   ├── differential_compare.py # 差异比较
│   ├── contradiction_check.py # 矛盾检查
│   ├── escalation_recommendation.py # 升级建议
│   ├── ack_scc_specialist.py  # AK/SCC 专家技能
│   ├── benign_mimic_specialist.py # 良性模拟专家技能
│   └── ...
│
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── label_canonicalizer.py # 标签规范化
│   ├── external_conservative_fusion.py # 保守融合
│   └── aligned_subset_compare.py # 对齐子集比较
│
├── scripts/                    # 脚本目录
│   ├── start_llama.sh         # 启动 Llama 模型
│   ├── start_dermatollama.sh  # 启动 DermatoLlama 模型
│   ├── start_hulumed.sh       # 启动 Hulu-Med 模型
│   ├── run_dataset_final_round.sh # 运行数据集最终轮次
│   └── ...
│
├── state/                      # 状态存储目录
│   ├── dataset_adaptation/    # 数据集适配状态
│   ├── policy/                # 策略状态
│   ├── split_states/          # 数据划分状态
│   └── trainable_components/  # 可训练组件状态
│
├── tests/                      # 测试目录
│   └── ...
│
├── design/                     # 设计文档
│   ├── experiments/           # 实验设计
│   └── skill_specs/           # 技能规范
│
├── paper/                      # 论文相关
│   ├── 参考文献/
│   ├── 论文各部分/
│   ├── figures/
│   └── scripts/
│
├── outputs/                    # 输出目录
│   ├── final_round/           # 最终轮次输出
│   ├── final_splits/          # 最终数据划分
│   └── ...
│
├── logs/                       # 日志目录
│
├── archive/                    # 归档目录（旧版本）
│   ├── v3-初步多数据集/
│   ├── v4继续多数据集改进/
│   ├── v7-scin数据集/
│   └── workflow改进/
│
├── data -> /root/mydata/data  # 数据目录（软链接）
│
├── setup.py                    # 安装配置
├── pyproject.toml             # 项目配置
├── requirements.txt           # 依赖列表
├── environment.yml            # Conda 环境配置
├── README.md                  # 项目说明
├── .gitignore                 # Git 忽略配置
├── 使用规范.md                # 使用规范
├── 当前最佳版本说明.md        # 版本说明
└── DermAgent自进化机制详解.md # 自进化机制文档
```

## 核心模块说明

### 1. Agent 模块 (`agent/`)
核心代理系统，包含：
- **规划器 (Planner)**: 任务分解和执行规划
- **聚合器 (Aggregator)**: 多技能结果聚合
- **反思机制 (Batch Reflection)**: 批量案例反思和改进
- **状态管理 (State)**: 代理状态跟踪
- **困难案例挖掘 (Hard Case Miner)**: 识别和分析困难案例
- **混淆聚类 (Confusion Clusters)**: 混淆案例聚类分析

### 2. Skills 模块 (`skills/`)
诊断技能库，包含：
- **基础技能**: 边界、颜色、分布分析
- **推理技能**: 差异比较、矛盾检查
- **专家技能**: 特定疾病专家（AK/SCC、良性模拟等）
- **风险评估**: 升级建议

### 3. Memory 模块 (`memory/`)
经验记忆系统，包含：
- **经验库 (Experience Bank)**: 存储历史诊断经验
- **经验检索 (Experience Retriever)**: 检索相似案例
- **经验整合 (Experience Consolidator)**: 整合和优化经验
- **经验转换 (Experience Transform)**: 经验格式转换

### 4. DataIO 模块 (`dataio/`)
数据加载和处理，支持多个数据集：
- HAM10000
- ISIC2019
- PAD-UFES-20
- SCIN
- SD-198
- Xiangya-SFT

### 5. Integrations 模块 (`integrations/`)
外部模型集成：
- **OpenAI Client**: 统一的 VLM 调用接口
- **Transformers VLM Server**: 本地 VLM 服务器

### 6. Meta Learning 模块 (`meta_learning/`)
元学习能力：
- **Few-shot Adapter**: 少样本学习适配

### 7. Cognition 模块 (`cognition/`)
认知状态管理：
- 跟踪代理的认知状态和决策过程

## 数据流

```
数据集 (data/)
    ↓
DataIO 加载器 (dataio/)
    ↓
Agent 规划器 (agent/planner.py)
    ↓
Skills 执行 (skills/)
    ↓
Memory 检索 (memory/)
    ↓
Agent 聚合器 (agent/aggregator.py)
    ↓
结果输出 (outputs/)
    ↓
反思和改进 (agent/batch_reflection.py)
    ↓
经验存储 (memory/experience_store.py)
```

## 安装和使用

### 安装
```bash
# 开发模式安装
pip install -e .

# 或使用 conda 环境
conda env create -f environment.yml
conda activate derm-qwen
```

### 使用
```bash
# 启动 VLM 服务器
bash scripts/start_llama.sh /root/DermAgent

# 运行诊断
python -m agent.main --config configs/run_profiles.py
```

## 开发规范

### 代码风格
- 使用 Black 格式化代码（行长度 100）
- 使用 isort 排序导入
- 遵循 PEP 8 规范

### 测试
```bash
# 运行测试
pytest tests/

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

### 提交规范
- 遵循 Conventional Commits 规范
- 每次提交前运行测试

## 扩展指南

### 添加新数据集
1. 在 `dataio/` 创建新的 loader 和 schema
2. 在 `configs/dataset_splits.py` 添加配置
3. 更新 `state/dataset_adaptation/` 添加适配状态

### 添加新技能
1. 在 `skills/` 创建新技能类，继承 `base.py`
2. 在 `skills/catalog.py` 注册新技能
3. 在 `design/skill_specs/` 添加技能规范

### 添加新模型
1. 在 `integrations/` 添加模型集成代码
2. 在 `scripts/` 创建启动脚本
3. 更新 `integrations/openai_client.py` 支持新模型

## 许可证

MIT License

## 联系方式

- 项目主页: https://github.com/yourusername/DermAgent
- 问题反馈: https://github.com/yourusername/DermAgent/issues
