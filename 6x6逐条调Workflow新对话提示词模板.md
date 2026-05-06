# 6x6 逐条调 Workflow 新对话提示词模板

下面这段提示词用于后续每次新开对话时复用。复制整段给 Codex，让它按当前 6x6 架构继续逐个调 `model x dataset` workflow。

```text
你现在在 /data/gh/DermAgent 项目中。我要继续按 6 个模型 x 6 个数据集逐条调 workflow，让每个 model×dataset cell 都有自己的可控 workflow，并尽量让 agent 在同批 case 上比 direct baseline 更好。

当前已知状态：
- qwen 这一整行已经调好，并已保存到 GitHub 分支 qwenOK。
- 当前架构已经支持 model×dataset 专属 workflow cell。
- workflow 优先级是：
  1. model×dataset override
  2. model override
  3. dataset workflow
  4. default workflow
- 主要实现位置：
  - agent/model_workflow_router.py
  - agent/workflow_profiles.py
  - agent/planner.py
  - agent/conservative_fusion.py
  - scripts/compare_agent_vs_qwen.py
- 当前中文状态文档在项目根目录：
  - 项目索引.md
  - 6x6工作流状态_20260506.md

模型：
- qwen
- medgemma
- skinvl
- llama
- hulumed
- dermatollama

数据集：
- ham10000
- isic2019
- pad20
- scin
- sd198
- xiangya_sft / xiangya

总体目标：
- 以后每次只选一个或少数几个效果差的 model×dataset cell 定向调。
- 不要泛泛重构。
- 优先改 workflow 的可控旋钮：
  - workflow_profile
  - workflow_capabilities
  - allowed_skills
  - force_enable_skills
  - force_disable_skills
  - disabled_specialist_skills
  - skip_specialist_skills
  - skip_experience_retrieval
  - enable_experience_retrieval
  - enable_skill_retrieval
  - conservative_fusion_weight
  - force_conservative_fusion
  - fallback_on_malformed_final
  - disable_legacy_final_path
- 如果必须新增 skill，只新增小而明确的 skill，并注册到现有 skill registry。

运行要求：
- compare 必须保持 frozen evaluation。
- 不要打开 test writeback。
- 不要污染 test split。
- 不要重置用户已有改动。
- 不要停止正在运行的大实验，除非我明确要求。
- 调参先 smoke，再扩大：
  1. 先跑 3-5 case smoke。
  2. smoke 没有格式错误/timeout/明显退化后，跑 30-50 case。
  3. 如果效果好，再跑完整 80 或该数据集可用完整 case。
- 判定“调好”的主标准：同批 case agent top1 > direct baseline top1。
- 同时记录 topk、malignant recall、regression/improvement case 数。
- 如果 top1 赢但 malignant recall 明显掉，要在文档里标注风险。

8 卡加速策略：
- 如果要调某一个模型，就尽量启动 8 个该模型 replica，占用 8 张卡。
- 每个 replica 一个 OpenAI-compatible 端口，例如：
  - 8000
  - 8100
  - 8101
  - 8102
  - 8103
  - 8104
  - 8105
  - 8106
- 如果已有该模型 replica，就复用；缺几个就补几个。
- 完整 80 case 可以切成 8 个 shard：
  - offset 0 limit 10
  - offset 10 limit 10
  - offset 20 limit 10
  - offset 30 limit 10
  - offset 40 limit 10
  - offset 50 limit 10
  - offset 60 limit 10
  - offset 70 limit 10
- 8 个 shard 分别打到 8 个端口。
- 跑完后必须聚合 8 个 shard 的 summary 和 records，确认：
  - 总 case = 80
  - unique case = 80
  - baseline top1/topk/malignant recall
  - agent top1/topk/malignant recall
  - workflow distribution
  - label_space distribution
  - model_profile distribution
  - workflow_cell distribution
  - agent helped / hurt / unchanged case 数

本次进入新对话后请先做：
1. 阅读根目录的：
   - 项目索引.md
   - 6x6工作流状态_20260506.md
2. 阅读关键代码：
   - agent/model_workflow_router.py
   - agent/workflow_profiles.py
   - agent/planner.py
   - agent/conservative_fusion.py
   - scripts/compare_agent_vs_qwen.py
3. 检查当前 git 分支和状态。
4. 根据 6x6工作流状态_20260506.md，选我指定的 cell 或自动挑一个最值得优先调的差 cell。
5. 分析该 cell 失败原因：
   - baseline 是否已经强
   - agent 是否经常改坏 baseline
   - top1 退化还是 topk 有收益
   - 是否 malignant recall 下降
   - 是否 JSON malformed / timeout / empty final
   - 是否 skill 太多、retrieval 太重、planner 自由度太高
   - 是否 label_space/workflow routing 错误
6. 做最小 workflow 修改。
7. 跑 smoke。
8. 根据结果决定是否跑 30-50 或 80 case。
9. 更新根目录 6x6工作流状态_20260506.md。
10. 如果该 cell 或该模型线调好了：
    - git commit
    - 推到 GitHub 新分支
    - 分支命名清晰，例如：
      - llama_isicOK
      - hulumed_pad20OK
      - medgemma_fix1
      - skinvl_round1
      - dermatollama_pad20OK
    - 如果整条模型线 6 个数据集都调好了，就推成：
      - llamaOK
      - hulumedOK
      - medgemmaOK
      - skinvlOK
      - dermatollamaOK

输出要求：
- 先说你选了哪个 model×dataset cell。
- 说明当前 workflow 是什么。
- 说明失败/退化在哪里。
- 说明改了哪些 workflow 旋钮。
- 给出 smoke/复检结果。
- 给出实际 workflow 分布。
- 给出是否已调好。
- 给出 commit hash 和分支名。
```
