# 6x6 逐条调 Workflow 新对话提示词模板

用法：每次新开对话时，复制下面整段，把 `【...】` 里的空位填上。没想好的地方可以保留“自动选择/待确认”，让新对话先读状态文档后决定。

```text
你现在在 /data/gh/DermAgent 项目中。我要继续按 6 个模型 x 6 个数据集逐条调 workflow，让每个 model×dataset cell 都有自己的可控 workflow，并尽量让 agent 在同批 case 上比 direct baseline 更好。

本次任务填写区：
- 本次要调的模型：`【填写模型名，例如 hulumed / llama / skinvl / medgemma / dermatollama；如果不指定，写“自动选择最值得优先调的差 cell”】`
- 本次要调的数据集：`【填写数据集名，例如 isic2019 / pad20 / ham10000 / scin / sd198 / xiangya_sft；如果不指定，写“自动选择”】`
- 本次目标 cell：`【填写 model x dataset，例如 hulumed x isic2019；如果不指定，写“自动选择”】`
- 本次优化目标：`【例如 top1 必须超过 baseline；或 top1 不退、topk 提升；或 malignant recall 不能掉；或优先修格式稳定性】`
- 本次可接受的取舍：`【例如 top1 赢但 malignant recall 小幅下降是否接受；不接受就写“不接受”】`
- 本次验证规模：`【例如 smoke 5 case -> 50 case -> 80 case；或直接 8 卡 80 case】`
- 本次是否使用 8 卡并行：`【是/否；默认是】`
- 本次模型服务端口：`【例如 8000,8100,8101,8102,8103,8104,8105,8106；如果没有就让 Codex 启动】`
- 本次输出目录前缀：`【例如 outputs/hulumed_isic_workflow_tune_$(date -u +%Y%m%dT%H%M%SZ)；也可以写“由 Codex 命名”】`
- 本次完成后分支名：`【例如 hulumed_isicOK / llama_sd198OK / skinvl_round1 / medgemma_fix1；如果整条模型线完成就用 hulumedOK 等】`
- 其他特别要求：`【例如不要动某些文件、不要停某个正在跑的进程、优先保守融合等】`

当前已知状态：
- qwen 这一整行已经调好，并已保存到 GitHub 分支 `qwenOK`。
- OK 分支是累积迭代 checkpoint，不是彼此独立的平行分支。后续新 OK 分支必须从当前最新 OK 分支继续，包含前面所有已经调好的 workflow。例如调完 qwen 后继续调 dermatollama，最终 `dermatollamaOK` 应包含 `qwenOK` 的全部内容以及 dermatollama 全行的新增 workflow。
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

模型范围：
- qwen
- medgemma
- skinvl
- llama
- hulumed
- dermatollama

数据集范围：
- ham10000
- isic2019
- pad20
- scin
- sd198
- xiangya_sft / xiangya

总体目标：
- 每次只选一个或少数几个效果差的 model×dataset cell 定向调。
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
测试时一定要把8个卡都用上一起跑，保持最快速度
- compare 必须保持 frozen evaluation。
- 不要打开 test writeback。
- 不要污染 test split。
- 不要重置用户已有改动。
- 不要停止正在运行的大实验，除非我明确要求。
- 调参先 smoke，再扩大，除非我在“本次验证规模”里明确要求直接 8 卡完整复检：
  1. 先跑 3-5 case smoke。
  2. smoke 没有格式错误/timeout/明显退化后，跑 30-50 case。
  3. 如果效果好，再跑完整 80 或该数据集可用完整 case。
- 判定“调好”的主标准：同批 case agent top1 > direct baseline top1。
- 同时记录 topk、malignant recall、regression/improvement case 数。
- 如果 top1 赢但 malignant recall 明显掉，要在文档里标注风险，并说明是否符合“本次可接受的取舍”。

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
  - 总 case 数
  - unique case 数
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
4. 根据“本次任务填写区”和 `6x6工作流状态_20260506.md`，确定本次 cell。
5. 分析该 cell 失败原因：
   - baseline 是否已经强
   - agent 是否经常改坏 baseline
   - top1 退化还是 topk 有收益
   - 是否 malignant recall 下降
   - 是否 JSON malformed / timeout / empty final
   - 是否 skill 太多、retrieval 太重、planner 自由度太高
   - 是否 label_space/workflow routing 错误
6. 做最小 workflow 修改。
7. 跑 smoke 或按“本次验证规模”执行。
8. 根据结果决定是否扩大复检。
9. 更新根目录 `6x6工作流状态_20260506.md`。
10. 如果该 cell 或该模型线调好了：
    - git commit
    - 推到 GitHub 新分支；注意新分支必须从当前最新 OK 分支继续，保留前面所有已调好的 workflow，不要从 `master`、旧基线或单独分叉开始
    - 分支名使用“本次完成后分支名”
    - 如果整条模型线 6 个数据集都调好了，就推成：
      - llamaOK
      - hulumedOK
      - medgemmaOK
      - skinvlOK
      - dermatollamaOK

输出要求：
- 先说本次选了哪个 model×dataset cell。
- 说明当前 workflow 是什么。
- 说明失败/退化在哪里。
- 说明改了哪些 workflow 旋钮。
- 给出 smoke/复检结果。
- 给出实际 workflow 分布。
- 给出是否已调好。
- 给出更新了哪个文档。
- 给出 commit hash 和分支名。
```

## 快速填写例子

```text
本次要调的模型：hulumed
本次要调的数据集：isic2019
本次目标 cell：hulumed x isic2019
本次优化目标：top1 必须超过 baseline，同时 malignant recall 不能明显下降
本次可接受的取舍：不接受 top1 回退；不接受 malignant recall 明显下降
本次验证规模：smoke 5 case -> 50 case；如果 50 case 赢，再 8 卡 80 case
本次是否使用 8 卡并行：是
本次模型服务端口：如果已有 hulumed replica 就复用，否则启动 8000,8100,8101,8102,8103,8104,8105,8106
本次输出目录前缀：由 Codex 命名
本次完成后分支名：hulumed_isicOK
其他特别要求：不要动 qwenOK 结果；不要停正在跑的其他模型实验
```
