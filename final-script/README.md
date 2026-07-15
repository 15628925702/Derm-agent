# Final Script 最终执行手册

这个目录不是通用开发文档，而是“最终实验执行清单”。

它的目标只有一个：
- 把最后要跑的实验按主次顺序整理清楚
- 把固定资产路径、输出目录、运行脚本收束到统一入口
- 避免在最终阶段混用不同模型线的状态、经验库、policy 或 checkpoint

所有最终实验输出默认写到：
- `final-score/`

## 一句话总览

真正主线只有一个：

- 本数据集上 `direct Qwen` vs `agent + Qwen`

其余内容都属于次级实验，用于补充分析、泛化验证或论文支撑：

- `Qwen + agent` 在本数据集上的消融
- `Qwen` 与 `Qwen + agent` 在外部数据集上的泛化
- `agent` 在 `MedGemma` 上的泛化
- `agent` 在 `SkinVL` 上的泛化
- `Qwen + agent` 在本数据集上的 paired statistics
- `Qwen + agent` 在本数据集上的 qualitative case study

所以不要把 `MedGemma` 和 `SkinVL` 这两条线理解成和 `Qwen` 主线同等级的主结果线。

## 实验优先级

### P0：唯一主线

1. 本数据集上 `direct Qwen` vs `agent + Qwen`

### P1：主线补充分析

2. `Qwen + agent` 的消融实验
3. `Qwen` 与 `Qwen + agent` 在外部数据集上的泛化
4. `Qwen + agent` 在本数据集上的 paired statistics
5. `Qwen + agent` 在本数据集上的 qualitative case study

### P2：模型泛化实验

6. `agent` 在 `MedGemma` 上的泛化
7. `agent` 在 `SkinVL` 上的泛化

这两个模型当前更适合作为：
- 泛化验证线
- exploratory 结果线

而不是主论文结果线。

## 先看总原则

### 1. 不同模型线绝对不能混状态

三条模型线：

- `Qwen`
- `MedGemma`
- `SkinVL`

必须分别使用各自独立的：

- model service
- `policy_root`
- `split_state_root`
- bootstrap / train artifacts
- final comparison output root

当前约定为：

- `Qwen`
  - `state/policy`
  - `state/split_states`
- `MedGemma`
  - `state/policy_medgemma`
  - `state_medgemma/split_states`
- `SkinVL`
  - `state/policy_skinvl`
  - `state_skinvl/split_states`

从当前版本开始，这个分仓约束是“端到端生效”的，而不只是最终 compare 时生效：

- `run_agent.py` 会按 `DERMAGENT_SPLIT_STATE_ROOT` 读取 split-aware state
- cognition writeback 也会写回当前 split-aware 根目录
- `debug_single_case.py`、bootstrap 脚本和其他底层入口，只要触发写回，都会真正落到当前环境变量指向的仓

所以老习惯里那种“先随手跑底层 bootstrap，反正 compare 时会切仓”的做法现在不能再依赖了。底层写回本身就会进入当前仓。

除了资产分仓之外，后续如果接入新数据集，还要把“标签空间”当成独立配置来管理：

- 不同数据集可以保留各自原始标签
- 不同数据集可以注册自己的 `label_space_id`
- 经验库、训练产物和评测结果应该与对应 `label_space` 保持一致

也就是说，未来新数据集实验线不只是：

- 独立 `policy_root`
- 独立 `split_state_root`

还应该是：

- 独立 `label_space_id`
- 独立数据集 loader / schema

### 2. 不要同时跑多个模型服务

推荐一次只保留一条模型线服务活着：

- 跑 `Qwen` 时，停掉 `MedGemma` / `SkinVL`
- 跑 `MedGemma` 时，停掉 `Qwen` / `SkinVL`
- 跑 `SkinVL` 时，停掉 `Qwen` / `MedGemma`

### 3. 服务必须在持久终端里运行

不要：

- 启动完服务后立刻关窗口
- 在会自动回收后台进程的临时会话里启动

推荐：

- 一个终端专门跑服务
- 一个终端专门跑实验

### 4. final-script 只负责最终实验，不替代底层脚本

底层训练/评测逻辑仍在：

- `scripts/`

`final-script/` 只做三件事：

- 固定资产路径
- 固定输出目录
- 固定最终实验入口

补充一条非常重要的操作规则：

- 用 `final-script/` 官方入口跑旧实验时，命令本身通常不用改
- 只要你要手动跑 `scripts/` 里的底层命令，就先 `source` 对应模型线的 `.env`
- 不要把 `Qwen` 线 shell 里加载过的根目录，继续拿去跑 `SkinVL` / `MedGemma` / 新数据集 bootstrap
- 如果是新数据集实验，除了切资产根目录，还要先确认该数据集的 `label_space` 已经注册，不要默认复用旧主线标签解释

## 目录说明

### 配置

- `final-script/configs/common.env`
  - 公共路径、默认输出目录、默认 split

- `final-script/configs/final_assets_registry.env`
  - 记录 Qwen / MedGemma / SkinVL 三条线使用的固定资产路径

- `final-script/configs/qwen_final.env`
  - Qwen 最终实验环境配置

- `final-script/configs/medgemma_final.env`
  - MedGemma 最终实验环境配置

- `final-script/configs/skinvl_final.env`
  - SkinVL 最终实验环境配置

### 服务脚本

- `final-script/servers/start_qwen_final.sh`
- `final-script/servers/start_medgemma_final.sh`
- `final-script/servers/start_skinvl_final.sh`

### 运行脚本

- `final-script/runs/run_main_qwen_vs_agent_qwen.sh`
- `final-script/runs/run_qwen_ablations.sh`
- `final-script/runs/run_external_qwen_vs_agent_qwen.sh`
- `final-script/runs/run_main_medgemma_vs_agent_medgemma.sh`
- `final-script/runs/run_main_skinvl_vs_agent_skinvl.sh`
- `final-script/runs/run_paired_statistics.sh`
- `final-script/runs/run_qualitative_case_study.sh`

### 导出脚本

- `final-script/exports/export_tables.sh`
- `final-script/exports/export_figures.sh`

## 推荐执行顺序

建议严格按下面顺序执行：

1. 启动 `Qwen` 服务
2. 跑本数据集上的 `direct Qwen` vs `agent + Qwen`
3. 跑 `Qwen + agent` 消融
4. 跑 `Qwen` 外部数据集泛化
5. 跑 paired statistics
6. 跑 qualitative case study
7. 停掉 `Qwen`
8. 启动 `MedGemma`
9. 跑 `MedGemma` 泛化线
10. 停掉 `MedGemma`
11. 启动 `SkinVL`
12. 跑 `SkinVL` 泛化线
13. 导出 paper tables / figures

## 一、P0 主线：本数据集上 Qwen vs agent+Qwen

这是唯一主线，也是优先级最高的实验。

### 1. 启动 Qwen 服务

```bash
cd /root/DermAgent
bash final-script/servers/start_qwen_final.sh
```

### 2. 跑主线 compare

```bash
cd /root/DermAgent
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh
```

这里的旧主线命令保持不变。因为：

- `final-script/configs/qwen_final.env`
- `final-script/configs/common.env`

已经把 `DERMAGENT_POLICY_ROOT` 和 `DERMAGENT_SPLIT_STATE_ROOT` 固定到了主线资产仓。

常用入口：

```bash
cd /root/DermAgent
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --smoke-10
```

```bash
cd /root/DermAgent
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --medium-40
```

```bash
cd /root/DermAgent
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --full-test
```

也支持显式控制：

```bash
cd /root/DermAgent
bash final-script/runs/run_main_qwen_vs_agent_qwen.sh --limit 10 --case-offset 171
```

### 3. 当前主线固定资产

默认主线资产：

- policy id: `mainline_parameterized_agent_20260331`
- learned controller / retrieval 复用自主线稳定资产
- frozen split-state 位于 `final-script/assets/qwen_mainline/`

如果你要在主线环境下手动运行底层脚本，推荐先显式进入这条线的环境：

```bash
cd /root/DermAgent
source final-script/configs/qwen_final.env
```

然后再执行：

- `python scripts/debug_single_case.py ...`
- `python scripts/evaluate_policy_candidate.py ...`
- 任何会写经验库或 cognition 的 bootstrap / debug 命令

### 4. 进度输出

会看到：

- `progress 0/3`
- `progress 1/3`
- `progress 2/3`
- `progress 3/3`

底层还会继续打印逐 case：

- `progress baseline i/N`
- `progress agent i/N`

## 二、P1 补充分析

### 1. Qwen 消融

```bash
cd /root/DermAgent
bash final-script/runs/run_qwen_ablations.sh
```

### 2. Qwen 外部数据集泛化

HAM10000：

```bash
cd /root/DermAgent
EXTERNAL_DATASET=ham10000 bash final-script/runs/run_external_qwen_vs_agent_qwen.sh
```

ISIC2019：

```bash
cd /root/DermAgent
EXTERNAL_DATASET=isic2019 bash final-script/runs/run_external_qwen_vs_agent_qwen.sh
```

### 3. paired statistics

```bash
cd /root/DermAgent
bash final-script/runs/run_paired_statistics.sh
```

### 4. qualitative case study

```bash
cd /root/DermAgent
bash final-script/runs/run_qualitative_case_study.sh
```

## 三、P2 泛化线：MedGemma

这条线用于验证 agent 设计是否能迁移到另一种基础模型，不是主线。

### 1. 启动 MedGemma 服务

```bash
cd /root/DermAgent
bash final-script/servers/start_medgemma_final.sh
```

### 2. 跑 MedGemma 泛化 compare

```bash
cd /root/DermAgent
bash final-script/runs/run_main_medgemma_vs_agent_medgemma.sh
```

如果只是做少量 smoke，可以在底层显式传：

```bash
cd /root/DermAgent
LIMIT=5 bash final-script/runs/run_main_medgemma_vs_agent_medgemma.sh
```

## 四、P2 泛化线：SkinVL

这条线同样是泛化验证线，不是主线。

### 1. 启动 SkinVL 服务

推荐更保守的启动：

```bash
cd /root/DermAgent
MAX_NEW_TOKENS_DEFAULT=256 bash final-script/servers/start_skinvl_final.sh
```

### 2. 跑 SkinVL 泛化 compare

```bash
cd /root/DermAgent
bash final-script/runs/run_main_skinvl_vs_agent_skinvl.sh --smoke-10
```

也可以：

```bash
cd /root/DermAgent
bash final-script/runs/run_main_skinvl_vs_agent_skinvl.sh --medium-40
```

### 3. SkinVL 独立 bootstrap / train / compare

如果继续做 `SkinVL+agent` 训练，必须保持完全隔离。

现在这里的“完全隔离”不只是 compare 阶段的要求，而是 bootstrap 和 writeback 阶段也必须做到。换句话说：

- 先设好 `DERMAGENT_POLICY_ROOT`
- 再设好 `DERMAGENT_SPLIT_STATE_ROOT`
- 然后再跑任何底层写回脚本

不要先在默认主仓里做 bootstrap，之后才切到 `SkinVL` 仓做 compare。

补后 24 case bootstrap：

```bash
cd /root/DermAgent

for i in $(seq 24 47); do
  echo "[bootstrap] case-index=${i}"
  DERMAGENT_POLICY_ROOT=/root/DermAgent/state/policy_skinvl \
  DERMAGENT_SPLIT_STATE_ROOT=/root/DermAgent/state_skinvl/split_states \
  OPENAI_BASE_URL=http://127.0.0.1:8011/v1 \
  OPENAI_API_KEY=EMPTY \
  OPENAI_MODEL=SkinVL-MM \
  python scripts/debug_single_case.py \
    --case-index "${i}" \
    --data-root /root/DermAgent/data/pad_ufes_20 \
    --output-dir /root/DermAgent/outputs/skinvl_bootstrap \
    --enable-writeback \
    --data-split train \
    --run-mode skinvl_train_bootstrap
done
```

训练：

```bash
cd /root/DermAgent

DERMAGENT_POLICY_ROOT=/root/DermAgent/state/policy_skinvl \
DERMAGENT_SPLIT_STATE_ROOT=/root/DermAgent/state_skinvl/split_states \
OPENAI_BASE_URL=http://127.0.0.1:8011/v1 \
OPENAI_API_KEY=EMPTY \
OPENAI_MODEL=SkinVL-MM \
python scripts/train_learned_components.py \
  --stages 0,1,2,3,4 \
  --run-id skinvl_learned_48_v1 \
  --records-root /root/DermAgent/outputs/skinvl_bootstrap \
  --data-root /root/DermAgent/data \
  --split-json /root/DermAgent/outputs/pad_ufes_20_split.json \
  --training-split train \
  --stage3-data-split val \
  --limit 48 \
  --epochs 8 \
  --output-dir /root/DermAgent/outputs/train_runs \
  --checkpoint-out-dir /root/DermAgent/outputs/checkpoints \
  --client-timeout 180 \
  --client-max-retries 2
```

训练后的 10 case compare：

```bash
cd /root/DermAgent

env \
  OPENAI_BASE_URL=http://127.0.0.1:8011/v1 \
  OPENAI_API_KEY=EMPTY \
  OPENAI_MODEL=SkinVL-MM \
  DERMAGENT_POLICY_ROOT=/root/DermAgent/state/policy_skinvl \
  DERMAGENT_SPLIT_STATE_ROOT=/root/DermAgent/state_skinvl/split_states \
  python scripts/compare_agent_vs_qwen.py \
    --data-root /root/DermAgent/data \
    --split-json /root/DermAgent/outputs/pad_ufes_20_split.json \
    --data-split test \
    --output-dir /root/DermAgent/final-score/final_runs/main_skinvl_vs_agent_skinvl \
    --policy-config /root/DermAgent/outputs/train_runs/skinvl_learned_48_v1/stage3_policy_candidate_evaluation/candidate_policy.json \
    --policy-label "SkinVL learned-48 candidate policy" \
    --limit 10
```

### 4. SkinVL 当前定位

当前更建议把 `SkinVL` 理解为：

- 一个 direct baseline 候选
- 一个 exploratory agent 泛化线

而不是和 `Qwen` 主线并列的最终主结果线。

## 五、导出

跑完主要实验和补充分析后，再导出 paper 产物：

```bash
cd /root/DermAgent
bash final-script/exports/export_tables.sh
bash final-script/exports/export_figures.sh
```

## 最后提醒

这份目录里最容易混淆的一点就是：

- `Qwen` 主线是唯一主结果
- `MedGemma` 和 `SkinVL` 都是泛化线

所以如果时间或资源有限，优先顺序永远是：

1. `Qwen` 主线
2. `Qwen` 补充分析
3. `MedGemma` / `SkinVL` 泛化线

再补一条操作层面的建议：

- 旧实验官方入口尽量继续走 `final-script/`
- 任何新数据集适配、手动 bootstrap、单病例写回调试，不要复用旧主线资产仓
- 如果要新开一条独立实验线，优先使用 [`scripts/manage_dataset_experiment_assets.py`](/root/DermAgent/scripts/manage_dataset_experiment_assets.py) 初始化隔离资产
- 新数据集实验线除了隔离资产，还应明确记录自己的 `label_space_id`，避免后续经验 consolidation、评测和导表时标签解释混乱
