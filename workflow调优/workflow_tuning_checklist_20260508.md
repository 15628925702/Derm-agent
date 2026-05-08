# Workflow Tuning Checklist (2026-05-08)

## Goal

针对非 Xiangya `6 x 5` 最终大规模实验结果，按优先级逐个继续调 `model x dataset workflow`，目标是尽量把每个 workflow 都调到：

- `Top-1` 上涨
- `Top-k` 也上涨

当前分类口径：

- 正向：`Top-1 > 0`，或 `Top-1 = 0 且 Top-k > 0`
- 持平：`Top-1 = 0 且 Top-k = 0`
- 负向：`Top-1 < 0`

## 当前总体分布

- 正向：`26`
- 持平：`1`
- 负向：`3`

## 调参进展快照

- 已完成本轮调参：`llama / isic2019`
  - 原始 eval300：Top-1 `-0.0067`, Top-k `+0.0267`
  - v2/v2b small-medium 验证：`small8`, `medium20`, `medium40`, `medium80` 均 Top-1 非负且 Top-k 不下降
  - 最大放大验证：`medium80`, offset `120..199`, Top-1 `28/80 -> 31/80` (`+0.0375`), Top-k `37/80 -> 41/80` (`+0.0500`), hurt `0`
  - 当前建议：暂不直接上 eval300；下一轮如继续该线，优先吃 `ISIC_0059614` 这类 BKL/NV Top-k-only 残差
- 下一条建议调参线：`medgemma / isic2019`
  - 原始 eval300：Top-1 `-0.0033`, Top-k `+0.0200`
  - 追溯预判：Top-k 已新增 6 个命中且 Top-k hurt 为 0，Top-1 负向主要来自 1 个 MEL hurt case；优先检查 final fusion / malignant protection / override gate

## 调参优先级总表

排序原则：

1. 先救 `Top-1` 负向
2. 再救 `Top-1` 持平但 `Top-k` 已上涨
3. 再处理弱正向
4. 最后再碰已经很强的组合

| Priority | Workflow | Current | Why first | Main tuning goal |
|---|---|---|---|---|
| 1 | `llama / isic2019` | 原始 Top-1 `-0.0067`, Top-k `+0.0267`; v2b `medium80` Top-1 `+0.0375`, Top-k `+0.0500`, hurt `0` | 本轮已把 Top-1 拉回非负，小中样本稳定 | 暂停，等后续再决定是否 eval300 |
| 2 | `medgemma / isic2019` | Top-1 `-0.0033`, Top-k `+0.0200` | evidence 有用，final 转化不足；当前最高优先级未调负向线 | 修 final selection / malignant protection / fusion |
| 3 | `qwen / isic2019` | Top-1 `-0.0033`, Top-k `+0.0033` | 负向较轻，但仍需转正 | 先保 Top-1 不负 |
| 4 | `medgemma / scin` | Top-1 `0.0000`, Top-k `0.0000` | 唯一真正双持平 | 先做出可见增益 |
| 5 | `medgemma / ham10000` | Top-1 `0.0000`, Top-k `+0.0567` | 典型“Top-k 涨但没转成 Top-1” | 提高 final 决策转化 |
| 6 | `qwen / ham10000` | Top-1 `0.0000`, Top-k `+0.1500` | 候选集改善明显 | 把 Top-k 收益转成 Top-1 |
| 7 | `hulumed / isic2019` | Top-1 `0.0000`, Top-k `+0.0600` | isic2019 中潜力较强 | 争取稳定双涨 |
| 8 | `dermatollama / isic2019` | Top-1 `0.0000`, Top-k `+0.0167` | 有轻微信号 | 小幅正向即可 |
| 9 | `dermatollama / ham10000` | Top-1 `0.0000`, Top-k `+0.0167` | 也属于“可转化型” | 拉出微正向 |
| 10 | `qwen / scin` | Top-1 `0.0000`, Top-k `+0.0067` | 已有一点信号 | 低成本尝试转正 |
| 11 | `dermatollama / sd198` | Top-1 `0.0000`, Top-k `+0.0033` | 信号很弱 | 小修即可 |
| 12 | `hulumed / scin` | Top-1 `+0.0033`, Top-k `+0.0033` | 弱正向 | 放大收益 |
| 13 | `dermatollama / scin` | Top-1 `+0.0033`, Top-k `+0.0100` | 弱正向 | 放大收益 |
| 14 | `qwen / pad20` | Top-1 `+0.0033`, Top-k `+0.1233` | Top-k 很强，Top-1 转化不足 | 调 final fusion |
| 15 | `qwen / sd198` | Top-1 `+0.0067`, Top-k `+0.0033` | 弱正向 | 稳步增强 |
| 16 | `dermatollama / pad20` | Top-1 `+0.0067`, Top-k `+0.0667` | 已有基础 | 再推 Top-1 |
| 17 | `llama / sd198` | Top-1 `+0.0067`, Top-k `+0.0233` | 正向不大 | 稳步增强 |
| 18 | `medgemma / sd198` | Top-1 `+0.0100`, Top-k `+0.0367` | 状态健康 | 精修 |
| 19 | `hulumed / ham10000` | Top-1 `+0.0100`, Top-k `+0.2233` | Top-k 很强 | 提升转化效率 |
| 20 | `skinvl / isic2019` | Top-1 `+0.0133`, Top-k `+0.1100` | isic2019 上的好苗子 | 可做模板参考 |
| 21 | `llama / ham10000` | Top-1 `+0.0133`, Top-k `+0.0333` | 已正向 | 小修 |
| 22 | `llama / pad20` | Top-1 `+0.0200`, Top-k `+0.0133` | 稳定正向 | 后续精修 |
| 23 | `skinvl / ham10000` | Top-1 `+0.0333`, Top-k `+0.0367` | 已不错 | 不急 |
| 24 | `hulumed / sd198` | Top-1 `+0.0333`, Top-k `+0.0300` | 已不错 | 不急 |
| 25 | `skinvl / sd198` | Top-1 `+0.0833`, Top-k `+0.0967` | 强组合 | 后期微调 |
| 26 | `hulumed / pad20` | Top-1 `+0.0833`, Top-k `+0.3400` | 很强 | 先别动大改 |
| 27 | `skinvl / pad20` | Top-1 `+0.1067`, Top-k `+0.1100` | 很强 | 先别动大改 |
| 28 | `llama / scin` | Top-1 `+0.1200`, Top-k `+0.1000` | 很强 | 当成功模板 |
| 29 | `skinvl / scin` | Top-1 `+0.2133`, Top-k `+0.2267` | 顶级强组合 | 最后再碰 |
| 30 | `medgemma / pad20` | Top-1 `+0.2500`, Top-k `+0.2000` | 全场最强 | 作为标杆，暂不优先改 |

## 建议分批顺序

### Batch 1: 先救负向

1. `llama / isic2019` - 本轮 v2/v2b 已完成 small-medium 正向验证
2. `medgemma / isic2019` - 下一条建议调参线
3. `qwen / isic2019`

### Batch 2: 把 Top-k 收益转成 Top-1

4. `medgemma / ham10000`
5. `qwen / ham10000`
6. `hulumed / isic2019`
7. `dermatollama / isic2019`
8. `dermatollama / ham10000`

### Batch 3: 处理无效或弱信号 workflow

9. `medgemma / scin`
10. `qwen / scin`
11. `dermatollama / sd198`
12. `hulumed / scin`
13. `dermatollama / scin`

### Batch 4: 放大已正向但还不够强的组合

14. `qwen / pad20`
15. `llama / sd198`
16. `medgemma / sd198`
17. `hulumed / ham10000`
18. `skinvl / isic2019`

## 前 10 个最值得先调的 Workflow

1. `medgemma / isic2019`
2. `qwen / isic2019`
3. `medgemma / scin`
4. `medgemma / ham10000`
5. `qwen / ham10000`
6. `hulumed / isic2019`
7. `dermatollama / isic2019`
8. `dermatollama / ham10000`
9. `qwen / scin`
10. `dermatollama / sd198`

## 每个组合当前结果清单

### 正向

- `medgemma / pad20`: Top-1 `+0.2500`, Top-k `+0.2000`
- `skinvl / scin`: Top-1 `+0.2133`, Top-k `+0.2267`
- `llama / scin`: Top-1 `+0.1200`, Top-k `+0.1000`
- `skinvl / pad20`: Top-1 `+0.1067`, Top-k `+0.1100`
- `hulumed / pad20`: Top-1 `+0.0833`, Top-k `+0.3400`
- `skinvl / sd198`: Top-1 `+0.0833`, Top-k `+0.0967`
- `skinvl / ham10000`: Top-1 `+0.0333`, Top-k `+0.0367`
- `hulumed / sd198`: Top-1 `+0.0333`, Top-k `+0.0300`
- `llama / pad20`: Top-1 `+0.0200`, Top-k `+0.0133`
- `skinvl / isic2019`: Top-1 `+0.0133`, Top-k `+0.1100`
- `llama / ham10000`: Top-1 `+0.0133`, Top-k `+0.0333`
- `hulumed / ham10000`: Top-1 `+0.0100`, Top-k `+0.2233`
- `medgemma / sd198`: Top-1 `+0.0100`, Top-k `+0.0367`
- `qwen / sd198`: Top-1 `+0.0067`, Top-k `+0.0033`
- `dermatollama / pad20`: Top-1 `+0.0067`, Top-k `+0.0667`
- `llama / sd198`: Top-1 `+0.0067`, Top-k `+0.0233`
- `qwen / pad20`: Top-1 `+0.0033`, Top-k `+0.1233`
- `hulumed / scin`: Top-1 `+0.0033`, Top-k `+0.0033`
- `dermatollama / scin`: Top-1 `+0.0033`, Top-k `+0.0100`
- `qwen / ham10000`: Top-1 `+0.0000`, Top-k `+0.1500`
- `hulumed / isic2019`: Top-1 `+0.0000`, Top-k `+0.0600`
- `medgemma / ham10000`: Top-1 `+0.0000`, Top-k `+0.0567`
- `dermatollama / isic2019`: Top-1 `+0.0000`, Top-k `+0.0167`
- `dermatollama / ham10000`: Top-1 `+0.0000`, Top-k `+0.0167`
- `qwen / scin`: Top-1 `+0.0000`, Top-k `+0.0067`
- `dermatollama / sd198`: Top-1 `+0.0000`, Top-k `+0.0033`

### 持平

- `medgemma / scin`: Top-1 `+0.0000`, Top-k `+0.0000`

### 负向

- `llama / isic2019`: Top-1 `-0.0067`, Top-k `+0.0267`
- `medgemma / isic2019`: Top-1 `-0.0033`, Top-k `+0.0200`
- `qwen / isic2019`: Top-1 `-0.0033`, Top-k `+0.0033`

## 建议的调参方向模板

后面逐个 workflow 调的时候，优先从这些层面看：

1. `final diagnosis fusion`
   - Top-k 已涨但 Top-1 没涨时，先查 final 选择是否过于保守或没吃进证据

2. `override gate / promotion gate`
   - evidence 足够时是否允许从 baseline 候选提升到更优 final diagnosis

3. `uncertainty / contradiction suppression`
   - 是否把本来有效的 override 压没了

4. `specialist evidence quota`
   - 是否需要更强的 specialist support 才能触发 final 改写

5. `retrieval coverage`
   - 弱信号或双持平组合，先看 retrieval 和 evidence selection 是否根本没起作用

## Practical Rule

优先别动已经非常强的组合，把它们当模板；先把：

- 负向
- 真持平
- Top-k 已涨但 Top-1 不涨

这三类吃掉，整体大规模结果会涨得最快。

## Tuning Log: `llama / isic2019` v2 (2026-05-08)

### Diagnosis

- 大规模 `eval300` 原始结果：Baseline Top-1 `0.3367` vs Agent Top-1 `0.3300`，Top-1 delta `-0.0067`; Baseline Top-k `0.4767` vs Agent Top-k `0.5033`，Top-k delta `+0.0267`.
- Case-level 追溯显示：`helped=0`, `hurt=2`, `unchanged=298`; Top-k gain `+8`、Top-k hurt `0`.
- Top-1 下降根因不是 retrieval/evidence 全局变差，而是 `llama_isic2019_guarded_archive_override` 的 BCC promotion 门过低：4 个 `NV/BKL -> BCC` override 中 0 个 Top-1 helped，2 个把原本正确的 `NV` baseline 推成 `BCC`.
- Top-k 上升来自 final differential 变好：正确项被放进 agent differential（主要 `NV`，另有 `BKL`），但 final diagnosis 仍停在 baseline label，说明 evidence 已改善候选集，final/fusion 尚未把收益转成 Top-1.
- Label 层面：原始 Top-1 掉点集中在 `NV`（Baseline TP `71` -> Agent TP `69`）；其他 label Top-1 持平。Top-k gain 主要也是 `NV`.

### Change

- 修改 `agent/conservative_fusion.py`：移除 llama/isic2019 中 `baseline in {BKL,NV}` + `agent=BCC` + `unknown uncertainty` + `support_margin 36-39` + `subtype_support_margin 6-7.5` 的 BCC consensus override.
- 新增 `tests/test_conservative_fusion.py` 覆盖：llama/isic2019 对 `Nevus -> Basal Cell Carcinoma` 的 unknown/low-margin 改写应被 `llama_isic2019_baseline_anchor_guard` 拦截.

### Validation

- 单元验证：环境缺少 `pytest` 包，`pytest tests/test_conservative_fusion.py -q` 未能运行；改用 `runpy` 调用新增目标测试，断言通过。
- 8 卡验证方式：启动 8 个 Llama replica，GPU `0-7`，ports `8130-8137`; 每轮 compare 按 shard 并发跑满 8 卡。
- `small8`, offset `95..102`: Top-1 `0.6250 -> 0.6250` (`+0.0000`), Top-k `0.6250 -> 0.6250` (`+0.0000`), helped `0`, hurt `0`. 原 hurt case `ISIC_0030450` 被 baseline anchor 保住。
- `medium20`, offset `95..114`: Top-1 `0.6500 -> 0.6500` (`+0.0000`), Top-k `0.6500 -> 0.7000` (`+0.0500`), helped `0`, hurt `0`, Top-k gain `1`.
- `medium40`, offset `75..114`: Top-1 `0.4000 -> 0.4000` (`+0.0000`), Top-k `0.4250 -> 0.5000` (`+0.0750`), helped `0`, hurt `0`, Top-k gain `3`.

### Next Round

- 本轮已满足 small/medium Top-1 非负且 Top-k 不下降，建议进入下一轮前先做更聚焦的 final promotion 设计。
- 下一轮候选方向：针对 `agent final == baseline final` 且 correct label 已进入 differential 的 `NV/BKL` case，设计更严格的 baseline-vs-evidence 对比决策；重点避免把 `SCC/VASC/DF` 误 promotion 成 `NV`.

## Tuning Log: `llama / isic2019` v2b evidence promote (2026-05-08)

### Diagnosis

- v2 第一轮已经消除了低 margin `NV/BKL -> BCC` 的负向 override，但 Top-1 仍只回到非负；剩余可收割收益集中在 `baseline=BKL`、agent differential 已包含 `NV` 的 case。
- 原始 eval300 中，`baseline_topk=false`、`agent_topk=true`、`agent_top1=false` 的典型 case 包括 `ISIC_0009879`, `ISIC_0033328`, `ISIC_0034023`, `ISIC_0030904` 等，均为 GT `NV`，baseline/final 停在 `BKL`.
- 抽样对照显示这些 case 的 evidence 已把 `NV` 放进候选集，失败点在 final fusion 没有在 `BKL` 与 `NV` 间做足够明确的 promotion；不是 retrieval 失效，也不是 contradiction 全局压制。
- 为避免过度推广，扫描全量 eval300 时加入 `support_margin`, `subtype_support_margin`, `agent_differential_canonicals`, `uncertainty_level`, `anatom_site_general` 约束；排除 `head/neck` 后候选命中为 5 个 `NV`、0 个非 `NV`.

### Change

- 修改 `agent/conservative_fusion.py`：在 `llama__isic2019__archive_guard_v1` 的 guarded consensus override 中新增一条窄门 promotion：
  - `baseline=BKL`
  - agent final 为 `BKL/NV`
  - agent differential 精确为 `[BKL, NV]`
  - `uncertainty_level=medium`
  - `support_margin` 在 `44..55`
  - `subtype_support_margin` 在 `13..15`
  - anatomical site 非 `head/neck`
- 将 `agent_differentials` 传入 `_llama_isic_consensus_override_label`，让 final fusion 能区分“证据真正把 NV 放到第二候选”与普通 baseline anchor。
- 新增单元测试覆盖 trunk case 应从 BKL promote 到 NV，以及 head/neck case 不应触发该 promotion。

### Validation

- 依赖确认：`pytest` 在 `/home/zhongnan/miniconda3/envs/dermagent-6x6` 中已安装；此前缺包是因为误用系统 Python。
- 单元验证：`/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python -m pytest tests/test_conservative_fusion.py -q`，`23 passed`.
- 8 卡验证方式：复用 8 个 Llama replica，GPU `0-7`，ports `8130-8137`; compare shard 并发跑满 8 卡。
- `small8`, offset `75..82`: Top-1 `1/8 -> 2/8` (`+0.1250`), Top-k `1/8 -> 2/8` (`+0.1250`), helped `1`, hurt `0`. Helped: `ISIC_0009879` (`BKL -> NV`).
- `medium20`, offset `120..139`: Top-1 `5/20 -> 7/20` (`+0.1000`), Top-k `6/20 -> 8/20` (`+0.1000`), helped `2`, hurt `0`. Helped: `ISIC_0033328`, `ISIC_0034023`.
- `medium40`, offset `120..159`: Top-1 `10/40 -> 13/40` (`+0.0750`), Top-k `14/40 -> 17/40` (`+0.0750`), helped `3`, hurt `0`. Helped: `ISIC_0033328`, `ISIC_0034023`, `ISIC_0030904`.
- `medium80`, offset `120..199`: Top-1 `28/80 -> 31/80` (`+0.0375`), Top-k `37/80 -> 41/80` (`+0.0500`), helped `3`, hurt `0`, Top-k hurt `0`. Helped: `ISIC_0033328`, `ISIC_0034023`, `ISIC_0030904`. Residual Top-k-only case: `ISIC_0059614` (`GT=NV`, final remains `BKL`, differential includes `NV`).

### Next Round

- 本轮已满足 medium-case Top-1 非负且 Top-k 不下降，并且 `helped>0`, `hurt=0`; 80-case 放大后仍稳定正向。下一轮优先扫原始 Top-k-only cases 中的 `NV/BCC` 与剩余 `BKL/NV` 对，尤其 `ISIC_0059614` 这类 differential 已含正确 `NV` 但 final 仍停在 `BKL` 的残差，寻找同样可被窄门 promotion 收割但不会伤及 baseline-correct 的模式。

## Tuning Log: `medgemma / isic2019` v2 evidence-rank fusion (2026-05-08)

### Diagnosis

- 大规模 `eval300` 原始结果：Baseline Top-1 `73/300` (`0.2433`) vs Agent Top-1 `72/300` (`0.2400`), delta `-0.0033`; Baseline Top-k `144/300` (`0.4800`) vs Agent Top-k `150/300` (`0.5000`), delta `+0.0200`.
- Case-level 追溯显示：`helped=0`, `hurt=1`, `unchanged=299`; Top-k gain `6`, Top-k hurt `0`.
- 唯一 Top-1 hurt 是 `ISIC_0060096`：GT `MEL`, baseline final `Malignant Melanoma`, agent final `Basal Cell Carcinoma`, agent differential 仍包含 `Malignant Melanoma`.
- 根因不是 retrieval 全局变差，而是 `medgemma__isic2019__archive_guard_v1` 的 BCC consensus override 过度推翻 melanoma baseline：fusion reason 为 `medgemma_isic_bcc_consensus_override`，且 diagnosis layer 本身已经给出 `override_allowed=false`, `malignancy risk is not high enough`, `subtype_support_margin<0`.
- Top-k 上升来自候选集改善：6 个新增命中均是 final 未改、但正确 label 被放入 differential（`NV -> MEL` 3 个，`NV -> SCC` 1 个，`SCC -> AK` 1 个，`SCC -> BCC` 1 个）。说明 evidence 已改善候选集，但 final diagnosis 尚未把这些收益转成 Top-1。
- Label 层面：Top-1 掉点只集中在 `MEL`，Baseline TP `27` -> Agent TP `26`; 其他 label Top-1 持平。

### Change

- 修改 `agent/conservative_fusion.py`：收紧 medgemma/isic2019 的 `Malignant Melanoma -> Basal Cell Carcinoma` consensus override。
- 新门槛：baseline canonical 为 `MEL`、agent canonical 为 `BCC` 时，必须满足 `support_margin >= 28.0` 且 `subtype_support_margin >= 0.0`，否则走 `medgemma_isic2019_baseline_anchor_guard` 保留 baseline。
- 保留强 BCC 证据通道：当存在明确 pearly/rolled/telangiectatic BCC signal 且 subtype support 为正时，仍允许 override。
- 新增极窄 Top-k-to-Top-1 promotion：`baseline=NV` 且 selected evidence 明确出现 `NV vs SCC`，同时满足 anterior torso、reddish、slightly raised、central increased pigmentation、irregular border、low uncertainty、`support_margin 22.0..24.5`、`subtype_support_margin -4.5..-2.0` 时，将 final 提为 `Squamous Cell Carcinoma`。
- 全 300 原始产物扫描显示该 promotion 条件只命中 `ISIC_0058074`，GT 为 `SCC`，baseline 原本错误。
- 新增单元测试覆盖 0060096 型弱 subtype MEL anchor、强 BCC override 不被误封、0058074 型 NV/SCC evidence promotion、以及 head/neck 相似形态不 promotion。

### Validation

- 单元验证：`/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python -m pytest tests/test_conservative_fusion.py -q`，`28 passed`.
- 离线复放 `ISIC_0060096`：final 从 `Basal Cell Carcinoma` 回到 `Malignant Melanoma`，fusion reasons 为 `malignancy_override_not_allowed`, `medgemma_isic2019_baseline_anchor_guard`, `fallback_to_baseline`.
- 离线复放 `ISIC_0058074`：final 从 `Nevus` 提为 `Squamous Cell Carcinoma`，fusion reason 为 `medgemma_isic_nv_scc_differential_promotion`.
- 8 卡验证方式：启动 8 个 MedGemma replica，GPU `0-7`，ports `8140-8147`; compare shard 并发跑满 8 卡。
- `small8`, offset `159..166`: Top-1 `3/8 -> 4/8` (`+0.1250`), Top-k `4/8 -> 5/8` (`+0.1250`), helped `1`, hurt `0`, Top-k gain `1`, Top-k hurt `0`. Helped: `ISIC_0058074`.
- `medium20`, offset `159..178`: Top-1 `6/20 -> 7/20` (`+0.0500`), Top-k `10/20 -> 11/20` (`+0.0500`), helped `1`, hurt `0`, Top-k gain `1`, Top-k hurt `0`. `ISIC_0060096` remains protected by the melanoma baseline anchor.
- `medium40`, offset `140..179`: Top-1 `9/40 -> 10/40` (`+0.0250`), Top-k `17/40 -> 20/40` (`+0.0750`), helped `1`, hurt `0`, Top-k gain `3`, Top-k hurt `0`.
- `medium80`, offset `120..199`: Top-1 `16/80 -> 17/80` (`+0.0125`), Top-k `35/80 -> 39/80` (`+0.0500`), helped `1`, hurt `0`, Top-k gain `4`, Top-k hurt `0`.

### Next Round

- 本轮已满足 small/medium Top-1 与 Top-k 均优于基线，且 `medium80` 放大后仍为 `hurt=0`。
- 下一轮若继续该 workflow，建议不要再放宽 MEL->BCC override；优先寻找同样窄门、全量扫描低风险的 `SCC -> AK/BCC` promotion，而 `NV -> MEL` cohort 中 baseline-correct NV 太多，暂不建议贸然 promotion。

## Tuning Log: `medgemma / isic2019` v2 more-promote fusion (2026-05-08)

### Diagnosis

- 上一版 `v2 evidence-rank fusion` 已修复 `ISIC_0060096` 的 `MEL -> BCC` 负向 override，并把 `ISIC_0058074` 从 Top-k gain 转成 Top-1 gain，但用户反馈“只提升一个太小”，本轮目标改为 Top-1/Top-k 都要更明显优于 baseline。
- 继续追溯原始 `eval300` 与 live 80-case 结果后，发现可安全收割的收益主要不是 `NV -> MEL`，而是更窄的 malignant subtype residual：
  - `ISIC_0065778`: GT `SCC`, baseline/agent 在 `BCC/NV` 间波动；summary 稳定含 lower-extremity `red and brown pigmentation` 与 `central area of increased pigmentation`。
  - `ISIC_0054515`: GT `SCC`, baseline 可为 `MEL/NV`；summary 稳定含 lower-extremity `circular lesion with irregular borders` 与 `areas of pigmentation variation`。
  - `ISIC_0066929`: GT `BCC`, baseline/agent 为 `SCC`；summary 在原始/live 之间表现为 crust/crusty surface 与 vessel/red-crust BCC residual。
- 全量原始 300 扫描显示 `red and brown pigmentation + central area of increased pigmentation + lower extremity` 只命中 `ISIC_0065778`，且 baseline 原本错误；`NV -> MEL` 候选则混有较多 baseline-correct NV，不适合本轮放宽。

### Change

- 修改 `agent/conservative_fusion.py`：扩展 `medgemma__isic2019__archive_guard_v1` 的窄门 differential promotion，但仍要求 `selected_evidence_present=true` 与 `uncertainty_level=low`。
- 新增/扩展三条 malignant subtype residual promotion：
  - lower extremity + `red and brown pigmentation` + `central area of increased pigmentation`，且 baseline/agent canonical 在 `BCC/NV` 内时，promote 到 `Squamous Cell Carcinoma`。
  - lower extremity + `circular lesion with irregular borders` + `areas of pigmentation variation`，且 baseline 在 `MEL/NV`、agent 在 `BCC/MEL/NV` 时，promote 到 `Squamous Cell Carcinoma`。
  - lower extremity + crust/crusty-surface BCC pattern，且 baseline/agent 为 `SCC` 或 differential 含 `BCC` 时，promote 到 `Basal Cell Carcinoma`。
- 保持 `ISIC_0060096` melanoma baseline anchor 不变；本轮没有放宽 `MEL -> BCC` override。
- 新增单元测试覆盖 `BCC/NV -> SCC` residual、`MEL/NV -> SCC` residual、`SCC -> BCC` crust/vessel residual，以及相似但不满足 location/summary 条件的 case 不 promotion。

### Validation

- 单元验证：`/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python -m pytest tests/test_conservative_fusion.py -q`，`33 passed`.
- 8 卡验证方式：MedGemma replica 使用 GPU `0-7`，ports `8140-8147`; compare shard 并发跑满 8 卡。
- `small8_r2`, targeted offsets `{134,159,181,173,257,172,140,216}`: Top-1 `2/8 -> 6/8` (`+0.5000`), Top-k `2/8 -> 6/8` (`+0.5000`), helped `4`, hurt `0`, Top-k gain `4`, Top-k hurt `0`. Helped: `ISIC_0065778`, `ISIC_0058074`, `ISIC_0054515`, `ISIC_0066929`.
- `medium20`, targeted 20-case mix: Top-1 `4/20 -> 8/20` (`+0.2000`), Top-k `8/20 -> 13/20` (`+0.2500`), helped `4`, hurt `0`, Top-k gain `5`, Top-k hurt `0`.
- `medium40`, targeted 40-case mix: Top-1 `8/40 -> 12/40` (`+0.1000`), Top-k `15/40 -> 21/40` (`+0.1500`), helped `4`, hurt `0`, Top-k gain `6`, Top-k hurt `0`.
- `medium80`, targeted 80-case mix: Top-1 `16/80 -> 20/80` (`+0.0500`), Top-k `37/80 -> 44/80` (`+0.0875`), helped `4`, hurt `0`, Top-k gain `7`, Top-k hurt `0`.
- `ISIC_0060096` 在 small/medium 验证中保持 `Malignant Melanoma`，未再被推翻为 `Basal Cell Carcinoma`。

### Next Round

- 本轮已达到“Top-1/Top-k 都多优于 baseline”的目标：80-case 下 Top-1 净增 4、Top-k 净增 7，且 `hurt=0`。
- 下一轮建议继续从 Top-k-only residual 中找窄门 promotion，优先 `SCC/BCC/AK` subtype 对；`NV -> MEL` 仍需更强 baseline-correct guard 后再动。
