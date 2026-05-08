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

## 调参优先级总表

排序原则：

1. 先救 `Top-1` 负向
2. 再救 `Top-1` 持平但 `Top-k` 已上涨
3. 再处理弱正向
4. 最后再碰已经很强的组合

| Priority | Workflow | Current | Why first | Main tuning goal |
|---|---|---|---|---|
| 1 | `llama / isic2019` | Top-1 `-0.0067`, Top-k `+0.0267` | 负向里跌幅最大，但 Top-k 已有帮助 | 先把 Top-1 拉回非负 |
| 2 | `medgemma / isic2019` | Top-1 `-0.0033`, Top-k `+0.0200` | evidence 有用，final 转化不足 | 修 final selection / fusion |
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

1. `llama / isic2019`
2. `medgemma / isic2019`
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

1. `llama / isic2019`
2. `medgemma / isic2019`
3. `qwen / isic2019`
4. `medgemma / scin`
5. `medgemma / ham10000`
6. `qwen / ham10000`
7. `hulumed / isic2019`
8. `dermatollama / isic2019`
9. `dermatollama / ham10000`
10. `qwen / scin`

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
