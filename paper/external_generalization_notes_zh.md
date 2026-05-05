# 外部泛化结果说明

## 当前结论

当前 DermAgent 的外部数据集结果不适合作为“稳定优于 direct baseline”的正面主结论。

在将外部保守泛化层真正接回最终外部评测链路后，外部随机分层子集上的结果表现为：

- 不再明显劣化 direct baseline
- 但也没有显示出稳定的额外增益
- 外部 agent 输出在多数情况下被保守层回退到 baseline

因此，当前外部保守层更适合被解释为：

- `baseline-preserving safeguard`
- 而不是 `stable external gain module`

## 为什么会这样

排查结果显示，外部结果的主要瓶颈不在融合阈值本身，而在：

- `consistent_retrieval` 不足
- 外部样本上的 override 资格不足
- `malignancy_override_allowed` 与 `subtype_override_allowed` 在外部分布上很难被满足

这说明当前主线 agent 的检索、证据组织和 override 判据主要仍然适配内部 PAD-UFES-20 分布。

## 对方法的解释

当前主线的 bootstrap、经验库、可学习组件训练与策略选择，主要来自内部数据集分布。

因此在外部数据集上出现以下现象是合理的：

- agent 可以形成风险提示
- 但缺少足够稳定的一致性检索支持
- 导致保守泛化层倾向于保留 direct baseline

换言之，当前外部结果并不否定内部主线贡献，而是提示：

- 外部分布迁移的主要瓶颈在 retrieval / evidence consistency
- 而不是最后一层 conservative fusion 门槛本身

## 为什么不继续调保守层

我们尝试过：

- 修复最终外部评测链路，确保真正调用旧版 conservative fusion
- 小步放宽 support margin / subtype support margin / contradiction gate
- 在风险升级场景下引入更窄的例外通道

结果表明：

- 轻微放宽阈值并不能稳定带来外部正增益
- 继续调节保守层会越来越接近“为结果而调规则”

因此，在论文最终版本中，更稳妥的做法是：

- 保留严格 conservative external fusion
- 将外部结果表述为“安全性保护有效，但未显示稳定增益”

## 论文中建议的写法

建议写法：

> 在外部随机分层子集上，恢复原始 conservative external fusion 逻辑后，DermAgent 不再劣化 direct baseline，但也未显示稳定的额外性能增益。这表明当前外部保守层主要充当 safety-preserving safeguard，而非稳定的外部增强模块。进一步分析表明，当前瓶颈主要来自外部分布下的一致性检索与证据支持不足，而不是融合阈值本身。

## 如果未来继续做

更合理的后续方向不是继续调 conservative fusion，而是：

- 做 external-specific bootstrap / retrieval adaptation
- 做 domain-aware retrieval consistency control
- 区分“纯外部泛化”和“外部适配”两种实验口径

这类工作可以作为后续研究方向，而不建议在当前论文收尾阶段继续修改主 agent 结构。
