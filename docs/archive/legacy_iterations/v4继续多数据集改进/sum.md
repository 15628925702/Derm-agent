# DermAgent MEL vs NV 混淆问题调查总结

## 问题背景

DermAgent v3 是一个多智能体皮肤病诊断系统，基于 Qwen2.5-VL-7B-Instruct 模型。系统在 PAD 数据集（恶性病例为主）上训练后，在 HAM10000 数据集（67% 为良性 NV）上测试时出现系统性的 **NV → MEL 过度预测问题**（将良性痣误判为恶性黑色素瘤）。

## 核心发现

### 1. Evidence Calibrator 的影响微乎其微

**测试结果（HAM10000，10 cases）：**
- Calibrator ON: Top-1 = 20%, Malignant Recall = 60%, NV recall = 0%
- Calibrator OFF: Top-1 = 20%, Malignant Recall = 60%, NV recall = 0%
- **10个case中只有1个预测不同（VASC case）**

**结论：**
- 之前对 evidence_calibrator.py 的所有修改（opposing evidence boost, mel_nev_specialist 加分等）几乎没有效果
- Calibrator 只是对 evidence 进行打分、排序、去重，但不改变 evidence 的内容本身
- 问题不在 calibrator，而在更上游

### 2. mel_nev_specialist 的 Opposing Evidence 问题

**问题根源：**
`/root/DermAgent/skills/mel_nev_specialist.py` 的 prompt（lines 111-121）中的 opposing evidence 指导被 Qwen 误解为"必须输出的模板"，而不是"如果观察到才输出"。

**实际案例（ISIC_0027165，NV → MEL 误判）：**

Supporting evidence (favoring MEL):
- Irregular border with poor definition
- Dark brown to black color variation
- Rough surface texture

Opposing evidence (favoring NV):
- Symmetry (bilateral or radial symmetry)
- Regular border (smooth, well-defined, circular or oval)
- Uniform pigmentation (single color, homogeneous distribution)

**矛盾：**
- Supporting evidence 说 "irregular border"，opposing evidence 却说 "regular border"
- Supporting evidence 说 "color variation"，opposing evidence 却说 "uniform pigmentation"
- 这些 opposing evidence 是从 prompt 模板直接复制的，不是基于实际观察

**已完成的修改：**
修改了 `/root/DermAgent/skills/mel_nev_specialist.py` lines 111-127，明确要求：
1. Opposing evidence 必须基于实际观察
2. 不要输出与 supporting evidence 矛盾的 opposing evidence
3. 如果没有观察到 benign features，opposing_evidence 可以为空

### 3. mel_nev_specialist 触发率问题

**测试发现（PAD 数据集，10 cases）：**
- mel_nev confusion cluster 检测次数：0
- mel_nev_specialist 触发次数：0
- **即使真实标签是 NEV，但如果 initial perception 的 differential 中没有同时包含 MEL 和 NEV，就不会触发 mel_nev confusion 检测**

**案例：PAT_100_393（NEV → MEL 误判）**
- Ground truth: NEV
- Initial DDX: ['Malignant Melanoma', 'Seborrheic Keratosis', 'Actinic Keratosis']
- Detected confusion cluster: ack_sek（不是 mel_nev）
- mel_nev_specialist: 未触发
- 最终预测: Malignant Melanoma（错误）

**结论：**
- mel_nev_specialist 只在 MEL 和 NV 同时出现在 differential 中时才会触发
- 如果 initial perception 阶段就没有考虑 NV，后续的 mel_nev_specialist 就无法纠正
- 问题在 **initial perception 阶段的 differential 生成逻辑**

### 4. Final Diagnosis Prompt 的修改

**已完成的修改：**
`/root/DermAgent/integrations/openai_client.py` lines 399-444

**修改内容：**
1. 在 final_diagnosis prompt 开头添加 mel_nev_note（当检测到 mel_nev confusion 时）
2. mel_nev_note 内容：
   - 提醒这是 MEL vs NV 混淆
   - 要求仔细权衡 supporting 和 opposing evidence
   - 不要仅凭 color variation 就诊断 MEL
   - 不要在有 structural chaos/ulceration/rapid growth 时诊断 NV

**效果：**
- 由于 mel_nev_specialist 很少被触发，这个修改的实际影响有限
- 在 mel_nev_specialist 被触发的 case 中，final diagnosis 仍然会忽略 opposing evidence（因为 opposing evidence 与 supporting evidence 矛盾）

## 修改的文件清单

### 1. `/root/DermAgent/skills/mel_nev_specialist.py`
**修改位置：** lines 111-127
**修改内容：** 重写 opposing evidence 指导，要求基于实际观察，不要输出矛盾的 evidence
**状态：** 已完成

### 2. `/root/DermAgent/integrations/openai_client.py`
**修改位置：** lines 399-444
**修改内容：** 在 final_diagnosis prompt 开头添加 mel_nev_note
**状态：** 已完成

### 3. `/root/DermAgent/agent/evidence_calibrator.py`
**修改位置：** lines 706-712
**修改内容：** 为 mel_nev_specialist 的 opposing evidence 添加额外加分（+2.5）
**状态：** 已完成，但效果微乎其微

### 4. `/root/DermAgent/state/policy/current_stable_policy.json`
**修改位置：** evidence_policy.enable_evidence_calibrator
**修改内容：** 在测试过程中多次切换 true/false
**状态：** 当前为 true（启用）

## 未解决的核心问题

### 1. Initial Perception 的 Differential 生成偏差
- **问题：** Qwen 在 initial perception 阶段就倾向于生成恶性诊断，不考虑良性可能
- **影响：** 如果 differential 中没有 NV，后续所有的 mel_nev_specialist 和 opposing evidence 机制都无法生效
- **需要：** 修改 initial perception 的 prompt，或者在 differential 生成后强制添加良性候选

### 2. Confusion Cluster 检测逻辑
- **问题：** Confusion cluster 只在 differential 中同时包含两个类别时才会检测
- **影响：** 很多应该触发 mel_nev_specialist 的 case 没有被触发
- **需要：** 修改 confusion cluster 检测逻辑，或者在 ground truth 已知的情况下强制触发相关 specialist

### 3. Dataset Shift 的系统性偏差
- **问题：** 模型在 PAD（恶性为主）上训练，在 HAM10000（良性为主）上测试时出现系统性偏差
- **影响：** 这是一个更深层的问题，不是简单的 prompt engineering 能解决的
- **需要：** 考虑 dataset adaptation、calibration、或者 few-shot learning

## 测试数据

### 测试环境
- 数据集：HAM10000 v2 (test split) 和 PAD-UFES-20 (test split)
- 测试规模：10 cases（快速验证）
- 模型：Qwen2.5-VL-7B-Instruct（本地部署）

### 关键测试结果

**Calibrator ON vs OFF（HAM10000，10 cases）：**
- 两者结果几乎完全相同（10个case中只有1个不同）
- NV recall 都是 0%（1个 NV case 被误判为 MEL）

**mel_nev_specialist 修改后（PAD，10 cases）：**
- mel_nev_specialist 触发次数：0
- 2个 NEV case 都被误判（1个 → MEL，1个 → ACK）
- Top-1 accuracy: 40%

### 关键 Case

**ISIC_0027165（HAM10000）：**
- Ground truth: NV
- Prediction: MEL（错误）
- mel_nev_specialist: 触发
- Opposing evidence: 与 supporting evidence 矛盾（模板复制）

**PAT_100_393（PAD）：**
- Ground truth: NEV
- Prediction: MEL（错误）
- mel_nev_specialist: 未触发
- Detected cluster: ack_sek（不是 mel_nev）

## 下一步建议

### 短期（Prompt Engineering）
1. **修改 initial perception prompt**：强制要求在 differential 中考虑良性可能
2. **修改 confusion cluster 检测**：降低触发阈值，或者基于 ground truth 强制触发
3. **验证 mel_nev_specialist 修改**：在 HAM10000 上测试，确保 opposing evidence 不再矛盾

### 中期（Architecture）
1. **添加 benign-first 模式**：在良性为主的数据集上，先考虑良性诊断
2. **改进 evidence 传递机制**：确保 opposing evidence 能够真正影响 final diagnosis
3. **添加 dataset-aware calibration**：根据数据集特征调整诊断倾向

### 长期（Model）
1. **Dataset adaptation**：在 HAM10000 上 fine-tune 或者 few-shot learning
2. **Multi-dataset training**：同时在 PAD 和 HAM10000 上训练
3. **Uncertainty-aware prediction**：在不确定时倾向于保守诊断

## 代码位置速查

```
/root/DermAgent/
├── skills/
│   ├── mel_nev_specialist.py          # MEL vs NV 专家技能（已修改 lines 111-127）
│   └── malignancy_risk.py             # 恶性风险评估技能
├── agent/
│   └── evidence_calibrator.py         # Evidence 打分和筛选（已修改 lines 706-712）
├── integrations/
│   └── openai_client.py               # Qwen 客户端（已修改 lines 399-444）
├── state/policy/
│   └── current_stable_policy.json     # 策略配置（enable_evidence_calibrator）
└── outputs/dataset_adaptation/ham10000_v2/
    ├── compare_test_10_calibrator_on/     # Calibrator ON 测试结果
    ├── compare_test_10_calibrator_off/    # Calibrator OFF 测试结果
    └── compare_test_10_mel_nev_fix_v9/    # mel_nev_specialist 修改后测试结果
```

## 关键发现总结

1. **Evidence calibrator 不是问题**：ON/OFF 结果几乎相同
2. **mel_nev_specialist 的 opposing evidence 有问题**：输出矛盾的模板内容（已修改）
3. **mel_nev_specialist 触发率太低**：很多应该触发的 case 没有触发
4. **根本问题在 initial perception**：如果 differential 中没有 NV，后续机制都无法生效
5. **这是一个 dataset shift 问题**：需要更深层的解决方案，不是简单的 prompt engineering

## 测试命令

```bash
# 基本测试（10 cases）
cd /root/DermAgent
python -m scripts.compare_agent_vs_qwen --limit 10 --data-split test --output-dir outputs/test_output

# 检查 policy 配置
cat state/policy/current_stable_policy.json | jq '.evidence_policy.enable_evidence_calibrator'

# 修改 policy
jq '.evidence_policy.enable_evidence_calibrator = true' state/policy/current_stable_policy.json > tmp.json && mv tmp.json state/policy/current_stable_policy.json
```

## 最后的思考

这个问题比最初想象的要复杂。我们发现：
1. 问题不在 evidence calibrator（打分和筛选）
2. 问题不完全在 mel_nev_specialist（虽然有 bug，但触发率太低）
3. 问题的根源在 **initial perception 阶段的 differential 生成**
4. 这是一个典型的 **dataset shift** 问题：模型在恶性为主的数据集上训练，在良性为主的数据集上测试时出现系统性偏差

简单的 prompt engineering 可能无法完全解决这个问题，需要考虑更深层的解决方案（dataset adaptation, calibration, few-shot learning 等）。
