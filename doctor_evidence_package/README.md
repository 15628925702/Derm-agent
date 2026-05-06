# 医生可读证据包出口

这个出口用于把 DermAgent 在最终诊断前组织好的结构化证据，再整理成医生可直接阅读的辅助证据包。它不是最终诊断结果，也不会改变 `final_diagnosis` 的融合策略。

## 默认状态

- 默认关闭，不额外消耗模型调用。
- 关闭时不会生成 `physician_evidence_summary` 字段内容。
- 开启后只在 DermAgent 路径多执行一次整理调用，direct baseline 不走这个出口。

## 输出内容

开启后，每个 case 会生成：

- `state.physician_evidence_summary`
- `case_execution_record.physician_evidence_summary`
- debug 目录下的 `physician_evidence_summary.json`

证据包字段包括：

- `evidence_overview`
- `key_observations`
- `supporting_evidence`
- `opposing_or_uncertain_evidence`
- `risk_flags`
- `differential_considerations`
- `information_gaps`
- `suggested_next_checks`
- `caveats`

## 安全边界

这个出口只面向医生辅助阅读，不暴露内部 prompt、系统指令、workflow 路由、policy 名称、source_id、retrieval score、隐藏标签、GT 或实现细节。

整理调用会明确要求模型不要给最终诊断，也不要替换最终诊断。如果模型返回了 `final_diagnosis` 或 `diagnosis` 之类字段，运行时会把它们从医生证据包里移除，并在 `caveats` 里记录。

## 开启方式

单次 compare 开启：

```bash
python scripts/compare_agent_vs_qwen.py \
  --data-root /path/to/dataset \
  --limit 8 \
  --enable-physician-evidence-summary
```

环境变量开启：

```bash
DERMAGENT_ENABLE_PHYSICIAN_EVIDENCE_SUMMARY=1 python scripts/compare_agent_vs_qwen.py --data-root /path/to/dataset --limit 8
```

代码内开启：

```python
run_agent(
    case_input=case_input,
    execution_overrides={"enable_physician_evidence_summary": True},
)
```

不开启时，诊断评估速度和原来一致。
