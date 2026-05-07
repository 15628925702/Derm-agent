# 医生可读证据包出口

这个出口用于把 DermAgent 在最终诊断前组织好的结构化证据，再整理成医生可直接阅读的辅助证据包。它不是最终诊断结果，也不会改变 `final_diagnosis` 的融合策略。

## 默认状态

- 默认关闭，不额外消耗模型调用。
- 关闭时不会生成 `physician_evidence_summary` 字段内容。
- 开启后只在 DermAgent 路径多执行一次整理调用，direct baseline 不走这个出口。
- 整理调用固定走独立 Qwen physician-summary 服务，不复用当前诊断 backbone，也不使用 extractive fallback。

## 输出内容

开启后，每个 case 会生成：

- `state.physician_evidence_summary`
- `case_execution_record.physician_evidence_summary`
- debug 目录下的 `physician_evidence_summary.json`

证据包字段包括：

- `evidence_overview`
- `clinical_context`
- `lesion_description`
- `evidence_by_domain`
- `key_observations`
- `supporting_evidence`
- `opposing_or_uncertain_evidence`
- `differential_reasoning`
- `risk_flags`
- `differential_considerations`
- `contradictions_or_tensions`
- `information_gaps`
- `suggested_next_checks`
- `caveats`

当前 schema 为 `physician_evidence_summary_v2_detailed`，支持两种详细度：

- `brief`：紧凑版，适合快速预览和论文正文中展示少量例子。
- `detailed`：详细版，尽量保留更多中间证据，并额外输出 `structured_evidence_appendix`，用于医生审阅、补充材料和机制展示。

这一层不是短摘要，而是把中间证据包整理成临床医生可读的证据工作台：保留形态、颜色/色素模式、分布/部位、风险、不确定性、矛盾张力、信息缺口和下一步检查建议。机器式风险标记会被转换成自然临床语言。

## 安全边界

这个出口只面向医生辅助阅读，不暴露内部 prompt、系统指令、workflow 路由、policy 名称、source_id、retrieval score、隐藏标签、GT 或实现细节。

整理调用会明确要求模型不要给最终诊断，也不要替换最终诊断。如果模型返回了 `final_diagnosis` 或 `diagnosis` 之类字段，运行时会把它们从医生证据包里移除，并在 `caveats` 里记录。

## 开启方式

先单独启动 Qwen 总结服务。默认端口是 `8200`，服务名是 `Qwen2.5-VL-7B-Instruct`：

```bash
cd /data/gh/DermAgent
bash scripts/start_qwen_physician_summary_server.sh
```

单次 compare 开启：

```bash
python scripts/compare_agent_vs_qwen.py \
  --data-root /path/to/dataset \
  --limit 8 \
  --enable-physician-evidence-summary \
  --physician-evidence-summary-base-url http://127.0.0.1:8200/v1 \
  --physician-evidence-summary-model Qwen2.5-VL-7B-Instruct \
  --physician-evidence-summary-detail detailed
```

把最后一行改成 `--physician-evidence-summary-detail brief` 即可生成紧凑版。

环境变量开启：

```bash
DERMAGENT_ENABLE_PHYSICIAN_EVIDENCE_SUMMARY=1 \
DERMAGENT_QWEN_SUMMARY_BASE_URL=http://127.0.0.1:8200/v1 \
DERMAGENT_QWEN_SUMMARY_MODEL=Qwen2.5-VL-7B-Instruct \
DERMAGENT_QWEN_SUMMARY_DETAIL=detailed \
python scripts/compare_agent_vs_qwen.py --data-root /path/to/dataset --limit 8
```

代码内开启：

```python
run_agent(
    case_input=case_input,
    execution_overrides={
        "enable_physician_evidence_summary": True,
        "physician_evidence_summary_base_url": "http://127.0.0.1:8200/v1",
        "physician_evidence_summary_model": "Qwen2.5-VL-7B-Instruct",
        "physician_evidence_summary_detail": "detailed",
    },
)
```

不开启时，诊断评估速度和原来一致。开启但 Qwen 总结服务不可用时，该 case 的 `physician_evidence_summary.status` 会记录为 `failed`；模型返回空内容时记录为 `empty_model_output`。运行时不会退回到当前诊断模型或模板拼接结果。
