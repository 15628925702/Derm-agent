# Physician Evidence Package Examples

This folder is reserved for real doctor-readable evidence-package examples generated from actual DermAgent runs.

Rules:

- Generate examples only with `DERMAGENT_ENABLE_PHYSICIAN_EVIDENCE_SUMMARY=1` or `--enable-physician-evidence-summary`.
- The physician summary call must use the separate Qwen service, usually `http://127.0.0.1:8200/v1`.
- Keep only examples with `physician_evidence_summary.status: "ok"` for paper-facing review.
- Do not add fallback, template, hand-written, or fabricated examples.

Current real example sets:

- `qwen_ham10000_brief_examples_20260507/`: compact doctor-readable summaries for the same three HAM10000 cases.
- `qwen_ham10000_detailed_examples_20260507/`: fuller summaries for the same cases, including `structured_evidence_appendix`.

Start the summary service with:

```bash
cd /data/gh/DermAgent
bash scripts/start_qwen_physician_summary_server.sh
```

Use `--physician-evidence-summary-detail brief` or `--physician-evidence-summary-detail detailed` during generation.
