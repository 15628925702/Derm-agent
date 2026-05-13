# HuluMed+Agent Workflow Tuning Tasks

Date: 2026-05-13
Branch: `hulumed-isic-fusion-top3-20260513`

## Current Refactor State

- HuluMed+Agent five-dataset fusion rules have been extracted from `memory/fusion_experience/workflow_fusion_decision.py`.
- New focused module: `memory/fusion_experience/hulumed_workflow_fusion.py`.
- Refactor verification used real report cases and compared old vs new `decide_conservative_agent_fusion` outputs:
  - HAM10000 current 30% test: 3004 cases, 0 mismatches.
  - PAD20 current 30% test: 689 cases, 0 mismatches.
  - ISIC2019 old eval300: 300 cases, 0 mismatches.
  - SCIN old eval300: 300 cases, 0 mismatches.
  - SD198 old eval300: 300 cases, 0 mismatches.

## Global Evaluation Rules

- TopK is fixed to Top3 for all future comparisons.
- Do not require HuluMed+Agent to beat pathological malignant recall caused by degenerate prediction behavior, especially SkinVL runs that nearly predict all cases as malignant.
- Normal direct baselines still must be beaten or matched closely.
- If Top1 and Top3 clearly exceed the direct baselines, a malignant recall difference within about 0-1 percentage point can be treated as acceptable only if per-class safety is not degraded.
- Every workflow change must check per-disease or per-group performance. Avoid a fix that raises aggregate metrics while leaving one disease class near zero recall or extremely high error.

## Dataset Tasks

### PAD20

Status: no tuning needed now.

- Current HuluMed+Agent already has strong Top1 and Top3 on the full 30% test split.
- Do not adjust PAD20 workflow unless a later regression appears.
- Keep PAD20 as a regression guard when changing shared code.

### HAM10000

Status: evidence incomplete; run small direct probes only.

- Current HuluMed+Agent is strong against the available same-split direct results, but Dermatollama and LLaMA same-split direct are missing.
- Run only 100 cases each for:
  - Dermatollama direct on HAM10000 current 30% test split.
  - LLaMA direct on HAM10000 current 30% test split.
- Do not launch large full runs for this check.
- Compare HuluMed+Agent against these 100case probes on:
  - Top1 / correctness.
  - Top3.
  - Malignant recall.
  - Per-class recall, especially MEL, BCC, AKIEC, BKL, NV.

### ISIC2019

Status: highest-priority workflow tuning target.

Current normal direct targets, excluding pathological SkinVL malignant recall:

- Top1 target: beat HuluMed direct 50.4%; preferably also beat Dermatollama partial 51.4%.
- Top3 target: beat HuluMed direct 74.6%.
- Malignant recall target: beat LLaMA direct 42.7%.

Known issue:

- Old HuluMed+Agent eval300 was Top1 44.0%, Top3 70.0%, malignant recall 19.3%.
- The main weakness is malignant or high-risk ISIC labels being diluted into benign labels, especially NV, BKL, DF, or VASC.

Workflow tuning focus in `memory/fusion_experience/hulumed_workflow_fusion.py`:

- Strengthen malignant-preserving and top3-to-top1 promotion for MEL, BCC, SCC, and AK.
- Check cases where archive/initial Top3 contains MEL, BCC, SCC, or AK but final diagnosis becomes NV/BKL/DF/VASC.
- Add guards so benign promotion does not erase malignant candidates unless evidence is very strong.
- Track per-class recall for MEL, BCC, SCC, AK, BKL, NV, DF, VASC. No class should collapse to near-zero recall after tuning.

### SCIN

Status: tune malignant recall.

Known issue:

- Old HuluMed+Agent eval300 had acceptable-ish Top1/Top3 but malignant recall was very low.
- SkinVL malignant recall should not be used as the target if it comes from degenerate malignant overprediction.

Workflow tuning focus:

- Confirm SCIN grouped malignant/high-risk group definition.
- Strengthen rescue/promotion into `MALIGNANT_PREMALIGNANT` when early differential, agent differential, metadata, or evidence text contains actinic keratosis, BCC, SCC, melanoma, cancer, or high-risk sun-damage signals.
- Audit false-benign malignant cases by group and reason.
- Ensure improvements do not destroy common non-malignant groups such as dermatitis/eczema, pigment/keratosis/nevus, infection, acne/follicular, and vascular/purpuric.

### SD198

Status: needs overall tuning, not only malignant recall.

Old evidence:

- HuluMed+Agent eval300: Top1 45.7%, Top3 51.3%, malignant recall 80.0%.
- Old direct strongest: Top1 62.3%, Top3 64.7%, malignant recall 86.7%.

Workflow tuning focus:

- First verify whether the comparison is using fine labels or grouped labels. If other models are evaluated at grouped/coarse label level, HuluMed+Agent must use the same label-space granularity.
- After label-space consistency is confirmed, tune grouped top3-to-top1 promotion.
- Prioritize groups with very low recall or extreme error, not only aggregate score.
- Watch malignant/high-risk groups, papulosquamous/keratotic, dermatitis/eczema, benign tumor/cyst, vascular/ulcer/purpura, infection, acne/folliculitis/rosacea, and sun-damage/actinic.

## Acceptance Checklist

- For each tuned dataset, produce a table comparing HuluMed+Agent against the strongest non-pathological direct model:
  - Top1 / correctness.
  - Top3.
  - Malignant recall.
  - Error rate.
- Produce per-class or per-group recall/error tables.
- Explicitly list any class with near-zero recall or very high error.
- Confirm PAD20 did not regress.
- Confirm `python -m py_compile memory/fusion_experience/workflow_fusion_decision.py memory/fusion_experience/hulumed_workflow_fusion.py` passes.
- For any workflow logic change, run an offline comparison or small probe before full evaluation.
