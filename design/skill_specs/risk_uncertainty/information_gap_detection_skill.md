# information_gap_detection_skill

## Purpose
Identify the clinically meaningful information that is still missing from the case and explain how those gaps limit reasoning.

## When to Use
Use when uncertainty remains meaningful after observation and comparison.
Use when the case still feels underdetermined and the reasoning chain would benefit from explicitly naming what is missing.
Do not use this skill to choose the final diagnosis.

## Clinical Pattern
This skill focuses on:
- what information is missing,
- why that information matters,
- how the missing data keeps the differential open,
- how much uncertainty will remain if the gaps stay unresolved.

Its value is that it turns vague uncertainty into explicit missing-information structure.

## Strategy Overview
Doctors often pause to ask what they still do not know and whether that missing information materially changes interpretation.
This skill should make that pause explicit.
Its explainability value is high because later reasoning and reflection can point to concrete information deficits instead of generic uncertainty language.

## Workflow
1. First review the current observations, comparisons, contradictions, and uncertainty outputs.
2. Then name the clinically useful information that is still absent or too weak.
3. Then explain why each missing item matters for narrowing or reordering the differential.
4. Then state how the differential remains open because the information is missing.
5. Finally estimate the residual uncertainty if those gaps remain unresolved.

## Evidence to Check
- current observation outputs,
- differential and exclusion outputs,
- contradiction and uncertainty outputs,
- metadata fields that are absent or weak,
- information that would clarify risk, morphology, timing, or comparison.

## Watch Out For
- Do not invent irrelevant tests or missing data.
- Do not confuse missing evidence with contradictory evidence.
- Do not turn information-gap reasoning into treatment advice.
- Do not hide why a gap matters clinically.

## Output Contract
Return structured missing-information reasoning only.

- `missing_information`: clinically relevant information that is still missing
- `why_it_matters`: why those missing items matter
- `impact_on_differential`: how those gaps keep the differential unresolved
- `uncertainty_if_missing`: `low`, `medium`, or `high`

The output should strengthen explainable CoT by making uncertainty traceable to explicit information gaps.
