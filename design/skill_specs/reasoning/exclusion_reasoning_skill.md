# exclusion_reasoning_skill

## Purpose
Make physician-style exclusion reasoning explicit by naming which current candidates look less likely, why they look less likely, and what missing evidence still limits strong exclusion.

## When to Use
Use when two or more current candidates remain active and the system needs transparent differential narrowing.
Use when negative evidence or missing evidence is clinically important to explain why some candidates should be downgraded.
Do not use this skill to produce the final winning diagnosis.

## Clinical Pattern
This skill focuses on exclusion logic:
- which already-mentioned candidates are currently less favored,
- what negative or opposing evidence weakens them,
- what important evidence is still missing,
- how confident the current exclusion stance really is.

Its main value is to expose how the model narrows the differential without hiding the role of absent or missing evidence.

## Strategy Overview
Doctors often reason by asking not only “what fits?” but also “what does not fit well enough, and why?”
This skill formalizes that pattern.
It must separate true negative evidence from mere lack of confirming evidence, because those are clinically different.
Its explainability value is especially strong for final CoT because it makes the exclusion chain and its limitations inspectable.

## Workflow
1. First limit the scope to candidates that are already active in the current differential or specialist comparison.
2. Then identify negative evidence: features that argue against a candidate, not just features that fail to support it.
3. Then identify missing evidence that would be required for stronger exclusion.
4. Then compare the weight of negative evidence versus the uncertainty created by the missing evidence.
5. Then state which candidates are currently unlikely without declaring a final winner.
6. Finally grade exclusion confidence conservatively as low, medium, or high.

## Evidence to Check
- current ddx candidates,
- negative morphology or pattern evidence,
- pairwise comparison outputs,
- specialist comparison outputs,
- metadata conflicts or weak support,
- unresolved uncertainty,
- evidence that is absent but would be necessary for stronger exclusion.

## Watch Out For
- Do not invent new candidate diseases.
- Do not treat missing evidence as if it were direct contradiction.
- Do not overstate exclusion confidence when key evidence is unavailable.
- Do not convert “unlikely” into “ruled out” unless the current evidence truly supports that level of certainty.
- Do not let exclusion reasoning silently become final diagnosis selection.

## Output Contract
Return structured exclusion reasoning only.

- `unlikely_candidates`: already-active candidates that currently look less likely
- `exclusion_evidence`: negative or opposing evidence that weakens those candidates
- `required_missing_evidence`: evidence still needed for stronger exclusion
- `exclusion_confidence`: `low`, `medium`, or `high`

The output should foreground negative evidence, missing evidence, and reasoning transparency rather than disease classification.
