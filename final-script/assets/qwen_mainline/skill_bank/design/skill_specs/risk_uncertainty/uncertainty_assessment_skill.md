# uncertainty_assessment_skill

## Purpose
Make ambiguity, missing information, and reasoning limits explicit so the system can remain clinically cautious and interpretable.

## When to Use
Use when visual ambiguity, unresolved differential conflict, missing metadata, or evidence weakness remains clinically meaningful.
Use after enough case context exists to judge where the reasoning chain is limited.
Do not use this skill as a generic fallback; use it to state specific uncertainty sources.

## Clinical Pattern
This skill characterizes:
- how uncertain the current reasoning state is,
- why that uncertainty exists,
- which missing observations or metadata matter most,
- which negative findings reduce or fail to reduce uncertainty.

The target is self-awareness of evidence quality, not diagnostic indecision phrased vaguely.

## Strategy Overview
In dermatologist-style reasoning, uncertainty is not failure; it is structured knowledge about the limits of the current evidence.
This skill should expose ambiguity rather than hide it behind confident wording.
Its explainability value is direct: later CoT can show not only what the model thinks, but also where and why the evidence remains weak, incomplete, or internally unstable.

## Workflow
1. First scan the image and prior skill outputs for ambiguous findings, weakly supported descriptors, or unresolved disagreements.
2. Then identify which metadata fields or clinical context elements are missing and why they matter.
3. Then check whether any reassuring negative evidence meaningfully lowers uncertainty.
4. Then separate uncertainty due to poor image quality from uncertainty due to genuine overlap between plausible explanations.
5. Then assign a low, medium, or high uncertainty level that matches the actual evidence burden.
6. Finally state the missing information explicitly so downstream reasoning can remain cautious and interpretable.

## Evidence to Check
- image ambiguity such as blur, crop, glare, or low contrast,
- unresolved tension between skill outputs,
- missing metadata for size, site, symptoms, or evolution,
- differential comparisons that remain open,
- negative evidence showing when a feared ambiguity is not actually present.

## Watch Out For
- Do not understate uncertainty just because the lesion looks superficially simple.
- Do not confuse uncertainty with contradiction.
- Do not use vague language like "not sure" without naming the reason.
- Do not let uncertainty drift into final diagnosis discussion.

## Output Contract
Return structured uncertainty evidence only.

- `uncertainty_level`: `low`, `medium`, `high`, or `unknown`
- `reasons`: explicit reasons supporting the uncertainty level
- `missing_information`: key information that would most improve reasoning quality

The output should strengthen CoT interpretability by making evidence limits explicit, inspectable, and clinically meaningful for downstream integration.
