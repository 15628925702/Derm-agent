# metadata_consistency_skill

## Purpose
Check whether image evidence and case metadata tell a coherent story so downstream reasoning can weight the evidence with appropriate trust.

## When to Use
Use when metadata credibility or image-metadata coherence could alter confidence in the reasoning process.
Use when size, site, symptom, or evolution fields appear important to the case.
Do not use this skill to punish uncertainty; use it to identify real mismatch versus simple lack of visibility.

## Clinical Pattern
This skill asks:
- does the image impression fit the stated site and lesion context,
- do metadata claims about size, symptoms, or change align with what is visible,
- are there direct contradictions,
- are there softer suspicious points that reduce trust without proving conflict.

The target is evidence reliability, not diagnosis.

## Strategy Overview
Clinical reasoning degrades when image and metadata are treated as equally valid without coherence checking.
This skill should distinguish hard contradiction, soft concern, and acceptable uncertainty.
Its explainability value is that later CoT can say not only what evidence exists, but also how trustworthy or internally coherent that evidence appears.

## Workflow
1. First identify which metadata fields are most relevant to visible lesion interpretation, such as region, diameter, symptoms, and elevation.
2. Then compare those fields against the current image summary and earlier observation skills.
3. Then separate direct conflicts from softer suspicious points or credibility concerns.
4. Then note important negative evidence, such as no clear mismatch between stated site and visible context.
5. Then assign an overall consistency score that reflects coherence, not certainty of diagnosis.
6. Finally preserve uncertainty when the image simply cannot verify a metadata claim.

## Evidence to Check
- site metadata,
- size metadata,
- symptom and change history,
- perception summary,
- outputs from morphology, border, color, and distribution skills,
- negative evidence showing where mismatch is not convincingly present.

## Watch Out For
- Weak visibility is not the same thing as contradiction.
- Do not mark uncertainty itself as a hard conflict.
- Benign-looking images may still have concerning metadata and vice versa.
- Do not convert poor consistency into a disease label.

## Output Contract
Return structured evidence-reliability information only.

- `consistency_score`: `high`, `medium`, `low`, or `unknown`
- `conflicts`: direct contradictions between image impression and metadata
- `suspicious_points`: softer mismatches, reliability concerns, or credibility questions

The output should improve CoT interpretability by showing how trustworthy the evidence chain is before downstream integration happens.
