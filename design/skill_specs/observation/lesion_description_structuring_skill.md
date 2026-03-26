# lesion_description_structuring_skill

## Purpose
Organize image findings and relevant metadata into a standard dermatologist-style lesion description that later reasoning can reuse directly.

## When to Use
Use after first-pass observation skills when the case needs a stable descriptive summary before comparison, exclusion, risk framing, or uncertainty auditing.
Use when the system has multiple descriptive fragments and needs them normalized into one reusable clinical description.
Do not use this skill to choose or imply a disease label.

## Clinical Pattern
This skill standardizes lesion description across these slots:
- primary lesion morphology,
- color,
- border,
- surface,
- size and count,
- distribution,
- associated clinical context.

Its value is that later chain-of-thought can cite an explicit structured lesion description instead of loosely restating the image.

## Strategy Overview
Doctors often restate a lesion in standard descriptive language before they narrow the differential.
This skill therefore acts like a clinical description formatter: it integrates image evidence with only the metadata that materially changes lesion interpretation.
Its explainability value is high because downstream reasoning can point to stable description slots rather than opaque intuition.

## Workflow
1. First decide what the primary lesion is and state the best standard morphology phrase.
2. Then organize visible evidence into color, border, surface, size/count, and distribution buckets.
3. Then compare image-derived description with useful metadata such as site, symptoms, or temporal context and place only clinically relevant items into associated context.
4. Then preserve important negative evidence such as no obvious scale, no clear ulceration, or no reliable evidence of multiplicity when those absences matter.
5. Then mark uncertainty indirectly through cautious phrasing if the image does not support a strong structural claim.
6. Finally ensure the description stays reusable and disease-agnostic.

## Evidence to Check
- dominant lesion form,
- dominant and secondary colors,
- border sharpness and regularity,
- surface character,
- size clues and lesion count,
- anatomic site and visible distribution,
- clinically relevant metadata that should travel with the description,
- important absent findings and weakly supported findings.

## Watch Out For
- Do not let description drift into diagnosis.
- Do not insert metadata just because it exists; only include context that changes lesion reading.
- Do not mix risk judgment into the description layer.
- Do not overstate subtle features that are weakly seen.
- Do not erase negative evidence that later exclusion reasoning will need.

## Output Contract
Return standardized lesion-description evidence only.

- `primary_lesion_morphology`: one concise dermatologist-style morphology phrase
- `color`: reusable color description elements
- `border`: reusable border description elements
- `surface`: reusable surface description elements
- `size_count`: standardized size and count descriptors
- `distribution`: standardized site and distribution descriptors
- `associated_context`: relevant contextual descriptors from metadata

The output should remain descriptive, reusable, and directly supportive of explainable CoT.
