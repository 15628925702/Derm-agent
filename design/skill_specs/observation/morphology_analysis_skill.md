# morphology_analysis_skill

## Purpose
Describe the lesion in dermatologist-style morphological terms so downstream reasoning starts from visible structure rather than disease naming.

## When to Use
Use early in almost every case before comparison, risk framing, or contradiction auditing.
Use when the system still needs a stable answer to the question: what kind of lesion form is actually visible?
Do not use this skill to decide whether the lesion is a specific disease.

## Clinical Pattern
This skill focuses on the dominant lesion form:
- whether the lesion appears flat or raised,
- whether it is best described as macule, papule, plaque, or nodule,
- whether the visible finding is solitary or part of a grouped pattern,
- whether there is enough visual context to estimate only a coarse size band.

The clinical value is descriptive anchoring, not diagnostic commitment.

## Strategy Overview
A dermatologist usually starts by naming morphology before interpreting significance.
This skill should therefore stabilize the observation layer, capture useful negative evidence, and preserve uncertainty when scale or elevation cannot be judged confidently.
Its main explainability value is that later reasoning can point back to explicit structure terms instead of hidden intuition.

## Workflow
1. First inspect the dominant lesion silhouette and choose the best gross structural description.
2. Then assess whether the lesion appears flat, raised, or mixed; avoid overcalling nodularity from shadow, crust, or oblique angle.
3. Then estimate only a coarse size range using image context and metadata when available.
4. Then check whether the visible pattern is solitary, multiple, or clustered.
5. Then record negative evidence that matters, such as no clear deep nodularity or no reliable scale reference.
6. Finally preserve uncertainty explicitly if morphology is limited by crop, blur, glare, or angle.

## Evidence to Check
- overall lesion silhouette,
- relative elevation cues,
- surface contour suggesting flat versus raised structure,
- available size clues from metadata or image context,
- solitary versus grouped visible presentation,
- negative evidence showing what is not clearly present.

## Watch Out For
- Do not convert morphology into diagnosis.
- Do not infer malignancy from elevation alone.
- Do not overestimate size when scale is unreliable.
- Do not confuse glare, crust, or shadow with true nodularity.
- Do not hide uncertainty when lesion form is visually ambiguous.

## Output Contract
Return structured morphology evidence only.

- `lesion_type`: primary morphological class such as `macule`, `papule`, `plaque`, `nodule`, or `unknown`
- `size_range`: coarse size band only
- `elevation`: `flat`, `raised`, `mixed`, or `unknown`
- `count`: `solitary`, `multiple`, `clustered`, or `unknown`

The output should remain descriptive, preserve weak-evidence uncertainty, and improve CoT interpretability by making the observation basis explicit.
