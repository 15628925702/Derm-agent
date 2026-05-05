# distribution_analysis_skill

## Purpose
Describe where the lesion is located and how the visible pattern is distributed so later reasoning can use anatomic and distribution context explicitly.

## When to Use
Use when body site or visible distribution pattern may shift how downstream reasoning weighs the lesion.
Use when metadata provides region information or when the image context suggests solitary, clustered, or broader involvement.
Do not use this skill to infer a diagnosis from location alone.

## Clinical Pattern
This skill focuses on:
- the likely body location,
- whether the visible pattern appears symmetric or asymmetric,
- whether the visible process seems localized or generalized,
- whether lesions appear solitary, clustered, or scattered.

The point is contextual description, not disease classification.

## Strategy Overview
Dermatologists routinely use anatomic site and pattern extent as reasoning anchors, but they also know that a single image can under-represent real distribution.
This skill should therefore separate what is visible from what is only suggested by metadata.
Its explainability value is that later reasoning can say exactly which site and distribution assumptions were made and which remained uncertain.

## Workflow
1. First identify the most likely anatomic site from metadata and visible context, preferring explicit metadata when image context is weak.
2. Then decide whether the visible pattern appears isolated to one area or suggests a broader process.
3. Then inspect whether visible lesions look solitary, clustered, or scattered.
4. Then judge whether symmetry can actually be assessed or whether the image crop makes that impossible.
5. Then record important negative evidence, such as no reliable evidence of generalized spread.
6. Finally preserve uncertainty when a single close-up image does not support confident distribution claims.

## Evidence to Check
- region metadata,
- background skin context,
- number and grouping of visible lesions,
- visible balance across the field,
- whether the crop is too narrow to judge true extent,
- negative evidence showing absence of reliable generalized or bilateral pattern.

## Watch Out For
- Do not infer generalized disease from a tightly cropped image.
- Do not overrule clear metadata with weak background cues.
- Do not treat symmetry as assessable when only one lesion is visible.
- Do not turn location into a diagnosis shortcut.

## Output Contract
Return structured distribution evidence only.

- `body_location`: most likely body site
- `symmetry`: `symmetric`, `asymmetric`, or `uncertain`
- `localized_vs_generalized`: `localized`, `generalized`, or `uncertain`
- `clustering_pattern`: `solitary`, `clustered`, `scattered`, or `unknown`

The output should improve CoT interpretability by making site and extent assumptions explicit instead of leaving them implicit inside later reasoning.
