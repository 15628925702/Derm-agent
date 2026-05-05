# color_pattern_analysis_skill

## Purpose
Describe lesion color and pigment organization in a way that helps downstream reasoning compare visual patterns without collapsing into disease naming.

## When to Use
Use when pigment, color heterogeneity, or asymmetry may help refine the current differential.
Use after basic morphology has been established or whenever color becomes a major point of disagreement.
Do not use this skill as a shortcut to declare a malignant diagnosis.

## Clinical Pattern
This skill looks at:
- the dominant lesion color,
- whether color is uniform or heterogeneous,
- how pigment is distributed across the lesion,
- whether color asymmetry is convincingly present.

The focus is pattern description, not pattern verdict.

## Strategy Overview
Dermatologists often use color as a supporting clue rather than an isolated answer.
This skill should separate dominant color, variation, and distribution pattern into explicit components so later reasoning can refer to them individually.
Its explainability value is that it turns a vague impression like "looks uneven" into traceable pigment evidence with stated uncertainty and negative findings.

## Workflow
1. First identify the dominant visible color without over-interpreting lighting artifacts.
2. Then compare the center, periphery, and different halves of the lesion to judge whether color is uniform or varied.
3. Then describe the pigment arrangement as homogeneous, patchy, reticular, mixed, or otherwise uncertain.
4. Then actively check whether color asymmetry is real or only produced by shadow, glare, hair, or compression.
5. Then record negative evidence, such as absence of striking multicolor change or absence of convincing asymmetry.
6. Finally preserve uncertainty if image quality, white balance, or partial occlusion weakens confidence.

## Evidence to Check
- dominant pigment tone,
- degree of color variation,
- center-to-edge pigment distribution,
- left-right or top-bottom color asymmetry,
- artifact sources such as shadow, blur, glare, and white balance shift,
- negative evidence showing what color irregularity is not convincingly present.

## Watch Out For
- Do not equate multicolor appearance with a final melanoma diagnosis.
- Do not let lighting or white balance create false heterogeneity.
- Do not call asymmetry from a single dark shadow zone without checking artifact.
- Do not ignore reassuring negative evidence just because one area looks darker.

## Output Contract
Return structured pigment evidence only.

- `primary_color`: dominant visible lesion color
- `color_variation`: whether color is `uniform`, mildly varied, markedly varied, or `unknown`
- `pigmentation_pattern`: structured description of pigment arrangement
- `asymmetry_color`: whether color asymmetry is `present`, `absent`, or `uncertain`

The output should make pigment reasoning auditable so later CoT can cite color evidence rather than rely on hidden visual intuition.
