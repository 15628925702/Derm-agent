# border_surface_analysis_skill

## Purpose
Describe lesion margin and surface behavior in dermatologist-style terms so downstream reasoning can weigh edge quality and surface change explicitly.

## When to Use
Use early when lesion margin and surface texture may meaningfully constrain the visual description.
Use whenever irregular edge, scale, keratotic change, ulceration, or surface roughness could alter later reasoning.
Do not use this skill to infer a disease label from border irregularity alone.

## Clinical Pattern
This skill examines:
- whether the lesion border is sharply demarcated or poorly defined,
- whether the contour is regular or irregular,
- whether the surface is smooth, rough, keratotic, crusted, or ulcerated,
- whether visible scaling is present.

The clinical target is structured edge and surface evidence, not a final disease conclusion.

## Strategy Overview
Dermatologists often inspect border and surface after defining the lesion form, because these findings can support or weaken later hypotheses.
This skill should make the reasoning path explicit: edge definition, contour quality, surface texture, then specific surface change.
Its explainability value is that later risk or differential steps can cite exactly which border or surface finding was present, absent, or uncertain.

## Workflow
1. First inspect the lesion boundary and decide how clearly the edge is demarcated from surrounding skin.
2. Then assess the contour shape and decide whether irregularity is convincing or only apparent because of crop, focus, or shadow.
3. Then inspect the surface itself for smoothness, roughness, keratosis, crust, or ulcerative change.
4. Then check specifically for scale and decide whether it is definite, absent, or uncertain.
5. Then record negative evidence, such as no convincing ulceration or no reliable surface break.
6. Finally preserve uncertainty if glare, hair, blur, or low contrast limits edge or surface judgment.

## Evidence to Check
- edge sharpness,
- contour regularity,
- border interruption or fading,
- smooth versus rough surface quality,
- visible scale, crust, keratosis, or ulceration,
- negative evidence showing which concerning surface changes are not clearly present.

## Watch Out For
- Do not infer malignancy from irregular border alone.
- Do not let blur or shadow masquerade as poor border definition.
- Do not mistake a specular highlight for surface smoothness or absence of scale.
- Do not call ulceration unless a true surface defect is visible.

## Output Contract
Return structured border and surface evidence only.

- `border_clarity`: clarity of lesion boundary
- `border_irregularity`: regularity versus irregularity of lesion outline
- `surface_texture`: visible surface category such as smooth, rough, keratotic, crusted, ulcerated, or unknown
- `scaling_presence`: whether visible scaling is `present`, `absent`, or `uncertain`

The output should strengthen CoT transparency by separating edge evidence from surface evidence and by preserving negative findings where clinically important.
