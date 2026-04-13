# Skill Specs

This directory stores the text-layer clinical specifications for DermAgent skills.

These files are intentionally separate from the Python implementation so the reasoning workflow remains:
- persistent,
- reviewable,
- versionable,
- reusable for prompt construction, evaluation, and future training data generation.

## Design Rules

- Each file describes one atomic clinical reasoning action.
- A skill spec must not act like a disease classifier.
- A skill spec must emphasize observable evidence, negative evidence, uncertainty, and reasoning transparency.
- The text layer exists to strengthen explainable chain-of-thought structure without exposing uncontrolled free-form diagnosis behavior.
- Output contracts must stay aligned with the corresponding `skills/*.py` implementation.

## Section Template

Each skill file uses the same section layout:

- `Purpose`
- `When to Use`
- `Clinical Pattern`
- `Strategy Overview`
- `Workflow`
- `Evidence to Check`
- `Watch Out For`
- `Output Contract`

## Directory Layout

- `observation/`: first-pass descriptive skills
- `reasoning/`: comparison, consistency, and temporal reasoning skills
- `risk_uncertainty/`: risk and uncertainty framing skills

## Code Mapping

- `observation/morphology_analysis_skill.md` -> [skills/morphology.py](/root/DermAgent/skills/morphology.py)
- `observation/color_pattern_analysis_skill.md` -> [skills/color_pattern.py](/root/DermAgent/skills/color_pattern.py)
- `observation/border_surface_analysis_skill.md` -> [skills/border_surface.py](/root/DermAgent/skills/border_surface.py)
- `observation/distribution_analysis_skill.md` -> [skills/distribution.py](/root/DermAgent/skills/distribution.py)
- `observation/lesion_description_structuring_skill.md` -> [skills/lesion_description_structuring.py](/root/DermAgent/skills/lesion_description_structuring.py)
- `reasoning/temporal_evolution_skill.md` -> [skills/temporal_evolution.py](/root/DermAgent/skills/temporal_evolution.py)
- `reasoning/metadata_consistency_skill.md` -> [skills/metadata_consistency.py](/root/DermAgent/skills/metadata_consistency.py)
- `reasoning/differential_compare_skill.md` -> [skills/differential_compare.py](/root/DermAgent/skills/differential_compare.py)
- `reasoning/exclusion_reasoning_skill.md` -> [skills/exclusion_reasoning.py](/root/DermAgent/skills/exclusion_reasoning.py)
- `reasoning/contradiction_check_skill.md` -> [skills/contradiction_check.py](/root/DermAgent/skills/contradiction_check.py)
- `risk_uncertainty/malignancy_risk_assessment_skill.md` -> [skills/malignancy_risk.py](/root/DermAgent/skills/malignancy_risk.py)
- `risk_uncertainty/uncertainty_assessment_skill.md` -> [skills/uncertainty.py](/root/DermAgent/skills/uncertainty.py)
- `risk_uncertainty/information_gap_detection_skill.md` -> [skills/information_gap_detection.py](/root/DermAgent/skills/information_gap_detection.py)
- `risk_uncertainty/escalation_recommendation_skill.md` -> [skills/escalation_recommendation.py](/root/DermAgent/skills/escalation_recommendation.py)

## Maintenance Note

When a Python skill changes its workflow intent or output schema, the matching Markdown file should be updated in the same change.
